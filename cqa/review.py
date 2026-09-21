"""PR / diff review mode.

Everything here is deterministic preparation — no LLM. Given a git range (or a pasted
unified diff) we work out *what changed* (files, hunks, which indexed symbols the changed
lines fall inside) and *what that could affect* (callers from the call graph, callers that
live in test/, files that #include a changed header). The result is one context block that
is handed to the normal QA graph in `mode="review"`, so the reviewer LLM can still use tools
(read_file, git_log, ...) and gets checkpointed follow-ups, citations, tracing, feedback and
metrics for free.

Line numbers: the symbol index describes the tree as of the last index build. If the diff's
head is not that commit, symbol mapping and caller lists are approximate — `warnings` says so.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from unidiff import PatchSet
from unidiff.errors import UnidiffParseError

from .callgraph import CallGraph
from .config import Config
from .gittools import run_git, valid_ref
from .symbol_index import SymbolDef, SymbolIndex

REVIEW_SYSTEM_PROMPT = """You are CodebaseQA in code-review mode for the LuxTrace C++ codebase.

You are given a change (a git diff) plus deterministic analysis of what it touches: the \
indexed symbols the changed lines fall inside, their callers, callers that live in test/, and \
files that include changed headers. Review the change like a careful senior engineer.

Rules:
- Ground every point in the diff or the code: cite `path:line` (for added/changed lines use \
the NEW file's line numbers; the hunk headers `@@ -a,b +c,d @@` give them). Never invent \
files, symbols, or behavior.
- The caller/test/include lists come from a static, name-based analysis — treat them as \
strong hints, and use `symbol_usages`, `read_file`, `git_log` etc. to verify anything your \
conclusion depends on. Read the surrounding code before claiming a bug.
- If the change looks fine, say so plainly. Do not pad the review with speculative issues.

Answer in this structure (Markdown):
**Summary** — 2-3 sentences: what the change does.
**Risk** — Low / Medium / High, with a one-line reason.
**Impact** — which callers / tests / includers are affected and how (cite them).
**Concerns** — a numbered list; each with severity (bug / risk / nit), a citation, and why.
**Missing tests** — behavior that changed but has no covering test you can find.
**Suggested checks** — concrete things to run or verify before merging.
"""

_SYMBOL_KINDS = ("function", "method")
MAX_RAW_DIFF_CHARS = 2_000_000  # hard ceiling on what we read from git / accept as pasted text


def _cut_at_file_boundary(raw: str, limit: int) -> tuple[str, int]:
    """Cut a unified diff to at most `limit` chars *between* files (never mid-hunk, which
    would not parse). Returns (text, files_dropped)."""
    if len(raw) <= limit:
        return raw, 0
    marker = "\ndiff --git "
    cut = raw.rfind(marker, 0, limit)
    if cut <= 0:
        # a single enormous file: keep it whole rather than return nothing parseable
        return raw, 0
    dropped = raw.count(marker, cut + 1)
    return raw[: cut + 1], dropped


@dataclass
class ReviewPrep:
    title: str
    context: str
    stats: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def get_diff(cfg: Config, base: str, head: str) -> str:
    """`git diff base...head` (merge-base form) with refs validated. Raises ValueError."""
    for label, ref in (("base", base), ("head", head)):
        if not valid_ref(ref) or ".." in ref:
            raise ValueError(f"invalid {label} ref: {ref!r}")
    ok, out = run_git(
        cfg,
        ["diff", "--no-color", "--unified=3", "--find-renames", f"{base}...{head}", "--"],
        timeout=60,
        max_chars=MAX_RAW_DIFF_CHARS,
    )
    if not ok:
        raise ValueError(f"git diff failed: {out}")
    if not out.strip():
        raise ValueError(f"no changes between {base} and {head}")
    return out


def _rev_parse(cfg: Config, ref: str) -> str | None:
    if not valid_ref(ref):
        return None
    ok, out = run_git(cfg, ["rev-parse", "--verify", "--quiet", ref + "^{commit}"])
    return out.strip() if ok and out.strip() else None


def _changed_target_lines(pf) -> list[int]:
    """New-file line numbers touched by a patched file. A pure deletion is attributed to the
    line right after it (where the removed code used to sit)."""
    lines: list[int] = []
    for hunk in pf:
        last = hunk.target_start - 1
        for ln in hunk:
            if ln.line_type == "+":
                lines.append(ln.target_line_no)
                last = ln.target_line_no
            elif ln.line_type == "-":
                lines.append(last + 1)
            else:
                last = ln.target_line_no
    return lines


def _enclosing(defs: list[SymbolDef], line: int) -> SymbolDef | None:
    inside = [d for d in defs if d.start_line <= line <= d.end_line]
    if not inside:
        return None
    callables = [d for d in inside if d.kind in _SYMBOL_KINDS]
    pool = callables or inside
    return min(pool, key=lambda d: d.end_line - d.start_line)


def _fmt_def(d: SymbolDef) -> str:
    return f"{d.name} ({d.path}:{d.start_line}-{d.end_line})"


_DOC_EXTS = {".md", ".html", ".htm", ".txt", ".json", ".rc", ".yml", ".yaml"}


def _priority(cfg: Config, path: str) -> int:
    """0 = indexed C++ source (what a code review is about), 1 = other code/build files,
    2 = docs, licences and data — the first to be dropped when the prompt budget runs out."""
    from pathlib import PurePosixPath

    p = PurePosixPath(path)
    if p.suffix in cfg.repo.cpp_extensions:
        return 0
    name = p.name.upper()
    if p.suffix.lower() in _DOC_EXTS or name.startswith(("LICENSE", "LICENSING", "COMMERCIAL", "CLA", "THIRD-PARTY", "README")):
        return 2
    return 1


def _budget_diff(cfg: Config, patch: PatchSet, raw: str, max_chars: int) -> tuple[str, list[str]]:
    """Whole diff if it fits; otherwise source files first (smaller first within a tier),
    each capped, listing what was left out so the reviewer knows the diff is partial."""
    if len(raw) <= max_chars:
        return raw, []
    per_file_cap = max(4000, max_chars // 6)
    files = sorted(patch, key=lambda pf: (_priority(cfg, pf.path), pf.added + pf.removed))
    parts: list[str] = []
    used = 0
    omitted: list[str] = []
    for pf in files:
        text = str(pf)
        if len(text) > per_file_cap:
            cut = len(str(pf)) - per_file_cap
            text = text[:per_file_cap] + "\n[... " + pf.path + f": diff truncated ({cut} more chars) ...]\n"
        if used + len(text) > max_chars:
            omitted.append(pf.path)
            continue
        parts.append(text)
        used += len(text)
    return "\n".join(parts), omitted


def prepare_review(
    cfg: Config,
    symbols: SymbolIndex,
    graph: CallGraph | None,
    *,
    base: str | None = None,
    head: str | None = None,
    diff: str | None = None,
) -> ReviewPrep:
    """Build the review context for a git range or a pasted diff. Raises ValueError."""
    warnings: list[str] = []
    if diff:
        raw = diff
        title = "Review of pasted diff"
    else:
        base = base or cfg.review.default_base
        head = head or cfg.review.default_head
        raw = get_diff(cfg, base, head)
        title = f"Review changes {base}...{head}"
        state_path = cfg.storage.index_state_file
        try:
            indexed = json.loads(state_path.read_text(encoding="utf-8")).get("git_commit")
        except (OSError, json.JSONDecodeError):
            indexed = None
        head_sha = _rev_parse(cfg, head)
        if indexed and head_sha and indexed != head_sha:
            warnings.append(
                f"The index was built at {indexed[:7]} but the diff head is {head_sha[:7]}: symbol "
                "line ranges and caller lists are approximate. Re-run scripts/index_luxtrace.py for exact mapping."
            )
    raw, dropped = _cut_at_file_boundary(raw, MAX_RAW_DIFF_CHARS)
    if dropped:
        warnings.append(f"The diff exceeds {MAX_RAW_DIFF_CHARS:,} characters: the last {dropped} file(s) were not analysed.")

    try:
        patch = PatchSet(raw)
    except (UnidiffParseError, Exception) as e:  # noqa: BLE001 — pasted text may not be a valid diff
        raise ValueError(f"could not parse the diff: {e}") from e
    if not len(patch):
        raise ValueError("the diff contains no file changes")

    n_add = sum(pf.added for pf in patch)
    n_del = sum(pf.removed for pf in patch)

    changed: list[tuple[SymbolDef, str]] = []  # (symbol, status)
    seen_syms: set[str] = set()
    other_files: list[str] = []
    header_changes: list[str] = []
    file_lines: list[str] = []
    for pf in sorted(patch, key=lambda pf: _priority(cfg, pf.path)):
        path = pf.path
        status = "added" if pf.is_added_file else "deleted" if pf.is_removed_file else "modified"
        if pf.is_rename:
            status = "renamed"
        file_lines.append(f"- {path} [{status}] +{pf.added} -{pf.removed}")
        defs = symbols.defs_in_path(path)
        if not defs:
            other_files.append(path)
            continue
        if path.endswith((".h", ".hpp", ".hxx", ".cuh")):
            header_changes.append(path)
        if pf.is_removed_file:
            continue
        for line in _changed_target_lines(pf):
            d = _enclosing(defs, line)
            if d and d.name not in seen_syms:
                seen_syms.add(d.name)
                changed.append((d, "added" if pf.is_added_file else "modified"))

    impact: list[str] = []
    n_tests = 0
    for d, st in changed[:20]:
        callers = graph.callers.get(d.name, []) if graph else []
        prod, tests = [], []
        for c in callers:
            cd = symbols.definitions(c)
            (tests if cd and cd[0].path.startswith("test/") else prod).append(_fmt_def(cd[0]) if cd else c)
        n_tests += len(tests)
        line = f"- {_fmt_def(d)} [{st}]"
        line += f"\n    callers ({len(prod)}): " + ("; ".join(prod[:8]) + (" …" if len(prod) > 8 else "") if prod else "none found (may be called through member access — verify with symbol_usages)")
        line += f"\n    callers in test/ ({len(tests)}): " + ("; ".join(tests[:6]) if tests else "none found")
        impact.append(line)
    if len(changed) > 20:
        impact.append(f"- … and {len(changed) - 20} more changed symbols (not listed)")

    includers: list[str] = []
    for h in header_changes[:8]:
        inc = graph.included_by.get(h, []) if graph else []
        includers.append(f"- {h} is included by {len(inc)} file(s): " + ", ".join(inc[:10]) + (" …" if len(inc) > 10 else ""))

    diff_text, omitted = _budget_diff(cfg, patch, raw, cfg.review.max_diff_chars)
    if omitted:
        warnings.append(f"Diff budget reached — {len(omitted)} file(s) omitted from the prompt: " + ", ".join(omitted[:10]))

    parts = [
        f"## Change under review\n{title}: {len(patch)} file(s), +{n_add} −{n_del} lines.\n" + "\n".join(file_lines[:60]),
    ]
    if warnings:
        parts.append("## Analysis caveats\n" + "\n".join(f"- {w}" for w in warnings))
    parts.append(
        "## Changed indexed symbols and their impact (static analysis)\n"
        + ("\n".join(impact) if impact else "(no indexed symbol contains a changed line — build files / docs / new code outside indexed definitions)")
    )
    if includers:
        parts.append("## Changed headers and their includers\n" + "\n".join(includers))
    if other_files:
        parts.append("## Changed files with no indexed symbols\n" + ", ".join(other_files[:30]))
    parts.append("## Diff\n```diff\n" + diff_text.rstrip() + "\n```")

    return ReviewPrep(
        title=title,
        context="\n\n".join(parts),
        stats={
            "files": len(patch),
            "additions": n_add,
            "deletions": n_del,
            "changed_symbols": len(changed),
            "tests_calling_changed": n_tests,
            "diff_chars": len(diff_text),
            "omitted_files": len(omitted),
        },
        warnings=warnings,
    )
