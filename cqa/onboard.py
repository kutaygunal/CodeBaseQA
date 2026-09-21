"""Onboarding tours: "explain this file / module to a new engineer".

Like review mode, the preparation is deterministic — a *skeleton* assembled from the symbol
index and the call/include graph (what's defined here, who depends on it, what it depends on,
where the entry points and tests are). The LLM turns that into a guided tour via the normal
QA graph in `mode="tour"`, and can still read code with its tools. Finished tours are cached
per (target, index build, provider, model) in the `tours` table — see `index_key`.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import PurePosixPath

from .callgraph import CallGraph
from .config import Config
from .symbol_index import SymbolIndex
from .tools import _safe_resolve

TOUR_SYSTEM_PROMPT = """You are CodebaseQA writing an onboarding tour of part of the LuxTrace \
C++ codebase for an engineer who has just joined.

You are given a deterministic skeleton (definitions, dependencies, entry points, tests) built \
from the index. Use your tools (read_file, search_symbols, symbol_usages, call_graph) to read the \
key code before you write — the skeleton tells you where to look, it is not the code.

Rules:
- Ground every claim in a citation `path:start-end` from the skeleton or from code you read. \
Never invent files, symbols, or behavior.
- Be concrete and skimmable; prefer a short table or bullets over paragraphs.

Write these sections (Markdown, with `##` headings):
## What it is for
2-3 sentences.
## Key types and functions
The handful that matter, each with a one-line role and a citation.
## Entry points
Where execution enters this code and who calls it.
## How it works
The main flow as numbered steps.
## Suggested reading order
An ordered list: which file/lines to open first, second, third, and why.
## Gotchas
Invariants, performance traps, threading or ownership rules, surprising behavior.
## Tests
Where it is exercised (path:line), or say none were found.

If a diagram would help, call `call_graph` on one or two central symbols and include its \
`mermaid` output verbatim in a ```mermaid block.
Keep it about 500 words for a file and 800 for a module."""


@dataclass
class TourPrep:
    target: str  # normalized: a repo-relative file path or directory
    kind: str  # "file" | "module"
    title: str
    context: str


def index_key(cfg: Config) -> str:
    """Identifies one index build. A tour cached under an older key is stale by definition:
    the commit changed, or the index was rebuilt (working-tree edits included)."""
    try:
        st = json.loads(cfg.storage.index_state_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "unknown"
    return f"{(st.get('git_commit') or 'nogit')[:12]}:{int(st.get('built_at', 0))}"


def _indexed_paths(symbols: SymbolIndex, graph: CallGraph | None) -> set[str]:
    paths: set[str] = set()
    for name in symbols.all_names():
        for d in symbols.definitions(name):
            paths.add(d.path)
    if graph:
        for a, bs in graph.includes.items():
            paths.add(a)
            paths.update(bs)
    return paths


def resolve_target(cfg: Config, symbols: SymbolIndex, graph: CallGraph | None, target: str) -> tuple[str, str]:
    """(kind, normalized target). Accepts a file path, a directory, or a module tag like 'core'."""
    target = (target or "").strip().strip("/").replace("\\", "/")
    if not target:
        raise ValueError("no tour target given")
    paths = _indexed_paths(symbols, graph)
    if target in paths:
        return "file", target
    for cand in (target, f"src/{target}"):
        prefix = cand + "/"
        if any(p.startswith(prefix) for p in paths):
            return "module", cand
    raise ValueError(f"'{target}' is not an indexed file or module")


_LICENSE_RE = re.compile(r"SPDX|copyright|licen[sc]e|all rights reserved", re.I)


def _header_comment(cfg: Config, path: str, max_chars: int = 180) -> str:
    """First non-licence comment block of a file — usually its one-line purpose."""
    p = _safe_resolve(cfg, path)
    if p is None or not p.is_file():
        return ""
    try:
        head = p.read_text(encoding="utf-8", errors="replace").split("\n")[:60]
    except OSError:
        return ""
    blocks: list[list[str]] = []
    cur: list[str] = []
    in_block = False
    for line in head:
        s = line.strip()
        if in_block:
            cur.append(s.rstrip("*/").lstrip("* ").strip())
            if "*/" in s:
                in_block = False
                blocks.append(cur)
                cur = []
        elif s.startswith("//"):
            cur.append(s.lstrip("/ ").strip())
        elif s.startswith("/*"):
            in_block = "*/" not in s
            cur.append(s.lstrip("/* ").rstrip("*/").strip())
            if not in_block:
                blocks.append(cur)
                cur = []
        else:
            if cur:
                blocks.append(cur)
                cur = []
    if cur:
        blocks.append(cur)
    for b in blocks:
        text = re.sub(r"\s+", " ", " ".join(x for x in b if x)).strip()
        if text and not _LICENSE_RE.search(text):
            return text[:max_chars]
    return ""


def _n_lines(cfg: Config, path: str) -> int:
    p = _safe_resolve(cfg, path)
    try:
        return len(p.read_text(encoding="utf-8", errors="replace").split("\n")) if p and p.is_file() else 0
    except OSError:
        return 0


def _sym_path(symbols: SymbolIndex, name: str) -> str:
    d = symbols.definitions(name)
    return d[0].path if d else ""


def _fmt(symbols: SymbolIndex, name: str) -> str:
    d = symbols.definitions(name)
    return f"{name} ({d[0].path}:{d[0].start_line}-{d[0].end_line})" if d else name


def _file_skeleton(cfg: Config, symbols: SymbolIndex, graph: CallGraph | None, path: str) -> str:
    defs = symbols.defs_in_path(path)
    lines = [f"File: {path} ({_n_lines(cfg, path)} lines)"]
    hc = _header_comment(cfg, path)
    if hc:
        lines.append(f"Header comment: {hc}")
    lines.append(f"\nDefinitions ({len(defs)}):")
    for d in defs[:60]:
        lines.append(f"- {d.kind} {d.name} — {d.path}:{d.start_line}-{d.end_line}")
    if len(defs) > 60:
        lines.append(f"- … and {len(defs) - 60} more")
    if graph:
        pair = graph.pairs.get(path)
        if pair:
            lines.append(f"\nHeader/implementation pair: {pair}")
        inc = graph.includes.get(path, [])
        if inc:
            lines.append(f"Includes (project headers): {', '.join(inc[:15])}")
        inc_by = graph.included_by.get(path, [])
        if inc_by:
            lines.append(f"Included by {len(inc_by)} file(s): {', '.join(inc_by[:12])}{' …' if len(inc_by) > 12 else ''}")
        entry: Counter = Counter()
        tests: list[str] = []
        deps: Counter = Counter()
        for d in defs:
            key = f"{d.name} ({d.path}:{d.start_line}-{d.end_line})"
            for c in graph.callers.get(d.name, []):
                cp = _sym_path(symbols, c)
                if cp and cp != path:
                    entry[key] += 1
                    if cp.startswith("test/") and len(tests) < 8:
                        tests.append(_fmt(symbols, c))
            for callee in graph.callees.get(d.name, []):
                if _sym_path(symbols, callee) not in ("", path):
                    deps[callee] += 1
        if entry:
            lines.append("\nMost-called from other files (likely entry points):")
            for key, c in entry.most_common(8):
                lines.append(f"- {key} — {c} external caller(s)")
        if deps:
            lines.append("\nMain dependencies on other files:")
            for n, c in deps.most_common(8):
                lines.append(f"- {_fmt(symbols, n)}")
        if tests:
            lines.append("\nCalled from tests:\n" + "\n".join(f"- {t}" for t in tests))
    return "\n".join(lines)


def _module_skeleton(cfg: Config, symbols: SymbolIndex, graph: CallGraph | None, directory: str) -> str:
    prefix = directory + "/"
    paths = sorted(p for p in _indexed_paths(symbols, graph) if p.startswith(prefix))
    lines = [f"Module: {directory} — {len(paths)} indexed files"]
    lines.append("\nFiles (definitions · lines · header comment):")
    for p in paths[:80]:
        hc = _header_comment(cfg, p, 110)
        lines.append(f"- {p} · {len(symbols.defs_in_path(p))} defs · {_n_lines(cfg, p)} lines" + (f" · {hc}" if hc else ""))
    if len(paths) > 80:
        lines.append(f"- … and {len(paths) - 80} more files")
    if graph:
        inside = set(paths)
        inc_counts = Counter({p: len([x for x in graph.included_by.get(p, []) if x in inside]) for p in paths})
        top = [(p, c) for p, c in inc_counts.most_common(8) if c]
        if top:
            lines.append("\nMost-included files inside the module (its shared core): " + ", ".join(f"{p} ({c})" for p, c in top))
        entry: Counter = Counter()
        for p in paths:
            for d in symbols.defs_in_path(p):
                key = f"{d.name} ({d.path}:{d.start_line}-{d.end_line})"
                for c in graph.callers.get(d.name, []):
                    cp = _sym_path(symbols, c)
                    if cp and not cp.startswith(prefix):
                        entry[key] += 1
        if entry:
            lines.append("\nMost-called from OUTSIDE the module (its public surface):")
            for key, c in entry.most_common(12):
                lines.append(f"- {key} — {c} external caller(s)")
        out_mods: Counter = Counter()
        for p in paths:
            for tgt in graph.includes.get(p, []):
                if not tgt.startswith(prefix):
                    out_mods["/".join(tgt.split("/")[:2])] += 1
        if out_mods:
            lines.append("\nDepends on (by #include): " + ", ".join(f"{m} ({c})" for m, c in out_mods.most_common(6)))
    return "\n".join(lines)


def prepare_tour(cfg: Config, symbols: SymbolIndex, graph: CallGraph | None, target: str) -> TourPrep:
    kind, norm = resolve_target(cfg, symbols, graph, target)
    skeleton = _file_skeleton(cfg, symbols, graph, norm) if kind == "file" else _module_skeleton(cfg, symbols, graph, norm)
    title = f"Explain {'file' if kind == 'file' else 'module'} {norm}"
    context = f"## Tour skeleton (from the index — read the code with your tools before writing)\n{skeleton}"
    return TourPrep(target=norm, kind=kind, title=title, context=context)
