"""Read-only git history tools for the agent: `git_log` and `git_blame`.

Answers "why was this changed?" / "who touches this?" — questions the source text alone
can't. Everything runs `git` with a fixed argument list (no shell), paths are confined to
the indexed repo by `_safe_resolve`, refs are validated, and output is size-capped.
Author *names* are returned; e-mail addresses are deliberately dropped.
"""
from __future__ import annotations

import re
import subprocess
import time

from .config import Config
from .tools import _safe_resolve

MAX_LOG = 20
MAX_BLAME_LINES = 400
MAX_OUTPUT_CHARS = 20000
# Refs/ranges we accept (sha, branch, tag, HEAD~3, main...feature). Never a leading '-'.
_REF_RE = re.compile(r"^[A-Za-z0-9_./~^@{}][A-Za-z0-9_./~^@{}\-]*(\.\.\.?[A-Za-z0-9_./~^@{}][A-Za-z0-9_./~^@{}\-]*)?$")


def valid_ref(ref: str) -> bool:
    return bool(ref) and len(ref) <= 200 and not ref.startswith("-") and bool(_REF_RE.match(ref))


def run_git(cfg: Config, args: list[str], timeout: int = 20, max_chars: int = MAX_OUTPUT_CHARS) -> tuple[bool, str]:
    """Run `git <args>` in the repo root. Returns (ok, stdout-or-error)."""
    try:
        out = subprocess.run(
            ["git", "-c", "core.quotepath=off", *args],
            cwd=cfg.repo.root, capture_output=True, text=True, timeout=timeout,
            encoding="utf-8", errors="replace",
        )
    except FileNotFoundError:
        return False, "git is not installed or not on PATH"
    except subprocess.TimeoutExpired:
        return False, "git timed out"
    if out.returncode != 0:
        return False, (out.stderr or out.stdout or "git failed").strip()[:500]
    return True, out.stdout[:max_chars]


def _tracked_rel(cfg: Config, path: str) -> tuple[str | None, str | None]:
    """(repo-relative posix path, error)."""
    p = _safe_resolve(cfg, path)
    if p is None or not p.is_file():
        return None, f"file not found or outside repo: {path}"
    return p.relative_to(cfg.repo.root.resolve()).as_posix(), None


def git_log(cfg: Config, path: str, start_line: int | None = None, end_line: int | None = None,
            limit: int = 8) -> dict:
    """Commits that touched a file — or, given a line range, that touched *those lines*
    (`git log -L`), newest first."""
    rel, err = _tracked_rel(cfg, path)
    if err:
        return {"error": err}
    limit = max(1, min(int(limit or 8), MAX_LOG))
    fmt = "--format=%h%x09%an%x09%ad%x09%s"
    if start_line:
        s = max(1, int(start_line))
        e = max(s, int(end_line or s))
        args = ["log", "-L", f"{s},{e}:{rel}", "-s", fmt, "--date=short", "-n", str(limit)]
        scope = f"lines {s}-{e}"
    else:
        args = ["log", "--follow", fmt, "--date=short", "-n", str(limit), "--", rel]
        scope = "whole file"
    ok, out = run_git(cfg, args)
    if not ok:
        return {"error": out}
    commits = []
    for line in out.splitlines():
        parts = line.split("\t", 3)
        if len(parts) == 4:
            commits.append({"commit": parts[0], "author": parts[1], "date": parts[2], "subject": parts[3]})
    return {"path": rel, "scope": scope, "commits": commits}


def git_blame(cfg: Config, path: str, start_line: int, end_line: int | None = None) -> dict:
    """Who last changed each line in a range — summarized per commit (line counts)."""
    rel, err = _tracked_rel(cfg, path)
    if err:
        return {"error": err}
    s = max(1, int(start_line))
    e = max(s, int(end_line or s))
    e = min(e, s + MAX_BLAME_LINES - 1)
    ok, out = run_git(cfg, ["blame", "-L", f"{s},{e}", "--porcelain", "--", rel])
    if not ok:
        return {"error": out}
    commits: dict[str, dict] = {}
    cur: str | None = None
    for line in out.splitlines():
        m = re.match(r"^([0-9a-f]{40}) \d+ \d+", line)
        if m:
            cur = m.group(1)
            commits.setdefault(cur, {"lines": 0})["lines"] += 1
        elif cur and line.startswith("author "):
            commits[cur]["author"] = line[7:]
        elif cur and line.startswith("author-time "):
            try:
                commits[cur]["date"] = time.strftime("%Y-%m-%d", time.gmtime(int(line[12:])))
            except ValueError:
                pass
        elif cur and line.startswith("summary "):
            commits[cur]["subject"] = line[8:]
    rows = [
        {"commit": sha[:7], **{k: v for k, v in info.items()}}
        for sha, info in sorted(commits.items(), key=lambda kv: -kv[1]["lines"])
    ]
    return {"path": rel, "start_line": s, "end_line": e, "commits": rows}


def list_refs(cfg: Config, n_commits: int = 15) -> dict:
    """Branches, tags and recent commits for the review form's pickers (read-only)."""
    ok, out = run_git(cfg, ["for-each-ref", "--format=%(refname:short)", "refs/heads", "refs/tags", "--count=60"])
    refs = [r for r in (out.splitlines() if ok else []) if valid_ref(r)]
    ok, out = run_git(cfg, ["log", f"-n{int(n_commits)}", "--format=%h%x09%ad%x09%s", "--date=short"])
    commits = []
    if ok:
        for line in out.splitlines():
            parts = line.split("\t", 2)
            if len(parts) == 3:
                commits.append({"sha": parts[0], "date": parts[1], "subject": parts[2][:90]})
    ok, cur = run_git(cfg, ["rev-parse", "--abbrev-ref", "HEAD"])
    return {"refs": refs, "commits": commits, "current": cur.strip() if ok else None}
