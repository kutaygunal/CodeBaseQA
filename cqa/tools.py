"""Agent tools: read_file, grep/search_symbols, list_dir. See PLAN.md §6 step 4."""
from __future__ import annotations

import re
from pathlib import Path

from .config import Config
from .symbol_index import SymbolIndex, find_usages

MAX_READ_LINES = 400


def _safe_resolve(cfg: Config, rel_path: str) -> Path | None:
    """Resolve rel_path under the repo root, refusing traversal outside it."""
    root = cfg.repo.root.resolve()
    candidate = (root / rel_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    return candidate


def read_file(cfg: Config, path: str, start_line: int = 1, end_line: int | None = None) -> dict:
    """Read a slice of a file in the indexed repo (path relative to repo root)."""
    p = _safe_resolve(cfg, path)
    if p is None or not p.is_file():
        return {"error": f"file not found or outside repo: {path}"}
    lines = p.read_text(encoding="utf-8", errors="replace").split("\n")
    start_line = max(1, start_line)
    end_line = min(len(lines), end_line or (start_line + MAX_READ_LINES - 1))
    end_line = min(end_line, start_line + MAX_READ_LINES - 1)
    snippet = "\n".join(lines[start_line - 1 : end_line])
    return {
        "path": path,
        "start_line": start_line,
        "end_line": end_line,
        "total_lines": len(lines),
        "text": snippet,
    }


MAX_VIEWER_LINES = 4000


def read_full_file(cfg: Config, path: str) -> dict:
    """Read a whole file for the web UI's source viewer (uncapped vs. the agent's
    read_file tool, up to a sanity ceiling for pathologically large files)."""
    p = _safe_resolve(cfg, path)
    if p is None or not p.is_file():
        return {"error": f"file not found or outside repo: {path}"}
    lines = p.read_text(encoding="utf-8", errors="replace").split("\n")
    truncated = len(lines) > MAX_VIEWER_LINES
    return {
        "path": path,
        "total_lines": len(lines),
        "truncated": truncated,
        "text": "\n".join(lines[:MAX_VIEWER_LINES]),
    }


def list_dir(cfg: Config, path: str = "") -> dict:
    """List immediate contents of a directory in the indexed repo."""
    p = _safe_resolve(cfg, path)
    if p is None or not p.is_dir():
        return {"error": f"directory not found or outside repo: {path}"}
    entries = []
    for child in sorted(p.iterdir()):
        if child.name in cfg.repo.ignore_dirs:
            continue
        entries.append(child.name + ("/" if child.is_dir() else ""))
    return {"path": path, "entries": entries}


def grep(cfg: Config, pattern: str, path_prefix: str = "", limit: int = 30) -> dict:
    """Regex/text search across the indexed repo tree (word-unanchored)."""
    try:
        rx = re.compile(pattern)
    except re.error as e:
        return {"error": f"invalid regex: {e}"}
    root = cfg.repo.root
    exts = cfg.repo.cpp_extensions | cfg.repo.text_extensions
    hits: list[dict] = []
    for inc in cfg.repo.include_dirs:
        base = root / inc
        if not base.exists():
            continue
        for fp in sorted(base.rglob("*")):
            if len(hits) >= limit:
                return {"hits": hits, "truncated": True}
            if not fp.is_file() or fp.suffix not in exts:
                continue
            if any(part in cfg.repo.ignore_dirs for part in fp.parts):
                continue
            rel = fp.relative_to(root).as_posix()
            if path_prefix and not rel.startswith(path_prefix):
                continue
            try:
                text = fp.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for i, line in enumerate(text.split("\n"), start=1):
                if rx.search(line):
                    hits.append({"path": rel, "line": i, "text": line.strip()[:200]})
                    if len(hits) >= limit:
                        break
    return {"hits": hits, "truncated": False}


def search_symbols(symbols: SymbolIndex, name: str, limit: int = 20) -> dict:
    """Exact + bare-name + substring lookup of function/class/method symbols."""
    exact = symbols.lookup(name)
    if exact:
        return {"query": name, "matches": [_def_dict(d) for d in exact[:limit]]}
    needle = name.lower()
    fuzzy = [n for n in symbols.all_names() if needle in n.lower()]
    out = []
    for n in fuzzy[:limit]:
        out.extend(symbols.lookup(n))
    return {"query": name, "matches": [_def_dict(d) for d in out[:limit]], "fuzzy": True}


def symbol_usages(cfg: Config, name: str, limit: int = 30) -> dict:
    hits = find_usages(name, cfg, limit=limit)
    return {"symbol": name, "usages": hits}


def _def_dict(d) -> dict:
    return {
        "name": d.name,
        "kind": d.kind,
        "path": d.path,
        "module": d.module,
        "start_line": d.start_line,
        "end_line": d.end_line,
    }


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read a slice of a source file from the indexed repository, by path "
                f"relative to the repo root. Returns at most {MAX_READ_LINES} lines."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "e.g. src/core/RayTracer.cpp"},
                    "start_line": {"type": "integer", "default": 1},
                    "end_line": {"type": "integer", "description": "optional"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List the immediate contents of a directory in the indexed repo.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "e.g. src/core (empty for repo root)"}
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "Regex search across the indexed repo's source and test files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "a Python regex"},
                    "path_prefix": {"type": "string", "description": "optional, e.g. src/gpu"},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_symbols",
            "description": (
                "Look up a function/class/struct/method by name in the exact symbol index. "
                "Falls back to substring matching if there's no exact/bare match, e.g. "
                "'RayFile' finds 'RayFileData'."
            ),
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "symbol_usages",
            "description": "Find call sites / references to a symbol across the indexed repo.",
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        },
    },
]
