"""Exact symbol index: name -> definition(s), plus live grep-based usage lookup.

Definitions are built once at ingest time from the AST chunks (cheap — they're
already there). Usages are resolved on demand via a repo grep rather than
precomputed, since "every symbol x every file" is unbounded and most symbols
are never asked about. See PLAN.md §6 step 3.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from .chunks_cpp import Chunk
from .config import Config

_DEF_KINDS = {"function", "method", "class", "struct"}


@dataclass
class SymbolDef:
    name: str  # bare name, e.g. "Trace" or "RayTracer::Trace"
    kind: str
    path: str
    module: str
    start_line: int
    end_line: int


class SymbolIndex:
    def __init__(self) -> None:
        # exact symbol string -> definitions (there can be overloads/multiple TUs)
        self._by_full_name: dict[str, list[SymbolDef]] = {}
        # bare method/function name (no "Class::" prefix) -> full names that end with it
        self._bare_to_full: dict[str, set[str]] = {}

    @classmethod
    def build_from_chunks(cls, chunks: list[Chunk]) -> "SymbolIndex":
        idx = cls()
        seen: set[tuple] = set()
        for c in chunks:
            if c.kind not in _DEF_KINDS or not c.symbol:
                continue
            if c.part != 1:
                continue  # definition site is the first part of a split unit
            key = (c.symbol, c.path, c.start_line)
            if key in seen:
                continue
            seen.add(key)
            d = SymbolDef(
                name=c.symbol,
                kind=c.kind,
                path=c.path,
                module=c.module,
                start_line=c.start_line,
                end_line=c.end_line,
            )
            idx._by_full_name.setdefault(c.symbol, []).append(d)
            bare = c.symbol.rsplit("::", 1)[-1]
            idx._bare_to_full.setdefault(bare, set()).add(c.symbol)
        return idx

    def lookup(self, name: str) -> list[SymbolDef]:
        """Exact match first, then bare-name resolution (e.g. 'Trace' -> 'TraceScene::Trace')."""
        if name in self._by_full_name:
            return self._by_full_name[name]
        out: list[SymbolDef] = []
        for full in sorted(self._bare_to_full.get(name, ())):
            out.extend(self._by_full_name[full])
        return out

    def all_names(self) -> list[str]:
        return sorted(self._by_full_name.keys())

    def full_names_for_bare(self, bare: str) -> list[str]:
        """Every indexed full name ('Class::method' or free function) ending in `bare`."""
        return sorted(self._bare_to_full.get(bare, ()))

    def definitions(self, full_name: str) -> list[SymbolDef]:
        return self._by_full_name.get(full_name, [])

    def defs_in_path(self, path: str) -> list[SymbolDef]:
        """Every definition in a file, ordered by start line (built lazily, then cached)."""
        if not hasattr(self, "_by_path"):
            by_path: dict[str, list[SymbolDef]] = {}
            for defs in self._by_full_name.values():
                for d in defs:
                    by_path.setdefault(d.path, []).append(d)
            for v in by_path.values():
                v.sort(key=lambda d: (d.start_line, d.end_line))
            self._by_path = by_path
        return self._by_path.get(path, [])

    def resolves(self, name: str) -> bool:
        return bool(self.lookup(name))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            name: [asdict(d) for d in defs] for name, defs in self._by_full_name.items()
        }
        path.write_text(json.dumps(data, indent=0), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "SymbolIndex":
        idx = cls()
        data = json.loads(path.read_text(encoding="utf-8"))
        for name, defs in data.items():
            parsed = [SymbolDef(**d) for d in defs]
            idx._by_full_name[name] = parsed
            bare = name.rsplit("::", 1)[-1]
            idx._bare_to_full.setdefault(bare, set()).add(name)
        return idx


_IDENT_BOUNDARY = r"(?<![A-Za-z0-9_])"
_IDENT_BOUNDARY_END = r"(?![A-Za-z0-9_])"


def find_usages(
    symbol: str,
    cfg: Config,
    limit: int = 30,
    exclude_def_lines: set[tuple[str, int]] | None = None,
) -> list[dict]:
    """Grep the indexed repo tree for word-boundary references to `symbol`.

    `symbol` may be bare ("Trace") or qualified ("TraceScene::Trace") — qualified
    names are matched literally; bare names match the identifier anywhere,
    including as part of a qualified call (`obj.Trace(...)`, `TraceScene::Trace`).
    """
    bare = symbol.rsplit("::", 1)[-1]
    pattern = re.compile(_IDENT_BOUNDARY + re.escape(bare) + _IDENT_BOUNDARY_END)
    root = cfg.repo.root
    hits: list[dict] = []
    exclude = exclude_def_lines or set()

    exts = cfg.repo.cpp_extensions | cfg.repo.text_extensions
    for inc in cfg.repo.include_dirs:
        base = root / inc
        if not base.exists():
            continue
        for p in base.rglob("*"):
            if len(hits) >= limit:
                return hits
            if not p.is_file() or p.suffix not in exts:
                continue
            if any(part in cfg.repo.ignore_dirs for part in p.parts):
                continue
            rel = p.relative_to(root).as_posix()
            try:
                text = p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for i, line in enumerate(text.split("\n"), start=1):
                if (rel, i) in exclude:
                    continue
                if pattern.search(line):
                    hits.append({"path": rel, "line": i, "text": line.strip()[:200]})
                    if len(hits) >= limit:
                        break
    return hits
