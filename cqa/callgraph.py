"""Static, name-based call graph + include graph, built at ingest time.

Edges come from tree-sitter `call_expression` nodes inside each function/method chunk,
resolved *by name* against the exact symbol index (no type resolution, so it is
approximate: overloads and same-named methods on different classes can be conflated).
Ambiguity is limited by: (a) skipping very common names, (b) skipping names with more
than MAX_CANDIDATES definitions, and (c) preferring candidates in the caller's own class /
file when a name has several. Consumers (retrieval expansion, the `call_graph` tool,
review, tours) treat the graph as a strong hint, never as ground truth.
"""
from __future__ import annotations

import json
import re
from collections import deque
from pathlib import Path

from .chunks_cpp import Chunk, _make_parser
from .symbol_index import SymbolIndex

MAX_CANDIDATES = 3
MIN_NAME_LEN = 4
_CALLABLE_KINDS = {"function", "method"}
# Names that collide with the standard library / ubiquitous member names — a call to
# `v.size()` must not become an edge to some project's own `size()`.
_COMMON_NAMES = {
    "size", "empty", "begin", "end", "clear", "data", "find", "count", "insert", "erase",
    "push_back", "emplace_back", "pop_back", "front", "back", "resize", "reserve", "swap",
    "reset", "get", "set", "at", "min", "max", "abs", "sqrt", "load", "save", "read",
    "write", "open", "close", "name", "value", "type", "next",
    "make_pair", "make_shared", "make_unique", "move", "forward", "sort", "fill", "copy",
    "assert", "printf",
}
_INCLUDE_RE = re.compile(r'^\s*#\s*include\s+"([^"]+)"', re.M)


def _call_name(fn_node) -> tuple[str | None, str] | None:
    """(qualifier, bare name) from a call_expression's `function` child.
    `A::B::f(...)` -> ("A::B", "f"); `obj.f()` / `obj->f()` / `f()` -> (None, "f")."""
    t = fn_node.type
    txt = lambda n: n.text.decode("utf-8", "replace")  # noqa: E731
    if t == "identifier":
        return None, txt(fn_node)
    if t == "field_identifier":
        return None, txt(fn_node)
    if t == "qualified_identifier":
        scope = fn_node.child_by_field_name("scope")
        name = fn_node.child_by_field_name("name")
        if name is None:
            return None
        inner = _call_name(name)
        if inner is None:
            return None
        q_inner, nm = inner
        scope_txt = txt(scope) if scope is not None else None
        if scope_txt and q_inner:
            scope_txt = f"{scope_txt}::{q_inner}"
        return (scope_txt or q_inner), nm
    if t == "template_function":
        name = fn_node.child_by_field_name("name")
        return _call_name(name) if name is not None else None
    if t == "field_expression":
        field = fn_node.child_by_field_name("field")
        return (None, txt(field)) if field is not None else None
    return None


def _extract_call_names(text: str, parser) -> set[tuple[str | None, str]]:
    tree = parser.parse(text.encode("utf-8", errors="replace"))
    names: set[tuple[str | None, str]] = set()
    stack = [tree.root_node]
    while stack:
        n = stack.pop()
        if n.type == "call_expression":
            fn = n.child_by_field_name("function")
            if fn is not None:
                nm = _call_name(fn)
                if nm:
                    names.add(nm)
        stack.extend(n.children)
    return names


class CallGraph:
    def __init__(self) -> None:
        self.callees: dict[str, list[str]] = {}
        self.callers: dict[str, list[str]] = {}
        self.includes: dict[str, list[str]] = {}
        self.included_by: dict[str, list[str]] = {}
        self.pairs: dict[str, str] = {}  # header <-> impl, both directions

    # --- build -----------------------------------------------------------

    @classmethod
    def build(cls, chunks: list[Chunk], symbols: SymbolIndex, repo_root: Path) -> "CallGraph":
        g = cls()
        parser = _make_parser()

        # 1. union of call names per symbol (a split function has several chunks)
        names_by_symbol: dict[str, set[tuple[str | None, str]]] = {}
        for c in chunks:
            if c.kind in _CALLABLE_KINDS and c.symbol:
                names_by_symbol.setdefault(c.symbol, set()).update(_extract_call_names(c.text, parser))

        # 2. resolve names -> full symbols
        callees: dict[str, list[str]] = {}
        for caller, names in names_by_symbol.items():
            caller_class = caller.rsplit("::", 1)[0] if "::" in caller else None
            caller_defs = symbols.definitions(caller)
            caller_path = caller_defs[0].path if caller_defs else None
            resolved: list[str] = []
            for qual, nm in sorted(names, key=lambda t: (t[0] or "", t[1])):
                if len(nm) < MIN_NAME_LEN or nm.lower() in _COMMON_NAMES:
                    continue
                cands = [
                    c for c in symbols.full_names_for_bare(nm)
                    if c != caller and _is_callable(symbols, c) and not _is_constructor(c)
                ]
                if qual:
                    qualified = [c for c in cands if c.startswith(qual + "::") or c.rsplit("::", 1)[0].endswith("::" + qual)]
                    if qualified:
                        cands = qualified
                if not cands or len(cands) > MAX_CANDIDATES:
                    continue
                if len(cands) > 1:
                    same_class = [c for c in cands if caller_class and c.startswith(caller_class + "::")]
                    if same_class:
                        cands = same_class
                    elif caller_path:
                        same_file = [c for c in cands if any(d.path == caller_path for d in symbols.definitions(c))]
                        cands = same_file or cands
                for c in cands:
                    if c not in resolved:
                        resolved.append(c)
            if resolved:
                callees[caller] = resolved
        g.callees = callees
        for a, bs in callees.items():
            for b in bs:
                g.callers.setdefault(b, [])
                if a not in g.callers[b]:
                    g.callers[b].append(a)

        # 3. include graph (read each indexed file's #include "..." lines)
        indexed = sorted({c.path for c in chunks})
        by_base: dict[str, list[str]] = {}
        for p in indexed:
            by_base.setdefault(Path(p).name, []).append(p)
        for p in indexed:
            try:
                text = (repo_root / p).read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for inc in _INCLUDE_RE.findall(text):
                target = _resolve_include(p, inc, indexed, by_base)
                if target and target != p:
                    g.includes.setdefault(p, [])
                    if target not in g.includes[p]:
                        g.includes[p].append(target)
                        g.included_by.setdefault(target, []).append(p)

        # 4. header <-> implementation pairs (same dir + stem)
        stems: dict[tuple[str, str], list[str]] = {}
        for p in indexed:
            pp = Path(p)
            stems.setdefault((pp.parent.as_posix(), pp.stem), []).append(p)
        for group in stems.values():
            hdr = [p for p in group if Path(p).suffix in (".h", ".hpp", ".hxx", ".cuh")]
            impl = [p for p in group if Path(p).suffix in (".cpp", ".cc", ".cxx", ".cu")]
            if len(hdr) == 1 and len(impl) == 1:
                g.pairs[hdr[0]] = impl[0]
                g.pairs[impl[0]] = hdr[0]
        return g

    # --- persistence -----------------------------------------------------

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "callees": self.callees, "includes": self.includes, "pairs": self.pairs,
                },
                indent=0,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "CallGraph":
        data = json.loads(path.read_text(encoding="utf-8"))
        g = cls()
        g.callees = data.get("callees", {})
        g.includes = data.get("includes", {})
        g.pairs = data.get("pairs", {})
        for a, bs in g.callees.items():
            for b in bs:
                g.callers.setdefault(b, []).append(a)
        for a, bs in g.includes.items():
            for b in bs:
                g.included_by.setdefault(b, []).append(a)
        return g

    # --- queries ---------------------------------------------------------

    def neighbors_of(self, symbol: str, direction: str = "both") -> list[tuple[str, str]]:
        """[(relation, symbol)] one hop out; relation is 'callee' or 'caller'."""
        out: list[tuple[str, str]] = []
        if direction in ("callees", "both"):
            out += [("callee", s) for s in self.callees.get(symbol, [])]
        if direction in ("callers", "both"):
            out += [("caller", s) for s in self.callers.get(symbol, [])]
        return out

    def subgraph(self, root: str, direction: str = "both", depth: int = 1, max_nodes: int = 25) -> dict:
        """BFS from `root`. Returns {"nodes": [...], "edges": [[caller, callee], ...], "truncated": bool}."""
        depth = max(1, min(depth, 3))
        nodes: list[str] = [root]
        seen = {root}
        edges: list[list[str]] = []
        edge_set: set[tuple[str, str]] = set()
        truncated = False
        q: deque[tuple[str, int]] = deque([(root, 0)])
        while q:
            cur, d = q.popleft()
            if d >= depth:
                continue
            for rel, nb in self.neighbors_of(cur, direction):
                edge = (cur, nb) if rel == "callee" else (nb, cur)
                if edge not in edge_set:
                    if nb not in seen and len(nodes) >= max_nodes:
                        truncated = True
                        continue
                    edge_set.add(edge)
                    edges.append(list(edge))
                if nb not in seen:
                    seen.add(nb)
                    nodes.append(nb)
                    q.append((nb, d + 1))
        return {"nodes": nodes, "edges": edges, "truncated": truncated}


def _is_callable(symbols: SymbolIndex, full_name: str) -> bool:
    return any(d.kind in _CALLABLE_KINDS for d in symbols.definitions(full_name))


def _is_constructor(full_name: str) -> bool:
    parts = full_name.split("::")
    return len(parts) >= 2 and parts[-1] == parts[-2]


def _resolve_include(from_path: str, inc: str, indexed: list[str], by_base: dict[str, list[str]]) -> str | None:
    inc_norm = inc.replace("\\", "/")
    # relative to the including file's directory
    rel = (Path(from_path).parent / inc_norm).as_posix()
    parts: list[str] = []
    for seg in rel.split("/"):
        if seg == "..":
            if parts:
                parts.pop()
        elif seg and seg != ".":
            parts.append(seg)
    rel = "/".join(parts)
    if rel in indexed:
        return rel
    # relative to an indexed root like src/ or test/
    for p in indexed:
        if p.endswith("/" + inc_norm) or p == inc_norm:
            return p
    cands = by_base.get(Path(inc_norm).name, [])
    return cands[0] if len(cands) == 1 else None


def to_mermaid(root: str, sub: dict) -> str:
    """Mermaid `flowchart LR` source for a `subgraph()` result. Labels are quoted, so
    `::` and other punctuation in C++ names are safe."""
    ids = {n: f"n{i}" for i, n in enumerate(sub["nodes"])}
    lines = ["flowchart LR"]
    for n, nid in ids.items():
        label = n.replace('"', "'")
        lines.append(f'    {nid}["{label}"]')
    for a, b in sub["edges"]:
        if a in ids and b in ids:
            lines.append(f"    {ids[a]} --> {ids[b]}")
    if root in ids:
        lines.append(f"    style {ids[root]} stroke-width:3px")
    return "\n".join(lines)
