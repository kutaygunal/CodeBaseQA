"""AST-aware C++ chunking via tree-sitter-cpp, with size rules and a CUDA fallback.

See PLAN.md §2.1 for the sizing rules and §5.2 for why tree-sitter over libclang.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import tree_sitter_cpp as tscpp
from tree_sitter import Language, Node, Parser

from .config import ChunkingConfig

_LANGUAGE = Language(tscpp.language())


@dataclass
class Chunk:
    chunk_id: str
    path: str  # POSIX-style, relative to repo root
    module: str  # top-level dir under the indexed root, e.g. "core", "gpu", "test"
    kind: str  # "function" | "method" | "class" | "struct" | "block" | "text"
    symbol: str | None
    start_line: int  # 1-indexed, inclusive
    end_line: int  # 1-indexed, inclusive
    text: str
    part: int = 1
    parts_total: int = 1
    language: str = "cpp"


def _make_parser() -> Parser:
    return Parser(_LANGUAGE)


def _token_estimate(text: str) -> int:
    """Rough token count — whitespace-ish words * 1.3 (config.py comment)."""
    words = len(text.split())
    return max(1, round(words * 1.3))


def _split_to_size(
    lines: list[str], start_line: int, cfg: ChunkingConfig
) -> list[tuple[str, int, int]]:
    """Split a unit's lines into overlapping windows honoring target/max tokens.

    Returns list of (text, start_line, end_line), all 1-indexed inclusive.
    Single-element result if the unit already fits under max_tokens.
    """
    full_text = "\n".join(lines)
    total_tokens = _token_estimate(full_text)
    if total_tokens <= cfg.max_tokens or len(lines) <= 1:
        return [(full_text, start_line, start_line + len(lines) - 1)]

    tokens_per_line = max(total_tokens / max(1, len(lines)), 0.01)
    window_lines = max(1, round(cfg.target_tokens / tokens_per_line))
    step = max(1, round(window_lines * (1 - cfg.overlap_ratio)))

    windows: list[tuple[str, int, int]] = []
    i = 0
    n = len(lines)
    while i < n:
        j = min(n, i + window_lines)
        chunk_lines = lines[i:j]
        windows.append(("\n".join(chunk_lines), start_line + i, start_line + j - 1))
        if j >= n:
            break
        i += step
    return windows


def _find_function_declarator(node: Node) -> Node | None:
    if node.type == "function_declarator":
        return node
    for child in node.children:
        found = _find_function_declarator(child)
        if found is not None:
            return found
    return None


def _declarator_name(declarator: Node) -> str | None:
    for child in declarator.children:
        if child.type in (
            "identifier",
            "field_identifier",
            "qualified_identifier",
            "destructor_name",
            "operator_name",
        ):
            return child.text.decode("utf-8", errors="replace")
    return None


def _enclosing_class_name(node: Node) -> str | None:
    parent = node.parent
    while parent is not None:
        if parent.type in ("class_specifier", "struct_specifier"):
            for child in parent.children:
                if child.type == "type_identifier":
                    return child.text.decode("utf-8", errors="replace")
            return None
        parent = parent.parent
    return None


def _collect_nodes(root: Node, types: set[str]) -> list[Node]:
    out: list[Node] = []

    def walk(n: Node):
        if n.type in types:
            out.append(n)
        for c in n.children:
            walk(c)

    walk(root)
    return out


def _is_top_level_class(node: Node) -> bool:
    """True if this class/struct isn't nested inside another class/struct."""
    parent = node.parent
    while parent is not None:
        if parent.type in ("class_specifier", "struct_specifier"):
            return False
        parent = parent.parent
    return True


def _collapse_function_bodies(source: bytes, node: Node) -> str:
    """Render a class/struct node's text with nested method bodies collapsed.

    Keeps field declarations and method signatures visible while dropping
    large inline bodies, so a class chunk stays a cheap "overview" and full
    method bodies live in their own function/method chunks (no duplication
    bloat on files like RayTracer.cpp).
    """
    func_defs = _collect_nodes(node, {"function_definition"})
    replacements: list[tuple[int, int]] = []
    for fd in func_defs:
        for child in fd.children:
            if child.type == "compound_statement":
                replacements.append((child.start_byte, child.end_byte))
                break
    # Apply from the end backwards so earlier byte offsets stay valid.
    replacements.sort(key=lambda r: r[0], reverse=True)
    buf = bytearray(source[node.start_byte : node.end_byte])
    base = node.start_byte
    placeholder = b"{ /* ... */ }"
    for start, end in replacements:
        s, e = start - base, end - base
        if 0 <= s <= e <= len(buf):
            buf[s:e] = placeholder
    return buf.decode("utf-8", errors="replace")


def _brace_depth0_fallback(source_text: str) -> list[tuple[str, str | None, int, int]]:
    """Heuristic splitter for files tree-sitter can't find functions in (CUDA).

    Returns list of (text, symbol_or_none, start_line, end_line), 1-indexed.
    """
    lines = source_text.split("\n")
    blocks: list[tuple[str, str | None, int, int]] = []
    depth = 0
    block_start = None
    preamble_start = 0
    sig_re = re.compile(r"[)\]]\s*(const\s*)?\{?\s*$")

    def flush_preamble(end_idx: int):
        if end_idx > preamble_start:
            text = "\n".join(lines[preamble_start:end_idx])
            if text.strip():
                blocks.append((text, None, preamble_start + 1, end_idx))

    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        opens = line.count("{")
        closes = line.count("}")
        if depth == 0 and opens > 0:
            flush_preamble(i)
            block_start = i
        depth += opens - closes
        if block_start is not None and depth <= 0:
            end = i
            text = "\n".join(lines[block_start : end + 1])
            # try to pull a symbol-ish name from the signature line(s) above the brace
            sym = None
            m = re.search(r"(\w+)\s*\([^;{]*\)\s*\{?\s*$", lines[block_start])
            if m:
                sym = m.group(1)
            blocks.append((text, sym, block_start + 1, end + 1))
            preamble_start = end + 1
            block_start = None
            depth = 0
        i += 1
    flush_preamble(n)
    if not blocks:
        # Truly nothing brace-shaped (rare) — index the whole file as one block.
        blocks.append((source_text, None, 1, n))
    return blocks


def chunk_cpp_file(path: Path, module: str, rel_path: str, cfg: ChunkingConfig) -> list[Chunk]:
    """Chunk one C++/CUDA source file into function/method/class/block units."""
    source_bytes = path.read_bytes()
    parser = _make_parser()
    tree = parser.parse(source_bytes)
    root = tree.root_node

    func_nodes = _collect_nodes(root, {"function_definition"})
    class_nodes = [
        n for n in _collect_nodes(root, {"class_specifier", "struct_specifier"}) if _is_top_level_class(n)
    ]

    chunks: list[Chunk] = []

    if not func_nodes and not class_nodes:
        # CUDA fallback — tree-sitter-cpp found no named units at all.
        source_text = source_bytes.decode("utf-8", errors="replace")
        for text, sym, start, end in _brace_depth0_fallback(source_text):
            lines = text.split("\n")
            for sub_text, s_line, e_line in _split_to_size(lines, start, cfg):
                chunks.append(
                    Chunk(
                        chunk_id="",
                        path=rel_path,
                        module=module,
                        kind="block",
                        symbol=sym,
                        start_line=s_line,
                        end_line=e_line,
                        text=sub_text,
                        language="cuda" if path.suffix in (".cu", ".cuh") else "cpp",
                    )
                )
        _finalize_ids(chunks)
        return chunks

    for fn in func_nodes:
        declarator = _find_function_declarator(fn)
        name = _declarator_name(declarator) if declarator else None
        class_name = _enclosing_class_name(fn)
        if class_name and name:
            symbol = f"{class_name}::{name}"
            kind = "method"
        elif name:
            symbol = name
            kind = "function"
        else:
            symbol = None
            kind = "function"
        text = fn.text.decode("utf-8", errors="replace")
        start_line = fn.start_point[0] + 1
        lines = text.split("\n")
        for sub_text, s_line, e_line in _split_to_size(lines, start_line, cfg):
            chunks.append(
                Chunk(
                    chunk_id="",
                    path=rel_path,
                    module=module,
                    kind=kind,
                    symbol=symbol,
                    start_line=s_line,
                    end_line=e_line,
                    text=sub_text,
                )
            )

    for cls in class_nodes:
        name = None
        for child in cls.children:
            if child.type == "type_identifier":
                name = child.text.decode("utf-8", errors="replace")
                break
        text = _collapse_function_bodies(source_bytes, cls)
        start_line = cls.start_point[0] + 1
        lines = text.split("\n")
        for sub_text, s_line, e_line in _split_to_size(lines, start_line, cfg):
            chunks.append(
                Chunk(
                    chunk_id="",
                    path=rel_path,
                    module=module,
                    kind=cls.type.replace("_specifier", ""),  # "class" | "struct"
                    symbol=name,
                    start_line=s_line,
                    end_line=e_line,
                    text=sub_text,
                )
            )

    if not chunks:
        # Parsed fine but found nothing (e.g. a pure-declarations header) —
        # still must not skip the file per PLAN.md §2.1.
        source_text = source_bytes.decode("utf-8", errors="replace")
        lines = source_text.split("\n")
        for sub_text, s_line, e_line in _split_to_size(lines, 1, cfg):
            chunks.append(
                Chunk(
                    chunk_id="",
                    path=rel_path,
                    module=module,
                    kind="text",
                    symbol=None,
                    start_line=s_line,
                    end_line=e_line,
                    text=sub_text,
                )
            )

    _finalize_ids(chunks)
    return chunks


def chunk_text_file(path: Path, module: str, rel_path: str, cfg: ChunkingConfig) -> list[Chunk]:
    """Plain-text chunking for non-C++ indexed files (CMake, .inc test suites, etc.)."""
    source_text = path.read_text(encoding="utf-8", errors="replace")
    lines = source_text.split("\n")
    chunks: list[Chunk] = []
    for sub_text, s_line, e_line in _split_to_size(lines, 1, cfg):
        chunks.append(
            Chunk(
                chunk_id="",
                path=rel_path,
                module=module,
                kind="text",
                symbol=None,
                start_line=s_line,
                end_line=e_line,
                text=sub_text,
                language="text",
            )
        )
    _finalize_ids(chunks)
    return chunks


def _finalize_ids(chunks: list[Chunk]) -> None:
    """Assign chunk_id + part/parts_total.

    _split_to_size emits parts for one unit contiguously and in order, so runs
    are detected by (path, kind, symbol) staying the same AND line ranges
    being adjacent/overlapping (true for windowed splits, false across units).
    """
    i = 0
    n = len(chunks)
    while i < n:
        j = i + 1
        while (
            j < n
            and chunks[j].path == chunks[i].path
            and chunks[j].kind == chunks[i].kind
            and chunks[j].symbol == chunks[i].symbol
            and chunks[j].start_line <= chunks[j - 1].end_line + 1
        ):
            j += 1
        total = j - i
        for part_idx, k in enumerate(range(i, j), start=1):
            chunks[k].part = part_idx
            chunks[k].parts_total = total
            sym_slug = (chunks[k].symbol or "block").replace("/", "_").replace(" ", "_")
            chunks[k].chunk_id = (
                f"{chunks[k].path}::{sym_slug}::{chunks[k].start_line}-{chunks[k].end_line}"
                f"::p{part_idx}of{total}"
            )
        i = j
