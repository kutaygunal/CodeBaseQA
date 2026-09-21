"""Walk the target repo, chunk it, embed it, and persist all indexes.

See PLAN.md §6 step 1.
"""
from __future__ import annotations

import json
import pickle
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

from rank_bm25 import BM25Okapi

from .callgraph import CallGraph
from .chunks_cpp import Chunk, chunk_cpp_file, chunk_text_file
from .config import Config, load_config
from .store import embed_texts, get_chroma_collection
from .symbol_index import SymbolIndex


def _tokenize(text: str) -> list[str]:
    """Simple code-aware tokenizer for BM25: split on non-alnum, lowercase,
    and also keep camelCase/PascalCase/snake_case sub-tokens for recall."""
    import re

    tokens: list[str] = []
    for raw in re.split(r"[^A-Za-z0-9_]+", text):
        if not raw:
            continue
        tokens.append(raw.lower())
        # split identifier parts: RayTracer -> ray, tracer ; trace_scene -> trace, scene
        parts = re.findall(r"[A-Z]?[a-z0-9]+|[A-Z]+(?![a-z])", raw)
        if len(parts) > 1:
            tokens.extend(p.lower() for p in parts)
    return tokens


def _iter_repo_files(cfg: Config):
    root = cfg.repo.root
    for inc in cfg.repo.include_dirs:
        base = root / inc
        if not base.exists():
            continue
        for p in sorted(base.rglob("*")):
            if not p.is_file():
                continue
            if any(part in cfg.repo.ignore_dirs for part in p.parts):
                continue
            yield p


def _module_for(root: Path, p: Path) -> str:
    """Top-level module tag: src/<dir>/... -> <dir>; src/main.cpp -> "core";
    anything under test/ -> "test" (never the filename — see PLAN.md §9)."""
    parts = p.relative_to(root).parts
    if parts[0] == "src":
        return parts[1] if len(parts) > 2 else "core"
    if parts[0] == "test":
        return "test"
    return parts[0]


def _git_commit(root: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, timeout=10
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def chunk_repo(cfg: Config) -> tuple[list[Chunk], list[str]]:
    """Chunk every indexed file. Returns (chunks, skipped_paths) — skipped should be empty."""
    chunks: list[Chunk] = []
    skipped: list[str] = []
    for p in _iter_repo_files(cfg):
        ext = p.suffix
        rel = p.relative_to(cfg.repo.root).as_posix()
        module = _module_for(cfg.repo.root, p)
        try:
            if ext in cfg.repo.cpp_extensions:
                chunks.extend(chunk_cpp_file(p, module, rel, cfg.chunking))
            elif ext in cfg.repo.text_extensions:
                chunks.extend(chunk_text_file(p, module, rel, cfg.chunking))
        except Exception as e:  # noqa: BLE001 — must not abort the whole run
            skipped.append(f"{rel}: {e}")
    return chunks, skipped


def build_index(cfg: Config | None = None, progress=print, contextual: bool | None = None) -> dict:
    """Full (re)build: chunk -> embed -> Chroma -> BM25 -> symbol index -> call graph -> state file.

    `contextual` (default: config `ingest.contextual.enabled`) prepends an LLM-written
    context sentence to each chunk before embedding/BM25 — see cqa/contextual.py."""
    cfg = cfg or load_config()
    t0 = time.time()
    use_contextual = cfg.ingest.contextual.enabled if contextual is None else contextual

    progress(f"Walking {cfg.repo.root} ...")
    chunks, skipped = chunk_repo(cfg)
    progress(f"  {len(chunks)} chunks from {len({c.path for c in chunks})} files"
              f" ({len(skipped)} skipped)")
    if skipped:
        for s in skipped[:20]:
            progress(f"  SKIPPED: {s}")

    progress("Building symbol index ...")
    symbols = SymbolIndex.build_from_chunks(chunks)
    symbols.save(cfg.storage.index_state_file.parent / "symbols.json")
    progress(f"  {len(symbols.all_names())} unique symbols")

    progress("Building call graph ...")
    graph = CallGraph.build(chunks, symbols, cfg.repo.root)
    graph.save(cfg.storage.index_state_file.parent / "graph.json")
    progress(
        f"  {sum(len(v) for v in graph.callees.values())} call edges, "
        f"{sum(len(v) for v in graph.includes.values())} include edges, {len(graph.pairs) // 2} header/impl pairs"
    )

    texts = [c.text for c in chunks]  # displayed to the LLM / user — never modified
    contexts: dict[str, str] = {}
    if use_contextual:
        progress("Generating contextual descriptions ...")
        from .contextual import generate_contexts

        contexts = generate_contexts(chunks, cfg, progress)
        progress(f"  {sum(1 for v in contexts.values() if v)}/{len(chunks)} chunks have context")
    # What is actually embedded and keyword-indexed: context (if any) + chunk text.
    index_texts = [
        (contexts[c.chunk_id] + "\n\n" + c.text if contexts.get(c.chunk_id) else c.text) for c in chunks
    ]

    progress(f"Embedding {len(chunks)} chunks via '{cfg.models.embed_model}' ...")
    embeddings = embed_texts(index_texts, cfg.models.embed_model)
    progress("  done")

    progress("Writing Chroma collection ...")
    coll = get_chroma_collection(cfg, reset=True)
    ids = [c.chunk_id for c in chunks]
    metadatas = [
        {
            "path": c.path,
            "module": c.module,
            "kind": c.kind,
            "symbol": c.symbol or "",
            "start_line": c.start_line,
            "end_line": c.end_line,
            "part": c.part,
            "parts_total": c.parts_total,
            "language": c.language,
        }
        for c in chunks
    ]
    batch = 512
    for i in range(0, len(chunks), batch):
        coll.upsert(
            ids=ids[i : i + batch],
            embeddings=embeddings[i : i + batch],
            documents=texts[i : i + batch],
            metadatas=metadatas[i : i + batch],
        )
    progress(f"  {coll.count()} vectors in Chroma")

    progress("Building BM25 index ...")
    tokenized = [_tokenize(t) for t in index_texts]
    bm25 = BM25Okapi(tokenized)
    bm25_path = cfg.storage.index_state_file.parent / "bm25.pkl"
    bm25_path.parent.mkdir(parents=True, exist_ok=True)
    with open(bm25_path, "wb") as f:
        pickle.dump({"bm25": bm25, "ids": ids, "metadatas": metadatas, "texts": texts}, f)
    progress("  done")

    state = {
        "built_at": time.time(),
        "repo_root": str(cfg.repo.root),
        "git_commit": _git_commit(cfg.repo.root),
        "num_files": len({c.path for c in chunks}),
        "num_chunks": len(chunks),
        "num_symbols": len(symbols.all_names()),
        "num_call_edges": sum(len(v) for v in graph.callees.values()),
        "contextual": use_contextual,
        "skipped": skipped,
        "duration_s": round(time.time() - t0, 2),
    }
    cfg.storage.index_state_file.parent.mkdir(parents=True, exist_ok=True)
    cfg.storage.index_state_file.write_text(json.dumps(state, indent=2), encoding="utf-8")

    per_module: dict[str, int] = {}
    for c in chunks:
        per_module[c.module] = per_module.get(c.module, 0) + 1
    state["per_module"] = per_module
    progress(f"Done in {state['duration_s']}s. Per-module chunk counts: {per_module}")
    return state
