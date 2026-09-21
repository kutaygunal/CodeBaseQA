"""Contextual chunk embeddings ("contextual retrieval").

A function chunk like `void bandWeights(...) {...}` embeds badly on its own: the embedding
doesn't know it belongs to the spectral-sampling code of a ray tracer. At index time we ask a
small LLM to write one or two sentences situating each chunk in its file/module, and embed +
BM25-index `context + chunk` instead of the bare chunk. The displayed text is unchanged.

Costs one LLM call per chunk, so results are cached on disk keyed by a hash of
(model, path, symbol, chunk text) — rebuilding an unchanged index re-uses every context.
"""
from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

from .chunks_cpp import Chunk
from .config import Config
from .providers import get_provider

PROMPT = """You are helping build a code search index for the {project} C++ codebase.

File: {path}
Module: {module}
Other symbols defined in this file: {siblings}

Here is one chunk of that file ({kind} {symbol}):
<chunk>
{text}
</chunk>

Write 1-2 plain-English sentences that situate this chunk within the file and the wider
system: what subsystem it belongs to, what it is for, and the key domain concepts it deals
with. Use words a developer would type when searching for it. Output ONLY those sentences."""

MAX_CHUNK_CHARS = 3500


def _key(model: str, c: Chunk) -> str:
    h = hashlib.sha256()
    for part in (model, c.path, c.symbol or "", c.text):
        h.update(part.encode("utf-8", errors="replace"))
        h.update(b"\x00")
    return h.hexdigest()


def load_cache(path: Path) -> dict[str, str]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def generate_contexts(
    chunks: list[Chunk],
    cfg: Config,
    progress: Callable[[str], None] = print,
    chat: Callable[[str], str] | None = None,
) -> dict[str, str]:
    """Return {chunk_id: context sentence(s)}. `chat` overrides the LLM call (for tests)."""
    ccfg = cfg.ingest.contextual
    model = ccfg.model or cfg.models.chat_model
    cache_path = cfg.storage.index_state_file.parent / "context_cache.json"
    cache = load_cache(cache_path)

    siblings: dict[str, list[str]] = {}
    for c in chunks:
        if c.symbol and c.part == 1:
            siblings.setdefault(c.path, []).append(c.symbol)

    if chat is None:
        provider = get_provider("ollama")

        def chat(prompt: str) -> str:  # noqa: F811
            return provider.chat(model, [{"role": "user", "content": prompt}], []).content

    project = cfg.repo.root.name
    todo = [c for c in chunks if _key(model, c) not in cache]
    progress(f"  contextual: {len(chunks) - len(todo)} cached, {len(todo)} to generate with '{model}'")

    def work(c: Chunk) -> tuple[str, str]:
        sibs = ", ".join(list(dict.fromkeys(siblings.get(c.path, [])))[:25])
        prompt = PROMPT.format(
            project=project, path=c.path, module=c.module, siblings=sibs or "(none)",
            kind=c.kind, symbol=c.symbol or "(unnamed)", text=c.text[:MAX_CHUNK_CHARS],
        )
        for attempt in range(2):
            try:
                out = (chat(prompt) or "").strip()
                if out:
                    return _key(model, c), " ".join(out.split())[:600]
            except Exception:  # noqa: BLE001 — one flaky call must not abort the whole index
                continue
        return _key(model, c), ""

    done = 0
    with ThreadPoolExecutor(max_workers=max(1, ccfg.concurrency)) as pool:
        for key, ctx in pool.map(work, todo):
            if ctx:
                cache[key] = ctx
            done += 1
            if done % 100 == 0:
                progress(f"    {done}/{len(todo)}")
                cache_path.write_text(json.dumps(cache), encoding="utf-8")
    if todo:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache), encoding="utf-8")

    return {c.chunk_id: cache.get(_key(model, c), "") for c in chunks}
