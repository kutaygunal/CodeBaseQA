"""Cross-encoder reranking (fastembed / ONNX, fully local).

A cross-encoder reads (query, chunk) *together*, so it separates "mentions the same
words" from "actually answers the question" far better than the bi-encoder + BM25 first
stage. It is too slow to run over the whole corpus, so it only re-scores the ~30
candidates the RRF merge produced. Loaded lazily; if fastembed or the model is
unavailable the retriever silently keeps the RRF order.
"""
from __future__ import annotations

import os
import threading

from .config import PROJECT_ROOT

_lock = threading.Lock()
_cache: dict[str, "Reranker | None"] = {}
_warned: set[str] = set()

MAX_DOC_CHARS = 1600  # ~400 tokens; bench: same recall as 2400 at ~20% less time


class Reranker:
    def __init__(self, model_name: str):
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self.model_name = model_name
        # Keep the model in the project (gitignored) — fastembed's default is the OS temp
        # dir, which Windows may clean, forcing a re-download.
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        self._enc = TextCrossEncoder(model_name=model_name, cache_dir=str(PROJECT_ROOT / ".cache" / "fastembed"))

    def score(self, query: str, docs: list[str]) -> list[float]:
        return [float(s) for s in self._enc.rerank(query, [d[:MAX_DOC_CHARS] for d in docs])]


def get_reranker(model_name: str) -> Reranker | None:
    """Cached reranker, or None (with a one-time warning) if it can't be loaded."""
    with _lock:
        if model_name in _cache:
            return _cache[model_name]
        try:
            _cache[model_name] = Reranker(model_name)
        except Exception as e:  # noqa: BLE001 — missing dep / download failure must not break search
            if model_name not in _warned:
                _warned.add(model_name)
                print(f"[cqa] reranker '{model_name}' unavailable ({type(e).__name__}: {e}); using RRF order")
            _cache[model_name] = None
        return _cache[model_name]
