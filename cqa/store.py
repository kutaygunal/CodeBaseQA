"""Shared storage helpers: Ollama embeddings + the Chroma client."""
from __future__ import annotations

import chromadb
import ollama
from chromadb.config import Settings

from .config import Config


def embed_texts(texts: list[str], model: str, batch_size: int = 64) -> list[list[float]]:
    """Embed a list of texts via the local Ollama daemon, batched."""
    out: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = ollama.embed(model=model, input=batch)
        out.extend(resp["embeddings"])
    return out


def get_chroma_collection(cfg: Config, reset: bool = False):
    client = chromadb.PersistentClient(
        path=str(cfg.storage.chroma_dir), settings=Settings(anonymized_telemetry=False)
    )
    if reset:
        try:
            client.delete_collection(cfg.storage.collection_name)
        except Exception:
            pass
    return client.get_or_create_collection(
        name=cfg.storage.collection_name, metadata={"hnsw:space": "cosine"}
    )
