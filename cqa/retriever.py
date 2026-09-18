"""Hybrid retrieval: BM25 (keyword) + dense embeddings (Chroma), RRF merge.

See PLAN.md §6 step 2 and §3.1 for the "confidence" hard-gate this feeds.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass

from .config import Config, load_config
from .ingest import _tokenize
from .store import embed_texts, get_chroma_collection


@dataclass
class Hit:
    chunk_id: str
    path: str
    module: str
    kind: str
    symbol: str
    start_line: int
    end_line: int
    text: str
    rrf_score: float
    dense_rank: int | None = None
    bm25_rank: int | None = None


class Retriever:
    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or load_config()
        self._coll = get_chroma_collection(self.cfg)
        bm25_path = self.cfg.storage.index_state_file.parent / "bm25.pkl"
        with open(bm25_path, "rb") as f:
            data = pickle.load(f)
        self._bm25 = data["bm25"]
        self._bm25_ids = data["ids"]
        self._bm25_meta = data["metadatas"]
        self._bm25_texts = data["texts"]
        self._id_to_row = {cid: i for i, cid in enumerate(self._bm25_ids)}
        self.modules = sorted({m["module"] for m in self._bm25_meta})

    def _dense_search(self, query: str, k: int, where: dict | None) -> list[str]:
        qvec = embed_texts([query], self.cfg.models.embed_model)[0]
        res = self._coll.query(
            query_embeddings=[qvec],
            n_results=k,
            where=where,
        )
        return res["ids"][0] if res["ids"] else []

    def _bm25_search(self, query: str, k: int, where: dict | None) -> list[str]:
        tokens = _tokenize(query)
        scores = self._bm25.get_scores(tokens)
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        out: list[str] = []
        for i in order:
            meta = self._bm25_meta[i]
            if where and not _matches_where(meta, where):
                continue
            out.append(self._bm25_ids[i])
            if len(out) >= k:
                break
        return out

    def search(
        self,
        query: str,
        top_k: int | None = None,
        module: str | None = None,
        path_prefix: str | None = None,
    ) -> list[Hit]:
        cfg = self.cfg
        top_k = top_k or cfg.retrieval.top_k
        fetch_k = max(top_k * 3, 15)
        where = {"module": module} if module else None

        dense_ids = self._dense_search(query, fetch_k, where)
        bm25_ids = self._bm25_search(query, fetch_k, where)

        if path_prefix:
            dense_ids = [i for i in dense_ids if self._path_of(i).startswith(path_prefix)]
            bm25_ids = [i for i in bm25_ids if self._path_of(i).startswith(path_prefix)]

        rrf_k = cfg.retrieval.rrf_k
        scores: dict[str, float] = {}
        dense_rank: dict[str, int] = {}
        bm25_rank: dict[str, int] = {}
        for rank, cid in enumerate(dense_ids, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + rank)
            dense_rank[cid] = rank
        for rank, cid in enumerate(bm25_ids, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + rank)
            bm25_rank[cid] = rank

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]

        hits: list[Hit] = []
        for cid, score in ranked:
            row = self._id_to_row.get(cid)
            if row is None:
                continue
            meta = self._bm25_meta[row]
            hits.append(
                Hit(
                    chunk_id=cid,
                    path=meta["path"],
                    module=meta["module"],
                    kind=meta["kind"],
                    symbol=meta["symbol"],
                    start_line=meta["start_line"],
                    end_line=meta["end_line"],
                    text=self._bm25_texts[row],
                    rrf_score=score,
                    dense_rank=dense_rank.get(cid),
                    bm25_rank=bm25_rank.get(cid),
                )
            )
        return hits

    def _path_of(self, chunk_id: str) -> str:
        row = self._id_to_row.get(chunk_id)
        return self._bm25_meta[row]["path"] if row is not None else ""

    def is_thin(self, hits: list[Hit], symbol_resolved: bool | None = None) -> bool:
        """PLAN.md §3.1 hard gate."""
        cfg = self.cfg.retrieval
        if len(hits) < cfg.min_hits:
            return True
        if not hits or hits[0].rrf_score < cfg.rrf_min_score:
            return True
        if symbol_resolved is False:
            return True
        return False


def _matches_where(meta: dict, where: dict) -> bool:
    return all(meta.get(k) == v for k, v in where.items())
