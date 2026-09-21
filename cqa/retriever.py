"""Hybrid retrieval: BM25 (keyword) + dense embeddings (Chroma), RRF merge, and optional
cross-encoder reranking, multi-query fusion and call-graph neighbor expansion.

See PLAN.md §6 step 2 and §3.1 for the "confidence" hard-gate this feeds.

Pipeline (each optional stage is off unless enabled in config or passed explicitly):
    queries -> [dense + BM25 per query] -> RRF fuse -> candidate pool (candidate_k)
            -> cross-encoder rerank -> top_k -> graph expansion (extra neighbor chunks)

The §3.1 confidence gate must not be perturbed by any of that reordering, so it is
computed from `gate_score`: the best RRF score, over the whole candidate pool, of the
*original question's own* dense+BM25 lists — the quantity `rrf_min_score` was calibrated on.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

from .callgraph import CallGraph
from .config import Config, load_config
from .ingest import _tokenize
from .rerank import get_reranker
from .store import embed_texts, get_chroma_collection

_DEF_KINDS = {"function", "method", "class", "struct"}


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
    # RRF score from the original question's own lists only (gate calibration, see module doc)
    primary_score: float = 0.0
    rerank_score: float | None = None
    # Set on graph-expansion hits, e.g. "callee of RayTracer::trace"; None for ranked hits.
    via: str | None = None


@dataclass
class SearchResult:
    hits: list[Hit]
    gate_score: float
    pool_size: int = 0
    reranked: bool = False
    expanded: int = 0
    queries: list[str] = field(default_factory=list)


class Retriever:
    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or load_config()
        self._coll = get_chroma_collection(self.cfg)
        index_dir = self.cfg.storage.index_state_file.parent
        with open(index_dir / "bm25.pkl", "rb") as f:
            data = pickle.load(f)
        self._bm25 = data["bm25"]
        self._bm25_ids = data["ids"]
        self._bm25_meta = data["metadatas"]
        self._bm25_texts = data["texts"]
        self._id_to_row = {cid: i for i, cid in enumerate(self._bm25_ids)}
        self.modules = sorted({m["module"] for m in self._bm25_meta})
        # symbol -> first row of its definition chunk (for graph expansion)
        self._symbol_row: dict[str, int] = {}
        for i, m in enumerate(self._bm25_meta):
            if m.get("symbol") and m.get("kind") in _DEF_KINDS and m.get("part", 1) == 1:
                self._symbol_row.setdefault(m["symbol"], i)
        graph_path = index_dir / "graph.json"
        self.graph: CallGraph | None = CallGraph.load(graph_path) if graph_path.exists() else None

    # --- first-stage searches -------------------------------------------

    def _dense_search(self, queries: list[str], k: int, where: dict | None) -> list[list[str]]:
        # Qwen3-Embedding is instruction-aware: a task instruction on the *query* side
        # (documents stay unprefixed) measurably improves retrieval. Leave empty to embed
        # queries exactly like documents (uniform nomic-style semantics).
        instr = (self.cfg.retrieval.embed_query_instruction or "").strip()
        if instr:
            queries = [f"{instr}\n{q}" if q else q for q in queries]
        qvecs = embed_texts(queries, self.cfg.models.embed_model)
        res = self._coll.query(query_embeddings=qvecs, n_results=k, where=where)
        return res["ids"] if res["ids"] else [[] for _ in queries]

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

    # --- public search ---------------------------------------------------

    def search(self, query: str, top_k: int | None = None, module: str | None = None,
               path_prefix: str | None = None, **kw) -> list[Hit]:
        """Back-compat wrapper: just the hits. See `search_ex` for the gate score etc."""
        return self.search_ex(query, top_k=top_k, module=module, path_prefix=path_prefix, **kw).hits

    def search_ex(
        self,
        query: str,
        top_k: int | None = None,
        module: str | None = None,
        path_prefix: str | None = None,
        extra_queries: list[str] | None = None,
        rerank: bool | None = None,
        expand: bool | None = None,
        rerank_fuse_weight: float | None = None,
    ) -> SearchResult:
        cfg = self.cfg
        rcfg = cfg.retrieval
        top_k = top_k or rcfg.top_k
        do_rerank = rcfg.rerank.enabled if rerank is None else rerank
        do_expand = rcfg.expand.enabled if expand is None else expand
        pool_k = max(rcfg.candidate_k, top_k) if do_rerank else top_k
        fetch_k = max(pool_k * 3, 15) if not do_rerank else max(rcfg.candidate_k * 2, 15)
        where = {"module": module} if module else None

        queries = [query] + [q for q in (extra_queries or []) if q and q.strip() and q != query]
        dense_lists = self._dense_search(queries, fetch_k, where)
        bm25_lists = [self._bm25_search(q, fetch_k, where) for q in queries]
        if path_prefix:
            dense_lists = [[i for i in l if self._path_of(i).startswith(path_prefix)] for l in dense_lists]
            bm25_lists = [[i for i in l if self._path_of(i).startswith(path_prefix)] for l in bm25_lists]

        rrf_k = rcfg.rrf_k
        scores: dict[str, float] = {}
        primary: dict[str, float] = {}
        dense_rank: dict[str, int] = {}
        bm25_rank: dict[str, int] = {}
        for qi in range(len(queries)):
            for rank, cid in enumerate(dense_lists[qi], start=1):
                w = 1.0 / (rrf_k + rank)
                scores[cid] = scores.get(cid, 0.0) + w
                if qi == 0:
                    primary[cid] = primary.get(cid, 0.0) + w
                    dense_rank[cid] = rank
            for rank, cid in enumerate(bm25_lists[qi], start=1):
                w = 1.0 / (rrf_k + rank)
                scores[cid] = scores.get(cid, 0.0) + w
                if qi == 0:
                    primary[cid] = primary.get(cid, 0.0) + w
                    bm25_rank[cid] = rank

        gate_score = max(primary.values(), default=0.0)
        pool = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:pool_k]

        hits: list[Hit] = []
        for cid, score in pool:
            row = self._id_to_row.get(cid)
            if row is None:
                continue
            hits.append(self._hit(cid, row, score, primary.get(cid, 0.0), dense_rank.get(cid), bm25_rank.get(cid)))

        reranked = False
        if do_rerank and len(hits) > 1:
            rr = get_reranker(rcfg.rerank.model)
            if rr is not None:
                try:
                    rs = rr.score(query, [f"{h.path} {h.symbol}\n{h.text}" for h in hits])
                    for h, s in zip(hits, rs):
                        h.rerank_score = s
                    w = rcfg.rerank.fuse_weight if rerank_fuse_weight is None else rerank_fuse_weight
                    if w > 0:
                        orig_rank = {h.chunk_id: i for i, h in enumerate(hits, 1)}  # pool is RRF-ordered
                        by_rerank = sorted(hits, key=lambda h: h.rerank_score, reverse=True)
                        fused = {
                            h.chunk_id: 1.0 / (rrf_k + r) + w / (rrf_k + orig_rank[h.chunk_id])
                            for r, h in enumerate(by_rerank, 1)
                        }
                        hits.sort(key=lambda h: fused[h.chunk_id], reverse=True)
                    else:
                        hits.sort(key=lambda h: h.rerank_score, reverse=True)
                    reranked = True
                except Exception as e:  # noqa: BLE001 — degrade to RRF order, never fail the question
                    print(f"[cqa] rerank failed ({type(e).__name__}: {e}); using RRF order")
        hits = hits[:top_k]

        extra: list[Hit] = []
        if do_expand and self.graph is not None:
            extra = self._expand(hits, rcfg.expand.seeds, rcfg.expand.max_extra, query)
        return SearchResult(
            hits=hits + extra, gate_score=gate_score, pool_size=len(pool), reranked=reranked,
            expanded=len(extra), queries=queries,
        )

    # --- graph expansion -------------------------------------------------

    def _expand(self, hits: list[Hit], n_seeds: int, budget: int, query: str) -> list[Hit]:
        """Neighbor chunks (callees / callers) of the top few function hits — the pieces a
        flat similarity search misses for "what happens when X is called" questions.

        Seeds skip the test module (a test's callees are noise); neighbors are ranked by
        how well their own chunk matches the *query* (BM25), not alphabetically."""
        have = {h.chunk_id for h in hits}
        seeds = [
            h for h in hits
            if h.symbol and h.kind in ("function", "method") and h.module != "test"
        ][:n_seeds]
        if not seeds:
            return []
        qscores = self._bm25.get_scores(_tokenize(query))
        per_seed: list[list[tuple[float, str, str, int]]] = []
        for s in seeds:
            cands = []
            for rel, sym in self.graph.neighbors_of(s.symbol, "both"):
                row = self._symbol_row.get(sym)
                if row is None or self._bm25_meta[row]["module"] == "test":
                    continue
                cands.append((float(qscores[row]), f"{rel} of {s.symbol}", sym, row))
            cands.sort(key=lambda t: -t[0])
            per_seed.append(cands)
        # round-robin across seeds so one chatty function can't use the whole budget
        out: list[Hit] = []
        while len(out) < budget and any(per_seed):
            for cands in per_seed:
                while cands and len(out) < budget:
                    _, via, _, row = cands.pop(0)
                    cid = self._bm25_ids[row]
                    if cid in have:
                        continue
                    have.add(cid)
                    out.append(self._hit(cid, row, 0.0, 0.0, None, None, via=via))
                    break
        return out

    # --- helpers ---------------------------------------------------------

    def _hit(self, cid: str, row: int, score: float, primary: float, d_rank, b_rank, via: str | None = None) -> Hit:
        meta = self._bm25_meta[row]
        return Hit(
            chunk_id=cid, path=meta["path"], module=meta["module"], kind=meta["kind"],
            symbol=meta["symbol"], start_line=meta["start_line"], end_line=meta["end_line"],
            text=self._bm25_texts[row], rrf_score=score, dense_rank=d_rank, bm25_rank=b_rank,
            primary_score=primary, via=via,
        )

    def hit_from_id(self, chunk_id: str, score: float = 0.0, primary: float = 0.0,
                    rerank_score: float | None = None, via: str | None = None) -> Hit | None:
        """Rebuild a Hit from a chunk id (parallel sub-retrievals pass ids through graph
        state instead of copying chunk text into every checkpoint)."""
        row = self._id_to_row.get(chunk_id)
        if row is None:
            return None
        h = self._hit(chunk_id, row, score, primary, None, None, via=via)
        h.rerank_score = rerank_score
        return h

    def _path_of(self, chunk_id: str) -> str:
        row = self._id_to_row.get(chunk_id)
        return self._bm25_meta[row]["path"] if row is not None else ""

    def is_thin(self, hits: list[Hit], symbol_resolved: bool | None = None,
                gate_score: float | None = None) -> bool:
        """PLAN.md §3.1 hard gate. Deliberately independent of rerank/expansion order:
        only *ranked* hits count toward `min_hits`, and the score compared against
        `rrf_min_score` is `gate_score` (best primary-RRF over the candidate pool)."""
        cfg = self.cfg.retrieval
        ranked = [h for h in hits if h.via is None]
        if len(ranked) < cfg.min_hits:
            return True
        top = gate_score if gate_score is not None else max((h.primary_score for h in ranked), default=0.0)
        if top < cfg.rrf_min_score:
            return True
        if symbol_resolved is False:
            return True
        return False


def _matches_where(meta: dict, where: dict) -> bool:
    return all(meta.get(k) == v for k, v in where.items())
