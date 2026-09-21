"""Retrieval-layer behavior of the new stages (rerank ordering, multi-query fusion, graph
expansion) against the built index. No chat LLM; the reranker is stubbed so the test is fast
and does not need the ONNX model."""
from __future__ import annotations

import pytest

from cqa.retriever import Retriever


@pytest.fixture(scope="module")
def retriever(cfg, index_dir):
    return Retriever(cfg)


Q = "How does the thread pool distribute parallel work across worker threads?"


def test_gate_score_is_unchanged_by_rerank_and_extra_queries(retriever, monkeypatch):
    base = retriever.search_ex(Q, rerank=False, expand=False)

    class Backwards:
        def score(self, query, docs):   # reverses the order: the worst possible reranker
            return [float(i) for i in range(len(docs))]

    monkeypatch.setattr("cqa.retriever.get_reranker", lambda model: Backwards())
    reranked = retriever.search_ex(Q, rerank=True, expand=False, rerank_fuse_weight=0.0)
    multi = retriever.search_ex(Q, extra_queries=["thread pool workers", "parallel runParallel"], rerank=False, expand=False)
    assert reranked.reranked and [h.chunk_id for h in reranked.hits] != [h.chunk_id for h in base.hits]
    # the §3.1 confidence gate must not care how the pool was ordered or how many queries fed it
    assert reranked.gate_score == pytest.approx(base.gate_score) == pytest.approx(multi.gate_score)
    assert retriever.is_thin(reranked.hits, None, gate_score=reranked.gate_score) == retriever.is_thin(base.hits, None, gate_score=base.gate_score)


def test_fusing_with_rrf_keeps_a_strong_first_stage_hit(retriever, monkeypatch):
    base = retriever.search_ex(Q, rerank=False, expand=False)
    top = base.hits[0].chunk_id

    class Backwards:
        def score(self, query, docs):
            return [float(i) for i in range(len(docs))]

    monkeypatch.setattr("cqa.retriever.get_reranker", lambda model: Backwards())
    pure = retriever.search_ex(Q, rerank=True, expand=False, rerank_fuse_weight=0.0)
    fused = retriever.search_ex(Q, rerank=True, expand=False, rerank_fuse_weight=1.0)
    assert top not in [h.chunk_id for h in pure.hits[:1]]        # pure rerank demotes it
    assert top in [h.chunk_id for h in fused.hits]                # fusion keeps it in the context


def test_extra_queries_can_only_add_candidates_not_shrink_the_result(retriever):
    res = retriever.search_ex(Q, extra_queries=["ThreadPool::runParallel"], rerank=False, expand=False)
    assert len(res.hits) == retriever.cfg.retrieval.top_k and res.queries[0] == Q and len(res.queries) == 2


def test_expansion_adds_labelled_neighbors_after_ranked_hits(retriever):
    q = "When Simulation::run is called, what does it use to actually trace the rays?"
    res = retriever.search_ex(q, rerank=False, expand=True)
    ranked = [h for h in res.hits if h.via is None]
    extra = [h for h in res.hits if h.via]
    assert len(ranked) == retriever.cfg.retrieval.top_k and 0 < len(extra) <= retriever.cfg.retrieval.expand.max_extra
    assert res.hits[: len(ranked)] == ranked                       # neighbors always come last
    assert all(h.via.startswith(("callee of ", "caller of ")) and h.module != "test" for h in extra)
    assert len({h.chunk_id for h in res.hits}) == len(res.hits)    # no duplicates
    # the gate only counts ranked hits
    assert retriever.is_thin(extra, None, gate_score=1.0) is True


def test_expansion_off_returns_only_ranked_hits(retriever):
    res = retriever.search_ex(Q, rerank=False, expand=False)
    assert all(h.via is None for h in res.hits) and res.expanded == 0
