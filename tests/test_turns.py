"""Per-turn store: recording, metrics, feedback, stats, tour cache, cost estimates."""
from __future__ import annotations

import sqlite3

import pytest

from cqa.turns import TurnStore, estimate_cost, summarize_trace

TRACE = [
    {"node": "analyze", "usage": {"in": 100, "out": 20}, "duration_ms": 900},
    {"node": "retrieve", "thin": False, "hits": [{"score": 0.031}, {"score": 0.02}], "duration_ms": 30},
    {"node": "reason", "usage": {"in": 1000, "out": 50}, "duration_ms": 1200},
    {"node": "tools", "calls": [{"tool": "read_file"}], "duration_ms": 5},
    {"node": "reason", "usage": {"in": 1500, "out": 300}, "duration_ms": 2000},
    {"node": "answer", "citations": []},
]


@pytest.fixture
def store():
    return TurnStore(sqlite3.connect(":memory:", check_same_thread=False), pricing={"ollama/*": {"in_per_mtok": 0, "out_per_mtok": 0}})


def _rec(store, thread="t1", q="q?", provider="ollama", model="m", latency=1234):
    return store.record(thread_id=thread, question=q, answer="a", citations=["src/a.cpp:1-2"], trace=TRACE,
                        provider=provider, model=model, module=None, mode="qa", latency_ms=latency)


def test_summarize_trace_counts_usage_calls_and_rounds():
    m = summarize_trace(TRACE)
    assert m == {"tokens_in": 2600, "tokens_out": 370, "llm_calls": 3, "tool_rounds": 1}


def test_estimate_cost_lookup_order_and_unknown_model():
    pricing = {"anthropic/x": {"in_per_mtok": 3, "out_per_mtok": 15}, "ollama/*": {"in_per_mtok": 0, "out_per_mtok": 0}}
    assert estimate_cost(pricing, "anthropic", "x", 1_000_000, 100_000) == pytest.approx(4.5)
    assert estimate_cost(pricing, "ollama", "anything", 5000, 5000) == 0
    assert estimate_cost(pricing, "openai", "gpt", 1, 1) is None   # never guessed


def test_record_and_history_roundtrip_restores_trace(store):
    meta = _rec(store)
    assert meta["tokens_in"] == 2600 and meta["cost_usd"] == 0
    [t] = store.for_thread("t1")
    assert t["trace"] == TRACE and t["citations"] == ["src/a.cpp:1-2"] and t["feedback"] is None
    assert t["metrics"]["latency_ms"] == 1234 and t["metrics"]["tool_rounds"] == 1


def test_feedback_upsert_clear_and_validation(store):
    tid = _rec(store)["turn_id"]
    assert store.set_feedback(tid, -1, reason="Wrong file", comment="c", correct_paths=["src/x.cpp:1-9", " "])
    fb = store.get(tid)["feedback"]
    assert fb["rating"] == -1 and fb["reason"] == "Wrong file" and fb["correct_paths"] == ["src/x.cpp:1-9"]
    assert store.set_feedback(tid, 1)   # change of mind overwrites
    assert store.get(tid)["feedback"]["rating"] == 1
    assert store.set_feedback(tid, 0) and store.get(tid)["feedback"] is None
    assert store.set_feedback("missing", 1) is False
    with pytest.raises(ValueError):
        store.set_feedback(tid, 5)


def test_export_includes_retrieval_stats_for_rated_turns_only(store):
    a, b = _rec(store)["turn_id"], _rec(store, q="other")["turn_id"]
    store.set_feedback(a, -1, reason="Hallucinated")
    rows = store.export_feedback()
    assert [r["turn_id"] for r in rows] == [a] and b not in [r["turn_id"] for r in rows]
    assert rows[0]["retrieval"] == {"thin": False, "top_score": 0.031, "n_hits": 2}


def test_stats_percentiles_and_thumbs(store):
    ids = [_rec(store, latency=ms)["turn_id"] for ms in (1000, 2000, 3000, 4000, 10000)]
    store.set_feedback(ids[0], 1)
    store.set_feedback(ids[1], -1)
    s = store.stats()
    assert s["overall"]["turns"] == 5 and s["overall"]["latency_p50_ms"] == 3000
    assert s["overall"]["latency_p95_ms"] == 10000
    assert s["overall"]["thumbs_up"] == 1 and s["overall"]["thumbs_up_rate"] == 0.5
    assert s["by_model"][0]["model"] == "m"


def test_delete_thread_removes_turns_and_their_feedback(store):
    tid = _rec(store, thread="gone")["turn_id"]
    store.set_feedback(tid, 1)
    store.delete_thread("gone")
    assert store.for_thread("gone") == [] and store.export_feedback() == []


def test_tour_cache_is_keyed_by_index_provider_and_model(store):
    store.put_tour("src/core/A.cpp", "abc:1", "ollama", "m", "# tour", ["src/core/A.cpp:1-2"])
    assert store.get_tour("src/core/A.cpp", "abc:1", "ollama", "m")["markdown"] == "# tour"
    assert store.get_tour("src/core/A.cpp", "abc:2", "ollama", "m") is None   # index rebuilt -> stale
    assert store.get_tour("src/core/A.cpp", "abc:1", "ollama", "other-model") is None
