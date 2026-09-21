"""Graph-level behavior with a scripted fake LLM: streaming, per-turn trace reset, the
thin-retrieval hard gate, query rewriting and the parallel sub-question planner.

Retrieval is real (needs the built index + local Ollama embeddings); the chat model is fake.
"""
from __future__ import annotations

import pytest
from langgraph.checkpoint.memory import MemorySaver

from cqa.graph import build_graph


@pytest.fixture
def app(cfg, index_dir):
    return build_graph(cfg, checkpointer=MemorySaver())


def _state(q, **kw):
    s = {"question": q, "provider": "ollama", "model": "fake", "module": None, "mode": "qa",
         "mode_system": None, "mode_context": None, "max_rounds_override": None}
    s.update(kw)
    return s


def _cfg(tid):
    return {"configurable": {"thread_id": tid}}


def test_answer_streams_tokens_and_records_usage(app, fake_llm):
    fake_llm(answer="The pool spawns workers (src/core/ThreadPool.cpp:84-89).")
    events = list(app.stream(_state("How does the thread pool distribute parallel work across workers?"),
                             _cfg("t1"), stream_mode=["updates", "custom"]))
    tokens = [p["text"] for m, p in events if m == "custom" and p.get("type") == "token"]
    assert "".join(tokens).strip().startswith("The pool spawns workers")
    final = app.get_state(_cfg("t1")).values
    assert final["final_answer"].startswith("The pool spawns workers")
    assert final["citations"] == ["src/core/ThreadPool.cpp:84-89"]
    reason = [e for e in final["trace"] if e["node"] == "reason"][0]
    assert reason["usage"] == {"in": 30, "out": 12} and reason["duration_ms"] >= 0


def test_trace_resets_each_turn_on_the_same_thread(app, fake_llm):
    fake_llm()
    for q in ("How does the thread pool distribute parallel work across workers?",
              "How is a diagnostics bundle saved for bug reports?"):
        app.invoke(_state(q), _cfg("t2"))
    trace = app.get_state(_cfg("t2")).values["trace"]
    assert [e["node"] for e in trace].count("analyze") == 1   # not 2: Overwrite reset the reducer
    assert [e["node"] for e in trace].count("retrieve") == 1


def test_thin_retrieval_still_forces_a_tool_call(app, fake_llm):
    fake_llm()   # the model never asks for a tool
    app.invoke(_state("What functions call Definitely::NotASymbol?"), _cfg("t3"))
    trace = app.get_state(_cfg("t3")).values["trace"]
    retrieve = next(e for e in trace if e["node"] == "retrieve")
    assert retrieve["thin"] is True and retrieve["symbol_resolved"] is False
    reason0 = next(e for e in trace if e["node"] == "reason")
    assert reason0["forced"] is True and reason0["tool_calls"] == ["search_symbols"]


def test_rewritten_symbol_never_makes_retrieval_thin(app, fake_llm):
    # The rewriter mentions a PascalCase word that is not a symbol; only the user's own words
    # may trip the unresolved-symbol gate.
    fake_llm(analysis={"standalone_question": "How does the thread pool in the FooBarProject codebase work?",
                       "search_queries": ["thread pool workers"], "hypothetical_answer": ""})
    app.invoke(_state("How does the thread pool distribute parallel work across worker threads?"), _cfg("t4"))
    retrieve = next(e for e in app.get_state(_cfg("t4")).values["trace"] if e["node"] == "retrieve")
    assert retrieve["symbol_resolved"] is None and retrieve["thin"] is False


def test_followup_is_rewritten_using_history(app, fake_llm):
    p = fake_llm(analysis={"standalone_question": "How does SimulationWorker handle cancellation?",
                           "search_queries": [], "hypothetical_answer": ""})
    app.invoke(_state("How does SimulationWorker run a simulation in the background?"), _cfg("t5"))
    app.invoke(_state("And how does it handle cancellation?"), _cfg("t5"))
    analyze = [e for e in app.get_state(_cfg("t5")).values["trace"] if e["node"] == "analyze"][0]
    assert analyze["standalone"] == "How does SimulationWorker handle cancellation?"
    # the analysis prompt for turn 2 carried the earlier conversation
    prompts = [c[-1]["content"] for c in p.calls if "You prepare a question" in c[-1].get("content", "")]
    assert "SimulationWorker" in prompts[-1] and "Conversation so far" in prompts[-1]


def test_compound_question_fans_out_and_merges(app, fake_llm):
    fake_llm(analysis={
        "standalone_question": "How does the GPU trace path differ from the CPU trace path?",
        "search_queries": [], "hypothetical_answer": "",
        "sub_questions": ["How does the CPU trace path work?", "How does the GPU trace path work?"],
    })
    events = list(app.stream(_state("How does the GPU path differ from the CPU trace?"), _cfg("t6"),
                             stream_mode="updates"))
    subs = [u for u in events if "sub_retrieve" in u]
    assert len(subs) == 3   # overall question + 2 sub-questions, retrieved in parallel
    trace = app.get_state(_cfg("t6")).values["trace"]
    join = next(e for e in trace if e["node"] == "retrieve")
    assert len(join["sub_questions"]) == 3
    paths = [(h["path"], h["start_line"]) for h in join["hits"]]
    assert len(paths) == len(set(paths)) <= 12   # deduplicated and capped
    assert len([e for e in trace if e["node"] == "sub_retrieve"]) == 3


def test_review_mode_skips_similarity_search(app, fake_llm):
    fake_llm(answer="**Summary** fine.")
    app.invoke(_state("Review changes A...B", mode="review", mode_system="You review.",
                      mode_context="## Diff\n```diff\n+x\n```", max_rounds_override=5), _cfg("t7"))
    s = app.get_state(_cfg("t7")).values
    retrieve = next(e for e in s["trace"] if e["node"] == "retrieve")
    assert retrieve["mode"] == "review" and retrieve["hits"] == []
    assert s["messages"][0]["content"] == "You review." and "## Diff" in s["messages"][1]["content"]
    assert s["max_rounds"] == 5
