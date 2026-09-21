"""Unit tests for the eval-harness scoring helpers (no LLM — pure functions)."""
from cqa.evalutil import expect_hit, is_refusal, path_of, percentile, retrieval_stats


def test_is_refusal_accepts_genuine_out_of_scope_answers():
    refusals = [
        "No login or authentication exists in this codebase. I searched and found nothing.",
        "There is no REST API server and no HTTP routing anywhere in this codebase.",
        "I can't answer this question from the LuxTrace codebase as retrieved.",
        "The repository does not contain any billing or subscription database code.",
        "LuxTrace does not implement user login or authentication. A grep returned no real hits.",
        "I could not find any ML training framework in the index.",
    ]
    for a in refusals:
        assert is_refusal(a), f"should have refused: {a!r}"


def test_is_refusal_rejects_business_as_usual_answers():
    normal = [
        "RayTracer::trace calls TraceScene::nearestHit (src/core/RayTracer.cpp:2130).",
        "The code is implemented in Simulation.cpp and reused across the app.",
    ]
    for a in normal:
        assert not is_refusal(a), f"should NOT have refused: {a!r}"


def test_path_of_strips_line_range():
    assert path_of("src/core/RayTracer.cpp:2130-2136") == "src/core/RayTracer.cpp"
    assert path_of("src/ui/SurfaceInspector.h:5") == "src/ui/SurfaceInspector.h"


def test_expect_hit_is_case_insensitive_substring():
    assert expect_hit(["src/core/Coating.cpp"], ["a src/core/Coating.cpp:10-20 b"])
    assert not expect_hit(["src/core/Coating.cpp"], ["src/core/MaterialFile.cpp"])


def test_retrieval_stats_from_trace():
    trace = [
        {"node": "retrieve", "hits": [
            {"path": "src/core/Other.cpp", "score": 0.5},
            {"path": "src/core/Coating.cpp", "score": 0.4},
        ]},
    ]
    s = retrieval_stats(trace, ["src/core/Coating.cpp"])
    assert s["recall@1"] == 0 and s["recall@3"] == 1 and round(s["mrr"], 3) == 0.5
    # negative case: no expected file -> rank None, all zeros
    s2 = retrieval_stats(trace, [])
    assert s2["rank"] is None and s2["recall@3"] == 0 and s2["mrr"] == 0.0


def test_percentile():
    assert percentile([1, 2, 3, 4], 50) == 3
    assert percentile([], 50) is None
