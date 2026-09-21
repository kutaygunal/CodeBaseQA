"""Unit tests for the follow-up suggestion generator (no LLM)."""
from cqa.suggest import suggest_followups


def test_symbol_based_followups():
    trace = [
        {"node": "retrieve", "hits": [
            {"symbol": "RayTracer::trace", "path": "src/core/RayTracer.cpp"},
            {"symbol": "RayTracer", "path": "src/core/RayTracer.h"},
        ]},
    ]
    qs = suggest_followups("How is tracing done?", trace, limit=3)
    assert "What calls RayTracer::trace?" in qs
    assert len(qs) <= 3
    assert qs, "should emit at least one follow-up"


def test_empty_or_none_trace_returns_empty():
    assert suggest_followups("Q", None) == []
    assert suggest_followups("Q", []) == []
    assert suggest_followups("Q", [{"node": "analyze"}]) == []


def test_never_echoes_the_question():
    trace = [{"node": "retrieve", "hits": [{"symbol": "MeshBuilder::meshShape", "path": "src/core/MeshBuilder.cpp"}]}]
    qs = suggest_followups("What calls MeshBuilder::meshShape?", trace)
    assert "What calls MeshBuilder::meshShape?" not in qs
    assert any("MeshBuilder::meshShape" in q for q in qs)


def test_path_based_fallback():
    trace = [{"node": "retrieve", "hits": [{"symbol": "", "path": "src/core/Coating.cpp"}]}]
    qs = suggest_followups("Coating?", trace)
    assert qs, "should still suggest something from the path"


def test_sub_retrieve_hits_count_too():
    trace = [
        {"node": "sub_retrieve", "hits": [{"symbol": "Simulation::run", "path": "src/core/Simulation.cpp"}]},
    ]
    qs = suggest_followups("How does a run happen?", trace)
    assert "Where is Simulation::run defined?" in qs
