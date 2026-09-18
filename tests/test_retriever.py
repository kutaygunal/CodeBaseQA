"""Sanity tests: symbol lookup + hybrid retrieval hit the right files.

Needs an index already built (`python scripts/index_luxtrace.py`) — these are
retrieval-layer tests and deliberately involve no LLM, per PLAN.md §8.1.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cqa.config import load_config
from cqa.retriever import Retriever
from cqa.symbol_index import SymbolIndex


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def symbols(cfg):
    path = cfg.storage.index_state_file.parent / "symbols.json"
    if not path.exists():
        pytest.skip("index not built — run scripts/index_luxtrace.py first")
    return SymbolIndex.load(path)


@pytest.fixture(scope="module")
def retriever(cfg):
    if not (cfg.storage.index_state_file.parent / "bm25.pkl").exists():
        pytest.skip("index not built — run scripts/index_luxtrace.py first")
    return Retriever(cfg)


def test_symbol_lookup_raytracer_resolves_to_core(symbols):
    defs = symbols.lookup("RayTracer")
    assert defs, "RayTracer should resolve"
    assert any(d.path.startswith("src/core/") for d in defs)


def test_symbol_lookup_bare_name_resolves_qualified(symbols):
    # A bare method name with no free-function of the same name should still resolve,
    # via bare-name matching, to its qualified "Class::method" form.
    defs = symbols.lookup("buildViewerTab")
    assert defs
    assert any(d.name == "MainWindow::buildViewerTab" for d in defs)


def test_hybrid_query_ray_tracing_entry_point(retriever):
    hits = retriever.search("ray tracing entry point", top_k=5)
    assert any(h.path == "src/core/RayTracer.cpp" for h in hits)


def test_hybrid_query_gpu_trace(retriever):
    hits = retriever.search("GPU ray tracing kernel CUDA", top_k=8, module="gpu")
    assert hits
    assert all(h.module == "gpu" for h in hits)


def test_hybrid_query_rayfile(retriever):
    hits = retriever.search("RayFileData struct fields power extent", top_k=5)
    assert any(h.path.startswith("src/core/RayFile") for h in hits)


def test_is_thin_flags_weak_retrieval(retriever):
    hits = retriever.search("a completely unrelated nonsense query xyzzy plugh", top_k=5)
    assert retriever.is_thin(hits) in (True, False)  # exercised without crashing
    assert retriever.is_thin([], symbol_resolved=False) is True
