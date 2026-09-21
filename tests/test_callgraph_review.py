"""Call graph construction, diff -> symbol mapping, and git-tool safety. All synthetic or
read-only; no LLM."""
from __future__ import annotations

import pytest

from cqa.callgraph import CallGraph, to_mermaid
from cqa.chunks_cpp import Chunk
from cqa.config import load_config
from cqa.gittools import git_blame, git_log, valid_ref
from cqa.review import _changed_target_lines, _cut_at_file_boundary, _enclosing, prepare_review
from cqa.symbol_index import SymbolIndex
from cqa.tools import call_graph_tool


def _chunk(symbol, kind, path, start, end, text, module="core"):
    return Chunk(chunk_id=f"{path}:{start}", path=path, module=module, kind=kind, symbol=symbol,
                 start_line=start, end_line=end, text=text)


CHUNKS = [
    _chunk("Simulation::run", "method", "src/core/Simulation.cpp", 10, 30,
           "void Simulation::run() { auto r = RayTracer::trace(scene); size(); Vec3(1,2,3); helperStep(); }"),
    _chunk("RayTracer::trace", "method", "src/core/RayTracer.cpp", 5, 50,
           "void RayTracer::trace(const Scene& s) { shade(s); }"),
    _chunk("Other::trace", "method", "src/core/Other.cpp", 1, 9, "void Other::trace() {}"),
    _chunk("shade", "function", "src/core/RayTracer.cpp", 60, 70, "void shade(const Scene&) {}"),
    _chunk("helperStep", "function", "src/core/Simulation.cpp", 40, 45, "void helperStep() {}"),
    _chunk("Vec3", "struct", "src/core/Vec3.h", 1, 5, "struct Vec3 {};"),
    _chunk("Vec3::Vec3", "method", "src/core/Vec3.h", 2, 2, "Vec3(double,double,double) {}"),
    _chunk("testTrace", "function", "test/TestMain.cpp", 1, 10, "void testTrace() { RayTracer::trace(s); }", module="test"),
]


@pytest.fixture(scope="module")
def graph_and_symbols(tmp_path_factory):
    syms = SymbolIndex.build_from_chunks(CHUNKS)
    root = tmp_path_factory.mktemp("repo")
    (root / "src/core").mkdir(parents=True)
    (root / "src/core/Simulation.cpp").write_text('#include "RayTracer.h"\n#include "Vec3.h"\n')
    (root / "src/core/RayTracer.cpp").write_text('#include "RayTracer.h"\n')
    (root / "src/core/RayTracer.h").write_text("")
    (root / "src/core/Vec3.h").write_text("")
    return CallGraph.build(CHUNKS, syms, root), syms


def test_call_edges_use_qualifiers_and_skip_noise(graph_and_symbols):
    g, _ = graph_and_symbols
    callees = g.callees["Simulation::run"]
    assert "RayTracer::trace" in callees and "Other::trace" not in callees   # `RayTracer::` qualifier narrows
    assert "helperStep" in callees
    assert not any(c.startswith("Vec3") for c in callees)                   # constructor calls are not edges
    assert "size" not in " ".join(callees)                                  # std-colliding names skipped


def test_callers_are_inverted_and_include_tests(graph_and_symbols):
    g, _ = graph_and_symbols
    assert set(g.callers["RayTracer::trace"]) == {"Simulation::run", "testTrace"}


def test_include_graph_and_pairs_resolve(graph_and_symbols):
    g, _ = graph_and_symbols
    assert "src/core/Vec3.h" in g.includes["src/core/Simulation.cpp"]
    assert "src/core/Simulation.cpp" in g.included_by["src/core/Vec3.h"]


def test_subgraph_and_mermaid_are_safe(graph_and_symbols):
    g, _ = graph_and_symbols
    sub = g.subgraph("Simulation::run", "callees", 2)
    assert ["Simulation::run", "RayTracer::trace"] in sub["edges"] and ["RayTracer::trace", "shade"] in sub["edges"]
    src = to_mermaid("Simulation::run", sub)
    assert src.startswith("flowchart LR") and '["Simulation::run"]' in src and "-->" in src


def test_call_graph_tool_resolves_bare_names_and_reports_unknown(graph_and_symbols):
    g, syms = graph_and_symbols
    ok = call_graph_tool(g, syms, "shade", "callers", 1)
    assert "RayTracer::trace -> shade" in ok["edges"] and ok["mermaid"].startswith("flowchart")
    assert "error" in call_graph_tool(g, syms, "NoSuchThing")
    assert "error" in call_graph_tool(None, syms, "shade")


# ---------------------------------------------------------------- review

DIFF = """diff --git a/src/core/RayTracer.cpp b/src/core/RayTracer.cpp
index 111..222 100644
--- a/src/core/RayTracer.cpp
+++ b/src/core/RayTracer.cpp
@@ -8,2 +8,3 @@ void RayTracer::trace(const Scene& s) {
     shade(s);
+    audit(s);
 }
@@ -62,1 +63,1 @@ void shade(const Scene&) {
-    old();
+    fresh();
"""


def test_changed_lines_map_to_enclosing_symbols(graph_and_symbols):
    from unidiff import PatchSet

    _, syms = graph_and_symbols
    pf = PatchSet(DIFF)[0]
    lines = _changed_target_lines(pf)
    assert sorted(set(lines)) == [9, 63]   # the added line, and the replaced line
    names = {_enclosing(syms.defs_in_path("src/core/RayTracer.cpp"), ln).name for ln in lines}
    assert names == {"RayTracer::trace", "shade"}


def test_prepare_review_from_pasted_diff_reports_callers(graph_and_symbols, cfg):
    g, syms = graph_and_symbols
    prep = prepare_review(cfg, syms, g, diff=DIFF)
    assert prep.stats["files"] == 1 and prep.stats["changed_symbols"] == 2
    assert prep.stats["tests_calling_changed"] == 1   # testTrace calls RayTracer::trace
    assert "callers in test/ (1)" in prep.context and "```diff" in prep.context


def test_prepare_review_rejects_garbage_and_bad_refs(graph_and_symbols, cfg):
    g, syms = graph_and_symbols
    with pytest.raises(ValueError):
        prepare_review(cfg, syms, g, diff="this is not a diff")
    for bad in ("-x", "HEAD;ls", "a..b", "$(id)", ""):
        with pytest.raises(ValueError):
            prepare_review(cfg, syms, g, base=bad or "HEAD~1", head="HEAD" if bad else "--")


def test_large_diffs_are_cut_between_files_never_mid_hunk():
    f1 = "diff --git a/a b/a\n--- a/a\n+++ b/a\n@@ -1 +1 @@\n-x\n+y\n"
    raw = f1 * 50
    cut, dropped = _cut_at_file_boundary(raw, len(f1) * 10 + 20)
    assert dropped > 0 and cut.endswith("+y\n") and cut.count("diff --git") == 10


# ---------------------------------------------------------------- git tools


def test_ref_validation():
    assert all(valid_ref(r) for r in ("HEAD", "HEAD~3", "main...feature", "origin/main", "a1b2c3d", "v1.2.0"))
    assert not any(valid_ref(r) for r in ("", "-rf", "a b", "x;y", "$(x)", "a" * 300, "--upload-pack=x"))


def test_git_tools_reject_paths_outside_the_repo():
    cfg = load_config()
    if not cfg.repo.root.exists():
        pytest.skip("target repo not present")
    for path in ("../../etc/passwd", "../CodebaseQA/README.md", "/etc/passwd", "does/not/exist.cpp"):
        assert "error" in git_log(cfg, path)
        assert "error" in git_blame(cfg, path, 1, 5)
