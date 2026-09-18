# HANDOVER — CodebaseQA → multi-repo, multi-project, multi-language target

**Written:** 2026-09-18. **Source system:** this repo, built and verified end-to-end against
`../LuxTrace` (single repo, single C++ project, 154 files, 1,983 chunks). **Target:** 2 repos,
each containing *many* sub-projects in *different languages*. This document is the complete
state of the working system plus the concrete adaptation plan — read it start to finish before
touching code; §5 is the part that actually changes for the new target, everything before it is
context you need to make §5's decisions correctly, and §7 will save you from re-discovering six
real bugs.

---

## 0. TL;DR for whoever (human or agent) picks this up

1. This system works and is proven: hybrid retrieval (BM25 + dense) + an exact symbol index +
   a bounded agentic tool loop (LangGraph) + a full-featured web UI, all local-first via Ollama
   with optional OpenAI/Anthropic/GitHub Models. **~54–85x fewer tokens per question** than
   dumping source into the prompt (measured, not estimated — see §9).
2. It is **hard-wired to one repo and one language (C++)** right now. That is the only thing
   that structurally has to change. The retrieval math, the graph, the confidence gate, the
   tool loop, and 100% of the web UI do not need to change in kind — only in the metadata
   schema they key off (§5.4).
3. The two things that are currently "nice to have, todo" — **incremental re-indexing** and a
   **faster keyword index than `rank_bm25`** — become **must-haves** at your target's scale.
   Don't skip §5.7.
4. Copy this repo wholesale, then work through §6 in order. Budget for language-plugin work
   (§5.2) being the single biggest chunk of new code; everything else is schema propagation.

---

## 1. What this system does (one paragraph)

A question comes in → **RETRIEVE** (hybrid BM25 + dense embeddings, reciprocal-rank-fusion
merge, plus an exact symbol-name index) → a deterministic **hard gate** decides if retrieval
was thin (few/weak hits, or a named symbol the index couldn't resolve) → **REASON** (an LLM,
behind a provider-agnostic interface — Ollama/OpenAI/Anthropic/OpenAI-compatible) either
answers or calls a tool → if thin, at least one tool call is *forced* even if the model doesn't
ask for one → **TOOLS** (`read_file` / `grep` / `search_symbols` / `symbol_usages` / `list_dir`,
all sandboxed to the indexed root) runs, feeds results back, loops to REASON (bounded, 3
rounds) → **ANSWER**, grounded, every claim cited as `path:start-end`. LangGraph checkpoints
the conversation (SQLite-backed, durable) so follow-ups keep context. A FastAPI + vanilla-JS
web UI streams all of this live via SSE, with a multi-thread sidebar, a resizable/undockable
syntax-highlighted source viewer that citations click through to, module filtering, a theme
toggle, and a Settings panel for picking the chat provider/model and pasting API keys.

---

## 2. Why this architecture — the pitch, with real numbers

Measured on this exact build (see full methodology in the conversation this repo came from,
reproducible via the snippet in §9):

- Dumping LuxTrace's indexed `src/`+`test/` as raw context: **2,773,146 characters ≈ 690,000
  tokens**, resent on every question.
- What this system actually sent an LLM to fully answer a real question (instrumented, live):
  **51,067 characters ≈ 12,800 tokens total across a 3-tool-round conversation**; a
  no-tool-needed question costs roughly **8,000 tokens**.
- **~54x–85x fewer tokens**, plus the latency win (prefilling 690k tokens takes real wall-clock
  seconds even before the model starts responding; 8–13k tokens is near-instant), plus the
  quality win (models are measurably worse at pulling one fact out of a huge context than being
  handed the 8 chunks that matter — "lost in the middle").
- The gap **does not shrink as your target grows** — retrieval cost stays flat per question
  (~8 chunks, capped tool reads) regardless of whether the corpus is 2,000 chunks or 200,000.
  Whole-repo dumping, by contrast, becomes physically impossible past your model's context
  window and prohibitively expensive well before that.

This is the argument for doing the (real, nontrivial) work in §5 rather than just widening the
context window and pasting more.

---

## 3. Complete file inventory (this repo, as built)

~3,670 lines total. Everything under `cqa/` has no import-time side effects except reading
`config/luxtrace.yaml`; everything is unit-testable without an LLM except `graph.py`/`agent.py`.

| File | LOC | Responsibility |
|---|---:|---|
| `config/luxtrace.yaml` | — | **The** target-specific file. Repo path, include/ignore dirs, extensions, chunk sizing, models, retrieval thresholds, storage paths. This is what you fork per target. |
| `cqa/config.py` | 92 | Loads the YAML into typed dataclasses, resolves paths absolute. |
| `cqa/chunks_cpp.py` | 370 | **The language-specific file.** tree-sitter-cpp AST chunking: function/method/class/struct units, size-based splitting with overlap, class-body method-body collapsing (keeps class chunks cheap), brace-depth-0 fallback for anything tree-sitter can't parse (built for CUDA, never actually triggered — see §7). **This is what gets generalized in §5.2.** |
| `cqa/store.py` | 32 | Ollama embedding calls (batched) + Chroma `PersistentClient` setup. |
| `cqa/ingest.py` | 171 | Walks the repo, dispatches to AST or text chunking per extension, embeds, writes Chroma + a `rank_bm25.BM25Okapi` pickle + the symbol index + an index-state JSON (git commit, counts, per-module breakdown). Always a **full rebuild** (§5.7 flags this). |
| `cqa/symbol_index.py` | 144 | Exact `name → definition(s)` index built from chunk metadata at ingest time (cheap, no re-parsing). Bare-name resolution (`"Trace"` → `"RayTracer::trace"`). Usages are **not** precomputed — `find_usages()` greps the live tree on demand, since precomputing symbol×file is unbounded. |
| `cqa/retriever.py` | 139 | Hybrid search: separate dense (Chroma) and BM25 top-k lists, merged by reciprocal rank fusion (`1/(k+rank)`, `k=60`). `is_thin()` is the confidence hard gate (§4). `where`-clause metadata filtering is already generic (`all(meta.get(k)==v for k,v in where.items())`) — this is why adding `repo`/`project` filters in §5.4 is cheap. |
| `cqa/tools.py` | 216 | Agent-callable tools (`read_file` capped at 400 lines, `grep`, `search_symbols` with exact→bare→substring fallback, `symbol_usages`, `list_dir`), plus `read_full_file` (uncapped, for the web UI's source viewer) — all path-sandboxed via `_safe_resolve` (refuses traversal outside the indexed root). `TOOL_SCHEMAS` is the OpenAI-style tool-calling schema shared by every provider. |
| `cqa/providers.py` | 261 | **Chat-model provider abstraction.** `OllamaProvider` / `OpenAIProvider` (also used for any OpenAI-compatible endpoint) / `AnthropicProvider`, each normalizing to `ChatResult{content, tool_calls}`. Message-format converters (`_to_openai_messages`, `_to_anthropic_messages`) so the rest of the system stays provider-agnostic. Runtime credential store (in-memory, never on disk) + `PROVIDER_SPECS` registry. |
| `cqa/graph.py` | 332 | The LangGraph `StateGraph`: `retrieve → reason ⇄ tools → answer`, `GraphState` TypedDict, the confidence hard-gate wiring, the system prompt, citation-regex extraction. Takes an injectable `checkpointer`. |
| `cqa/agent.py` | 154 | `Agent` class: owns the compiled graph, a `SqliteSaver` checkpointer (durable), a `ThreadStore`. `ask()` (sync), `stream()` (generator of progress events for SSE), `history()` (reconstructs past turns from a checkpoint for the sidebar). |
| `cqa/threads.py` | 80 | Conversation metadata (title, timestamps) in a small SQLite table living in the *same* file as the LangGraph checkpoints. Auto-titles from the first question. |
| `cqa/cli.py` | 67 | Interactive REPL + one-shot `--trace` mode. |
| `web/server.py` | 216 | FastAPI: `/api/health`, `/api/models` (provider status), `/api/modules`, `/api/source`, `/api/tree`, `/api/threads*` (CRUD), `/api/credentials` (POST/DELETE), `/api/chat` (sync), `/api/chat/stream` (**GET**, not POST — see §7 for why). |
| `web/static/index.html` | 1,097 | The whole chat UI, self-contained: sidebar, settings popover, theme system, resizable source panel, module chips, composer. No build step, no framework. |
| `web/static/source-viewer.js` | 82 | Shared between the docked panel and the standalone undock page: hljs highlighting + the "re-open spans at line breaks" line-splitter (a real, nontrivial little algorithm — see §7) + fetch/render. |
| `web/static/source.html` | 67 | Standalone undocked source viewer — same rendering, own tab/window, for a second monitor. |
| `scripts/index_luxtrace.py` | 32 | One-shot ingest CLI. |
| `scripts/demo_questions.py` | 43 | Runs the scripted demo questions, records `demo_output.json`. |
| `tests/test_retriever.py` | 73 | Retrieval-layer tests, **no LLM involved** — symbol lookup + hybrid query top-k assertions. This is the pattern to replicate per-target (§10). |

---

## 4. Core design decisions — preserve these, they're load-bearing

1. **tree-sitter over libclang/native parsers.** No toolchain, no compile database, pip
   wheels only. You only need *named function/class nodes with line ranges* — not full
   semantic resolution. This decision generalizes cleanly: every mainstream language has a
   `tree-sitter-<lang>` PyPI package with the identical `Language(pkg.language())` /
   `Parser(lang)` API.
2. **Chunk size rules, not just AST units.** Target 800 tokens, max 1200, min 200, 15%
   overlap on oversized splits. A single giant function (LuxTrace had a 3,143-line file) must
   not become one multi-thousand-token chunk. Never silently drop a file — brace-depth-0
   fallback for anything the parser can't handle (or, in the general case, a plain
   sliding-window text chunker — already exists as `chunk_text_file`, reuse it as the
   universal fallback for languages you haven't built an AST plugin for yet).
3. **Hybrid retrieval, RRF merge.** BM25 alone misses semantic queries ("ray tracing entry
   point" → `RayTracer.cpp` even without the literal word "entry"); dense alone misses exact
   identifier matches. RRF (`retriever.py`) needs no score normalization between the two,
   which is why it was chosen over a weighted-sum merge.
4. **The confidence hard gate is deterministic, not model-trusted.** `Retriever.is_thin()`
   forces a tool round when `len(hits) < min_hits` OR top RRF score `< rrf_min_score` OR a
   symbol-shaped token in the question didn't resolve. `graph.py`'s `reason_node` *also*
   force-injects a synthetic `search_symbols` tool call if the model ignores the nudge. This
   two-layer design is why you can *prove* tool use happens on thin retrieval rather than
   hoping the model cooperates — keep both layers.
5. **Bounded tool rounds (3).** `answer_node` force-answers with whatever it has once the
   budget is spent, explicitly telling the model to flag what it couldn't confirm. Never let
   this loop be unbounded.
6. **Provider abstraction returns a normalized shape, always.** `{content, tool_calls: [{id,
   function: {name, arguments: dict}}]}` — every provider's SDK-specific response gets
   converted to this at the edge (`providers.py`), so `graph.py` never branches on which
   provider answered.
7. **Every claim is grounded with a `path:start-end` citation**, extracted via regex from the
   final answer text and independently verifiable (the web UI makes them clickable — this
   is the trust mechanism, not just a nice UX feature). The system prompt is explicit that
   ungrounded claims are not allowed.
8. **LangGraph checkpointing is durable (SQLite), not in-memory.** `MemorySaver` (the
   original default) loses every conversation on restart. Swapped to `SqliteSaver` — do not
   regress this for the new target; multi-repo means longer-lived, more valuable sessions.

---

## 5. What actually changes for "2 repos, many projects each, different languages"

This is the real work. Everything here is additive to the design above, not a rewrite.

### 5.1 Config schema: from one repo to N repos of M projects

Current `config/luxtrace.yaml` has one `repo.root`. Replace with a list, and add a
project-detection strategy per repo:

```yaml
repos:
  - name: "backend-monorepo"          # short, stable id — becomes the "repo" metadata tag
    root: "../BackendRepo"
    include_dirs: ["."]                # or specific top-level dirs if you don't want everything
    ignore_dirs: ["node_modules", ".git", "vendor", "dist", "build", "__pycache__", ".venv"]
    project_markers:                   # files that mark "this directory is a project root"
      - "package.json"
      - "pyproject.toml"
      - "go.mod"
      - "Cargo.toml"
      - "pom.xml"
      - "*.csproj"
      - "CMakeLists.txt"
  - name: "frontend-monorepo"
    root: "../FrontendRepo"
    include_dirs: ["."]
    ignore_dirs: ["node_modules", ".git", "dist", "build"]
    project_markers: ["package.json", "pnpm-workspace.yaml"]

# Per-extension language routing replaces the single cpp_extensions/text_extensions pair —
# see §5.2.
languages:
  ".py":  { chunker: "tree_sitter", pkg: "tree_sitter_python" }
  ".go":  { chunker: "tree_sitter", pkg: "tree_sitter_go" }
  ".ts":  { chunker: "tree_sitter", pkg: "tree_sitter_typescript", submodule: "typescript" }
  ".tsx": { chunker: "tree_sitter", pkg: "tree_sitter_typescript", submodule: "tsx" }
  ".java":{ chunker: "tree_sitter", pkg: "tree_sitter_java" }
  ".rs":  { chunker: "tree_sitter", pkg: "tree_sitter_rust" }
  ".rb":  { chunker: "tree_sitter", pkg: "tree_sitter_ruby" }
  ".cpp": { chunker: "cpp_custom" }    # keep the existing hand-tuned C++ logic
  ".h":   { chunker: "cpp_custom" }
  # anything not listed here → chunk_text_file (sliding-window fallback), never skipped
```

**First step before writing any of this: inventory what's actually in the two repos.**
Don't guess the language list. Run per repo:
```bash
find <repo_root> -type f | sed 's/.*\.//' | sort | uniq -c | sort -rn | head -40
```
and size each candidate language by LOC (`find <repo_root> -name '*.py' | xargs wc -l | tail -1`)
before deciding which get a real AST plugin (§5.2) vs. the plain-text fallback. Prioritize by
LOC × how often you expect "how does X work" questions about it, not by novelty.

### 5.2 Multi-language AST chunking — the real engineering work

`chunks_cpp.py`'s C++-specific logic (`_find_function_declarator`, `_declarator_name`,
`_enclosing_class_name`) exists because **C++'s tree-sitter grammar doesn't put a function's
name in a `name` field** — it's buried in a declarator subtree (return-type + pointer/ref
wrapping + identifier). This is the *hard* case, not the typical one.

**Good news: most other languages are easier.** Python, JavaScript/TypeScript, Go, Java, Rust,
Ruby, C# all expose the definition name via a **named field** you get directly:
```python
name_node = fn_node.child_by_field_name("name")
```
So the generalized design is a small per-language spec table plus one generic extraction path,
with the C++ declarator-walk kept as a documented special case:

```python
# cqa/chunks_ast.py (new, replaces the cpp-only chunks_cpp.py — keep chunks_cpp.py's
# functions as the C++ special-case implementation, imported by this dispatcher)

from dataclasses import dataclass

@dataclass
class LangSpec:
    pkg_module: str            # e.g. "tree_sitter_python"
    func_types: set[str]       # AST node types treated as "function" units
    class_types: set[str]      # AST node types treated as "class" units
    name_field: str = "name"   # field name to pull the identifier from (None = C++-style walk)
    uses_declarator_walk: bool = False   # True only for the C family

LANG_TABLE: dict[str, LangSpec] = {
    "python": LangSpec("tree_sitter_python", {"function_definition"}, {"class_definition"}),
    "javascript": LangSpec("tree_sitter_javascript",
        {"function_declaration", "method_definition"}, {"class_declaration"}),
    "typescript": LangSpec("tree_sitter_typescript",
        {"function_declaration", "method_definition"}, {"class_declaration", "interface_declaration"}),
    "go": LangSpec("tree_sitter_go", {"function_declaration", "method_declaration"}, set()),
    "java": LangSpec("tree_sitter_java",
        {"method_declaration", "constructor_declaration"}, {"class_declaration", "interface_declaration"}),
    "rust": LangSpec("tree_sitter_rust", {"function_item"}, {"struct_item", "impl_item", "enum_item"}),
    "ruby": LangSpec("tree_sitter_ruby", {"method"}, {"class"}),
    "csharp": LangSpec("tree_sitter_c_sharp", {"method_declaration", "constructor_declaration"}, {"class_declaration"}),
    "cpp": LangSpec("tree_sitter_cpp", {"function_definition"}, {"class_specifier", "struct_specifier"},
        uses_declarator_walk=True),
}
```

The rest of the pipeline (size-based splitting with overlap, "never skip a file", enclosing-class
lookup by walking `node.parent` until a `class_types` match, the brace-depth-0 / plain-text
fallback for languages without an entry) is **already language-agnostic in the existing code** —
`_split_to_size`, `_collect_nodes`, `_is_top_level_class`, `_collapse_function_bodies` (adjust the
placeholder text per language: `{ /* ... */ }` vs `: ...` for Python) all operate on generic
tree-sitter `Node` objects and don't need C++-specific changes. Budget most of your time on:

- Installing and pip-testing each `tree-sitter-<lang>` package individually before wiring it in
  (same smoke test pattern as this repo's initial tree-sitter-cpp verification — parse a small
  synthetic snippet, print the tree, confirm `function_definition`-equivalent nodes appear).
- TypeScript needs the `tree-sitter-typescript` package's `language_typescript()` /
  `language_tsx()` **two separate exports** — it's one PyPI package with two grammars, not one.
- Anything without an AST plugin still gets indexed via `chunk_text_file` (sliding-window,
  already generic, already used for `.cmake`/`.inc` here) — this guarantees the "never skip a
  file" promise holds across the whole language matrix on day one, even before every language
  has a real plugin. Ship with text-chunking for the long tail, upgrade the top 3–4 languages
  by LOC to real AST chunking first.

### 5.3 Project-boundary detection inside each repo

"Each repo has many projects inside it" needs a project tag, not just the flat `module` tag
this build used (`src/<dir>` → `core`/`gpu`/etc.). Walk from each repo root; the first
directory (top-down) containing one of `project_markers` (§5.1) **owns** every file under it
until a *nested* marker is found (monorepo packages, e.g. `packages/*/package.json`). That
directory's name (or its manifest's declared name, e.g. `package.json`'s `"name"` field, more
readable than a folder name) becomes the `project` tag. Keep `module` as an optional
finer-grained tag *within* a project, computed the same way `_module_for` does today but scoped
to that project's root instead of the repo root — useful for a big single project
(`services/api/src/{handlers,db,queue}`) but not mandatory for small ones.

### 5.4 Metadata schema propagation (the mechanical part)

Every chunk currently carries `{path, module, kind, symbol, start_line, end_line, part,
parts_total, language}`. Add `repo` and `project`:

- **`chunks_ast.py`** (or whatever replaces `chunks_cpp.py`): add `repo` + `project` params,
  same as `module` today.
- **`ingest.py`**: iterate `cfg.repos` (plural) instead of one `cfg.repo`; tag every chunk with
  its repo name and detected project; the Chroma `metadatas` dict and the BM25 pickle's
  `metadatas` list both need the two new keys (mechanical — same pattern as adding `module`
  originally).
- **`retriever.py`**: `Retriever.search()` gains `repo: str | None` and `project: str | None`
  kwargs alongside the existing `module`. **No change needed to the actual filtering logic** —
  `_matches_where` already does `all(meta.get(k) == v for k, v in where.items())`, so building
  `where = {k: v for k, v in [("repo", repo), ("project", project), ("module", module)] if v}`
  and passing it straight to both `_dense_search`'s Chroma `where=` and `_bm25_search`'s manual
  filter loop just works. Also expose `Retriever.repos` / `.projects_by_repo` (mirrors the
  existing `.modules` list, same one-line set-comprehension over `_bm25_meta`).
- **`symbol_index.py`**: `SymbolDef` gains `repo`/`project` fields. **Name-collision risk is
  real now** — a `main()` or `Server.start()` can legitimately exist in multiple unrelated
  projects. `lookup()`'s bare-name fallback will return all of them; that's *correct* behavior
  (the LLM sees each match's `path`, which disambiguates), but add an optional `project=`
  filter param to `lookup()` for tools/UI to narrow explicitly when the question names a
  project, and surface `repo`/`project` in `search_symbols`'s tool-result dict so the model can
  reason about which match is relevant.
- **`tools.py`**: `_safe_resolve` currently resolves against one `cfg.repo.root`. Generalize to
  resolve `path` as `"<repo_name>/<relative_path>"` and look up the right repo root from
  `cfg.repos` by name before resolving — otherwise `read_file`/`grep`/`list_dir` have no way to
  know which of the two repos a path belongs to. This also means every `path` stored in chunk
  metadata and every citation the model emits should be repo-prefixed
  (`backend-monorepo/services/api/src/handlers.py:42`), which also fixes the cross-repo
  path-collision risk (two repos can both have a `src/main.py`).
- **`graph.py`**: `GraphState` gains `repo`/`project` (mirrors the existing `module` field,
  same plumbing through `retrieve_node`).
- **`web/server.py`**: replace `/api/modules` with `/api/facets` returning
  `{"repos": [...], "projects": {repo: [...]}, "modules": {repo: {project: [...]}}}` for a
  cascading filter UI; `/api/source` needs the repo-prefixed path scheme from the point above.
- **`web/static/index.html`**: the flat module-chip row becomes two cascading pickers (Repo →
  Project, with Module as an optional third level) — same chip-rendering pattern, just fed from
  `/api/facets` instead of `/api/modules` and wired to update the `repo`/`project`/`module`
  query params sent to `/api/chat/stream`.

### 5.5 Source viewer: multi-language syntax highlighting

`source-viewer.js` calls `hljs.highlight(data.text, { language: 'cpp' })` — hardcoded. Fix:
`GET /api/source` should return the chunk/file's detected language (already known from the
extension → language table in §5.2), and the frontend should call
`hljs.highlight(data.text, { language: data.language })` — hljs ships grammars for every
mainstream language; load the ones you need from the same cdnjs highlight.js package this
repo already uses (add `<script>` tags for each language pack, same pattern as the existing
`languages/cpp.min.js` include). The hand-rolled `.hljs-*` CSS theme in `index.html`/
`source.html` (written because external stylesheets aren't on the CDN allowlist — only cdnjs
*scripts*) already covers the token classes hljs uses generically (`keyword`, `string`,
`comment`, `number`, `title`, `type`, `meta`) so it should need little to no change across
languages — verify visually per language, tweak only if a grammar emits a class the current
theme doesn't style.

### 5.6 Web UI: everything else carries over unchanged

The theme system (CSS custom properties + `data-theme` attribute + `prefers-color-scheme`
fallback), the settings popover (provider/model picker, credential inputs), the multi-thread
sidebar (SQLite-backed, `threads.py` unchanged), the resizable/undockable source panel, the
fun status-message array, the trace panel — **none of this is repo/language-specific.** Don't
touch it beyond the facet-filter change in §5.4 and the syntax-highlighting language param in
§5.5.

### 5.7 Scaling — the two must-haves at this target's size

LuxTrace (47.8k LOC, 154 files, 1,983 chunks) never stressed either of these. Two repos of
many projects will, likely by 10–100x on chunk count. Both were explicitly deferred as
"nice/todo" for LuxTrace — **don't defer them for this target.**

1. **`rank_bm25` doesn't scale.** `BM25Okapi` is pure Python, holds the whole tokenized corpus
   in memory, and `get_scores()` is an unindexed O(N) scan per query with no early
   termination. Fine at ~2,000 documents; starts costing real per-query latency somewhere in
   the tens of thousands, and gets worse from there. Chroma's dense side (HNSW) does **not**
   have this problem — it's built for hundreds of thousands of vectors. **Recommended fix:**
   swap to [`bm25s`](https://github.com/xhluca/bm25s) (numpy/scipy-backed, same BM25 math, an
   order of magnitude faster, near drop-in — `retriever.py`'s `_bm25_search` is the only
   function that touches the BM25 object directly, so this is a contained change) once you
   have real chunk counts to benchmark against. Don't pre-optimize before you know the number;
   do measure before you ship.
2. **There is no incremental re-indexing.** `ingest.build_index()` always does a full rebuild
   (`reset=True` on the Chroma collection, a fresh BM25 corpus, a fresh symbol index). At
   LuxTrace's scale this was 17–47s, a non-issue. At two big multi-project repos, re-embedding
   everything because one file changed is a real cost (both in wall-clock and — if you're using
   a paid embedding API for anything — money) and a real friction point for keeping the index
   fresh. **Recommended design** (the mechanism was already sketched but never built — see
   PLAN.md §9's decisions log for the original intent):
   - Store the last-indexed git commit **per repo** in the index-state file (already captured
     per-run via `_git_commit()`, just needs to move from "informational" to "compared against").
   - On re-index, run `git diff --name-only <last_commit> HEAD -- <include_dirs>` per repo to
     get changed/added/deleted files.
   - For changed/deleted files: delete their existing chunks from Chroma (`collection.delete(
     where={"path": path, "repo": repo_name})`) and drop them from the in-memory chunk list
     before rebuilding BM25.
   - For changed/added files: re-chunk + re-embed only those, `upsert` into Chroma.
   - Rebuild the BM25 object from the updated full chunk list (`rank_bm25`/`bm25s` have no
     incremental-add API, but re-tokenizing + rebuilding the index object is CPU-only and cheap
     even at 100k+ docs — it's the *embedding* calls you're saving, not the BM25 rebuild).
   - Rebuild the symbol index the same way (it's already built fresh from the in-memory chunk
     list every run — just needs that list to be the merged changed+unchanged set, not
     everything re-chunked from scratch).
   - Expose both modes: `scripts/index_luxtrace.py --full` (today's behavior, for first run or
     `--reset`) and the default incremental path once a prior index-state file exists.

---

## 6. Build order for the new target

Mirrors what worked here (`PLAN.md` §6), adjusted for the new scope:

1. **Inventory** both repos: language/extension breakdown, LOC per language, manifest files
   present (to validate your `project_markers` list), rough chunk-count estimate (LOC / ~35
   avg chars-per-chunk-line-target — see the estimation snippet in §9). Do this *before*
   writing `config/luxtrace.yaml`'s replacement — it tells you which languages need a real
   AST plugin (§5.2) on day one vs. can ride the text-chunk fallback initially.
2. **Config schema** (§5.1) — get `repos:` + `languages:` + `project_markers` right and
   validated (a small script that just walks and prints detected `{repo, project, module}`
   per file, no chunking/embedding yet, is worth writing first — cheap to get wrong silently).
3. **Chunker generalization** (§5.2) — start with the fallback (`chunk_text_file`, already
   works for anything) wired to the new repo/project/language tags, confirm **zero files
   skipped** across both repos before adding a single AST plugin. Then add AST plugins for
   your top 2–4 languages by LOC, smoke-testing each against real files from the target repos
   the same way this repo's tree-sitter-cpp was verified (`__global__`/CUDA parsing check,
   etc. — see the conversation history for the exact verification pattern).
4. **Ingest + retriever** (§5.4's `ingest.py`/`retriever.py`/`symbol_index.py` changes) — run a
   full index, sanity-check chunk/symbol counts per repo/project, and **immediately benchmark
   BM25 query latency at real scale** (§5.7 point 1) before building anything on top of it.
5. **Tools + graph** (§5.4's `tools.py`/`graph.py` changes, repo-prefixed paths) — re-run this
   repo's exact verification pattern: ask a question expected to need tools, confirm the trace
   shows forced tool use on thin retrieval, confirm citations are repo-prefixed and correct.
6. **Web UI facets + syntax highlighting** (§5.4/§5.5) — cascading repo→project→module
   filters, per-language hljs grammar loading.
7. **Acceptance criteria** (§10) — write the PLAN.md-§8.1-style table for this target: pick
   5–10 real questions spanning both repos and multiple projects/languages, assert which
   file each must cite, keep these as regression tests.
8. **Incremental re-indexing** (§5.7 point 2) — build this once step 4–7 are stable and you
   have a real editing cadence to test against (change a file, re-index, confirm only that
   file's chunks changed and the rest of the index is untouched).

---

## 7. Known gotchas — already hit and fixed here, will recur

Save yourself the debugging time:

1. **Module/project tagging from path parts is easy to get subtly wrong.** First pass here
   used `parts[1]` blindly, which tagged `src/main.cpp` as module `"main.cpp"` (a filename,
   not a module) and everything under `test/` by filename instead of a shared `"test"` tag.
   Fixed in `_module_for` (`ingest.py`) — the lesson generalizes: **write a quick script that
   prints the detected tag for every file and eyeball it** before running a full (expensive)
   ingest. This bites harder with the §5.3 project-marker walk — test it on both repos'
   actual directory shapes, don't assume.
2. **LangGraph + msgpack checkpointing chokes on custom dataclasses.** Storing a raw `Hit`
   dataclass instance in graph state triggered `"Deserializing unregistered type"` warnings
   (will hard-fail in a future langgraph version). Fix: convert to plain dicts before they go
   into any state field that gets checkpointed. Applies to any new dataclass you put in
   `GraphState` (e.g. a `ProjectRef` type, if you add one).
3. **Follow-up questions silently lose history if you're not careful.** `retrieve_node`
   originally rebuilt `messages` from scratch every turn, discarding prior conversation. Fix:
   check `state.get("messages")` for existing history and *append* the new turn instead of
   replacing. Easy to regress if you refactor `retrieve_node`.
4. **Windows console + non-ASCII output = crash, not mojibake.** `print()`ing an em-dash or
   arrow character through the default Windows `cp1252` stdout encoding raises
   `UnicodeEncodeError` (not just ugly output). Fixed with
   `sys.stdout.reconfigure(encoding="utf-8")` at the top of `cli.py` and
   `scripts/demo_questions.py`. Do this in any new CLI entry point too.
5. **POST + `fetch()` + `ReadableStream` for SSE is fragile in sandboxed/proxied browser
   contexts** (this was hit in an automated-browser testing sandbox specifically, not
   necessarily in a real end-user Chrome — but it's cheap insurance either way). The fetch
   reading loop got silently `ERR_ABORTED` even though the backend completed correctly
   (verified via `curl`). **Fixed by switching `/api/chat/stream` from POST to GET and the
   frontend from manual `fetch`+`ReadableStream` to the browser-native `EventSource` API** —
   more standard, more broadly compatible, no custom reconnect/parsing logic needed. Keep this
   as GET; don't "fix" it back to POST for the sake of hiding the question text from the URL —
   query params here are the question text and filter selections, not secrets (secrets — API
   keys — are handled separately via `POST /api/credentials`, stored server-side, never in a
   URL; preserve that separation for any new secret-bearing endpoint).
6. **tree-sitter-cpp actually handles `__global__` (CUDA) and similar unknown-macro tokens
   gracefully** — it produces an `ERROR` node as a child but still yields a correctly-ranged
   `function_definition` node around it. The brace-depth-0 fallback built for this case never
   actually fired on real LuxTrace CUDA code. **Don't assume a language needs its own special
   parsing path until you've actually tested tree-sitter against real files from that
   language/framework** — grammars are more fault-tolerant than you'd guess, and premature
   fallback logic is wasted effort. Test first (§6 step 3), special-case only if needed.
7. **`SqliteSaver` needs `check_same_thread=False` on the sqlite3 connection** since the web
   server's SSE streaming runs each request's graph execution in a background thread separate
   from the one that constructed the `Agent`/connection. `SqliteSaver` has an internal lock
   making this safe (confirmed via its docstring, not just assumed) — don't drop the flag or
   swap in a raw `sqlite3.connect()` elsewhere without it if you add more threaded access to
   the same DB file (e.g. the `ThreadStore` in `threads.py` reuses the *same* connection
   object for exactly this reason — don't open a second connection to the same file from a
   different thread without checking this).

---

## 8. Environment & dependencies (versions that worked)

```
Python 3.11+ (built/tested on 3.12.11 via `uv venv`)
uv 0.12.1
```

```toml
# pyproject.toml — core deps that carry over unchanged; add tree-sitter-<lang> per §5.2
langchain>=0.3.0
langchain-core>=0.3.0
langgraph>=0.2.0
langchain-chroma>=0.1.4
chromadb>=0.5.0
ollama>=0.4.0
rank-bm25>=0.2.2              # ← replace with bm25s at scale, see §5.7
tree-sitter>=0.23.0
tree-sitter-cpp>=0.23.0       # ← add one line per language, see §5.2
pyyaml>=6.0
pydantic>=2.7
fastapi>=0.115.0
uvicorn[standard]>=0.30.0
sse-starlette>=2.1.0
openai>=1.50.0
anthropic>=0.40.0
langgraph-checkpoint-sqlite>=2.0.0
```

Ollama daemon must be running with `nomic-embed-text` pulled (embeddings, always local) and
whichever chat model(s) you configure available (`ollama list` to check). No API keys required
for the baseline; OpenAI/Anthropic/GitHub Models are optional, added via `.env` or pasted into
the web UI's Settings panel (kept in server memory only, never written to disk).

---

## 9. Baseline metrics from the LuxTrace build (sanity-check the new build against these)

```
Repo:        154 files, 47,858 LOC (src/) — 61,842 LOC including test/
Chunking:    1,983 chunks, 0 skipped, <1s
Full ingest: 17-47s (embedding-call-bound; faster once Ollama has the model warm)
Symbols:     1,540 unique
Tests:       6/6 passing, retrieval-layer only, no LLM
Per-question cost (measured, real instrumented run):
  no-tool-needed question:  ~8,000 tokens
  3-tool-round question:    ~12,800 tokens total across the conversation
  vs. whole-repo dump:      ~690,000 tokens
```

Reproduce the per-question measurement on the new target with:
```python
import cqa.providers as providers
orig = providers.OllamaProvider.chat
calls = []
def wrapped(self, model, messages, tools):
    calls.append(sum(len(m.get('content') or '') for m in messages))
    return orig(self, model, messages, tools)
providers.OllamaProvider.chat = wrapped

from cqa.agent import Agent
a = Agent()
r = a.ask("<a real question>")
print(calls, sum(calls) // 4, "approx tokens")
```

---

## 10. Testing pattern to replicate (`tests/test_retriever.py`)

No-LLM, retrieval-layer-only tests: exact symbol lookup resolves to the right file, bare-name
lookup resolves through to a qualified form, a semantic hybrid query returns the expected file
in its top-k, a module/project-scoped query only returns that scope, `is_thin()` doesn't crash
on edge cases. For the new target, add one test file **per repo** (or per major project) with
5–10 questions where you know the ground-truth file, exactly like PLAN.md §8.1's
question→must-cite-file table — this is what makes "did the multi-language chunker actually
work" a fast, LLM-free, CI-able check instead of a vibe check.

---

## 11. Explicitly not built — don't assume these exist

- Scoped query syntax (`@file`, `#dir`, `@lang` typed directly into the chat box) — the
  retriever supports the filters programmatically (`module`/`repo`/`project` kwargs); there's
  no parser turning `@RayTracer.cpp` typed in the composer into that filter. The web UI's chip
  filters are the only exposed UX for this today.
- Incremental re-indexing (§5.7 — build this for the new target, it wasn't needed here).
- A faster-than-`rank_bm25` keyword index (§5.7 — same).
- Any multi-language support at all (§5.2 — this is the actual net-new work).
- Cross-repo symbol resolution beyond "the bare-name fallback naturally returns matches from
  both repos, disambiguated by the now-repo-prefixed `path`" (§5.4) — there's no special
  "did you mean the one in repo A or repo B" UX; the citations already carry that information,
  which was judged sufficient rather than building a disambiguation prompt.
- A real GitHub Copilot integration — deliberately not built. Copilot Chat has no public
  chat-completions API for third-party apps as of 2026; only unofficial reverse-engineered
  proxies exist, which this project avoided on ToS grounds. GitHub Models
  (`models.github.ai/inference`) is the supported provider that gets you the closest thing.
