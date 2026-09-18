# CodebaseQA — Code-Centric Codebase Q&A Assistant

**Target showcase repo:** `../LuxTrace` — a C++ optics/ray-tracing simulation application.
**Goal:** A bot that answers natural-language questions about LuxTrace's source using real
repository structure (not README blobs), with agentic tool use when retrieval is thin.

---

## 1. Why LuxTrace is the right target

- **Single, consistent language (C++)** → clean AST-based chunking story.
- **Clear module boundaries:** `src/core` (optics simulation, 77 files), `src/gpu`
  (CUDA kernels, 5), `src/render` (12), `src/ui` (Qt panels, 39).
- **Rich cross-file call chains** to chase: e.g. Simulation → SimulationWorker →
  RayTracer → TraceScene(SIMD) → GpuTrace; UI MainWindow → Simulation → JobRunner.
- **~48k LOC of hand-written source** (134 files) — large enough to be interesting, small
  enough to index/answer quickly on a single machine.
- **Has a build system + tests** (CMake, `test/`) which the assistant can later summarize.
- **Is a git repo** — `git diff --name-only` gives incremental re-indexing for free.

---

## 2. Scope (kept mid-size)

| Feature | Priority | Status |
|---|---|---|
| AST/path-aware chunking for C++ (`function`/`class`/`method` units) | **must** | **done** |
| Max-size chunk splitting + CUDA fallback (§2.1) | **must** | **done** |
| Vector store (Chroma) + metadata (path, symbol, line range, module) | **must** | **done** |
| Hybrid retrieval = BM25 (keyword) + dense embeddings, RRF merge | **must** | **done** |
| Exact symbol index (`function`/`class` names → definition + usages) | **must** | **done** |
| Agent tools: `read_file`, `grep/search_symbols`, `list_dir` | **must** | **done** (+ `symbol_usages`) |
| Graph loop: retrieve → confidence low → tool call → re-answer (bounded) | **must** | **done** |
| Grounded citations (`path:line`, symbol names) in every answer | **must** | **done** |
| CLI chat interface with follow-up memory (LangGraph checkpointing) | **must** | **done** |
| Scoped queries (`@file`, `#dir`, `@lang`) — metadata filters | nice | **done** as a module filter (UI chips: core/gpu/render/ui/test); no `@file`/`#dir` query-syntax parser |
| Incremental re-indexing (only changed files, via git) | nice | todo |
| Web UI | later | **done** — FastAPI + SSE, live trace panel, multi-provider model picker, clickable source viewer, multi-thread sidebar, theme toggle |
| Logging / "show the trace" of retrievals + tool calls | nice | **done** — `--trace` CLI flag + web trace panel |
| Multi-provider chat (Ollama/OpenAI/Anthropic/GitHub Models) | (added post-build-out) | **done** — `cqa/providers.py` |
| Click-through source viewer (citation → highlighted file) | (added post-build-out) | **done** — `/api/source` + side panel, hand-rolled hljs theme |
| Durable multi-thread conversation sidebar | (added post-build-out) | **done** — `SqliteSaver` + `cqa/threads.py` |

### 2.1 Chunk sizing rules

AST units alone are not enough — `src/core/RayTracer.cpp` is 3,143 lines and a single
function there can exceed any sane embedding window.

- **Target** chunk ≈ 200–1,200 tokens.
- A function/class node **over the max** is split into overlapping sub-chunks (~15% overlap)
  that all carry the **same** `symbol` / `path` metadata, plus `part: i/n`.
- Tiny adjacent nodes (one-line getters) may be **merged** up to the target size.
- **CUDA fallback:** `tree-sitter-cpp` does not understand `__global__` / `<<<>>>` launch
  syntax, and `src/gpu/GpuTrace.cu` is 2,389 lines — the second-largest file in the repo.
  If an AST parse yields **zero** function nodes for a file, fall back to brace-depth-0
  heuristic splitting rather than dropping the file. Never silently skip a source file.

---

## 3. Architecture (LangGraph + LangChain, per current API)

```
User question
      │  invoke()
      ▼
┌─────────────────────────────┐
│  RETRIEVE node              │
│  hybrid: BM25 + dense (RRF) │  ── hits with metadata
│  + exact symbol lookup      │
└──────────────┬──────────────┘
               │  context + confidence
               ▼
┌─────────────────────────────┐
│  REASON node (LLM)          │
│  enough? → answer with refs │
│  thin?    → signal tool call│
└──────┬──────────────┬───────┘
       │ no           │ yes
       ▼              ▼
  ANSWER node   ┌─────────────────────────────┐
  (citations)   │  TOOLS node                │
                │  read_file / grep / list   │
                │  results feed back         │
                └──────────┬────────────────┘
                           │  loop to REASON (bounded, max N)
```

- **State:** `TypedDict {messages, question, retrieved_chunks, tool_results, step}`.
- **Checkpointing** (LangGraph `MemorySaver`) → follow-up "and what calls that?" context.
- **Loop guard:** max tool rounds (**3**) per question; abort → answer with what it has.

### 3.1 "Confidence" — concrete definition

The REASON node branches on retrieval being "thin". That is decided by **two** independent
triggers, either of which routes to TOOLS:

1. **Hard gate (deterministic, evaluated in RETRIEVE):** force a tool round if
   `len(hits) < 3`, **or** the top RRF score is below `RRF_MIN` (config, start at `0.03`),
   **or** the question names a symbol the exact symbol index did not resolve.
2. **Model gate:** the LLM emits a tool call. Tools are bound to the model, so "I need to
   look at that file" arrives as a `tool_calls` message, never parsed out of prose.

Both gates share the same 3-round budget. The hard gate exists so the demo can *prove*
tool use happens on thin retrieval without depending on model whim.

---

## 4. Project layout

```
CodebaseQA/
├── PLAN.md                  # this file
├── README.md                # usage + how it works + sample Q&A against LuxTrace
├── pyproject.toml           # deps: langchain, langgraph, langchain-chroma, ollama,
│                            #       tree-sitter, tree-sitter-cpp, rank_bm25
├── .gitignore               # .chroma/, .index/, __pycache__, .venv, .env
├── .env.example             # optional API keys; default is Ollama
├── config/
│   └── luxtrace.yaml        # repo target, languages, ignore dirs, chunk size, model names
├── cqa/
│   ├── __init__.py
│   ├── ingest.py            # walk repo, AST split of C++, build Chroma + symbol index
│   ├── chunks_cpp.py        # tree-sitter-cpp function/class splitting + size rules (§2.1)
│   ├── store.py             # Ollama embeddings + Chroma client
│   ├── retriever.py         # hybrid: BM25 + dense (RRF) + metadata filter
│   ├── symbol_index.py      # function/class definitions + usage lookup
│   ├── tools.py             # read_file / grep / search_symbols / symbol_usages / list_dir
│   ├── providers.py         # chat-model abstraction: Ollama / OpenAI / Anthropic / OpenAI-compatible
│   ├── graph.py             # LangGraph: RETRIEVE → REASON → (TOOLS)* → ANSWER
│   ├── agent.py             # high-level ask() / stream() / history()
│   ├── threads.py           # conversation (thread) metadata store, sqlite-backed
│   └── cli.py                # interactive chat REPL
├── web/
│   ├── server.py               # FastAPI: chat (SSE), threads, source viewer, models, credentials
│   └── static/
│       ├── index.html          # chat UI: sidebar, settings, theme toggle, resizable source panel
│       ├── source.html         # standalone undocked source viewer (own tab/window)
│       └── source-viewer.js    # shared hljs-based line-highlighting logic (both pages)
├── scripts/
│   ├── index_luxtrace.py    # one-shot ingestion of ../LuxTrace
│   └── demo_questions.py    # scripted sample questions for the README demo
└── tests/
    └── test_retriever.py    # sanity: symbol lookup + hybrid query hit correct files
```

> `.gitignore` must exist **before** the first index run — Chroma writes a persistent
> store directory into the project.

---

## 5. Technology choices (Ollama-backed, no API keys)

| Layer | Choice | Why |
|---|---|---|
| Embeddings | Ollama `nomic-embed-text` — installed ✅ | cheap, CPU-friendly, no key |
| Chat/agent model | Ollama **`deepseek-v4.1-flash:cloud`** — installed ✅ | 1M context, native tool calling, no local VRAM cost |
| Offline fallback | Ollama `qwen3-vl:8b` — installed ✅ | keeps the project runnable with no network |
| Vector store | **Chroma** | small, local, easy |
| BM25 hybrid | **rank_bm25** | tiny, no external service |
| Orchestration | **LangGraph** (`langgraph`) | current API, checkpointing, loop control |
| C++ parsing | **`tree-sitter-cpp`** | pip wheel, no toolchain — see §5.2 |

### 5.1 Model decision — why `deepseek-v4.1-flash:cloud`

Verified directly against the local Ollama daemon:

- `ollama show` reports `architecture deepseek_v41`, **context length 1,048,576**,
  capabilities `completion / thinking / tools / vision`.
- A live `/api/chat` call with a bound `grep_symbol` tool came back with a proper
  `tool_calls` message — **tool calling works**, which the entire TOOLS node depends on.
- `/api/embed` with `nomic-embed-text` returns vectors.

The 1M context is the decisive win: a whole 3,000-line file can be handed to the REASON
node after a `read_file` tool call, with no windowing gymnastics.

**Honest caveat:** this is a *cloud-routed* model served through the Ollama client. It
needs an Ollama sign-in and network access. There is no API key to manage, but the earlier
"runs fully offline" framing does not hold for this choice. `config/luxtrace.yaml`
therefore carries a `chat_model` key — set it to `qwen3-vl:8b` for a fully local run.
Embeddings stay local either way, so the index builds offline regardless.

### 5.2 Parser decision — tree-sitter, not clang

The previous draft said "clang-based" in §4 while §5 offered "regex/clang-tidy **or**
tree-sitter". Resolved in favour of **tree-sitter-cpp**:

- `libclang` on Windows needs a matching DLL *and* a compile database to parse LuxTrace's
  Qt + CUDA headers correctly — a large setup cost for this project's needs.
- Nothing here needs full semantic resolution; it needs *named function/class nodes with
  line ranges*, which tree-sitter gives from a pip wheel.
- Symbol **usages** come from the symbol index + grep, not from a type-resolved AST.

---

## 6. Build order (learning in stages)

1. **Ingest** — walk `../LuxTrace/src`, chunk by C++ function/class via tree-sitter,
   apply the §2.1 size rules, store chunks + metadata. *Deliverable:* counts (`n chunks`,
   per module) and **zero skipped source files**.
2. **Retriever** — dense (Chroma) + BM25 hybrid, RRF merge, metadata filter.
   *Deliverable:* top-k query answers with file paths; unit check a known symbol.
3. **Symbol index** — definitions + usages for "what calls X / where is X". *Deliverable:*
   answering `where is Simulation` / `what calls RayTracer::Trace`.
4. **Tools + graph** — `read_file`, `grep`, `list_dir`; RETRIEVE → REASON → TOOLS → REASON
   loop with the §3.1 gates. *Deliverable:* end-to-end answers with `path:line` citations.
5. **CLI + memory** — interactive REPL, checkpointing for follow-ups, "show trace" flag.
6. **Demo & README** — run scripted questions against LuxTrace, record outputs.

Steps 1–3 involve no LLM and are independently testable.

---

## 7. Example questions we want to answer (from the real repo)

Every symbol below was confirmed present in `../LuxTrace/src`.

- "How does a simulation run from the UI to the result?" — MainWindow → SimulationWorker
  → RayTracer → JobRunner.
- "What functions call `TraceScene::Trace`?"
- "Where is `SimulationResult` used and how is it produced?" (38 files reference it)
- "How does the GPU path differ from the CPU trace?" — GpuTrace vs TraceSceneSimd.
- "What does `RayFile` store and where is it written/read?"

---

## 8. Definition of done — ✅ met

- ✅ `scripts/index_luxtrace.py` builds Chroma + symbol index over `../LuxTrace/src` and
  `test/` with **0 files skipped** (154 files, 1,983 chunks, 1,540 symbols; CUDA included
  per §2.1 — tree-sitter actually parses `GpuTrace.cu`'s functions directly, so the
  brace-depth fallback exists as a safety net but wasn't needed on this repo).
- ✅ CLI/web UI answer the demo questions with **grounded citations** (path:line + symbol
  names) — verified programmatically against the §8.1 table below.
- ✅ When retrieval is thin, the agent visibly calls a tool (trace shows `search_symbols`/
  `grep`/`read_file` rounds; see `demo_output.json`).
- ✅ Follow-up questions reuse prior context (checkpointing) — verified interactively in
  both the CLI and the web UI ("what does it use JobRunner.h for?" correctly resolved
  "it").
- ✅ README shows recorded Q&A + a run trace; `pytest tests/` passes (6/6).

### 8.1 Testable acceptance criteria

Each demo question maps to a file that **must** appear in the citations, so
`tests/test_retriever.py` can actually fail:

| Question | Must cite |
|---|---|
| "What calls `TraceScene::Trace`?" | `src/core/RayTracer.cpp` |
| "How does the GPU path differ from the CPU trace?" | `src/gpu/GpuTrace.cu` |
| "How does a simulation run from the UI to the result?" | `src/ui/MainWindow.cpp` |
| "What does `RayFile` store?" | a `src/core/RayFile.*` file |

Plus retrieval-level asserts that need no LLM at all: `symbol_index.lookup("RayTracer")`
resolves to a definition under `src/core/`, and a hybrid query for
`"ray tracing entry point"` returns `src/core/RayTracer.cpp` in the top 5.

---

## 9. Notes / decisions log

- Chat is Ollama **`deepseek-v4.1-flash:cloud`** (tool calling verified against the live
  daemon); embeddings are local `nomic-embed-text`. No API keys, but chat needs an Ollama
  sign-in + network. `qwen3-vl:8b` is the configured offline fallback. (§5.1)
- C++ parsing is **tree-sitter-cpp**, not libclang. (§5.2)
- `src/gpu/GpuTrace.cu` and `.h.in` templates included; build output ignored.
- Ignore `build/`, `.git/`, `resources/`, `docs/` for indexing.
- **Index `test/` as well** (`test/*.inc`, ~40 suites). Test files are strong evidence for
  "how is X used" questions, and §1 already promises the assistant can summarize them.
  Tag them `module: test` so they can be filtered out when unwanted.
- Incremental re-indexing keys off `git diff --name-only` against the last indexed commit,
  stored alongside the Chroma collection.
