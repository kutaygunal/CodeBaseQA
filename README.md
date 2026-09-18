# CodebaseQA

Code-Centric Codebase Q&A Assistant.

Answers natural-language questions about a real C++ codebase (`../LuxTrace`) by combining
AST-aware chunking, hybrid retrieval (BM25 + embeddings), an exact symbol index, and
agentic tool use (`read_file` / `grep` / `search_symbols` / `symbol_usages` / `list_dir`)
when retrieval is thin. Ships a CLI and a web chat UI with a live reasoning trace.

- Full plan: [PLAN.md](PLAN.md)
- Tech: LangGraph + LangChain, Ollama models, Chroma vector store, tree-sitter-cpp
  parsing, FastAPI + SSE web UI — no API keys.

**Status: implemented and working end-to-end** against `../LuxTrace` (154 files, 1,983
chunks, 1,540 symbols, 0 skipped). All PLAN.md §7 demo questions pass their §8.1
acceptance criteria — see [demo_output.json](demo_output.json) for full recorded Q&A.

## Models (all already installed locally via Ollama)

| Role | Model | Notes |
|---|---|---|
| Chat / agent | `deepseek-v4.1-flash:cloud` | 1M context, native tool calling; cloud-routed, needs Ollama sign-in |
| Embeddings | `nomic-embed-text` | fully local |
| Offline fallback chat | `qwen3-vl:8b` | set `chat_model` in `config/luxtrace.yaml` |

## Setup

```bash
uv venv .venv
uv pip install -e ".[dev]" --python .venv
```

Requires a running Ollama daemon with `nomic-embed-text` pulled (`ollama pull nomic-embed-text`)
and access to `deepseek-v4.1-flash:cloud` (or edit `config/luxtrace.yaml` to point at a
different chat model, e.g. the local `qwen3-vl:8b`). Copy `.env.example` to `.env` to add
OpenAI/Anthropic/GitHub Models API keys — optional, only needed to use those providers from
the web UI's Settings panel.

## Build the index

```bash
.venv/Scripts/python.exe scripts/index_luxtrace.py
```

Walks `../LuxTrace/src` and `../LuxTrace/test`, AST-chunks every C++/CUDA file via
tree-sitter (with a brace-depth fallback so nothing is ever silently skipped), embeds
every chunk, and writes the Chroma vector store, BM25 index, and exact symbol index under
`.chroma/` / `.index/`.

## Ask questions

**CLI:**
```bash
.venv/Scripts/python.exe -m cqa.cli                     # interactive REPL
.venv/Scripts/python.exe -m cqa.cli --trace "your question here"   # one-shot, with trace
```

**Web UI**:
```bash
.venv/Scripts/python.exe -m uvicorn web.server:app --host 127.0.0.1 --port 8000
```
Then open http://localhost:8000. Features:

- **Multi-provider model picker** (Settings → gear icon) — Ollama (every locally installed
  chat model, auto-listed), OpenAI, Anthropic (Claude), and GitHub Models (an
  OpenAI-compatible endpoint, `models.github.ai/inference`). Paste an API key/token right
  in the panel — kept in memory on the server for that session only, never written to
  disk or put in a URL — or set `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `GITHUB_TOKEN` in
  `.env` for a persistent default. Anthropic also accepts a Claude Pro/Max subscription
  token (`sk-ant-oat01-...`, from `claude setup-token`) in place of an API key. GitHub
  Copilot itself has no public chat-completions API as of 2026 (only unofficial
  reverse-engineered proxies exist) — GitHub Models is the legitimate equivalent.
- **Click any citation to open the source** — a resizable slide-over panel (drag its left
  edge) shows the real file, syntax-highlighted, scrolled to and highlighting the cited
  lines. Works from inline citations, the citation chips under an answer, and the
  retrieved-chunk rows in the trace. The ⧉ button undocks it into a standalone browser
  window/tab (`/static/source.html?path=...&start=...&end=...`) for a second monitor.
- **Light / dark / system theme toggle**, persisted.
- **Conversation sidebar** — multiple threads, rename, delete, durably persisted
  (SQLite-backed `SqliteSaver`, survives a server restart) and restorable.
- **Module filter** — scope retrieval to `core`/`gpu`/`render`/`ui`/`test`.
- Live retrieve → reason → tools trace panel per answer, with (mostly) less boring status
  text while tools are running.

**Run the demo questions** from PLAN.md §7 and record output to `demo_output.json`:
```bash
.venv/Scripts/python.exe scripts/demo_questions.py
```

**Tests** (retrieval-layer only, no LLM — PLAN.md §8.1):
```bash
.venv/Scripts/python.exe -m pytest tests/ -v
```

## Sample Q&A (recorded, see `demo_output.json` for the full run)

> **Q: What functions call `TraceScene::Trace`?**
>
> There is no `TraceScene::Trace` symbol in this codebase — `TraceScene`
> (`src/core/TraceScene.h:138-210`) exposes `build`, `clear`, `nearestHit`, `occluded`,
> etc., but no trace method. The function you likely mean is `RayTracer::trace`, which
> takes a `const TraceScene&`. Its callers include `Simulation::run`
> (`src/core/Simulation.cpp:494`), `Studies.cpp:1057/1103/1141/1312`, and the CLI harness
> (`src/main.cpp:197`)...

This is the agent correctly refusing to hallucinate a symbol that doesn't exist, resolving
via `search_symbols`/`grep` tool calls (retrieval alone was flagged "thin" — see
PLAN.md §3.1), and citing the real entry point instead. 3 tool rounds, ~4s.

## How it works

See [PLAN.md](PLAN.md) §3 for the full architecture. In short:

```
question → RETRIEVE (hybrid BM25+dense, RRF merge, exact symbol lookup)
         → REASON (LLM decides: answer, or call a tool)
         → [TOOLS: read_file / grep / search_symbols / symbol_usages / list_dir]*  (≤3 rounds)
         → ANSWER (grounded, path:line citations)
```

Retrieval that's "thin" (few/weak hits, or a named symbol the exact index couldn't
resolve) hard-forces at least one tool call on the first round, so the demo can prove
tool use happens rather than depending on model whim (PLAN.md §3.1). Follow-up questions
reuse the same LangGraph thread (`MemorySaver` checkpointing) so "and what calls that?"
carries prior context.
