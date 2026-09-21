"""LangGraph state machine:

    ANALYZE -> RETRIEVE -> REASON -> (TOOLS)* -> ANSWER
       \\-> SUB_RETRIEVE x N (parallel, compound questions) -/

See PLAN.md §3 (architecture) and §3.1 (the "confidence"/thin-retrieval gates).

* ANALYZE   one small LLM call: standalone rewrite, extra queries / HyDE, sub-question plan
            (cqa/query_analysis.py). Skipped for short questions; failures are harmless.
* RETRIEVE  hybrid search (+rerank, +optional graph expansion) and the deterministic
            confidence gate. For compound questions it joins the parallel SUB_RETRIEVE results.
* REASON    the LLM answers or calls a tool; streams answer text as it is generated.
* TOOLS     read_file / grep / search_symbols / symbol_usages / list_dir / call_graph /
            git_log / git_blame (bounded rounds).
* ANSWER    final answer with `path:start-end` citations.

`trace` and `sub_results` use an appending reducer so parallel branches can't clobber each
other; the first node of every turn resets them with `Overwrite`. Each trace entry carries
`duration_ms` and (for LLM calls) `usage` — that is the observability data (cqa/turns.py).
"""
from __future__ import annotations

import json
import operator
import re
import time
from typing import Annotated, TypedDict

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.config import get_stream_writer
from langgraph.graph import END, StateGraph
from langgraph.types import Overwrite, Send

from .config import Config, load_config
from .gittools import git_blame, git_log
from .llm import call_model
from .query_analysis import Analysis, analyze_question, should_analyze
from .retriever import Hit, Retriever
from .symbol_index import SymbolIndex
from .tools import (
    TOOL_SCHEMAS,
    call_graph_tool,
    grep,
    list_dir,
    read_file,
    search_symbols,
    symbol_usages,
)

SYSTEM_PROMPT = """You are CodebaseQA, an assistant that answers questions about the \
LuxTrace C++ codebase using ONLY the provided context and tool results — never invent \
file paths, symbols, or behavior.

Rules:
- Every factual claim about the code must be grounded in a citation of the form \
`path:start_line-end_line` (or `path:line`), taken from the retrieved context or a tool \
result you actually received in this conversation.
- If the retrieved context is insufficient, call a tool (read_file / grep / search_symbols \
/ symbol_usages / list_dir) instead of guessing.
- Prefer `search_symbols` for "where is X defined", `symbol_usages` / `grep` for "what \
calls X" or "where is X used", and `read_file` to see a specific region in full.
- For "why was this changed / who wrote this / when was this added" use `git_log` / \
`git_blame`; cite the commit hash and subject.
- When the user asks for a diagram, flow, or call graph, call `call_graph` and paste its \
`mermaid` output VERBATIM inside a ```mermaid fenced code block, then explain it briefly with \
citations. Never draw edges you did not get from a tool result. The graph is static and \
name-based — say so if an edge you rely on is uncertain.
- Some retrieved chunks are tagged `(callee of X)` / `(caller of X)`: they were added by \
call-graph expansion, not by similarity — use them for context but do not assume they answer \
the question by themselves.
- When you have enough information, answer directly and concisely, with citations inline, \
e.g. "RayTracer::trace (src/core/RayTracer.cpp:2130-2136) calls TraceScene::nearestHit...".
- Do not call more tools once you can answer the question; do not repeat an identical \
tool call.
"""

_SYMBOL_TOKEN_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*(?:::[A-Za-z_][A-Za-z0-9_]*)?\b")
_STOPWORDS = {
    "the", "a", "an", "is", "are", "what", "where", "how", "does", "do", "in", "of",
    "and", "to", "for", "this", "that", "it", "used", "call", "calls", "store", "stores",
    "run", "from", "path", "differ", "result", "function", "functions",
}
CITATION_RE = re.compile(r"[\w./]+\.(?:cpp|h|hpp|cu|cuh|cc|inc):\d+(?:-\d+)?")

TOTAL_SUB_CONTEXT_CHUNKS = 12


class GraphState(TypedDict, total=False):
    question: str
    mode: str  # "qa" | "review" | "tour"
    mode_system: str | None  # replaces SYSTEM_PROMPT for review/tour turns
    mode_context: str | None  # deterministic pre-built context (diff + impact, file skeleton...)
    max_rounds_override: int | None
    messages: list[dict]
    analysis: dict
    hits: list[dict]
    thin: bool
    symbol_resolved: bool | None
    step: int
    max_rounds: int
    forced_tool_this_round: bool
    final_answer: str
    citations: list[str]
    trace: Annotated[list[dict], operator.add]
    sub_results: Annotated[list[dict], operator.add]
    provider: str
    model: str
    module: str | None


def _guess_symbol_candidates(question: str, ignore: set[str] | None = None) -> list[str]:
    tokens = _SYMBOL_TOKEN_RE.findall(question)
    out = []
    for t in tokens:
        if t.lower() in _STOPWORDS or len(t) < 3 or (ignore and t.lower() in ignore):
            continue
        if "::" in t or (t[0].isupper() and any(c.islower() for c in t)) or "_" in t:
            out.append(t)
    return out


def _format_context(hits: list[Hit]) -> str:
    if not hits:
        return "(no retrieval hits)"
    lines = []
    for h in hits:
        via = f" ({h.via})" if h.via else ""
        lines.append(
            f"--- {h.path}:{h.start_line}-{h.end_line} "
            f"[{h.kind} {h.symbol or ''}]{via} ---\n{h.text}"
        )
    return "\n\n".join(lines)


def _hit_trace(h: Hit) -> dict:
    return {
        "path": h.path,
        "start_line": h.start_line,
        "end_line": h.end_line,
        "lines": f"{h.start_line}-{h.end_line}",
        "symbol": h.symbol,
        "score": round(h.rrf_score, 4),
        "rerank": None if h.rerank_score is None else round(h.rerank_score, 3),
        "via": h.via,
    }


def _hit_state(h: Hit) -> dict:
    return {
        "chunk_id": h.chunk_id,
        "path": h.path,
        "start_line": h.start_line,
        "end_line": h.end_line,
        "kind": h.kind,
        "symbol": h.symbol,
        "rrf_score": h.rrf_score,
        "rerank_score": h.rerank_score,
        "via": h.via,
    }


def _ms(t0: float) -> int:
    return int((time.perf_counter() - t0) * 1000)


def _writer():
    """Custom-stream writer (token events), or None when not running under .stream()."""
    try:
        return get_stream_writer()
    except Exception:  # noqa: BLE001 — outside a streaming run
        return None


def build_graph(cfg: Config | None = None, checkpointer: BaseCheckpointSaver | None = None):
    cfg = cfg or load_config()
    retriever = Retriever(cfg)
    symbols = SymbolIndex.load(cfg.storage.index_state_file.parent / "symbols.json")
    acfg = cfg.retrieval.analyze
    # the repo's own name (e.g. "LuxTrace") looks like a PascalCase symbol but never is one
    repo_words = {cfg.repo.root.name.lower()}

    def guess(text: str) -> list[str]:
        return _guess_symbol_candidates(text, ignore=repo_words)

    # ------------------------------------------------------------------ analyze

    def analyze_node(state: GraphState) -> GraphState:
        t0 = time.perf_counter()
        question = state["question"]
        mode = state.get("mode") or "qa"
        provider_id = state.get("provider") or "ollama"
        model = state.get("model") or cfg.models.chat_model
        prior = state.get("messages") or []

        analysis = Analysis(standalone=question)
        entry: dict = {"node": "analyze", "skipped": True}
        if mode == "qa" and acfg.enabled and should_analyze(question, bool(prior), cfg):
            analysis = analyze_question(question, prior, cfg, provider_id, model)
            entry = {
                "node": "analyze",
                "skipped": False,
                "used_llm": analysis.used_llm,
                "standalone": analysis.standalone,
                "queries": analysis.queries,
                "sub_questions": analysis.sub_questions,
                "usage": analysis.usage,
                "error": analysis.error,
            }
        entry["duration_ms"] = _ms(t0)
        return {
            # First node of the turn: reset the per-turn accumulators.
            "trace": Overwrite([entry]),
            "sub_results": Overwrite([]),
            "analysis": {
                "standalone": analysis.standalone,
                "queries": analysis.queries,
                "sub_questions": analysis.sub_questions,
            },
        }

    def route_after_analyze(state: GraphState):
        a = state.get("analysis") or {}
        subs = a.get("sub_questions") or []
        if state.get("mode", "qa") != "qa" or len(subs) < 2:
            return "retrieve"
        module = state.get("module") or None
        # the overall question (idx 0) plus each sub-question, retrieved in parallel
        tasks = [{"idx": 0, "sub_q": a.get("standalone") or state["question"], "queries": a.get("queries") or [],
                  "module": module, "overall": True}]
        tasks += [
            {"idx": i, "sub_q": q, "queries": [], "module": module, "overall": False}
            for i, q in enumerate(subs, start=1)
        ]
        return [Send("sub_retrieve", t) for t in tasks]

    def sub_retrieve_node(task: dict) -> dict:
        t0 = time.perf_counter()
        res = retriever.search_ex(task["sub_q"], extra_queries=task.get("queries") or None,
                                  module=task.get("module"))
        cands = guess(task["sub_q"])
        resolved: bool | None = None
        if cands:
            resolved = any(symbols.lookup(c) for c in cands)
        thin = retriever.is_thin(res.hits, resolved, gate_score=res.gate_score)
        sr = {
            "idx": task["idx"],
            "sub_q": task["sub_q"],
            "overall": task.get("overall", False),
            "hit_ids": [{"id": h.chunk_id, "rrf": h.rrf_score, "rerank": h.rerank_score, "via": h.via}
                        for h in res.hits],
            "thin": thin,
            "gate_score": res.gate_score,
            "symbol_resolved": resolved,
        }
        entry = {
            "node": "sub_retrieve",
            "idx": task["idx"],
            "sub_question": task["sub_q"],
            "overall": task.get("overall", False),
            "hits": [_hit_trace(h) for h in res.hits],
            "thin": thin,
            "reranked": res.reranked,
            "duration_ms": _ms(t0),
        }
        return {"sub_results": [sr], "trace": [entry]}

    # ----------------------------------------------------------------- retrieve

    def retrieve_node(state: GraphState) -> GraphState:
        t0 = time.perf_counter()
        question = state["question"]
        mode = state.get("mode") or "qa"
        module = state.get("module") or None
        analysis = state.get("analysis") or {}
        sub_results = sorted(state.get("sub_results") or [], key=lambda r: r["idx"])

        hits: list[Hit] = []
        groups: list[tuple[str, list[Hit]]] = []
        thin = False
        symbol_resolved: bool | None = None
        reranked = False
        sub_trace: list[dict] = []

        # Only the user's own words can make a symbol "unresolved" (the thin gate): a name the
        # query rewriter introduced must never trigger a needless forced tool round. Names from
        # the rewritten question can only *add* resolved definitions (e.g. "that" -> RayTracer).
        candidates = guess(question)
        standalone = analysis.get("standalone") or question
        extra_candidates = [c for c in guess(standalone) if c not in candidates] if standalone != question else []

        if mode != "qa":
            pass  # review/tour: the deterministic mode_context carries the evidence
        elif sub_results:
            per_sub = max(3, TOTAL_SUB_CONTEXT_CHUNKS // len(sub_results))
            seen: set[str] = set()
            for sr in sub_results:
                gh: list[Hit] = []
                for item in sr["hit_ids"]:
                    if len(gh) >= per_sub:
                        break
                    if item["id"] in seen:
                        continue
                    h = retriever.hit_from_id(item["id"], score=item["rrf"], rerank_score=item["rerank"], via=item["via"])
                    if h is None:
                        continue
                    seen.add(item["id"])
                    gh.append(h)
                label = "Overall question" if sr["overall"] else f"Sub-question {sr['idx']}: {sr['sub_q']}"
                groups.append((label, gh))
                hits.extend(gh)
                sub_trace.append({"idx": sr["idx"], "sub_question": sr["sub_q"], "thin": sr["thin"]})
                for c in guess(sr["sub_q"]):
                    if c not in candidates and c not in extra_candidates:
                        extra_candidates.append(c)
            # A compound question is "thin" if any *sub*-question's own retrieval is weak
            # (the overall phrasing is naturally diffuse, so it is excluded from the gate).
            thin = any(sr["thin"] for sr in sub_results if not sr["overall"])
            reranked = True
        else:
            res = retriever.search_ex(standalone, extra_queries=analysis.get("queries") or None, module=module)
            hits = res.hits
            reranked = res.reranked

        resolved_defs = []
        if mode == "qa":
            if candidates:
                symbol_resolved = False
            for c in candidates + extra_candidates:
                defs = symbols.lookup(c)
                if defs:
                    if candidates:
                        symbol_resolved = True
                    resolved_defs.extend(defs)
        if mode == "qa":
            if sub_results:
                if symbol_resolved is False:
                    thin = True
            else:
                thin = retriever.is_thin(hits, symbol_resolved, gate_score=res.gate_score)

        symbol_block = ""
        if resolved_defs:
            symbol_block = "\n\nExact symbol matches:\n" + "\n".join(
                f"- {d.name} ({d.kind}) {d.path}:{d.start_line}-{d.end_line}" for d in resolved_defs[:10]
            )

        if mode != "qa":
            user_block = f"{question}\n\n{state.get('mode_context') or ''}"
        elif groups:
            body = "\n\n".join(f"### {label}\n{_format_context(gh)}" for label, gh in groups)
            user_block = (
                f"Question: {question}\n\nRetrieved context, grouped by sub-question "
                f"(hybrid BM25+dense, {len(hits)} chunks):\n{body}{symbol_block}"
            )
        else:
            n_ranked = sum(1 for h in hits if h.via is None)
            extra = f" + {len(hits) - n_ranked} via call graph" if len(hits) > n_ranked else ""
            user_block = (
                f"Question: {question}\n\n"
                f"Retrieved context (hybrid BM25+dense{'+rerank' if reranked else ''}, "
                f"top {n_ranked}{extra}):\n{_format_context(hits)}{symbol_block}"
            )

        prior_messages = state.get("messages") or []
        if prior_messages:
            # Follow-up on an existing checkpointed thread — keep the conversation
            # history (including the prior answer) and add this turn on top.
            messages = prior_messages + [{"role": "user", "content": user_block}]
        else:
            system = state.get("mode_system") or SYSTEM_PROMPT
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user_block},
            ]

        entry = {
            "node": "retrieve",
            "mode": mode,
            "hits": [_hit_trace(h) for h in hits],
            "thin": thin,
            "symbol_resolved": symbol_resolved,
            "reranked": reranked,
            "expanded": sum(1 for h in hits if h.via),
            "sub_questions": sub_trace,
            "duration_ms": _ms(t0),
        }
        return {
            "messages": messages,
            "hits": [_hit_state(h) for h in hits],
            "thin": thin,
            "symbol_resolved": symbol_resolved,
            "step": 0,
            "max_rounds": state.get("max_rounds_override") or cfg.graph.max_tool_rounds,
            "trace": [entry],
            "provider": state.get("provider") or "ollama",
            "model": state.get("model") or cfg.models.chat_model,
        }

    # ------------------------------------------------------------------- reason

    def reason_node(state: GraphState) -> GraphState:
        t0 = time.perf_counter()
        messages = list(state["messages"])
        step = state["step"]
        thin = state["thin"] and step == 0  # hard gate only applies to the first round
        provider_id = state.get("provider") or "ollama"
        model = state.get("model") or cfg.models.chat_model

        if thin:
            messages = messages + [
                {
                    "role": "system",
                    "content": (
                        "Retrieval confidence is low for this question (few/weak hits or "
                        "an unresolved symbol name). You MUST call at least one tool "
                        "(search_symbols, grep, symbol_usages, or read_file) before answering."
                    ),
                }
            ]

        w = _writer()
        result = call_model(
            cfg, provider_id, model, messages, TOOL_SCHEMAS,
            on_token=(lambda t: w({"type": "token", "text": t})) if w else None,
            on_reset=(lambda: w({"type": "token_reset"})) if w else None,
        )
        assistant_entry = {"role": "assistant", "content": result.content or ""}
        tool_calls = result.tool_calls

        forced = False
        if thin and not tool_calls:
            # Hard-gate backstop: the model ignored the nudge — force one tool call
            # ourselves so "thin retrieval => visible tool use" always holds (§3.1).
            forced = True
            candidates = guess(state["question"])
            args = {"name": candidates[0]} if candidates else {"name": state["question"][:40]}
            tool_calls = [
                {
                    "function": {"name": "search_symbols", "arguments": args},
                    "id": "forced_0",
                }
            ]
            if not assistant_entry["content"]:
                assistant_entry["content"] = "(low-confidence retrieval — checking the symbol index)"

        assistant_entry["tool_calls"] = tool_calls
        # `messages` already carries the thin-retrieval nudge (if any) — reuse it so the
        # conversation the model saw matches what gets persisted.
        new_messages = messages + [assistant_entry]

        entry = {
            "node": "reason",
            "step": step,
            "tool_calls": [tc.get("function", {}).get("name") for tc in tool_calls],
            "forced": forced,
            "content_preview": (assistant_entry["content"] or "")[:200],
            "usage": result.usage,
            "duration_ms": _ms(t0),
        }
        return {"messages": new_messages, "trace": [entry], "forced_tool_this_round": forced}

    # -------------------------------------------------------------------- tools

    def _run_tool(name: str, args: dict) -> dict:
        try:
            if name == "read_file":
                return read_file(cfg, **args)
            if name == "list_dir":
                return list_dir(cfg, **args)
            if name == "grep":
                return grep(cfg, **args)
            if name == "search_symbols":
                return search_symbols(symbols, **args)
            if name == "symbol_usages":
                return symbol_usages(cfg, **args)
            if name == "call_graph":
                return call_graph_tool(retriever.graph, symbols, **args)
            if name == "git_log":
                return git_log(cfg, **args)
            if name == "git_blame":
                return git_blame(cfg, **args)
            return {"error": f"unknown tool {name}"}
        except TypeError as e:
            return {"error": f"bad arguments for {name}: {e}"}
        except Exception as e:  # noqa: BLE001 — a tool failure is a tool result, not a crash
            return {"error": f"{name} failed: {type(e).__name__}: {e}"}

    def tools_node(state: GraphState) -> GraphState:
        t0 = time.perf_counter()
        messages = list(state["messages"])
        last = messages[-1]
        tool_calls = last.get("tool_calls") or []
        results_trace = []

        for i, tc in enumerate(tool_calls):
            fn = tc.get("function", {})
            name = fn.get("name")
            args = fn.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            tt = time.perf_counter()
            result = _run_tool(name, args)
            content = json.dumps(result)
            results_trace.append(
                {
                    "tool": name,
                    "args": args,
                    "result_preview": str(result)[:300],
                    "result_chars": len(content),
                    "ms": _ms(tt),
                }
            )
            messages.append(
                {
                    "role": "tool",
                    # keep call_graph output whole: its mermaid block must not be cut mid-edge
                    "content": content[:12000] if name == "call_graph" else content[:6000],
                    "tool_call_id": tc.get("id", f"call_{i}"),
                }
            )

        entry = {"node": "tools", "calls": results_trace, "duration_ms": _ms(t0)}
        return {"messages": messages, "step": state["step"] + 1, "trace": [entry]}

    # ------------------------------------------------------------------- answer

    def answer_node(state: GraphState) -> GraphState:
        t0 = time.perf_counter()
        messages = state["messages"]
        provider_id = state.get("provider") or "ollama"
        model = state.get("model") or cfg.models.chat_model
        usage: dict = {}
        # If we hit the round budget without a final answer, ask once more with no tools.
        if state["step"] >= state["max_rounds"]:
            budget_messages = messages + [
                {
                    "role": "system",
                    "content": "Tool-call budget reached. Answer now with what you have, "
                    "citing sources, and note if something couldn't be confirmed.",
                }
            ]
            w = _writer()
            result = call_model(
                cfg, provider_id, model, budget_messages, [],
                on_token=(lambda t: w({"type": "token", "text": t})) if w else None,
                on_reset=(lambda: w({"type": "token_reset"})) if w else None,
            )
            content, usage = result.content, result.usage
        else:
            content = messages[-1].get("content", "") if messages[-1]["role"] == "assistant" else ""
            if not content:
                result = call_model(cfg, provider_id, model, messages, TOOL_SCHEMAS)
                content, usage = result.content, result.usage

        citations = CITATION_RE.findall(content)
        entry = {"node": "answer", "citations": citations, "duration_ms": _ms(t0)}
        if usage:
            entry["usage"] = usage
        # The state's message history must end with the final assistant turn so follow-ups
        # (and the budget path, where the answer was produced here) keep it.
        out: dict = {"final_answer": content, "citations": citations, "trace": [entry]}
        if state["step"] >= state["max_rounds"] and content:
            out["messages"] = list(messages) + [{"role": "assistant", "content": content}]
        return out

    # ------------------------------------------------------------------ routing

    def route_after_reason(state: GraphState) -> str:
        last = state["messages"][-1]
        if last.get("role") == "assistant" and last.get("tool_calls"):
            return "tools"
        return "answer"

    def route_after_tools(state: GraphState) -> str:
        if state["step"] >= state["max_rounds"]:
            return "answer"
        return "reason"

    graph = StateGraph(GraphState)
    graph.add_node("analyze", analyze_node)
    graph.add_node("sub_retrieve", sub_retrieve_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("reason", reason_node)
    graph.add_node("tools", tools_node)
    graph.add_node("answer", answer_node)

    graph.set_entry_point("analyze")
    graph.add_conditional_edges("analyze", route_after_analyze, ["retrieve", "sub_retrieve"])
    graph.add_edge("sub_retrieve", "retrieve")
    graph.add_edge("retrieve", "reason")
    graph.add_conditional_edges("reason", route_after_reason, {"tools": "tools", "answer": "answer"})
    graph.add_conditional_edges("tools", route_after_tools, {"reason": "reason", "answer": "answer"})
    graph.add_edge("answer", END)

    return graph.compile(checkpointer=checkpointer or MemorySaver())
