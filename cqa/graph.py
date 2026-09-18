"""LangGraph state machine: RETRIEVE -> REASON -> (TOOLS)* -> ANSWER.

See PLAN.md §3 (architecture) and §3.1 (the "confidence"/thin-retrieval gates).
"""
from __future__ import annotations

import json
import re
from typing import TypedDict

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from .config import Config, load_config
from .providers import ChatResult, get_provider
from .retriever import Hit, Retriever
from .symbol_index import SymbolIndex
from .tools import TOOL_SCHEMAS, grep, list_dir, read_file, search_symbols, symbol_usages

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


class GraphState(TypedDict, total=False):
    question: str
    messages: list[dict]
    hits: list[dict]
    thin: bool
    symbol_resolved: bool | None
    step: int
    max_rounds: int
    forced_tool_this_round: bool
    final_answer: str
    citations: list[str]
    trace: list[dict]
    provider: str
    model: str
    module: str | None


def _guess_symbol_candidates(question: str) -> list[str]:
    tokens = _SYMBOL_TOKEN_RE.findall(question)
    out = []
    for t in tokens:
        if t.lower() in _STOPWORDS or len(t) < 3:
            continue
        if "::" in t or (t[0].isupper() and any(c.islower() for c in t)) or "_" in t:
            out.append(t)
    return out


def _format_context(hits: list[Hit]) -> str:
    if not hits:
        return "(no retrieval hits)"
    lines = []
    for h in hits:
        lines.append(
            f"--- {h.path}:{h.start_line}-{h.end_line} "
            f"[{h.kind} {h.symbol or ''}] ---\n{h.text}"
        )
    return "\n\n".join(lines)


def _call_model(cfg: Config, provider_id: str, model: str, messages: list[dict]) -> ChatResult:
    try:
        return get_provider(provider_id).chat(model, messages, TOOL_SCHEMAS)
    except Exception:
        if provider_id == "ollama" and model == cfg.models.chat_model_fallback:
            raise
        # Fall back to the local Ollama model so a bad/unavailable provider
        # selection never hard-fails the whole answer.
        return get_provider("ollama").chat(cfg.models.chat_model_fallback, messages, TOOL_SCHEMAS)


def build_graph(cfg: Config | None = None, checkpointer: BaseCheckpointSaver | None = None):
    cfg = cfg or load_config()
    retriever = Retriever(cfg)
    symbols = SymbolIndex.load(cfg.storage.index_state_file.parent / "symbols.json")

    def retrieve_node(state: GraphState) -> GraphState:
        question = state["question"]
        module = state.get("module") or None
        hits = retriever.search(question, module=module)
        candidates = _guess_symbol_candidates(question)
        symbol_resolved: bool | None = None
        resolved_defs = []
        if candidates:
            symbol_resolved = False
            for c in candidates:
                defs = symbols.lookup(c)
                if defs:
                    symbol_resolved = True
                    resolved_defs.extend(defs)
        thin = retriever.is_thin(hits, symbol_resolved)

        context = _format_context(hits)
        symbol_block = ""
        if resolved_defs:
            symbol_block = "\n\nExact symbol matches:\n" + "\n".join(
                f"- {d.name} ({d.kind}) {d.path}:{d.start_line}-{d.end_line}"
                for d in resolved_defs[:10]
            )

        user_block = (
            f"Question: {question}\n\n"
            f"Retrieved context (hybrid BM25+dense, top {len(hits)}):\n{context}"
            f"{symbol_block}"
        )
        prior_messages = state.get("messages") or []
        if prior_messages:
            # Follow-up on an existing checkpointed thread — keep the conversation
            # history (including the prior answer) and add this turn on top.
            messages = prior_messages + [{"role": "user", "content": user_block}]
        else:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_block},
            ]
        trace = [
            {
                "node": "retrieve",
                "hits": [
                    {
                        "path": h.path,
                        "start_line": h.start_line,
                        "end_line": h.end_line,
                        "lines": f"{h.start_line}-{h.end_line}",
                        "symbol": h.symbol,
                        "score": round(h.rrf_score, 4),
                    }
                    for h in hits
                ],
                "thin": thin,
                "symbol_resolved": symbol_resolved,
            }
        ]
        return {
            "messages": messages,
            "hits": [
                {
                    "chunk_id": h.chunk_id,
                    "path": h.path,
                    "start_line": h.start_line,
                    "end_line": h.end_line,
                    "kind": h.kind,
                    "symbol": h.symbol,
                    "rrf_score": h.rrf_score,
                }
                for h in hits
            ],
            "thin": thin,
            "symbol_resolved": symbol_resolved,
            "step": 0,
            "max_rounds": cfg.graph.max_tool_rounds,
            "trace": trace,
            "provider": state.get("provider") or "ollama",
            "model": state.get("model") or cfg.models.chat_model,
        }

    def reason_node(state: GraphState) -> GraphState:
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

        result = _call_model(cfg, provider_id, model, messages)
        assistant_entry = {"role": "assistant", "content": result.content or ""}
        tool_calls = result.tool_calls

        forced = False
        if thin and not tool_calls:
            # Hard-gate backstop: the model ignored the nudge — force one tool call
            # ourselves so "thin retrieval => visible tool use" always holds (§3.1).
            forced = True
            candidates = _guess_symbol_candidates(state["question"])
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

        trace = list(state.get("trace", []))
        trace.append(
            {
                "node": "reason",
                "step": step,
                "tool_calls": [tc.get("function", {}).get("name") for tc in tool_calls],
                "forced": forced,
                "content_preview": (assistant_entry["content"] or "")[:200],
            }
        )
        return {"messages": new_messages, "trace": trace, "forced_tool_this_round": forced}

    def tools_node(state: GraphState) -> GraphState:
        messages = list(state["messages"])
        last = messages[-1]
        tool_calls = last.get("tool_calls") or []
        trace = list(state.get("trace", []))
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
            try:
                if name == "read_file":
                    result = read_file(cfg, **args)
                elif name == "list_dir":
                    result = list_dir(cfg, **args)
                elif name == "grep":
                    result = grep(cfg, **args)
                elif name == "search_symbols":
                    result = search_symbols(symbols, **args)
                elif name == "symbol_usages":
                    result = symbol_usages(cfg, **args)
                else:
                    result = {"error": f"unknown tool {name}"}
            except TypeError as e:
                result = {"error": f"bad arguments for {name}: {e}"}

            results_trace.append({"tool": name, "args": args, "result_preview": str(result)[:300]})
            messages.append(
                {
                    "role": "tool",
                    "content": json.dumps(result)[:6000],
                    "tool_call_id": tc.get("id", f"call_{i}"),
                }
            )

        trace.append({"node": "tools", "calls": results_trace})
        return {"messages": messages, "step": state["step"] + 1, "trace": trace}

    def answer_node(state: GraphState) -> GraphState:
        messages = state["messages"]
        provider_id = state.get("provider") or "ollama"
        model = state.get("model") or cfg.models.chat_model
        # If we hit the round budget without a final answer, ask once more with no tools.
        if state["step"] >= state["max_rounds"]:
            budget_messages = messages + [
                {
                    "role": "system",
                    "content": "Tool-call budget reached. Answer now with what you have, "
                    "citing sources, and note if something couldn't be confirmed.",
                }
            ]
            result = get_provider(provider_id).chat(model, budget_messages, [])
            content = result.content
        else:
            content = messages[-1].get("content", "") if messages[-1]["role"] == "assistant" else ""
            if not content:
                result = _call_model(cfg, provider_id, model, messages)
                content = result.content

        citations = re.findall(r"[\w./]+\.(?:cpp|h|hpp|cu|cuh|cc|inc):\d+(?:-\d+)?", content)
        trace = list(state.get("trace", []))
        trace.append({"node": "answer", "citations": citations})
        return {"final_answer": content, "citations": citations, "trace": trace}

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
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("reason", reason_node)
    graph.add_node("tools", tools_node)
    graph.add_node("answer", answer_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "reason")
    graph.add_conditional_edges("reason", route_after_reason, {"tools": "tools", "answer": "answer"})
    graph.add_conditional_edges("tools", route_after_tools, {"reason": "reason", "answer": "answer"})
    graph.add_edge("answer", END)

    return graph.compile(checkpointer=checkpointer or MemorySaver())
