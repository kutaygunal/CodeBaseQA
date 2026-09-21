"""High-level ask()/stream() entry points over the compiled graph, with
durable checkpointed follow-ups, a thread (conversation) store, and a per-turn
store that records every answer with its trace, timing, token usage and cost."""
from __future__ import annotations

import re
import sqlite3
import time
import uuid
from typing import Iterator

from langgraph.checkpoint.sqlite import SqliteSaver

from .callgraph import CallGraph
from .config import Config, load_config
from .graph import CITATION_RE, build_graph
from .onboard import TOUR_SYSTEM_PROMPT, TourPrep, index_key, prepare_tour
from .retriever import Retriever
from .review import REVIEW_SYSTEM_PROMPT, ReviewPrep, prepare_review
from .symbol_index import SymbolIndex
from .threads import ThreadStore
from .turns import TurnStore

_CITATION_RE = CITATION_RE


class Agent:
    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or load_config()
        db_path = self.cfg.storage.index_state_file.parent / "conversations.sqlite"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        checkpointer = SqliteSaver(self._conn)
        checkpointer.setup()
        self.threads = ThreadStore(self._conn)
        self.turns = TurnStore(self._conn)
        self.app = build_graph(self.cfg, checkpointer=checkpointer)
        self.modules = Retriever(self.cfg).modules
        index_dir = self.cfg.storage.index_state_file.parent
        self.symbols = SymbolIndex.load(index_dir / "symbols.json")
        graph_path = index_dir / "graph.json"
        self.callgraph: CallGraph | None = CallGraph.load(graph_path) if graph_path.exists() else None

    # --- helpers ---------------------------------------------------------

    def _input(
        self,
        question: str,
        provider: str,
        model: str | None,
        module: str | None,
        extra: dict | None,
    ) -> dict:
        state = {
            "question": question,
            "provider": provider,
            "model": model or self.cfg.models.chat_model,
            "module": module,
            "mode": "qa",
            "mode_system": None,
            "mode_context": None,
            "max_rounds_override": None,
        }
        state.update(extra or {})
        return state

    def _finish(
        self, thread_id: str, question: str, state_in: dict, answer: str, citations: list[str],
        trace: list[dict], t0: float,
    ) -> dict:
        self.threads.touch(thread_id, first_question=question)
        meta = self.turns.record(
            thread_id=thread_id, question=question, answer=answer, citations=citations, trace=trace,
            provider=state_in["provider"], model=state_in["model"], module=state_in["module"],
            mode=state_in["mode"], latency_ms=int((time.perf_counter() - t0) * 1000),
        )
        return meta

    def delete_thread(self, thread_id: str) -> None:
        self.turns.delete_thread(thread_id)
        self.threads.delete(thread_id)

    # --- ask / stream ----------------------------------------------------

    def ask(
        self,
        question: str,
        thread_id: str | None = None,
        provider: str = "ollama",
        model: str | None = None,
        module: str | None = None,
        extra: dict | None = None,
    ) -> dict:
        """Ask a question. Pass the same thread_id back for follow-ups (checkpointed)."""
        thread_id = thread_id or str(uuid.uuid4())
        self.threads.ensure(thread_id)
        config = {"configurable": {"thread_id": thread_id}}
        state_in = self._input(question, provider, model, module, extra)
        t0 = time.perf_counter()
        result = self.app.invoke(state_in, config=config)
        answer = result.get("final_answer", "")
        citations = result.get("citations", [])
        trace = result.get("trace", [])
        meta = self._finish(thread_id, question, state_in, answer, citations, trace, t0)
        return {
            "thread_id": thread_id,
            "answer": answer,
            "citations": citations,
            "trace": trace,
            "turn_id": meta["turn_id"],
            "metrics": {k: v for k, v in meta.items() if k != "turn_id"},
        }

    def stream(
        self,
        question: str,
        thread_id: str | None = None,
        provider: str = "ollama",
        model: str | None = None,
        module: str | None = None,
        extra: dict | None = None,
    ) -> Iterator[dict]:
        """Yield live progress events {type, ...} as the graph runs:

            analyze | retrieve | reason | tools    node progress (one per node update)
            token / token_reset                     answer text as it is generated
            final                                   {thread_id, answer, citations, trace, turn_id, metrics}
            error

        Used by the web UI to show retrieval/tool-use steps and stream the answer."""
        thread_id = thread_id or str(uuid.uuid4())
        self.threads.ensure(thread_id)
        config = {"configurable": {"thread_id": thread_id}}
        state_in = self._input(question, provider, model, module, extra)
        t0 = time.perf_counter()
        try:
            for mode, payload in self.app.stream(state_in, config=config, stream_mode=["updates", "custom"]):
                if mode == "custom":
                    if isinstance(payload, dict) and payload.get("type") in ("token", "token_reset"):
                        yield payload
                    continue
                for node_name, partial in payload.items():
                    partial = partial or {}
                    if node_name == "analyze":
                        entry = _first(partial.get("trace"))
                        yield {
                            "type": "analyze",
                            "skipped": entry.get("skipped", True),
                            "standalone": entry.get("standalone"),
                            "queries": entry.get("queries", []),
                            "sub_questions": entry.get("sub_questions", []),
                            "error": entry.get("error"),
                        }
                    elif node_name == "sub_retrieve":
                        entry = _first(partial.get("trace"))
                        yield {
                            "type": "retrieve",
                            "sub_question": entry.get("sub_question"),
                            "hits": entry.get("hits", []),
                            "thin": entry.get("thin"),
                        }
                    elif node_name == "retrieve":
                        entry = _first(partial.get("trace"))
                        yield {
                            "type": "retrieve",
                            "hits": entry.get("hits", []),
                            "thin": partial.get("thin"),
                            "symbol_resolved": partial.get("symbol_resolved"),
                            "reranked": entry.get("reranked"),
                            "expanded": entry.get("expanded", 0),
                            "joined": bool(entry.get("sub_questions")),
                            "mode": entry.get("mode"),
                        }
                    elif node_name == "reason":
                        msgs = partial.get("messages", [])
                        last_msg = msgs[-1] if msgs else {}
                        tool_calls = last_msg.get("tool_calls") or []
                        yield {
                            "type": "reason",
                            "forced": partial.get("forced_tool_this_round", False),
                            "tool_calls": [(tc.get("function") or {}).get("name") for tc in tool_calls],
                            "content_preview": (last_msg.get("content") or "")[:200],
                        }
                    elif node_name == "tools":
                        entry = _first(partial.get("trace"))
                        yield {"type": "tools", "calls": entry.get("calls", [])}
                    elif node_name == "answer":
                        # The checkpoint holds the fully reduced per-turn trace.
                        snap = self.app.get_state(config)
                        trace = (snap.values or {}).get("trace", [])
                        answer = partial.get("final_answer", "")
                        citations = partial.get("citations", [])
                        meta = self._finish(thread_id, question, state_in, answer, citations, trace, t0)
                        yield {
                            "type": "final",
                            "thread_id": thread_id,
                            "answer": answer,
                            "citations": citations,
                            "trace": trace,
                            "turn_id": meta["turn_id"],
                            "metrics": {k: v for k, v in meta.items() if k != "turn_id"},
                        }
        except Exception as e:  # noqa: BLE001
            yield {"type": "error", "message": str(e)}

    # --- review mode -----------------------------------------------------

    def prepare_review(self, base: str | None = None, head: str | None = None, diff: str | None = None) -> ReviewPrep:
        """Deterministic diff analysis (no LLM). Raises ValueError with a user-facing message."""
        return prepare_review(self.cfg, self.symbols, self.callgraph, base=base, head=head, diff=diff)

    def review_extra(self, prep: ReviewPrep) -> dict:
        return {
            "mode": "review",
            "mode_system": REVIEW_SYSTEM_PROMPT,
            "mode_context": prep.context,
            "max_rounds_override": self.cfg.review.max_tool_rounds,
        }

    def stream_review(self, prep: ReviewPrep, thread_id: str | None = None, provider: str = "ollama",
                      model: str | None = None) -> Iterator[dict]:
        yield from self.stream(prep.title, thread_id=thread_id, provider=provider, model=model,
                               extra=self.review_extra(prep))

    # --- tour mode -------------------------------------------------------

    def prepare_tour(self, target: str) -> TourPrep:
        return prepare_tour(self.cfg, self.symbols, self.callgraph, target)

    def stream_tour(self, prep: TourPrep, thread_id: str | None = None, provider: str = "ollama",
                    model: str | None = None, refresh: bool = False) -> Iterator[dict]:
        """Tour for a file/module. Cached per (target, index build, provider, model): a hit
        returns instantly, and still seeds the thread so follow-up questions have the tour as context."""
        model = model or self.cfg.models.chat_model
        key = index_key(self.cfg)
        thread_id = thread_id or str(uuid.uuid4())
        cached = None if refresh else self.turns.get_tour(prep.target, key, provider, model)
        extra = {
            "mode": "tour",
            "mode_system": TOUR_SYSTEM_PROMPT,
            "mode_context": prep.context,
            "max_rounds_override": self.cfg.review.max_tool_rounds,
        }
        if cached:
            t0 = time.perf_counter()
            self.threads.ensure(thread_id)
            config = {"configurable": {"thread_id": thread_id}}
            snap = self.app.get_state(config)
            prior = list((snap.values or {}).get("messages", [])) if snap else []
            seeded = prior or [{"role": "system", "content": TOUR_SYSTEM_PROMPT}]
            seeded = seeded + [
                {"role": "user", "content": prep.title + "\n\n" + prep.context},
                {"role": "assistant", "content": cached["markdown"]},
            ]
            trace = [{"node": "cache", "cached": True, "target": prep.target, "duration_ms": 0}]
            self.app.update_state(
                config,
                {"messages": seeded, "final_answer": cached["markdown"], "citations": cached["citations"], "trace": trace},
                as_node="answer",
            )
            state_in = self._input(prep.title, provider, model, None, extra)
            meta = self._finish(thread_id, prep.title, state_in, cached["markdown"], cached["citations"], trace, t0)
            yield {
                "type": "final", "thread_id": thread_id, "answer": cached["markdown"],
                "citations": cached["citations"], "trace": trace, "turn_id": meta["turn_id"],
                "metrics": {k: v for k, v in meta.items() if k != "turn_id"}, "cached": True,
            }
            return
        for ev in self.stream(prep.title, thread_id=thread_id, provider=provider, model=model, extra=extra):
            if ev.get("type") == "final" and ev.get("answer"):
                self.turns.put_tour(prep.target, key, provider, model, ev["answer"], ev.get("citations", []))
            yield ev

    # --- history ---------------------------------------------------------

    def history(self, thread_id: str) -> list[dict]:
        """A thread's turns for restoring a conversation in the UI. Prefers the per-turn
        store (full trace, metrics, feedback state); falls back to reconstructing
        question/answer text from the LangGraph checkpoint for threads that predate it."""
        stored = self.turns.for_thread(thread_id)
        if stored:
            out: list[dict] = []
            for t in stored:
                out.append({"role": "user", "content": t["question"], "mode": t["mode"]})
                out.append(
                    {
                        "role": "assistant",
                        "content": t["answer"],
                        "citations": t["citations"],
                        "trace": t["trace"],
                        "turn_id": t["turn_id"],
                        "metrics": t["metrics"],
                        "feedback": t["feedback"],
                    }
                )
            return out
        return self._history_from_checkpoint(thread_id)

    def _history_from_checkpoint(self, thread_id: str) -> list[dict]:
        config = {"configurable": {"thread_id": thread_id}}
        snapshot = self.app.get_state(config)
        messages = (snapshot.values or {}).get("messages", []) if snapshot else []
        turns: list[dict] = []
        for m in messages:
            role = m.get("role")
            if role == "user":
                content = m.get("content") or ""
                # The first user turn is wrapped with retrieved-context boilerplate —
                # pull just the original question back out.
                q_match = re.match(r"Question: (.*?)\n\nRetrieved context", content, re.S)
                turns.append({"role": "user", "content": q_match.group(1) if q_match else content})
            elif role == "assistant" and not m.get("tool_calls"):
                content = m.get("content") or ""
                if content:
                    turns.append(
                        {"role": "assistant", "content": content, "citations": _CITATION_RE.findall(content)}
                    )
        return turns


def _first(v) -> dict:
    """First trace entry of a node update (the update may wrap it in Overwrite)."""
    if v is None:
        return {}
    v = getattr(v, "value", v)  # unwrap langgraph.types.Overwrite
    return v[0] if v else {}
