"""High-level ask()/stream() entry points over the compiled graph, with
durable checkpointed follow-ups and a thread (conversation) store."""
from __future__ import annotations

import re
import sqlite3
import uuid
from typing import Iterator

from langgraph.checkpoint.sqlite import SqliteSaver

from .config import Config, load_config
from .graph import build_graph
from .retriever import Retriever
from .threads import ThreadStore

_CITATION_RE = re.compile(r"[\w./]+\.(?:cpp|h|hpp|cu|cuh|cc|inc):\d+(?:-\d+)?")


class Agent:
    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or load_config()
        db_path = self.cfg.storage.index_state_file.parent / "conversations.sqlite"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        checkpointer = SqliteSaver(self._conn)
        checkpointer.setup()
        self.threads = ThreadStore(self._conn)
        self.app = build_graph(self.cfg, checkpointer=checkpointer)
        self.modules = Retriever(self.cfg).modules

    def ask(
        self,
        question: str,
        thread_id: str | None = None,
        provider: str = "ollama",
        model: str | None = None,
        module: str | None = None,
    ) -> dict:
        """Ask a question. Pass the same thread_id back for follow-ups (checkpointed)."""
        thread_id = thread_id or str(uuid.uuid4())
        self.threads.ensure(thread_id)
        config = {"configurable": {"thread_id": thread_id}}
        result = self.app.invoke(
            {
                "question": question,
                "provider": provider,
                "model": model or self.cfg.models.chat_model,
                "module": module,
            },
            config=config,
        )
        self.threads.touch(thread_id, first_question=question)
        return {
            "thread_id": thread_id,
            "answer": result.get("final_answer", ""),
            "citations": result.get("citations", []),
            "trace": result.get("trace", []),
        }

    def stream(
        self,
        question: str,
        thread_id: str | None = None,
        provider: str = "ollama",
        model: str | None = None,
        module: str | None = None,
    ) -> Iterator[dict]:
        """Yield live progress events {type, ...} as the graph runs, node by node,
        ending with one {"type": "final", "thread_id", "answer", "citations", "trace"}.
        Used by the web UI to show retrieval/tool-use steps as they happen."""
        thread_id = thread_id or str(uuid.uuid4())
        self.threads.ensure(thread_id)
        config = {"configurable": {"thread_id": thread_id}}
        last_state: dict = {}
        try:
            for update in self.app.stream(
                {
                    "question": question,
                    "provider": provider,
                    "model": model or self.cfg.models.chat_model,
                    "module": module,
                },
                config=config,
                stream_mode="updates",
            ):
                for node_name, partial in update.items():
                    last_state.update(partial)
                    if node_name == "retrieve":
                        yield {
                            "type": "retrieve",
                            "hits": partial.get("hits", []),
                            "thin": partial.get("thin"),
                            "symbol_resolved": partial.get("symbol_resolved"),
                        }
                    elif node_name == "reason":
                        msgs = partial.get("messages", [])
                        last_msg = msgs[-1] if msgs else {}
                        tool_calls = last_msg.get("tool_calls") or []
                        yield {
                            "type": "reason",
                            "forced": partial.get("forced_tool_this_round", False),
                            "tool_calls": [
                                (tc.get("function") or {}).get("name")
                                if hasattr(tc, "get")
                                else tc["function"]["name"]
                                for tc in tool_calls
                            ],
                            "content_preview": (last_msg.get("content") or "")[:200],
                        }
                    elif node_name == "tools":
                        trace = partial.get("trace", [])
                        last_tool_entry = trace[-1] if trace else {}
                        yield {"type": "tools", "calls": last_tool_entry.get("calls", [])}
                    elif node_name == "answer":
                        self.threads.touch(thread_id, first_question=question)
                        yield {
                            "type": "final",
                            "thread_id": thread_id,
                            "answer": partial.get("final_answer", ""),
                            "citations": partial.get("citations", []),
                            "trace": last_state.get("trace", []),
                        }
        except Exception as e:  # noqa: BLE001
            yield {"type": "error", "message": str(e)}

    def history(self, thread_id: str) -> list[dict]:
        """Reconstruct a thread's user/assistant turns from the checkpointed state,
        for restoring a conversation when the web UI switches to it. The original
        per-turn trace isn't stored separately, so only the question/answer text and
        re-derived citations come back (no tool-call trace for past turns)."""
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
                        {
                            "role": "assistant",
                            "content": content,
                            "citations": _CITATION_RE.findall(content),
                        }
                    )
        return turns
