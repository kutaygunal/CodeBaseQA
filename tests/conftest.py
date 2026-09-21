"""Shared test setup: make `cqa` importable and provide a scripted fake chat provider."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cqa.config import load_config  # noqa: E402
from cqa.providers import ChatResult  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def index_dir(cfg):
    d = cfg.storage.index_state_file.parent
    if not (d / "bm25.pkl").exists() or not (d / "graph.json").exists():
        pytest.skip("index not built — run scripts/index_luxtrace.py first")
    return d


class FakeProvider:
    """Scripted chat provider. `analysis` is the JSON dict returned to the query-analysis
    prompt; every other call is treated as the reasoning step: it returns `tool_script`
    calls first (one list per round) and then `answer`. Streams the answer word by word."""

    def __init__(self, answer="It works (src/core/ThreadPool.cpp:51-81).", analysis=None, tool_script=None):
        self.answer = answer
        self.analysis = analysis
        self.tool_script = list(tool_script or [])
        self.calls: list[list[dict]] = []

    def _next(self, messages, tools) -> ChatResult:
        self.calls.append(messages)
        last = messages[-1].get("content", "") if messages else ""
        if "You prepare a question" in last:
            import json

            return ChatResult(content=json.dumps(self.analysis or {}), usage={"in": 10, "out": 5})
        if tools and self.tool_script:
            calls = self.tool_script.pop(0)
            return ChatResult(
                content="",
                tool_calls=[
                    {"id": f"c{i}", "function": {"name": n, "arguments": a}} for i, (n, a) in enumerate(calls)
                ],
                usage={"in": 20, "out": 3},
            )
        return ChatResult(content=self.answer, usage={"in": 30, "out": 12})

    def chat(self, model, messages, tools):
        return self._next(messages, tools)

    def chat_stream(self, model, messages, tools):
        res = self._next(messages, tools)
        for w in res.content.split(" "):
            yield w + " "
        yield res


@pytest.fixture
def fake_llm(monkeypatch):
    """Install a FakeProvider for every provider id; returns a factory to configure it."""

    def install(**kw) -> FakeProvider:
        p = FakeProvider(**kw)
        monkeypatch.setattr("cqa.llm.get_provider", lambda provider_id: p)
        return p

    return install
