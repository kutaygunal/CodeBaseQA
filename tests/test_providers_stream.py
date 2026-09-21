"""Provider streaming: delta accumulation, tool-call reassembly and usage, on canned chunks."""
from __future__ import annotations

from types import SimpleNamespace as NS

from cqa import providers
from cqa.providers import ChatResult, OllamaProvider, OpenAIProvider, chat_with_optional_stream


def test_ollama_stream_accumulates_text_tools_and_usage(monkeypatch):
    chunks = [
        {"message": {"content": "Hel"}, "done": False},
        {"message": {"content": "lo"}, "done": False},
        {"message": {"content": "", "tool_calls": [{"function": {"name": "grep", "arguments": {"pattern": "x"}}}]}, "done": False},
        {"message": {"content": ""}, "done": True, "prompt_eval_count": 12, "eval_count": 7},
    ]
    monkeypatch.setattr(providers.ollama, "chat", lambda **kw: iter(chunks))
    seen = []
    res = chat_with_optional_stream(OllamaProvider(), "m", [], [], seen.append)
    assert seen == ["Hel", "lo"] and res.content == "Hello"
    assert res.tool_calls == [{"id": "call_0", "function": {"name": "grep", "arguments": {"pattern": "x"}}}]
    assert res.usage == {"in": 12, "out": 7}


def _oi_chunk(content=None, tool=None, usage=None):
    choices = [NS(delta=NS(content=content, tool_calls=tool))] if (content is not None or tool) else []
    return NS(choices=choices, usage=usage)


def test_openai_stream_reassembles_split_tool_call_arguments():
    tc = lambda idx, id=None, name=None, args=None: NS(index=idx, id=id, function=NS(name=name, arguments=args))  # noqa: E731
    chunks = [
        _oi_chunk("Working"),
        _oi_chunk(tool=[tc(0, "call_a", "read_file", '{"path": "src/')]),
        _oi_chunk(tool=[tc(0, None, None, 'core/A.cpp", "start_line": 3}')]),
        _oi_chunk(tool=[tc(1, "call_b", "grep", '{"pattern": "y"}')]),
        _oi_chunk(usage=NS(prompt_tokens=50, completion_tokens=9)),
    ]
    prov = OpenAIProvider.__new__(OpenAIProvider)
    prov._client = NS(chat=NS(completions=NS(create=lambda **kw: iter(chunks))))
    seen = []
    res = chat_with_optional_stream(prov, "m", [{"role": "user", "content": "hi"}], [{"type": "function"}], seen.append)
    assert seen == ["Working"] and res.usage == {"in": 50, "out": 9}
    assert [t["function"]["name"] for t in res.tool_calls] == ["read_file", "grep"]
    assert res.tool_calls[0]["function"]["arguments"] == {"path": "src/core/A.cpp", "start_line": 3}


def test_without_a_token_callback_the_blocking_path_is_used():
    class OnlyChat:
        def chat(self, model, messages, tools):
            return ChatResult(content="plain")

    assert chat_with_optional_stream(OnlyChat(), "m", [], [], None).content == "plain"
    assert chat_with_optional_stream(OnlyChat(), "m", [], [], lambda t: None).content == "plain"  # no chat_stream


def test_call_model_falls_back_and_resets_partial_stream(monkeypatch, cfg):
    from cqa import llm

    class Flaky:
        def chat_stream(self, model, messages, tools):
            yield "partial "
            raise RuntimeError("boom")

    class Good:
        def chat_stream(self, model, messages, tools):
            yield "clean"
            yield ChatResult(content="clean")

    monkeypatch.setattr(llm, "get_provider", lambda pid: Flaky() if pid == "openai" else Good())
    seen, resets = [], []
    res = llm.call_model(cfg, "openai", "gpt", [], on_token=seen.append, on_reset=lambda: resets.append(1))
    assert res.content == "clean" and resets == [1] and seen == ["partial ", "clean"]
