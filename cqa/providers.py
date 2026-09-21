"""Chat-model provider abstraction: Ollama (local + cloud-routed), OpenAI,
Anthropic, and any OpenAI-compatible endpoint (e.g. GitHub Models), behind one
normalized interface so cqa/graph.py doesn't care who's answering.

Normalized response shape: {"content": str, "tool_calls": [{"id": str,
"function": {"name": str, "arguments": dict}}], "usage": {"in": int, "out": int}}.
Internal message history stays in that same OpenAI-ish shape everywhere; each
provider converts to/from its own wire format at the edges.

Every provider has a blocking `chat()` and a streaming `chat_stream()` that yields
text deltas followed by one final ChatResult.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Callable, Iterator, Protocol

import ollama


@dataclass
class ChatResult:
    content: str
    tool_calls: list[dict] = field(default_factory=list)
    # Token accounting for observability: {"in": prompt_tokens, "out": completion_tokens}.
    usage: dict = field(default_factory=dict)


class ChatProvider(Protocol):
    def chat(self, model: str, messages: list[dict], tools: list[dict]) -> ChatResult: ...

    def chat_stream(
        self, model: str, messages: list[dict], tools: list[dict]
    ) -> Iterator["str | ChatResult"]:
        """Yield text deltas (str) as they arrive, then one final ChatResult."""
        ...


def _norm_tool_args(args) -> dict:
    if isinstance(args, str):
        try:
            return json.loads(args) if args else {}
        except json.JSONDecodeError:
            return {}
    return args or {}


def chat_with_optional_stream(
    provider: ChatProvider,
    model: str,
    messages: list[dict],
    tools: list[dict],
    on_token: Callable[[str], None] | None = None,
) -> ChatResult:
    """Run one chat call. With `on_token`, stream text deltas through it as they arrive
    (falling back to the plain non-streaming call for providers that can't stream)."""
    if on_token is None or not hasattr(provider, "chat_stream"):
        return provider.chat(model, messages, tools)
    final: ChatResult | None = None
    for item in provider.chat_stream(model, messages, tools):
        if isinstance(item, ChatResult):
            final = item
        elif item:
            on_token(item)
    return final or ChatResult(content="")


class OllamaProvider:
    id = "ollama"

    @staticmethod
    def _usage(resp) -> dict:
        get = resp.get if hasattr(resp, "get") else (lambda k, d=None: getattr(resp, k, d))
        return {"in": int(get("prompt_eval_count", 0) or 0), "out": int(get("eval_count", 0) or 0)}

    @staticmethod
    def _tool_calls(msg) -> list[dict]:
        out = []
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function") or {}
            out.append(
                {
                    "id": tc.get("id") or f"call_{len(out)}",
                    "function": {"name": fn.get("name"), "arguments": _norm_tool_args(fn.get("arguments"))},
                }
            )
        return out

    def chat(self, model: str, messages: list[dict], tools: list[dict]) -> ChatResult:
        resp = ollama.chat(model=model, messages=messages, tools=tools, think=False, stream=False)
        msg = resp["message"]
        return ChatResult(
            content=msg.get("content") or "", tool_calls=self._tool_calls(msg), usage=self._usage(resp)
        )

    def chat_stream(
        self, model: str, messages: list[dict], tools: list[dict]
    ) -> Iterator["str | ChatResult"]:
        content_parts: list[str] = []
        tool_calls: list[dict] = []
        usage: dict = {}
        for chunk in ollama.chat(model=model, messages=messages, tools=tools, think=False, stream=True):
            msg = chunk["message"]
            delta = msg.get("content") or ""
            if delta:
                content_parts.append(delta)
                yield delta
            for tc in self._tool_calls(msg):
                tc["id"] = f"call_{len(tool_calls)}"
                tool_calls.append(tc)
            if chunk.get("done"):
                usage = self._usage(chunk)
        yield ChatResult(content="".join(content_parts), tool_calls=tool_calls, usage=usage)

    @staticmethod
    def list_models() -> list[str]:
        try:
            resp = ollama.list()
            return sorted(m.model for m in resp.models if "embed" not in m.model.lower())
        except Exception:
            return []


class OpenAIProvider:
    """Also used for any OpenAI-compatible endpoint (GitHub Models, Azure, local vLLM, ...)."""

    id = "openai"

    def __init__(self, api_key: str | None = None, base_url: str | None = None):
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def chat(self, model: str, messages: list[dict], tools: list[dict]) -> ChatResult:
        oi_messages = _to_openai_messages(messages)
        oi_tools = tools or None
        resp = self._client.chat.completions.create(
            model=model, messages=oi_messages, tools=oi_tools, tool_choice="auto" if oi_tools else None
        )
        msg = resp.choices[0].message
        tool_calls = [
            {
                "id": tc.id,
                "function": {"name": tc.function.name, "arguments": _norm_tool_args(tc.function.arguments)},
            }
            for tc in (msg.tool_calls or [])
        ]
        u = getattr(resp, "usage", None)
        usage = (
            {"in": getattr(u, "prompt_tokens", 0) or 0, "out": getattr(u, "completion_tokens", 0) or 0}
            if u
            else {}
        )
        return ChatResult(content=msg.content or "", tool_calls=tool_calls, usage=usage)

    def chat_stream(
        self, model: str, messages: list[dict], tools: list[dict]
    ) -> Iterator["str | ChatResult"]:
        kwargs: dict = dict(model=model, messages=_to_openai_messages(messages), stream=True)
        if tools:
            kwargs.update(tools=tools, tool_choice="auto")
        try:
            stream = self._client.chat.completions.create(**kwargs, stream_options={"include_usage": True})
        except Exception:
            # Some OpenAI-compatible endpoints (e.g. GitHub Models) reject stream_options.
            stream = self._client.chat.completions.create(**kwargs)
        content_parts: list[str] = []
        acc: dict[int, dict] = {}  # tool-call index -> {id, name, args-string}
        usage: dict = {}
        for chunk in stream:
            u = getattr(chunk, "usage", None)
            if u:
                usage = {
                    "in": getattr(u, "prompt_tokens", 0) or 0,
                    "out": getattr(u, "completion_tokens", 0) or 0,
                }
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                content_parts.append(delta.content)
                yield delta.content
            for tc in delta.tool_calls or []:
                slot = acc.setdefault(tc.index, {"id": None, "name": "", "args": ""})
                if tc.id:
                    slot["id"] = tc.id
                if tc.function and tc.function.name:
                    slot["name"] += tc.function.name
                if tc.function and tc.function.arguments:
                    slot["args"] += tc.function.arguments
        tool_calls = [
            {"id": v["id"] or f"call_{i}", "function": {"name": v["name"], "arguments": _norm_tool_args(v["args"])}}
            for i, v in sorted(acc.items())
        ]
        yield ChatResult(content="".join(content_parts), tool_calls=tool_calls, usage=usage)


class AnthropicProvider:
    id = "anthropic"

    def __init__(self, api_key: str | None = None):
        from anthropic import Anthropic

        # Anthropic accepts either a platform API key (sk-ant-api...) via x-api-key,
        # or a Claude Pro/Max subscription OAuth token (sk-ant-oat01-..., minted by
        # `claude setup-token`) via a bearer Authorization header — the SDK picks the
        # header based on which constructor arg is used.
        if api_key and api_key.startswith("sk-ant-oat"):
            self._client = Anthropic(auth_token=api_key)
        else:
            self._client = Anthropic(api_key=api_key)

    @staticmethod
    def _request(model: str, messages: list[dict], tools: list[dict]) -> dict:
        system_text, anthro_messages = _to_anthropic_messages(messages)
        anthro_tools = [
            {"name": t["function"]["name"], "description": t["function"]["description"], "input_schema": t["function"]["parameters"]}
            for t in (tools or [])
        ]
        kwargs: dict = dict(model=model, max_tokens=4096, system=system_text or "", messages=anthro_messages)
        if anthro_tools:
            kwargs["tools"] = anthro_tools
        return kwargs

    @staticmethod
    def _parse(resp) -> ChatResult:
        content = ""
        tool_calls = []
        for block in resp.content:
            if block.type == "text":
                content += block.text
            elif block.type == "tool_use":
                tool_calls.append({"id": block.id, "function": {"name": block.name, "arguments": block.input}})
        u = getattr(resp, "usage", None)
        usage = (
            {"in": getattr(u, "input_tokens", 0) or 0, "out": getattr(u, "output_tokens", 0) or 0}
            if u
            else {}
        )
        return ChatResult(content=content, tool_calls=tool_calls, usage=usage)

    def chat(self, model: str, messages: list[dict], tools: list[dict]) -> ChatResult:
        return self._parse(self._client.messages.create(**self._request(model, messages, tools)))

    def chat_stream(
        self, model: str, messages: list[dict], tools: list[dict]
    ) -> Iterator["str | ChatResult"]:
        with self._client.messages.stream(**self._request(model, messages, tools)) as stream:
            for text in stream.text_stream:
                if text:
                    yield text
            final = stream.get_final_message()
        yield self._parse(final)


def _to_openai_messages(messages: list[dict]) -> list[dict]:
    out = []
    for m in messages:
        if m["role"] == "assistant" and m.get("tool_calls"):
            out.append(
                {
                    "role": "assistant",
                    "content": m.get("content") or None,
                    "tool_calls": [
                        {
                            "id": tc.get("id") or f"call_{i}",
                            "type": "function",
                            "function": {
                                "name": (tc.get("function") or {}).get("name"),
                                "arguments": json.dumps((tc.get("function") or {}).get("arguments") or {}),
                            },
                        }
                        for i, tc in enumerate(m["tool_calls"])
                    ],
                }
            )
        else:
            out.append({"role": m["role"], "content": m.get("content") or "", **({"tool_call_id": m["tool_call_id"]} if "tool_call_id" in m else {})})
    return out


def _to_anthropic_messages(messages: list[dict]) -> tuple[str, list[dict]]:
    system_parts: list[str] = []
    out: list[dict] = []
    for m in messages:
        role = m["role"]
        if role == "system":
            system_parts.append(m.get("content") or "")
        elif role == "user":
            out.append({"role": "user", "content": m.get("content") or ""})
        elif role == "assistant":
            blocks = []
            if m.get("content"):
                blocks.append({"type": "text", "text": m["content"]})
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function") or {}
                blocks.append({"type": "tool_use", "id": tc.get("id"), "name": fn.get("name"), "input": fn.get("arguments") or {}})
            out.append({"role": "assistant", "content": blocks or (m.get("content") or "")})
        elif role == "tool":
            block = {"type": "tool_result", "tool_use_id": m.get("tool_call_id"), "content": m.get("content") or ""}
            if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list):
                out[-1]["content"].append(block)
            else:
                out.append({"role": "user", "content": [block]})
    return "\n\n".join(p for p in system_parts if p), out


# --- Registry -----------------------------------------------------------

PROVIDER_SPECS = {
    "ollama": {
        "label": "Ollama (local + cloud-routed)",
        "env": None,
        "base_url_env": "OLLAMA_HOST",
    },
    "openai": {
        "label": "OpenAI",
        "env": "OPENAI_API_KEY",
        "base_url": None,
        "default_models": ["gpt-5.1", "gpt-5.1-mini", "gpt-4.1", "gpt-4o-mini"],
        "key_hint": "sk-... — platform.openai.com/api-keys (no OAuth exists for the API in 2026)",
    },
    "anthropic": {
        "label": "Anthropic (Claude)",
        "env": "ANTHROPIC_API_KEY",
        "default_models": ["claude-opus-5", "claude-sonnet-5", "claude-haiku-4-5-20251001"],
        "key_hint": "sk-ant-api... (API key) or sk-ant-oat01-... (Claude Pro/Max token from `claude setup-token`)",
    },
    "github_models": {
        "label": "GitHub Models",
        # 2026 endpoint — the old models.inference.ai.azure.com host still works but is
        # being phased out. Note: this is GitHub *Models* (a model catalog/inference
        # API), not GitHub Copilot — Copilot Chat has no public chat-completions API as
        # of 2026, only unofficial reverse-engineered proxies, which we don't use here.
        "env": "GITHUB_TOKEN",
        "base_url": "https://models.github.ai/inference",
        "default_models": ["openai/gpt-4o", "openai/gpt-4o-mini", "meta/Meta-Llama-3.1-70B-Instruct"],
        "key_hint": "a GitHub personal access token — classic (no scopes needed) or fine-grained with 'models: read'",
    },
}

# Credentials entered in the Settings panel — kept in memory only for this server
# process's lifetime (never written to disk, never put in a URL/query string).
_runtime_keys: dict[str, str] = {}


def set_runtime_key(provider_id: str, api_key: str | None) -> None:
    if provider_id not in PROVIDER_SPECS or provider_id == "ollama":
        raise ValueError(f"provider '{provider_id}' does not take a key")
    if api_key:
        _runtime_keys[provider_id] = api_key
    else:
        _runtime_keys.pop(provider_id, None)


def _resolve_key(provider_id: str) -> str | None:
    if provider_id in _runtime_keys:
        return _runtime_keys[provider_id]
    env = PROVIDER_SPECS[provider_id].get("env")
    return os.environ.get(env) if env else None


def provider_status() -> dict:
    """What's available right now, for the Settings panel."""
    out = {}
    for pid, spec in PROVIDER_SPECS.items():
        if pid == "ollama":
            models = OllamaProvider.list_models()
            out[pid] = {
                "label": spec["label"], "available": bool(models), "models": models,
                "needs_env": None, "has_key": False, "key_hint": None,
            }
        else:
            key = _resolve_key(pid)
            out[pid] = {
                "label": spec["label"],
                "available": bool(key),
                "models": spec.get("default_models", []),
                "needs_env": spec.get("env"),
                "has_key": pid in _runtime_keys,
                "key_hint": spec.get("key_hint"),
            }
    return out


def get_provider(provider_id: str) -> ChatProvider:
    spec = PROVIDER_SPECS.get(provider_id)
    if spec is None:
        raise ValueError(f"unknown provider: {provider_id}")
    if provider_id == "ollama":
        return OllamaProvider()
    if provider_id == "anthropic":
        return AnthropicProvider(api_key=_resolve_key(provider_id))
    # openai + any OpenAI-compatible preset (github_models, ...)
    return OpenAIProvider(api_key=_resolve_key(provider_id), base_url=spec.get("base_url"))
