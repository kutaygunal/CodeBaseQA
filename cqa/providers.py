"""Chat-model provider abstraction: Ollama (local + cloud-routed), OpenAI,
Anthropic, and any OpenAI-compatible endpoint (e.g. GitHub Models), behind one
normalized interface so cqa/graph.py doesn't care who's answering.

Normalized response shape: {"content": str, "tool_calls": [{"id": str,
"function": {"name": str, "arguments": dict}}]}. Internal message history
stays in that same OpenAI-ish shape everywhere; each provider converts to/from
its own wire format at the edges.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Protocol

import ollama


@dataclass
class ChatResult:
    content: str
    tool_calls: list[dict] = field(default_factory=list)


class ChatProvider(Protocol):
    def chat(self, model: str, messages: list[dict], tools: list[dict]) -> ChatResult: ...


class OllamaProvider:
    id = "ollama"

    def chat(self, model: str, messages: list[dict], tools: list[dict]) -> ChatResult:
        resp = ollama.chat(model=model, messages=messages, tools=tools, think=False, stream=False)
        msg = resp["message"]
        tool_calls = []
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function") or {}
            args = fn.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            tool_calls.append({"id": tc.get("id") or f"call_{len(tool_calls)}", "function": {"name": fn.get("name"), "arguments": args}})
        return ChatResult(content=msg.get("content") or "", tool_calls=tool_calls)

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
        tool_calls = []
        for tc in msg.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments) if tc.function.arguments else {}
            except json.JSONDecodeError:
                args = {}
            tool_calls.append({"id": tc.id, "function": {"name": tc.function.name, "arguments": args}})
        return ChatResult(content=msg.content or "", tool_calls=tool_calls)


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

    def chat(self, model: str, messages: list[dict], tools: list[dict]) -> ChatResult:
        system_text, anthro_messages = _to_anthropic_messages(messages)
        anthro_tools = [
            {"name": t["function"]["name"], "description": t["function"]["description"], "input_schema": t["function"]["parameters"]}
            for t in (tools or [])
        ]
        resp = self._client.messages.create(
            model=model,
            max_tokens=4096,
            system=system_text or "",
            messages=anthro_messages,
            tools=anthro_tools or None,
        )
        content = ""
        tool_calls = []
        for block in resp.content:
            if block.type == "text":
                content += block.text
            elif block.type == "tool_use":
                tool_calls.append({"id": block.id, "function": {"name": block.name, "arguments": block.input}})
        return ChatResult(content=content, tool_calls=tool_calls)


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
