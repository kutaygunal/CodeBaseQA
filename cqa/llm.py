"""One place that calls a chat model with the project's fallback + streaming rules.

Used by the QA graph, query analysis, review and tour generation, so a bad/unavailable
provider selection never hard-fails an answer (it degrades to the local Ollama model).
"""
from __future__ import annotations

from typing import Callable

from .config import Config
from .providers import ChatResult, chat_with_optional_stream, get_provider


def call_model(
    cfg: Config,
    provider_id: str,
    model: str,
    messages: list[dict],
    tools: list[dict] | None = None,
    on_token: Callable[[str], None] | None = None,
    on_reset: Callable[[], None] | None = None,
) -> ChatResult:
    """Chat once. Streams text deltas through `on_token` when given. If the selected
    provider fails, retry on the local fallback model; `on_reset` is called first when a
    partial stream had already been emitted, so the UI can discard the abandoned text."""
    tools = tools or []
    emitted = False

    def tap(text: str) -> None:
        nonlocal emitted
        emitted = True
        if on_token:
            on_token(text)

    try:
        return chat_with_optional_stream(
            get_provider(provider_id), model, messages, tools, tap if on_token else None
        )
    except Exception:
        if provider_id == "ollama" and model == cfg.models.chat_model_fallback:
            raise
        if emitted and on_reset:
            on_reset()
        return chat_with_optional_stream(
            get_provider("ollama"), cfg.models.chat_model_fallback, messages, tools,
            on_token if on_token else None,
        )
