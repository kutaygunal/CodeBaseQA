"""Cheap, deterministic follow-up suggestions derived from the retrieval trace.

No LLM call — we turn the symbols and files that were actually retrieved into 2-3
exploratory "next question" prompts ("What calls X?", "Where is X defined?"). This keeps
follow-up chips instant and free, and they drive a natural exploration loop: an answer about
`SurfaceInspector` leads to "What calls SurfaceInspector::optics?".

Used by cqa/agent.py (returned with every answer) and rendered as clickable chips in the
web UI (which re-submit the suggestion as the next question in the same thread).
"""
from __future__ import annotations

_HIT_NODES = ("retrieve", "sub_retrieve")


def _collect(trace) -> tuple[list[str], list[str]]:
    """Return (symbols, paths) seen in the retrieved hits, in first-appearance order."""
    symbols: list[str] = []
    paths: list[str] = []
    for e in trace or []:
        if e.get("node") not in _HIT_NODES:
            continue
        for h in e.get("hits") or []:
            s = (h.get("symbol") or "").strip()
            if s and s not in symbols:
                symbols.append(s)
            p = (h.get("path") or "").strip()
            if p and p not in paths:
                paths.append(p)
    return symbols, paths


def suggest_followups(question: str, trace: list[dict] | None = None, limit: int = 3) -> list[str]:
    """Build <= `limit` follow-up question strings from the retrieved evidence."""
    ql = (question or "").strip().lower()
    symbols, paths = _collect(trace)

    cands: list[str] = []
    add = lambda t: (cands.append(t) if t and t not in cands else None)

    if symbols:
        s = symbols[0]
        if f"what calls {s.lower()}" not in ql:
            add(f"What calls {s}?")
        add(f"Where is {s} defined?")
        if len(symbols) >= 2:
            add(f"How does {s} relate to {symbols[1]}?")
        elif len(symbols) == 1:
            add(f"What symbols does {s} depend on?")

    # Fill the remaining slu with path-scoped questions (useful when the top hit is a file).
    if len(cands) < limit and paths:
        p0 = paths[0]
        base = p0.rsplit("/", 1)[-1]
        if f"what is {p0.lower()} responsible for" not in ql:
            add(f"What is {p0} responsible for?")
        if len(cands) < limit and f"defined in {base.lower()}" not in ql:
            add(f"What symbols are defined in {base}?")

    # Never echo the question itself back as a "next" suggestion.
    return [t for t in cands if t.lower() != ql][:limit]
