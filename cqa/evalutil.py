"""Deterministic scoring helpers for the answer-level eval harness (scripts/eval_answers.py).

Kept as a separate module so the refusal detector / match helpers can be unit-tested
without importing the script. No LLM calls here — everything is a deterministic function
of the answer text, citations and turn trace.
"""
from __future__ import annotations

import re

KS = (1, 3, 8)

# --- refusal (not-in-codebase) detector --------------------------------------
# A correct not-in-codebase answer often CITIES its search evidence (grep/symbol results),
# so we do NOT cap on citation count. The detector only runs on negative cases, so broad
# recall is the goal over precision. It is a heuristic; an LLM-judge is a possible upgrade.
_REFUSE = re.compile(
    r"(?:\bnot (?:a |an |any )?(?:part |present|found|included|indexed|implemented|realized|supported|available|handled|provided|in (?:this |scope |the )?\w*|within)\b)|"
    r"(?:\bdoes(?:n't| not)? (?:exist|appear|contain|include|have|implement|support|provide|build|ship|use)\b)|"
    r"(?:\b(?:i |we )?(?:can't|cannot|couldn't|could not|unable to)\s+(?:answer|find|locate|say|confirm)\b)|"
    r"(?:\b(?:there is no|there's no)\s+[\w .,\-]{0,40}\b(?:server|api|service|framework|schema|database|login|authentication|auth|subscription|billing|training|ml|model|routing)\b)|"
    r"(?:\bno\s+(?:real|relevant|genuine|direct|actual|meaningful|true)?\s*(?:matches|results|hits|symbol|definition|evidence|entry|record)\b)|"
    r"(?:\b(?:nothing|zero)\b)|"
    r"(?:\b(?:i |we )?(?:couldn't|could not|was unable|weren't able|don't|do not)\s+(?:find|locate)\b)|"
    r"\bnot (?:in|included in|present in|found in|part of)\s+(?:this )?(?:codebase|code base|repository|repo|project)\b",
    re.I,
)
# words that almost never indicate a genuine refusal on their own
_INNOCENT = re.compile(r"\b(don't know|doesn't do|isn't it|not have to)\b", re.I)


def is_refusal(answer: str) -> bool:
    """True if the answer text expresses a not-in-codebase / can't-answer response."""
    if not answer:
        return False
    return bool(_REFUSE.search(answer)) and not _INNOCENT.search(answer)


def path_of(cite: str) -> str:
    """'src/foo.cpp:10-20' -> 'src/foo.cpp' (strip the line range)."""
    m = re.match(r"([\w./\\\-]+\.(?:cpp|h|hpp|cu|cuh|cc|inc|cmake|txt|in))", cite)
    return m.group(1) if m else cite.split(":")[0].strip()


def expect_hit(expect: list[str], sources: list[str]) -> bool:
    """Any expected path is a substring of any source string (case-insensitive)."""
    norm = [s.lower() for s in sources]
    return any(e.lower() in s for s in norm for e in expect)


def retrieval_stats(trace: list[dict], expect: list[str]) -> dict:
    """recall@k / MRR over expected paths, from the retrieve/sub_retrieve trace entries."""
    hits: list[str] = []  # preserve first-appearance order across sub-queries
    seen: set[str] = set()
    for e in trace or []:
        if e.get("node") not in ("retrieve", "sub_retrieve"):
            continue
        for h in e.get("hits", []):
            p = h.get("path")
            if p and p not in seen:
                seen.add(p)
                hits.append(p)
    rank = next((i for i, p in enumerate(hits, 1)
                 if any(e.lower() in p.lower() for e in expect)), None)
    return {
        "recall@1": int(rank is not None and rank <= 1),
        "recall@3": int(rank is not None and rank <= 3),
        "recall@8": int(rank is not None and rank <= 8),
        "mrr": (1.0 / rank) if rank else 0.0,
        "rank": rank,
    }


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    k = min(len(s) - 1, max(0, int(round(pct / 100 * (len(s) - 1)))))
    return round(s[k], 0)
