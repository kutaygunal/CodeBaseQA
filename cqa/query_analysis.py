"""Question analysis: standalone rewrite, extra search queries, HyDE, sub-question plan.

One small JSON-producing LLM call that fixes the retrieval-side weaknesses of raw user text:

* follow-ups ("and what calls that?") are rewritten into a standalone question using the
  recent conversation, since retrieval otherwise embeds the bare fragment;
* vague questions get 2-3 alternative search queries plus a short hypothetical answer
  (HyDE) that uses the vocabulary the code would use;
* compound questions ("compare the GPU and CPU paths") are split into <=N sub-questions
  that the graph retrieves for in parallel.

Any failure degrades to "no analysis" — the raw question is then used as before.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .config import Config
from .llm import call_model

_COMPOUND_RE = re.compile(
    r"\b(compare|comparison|versus|vs\.?|differ(?:s|ent|ence|ences)?|contrast|both|"
    r"as well as|each of|respectively)\b",
    re.I,
)

PROMPT = """You prepare a question about the {project} C++ codebase for code search.

{history_block}Question: {question}

Reply with ONE JSON object and nothing else, with these keys:
"standalone_question": the question rewritten so it makes sense without the conversation (resolve "that", "it", "the same function" using the history). If it is already standalone, repeat it unchanged.
"search_queries": a list of up to {max_queries} short alternative search queries (identifiers, file/module names, or domain terms a developer would grep for). Different wording from each other.
{hyde_line}"sub_questions": {sub_line}

Do not invent symbol names you are not reasonably sure exist; prefer plain descriptive terms."""

_HYDE_LINE = '"hypothetical_answer": a 2-4 line plausible answer written like the codebase would express it (mention likely function/class names in a natural way). It is only used for search.\n'
_SUB_PLAN = (
    "if the question truly asks about several distinct things that need separate lookups "
    "(e.g. a comparison of two subsystems), a list of at most {n} self-contained sub-questions; otherwise []."
)
_SUB_NONE = "[]"


@dataclass
class Analysis:
    standalone: str
    queries: list[str] = field(default_factory=list)  # extra queries (incl. HyDE text)
    sub_questions: list[str] = field(default_factory=list)
    used_llm: bool = False
    usage: dict = field(default_factory=dict)
    error: str | None = None


def looks_compound(question: str) -> bool:
    if _COMPOUND_RE.search(question):
        return True
    return question.count("?") >= 2


def should_analyze(question: str, has_history: bool, cfg: Config) -> bool:
    if has_history:
        return True  # follow-ups need context resolution
    return len(question.split()) >= cfg.retrieval.analyze.min_words


def _history_block(history: list[dict]) -> str:
    """Compact recent Q/A pairs from the checkpointed thread messages."""
    turns: list[str] = []
    for m in history[-8:]:
        role = m.get("role")
        content = (m.get("content") or "").strip()
        if role == "user":
            q = re.match(r"Question: (.*?)\n\nRetrieved context", content, re.S)
            turns.append("User: " + (q.group(1) if q else content)[:300])
        elif role == "assistant" and not m.get("tool_calls") and content:
            turns.append("Assistant: " + content[:400])
    if not turns:
        return ""
    return "Conversation so far:\n" + "\n".join(turns[-4:]) + "\n\n"


def _parse_json(text: str) -> dict | None:
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I)
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                obj = json.loads(m.group(0))
                return obj if isinstance(obj, dict) else None
            except json.JSONDecodeError:
                return None
    return None


def _clean_list(v, n: int, max_len: int = 300) -> list[str]:
    if not isinstance(v, list):
        return []
    out = []
    for x in v:
        if isinstance(x, str) and x.strip() and x.strip() not in out:
            out.append(x.strip()[:max_len])
    return out[:n]


def analyze_question(
    question: str,
    history: list[dict],
    cfg: Config,
    provider_id: str,
    model: str,
) -> Analysis:
    acfg = cfg.retrieval.analyze
    allow_plan = looks_compound(question)
    prompt = PROMPT.format(
        project=cfg.repo.root.name,
        history_block=_history_block(history),
        question=question,
        max_queries=acfg.max_queries,
        hyde_line=_HYDE_LINE if acfg.use_hyde else "",
        sub_line=_SUB_PLAN.format(n=acfg.max_sub_questions) if allow_plan else _SUB_NONE,
    )
    try:
        res = call_model(cfg, provider_id, model, [{"role": "user", "content": prompt}])
    except Exception as e:  # noqa: BLE001
        return Analysis(standalone=question, error=f"{type(e).__name__}: {e}")
    obj = _parse_json(res.content)
    if not obj:
        return Analysis(standalone=question, used_llm=True, usage=res.usage, error="unparseable analysis")

    standalone = obj.get("standalone_question")
    standalone = standalone.strip()[:500] if isinstance(standalone, str) and standalone.strip() else question
    queries = _clean_list(obj.get("search_queries"), acfg.max_queries)
    hyde = obj.get("hypothetical_answer")
    if acfg.use_hyde and isinstance(hyde, str) and hyde.strip():
        queries.append(hyde.strip()[:600])
    subs = _clean_list(obj.get("sub_questions"), acfg.max_sub_questions) if allow_plan else []
    if len(subs) < 2:  # a "plan" of one sub-question is just the question
        subs = []
    return Analysis(standalone=standalone, queries=queries, sub_questions=subs, used_llm=True, usage=res.usage)
