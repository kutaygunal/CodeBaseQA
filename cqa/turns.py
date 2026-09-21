"""Per-turn persistence: every answered question is stored with its full trace and
metrics (latency, tokens, cost), plus user feedback and cached "tour" documents.

Lives in the same sqlite file as the LangGraph checkpoints and the thread list.
This is the single source of truth for observability (#18), the feedback loop
(#17), restoring past traces when a conversation is reopened, and the tour cache.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path

import yaml

from .config import PROJECT_ROOT

PRICING_PATH = PROJECT_ROOT / "config" / "pricing.yaml"


def load_pricing(path: Path | None = None) -> dict:
    p = path or PRICING_PATH
    if not p.exists():
        return {}
    try:
        return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return {}


def estimate_cost(pricing: dict, provider: str, model: str, tokens_in: int, tokens_out: int) -> float | None:
    """USD estimate from a $/Mtok table, or None when the model has no price entry.
    Lookup order: "provider/model", "provider/*"."""
    entry = pricing.get(f"{provider}/{model}") or pricing.get(f"{provider}/*")
    if not entry:
        return None
    return round(
        tokens_in / 1e6 * float(entry.get("in_per_mtok", 0)) + tokens_out / 1e6 * float(entry.get("out_per_mtok", 0)),
        6,
    )


def summarize_trace(trace: list[dict]) -> dict:
    """Aggregate token usage / LLM calls / tool rounds out of a turn's trace entries."""
    tin = tout = calls = rounds = 0
    for e in trace or []:
        u = e.get("usage") or {}
        if u:
            tin += int(u.get("in", 0) or 0)
            tout += int(u.get("out", 0) or 0)
            calls += 1
        if e.get("node") == "tools":
            rounds += 1
    return {"tokens_in": tin, "tokens_out": tout, "llm_calls": calls, "tool_rounds": rounds}


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    k = min(len(s) - 1, max(0, int(round(pct / 100 * (len(s) - 1)))))
    return s[k]


class TurnStore:
    def __init__(self, conn: sqlite3.Connection, pricing: dict | None = None):
        self._conn = conn
        self._lock = threading.Lock()  # one shared connection across worker threads
        self.pricing = pricing if pricing is not None else load_pricing()
        with self._conn:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS turns (
                    id TEXT PRIMARY KEY,
                    thread_id TEXT NOT NULL,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    citations TEXT NOT NULL,
                    trace TEXT NOT NULL,
                    provider TEXT, model TEXT, module TEXT,
                    mode TEXT NOT NULL DEFAULT 'qa',
                    latency_ms INTEGER, tokens_in INTEGER, tokens_out INTEGER,
                    llm_calls INTEGER, tool_rounds INTEGER,
                    cost_usd REAL,
                    created_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_turns_thread ON turns(thread_id, created_at);
                CREATE TABLE IF NOT EXISTS feedback (
                    turn_id TEXT PRIMARY KEY,
                    rating INTEGER NOT NULL,
                    reason TEXT, comment TEXT, correct_paths TEXT,
                    created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS tours (
                    target TEXT NOT NULL, index_key TEXT NOT NULL,
                    provider TEXT NOT NULL, model TEXT NOT NULL,
                    markdown TEXT NOT NULL, citations TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    PRIMARY KEY (target, index_key, provider, model)
                );
                """
            )
            # Backward-compatible migration: add the `suggestions` column if missing.
            cols = {r[1] for r in self._conn.execute("PRAGMA table_info(turns)").fetchall()}
            if "suggestions" not in cols:
                self._conn.execute("ALTER TABLE turns ADD COLUMN suggestions TEXT")

    # --- turns -----------------------------------------------------------

    def record(
        self,
        *,
        thread_id: str,
        question: str,
        answer: str,
        citations: list[str],
        trace: list[dict],
        provider: str,
        model: str,
        module: str | None,
        mode: str,
        latency_ms: int,
        suggestions: list[str] | None = None,
    ) -> dict:
        m = summarize_trace(trace)
        cost = estimate_cost(self.pricing, provider, model, m["tokens_in"], m["tokens_out"])
        turn_id = uuid.uuid4().hex
        with self._lock, self._conn:
            self._conn.execute(
                """INSERT INTO turns (id, thread_id, question, answer, citations, trace, provider, model,
                   module, mode, latency_ms, tokens_in, tokens_out, llm_calls, tool_rounds, cost_usd, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    turn_id, thread_id, question, answer, json.dumps(citations), json.dumps(trace),
                    provider, model, module, mode, latency_ms, m["tokens_in"], m["tokens_out"],
                    m["llm_calls"], m["tool_rounds"], cost, time.time(),
                ),
            )
            if suggestions:
                self._conn.execute(
                    "UPDATE turns SET suggestions=? WHERE id=?", (json.dumps(suggestions), turn_id)
                )
        return {"turn_id": turn_id, "latency_ms": latency_ms, "cost_usd": cost, **m}

    @staticmethod
    def _row_to_turn(r: sqlite3.Row | tuple) -> dict:
        (tid, thread_id, question, answer, citations, trace, suggestions, provider, model, module, mode,
         latency_ms, tin, tout, calls, rounds, cost, created_at, rating, reason, comment, paths) = r
        return {
            "turn_id": tid, "thread_id": thread_id, "question": question, "answer": answer,
            "citations": json.loads(citations), "trace": json.loads(trace),
            "suggestions": json.loads(suggestions) if suggestions else [],
            "provider": provider, "model": model, "module": module, "mode": mode,
            "metrics": {
                "latency_ms": latency_ms, "tokens_in": tin, "tokens_out": tout,
                "llm_calls": calls, "tool_rounds": rounds, "cost_usd": cost,
            },
            "created_at": created_at,
            "feedback": None if rating is None else {
                "rating": rating, "reason": reason, "comment": comment,
                "correct_paths": json.loads(paths) if paths else [],
            },
        }

    _SELECT = """SELECT t.id, t.thread_id, t.question, t.answer, t.citations, t.trace, t.suggestions, t.provider, t.model,
                 t.module, t.mode, t.latency_ms, t.tokens_in, t.tokens_out, t.llm_calls, t.tool_rounds,
                 t.cost_usd, t.created_at, f.rating, f.reason, f.comment, f.correct_paths
                 FROM turns t LEFT JOIN feedback f ON f.turn_id = t.id"""

    def for_thread(self, thread_id: str) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                self._SELECT + " WHERE t.thread_id = ? ORDER BY t.created_at", (thread_id,)
            ).fetchall()
        return [self._row_to_turn(r) for r in rows]

    def get(self, turn_id: str) -> dict | None:
        with self._lock:
            r = self._conn.execute(self._SELECT + " WHERE t.id = ?", (turn_id,)).fetchone()
        return self._row_to_turn(r) if r else None

    def delete_thread(self, thread_id: str) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "DELETE FROM feedback WHERE turn_id IN (SELECT id FROM turns WHERE thread_id = ?)", (thread_id,)
            )
            self._conn.execute("DELETE FROM turns WHERE thread_id = ?", (thread_id,))

    # --- feedback --------------------------------------------------------

    def set_feedback(
        self,
        turn_id: str,
        rating: int,
        reason: str | None = None,
        comment: str | None = None,
        correct_paths: list[str] | None = None,
    ) -> bool:
        """rating: 1 (👍), -1 (👎), or 0 to clear. Returns False for an unknown turn."""
        if rating not in (-1, 0, 1):
            raise ValueError("rating must be -1, 0 or 1")
        with self._lock, self._conn:
            if not self._conn.execute("SELECT 1 FROM turns WHERE id = ?", (turn_id,)).fetchone():
                return False
            if rating == 0:
                self._conn.execute("DELETE FROM feedback WHERE turn_id = ?", (turn_id,))
                return True
            self._conn.execute(
                """INSERT INTO feedback (turn_id, rating, reason, comment, correct_paths, created_at)
                   VALUES (?,?,?,?,?,?)
                   ON CONFLICT(turn_id) DO UPDATE SET rating=excluded.rating, reason=excluded.reason,
                   comment=excluded.comment, correct_paths=excluded.correct_paths, created_at=excluded.created_at""",
                (
                    turn_id, rating, (reason or "")[:60], (comment or "")[:2000],
                    json.dumps([p.strip()[:300] for p in (correct_paths or []) if p.strip()][:10]), time.time(),
                ),
            )
        return True

    def export_feedback(self) -> list[dict]:
        """Rated turns with their question/answer and the retrieval trace entry — the raw
        material for scripts/feedback_report.py and for growing an eval set."""
        with self._lock:
            rows = self._conn.execute(
                self._SELECT + " WHERE f.rating IS NOT NULL ORDER BY f.created_at"
            ).fetchall()
        out = []
        for r in rows:
            t = self._row_to_turn(r)
            retrieve = next((e for e in t["trace"] if e.get("node") == "retrieve"), {})
            out.append(
                {
                    "turn_id": t["turn_id"], "question": t["question"], "answer": t["answer"],
                    "citations": t["citations"], "provider": t["provider"], "model": t["model"],
                    "mode": t["mode"], **t["feedback"],
                    "retrieval": {
                        "thin": retrieve.get("thin"),
                        "top_score": max((h.get("score", 0) for h in retrieve.get("hits", [])), default=None),
                        "n_hits": len(retrieve.get("hits", [])),
                    },
                }
            )
        return out

    # --- stats -----------------------------------------------------------

    def stats(self) -> dict:
        with self._lock:
            rows = self._conn.execute(
                """SELECT t.provider, t.model, t.latency_ms, t.tokens_in, t.tokens_out, t.cost_usd,
                   t.tool_rounds, f.rating FROM turns t LEFT JOIN feedback f ON f.turn_id = t.id"""
            ).fetchall()
        groups: dict[tuple, list] = {}
        for r in rows:
            groups.setdefault((r[0] or "?", r[1] or "?"), []).append(r)

        def agg(rs: list) -> dict:
            lat = [r[2] for r in rs if r[2] is not None]
            costs = [r[5] for r in rs if r[5] is not None]
            up = sum(1 for r in rs if r[7] == 1)
            down = sum(1 for r in rs if r[7] == -1)
            return {
                "turns": len(rs),
                "latency_p50_ms": _percentile(lat, 50),
                "latency_p95_ms": _percentile(lat, 95),
                "avg_tokens_in": round(sum(r[3] or 0 for r in rs) / len(rs)) if rs else 0,
                "avg_tokens_out": round(sum(r[4] or 0 for r in rs) / len(rs)) if rs else 0,
                "total_cost_usd": round(sum(costs), 6) if costs else None,
                "avg_tool_rounds": round(sum(r[6] or 0 for r in rs) / len(rs), 2) if rs else 0,
                "thumbs_up": up,
                "thumbs_down": down,
                "thumbs_up_rate": round(up / (up + down), 3) if (up + down) else None,
            }

        return {
            "overall": agg(rows) if rows else agg([]),
            "by_model": [
                {"provider": p, "model": m, **agg(rs)}
                for (p, m), rs in sorted(groups.items(), key=lambda kv: -len(kv[1]))
            ],
        }

    # --- tour cache ------------------------------------------------------

    def get_tour(self, target: str, index_key: str, provider: str, model: str) -> dict | None:
        with self._lock:
            r = self._conn.execute(
                "SELECT markdown, citations, created_at FROM tours WHERE target=? AND index_key=? AND provider=? AND model=?",
                (target, index_key, provider, model),
            ).fetchone()
        return {"markdown": r[0], "citations": json.loads(r[1]), "created_at": r[2]} if r else None

    def put_tour(self, target: str, index_key: str, provider: str, model: str, markdown: str, citations: list[str]) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                """INSERT OR REPLACE INTO tours (target, index_key, provider, model, markdown, citations, created_at)
                   VALUES (?,?,?,?,?,?,?)""",
                (target, index_key, provider, model, markdown, json.dumps(citations), time.time()),
            )
