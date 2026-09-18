"""Conversation (thread) metadata: title, timestamps — for the sidebar.

Lives in the same sqlite file as the LangGraph checkpoints (different tables),
so one file backs both. See PLAN.md — extended (web UI, multi-thread sidebar).
"""
from __future__ import annotations

import sqlite3
import time
import uuid
from pathlib import Path


class ThreadStore:
    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS threads (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )"""
        )
        self._conn.commit()

    def create(self, title: str = "New chat") -> dict:
        tid = str(uuid.uuid4())
        now = time.time()
        with self._conn:
            self._conn.execute(
                "INSERT INTO threads (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (tid, title, now, now),
            )
        return {"id": tid, "title": title, "created_at": now, "updated_at": now}

    def ensure(self, thread_id: str, title_hint: str = "New chat") -> None:
        """Create a row if this thread_id doesn't have one yet (first message on a
        client-generated id)."""
        now = time.time()
        with self._conn:
            self._conn.execute(
                "INSERT OR IGNORE INTO threads (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (thread_id, title_hint, now, now),
            )

    def touch(self, thread_id: str, first_question: str | None = None) -> None:
        now = time.time()
        with self._conn:
            if first_question is not None:
                row = self._conn.execute(
                    "SELECT title FROM threads WHERE id = ?", (thread_id,)
                ).fetchone()
                if row and row[0] == "New chat":
                    title = first_question.strip()[:60]
                    self._conn.execute(
                        "UPDATE threads SET title = ?, updated_at = ? WHERE id = ?",
                        (title, now, thread_id),
                    )
                    return
            self._conn.execute("UPDATE threads SET updated_at = ? WHERE id = ?", (now, thread_id))

    def list(self) -> list[dict]:
        rows = self._conn.execute(
            "SELECT id, title, created_at, updated_at FROM threads ORDER BY updated_at DESC"
        ).fetchall()
        return [{"id": r[0], "title": r[1], "created_at": r[2], "updated_at": r[3]} for r in rows]

    def rename(self, thread_id: str, title: str) -> None:
        with self._conn:
            self._conn.execute(
                "UPDATE threads SET title = ?, updated_at = ? WHERE id = ?",
                (title.strip()[:100] or "Untitled", time.time(), thread_id),
            )

    def delete(self, thread_id: str) -> None:
        with self._conn:
            self._conn.execute("DELETE FROM threads WHERE id = ?", (thread_id,))
            self._conn.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
            self._conn.execute("DELETE FROM writes WHERE thread_id = ?", (thread_id,))
