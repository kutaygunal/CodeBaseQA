#!/usr/bin/env python
"""Report on user feedback: how did 👍 vs 👎 answers differ in retrieval confidence?

    python scripts/feedback_report.py             # summary + suggested thresholds
    python scripts/feedback_report.py --jsonl out.jsonl   # also dump the raw rows

Reads the rated turns from the per-turn store (.index/conversations.sqlite) and compares, for
thumbs-up vs thumbs-down answers, the top retrieval score and how often the retrieval was
flagged `thin`. It only REPORTS: `retrieval.rrf_min_score` / `min_hits` in config/luxtrace.yaml
stay whatever you set — with few ratings the numbers are noisy and a threshold should never
be tuned automatically on them.

Down-voted turns that carry a "correct file(s)" correction are also listed as candidate cases
for eval/bench_cases.yaml (question -> expected paths).
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
import sys
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cqa.config import load_config
from cqa.turns import TurnStore


def _fmt(vals: list[float]) -> str:
    return "n/a" if not vals else f"median {statistics.median(vals):.4f} · min {min(vals):.4f} · max {max(vals):.4f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default=None, help="write the raw rated turns to this file")
    args = ap.parse_args()

    cfg = load_config()
    db = cfg.storage.index_state_file.parent / "conversations.sqlite"
    if not db.exists():
        sys.exit(f"no conversation database at {db}")
    store = TurnStore(sqlite3.connect(str(db)))
    rows = store.export_feedback()
    if args.jsonl:
        Path(args.jsonl).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")

    up = [r for r in rows if r["rating"] == 1]
    down = [r for r in rows if r["rating"] == -1]
    print(f"{len(rows)} rated answers: {len(up)} 👍, {len(down)} 👎\n")
    if not rows:
        print("Nothing rated yet — use the 👍/👎 buttons under answers in the web UI.")
        return

    for label, group in (("👍", up), ("👎", down)):
        scores = [r["retrieval"]["top_score"] for r in group if r["retrieval"]["top_score"] is not None]
        thin = [r for r in group if r["retrieval"]["thin"]]
        print(f"{label} top retrieval score: {_fmt(scores)}")
        if group:
            print(f"   flagged thin: {len(thin)}/{len(group)} ({len(thin) / len(group):.0%})")
    reasons: dict[str, int] = {}
    for r in down:
        reasons[r.get("reason") or "(none)"] = reasons.get(r.get("reason") or "(none)", 0) + 1
    if reasons:
        print("\n👎 reasons: " + ", ".join(f"{k} ×{v}" for k, v in sorted(reasons.items(), key=lambda kv: -kv[1])))

    print("\nConfig today: min_hits={}, rrf_min_score={}".format(cfg.retrieval.min_hits, cfg.retrieval.rrf_min_score))
    up_s = [r["retrieval"]["top_score"] for r in up if r["retrieval"]["top_score"] is not None]
    down_s = [r["retrieval"]["top_score"] for r in down if r["retrieval"]["top_score"] is not None]
    if len(up_s) >= 5 and len(down_s) >= 5:
        gap = statistics.median(up_s) - statistics.median(down_s)
        print(f"Median top-score gap 👍 − 👎: {gap:+.4f}. " + (
            "Down-voted answers tend to have weaker retrieval — consider raising rrf_min_score slightly "
            "so more of them are treated as thin (forcing tool use)." if gap > 0.003 else
            "No clear retrieval-confidence signal separates good from bad answers; the failures are "
            "probably in reasoning or missing chunks, not in the thin gate."))
    else:
        print("Need at least 5 ratings of each kind before a threshold comparison means anything.")

    fixes = [r for r in down if r.get("correct_paths")]
    if fixes:
        print("\nCandidate eval cases (👎 with a correction) — paste into eval/bench_cases.yaml after checking:")
        for r in fixes:
            paths = [p.split(":")[0] for p in r["correct_paths"]]
            print(f'  - q: {json.dumps(r["question"], ensure_ascii=False)}\n    expect: {json.dumps(paths)}')


if __name__ == "__main__":
    main()
