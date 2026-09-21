#!/usr/bin/env python
"""Answer-level evaluation harness for CodebaseQA (Tier-1 eval).

Runs the FULL agent on golden questions (eval/answer_cases.yaml) and scores the FINAL
ANSWER, not just chunk recall. It reports, per case and aggregate:

  * retrieval quality    recall@k / MRR over `expect` (from the turn trace, free)
  * groundedness         did the answer cite / mention an expected file ?
  * refusal              negative cases: did it say "not in the codebase" instead of
                         hallucinating code?   (heuristic detector — cqa/evalutil.py)
  * citation validity    fraction of emitted citations that reference a real indexed file
  * latency / tokens / cost / llm calls / tool rounds   (budget per case)
  * per-case PASS        category-dependent, and a CI gate (--gate) that fails the run
                         (exit != 0) when aggregate thresholds (eval/gates.yaml) are missed.

Usage:
  python scripts/eval_answers.py                          # all cases, full agent
  python scripts/eval_answers.py --categories definition comparison
  python scripts/eval_answers.py --ids def-envmap neg-login
  python scripts/eval_answers.py --limit 8
  python scripts/eval_answers.py --gate                   # enforce thresholds (CI)
  python scripts/eval_answers.py --report eval/report.json
  Full run is N x answer latency (the chat model answers each case). Use --limit /
  --categories / --ids to keep a gate run short.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from cqa.agent import Agent
from cqa.config import load_config
from cqa.evalutil import expect_hit, is_refusal, path_of, percentile, retrieval_stats
from cqa.retriever import Retriever

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CASES = PROJECT_ROOT / "eval" / "answer_cases.yaml"
GATES_PATH = PROJECT_ROOT / "eval" / "gates.yaml"
DEFAULT_REPORT = PROJECT_ROOT / "eval" / "report.json"


def _gates() -> dict:
    if GATES_PATH.exists():
        try:
            return yaml.safe_load(GATES_PATH.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError):
            pass
    return {}


def main() -> None:
    ap = argparse.ArgumentParser(description="Answer-level eval harness for CodebaseQA")
    ap.add_argument("--case-file", default=str(DEFAULT_CASES))
    ap.add_argument("--categories", nargs="*", default=None,
                    help="only these categories (definition|usage|comparison|flow|follow-up|negative)")
    ap.add_argument("--ids", nargs="*", default=None, help="only these case ids")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--model", default=None, help="chat model (default: cfg.models.chat_model)")
    ap.add_argument("--provider", default="ollama")
    ap.add_argument("--gate", action="store_true", help="enforce eval/gates.yaml; exit != 0 on miss")
    ap.add_argument("--report", default=str(DEFAULT_REPORT))
    args = ap.parse_args()

    cfg = load_config()
    model = args.model or cfg.models.chat_model
    data = yaml.safe_load(Path(args.case_file).read_text(encoding="utf-8"))
    cases = data["cases"]
    default_budget = int((data.get("defaults") or {}).get("budget_ms", 90000))

    if args.categories:
        cases = [c for c in cases if c.get("category") in args.categories]
    if args.ids:
        cases = [c for c in cases if c.get("id") in args.ids]
    if args.limit:
        cases = cases[: args.limit]

    agent = Agent(cfg)
    fileset = {m["path"].lower() for m in Retriever(cfg)._bm25_meta}

    print(f"{len(cases)} cases · chat model: {model} · index: {cfg.storage.index_state_file.parent}\n")

    rows: list[dict] = []
    for c in cases:
        cid = c["id"]
        category = c.get("category", "definition")
        expect = c.get("expect") or []
        negative = category == "negative"
        history = c.get("history") or []
        budget = int(c.get("budget_ms", default_budget))

        thread_id = str(uuid.uuid4())
        # seed the thread for follow-ups so rewriting has real context
        if history:
            for turn in history[:-1]:
                if turn.get("role") == "user" and turn.get("content"):
                    agent.ask(turn["content"], thread_id=thread_id, provider=args.provider, model=model)

        row = {"id": cid, "category": category, "question": c["q"], "expect": expect, "budget_ms": budget}
        t0 = time.time()
        try:
            last_user = history[-1]["content"] if (history and history[-1].get("role") == "user") else c["q"]
            res = agent.ask(last_user, thread_id=thread_id, provider=args.provider, model=model)
            answer = res["answer"]
            citations = res.get("citations") or []
            m = res.get("metrics") or {}
            latency_ms = int(m.get("latency_ms", (time.time() - t0) * 1000))
            row.update({
                "latency_ms": latency_ms,
                "tokens_in": m.get("tokens_in"), "tokens_out": m.get("tokens_out"),
                "llm_calls": m.get("llm_calls"), "tool_rounds": m.get("tool_rounds"),
                "cost_usd": m.get("cost_usd"),
            })
            row["recall"] = retrieval_stats(res.get("trace", []), expect)
            cited = expect_hit(expect, citations)
            mentioned = expect_hit(expect, [answer or ""])
            grounded = cited or mentioned
            refused = is_refusal(answer) if negative else False
            cit_paths = [path_of(c) for c in citations]
            valid_cites = sum(1 for p in cit_paths if p.lower() in fileset)
            row.update({
                "answer": answer,
                "citations": citations,
                "n_citations": len(citations),
                "cited_expected": cited,
                "mentioned_expected": mentioned,
                "grounded": grounded,
                "refused": refused,
                "citation_validity": (valid_cites / len(cit_paths)) if cit_paths else 1.0,
                "error": None,
            })
            row["pass"] = refused if negative else grounded
            row["budget_ok"] = latency_ms <= budget
        except Exception as e:  # noqa: BLE001 — one bad case must not kill the run
            row.update({"error": f"{type(e).__name__}: {e}", "latency_ms": int((time.time() - t0) * 1000)})
            row.setdefault("pass", False)
            row.setdefault("budget_ok", False)
        rows.append(row)

        mark = "PASS" if row["pass"] else ("ERROR" if row.get("error") else "FAIL")
        if not row["pass"]:
            print(f"[{mark}] {cid:<14} {category:<10} {row.get('latency_ms', '?')}ms "
                  f"{'refused' if row.get('refused') else 'grounded' if row.get('grounded') else ''}")
            if row.get("error"):
                print(f"      error: {row['error']}")

    # ------------------------------------------------------------ aggregate
    def pct(vals):
        return round(sum(vals) / len(vals), 3) if vals else None

    lat = [r["latency_ms"] for r in rows if r.get("latency_ms") is not None]
    # retrieval recall only makes sense for content cases (negatives have no expected file)
    recs = [r["recall"] for r in rows if "recall" in r and r.get("category") != "negative"]
    negs = [r for r in rows if r.get("category") == "negative"]
    non_neg = [r for r in rows if r.get("category") != "negative"]

    summary = {
        "n": len(rows),
        "pass_rate": pct([1.0 if r["pass"] else 0.0 for r in rows if "pass" in r]),
        "budget_ok_rate": pct([1.0 if r["budget_ok"] else 0.0 for r in rows if "budget_ok" in r]),
        "budget_violations": sum(1 for r in rows if not r.get("budget_ok")),
        "latency_p50_ms": percentile(lat, 50), "latency_p95_ms": percentile(lat, 95),
        "recall@1": pct([r["recall@1"] / 1.0 for r in recs]),
        "recall@3": pct([r["recall@3"] / 1.0 for r in recs]),
        "recall@8": pct([r["recall@8"] / 1.0 for r in recs]),
        "mrr": pct([r["mrr"] for r in recs]),
        "grounded_rate": pct([1.0 if r.get("grounded") else 0.0 for r in non_neg]),
        "refusal_precision": pct([1.0 if r.get("refused") else 0.0 for r in negs]) if negs else None,
        "citation_validity": pct([r.get("citation_validity", 1.0) for r in rows if "citations" in r]),
        "total_tokens_in": sum(r.get("tokens_in") or 0 for r in rows),
        "total_tokens_out": sum(r.get("tokens_out") or 0 for r in rows),
        "total_cost_usd": sum(r.get("cost_usd") or 0 for r in rows),
        "errors": sum(1 for r in rows if r.get("error")),
        "model": model,
        "provider": args.provider,
    }
    by_cat: dict[str, dict] = {}
    for r in rows:
        grp = by_cat.setdefault(r.get("category") or "?", {"n": 0, "pass": 0, "lat_ms": []})
        grp["n"] += 1
        grp["pass"] += 1 if r["pass"] else 0
        if r.get("latency_ms") is not None:
            grp["lat_ms"].append(r["latency_ms"])
    for k, v in by_cat.items():
        v["pass_rate"] = round(v["pass"] / v["n"], 3) if v["n"] else None
        v["latency_p50_ms"] = percentile(v["lat_ms"], 50)
        v.pop("lat_ms", None)
    summary["by_category"] = by_cat

    print("-" * 66)
    print(f"{'':22}{'pass%':>6}{'r@1':>6}{'r@3':>6}{'r@8':>6}{'MRR':>6}{'p50ms':>8}")
    print(f"{'OVERALL':22}{summary['pass_rate'] or 0:>6.2f}"
          f"{summary['recall@1'] or 0:>6.2f}{summary['recall@3'] or 0:>6.2f}"
          f"{summary['recall@8'] or 0:>6.2f}{summary['mrr'] or 0:>6.3f}"
          f"{summary['latency_p50_ms'] or 0:>8.0f}")
    for cat, v in sorted(by_cat.items()):
        print(f"{cat+':':22}{v['pass_rate'] or 0:>6.2f}{'':>6}{'':>6}{'':>6}{'':>6}"
              f"{v['latency_p50_ms'] or 0:>8.0f}")
    print("-" * 66)
    print(f"budget violations: {summary['budget_violations']}/{summary['n']} · "
          f"grounded: {summary['grounded_rate']} · "
          f"refusal precision: {summary['refusal_precision']} · "
          f"citation validity: {summary['citation_validity']}")
    print(f"tokens in/out: {summary['total_tokens_in']}/{summary['total_tokens_out']} · "
          f"cost: {summary['total_cost_usd']} · errors: {summary['errors']}")

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps({"summary": summary, "cases": rows}, indent=2), encoding="utf-8")
    print(f"\nReport: {args.report}")

    if args.gate:
        g = _gates().get("gates", {})
        failures = []
        checks = [
            ("pass_rate", summary["pass_rate"], g.get("pass_rate", 0.70)),
            ("recall@3", summary["recall@3"], g.get("recall@3", 0.85)),
            ("refusal_precision", summary["refusal_precision"], g.get("refusal_precision", 0.75)),
            ("citation_validity", summary["citation_validity"], g.get("citation_validity", 0.85)),
        ]
        for name, got, want in checks:
            if got is None:
                got = 0.0
            if float(got) < float(want):
                failures.append(f"  {name}: got {got:.3f} < want {want:.3f}")
        if summary["budget_violations"] > int(g.get("budget_violations_max", 2)):
            failures.append(f"  budget violations: {summary['budget_violations']} > {g.get('budget_violations_max', 2)}")
        if failures:
            print("GATE FAILED:")
            print("\n".join(failures))
            sys.exit(1)
        print("GATE PASSED")


if __name__ == "__main__":
    main()
