#!/usr/bin/env python
"""Tiny retrieval benchmark: recall@k / MRR for each retrieval feature toggle.

    python scripts/retrieval_bench.py                 # all configurations
    python scripts/retrieval_bench.py --only baseline rerank
    python scripts/retrieval_bench.py --config config/other.yaml   # e.g. a contextual index
    python scripts/retrieval_bench.py -v              # list the cases each config misses

Cases live in eval/bench_cases.yaml (hand-labeled: a case hits when ANY expected path
appears in the retrieved list). This measures the retrieval layer only — no answer
generation — so it is cheap and repeatable. The `analyze` configurations make one small
LLM call per question (cached across configs within a run).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from cqa.config import PROJECT_ROOT, load_config
from cqa.query_analysis import analyze_question
from cqa.retriever import Retriever

# name -> (rerank, expand, analyze, rerank_fuse_weight)
CONFIGS = {
    "baseline": (False, False, False, 0.0),
    "rerank": (True, False, False, 0.0),
    "rerank-fused": (True, False, False, 1.0),
    "analyze": (False, False, True, 0.0),
    "expand": (False, True, False, 0.0),
    "rerank+analyze": (True, False, True, 0.0),
    "fused+analyze": (True, False, True, 1.0),
    "all": (True, True, True, 0.0),
    "all-fused": (True, True, True, 1.0),
}
KS = (1, 3, 8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--cases", default=str(PROJECT_ROOT / "eval" / "bench_cases.yaml"))
    ap.add_argument("--only", nargs="*", default=None, help="configuration names to run")
    ap.add_argument(
        "--qi", default=None, metavar="TEXT",
        help="experimental override for retrieval.embed_query_instruction (empty string disables)",
    )
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cases = yaml.safe_load(Path(args.cases).read_text(encoding="utf-8"))["cases"]
    retriever = Retriever(cfg)
    if args.qi is not None:
        retriever.cfg.retrieval.embed_query_instruction = args.qi
    provider, model = "ollama", cfg.models.chat_model
    analyses: dict[int, tuple] = {}  # case index -> (Analysis, seconds)
    names = args.only or list(CONFIGS)
    print(f"{len(cases)} cases · index: {cfg.storage.index_state_file.parent} · chat model for analyze: {model}")
    if args.qi is not None:
        print("query instruction: (none)" if not args.qi else f"query instruction: {args.qi[:44]!r}…")
    print()
    header = f"{'config':<16}" + "".join(f"recall@{k:<3}" for k in KS) + "r@all    MRR    avg_ms"
    print(header)
    print("-" * len(header))
    for name in names:
        rerank, expand, analyze, fw = CONFIGS[name]
        hits_at = {k: 0 for k in KS}
        rr_sum = 0.0
        elapsed = 0.0
        missed: list[str] = []
        found_any = 0
        for ci, case in enumerate(cases):
            q = case["q"]
            expect = set(case["expect"])
            extra: list[str] = []
            search_q = q
            if analyze:
                if ci not in analyses:
                    t0 = time.time()
                    a = analyze_question(q, case.get("history") or [], cfg, provider, model)
                    analyses[ci] = (a, time.time() - t0)
                a, a_secs = analyses[ci]
                search_q, extra = a.standalone, a.queries
                elapsed += a_secs
            t1 = time.time()
            res = retriever.search_ex(search_q, extra_queries=extra, rerank=rerank, expand=expand, rerank_fuse_weight=fw)
            elapsed += time.time() - t1
            rank = next((i for i, h in enumerate(res.hits, 1) if h.path in expect), None)
            for k in KS:
                if rank is not None and rank <= k:
                    hits_at[k] += 1
            rr_sum += 1.0 / rank if rank else 0.0
            if rank is not None:
                found_any += 1
            if rank is None or rank > 8:
                missed.append(q[:70] + ("" if rank is None else f"  (found at rank {rank})"))
        n = len(cases)
        row = f"{name:<16}" + "".join(f"{hits_at[k] / n:<10.2f}" for k in KS)
        row += f"{found_any / n:<8.2f}{rr_sum / n:<7.3f}{elapsed / n * 1000:<7.0f}"
        print(row)
        if args.verbose and missed:
            for m in missed:
                print(f"    miss: {m}")


if __name__ == "__main__":
    main()
