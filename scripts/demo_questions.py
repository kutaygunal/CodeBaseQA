#!/usr/bin/env python
"""Run the PLAN.md §7 demo questions and record Q&A + trace for the README."""
import json
import sys
import time
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cqa.agent import Agent

QUESTIONS = [
    "How does a simulation run from the UI to the result?",
    "What functions call TraceScene::Trace?",
    "Where is SimulationResult used and how is it produced?",
    "How does the GPU path differ from the CPU trace?",
    "What does RayFile store and where is it written/read?",
]


def main():
    agent = Agent()
    out = []
    for q in QUESTIONS:
        print(f"\n{'=' * 70}\nQ: {q}\n{'=' * 70}")
        t0 = time.time()
        result = agent.ask(q)
        dt = time.time() - t0
        print(result["answer"])
        print(f"\n[{dt:.1f}s, {len(result['citations'])} citations, "
              f"{sum(1 for t in result['trace'] if t['node']=='tools')} tool rounds]")
        out.append({"question": q, **result, "duration_s": round(dt, 1)})

    out_path = Path(__file__).resolve().parent.parent / "demo_output.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
