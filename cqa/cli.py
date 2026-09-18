"""Interactive chat REPL. See PLAN.md §6 step 5."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from .agent import Agent

BANNER = """CodebaseQA — ask about LuxTrace's source.
Type 'exit' or Ctrl-C to quit. Pass --trace to see retrieval/tool trace.
"""


def main():
    parser = argparse.ArgumentParser(description="CodebaseQA interactive chat REPL")
    parser.add_argument("--trace", action="store_true", help="show the retrieve/reason/tools trace")
    parser.add_argument("question", nargs="*", help="ask one question and exit (non-interactive)")
    args = parser.parse_args()

    agent = Agent()

    if args.question:
        q = " ".join(args.question)
        result = agent.ask(q)
        print(result["answer"])
        if args.trace:
            _print_trace(result["trace"])
        return

    print(BANNER)
    thread_id = None
    while True:
        try:
            q = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q.lower() in ("exit", "quit"):
            break
        result = agent.ask(q, thread_id=thread_id)
        thread_id = result["thread_id"]
        print()
        print(result["answer"])
        print()
        if args.trace:
            _print_trace(result["trace"])


def _print_trace(trace: list[dict]) -> None:
    print("--- trace ---", file=sys.stderr)
    for t in trace:
        print(t, file=sys.stderr)
    print("-------------", file=sys.stderr)


if __name__ == "__main__":
    main()
