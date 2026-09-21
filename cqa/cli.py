"""CLI: interactive chat REPL, one-shot questions, diff review and file/module tours.

    python -m cqa.cli                                   interactive REPL
    python -m cqa.cli --trace "how does X work?"        one-shot question
    python -m cqa.cli review --base main --head HEAD    review a git range (or --diff file.patch)
    python -m cqa.cli tour src/core/ThreadPool.cpp      onboarding tour of a file or module

See PLAN.md §6 step 5.
"""
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


def _metrics_line(m: dict | None, cached: bool = False) -> str:
    if cached:
        return "[cached tour · instant]"
    if not m:
        return ""
    tok = (m.get("tokens_in") or 0) + (m.get("tokens_out") or 0)
    cost = m.get("cost_usd")
    return (
        f"[{(m.get('latency_ms') or 0) / 1000:.1f}s · {tok} tok"
        + ("" if cost is None else (" · local" if cost == 0 else f" · ${cost:.4f}"))
        + f" · {m.get('llm_calls', 0)} LLM calls · {m.get('tool_rounds', 0)} tool rounds]"
    )


def _run_stream(events, show_trace: bool) -> None:
    """Print a streamed turn: answer text as it arrives, then metrics (stderr)."""
    streamed = False
    for ev in events:
        t = ev.get("type")
        if t == "token":
            streamed = True
            print(ev["text"], end="", flush=True)
        elif t == "token_reset":
            streamed = False
            print("\n[restarting answer on the fallback model]", file=sys.stderr)
        elif t == "reason" and ev.get("tool_calls"):
            streamed = False   # text before a tool call was thinking, not the answer
            print(f"\n… {', '.join(ev['tool_calls'])}", file=sys.stderr)
        elif t == "error":
            print(f"error: {ev['message']}", file=sys.stderr)
            sys.exit(1)
        elif t == "final":
            if not streamed:
                print(ev["answer"], end="")
            print()
            print(_metrics_line(ev.get("metrics"), ev.get("cached", False)), file=sys.stderr)
            if show_trace:
                _print_trace(ev.get("trace", []))


def _cmd_review(args) -> None:
    agent = Agent()
    diff = Path(args.diff).read_text(encoding="utf-8", errors="replace") if args.diff else None
    try:
        prep = agent.prepare_review(base=args.base, head=args.head, diff=diff)
    except ValueError as e:
        sys.exit(f"review: {e}")
    s = prep.stats
    print(f"{prep.title}: {s['files']} files, +{s['additions']} -{s['deletions']}, "
          f"{s['changed_symbols']} changed symbols", file=sys.stderr)
    for w in prep.warnings:
        print(f"warning: {w}", file=sys.stderr)
    _run_stream(agent.stream_review(prep, provider=args.provider, model=args.model), args.trace)


def _cmd_tour(args) -> None:
    agent = Agent()
    try:
        prep = agent.prepare_tour(args.target)
    except ValueError as e:
        sys.exit(f"tour: {e}")
    _run_stream(agent.stream_tour(prep, provider=args.provider, model=args.model, refresh=args.refresh), args.trace)


def _chat_main() -> None:
    parser = argparse.ArgumentParser(description="CodebaseQA interactive chat REPL")
    parser.add_argument("--trace", action="store_true", help="show the retrieve/reason/tools trace")
    parser.add_argument("question", nargs="*", help="ask one question and exit (non-interactive)")
    args = parser.parse_args()

    agent = Agent()

    if args.question:
        _run_stream(agent.stream(" ".join(args.question)), args.trace)
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
        thread_id = thread_id or __import__("uuid").uuid4().hex
        print()
        _run_stream(agent.stream(q, thread_id=thread_id), args.trace)
        print()


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] in ("review", "tour"):
        cmd = sys.argv.pop(1)
        p = argparse.ArgumentParser(prog=f"cqa.cli {cmd}")
        p.add_argument("--provider", default="ollama")
        p.add_argument("--model", default=None)
        p.add_argument("--trace", action="store_true")
        if cmd == "review":
            p.add_argument("--base", default=None, help="base ref (default: config review.default_base)")
            p.add_argument("--head", default=None, help="head ref (default: config review.default_head)")
            p.add_argument("--diff", default=None, help="path to a unified diff file instead of a git range")
            _cmd_review(p.parse_args())
        else:
            p.add_argument("target", help="an indexed file (src/core/ThreadPool.cpp) or module (core, src/gpu)")
            p.add_argument("--refresh", action="store_true", help="ignore the cached tour and regenerate")
            _cmd_tour(p.parse_args())
        return
    _chat_main()


def _print_trace(trace: list[dict]) -> None:
    print("--- trace ---", file=sys.stderr)
    for t in trace:
        print(t, file=sys.stderr)
    print("-------------", file=sys.stderr)


if __name__ == "__main__":
    main()
