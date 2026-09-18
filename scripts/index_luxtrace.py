#!/usr/bin/env python
"""One-shot ingestion of ../LuxTrace. See PLAN.md §6 step 1 / §8 definition of done."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cqa.config import load_config
from cqa.ingest import build_index


def main():
    cfg = load_config()
    state = build_index(cfg)
    print()
    print("=== Index state ===")
    print(f"repo_root   : {state['repo_root']}")
    print(f"git_commit  : {state['git_commit']}")
    print(f"files       : {state['num_files']}")
    print(f"chunks      : {state['num_chunks']}")
    print(f"symbols     : {state['num_symbols']}")
    print(f"skipped     : {len(state['skipped'])}")
    print(f"duration    : {state['duration_s']}s")
    if state["skipped"]:
        print("\nWARNING: some files were skipped:")
        for s in state["skipped"]:
            print(f"  - {s}")
        sys.exit(1)


if __name__ == "__main__":
    main()
