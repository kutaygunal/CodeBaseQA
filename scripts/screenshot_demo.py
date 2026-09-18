#!/usr/bin/env python
"""Regenerate the README screenshots in docs/screenshots/.

Requires the `screenshots` extra (`uv pip install -e ".[screenshots]" --python .venv`) and
a running web server (`uvicorn web.server:app`). Uses the system-installed Edge via
Playwright's `channel="msedge"` — no extra browser download.

Note: `--disable-features=WebContentsForceDark,...` is required — Chromium's built-in
auto-dark-mode heuristic (triggered by content complexity, e.g. many syntax-highlighted
spans) will otherwise force-invert the page to dark regardless of the app's own light
theme, even though the DOM/CSSOM correctly report light colors throughout. Cost real time
to track down; see HANDOVER.md if this regresses.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "docs" / "screenshots"
BASE = "http://localhost:8000/"
THREAD = "b076ff16-120a-4bc8-9f77-7caea554554b"  # "What does RayFile store..." — see README

SHOTS = [
    (f"?thread={THREAD}", "01_chat_answer.png", True),
    (f"?thread={THREAD}&source=src/core/RayFile.h:50-96", "02_source_panel.png", False),
    ("?settings=1", "03_settings.png", False),
    (f"?theme=light&thread={THREAD}&source=src/core/RayFile.h:50-96", "04_light_theme.png", False),
]


def main():
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print('Missing playwright. Run: uv pip install -e ".[screenshots]" --python .venv')
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(
            channel="msedge",
            headless=True,
            args=[
                "--disable-features=WebContentsForceDark,ForceDarkMode,AutoDarkMode",
                "--force-color-profile=srgb",
            ],
        )
        for query, filename, scroll_top in SHOTS:
            page = browser.new_page(viewport={"width": 1280, "height": 900}, color_scheme="dark")
            page.goto(BASE + query, wait_until="networkidle")
            page.wait_for_timeout(800)
            if scroll_top:
                page.evaluate("document.querySelector('main').scrollTop = 0")
                page.wait_for_timeout(200)
            out = OUT_DIR / filename
            page.screenshot(path=str(out))
            page.close()
            print(f"saved {out}")
        browser.close()


if __name__ == "__main__":
    main()
