"""
Quick smoke test for the You.com Contents API integration.

Usage:
    python -m scripts.check_you_api --url https://example.com/article
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dashboard.content_api import ContentAPI  # noqa: E402


async def run_smoke(urls: List[str], content_format: str, force_live: bool) -> int:
    """Run a quick fetch and print diagnostic information."""
    mock_mode = None if not force_live else False
    api = ContentAPI(mock_mode=mock_mode)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.info("YOU_API_KEY configured: %s", "yes" if api.api_key else "no")
    logging.info("Mock mode: %s", api.mock_mode)
    logging.info("Target URLs: %s", urls)
    logging.info("Requested format: %s", content_format)

    pages = await api.fetch(urls, format=content_format)

    for page in pages:
        status = "OK" if page.error is None else f"ERROR ({page.error})"
        preview = (page.content or "").strip().splitlines()
        snippet = preview[0][:120] if preview else "<empty>"
        logging.info("- %s -> %s | %s", page.url, status, snippet)

    stats = api.get_stats()
    logging.info("Stats: %s", stats)
    logging.info("Budget: %s", api.get_budget_status())

    errors = sum(1 for page in pages if page.error)
    return 0 if errors == 0 else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test You.com Content API wrapper.")
    parser.add_argument("--url", action="append", required=True, help="URL to fetch (can be provided multiple times)")
    parser.add_argument("--format", default="markdown", help="Output format (markdown or html). Default: markdown")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Force live mode (fail if YOU_API_KEY missing). Without this flag the wrapper auto-detects mock/live.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return asyncio.run(run_smoke(args.url, args.format, force_live=args.live))


if __name__ == "__main__":
    raise SystemExit(main())
