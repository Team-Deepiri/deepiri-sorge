#!/usr/bin/env python3
"""Run deepiri-sorge locally for testing"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from bot.main import main as bot_main


def main():
    parser = argparse.ArgumentParser(
        description="Run deepiri-sorge locally"
    )
    parser.add_argument(
        "--diff",
        "-d",
        type=str,
        help="Path to diff file",
        required=True,
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="sorge.toml",
        help="Config file path",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't post to GitHub",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    sys.argv = [
        "sorge",
        "--diff", args.diff,
        "--config", args.config,
    ]

    if args.dry_run:
        sys.argv.append("--dry-run")

    if args.verbose:
        sys.argv.append("--verbose")

    bot_main()


if __name__ == "__main__":
    main()
