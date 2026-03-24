#!/usr/bin/env python3
"""Download quantized models for deepiri-sorge"""

import argparse
import sys
from pathlib import Path

try:
    from bot.cpu_reviewer import MODEL_URLS, download_model
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from bot.cpu_reviewer import MODEL_URLS, download_model


def main():
    parser = argparse.ArgumentParser(
        description="Download quantized models for deepiri-sorge"
    )
    parser.add_argument(
        "--model",
        "-m",
        choices=list(MODEL_URLS.keys()),
        default="codellama-7b-q4",
        help="Model to download (default: codellama-7b-q4)",
    )
    parser.add_argument(
        "--target",
        "-t",
        type=str,
        help="Target directory for model",
    )
    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="List available models",
    )

    args = parser.parse_args()

    if args.list:
        print("Available models:")
        for name, _url in MODEL_URLS.items():
            size = "4.8GB"
            print(f"  {name}: {size}")
        print()
        print("Default: codellama-7b-q4 (Code-Specific)")
        return

    target_dir = Path(args.target) if args.target else None

    print(f"Downloading {args.model}...")
    print(f"URL: {MODEL_URLS[args.model]}")

    try:
        path = download_model(args.model, target_dir)
        print(f"\nModel downloaded to: {path}")
        print("\nTo use this model, set SORGE_MODEL_PATH environment variable:")
        print(f"  export SORGE_MODEL_PATH={path}")
        print("\nOr add to your sorge.toml:")
        print("  [model]")
        print(f'  path = "{path}"')
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
