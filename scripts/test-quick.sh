#!/bin/bash
# Quick test script - runs tests without coverage

set -e

source .venv/bin/activate

echo "Running tests..."
pytest tests/ -v

echo ""
echo "All tests passed!"
