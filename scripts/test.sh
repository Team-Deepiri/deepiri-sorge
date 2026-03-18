#!/bin/bash
# Test runner script

set -e

echo "Running deepiri-sorge tests..."

# Check if venv exists
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -q -r requirements.txt
    pip install -q -e ".[dev]"
fi

# Activate venv
source .venv/bin/activate

# Run tests
pytest tests/ -v --cov=bot --cov-report=term-missing

echo ""
echo "Tests complete!"
