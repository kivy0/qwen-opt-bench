set -e
uv run scripts/train.py
uv run scripts/evaluate.py
