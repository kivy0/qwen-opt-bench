set -e
uv run scripts/train.py --config configs/hybrid.yaml
uv run scripts/evaluate.py --config configs/hybrid.yaml
