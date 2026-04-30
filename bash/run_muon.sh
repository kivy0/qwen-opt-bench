set -e
uv run scripts/train.py --config configs/muon.yaml
uv run scripts/evaluate.py --config configs/muon.yaml
