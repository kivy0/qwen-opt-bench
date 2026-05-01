# Qwen optimizers benchmark

## Results

All three optimization strategies (AdamW, Muon, Hybrid) show nearly identical convergence behavior over 1500 training steps. Differences in downstream performance are marginal (≤0.3 percentage points in average accuracy).

**Average benchmark accuracy:**

- **Hybrid:** 0.5260  
- **AdamW:** 0.5254  
- **Muon:** 0.5230  
- **Base model:** 0.5208  

The Hybrid configuration achieves the highest aggregate score, but the gap relative to AdamW is minimal and likely within single-run variance. Importantly, neither Muon nor Hybrid introduces training instability, confirming their safe applicability even in small-batch, resource-constrained settings.

### Artifacts

**All artifacts are available on Hugging Face:**
- [🤗 Results Repository](https://huggingface.co/kivy0/qwen-opt-bench-result)

Includes: training logs, metrics, model checkpoints, evaluation results for every experiment, and base model evaluation.


## Installation and launch

### 1. Install package manager `uv`
```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Clone the repository

```bash
git clone https://github.com/kivy0/qwen-opt-bench.git
cd qwen-opt-bench
```

---

## Running Experiments

### Quick Start with Bash Scripts
We provide pre-configured scripts in the `bash/` directory for easy reproduction.

**Run AdamW baseline:**
```bash
bash bash/run_adamw.sh
```

**Run full reproduction (all optimizers):**
```bash
bash bash/run_fullreproduce.sh
```

All artifacts, from logs and weights to metrics, will be saved in the experiment directory `runs/<experiment_name>`.


---

### Custom Training
To launch training manually with a specific configuration, use the `scripts/train.py` script:

```bash
uv run scripts/train.py --config <path_to_config>
```

**Arguments:**
*   `--config`: Path to the YAML configuration file.

---

### Evaluation
Use the `scripts/evaluate.py` script to test your models on benchmark datasets.

```bash
uv run scripts/evaluate.py [ARGUMENTS]
```

**Arguments:**
*   `--config`: Path to the config file.
*   `--checkpoint`: Path to a specific model checkpoint. If not provided, it defaults to `runs/<experiment_name>`.
*   `--baseline`: Flag to evaluate the raw baseline model (defined in `config.model.name_or_path`) instead of a trained local checkpoint.
