# ColBERTv2 + Learned Token Weights (Strategy 1)

Freeze the entire ColBERTv2 encoder. Train only a lightweight weight head (129 params) that learns per-query-token importance. Compare weighted MaxSim vs vanilla MaxSim on identical representations.

## Data

Auto-loaded from HuggingFace:
- `Tevatron/msmarco-passage` — same MS MARCO Passage Ranking data ColBERTv2 was trained on

## Setup

```bash
pip install -r colbert_weighted/requirements.txt
```

## Run

```bash
# Train weight head (encoder frozen)
python run_experiment.py --max_steps 5000 --batch_size 32

# Quick debug
python run_experiment.py --max_rows 100 --max_steps 50

# Try sigmoid normalization
python run_experiment.py --norm sigmoid --output_dir outputs/weighted_sigmoid

# Diagnostics only (load saved weight head)
python run_experiment.py --diagnostics_only --output_dir outputs/weighted
```

## What gets saved

In `outputs/weighted/`:
- `weight_head.pt` — just the 129 learned params (bolt onto any ColBERTv2)
- `model.pt` — full state dict
- `train_log.json` — loss curve
- `token_weights.json` — per-token weights for sample queries
- `weight_dist.json` — entropy, max weight, active token counts
- `pruning_test.json` — scores at top-k pruning levels
