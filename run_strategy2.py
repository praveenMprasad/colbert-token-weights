#!/usr/bin/env python3
"""Strategy 2: Query encoder + weight head co-train, doc encoder frozen.

Trains both the query-side BERT encoder and the weight head together.
Doc encoder stays frozen — no re-indexing needed.
Evaluates on held-out queries comparing:
  - Baseline ColBERTv2 (vanilla MaxSim)
  - S2 with weighted MaxSim
  - S2 with vanilla MaxSim (isolates encoder vs weight contribution)
"""
import argparse
import json
import os
import torch
from transformers import AutoTokenizer

from strategy2.config import S2Config
from strategy2.train import train
from strategy2.evaluate import evaluate
from strategy2.model import ColBERTWeightedS2
from colbert_weighted.model import ColBERTWeighted
from colbert_weighted.config import ExpConfig
from colbert_weighted.diagnostics import (
    inspect_token_weights,
    weight_distribution_stats,
    pruning_test,
)

SAMPLE_QUERIES = [
    "what is the capital of france",
    "how many people live in new york city",
    "who invented the telephone",
    "average temperature in death valley",
    "what causes earthquakes",
]


def run_diagnostics(model, config, output_dir, device="cpu"):
    """Reuse S1 diagnostics — model just needs .encode() and .weight_head."""
    tokenizer = AutoTokenizer.from_pretrained(config.checkpoint)
    os.makedirs(output_dir, exist_ok=True)

    # Wrap S2 model to have .encode() for diagnostics compatibility
    class S2Wrapper:
        def __init__(self, m, c):
            self.config = c
            self.weight_head = m.weight_head
        def encode(self, input_ids, attention_mask):
            return m.encode_query(input_ids, attention_mask)

    wrapper = S2Wrapper(model, config)

    inspection = inspect_token_weights(wrapper, tokenizer, SAMPLE_QUERIES, device)
    with open(os.path.join(output_dir, "token_weights.json"), "w") as f:
        json.dump(inspection, f, indent=2)
    print("\n--- Token Weight Inspection ---")
    for item in inspection:
        print(f"\nQuery: {item['query']}")
        for tok, w in item["token_weights"]:
            print(f"  {tok:15s} {w:.4f}")

    dist = weight_distribution_stats(wrapper, tokenizer, SAMPLE_QUERIES, device)
    with open(os.path.join(output_dir, "weight_dist.json"), "w") as f:
        json.dump(dist, f, indent=2)
    print("\n--- Weight Distribution ---")
    for i, q in enumerate(SAMPLE_QUERIES):
        print(f"  {q[:40]:40s}  max_w={dist['max_weight'][i]:.4f}  "
              f"entropy={dist['entropy'][i]:.4f}  active={dist['active_tokens'][i]}")


def main():
    parser = argparse.ArgumentParser(description="Strategy 2: co-train encoder + weights")
    parser.add_argument("--norm", default="softmax", choices=["softmax", "sigmoid"])
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--output_dir", default="outputs/strategy2")
    parser.add_argument("--num_eval", type=int, default=1000)
    parser.add_argument("--eval_only", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = S2Config(
        weight_norm=args.norm,
        batch_size=args.batch_size,
        epochs=args.epochs,
    )

    if args.eval_only:
        print("Loading saved S2 model...")
        model = ColBERTWeightedS2(config).to(device)
        model.load_state_dict(torch.load(
            os.path.join(args.output_dir, "model.pt"), map_location=device))
    else:
        print("=" * 60)
        print(f"Strategy 2: encoder + weight head (norm={args.norm})")
        print("Query encoder: trainable | Doc encoder: frozen")
        print("=" * 60)
        model = train(config, args.output_dir, max_steps=args.max_steps)

    model.eval()
    run_diagnostics(model, config, args.output_dir, device=str(device))

    # Evaluation
    print("\n" + "=" * 60)
    print(f"Reranking eval on {args.num_eval} held-out queries")
    print("=" * 60)

    baseline_config = ExpConfig(use_token_weights=False)
    baseline_model = ColBERTWeighted(baseline_config).to(device)
    baseline_model.eval()

    results = evaluate(model, baseline_model, config, device, num_eval=args.num_eval)
    with open(os.path.join(args.output_dir, "eval_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  Baseline (vanilla ColBERTv2)  MRR@10: {results['baseline']:.4f}")
    print(f"  S2 weighted MaxSim           MRR@10: {results['s2_weighted']:.4f}")
    print(f"  S2 vanilla MaxSim (no wts)   MRR@10: {results['s2_vanilla']:.4f}")
    print(f"  Queries: {results['num_queries']}")


if __name__ == "__main__":
    main()
