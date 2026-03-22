#!/usr/bin/env python3
"""Strategy 1: Freeze ColBERTv2 encoder, train only the weight head.

Baseline = vanilla ColBERTv2 (uniform weights, no training needed)
Proposed = same encoder + learned token weights (129 trainable params)

Both use identical token embeddings — the only variable is the weighting.
"""
import argparse
import json
import os

import torch
from transformers import AutoTokenizer

from colbert_weighted.config import ExpConfig
from colbert_weighted.model import ColBERTWeighted
from colbert_weighted.train import train
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
    tokenizer = AutoTokenizer.from_pretrained(config.checkpoint)
    os.makedirs(output_dir, exist_ok=True)

    # 1. Token weight inspection
    inspection = inspect_token_weights(model, tokenizer, SAMPLE_QUERIES, device)
    with open(os.path.join(output_dir, "token_weights.json"), "w") as f:
        json.dump(inspection, f, indent=2)
    print("\n--- Token Weight Inspection ---")
    for item in inspection:
        print(f"\nQuery: {item['query']}")
        for tok, w in item["token_weights"]:
            print(f"  {tok:15s} {w:.4f}")

    # 2. Weight distribution
    dist = weight_distribution_stats(model, tokenizer, SAMPLE_QUERIES, device)
    with open(os.path.join(output_dir, "weight_dist.json"), "w") as f:
        json.dump(dist, f, indent=2)
    print("\n--- Weight Distribution ---")
    for i, q in enumerate(SAMPLE_QUERIES):
        print(f"  {q[:40]:40s}  max_w={dist['max_weight'][i]:.4f}  "
              f"entropy={dist['entropy'][i]:.4f}  active={dist['active_tokens'][i]}")

    # 3. Pruning test
    dummy_doc = "Paris is the capital and most populous city of France"
    pruning = pruning_test(model, tokenizer, SAMPLE_QUERIES[0], dummy_doc, device=device)
    with open(os.path.join(output_dir, "pruning_test.json"), "w") as f:
        json.dump(pruning, f, indent=2)
    print("\n--- Pruning Test ---")
    print(f"  Query: {SAMPLE_QUERIES[0]}")
    for k, score in pruning.items():
        print(f"  top-{k}: {score:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Strategy 1: Frozen encoder + weight head")
    parser.add_argument("--norm", default="softmax", choices=["softmax", "sigmoid"])
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--output_dir", default="outputs/weighted")
    parser.add_argument("--diagnostics_only", action="store_true",
                        help="Skip training, load saved weight head and run diagnostics")
    parser.add_argument("--eval_queries", type=int, default=None,
                        help="Limit dev queries for eval (default: all 6980)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = ExpConfig(
        use_token_weights=True,
        weight_norm=args.norm,
        batch_size=args.batch_size,
        epochs=args.epochs,
    )

    if args.diagnostics_only:
        print("Loading saved model for diagnostics...")
        model = ColBERTWeighted(config).to(device)
        wh_path = os.path.join(args.output_dir, "weight_head.pt")
        model.weight_head.load_state_dict(torch.load(wh_path, map_location=device))
    else:
        print("=" * 60)
        print(f"Training weight head (norm={args.norm})")
        print("Encoder: frozen ColBERTv2 | Trainable: weight head only")
        print("=" * 60)
        model = train(config, args.output_dir,
                      max_steps=args.max_steps, max_rows=args.max_rows)

    run_diagnostics(model, config, args.output_dir, device=str(device))

    # Dev set reranking evaluation: vanilla vs weighted MRR@10
    from colbert_weighted.eval_rerank import evaluate_reranking
    print("\n" + "=" * 60)
    print("Reranking evaluation on dev set")
    print("=" * 60)
    model.eval()
    results = evaluate_reranking(model, config, device, max_queries=args.eval_queries)
    with open(os.path.join(args.output_dir, "eval_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Vanilla  MRR@10: {results['vanilla_mrr@10']:.4f}")
    print(f"  Weighted MRR@10: {results['weighted_mrr@10']:.4f}")
    print(f"  Vanilla  Acc@1:  {results['vanilla_acc@1']:.4f}")
    print(f"  Weighted Acc@1:  {results['weighted_acc@1']:.4f}")
    print(f"  Queries evaluated: {results['num_queries']}")


if __name__ == "__main__":
    main()
