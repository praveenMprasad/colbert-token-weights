#!/usr/bin/env python3
"""ESCI experiment orchestrator.

Run 1: Vanilla ColBERT baseline (no weights) — both encoders train
Run 2: ColBERT + weight head — everything trains together
Run 3 (optional): Weight head only on frozen Run 1 checkpoint

Usage:
  python3 run_esci.py --run baseline          # Run 1
  python3 run_esci.py --run weighted           # Run 2
  python3 run_esci.py --run eval --eval_model weighted  # Eval only
  python3 run_esci.py --run all                # Run 1 + 2 + eval
"""
import argparse
import json
import os
import torch
from transformers import AutoTokenizer

from esci.config import ESCIConfig
from esci.model import ColBERTESCI
from esci.train import train
from esci.evaluate import evaluate, pruning_eval
from colbert_weighted.diagnostics import inspect_token_weights


SAMPLE_QUERIES = [
    "men's black leather jacket",
    "women's running shoes size 8",
    "blue wireless bluetooth headphones",
    "organic green tea bags",
    "kids waterproof winter boots",
    "stainless steel water bottle 32 oz",
    "red dress for wedding guest",
    "laptop stand adjustable height",
]


def run_diagnostics(model, config, output_dir, device="cpu"):
    tokenizer = AutoTokenizer.from_pretrained(config.checkpoint)

    class Wrapper:
        def __init__(self, m, c):
            self.config = c
            self.weight_head = m.weight_head
            self._model = m
        def encode(self, input_ids, attention_mask):
            return self._model.encode(input_ids, attention_mask)
        def eval(self):
            self._model.eval()
            return self

    wrapper = Wrapper(model, config)
    inspection = inspect_token_weights(wrapper, tokenizer, SAMPLE_QUERIES, device)
    with open(os.path.join(output_dir, "token_weights.json"), "w") as f:
        json.dump(inspection, f, indent=2)

    print("\n--- Token Weight Inspection ---")
    for item in inspection:
        print(f"\nQuery: {item['query']}")
        for tok, w in item["token_weights"]:
            print(f"  {tok:15s} {w:.4f}")


def run_baseline(args):
    """Run 1: Vanilla ColBERT, no weight head."""
    out = os.path.join(args.output_dir, "baseline")
    print("=" * 60)
    print("Run 1: Vanilla ColBERT baseline (no weights)")
    print("=" * 60)
    config = ESCIConfig(
        use_token_weights=False,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    model = train(config, out, max_steps=args.max_steps, max_rows=args.max_rows)
    return model, config, out


def run_weighted(args):
    """Run 2: ColBERT + weight head, everything trains."""
    out = os.path.join(args.output_dir, "weighted")
    print("=" * 60)
    print("Run 2: ColBERT + weight head (all trainable)")
    print("=" * 60)
    config = ESCIConfig(
        use_token_weights=True,
        weight_norm=args.norm,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    model = train(config, out, max_steps=args.max_steps, max_rows=args.max_rows)
    return model, config, out


def run_eval(model, config, out, device, args):
    """Evaluate a trained model."""
    model.eval()
    print(f"\n{'=' * 60}")
    print(f"Evaluating: {out}")
    print(f"{'=' * 60}")

    results = evaluate(model, config, device, max_queries=args.num_eval, split="test")
    with open(os.path.join(out, "eval_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  Vanilla  MRR@10: {results['vanilla_mrr@10']:.4f}")
    print(f"  Vanilla NDCG@10: {results['vanilla_ndcg@10']:.4f}")
    if results.get("vanilla_attr_mrr@10") is not None:
        print(f"  Vanilla Attr-query MRR@10: {results['vanilla_attr_mrr@10']:.4f}")
        print(f"  Vanilla Non-attr  MRR@10: {results['vanilla_non_attr_mrr@10']:.4f}")
    if results.get("vanilla_separation") is not None:
        print(f"  Vanilla E>S separation: {results['vanilla_separation']:.4f}")

    if config.use_token_weights:
        print(f"\n  Weighted  MRR@10: {results['weighted_mrr@10']:.4f}")
        print(f"  Weighted NDCG@10: {results['weighted_ndcg@10']:.4f}")
        if results.get("weighted_attr_mrr@10") is not None:
            print(f"  Weighted Attr-query MRR@10: {results['weighted_attr_mrr@10']:.4f}")
            print(f"  Weighted Non-attr  MRR@10: {results['weighted_non_attr_mrr@10']:.4f}")
        if results.get("weighted_separation") is not None:
            print(f"  Weighted E>S separation: {results['weighted_separation']:.4f}")

    print(f"  Queries: {results['num_queries']}")

    # Pruning eval for weighted model
    if config.use_token_weights:
        print(f"\n--- Pruning Test ---")
        prune_results = pruning_eval(model, config, device, max_queries=min(args.num_eval, 200))
        if prune_results:
            with open(os.path.join(out, "pruning_results.json"), "w") as f:
                json.dump(prune_results, f, indent=2)
            for k, v in prune_results.items():
                if v is not None:
                    print(f"  {k}: {v:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="ESCI experiment")
    parser.add_argument("--run", required=True,
                        choices=["baseline", "weighted", "eval", "original", "all"])
    parser.add_argument("--norm", default="softmax", choices=["softmax", "sigmoid"])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--num_eval", type=int, default=1000)
    parser.add_argument("--output_dir", default="outputs/esci")
    parser.add_argument("--eval_model", default="weighted",
                        choices=["baseline", "weighted"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.run == "baseline":
        model, config, out = run_baseline(args)
        run_eval(model, config, out, device, args)

    elif args.run == "original":
        print("=" * 60)
        print("Original ColBERTv2 (no ESCI fine-tuning)")
        print("=" * 60)
        config = ESCIConfig(use_token_weights=False)
        model = ColBERTESCI(config).to(device)
        out = os.path.join(args.output_dir, "original")
        os.makedirs(out, exist_ok=True)
        run_eval(model, config, out, device, args)

    elif args.run == "weighted":
        model, config, out = run_weighted(args)
        run_diagnostics(model, config, out, device=str(device))
        run_eval(model, config, out, device, args)

    elif args.run == "eval":
        out = os.path.join(args.output_dir, args.eval_model)
        use_weights = (args.eval_model == "weighted")
        config = ESCIConfig(
            use_token_weights=use_weights,
            weight_norm=args.norm,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
        model = ColBERTESCI(config).to(device)
        model.load_state_dict(torch.load(
            os.path.join(out, "model.pt"), map_location=device))
        if use_weights:
            run_diagnostics(model, config, out, device=str(device))
        run_eval(model, config, out, device, args)

    elif args.run == "all":
        # Run 0: Original ColBERTv2 (no fine-tuning)
        print("=" * 60)
        print("Run 0: Original ColBERTv2 (no ESCI fine-tuning)")
        print("=" * 60)
        orig_config = ESCIConfig(use_token_weights=False)
        orig_model = ColBERTESCI(orig_config).to(device)
        orig_out = os.path.join(args.output_dir, "original")
        os.makedirs(orig_out, exist_ok=True)
        orig_results = run_eval(orig_model, orig_config, orig_out, device, args)
        del orig_model
        torch.cuda.empty_cache()

        # Run 1: baseline
        b_model, b_config, b_out = run_baseline(args)
        b_results = run_eval(b_model, b_config, b_out, device, args)
        del b_model
        torch.cuda.empty_cache()

        # Run 2: weighted
        w_model, w_config, w_out = run_weighted(args)
        run_diagnostics(w_model, w_config, w_out, device=str(device))
        w_results = run_eval(w_model, w_config, w_out, device, args)

        # Summary
        print(f"\n{'=' * 60}")
        print("ESCI Experiment Summary")
        print(f"{'=' * 60}")
        print(f"  Original ColBERTv2 MRR@10: {orig_results['vanilla_mrr@10']:.4f}")
        print(f"  Finetuned baseline MRR@10: {b_results['vanilla_mrr@10']:.4f}")
        print(f"  Finetuned weighted MRR@10: {w_results.get('weighted_mrr@10', 'N/A')}")
        print(f"  Original ColBERTv2 NDCG@10: {orig_results['vanilla_ndcg@10']:.4f}")
        print(f"  Finetuned baseline NDCG@10: {b_results['vanilla_ndcg@10']:.4f}")
        print(f"  Finetuned weighted NDCG@10: {w_results.get('weighted_ndcg@10', 'N/A')}")


if __name__ == "__main__":
    main()
