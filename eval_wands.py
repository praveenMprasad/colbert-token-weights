#!/usr/bin/env python3
"""WANDS generalization eval: ESCI-trained weight head on Wayfair data.

Usage:
  # Clone WANDS data first:
  git clone https://github.com/wayfair/WANDS.git wands_repo
  cp -r wands_repo/dataset wands/dataset

  # Run eval:
  python3 eval_wands.py --model_path outputs/esci/gap_multineg/model.pt \
      --weight_head_path outputs/esci/gap_multineg/weight_head_step2000.pt
"""
import argparse
import json
import os
import torch
from transformers import AutoTokenizer

from esci.config import ESCIConfig
from esci.model import ColBERTESCI
from wands.evaluate import evaluate_wands

SAMPLE_QUERIES = [
    "blue velvet sofa",
    "king size bed frame wood",
    "white marble coffee table",
    "outdoor patio furniture set",
    "small bathroom vanity",
    "modern black desk lamp",
    "round dining table for 6",
    "kids bunk bed with storage",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="outputs/esci/gap_multineg/model.pt")
    parser.add_argument("--weight_head_path", default=None)
    parser.add_argument("--data_dir", default="wands/dataset")
    parser.add_argument("--max_queries", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = ESCIConfig(
        use_token_weights=True,
        weight_norm="softmax",
        softmax_temperature=args.temperature,
    )

    model = ColBERTESCI(config).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    if args.weight_head_path:
        print(f"Loading weight head from: {args.weight_head_path}")
        model.weight_head.load_state_dict(
            torch.load(args.weight_head_path, map_location=device))
    model.eval()

    # Weight inspection on WANDS-style queries
    tokenizer = AutoTokenizer.from_pretrained(config.checkpoint)
    print("\n--- Weight Inspection (Wayfair-style queries) ---")
    with torch.no_grad():
        for q in SAMPLE_QUERIES:
            enc = tokenizer(q, return_tensors="pt", padding="max_length",
                           truncation=True, max_length=config.query_maxlen)
            enc = {k: v.to(device) for k, v in enc.items()}
            _, mask, hidden = model.encode(enc["input_ids"], enc["attention_mask"])
            w = model.weight_head(hidden, mask)[0].cpu()
            tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
            m = mask[0].cpu()
            parts = []
            for t, wi, mi in zip(tokens, w, m):
                if mi and t not in ("[CLS]", "[SEP]", "[PAD]"):
                    parts.append(f"{t}={wi:.3f}")
            print(f"  {q:40s} → {', '.join(parts)}")

    # Full eval
    print(f"\n{'='*60}")
    print("WANDS Generalization Eval (zero-shot transfer from ESCI)")
    print(f"{'='*60}")

    results = evaluate_wands(model, config, device,
                             data_dir=args.data_dir,
                             max_queries=args.max_queries)

    print(f"\nVanilla  MRR@10: {results['vanilla_mrr@10']:.4f}")
    print(f"Vanilla NDCG@10: {results['vanilla_ndcg@10']:.4f}")
    if results.get("vanilla_attr_mrr@10"):
        print(f"Vanilla Attr MRR@10: {results['vanilla_attr_mrr@10']:.4f}")
        print(f"Vanilla Non-attr MRR@10: {results['vanilla_non_attr_mrr@10']:.4f}")
    if results.get("vanilla_separation"):
        print(f"Vanilla E>P separation: {results['vanilla_separation']:.4f}")

    if results.get("weighted_mrr@10"):
        print(f"\nWeighted  MRR@10: {results['weighted_mrr@10']:.4f}")
        print(f"Weighted NDCG@10: {results['weighted_ndcg@10']:.4f}")
        if results.get("weighted_attr_mrr@10"):
            print(f"Weighted Attr MRR@10: {results['weighted_attr_mrr@10']:.4f}")
            print(f"Weighted Non-attr MRR@10: {results['weighted_non_attr_mrr@10']:.4f}")
        if results.get("weighted_separation"):
            print(f"Weighted E>P separation: {results['weighted_separation']:.4f}")

    print(f"Queries: {results['num_queries']}")

    # Save
    os.makedirs("outputs/wands", exist_ok=True)
    with open("outputs/wands/eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved to outputs/wands/eval_results.json")


if __name__ == "__main__":
    main()
