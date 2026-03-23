#!/usr/bin/env python3
"""Eval-only for gap-trained model."""
import argparse
import json
import os
import torch
from transformers import AutoTokenizer

from esci.config import ESCIConfig
from esci.model import ColBERTESCI
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="outputs/esci/gap_weighted/model.pt")
    parser.add_argument("--weight_head_path", default=None,
                        help="Override weight head from a specific checkpoint")
    parser.add_argument("--num_eval", type=int, default=50)
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
        print(f"Overriding weight head from: {args.weight_head_path}")
        model.weight_head.load_state_dict(torch.load(args.weight_head_path, map_location=device))
    model.eval()

    # Token weight inspection
    tokenizer = AutoTokenizer.from_pretrained(config.checkpoint)
    print("\n--- Token Weight Inspection ---")
    with torch.no_grad():
        for q in SAMPLE_QUERIES:
            enc = tokenizer(q, return_tensors="pt", padding="max_length",
                           truncation=True, max_length=config.query_maxlen)
            enc = {k: v.to(device) for k, v in enc.items()}
            _, mask, hidden = model.encode(enc["input_ids"], enc["attention_mask"])
            w = model.weight_head(hidden, mask)[0].cpu()
            tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
            m = mask[0].cpu()
            print(f"\nQuery: {q}")
            for t, wi, mi in zip(tokens, w, m):
                if mi and t not in ("[PAD]"):
                    print(f"  {t:15s} {wi:.4f}")

    # Full eval
    print(f"\n{'='*60}")
    print(f"Evaluating gap-trained model (temp={args.temperature})")
    print(f"{'='*60}")
    results = evaluate(model, config, device, max_queries=args.num_eval, split="test")

    print(f"\nVanilla  MRR@10: {results['vanilla_mrr@10']:.4f}")
    print(f"Vanilla NDCG@10: {results['vanilla_ndcg@10']:.4f}")
    if results.get("vanilla_attr_mrr@10"):
        print(f"Vanilla Attr MRR@10: {results['vanilla_attr_mrr@10']:.4f}")
        print(f"Vanilla Non-attr MRR@10: {results['vanilla_non_attr_mrr@10']:.4f}")
    if results.get("vanilla_separation"):
        print(f"Vanilla E>S separation: {results['vanilla_separation']:.4f}")

    print(f"\nWeighted  MRR@10: {results['weighted_mrr@10']:.4f}")
    print(f"Weighted NDCG@10: {results['weighted_ndcg@10']:.4f}")
    if results.get("weighted_attr_mrr@10"):
        print(f"Weighted Attr MRR@10: {results['weighted_attr_mrr@10']:.4f}")
        print(f"Weighted Non-attr MRR@10: {results['weighted_non_attr_mrr@10']:.4f}")
    if results.get("weighted_separation"):
        print(f"Weighted E>S separation: {results['weighted_separation']:.4f}")
    print(f"Queries: {results['num_queries']}")

    # Pruning
    print(f"\n--- Pruning Test ---")
    prune = pruning_eval(model, config, device, max_queries=args.num_eval)
    if prune:
        for k, v in prune.items():
            if v is not None:
                print(f"  {k}: {v:.4f}")

if __name__ == "__main__":
    main()
