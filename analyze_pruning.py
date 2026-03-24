#!/usr/bin/env python3
"""Analyze pruning efficiency on ESCI and WANDS queries.

Shows: actual token counts, compute savings at each top-k, MRR tradeoffs.
"""
import argparse
import numpy as np
from transformers import AutoTokenizer
from collections import Counter


def analyze(dataset, data_dir=None):
    tokenizer = AutoTokenizer.from_pretrained("colbert-ir/colbertv2.0")

    if dataset == "esci":
        from esci.data import ESCIRerankDataset
        rerank_data = list(ESCIRerankDataset(split="test", locale="us", max_queries=1000))
        queries = [item["query"] for item in rerank_data]
        vanilla_mrr = 0.8305
        gap_results = {2: 0.8072, 4: 0.8341, 8: 0.8352, "all": 0.8326}
    else:
        from wands.evaluate import load_wands
        data = load_wands(data_dir)
        queries = [item["query"] for item in data]
        vanilla_mrr = 0.5696
        gap_results = {2: 0.5237, 4: 0.5712, 8: 0.5769, "all": 0.5758}

    special = {"[CLS]", "[SEP]", "[PAD]", "[Q]", "[MASK]"}
    token_counts = []

    for query in queries:
        enc = tokenizer(query, padding="max_length", truncation=True,
                        max_length=32, return_tensors="pt")
        tokens = tokenizer.convert_ids_to_tokens(enc.input_ids[0])
        mask = enc.attention_mask[0]
        real = sum(1 for t, m in zip(tokens, mask) if m == 1 and t not in special)
        active = int(mask.sum())
        token_counts.append({"real": real, "active": active})

    real_counts = [t["real"] for t in token_counts]
    active_counts = [t["active"] for t in token_counts]
    name = "ESCI" if dataset == "esci" else "WANDS"

    print("=" * 60)
    print(f"{name} Query Token Analysis ({len(queries)} queries)")
    print("=" * 60)
    print(f"\nQuery maxlen (padded): 32")
    print(f"\nReal content tokens (excluding special tokens):")
    print(f"  Mean:   {np.mean(real_counts):.1f}")
    print(f"  Median: {np.median(real_counts):.1f}")
    print(f"  Min:    {np.min(real_counts)}")
    print(f"  Max:    {np.max(real_counts)}")
    print(f"  Std:    {np.std(real_counts):.1f}")

    print(f"\nActive tokens (incl. special):")
    print(f"  Mean:   {np.mean(active_counts):.1f}")
    print(f"  Median: {np.median(active_counts):.1f}")

    print(f"\nReal token count distribution:")
    dist = Counter(real_counts)
    for k in sorted(dist.keys()):
        bar = "#" * (dist[k] // max(1, len(queries) // 200))
        print(f"  {k:3d} tokens: {dist[k]:4d} queries  {bar}")

    mean_active = np.mean(active_counts)

    print(f"\n{'=' * 60}")
    print(f"Pruning Efficiency — {name} (Gap method)")
    print(f"{'=' * 60}")
    print(f"\n{'Tokens kept':<15} {'MRR@10':<10} {'vs Vanilla':<12} {'Vectors':<18} {'Compute saving'}")
    print("-" * 75)

    for k in [2, 4, 8, "all"]:
        mrr = gap_results[k]
        delta = mrr - vanilla_mrr
        if k == "all":
            kept = mean_active
            label = f"All ({mean_active:.0f} avg)"
        else:
            kept = k
            label = f"Top {k}"
        pct_saved = ((mean_active - kept) / mean_active) * 100
        speedup = mean_active / kept if kept > 0 else 0
        print(f"{label:<15} {mrr:.4f}     {delta:+.4f}       {kept:.0f}/{mean_active:.0f} tokens       {pct_saved:.0f}% fewer ops ({speedup:.1f}x)")

    # MaxSim compute
    doc_tokens = 128
    print(f"\n{'=' * 60}")
    print(f"Compute Impact (MaxSim ops per query-doc pair)")
    print(f"{'=' * 60}")
    print(f"\nFull ({mean_active:.0f} tokens): {mean_active:.0f} × {doc_tokens} = {mean_active * doc_tokens:.0f} ops")
    for k in [2, 4, 8]:
        ops = k * doc_tokens
        speedup = (mean_active * doc_tokens) / ops
        print(f"Top-{k}:            {k} × {doc_tokens} = {ops} ops  ({speedup:.1f}x speedup)")

    # Reranking latency estimate
    print(f"\n{'=' * 60}")
    print(f"Reranking Latency Impact")
    print(f"{'=' * 60}")
    for n_candidates in [20, 50, 100, 500]:
        full_ops = mean_active * doc_tokens * n_candidates
        top4_ops = 4 * doc_tokens * n_candidates
        print(f"  {n_candidates} candidates: {full_ops/1000:.0f}K ops (full) → {top4_ops/1000:.0f}K ops (top-4)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="esci", choices=["esci", "wands"])
    parser.add_argument("--data_dir", default="~/wands_repo/dataset")
    args = parser.parse_args()
    import os
    args.data_dir = os.path.expanduser(args.data_dir)
    analyze(args.dataset, args.data_dir)


if __name__ == "__main__":
    main()
