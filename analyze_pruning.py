#!/usr/bin/env python3
"""Analyze pruning efficiency on ESCI queries.

Shows: actual token counts, storage savings at each top-k, MRR tradeoffs.
"""
import numpy as np
from transformers import AutoTokenizer
from collections import Counter

from esci.data import ESCIRerankDataset


def main():
    tokenizer = AutoTokenizer.from_pretrained("colbert-ir/colbertv2.0")
    rerank_data = list(ESCIRerankDataset(split="test", locale="us", max_queries=1000))

    # Count real tokens per query (excluding [CLS], [SEP], [PAD])
    token_counts = []
    special = {"[CLS]", "[SEP]", "[PAD]", "[Q]", "[MASK]"}

    for item in rerank_data:
        enc = tokenizer(item["query"], padding="max_length", truncation=True,
                        max_length=32, return_tensors="pt")
        tokens = tokenizer.convert_ids_to_tokens(enc.input_ids[0])
        mask = enc.attention_mask[0]
        # Count real content tokens (non-special, non-pad)
        real = sum(1 for t, m in zip(tokens, mask) if m == 1 and t not in special)
        total_active = int(mask.sum())  # includes [CLS], [SEP] etc
        token_counts.append({"real": real, "active": total_active})

    real_counts = [t["real"] for t in token_counts]
    active_counts = [t["active"] for t in token_counts]

    print("=" * 60)
    print("ESCI Query Token Analysis (1000 test queries)")
    print("=" * 60)
    print(f"\nQuery maxlen (padded): 32")
    print(f"\nReal content tokens (excluding [CLS]/[SEP]/[PAD]/[Q]/[MASK]):")
    print(f"  Mean:   {np.mean(real_counts):.1f}")
    print(f"  Median: {np.median(real_counts):.1f}")
    print(f"  Min:    {np.min(real_counts)}")
    print(f"  Max:    {np.max(real_counts)}")
    print(f"  Std:    {np.std(real_counts):.1f}")

    print(f"\nActive tokens (attention_mask=1, includes special tokens):")
    print(f"  Mean:   {np.mean(active_counts):.1f}")
    print(f"  Median: {np.median(active_counts):.1f}")

    # Distribution
    print(f"\nReal token count distribution:")
    dist = Counter(real_counts)
    for k in sorted(dist.keys()):
        bar = "#" * (dist[k] // 5)
        print(f"  {k:3d} tokens: {dist[k]:4d} queries  {bar}")

    # Pruning efficiency table
    print(f"\n{'=' * 60}")
    print("Pruning Efficiency Analysis (Gap method)")
    print(f"{'=' * 60}")

    # From esci_eval_all.json
    vanilla_mrr = 0.8305
    gap_results = {
        "all": 0.8326,
        2: 0.8072,
        4: 0.8341,
        8: 0.8352,
    }

    mean_active = np.mean(active_counts)
    mean_real = np.mean(real_counts)

    print(f"\n{'Tokens kept':<15} {'MRR@10':<10} {'vs Vanilla':<12} {'Vectors kept':<15} {'Vectors saved':<15} {'Storage saving'}")
    print("-" * 85)

    for k in [2, 4, 8, "all"]:
        mrr = gap_results[k]
        delta = mrr - vanilla_mrr

        if k == "all":
            kept = mean_active
            label = f"All ({mean_active:.0f} avg)"
        else:
            kept = k
            label = f"Top {k}"

        saved = mean_active - kept
        pct_saved = (saved / mean_active) * 100
        pct_kept = (kept / mean_active) * 100

        print(f"{label:<15} {mrr:.4f}     {delta:+.4f}       {kept:.0f}/{mean_active:.0f} ({pct_kept:.0f}%)      {saved:.0f} ({pct_saved:.0f}%)         {pct_saved:.0f}%")

    # ColBERT storage math
    print(f"\n{'=' * 60}")
    print("Storage Impact (ColBERT index)")
    print(f"{'=' * 60}")
    dim = 128
    bytes_per_vec = dim * 2  # fp16
    print(f"\nPer query vector: {dim}d × 2 bytes = {bytes_per_vec} bytes")
    print(f"Per query (all tokens): {mean_active:.0f} vectors × {bytes_per_vec} bytes = {mean_active * bytes_per_vec:.0f} bytes")
    print(f"Per query (top-4):      4 vectors × {bytes_per_vec} bytes = {4 * bytes_per_vec} bytes")
    print(f"Saving per query: {(mean_active - 4) * bytes_per_vec:.0f} bytes ({(1 - 4/mean_active)*100:.0f}% reduction)")

    # At scale
    for n_queries in [1_000_000, 10_000_000, 100_000_000]:
        full_gb = (n_queries * mean_active * bytes_per_vec) / (1024**3)
        top4_gb = (n_queries * 4 * bytes_per_vec) / (1024**3)
        print(f"\n  {n_queries/1e6:.0f}M queries:")
        print(f"    Full:  {full_gb:.1f} GB")
        print(f"    Top-4: {top4_gb:.1f} GB  (saves {full_gb - top4_gb:.1f} GB)")

    # Compute savings (MaxSim operations)
    print(f"\n{'=' * 60}")
    print("Compute Impact (MaxSim operations per query-doc pair)")
    print(f"{'=' * 60}")
    doc_tokens = 128  # max
    print(f"\nMaxSim = for each query token, find max similarity across all doc tokens")
    print(f"Full:  {mean_active:.0f} query tokens × {doc_tokens} doc tokens = {mean_active * doc_tokens:.0f} similarity ops")
    print(f"Top-4: 4 query tokens × {doc_tokens} doc tokens = {4 * doc_tokens} similarity ops")
    print(f"Speedup: {mean_active / 4:.1f}x fewer operations")


if __name__ == "__main__":
    main()
