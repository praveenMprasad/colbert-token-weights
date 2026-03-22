"""Reranking evaluation on MS MARCO dev set.

Compares vanilla MaxSim (baseline) vs weighted MaxSim on the same
frozen ColBERTv2 representations. Reports MRR@10 and accuracy@1.

Uses the Tevatron/msmarco-passage dev split which has positive and
negative passages per query — no corpus indexing needed.
"""
import argparse
import json
import os
import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer

from .config import ExpConfig
from .model import ColBERTWeighted
from .scoring import maxsim, weighted_maxsim


def load_dev_data(max_queries=None, cache_dir=None):
    """Load dev set with positive + negative passages."""
    ds = load_dataset("Tevatron/msmarco-passage", split="validation", cache_dir=cache_dir)
    if max_queries:
        ds = ds.select(range(min(max_queries, len(ds))))
    return ds


def score_query_docs(model, tokenizer, query, docs, device, config):
    """Score a query against multiple docs. Returns (vanilla_scores, weighted_scores)."""
    q_enc = tokenizer(
        query, padding="max_length", truncation=True,
        max_length=config.query_maxlen, return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        Q, q_mask, q_hidden = model.encode(q_enc.input_ids, q_enc.attention_mask)

        # Get weights from trained head
        weights = None
        if model.weight_head is not None:
            weights = model.weight_head(q_hidden, q_mask)

    vanilla_scores = []
    weighted_scores = []

    # Score each doc
    for doc_text in docs:
        d_enc = tokenizer(
            doc_text, padding="max_length", truncation=True,
            max_length=config.doc_maxlen, return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            D, d_mask, _ = model.encode(d_enc.input_ids, d_enc.attention_mask)

            v_score = maxsim(Q, D, q_mask, d_mask)
            vanilla_scores.append(v_score.item())

            if weights is not None:
                w_score = weighted_maxsim(Q, D, q_mask, d_mask, weights)
                weighted_scores.append(w_score.item())

    return vanilla_scores, weighted_scores


def mrr_at_k(ranked_labels, k=10):
    """Compute MRR@k given a list of labels sorted by score (desc)."""
    for i, label in enumerate(ranked_labels[:k]):
        if label == 1:
            return 1.0 / (i + 1)
    return 0.0


def evaluate_reranking(model, config, device, max_queries=None):
    """Run reranking eval on dev set. Returns baseline and weighted MRR@10."""
    tokenizer = AutoTokenizer.from_pretrained(config.checkpoint)
    dev_data = load_dev_data(max_queries=max_queries)

    vanilla_mrrs = []
    weighted_mrrs = []
    vanilla_acc1 = []
    weighted_acc1 = []

    for row in tqdm(dev_data, desc="Evaluating dev set"):
        query = row["query"]
        pos_docs = [p["text"] for p in (row["positive_passages"] or [])]
        neg_docs = [p["text"] for p in (row["negative_passages"] or [])]

        if not pos_docs or not neg_docs:
            continue

        all_docs = pos_docs + neg_docs
        labels = [1] * len(pos_docs) + [0] * len(neg_docs)

        v_scores, w_scores = score_query_docs(
            model, tokenizer, query, all_docs, device, config,
        )

        # Vanilla ranking
        v_order = np.argsort(v_scores)[::-1]
        v_ranked_labels = [labels[i] for i in v_order]
        vanilla_mrrs.append(mrr_at_k(v_ranked_labels, k=10))
        vanilla_acc1.append(1.0 if v_ranked_labels[0] == 1 else 0.0)

        # Weighted ranking
        if w_scores:
            w_order = np.argsort(w_scores)[::-1]
            w_ranked_labels = [labels[i] for i in w_order]
            weighted_mrrs.append(mrr_at_k(w_ranked_labels, k=10))
            weighted_acc1.append(1.0 if w_ranked_labels[0] == 1 else 0.0)

    results = {
        "vanilla_mrr@10": float(np.mean(vanilla_mrrs)),
        "vanilla_acc@1": float(np.mean(vanilla_acc1)),
        "weighted_mrr@10": float(np.mean(weighted_mrrs)) if weighted_mrrs else None,
        "weighted_acc@1": float(np.mean(weighted_acc1)) if weighted_acc1 else None,
        "num_queries": len(vanilla_mrrs),
    }
    return results


def main():
    parser = argparse.ArgumentParser(description="Reranking eval: vanilla vs weighted MaxSim")
    parser.add_argument("--weight_head_path", default=None,
                        help="Path to trained weight_head.pt (omit for untrained baseline)")
    parser.add_argument("--norm", default="softmax", choices=["softmax", "sigmoid"])
    parser.add_argument("--max_queries", type=int, default=None,
                        help="Limit dev queries for quick test")
    parser.add_argument("--output", default=None, help="Save results JSON")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = ExpConfig(use_token_weights=True, weight_norm=args.norm)
    model = ColBERTWeighted(config).to(device)

    if args.weight_head_path:
        model.weight_head.load_state_dict(
            torch.load(args.weight_head_path, map_location=device)
        )
        print(f"Loaded weight head from {args.weight_head_path}")

    model.eval()
    results = evaluate_reranking(model, config, device, max_queries=args.max_queries)

    print(f"\n{'='*50}")
    print(f"Results ({results['num_queries']} queries)")
    print(f"{'='*50}")
    print(f"  Vanilla  MRR@10: {results['vanilla_mrr@10']:.4f}")
    print(f"  Weighted MRR@10: {results['weighted_mrr@10']:.4f}")
    print(f"  Vanilla  Acc@1:  {results['vanilla_acc@1']:.4f}")
    print(f"  Weighted Acc@1:  {results['weighted_acc@1']:.4f}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
