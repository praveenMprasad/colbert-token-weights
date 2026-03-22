"""Reranking evaluation: vanilla MaxSim vs weighted MaxSim.

Uses the last N queries from the train split (which has 1 positive +
30 negatives per query). The validation split has no passages.
Reports MRR@10 and Acc@1 for both scoring methods.
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


def evaluate_reranking(model, config, device, num_eval=1000):
    """Rerank last `num_eval` train queries with vanilla and weighted MaxSim."""
    tokenizer = AutoTokenizer.from_pretrained(config.checkpoint)

    ds = load_dataset("Tevatron/msmarco-passage", split="train")
    eval_ds = ds.select(range(len(ds) - num_eval, len(ds)))

    vanilla_mrrs, weighted_mrrs = [], []
    vanilla_acc1, weighted_acc1 = [], []

    for row in tqdm(eval_ds, desc="Evaluating"):
        query = row["query"]
        pos_docs = [p["text"] for p in (row["positive_passages"] or [])]
        neg_docs = [p["text"] for p in (row["negative_passages"] or [])]
        if not pos_docs or not neg_docs:
            continue

        all_docs = pos_docs + neg_docs
        labels = [1] * len(pos_docs) + [0] * len(neg_docs)

        q_enc = tokenizer(
            query, padding="max_length", truncation=True,
            max_length=config.query_maxlen, return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            Q, q_mask, q_hidden = model.encode(q_enc.input_ids, q_enc.attention_mask)
            weights = model.weight_head(q_hidden, q_mask) if model.weight_head else None

        v_scores, w_scores = [], []
        for doc_text in all_docs:
            d_enc = tokenizer(
                doc_text, padding="max_length", truncation=True,
                max_length=config.doc_maxlen, return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                D, d_mask, _ = model.encode(d_enc.input_ids, d_enc.attention_mask)
                v_scores.append(maxsim(Q, D, q_mask, d_mask).item())
                if weights is not None:
                    w_scores.append(weighted_maxsim(Q, D, q_mask, d_mask, weights).item())

        # Vanilla MRR@10
        v_order = np.argsort(v_scores)[::-1]
        v_ranked = [labels[i] for i in v_order]
        vanilla_mrrs.append(_mrr(v_ranked, 10))
        vanilla_acc1.append(1.0 if v_ranked[0] == 1 else 0.0)

        # Weighted MRR@10
        if w_scores:
            w_order = np.argsort(w_scores)[::-1]
            w_ranked = [labels[i] for i in w_order]
            weighted_mrrs.append(_mrr(w_ranked, 10))
            weighted_acc1.append(1.0 if w_ranked[0] == 1 else 0.0)

    return {
        "vanilla_mrr@10": float(np.mean(vanilla_mrrs)),
        "weighted_mrr@10": float(np.mean(weighted_mrrs)) if weighted_mrrs else None,
        "vanilla_acc@1": float(np.mean(vanilla_acc1)),
        "weighted_acc@1": float(np.mean(weighted_acc1)) if weighted_acc1 else None,
        "num_queries": len(vanilla_mrrs),
    }


def _mrr(ranked_labels, k):
    for i, label in enumerate(ranked_labels[:k]):
        if label == 1:
            return 1.0 / (i + 1)
    return 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight_head_path", default=None)
    parser.add_argument("--norm", default="softmax", choices=["softmax", "sigmoid"])
    parser.add_argument("--num_eval", type=int, default=1000)
    parser.add_argument("--output", default=None)
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
    results = evaluate_reranking(model, config, device, num_eval=args.num_eval)

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
