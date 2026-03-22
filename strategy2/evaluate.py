"""Strategy 2 evaluation: reranking on held-out train queries.

Compares three scoring methods:
  1. Vanilla MaxSim (baseline ColBERTv2, no training)
  2. Weighted MaxSim with Strategy 2 model (encoder + weights co-trained)
  3. Vanilla MaxSim with Strategy 2 encoder (weights disabled)
     — isolates encoder improvement from weight contribution
"""
import argparse
import json
import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer

from .config import S2Config
from .model import ColBERTWeightedS2
from colbert_weighted.model import ColBERTWeighted
from colbert_weighted.config import ExpConfig
from colbert_weighted.scoring import maxsim, weighted_maxsim


def _mrr(ranked_labels, k=10):
    for i, label in enumerate(ranked_labels[:k]):
        if label == 1:
            return 1.0 / (i + 1)
    return 0.0


def evaluate(s2_model, baseline_model, config, device, num_eval=1000):
    """Compare baseline vs S2 weighted vs S2 vanilla on held-out queries."""
    tokenizer = AutoTokenizer.from_pretrained(config.checkpoint)

    ds = load_dataset("Tevatron/msmarco-passage", split="train")
    eval_ds = ds.select(range(len(ds) - num_eval, len(ds)))

    results = {"baseline": [], "s2_weighted": [], "s2_vanilla": []}

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
            # Baseline: frozen ColBERTv2
            bQ, bq_mask, _ = baseline_model.encode(q_enc.input_ids, q_enc.attention_mask)
            # S2: trained query encoder
            sQ, sq_mask, sq_hidden = s2_model.encode_query(q_enc.input_ids, q_enc.attention_mask)
            s_weights = s2_model.weight_head(sq_hidden, sq_mask)

        baseline_scores, s2w_scores, s2v_scores = [], [], []

        for doc_text in all_docs:
            d_enc = tokenizer(
                doc_text, padding="max_length", truncation=True,
                max_length=config.doc_maxlen, return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                # Use baseline encoder for doc (both models share same frozen doc encoder)
                bD, bd_mask, _ = baseline_model.encode(d_enc.input_ids, d_enc.attention_mask)
                sD, sd_mask = s2_model.encode_doc(d_enc.input_ids, d_enc.attention_mask)

                baseline_scores.append(maxsim(bQ, bD, bq_mask, bd_mask).item())
                s2w_scores.append(weighted_maxsim(sQ, sD, sq_mask, sd_mask, s_weights).item())
                s2v_scores.append(maxsim(sQ, sD, sq_mask, sd_mask).item())

        for scores, key in [(baseline_scores, "baseline"),
                            (s2w_scores, "s2_weighted"),
                            (s2v_scores, "s2_vanilla")]:
            order = np.argsort(scores)[::-1]
            ranked = [labels[i] for i in order]
            results[key].append(_mrr(ranked, 10))

    summary = {k: float(np.mean(v)) for k, v in results.items()}
    summary["num_queries"] = len(results["baseline"])
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="Path to S2 model.pt")
    parser.add_argument("--norm", default="softmax", choices=["softmax", "sigmoid"])
    parser.add_argument("--num_eval", type=int, default=1000)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load S2 model
    s2_config = S2Config(weight_norm=args.norm)
    s2_model = ColBERTWeightedS2(s2_config).to(device)
    s2_model.load_state_dict(torch.load(args.model_path, map_location=device))
    s2_model.eval()

    # Load baseline (frozen ColBERTv2, no weight head needed)
    baseline_config = ExpConfig(use_token_weights=False)
    baseline_model = ColBERTWeighted(baseline_config).to(device)
    baseline_model.eval()

    results = evaluate(s2_model, baseline_model, s2_config, device, num_eval=args.num_eval)

    print(f"\n{'='*50}")
    print(f"Results ({results['num_queries']} queries)")
    print(f"{'='*50}")
    print(f"  Baseline (vanilla ColBERTv2)  MRR@10: {results['baseline']:.4f}")
    print(f"  S2 weighted MaxSim           MRR@10: {results['s2_weighted']:.4f}")
    print(f"  S2 vanilla MaxSim (no wts)   MRR@10: {results['s2_vanilla']:.4f}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
