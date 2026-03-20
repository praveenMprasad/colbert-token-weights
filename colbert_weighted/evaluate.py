"""Evaluation: MRR@10, Recall@k, NDCG@10 using pytrec_eval.

Supports both:
  - HuggingFace qrels (from Tevatron/msmarco-passage dev split)
  - Local TREC-format qrels files
"""
import argparse
import json
import pytrec_eval
from collections import defaultdict


def load_qrels_from_hf(cache_dir=None):
    """Load dev qrels directly from HuggingFace."""
    from .data import load_dev_qrels
    return load_dev_qrels(cache_dir=cache_dir)


def load_qrels_from_file(path):
    """Load TREC-format qrels: qid 0 docid relevance."""
    qrels = defaultdict(dict)
    with open(path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 4:
                qid, _, docid, rel = parts
            else:
                parts = line.strip().split()
                qid, _, docid, rel = parts
            qrels[qid][docid] = int(rel)
    return dict(qrels)


def load_ranking(path):
    """Load ranking: qid docid rank score (TSV)."""
    run = defaultdict(dict)
    with open(path) as f:
        for line in f:
            parts = line.strip().split("\t")
            qid, docid = parts[0], parts[1]
            score = float(parts[3]) if len(parts) > 3 else float(parts[2])
            run[qid][docid] = score
    return dict(run)


def evaluate(qrels, ranking_path, metrics=None):
    """Compute IR metrics.

    Args:
        qrels: dict {qid: {docid: rel}} or str path to TREC qrels file
        ranking_path: path to ranking TSV
        metrics: set of pytrec_eval metric strings
    """
    if metrics is None:
        metrics = {"map", "ndcg_cut_10", "recip_rank", "recall_10", "recall_50",
                   "recall_100", "recall_200", "recall_1000"}

    if isinstance(qrels, str):
        qrels = load_qrels_from_file(qrels)

    run = load_ranking(ranking_path)

    evaluator = pytrec_eval.RelevanceEvaluator(qrels, metrics)
    results = evaluator.evaluate(run)

    # Aggregate
    agg = defaultdict(list)
    for qid, qmetrics in results.items():
        for m, v in qmetrics.items():
            agg[m].append(v)

    summary = {m: sum(vs) / len(vs) for m, vs in agg.items()}
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qrels", default=None, help="Path to TREC qrels file (omit to use HuggingFace)")
    parser.add_argument("--ranking", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.qrels:
        qrels = load_qrels_from_file(args.qrels)
    else:
        print("Loading qrels from HuggingFace (Tevatron/msmarco-passage dev)...")
        qrels = load_qrels_from_hf()

    summary = evaluate(qrels, args.ranking)
    for m, v in sorted(summary.items()):
        print(f"{m:25s} {v:.4f}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
