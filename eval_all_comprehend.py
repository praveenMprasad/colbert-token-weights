#!/usr/bin/env python3
"""Re-split eval_all results by Comprehend-based attribute detection.

Uses cached Comprehend syntax to classify queries as attribute vs generic.
Attribute query = has any ADJ token OR gender noun (men, women, boys, girls).
Does NOT re-run model inference — loads per-query scores from eval_all results
and the weight analysis cache, then re-computes attr/non-attr MRR splits.

Usage:
  python3 eval_all_comprehend.py --dataset esci
  python3 eval_all_comprehend.py --dataset wands
  python3 eval_all_comprehend.py --dataset both --data_dir ~/wands_repo/dataset
"""
import argparse
import json
import os
import numpy as np
import torch
from transformers import AutoTokenizer
from tqdm import tqdm

from esci.config import ESCIConfig
from esci.model import ColBERTESCI
from esci.data import ESCIRerankDataset
from esci.evaluate import mrr_at_k, ndcg_at_k, exact_substitute_separation, score_query_products
from colbert_weighted.scoring import maxsim, weighted_maxsim

GENDER_NOUNS = {"men", "mens", "men's", "women", "womens", "women's",
                "boy", "boys", "girl", "girls", "ladies", "male", "female"}


def load_comprehend_cache(dataset):
    path = f"results/comprehend_syntax_cache_{dataset}.json"
    if not os.path.exists(path):
        print(f"ERROR: No Comprehend cache at {path}. Run analyze_weights.py --dataset {dataset} first.")
        return None
    with open(path) as f:
        cache = json.load(f)
    return cache


def is_attribute_query_comprehend(query, pos_map):
    """Attribute query if it has any ADJ token or gender noun."""
    # Check for adjectives from Comprehend
    for word, pos in pos_map.items():
        if pos == "ADJ":
            return True
    # Check for gender nouns
    words = set(query.lower().split())
    if words & GENDER_NOUNS:
        return True
    return False


def load_model(config, model_path, weight_head_path, device):
    model = ColBERTESCI(config).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    if weight_head_path:
        model.weight_head.load_state_dict(
            torch.load(weight_head_path, map_location=device))
    model.eval()
    return model


def eval_esci_comprehend(gap_model, loo_model, config, device, num_eval, tokenizer):
    """Eval on ESCI with Comprehend-based attribute classification."""
    cache = load_comprehend_cache("esci")
    if not cache:
        return None

    rerank_data = list(ESCIRerankDataset(split="test", locale="us", max_queries=num_eval))
    query_to_pos = {q: pm for q, pm in zip(cache["queries"], cache["pos_maps"])}

    results = {"vanilla": [], "gap": [], "loo": []}
    attr_results = {"vanilla": [], "gap": [], "loo": []}
    non_attr_results = {"vanilla": [], "gap": [], "loo": []}
    ndcg = {"vanilla": [], "gap": [], "loo": []}
    sep = {"vanilla": [], "gap": [], "loo": []}

    n_attr = 0
    n_generic = 0

    for item in tqdm(rerank_data, desc="ESCI eval (Comprehend attr)"):
        query = item["query"]
        products = item["products"]
        if len(products) < 2:
            continue

        pos_map = query_to_pos.get(query, {})
        is_attr = is_attribute_query_comprehend(query, pos_map)
        if is_attr:
            n_attr += 1
        else:
            n_generic += 1

        v_scores, gw_scores = score_query_products(
            gap_model, tokenizer, query, products, config, device)
        _, lw_scores = score_query_products(
            loo_model, tokenizer, query, products, config, device)

        for name, scores in [("vanilla", v_scores), ("gap", gw_scores), ("loo", lw_scores)]:
            order = np.argsort(scores)[::-1]
            labels = [products[i]["label"] for i in order]
            esci_labels = [products[i]["esci_label"] for i in order]

            mrr = mrr_at_k(labels)
            results[name].append(mrr)
            ndcg[name].append(ndcg_at_k(labels))
            s = exact_substitute_separation(esci_labels)
            if s is not None:
                sep[name].append(s)
            if is_attr:
                attr_results[name].append(mrr)
            else:
                non_attr_results[name].append(mrr)

    out = {}
    for name in ["vanilla", "gap", "loo"]:
        out[name] = {
            "mrr@10": float(np.mean(results[name])),
            "ndcg@10": float(np.mean(ndcg[name])),
            "attr_mrr@10": float(np.mean(attr_results[name])) if attr_results[name] else None,
            "non_attr_mrr@10": float(np.mean(non_attr_results[name])) if non_attr_results[name] else None,
            "e_vs_s_separation": float(np.mean(sep[name])) if sep[name] else None,
        }
    out["num_queries"] = len(results["vanilla"])
    out["num_attr_queries"] = n_attr
    out["num_generic_queries"] = n_generic
    out["attr_method"] = "comprehend_adj + gender_nouns"
    return out


def eval_wands_comprehend(gap_model, loo_model, config, device, tokenizer, data_dir):
    """Eval on WANDS with Comprehend-based attribute classification."""
    cache = load_comprehend_cache("wands")
    if not cache:
        return None

    from wands.evaluate import load_wands, mrr_at_k as wands_mrr, ndcg_at_k as wands_ndcg

    data = load_wands(data_dir)
    query_to_pos = {q: pm for q, pm in zip(cache["queries"], cache["pos_maps"])}

    results = {"vanilla": [], "gap": [], "loo": []}
    attr_results = {"vanilla": [], "gap": [], "loo": []}
    non_attr_results = {"vanilla": [], "gap": [], "loo": []}
    ndcg_r = {"vanilla": [], "gap": [], "loo": []}
    sep = {"vanilla": [], "gap": [], "loo": []}

    n_attr = 0
    n_generic = 0

    for item in tqdm(data, desc="WANDS eval (Comprehend attr)"):
        query = item["query"]
        products = item["products"]
        if len(products) < 2:
            continue

        pos_map = query_to_pos.get(query, {})
        is_attr = is_attribute_query_comprehend(query, pos_map)
        if is_attr:
            n_attr += 1
        else:
            n_generic += 1

        q_enc = tokenizer(query, return_tensors="pt", padding="max_length",
                          truncation=True, max_length=config.query_maxlen).to(device)

        with torch.no_grad():
            Q_g, qm_g, qh_g = gap_model.encode(q_enc.input_ids, q_enc.attention_mask)
            w_gap = gap_model.weight_head(qh_g, qm_g)
            Q_l, qm_l, qh_l = loo_model.encode(q_enc.input_ids, q_enc.attention_mask)
            w_loo = loo_model.weight_head(qh_l, qm_l)

        titles = [p["title"] for p in products]
        labels = [p["label"] for p in products]
        label_strs = [p.get("label_str", "") for p in products]

        v_scores, gw_scores, lw_scores = [], [], []
        BATCH = 64

        for start in range(0, len(titles), BATCH):
            bt = titles[start:start + BATCH]
            d_enc = tokenizer(bt, return_tensors="pt", padding="max_length",
                              truncation=True, max_length=config.doc_maxlen).to(device)
            with torch.no_grad():
                D_g, dm_g, _ = gap_model.encode(d_enc.input_ids, d_enc.attention_mask)
                bsz = D_g.size(0)
                Q_ge = Q_g.expand(bsz, -1, -1)
                qm_ge = qm_g.expand(bsz, -1)
                sim_g = torch.bmm(Q_ge, D_g.transpose(1, 2))
                sim_g = sim_g.masked_fill(~dm_g.unsqueeze(1), float("-inf"))
                ms_g, _ = sim_g.max(dim=-1)
                v_batch = (ms_g * qm_ge.float()).sum(dim=-1).cpu().tolist()
                v_scores.extend(v_batch)
                gw_batch = (ms_g * w_gap.expand(bsz, -1)).sum(dim=-1).cpu().tolist()
                gw_scores.extend(gw_batch)

                D_l, dm_l, _ = loo_model.encode(d_enc.input_ids, d_enc.attention_mask)
                Q_le = Q_l.expand(bsz, -1, -1)
                sim_l = torch.bmm(Q_le, D_l.transpose(1, 2))
                sim_l = sim_l.masked_fill(~dm_l.unsqueeze(1), float("-inf"))
                ms_l, _ = sim_l.max(dim=-1)
                lw_batch = (ms_l * w_loo.expand(bsz, -1)).sum(dim=-1).cpu().tolist()
                lw_scores.extend(lw_batch)

        for name, scores in [("vanilla", v_scores), ("gap", gw_scores), ("loo", lw_scores)]:
            order = np.argsort(scores)[::-1]
            rl = [labels[i] for i in order]
            results[name].append(wands_mrr(rl))
            ndcg_r[name].append(wands_ndcg(rl))
            if is_attr:
                attr_results[name].append(wands_mrr(rl))
            else:
                non_attr_results[name].append(wands_mrr(rl))
            rls = [label_strs[i] for i in order]
            correct, total = 0, 0
            for i, li in enumerate(rls):
                for j, lj in enumerate(rls):
                    if li == "Exact" and lj == "Partial":
                        total += 1
                        if i < j:
                            correct += 1
            if total > 0:
                sep[name].append(correct / total)

    out = {}
    for name in ["vanilla", "gap", "loo"]:
        out[name] = {
            "mrr@10": float(np.mean(results[name])),
            "ndcg@10": float(np.mean(ndcg_r[name])),
            "attr_mrr@10": float(np.mean(attr_results[name])) if attr_results[name] else None,
            "non_attr_mrr@10": float(np.mean(non_attr_results[name])) if non_attr_results[name] else None,
            "exact_vs_partial_sep": float(np.mean(sep[name])) if sep[name] else None,
        }
    out["num_queries"] = len(results["vanilla"])
    out["num_attr_queries"] = n_attr
    out["num_generic_queries"] = n_generic
    out["attr_method"] = "comprehend_adj + gender_nouns"
    return out


def _print_results(name, r):
    print(f"\n--- {name} Results ({r['num_queries']} queries) ---")
    print(f"  Attribute queries: {r['num_attr_queries']}, Generic: {r['num_generic_queries']}")
    print(f"  Attr method: {r['attr_method']}")
    for model in ["vanilla", "gap", "loo"]:
        m = r[model]
        print(f"\n  {model.upper()}:")
        print(f"    MRR@10:          {m['mrr@10']:.4f}")
        print(f"    NDCG@10:         {m['ndcg@10']:.4f}")
        if m.get("attr_mrr@10") is not None:
            print(f"    Attr MRR@10:     {m['attr_mrr@10']:.4f}")
        if m.get("non_attr_mrr@10") is not None:
            print(f"    Non-attr MRR@10: {m['non_attr_mrr@10']:.4f}")
        sep_key = "e_vs_s_separation" if "e_vs_s_separation" in m else "exact_vs_partial_sep"
        if m.get(sep_key) is not None:
            print(f"    Separation:      {m[sep_key]:.4f}")

    # Delta table
    print(f"\n  Deltas vs Vanilla:")
    v = r["vanilla"]
    for model in ["gap", "loo"]:
        m = r[model]
        print(f"    {model.upper()}: MRR {m['mrr@10'] - v['mrr@10']:+.4f}", end="")
        if m.get("attr_mrr@10") and v.get("attr_mrr@10"):
            print(f"  Attr {m['attr_mrr@10'] - v['attr_mrr@10']:+.4f}", end="")
        if m.get("non_attr_mrr@10") and v.get("non_attr_mrr@10"):
            print(f"  Non-attr {m['non_attr_mrr@10'] - v['non_attr_mrr@10']:+.4f}", end="")
        print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["esci", "wands", "both"])
    parser.add_argument("--gap_model", default="outputs/esci/gap_multineg/model.pt")
    parser.add_argument("--gap_wh", default="outputs/esci/gap_multineg/weight_head_step2000.pt")
    parser.add_argument("--loo_model", default="outputs/esci/loo_weighted/model.pt")
    parser.add_argument("--loo_wh", default="outputs/esci/loo_weighted/weight_head_step2500.pt")
    parser.add_argument("--num_eval", type=int, default=1000)
    parser.add_argument("--data_dir", default="~/wands_repo/dataset")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--output_dir", default="results")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.data_dir = os.path.expanduser(args.data_dir)

    config = ESCIConfig(
        use_token_weights=True,
        weight_norm="softmax",
        softmax_temperature=args.temperature,
    )
    tokenizer = AutoTokenizer.from_pretrained(config.checkpoint)

    print("Loading Gap model...")
    gap_model = load_model(config, args.gap_model, args.gap_wh, device)
    print("Loading LOO model...")
    loo_model = load_model(config, args.loo_model, args.loo_wh, device)

    os.makedirs(args.output_dir, exist_ok=True)

    if args.dataset in ("esci", "both"):
        print(f"\n{'='*60}")
        print(f"ESCI — Comprehend Attribute Classification")
        print(f"{'='*60}")
        esci_results = eval_esci_comprehend(
            gap_model, loo_model, config, device, args.num_eval, tokenizer)
        if esci_results:
            out_path = os.path.join(args.output_dir, "esci_eval_comprehend.json")
            with open(out_path, "w") as f:
                json.dump(esci_results, f, indent=2)
            print(f"\nSaved to {out_path}")
            _print_results("ESCI (Comprehend)", esci_results)

    if args.dataset in ("wands", "both"):
        print(f"\n{'='*60}")
        print(f"WANDS — Comprehend Attribute Classification")
        print(f"{'='*60}")
        wands_results = eval_wands_comprehend(
            gap_model, loo_model, config, device, tokenizer, args.data_dir)
        if wands_results:
            out_path = os.path.join(args.output_dir, "wands_eval_comprehend.json")
            with open(out_path, "w") as f:
                json.dump(wands_results, f, indent=2)
            print(f"\nSaved to {out_path}")
            _print_results("WANDS (Comprehend)", wands_results)


if __name__ == "__main__":
    main()
