#!/usr/bin/env python3
"""Run both Gap and LOO models on ESCI and WANDS, save all results.

Evaluates vanilla ColBERTv2, Gap-weighted, and LOO-weighted on the same
queries so baselines are identical. Includes pruning.

Usage:
  python3 eval_all.py --dataset esci --num_eval 1000
  python3 eval_all.py --dataset wands --data_dir ~/wands_repo/dataset
  python3 eval_all.py --dataset both --num_eval 1000 --data_dir ~/wands_repo/dataset
"""
import argparse
import json
import os
import torch
import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm

from esci.config import ESCIConfig
from esci.model import ColBERTESCI
from esci.evaluate import (evaluate, pruning_eval, ESCIRerankDataset,
                           score_query_products, mrr_at_k, ndcg_at_k,
                           exact_substitute_separation, has_attribute_terms)
from colbert_weighted.scoring import maxsim, weighted_maxsim


def load_model(config, model_path, weight_head_path, device):
    model = ColBERTESCI(config).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    if weight_head_path:
        model.weight_head.load_state_dict(
            torch.load(weight_head_path, map_location=device))
    model.eval()
    return model


def eval_esci(gap_model, loo_model, config, device, num_eval, tokenizer):
    """Eval both models on the same ESCI queries."""
    rerank_data = list(ESCIRerankDataset(split="test", locale="us", max_queries=num_eval))

    results = {"vanilla": [], "gap": [], "loo": []}
    attr_results = {"vanilla": [], "gap": [], "loo": []}
    non_attr_results = {"vanilla": [], "gap": [], "loo": []}
    ndcg = {"vanilla": [], "gap": [], "loo": []}
    sep = {"vanilla": [], "gap": [], "loo": []}

    for item in tqdm(rerank_data, desc="ESCI eval"):
        query = item["query"]
        products = item["products"]
        if len(products) < 2:
            continue

        # Score with gap model (vanilla + weighted)
        v_scores, gw_scores = score_query_products(
            gap_model, tokenizer, query, products, config, device)

        # Score with loo model (weighted only, vanilla same encoder)
        _, lw_scores = score_query_products(
            loo_model, tokenizer, query, products, config, device)

        is_attr = has_attribute_terms(query)[0]

        for name, scores in [("vanilla", v_scores), ("gap", gw_scores), ("loo", lw_scores)]:
            order = np.argsort(scores)[::-1]
            labels = [products[i]["label"] for i in order]
            esci_labels = [products[i]["esci_label"] for i in order]

            results[name].append(mrr_at_k(labels))
            ndcg[name].append(ndcg_at_k(labels))
            s = exact_substitute_separation(esci_labels)
            if s is not None:
                sep[name].append(s)
            if is_attr:
                attr_results[name].append(mrr_at_k(labels))
            else:
                non_attr_results[name].append(mrr_at_k(labels))

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

    # Pruning for gap and loo
    for name, model in [("gap", gap_model), ("loo", loo_model)]:
        prune = pruning_eval(model, config, device, max_queries=num_eval)
        if prune:
            out[name]["pruning"] = prune

    return out


def eval_wands(gap_model, loo_model, config, device, tokenizer, data_dir):
    """Eval both models on the same WANDS queries."""
    from wands.evaluate import load_wands, mrr_at_k, ndcg_at_k, has_attribute_terms

    data = load_wands(data_dir)

    results = {"vanilla": [], "gap": [], "loo": []}
    attr_results = {"vanilla": [], "gap": [], "loo": []}
    non_attr_results = {"vanilla": [], "gap": [], "loo": []}
    ndcg_r = {"vanilla": [], "gap": [], "loo": []}
    sep = {"vanilla": [], "gap": [], "loo": []}

    # Pruning accumulators
    prune_ks = (2, 4, 8)
    prune_gap = {k: [] for k in prune_ks}
    prune_gap["all"] = []
    prune_loo = {k: [] for k in prune_ks}
    prune_loo["all"] = []

    for item in tqdm(data, desc="WANDS eval"):
        query = item["query"]
        products = item["products"]
        if len(products) < 2:
            continue

        # Encode query with gap model
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
        is_attr = has_attribute_terms(query)

        v_scores, gw_scores, lw_scores = [], [], []
        all_ms_gap, all_ms_loo = [], []
        BATCH = 64

        for start in range(0, len(titles), BATCH):
            bt = titles[start:start + BATCH]
            d_enc = tokenizer(bt, return_tensors="pt", padding="max_length",
                              truncation=True, max_length=config.doc_maxlen).to(device)
            with torch.no_grad():
                # Gap model encode
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
                for i in range(bsz):
                    all_ms_gap.append(ms_g[i].cpu())

                # LOO model encode
                D_l, dm_l, _ = loo_model.encode(d_enc.input_ids, d_enc.attention_mask)
                Q_le = Q_l.expand(bsz, -1, -1)
                sim_l = torch.bmm(Q_le, D_l.transpose(1, 2))
                sim_l = sim_l.masked_fill(~dm_l.unsqueeze(1), float("-inf"))
                ms_l, _ = sim_l.max(dim=-1)
                lw_batch = (ms_l * w_loo.expand(bsz, -1)).sum(dim=-1).cpu().tolist()
                lw_scores.extend(lw_batch)
                for i in range(bsz):
                    all_ms_loo.append(ms_l[i].cpu())

        for name, scores in [("vanilla", v_scores), ("gap", gw_scores), ("loo", lw_scores)]:
            order = np.argsort(scores)[::-1]
            rl = [labels[i] for i in order]
            results[name].append(mrr_at_k(rl))
            ndcg_r[name].append(ndcg_at_k(rl))
            if is_attr:
                attr_results[name].append(mrr_at_k(rl))
            else:
                non_attr_results[name].append(mrr_at_k(rl))
            # Separation
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

        # Pruning
        for w, all_ms, prune_acc in [(w_gap, all_ms_gap, prune_gap),
                                      (w_loo, all_ms_loo, prune_loo)]:
            for k in list(prune_ks) + [None]:
                if k is not None:
                    qm = qm_g if w is w_gap else qm_l
                    _, topk_idx = w[0].topk(min(k, int(qm.sum())))
                    pmask = torch.zeros(w.size(1), device="cpu")
                    pmask[topk_idx.cpu()] = 1
                else:
                    pmask = torch.ones(w.size(1), device="cpu")
                wp = w[0].cpu() * pmask
                denom = wp.sum().clamp(min=1e-9)
                wp = wp / denom
                scores_p = [(ms_i * wp).sum().item() for ms_i in all_ms]
                order = np.argsort(scores_p)[::-1]
                rl = [labels[i] for i in order]
                key = k if k is not None else "all"
                prune_acc[key].append(mrr_at_k(rl))

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

    for name, prune_acc in [("gap", prune_gap), ("loo", prune_loo)]:
        out[name]["pruning"] = {}
        for k, vals in prune_acc.items():
            out[name]["pruning"][f"top{k}"] = float(np.mean(vals)) if vals else None

    return out


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
        print(f"ESCI Evaluation ({args.num_eval} queries)")
        print(f"{'='*60}")
        esci_results = eval_esci(gap_model, loo_model, config, device, args.num_eval, tokenizer)
        out_path = os.path.join(args.output_dir, "esci_eval_all.json")
        with open(out_path, "w") as f:
            json.dump(esci_results, f, indent=2)
        print(f"\nSaved to {out_path}")
        _print_results("ESCI", esci_results)

    if args.dataset in ("wands", "both"):
        print(f"\n{'='*60}")
        print(f"WANDS Evaluation (zero-shot transfer)")
        print(f"{'='*60}")
        wands_results = eval_wands(gap_model, loo_model, config, device, tokenizer, args.data_dir)
        out_path = os.path.join(args.output_dir, "wands_eval_all.json")
        with open(out_path, "w") as f:
            json.dump(wands_results, f, indent=2)
        print(f"\nSaved to {out_path}")
        _print_results("WANDS", wands_results)


def _print_results(name, r):
    print(f"\n--- {name} Results ({r['num_queries']} queries) ---")
    for model in ["vanilla", "gap", "loo"]:
        m = r[model]
        print(f"\n  {model.upper()}:")
        print(f"    MRR@10: {m['mrr@10']:.4f}")
        print(f"    NDCG@10: {m['ndcg@10']:.4f}")
        if m.get("attr_mrr@10"):
            print(f"    Attr MRR@10: {m['attr_mrr@10']:.4f}")
        if m.get("non_attr_mrr@10"):
            print(f"    Non-attr MRR@10: {m['non_attr_mrr@10']:.4f}")
        sep_key = "e_vs_s_separation" if "e_vs_s_separation" in m else "exact_vs_partial_sep"
        if m.get(sep_key):
            print(f"    Separation: {m[sep_key]:.4f}")
        if m.get("pruning"):
            print(f"    Pruning: {m['pruning']}")


if __name__ == "__main__":
    main()
