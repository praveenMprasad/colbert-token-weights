#!/usr/bin/env python3
"""Statistical significance tests for weighted vs vanilla ColBERT.

Paired bootstrap + paired t-test on per-query MRR@10.
Tests: gap vs vanilla, loo vs vanilla, gap_top4 vs vanilla.

Usage:
  python3 eval_significance.py --dataset esci
  python3 eval_significance.py --dataset wands --data_dir ~/wands_repo/dataset
  python3 eval_significance.py --dataset both --data_dir ~/wands_repo/dataset
"""
import argparse
import json
import os
import math
import torch
import numpy as np
from scipy import stats
from transformers import AutoTokenizer
from tqdm import tqdm

from esci.config import ESCIConfig
from esci.model import ColBERTESCI
from esci.data import ESCIRerankDataset
from esci.evaluate import mrr_at_k, ndcg_at_k, exact_substitute_separation, score_query_products
from colbert_weighted.scoring import maxsim, weighted_maxsim


def load_model(config, model_path, weight_head_path, device):
    model = ColBERTESCI(config).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    if weight_head_path:
        model.weight_head.load_state_dict(
            torch.load(weight_head_path, map_location=device))
    model.eval()
    return model


def bootstrap_test(a, b, n_bootstrap=10000, seed=42):
    """Paired bootstrap significance test.

    Tests H0: mean(a) == mean(b).
    Returns p-value (two-sided).
    """
    rng = np.random.RandomState(seed)
    a = np.array(a)
    b = np.array(b)
    observed_diff = np.mean(a) - np.mean(b)
    diffs = a - b
    count = 0
    for _ in range(n_bootstrap):
        sample = rng.choice(diffs, size=len(diffs), replace=True)
        if abs(np.mean(sample)) >= abs(observed_diff):
            count += 1
    # Actually test if 0 is plausible:
    # Center diffs at 0, resample, count how often we see >= observed
    centered = diffs - np.mean(diffs)
    count = 0
    for _ in range(n_bootstrap):
        sample = rng.choice(centered, size=len(centered), replace=True)
        if abs(np.mean(sample)) >= abs(observed_diff):
            count += 1
    return count / n_bootstrap


def bootstrap_ci(a, b, n_bootstrap=10000, alpha=0.05, seed=42):
    """Bootstrap confidence interval for mean(a) - mean(b)."""
    rng = np.random.RandomState(seed)
    a = np.array(a)
    b = np.array(b)
    diffs = a - b
    boot_means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(diffs, size=len(diffs), replace=True)
        boot_means.append(np.mean(sample))
    boot_means = np.sort(boot_means)
    lo = np.percentile(boot_means, 100 * alpha / 2)
    hi = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return lo, hi


def eval_esci_perquery(gap_model, loo_model, config, device, num_eval, tokenizer):
    """Get per-query MRR for all models on ESCI."""
    rerank_data = list(ESCIRerankDataset(split="test", locale="us", max_queries=num_eval))

    vanilla_mrrs, gap_mrrs, loo_mrrs = [], [], []
    gap_top4_mrrs = []

    for item in tqdm(rerank_data, desc="ESCI per-query"):
        query = item["query"]
        products = item["products"]
        if len(products) < 2:
            continue

        v_scores, gw_scores = score_query_products(
            gap_model, tokenizer, query, products, config, device)
        _, lw_scores = score_query_products(
            loo_model, tokenizer, query, products, config, device)

        # Vanilla MRR
        v_order = np.argsort(v_scores)[::-1]
        v_labels = [products[i]["label"] for i in v_order]
        vanilla_mrrs.append(mrr_at_k(v_labels))

        # Gap weighted MRR
        gw_order = np.argsort(gw_scores)[::-1]
        gw_labels = [products[i]["label"] for i in gw_order]
        gap_mrrs.append(mrr_at_k(gw_labels))

        # LOO weighted MRR
        lw_order = np.argsort(lw_scores)[::-1]
        lw_labels = [products[i]["label"] for i in lw_order]
        loo_mrrs.append(mrr_at_k(lw_labels))

        # Gap top-4 pruning MRR
        q_enc = tokenizer(query, return_tensors="pt", padding="max_length",
                          truncation=True, max_length=config.query_maxlen).to(device)
        with torch.no_grad():
            Q, q_mask, q_hidden = gap_model.encode(q_enc.input_ids, q_enc.attention_mask)
            weights = gap_model.weight_head(q_hidden, q_mask)
            _, topk_idx = weights[0].topk(min(4, int(q_mask.sum())))
            pruned_mask = torch.zeros_like(q_mask)
            pruned_mask[0, topk_idx] = 1
            w_pruned = weights * pruned_mask.float()
            w_pruned = w_pruned / w_pruned.sum(dim=-1, keepdim=True).clamp(min=1e-9)

        top4_scores = []
        for prod in products:
            d_enc = tokenizer(prod["title"], return_tensors="pt", padding="max_length",
                              truncation=True, max_length=config.doc_maxlen).to(device)
            with torch.no_grad():
                D, d_mask, _ = gap_model.encode(d_enc.input_ids, d_enc.attention_mask)
                s = weighted_maxsim(Q, D, pruned_mask.bool(), d_mask, w_pruned)
                top4_scores.append(s.item())

        t4_order = np.argsort(top4_scores)[::-1]
        t4_labels = [products[i]["label"] for i in t4_order]
        gap_top4_mrrs.append(mrr_at_k(t4_labels))

    return vanilla_mrrs, gap_mrrs, loo_mrrs, gap_top4_mrrs


def eval_wands_perquery(gap_model, loo_model, config, device, tokenizer, data_dir):
    """Get per-query MRR for all models on WANDS."""
    from wands.evaluate import load_wands, mrr_at_k as wands_mrr

    data = load_wands(data_dir)
    vanilla_mrrs, gap_mrrs, loo_mrrs = [], [], []

    for item in tqdm(data, desc="WANDS per-query"):
        query = item["query"]
        products = item["products"]
        if len(products) < 2:
            continue

        q_enc = tokenizer(query, return_tensors="pt", padding="max_length",
                          truncation=True, max_length=config.query_maxlen).to(device)

        with torch.no_grad():
            Q_g, qm_g, qh_g = gap_model.encode(q_enc.input_ids, q_enc.attention_mask)
            w_gap = gap_model.weight_head(qh_g, qm_g)
            Q_l, qm_l, qh_l = loo_model.encode(q_enc.input_ids, q_enc.attention_mask)
            w_loo = loo_model.weight_head(qh_l, qm_l)

        titles = [p["title"] for p in products]
        labels = [p["label"] for p in products]
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
                v_scores.extend((ms_g * qm_ge.float()).sum(dim=-1).cpu().tolist())
                gw_scores.extend((ms_g * w_gap.expand(bsz, -1)).sum(dim=-1).cpu().tolist())

                D_l, dm_l, _ = loo_model.encode(d_enc.input_ids, d_enc.attention_mask)
                Q_le = Q_l.expand(bsz, -1, -1)
                sim_l = torch.bmm(Q_le, D_l.transpose(1, 2))
                sim_l = sim_l.masked_fill(~dm_l.unsqueeze(1), float("-inf"))
                ms_l, _ = sim_l.max(dim=-1)
                lw_scores.extend((ms_l * w_loo.expand(bsz, -1)).sum(dim=-1).cpu().tolist())

        for name, scores, acc in [("v", v_scores, vanilla_mrrs),
                                    ("g", gw_scores, gap_mrrs),
                                    ("l", lw_scores, loo_mrrs)]:
            order = np.argsort(scores)[::-1]
            rl = [labels[i] for i in order]
            acc.append(wands_mrr(rl))

    return vanilla_mrrs, gap_mrrs, loo_mrrs


def run_tests(name, vanilla, gap, loo, gap_top4=None):
    """Run all significance tests and print results."""
    print(f"\n{'='*70}")
    print(f"Statistical Significance — {name}")
    print(f"{'='*70}")

    comparisons = [
        ("Gap weighted vs Vanilla", gap, vanilla),
        ("LOO weighted vs Vanilla", loo, vanilla),
    ]
    if gap_top4 is not None:
        comparisons.append(("Gap top-4 pruned vs Vanilla", gap_top4, vanilla))

    for label, a, b in comparisons:
        a = np.array(a)
        b = np.array(b)
        diff = a - b
        mean_a = np.mean(a)
        mean_b = np.mean(b)
        mean_diff = np.mean(diff)
        n = len(a)

        # Paired t-test
        t_stat, t_pval = stats.ttest_rel(a, b)

        # Wilcoxon signed-rank (non-parametric)
        try:
            w_stat, w_pval = stats.wilcoxon(a, b)
        except ValueError:
            w_stat, w_pval = None, None

        # Bootstrap
        boot_pval = bootstrap_test(a, b)
        boot_lo, boot_hi = bootstrap_ci(a, b)

        # Win/loss/tie
        wins = np.sum(diff > 0)
        losses = np.sum(diff < 0)
        ties = np.sum(diff == 0)

        print(f"\n  {label}")
        print(f"  {'─'*60}")
        print(f"  N queries:        {n}")
        print(f"  Mean A:           {mean_a:.6f}")
        print(f"  Mean B (vanilla): {mean_b:.6f}")
        print(f"  Mean diff:        {mean_diff:+.6f}")
        print(f"  Std diff:         {np.std(diff):.6f}")
        print(f"")
        print(f"  Paired t-test:    t={t_stat:.4f}  p={t_pval:.6f}  {'*' if t_pval < 0.05 else 'ns'}")
        if w_pval is not None:
            print(f"  Wilcoxon:         W={w_stat:.0f}  p={w_pval:.6f}  {'*' if w_pval < 0.05 else 'ns'}")
        print(f"  Bootstrap:        p={boot_pval:.6f}  {'*' if boot_pval < 0.05 else 'ns'}")
        print(f"  95% CI (boot):    [{boot_lo:+.6f}, {boot_hi:+.6f}]")
        ci_excludes_zero = (boot_lo > 0) or (boot_hi < 0)
        print(f"  CI excludes 0:    {'YES → significant' if ci_excludes_zero else 'NO → not significant'}")
        print(f"")
        print(f"  Wins/Losses/Ties: {wins}/{losses}/{ties}")
        print(f"  Win rate:         {wins/n*100:.1f}%")

        # Effect size (Cohen's d for paired)
        d = mean_diff / np.std(diff) if np.std(diff) > 0 else 0
        print(f"  Cohen's d:        {d:.4f} ({'negligible' if abs(d) < 0.2 else 'small' if abs(d) < 0.5 else 'medium' if abs(d) < 0.8 else 'large'})")

        # Significance summary
        sig = t_pval < 0.05 and boot_pval < 0.05
        print(f"\n  ► {'SIGNIFICANT (p<0.05)' if sig else 'NOT SIGNIFICANT'}")


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
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.data_dir = os.path.expanduser(args.data_dir)

    config = ESCIConfig(
        use_token_weights=True, weight_norm="softmax",
        softmax_temperature=args.temperature,
    )
    tokenizer = AutoTokenizer.from_pretrained(config.checkpoint)

    print("Loading models...")
    gap_model = load_model(config, args.gap_model, args.gap_wh, device)
    loo_model = load_model(config, args.loo_model, args.loo_wh, device)

    if args.dataset in ("esci", "both"):
        print("\nRunning ESCI per-query eval...")
        v, g, l, g4 = eval_esci_perquery(
            gap_model, loo_model, config, device, args.num_eval, tokenizer)
        run_tests("ESCI", v, g, l, g4)

        with open("results/significance_esci.json", "w") as f:
            json.dump({"vanilla": v, "gap": g, "loo": l, "gap_top4": g4}, f)
        print("\nSaved per-query scores to results/significance_esci.json")

    if args.dataset in ("wands", "both"):
        print("\nRunning WANDS per-query eval...")
        v, g, l = eval_wands_perquery(
            gap_model, loo_model, config, device, tokenizer, args.data_dir)
        run_tests("WANDS", v, g, l)

        with open("results/significance_wands.json", "w") as f:
            json.dump({"vanilla": v, "gap": g, "loo": l}, f)
        print("\nSaved per-query scores to results/significance_wands.json")


if __name__ == "__main__":
    main()
