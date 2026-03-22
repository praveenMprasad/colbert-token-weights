"""ESCI evaluation: reranking with attribute-specific analysis.

Compares baseline (no weights) vs weighted ColBERT on ESCI test split.
Metrics: MRR@10, NDCG@10, Exact-vs-Substitute separation, attribute analysis, pruning.
"""
import re
import math
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from torch.cuda.amp import autocast

from .config import ESCIConfig
from .model import ColBERTESCI
from .data import ESCIRerankDataset
from colbert_weighted.scoring import maxsim, weighted_maxsim


# Attribute keyword lists for analysis
GENDER_TERMS = {"men", "mens", "men's", "women", "womens", "women's", "boy", "boys",
                "girl", "girls", "male", "female", "unisex", "ladies"}
COLOR_TERMS = {"black", "white", "red", "blue", "green", "yellow", "pink", "purple",
               "brown", "grey", "gray", "orange", "navy", "beige", "gold", "silver"}
SIZE_TERMS = {"small", "medium", "large", "xl", "xxl", "xs", "petite", "plus",
              "tall", "short", "mini", "king", "queen", "twin", "full"}
ATTRIBUTE_TERMS = GENDER_TERMS | COLOR_TERMS | SIZE_TERMS


def mrr_at_k(ranked_labels, k=10):
    """MRR@k from a list of relevance labels sorted by score desc."""
    for i, label in enumerate(ranked_labels[:k]):
        if label >= 3:  # E = 3 is relevant
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(ranked_labels, k=10):
    """NDCG@k from a list of relevance labels sorted by score desc."""
    dcg = sum(l / math.log2(i + 2) for i, l in enumerate(ranked_labels[:k]))
    ideal = sorted(ranked_labels, reverse=True)[:k]
    idcg = sum(l / math.log2(i + 2) for i, l in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def exact_substitute_separation(ranked_labels_esci):
    """Measures how well Exact products rank above Substitute products.

    Returns: fraction of (E, S) pairs where E ranks higher than S.
    """
    correct, total = 0, 0
    for i, li in enumerate(ranked_labels_esci):
        for j, lj in enumerate(ranked_labels_esci):
            if li == "E" and lj == "S":
                total += 1
                if i < j:
                    correct += 1
    return correct / total if total > 0 else None


def has_attribute_terms(query):
    """Check if query contains attribute filter terms."""
    words = set(re.findall(r"[a-z']+", query.lower()))
    found = words & ATTRIBUTE_TERMS
    return len(found) > 0, found


def score_query_products(model, tokenizer, query, products, config, device):
    """Score all products for a query. Returns (vanilla_scores, weighted_scores, weights)."""
    q_enc = tokenizer(query, return_tensors="pt", padding="max_length",
                      truncation=True, max_length=config.query_maxlen).to(device)

    with torch.no_grad():
        Q, q_mask, q_hidden = model.encode(q_enc.input_ids.to(device), q_enc.attention_mask.to(device))

    vanilla_scores = []
    weighted_scores = []
    for prod in products:
        d_enc = tokenizer(prod["title"], return_tensors="pt", padding="max_length",
                          truncation=True, max_length=config.doc_maxlen).to(device)
        with torch.no_grad():
            D, d_mask, _ = model.encode(d_enc.input_ids, d_enc.attention_mask)
            v_score = maxsim(Q, D, q_mask, d_mask)
            vanilla_scores.append(v_score.item())

            if model.weight_head is not None:
                weights = model.weight_head(q_hidden, q_mask)
                w_score = weighted_maxsim(Q, D, q_mask, d_mask, weights)
                weighted_scores.append(w_score.item())

    return vanilla_scores, weighted_scores


def evaluate(model, config, device, max_queries=None, split="test"):
    """Full ESCI evaluation with attribute analysis."""
    tokenizer = AutoTokenizer.from_pretrained(config.checkpoint)
    rerank_data = ESCIRerankDataset(split=split, locale=config.locale, max_queries=max_queries)

    model.eval()
    has_weights = model.weight_head is not None

    all_mrr_vanilla, all_mrr_weighted = [], []
    all_ndcg_vanilla, all_ndcg_weighted = [], []
    attr_mrr_vanilla, attr_mrr_weighted = [], []
    non_attr_mrr_vanilla, non_attr_mrr_weighted = [], []
    separation_vanilla, separation_weighted = [], []

    for item in tqdm(rerank_data, desc="Evaluating"):
        query = item["query"]
        products = item["products"]
        if len(products) < 2:
            continue

        v_scores, w_scores = score_query_products(model, tokenizer, query, products, config, device)

        # Rank by vanilla scores
        v_order = np.argsort(v_scores)[::-1]
        v_labels = [products[i]["label"] for i in v_order]
        v_esci = [products[i]["esci_label"] for i in v_order]

        all_mrr_vanilla.append(mrr_at_k(v_labels))
        all_ndcg_vanilla.append(ndcg_at_k(v_labels))

        sep_v = exact_substitute_separation(v_esci)
        if sep_v is not None:
            separation_vanilla.append(sep_v)

        has_attr, _ = has_attribute_terms(query)
        if has_attr:
            attr_mrr_vanilla.append(mrr_at_k(v_labels))
        else:
            non_attr_mrr_vanilla.append(mrr_at_k(v_labels))

        # Rank by weighted scores
        if has_weights and w_scores:
            w_order = np.argsort(w_scores)[::-1]
            w_labels = [products[i]["label"] for i in w_order]
            w_esci = [products[i]["esci_label"] for i in w_order]

            all_mrr_weighted.append(mrr_at_k(w_labels))
            all_ndcg_weighted.append(ndcg_at_k(w_labels))

            sep_w = exact_substitute_separation(w_esci)
            if sep_w is not None:
                separation_weighted.append(sep_w)

            if has_attr:
                attr_mrr_weighted.append(mrr_at_k(w_labels))
            else:
                non_attr_mrr_weighted.append(mrr_at_k(w_labels))

    results = {
        "num_queries": len(all_mrr_vanilla),
        "vanilla_mrr@10": float(np.mean(all_mrr_vanilla)) if all_mrr_vanilla else None,
        "vanilla_ndcg@10": float(np.mean(all_ndcg_vanilla)) if all_ndcg_vanilla else None,
        "vanilla_separation": float(np.mean(separation_vanilla)) if separation_vanilla else None,
        "vanilla_attr_mrr@10": float(np.mean(attr_mrr_vanilla)) if attr_mrr_vanilla else None,
        "vanilla_non_attr_mrr@10": float(np.mean(non_attr_mrr_vanilla)) if non_attr_mrr_vanilla else None,
    }
    if has_weights:
        results.update({
            "weighted_mrr@10": float(np.mean(all_mrr_weighted)) if all_mrr_weighted else None,
            "weighted_ndcg@10": float(np.mean(all_ndcg_weighted)) if all_ndcg_weighted else None,
            "weighted_separation": float(np.mean(separation_weighted)) if separation_weighted else None,
            "weighted_attr_mrr@10": float(np.mean(attr_mrr_weighted)) if attr_mrr_weighted else None,
            "weighted_non_attr_mrr@10": float(np.mean(non_attr_mrr_weighted)) if non_attr_mrr_weighted else None,
        })

    return results


def pruning_eval(model, config, device, max_queries=100, ks=(2, 4, 8)):
    """Evaluate retrieval quality when pruning to top-k weighted query tokens."""
    tokenizer = AutoTokenizer.from_pretrained(config.checkpoint)
    rerank_data = ESCIRerankDataset(split="test", locale=config.locale, max_queries=max_queries)

    model.eval()
    if model.weight_head is None:
        print("No weight head — skipping pruning eval.")
        return None

    mrr_by_k = {k: [] for k in ks}
    mrr_by_k["all"] = []

    for item in tqdm(rerank_data, desc="Pruning eval"):
        query = item["query"]
        products = item["products"]
        if len(products) < 2:
            continue

        q_enc = tokenizer(query, return_tensors="pt", padding="max_length",
                          truncation=True, max_length=config.query_maxlen).to(device)
        with torch.no_grad():
            Q, q_mask, q_hidden = model.encode(q_enc.input_ids, q_enc.attention_mask)
            weights = model.weight_head(q_hidden, q_mask)

        # Encode all products once
        d_embs, d_masks = [], []
        for prod in products:
            d_enc = tokenizer(prod["title"], return_tensors="pt", padding="max_length",
                              truncation=True, max_length=config.doc_maxlen).to(device)
            with torch.no_grad():
                D, d_mask, _ = model.encode(d_enc.input_ids, d_enc.attention_mask)
            d_embs.append(D)
            d_masks.append(d_mask)

        labels = [p["label"] for p in products]

        for k in list(ks) + [None]:
            if k is not None:
                _, topk_idx = weights[0].topk(min(k, int(q_mask.sum())))
                pruned_mask = torch.zeros_like(q_mask)
                pruned_mask[0, topk_idx] = 1
                pruned_mask = pruned_mask.bool()
            else:
                pruned_mask = q_mask.clone()

            w_pruned = weights * pruned_mask.float()
            denom = w_pruned.sum(dim=-1, keepdim=True).clamp(min=1e-9)
            w_pruned = w_pruned / denom

            scores = []
            for D, d_mask in zip(d_embs, d_masks):
                with torch.no_grad():
                    s = weighted_maxsim(Q, D, pruned_mask, d_mask, w_pruned)
                scores.append(s.item())

            order = np.argsort(scores)[::-1]
            ranked_labels = [labels[i] for i in order]
            key = k if k is not None else "all"
            mrr_by_k[key].append(mrr_at_k(ranked_labels))

    results = {}
    for k, vals in mrr_by_k.items():
        results[f"mrr@10_top{k}"] = float(np.mean(vals)) if vals else None
    return results
