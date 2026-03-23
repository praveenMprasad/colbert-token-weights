"""WANDS generalization eval: test ESCI-trained weight head on Wayfair data.

Zero-shot transfer: weight head trained on Amazon ESCI, evaluated on Wayfair WANDS.
No training on WANDS — pure generalization test.

WANDS data from: https://github.com/wayfair/WANDS
Labels: Exact=2, Partial=1, Irrelevant=0
"""
import os
import csv
import math
import re
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from collections import defaultdict

from esci.config import ESCIConfig
from esci.model import ColBERTESCI
from colbert_weighted.scoring import maxsim, weighted_maxsim


# Reuse attribute terms from ESCI eval
GENDER_TERMS = {"men", "mens", "men's", "women", "womens", "women's", "boy", "boys",
                "girl", "girls", "male", "female", "unisex", "ladies"}
COLOR_TERMS = {"black", "white", "red", "blue", "green", "yellow", "pink", "purple",
               "brown", "grey", "gray", "orange", "navy", "beige", "gold", "silver"}
SIZE_TERMS = {"small", "medium", "large", "xl", "xxl", "xs", "petite", "plus",
              "tall", "short", "mini", "king", "queen", "twin", "full"}
MATERIAL_TERMS = {"wood", "wooden", "metal", "leather", "velvet", "cotton", "linen",
                  "marble", "glass", "ceramic", "steel", "iron", "brass", "oak",
                  "walnut", "bamboo", "rattan", "wicker", "fabric", "upholstered"}
ATTRIBUTE_TERMS = GENDER_TERMS | COLOR_TERMS | SIZE_TERMS | MATERIAL_TERMS

LABEL_MAP = {"Exact": 2, "Partial": 1, "Irrelevant": 0}


def has_attribute_terms(query):
    words = set(re.findall(r"[a-z']+", query.lower()))
    return len(words & ATTRIBUTE_TERMS) > 0


def mrr_at_k(ranked_labels, k=10, threshold=2):
    """MRR@k. threshold=2 means only Exact is relevant."""
    for i, label in enumerate(ranked_labels[:k]):
        if label >= threshold:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(ranked_labels, k=10):
    dcg = sum(l / math.log2(i + 2) for i, l in enumerate(ranked_labels[:k]))
    ideal = sorted(ranked_labels, reverse=True)[:k]
    idcg = sum(l / math.log2(i + 2) for i, l in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def load_wands(data_dir="wands/dataset"):
    """Load WANDS CSV files into query-product groups."""
    # Load products
    products = {}
    with open(os.path.join(data_dir, "product.csv"), "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            products[row["product_id"]] = row["product_name"]

    # Load queries
    queries = {}
    with open(os.path.join(data_dir, "query.csv"), "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            queries[row["query_id"]] = row["query"]

    # Load labels and group by query
    query_products = defaultdict(list)
    with open(os.path.join(data_dir, "label.csv"), "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = row["query_id"]
            pid = row["product_id"]
            label = LABEL_MAP.get(row["label"], 0)
            if pid in products and qid in queries:
                query_products[qid].append({
                    "title": products[pid],
                    "label": label,
                    "label_str": row["label"],
                })

    data = []
    for qid, prods in query_products.items():
        data.append({"query": queries[qid], "products": prods})

    print(f"Loaded WANDS: {len(data)} queries, {sum(len(d['products']) for d in data)} judgments")
    return data


def evaluate_wands(model, config, device, data_dir="wands/dataset", max_queries=None):
    """Evaluate ColBERT + weight head on WANDS. Zero-shot transfer."""
    tokenizer = AutoTokenizer.from_pretrained(config.checkpoint)
    data = load_wands(data_dir)

    if max_queries:
        data = data[:max_queries]

    model.eval()
    has_weights = model.weight_head is not None

    all_mrr_v, all_mrr_w = [], []
    all_ndcg_v, all_ndcg_w = [], []
    attr_mrr_v, attr_mrr_w = [], []
    non_attr_mrr_v, non_attr_mrr_w = [], []
    # Exact vs Partial separation
    sep_v, sep_w = [], []

    for item in tqdm(data, desc="WANDS eval"):
        query = item["query"]
        products = item["products"]
        if len(products) < 2:
            continue

        q_enc = tokenizer(query, return_tensors="pt", padding="max_length",
                          truncation=True, max_length=config.query_maxlen).to(device)

        with torch.no_grad():
            Q, q_mask, q_hidden = model.encode(q_enc.input_ids, q_enc.attention_mask)

        v_scores, w_scores = [], []
        for prod in products:
            d_enc = tokenizer(prod["title"], return_tensors="pt", padding="max_length",
                              truncation=True, max_length=config.doc_maxlen).to(device)
            with torch.no_grad():
                D, d_mask, _ = model.encode(d_enc.input_ids, d_enc.attention_mask)
                v_scores.append(maxsim(Q, D, q_mask, d_mask).item())
                if has_weights:
                    weights = model.weight_head(q_hidden, q_mask)
                    w_scores.append(weighted_maxsim(Q, D, q_mask, d_mask, weights).item())

        # Vanilla ranking
        v_order = np.argsort(v_scores)[::-1]
        v_labels = [products[i]["label"] for i in v_order]
        all_mrr_v.append(mrr_at_k(v_labels))
        all_ndcg_v.append(ndcg_at_k(v_labels))

        is_attr = has_attribute_terms(query)
        if is_attr:
            attr_mrr_v.append(mrr_at_k(v_labels))
        else:
            non_attr_mrr_v.append(mrr_at_k(v_labels))

        # Exact vs Partial separation
        v_labels_str = [products[i]["label_str"] for i in v_order]
        correct, total = 0, 0
        for i, li in enumerate(v_labels_str):
            for j, lj in enumerate(v_labels_str):
                if li == "Exact" and lj == "Partial":
                    total += 1
                    if i < j:
                        correct += 1
        if total > 0:
            sep_v.append(correct / total)

        # Weighted ranking
        if has_weights and w_scores:
            w_order = np.argsort(w_scores)[::-1]
            w_labels = [products[i]["label"] for i in w_order]
            all_mrr_w.append(mrr_at_k(w_labels))
            all_ndcg_w.append(ndcg_at_k(w_labels))

            if is_attr:
                attr_mrr_w.append(mrr_at_k(w_labels))
            else:
                non_attr_mrr_w.append(mrr_at_k(w_labels))

            w_labels_str = [products[i]["label_str"] for i in w_order]
            correct, total = 0, 0
            for i, li in enumerate(w_labels_str):
                for j, lj in enumerate(w_labels_str):
                    if li == "Exact" and lj == "Partial":
                        total += 1
                        if i < j:
                            correct += 1
            if total > 0:
                sep_w.append(correct / total)

    results = {
        "num_queries": len(all_mrr_v),
        "vanilla_mrr@10": float(np.mean(all_mrr_v)) if all_mrr_v else None,
        "vanilla_ndcg@10": float(np.mean(all_ndcg_v)) if all_ndcg_v else None,
        "vanilla_attr_mrr@10": float(np.mean(attr_mrr_v)) if attr_mrr_v else None,
        "vanilla_non_attr_mrr@10": float(np.mean(non_attr_mrr_v)) if non_attr_mrr_v else None,
        "vanilla_separation": float(np.mean(sep_v)) if sep_v else None,
    }
    if has_weights:
        results.update({
            "weighted_mrr@10": float(np.mean(all_mrr_w)) if all_mrr_w else None,
            "weighted_ndcg@10": float(np.mean(all_ndcg_w)) if all_ndcg_w else None,
            "weighted_attr_mrr@10": float(np.mean(attr_mrr_w)) if attr_mrr_w else None,
            "weighted_non_attr_mrr@10": float(np.mean(non_attr_mrr_w)) if non_attr_mrr_w else None,
            "weighted_separation": float(np.mean(sep_w)) if sep_w else None,
        })

    return results
