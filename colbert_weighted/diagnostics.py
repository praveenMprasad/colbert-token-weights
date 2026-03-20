"""Diagnostics: token weight inspection, distribution stats, pruning test."""
import json
import torch
import numpy as np
from transformers import AutoTokenizer
from .config import ExpConfig
from .model import ColBERTWeighted


def inspect_token_weights(model, tokenizer, queries, device="cpu"):
    """Log tokens + weights for sample queries."""
    model.eval()
    results = []
    for query in queries:
        enc = tokenizer(query, return_tensors="pt", padding="max_length",
                        truncation=True, max_length=model.config.query_maxlen).to(device)
        with torch.no_grad():
            _, mask, hidden = model.encode(enc.input_ids, enc.attention_mask)
            if model.weight_head is not None:
                weights = model.weight_head(hidden, mask)
            else:
                weights = mask.float() / mask.float().sum(dim=-1, keepdim=True)

        tokens = tokenizer.convert_ids_to_tokens(enc.input_ids[0])
        w = weights[0].cpu().numpy()
        m = mask[0].cpu().numpy()

        token_weights = []
        for t, wi, mi in zip(tokens, w, m):
            if mi:
                token_weights.append((t, float(wi)))
        results.append({"query": query, "token_weights": token_weights})
    return results


def weight_distribution_stats(model, tokenizer, queries, device="cpu"):
    """Track max weight, entropy, active tokens per query."""
    model.eval()
    stats = {"max_weight": [], "entropy": [], "active_tokens": []}
    for query in queries:
        enc = tokenizer(query, return_tensors="pt", padding="max_length",
                        truncation=True, max_length=model.config.query_maxlen).to(device)
        with torch.no_grad():
            _, mask, hidden = model.encode(enc.input_ids, enc.attention_mask)
            if model.weight_head is not None:
                weights = model.weight_head(hidden, mask)
            else:
                n = mask.float().sum(dim=-1, keepdim=True)
                weights = mask.float() / n

        w = weights[0].cpu().numpy()
        m = mask[0].cpu().numpy().astype(bool)
        w_active = w[m]

        stats["max_weight"].append(float(w_active.max()))
        # Entropy
        w_clipped = np.clip(w_active, 1e-12, None)
        entropy = -np.sum(w_clipped * np.log(w_clipped))
        stats["entropy"].append(float(entropy))
        # Active tokens (weight > 1/(2*num_tokens) — above uniform/2)
        threshold = 1.0 / (2 * len(w_active))
        stats["active_tokens"].append(int((w_active > threshold).sum()))

    return stats


def pruning_test(model, tokenizer, query, doc, ks=(2, 4, 8, None), device="cpu"):
    """Score a (query, doc) pair keeping only top-k weighted query tokens.

    Returns dict: k -> score
    """
    model.eval()
    q_enc = tokenizer(query, return_tensors="pt", padding="max_length",
                      truncation=True, max_length=model.config.query_maxlen).to(device)
    d_enc = tokenizer(doc, return_tensors="pt", padding="max_length",
                      truncation=True, max_length=model.config.doc_maxlen).to(device)

    with torch.no_grad():
        Q, q_mask, q_hidden = model.encode(q_enc.input_ids, q_enc.attention_mask)
        D, d_mask, _ = model.encode(d_enc.input_ids, d_enc.attention_mask)

        if model.weight_head is not None:
            weights = model.weight_head(q_hidden, q_mask)
        else:
            weights = q_mask.float() / q_mask.float().sum(dim=-1, keepdim=True)

    results = {}
    w = weights[0]
    for k in ks:
        if k is None:
            pruned_mask = q_mask.clone()
        else:
            _, topk_idx = w.topk(min(k, int(q_mask.sum())))
            pruned_mask = torch.zeros_like(q_mask)
            pruned_mask[0, topk_idx] = 1
            pruned_mask = pruned_mask.bool()

        # Re-normalize weights for kept tokens
        w_pruned = weights * pruned_mask.float()
        denom = w_pruned.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        w_pruned = w_pruned / denom

        from .scoring import weighted_maxsim
        score = weighted_maxsim(Q, D, pruned_mask, d_mask, w_pruned)
        results[k if k else "all"] = float(score.item())

    return results
