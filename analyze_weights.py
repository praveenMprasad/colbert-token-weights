#!/usr/bin/env python3
"""Weight distribution analysis with AWS Comprehend POS tagging.

Runs on ALL ESCI test queries for aggregate stats.
Shows detailed per-query output for a curated sample.

Categories:
  - Head noun: core product/subject (last noun(s) in main phrase)
  - Modifier: adjectives, adverbs, attributes, pre-head nouns
  - Function: determiners, prepositions, conjunctions, punctuation
  - Subword: wordpiece continuations (##xyz)
  - Number: numeric tokens
"""
import torch
import numpy as np
import json
import re
import boto3
from transformers import AutoTokenizer
from tqdm import tqdm
from esci.config import ESCIConfig
from esci.model import ColBERTESCI
from esci.data import ESCIRerankDataset

comprehend = boto3.client("comprehend", region_name="us-east-1")

STOPWORDS = {"the", "a", "an", "for", "with", "in", "of", "to", "and", "or",
             "on", "at", "by", "from", "is", "it", "that", "this", "as", "be",
             "'s", "'", "s"}

MODIFIER_POS = {"ADJ", "ADV", "VERB"}
NOUN_POS = {"NOUN", "PROPN"}
FUNCTION_POS = {"ADP", "DET", "CONJ", "CCONJ", "SCONJ", "PART", "PRON",
                "AUX", "INTJ", "PUNCT"}

# Batch Comprehend: up to 25 texts per call
COMPREHEND_BATCH = 25


def comprehend_pos_batch(queries):
    """Batch POS tagging via Comprehend. Returns list of {word: pos} dicts."""
    results = []
    for i in range(0, len(queries), COMPREHEND_BATCH):
        batch = queries[i:i + COMPREHEND_BATCH]
        resp = comprehend.batch_detect_syntax(
            TextList=batch, LanguageCode="en")
        for item in resp["ResultList"]:
            pos_map = {}
            for token in item["SyntaxTokens"]:
                pos_map[token["Text"].lower()] = token["PartOfSpeech"]["Tag"]
            results.append(pos_map)
    return results


def find_head_nouns(pos_map, query):
    """Head noun = last noun(s) in main phrase (before preposition)."""
    resp = comprehend.detect_syntax(Text=query, LanguageCode="en")
    tokens = resp["SyntaxTokens"]

    main_tokens = []
    for t in tokens:
        if t["PartOfSpeech"]["Tag"] == "ADP":
            break
        main_tokens.append(t)

    if not main_tokens:
        main_tokens = tokens

    head_nouns = set()
    for t in reversed(main_tokens):
        pos = t["PartOfSpeech"]["Tag"]
        if pos in ("NOUN", "PROPN"):
            head_nouns.add(t["Text"].lower())
        elif head_nouns:
            break

    return head_nouns


def categorize_token(token_text, pos_map, head_nouns):
    if token_text.startswith("##"):
        return "subword"
    t = token_text.lower()
    if t in STOPWORDS:
        return "function"
    if re.match(r"^\d+$", t):
        return "number"
    if t in head_nouns:
        return "head_noun"
    pos = pos_map.get(t, "")
    if pos in MODIFIER_POS:
        return "modifier"
    if pos in NOUN_POS:
        return "modifier"  # pre-head noun = modifier
    if pos in FUNCTION_POS:
        return "function"
    return "modifier"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = ESCIConfig(
        use_token_weights=True, weight_norm="softmax",
        softmax_temperature=0.3,
    )
    tokenizer = AutoTokenizer.from_pretrained(config.checkpoint)

    model = ColBERTESCI(config).to(device)
    model.load_state_dict(torch.load("outputs/esci/gap_multineg/model.pt",
                                      map_location=device))
    model.weight_head.load_state_dict(
        torch.load("outputs/esci/gap_multineg/weight_head_step2000.pt",
                    map_location=device))
    model.eval()

    # Load all test queries
    rerank_data = list(ESCIRerankDataset(split="test", locale="us", max_queries=1000))
    all_queries = [item["query"] for item in rerank_data]
    print(f"Loaded {len(all_queries)} queries")

    # Sample queries for detailed output
    SHOW_QUERIES = {
        "men's black leather jacket",
        "women's red silk dress size 8",
        "blue velvet sofa",
        "large wooden dining table",
        "gold metal wall mirror",
        "boys white cotton shirt",
        "pink running shoes for girls",
        "laptop stand",
        "wireless headphones",
        "coffee table",
        "water bottle for gym",
        "stainless steel kitchen knife set",
        "queen size grey bed sheets",
        "men's waterproof hiking boots brown",
        "organic green tea bags",
    }

    # Check for cached syntax
    syntax_cache_path = "results/comprehend_syntax_cache.json"
    import os
    if os.path.exists(syntax_cache_path):
        print(f"Loading cached Comprehend syntax from {syntax_cache_path}")
        with open(syntax_cache_path) as f:
            cache = json.load(f)
        all_pos_maps = cache["pos_maps"]
        all_head_nouns = [set(hn) for hn in cache["head_nouns"]]
    else:
        # Batch POS tag all queries
        print("Running Comprehend POS tagging...")
        all_pos_maps = comprehend_pos_batch(all_queries)

        # Get head nouns for all queries
        print("Finding head nouns...")
        all_head_nouns = []
        for q in tqdm(all_queries, desc="Head noun detection"):
            all_head_nouns.append(find_head_nouns(all_pos_maps[len(all_head_nouns)], q))

        # Save cache
        with open(syntax_cache_path, "w") as f:
            json.dump({
                "num_queries": len(all_queries),
                "queries": all_queries,
                "pos_maps": all_pos_maps,
                "head_nouns": [list(hn) for hn in all_head_nouns],
            }, f, indent=2)
        print(f"Saved Comprehend syntax cache to {syntax_cache_path}")

    # Run weight head on all queries
    print("Computing weights...")
    category_weights = {}
    all_query_data = []

    print("\n" + "=" * 75)
    print("Token Weight Analysis — Comprehend POS (all 1000 queries)")
    print("=" * 75)

    for idx, query in enumerate(tqdm(all_queries, desc="Weight analysis")):
        pos_map = all_pos_maps[idx]
        head_nouns = all_head_nouns[idx]

        enc = tokenizer(query, return_tensors="pt", padding="max_length",
                        truncation=True, max_length=config.query_maxlen).to(device)

        with torch.no_grad():
            Q, q_mask, q_hidden = model.encode(enc.input_ids, enc.attention_mask)
            weights = model.weight_head(q_hidden, q_mask)[0].cpu().numpy()

        tokens = tokenizer.convert_ids_to_tokens(enc.input_ids[0])
        mask = q_mask[0].cpu().numpy()

        show = query in SHOW_QUERIES
        if show:
            print(f"\nQuery: \"{query}\"")
            print(f"  POS: {pos_map}")
            print(f"  Head nouns: {head_nouns}")
            print(f"  {'Token':<15} {'Category':<12} {'Weight':<8} {'Bar'}")
            print(f"  {'-'*60}")

        query_tokens = []
        for t, w, m in zip(tokens, weights, mask):
            if not m or t in ("[CLS]", "[SEP]", "[PAD]", "[Q]", "[MASK]"):
                continue
            cat = categorize_token(t, pos_map, head_nouns)
            if show:
                bar = "█" * int(w * 200)
                marker = " ◄HEAD" if cat == "head_noun" else ""
                print(f"  {t:<15} {cat:<12} {w:.4f}   {bar}{marker}")

            query_tokens.append({"token": t, "category": cat, "weight": float(w)})
            category_weights.setdefault(cat, []).append(float(w))

        all_query_data.append({"query": query, "head_nouns": list(head_nouns),
                                "tokens": query_tokens})

    # Aggregate
    print(f"\n{'=' * 75}")
    print(f"Aggregate: Average Weight by Token Role ({len(all_queries)} queries)")
    print(f"{'=' * 75}")
    print(f"\n  {'Role':<15} {'Avg Weight':<12} {'Count':<8} {'Std':<8} {'Bar'}")
    print(f"  {'-'*60}")

    order = ["modifier", "head_noun", "number", "function", "subword"]
    for cat in order:
        ws = category_weights.get(cat, [])
        if not ws:
            continue
        avg = np.mean(ws)
        std = np.std(ws)
        bar = "█" * int(avg * 200)
        print(f"  {cat:<15} {avg:.4f}       {len(ws):<8} {std:.4f}   {bar}")

    # Summary
    mod_w = category_weights.get("modifier", [])
    head_w = category_weights.get("head_noun", [])
    func_w = (category_weights.get("function", []) +
              category_weights.get("subword", []))

    print(f"\n{'=' * 75}")
    print("Summary")
    print(f"{'=' * 75}")
    if mod_w:
        print(f"  Modifiers (adj/adv/attr):  {np.mean(mod_w):.4f} avg  (n={len(mod_w)})")
    if head_w:
        print(f"  Head nouns (product type): {np.mean(head_w):.4f} avg  (n={len(head_w)})")
    if func_w:
        print(f"  Function (stop/subword):   {np.mean(func_w):.4f} avg  (n={len(func_w)})")

    if mod_w and func_w:
        print(f"\n  Modifier / Function ratio: {np.mean(mod_w) / np.mean(func_w):.2f}x")
    if head_w and func_w:
        print(f"  Head noun / Function ratio: {np.mean(head_w) / np.mean(func_w):.2f}x")
    if mod_w and head_w:
        print(f"  Modifier / Head noun ratio: {np.mean(mod_w) / np.mean(head_w):.2f}x")

    with open("results/weight_analysis.json", "w") as f:
        json.dump({
            "num_queries": len(all_queries),
            "queries": all_query_data,
            "category_stats": {cat: {"mean": float(np.mean(ws)), "std": float(np.std(ws)),
                                      "count": len(ws)}
                                for cat, ws in category_weights.items() if ws},
        }, f, indent=2)
    print(f"\nSaved to results/weight_analysis.json")


if __name__ == "__main__":
    main()
