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


def _print_category_table(cat_weights):
    print(f"\n  {'Role':<15} {'Avg Weight':<12} {'Count':<8} {'Std':<8} {'Bar'}")
    print(f"  {'-'*60}")
    for cat in ["modifier", "head_noun", "number", "function", "subword"]:
        ws = cat_weights.get(cat, [])
        if not ws:
            continue
        avg = np.mean(ws)
        std = np.std(ws)
        bar = "█" * int(avg * 200)
        print(f"  {cat:<15} {avg:.4f}       {len(ws):<8} {std:.4f}   {bar}")


def _print_summary(cat_weights):
    mod_w = cat_weights.get("modifier", [])
    head_w = cat_weights.get("head_noun", [])
    func_w = (cat_weights.get("function", []) + cat_weights.get("subword", []))
    if mod_w:
        print(f"  Modifiers:  {np.mean(mod_w):.4f} avg  (n={len(mod_w)})")
    if head_w:
        print(f"  Head nouns: {np.mean(head_w):.4f} avg  (n={len(head_w)})")
    if func_w:
        print(f"  Function:   {np.mean(func_w):.4f} avg  (n={len(func_w)})")
    if mod_w and func_w:
        print(f"  Modifier / Function: {np.mean(mod_w) / np.mean(func_w):.2f}x")
    if head_w and func_w:
        print(f"  Head / Function:     {np.mean(head_w) / np.mean(func_w):.2f}x")
    if mod_w and head_w:
        print(f"  Modifier / Head:     {np.mean(mod_w) / np.mean(head_w):.2f}x")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="esci", choices=["esci", "wands"])
    parser.add_argument("--data_dir", default="~/wands_repo/dataset")
    args = parser.parse_args()

    import os
    args.data_dir = os.path.expanduser(args.data_dir)

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

    # Load queries based on dataset
    if args.dataset == "esci":
        rerank_data = list(ESCIRerankDataset(split="test", locale="us", max_queries=1000))
        all_queries = [item["query"] for item in rerank_data]
    else:
        from wands.evaluate import load_wands
        wands_data = load_wands(args.data_dir)
        all_queries = [item["query"] for item in wands_data]

    dataset_name = "ESCI" if args.dataset == "esci" else "WANDS"
    print(f"Loaded {len(all_queries)} {dataset_name} queries")

    # Check for cached syntax
    syntax_cache_path = f"results/comprehend_syntax_cache_{args.dataset}.json"
    if os.path.exists(syntax_cache_path):
        print(f"Loading cached Comprehend syntax from {syntax_cache_path}")
        with open(syntax_cache_path) as f:
            cache = json.load(f)
        all_pos_maps = cache["pos_maps"]
        all_head_nouns = [set(hn) for hn in cache["head_nouns"]]
    else:
        print("Running Comprehend POS tagging...")
        all_pos_maps = comprehend_pos_batch(all_queries)

        print("Finding head nouns...")
        all_head_nouns = []
        for q in tqdm(all_queries, desc="Head noun detection"):
            all_head_nouns.append(find_head_nouns(all_pos_maps[len(all_head_nouns)], q))

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

    print(f"\n{'=' * 75}")
    print(f"Token Weight Analysis — Comprehend POS — {dataset_name} ({len(all_queries)} queries)")
    print(f"{'=' * 75}")

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

        query_tokens = []
        for t, w, m in zip(tokens, weights, mask):
            if not m or t in ("[CLS]", "[SEP]", "[PAD]", "[Q]", "[MASK]"):
                continue
            cat = categorize_token(t, pos_map, head_nouns)
            query_tokens.append({"token": t, "category": cat, "weight": float(w)})
            category_weights.setdefault(cat, []).append(float(w))

        all_query_data.append({"query": query, "head_nouns": list(head_nouns),
                                "tokens": query_tokens})

    # Attribute detection — use WANDS version for WANDS
    if args.dataset == "wands":
        from wands.evaluate import has_attribute_terms as has_attr_fn
    else:
        from esci.evaluate import has_attribute_terms
        has_attr_fn = lambda q: has_attribute_terms(q)[0]

    attr_cat_weights = {}
    generic_cat_weights = {}
    for qdata in all_query_data:
        is_attr = has_attr_fn(qdata["query"])
        target = attr_cat_weights if is_attr else generic_cat_weights
        for tok in qdata["tokens"]:
            target.setdefault(tok["category"], []).append(tok["weight"])

    # Aggregate
    print(f"\n{'=' * 75}")
    print(f"Aggregate: Average Weight by Token Role ({len(all_queries)} queries)")
    print(f"{'=' * 75}")
    _print_category_table(category_weights)

    n_attr = sum(1 for q in all_query_data if has_attr_fn(q["query"]))
    n_generic = len(all_query_data) - n_attr

    print(f"\n{'=' * 75}")
    print(f"Attribute Queries Only ({n_attr} queries)")
    print(f"{'=' * 75}")
    _print_category_table(attr_cat_weights)
    _print_summary(attr_cat_weights)

    print(f"\n{'=' * 75}")
    print(f"Generic Queries Only ({n_generic} queries)")
    print(f"{'=' * 75}")
    _print_category_table(generic_cat_weights)
    _print_summary(generic_cat_weights)

    print(f"\n{'=' * 75}")
    print("Overall Summary")
    print(f"{'=' * 75}")
    _print_summary(category_weights)

    # Per-query top token analysis
    print(f"\n{'=' * 75}")
    print("Per-Query: Which token role gets the highest weight?")
    print(f"{'=' * 75}")
    top_cat_counts = {}
    for qdata in all_query_data:
        if not qdata["tokens"]:
            continue
        top_tok = max(qdata["tokens"], key=lambda x: x["weight"])
        cat = top_tok["category"]
        top_cat_counts[cat] = top_cat_counts.get(cat, 0) + 1

    total = sum(top_cat_counts.values())
    print(f"\n  All queries ({total}):")
    for cat in ["modifier", "head_noun", "number", "function", "subword"]:
        c = top_cat_counts.get(cat, 0)
        pct = c / total * 100 if total else 0
        bar = "█" * int(pct / 2)
        print(f"    {cat:<15} {c:4d} queries ({pct:5.1f}%)  {bar}")

    def get_present_cats(tokens):
        return set(t["category"] for t in tokens)

    # Modifier + head noun, no numbers
    mod_head_only = [q for q in all_query_data if q["tokens"] and
                     {"modifier", "head_noun"}.issubset(get_present_cats(q["tokens"])) and
                     "number" not in get_present_cats(q["tokens"])]
    if mod_head_only:
        counts = {}
        for q in mod_head_only:
            top = max(q["tokens"], key=lambda x: x["weight"])
            counts[top["category"]] = counts.get(top["category"], 0) + 1
        n = len(mod_head_only)
        print(f"\n  Queries with BOTH modifier + head noun, NO numbers ({n}):")
        for cat in ["modifier", "head_noun", "function", "subword"]:
            c = counts.get(cat, 0)
            pct = c / n * 100 if n else 0
            bar = "█" * int(pct / 2)
            print(f"    {cat:<15} {c:4d} queries ({pct:5.1f}%)  {bar}")

    # All three types
    all_three = [q for q in all_query_data if q["tokens"] and
                 {"modifier", "head_noun", "number"}.issubset(get_present_cats(q["tokens"]))]
    if all_three:
        counts = {}
        for q in all_three:
            top = max(q["tokens"], key=lambda x: x["weight"])
            counts[top["category"]] = counts.get(top["category"], 0) + 1
        n = len(all_three)
        print(f"\n  Queries with modifier + head noun + number ({n}):")
        for cat in ["number", "modifier", "head_noun", "function", "subword"]:
            c = counts.get(cat, 0)
            pct = c / n * 100 if n else 0
            bar = "█" * int(pct / 2)
            print(f"    {cat:<15} {c:4d} queries ({pct:5.1f}%)  {bar}")

    # Head noun only
    head_only = [q for q in all_query_data if q["tokens"] and
                 "head_noun" in get_present_cats(q["tokens"]) and
                 "modifier" not in get_present_cats(q["tokens"]) and
                 "number" not in get_present_cats(q["tokens"])]
    if head_only:
        print(f"\n  Queries with ONLY head nouns ({len(head_only)}):")
        counts = {}
        for q in head_only:
            top = max(q["tokens"], key=lambda x: x["weight"])
            counts[top["category"]] = counts.get(top["category"], 0) + 1
        for cat, c in sorted(counts.items(), key=lambda x: -x[1]):
            print(f"    {cat:<15} {c:4d}")

    # Head noun won — had modifiers?
    head_won = [q for q in all_query_data if q["tokens"] and
                max(q["tokens"], key=lambda x: x["weight"])["category"] == "head_noun"]
    if head_won:
        has_mod = sum(1 for q in head_won if "modifier" in get_present_cats(q["tokens"]))
        no_mod = len(head_won) - has_mod
        print(f"\n  Queries where HEAD NOUN had highest weight ({len(head_won)}):")
        print(f"    Had modifiers present:     {has_mod} ({has_mod/len(head_won)*100:.0f}%)")
        print(f"    No modifiers present:      {no_mod} ({no_mod/len(head_won)*100:.0f}%)")
        if has_mod > 0:
            print(f"    Examples where head beat modifiers:")
            shown = 0
            for q in head_won:
                if "modifier" in get_present_cats(q["tokens"]):
                    toks = " | ".join(f"{t['token']}({t['category'][0]})={t['weight']:.3f}"
                                       for t in q["tokens"])
                    print(f"      \"{q['query']}\" → {toks}")
                    shown += 1
                    if shown >= 5:
                        break

    # Entropy
    print(f"\n{'=' * 75}")
    print("Weight Entropy")
    print(f"{'=' * 75}")
    attr_entropies = []
    generic_entropies = []
    for qdata in all_query_data:
        ws = np.array([t["weight"] for t in qdata["tokens"]])
        if len(ws) < 2:
            continue
        ws = ws[ws > 0]
        entropy = -np.sum(ws * np.log(ws + 1e-10))
        max_ent = np.log(len(ws))
        norm_entropy = entropy / max_ent if max_ent > 0 else 0

        is_attr = has_attr_fn(qdata["query"])
        if is_attr:
            attr_entropies.append(norm_entropy)
        else:
            generic_entropies.append(norm_entropy)

    if attr_entropies:
        print(f"  Attribute queries:  entropy={np.mean(attr_entropies):.4f} (n={len(attr_entropies)})")
    if generic_entropies:
        print(f"  Generic queries:    entropy={np.mean(generic_entropies):.4f} (n={len(generic_entropies)})")
    print(f"  (Lower = more peaked = model more confident)")

    # Discriminator analysis: for queries where modifier won,
    # show the high-weight modifier and whether it distinguishes Exact from Substitute
    print(f"\n{'=' * 75}")
    print("Modifier Discriminator Analysis")
    print("Queries where top-weighted modifier is the discriminating token")
    print(f"{'=' * 75}")

    if args.dataset == "esci":
        rerank_items = list(ESCIRerankDataset(split="test", locale="us", max_queries=1000))
        query_to_products = {item["query"]: item["products"] for item in rerank_items}
    else:
        from wands.evaluate import load_wands
        wands_items = load_wands(args.data_dir)
        query_to_products = {}
        for item in wands_items:
            query_to_products[item["query"]] = item["products"]

    discriminator_count = 0
    non_discriminator_count = 0
    shown = 0

    for qdata in all_query_data:
        if not qdata["tokens"]:
            continue
        top_tok = max(qdata["tokens"], key=lambda x: x["weight"])
        if top_tok["category"] != "modifier":
            continue

        query = qdata["query"]
        products = query_to_products.get(query, [])
        if len(products) < 2:
            continue

        mod_token = top_tok["token"].lower().replace("##", "")

        # Check: does this modifier appear more in Exact titles than Substitute/Partial?
        if args.dataset == "esci":
            exact_titles = [p["title"].lower() for p in products if p["esci_label"] == "E"]
            sub_titles = [p["title"].lower() for p in products if p["esci_label"] == "S"]
        else:
            exact_titles = [p["title"].lower() for p in products if p["label"] == 2]
            sub_titles = [p["title"].lower() for p in products if p["label"] == 1]

        if not exact_titles or not sub_titles:
            continue

        exact_has = sum(1 for t in exact_titles if mod_token in t) / len(exact_titles)
        sub_has = sum(1 for t in sub_titles if mod_token in t) / len(sub_titles)

        is_discriminator = exact_has > sub_has
        if is_discriminator:
            discriminator_count += 1
        else:
            non_discriminator_count += 1

        if shown < 15:
            marker = "✓ DISC" if is_discriminator else "✗ not disc"
            toks = " | ".join(f"{t['token']}({t['category'][0]})={t['weight']:.3f}"
                               for t in qdata["tokens"])
            print(f"\n  \"{query}\"")
            print(f"    Top modifier: \"{mod_token}\" (w={top_tok['weight']:.3f})")
            print(f"    In Exact titles: {exact_has*100:.0f}% ({len(exact_titles)} titles)")
            print(f"    In Substitute titles: {sub_has*100:.0f}% ({len(sub_titles)} titles)")
            print(f"    → {marker}")
            shown += 1

    total_checked = discriminator_count + non_discriminator_count
    if total_checked > 0:
        print(f"\n  Summary ({total_checked} queries with modifier as top weight):")
        print(f"    Modifier IS discriminator (more in Exact): {discriminator_count} ({discriminator_count/total_checked*100:.0f}%)")
        print(f"    Modifier NOT discriminator:                {non_discriminator_count} ({non_discriminator_count/total_checked*100:.0f}%)")

    out_path = f"results/weight_analysis_{args.dataset}.json"
    with open(out_path, "w") as f:
        json.dump({
            "dataset": dataset_name,
            "num_queries": len(all_queries),
            "queries": all_query_data,
            "category_stats": {cat: {"mean": float(np.mean(ws)), "std": float(np.std(ws)),
                                      "count": len(ws)}
                                for cat, ws in category_weights.items() if ws},
        }, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
