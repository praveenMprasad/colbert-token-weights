#!/usr/bin/env python3
"""Weight distribution analysis with POS-based token categorization.

Categories:
  - Head noun: core subject of the query (last noun in noun phrase, product type)
  - Modifier: adjectives, adverbs, attributes that narrow the search
  - Function: stopwords, prepositions, determiners, punctuation
  - Subword: wordpiece continuations (##xyz)
  - Number: numeric tokens
"""
import torch
import numpy as np
import json
import re
import nltk
from transformers import AutoTokenizer
from esci.config import ESCIConfig
from esci.model import ColBERTESCI

nltk.download("averaged_perceptron_tagger_eng", quiet=True)
nltk.download("punkt_tab", quiet=True)

STOPWORDS = {"the", "a", "an", "for", "with", "in", "of", "to", "and", "or",
             "on", "at", "by", "from", "is", "it", "that", "this", "as", "be",
             "'s", "'", "s"}

# POS tags → category mapping
# NN/NNS/NNP = nouns, JJ/JJR/JJS = adjectives, RB = adverbs
# VBG can be modifier ("running shoes") or head ("running")
MODIFIER_POS = {"JJ", "JJR", "JJS", "RB", "RBR", "RBS", "VBG", "VBN", "CD"}
NOUN_POS = {"NN", "NNS", "NNP", "NNPS"}
FUNCTION_POS = {"IN", "DT", "CC", "TO", "PRP", "PRP$", "WDT", "WP", "EX",
                "MD", "POS", "RP"}


def get_query_pos_tags(query):
    """POS tag a query and return {word: pos} mapping."""
    words = nltk.word_tokenize(query)
    tagged = nltk.pos_tag(words)
    return {w.lower(): pos for w, pos in tagged}


def find_head_nouns(query):
    """Identify head nouns — the core product/subject tokens.
    
    Heuristic: the last noun(s) before a preposition or end of query.
    E.g., "men's black leather jacket" → jacket is head noun
          "running shoes for girls" → shoes is head noun
          "stainless steel kitchen knife set" → knife, set are head nouns
    """
    words = nltk.word_tokenize(query)
    tagged = nltk.pos_tag(words)
    
    head_nouns = set()
    # Walk backwards, first noun(s) from the end (before any prep phrase)
    # Split on prepositions to find main noun phrase
    chunks = []
    current = []
    for w, pos in tagged:
        if pos in ("IN", "TO") and current:
            chunks.append(current)
            current = []
        else:
            current.append((w, pos))
    if current:
        chunks.append(current)
    
    # Main chunk is the first one (before any "for X" / "in X")
    main_chunk = chunks[0] if chunks else tagged
    
    # Head noun = last noun in main chunk
    for w, pos in reversed(main_chunk):
        if pos in NOUN_POS:
            head_nouns.add(w.lower())
            break
    
    # If query is like "knife set", both are head nouns
    # Check if last 2 tokens are both nouns
    nouns_at_end = []
    for w, pos in reversed(main_chunk):
        if pos in NOUN_POS:
            nouns_at_end.append(w.lower())
        else:
            break
    if len(nouns_at_end) >= 2:
        head_nouns.update(nouns_at_end)
    
    return head_nouns


def categorize_token_pos(token_text, query, pos_map, head_nouns):
    """Categorize using POS tags + head noun detection."""
    if token_text.startswith("##"):
        return "subword"
    
    t = token_text.lower()
    
    if t in STOPWORDS or t in ("'", "s", "'s"):
        return "function"
    
    if re.match(r"^\d+$", t):
        return "number"
    
    # Check if it's a head noun
    if t in head_nouns:
        return "head_noun"
    
    # Use POS tag
    pos = pos_map.get(t, "")
    if pos in MODIFIER_POS:
        return "modifier"
    if pos in NOUN_POS:
        # Noun but not head → it's a modifier noun (e.g., "leather" in "leather jacket")
        return "modifier"
    if pos in FUNCTION_POS:
        return "function"
    
    # Default: if it's before the head noun, it's a modifier
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

    SAMPLE_QUERIES = [
        # Attribute-heavy
        "men's black leather jacket",
        "women's red silk dress size 8",
        "blue velvet sofa",
        "large wooden dining table",
        "gold metal wall mirror",
        "boys white cotton shirt",
        "small ceramic flower pot",
        "pink running shoes for girls",
        # Generic
        "laptop stand",
        "wireless headphones",
        "coffee table",
        "phone case",
        "yoga mat",
        "desk lamp",
        "water bottle for gym",
        "birthday gift for mom",
        # Mixed
        "stainless steel kitchen knife set",
        "queen size grey bed sheets",
        "men's waterproof hiking boots brown",
        "organic green tea bags",
    ]

    category_weights = {"head_noun": [], "modifier": [], "function": [],
                        "subword": [], "number": []}
    all_query_data = []

    print("=" * 75)
    print("Token Weight Analysis — Head Nouns vs Modifiers vs Function Words")
    print("=" * 75)

    for query in SAMPLE_QUERIES:
        pos_map = get_query_pos_tags(query)
        head_nouns = find_head_nouns(query)

        enc = tokenizer(query, return_tensors="pt", padding="max_length",
                        truncation=True, max_length=config.query_maxlen).to(device)

        with torch.no_grad():
            Q, q_mask, q_hidden = model.encode(enc.input_ids, enc.attention_mask)
            weights = model.weight_head(q_hidden, q_mask)[0].cpu().numpy()

        tokens = tokenizer.convert_ids_to_tokens(enc.input_ids[0])
        mask = q_mask[0].cpu().numpy()

        print(f"\nQuery: \"{query}\"")
        print(f"  Head nouns: {head_nouns}")
        print(f"  {'Token':<15} {'Category':<12} {'Weight':<8} {'Bar'}")
        print(f"  {'-'*60}")

        query_tokens = []
        for t, w, m in zip(tokens, weights, mask):
            if not m or t in ("[CLS]", "[SEP]", "[PAD]", "[Q]", "[MASK]"):
                continue
            cat = categorize_token_pos(t, query, pos_map, head_nouns)
            bar = "█" * int(w * 200)
            marker = " ◄HEAD" if cat == "head_noun" else ""
            print(f"  {t:<15} {cat:<12} {w:.4f}   {bar}{marker}")

            query_tokens.append({"token": t, "category": cat, "weight": float(w)})
            category_weights.setdefault(cat, []).append(float(w))

        all_query_data.append({"query": query, "head_nouns": list(head_nouns),
                                "tokens": query_tokens})

    # Aggregate
    print(f"\n{'=' * 75}")
    print("Aggregate: Average Weight by Token Role")
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

    # Summary comparison
    mod_w = category_weights.get("modifier", [])
    head_w = category_weights.get("head_noun", [])
    func_w = category_weights.get("function", []) + category_weights.get("subword", [])

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

    # Save
    with open("results/weight_analysis.json", "w") as f:
        json.dump({
            "queries": all_query_data,
            "category_stats": {cat: {"mean": float(np.mean(ws)), "std": float(np.std(ws)),
                                      "count": len(ws)}
                                for cat, ws in category_weights.items() if ws},
        }, f, indent=2)
    print(f"\nSaved to results/weight_analysis.json")


if __name__ == "__main__":
    main()
