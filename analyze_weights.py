#!/usr/bin/env python3
"""Weight distribution analysis: what tokens get high/low weights?

Runs the gap weight head on sample queries and categorizes tokens by type.
Outputs per-query weight bars + aggregate stats by token type.
"""
import torch
import numpy as np
import json
import re
from transformers import AutoTokenizer
from esci.config import ESCIConfig
from esci.model import ColBERTESCI

# Token categories
STOPWORDS = {"the", "a", "an", "for", "with", "in", "of", "to", "and", "or",
             "on", "at", "by", "from", "is", "it", "that", "this", "as", "be"}
GENDER_TERMS = {"men", "mens", "men's", "women", "womens", "women's", "boy",
                "boys", "girl", "girls", "male", "female", "unisex", "ladies"}
COLOR_TERMS = {"black", "white", "red", "blue", "green", "yellow", "pink",
               "purple", "brown", "grey", "gray", "orange", "navy", "beige",
               "gold", "silver"}
SIZE_TERMS = {"small", "medium", "large", "xl", "xxl", "xs", "petite", "plus",
              "tall", "short", "mini", "king", "queen", "twin", "full"}
MATERIAL_TERMS = {"wood", "wooden", "metal", "leather", "velvet", "cotton",
                  "linen", "marble", "glass", "ceramic", "steel", "iron",
                  "brass", "oak", "walnut", "bamboo", "rattan", "wicker",
                  "fabric", "upholstered", "silk", "polyester", "suede"}
NUMBER_PATTERN = re.compile(r"^\d+$")

ATTRIBUTE_TERMS = GENDER_TERMS | COLOR_TERMS | SIZE_TERMS | MATERIAL_TERMS


def categorize_token(token_text):
    """Categorize a token into a type."""
    t = token_text.lower().replace("##", "")
    if t in STOPWORDS:
        return "stopword"
    if t in COLOR_TERMS:
        return "color"
    if t in GENDER_TERMS:
        return "gender"
    if t in SIZE_TERMS:
        return "size"
    if t in MATERIAL_TERMS:
        return "material"
    if NUMBER_PATTERN.match(t):
        return "number"
    if token_text.startswith("##"):
        return "subword"
    return "content"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = ESCIConfig(
        use_token_weights=True,
        weight_norm="softmax",
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

    # Sample queries — mix of attribute-heavy and generic
    SAMPLE_QUERIES = [
        # Attribute-heavy (color, gender, size, material)
        "men's black leather jacket",
        "women's red silk dress size 8",
        "blue velvet sofa",
        "large wooden dining table",
        "gold metal wall mirror",
        "boys white cotton shirt",
        "small ceramic flower pot",
        "pink running shoes for girls",
        # Generic / fewer attributes
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

    # Aggregate stats by category
    category_weights = {}
    all_query_data = []

    print("=" * 70)
    print("Token Weight Distribution Analysis (Gap model)")
    print("=" * 70)

    for query in SAMPLE_QUERIES:
        enc = tokenizer(query, return_tensors="pt", padding="max_length",
                        truncation=True, max_length=config.query_maxlen).to(device)

        with torch.no_grad():
            Q, q_mask, q_hidden = model.encode(enc.input_ids, enc.attention_mask)
            weights = model.weight_head(q_hidden, q_mask)[0].cpu().numpy()

        tokens = tokenizer.convert_ids_to_tokens(enc.input_ids[0])
        mask = q_mask[0].cpu().numpy()

        print(f"\nQuery: \"{query}\"")
        print(f"  {'Token':<15} {'Category':<12} {'Weight':<8} {'Bar'}")
        print(f"  {'-'*55}")

        query_tokens = []
        for t, w, m in zip(tokens, weights, mask):
            if not m or t in ("[CLS]", "[SEP]", "[PAD]", "[Q]", "[MASK]"):
                continue
            cat = categorize_token(t)
            bar = "█" * int(w * 200)  # scale for visibility
            print(f"  {t:<15} {cat:<12} {w:.4f}   {bar}")

            query_tokens.append({"token": t, "category": cat, "weight": float(w)})

            if cat not in category_weights:
                category_weights[cat] = []
            category_weights[cat].append(float(w))

        all_query_data.append({"query": query, "tokens": query_tokens})

    # Aggregate analysis
    print(f"\n{'=' * 70}")
    print("Aggregate: Average Weight by Token Category")
    print(f"{'=' * 70}")
    print(f"\n  {'Category':<15} {'Avg Weight':<12} {'Count':<8} {'Std':<8} {'Bar'}")
    print(f"  {'-'*55}")

    sorted_cats = sorted(category_weights.items(),
                         key=lambda x: np.mean(x[1]), reverse=True)
    for cat, ws in sorted_cats:
        avg = np.mean(ws)
        std = np.std(ws)
        bar = "█" * int(avg * 200)
        print(f"  {cat:<15} {avg:.4f}       {len(ws):<8} {std:.4f}   {bar}")

    # Attribute vs non-attribute
    attr_weights = []
    non_attr_weights = []
    for cat, ws in category_weights.items():
        if cat in ("color", "gender", "size", "material"):
            attr_weights.extend(ws)
        elif cat in ("stopword", "subword"):
            non_attr_weights.extend(ws)

    print(f"\n{'=' * 70}")
    print("Attribute vs Function Token Weights")
    print(f"{'=' * 70}")
    if attr_weights:
        print(f"  Attribute tokens (color/gender/size/material): {np.mean(attr_weights):.4f} avg  (n={len(attr_weights)})")
    if non_attr_weights:
        print(f"  Function tokens (stopwords/subwords):          {np.mean(non_attr_weights):.4f} avg  (n={len(non_attr_weights)})")
    if attr_weights and non_attr_weights:
        ratio = np.mean(attr_weights) / np.mean(non_attr_weights)
        print(f"  Ratio (attribute / function):                  {ratio:.1f}x")

    # Save for plotting
    with open("results/weight_analysis.json", "w") as f:
        json.dump({
            "queries": all_query_data,
            "category_stats": {cat: {"mean": float(np.mean(ws)), "std": float(np.std(ws)),
                                      "count": len(ws)}
                                for cat, ws in category_weights.items()},
        }, f, indent=2)
    print(f"\nSaved to results/weight_analysis.json")


if __name__ == "__main__":
    main()
