"""ESCI data loading.

Loads Amazon Shopping Queries Dataset and creates training triples:
  - Query + Exact product (positive)
  - Query + Substitute product (hard negative)

Substitute negatives force the model to learn fine-grained attribute
differences (gender, color, size) rather than trivial product-type distinctions.
Falls back to Irrelevant negatives when no Substitutes are available.
"""
import random
from collections import defaultdict
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader


class ESCITriplesDataset(Dataset):
    """Creates (query, positive, negative) triples from ESCI data."""

    def __init__(self, split="train", locale="us", max_rows=None):
        print(f"Loading ESCI ({split}, locale={locale})...")
        ds = load_dataset("alvations/esci-data-task2", split=split)

        # Filter to locale and group by query
        queries = defaultdict(lambda: {"exact": [], "substitute": [], "irrelevant": []})
        for row in ds:
            if row["product_locale"] != locale:
                continue
            qid = row["query_id"]
            query_text = row["query"]
            title = row["product_title"] or ""
            label = row["esci_label"]

            if label == "E":
                queries[qid]["exact"].append(title)
            elif label == "S":
                queries[qid]["substitute"].append(title)
            elif label == "I":
                queries[qid]["irrelevant"].append(title)
            queries[qid]["query"] = query_text

        # Build triples: exact=positive, substitute=hard negative (fallback to irrelevant)
        self.triples = []
        for qid, data in queries.items():
            if not data["exact"]:
                continue
            # Prefer substitute negatives (hard), fall back to irrelevant (easy)
            neg_pool = data["substitute"] if data["substitute"] else data["irrelevant"]
            if not neg_pool:
                continue
            for pos in data["exact"]:
                neg = random.choice(neg_pool)
                self.triples.append((data["query"], pos, neg))
            if max_rows and len(self.triples) >= max_rows:
                break

        if max_rows:
            self.triples = self.triples[:max_rows]
        print(f"Built {len(self.triples)} triples from {len(queries)} queries.")

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        return self.triples[idx]


class ESCIRerankDataset:
    """Loads full query-product lists for reranking evaluation.

    Each item: (query, [(product_title, label), ...])
    Labels: E=3, S=2, C=1, I=0 (for NDCG computation)
    """
    LABEL_MAP = {"E": 3, "S": 2, "C": 1, "I": 0}

    def __init__(self, split="test", locale="us", max_queries=None):
        print(f"Loading ESCI rerank data ({split}, locale={locale})...")
        ds = load_dataset("alvations/esci-data-task2", split=split)

        queries = defaultdict(lambda: {"query": "", "products": []})
        for row in ds:
            if row["product_locale"] != locale:
                continue
            qid = row["query_id"]
            queries[qid]["query"] = row["query"]
            queries[qid]["products"].append({
                "title": row["product_title"] or "",
                "label": self.LABEL_MAP.get(row["esci_label"], 0),
                "esci_label": row["esci_label"],
            })

        self.data = list(queries.values())
        if max_queries:
            self.data = self.data[:max_queries]
        print(f"Loaded {len(self.data)} queries for reranking.")

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)


def get_collator(tokenizer, query_maxlen=32, doc_maxlen=128):
    """Returns a collator for (query, pos, neg) triples."""
    from colbert_weighted.data import ColBERTCollator
    return ColBERTCollator(tokenizer, query_maxlen, doc_maxlen)


def get_dataloader(config, tokenizer, split="train", max_rows=None):
    ds = ESCITriplesDataset(split=split, locale=config.locale, max_rows=max_rows)
    collator = get_collator(tokenizer, config.query_maxlen, config.doc_maxlen)
    return DataLoader(
        ds, batch_size=config.batch_size, shuffle=(split == "train"),
        collate_fn=collator, num_workers=4, pin_memory=True,
    )
