"""Data loading from HuggingFace: Tevatron/msmarco-passage.

ColBERTv2 was trained on MS MARCO Passage Ranking (NAACL'22 paper).
We use the same underlying dataset via HuggingFace for reproducibility.

Training: Tevatron/msmarco-passage (train split) — 401k queries with
          positive and negative passages (BM25 negatives).
Dev eval: Tevatron/msmarco-passage (dev split) — 6.98k queries.
Corpus:   Tevatron/msmarco-passage-corpus — 8.84M passages.

Uses streaming mode to avoid downloading the full dataset to disk.
"""
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from datasets import load_dataset
from transformers import AutoTokenizer


class MSMARCOTriplesDataset(Dataset):
    """Loads triples via streaming, materializes only what's needed."""

    def __init__(self, split: str = "train", max_rows: int = None, cache_dir: str = None):
        self.triples = []
        ds = load_dataset("Tevatron/msmarco-passage", split=split, streaming=True)
        for i, row in enumerate(ds):
            if max_rows is not None and i >= max_rows:
                break
            query = row["query"]
            pos = row["positive_passages"][0]["text"]
            negs = row.get("negative_passages", [])
            neg = negs[0]["text"] if negs else pos
            self.triples.append((query, pos, neg))

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        return self.triples[idx]


class ColBERTCollator:
    """Tokenizes (query, pos_doc, neg_doc) triples for ColBERT."""

    def __init__(self, tokenizer, query_maxlen=32, doc_maxlen=180):
        self.tokenizer = tokenizer
        self.query_maxlen = query_maxlen
        self.doc_maxlen = doc_maxlen

    def __call__(self, batch):
        queries, pos_docs, neg_docs = zip(*batch)

        q_enc = self.tokenizer(
            list(queries), padding="max_length", truncation=True,
            max_length=self.query_maxlen, return_tensors="pt",
        )
        p_enc = self.tokenizer(
            list(pos_docs), padding="max_length", truncation=True,
            max_length=self.doc_maxlen, return_tensors="pt",
        )
        n_enc = self.tokenizer(
            list(neg_docs), padding="max_length", truncation=True,
            max_length=self.doc_maxlen, return_tensors="pt",
        )
        return {
            "q_ids": q_enc.input_ids, "q_mask": q_enc.attention_mask,
            "d_pos_ids": p_enc.input_ids, "d_pos_mask": p_enc.attention_mask,
            "d_neg_ids": n_enc.input_ids, "d_neg_mask": n_enc.attention_mask,
        }


def get_dataloader(config, tokenizer, split="train", max_rows=None):
    """Create a DataLoader for MS MARCO passage ranking triples.

    Streams from HuggingFace — no full dataset download required.
    """
    dataset = MSMARCOTriplesDataset(split=split, max_rows=max_rows)
    collator = ColBERTCollator(tokenizer, config.query_maxlen, config.doc_maxlen)
    return DataLoader(
        dataset, batch_size=config.batch_size, shuffle=(split == "train"),
        collate_fn=collator, num_workers=0, pin_memory=True,
    )


def load_dev_qrels(cache_dir=None):
    """Load dev qrels from the Tevatron dataset for evaluation.

    Returns:
        dict: {query_id: {doc_id: relevance}}
    """
    ds = load_dataset("Tevatron/msmarco-passage", split="dev", streaming=True)
    qrels = {}
    for row in ds:
        qid = str(row["query_id"])
        qrels[qid] = {}
        for p in row["positive_passages"]:
            qrels[qid][str(p["docid"])] = 1
    return qrels
