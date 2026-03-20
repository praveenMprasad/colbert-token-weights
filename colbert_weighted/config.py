"""Experiment configuration."""
from dataclasses import dataclass
from typing import Literal


@dataclass
class ExpConfig:
    # Model
    checkpoint: str = "colbert-ir/colbertv2.0"
    dim: int = 128  # ColBERT embedding dim
    query_maxlen: int = 32
    doc_maxlen: int = 180

    # Weight head
    use_token_weights: bool = True
    weight_norm: Literal["softmax", "sigmoid"] = "softmax"

    # Training — only the weight head trains
    weight_head_lr: float = 1e-4
    epochs: int = 1
    batch_size: int = 32
    accumulation_steps: int = 1
    warmup_steps: int = 500

    # Data (all from HuggingFace — no local files needed)
    # Tevatron/msmarco-passage: train=401k queries, dev=6.98k queries
    hf_cache_dir: str = None

    # Eval
    top_k_eval: list = None

    def __post_init__(self):
        if self.top_k_eval is None:
            self.top_k_eval = [10, 50, 100, 200, 1000]
