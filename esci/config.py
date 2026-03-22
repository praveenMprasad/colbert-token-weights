"""ESCI experiment config."""
from dataclasses import dataclass
from typing import Literal


@dataclass
class ESCIConfig:
    # Model
    checkpoint: str = "colbert-ir/colbertv2.0"
    dim: int = 128
    query_maxlen: int = 32
    doc_maxlen: int = 128  # product titles are shorter than passages

    # Weight head
    use_token_weights: bool = True
    weight_norm: Literal["softmax", "sigmoid"] = "softmax"

    # Training
    encoder_lr: float = 2e-5
    weight_head_lr: float = 1e-3
    epochs: int = 3
    batch_size: int = 64
    accumulation_steps: int = 1
    warmup_ratio: float = 0.1

    # Data
    dataset: str = "alvations/esci-data-task2"
    locale: str = "us"  # English only
