"""Strategy 2 config: query encoder + weight head train, doc encoder frozen."""
from dataclasses import dataclass
from typing import Literal


@dataclass
class S2Config:
    # Model
    checkpoint: str = "colbert-ir/colbertv2.0"
    dim: int = 128
    query_maxlen: int = 32
    doc_maxlen: int = 180

    # Weight head
    weight_norm: Literal["softmax", "sigmoid"] = "softmax"

    # Training
    encoder_lr: float = 3e-6       # query encoder (BERT + linear)
    weight_head_lr: float = 1e-4   # weight head (randomly initialized)
    epochs: int = 1
    batch_size: int = 32
    accumulation_steps: int = 1
    warmup_steps: int = 500

    # Data
    hf_cache_dir: str = None
    eval_holdout: int = 5000  # last N train queries held out for eval
