"""ColBERT model for ESCI: encoder + weight head train together.

Both query and doc encoders train (same dataset, no pretrained asymmetry).
Weight head is optional — disabled for baseline run.
"""
import torch
import torch.nn as nn
from transformers import AutoModel

from .config import ESCIConfig
from colbert_weighted.weight_head import TokenWeightHead
from colbert_weighted.scoring import maxsim, weighted_maxsim


class ColBERTESCI(nn.Module):
    def __init__(self, config: ESCIConfig):
        super().__init__()
        self.config = config
        self.bert = AutoModel.from_pretrained(config.checkpoint)
        self.linear = nn.Linear(self.bert.config.hidden_size, config.dim, bias=False)

        self.weight_head = None
        if config.use_token_weights:
            self.weight_head = TokenWeightHead(
                hidden_dim=self.bert.config.hidden_size,
                norm=config.weight_norm,
            )

    def encode(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state
        emb = self.linear(hidden)
        emb = nn.functional.normalize(emb, p=2, dim=-1)
        return emb, attention_mask.bool(), hidden

    def score(self, Q, D, q_mask, d_mask, q_hidden):
        if self.weight_head is not None:
            weights = self.weight_head(q_hidden, q_mask)
            return weighted_maxsim(Q, D, q_mask, d_mask, weights), weights
        else:
            return maxsim(Q, D, q_mask, d_mask), None

    def forward(self, q_ids, q_mask, d_pos_ids, d_pos_mask, d_neg_ids, d_neg_mask):
        Q, q_m, q_hidden = self.encode(q_ids, q_mask)
        D_pos, d_pos_m, _ = self.encode(d_pos_ids, d_pos_mask)
        D_neg, d_neg_m, _ = self.encode(d_neg_ids, d_neg_mask)

        pos_scores, weights = self.score(Q, D_pos, q_m, d_pos_m, q_hidden)
        neg_scores, _ = self.score(Q, D_neg, q_m, d_neg_m, q_hidden)

        return pos_scores, neg_scores, weights
