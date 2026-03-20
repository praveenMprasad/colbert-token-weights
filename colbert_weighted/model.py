"""ColBERT model with frozen encoder + trainable token weight head.

Strategy 1: The entire ColBERTv2 encoder (BERT + linear projection) is frozen.
Only the weight head (129 params) is trainable. This isolates the question:
do learned query-token weights improve retrieval on fixed representations?
"""
import torch
import torch.nn as nn
from transformers import AutoModel

from .config import ExpConfig
from .weight_head import TokenWeightHead
from .scoring import maxsim, weighted_maxsim


class ColBERTWeighted(nn.Module):
    def __init__(self, config: ExpConfig):
        super().__init__()
        self.config = config

        # Encoder — loaded from ColBERTv2, fully frozen
        self.bert = AutoModel.from_pretrained(config.checkpoint)
        self.linear = nn.Linear(self.bert.config.hidden_size, config.dim, bias=False)

        # Freeze entire encoder
        for param in self.bert.parameters():
            param.requires_grad = False
        for param in self.linear.parameters():
            param.requires_grad = False

        # Weight head — the only trainable component
        self.weight_head = None
        if config.use_token_weights:
            self.weight_head = TokenWeightHead(
                hidden_dim=self.bert.config.hidden_size,
                norm=config.weight_norm,
            )

    @torch.no_grad()
    def encode(self, input_ids, attention_mask):
        """Encode tokens through frozen BERT + linear. Returns (emb, mask, hidden)."""
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state  # (B, L, H)
        emb = self.linear(hidden)  # (B, L, dim)
        emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
        return emb, attention_mask.bool(), hidden

    def score(self, Q, D, q_mask, d_mask, q_hidden):
        """Compute relevance scores with optional token weights."""
        if self.weight_head is not None:
            weights = self.weight_head(q_hidden, q_mask)
            return weighted_maxsim(Q, D, q_mask, d_mask, weights), weights
        else:
            return maxsim(Q, D, q_mask, d_mask), None

    def forward(self, q_ids, q_mask, d_pos_ids, d_pos_mask, d_neg_ids, d_neg_mask):
        """Training forward: returns pos_scores, neg_scores, weights.

        Encoder is frozen (no_grad). Only weight head gets gradients.
        """
        # All encoding is no_grad — frozen
        Q, q_m, q_hidden = self.encode(q_ids, q_mask)
        D_pos, d_pos_m, _ = self.encode(d_pos_ids, d_pos_mask)
        D_neg, d_neg_m, _ = self.encode(d_neg_ids, d_neg_mask)

        # Detach embeddings but keep q_hidden connected for weight head grad
        Q = Q.detach()
        D_pos = D_pos.detach()
        D_neg = D_neg.detach()

        # Re-enable grad for q_hidden so weight head can backprop
        q_hidden = q_hidden.detach().requires_grad_(True)

        pos_scores, weights = self.score(Q, D_pos, q_m, d_pos_m, q_hidden)
        neg_scores, _ = self.score(Q, D_neg, q_m, d_neg_m, q_hidden)

        return pos_scores, neg_scores, weights
