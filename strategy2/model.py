"""Strategy 2: Query encoder + weight head train together, doc encoder frozen.

The query encoder adapts its representations so the weight head has richer
signals. The doc encoder stays frozen — no re-indexing needed.
"""
import torch
import torch.nn as nn
from transformers import AutoModel

from .config import S2Config
from colbert_weighted.weight_head import TokenWeightHead
from colbert_weighted.scoring import maxsim, weighted_maxsim


class ColBERTWeightedS2(nn.Module):
    def __init__(self, config: S2Config):
        super().__init__()
        self.config = config

        self.bert = AutoModel.from_pretrained(config.checkpoint)
        self.linear = nn.Linear(self.bert.config.hidden_size, config.dim, bias=False)

        # Weight head — trainable
        self.weight_head = TokenWeightHead(
            hidden_dim=self.bert.config.hidden_size,
            norm=config.weight_norm,
        )

        # Doc encoder uses same weights but frozen at forward time
        # Query encoder (bert + linear) is trainable

    def encode_query(self, input_ids, attention_mask):
        """Query encoding — gradients flow through."""
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state
        emb = self.linear(hidden)
        emb = nn.functional.normalize(emb, p=2, dim=-1)
        return emb, attention_mask.bool(), hidden

    @torch.no_grad()
    def encode_doc(self, input_ids, attention_mask):
        """Doc encoding — fully frozen, no gradients."""
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state
        emb = self.linear(hidden)
        emb = nn.functional.normalize(emb, p=2, dim=-1)
        return emb.detach(), attention_mask.bool()

    def score(self, Q, D, q_mask, d_mask, q_hidden):
        weights = self.weight_head(q_hidden, q_mask)
        return weighted_maxsim(Q, D, q_mask, d_mask, weights), weights

    def forward(self, q_ids, q_mask, d_pos_ids, d_pos_mask, d_neg_ids, d_neg_mask):
        """Training forward.

        Query side: full gradients (encoder + weight head).
        Doc side: frozen (no gradients).
        """
        Q, q_m, q_hidden = self.encode_query(q_ids, q_mask)
        D_pos, d_pos_m = self.encode_doc(d_pos_ids, d_pos_mask)
        D_neg, d_neg_m = self.encode_doc(d_neg_ids, d_neg_mask)

        pos_scores, weights = self.score(Q, D_pos, q_m, d_pos_m, q_hidden)
        neg_scores, _ = self.score(Q, D_neg, q_m, d_neg_m, q_hidden)

        return pos_scores, neg_scores, weights
