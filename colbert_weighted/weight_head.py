"""Query-token weight head for weighted MaxSim."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenWeightHead(nn.Module):
    """Linear head that produces a scalar weight per query token.

    Input:  query hidden states (B, L, D)
    Output: normalized weights  (B, L)
    """

    def __init__(self, hidden_dim: int, norm: str = "softmax"):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, 1)
        self.norm = norm
        nn.init.zeros_(self.linear.bias)
        nn.init.normal_(self.linear.weight, std=0.02)

    def forward(self, hidden_states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: (B, L, D) query encoder hidden states
            mask: (B, L) boolean, True for real tokens
        Returns:
            weights: (B, L) normalized per-token weights
        """
        logits = self.linear(hidden_states).squeeze(-1)  # (B, L)

        if self.norm == "softmax":
            # Masked softmax
            logits = logits.masked_fill(~mask, float("-inf"))
            weights = F.softmax(logits, dim=-1)
        elif self.norm == "sigmoid":
            weights = torch.sigmoid(logits)
            weights = weights * mask.float()
            denom = weights.sum(dim=-1, keepdim=True).clamp(min=1e-9)
            weights = weights / denom
        else:
            raise ValueError(f"Unknown norm: {self.norm}")

        return weights
