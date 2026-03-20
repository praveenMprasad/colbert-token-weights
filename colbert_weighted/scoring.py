"""Scoring functions: vanilla MaxSim and weighted MaxSim."""
import torch


def maxsim(Q: torch.Tensor, D: torch.Tensor, q_mask: torch.Tensor, d_mask: torch.Tensor) -> torch.Tensor:
    """Standard ColBERT MaxSim.

    Args:
        Q: (B, Lq, dim) query embeddings
        D: (B, Ld, dim) doc embeddings
        q_mask: (B, Lq) query token mask
        d_mask: (B, Ld) doc token mask
    Returns:
        scores: (B,)
    """
    # (B, Lq, Ld)
    sim = torch.bmm(Q, D.transpose(1, 2))
    # mask out pad doc tokens
    sim = sim.masked_fill(~d_mask.unsqueeze(1), float("-inf"))
    # max over doc tokens per query token
    max_sim, _ = sim.max(dim=-1)  # (B, Lq)
    # mask out pad query tokens and sum
    max_sim = max_sim * q_mask.float()
    return max_sim.sum(dim=-1)  # (B,)


def weighted_maxsim(
    Q: torch.Tensor,
    D: torch.Tensor,
    q_mask: torch.Tensor,
    d_mask: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Weighted MaxSim: Score = sum_i w_i * max_j sim(q_i, d_j).

    Args:
        Q, D, q_mask, d_mask: same as maxsim
        weights: (B, Lq) normalized query-token weights
    Returns:
        scores: (B,)
    """
    sim = torch.bmm(Q, D.transpose(1, 2))
    sim = sim.masked_fill(~d_mask.unsqueeze(1), float("-inf"))
    max_sim, _ = sim.max(dim=-1)  # (B, Lq)
    # weight and sum (weights already handle masking via normalization)
    max_sim = max_sim * weights
    return max_sim.sum(dim=-1)  # (B,)
