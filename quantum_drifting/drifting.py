"""
Drifting field computation (Algorithm 2 from the paper).

The drifting field V governs how generated samples should move:
- Attracted toward positive samples (real measurement data)
- Repelled from negative samples (other generated data)
- V = 0 at equilibrium when generated distribution matches data

This module implements the kernel-based drifting field with:
- Multi-temperature support
- Drift normalization
- Feature normalization
"""

import torch
from typing import List, Optional


def compute_drifting_field(
    x: torch.Tensor,
    y_pos: torch.Tensor,
    y_neg: torch.Tensor,
    temperature: float,
    mask_self: bool = True
) -> torch.Tensor:
    """
    Compute the drifting field V following Algorithm 2.
    
    For each generated sample x_i:
      V(x_i) = sum_j,k  A(x_i, y_pos_j) * A(x_i, y_neg_k) * (y_pos_j - y_neg_k)
    
    where A is a doubly-normalized kernel (softmax over both x and y axes).
    
    Args:
        x: [N, D] generated samples
        y_pos: [N_pos, D] positive samples (real data)
        y_neg: [N_neg, D] negative samples (generated data)
        temperature: kernel bandwidth parameter tau
        mask_self: if True and N == N_neg, mask diagonal (self-pairs)
    
    Returns:
        V: [N, D] drifting field vectors
    """
    N, D = x.shape
    N_pos = y_pos.shape[0]
    N_neg = y_neg.shape[0]
    
    # Pairwise L2 distances
    dist_pos = torch.cdist(x, y_pos, p=2)  # [N, N_pos]
    dist_neg = torch.cdist(x, y_neg, p=2)  # [N, N_neg]
    
    # Self-masking: ignore x_i vs x_i when negatives come from x
    if mask_self and N == N_neg:
        dist_neg = dist_neg + torch.eye(N, device=x.device) * 1e6
    
    # Logits: -distance / temperature (Eq. 12)
    logit_pos = -dist_pos / temperature
    logit_neg = -dist_neg / temperature
    logit = torch.cat([logit_pos, logit_neg], dim=1)  # [N, N_pos + N_neg]
    
    # Double softmax normalization (Algorithm 2)
    A_row = torch.softmax(logit, dim=1)   # over y (columns)
    A_col = torch.softmax(logit, dim=0)   # over x (rows)
    A = torch.sqrt(A_row * A_col + 1e-12)
    
    # Split back
    A_pos = A[:, :N_pos]  # [N, N_pos]
    A_neg = A[:, N_pos:]  # [N, N_neg]
    
    # Weights (from Eq. 11)
    W_pos = A_pos * A_neg.sum(dim=1, keepdim=True)
    W_neg = A_neg * A_pos.sum(dim=1, keepdim=True)
    
    # Drift
    drift_pos = W_pos @ y_pos  # [N, D]
    drift_neg = W_neg @ y_neg  # [N, D]
    V = drift_pos - drift_neg
    
    return V


def compute_multi_temperature_drift(
    x: torch.Tensor,
    y_pos: torch.Tensor,
    y_neg: torch.Tensor,
    temperatures: List[float],
    normalize_drift: bool = True,
    mask_self: bool = True
) -> torch.Tensor:
    """
    Compute drifting field with multiple temperatures (Sec. A.6).
    
    Using multiple temperatures improves robustness across different
    feature scales and similarity structures. Each temperature captures
    a different notion of "nearby".
    
    Args:
        x: [N, D] generated samples
        y_pos: [N_pos, D] positive samples
        y_neg: [N_neg, D] negative samples
        temperatures: list of temperature values
        normalize_drift: if True, normalize each V_tau to unit RMS (Sec. A.6)
        mask_self: if True, mask self-pairs
    
    Returns:
        V_total: [N, D] aggregated drifting field
    """
    V_total = torch.zeros_like(x)
    
    for tau in temperatures:
        V = compute_drifting_field(x, y_pos, y_neg, tau, mask_self=mask_self)
        
        if normalize_drift:
            # Drift normalization (Eq. 23-25)
            drift_rms = torch.sqrt(torch.mean(V ** 2) + 1e-8)
            V = V / drift_rms
        
        V_total = V_total + V
    
    return V_total


def drifting_loss(
    x: torch.Tensor,
    V: torch.Tensor
) -> torch.Tensor:
    """
    Compute the drifting loss (Eq. 6).
    
    L = E[||x - stopgrad(x + V)||^2]
    
    The stop-gradient means V acts as a frozen target. The loss
    moves x toward x + V, effectively applying the drift.
    
    Args:
        x: [N, D] generator output (requires grad)
        V: [N, D] drifting field (will be detached)
    
    Returns:
        Scalar loss value
    """
    target = (x + V).detach()  # frozen target (stop-gradient)
    loss = torch.mean((x - target) ** 2)
    return loss
