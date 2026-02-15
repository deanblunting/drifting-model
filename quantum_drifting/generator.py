"""
Generator network for quantum tomography.

Maps (noise, measurement_basis_label) -> outcome distribution representation.

For n qubits with outcome space of dimension 2^n, the generator outputs
a 2^n-dimensional vector. The sign/magnitude at each position encodes
the probability of that outcome.
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class SimpleGenerator(nn.Module):
    """
    MLP generator for small qubit counts (1-4 qubits).
    
    Input: noise (dim C) + basis embedding (dim E)
    Output: 2^n dimensional vector (one value per measurement outcome)
    
    The output uses tanh activation so values are in [-1, 1].
    Outcome k has probability proportional to how much mass is near +1
    at position k.
    """
    
    def __init__(
        self,
        n_qubits: int,
        noise_dim: int = 32,
        hidden_dim: int = 128,
        n_layers: int = 4,
        n_bases: int = 9,  # 3^n_qubits for default
        basis_embed_dim: int = 32,
    ):
        super().__init__()
        self.n_qubits = n_qubits
        self.noise_dim = noise_dim
        self.output_dim = 2 ** n_qubits
        self.n_bases = n_bases
        
        # Basis embedding: each measurement basis gets a learned embedding
        self.basis_embed = nn.Embedding(n_bases, basis_embed_dim)
        
        # Build MLP
        input_dim = noise_dim + basis_embed_dim
        layers = []
        for i in range(n_layers):
            out_dim = hidden_dim if i < n_layers - 1 else self.output_dim
            layers.append(nn.Linear(input_dim if i == 0 else hidden_dim, out_dim))
            if i < n_layers - 1:
                layers.append(nn.SiLU())
        
        self.net = nn.Sequential(*layers)
        self.output_act = nn.Tanh()
        
        # Initialize last layer with small weights (near zero output initially)
        nn.init.zeros_(self.net[-1].bias)
        nn.init.normal_(self.net[-1].weight, std=0.01)
    
    def forward(
        self,
        noise: torch.Tensor,
        basis_idx: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            noise: [N, noise_dim]
            basis_idx: [N] integer indices into the basis embedding table
        Returns:
            x: [N, 2^n_qubits] output in [-1, 1]
        """
        emb = self.basis_embed(basis_idx)  # [N, E]
        inp = torch.cat([noise, emb], dim=-1)
        return self.output_act(self.net(inp))



def make_generator(
    n_qubits: int,
    n_bases: int,
    noise_dim: int = 32,
    hidden_dim: int = 128,
    n_layers: int = 4,
) -> nn.Module:
    return SimpleGenerator(
        n_qubits=n_qubits,
        noise_dim=noise_dim,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        n_bases=n_bases,
    )
