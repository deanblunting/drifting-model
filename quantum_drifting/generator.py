"""
Generator networks for quantum tomography.

Maps (noise, measurement_basis_label) -> outcome distribution representation.

For n qubits with outcome space of dimension 2^n, the generator outputs
a 2^n-dimensional vector. The sign/magnitude at each position encodes
the probability of that outcome.

Two architectures:
- SimpleGenerator: MLP for 1-3 qubits (small outcome space)
- TransformerGenerator: for 4+ qubits (structured outcome space)
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


class TransformerGenerator(nn.Module):
    """
    Transformer-based generator for larger qubit counts (4+ qubits).
    
    Uses a small transformer to process noise tokens conditioned on the
    measurement basis. This is loosely inspired by DiT (Peebles & Xie, 2023)
    used in the original drifting model paper.
    
    The noise is split into tokens, processed by transformer blocks with
    basis conditioning via adaptive layer norm, then projected to the
    output dimension.
    """
    
    def __init__(
        self,
        n_qubits: int,
        noise_dim: int = 64,
        hidden_dim: int = 128,
        n_heads: int = 4,
        n_blocks: int = 4,
        n_bases: int = 9,
        basis_embed_dim: int = 64,
        n_tokens: int = 8,
    ):
        super().__init__()
        self.n_qubits = n_qubits
        self.noise_dim = noise_dim
        self.output_dim = 2 ** n_qubits
        self.n_tokens = n_tokens
        self.hidden_dim = hidden_dim
        
        # Basis embedding
        self.basis_embed = nn.Embedding(n_bases, basis_embed_dim)
        
        # Project noise chunks into token embeddings
        assert noise_dim % n_tokens == 0
        chunk_dim = noise_dim // n_tokens
        self.token_proj = nn.Linear(chunk_dim, hidden_dim)
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.randn(1, n_tokens, hidden_dim) * 0.02)
        
        # Transformer blocks with adaptive layer norm (adaLN)
        self.blocks = nn.ModuleList([
            AdaLNTransformerBlock(hidden_dim, n_heads, basis_embed_dim)
            for _ in range(n_blocks)
        ])
        
        # Condition projection
        self.cond_proj = nn.Sequential(
            nn.Linear(basis_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, basis_embed_dim),
        )
        
        # Output projection
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim * n_tokens, self.output_dim)
        self.output_act = nn.Tanh()
        
        nn.init.zeros_(self.output_proj.bias)
        nn.init.normal_(self.output_proj.weight, std=0.01)
    
    def forward(self, noise: torch.Tensor, basis_idx: torch.Tensor) -> torch.Tensor:
        N = noise.shape[0]
        
        # Condition
        cond = self.basis_embed(basis_idx)  # [N, E]
        cond = self.cond_proj(cond)  # [N, E]
        
        # Tokenize noise
        chunks = noise.view(N, self.n_tokens, -1)  # [N, T, chunk_dim]
        tokens = self.token_proj(chunks) + self.pos_embed  # [N, T, H]
        
        # Transformer blocks
        for block in self.blocks:
            tokens = block(tokens, cond)
        
        # Output
        tokens = self.output_norm(tokens)
        flat = tokens.reshape(N, -1)  # [N, T * H]
        return self.output_act(self.output_proj(flat))


class AdaLNTransformerBlock(nn.Module):
    """Transformer block with adaptive layer norm conditioning."""
    
    def __init__(self, hidden_dim: int, n_heads: int, cond_dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        
        # adaLN: condition -> (scale, shift) for each norm
        self.ada_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, hidden_dim * 4),  # gamma1, beta1, gamma2, beta2
        )
        nn.init.zeros_(self.ada_proj[-1].weight)
        nn.init.zeros_(self.ada_proj[-1].bias)
    
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # Compute adaptive parameters
        params = self.ada_proj(cond)  # [N, 4*H]
        gamma1, beta1, gamma2, beta2 = params.chunk(4, dim=-1)
        
        # Self-attention with adaLN
        h = self.norm1(x)
        h = h * (1 + gamma1.unsqueeze(1)) + beta1.unsqueeze(1)
        h, _ = self.attn(h, h, h)
        x = x + h
        
        # FFN with adaLN
        h = self.norm2(x)
        h = h * (1 + gamma2.unsqueeze(1)) + beta2.unsqueeze(1)
        h = self.ffn(h)
        x = x + h
        
        return x


def make_generator(
    n_qubits: int,
    n_bases: int,
    noise_dim: int = 32,
    hidden_dim: int = 128,
    use_transformer: bool = False,
) -> nn.Module:
    """
    Factory function to create the appropriate generator.
    
    For <= 3 qubits or when use_transformer=False: SimpleGenerator
    For >= 4 qubits with use_transformer=True: TransformerGenerator
    """
    if use_transformer and n_qubits >= 4:
        # Ensure noise_dim divisible by n_tokens
        n_tokens = 8
        noise_dim = max(noise_dim, n_tokens * 8)
        noise_dim = (noise_dim // n_tokens) * n_tokens
        return TransformerGenerator(
            n_qubits=n_qubits,
            noise_dim=noise_dim,
            hidden_dim=hidden_dim,
            n_bases=n_bases,
            n_tokens=n_tokens,
        )
    else:
        return SimpleGenerator(
            n_qubits=n_qubits,
            noise_dim=noise_dim,
            hidden_dim=hidden_dim,
            n_bases=n_bases,
        )
