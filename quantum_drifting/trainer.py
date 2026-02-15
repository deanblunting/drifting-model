"""
Training loop for quantum tomography via drifting models.

Handles:
- Measurement data management (positive samples per basis)
- Per-basis drifting loss computation
- Multi-temperature drift aggregation
- Evaluation: probability error, fidelity, Bloch vector reconstruction
- Logging and checkpointing
"""

import torch
import torch.optim as optim
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from tqdm import tqdm

from .states import QuantumState
from .generator import make_generator
from .drifting import compute_multi_temperature_drift, drifting_loss
from .tomography import reconstruct_density_matrix, state_fidelity


@dataclass
class TrainConfig:
    """Training configuration."""
    # Generator
    noise_dim: int = 32
    hidden_dim: int = 128
    n_layers: int = 4
    # Drifting
    n_pos: int = 64          # positive samples per basis per step
    n_neg: int = 64          # generated (negative) samples per basis per step
    temperatures: List[float] = field(default_factory=lambda: [0.05, 0.15, 0.4])
    normalize_drift: bool = True
    
    # Optimization
    lr: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 2.0
    n_epochs: int = 800
    
    # Data
    n_shots: int = 5000      # measurement shots per basis
    
    # Logging
    log_every: int = 100
    eval_every: int = 50
    
    # Device
    device: str = "auto"     # "auto", "mps", "cuda", "cpu"


def get_device(config: TrainConfig) -> torch.device:
    """Select best available device."""
    if config.device != "auto":
        return torch.device(config.device)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class MeasurementDataStore:
    """
    Stores pre-collected measurement data for all bases.
    Provides efficient sampling of positive examples.
    """
    
    def __init__(self, state: QuantumState, bases: List[str], 
                 n_shots: int, device: torch.device):
        self.bases = bases
        self.n_bases = len(bases)
        self.basis_to_idx = {b: i for i, b in enumerate(bases)}
        self.dim = state.dim  # 2^n
        self.device = device
        
        # Collect measurement data
        self.data = {}       # basis -> [n_shots, dim] encoded outcomes
        self.true_probs = {} # basis -> [dim] true probabilities
        
        for basis in bases:
            encoded = state.sample_encoded(basis, n_shots)
            self.data[basis] = torch.tensor(encoded, dtype=torch.float32, device=device)
            self.true_probs[basis] = state.measure_probs(basis)
    
    def sample_positive(self, basis: str, n: int) -> torch.Tensor:
        """Sample n positive examples for a given basis."""
        data = self.data[basis]
        idx = torch.randint(0, data.shape[0], (n,), device=self.device)
        return data[idx]


def train(
    state: QuantumState,
    bases: List[str],
    config: TrainConfig = None,
    verbose: bool = True,
) -> Dict:
    """
    Train a Drifting Model for quantum state tomography.
    
    Args:
        state: the quantum state to reconstruct
        bases: list of measurement basis strings (e.g., ["XX", "XY", ...])
        config: training configuration
        verbose: print progress
    
    Returns:
        Dictionary with trained generator, history, and evaluation results
    """
    if config is None:
        config = TrainConfig()
    
    device = get_device(config)
    
    if verbose:
        print(f"State: {state}")
        print(f"Qubits: {state.n_qubits}, Outcome dim: {state.dim}")
        print(f"Measurement bases: {len(bases)}")
        print(f"Device: {device}")
    
    # Prepare data
    data_store = MeasurementDataStore(state, bases, config.n_shots, device)
    
    # Create generator
    generator = make_generator(
        n_qubits=state.n_qubits,
        n_bases=len(bases),
        noise_dim=config.noise_dim,
        hidden_dim=config.hidden_dim,
        n_layers=config.n_layers,
    ).to(device)
    
    n_params = sum(p.numel() for p in generator.parameters())
    if verbose:
        print(f"Generator params: {n_params:,}")
        print("-" * 60)
    
    # Optimizer
    optimizer = optim.AdamW(
        generator.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95),
    )
    
    # Training history
    history = {
        'loss': [], 'drift_norm': [],
        'prob_error': [], 'fidelity': [],
    }
    
    # Training loop
    iterator = range(config.n_epochs)
    if verbose:
        iterator = tqdm(iterator, desc="Training", leave=True)
    
    for epoch in iterator:
        epoch_loss = 0.0
        epoch_drift = 0.0
        
        # Shuffle basis order each epoch
        basis_order = np.random.permutation(len(bases))
        
        for bi in basis_order:
            basis = bases[bi]
            basis_idx_tensor = torch.full(
                (config.n_neg,), bi, dtype=torch.long, device=device
            )
            
            # Generate samples
            noise = torch.randn(config.n_neg, config.noise_dim, device=device)
            x = generator(noise, basis_idx_tensor)  # [N_neg, 2^n]
            
            # Positive samples (real measurement data)
            y_pos = data_store.sample_positive(basis, config.n_pos)  # [N_pos, 2^n]
            
            # Negative samples = generated samples (detached for V computation)
            y_neg = x.detach()
            
            # Compute drifting field
            V = compute_multi_temperature_drift(
                x, y_pos, y_neg,
                temperatures=config.temperatures,
                normalize_drift=config.normalize_drift,
            )
            
            # Compute loss
            loss = drifting_loss(x, V)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), config.grad_clip)
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_drift += torch.mean(V ** 2).item()
        
        avg_loss = epoch_loss / len(bases)
        avg_drift = epoch_drift / len(bases)
        history['loss'].append(avg_loss)
        history['drift_norm'].append(avg_drift)
        
        # Evaluate periodically
        if epoch % config.eval_every == 0 or epoch == config.n_epochs - 1:
            metrics = evaluate(generator, state, data_store, config, device)
            history['prob_error'].append(metrics['prob_error'])
            history['fidelity'].append(metrics['fidelity'])
            
            if verbose and (epoch % config.log_every == 0 or epoch == config.n_epochs - 1):
                tqdm.write(
                    f"Ep {epoch:4d} | Loss: {avg_loss:.5f} | ||V||²: {avg_drift:.5f} | "
                    f"P_err: {metrics['prob_error']:.4f} | Fidelity: {metrics['fidelity']:.4f}"
                )
        else:
            # Fill with last value for smooth plotting
            if history['prob_error']:
                history['prob_error'].append(history['prob_error'][-1])
                history['fidelity'].append(history['fidelity'][-1])
            else:
                history['prob_error'].append(1.0)
                history['fidelity'].append(0.0)
    
    # Final evaluation with more samples
    final_metrics = evaluate(generator, state, data_store, config, device, n_eval=5000)
    
    if verbose:
        print(f"\nFinal probability error: {final_metrics['prob_error']:.4f}")
        print(f"Final fidelity: {final_metrics['fidelity']:.4f}")
    
    return {
        'generator': generator,
        'data_store': data_store,
        'history': history,
        'config': config,
        'state': state,
        'bases': bases,
        'final_metrics': final_metrics,
        'device': device,
    }


@torch.no_grad()
def evaluate(
    generator: torch.nn.Module,
    state: QuantumState,
    data_store: MeasurementDataStore,
    config: TrainConfig,
    device: torch.device,
    n_eval: int = 3000,
) -> Dict:
    """
    Evaluate the generator by sampling and comparing distributions.
    
    Returns:
        Dictionary with prob_error, fidelity, and per-basis generated probs
    """
    generator.eval()
    
    gen_probs = {}
    for bi, basis in enumerate(data_store.bases):
        basis_idx = torch.full((n_eval,), bi, dtype=torch.long, device=device)
        noise = torch.randn(n_eval, config.noise_dim, device=device)
        x = generator(noise, basis_idx)  # [n_eval, 2^n]
        
        # Convert to probabilities: which outcome has the max value per sample
        outcomes = torch.argmax(x, dim=1)  # [n_eval]
        counts = torch.zeros(state.dim, device=device)
        for k in range(state.dim):
            counts[k] = (outcomes == k).float().sum()
        gen_probs[basis] = (counts / n_eval).cpu().numpy()
    
    # Average probability error across bases
    prob_error = 0.0
    for basis in data_store.bases:
        true_p = data_store.true_probs[basis]
        gen_p = gen_probs[basis]
        prob_error += np.sum(np.abs(true_p - gen_p))
    prob_error /= len(data_store.bases)
    
    # Reconstruct density matrix and compute fidelity
    try:
        rho_recon = reconstruct_density_matrix(
            gen_probs, state.n_qubits, data_store.bases
        )
        fid = state_fidelity(state.rho, rho_recon)
    except Exception:
        fid = 0.0
    
    generator.train()
    
    return {
        'prob_error': prob_error,
        'fidelity': fid,
        'gen_probs': gen_probs,
    }
