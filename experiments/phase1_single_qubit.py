"""
Phase 1: Single-qubit tomography experiments.
Verifies the drifting model framework on the simplest quantum system.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from quantum_drifting.states import QuantumState, random_haar_state
from quantum_drifting.trainer import train, TrainConfig
from quantum_drifting.utils import plot_training, plot_probability_comparison


def single_qubit_state(theta: float, phi: float, name: str = "") -> QuantumState:
    """Create a single qubit state from Bloch sphere angles."""
    sv = np.array([np.cos(theta/2), np.exp(1j*phi) * np.sin(theta/2)])
    return QuantumState(sv, n_qubits=1, name=name)


def main():
    os.makedirs("results/phase1", exist_ok=True)
    
    config = TrainConfig(
        noise_dim=16, hidden_dim=64,
        n_pos=64, n_neg=64,
        lr=3e-4, n_epochs=600,
        log_every=200, eval_every=50,
    )
    
    states = [
        single_qubit_state(np.pi/2, 0, "|+>"),
        single_qubit_state(np.pi/3, np.pi/4, "arbitrary"),
        single_qubit_state(0.3, 0, "near |0>"),
    ]
    
    for state in states:
        print(f"\n{'='*60}")
        print(f"  {state.name}")
        print(f"{'='*60}")
        
        bases = state.all_pauli_bases()  # ['X', 'Y', 'Z']
        result = train(state, bases, config)
        
        safe = state.name.replace(' ', '_').replace('|', '').replace('>', '')
        plot_training(result, f"results/phase1/train_{safe}.png")
        plot_probability_comparison(result, f"results/phase1/probs_{safe}.png")
    
    print("\nPhase 1 complete. Results in results/phase1/")


if __name__ == "__main__":
    main()
