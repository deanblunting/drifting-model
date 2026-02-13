"""
Phase 2: Multi-qubit tomography experiments.

Tests the drifting model on:
- 2-qubit Bell states (entangled)
- 3-qubit GHZ and W states
- 4-qubit GHZ state
- Random Haar states (2-4 qubits)

Key questions:
1. Can the drifting model learn entangled state statistics?
2. How does performance scale with qubit count?
3. Does it correctly reconstruct correlations between qubits?
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from quantum_drifting.states import (
    bell_state, ghz_state, w_state, random_haar_state,
    random_product_state, select_random_bases,
)
from quantum_drifting.trainer import train, TrainConfig
from quantum_drifting.tomography import reconstruct_density_matrix, state_fidelity, trace_distance
from quantum_drifting.utils import (
    plot_training, plot_probability_comparison, plot_density_matrix,
)


def run_experiment(state, config, label, out_dir):
    """Run a single tomography experiment and save results."""
    print(f"\n{'='*60}")
    print(f"  {label}: {state}")
    print(f"{'='*60}")
    
    bases = state.all_pauli_bases()
    # For 4+ qubits, 3^n bases is a lot. Use a subset.
    if len(bases) > 27:
        bases = select_random_bases(state.n_qubits, 27, seed=42)
        print(f"  Using {len(bases)}/{3**state.n_qubits} random bases")
    
    result = train(state, bases, config)
    
    safe = label.replace(' ', '_').replace('|', '').replace('>', '')
    plot_training(result, f"{out_dir}/train_{safe}.png")
    plot_probability_comparison(result, f"{out_dir}/probs_{safe}.png",
                                max_bases=min(9, len(bases)))
    
    # Density matrix visualization for 2-qubit states
    if state.n_qubits <= 3:
        gen_probs = result['final_metrics']['gen_probs']
        rho_recon = reconstruct_density_matrix(gen_probs, state.n_qubits, bases)
        
        plot_density_matrix(state.rho, f"True rho — {label}",
                           f"{out_dir}/rho_true_{safe}.png")
        plot_density_matrix(rho_recon, f"Reconstructed rho — {label}",
                           f"{out_dir}/rho_recon_{safe}.png")
        
        fid = state_fidelity(state.rho, rho_recon)
        td = trace_distance(state.rho, rho_recon)
        print(f"  Density matrix fidelity: {fid:.4f}")
        print(f"  Trace distance: {td:.4f}")
    
    return result


def main():
    out_dir = "results/phase2"
    os.makedirs(out_dir, exist_ok=True)
    
    # ---- 2-qubit experiments ----
    config_2q = TrainConfig(
        noise_dim=32, hidden_dim=128,
        n_pos=64, n_neg=64,
        lr=3e-4, n_epochs=1000,
        log_every=200, eval_every=50,
    )
    
    results = {}
    
    # Bell states
    for i, name in enumerate(['Phi+', 'Phi-', 'Psi+', 'Psi-']):
        state = bell_state(i)
        results[name] = run_experiment(state, config_2q, f"Bell_{name}", out_dir)
    
    # Random entangled 2-qubit
    state = random_haar_state(2, seed=42)
    results['Haar_2q'] = run_experiment(state, config_2q, "Haar_2q", out_dir)
    
    # Random product (unentangled) 2-qubit for comparison
    state = random_product_state(2, seed=42)
    results['Product_2q'] = run_experiment(state, config_2q, "Product_2q", out_dir)
    
    # ---- 3-qubit experiments ----
    config_3q = TrainConfig(
        noise_dim=32, hidden_dim=192,
        n_pos=96, n_neg=96,
        lr=2e-4, n_epochs=1500,
        log_every=300, eval_every=100,
    )
    
    state = ghz_state(3)
    results['GHZ_3'] = run_experiment(state, config_3q, "GHZ_3", out_dir)
    
    state = w_state(3)
    results['W_3'] = run_experiment(state, config_3q, "W_3", out_dir)
    
    # ---- 4-qubit experiment (stretch goal) ----
    config_4q = TrainConfig(
        noise_dim=64, hidden_dim=256,
        n_pos=128, n_neg=128,
        lr=2e-4, n_epochs=2000,
        log_every=400, eval_every=100,
        use_transformer=True,
    )
    
    state = ghz_state(4)
    results['GHZ_4'] = run_experiment(state, config_4q, "GHZ_4", out_dir)
    
    # ---- Summary ----
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    print(f"{'State':<16} {'Qubits':>6} {'Bases':>6} {'P_err':>8} {'Fidelity':>10}")
    print("-" * 50)
    for name, r in results.items():
        nq = r['state'].n_qubits
        nb = len(r['bases'])
        pe = r['final_metrics']['prob_error']
        fid = r['final_metrics']['fidelity']
        print(f"{name:<16} {nq:>6} {nb:>6} {pe:>8.4f} {fid:>10.4f}")
    
    print(f"\nResults saved to {out_dir}/")


if __name__ == "__main__":
    main()
