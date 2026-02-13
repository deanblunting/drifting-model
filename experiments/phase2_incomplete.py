"""
Phase 2: Incomplete tomography experiments.

The key research question: can a drifting model reconstruct quantum state
statistics from FEWER measurement bases than standard tomography requires?

Full tomography of n qubits needs 3^n measurement settings.
We test reconstruction quality with 25%, 50%, 75%, and 100% of bases.

If the drifting model can generalize (learn the underlying state structure
rather than memorizing individual basis statistics), it should degrade
gracefully with fewer measurements — potentially outperforming simple
frequency estimation for incomplete data.

This is where the approach could offer genuine value over standard methods.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from quantum_drifting.states import (
    bell_state, ghz_state, w_state, random_haar_state,
    select_random_bases, select_local_bases,
)
from quantum_drifting.trainer import train, TrainConfig
from quantum_drifting.tomography import reconstruct_density_matrix, state_fidelity
from quantum_drifting.utils import plot_training, plot_incomplete_comparison


def run_incomplete_sweep(state, config, fractions, out_dir, label):
    """
    Train with varying fractions of the complete measurement set.
    """
    all_bases = state.all_pauli_bases()
    n_total = len(all_bases)
    
    results = {}
    
    for frac in fractions:
        n_bases = max(1, int(frac * n_total))
        if n_bases >= n_total:
            bases = all_bases
            tag = f"100%({n_total})"
        else:
            bases = select_random_bases(state.n_qubits, n_bases, seed=42)
            tag = f"{int(frac*100)}%({n_bases})"
        
        print(f"\n--- {label} with {tag} bases ---")
        result = train(state, bases, config, verbose=True)
        results[tag] = result
        
        safe = f"{label}_{tag}".replace(' ', '_').replace('|', '').replace('>', '').replace('%', 'pct').replace('(', '_').replace(')', '')
        plot_training(result, f"{out_dir}/train_{safe}.png")
    
    # Also test with only local bases (3n instead of 3^n)
    local_bases = select_local_bases(state.n_qubits)
    n_local = len(local_bases)
    print(f"\n--- {label} with LOCAL bases ({n_local}) ---")
    result = train(state, local_bases, config, verbose=True)
    tag = f"local({n_local})"
    results[tag] = result
    
    safe_label = label.replace(' ', '_').replace('|', '').replace('>', '')
    plot_training(result, f"{out_dir}/train_{safe_label}_local.png")
    
    return results


def main():
    out_dir = "results/phase2_incomplete"
    os.makedirs(out_dir, exist_ok=True)
    
    fractions = [0.25, 0.50, 0.75, 1.0]
    
    # ---- 2-qubit Bell state ----
    config_2q = TrainConfig(
        noise_dim=32, hidden_dim=128,
        n_pos=64, n_neg=64,
        lr=3e-4, n_epochs=800,
        log_every=200, eval_every=50,
    )
    
    state = bell_state(0)  # |Phi+>
    results_bell = run_incomplete_sweep(state, config_2q, fractions, out_dir, "Bell_PhiPlus")
    plot_incomplete_comparison(results_bell, f"{out_dir}/completeness_Bell.png")
    
    # ---- 3-qubit GHZ ----
    config_3q = TrainConfig(
        noise_dim=32, hidden_dim=192,
        n_pos=96, n_neg=96,
        lr=2e-4, n_epochs=1200,
        log_every=300, eval_every=100,
    )
    
    state = ghz_state(3)
    results_ghz = run_incomplete_sweep(state, config_3q, fractions, out_dir, "GHZ_3")
    plot_incomplete_comparison(results_ghz, f"{out_dir}/completeness_GHZ3.png")
    
    # ---- 3-qubit random Haar ----
    state = random_haar_state(3, seed=123)
    results_haar = run_incomplete_sweep(state, config_3q, fractions, out_dir, "Haar_3q")
    plot_incomplete_comparison(results_haar, f"{out_dir}/completeness_Haar3.png")
    
    # ---- Summary ----
    print(f"\n{'='*70}")
    print("  INCOMPLETE TOMOGRAPHY SUMMARY")
    print(f"{'='*70}")
    
    for exp_name, results in [("Bell |Phi+>", results_bell), 
                               ("GHZ(3)", results_ghz),
                               ("Haar(3)", results_haar)]:
        print(f"\n  {exp_name}:")
        print(f"  {'Bases':<20} {'P_err':>8} {'Fidelity':>10}")
        print("  " + "-" * 40)
        for tag, r in sorted(results.items(), key=lambda x: len(x[1]['bases'])):
            pe = r['final_metrics']['prob_error']
            fid = r['final_metrics']['fidelity']
            print(f"  {tag:<20} {pe:>8.4f} {fid:>10.4f}")
    
    print(f"\nResults saved to {out_dir}/")


if __name__ == "__main__":
    main()
