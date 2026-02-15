"""
4-qubit tomography experiment using the AutoregressiveGenerator architecture.

Improvements over the flat MLP:
- Autoregressive factorization: P(k1,...,kn) = P(k1)*P(k2|k1)*... avoids 2^n output
- Per-qubit basis encoding: X/Y/Z embeddings generalize across bases sharing Pauli factors
- Basis sampling: random subset of bases per epoch for efficient training over 3^4=81 bases
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantum_drifting.states import ghz_state
from quantum_drifting.trainer import train, TrainConfig
from quantum_drifting.tomography import reconstruct_density_matrix, state_fidelity, trace_distance
from quantum_drifting.utils import plot_training, plot_probability_comparison, plot_density_matrix


def main():
    out_dir = "results/phase3"
    os.makedirs(out_dir, exist_ok=True)

    state = ghz_state(4)
    bases = state.all_pauli_bases()
    print(f"State: {state}")
    print(f"Bases: {len(bases)}/{3**state.n_qubits}")

    config = TrainConfig(
        autoregressive=True,
        noise_dim=64, hidden_dim=256,
        n_layers=6,
        n_pos=256, n_neg=256,
        bases_per_epoch=27,  # sample 27 of 81 bases per epoch
        lr=1e-4, n_epochs=2000,
        log_every=400, eval_every=100,
    )

    result = train(state, bases, config)

    plot_training(result, f"{out_dir}/train_GHZ4.png")
    plot_probability_comparison(result, f"{out_dir}/probs_GHZ4.png", max_bases=9)

    gen_probs = result['final_metrics']['gen_probs']
    rho_recon = reconstruct_density_matrix(gen_probs, state.n_qubits, bases)

    plot_density_matrix(state.rho, "True rho: GHZ(4)", f"{out_dir}/rho_true_GHZ4.png")
    plot_density_matrix(rho_recon, "Reconstructed rho: GHZ(4)", f"{out_dir}/rho_recon_GHZ4.png")

    fid = state_fidelity(state.rho, rho_recon)
    td = trace_distance(state.rho, rho_recon)
    print(f"\nFidelity:       {fid:.4f}")
    print(f"Trace distance: {td:.4f}")
    print(f"Prob error:     {result['final_metrics']['prob_error']:.4f}")
    print(f"\nResults saved to {out_dir}/")


if __name__ == "__main__":
    main()
