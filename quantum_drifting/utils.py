"""
Visualization utilities for quantum tomography experiments.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Dict, List, Optional


def plot_training(result: Dict, save_path: str):
    """Plot training curves: loss, drift, prob error, fidelity."""
    h = result['history']
    state = result['state']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(
        f'Quantum Tomography via Drifting: {state.name}\n'
        f'{state.n_qubits} qubits, {len(result["bases"])} bases',
        fontsize=13, fontweight='bold'
    )
    
    ax = axes[0, 0]
    ax.semilogy(h['loss'], color='#2c3e50', linewidth=0.8, alpha=0.7)
    # Smoothed version
    if len(h['loss']) > 20:
        window = min(50, len(h['loss']) // 5)
        smoothed = np.convolve(h['loss'], np.ones(window)/window, mode='valid')
        ax.semilogy(range(window-1, len(h['loss'])), smoothed, color='#2c3e50', linewidth=2)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.set_title('Training Loss'); ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    ax.semilogy(h['drift_norm'], color='#e74c3c', linewidth=0.8, alpha=0.7)
    if len(h['drift_norm']) > 20:
        window = min(50, len(h['drift_norm']) // 5)
        smoothed = np.convolve(h['drift_norm'], np.ones(window)/window, mode='valid')
        ax.semilogy(range(window-1, len(h['drift_norm'])), smoothed, color='#e74c3c', linewidth=2)
    ax.set_xlabel('Epoch'); ax.set_ylabel('||V||^2')
    ax.set_title('Drift Norm'); ax.grid(True, alpha=0.3)
    
    ax = axes[1, 0]
    ax.plot(h['prob_error'], color='#27ae60', linewidth=1.5)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Avg L1 Error')
    ax.set_title('Probability Error'); ax.grid(True, alpha=0.3)
    
    ax = axes[1, 1]
    ax.plot(h['fidelity'], color='#8e44ad', linewidth=1.5)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Fidelity')
    ax.set_title('State Fidelity'); ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_probability_comparison(result: Dict, save_path: str, max_bases: int = 9):
    """Bar chart comparing true vs generated probabilities for each basis."""
    state = result['state']
    data_store = result['data_store']
    gen_probs = result['final_metrics']['gen_probs']
    bases = result['bases'][:max_bases]
    dim = state.dim
    
    n_bases_show = len(bases)
    fig, axes = plt.subplots(1, n_bases_show, figsize=(4 * n_bases_show, 4))
    if n_bases_show == 1:
        axes = [axes]
    
    fig.suptitle(f'Probability Distributions: {state.name}', fontsize=12, fontweight='bold')
    
    for idx, basis in enumerate(bases):
        ax = axes[idx]
        true_p = data_store.true_probs[basis]
        gen_p = gen_probs[basis]
        
        x_pos = np.arange(dim)
        w = 0.35
        ax.bar(x_pos - w/2, true_p, w, label='True', color='#3498db', alpha=0.85)
        ax.bar(x_pos + w/2, gen_p, w, label='Generated', color='#e74c3c', alpha=0.85)
        
        ax.set_title(f'Basis: {basis}', fontsize=10)
        ax.set_xlabel('Outcome')
        if idx == 0:
            ax.set_ylabel('Probability')
        ax.set_xticks(x_pos)
        labels = [f'{k:0{state.n_qubits}b}' for k in range(dim)]
        ax.set_xticklabels(labels, fontsize=7, rotation=45)
        ax.legend(fontsize=7)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_density_matrix(rho: np.ndarray, title: str, save_path: str):
    """Visualize real and imaginary parts of a density matrix."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(title, fontsize=12, fontweight='bold')
    
    im1 = ax1.imshow(np.real(rho), cmap='RdBu_r', vmin=-0.5, vmax=0.5)
    ax1.set_title('Re(rho)')
    plt.colorbar(im1, ax=ax1, shrink=0.8)
    
    im2 = ax2.imshow(np.imag(rho), cmap='RdBu_r', vmin=-0.5, vmax=0.5)
    ax2.set_title('Im(rho)')
    plt.colorbar(im2, ax=ax2, shrink=0.8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_incomplete_comparison(results: Dict[str, Dict], save_path: str):
    """
    Compare reconstruction quality vs number of measurement bases.
    Shows fidelity and probability error as a function of measurement completeness.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Effect of Measurement Completeness', fontsize=12, fontweight='bold')
    
    n_bases_list = []
    fidelities = []
    prob_errors = []
    labels = []
    
    for label, result in sorted(results.items(), key=lambda x: len(x[1]['bases'])):
        n_b = len(result['bases'])
        fid = result['final_metrics']['fidelity']
        perr = result['final_metrics']['prob_error']
        n_bases_list.append(n_b)
        fidelities.append(fid)
        prob_errors.append(perr)
        labels.append(label)
    
    ax1.plot(n_bases_list, fidelities, 'o-', color='#8e44ad', linewidth=2, markersize=8)
    ax1.set_xlabel('Number of Measurement Bases')
    ax1.set_ylabel('Fidelity')
    ax1.set_title('Reconstruction Fidelity')
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3)
    for i, label in enumerate(labels):
        ax1.annotate(label, (n_bases_list[i], fidelities[i]), fontsize=7,
                     textcoords="offset points", xytext=(5, 5))
    
    ax2.plot(n_bases_list, prob_errors, 'o-', color='#e74c3c', linewidth=2, markersize=8)
    ax2.set_xlabel('Number of Measurement Bases')
    ax2.set_ylabel('Avg L1 Error')
    ax2.set_title('Probability Error')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")
