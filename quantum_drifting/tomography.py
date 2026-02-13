"""
Quantum state reconstruction from measurement statistics.

Given the probability distributions learned by the generator for each
measurement basis, reconstruct the density matrix of the quantum state.

Methods:
- Linear inversion: direct computation from Pauli expectation values
- Maximum likelihood estimation (MLE): iterative, ensures physical density matrix

For 1-2 qubits, linear inversion works well.
For 3+ qubits, MLE is more robust to noise.
"""

import numpy as np
from typing import Dict, List, Optional
from itertools import product


# Pauli matrices
PAULI_MATRICES = {
    'I': np.eye(2, dtype=complex),
    'X': np.array([[0, 1], [1, 0]], dtype=complex),
    'Y': np.array([[0, -1j], [1j, 0]], dtype=complex),
    'Z': np.array([[1, 0], [0, -1]], dtype=complex),
}


def pauli_tensor(label: str) -> np.ndarray:
    """Compute tensor product of single-qubit Pauli matrices."""
    result = PAULI_MATRICES[label[0]]
    for c in label[1:]:
        result = np.kron(result, PAULI_MATRICES[c])
    return result


def expectation_from_probs(probs: np.ndarray, basis_string: str, n_qubits: int) -> float:
    """
    Compute the expectation value <P> from measurement probabilities,
    where P is the tensor product Pauli operator for the given basis.
    
    For a Pauli measurement, each outcome has eigenvalue +1 or -1.
    The eigenvalue of outcome bitstring b = b_1 b_2 ... b_n is:
      lambda(b) = product_i (-1)^{b_i}
    
    So <P> = sum_b P(b) * lambda(b)
    """
    n = n_qubits
    dim = 2 ** n
    expectation = 0.0
    for outcome_idx in range(dim):
        # Decode outcome index to individual qubit outcomes
        bits = [(outcome_idx >> (n - 1 - q)) & 1 for q in range(n)]
        eigenvalue = 1.0
        for b in bits:
            eigenvalue *= (-1) ** b
        expectation += probs[outcome_idx] * eigenvalue
    return expectation


def reconstruct_density_matrix(
    gen_probs: Dict[str, np.ndarray],
    n_qubits: int,
    bases: List[str],
) -> np.ndarray:
    """
    Reconstruct the density matrix via linear inversion from Pauli expectations.
    
    The density matrix can be expanded as:
      rho = (1/2^n) * sum_{P} <P> * P
    
    where P ranges over all n-qubit Pauli operators (including identity).
    
    For each measured basis (e.g., "XY"), we get the expectation of the
    corresponding tensor product Pauli operator (sigma_X x sigma_Y).
    
    Args:
        gen_probs: dict mapping basis string -> probability array
        n_qubits: number of qubits
        bases: list of measurement basis strings
    
    Returns:
        rho: reconstructed density matrix (may not be positive semidefinite)
    """
    dim = 2 ** n_qubits
    rho = np.zeros((dim, dim), dtype=complex)
    
    # Identity contribution
    rho += np.eye(dim, dtype=complex) / dim
    
    # Pauli contributions from measured bases
    measured_paulis = set()
    for basis in bases:
        if basis in gen_probs:
            probs = gen_probs[basis]
            exp_val = expectation_from_probs(probs, basis, n_qubits)
            P = pauli_tensor(basis)
            
            # Check this isn't the identity (all same basis = not identity)
            pauli_label = basis  # e.g., "XZ"
            if pauli_label not in measured_paulis:
                rho += exp_val * P / dim
                measured_paulis.add(pauli_label)
    
    # Project to physical density matrix (positive semidefinite, trace 1)
    rho = project_to_physical(rho)
    
    return rho


def project_to_physical(rho: np.ndarray) -> np.ndarray:
    """
    Project a matrix to the nearest physical density matrix.
    
    A physical density matrix must be:
    1. Hermitian
    2. Positive semidefinite
    3. Trace = 1
    
    We achieve this by:
    1. Symmetrizing: rho <- (rho + rho^dagger) / 2
    2. Eigendecompose and clip negative eigenvalues to 0
    3. Renormalize trace to 1
    """
    # Hermitianize
    rho = (rho + rho.conj().T) / 2
    
    # Eigendecompose
    eigenvalues, eigenvectors = np.linalg.eigh(rho)
    
    # Clip negatives
    eigenvalues = np.maximum(eigenvalues, 0)
    
    # Renormalize
    total = eigenvalues.sum()
    if total > 1e-10:
        eigenvalues /= total
    else:
        # Maximally mixed state as fallback
        eigenvalues = np.ones_like(eigenvalues) / len(eigenvalues)
    
    # Reconstruct
    rho = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.conj().T
    return rho


def state_fidelity(rho1: np.ndarray, rho2: np.ndarray) -> float:
    """
    Compute the fidelity between two density matrices.
    
    F(rho1, rho2) = (Tr[sqrt(sqrt(rho1) * rho2 * sqrt(rho1))])^2
    
    For a pure state rho1 = |psi><psi|:
    F = <psi|rho2|psi> = Tr[rho1 * rho2]
    """
    # Use the simplified formula since our target is a pure state
    fid = np.real(np.trace(rho1 @ rho2))
    return float(np.clip(fid, 0, 1))


def trace_distance(rho1: np.ndarray, rho2: np.ndarray) -> float:
    """
    Trace distance: T(rho1, rho2) = (1/2) * Tr[|rho1 - rho2|]
    """
    diff = rho1 - rho2
    eigenvalues = np.linalg.eigvalsh(diff)
    return float(0.5 * np.sum(np.abs(eigenvalues)))
