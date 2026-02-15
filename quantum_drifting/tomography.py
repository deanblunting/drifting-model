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


def marginal_expectation(probs: np.ndarray, n_qubits: int, active_qubits: List[int]) -> float:
    """
    Compute the expectation value of a Pauli sub-operator acting on active_qubits,
    by marginalizing over the remaining qubits.

    For active qubits, each outcome contributes eigenvalue (-1)^{b_i}.
    For inactive qubits (identity), we sum over both outcomes (marginalize).

    Args:
        probs: [2^n] probability array from a full n-qubit measurement
        n_qubits: number of qubits
        active_qubits: list of qubit indices where a non-identity Pauli acts

    Returns:
        expectation value of the marginal operator
    """
    dim = 2 ** n_qubits
    expectation = 0.0
    for outcome_idx in range(dim):
        eigenvalue = 1.0
        for q in active_qubits:
            bit = (outcome_idx >> (n_qubits - 1 - q)) & 1
            eigenvalue *= (-1) ** bit
        expectation += probs[outcome_idx] * eigenvalue
    return expectation


def extract_all_expectations(
    probs: np.ndarray,
    basis_string: str,
    n_qubits: int,
) -> Dict[str, float]:
    """
    Extract all marginal Pauli expectations from a single basis measurement.

    A measurement in basis "XYZ" yields expectations for all 2^n - 1 nontrivial
    sub-operators: XYZ, XYI, XIZ, IYZ, XII, IYI, IIZ. Each is obtained by
    marginalizing over the qubits where we place I.

    Args:
        probs: [2^n] probability array
        basis_string: measurement basis (e.g., "XYZ")
        n_qubits: number of qubits

    Returns:
        dict mapping Pauli label (e.g., "XIZ") -> expectation value
    """
    expectations = {}
    # Iterate over all non-empty subsets of qubit positions
    for mask in range(1, 2 ** n_qubits):
        active_qubits = [q for q in range(n_qubits) if (mask >> (n_qubits - 1 - q)) & 1]
        label = ''.join(
            basis_string[q] if (mask >> (n_qubits - 1 - q)) & 1 else 'I'
            for q in range(n_qubits)
        )
        exp_val = marginal_expectation(probs, n_qubits, active_qubits)
        expectations[label] = exp_val
    return expectations


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

    For each measured basis (e.g., "XYZ"), we extract all 2^n - 1 marginal
    Pauli expectations (e.g., XYZ, XYI, XIZ, IYZ, XII, IYI, IIZ) by
    summing over qubits where the identity acts.

    When multiple bases provide the same Pauli expectation, we average them.

    Args:
        gen_probs: dict mapping basis string -> probability array
        n_qubits: number of qubits
        bases: list of measurement basis strings

    Returns:
        rho: reconstructed density matrix (may not be positive semidefinite)
    """
    dim = 2 ** n_qubits

    # Collect all Pauli expectations, averaging when multiple bases give the same one
    pauli_expectations: Dict[str, List[float]] = {}
    for basis in bases:
        if basis in gen_probs:
            probs = gen_probs[basis]
            marginals = extract_all_expectations(probs, basis, n_qubits)
            for label, exp_val in marginals.items():
                if label not in pauli_expectations:
                    pauli_expectations[label] = []
                pauli_expectations[label].append(exp_val)

    # Build rho
    rho = np.eye(dim, dtype=complex) / dim  # identity contribution

    for label, values in pauli_expectations.items():
        if label == 'I' * n_qubits:
            continue
        avg_exp = np.mean(values)
        P = pauli_tensor(label)
        rho += avg_exp * P / dim
    
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
