"""
Quantum state simulation for 1-N qubit systems.

Supports:
- Arbitrary pure states parameterized by state vectors
- Standard states: computational basis, Bell, GHZ, W, random Haar
- Pauli basis measurements (tensor products of X, Y, Z)
- Partial/incomplete measurement sets for testing data efficiency
- Born rule sampling
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from itertools import product


# ============================================================================
# Pauli matrices and measurement bases
# ============================================================================

PAULI = {
    'I': np.eye(2, dtype=complex),
    'X': np.array([[0, 1], [1, 0]], dtype=complex),
    'Y': np.array([[0, -1j], [1j, 0]], dtype=complex),
    'Z': np.array([[1, 0], [0, -1]], dtype=complex),
}

# Eigenstates of each Pauli
EIGENSTATES = {
    'X': [np.array([1, 1]) / np.sqrt(2), np.array([1, -1]) / np.sqrt(2)],
    'Y': [np.array([1, 1j]) / np.sqrt(2), np.array([1, -1j]) / np.sqrt(2)],
    'Z': [np.array([1, 0]), np.array([0, 1])],
}


def tensor_product(*matrices):
    """Compute tensor (Kronecker) product of multiple matrices."""
    result = matrices[0]
    for m in matrices[1:]:
        result = np.kron(result, m)
    return result


def get_measurement_projectors(basis_string: str) -> List[np.ndarray]:
    """
    Get projectors for a multi-qubit Pauli measurement.
    
    Args:
        basis_string: e.g., "XZ" for measuring qubit 0 in X, qubit 1 in Z
    
    Returns:
        List of 2^n projectors, one per measurement outcome.
        Outcomes are ordered as tensor products of single-qubit eigenstates.
    """
    n = len(basis_string)
    single_qubit_eigenstates = [EIGENSTATES[b] for b in basis_string]
    
    projectors = []
    for outcome_combo in product(range(2), repeat=n):
        # Tensor product of single-qubit eigenstates
        state = single_qubit_eigenstates[0][outcome_combo[0]]
        for q in range(1, n):
            state = np.kron(state, single_qubit_eigenstates[q][outcome_combo[q]])
        projectors.append(np.outer(state, np.conj(state)))
    
    return projectors


# ============================================================================
# Quantum States
# ============================================================================

class QuantumState:
    """
    N-qubit pure quantum state.
    
    The state is stored as a complex vector of dimension 2^N.
    Measurements are performed via Born's rule: P(k) = |<k|psi>|^2
    where |k> is the k-th eigenstate of the measurement basis.
    """
    
    def __init__(self, state_vector: np.ndarray, n_qubits: int, name: str = ""):
        """
        Args:
            state_vector: complex vector of dimension 2^n_qubits (will be normalized)
            n_qubits: number of qubits
            name: descriptive name for this state
        """
        self.n_qubits = n_qubits
        self.dim = 2 ** n_qubits
        assert len(state_vector) == self.dim, f"Expected dim {self.dim}, got {len(state_vector)}"
        
        # Normalize
        self.state_vector = state_vector.astype(complex)
        self.state_vector /= np.linalg.norm(self.state_vector)
        self.name = name
        
        # Density matrix
        self.rho = np.outer(self.state_vector, np.conj(self.state_vector))
    
    def measure_probs(self, basis_string: str) -> np.ndarray:
        """
        Born rule probabilities for a multi-qubit Pauli measurement.
        
        Args:
            basis_string: e.g., "XZ" for 2 qubits measured in X and Z respectively
        
        Returns:
            Array of 2^n probabilities, one per outcome bitstring
        """
        assert len(basis_string) == self.n_qubits
        projectors = get_measurement_projectors(basis_string)
        probs = np.array([np.real(np.trace(P @ self.rho)) for P in projectors])
        probs = np.clip(probs, 0, 1)
        probs /= probs.sum()  # numerical safety
        return probs
    
    def sample(self, basis_string: str, n_shots: int) -> np.ndarray:
        """
        Sample measurement outcomes.
        
        Returns:
            Array of shape [n_shots] with integer outcomes in [0, 2^n - 1]
        """
        probs = self.measure_probs(basis_string)
        return np.random.choice(self.dim, size=n_shots, p=probs)
    
    def sample_encoded(self, basis_string: str, n_shots: int) -> np.ndarray:
        """
        Sample and encode as one-hot-like continuous vectors for the drifting model.
        
        Returns:
            Array of shape [n_shots, 2^n] with soft one-hot encoding:
            outcome k -> vector with +1 at position k, -1 elsewhere
        """
        outcomes = self.sample(basis_string, n_shots)
        encoded = np.full((n_shots, self.dim), -1.0)
        encoded[np.arange(n_shots), outcomes] = 1.0
        return encoded
    
    def all_pauli_bases(self) -> List[str]:
        """Return all 3^n Pauli measurement bases (tensor products of X,Y,Z)."""
        return [''.join(combo) for combo in product('XYZ', repeat=self.n_qubits)]
    
    def info_complete_bases(self) -> List[str]:
        """
        Return a minimal informationally complete set of bases.
        For n qubits, we need at least 3^n - 1 independent real parameters
        (since rho is a 2^n x 2^n Hermitian matrix with trace 1).
        We return all 3^n Pauli bases, which is overcomplete but standard.
        """
        return self.all_pauli_bases()
    
    def __repr__(self):
        return f"QuantumState(n_qubits={self.n_qubits}, name='{self.name}')"


# ============================================================================
# Standard State Constructors
# ============================================================================

def computational_basis(n_qubits: int, index: int) -> QuantumState:
    """Computational basis state |index> for n qubits."""
    dim = 2 ** n_qubits
    sv = np.zeros(dim, dtype=complex)
    sv[index] = 1.0
    return QuantumState(sv, n_qubits, name=f"|{index:0{n_qubits}b}>")


def plus_state(n_qubits: int) -> QuantumState:
    """|+>^n = uniform superposition."""
    dim = 2 ** n_qubits
    sv = np.ones(dim, dtype=complex) / np.sqrt(dim)
    return QuantumState(sv, n_qubits, name="|+>^" + str(n_qubits))


def bell_state(which: int = 0) -> QuantumState:
    """
    Bell states for 2 qubits.
    0: |Phi+> = (|00> + |11>) / sqrt(2)
    1: |Phi-> = (|00> - |11>) / sqrt(2)
    2: |Psi+> = (|01> + |10>) / sqrt(2)
    3: |Psi-> = (|01> - |10>) / sqrt(2)
    """
    names = ['Phi+', 'Phi-', 'Psi+', 'Psi-']
    svs = [
        np.array([1, 0, 0, 1]) / np.sqrt(2),
        np.array([1, 0, 0, -1]) / np.sqrt(2),
        np.array([0, 1, 1, 0]) / np.sqrt(2),
        np.array([0, 1, -1, 0]) / np.sqrt(2),
    ]
    return QuantumState(svs[which].astype(complex), 2, name=f"|{names[which]}>")


def ghz_state(n_qubits: int) -> QuantumState:
    """GHZ state: (|00...0> + |11...1>) / sqrt(2)."""
    dim = 2 ** n_qubits
    sv = np.zeros(dim, dtype=complex)
    sv[0] = 1.0 / np.sqrt(2)       # |00...0>
    sv[-1] = 1.0 / np.sqrt(2)      # |11...1>
    return QuantumState(sv, n_qubits, name=f"GHZ({n_qubits})")


def w_state(n_qubits: int) -> QuantumState:
    """W state: (|100...0> + |010...0> + ... + |000...1>) / sqrt(n)."""
    dim = 2 ** n_qubits
    sv = np.zeros(dim, dtype=complex)
    for i in range(n_qubits):
        idx = 1 << (n_qubits - 1 - i)
        sv[idx] = 1.0 / np.sqrt(n_qubits)
    return QuantumState(sv, n_qubits, name=f"W({n_qubits})")


def random_haar_state(n_qubits: int, seed: Optional[int] = None) -> QuantumState:
    """Random state drawn from the Haar measure."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    dim = 2 ** n_qubits
    real = rng.randn(dim)
    imag = rng.randn(dim)
    sv = (real + 1j * imag)
    sv /= np.linalg.norm(sv)
    return QuantumState(sv, n_qubits, name=f"Haar({n_qubits})")


def random_product_state(n_qubits: int, seed: Optional[int] = None) -> QuantumState:
    """Random product (unentangled) state."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    
    sv = np.array([1.0], dtype=complex)
    for _ in range(n_qubits):
        theta = rng.uniform(0, np.pi)
        phi = rng.uniform(0, 2 * np.pi)
        qubit = np.array([np.cos(theta/2), np.exp(1j*phi) * np.sin(theta/2)])
        sv = np.kron(sv, qubit)
    
    return QuantumState(sv, n_qubits, name=f"Product({n_qubits})")


# ============================================================================
# Measurement Set Selection
# ============================================================================

def select_random_bases(n_qubits: int, n_bases: int, 
                        seed: Optional[int] = None) -> List[str]:
    """Select a random subset of Pauli measurement bases."""
    rng = np.random.RandomState(seed) if seed is not None else np.random
    all_bases = [''.join(combo) for combo in product('XYZ', repeat=n_qubits)]
    n_bases = min(n_bases, len(all_bases))
    indices = rng.choice(len(all_bases), size=n_bases, replace=False)
    return [all_bases[i] for i in sorted(indices)]


def select_local_bases(n_qubits: int) -> List[str]:
    """
    Select only the 3n local (single-qubit) measurement bases.
    E.g., for 2 qubits: XI, YI, ZI, IX, IY, IZ -> but since we don't
    use I, we approximate with: XZ, YZ, ZZ, ZX, ZY, ZZ (deduplicated).
    
    Actually, for local tomography we measure each qubit individually,
    so the bases are: for each qubit position, measure in X, Y, Z
    while measuring all others in Z (a common convention).
    """
    bases = []
    for qubit_idx in range(n_qubits):
        for pauli in 'XYZ':
            basis = ['Z'] * n_qubits
            basis[qubit_idx] = pauli
            b = ''.join(basis)
            if b not in bases:
                bases.append(b)
    return bases
