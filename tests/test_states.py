"""Tests for quantum state simulation."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from quantum_drifting.states import (
    QuantumState, bell_state, ghz_state, w_state,
    random_haar_state, computational_basis, plus_state,
    get_measurement_projectors,
)


class TestSingleQubit:
    def test_z_basis_zero_state(self):
        """Measuring |0> in Z should give P(0)=1."""
        state = computational_basis(1, 0)
        probs = state.measure_probs('Z')
        assert np.isclose(probs[0], 1.0)
        assert np.isclose(probs[1], 0.0)

    def test_x_basis_plus_state(self):
        """Measuring |+> in X should give P(+)=1."""
        state = plus_state(1)
        probs = state.measure_probs('X')
        assert np.isclose(probs[0], 1.0, atol=1e-10)

    def test_z_basis_plus_state(self):
        """Measuring |+> in Z should give 50/50."""
        state = plus_state(1)
        probs = state.measure_probs('Z')
        assert np.allclose(probs, [0.5, 0.5], atol=1e-10)

    def test_probabilities_sum_to_one(self):
        state = random_haar_state(1, seed=42)
        for basis in ['X', 'Y', 'Z']:
            probs = state.measure_probs(basis)
            assert np.isclose(probs.sum(), 1.0)


class TestBellStates:
    def test_bell_phi_plus_zz(self):
        """Bell |Phi+> measured in ZZ: P(00) = P(11) = 0.5."""
        state = bell_state(0)
        probs = state.measure_probs('ZZ')
        assert np.isclose(probs[0], 0.5, atol=1e-10)  # |00>
        assert np.isclose(probs[3], 0.5, atol=1e-10)  # |11>
        assert np.isclose(probs[1], 0.0, atol=1e-10)  # |01>
        assert np.isclose(probs[2], 0.0, atol=1e-10)  # |10>

    def test_bell_phi_plus_xx(self):
        """Bell |Phi+> measured in XX: P(++) = P(--) = 0.5."""
        state = bell_state(0)
        probs = state.measure_probs('XX')
        assert np.isclose(probs[0], 0.5, atol=1e-10)
        assert np.isclose(probs[3], 0.5, atol=1e-10)

    def test_bell_psi_minus_zz(self):
        """Bell |Psi-> in ZZ: P(01) = P(10) = 0.5."""
        state = bell_state(3)
        probs = state.measure_probs('ZZ')
        assert np.isclose(probs[1], 0.5, atol=1e-10)
        assert np.isclose(probs[2], 0.5, atol=1e-10)

    def test_all_bases_sum_to_one(self):
        state = bell_state(0)
        for basis in state.all_pauli_bases():
            probs = state.measure_probs(basis)
            assert np.isclose(probs.sum(), 1.0), f"Failed for basis {basis}"


class TestGHZ:
    def test_ghz_3_zzz(self):
        """GHZ(3) in ZZZ: P(000) = P(111) = 0.5."""
        state = ghz_state(3)
        probs = state.measure_probs('ZZZ')
        assert np.isclose(probs[0], 0.5, atol=1e-10)
        assert np.isclose(probs[7], 0.5, atol=1e-10)

    def test_ghz_normalization(self):
        for n in [2, 3, 4]:
            state = ghz_state(n)
            assert np.isclose(np.linalg.norm(state.state_vector), 1.0)


class TestWState:
    def test_w_3_zzz(self):
        """W(3) in ZZZ: P(001)=P(010)=P(100)=1/3."""
        state = w_state(3)
        probs = state.measure_probs('ZZZ')
        assert np.isclose(probs[1], 1/3, atol=1e-10)  # 001
        assert np.isclose(probs[2], 1/3, atol=1e-10)  # 010
        assert np.isclose(probs[4], 1/3, atol=1e-10)  # 100


class TestProjectors:
    def test_projectors_complete(self):
        """Projectors for any basis should sum to identity."""
        for basis in ['XX', 'XY', 'ZZ', 'YX']:
            projs = get_measurement_projectors(basis)
            total = sum(projs)
            assert np.allclose(total, np.eye(4), atol=1e-10), f"Failed for {basis}"

    def test_projectors_orthogonal(self):
        """Projectors should be orthogonal: P_i P_j = 0 for i != j."""
        projs = get_measurement_projectors('XZ')
        for i in range(len(projs)):
            for j in range(len(projs)):
                prod = projs[i] @ projs[j]
                if i == j:
                    assert np.allclose(prod, projs[i], atol=1e-10)
                else:
                    assert np.allclose(prod, 0, atol=1e-10)


class TestSampling:
    def test_sample_statistics(self):
        """Large sample should approximate true probabilities."""
        state = bell_state(0)
        n = 100000
        outcomes = state.sample('ZZ', n)
        counts = np.bincount(outcomes, minlength=4)
        empirical = counts / n
        true_probs = state.measure_probs('ZZ')
        assert np.allclose(empirical, true_probs, atol=0.02)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
