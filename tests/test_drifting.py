"""Tests for drifting field computation."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytest
from quantum_drifting.drifting import (
    compute_drifting_field, compute_multi_temperature_drift, drifting_loss,
)


class TestDriftingField:
    def test_equilibrium_identical_distributions(self):
        """When p == q (same samples), V should be approximately zero."""
        torch.manual_seed(42)
        N = 64
        D = 4
        samples = torch.randn(N, D)
        
        V = compute_drifting_field(samples, samples, samples, temperature=0.1)
        # V should be near zero (not exact due to self-masking and finite samples)
        assert V.abs().mean() < 0.1, f"V should be near zero, got mean abs {V.abs().mean()}"

    def test_antisymmetry(self):
        """V(p,q) should equal -V(q,p) (anti-symmetry property)."""
        torch.manual_seed(42)
        N = 32
        D = 2
        p_samples = torch.randn(N, D) + 1.0
        q_samples = torch.randn(N, D) - 1.0
        x = torch.randn(N, D)
        
        V_pq = compute_drifting_field(x, p_samples, q_samples, temperature=0.1, mask_self=False)
        V_qp = compute_drifting_field(x, q_samples, p_samples, temperature=0.1, mask_self=False)
        
        # V_pq + V_qp should be near zero
        diff = (V_pq + V_qp).abs().mean()
        assert diff < 0.1, f"Anti-symmetry violated: mean |V_pq + V_qp| = {diff}"

    def test_drift_toward_data(self):
        """Generated samples should drift toward the data distribution."""
        torch.manual_seed(42)
        # Data clustered at +2, generated at -2
        y_pos = torch.randn(64, 1) + 2.0
        x = torch.randn(32, 1) - 2.0
        y_neg = x.clone()
        
        V = compute_drifting_field(x, y_pos, y_neg, temperature=0.5)
        # V should point toward positive direction (toward data)
        assert V.mean() > 0, f"Drift should be positive, got {V.mean()}"

    def test_output_shape(self):
        x = torch.randn(16, 4)
        y_pos = torch.randn(32, 4)
        y_neg = torch.randn(16, 4)
        V = compute_drifting_field(x, y_pos, y_neg, temperature=0.1)
        assert V.shape == x.shape


class TestMultiTemperature:
    def test_multi_temp_output_shape(self):
        x = torch.randn(16, 4)
        y_pos = torch.randn(32, 4)
        y_neg = torch.randn(16, 4)
        V = compute_multi_temperature_drift(x, y_pos, y_neg, [0.05, 0.1, 0.3])
        assert V.shape == x.shape

    def test_normalized_drift_unit_scale(self):
        """With normalization, each temperature's V should have unit RMS."""
        torch.manual_seed(42)
        x = torch.randn(32, 4)
        y_pos = torch.randn(64, 4) + 1.0
        y_neg = x.clone()
        
        V = compute_multi_temperature_drift(
            x, y_pos, y_neg, [0.1], normalize_drift=True
        )
        # Single temp with normalization -> RMS should be ~1
        rms = torch.sqrt(torch.mean(V ** 2))
        assert 0.5 < rms < 2.0, f"Normalized drift RMS should be ~1, got {rms}"


class TestDriftingLoss:
    def test_loss_is_scalar(self):
        x = torch.randn(16, 4, requires_grad=True)
        V = torch.randn(16, 4)
        loss = drifting_loss(x, V)
        assert loss.shape == ()

    def test_loss_is_v_squared(self):
        """Loss should equal E[||V||^2] (since target = x + V detached)."""
        x = torch.randn(16, 4, requires_grad=True)
        V = torch.randn(16, 4)
        loss = drifting_loss(x, V)
        expected = torch.mean(V ** 2)
        assert torch.isclose(loss, expected, atol=1e-5)

    def test_gradient_flows_through_x(self):
        x = torch.randn(16, 4, requires_grad=True)
        V = torch.randn(16, 4)
        loss = drifting_loss(x, V)
        loss.backward()
        assert x.grad is not None
        assert not torch.all(x.grad == 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
