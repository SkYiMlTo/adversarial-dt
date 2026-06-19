"""
Tests for CUSUM and ISWT detectors.
"""

import numpy as np
import pytest
import sys
from pathlib import Path
from scipy.stats import chi2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import CUSUMConfig, ISWTConfig
from core.cusum import CUSUMDetector
from core.iswt import ISWTDetector


class TestCUSUM:
    """Tests for the CUSUM detector."""

    def test_zero_on_white_noise(self):
        """CUSUM should stay near zero on white noise (no mean shift)."""
        config = CUSUMConfig(k=0.5, h=5.0)
        cusum = CUSUMDetector(6, config)

        rng = np.random.default_rng(42)
        T = 1000
        innovations = rng.standard_normal((T, 6))

        results = cusum.run_batch(innovations)

        # G should stay below threshold most of the time
        max_G = np.max(results['G'])
        n_alarms = np.sum(np.any(results['alarm'], axis=1))

        # With ARL0 ≈ 148, expect roughly T/148 ≈ 6.7 alarms
        assert n_alarms < 50, f"Too many alarms on white noise: {n_alarms}"

    def test_detects_mean_shift(self):
        """CUSUM should detect a persistent mean shift."""
        config = CUSUMConfig(k=0.5, h=5.0)
        cusum = CUSUMDetector(1, config)

        rng = np.random.default_rng(42)
        T = 200

        # Normal for first 100 steps, then shift of 2σ
        innovations = rng.standard_normal((T, 1))
        innovations[100:, 0] += 2.0  # 2σ shift

        results = cusum.run_batch(innovations)

        # Should alarm after the shift
        alarm_after_shift = np.any(results['alarm'][100:])
        assert alarm_after_shift, "CUSUM failed to detect 2σ mean shift"

    def test_cusum_reset(self):
        """CUSUM should reset to zero."""
        cusum = CUSUMDetector(3)
        cusum.update(np.array([2.0, 2.0, 2.0]))
        assert np.all(cusum.G > 0)
        cusum.reset()
        assert np.all(cusum.G == 0)

    def test_phi_boundary_values(self):
        """φ_i should be 1 at G=0 and 0 at G≥h."""
        from core.sds import compute_phi

        h = 5.0
        # G = 0 → φ = 1
        assert compute_phi(np.array([0.0]), h)[0] == 1.0
        # G = h → φ = 0
        assert compute_phi(np.array([5.0]), h)[0] == 0.0
        # G > h → φ = 0
        assert compute_phi(np.array([10.0]), h)[0] == 0.0
        # G = h/2 → φ = 0.5
        assert abs(compute_phi(np.array([2.5]), h)[0] - 0.5) < 1e-10


class TestISWT:
    """Tests for the ISWT detector."""

    def test_no_alarm_on_white_noise(self):
        """ISWT should not alarm on spatially white innovations."""
        iswt = ISWTDetector(6, ISWTConfig(W=200, alpha=0.05))

        rng = np.random.default_rng(42)
        T = 500
        innovations = rng.standard_normal((T, 6))

        results = iswt.run_batch(innovations)

        # FPR should be near α = 0.05
        alarm_rate = np.mean(results['alarm'][200:])
        assert alarm_rate < 0.15, \
            f"ISWT FPR too high: {alarm_rate:.3f} (expected ≈ 0.05)"

    def test_detects_cross_correlation(self):
        """ISWT should alarm on correlated innovations."""
        iswt = ISWTDetector(6, ISWTConfig(W=200, alpha=0.05))

        rng = np.random.default_rng(42)
        T = 500

        # Create innovations with strong cross-correlation
        base = rng.standard_normal((T, 1))
        innovations = rng.standard_normal((T, 6)) * 0.5
        innovations[:, 0] += base[:, 0]
        innovations[:, 1] += 0.8 * base[:, 0]  # Correlated with sensor 0

        results = iswt.run_batch(innovations)

        # Should alarm after window fills
        alarm_rate = np.mean(results['alarm'][200:])
        assert alarm_rate > 0.3, \
            f"ISWT failed to detect correlation: alarm rate = {alarm_rate:.3f}"

    def test_stein_divergence_nonneg(self):
        """Stein divergence should be non-negative."""
        iswt = ISWTDetector(4, ISWTConfig(W=100, alpha=0.05))

        rng = np.random.default_rng(42)
        innovations = rng.standard_normal((200, 4))

        results = iswt.run_batch(innovations)

        assert np.all(results['lambda_iw'] >= -1e-10), \
            "Stein divergence should be non-negative"

    def test_chi_squared_critical_value(self):
        """Critical value should match scipy chi2."""
        N = 6
        alpha = 0.05
        dof = N * (N + 1) // 2  # 21 (full covariance matrix)

        iswt = ISWTDetector(N, ISWTConfig(W=200, alpha=alpha))
        expected = chi2.ppf(1 - alpha, dof)

        assert abs(iswt.critical - expected) < 1e-10, \
            f"Critical value mismatch: {iswt.critical} vs {expected}"

