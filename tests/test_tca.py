"""
Tests for TCA attack and SDS metric.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import SystemConfig, EKFConfig, CUSUMConfig, ISWTConfig, TCAConfig
from core.process_model import TwoTankProcess
from core.ekf import ExtendedKalmanFilter
from core.cusum import CUSUMDetector
from core.iswt import ISWTDetector
from core.sds import compute_sds, compute_sds_timeseries
from core.tca import TargetedConsistencyAttack


class TestSDS:
    """Tests for the Sensor Deception Score."""

    def test_sds_boundary_perfect(self):
        """SDS = 1 when all CUSUM at 0 and ISWT at 0."""
        G = np.zeros(6)
        test_stat = 0.0
        compromised = np.array([5])

        result = compute_sds(G, test_stat, compromised, h=5.0,
                             n_sensors=6, alpha=0.05)
        assert abs(result['sds'] - 1.0) < 1e-10

    def test_sds_boundary_cusum_alarm(self):
        """SDS = 0 when any CUSUM exceeds threshold."""
        G = np.array([0, 0, 0, 0, 0, 6.0])  # Sensor 5 exceeds h=5
        test_stat = 0.0
        compromised = np.array([5])

        result = compute_sds(G, test_stat, compromised, h=5.0)
        assert result['sds'] == 0.0

    def test_sds_boundary_iswt_alarm(self):
        """SDS = 0 when ISWT exceeds critical value."""
        G = np.zeros(6)
        test_stat = 1000.0  # Way above critical
        compromised = np.array([5])

        result = compute_sds(G, test_stat, compromised, h=5.0)
        assert result['sds'] == 0.0

    def test_sds_multiplicative(self):
        """SDS should be product of phi_mean and psi."""
        G = np.array([0, 0, 0, 0, 0, 2.5])  # G_5 = h/2 → φ = 0.5
        test_stat = 0.0  # Perfect whiteness → ψ = 1
        compromised = np.array([5])

        result = compute_sds(G, test_stat, compromised, h=5.0)
        assert abs(result['sds'] - 0.5) < 1e-10
        assert abs(result['phi_mean'] - 0.5) < 1e-10
        assert abs(result['psi'] - 1.0) < 1e-10


class TestTCA:
    """Tests for the Targeted Consistency Attack."""

    @pytest.fixture
    def setup(self):
        """Create test fixtures."""
        sys_cfg = SystemConfig()
        ekf_cfg = EKFConfig()
        cusum_cfg = CUSUMConfig()
        iswt_cfg = ISWTConfig(W=50)  # Smaller window for faster tests
        tca_cfg = TCAConfig(K=10, eta=0.01)  # Fewer iterations for speed

        process = TwoTankProcess(sys_cfg)
        T = 100  # Short for testing

        sim = process.simulate(T, fault_config={
            'sensor_idx': [5],
            'fault_start': 0,
            'fault_magnitude': 2.0,
        }, seed=42)

        return {
            'sys': sys_cfg, 'ekf': ekf_cfg, 'cusum': cusum_cfg,
            'iswt': iswt_cfg, 'tca': tca_cfg,
            'Y': sim['y_faulted'], 'U': sim['u'],
        }

    def test_greybox_improves_sds(self, setup):
        """Grey-box TCA should improve SDS over no perturbation."""
        tca = TargetedConsistencyAttack(
            setup['sys'], setup['ekf'], setup['cusum'],
            setup['iswt'], setup['tca']
        )

        # Baseline SDS (no perturbation)
        sds_base = tca._evaluate_surrogate(
            setup['Y'], setup['U'],
            np.zeros_like(setup['Y']), [5]
        )[0]

        # TCA grey-box
        result = tca.run_greybox(
            setup['Y'], setup['U'],
            list(range(6)), [5],
            epsilon=0.75 * setup['sys'].sigma[5],
            verbose=False
        )

        assert result['sds_final'] >= sds_base - 0.01, \
            f"TCA should not decrease SDS: " \
            f"{result['sds_final']:.4f} < {sds_base:.4f}"

    def test_perturbation_budget(self, setup):
        """TCA perturbation should respect L∞ budget."""
        tca = TargetedConsistencyAttack(
            setup['sys'], setup['ekf'], setup['cusum'],
            setup['iswt'], setup['tca']
        )

        epsilon = 0.5 * setup['sys'].sigma[5]
        result = tca.run_greybox(
            setup['Y'], setup['U'],
            list(range(6)), [5], epsilon, verbose=False
        )

        max_pert = np.max(np.abs(result['delta']))
        assert max_pert <= epsilon + 1e-10, \
            f"Budget violation: max |δ| = {max_pert:.6f} > ε = {epsilon:.6f}"

    def test_sds_monotonic(self, setup):
        """SDS should generally improve over TCA iterations."""
        tca = TargetedConsistencyAttack(
            setup['sys'], setup['ekf'], setup['cusum'],
            setup['iswt'], TCAConfig(K=20, eta=0.005)
        )

        result = tca.run_greybox(
            setup['Y'], setup['U'],
            list(range(6)), [5],
            epsilon=1.0 * setup['sys'].sigma[5],
            verbose=False
        )

        history = result['sds_history']
        # Final SDS should be >= initial SDS (or close)
        assert history[-1] >= history[0] - 0.1, \
            f"SDS did not improve: {history[0]:.4f} → {history[-1]:.4f}"
