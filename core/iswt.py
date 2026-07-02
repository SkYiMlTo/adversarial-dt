"""
Innovation Spatial Whiteness Test (ISWT) and Innovation Whiteness Detector (IWD).

Implements the ISWT from Eq. 6-8 (Sec. 2.3) and the IWD defense from Sec. 4.4:

1. Sample spatial covariance over sliding window W:
       C_hat(t) = (1/W) sum_{s=t-W+1}^{t} nu_hat(s) * nu_hat(s)^T      [Eq. 5]

2. Generalized Stein matrix divergence relative to baseline C_0:
       Lambda^IW(t) = tr(C_0^{-1} C_hat) - ln det(C_0^{-1} C_hat) - N  [Eq. 6']

   When C_0 = I_N, this reduces to the standard Stein divergence:
       Lambda^IW(t) = tr(C_hat) - ln det(C_hat) - N                     [Eq. 6]

3. Chi-squared alarm (Bartlett's theorem):
       alarm(t) = 1[W * Lambda^IW(t) > chi2_{1-alpha}(N(N+1)/2)]       [Eq. 7]

The use of a baseline covariance C_0 (estimated from clean calibration
data) accounts for residual EKF linearization artifacts in the
innovation cross-correlation structure. Under H0, C_hat(t) -> C_0 as
W -> infinity, so Lambda^IW(t) -> 0. Under attack, the adversarial
perturbation alters the cross-correlation structure, inflating
Lambda^IW above the chi-squared critical value.

This approach is equivalent to whitening the innovations with
C_0^{-1/2} before applying the standard ISWT, and is the standard
practice in multivariate change detection (cf. Basseville & Nikiforov).
"""

import numpy as np
from typing import Optional
from scipy.stats import chi2
from .config import ISWTConfig


class ISWTDetector:
    """Innovation Spatial Whiteness Test detector.

    Monitors the sample spatial covariance of standardized innovations
    for departures from the baseline covariance C_0, using the
    generalized Stein divergence.

    When no baseline is provided, falls back to the identity matrix
    (standard ISWT).
    """

    def __init__(self, n_sensors: int,
                 config: Optional[ISWTConfig] = None,
                 baseline_cov: Optional[np.ndarray] = None):
        """
        Args:
            n_sensors: Number of sensors (N).
            config: ISWT parameters (W, alpha).
            baseline_cov: (N, N) baseline covariance from clean data.
                          If None, uses I_N (standard ISWT).
        """
        self.cfg = config or ISWTConfig()
        self.N = n_sensors

        # Degrees of freedom: N(N+1)/2
        self.dof = n_sensors * (n_sensors + 1) // 2

        # Critical value (supports empirical calibration)
        self.critical = self.cfg.critical_value(n_sensors)

        # Baseline covariance and its inverse
        if baseline_cov is not None:
            self.C0 = baseline_cov.copy()
            self.C0_inv = np.linalg.inv(
                self.C0 + np.eye(n_sensors) * 1e-10)
        else:
            self.C0 = np.eye(n_sensors)
            self.C0_inv = np.eye(n_sensors)

        # Sliding window buffer for standardized innovations
        self.W = self.cfg.W
        self.buffer = np.zeros((self.W, n_sensors))
        self.buffer_idx = 0
        self.buffer_full = False

        # Current statistics
        self.C_hat = np.eye(n_sensors)     # Sample covariance
        self.lambda_iw = 0.0               # Stein divergence
        self.test_stat = 0.0               # W * Lambda^IW
        self.alarm = False

    def reset(self):
        """Reset the detector state."""
        self.buffer[:] = 0.0
        self.buffer_idx = 0
        self.buffer_full = False
        self.C_hat = np.eye(self.N)
        self.lambda_iw = 0.0
        self.test_stat = 0.0
        self.alarm = False

    def update(self, std_innovation: np.ndarray) -> dict:
        """Update ISWT with a new standardized innovation vector.

        Args:
            std_innovation: nu_hat(t) in R^N (standardized innovation).

        Returns:
            Dictionary with:
                - 'C_hat': sample spatial covariance matrix
                - 'lambda_iw': generalized Stein divergence
                - 'test_stat': W * Lambda^IW
                - 'critical': chi-squared critical value
                - 'alarm': True if ISWT alarm is triggered
                - 'ready': True if buffer is full (enough data)
        """
        # Add to circular buffer
        self.buffer[self.buffer_idx] = std_innovation
        self.buffer_idx = (self.buffer_idx + 1) % self.W
        if self.buffer_idx == 0:
            self.buffer_full = True

        if not self.buffer_full:
            # Not enough data yet
            return {
                'C_hat': self.C_hat.copy(),
                'lambda_iw': 0.0,
                'test_stat': 0.0,
                'critical': self.critical,
                'alarm': False,
                'ready': False,
            }

        # Compute sample spatial covariance: C_hat = (1/W) sum nu_hat * nu_hat^T
        self.C_hat = (self.buffer.T @ self.buffer) / self.W

        # Regularize for numerical stability (ensure positive definite)
        self.C_hat += np.eye(self.N) * 1e-10

        # Generalized Stein divergence: D(C_hat || C_0)
        # = tr(C_0^{-1} C_hat) - ln det(C_0^{-1} C_hat) - N
        M = self.C0_inv @ self.C_hat
        sign, logdet = np.linalg.slogdet(M)
        if sign <= 0:
            # Matrix product is not positive definite
            self.lambda_iw = 1e6
        else:
            self.lambda_iw = np.trace(M) - logdet - self.N

        # Test statistic: W * Lambda^IW
        self.test_stat = self.W * self.lambda_iw

        # Alarm decision
        self.alarm = self.test_stat > self.critical

        return {
            'C_hat': self.C_hat.copy(),
            'lambda_iw': self.lambda_iw,
            'test_stat': self.test_stat,
            'critical': self.critical,
            'alarm': self.alarm,
            'ready': True,
        }

    def run_batch(self, std_innovations: np.ndarray) -> dict:
        """Run ISWT over a batch of standardized innovations.

        Args:
            std_innovations: (T, N) array.

        Returns:
            Dictionary with arrays indexed by time:
                - 'lambda_iw': (T,) Stein divergence
                - 'test_stat': (T,) test statistic W*Lambda^IW
                - 'alarm': (T,) alarm flags
        """
        self.reset()
        T, N = std_innovations.shape

        lambda_iw = np.zeros(T)
        test_stat = np.zeros(T)
        alarm = np.zeros(T, dtype=bool)

        for t in range(T):
            result = self.update(std_innovations[t])
            lambda_iw[t] = result['lambda_iw']
            test_stat[t] = result['test_stat']
            alarm[t] = result['alarm']

        return {
            'lambda_iw': lambda_iw,
            'test_stat': test_stat,
            'alarm': alarm,
            'critical': self.critical,
        }


# ======================================================================
# Combined IWD | CUSUM decision (Eq. 13)
# ======================================================================

def combined_alarm(cusum_alarm: np.ndarray, iswt_alarm: bool) -> bool:
    """Combined IWD|CUSUM authentication decision (Eq. 13).

    a(t) = 1[max_i G_i(t) > h] | a^IWD(t)

    Args:
        cusum_alarm: Per-sensor CUSUM alarm flags.
        iswt_alarm: ISWT alarm flag.

    Returns:
        True if any alarm is triggered.
    """
    return bool(np.any(cusum_alarm) or iswt_alarm)


def combined_alarm_full(cusum_alarm: np.ndarray,
                        iswt_alarm: bool,
                        lstm_alarm: bool = False) -> bool:
    """Extended combined alarm including neural LSTM detector (Eq. 13+).

    a(t) = 1[max_i G_i(t) > h] | a^IWD(t) | a^LSTM(t)

    Args:
        cusum_alarm: Per-sensor CUSUM alarm flags.
        iswt_alarm: ISWT alarm flag.
        lstm_alarm: LSTM detector alarm flag.

    Returns:
        True if any alarm is triggered.
    """
    return bool(np.any(cusum_alarm) or iswt_alarm or lstm_alarm)


# ======================================================================
# PyTorch-differentiable ISWT (for TCA)
# ======================================================================

def iswt_torch(std_innovations, W=200, alpha=0.05, n_sensors=6,
               baseline_cov=None):
    """Differentiable ISWT computation in PyTorch.

    Computes the generalized Stein divergence over a sliding window.
    When baseline_cov is provided, computes D(C_hat || C_0) instead
    of D(C_hat || I).

    Args:
        std_innovations: (T, N) tensor of standardized innovations.
        W: Window size.
        alpha: Significance level.
        n_sensors: Number of sensors.
        baseline_cov: (N, N) numpy array baseline covariance.

    Returns:
        lambda_iw: (T,) tensor of generalized Stein divergence values.
    """
    import torch

    T, N = std_innovations.shape

    # Baseline inverse
    if baseline_cov is not None:
        C0_inv = torch.tensor(
            np.linalg.inv(baseline_cov + np.eye(N) * 1e-10),
            dtype=std_innovations.dtype,
            device=std_innovations.device)
    else:
        C0_inv = torch.eye(N, dtype=std_innovations.dtype,
                           device=std_innovations.device)

    # If T < W, return zeros
    if T < W:
        return torch.zeros(T, dtype=std_innovations.dtype,
                           device=std_innovations.device)

    # Unfold dimension 0 to create sliding windows of size W
    # permute(0, 2, 1) converts shape (T - W + 1, N, W) to (T - W + 1, W, N)
    windows = std_innovations.unfold(0, W, 1).permute(0, 2, 1)

    # Compute sample covariance matrices in parallel
    C_hat = torch.bmm(windows.transpose(1, 2), windows) / W

    # Regularize
    eye = torch.eye(N, dtype=std_innovations.dtype,
                    device=std_innovations.device).unsqueeze(0)
    C_hat = C_hat + eye * 1e-10

    # Batched generalized Stein divergence
    M = C0_inv.unsqueeze(0) @ C_hat
    tr_M = torch.diagonal(M, dim1=-2, dim2=-1).sum(dim=-1)
    logdet_M = torch.logdet(M)

    lambda_iw_valid = tr_M - logdet_M - N

    # Pad with W-1 zeros at the beginning for alignment
    padding = torch.zeros(W - 1, dtype=std_innovations.dtype,
                          device=std_innovations.device)
    lambda_iw = torch.cat([padding, lambda_iw_valid])

    return lambda_iw
