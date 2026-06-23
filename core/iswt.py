"""
Innovation Spatial Whiteness Test (ISWT) and Innovation Whiteness Detector (IWD).

Implements the ISWT from Eq. 6–8 (Sec. 2.3) and the IWD defense from Sec. 4.4:

1. Sample spatial covariance over sliding window W:
       Ĉ(t) = (1/W) Σ_{s=t-W+1}^{t} ν̂(s)·ν̂(s)^T         [Eq. 5]

2. Stein matrix divergence:
       Λ^IW(t) = tr(Ĉ) - ln det(Ĉ) - N                    [Eq. 6]

3. Chi-squared alarm (Bartlett's theorem):
       alarm(t) = 1[W·Λ^IW(t) > χ²_{1-α}(N(N+1)/2)]       [Eq. 7]

Under H0 (nominal operation), Ĉ → I_N as W → ∞, so Λ^IW → 0.
Under TCA, the cross-sensor innovation correlations introduced by the
Kalman gain coupling produce off-diagonal mass in Ĉ, inflating Λ^IW
above the chi-squared critical value (Proposition 2).

The IWD's false alarm rate is analytically controlled by α through the
known chi-squared null distribution — no empirical calibration required.
"""

import numpy as np
from typing import Optional
from scipy.stats import chi2
from .config import ISWTConfig


class ISWTDetector:
    """Innovation Spatial Whiteness Test detector.

    Monitors the sample spatial covariance of standardized innovations
    for departures from the identity matrix, using the Stein divergence.
    """

    def __init__(self, n_sensors: int,
                 config: Optional[ISWTConfig] = None):
        """
        Args:
            n_sensors: Number of sensors (N).
            config: ISWT parameters (W, α).
        """
        self.cfg = config or ISWTConfig()
        self.N = n_sensors

        # Degrees of freedom: N(N+1)/2
        # The Stein divergence tests the full covariance matrix
        # (diagonal + off-diagonal), giving N(N+1)/2 unique entries.
        self.dof = n_sensors * (n_sensors + 1) // 2

        # Critical value (supports empirical calibration)
        self.critical = self.cfg.critical_value(n_sensors)

        # Sliding window buffer for standardized innovations
        self.W = self.cfg.W
        self.buffer = np.zeros((self.W, n_sensors))
        self.buffer_idx = 0
        self.buffer_full = False

        # Current statistics
        self.C_hat = np.eye(n_sensors)     # Sample covariance
        self.lambda_iw = 0.0               # Stein divergence
        self.test_stat = 0.0               # W · Λ^IW
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
            std_innovation: ν̂(t) ∈ R^N (standardized innovation).

        Returns:
            Dictionary with:
                - 'C_hat': sample spatial covariance matrix
                - 'lambda_iw': Stein divergence Λ^IW
                - 'test_stat': W · Λ^IW
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

        # Compute sample spatial covariance: Ĉ = (1/W) Σ ν̂·ν̂^T
        self.C_hat = (self.buffer.T @ self.buffer) / self.W

        # Regularize for numerical stability (ensure positive definite)
        self.C_hat += np.eye(self.N) * 1e-10

        # Stein matrix divergence: Λ^IW = tr(Ĉ) - ln det(Ĉ) - N
        sign, logdet = np.linalg.slogdet(self.C_hat)
        if sign <= 0:
            # Matrix is not positive definite (numerical issue)
            # Set a large divergence to trigger alarm
            self.lambda_iw = 1e6
        else:
            self.lambda_iw = np.trace(self.C_hat) - logdet - self.N

        # Test statistic: W · Λ^IW
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
                - 'test_stat': (T,) test statistic W·Λ^IW
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
# Combined IWD ∨ CUSUM decision (Eq. 13)
# ======================================================================

def combined_alarm(cusum_alarm: np.ndarray, iswt_alarm: bool) -> bool:
    """Combined IWD∨CUSUM authentication decision (Eq. 13).

    a(t) = 1[max_i G_i(t) > h] ∨ a^IWD(t)

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

    a(t) = 1[max_i G_i(t) > h] ∨ a^IWD(t) ∨ a^LSTM(t)

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

def iswt_torch(std_innovations, W=200, alpha=0.05, n_sensors=6):
    """Differentiable ISWT computation in PyTorch.

    Computes the Stein divergence over a sliding window for gradient flow.

    Args:
        std_innovations: (T, N) tensor of standardized innovations.
        W: Window size.
        alpha: Significance level.
        n_sensors: Number of sensors.

    Returns:
        lambda_iw: (T,) tensor of Stein divergence values.
        test_stat: (T,) tensor of W·Λ^IW values.
    """
    import torch

    T, N = std_innovations.shape
    dof = N * (N + 1) // 2
    critical = chi2.ppf(1 - alpha, dof)

    lambda_iw_list = []

    for t in range(T):
        if t < W - 1:
            # Not enough data
            lambda_iw_list.append(torch.tensor(0.0, dtype=std_innovations.dtype,
                                                device=std_innovations.device))
            continue

        # Window of innovations [t-W+1, ..., t]
        window = std_innovations[t - W + 1: t + 1]  # (W, N)

        # Sample covariance
        C_hat = (window.T @ window) / W

        # Regularize
        C_hat = C_hat + torch.eye(N, dtype=std_innovations.dtype,
                                   device=std_innovations.device) * 1e-10

        # Stein divergence: tr(C) - ln det(C) - N
        tr_C = torch.trace(C_hat)
        logdet_C = torch.logdet(C_hat)

        lambda_val = tr_C - logdet_C - N
        lambda_iw_list.append(lambda_val)

    return torch.stack(lambda_iw_list)
