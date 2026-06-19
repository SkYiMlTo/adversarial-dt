"""
Per-sensor CUSUM sequential detector.

Implements a two-sided CUSUM test based on Eq. 4 of the paper:

    G⁺_i(t) = max(0, G⁺_i(t-1) + ν̂_i(t) - k)
    G⁻_i(t) = max(0, G⁻_i(t-1) - ν̂_i(t) - k)
    G_i(t)  = max(G⁺_i(t), G⁻_i(t))

where ν̂_i(t) = ν_i(t) / √S_i(t) is the standardized innovation.

Under H₀, ν̂_i ~ N(0,1) so E[ν̂_i] = 0, giving zero drift in both
accumulators. With k = 0.5 and h = 5.0 the ARL₀ matches the
Wald-Lorden bound at e^{2kh} ≈ 148 steps per sensor.

A sensor is flagged when G_i(t) > h.
"""

import numpy as np
from typing import Optional
from .config import CUSUMConfig


class CUSUMDetector:
    """Two-sided per-sensor CUSUM detector.

    Monitors the signed standardized innovation for persistent mean
    shifts in either direction, indicative of sensor compromise.
    """

    def __init__(self, n_sensors: int,
                 config: Optional[CUSUMConfig] = None):
        """
        Args:
            n_sensors: Number of sensors to monitor.
            config: CUSUM parameters (k, h).
        """
        self.cfg = config or CUSUMConfig()
        self.n_sensors = n_sensors

        # Two-sided CUSUM accumulators
        self.G_plus = np.zeros(n_sensors)   # Upper (positive shift)
        self.G_minus = np.zeros(n_sensors)  # Lower (negative shift)

        # Combined statistic and alarm state
        self.G = np.zeros(n_sensors)
        self.alarm = np.zeros(n_sensors, dtype=bool)

    def reset(self):
        """Reset all CUSUM statistics to zero."""
        self.G_plus[:] = 0.0
        self.G_minus[:] = 0.0
        self.G[:] = 0.0
        self.alarm[:] = False

    def update(self, std_innovation: np.ndarray) -> dict:
        """Update CUSUM statistics with new standardized innovations.

        Args:
            std_innovation: ν̂(t) = ν(t) / √S(t) — standardized
                            innovation vector of length n_sensors
                            (signed values, zero-mean under H₀).

        Returns:
            Dictionary with:
                - 'G': current CUSUM statistics max(G⁺, G⁻)
                - 'alarm': per-sensor alarm flags
                - 'any_alarm': True if any sensor exceeds threshold
        """
        # Two-sided CUSUM update
        self.G_plus = np.maximum(0.0,
                                  self.G_plus + std_innovation - self.cfg.k)
        self.G_minus = np.maximum(0.0,
                                   self.G_minus - std_innovation - self.cfg.k)

        # Combined statistic
        self.G = np.maximum(self.G_plus, self.G_minus)

        # Alarm: G_i(t) > h
        self.alarm = self.G > self.cfg.h

        return {
            'G': self.G.copy(),
            'alarm': self.alarm.copy(),
            'any_alarm': np.any(self.alarm),
        }

    def run_batch(self, std_innovations: np.ndarray) -> dict:
        """Run CUSUM over a batch of standardized innovations.

        Args:
            std_innovations: (T, N) array of standardized innovations.

        Returns:
            Dictionary with (T, N) arrays:
                - 'G': CUSUM statistics at each timestep
                - 'alarm': alarm flags at each timestep
        """
        self.reset()
        T, N = std_innovations.shape
        assert N == self.n_sensors

        G_all = np.zeros((T, N))
        alarm_all = np.zeros((T, N), dtype=bool)

        for t in range(T):
            result = self.update(std_innovations[t])
            G_all[t] = result['G']
            alarm_all[t] = result['alarm']

        return {
            'G': G_all,
            'alarm': alarm_all,
        }


# ======================================================================
# PyTorch-differentiable CUSUM (for TCA)
# ======================================================================

def cusum_torch(std_innovations, k=0.5, h=5.0):
    """Differentiable two-sided CUSUM computation in PyTorch.

    Uses torch.clamp instead of np.maximum for gradient flow.
    Returns the combined G = max(G⁺, G⁻) trajectory for SDS.

    Args:
        std_innovations: (T, N) tensor of standardized innovations.
        k: Reference value.
        h: Alarm threshold.

    Returns:
        G: (T, N) tensor of CUSUM statistics.
    """
    import torch

    T, N = std_innovations.shape
    G_list = []
    G_plus_prev = torch.zeros(N, dtype=std_innovations.dtype,
                               device=std_innovations.device)
    G_minus_prev = torch.zeros(N, dtype=std_innovations.dtype,
                                device=std_innovations.device)

    for t in range(T):
        nu = std_innovations[t]
        G_plus = torch.clamp(G_plus_prev + nu - k, min=0.0)
        G_minus = torch.clamp(G_minus_prev - nu - k, min=0.0)
        G = torch.maximum(G_plus, G_minus)
        G_list.append(G)
        G_plus_prev = G_plus
        G_minus_prev = G_minus

    return torch.stack(G_list)
