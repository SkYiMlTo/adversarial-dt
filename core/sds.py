"""
Sensor Deception Score (SDS) metric.

Implements the SDS from Sec. 4.2 (Eqs. 9–11):

    φ_i(δ) = max(0, 1 - G_i(t;δ) / h)     ∈ [0,1]   CUSUM evasion    [Eq. 9]
    ψ(δ)   = max(0, 1 - W·Λ^IW(t;δ) / χ²)  ∈ [0,1]   Whiteness pres.  [Eq. 10]
    SDS(δ) = ψ(δ) · min_i φ_i(δ)           ∈ [0,1]                   [Eq. 11]

Properties:
    - SDS = 1  ⟺  all CUSUM at 0 AND innovation covariance = I_N
    - SDS = 0  ⟺  any CUSUM alarm OR ISWT alarm
    - Multiplicative form reflects AND structure of the deception objective
"""

import numpy as np
from typing import Optional
from scipy.stats import chi2


def compute_phi(G: np.ndarray, h: float = 5.0) -> np.ndarray:
    """CUSUM evasion component (Eq. 9).

    φ_i = max(0, 1 - G_i / h)

    Args:
        G: CUSUM statistics, shape (N,) or (T, N).
        h: CUSUM alarm threshold.

    Returns:
        phi: CUSUM evasion scores, same shape as G.
    """
    return np.maximum(0.0, 1.0 - G / h)


def compute_psi(test_stat: float, critical: float) -> float:
    """Whiteness preservation component (Eq. 10).

    ψ = max(0, 1 - W·Λ^IW / χ²_{1-α})

    Args:
        test_stat: W · Λ^IW (ISWT test statistic).
        critical: Chi-squared critical value.

    Returns:
        psi: Whiteness preservation score ∈ [0, 1].
    """
    return max(0.0, 1.0 - test_stat / critical)


def compute_sds(G: np.ndarray, test_stat: float,
                compromised_idx: np.ndarray,
                h: float = 5.0,
                n_sensors: int = 6,
                alpha: float = 0.05,
                custom_critical: Optional[float] = None) -> dict:
    """Compute the Sensor Deception Score (Eq. 11).

    SDS = ψ · min_i φ_i

    Args:
        G: Per-sensor CUSUM statistics, shape (N,).
        test_stat: ISWT test statistic (W · Λ^IW).
        compromised_idx: Indices of compromised sensors (B).
        h: CUSUM alarm threshold.
        n_sensors: Total number of sensors (N).
        alpha: ISWT significance level.
        custom_critical: Calibrated critical value.

    Returns:
        Dictionary with:
            - 'sds': Sensor Deception Score ∈ [0, 1]
            - 'phi': per-sensor CUSUM evasion scores
            - 'phi_mean': average CUSUM evasion for compromised sensors
            - 'psi': whiteness preservation score
    """
    if custom_critical is not None:
        critical = custom_critical
    else:
        # Degrees of freedom for ISWT (full covariance matrix)
        dof = n_sensors * (n_sensors + 1) // 2
        critical = chi2.ppf(1 - alpha, dof)

    # CUSUM evasion per sensor
    phi = compute_phi(G, h)

    # Evasion fails if ANY sensor alarms
    phi_min = np.min(phi)

    # Whiteness preservation
    psi = compute_psi(test_stat, critical)

    # SDS = ψ · min(φ_i)
    sds = psi * phi_min

    return {
        'sds': sds,
        'phi': phi,
        'phi_mean': phi_min,  # Kept as phi_mean key for compatibility
        'psi': psi,
        'critical': critical,
    }


def compute_sds_timeseries(G_all: np.ndarray,
                           test_stat_all: np.ndarray,
                           compromised_idx: np.ndarray,
                           h: float = 5.0,
                           n_sensors: int = 6,
                           alpha: float = 0.05,
                           custom_critical: Optional[float] = None) -> dict:
    """Compute SDS over a time series.

    Args:
        G_all: (T, N) CUSUM statistics.
        test_stat_all: (T,) ISWT test statistics.
        compromised_idx: Indices of compromised sensors.
        h: CUSUM alarm threshold.
        n_sensors: Total number of sensors.
        alpha: ISWT significance level.
        custom_critical: Empirically calibrated threshold.

    Returns:
        Dictionary with (T,) arrays.
    """
    T = G_all.shape[0]
    sds_all = np.zeros(T)
    phi_mean_all = np.zeros(T)
    psi_all = np.zeros(T)

    for t in range(T):
        result = compute_sds(G_all[t], test_stat_all[t],
                             compromised_idx, h, n_sensors, alpha, custom_critical)
        sds_all[t] = result['sds']
        phi_mean_all[t] = result['phi_mean']
        psi_all[t] = result['psi']

    return {
        'sds': sds_all,
        'phi_mean': phi_mean_all,
        'psi': psi_all,
        'sds_mean': np.mean(sds_all),
    }


# ======================================================================
# PyTorch-differentiable SDS (for TCA optimization objective)
# ======================================================================

def sds_torch(G, lambda_iw, compromised_idx, h=5.0,
              n_sensors=6, alpha=0.05, W=200, custom_critical=None):
    """Differentiable SDS for TCA optimization.

    Args:
        G: (T, N) CUSUM tensor.
        lambda_iw: (T,) Stein divergence tensor.
        compromised_idx: list of compromised sensor indices.
        h: CUSUM threshold.
        n_sensors: Number of sensors.
        alpha: Significance level.
        W: ISWT window size.
        custom_critical: Empirically calibrated threshold.

    Returns:
        sds_mean: Scalar mean SDS (optimization objective).
    """
    import torch

    if custom_critical is not None:
        critical = custom_critical
    else:
        dof = n_sensors * (n_sensors + 1) // 2
        critical = chi2.ppf(1 - alpha, dof)

    # φ_i = clamp(1 - G_i/h, min=0)
    phi = torch.clamp(1.0 - G / h, min=0.0)  # (T, N)

    # Evasion fails if ANY sensor alarms
    phi_min = phi.min(dim=1)[0]                # (T,)

    # ψ = clamp(1 - W·Λ^IW / χ², min=0)
    psi = torch.clamp(1.0 - W * lambda_iw / critical, min=0.0)  # (T,)

    # SDS = ψ · min(φ)
    sds = psi * phi_min  # (T,)

    # Mean SDS over time (optimization objective)
    # Skip the first W steps where ISWT is not yet ready
    valid_start = W
    if sds.shape[0] > valid_start:
        sds_mean = sds[valid_start:].mean()
    else:
        sds_mean = sds.mean()

    return sds_mean
