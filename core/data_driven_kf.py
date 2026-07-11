"""
Data-driven Kalman Filter for external dataset evaluation.

Replaces the physics-based two-tank process model with a linear
state-space model learned from training data via ridge regression:

    x_{t+1} = A @ x_t + B @ u_t + w_t,   w ~ N(0, Q)
    y_t     = x_t + v_t,                  v ~ N(0, R)

The learned A, B matrices encode the dynamics of any CPS dataset
(BATADAL, SWaT, etc.) without requiring manual physical parameter
specification. The Kalman Filter framework, CUSUM, ISWT, TCA,
and SDS all operate identically on the resulting innovation sequence.

Key advantage: the Jacobian of the linear model is simply A (constant),
which makes gradient computation for TCA trivial and exact.

Reference: Ljung (1999), System Identification: Theory for the User.
"""

import numpy as np
from typing import Optional, Tuple
from .config import SystemConfig, EKFConfig


class DataDrivenKalmanFilter:
    """Linear Kalman Filter with data-driven state-space model.

    Provides the same interface as ExtendedKalmanFilter so it can be
    used as a drop-in replacement throughout the experiment pipeline.
    """

    def __init__(self, A: np.ndarray, B: np.ndarray,
                 Q: np.ndarray, R: np.ndarray,
                 x0: Optional[np.ndarray] = None,
                 P0: Optional[np.ndarray] = None):
        """
        Args:
            A: (N, N) state transition matrix (learned).
            B: (N, M) input matrix (learned).
            Q: (N, N) process noise covariance (learned).
            R: (N, N) measurement noise covariance (learned).
            x0: (N,) initial state estimate.
            P0: (N, N) initial error covariance.
        """
        self.A = A.copy()
        self.B = B.copy()

        self.n = A.shape[0]
        self.m = self.n  # H = I, so n_sensors = n_states

        self.H = np.eye(self.n)
        self.Q = Q.copy()
        self.R = R.copy()

        self._x0 = x0.copy() if x0 is not None else np.zeros(self.n)
        self._P0 = P0.copy() if P0 is not None else np.eye(self.n) * 0.1

        self.x_hat = self._x0.copy()
        self.P = self._P0.copy()
        self.x_pred = self.x_hat.copy()
        self.P_pred = self.P.copy()
        self.innovation = np.zeros(self.m)
        self.S = self.R.copy()
        self.K = np.zeros((self.n, self.m))

    def reset(self, x0: Optional[np.ndarray] = None,
              P0: Optional[np.ndarray] = None):
        """Reset the filter state."""
        self.x_hat = x0.copy() if x0 is not None else self._x0.copy()
        self.P = P0.copy() if P0 is not None else self._P0.copy()
        self.x_pred = self.x_hat.copy()
        self.P_pred = self.P.copy()
        self.innovation = np.zeros(self.m)
        self.S = self.R.copy()
        self.K = np.zeros((self.n, self.m))

    def set_noise_covariances(self, Q: np.ndarray, R: np.ndarray):
        """Update noise covariances."""
        self.Q = Q.copy()
        self.R = R.copy()

    def predict(self, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Prediction step using learned linear model.

        x_pred = A @ x_hat + B @ u
        P_pred = A @ P @ A^T + Q
        """
        self.x_pred = self.A @ self.x_hat + self.B @ u
        self.P_pred = self.A @ self.P @ self.A.T + self.Q
        return self.x_pred.copy(), self.P_pred.copy()

    def update(self, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Update step: incorporate measurement y(t)."""
        self.innovation = y - self.H @ self.x_pred
        self.S = self.H @ self.P_pred @ self.H.T + self.R
        self.K = self.P_pred @ self.H.T @ np.linalg.inv(self.S)
        self.x_hat = self.x_pred + self.K @ self.innovation
        IKH = np.eye(self.n) - self.K @ self.H
        self.P = IKH @ self.P_pred @ IKH.T + self.K @ self.R @ self.K.T
        return self.x_hat.copy(), self.innovation.copy()

    def step(self, y: np.ndarray, u: np.ndarray) -> dict:
        """Full KF step: predict + update."""
        self.predict(u)
        self.update(y)

        S_diag = np.diag(self.S)
        std_innov = self.innovation / np.sqrt(np.maximum(S_diag, 1e-12))

        return {
            'x_hat': self.x_hat.copy(),
            'x_pred': self.x_pred.copy(),
            'innovation': self.innovation.copy(),
            'S': self.S.copy(),
            'S_diag': S_diag.copy(),
            'K': self.K.copy(),
            'P': self.P.copy(),
            'std_innovation': std_innov.copy(),
        }

    def run_batch(self, Y: np.ndarray, U: np.ndarray,
                  x0: Optional[np.ndarray] = None,
                  P0: Optional[np.ndarray] = None) -> dict:
        """Run the KF over a batch of measurements."""
        self.reset(x0, P0)

        T, N = Y.shape
        results = {
            'x_hat': np.zeros((T, self.n)),
            'x_pred': np.zeros((T, self.n)),
            'innovation': np.zeros((T, N)),
            'S_diag': np.zeros((T, N)),
            'std_innovation': np.zeros((T, N)),
            'K': np.zeros((T, self.n, N)),
        }

        for t in range(T):
            out = self.step(Y[t], U[t])
            results['x_hat'][t] = out['x_hat']
            results['x_pred'][t] = out['x_pred']
            results['innovation'][t] = out['innovation']
            results['S_diag'][t] = out['S_diag']
            results['std_innovation'][t] = out['std_innovation']
            results['K'][t] = out['K']

        return results


# ==================================================================
# System identification: learn A, B, Q, R from training data
# ==================================================================

def identify_linear_system(Y_train: np.ndarray,
                           U_train: np.ndarray,
                           alpha: float = 0.5,
                           r_ratio: float = 0.1) -> dict:
    """Learn a linear state-space model from training data.

    Fits x_{t+1} = A @ x_t + B @ u_t via ridge regression on
    z-score normalized sensor data.

    Args:
        Y_train: (T, N) raw sensor measurements (clean, normal).
        U_train: (T, M) actuator inputs.
        alpha: Ridge regularization strength. Higher values
               ensure spectral radius < 1 (model stability).
        r_ratio: Fraction of process noise variance attributed
                 to measurement noise.

    Returns:
        Dictionary with:
            - 'A', 'B', 'Q', 'R': learned model parameters
            - 'y_mean', 'y_std': normalization statistics
            - 'u_mean', 'u_std': normalization statistics
            - 'spectral_radius': max |eigenvalue(A)|
            - 'residual_std': per-sensor residual standard deviation
    """
    N = Y_train.shape[1]
    M = U_train.shape[1]

    # Z-score normalize
    y_mean = Y_train.mean(axis=0)
    y_std = np.maximum(Y_train.std(axis=0), 1e-10)
    u_mean = U_train.mean(axis=0)
    u_std = np.maximum(U_train.std(axis=0), 1e-10)

    Y_n = (Y_train - y_mean) / y_std
    U_n = (U_train - u_mean) / u_std

    # Ridge regression: x_{t+1} = A @ x_t + B @ u_t
    X_feat = np.hstack([Y_n[:-1], U_n[:-1]])  # (T-1, N+M)
    Y_tgt = Y_n[1:]                             # (T-1, N)

    XtX = X_feat.T @ X_feat
    reg = alpha * np.eye(XtX.shape[0])
    coeffs = np.linalg.solve(XtX + reg, X_feat.T @ Y_tgt)

    A = coeffs[:N].T
    B = coeffs[N:].T

    # Residuals -> process noise covariance
    residuals = Y_tgt - X_feat @ coeffs
    Q = np.cov(residuals.T)

    # Measurement noise: fraction of process noise
    R = np.diag(np.diag(Q) * r_ratio)

    sr = np.max(np.abs(np.linalg.eigvals(A)))
    res_std = np.std(residuals, axis=0)

    return {
        'A': A, 'B': B, 'Q': Q, 'R': R,
        'y_mean': y_mean, 'y_std': y_std,
        'u_mean': u_mean, 'u_std': u_std,
        'spectral_radius': sr,
        'residual_std': res_std,
    }


def normalize_data(Y: np.ndarray, U: np.ndarray,
                   sysid: dict) -> Tuple[np.ndarray, np.ndarray]:
    """Normalize data using training statistics from system identification."""
    Y_n = (Y - sysid['y_mean']) / sysid['y_std']
    U_n = (U - sysid['u_mean']) / sysid['u_std']
    return Y_n, U_n


# ==================================================================
# PyTorch DifferentiableEKF wrapper for data-driven model
# ==================================================================

class DifferentiableDataDrivenKF:
    """PyTorch-based data-driven KF for TCA gradient computation.

    Uses the same frozen-linearization approach as the physics-based
    DifferentiableEKF: run the NumPy KF on unperturbed data, then
    compute nu = Y + delta - x_pred where gradient flows through delta.
    """

    def __init__(self, dd_kf: DataDrivenKalmanFilter, device=None):
        import torch

        self.dd_kf = dd_kf
        self.n = dd_kf.n
        self.dtype = torch.float64

        if device is None:
            self.device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

    def forward_pass(self, Y, U, delta=None):
        """Run the KF forward pass with optional perturbation.

        Same interface as DifferentiableEKF.forward_pass().
        """
        import torch

        T, N = Y.shape
        assert N == self.n

        # Pass 1: NumPy KF on unperturbed data
        Y_np = Y.detach().cpu().numpy()
        U_np = U.detach().cpu().numpy()

        batch = self.dd_kf.run_batch(Y_np, U_np)

        x_pred_all = torch.tensor(batch['x_pred'], dtype=self.dtype,
                                  device=self.device)
        S_diag_all = torch.tensor(batch['S_diag'], dtype=self.dtype,
                                  device=self.device)

        # Pass 2: differentiable innovation computation
        if delta is not None:
            nu_all = Y + delta - x_pred_all
        else:
            nu_all = Y - x_pred_all

        std_nu_all = nu_all / torch.sqrt(
            torch.clamp(S_diag_all, min=1e-12))

        return {
            'innovations': nu_all,
            'std_innovations': std_nu_all,
            'S_diag': S_diag_all,
            'K': [],
        }
