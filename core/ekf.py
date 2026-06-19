"""
Extended Kalman Filter for DT state estimation.

Implements the EKF described in Sec. 2.1 of the paper:
    - Process model:  x(t) = f_p(x(t-1), u(t-1)) + w(t),  w ~ N(0, Q)
    - Measurement:    y(t) = h(x(t)) + η(t),               η ~ N(0, R)
    - Innovation:     ν_i(t) = y_i(t) - ĥ_i(x̂(t|t-1))     [Eq. 3]
    - Innovation cov: S_i(t) = H_i P(t|t-1) H_i^T + σ_i²  [Sec. 2.1]

Two implementations:
    1. NumPy EKF   — used by the DT service and offline experiments
    2. PyTorch EKF — differentiable forward pass for TCA white-box
"""

import numpy as np
from typing import Optional, Tuple
from .config import SystemConfig, EKFConfig
from .process_model import TwoTankProcess


class ExtendedKalmanFilter:
    """EKF for the two-tank process with direct measurement (H = I).

    The DT runs this filter to produce:
        - Posterior state estimate x̂(t)
        - Prior prediction x̂(t|t-1)
        - Innovation ν(t) = y(t) - x̂(t|t-1)
        - Innovation covariance S(t) = P(t|t-1) + R
        - Kalman gain K(t)
    """

    def __init__(self, sys_config: Optional[SystemConfig] = None,
                 ekf_config: Optional[EKFConfig] = None):
        self.sys = sys_config or SystemConfig()
        self.ekf = ekf_config or EKFConfig()
        self.process = TwoTankProcess(self.sys)

        # Dimensions
        self.n = self.sys.n_states
        self.m = self.sys.n_sensors

        # Measurement Jacobian (identity for direct measurement)
        self.H = np.eye(self.n)

        # Noise covariances
        self.Q = self.ekf.Q.copy()
        self.R = self.sys.R.copy()

        # State estimate and covariance
        self.x_hat = self.sys.x0.copy()           # Posterior estimate
        self.P = self.ekf.P0.copy()                # Posterior covariance

        # Prior (prediction) quantities
        self.x_pred = self.x_hat.copy()            # x̂(t|t-1)
        self.P_pred = self.P.copy()                # P(t|t-1)

        # Innovation quantities
        self.innovation = np.zeros(self.m)         # ν(t)
        self.S = self.R.copy()                     # Innovation covariance
        self.K = np.zeros((self.n, self.m))         # Kalman gain

        # Last control input
        self._u = np.array([1.0, self.sys.valve_open])

    def reset(self, x0: Optional[np.ndarray] = None,
              P0: Optional[np.ndarray] = None):
        """Reset the filter state."""
        self.x_hat = x0.copy() if x0 is not None else self.sys.x0.copy()
        self.P = P0.copy() if P0 is not None else self.ekf.P0.copy()
        self.x_pred = self.x_hat.copy()
        self.P_pred = self.P.copy()
        self.innovation = np.zeros(self.m)
        self.S = self.R.copy()
        self.K = np.zeros((self.n, self.m))

    def set_noise_covariances(self, Q: np.ndarray, R: np.ndarray):
        """Update noise covariances (after calibration)."""
        self.Q = Q.copy()
        self.R = R.copy()

    def predict(self, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Prediction step: x̂(t|t-1), P(t|t-1).

        Uses the process model (RK4) for state prediction and the
        analytical Jacobian for covariance prediction.

        Args:
            u: Control input [u_pump, u_valve].

        Returns:
            x_pred: Prior state prediction.
            P_pred: Prior error covariance.
        """
        self._u = u.copy()

        # State prediction via process model (RK4 integration)
        self.x_pred = self.process.integrate(self.x_hat, u)

        # Covariance prediction via discrete Jacobian
        # Φ ≈ I + F·dt + (F·dt)²/2
        Phi = self.process.discrete_jacobian(self.x_hat, u)
        self.P_pred = Phi @ self.P @ Phi.T + self.Q

        return self.x_pred.copy(), self.P_pred.copy()

    def update(self, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Update step: incorporate measurement y(t).

        Computes innovation, Kalman gain, and posterior state estimate.

        Args:
            y: Measurement vector at time t.

        Returns:
            x_hat: Posterior state estimate.
            innovation: Innovation vector ν(t) = y - H·x̂(t|t-1).
        """
        # Innovation: ν(t) = y(t) - H·x̂(t|t-1)
        # Since H = I: ν(t) = y(t) - x̂(t|t-1)
        self.innovation = y - self.H @ self.x_pred

        # Innovation covariance: S = H·P(t|t-1)·H^T + R
        # Since H = I: S = P(t|t-1) + R
        self.S = self.H @ self.P_pred @ self.H.T + self.R

        # Kalman gain: K = P(t|t-1)·H^T·S^{-1}
        self.K = self.P_pred @ self.H.T @ np.linalg.inv(self.S)

        # Posterior state update: x̂(t) = x̂(t|t-1) + K·ν(t)
        self.x_hat = self.x_pred + self.K @ self.innovation

        # Posterior covariance (Joseph form for numerical stability)
        # P = (I - K·H)·P_pred·(I - K·H)^T + K·R·K^T
        IKH = np.eye(self.n) - self.K @ self.H
        self.P = IKH @ self.P_pred @ IKH.T + self.K @ self.R @ self.K.T

        return self.x_hat.copy(), self.innovation.copy()

    def step(self, y: np.ndarray,
             u: np.ndarray) -> dict:
        """Full EKF step: predict + update.

        Args:
            y: Measurement vector at time t.
            u: Control input at time t-1.

        Returns:
            Dictionary with:
                - 'x_hat': posterior state estimate
                - 'x_pred': prior prediction
                - 'innovation': ν(t) = y - x̂(t|t-1)
                - 'S': innovation covariance matrix
                - 'S_diag': diagonal of S (per-sensor innovation variance)
                - 'K': Kalman gain
                - 'P': posterior error covariance
                - 'std_innovation': standardized innovations ν̂_i = ν_i/√S_ii
        """
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

    # ------------------------------------------------------------------
    # Batch processing (for offline experiments)
    # ------------------------------------------------------------------

    def run_batch(self, Y: np.ndarray, U: np.ndarray,
                  x0: Optional[np.ndarray] = None,
                  P0: Optional[np.ndarray] = None) -> dict:
        """Run the EKF over a batch of measurements.

        Args:
            Y: (T, N) measurement matrix.
            U: (T, 2) control input matrix.
            x0: Initial state estimate (optional).
            P0: Initial error covariance (optional).

        Returns:
            Dictionary with (T, ...) arrays for all EKF outputs.
        """
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


# ======================================================================
# PyTorch Differentiable EKF (for TCA white-box mode)
# ======================================================================

def _torch_available():
    try:
        import torch
        return True
    except ImportError:
        return False


class DifferentiableEKF:
    """PyTorch-based EKF for automatic differentiation.

    This class wraps the same EKF mathematics using PyTorch tensors,
    enabling gradient computation ∂SDS/∂δ via backpropagation through
    the entire EKF forward pass.

    Used exclusively by TCA white-box mode.
    """

    def __init__(self, sys_config: Optional[SystemConfig] = None,
                 ekf_config: Optional[EKFConfig] = None):
        import torch

        self.sys = sys_config or SystemConfig()
        self.ekf = ekf_config or EKFConfig()
        self.dtype = torch.float64
        self.device = torch.device('cpu')

        self.n = self.sys.n_states
        self.m = self.sys.n_sensors

        # Convert parameters to tensors
        self.Q = torch.tensor(self.ekf.Q, dtype=self.dtype, device=self.device)
        self.R = torch.tensor(self.sys.R, dtype=self.dtype, device=self.device)
        self.H = torch.eye(self.n, dtype=self.dtype, device=self.device)
        self.x0 = torch.tensor(self.sys.x0, dtype=self.dtype, device=self.device)
        self.P0 = torch.tensor(self.ekf.P0, dtype=self.dtype, device=self.device)

        # Process model parameters
        self._A1 = self.sys.A1
        self._A2 = self.sys.A2
        self._Cv12 = self.sys.Cv12
        self._Cv_out = self.sys.Cv_out
        self._H0 = self.sys.H0
        self._a_pump = self.sys.a_pump
        self._Q_nom = self.sys.Q_nom
        self._tau_p = self.sys.tau_p
        self._tau_q = self.sys.tau_q
        self._dt_sample = self.sys.dt_sample
        self._EPS = 1e-8

    def _f_torch(self, x, u):
        """Differentiable process model (Euler integration over dt_sample).

        Uses Euler integration (not RK4) for gradient computation
        simplicity; the approximation error is small for dt_sample = 1s
        with the relaxation dynamics.
        """
        import torch

        L1 = torch.clamp(x[0], min=self._EPS)
        L2 = torch.clamp(x[1], min=self._EPS)
        P_in = x[2]
        P_out = x[3]
        Q12 = x[4]
        Q_pump = torch.clamp(x[5], min=0.0)
        u_pump, u_valve = u[0], u[1]

        dt = self._dt_sample

        # Steady-state targets
        Q_out = self._Cv_out * u_valve * torch.sqrt(L2 + self._EPS)
        P_in_ss = L1
        H_pump = torch.clamp(self._H0 - self._a_pump * Q_pump ** 2, min=0.0)
        P_out_ss = P_in + H_pump
        dL = L1 - L2
        Q12_ss = self._Cv12 * torch.sign(dL) * torch.sqrt(torch.abs(dL) + self._EPS)
        Q_pump_ss = u_pump * self._Q_nom

        # Euler integration
        L1_next = L1 + dt * (Q_pump - Q12) / self._A1
        L2_next = L2 + dt * (Q12 - Q_out) / self._A2
        P_in_next = P_in + dt * (P_in_ss - P_in) / self._tau_p
        P_out_next = P_out + dt * (P_out_ss - P_out) / self._tau_p
        Q12_next = Q12 + dt * (Q12_ss - Q12) / self._tau_q
        Q_pump_next = Q_pump + dt * (Q_pump_ss - Q_pump) / self._tau_q

        return torch.stack([L1_next, L2_next, P_in_next, P_out_next,
                            Q12_next, Q_pump_next])

    def _jacobian_torch(self, x, u):
        """Compute discrete Jacobian using torch.autograd.functional.jacobian."""
        import torch

        def f_wrap(x_in):
            return self._f_torch(x_in, u)

        return torch.autograd.functional.jacobian(f_wrap, x)

    def forward_pass(self, Y, U, delta=None):
        """Run the full EKF forward pass with optional perturbation.

        All operations are differentiable with respect to delta.

        Args:
            Y: (T, N) measurement tensor (raw, without perturbation).
            U: (T, 2) control input tensor.
            delta: (T, N) perturbation tensor (requires_grad=True for TCA).
                   Only entries for attacked sensors should be non-zero.

        Returns:
            Dictionary with:
                - 'innovations': (T, N) innovation tensor
                - 'std_innovations': (T, N) standardized innovations
                - 'S_diag': (T, N) innovation variance diagonal
                - 'K': list of (N, N) Kalman gain tensors
        """
        import torch

        T, N = Y.shape
        assert N == self.n

        # Apply perturbation
        if delta is not None:
            Y_pert = Y + delta
        else:
            Y_pert = Y

        # Initialize
        x_hat = self.x0.clone()
        P = self.P0.clone()

        innovations = []
        std_innovations = []
        S_diags = []
        K_list = []

        for t in range(T):
            u = U[t]
            y = Y_pert[t]

            # Predict
            x_pred = self._f_torch(x_hat, u)
            Phi = self._jacobian_torch(x_hat, u)
            P_pred = Phi @ P @ Phi.T + self.Q

            # Innovation
            nu = y - x_pred  # H = I

            # Innovation covariance
            S = P_pred + self.R  # H = I
            S_diag = torch.diag(S)

            # Kalman gain
            K = P_pred @ torch.linalg.inv(S)

            # Update
            x_hat = x_pred + K @ nu
            IKH = torch.eye(self.n, dtype=self.dtype, device=self.device) - K
            P = IKH @ P_pred @ IKH.T + K @ self.R @ K.T

            # Standardized innovation
            std_nu = nu / torch.sqrt(torch.clamp(S_diag, min=1e-12))

            innovations.append(nu)
            std_innovations.append(std_nu)
            S_diags.append(S_diag)
            K_list.append(K)

        return {
            'innovations': torch.stack(innovations),
            'std_innovations': torch.stack(std_innovations),
            'S_diag': torch.stack(S_diags),
            'K': K_list,
        }
