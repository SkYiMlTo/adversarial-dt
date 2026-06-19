"""
Targeted Consistency Attack (TCA).

Implements Algorithm 1 from Sec. 4.3:

    δ* = argmax_{‖δ(t)‖_∞ ≤ ε, ∀t}  SDS(δ)

TCA is a projected gradient ascent algorithm that maximizes SDS
subject to the physical plausibility budget:

    1. Perturb attacked sensor channels: ỹ_i(t) = y_i(t) + δ_i(t)
    2. Run EKF on perturbed measurements → innovations, S_diag
    3. Compute CUSUM statistics G_i(t)
    4. Compute ISWT statistic Λ^IW(t)
    5. Compute SDS from φ_i, ψ components
    6. Compute gradient ∇_δ SDS (autodiff or finite differences)
    7. Update: δ ← clip(δ + η·∇SDS, -ε, ε)

Two adversary regimes:
    - White-box:  Full EKF access, gradient via PyTorch autodiff
    - Grey-box:   Innovation-output-only, gradient via finite differences
"""

import numpy as np
from typing import Optional, List
from .config import SystemConfig, EKFConfig, CUSUMConfig, ISWTConfig, TCAConfig
from .ekf import ExtendedKalmanFilter
from .cusum import CUSUMDetector
from .iswt import ISWTDetector
from .sds import compute_sds_timeseries


class TargetedConsistencyAttack:
    """TCA: Projected Gradient Ascent on SDS.

    Computes the optimal perturbation sequence δ*(t) that maximizes
    the Sensor Deception Score within the ε-budget constraint.
    """

    def __init__(self,
                 sys_config: Optional[SystemConfig] = None,
                 ekf_config: Optional[EKFConfig] = None,
                 cusum_config: Optional[CUSUMConfig] = None,
                 iswt_config: Optional[ISWTConfig] = None,
                 tca_config: Optional[TCAConfig] = None):
        self.sys = sys_config or SystemConfig()
        self.ekf_cfg = ekf_config or EKFConfig()
        self.cusum_cfg = cusum_config or CUSUMConfig()
        self.iswt_cfg = iswt_config or ISWTConfig()
        self.tca_cfg = tca_config or TCAConfig()

    # ------------------------------------------------------------------
    # White-box TCA (PyTorch autodiff)
    # ------------------------------------------------------------------

    def run_whitebox(self,
                     Y: np.ndarray,
                     U: np.ndarray,
                     attacked_idx: List[int],
                     compromised_idx: List[int],
                     epsilon: float,
                     fault_vector: Optional[np.ndarray] = None,
                     verbose: bool = False) -> dict:
        """White-box TCA using PyTorch automatic differentiation.

        The adversary has full access to the EKF model, parameters,
        and internal state. Gradients are computed by backpropagating
        through the differentiable EKF forward pass.

        Args:
            Y: (T, N) raw measurement matrix (before perturbation).
            U: (T, 2) control input matrix.
            attacked_idx: Sensor indices the adversary can perturb (A).
            compromised_idx: Sensor indices with physical fault (B ⊆ A).
            epsilon: Perturbation budget ‖δ‖_∞ ≤ ε (absolute units).
            fault_vector: (T, N) fault added to measurements. If None,
                          assumes fault is already in Y.
            verbose: Print progress.

        Returns:
            Dictionary with:
                - 'delta': (T, N) optimal perturbation matrix
                - 'sds_history': SDS at each PGD iteration
                - 'sds_final': final mean SDS
        """
        import torch
        from .ekf import DifferentiableEKF
        from .cusum import cusum_torch
        from .iswt import iswt_torch
        from .sds import sds_torch

        T, N = Y.shape
        cfg = self.tca_cfg

        # Convert to tensors
        Y_t = torch.tensor(Y, dtype=torch.float64)
        U_t = torch.tensor(U, dtype=torch.float64)

        # Initialize perturbation (requires_grad for autodiff)
        delta = torch.zeros(T, N, dtype=torch.float64, requires_grad=True)

        # Create attack mask (only attacked sensors can be perturbed)
        mask = torch.zeros(N, dtype=torch.float64)
        mask[attacked_idx] = 1.0

        # Differentiable EKF
        diff_ekf = DifferentiableEKF(self.sys, self.ekf_cfg)

        sds_history = []
        surr_history = []
        best_delta = None
        best_sds = -1.0

        for k in range(cfg.K):
            # Zero gradient
            if delta.grad is not None:
                delta.grad.zero_()

            # Apply mask: only attacked sensors get perturbation
            delta_masked = delta * mask.unsqueeze(0)

            # Run EKF forward pass
            ekf_out = diff_ekf.forward_pass(Y_t, U_t, delta=delta_masked)

            # Compute CUSUM
            G = cusum_torch(ekf_out['std_innovations'],
                           k=self.cusum_cfg.k, h=self.cusum_cfg.h)

            # Compute ISWT
            lambda_iw = iswt_torch(ekf_out['std_innovations'],
                                    W=self.iswt_cfg.W,
                                    alpha=self.iswt_cfg.alpha,
                                    n_sensors=N)

            # True SDS metric (for tracking, requires no gradients)
            with torch.no_grad():
                sds_val = sds_torch(G, lambda_iw, compromised_idx,
                                    h=self.cusum_cfg.h,
                                    n_sensors=N,
                                    alpha=self.iswt_cfg.alpha,
                                    W=self.iswt_cfg.W,
                                    custom_critical=self.iswt_cfg.critical_value(N))
                sds_scalar = sds_val.item()

            # Smooth surrogate objective for PGD (additive, no clipping)
            # Maximize this: keep CUSUM low, keep ISWT low.
            critical = self.iswt_cfg.critical_value(N)
            
            # Penalize CUSUM of attacked sensors + ISWT globally
            cusum_penalty = torch.mean(G[:, compromised_idx]) / self.cusum_cfg.h
            iswt_penalty = torch.mean(self.iswt_cfg.W * lambda_iw) / critical
            
            surrogate_loss = -(cusum_penalty + 2.0 * iswt_penalty)

            surrogate_scalar = surrogate_loss.item()
            surr_history.append(surrogate_scalar)
            sds_history.append(sds_scalar)

            if sds_scalar > best_sds:
                best_sds = sds_scalar
                best_delta = delta.detach().clone()

            if verbose and (k % 10 == 0 or k == cfg.K - 1):
                print(f"  TCA iter {k:3d}/{cfg.K}: SDS = {sds_scalar:.4f}, Surr = {surrogate_scalar:.4f}")

            # Backward pass: compute ∇_δ surrogate_loss
            surrogate_loss.backward()

            if delta.grad is None:
                break

            # Projected gradient ascent with L∞ clipping
            with torch.no_grad():
                grad = delta.grad.clone()

                # Armijo line search for step size
                eta = cfg.eta
                for _ in range(10):
                    delta_candidate = delta + eta * grad
                    delta_candidate = delta_candidate * mask.unsqueeze(0)
                    delta_candidate = torch.clamp(delta_candidate,
                                                   -epsilon, epsilon)
                    # Simple step without full Armijo check for efficiency
                    break

                delta_new = torch.clamp(delta + eta * grad,
                                         -epsilon, epsilon)
                delta_new = delta_new * mask.unsqueeze(0)

                # Update (create new tensor with requires_grad)
                delta = delta_new.detach().requires_grad_(True)

        # Use best delta found
        if best_delta is None:
            best_delta = delta.detach()

        return {
            'delta': best_delta.numpy(),
            'sds_history': sds_history,
            'surr_history': surr_history,
            'sds_final': best_sds,
        }

    # ------------------------------------------------------------------
    # Grey-box TCA (finite differences)
    # ------------------------------------------------------------------

    def run_greybox(self,
                    Y: np.ndarray,
                    U: np.ndarray,
                    attacked_idx: List[int],
                    compromised_idx: List[int],
                    epsilon: float,
                    verbose: bool = False) -> dict:
        """Grey-box TCA using finite difference gradient estimation.

        The adversary observes only the EKF's innovation outputs
        (no access to internal model or state estimate). Gradients
        are estimated by perturbing each attacked sensor and measuring
        the SDS change.

        Query cost: O(|A| · T) per iteration.

        Args:
            Y: (T, N) measurement matrix.
            U: (T, 2) control inputs.
            attacked_idx: Sensor indices the adversary can perturb.
            compromised_idx: Sensor indices with physical fault.
            epsilon: Perturbation budget.
            verbose: Print progress.

        Returns:
            Same format as run_whitebox.
        """
        T, N = Y.shape
        cfg = self.tca_cfg

        # Initialize perturbation
        delta = np.zeros((T, N))
        sds_history = []
        surr_history = []
        best_delta = None
        best_sds = -1.0

        for k in range(cfg.K):
            # Evaluate current SDS and surrogate
            sds_base, surr_base = self._evaluate_surrogate(Y, U, delta, compromised_idx)
            sds_history.append(sds_base)
            surr_history.append(surr_base)

            if sds_base > best_sds:
                best_sds = sds_base
                best_delta = delta.copy()

            if verbose and (k % 10 == 0 or k == cfg.K - 1):
                print(f"  TCA-GB iter {k:3d}/{cfg.K}: SDS = {sds_base:.4f}, Surr = {surr_base:.4f}")

            # Estimate gradient via finite differences
            grad = np.zeros_like(delta)
            h_fd = cfg.fd_step

            for i in attacked_idx:
                # Positive perturbation
                delta_plus = delta.copy()
                delta_plus[:, i] += h_fd
                delta_plus[:, i] = np.clip(delta_plus[:, i],
                                            -epsilon, epsilon)
                _, surr_plus = self._evaluate_surrogate(Y, U, delta_plus,
                                              compromised_idx)

                # Negative perturbation
                delta_minus = delta.copy()
                delta_minus[:, i] -= h_fd
                delta_minus[:, i] = np.clip(delta_minus[:, i],
                                             -epsilon, epsilon)
                _, surr_minus = self._evaluate_surrogate(Y, U, delta_minus,
                                               compromised_idx)

                # Central difference on surrogate
                grad[:, i] = (surr_plus - surr_minus) / (2 * h_fd)

            # Projected gradient ascent
            delta = delta + cfg.eta * grad

            # L∞ projection
            delta = np.clip(delta, -epsilon, epsilon)

            # Zero non-attacked sensors
            mask = np.zeros(N)
            mask[attacked_idx] = 1.0
            delta = delta * mask[np.newaxis, :]

        if best_delta is None:
            best_delta = delta

        return {
            'delta': best_delta,
            'sds_history': sds_history,
            'surr_history': surr_history,
            'sds_final': best_sds,
        }

    # ------------------------------------------------------------------
    # SDS evaluation helper
    # ------------------------------------------------------------------

    def _evaluate_surrogate(self, Y: np.ndarray, U: np.ndarray,
                      delta: np.ndarray,
                      compromised_idx: List[int]) -> tuple:
        """Evaluate true SDS and surrogate loss for a given perturbation.

        Runs the full pipeline: EKF → CUSUM → ISWT → SDS/Surrogate.

        Args:
            Y: (T, N) raw measurements.
            U: (T, 2) control inputs.
            delta: (T, N) perturbation matrix.
            compromised_idx: list of compromised sensor indices.

        Returns:
            Mean SDS over the evaluation window.
        """
        T, N = Y.shape

        # Apply perturbation
        Y_pert = Y + delta

        # Run EKF
        ekf = ExtendedKalmanFilter(self.sys, self.ekf_cfg)
        ekf_results = ekf.run_batch(Y_pert, U)

        # Run CUSUM
        cusum = CUSUMDetector(N, self.cusum_cfg)
        cusum_results = cusum.run_batch(ekf_results['std_innovation'])

        # Run ISWT
        iswt = ISWTDetector(N, self.iswt_cfg)
        iswt_results = iswt.run_batch(ekf_results['std_innovation'])

        # Compute SDS time series
        sds_results = compute_sds_timeseries(
            cusum_results['G'],
            iswt_results['test_stat'],
            np.array(compromised_idx),
            h=self.cusum_cfg.h,
            n_sensors=N,
            alpha=self.iswt_cfg.alpha,
            custom_critical=self.iswt_cfg.critical_value(N)
        )

        critical = self.iswt_cfg.critical_value(N)

        cusum_pen = np.mean(cusum_results['G'][:, compromised_idx]) / self.cusum_cfg.h
        iswt_pen = np.mean(self.iswt_cfg.W * iswt_results['test_stat']) / critical
        surrogate = -(cusum_pen + 2.0 * iswt_pen)

        return sds_results['sds_mean'], surrogate

    # ------------------------------------------------------------------
    # Convenience: run attack with budget sweep
    # ------------------------------------------------------------------

    def sweep_budget(self,
                     Y: np.ndarray,
                     U: np.ndarray,
                     attacked_idx: List[int],
                     compromised_idx: List[int],
                     mode: str = 'whitebox',
                     verbose: bool = False) -> dict:
        """Run TCA across all budget levels in the sweep.

        Args:
            Y, U: Measurement and control data.
            attacked_idx, compromised_idx: Sensor indices.
            mode: 'whitebox' or 'greybox'.
            verbose: Print progress.

        Returns:
            Dictionary mapping ε/σ_η ratio to TCA result dict.
        """
        results = {}
        # Use the mean σ as reference for ε ratios
        sigma_ref = np.mean(self.sys.sigma[compromised_idx])

        for ratio in self.tca_cfg.epsilon_ratios:
            epsilon = ratio * sigma_ref
            if verbose:
                print(f"\n=== Budget ε/σ_η = {ratio:.2f} "
                      f"(ε = {epsilon:.6f}) ===")

            if mode == 'whitebox':
                result = self.run_whitebox(Y, U, attacked_idx,
                                           compromised_idx, epsilon,
                                           verbose=verbose)
            else:
                result = self.run_greybox(Y, U, attacked_idx,
                                          compromised_idx, epsilon,
                                          verbose=verbose)

            results[ratio] = result

        return results
