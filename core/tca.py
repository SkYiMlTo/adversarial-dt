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
                     epsilon: np.ndarray,
                     iswt_weight: float = 2.0,
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
            epsilon: (N,) array Perturbation budget ‖δ_i‖_∞ ≤ ε_i (absolute units).
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
        
        # Convert epsilon to tensor for broadcasting
        eps_t = torch.tensor(epsilon, dtype=torch.float64, device=Y_t.device).unsqueeze(0)

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
            
            # Penalize maximum CUSUM across all sensors + ISWT globally
            cusum_penalty = torch.mean(torch.max(G, dim=1)[0]) / self.cusum_cfg.h
            iswt_penalty = torch.mean(self.iswt_cfg.W * lambda_iw) / critical
            
            surrogate_loss = -(cusum_penalty + iswt_weight * iswt_penalty)

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
                    delta_candidate = torch.max(torch.min(delta_candidate, eps_t), -eps_t)
                    # Simple step without full Armijo check for efficiency
                    break

                delta_new = torch.max(torch.min(delta + eta * grad, eps_t), -eps_t)
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
    # White-box TCA with neural defense (PGD on CUSUM + ISWT + LSTM)
    # ------------------------------------------------------------------

    def run_whitebox_neural(self,
                            Y: np.ndarray,
                            U: np.ndarray,
                            attacked_idx: List[int],
                            compromised_idx: List[int],
                            epsilon: np.ndarray,
                            lstm_model,
                            lstm_config=None,
                            iswt_weight: float = 2.0,
                            lstm_weight: float = 1.0,
                            verbose: bool = False) -> dict:
        """White-box TCA evading CUSUM + ISWT + LSTM simultaneously.

        Extends run_whitebox() by adding the LSTM reconstruction error
        to the surrogate loss function. Gradients flow through both
        the differentiable EKF and the LSTM autoencoder.

        The surrogate loss becomes:
            L = -(cusum_penalty + λ_iswt · iswt_penalty + λ_lstm · lstm_penalty)

        where lstm_penalty = mean(recon_error) / threshold.

        Args:
            Y: (T, N) raw measurement matrix.
            U: (T, 2) control input matrix.
            attacked_idx: Sensor indices the adversary can perturb.
            compromised_idx: Sensor indices with physical fault.
            epsilon: (N,) perturbation budget.
            lstm_model: Trained LSTMAutoencoder (nn.Module).
            lstm_config: LSTMDetectorConfig (for seq_len).
            iswt_weight: Weight for ISWT penalty.
            lstm_weight: Weight for LSTM penalty.
            verbose: Print progress.

        Returns:
            Dictionary with 'delta', 'sds_history', 'surr_history',
            'sds_final', 'lstm_score_history'.
        """
        import torch
        from .ekf import DifferentiableEKF
        from .cusum import cusum_torch
        from .iswt import iswt_torch
        from .sds import sds_torch
        from .lstm_detector import lstm_anomaly_torch
        from .config import LSTMDetectorConfig

        T, N = Y.shape
        cfg = self.tca_cfg
        lstm_cfg = lstm_config or LSTMDetectorConfig()

        # Convert to tensors
        Y_t = torch.tensor(Y, dtype=torch.float64)
        U_t = torch.tensor(U, dtype=torch.float64)

        # Initialize perturbation
        delta = torch.zeros(T, N, dtype=torch.float64, requires_grad=True)

        # Budget tensor
        eps_t = torch.tensor(epsilon, dtype=torch.float64).unsqueeze(0)

        # Attack mask
        mask = torch.zeros(N, dtype=torch.float64)
        mask[attacked_idx] = 1.0

        # Differentiable EKF
        diff_ekf = DifferentiableEKF(self.sys, self.ekf_cfg)

        # LSTM model in eval mode (but with gradient flow)
        lstm_model_f64 = _cast_lstm_to_float64(lstm_model)
        lstm_model_f64.eval()

        # Get LSTM threshold from training data (fallback: 0.01)
        lstm_threshold = getattr(lstm_model, '_threshold', 0.01)

        sds_history = []
        surr_history = []
        lstm_score_history = []
        best_delta = None
        best_sds = -1.0

        for k in range(cfg.K):
            if delta.grad is not None:
                delta.grad.zero_()

            delta_masked = delta * mask.unsqueeze(0)

            # EKF forward pass
            ekf_out = diff_ekf.forward_pass(Y_t, U_t, delta=delta_masked)

            # CUSUM
            G = cusum_torch(ekf_out['std_innovations'],
                            k=self.cusum_cfg.k, h=self.cusum_cfg.h)

            # ISWT
            lambda_iw = iswt_torch(ekf_out['std_innovations'],
                                   W=self.iswt_cfg.W,
                                   alpha=self.iswt_cfg.alpha,
                                   n_sensors=N)

            # LSTM anomaly scores (differentiable)
            lstm_scores = lstm_anomaly_torch(
                lstm_model_f64,
                ekf_out['std_innovations'],
                seq_len=lstm_cfg.seq_len)

            # True SDS (for tracking)
            with torch.no_grad():
                sds_val = sds_torch(G, lambda_iw, compromised_idx,
                                    h=self.cusum_cfg.h,
                                    n_sensors=N,
                                    alpha=self.iswt_cfg.alpha,
                                    W=self.iswt_cfg.W,
                                    custom_critical=self.iswt_cfg.critical_value(N))
                sds_scalar = sds_val.item()
                lstm_mean_score = lstm_scores[lstm_cfg.seq_len:].mean().item() \
                    if T > lstm_cfg.seq_len else 0.0

            # Surrogate loss: CUSUM + ISWT + LSTM
            critical = self.iswt_cfg.critical_value(N)
            cusum_penalty = torch.mean(torch.max(G, dim=1)[0]) / self.cusum_cfg.h
            iswt_penalty = torch.mean(self.iswt_cfg.W * lambda_iw) / critical

            # LSTM penalty: keep reconstruction error below threshold
            valid_lstm = lstm_scores[lstm_cfg.seq_len:]
            if len(valid_lstm) > 0:
                lstm_penalty = torch.mean(valid_lstm) / max(lstm_threshold, 1e-8)
            else:
                lstm_penalty = torch.tensor(0.0, dtype=torch.float64)

            surrogate_loss = -(cusum_penalty
                               + iswt_weight * iswt_penalty
                               + lstm_weight * lstm_penalty)

            surrogate_scalar = surrogate_loss.item()
            surr_history.append(surrogate_scalar)
            sds_history.append(sds_scalar)
            lstm_score_history.append(lstm_mean_score)

            if sds_scalar > best_sds:
                best_sds = sds_scalar
                best_delta = delta.detach().clone()

            if verbose and (k % 10 == 0 or k == cfg.K - 1):
                print(f"  TCA-Neural iter {k:3d}/{cfg.K}: "
                      f"SDS={sds_scalar:.4f}, Surr={surrogate_scalar:.4f}, "
                      f"LSTM={lstm_mean_score:.6f}")

            # Backward
            surrogate_loss.backward()

            if delta.grad is None:
                break

            # Projected gradient ascent
            with torch.no_grad():
                grad = delta.grad.clone()
                delta_new = delta + cfg.eta * grad
                delta_new = delta_new * mask.unsqueeze(0)
                delta_new = torch.max(torch.min(delta_new, eps_t), -eps_t)
                delta = delta_new.detach().requires_grad_(True)

        if best_delta is None:
            best_delta = delta.detach()

        return {
            'delta': best_delta.numpy(),
            'sds_history': sds_history,
            'surr_history': surr_history,
            'sds_final': best_sds,
            'lstm_score_history': lstm_score_history,
        }

    # ------------------------------------------------------------------
    # Grey-box TCA (finite differences)
    # ------------------------------------------------------------------

    def run_greybox(self,
                    Y: np.ndarray,
                    U: np.ndarray,
                    attacked_idx: List[int],
                    compromised_idx: List[int],
                    epsilon: np.ndarray,
                    iswt_weight: float = 2.0,
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

        # Convert epsilon to array
        if isinstance(epsilon, (float, int)):
            epsilon_vec = np.full(N, float(epsilon))
        else:
            epsilon_vec = np.asarray(epsilon)

        # Initialize perturbation
        delta = np.zeros((T, N))
        sds_history = []
        surr_history = []
        best_delta = None
        best_sds = -1.0

        # Create helper bound to iswt_weight
        def _surrogate(delta_eval):
            return self._evaluate_surrogate(Y, U, delta_eval, compromised_idx, iswt_weight)

        for k in range(cfg.K):
            # Evaluate current SDS and surrogate
            sds_base, surr_base = _surrogate(delta)
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
                                            -epsilon_vec[i], epsilon_vec[i])
                _, surr_plus = _surrogate(delta_plus)

                # Negative perturbation
                delta_minus = delta.copy()
                delta_minus[:, i] -= h_fd
                delta_minus[:, i] = np.clip(delta_minus[:, i],
                                             -epsilon_vec[i], epsilon_vec[i])
                _, surr_minus = _surrogate(delta_minus)

                # Central difference on surrogate
                grad[:, i] = (surr_plus - surr_minus) / (2 * h_fd)

            # Projected gradient ascent
            delta = delta + cfg.eta * grad

            # L∞ projection
            delta = np.clip(delta, -epsilon_vec, epsilon_vec)

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
                      compromised_idx: List[int],
                      iswt_weight: float = 2.0) -> tuple:
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

        # Surrogate loss components
        # Penalize maximum CUSUM across all sensors
        cusum_pen = np.mean(np.max(cusum_results['G'], axis=1)) / self.cusum_cfg.h
        
        # ISWT test_stat is ALREADY W * lambda_iw
        iswt_pen = np.mean(iswt_results['test_stat']) / critical
        surrogate = -(cusum_pen + iswt_weight * iswt_pen)

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
            epsilon = ratio * self.sys.sigma
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


# ======================================================================
# Helper: cast LSTM model to float64 for EKF compatibility
# ======================================================================

def _cast_lstm_to_float64(model):
    """Cast an LSTM model's parameters to float64.

    The differentiable EKF operates in float64 for numerical precision.
    The LSTM (trained in float32) must be cast to match dtypes for
    gradient flow through the combined EKF → LSTM pipeline.

    Args:
        model: LSTMAutoencoder (nn.Module) in float32.

    Returns:
        model_f64: Same model with all parameters in float64.
    """
    import copy
    model_f64 = copy.deepcopy(model)
    model_f64 = model_f64.double()
    return model_f64

