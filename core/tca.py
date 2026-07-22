"""
Targeted Consistency Attack (TCA).

Implements Algorithm 1 from Sec. 4.3:

    delta* = argmax_{||delta(t)||_inf <= eps, forall t}  SDS(delta)

TCA is a projected gradient ascent algorithm that maximizes SDS
subject to the physical plausibility budget:

    1. Perturb attacked sensor channels: y_tilde_i(t) = y_i(t) + delta_i(t)
    2. Run EKF on perturbed measurements -> innovations, S_diag
    3. Compute CUSUM statistics G_i(t)
    4. Compute ISWT statistic Lambda^IW(t)
    5. Compute SDS from phi_i, psi components
    6. Compute gradient nabla_delta SDS (autodiff or finite differences)
    7. Update: delta <- clip(delta + eta * nabla SDS, -eps, eps)

Two adversary regimes:
    - White-box:  Full EKF access, gradient via PyTorch autodiff
    - Grey-box:   Innovation-output-only, gradient via finite differences

Surrogate loss design:
    The true SDS objective contains hard clipping (max(0, ...)) which
    causes zero gradients in the alarmed regime. We use a smooth
    surrogate that provides gradient signal everywhere:

    L_surr = logsumexp(G / h) + w_iw * mean(exp(W*lambda_iw / chi2 - 1))

    This is minimized (via negation + gradient ascent) to keep CUSUM
    and ISWT statistics below their respective thresholds.

    Adam momentum prevents stalling in flat surrogate regions.
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

    Computes the optimal perturbation sequence delta*(t) that maximizes
    the Sensor Deception Score within the eps-budget constraint.
    """

    def __init__(self,
                 sys_config: SystemConfig,
                 ekf_config: EKFConfig,
                 cusum_config: CUSUMConfig,
                 iswt_config: ISWTConfig,
                 tca_config: TCAConfig,
                 baseline_cov: Optional[np.ndarray] = None,
                 kf_model=None):
        self.sys = sys_config
        self.ekf_cfg = ekf_config
        self.cusum_cfg = cusum_config
        self.iswt_cfg = iswt_config
        self.tca_cfg = tca_config
        self.baseline_cov = baseline_cov
        self.kf_model = kf_model

    # ------------------------------------------------------------------
    # Smooth surrogate loss (shared between whitebox and neural)
    # ------------------------------------------------------------------

    @staticmethod
    def _smooth_surrogate_torch(G, lambda_iw, h, W, critical,
                                iswt_weight=2.0, temperature=2.0):
        """Compute smooth surrogate loss for PGD optimization.

        Uses logsumexp over CUSUM statistics for smooth gradient flow
        across all sensors and timesteps, plus an exponential barrier
        for the ISWT statistic.

        The logsumexp provides a smooth approximation to the max
        function: logsumexp(x/tau) ~ max(x) as tau -> 0. With
        temperature tau=2.0, the surrogate is differentiable and the
        gradient signal reaches all sensors proportional to exp(G_i/tau).

        Args:
            G: (T, N) CUSUM statistics tensor.
            lambda_iw: (T,) Stein divergence tensor.
            h: CUSUM alarm threshold.
            W: ISWT window size.
            critical: chi-squared critical value.
            iswt_weight: Relative weight for the ISWT penalty.
            temperature: logsumexp temperature (lower = sharper).

        Returns:
            surrogate_loss: Scalar tensor (to be negated for ascent).
        """
        import torch

        T, N = G.shape

        # CUSUM penalty: logsumexp across sensors at each timestep,
        # then mean across time. Normalized by threshold h.
        G_scaled = G / (h * temperature)
        cusum_lse = temperature * torch.logsumexp(G_scaled, dim=1)  # (T,)
        cusum_penalty = torch.mean(cusum_lse) / h

        # ISWT penalty: exponential barrier on W*lambda_iw / critical.
        # exp(ratio - 1) is ~0 when ratio << 1, and grows exponentially
        # when the test statistic approaches/exceeds the critical value.
        # Only consider timesteps where ISWT is valid (t >= W).
        valid_start = min(W, T)
        if T > valid_start:
            ratio = W * lambda_iw[valid_start:] / max(critical, 1e-6)
            ratio = torch.clamp(ratio - 1.0, min=-20.0, max=20.0)
            iswt_penalty = torch.mean(torch.exp(ratio))
        else:
            iswt_penalty = torch.tensor(0.0, dtype=G.dtype, device=G.device)

        # Combined surrogate (to be minimized by PGD)
        loss = cusum_penalty + iswt_weight * iswt_penalty

        return loss

    # ------------------------------------------------------------------
    # White-box TCA (PyTorch autodiff + Adam momentum)
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

        Uses Adam momentum to escape flat surrogate regions where
        vanilla PGD stalls.

        Args:
            Y: (T, N) raw measurement matrix (before perturbation).
            U: (T, 2) control input matrix.
            attacked_idx: Sensor indices the adversary can perturb (A).
            compromised_idx: Sensor indices with physical fault (B <= A).
            epsilon: (N,) array Perturbation budget ||delta_i||_inf <= eps_i.
            iswt_weight: Relative weight for ISWT in surrogate loss.
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

        # Auto-select device (GPU if available)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Convert to tensors on device
        Y_t = torch.tensor(Y, dtype=torch.float64, device=device)
        U_t = torch.tensor(U, dtype=torch.float64, device=device)

        # Initialize perturbation (small random for symmetry breaking)
        eps_np = np.asarray(epsilon)
        delta_init = np.random.default_rng(42).uniform(
            -0.01 * eps_np, 0.01 * eps_np, size=(T, N))
        # Zero non-attacked sensors
        attack_mask_np = np.zeros(N)
        attack_mask_np[attacked_idx] = 1.0
        delta_init *= attack_mask_np[np.newaxis, :]

        delta = torch.tensor(delta_init, dtype=torch.float64,
                             device=device, requires_grad=True)

        # Convert epsilon to tensor for broadcasting
        eps_t = torch.tensor(eps_np, dtype=torch.float64, device=device).unsqueeze(0)

        # Create attack mask
        mask = torch.zeros(N, dtype=torch.float64, device=device)
        mask[attacked_idx] = 1.0

        # Differentiable EKF or DataDrivenKF (on same device)
        if hasattr(self, 'kf_model') and self.kf_model is not None:
            from .data_driven_kf import DifferentiableDataDrivenKF
            diff_ekf = DifferentiableDataDrivenKF(self.kf_model, device=device)
        else:
            diff_ekf = DifferentiableEKF(self.sys, self.ekf_cfg, device=device)

        # Adam optimizer state
        m = torch.zeros_like(delta)  # First moment
        v = torch.zeros_like(delta)  # Second moment
        beta1, beta2 = 0.9, 0.999
        adam_eps = 1e-8

        sds_history = []
        surr_history = []
        best_delta = None
        best_sds = -1.0

        critical = self.iswt_cfg.critical_value(N)

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
                                    n_sensors=N,
                                    baseline_cov=self.baseline_cov)

            # True SDS metric (for tracking only, no gradients)
            with torch.no_grad():
                sds_val = sds_torch(G, lambda_iw, compromised_idx,
                                    h=self.cusum_cfg.h,
                                    n_sensors=N,
                                    alpha=self.iswt_cfg.alpha,
                                    W=self.iswt_cfg.W,
                                    custom_critical=critical)
                sds_scalar = sds_val.item()

            # Smooth surrogate loss
            surrogate_loss = self._smooth_surrogate_torch(
                G, lambda_iw,
                h=self.cusum_cfg.h,
                W=self.iswt_cfg.W,
                critical=critical,
                iswt_weight=iswt_weight,
                temperature=2.0)

            surrogate_scalar = surrogate_loss.item()
            surr_history.append(surrogate_scalar)
            sds_history.append(sds_scalar)

            if sds_scalar > best_sds:
                best_sds = sds_scalar
                best_delta = delta.detach().clone()

            if verbose and (k % 10 == 0 or k == cfg.K - 1):
                print(f"  TCA iter {k:3d}/{cfg.K}: "
                      f"SDS = {sds_scalar:.4f}, Surr = {surrogate_scalar:.4f}")

            # Backward pass: compute gradient of surrogate w.r.t. delta
            surrogate_loss.backward()

            if delta.grad is None:
                break

            # Adam-accelerated projected gradient descent (minimize surr)
            # which is equivalent to projected gradient ascent on SDS
            with torch.no_grad():
                grad = delta.grad.clone()

                # Adam moment updates
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * grad ** 2
                m_hat = m / (1 - beta1 ** (k + 1))
                v_hat = v / (1 - beta2 ** (k + 1))

                # Adam step (descending the surrogate = ascending SDS)
                step = cfg.eta * m_hat / (torch.sqrt(v_hat) + adam_eps)

                # PGD: delta <- delta - step (descent on surrogate)
                delta_new = delta - step
                delta_new = delta_new * mask.unsqueeze(0)
                delta_new = torch.clamp(delta_new, -eps_t, eps_t)

                # Update (create new tensor with requires_grad)
                delta = delta_new.detach().requires_grad_(True)

        # Use best delta found
        if best_delta is None:
            best_delta = delta.detach()

        return {
            'delta': best_delta.cpu().numpy(),
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
            L = logsumexp(G/h) + w_iw * exp_barrier(ISWT) + w_lstm * lstm_pen

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

        # Auto-select device (GPU if available)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Convert to tensors on device
        Y_t = torch.tensor(Y, dtype=torch.float64, device=device)
        U_t = torch.tensor(U, dtype=torch.float64, device=device)

        # Initialize perturbation with small random noise
        eps_np = np.asarray(epsilon)
        delta_init = np.random.default_rng(42).uniform(
            -0.01 * eps_np, 0.01 * eps_np, size=(T, N))
        attack_mask_np = np.zeros(N)
        attack_mask_np[attacked_idx] = 1.0
        delta_init *= attack_mask_np[np.newaxis, :]

        delta = torch.tensor(delta_init, dtype=torch.float64,
                             device=device, requires_grad=True)

        # Budget tensor
        eps_t = torch.tensor(eps_np, dtype=torch.float64, device=device).unsqueeze(0)

        # Attack mask
        mask = torch.zeros(N, dtype=torch.float64, device=device)
        mask[attacked_idx] = 1.0

        # Differentiable EKF (on same device)
        diff_ekf = DifferentiableEKF(self.sys, self.ekf_cfg, device=device)

        # LSTM model in eval mode (float64, on same device, with gradient flow)
        lstm_model_f64 = _cast_lstm_to_float64(lstm_model, device=device)
        lstm_model_f64.eval()

        # Get LSTM threshold from training data (fallback: 0.01)
        lstm_threshold = getattr(lstm_model, '_threshold', 0.01)

        # Adam optimizer state
        m = torch.zeros_like(delta)
        v = torch.zeros_like(delta)
        beta1, beta2 = 0.9, 0.999
        adam_eps = 1e-8

        critical = self.iswt_cfg.critical_value(N)

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
                                   n_sensors=N,
                                   baseline_cov=self.baseline_cov)

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
                                    custom_critical=critical)
                sds_scalar = sds_val.item()
                lstm_mean_score = lstm_scores[lstm_cfg.seq_len:].mean().item() \
                    if T > lstm_cfg.seq_len else 0.0

            # Base surrogate (CUSUM + ISWT)
            base_loss = self._smooth_surrogate_torch(
                G, lambda_iw,
                h=self.cusum_cfg.h,
                W=self.iswt_cfg.W,
                critical=critical,
                iswt_weight=iswt_weight)

            # LSTM penalty: keep reconstruction error below threshold
            valid_lstm = lstm_scores[lstm_cfg.seq_len:]
            if len(valid_lstm) > 0:
                lstm_penalty = torch.mean(valid_lstm) / max(lstm_threshold, 1e-8)
            else:
                lstm_penalty = torch.tensor(0.0, dtype=torch.float64,
                                            device=device)

            surrogate_loss = base_loss + lstm_weight * lstm_penalty

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

            # Adam-accelerated PGD
            with torch.no_grad():
                grad = delta.grad.clone()
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * grad ** 2
                m_hat = m / (1 - beta1 ** (k + 1))
                v_hat = v / (1 - beta2 ** (k + 1))
                step = cfg.eta * m_hat / (torch.sqrt(v_hat) + adam_eps)
                delta_new = delta - step
                delta_new = delta_new * mask.unsqueeze(0)
                delta_new = torch.clamp(delta_new, -eps_t, eps_t)
                delta = delta_new.detach().requires_grad_(True)

        if best_delta is None:
            best_delta = delta.detach()

        return {
            'delta': best_delta.cpu().numpy(),
            'sds_history': sds_history,
            'surr_history': surr_history,
            'sds_final': best_sds,
            'lstm_score_history': lstm_score_history,
        }

    # ------------------------------------------------------------------
    # Grey-box TCA (block-coordinate finite differences + restarts)
    # ------------------------------------------------------------------

    def run_greybox(self,
                    Y: np.ndarray,
                    U: np.ndarray,
                    attacked_idx: List[int],
                    compromised_idx: List[int],
                    epsilon: np.ndarray,
                    iswt_weight: float = 2.0,
                    n_restarts: int = 1,
                    verbose: bool = False) -> dict:
        """Grey-box TCA using block-coordinate finite difference gradients.

        The adversary observes only the EKF's innovation outputs
        (no access to internal model or state estimate). Gradients
        are estimated by perturbing temporal blocks per sensor and
        measuring the surrogate change.

        Improvements over uniform-shift FD:
        * Block-coordinate perturbation: perturbs blocks of ~20
          timesteps per sensor, giving temporal gradient resolution.
        * Random restarts: runs n_restarts independent optimizations
          from different initial perturbations, keeping the best.
        * Adam momentum on estimated gradients.

        Query cost: O(|A| * ceil(T/block_size) * K) per restart.

        Args:
            Y: (T, N) measurement matrix.
            U: (T, 2) control inputs.
            attacked_idx: Sensor indices the adversary can perturb.
            compromised_idx: Sensor indices with physical fault.
            epsilon: Perturbation budget.
            iswt_weight: Weight for ISWT in surrogate.
            n_restarts: Number of random restarts.
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

        # Attack mask
        mask = np.zeros(N)
        mask[attacked_idx] = 1.0

        # Block size for temporal resolution (larger = fewer FD evals)
        block_size = max(200, T // 3)

        best_delta_global = None
        best_sds_global = -1.0
        all_sds_history = []
        all_surr_history = []

        for restart in range(n_restarts):
            rng = np.random.default_rng(cfg.K * restart + 7)

            # Initialize with small random perturbation
            delta = rng.uniform(-0.1 * epsilon_vec, 0.1 * epsilon_vec,
                                size=(T, N))
            delta *= mask[np.newaxis, :]
            delta = np.clip(delta, -epsilon_vec, epsilon_vec)

            # Adam state
            m_adam = np.zeros_like(delta)
            v_adam = np.zeros_like(delta)
            beta1, beta2 = 0.9, 0.999
            adam_eps_val = 1e-8

            sds_history = []
            surr_history = []
            best_delta = None
            best_sds = -1.0

            for k in range(cfg.K_greybox):
                # Evaluate current SDS and surrogate
                sds_base, surr_base = self._evaluate_surrogate(
                    Y, U, delta, compromised_idx, iswt_weight)
                sds_history.append(sds_base)
                surr_history.append(surr_base)

                if sds_base > best_sds:
                    best_sds = sds_base
                    best_delta = delta.copy()

                if verbose and (k % 20 == 0 or k == cfg.K_greybox - 1):
                    tag = f"R{restart}" if n_restarts > 1 else "GB"
                    print(f"  TCA-{tag} iter {k:3d}/{cfg.K_greybox}: "
                          f"SDS = {sds_base:.4f}, Surr = {surr_base:.4f}")

                # Block-coordinate finite difference gradient estimation
                grad = np.zeros_like(delta)
                h_fd = cfg.fd_step

                for i in attacked_idx:
                    # Divide time axis into blocks
                    for b_start in range(0, T, block_size):
                        b_end = min(b_start + block_size, T)

                        # Positive perturbation on this block
                        delta_plus = delta.copy()
                        delta_plus[b_start:b_end, i] += h_fd
                        delta_plus[b_start:b_end, i] = np.clip(
                            delta_plus[b_start:b_end, i],
                            -epsilon_vec[i], epsilon_vec[i])
                        _, surr_plus = self._evaluate_surrogate(
                            Y, U, delta_plus, compromised_idx, iswt_weight)

                        # Negative perturbation on this block
                        delta_minus = delta.copy()
                        delta_minus[b_start:b_end, i] -= h_fd
                        delta_minus[b_start:b_end, i] = np.clip(
                            delta_minus[b_start:b_end, i],
                            -epsilon_vec[i], epsilon_vec[i])
                        _, surr_minus = self._evaluate_surrogate(
                            Y, U, delta_minus, compromised_idx, iswt_weight)

                        # Central difference (gradient of surrogate)
                        grad[b_start:b_end, i] = (
                            surr_plus - surr_minus) / (2 * h_fd)

                # Adam moment update
                m_adam = beta1 * m_adam + (1 - beta1) * grad
                v_adam = beta2 * v_adam + (1 - beta2) * grad ** 2
                m_hat = m_adam / (1 - beta1 ** (k + 1))
                v_hat = v_adam / (1 - beta2 ** (k + 1))

                # Adam step (descend surrogate = ascend SDS)
                step = cfg.eta * m_hat / (np.sqrt(v_hat) + adam_eps_val)
                delta = delta - step

                # L-inf projection
                delta = np.clip(delta, -epsilon_vec, epsilon_vec)

                # Zero non-attacked sensors
                delta = delta * mask[np.newaxis, :]

            if best_delta is None:
                best_delta = delta

            if best_sds > best_sds_global:
                best_sds_global = best_sds
                best_delta_global = best_delta.copy()

            all_sds_history.extend(sds_history)
            all_surr_history.extend(surr_history)

        if best_delta_global is None:
            best_delta_global = np.zeros((T, N))

        return {
            'delta': best_delta_global,
            'sds_history': all_sds_history,
            'surr_history': all_surr_history,
            'sds_final': best_sds_global,
        }

    # ------------------------------------------------------------------
    # SDS evaluation helper (used by grey-box)
    # ------------------------------------------------------------------

    def _evaluate_surrogate(self, Y: np.ndarray, U: np.ndarray,
                      delta: np.ndarray,
                      compromised_idx: List[int],
                      iswt_weight: float = 2.0) -> tuple:
        """Evaluate true SDS and surrogate loss for a given perturbation.

        Runs the full pipeline: EKF -> CUSUM -> ISWT -> SDS/Surrogate.

        Args:
            Y: (T, N) raw measurements.
            U: (T, 2) control inputs.
            delta: (T, N) perturbation matrix.
            compromised_idx: list of compromised sensor indices.
            iswt_weight: Relative weight for ISWT in surrogate.

        Returns:
            (sds_mean, surrogate_value) tuple.
        """
        T, N = Y.shape

        # Apply perturbation
        Y_pert = Y + delta

        # Run EKF or DataDrivenKF
        if hasattr(self, 'kf_model') and self.kf_model is not None:
            ekf_results = self.kf_model.run_batch(Y_pert, U)
        else:
            ekf = ExtendedKalmanFilter(self.sys, self.ekf_cfg)
            ekf_results = ekf.run_batch(Y_pert, U)

        # Run CUSUM
        cusum = CUSUMDetector(N, self.cusum_cfg)
        cusum_results = cusum.run_batch(ekf_results['std_innovation'])

        # Run ISWT
        iswt = ISWTDetector(N, self.iswt_cfg, baseline_cov=self.baseline_cov)
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
        h = self.cusum_cfg.h
        W = self.iswt_cfg.W

        # Smooth surrogate (matching torch version, but in NumPy)
        # CUSUM: logsumexp across sensors, mean across time
        temperature = 2.0
        G = cusum_results['G']
        G_scaled = G / (h * temperature)
        # Numerically stable logsumexp per row
        G_max = np.max(G_scaled, axis=1, keepdims=True)
        cusum_lse = temperature * (
            G_max.squeeze() + np.log(
                np.sum(np.exp(G_scaled - G_max), axis=1)))
        cusum_pen = np.mean(cusum_lse) / h

        # ISWT: exponential barrier
        valid_start = min(W, T)
        if T > valid_start:
            ratio = iswt_results['test_stat'][valid_start:] / max(critical, 1e-6)
            ratio = np.clip(ratio - 1.0, -20.0, 20.0)
            iswt_pen = np.mean(np.exp(ratio))
        else:
            iswt_pen = 0.0

        surrogate = cusum_pen + iswt_weight * iswt_pen

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
            Dictionary mapping eps/sigma ratio to TCA result dict.
        """
        results = {}

        for ratio in self.tca_cfg.epsilon_ratios:
            epsilon = ratio * self.sys.sigma
            if verbose:
                print(f"\n=== Budget eps/sigma = {ratio:.2f} ===")

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

def _cast_lstm_to_float64(model, device=None):
    """Cast an LSTM model's parameters to float64 and move to device.

    The differentiable EKF operates in float64 for numerical precision.
    The LSTM (trained in float32) must be cast to match dtypes for
    gradient flow through the combined EKF -> LSTM pipeline.

    Args:
        model: LSTMAutoencoder (nn.Module) in float32.
        device: Target torch.device. Defaults to CUDA if available, else CPU.

    Returns:
        model_f64: Same model with all parameters in float64 on `device`.
    """
    import copy
    import torch
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_f64 = copy.deepcopy(model)
    model_f64 = model_f64.double().to(device)
    return model_f64
