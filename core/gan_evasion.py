"""
Conditional GAN for Evasion of Digital Twin Detection Pipeline.

Implements a Generative Adversarial Network where:

    Generator G(z, c):
        Produces perturbation sequences δ(t) ∈ [-ε, ε]^N conditioned on
        attack context c = [ε, fault_magnitude, operating_point].
        Architecture: MLP → LSTM → Linear → tanh × ε

    Discriminator D(Y, U, δ):
        Wraps the full differentiable detection pipeline:
        EKF → CUSUM → ISWT → LSTM → combined alarm probability.
        This is a *physics-informed* discriminator — it uses the actual
        detection statistics rather than learning features from scratch.

Key insight:
    Unlike PGD-based TCA which optimizes δ per-session (O(K·T) per attack),
    the trained generator produces δ in a single forward pass (O(1)).
    If it converges, it has learned a *distribution* of stealthy perturbations,
    not just a single point solution.

Training:
    1. Sample random operating conditions (valve positions, fault magnitudes)
    2. Generator produces δ for each condition
    3. Differentiable pipeline evaluates detection statistics
    4. Generator minimizes alarm probability + physics constraints
    5. Discriminator maximizes classification accuracy (clean vs. attacked)
"""

import numpy as np
from typing import Optional, List, Dict
import torch
import torch.nn as nn

from .config import (SystemConfig, EKFConfig, CUSUMConfig,
                     ISWTConfig, GANConfig, LSTMDetectorConfig)


class EvasionGenerator(nn.Module):
    """Conditional generator for stealthy sensor perturbations.

    Produces δ(t) ∈ [-ε, ε]^(seq_len × N) given:
        - z: latent noise vector (for diversity)
        - c: conditioning vector [ε_ratio, fault_magnitude, ...operating context]

    The tanh output layer ensures the L∞ budget constraint by construction:
        δ = ε · tanh(output)
    """

    def __init__(self, n_sensors: int,
                 config: Optional[GANConfig] = None):
        """
        Args:
            n_sensors: Number of sensors (N).
            config: GAN architecture parameters.
        """
        super().__init__()
        self.cfg = config or GANConfig()
        self.n_sensors = n_sensors

        # Conditioning vector size: [ε_ratio, fault_mag, n_operating_params]
        self.cond_dim = 2 + n_sensors  # ε_ratio + fault_mag + per-sensor context

        # Input projection: z + c → hidden
        input_dim = self.cfg.latent_dim + self.cond_dim
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, self.cfg.hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(self.cfg.hidden_dim, self.cfg.hidden_dim),
            nn.LeakyReLU(0.2),
        )

        # Temporal generation via LSTM
        self.lstm = nn.LSTM(
            input_size=self.cfg.hidden_dim,
            hidden_size=self.cfg.hidden_dim,
            num_layers=self.cfg.n_layers,
            batch_first=True,
            dropout=0.1 if self.cfg.n_layers > 1 else 0.0,
        )

        # Output projection: hidden → N sensors
        self.output_proj = nn.Sequential(
            nn.Linear(self.cfg.hidden_dim, self.cfg.hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(self.cfg.hidden_dim // 2, n_sensors),
            nn.Tanh(),  # Output ∈ [-1, 1], scaled by ε later
        )

    def forward(self, z: torch.Tensor,
                conditioning: torch.Tensor,
                epsilon: torch.Tensor) -> torch.Tensor:
        """Generate perturbation sequence.

        Args:
            z: (batch, latent_dim) noise vector.
            conditioning: (batch, cond_dim) attack context.
            epsilon: (batch, N) per-sensor budget (absolute units).

        Returns:
            delta: (batch, seq_len, N) perturbation ∈ [-ε, ε].
        """
        batch_size = z.shape[0]
        seq_len = self.cfg.seq_len

        # Concatenate noise and conditioning
        zc = torch.cat([z, conditioning], dim=-1)  # (batch, latent_dim + cond_dim)

        # Project to hidden space
        h = self.input_proj(zc)  # (batch, hidden_dim)

        # Repeat across timesteps for LSTM input
        h_seq = h.unsqueeze(1).repeat(1, seq_len, 1)  # (batch, seq_len, hidden_dim)

        # Generate temporal structure
        lstm_out, _ = self.lstm(h_seq)  # (batch, seq_len, hidden_dim)

        # Project to sensor space with tanh activation
        raw_delta = self.output_proj(lstm_out)  # (batch, seq_len, N) ∈ [-1, 1]

        # Scale by epsilon to enforce budget constraint
        delta = raw_delta * epsilon.unsqueeze(1)  # (batch, seq_len, N) ∈ [-ε, ε]

        return delta


class PipelineDiscriminator(nn.Module):
    """Physics-informed discriminator wrapping the detection pipeline.

    Instead of learning arbitrary features, this discriminator:
    1. Runs the differentiable EKF on perturbed measurements
    2. Computes CUSUM, ISWT, and LSTM detection statistics
    3. Feeds the statistics into a small MLP classifier

    This ensures the discriminator's gradients carry physically meaningful
    information about what makes a perturbation detectable.
    """

    def __init__(self, n_sensors: int,
                 hidden_dim: int = 64):
        """
        Args:
            n_sensors: Number of sensors.
            hidden_dim: MLP hidden dimension.
        """
        super().__init__()
        self.n_sensors = n_sensors

        # Input: [max_cusum, mean_cusum, iswt_stat, lstm_score, ε_ratio, fault_mag]
        stat_dim = n_sensors + 3 + 2  # per-sensor CUSUM + aggregate stats + context

        self.classifier = nn.Sequential(
            nn.Linear(stat_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),  # Output: P(clean)
        )

    def forward(self, stats: torch.Tensor) -> torch.Tensor:
        """Classify detection statistics as clean or attacked.

        Args:
            stats: (batch, stat_dim) concatenated detection statistics.

        Returns:
            p_clean: (batch, 1) probability of being clean ∈ [0, 1].
        """
        return self.classifier(stats)


class GANTrainer:
    """End-to-end GAN training for evasion generation.

    Orchestrates the training loop:
    1. Generate operating conditions and faults
    2. Generator produces perturbations
    3. Differentiable pipeline evaluates detection
    4. Update generator (minimize detection) and discriminator (maximize accuracy)
    """

    def __init__(self,
                 generator: EvasionGenerator,
                 discriminator: PipelineDiscriminator,
                 sys_config: Optional[SystemConfig] = None,
                 ekf_config: Optional[EKFConfig] = None,
                 cusum_config: Optional[CUSUMConfig] = None,
                 iswt_config: Optional[ISWTConfig] = None,
                 lstm_model=None,
                 lstm_config: Optional[LSTMDetectorConfig] = None,
                 gan_config: Optional[GANConfig] = None):
        """
        Args:
            generator: EvasionGenerator network.
            discriminator: PipelineDiscriminator network.
            sys_config: System configuration.
            ekf_config: EKF parameters.
            cusum_config: CUSUM parameters.
            iswt_config: ISWT parameters.
            lstm_model: Trained LSTMAutoencoder (optional).
            lstm_config: LSTM detector configuration.
            gan_config: GAN training parameters.
        """
        # Auto-detect device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.G = generator.to(self.device)
        self.D = discriminator.to(self.device)
        self.sys = sys_config or SystemConfig()
        self.ekf_cfg = ekf_config or EKFConfig()
        self.cusum_cfg = cusum_config or CUSUMConfig()
        self.iswt_cfg = iswt_config or ISWTConfig()
        
        if lstm_model is not None:
            self.lstm_model = lstm_model.to(self.device)
        else:
            self.lstm_model = None
            
        self.lstm_cfg = lstm_config or LSTMDetectorConfig()
        self.gan_cfg = gan_config or GANConfig()

        # Optimizers (must be created AFTER moving parameters to device)
        self.opt_G = torch.optim.Adam(
            self.G.parameters(), lr=self.gan_cfg.learning_rate_g,
            betas=(0.5, 0.999))
        self.opt_D = torch.optim.Adam(
            self.D.parameters(), lr=self.gan_cfg.learning_rate_d,
            betas=(0.5, 0.999))

        self.training_history = {
            'g_loss': [], 'd_loss': [], 'evasion_rate': [],
            'mean_sds': [],
        }

    def _compute_pipeline_stats(self,
                                Y: torch.Tensor,
                                U: torch.Tensor,
                                delta: torch.Tensor,
                                epsilon_ratio: float,
                                fault_mag: float) -> torch.Tensor:
        """Run the differentiable detection pipeline and extract statistics.

        Args:
            Y: (seq_len, N) measurement tensor.
            U: (seq_len, 2) control tensor.
            delta: (seq_len, N) perturbation tensor.
            epsilon_ratio: Budget ratio for conditioning.
            fault_mag: Fault magnitude for conditioning.

        Returns:
            stats: (stat_dim,) detection statistics tensor.
        """
        from .ekf import DifferentiableEKF
        from .cusum import cusum_torch
        from .iswt import iswt_torch

        N = self.sys.n_sensors

        # Run differentiable EKF on device
        diff_ekf = DifferentiableEKF(self.sys, self.ekf_cfg, device=self.device)
        
        # Ensure inputs are on the same device
        Y_d = Y.to(self.device)
        U_d = U.to(self.device)
        delta_d = delta.to(self.device)
        
        ekf_out = diff_ekf.forward_pass(
            Y_d.squeeze(0) if Y_d.dim() == 3 else Y_d,
            U_d.squeeze(0) if U_d.dim() == 3 else U_d,
            delta=delta_d.squeeze(0) if delta_d.dim() == 3 else delta_d
        )

        std_innov = ekf_out['std_innovations']
        if std_innov.dim() == 3:
            std_innov = std_innov.squeeze(0)

        # CUSUM statistics
        G = cusum_torch(std_innov,
                        k=self.cusum_cfg.k, h=self.cusum_cfg.h)
        max_cusum_per_sensor = G.max(dim=0)[0]  # (N,)
        mean_cusum = G.mean()

        # ISWT statistic
        lambda_iw = iswt_torch(std_innov,
                                W=self.iswt_cfg.W,
                                alpha=self.iswt_cfg.alpha,
                                n_sensors=N)
        mean_iswt = lambda_iw.mean()

        # LSTM anomaly score (if available)
        if self.lstm_model is not None:
            from .lstm_detector import lstm_anomaly_torch
            lstm_scores = lstm_anomaly_torch(
                self.lstm_model, std_innov.float(),
                seq_len=self.lstm_cfg.seq_len)
            mean_lstm = lstm_scores.mean().double()
        else:
            mean_lstm = torch.tensor(0.0, dtype=torch.float64)

        # Concatenate all statistics on the device
        context = torch.tensor([epsilon_ratio, fault_mag],
                               dtype=torch.float64, device=self.device)
        stats = torch.cat([
            max_cusum_per_sensor,  # (N,)
            mean_cusum.unsqueeze(0),  # (1,)
            mean_iswt.unsqueeze(0),   # (1,)
            mean_lstm.unsqueeze(0),   # (1,)
            context,                  # (2,)
        ])  # Total: N + 3 + 2

        return stats

    def _generate_training_data(self, n_sessions: int,
                                seed: int = 42) -> List[Dict]:
        """Generate diverse training scenarios.

        Each scenario has different operating conditions and fault configs.

        Args:
            n_sessions: Number of scenarios to generate.
            seed: Random seed.

        Returns:
            List of scenario dictionaries with Y, U, fault info.
        """
        from .process_model import TwoTankProcess

        rng = np.random.RandomState(seed)
        scenarios = []
        seq_len = self.gan_cfg.seq_len

        for i in range(n_sessions):
            process = TwoTankProcess(self.sys)
            s = rng.randint(0, 100000)
            process.set_seed(s)

            # Random fault magnitude (1σ to 4σ)
            fault_mag = rng.uniform(1.0, 4.0)

            # Random attacked sensor
            attack_sensor = rng.randint(0, self.sys.n_sensors)

            sim = process.simulate(
                seq_len + 100,  # Extra for EKF warm-up
                fault_config={
                    'sensor_idx': [attack_sensor],
                    'fault_start': 0,
                    'fault_magnitude': fault_mag,
                },
                seed=s
            )

            # Take a window after warm-up
            start = 50
            Y = sim['y_faulted'][start:start + seq_len]
            U = sim['u'][start:start + seq_len]

            scenarios.append({
                'Y': Y,
                'U': U,
                'fault_mag': fault_mag,
                'attack_sensor': attack_sensor,
                'epsilon_ratio': rng.uniform(0.25, 1.5),
            })

        return scenarios

    def train(self, n_training_sessions: int = 50,
              seed: int = 42,
              verbose: bool = False) -> dict:
        """Train the GAN.

        Args:
            n_training_sessions: Number of training scenarios.
            seed: Random seed.
            verbose: Print progress.

        Returns:
            Training history dictionary.
        """
        cfg = self.gan_cfg

        if verbose:
            print("Generating training scenarios...")
        scenarios = self._generate_training_data(n_training_sessions, seed)

        if verbose:
            print(f"Training GAN for {cfg.n_epochs} epochs "
                  f"on {len(scenarios)} scenarios...")

        for epoch in range(cfg.n_epochs):
            epoch_g_loss = 0.0
            epoch_d_loss = 0.0
            epoch_evasion = 0.0
            n_batches = 0

            # Shuffle scenarios each epoch
            rng = np.random.RandomState(seed + epoch)
            indices = rng.permutation(len(scenarios))

            for idx in indices:
                scenario = scenarios[idx]
                Y_t = torch.tensor(scenario['Y'], dtype=torch.float64, device=self.device)
                U_t = torch.tensor(scenario['U'], dtype=torch.float64, device=self.device)
                eps_ratio = scenario['epsilon_ratio']
                fault_mag = scenario['fault_mag']
                epsilon = eps_ratio * torch.tensor(
                    self.sys.sigma, dtype=torch.float64, device=self.device)

                # --- Train Discriminator ---
                self.opt_D.zero_grad()

                # Real (clean) statistics — no perturbation
                with torch.no_grad():
                    real_stats = self._compute_pipeline_stats(
                        Y_t, U_t,
                        torch.zeros_like(Y_t, device=self.device),
                        eps_ratio, fault_mag).float()

                d_real = self.D(real_stats.unsqueeze(0))
                d_loss_real = -torch.log(d_real + 1e-8).mean()

                # Fake (GAN-perturbed) statistics
                z = torch.randn(1, cfg.latent_dim, dtype=torch.float32, device=self.device)
                cond = self._make_conditioning(eps_ratio, fault_mag)
                delta = self.G(z, cond, epsilon.float().unsqueeze(0))
                delta_2d = delta.squeeze(0).double()

                with torch.no_grad():
                    fake_stats = self._compute_pipeline_stats(
                        Y_t, U_t, delta_2d,
                        eps_ratio, fault_mag).float()

                d_fake = self.D(fake_stats.unsqueeze(0))
                d_loss_fake = -torch.log(1 - d_fake + 1e-8).mean()

                d_loss = d_loss_real + d_loss_fake
                d_loss.backward()
                self.opt_D.step()

                # --- Train Generator ---
                self.opt_G.zero_grad()

                z = torch.randn(1, cfg.latent_dim, dtype=torch.float32, device=self.device)
                delta = self.G(z, cond, epsilon.float().unsqueeze(0))
                delta_2d = delta.squeeze(0).double()

                # We need gradients through the pipeline for the generator
                # Use a simplified loss: minimize detection statistics directly
                g_loss = self._generator_loss(
                    Y_t, U_t, delta_2d, eps_ratio, fault_mag)
                g_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.G.parameters(), max_norm=1.0)
                self.opt_G.step()

                epoch_g_loss += g_loss.item()
                epoch_d_loss += d_loss.item()

                # Track evasion (no alarm = success)
                with torch.no_grad():
                    stats = self._compute_pipeline_stats(
                        Y_t, U_t, delta_2d,
                        eps_ratio, fault_mag)
                    max_cusum = stats[:self.sys.n_sensors].max().item()
                    evaded = max_cusum < self.cusum_cfg.h
                    epoch_evasion += float(evaded)

                n_batches += 1

            avg_g = epoch_g_loss / max(n_batches, 1)
            avg_d = epoch_d_loss / max(n_batches, 1)
            avg_ev = epoch_evasion / max(n_batches, 1)

            self.training_history['g_loss'].append(avg_g)
            self.training_history['d_loss'].append(avg_d)
            self.training_history['evasion_rate'].append(avg_ev)

            if verbose and (epoch % 20 == 0 or epoch == cfg.n_epochs - 1):
                print(f"  GAN epoch {epoch:3d}/{cfg.n_epochs}: "
                      f"G_loss={avg_g:.4f}, D_loss={avg_d:.4f}, "
                      f"evasion={avg_ev:.1%}")

        return self.training_history

    def _make_conditioning(self, eps_ratio: float,
                           fault_mag: float) -> torch.Tensor:
        """Create conditioning vector for the generator.

        Args:
            eps_ratio: Budget ratio ε/σ.
            fault_mag: Fault magnitude in σ units.

        Returns:
            cond: (1, cond_dim) conditioning tensor.
        """
        N = self.sys.n_sensors
        cond = torch.zeros(1, 2 + N, dtype=torch.float32, device=self.device)
        cond[0, 0] = eps_ratio
        cond[0, 1] = fault_mag
        # Operating context: normalized sensor noise levels
        cond[0, 2:] = torch.tensor(self.sys.sigma / np.max(self.sys.sigma),
                                    dtype=torch.float32, device=self.device)
        return cond

    def _generator_loss(self,
                        Y: torch.Tensor,
                        U: torch.Tensor,
                        delta: torch.Tensor,
                        eps_ratio: float,
                        fault_mag: float) -> torch.Tensor:
        """Compute generator loss (minimize detection statistics).

        The generator wants to minimize:
        1. CUSUM statistics (stay below threshold h)
        2. ISWT statistics (stay below chi-squared critical)
        3. LSTM anomaly score (stay below learned threshold)
        4. Budget violation penalty (redundant with tanh, but regularizes)

        Args:
            Y, U: Measurement and control tensors.
            delta: Generated perturbation (requires gradient through G).
            eps_ratio, fault_mag: Conditioning context.

        Returns:
            loss: Scalar generator loss.
        """
        from .ekf import DifferentiableEKF
        from .cusum import cusum_torch
        from .iswt import iswt_torch

        N = self.sys.n_sensors

        # Run pipeline (gradients flow through delta → G)
        diff_ekf = DifferentiableEKF(self.sys, self.ekf_cfg, device=self.device)
        
        # Ensure inputs are on self.device
        Y_d = Y.to(self.device)
        U_d = U.to(self.device)
        delta_d = delta.to(self.device)
        
        ekf_out = diff_ekf.forward_pass(
            Y_d.squeeze(0) if Y_d.dim() == 3 else Y_d,
            U_d.squeeze(0) if U_d.dim() == 3 else U_d,
            delta=delta_d.squeeze(0) if delta_d.dim() == 3 else delta_d)

        std_innov = ekf_out['std_innovations']
        if std_innov.dim() == 3:
            std_innov = std_innov.squeeze(0)

        # CUSUM penalty
        G = cusum_torch(std_innov,
                        k=self.cusum_cfg.k, h=self.cusum_cfg.h)
        cusum_loss = torch.mean(torch.max(G, dim=1)[0]) / self.cusum_cfg.h

        # ISWT penalty
        lambda_iw = iswt_torch(std_innov,
                                W=self.iswt_cfg.W,
                                alpha=self.iswt_cfg.alpha,
                                n_sensors=N)
        critical = self.iswt_cfg.critical_value(N)
        iswt_loss = torch.mean(self.iswt_cfg.W * lambda_iw) / critical

        # LSTM penalty (if available)
        lstm_loss = torch.tensor(0.0, dtype=torch.float64)
        if self.lstm_model is not None:
            from .lstm_detector import lstm_anomaly_torch
            lstm_scores = lstm_anomaly_torch(
                self.lstm_model, std_innov.float(),
                seq_len=self.lstm_cfg.seq_len)
            lstm_loss = lstm_scores.mean().double()

        # Combined loss (generator minimizes detection)
        total_loss = (cusum_loss
                      + 2.0 * iswt_loss
                      + lstm_loss)

        return total_loss.float()

    def generate_perturbation(self,
                              epsilon_ratio: float,
                              fault_mag: float,
                              n_samples: int = 1) -> np.ndarray:
        """Generate perturbations using the trained generator.

        Args:
            epsilon_ratio: Budget ratio ε/σ.
            fault_mag: Fault magnitude in σ units.
            n_samples: Number of perturbation samples.

        Returns:
            delta: (n_samples, seq_len, N) perturbation array.
        """
        self.G.eval()
        epsilon = torch.tensor(
            epsilon_ratio * self.sys.sigma,
            dtype=torch.float32, device=self.device).unsqueeze(0).repeat(n_samples, 1)

        z = torch.randn(n_samples, self.gan_cfg.latent_dim,
                         dtype=torch.float32, device=self.device)
        cond = self._make_conditioning(epsilon_ratio, fault_mag)
        cond = cond.repeat(n_samples, 1)

        with torch.no_grad():
            delta = self.G(z, cond, epsilon)

        return delta.cpu().numpy()


# ======================================================================
# Convenience: end-to-end GAN training function
# ======================================================================

def train_evasion_gan(sys_config: SystemConfig,
                      ekf_config: EKFConfig,
                      cusum_config: CUSUMConfig,
                      iswt_config: ISWTConfig,
                      gan_config: Optional[GANConfig] = None,
                      lstm_model=None,
                      lstm_config: Optional[LSTMDetectorConfig] = None,
                      n_training_sessions: int = 50,
                      seed: int = 42,
                      verbose: bool = False) -> GANTrainer:
    """Train an evasion GAN from scratch.

    Convenience function that creates the generator, discriminator,
    and trainer, then runs the full training loop.

    Args:
        sys_config: System configuration.
        ekf_config: Calibrated EKF parameters.
        cusum_config: CUSUM detector parameters.
        iswt_config: ISWT detector parameters.
        gan_config: GAN architecture/training parameters.
        lstm_model: Optional trained LSTMAutoencoder.
        lstm_config: LSTM detector configuration.
        n_training_sessions: Number of training scenarios.
        seed: Random seed.
        verbose: Print training progress.

    Returns:
        trainer: Trained GANTrainer with generator ready for inference.
    """
    gan_cfg = gan_config or GANConfig()
    N = sys_config.n_sensors

    generator = EvasionGenerator(N, gan_cfg)
    discriminator = PipelineDiscriminator(N, hidden_dim=gan_cfg.hidden_dim)

    trainer = GANTrainer(
        generator=generator,
        discriminator=discriminator,
        sys_config=sys_config,
        ekf_config=ekf_config,
        cusum_config=cusum_config,
        iswt_config=iswt_config,
        lstm_model=lstm_model,
        lstm_config=lstm_config,
        gan_config=gan_cfg,
    )

    trainer.train(
        n_training_sessions=n_training_sessions,
        seed=seed,
        verbose=verbose,
    )

    return trainer
