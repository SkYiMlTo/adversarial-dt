"""
LSTM Autoencoder Anomaly Detector for Digital Twin Innovation Sequences.

Implements a neural anomaly detector that monitors the EKF standardized
innovation sequence ν̂(t) for deviations from the learned nominal pattern.

Architecture:
    Encoder:  LSTM(N, hidden_dim, n_layers) → latent ∈ R^latent_dim
    Decoder:  LSTM(latent_dim, hidden_dim, n_layers) → reconstructed ∈ R^(seq_len × N)
    Score:    anomaly_score(t) = MSE(ν̂_window, ν̂_reconstructed)

Under nominal operation (H₀):
    - Innovations are i.i.d. N(0, 1), so the autoencoder learns to
      reproduce white noise patterns with low reconstruction error.

Under attack (H₁):
    - TCA perturbations introduce temporal structure and cross-sensor
      correlations that deviate from the learned nominal distribution.
    - The reconstruction error spikes above the calibrated threshold.

The detector exposes a differentiable interface (via PyTorch) so that
adversarial attacks (PGD, GAN) can compute gradients through the
detection decision — enabling the study of adversarial robustness.
"""

import numpy as np
from typing import Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .config import LSTMDetectorConfig


class LSTMAutoencoder(nn.Module):
    """LSTM encoder-decoder for innovation sequence reconstruction.

    The bottleneck forces the network to learn a compressed representation
    of nominal innovation dynamics. Anomalous sequences (under attack)
    cannot be well-reconstructed, producing high MSE.
    """

    def __init__(self, n_sensors: int,
                 config: Optional[LSTMDetectorConfig] = None):
        """
        Args:
            n_sensors: Number of sensors (N) — input feature dimension.
            config: LSTM architecture and training parameters.
        """
        super().__init__()
        self.cfg = config or LSTMDetectorConfig()
        self.n_sensors = n_sensors

        # Encoder: maps (seq_len, N) → latent_dim
        self.encoder = nn.LSTM(
            input_size=n_sensors,
            hidden_size=self.cfg.hidden_dim,
            num_layers=self.cfg.n_layers,
            batch_first=True,
            dropout=self.cfg.dropout if self.cfg.n_layers > 1 else 0.0,
        )
        self.enc_to_latent = nn.Linear(self.cfg.hidden_dim, self.cfg.latent_dim)

        # Decoder: maps latent_dim → (seq_len, N)
        self.latent_to_dec = nn.Linear(self.cfg.latent_dim, self.cfg.hidden_dim)
        self.decoder = nn.LSTM(
            input_size=self.cfg.hidden_dim,
            hidden_size=self.cfg.hidden_dim,
            num_layers=self.cfg.n_layers,
            batch_first=True,
            dropout=self.cfg.dropout if self.cfg.n_layers > 1 else 0.0,
        )
        self.output_layer = nn.Linear(self.cfg.hidden_dim, n_sensors)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: encode → decode → reconstruct.

        Args:
            x: (batch, seq_len, N) input innovation windows.

        Returns:
            reconstructed: (batch, seq_len, N) reconstructed innovations.
            latent: (batch, latent_dim) bottleneck representation.
        """
        batch_size, seq_len, _ = x.shape

        # Encode: use final hidden state as sequence summary
        _, (h_n, _) = self.encoder(x)
        # h_n shape: (n_layers, batch, hidden_dim) — take last layer
        encoder_out = h_n[-1]  # (batch, hidden_dim)

        # Bottleneck
        latent = self.enc_to_latent(encoder_out)  # (batch, latent_dim)

        # Decode: repeat latent across timesteps
        dec_input = self.latent_to_dec(latent)  # (batch, hidden_dim)
        dec_input = dec_input.unsqueeze(1).repeat(1, seq_len, 1)  # (batch, seq_len, hidden_dim)

        dec_output, _ = self.decoder(dec_input)  # (batch, seq_len, hidden_dim)

        # Output projection
        reconstructed = self.output_layer(dec_output)  # (batch, seq_len, N)

        return reconstructed, latent

    def compute_anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """Compute per-window anomaly score (differentiable).

        Args:
            x: (batch, seq_len, N) input innovation windows.

        Returns:
            scores: (batch,) MSE reconstruction error per window.
        """
        reconstructed, _ = self.forward(x)
        # MSE per window (mean over seq_len and sensors)
        scores = torch.mean((x - reconstructed) ** 2, dim=(1, 2))
        return scores


class LSTMDetector:
    """LSTM-based anomaly detector with training and inference.

    Wraps the LSTMAutoencoder with:
    - Training on clean calibration data
    - Threshold calibration from reconstruction error distribution
    - Online sliding-window detection
    - Batch detection for offline experiments
    """

    def __init__(self, n_sensors: int,
                 config: Optional[LSTMDetectorConfig] = None,
                 device=None):
        """
        Args:
            n_sensors: Number of sensors (N).
            config: LSTM configuration.
            device: torch.device (defaults to CUDA if available, else CPU).
        """
        self.cfg = config or LSTMDetectorConfig()
        self.n_sensors = n_sensors

        # Device selection
        if device is None:
            self.device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model = LSTMAutoencoder(n_sensors, self.cfg).to(self.device)
        self.threshold = float('inf')  # Set after training
        self.trained = False

        # Online sliding window buffer
        self.buffer = np.zeros((self.cfg.seq_len, n_sensors))
        self.buffer_idx = 0
        self.buffer_full = False

        # Current state
        self.anomaly_score = 0.0
        self.alarm = False

    def _create_windows(self, innovations: np.ndarray) -> np.ndarray:
        """Create sliding windows from innovation time series.

        Args:
            innovations: (T, N) standardized innovation matrix.

        Returns:
            windows: (T - seq_len + 1, seq_len, N) windowed data.
        """
        T, N = innovations.shape
        seq_len = self.cfg.seq_len
        if T < seq_len:
            raise ValueError(
                f"Time series length {T} < seq_len {seq_len}")

        n_windows = T - seq_len + 1
        windows = np.zeros((n_windows, seq_len, N))
        for i in range(n_windows):
            windows[i] = innovations[i:i + seq_len]
        return windows

    def train(self, clean_innovations: np.ndarray,
              verbose: bool = False) -> dict:
        """Train the LSTM autoencoder on clean innovation data.

        After training, the anomaly threshold is calibrated from the
        reconstruction error distribution on the training data.

        Args:
            clean_innovations: (T, N) standardized innovations from clean
                               calibration data.
            verbose: Print training progress.

        Returns:
            Dictionary with training history:
                - 'train_losses': per-epoch training loss
                - 'threshold': calibrated anomaly threshold
                - 'train_scores': reconstruction errors on training data
        """
        cfg = self.cfg

        # Create windowed training data
        windows = self._create_windows(clean_innovations)
        X = torch.tensor(windows, dtype=torch.float32)

        # Train/val split (80/20)
        n_train = int(0.8 * len(X))
        X_train = X[:n_train]
        X_val = X[n_train:]

        dataset = TensorDataset(X_train)
        loader = DataLoader(dataset, batch_size=cfg.batch_size,
                            shuffle=True, drop_last=False)

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=cfg.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=10, factor=0.5)

        if verbose and self.device.type == 'cuda':
            print(f"  [LSTM] Training on {torch.cuda.get_device_name(0)}")

        self.model.train()
        train_losses = []
        val_losses = []

        for epoch in range(cfg.n_epochs):
            epoch_loss = 0.0
            n_batches = 0

            for (batch_x,) in loader:
                batch_x = batch_x.to(self.device)
                optimizer.zero_grad()
                recon, _ = self.model(batch_x)
                loss = nn.functional.mse_loss(recon, batch_x)
                loss.backward()
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            train_losses.append(avg_loss)

            # Validation loss
            self.model.eval()
            with torch.no_grad():
                X_val_dev = X_val.to(self.device)
                val_recon, _ = self.model(X_val_dev)
                val_loss = nn.functional.mse_loss(val_recon, X_val_dev).item()
                val_losses.append(val_loss)
            self.model.train()

            scheduler.step(val_loss)

            if verbose and (epoch % 20 == 0 or epoch == cfg.n_epochs - 1):
                print(f"  LSTM epoch {epoch:3d}/{cfg.n_epochs}: "
                      f"train_loss={avg_loss:.6f}, val_loss={val_loss:.6f}")

        # Calibrate threshold from training data reconstruction errors
        self.model.eval()
        with torch.no_grad():
            X_dev = X.to(self.device)
            train_scores = self.model.compute_anomaly_score(X_dev).cpu().numpy()

        self.threshold = float(np.percentile(
            train_scores, cfg.threshold_percentile))
        self.trained = True

        if verbose:
            print(f"  LSTM threshold (p{cfg.threshold_percentile}): "
                  f"{self.threshold:.6f}")

        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'threshold': self.threshold,
            'train_scores': train_scores,
        }

    def reset(self):
        """Reset online detection state."""
        self.buffer[:] = 0.0
        self.buffer_idx = 0
        self.buffer_full = False
        self.anomaly_score = 0.0
        self.alarm = False

    def update(self, std_innovation: np.ndarray) -> dict:
        """Online update with a new standardized innovation vector.

        Args:
            std_innovation: ν̂(t) ∈ R^N.

        Returns:
            Dictionary with:
                - 'anomaly_score': reconstruction error
                - 'threshold': calibrated threshold
                - 'alarm': True if anomaly detected
                - 'ready': True if buffer is full (enough data)
        """
        # Add to circular buffer
        idx = self.buffer_idx % self.cfg.seq_len
        self.buffer[idx] = std_innovation
        self.buffer_idx += 1

        if self.buffer_idx < self.cfg.seq_len:
            return {
                'anomaly_score': 0.0,
                'threshold': self.threshold,
                'alarm': False,
                'ready': False,
            }

        self.buffer_full = True

        # Reconstruct the ordered window from circular buffer
        start = self.buffer_idx % self.cfg.seq_len
        window = np.concatenate([
            self.buffer[start:],
            self.buffer[:start]
        ], axis=0)

        # Compute anomaly score
        self.model.eval()
        with torch.no_grad():
            x = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(self.device)
            self.anomaly_score = float(
                self.model.compute_anomaly_score(x).item())

        self.alarm = self.anomaly_score > self.threshold

        return {
            'anomaly_score': self.anomaly_score,
            'threshold': self.threshold,
            'alarm': self.alarm,
            'ready': True,
        }

    def run_batch(self, std_innovations: np.ndarray) -> dict:
        """Run LSTM detector over a batch of standardized innovations.

        Args:
            std_innovations: (T, N) array.

        Returns:
            Dictionary with arrays indexed by time:
                - 'anomaly_score': (T,) reconstruction errors
                - 'alarm': (T,) alarm flags
                - 'threshold': scalar threshold value
        """
        self.reset()
        T, N = std_innovations.shape
        seq_len = self.cfg.seq_len

        anomaly_scores = np.zeros(T)
        alarms = np.zeros(T, dtype=bool)

        if T < seq_len:
            return {
                'anomaly_score': anomaly_scores,
                'alarm': alarms,
                'threshold': self.threshold,
            }

        # Efficient batch computation (instead of step-by-step)
        windows = self._create_windows(std_innovations)
        self.model.eval()
        with torch.no_grad():
            X = torch.tensor(windows, dtype=torch.float32).to(self.device)
            # Process in chunks to avoid GPU OOM
            chunk_size = 256
            scores_list = []
            for i in range(0, len(X), chunk_size):
                chunk = X[i:i + chunk_size]
                scores_list.append(
                    self.model.compute_anomaly_score(chunk).cpu().numpy())
            scores = np.concatenate(scores_list)

        # Map window scores back to timesteps
        # Window i corresponds to timestep i + seq_len - 1
        for i, score in enumerate(scores):
            t = i + seq_len - 1
            anomaly_scores[t] = score
            alarms[t] = score > self.threshold

        return {
            'anomaly_score': anomaly_scores,
            'alarm': alarms,
            'threshold': self.threshold,
        }

    def get_model(self) -> LSTMAutoencoder:
        """Return the underlying PyTorch model for adversarial access."""
        return self.model


# ======================================================================
# Differentiable LSTM anomaly score (for TCA/GAN gradient flow)
# ======================================================================

def lstm_anomaly_torch(model: LSTMAutoencoder,
                       std_innovations: torch.Tensor,
                       seq_len: int = 50) -> torch.Tensor:
    """Compute differentiable LSTM anomaly scores for gradient-based attacks.

    This function is used by TCA's run_whitebox_neural() and the GAN
    discriminator to backpropagate through the LSTM detection decision.

    Args:
        model: Trained LSTMAutoencoder in eval mode.
        std_innovations: (T, N) tensor of standardized innovations
                         (must have requires_grad through delta).
        seq_len: Window length (must match model's training config).

    Returns:
        scores: (T,) tensor of anomaly scores (differentiable).
                First (seq_len - 1) entries are zero (insufficient data).
    """
    T, N = std_innovations.shape
    scores = torch.zeros(T, dtype=std_innovations.dtype,
                         device=std_innovations.device)

    if T < seq_len:
        return scores

    # Create windows (differentiable indexing)
    n_windows = T - seq_len + 1
    windows = torch.stack([
        std_innovations[i:i + seq_len] for i in range(n_windows)
    ])  # (n_windows, seq_len, N)

    # Compute anomaly scores through the model
    window_scores = model.compute_anomaly_score(windows)  # (n_windows,)

    # Map back to timesteps using vectorised torch.cat (very fast, graph-friendly)
    padding = torch.zeros(seq_len - 1, dtype=std_innovations.dtype,
                          device=std_innovations.device)
    scores = torch.cat([padding, window_scores])

    return scores
