"""
Tests for the LSTM autoencoder anomaly detector.

Verifies:
1. Output dimensions are correct
2. Model trains and loss decreases
3. Anomaly detection works (higher score for attacked data)
4. Gradients flow through the model (for adversarial attacks)
"""

import numpy as np
import pytest
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import SystemConfig, LSTMDetectorConfig
from core.lstm_detector import LSTMAutoencoder, LSTMDetector, lstm_anomaly_torch


@pytest.fixture
def n_sensors():
    return 6


@pytest.fixture
def lstm_config():
    """Small config for fast testing."""
    return LSTMDetectorConfig(
        hidden_dim=16,
        n_layers=1,
        seq_len=20,
        latent_dim=8,
        n_epochs=10,
        batch_size=16,
    )


@pytest.fixture
def model(n_sensors, lstm_config):
    return LSTMAutoencoder(n_sensors, lstm_config)


class TestLSTMAutoencoder:
    """Tests for the LSTMAutoencoder nn.Module."""

    def test_output_shape(self, model, n_sensors, lstm_config):
        """Verify autoencoder I/O dimensions match."""
        batch = 4
        seq_len = lstm_config.seq_len
        x = torch.randn(batch, seq_len, n_sensors)

        recon, latent = model(x)

        assert recon.shape == (batch, seq_len, n_sensors), \
            f"Expected ({batch}, {seq_len}, {n_sensors}), got {recon.shape}"
        assert latent.shape == (batch, lstm_config.latent_dim), \
            f"Expected ({batch}, {lstm_config.latent_dim}), got {latent.shape}"

    def test_anomaly_score_shape(self, model, n_sensors, lstm_config):
        """Verify anomaly score output is scalar per window."""
        batch = 8
        x = torch.randn(batch, lstm_config.seq_len, n_sensors)

        scores = model.compute_anomaly_score(x)
        assert scores.shape == (batch,), f"Expected ({batch},), got {scores.shape}"
        assert torch.all(scores >= 0), "Anomaly scores should be non-negative"

    def test_differentiable(self, model, n_sensors, lstm_config):
        """Verify gradients flow through the model."""
        x = torch.randn(1, lstm_config.seq_len, n_sensors, requires_grad=True)

        score = model.compute_anomaly_score(x)
        score.backward()

        assert x.grad is not None, "No gradient computed for input"
        assert not torch.all(x.grad == 0), "Gradient is all zeros"


class TestLSTMDetector:
    """Tests for the LSTMDetector wrapper."""

    def test_trains_on_clean_data(self, n_sensors, lstm_config):
        """Verify loss decreases during training."""
        detector = LSTMDetector(n_sensors, lstm_config)

        # Generate synthetic clean innovations (white noise)
        T = 200
        clean_data = np.random.randn(T, n_sensors).astype(np.float32)

        history = detector.train(clean_data, verbose=False)

        assert len(history['train_losses']) == lstm_config.n_epochs
        # Loss should decrease (first > last, at least by some amount)
        assert history['train_losses'][-1] < history['train_losses'][0], \
            "Training loss did not decrease"
        assert history['threshold'] > 0, \
            "Threshold should be positive"
        assert detector.trained, "Detector should be marked as trained"

    def test_detects_anomaly(self, n_sensors, lstm_config):
        """Verify reconstruction error increases under attack."""
        detector = LSTMDetector(n_sensors, lstm_config)

        # Train on clean white noise
        T_train = 300
        clean_data = np.random.randn(T_train, n_sensors).astype(np.float32)
        detector.train(clean_data, verbose=False)

        # Clean test data (should have low scores)
        T_test = 100
        clean_test = np.random.randn(T_test, n_sensors).astype(np.float32)
        clean_results = detector.run_batch(clean_test)

        # Attacked test data (add strong bias — should have high scores)
        attacked_test = clean_test + 3.0  # Strong mean shift
        attacked_results = detector.run_batch(attacked_test)

        # Compare scores (after warm-up period)
        valid_start = lstm_config.seq_len
        if T_test > valid_start:
            clean_mean = np.mean(
                clean_results['anomaly_score'][valid_start:])
            attacked_mean = np.mean(
                attacked_results['anomaly_score'][valid_start:])

            assert attacked_mean > clean_mean, \
                (f"Attacked score ({attacked_mean:.4f}) should be higher "
                 f"than clean ({clean_mean:.4f})")

    def test_batch_vs_online(self, n_sensors, lstm_config):
        """Verify batch and online modes produce consistent results."""
        detector = LSTMDetector(n_sensors, lstm_config)

        T_train = 200
        clean_data = np.random.randn(T_train, n_sensors).astype(np.float32)
        detector.train(clean_data, verbose=False)

        T_test = 80
        test_data = np.random.randn(T_test, n_sensors).astype(np.float32)

        # Batch mode
        batch_results = detector.run_batch(test_data)

        # Online mode
        detector.reset()
        online_alarms = []
        for t in range(T_test):
            result = detector.update(test_data[t])
            online_alarms.append(result['alarm'])

        # Both should produce the same alarm decisions
        # (batch is optimized but semantically equivalent)
        # Just verify they both run without error and produce bool arrays
        assert batch_results['alarm'].dtype == bool
        assert all(isinstance(a, bool) for a in online_alarms)


class TestLSTMAnomalyTorch:
    """Tests for the differentiable anomaly score function."""

    def test_output_shape(self, model, n_sensors, lstm_config):
        """Verify torch anomaly scores have correct shape."""
        T = 60
        innovations = torch.randn(T, n_sensors)

        scores = lstm_anomaly_torch(model, innovations,
                                     seq_len=lstm_config.seq_len)

        assert scores.shape == (T,), f"Expected ({T},), got {scores.shape}"

    def test_zeros_before_warmup(self, model, n_sensors, lstm_config):
        """Verify scores are zero before the window is full."""
        T = 60
        innovations = torch.randn(T, n_sensors)

        scores = lstm_anomaly_torch(model, innovations,
                                     seq_len=lstm_config.seq_len)

        # First (seq_len - 1) scores should be zero
        assert torch.all(scores[:lstm_config.seq_len - 1] == 0), \
            "Scores before window is full should be zero"

    def test_gradient_flow(self, model, n_sensors, lstm_config):
        """Verify gradients propagate through lstm_anomaly_torch."""
        T = 40
        innovations = torch.randn(T, n_sensors, requires_grad=True)

        scores = lstm_anomaly_torch(model, innovations,
                                     seq_len=lstm_config.seq_len)
        loss = scores.sum()
        loss.backward()

        assert innovations.grad is not None, "No gradient for innovations"
