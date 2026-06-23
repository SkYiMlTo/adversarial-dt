"""
Tests for the GAN evasion generator.

Verifies:
1. Generator output dimensions and budget constraints
2. Discriminator output range
3. GAN training loop runs without errors
4. Generated perturbations respect L∞ budget
"""

import numpy as np
import pytest
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import SystemConfig, GANConfig
from core.gan_evasion import (EvasionGenerator, PipelineDiscriminator,
                               GANTrainer)


@pytest.fixture
def n_sensors():
    return 6


@pytest.fixture
def gan_config():
    """Small config for fast testing."""
    return GANConfig(
        latent_dim=16,
        hidden_dim=32,
        n_layers=2,
        seq_len=20,
        n_epochs=2,
        batch_size=4,
    )


@pytest.fixture
def sys_config():
    return SystemConfig()


@pytest.fixture
def generator(n_sensors, gan_config):
    return EvasionGenerator(n_sensors, gan_config)


@pytest.fixture
def discriminator(n_sensors):
    return PipelineDiscriminator(n_sensors, hidden_dim=32)


class TestEvasionGenerator:
    """Tests for the EvasionGenerator network."""

    def test_output_shape(self, generator, n_sensors, gan_config):
        """Verify generator produces correctly shaped perturbations."""
        batch = 4
        z = torch.randn(batch, gan_config.latent_dim)
        cond = torch.randn(batch, 2 + n_sensors)
        epsilon = torch.ones(batch, n_sensors) * 0.01

        delta = generator(z, cond, epsilon)

        assert delta.shape == (batch, gan_config.seq_len, n_sensors), \
            f"Expected ({batch}, {gan_config.seq_len}, {n_sensors}), " \
            f"got {delta.shape}"

    def test_budget_constraint(self, generator, n_sensors, gan_config):
        """Verify |δ_i| ≤ ε_i ∀i,t by construction (tanh × ε)."""
        batch = 8
        z = torch.randn(batch, gan_config.latent_dim)
        cond = torch.randn(batch, 2 + n_sensors)

        # Different epsilon per sensor
        epsilon = torch.tensor([0.01, 0.02, 0.03, 0.15, 0.001, 0.0005],
                               dtype=torch.float32)
        epsilon = epsilon.unsqueeze(0).repeat(batch, 1)

        with torch.no_grad():
            delta = generator(z, cond, epsilon)

        # Check L∞ constraint
        for i in range(n_sensors):
            max_delta_i = torch.abs(delta[:, :, i]).max().item()
            eps_i = epsilon[0, i].item()
            assert max_delta_i <= eps_i + 1e-6, \
                (f"Sensor {i}: max |δ| = {max_delta_i:.6f} > "
                 f"ε = {eps_i:.6f}")

    def test_differentiable(self, generator, n_sensors, gan_config):
        """Verify gradients flow through the generator."""
        z = torch.randn(1, gan_config.latent_dim)
        cond = torch.randn(1, 2 + n_sensors)
        epsilon = torch.ones(1, n_sensors) * 0.01

        delta = generator(z, cond, epsilon)
        loss = delta.sum()
        loss.backward()

        # Check that generator parameters have gradients
        has_grad = False
        for param in generator.parameters():
            if param.grad is not None and torch.any(param.grad != 0):
                has_grad = True
                break
        assert has_grad, "No gradients found in generator parameters"


class TestPipelineDiscriminator:
    """Tests for the PipelineDiscriminator network."""

    def test_output_range(self, discriminator, n_sensors):
        """Verify discriminator output ∈ [0, 1]."""
        batch = 4
        stat_dim = n_sensors + 3 + 2
        stats = torch.randn(batch, stat_dim)

        with torch.no_grad():
            p_clean = discriminator(stats)

        assert p_clean.shape == (batch, 1), \
            f"Expected ({batch}, 1), got {p_clean.shape}"
        assert torch.all(p_clean >= 0) and torch.all(p_clean <= 1), \
            f"Output should be in [0, 1], got range " \
            f"[{p_clean.min():.4f}, {p_clean.max():.4f}]"

    def test_differentiable(self, discriminator, n_sensors):
        """Verify gradients flow through the discriminator."""
        stat_dim = n_sensors + 3 + 2
        stats = torch.randn(1, stat_dim, requires_grad=True)

        p_clean = discriminator(stats)
        p_clean.backward()

        assert stats.grad is not None, "No gradient for input stats"


class TestGANTrainer:
    """Tests for the GANTrainer training loop."""

    def test_training_step_runs(self, generator, discriminator,
                                 sys_config, gan_config):
        """Verify one training step runs without errors."""
        from core.config import EKFConfig, CUSUMConfig, ISWTConfig

        trainer = GANTrainer(
            generator=generator,
            discriminator=discriminator,
            sys_config=sys_config,
            ekf_config=EKFConfig(),
            cusum_config=CUSUMConfig(),
            iswt_config=ISWTConfig(),
            gan_config=gan_config,
        )

        # Run training with minimal data (2 sessions, 2 epochs)
        history = trainer.train(
            n_training_sessions=2,
            seed=42,
            verbose=False,
        )

        assert 'g_loss' in history
        assert 'd_loss' in history
        assert len(history['g_loss']) == gan_config.n_epochs

    def test_generate_perturbation(self, generator, discriminator,
                                    sys_config, gan_config):
        """Verify perturbation generation produces valid output."""
        from core.config import EKFConfig, CUSUMConfig, ISWTConfig

        trainer = GANTrainer(
            generator=generator,
            discriminator=discriminator,
            sys_config=sys_config,
            ekf_config=EKFConfig(),
            cusum_config=CUSUMConfig(),
            iswt_config=ISWTConfig(),
            gan_config=gan_config,
        )

        delta = trainer.generate_perturbation(
            epsilon_ratio=1.0,
            fault_mag=2.0,
            n_samples=3,
        )

        assert delta.shape == (3, gan_config.seq_len, sys_config.n_sensors), \
            f"Expected (3, {gan_config.seq_len}, {sys_config.n_sensors}), " \
            f"got {delta.shape}"

        # Check budget constraint (within ε = 1.0 * sigma)
        sigma = sys_config.sigma
        for i in range(sys_config.n_sensors):
            max_delta_i = np.abs(delta[:, :, i]).max()
            assert max_delta_i <= sigma[i] + 1e-5, \
                (f"Sensor {i}: max |δ| = {max_delta_i:.6f} > "
                 f"ε = {sigma[i]:.6f}")
