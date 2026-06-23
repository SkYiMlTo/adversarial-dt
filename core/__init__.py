"""
Core library for the state deception attack / defense framework.

Implements the mathematical components from:
"Blinding the Oracle: Adversarial Sensor Manipulation Against
Digital Twin State Estimation in Cyber-Physical Systems"

Components:
    - config: Physical and algorithm parameters
    - process_model: Two-tank water distribution ODE
    - ekf: Extended Kalman Filter (NumPy + PyTorch differentiable)
    - cusum: Per-sensor CUSUM sequential detector
    - iswt: Innovation Spatial Whiteness Test (Stein divergence)
    - sds: Sensor Deception Score metric
    - tca: Targeted Consistency Attack (white-box + grey-box + neural)
    - calibration: EKF calibration and whiteness validation
    - lstm_detector: LSTM autoencoder anomaly detector (neural defense)
    - gan_evasion: Conditional GAN evasion generator (neural attack)
"""

from .config import (SystemConfig, EKFConfig, TCAConfig, ExperimentConfig,
                     LSTMDetectorConfig, GANConfig)
from .process_model import TwoTankProcess
from .ekf import ExtendedKalmanFilter
from .cusum import CUSUMDetector
from .iswt import ISWTDetector, combined_alarm_full
from .sds import compute_sds, compute_phi, compute_psi
from .tca import TargetedConsistencyAttack
from .calibration import calibrate_ekf, validate_whiteness

# Neural components require PyTorch — import conditionally
try:
    from .lstm_detector import LSTMAutoencoder, LSTMDetector
    from .gan_evasion import EvasionGenerator, GANTrainer, train_evasion_gan
except ImportError:
    pass  # torch not installed; neural components unavailable


