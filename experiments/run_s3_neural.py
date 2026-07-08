"""
S3 Neural Attack/Defense Evaluation.

Runs all S3 experiments for the LSTM + GAN expansion:

Phase A — LSTM Training & Baseline (Table 6):
    1. Train LSTM autoencoder on clean calibration innovations
    2. Evaluate detection: CUSUM-only vs ISWT-only vs LSTM-only vs Combined
    3. Compare TPR (on faulted data) and FPR (on clean data)

Phase B — Adversarial PGD Attack on LSTM (Table 7):
    1. Run existing TCA (evades CUSUM + ISWT only)
    2. Run new TCA-Neural (evades CUSUM + ISWT + LSTM)
    3. Compare evasion rates across fault magnitudes and budgets

Phase C — GAN vs PGD Comparison (Table 8):
    1. Train GAN evasion generator
    2. Generate perturbations with GAN (single forward pass)
    3. Generate perturbations with PGD (100 iterations)
    4. Compare evasion rate, SDS, computational cost

Output:
    results/s3/table6_detection.json
    results/s3/table7_adversarial_lstm.json
    results/s3/table8_gan_vs_pgd.json
    results/s3/lstm_model.pt  (trained LSTM checkpoint)
    results/s3/gan_model.pt   (trained GAN checkpoint)
"""

import os
import sys
import json
import time
import numpy as np
from pathlib import Path

# Force UTF-8 output on Windows (avoids cp1252 UnicodeEncodeError for Greek chars)
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from core.config import (ExperimentConfig, SystemConfig, EKFConfig,
                         CUSUMConfig, ISWTConfig, TCAConfig,
                         LSTMDetectorConfig, GANConfig)
from core.process_model import TwoTankProcess
from core.ekf import ExtendedKalmanFilter
from core.cusum import CUSUMDetector
from core.iswt import ISWTDetector, combined_alarm, combined_alarm_full
from core.sds import compute_sds_timeseries
from core.tca import TargetedConsistencyAttack
from core.calibration import full_calibration
from core.lstm_detector import LSTMDetector
from core.gan_evasion import train_evasion_gan

# Output directory
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "s3"

# ---------------------------------------------------------------------------
# GPU / device selection
# ---------------------------------------------------------------------------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if DEVICE.type == 'cuda':
    print(f"[GPU] Using CUDA device: {torch.cuda.get_device_name(0)}")
    # Disable cuDNN to allow RNN/LSTM backpropagation in eval mode
    torch.backends.cudnn.enabled = False
else:
    print("[CPU] CUDA not available — running on CPU.")

# ---------------------------------------------------------------------------
# Wall-clock budget: hard limit of 2 hours (7200 seconds).
# Phases that respect this budget will skip remaining configs when time is up.
# ---------------------------------------------------------------------------
WALL_CLOCK_BUDGET_SEC = 14400  # 4 hours
_EXPERIMENT_START: float = 0.0  # set in main()

def time_budget_remaining() -> float:
    """Return seconds remaining in the 2-hour budget."""
    return max(0.0, WALL_CLOCK_BUDGET_SEC - (time.time() - _EXPERIMENT_START))

# ---------------------------------------------------------------------------
# FAST_MODE: set to True for a quick smoke-test (~10 min on CPU).
# Set to False for a full GPU-accelerated run within the 2-hour budget.
# GPU-mode parameters below replace the fast-mode overrides.
# ---------------------------------------------------------------------------
FAST_MODE = False

# Fast-mode overrides (used only when FAST_MODE=True)
_FAST_K_PGD       = 15    # PGD iterations (vs full)
_FAST_T_EVAL      = 200   # Eval session length in steps
_FAST_FAULT_MULTS = [2.0, 4.0]          # Subset of fault magnitudes
_FAST_EPS_RATIOS  = [0.50, 1.50]        # Subset of budget ratios
_FAST_GAN_SESSIONS = 10   # GAN training sessions

# GPU-mode overrides (used when FAST_MODE=False).
# These are tuned to finish within ~2 hours on a modern NVIDIA GPU.
_GPU_K_PGD        = 100   # Full PGD iterations
_GPU_T_EVAL       = 600   # Full session length (600 steps = 10 min at 1 Hz)
_GPU_FAULT_MULTS  = [1.0, 2.0, 3.0, 4.0]   # All fault magnitudes
_GPU_EPS_RATIOS   = [0.50, 1.00, 1.50]      # Budget sweep
_GPU_GAN_SESSIONS = 50    # Full GAN training


# ======================================================================
# Phase A: LSTM Training & Detection Baseline (Table 6)
# ======================================================================

def run_phase_a(config: ExperimentConfig,
                verbose: bool = True) -> dict:
    """Train LSTM and evaluate detection performance.

    Returns:
        Dictionary with:
            - 'lstm_detector': trained LSTMDetector
            - 'calib': calibration results
            - 'table6': detection comparison results
    """
    print("\n" + "=" * 70)
    print("PHASE A: LSTM Training & Detection Baseline (Table 6)")
    print("=" * 70)

    sys_cfg = config.system
    N = sys_cfg.n_sensors
    T_train = config.s3_lstm_train_steps
    T_eval = config.s1_session_duration_steps

    # --- Step 1: Calibration ---
    print("\n[A.1] Running calibration...")
    calib = full_calibration(sys_cfg, config.s1_calibration_steps,
                             seed=config.seed)
    ekf_cfg = calib['ekf_config']
    iswt_cfg = calib['iswt_config']
    Q_hat = calib['Q']
    R = calib['R']

    if not calib['whiteness_validation']['passed']:
        print("  WARNING: Whiteness validation failed on clean data")

    # Process model needed for both training and evaluation branches
    process = TwoTankProcess(sys_cfg)

    # --- Step 2: Generate LSTM training data and train (or resume) ---
    lstm_ckpt_path = RESULTS_DIR / "lstm_model.pt"
    if lstm_ckpt_path.exists():
        # Resume from checkpoint — skip expensive retraining
        print(f"\n[A.2] Checkpoint found at {lstm_ckpt_path} — skipping training.")
        lstm_detector = LSTMDetector(N, config.lstm)
        ckpt = torch.load(lstm_ckpt_path, map_location='cpu')
        lstm_detector.model.load_state_dict(ckpt['model_state'])
        lstm_detector.threshold = ckpt['threshold']
        lstm_detector.model.to(lstm_detector.device)
        print(f"  Loaded LSTM threshold: {lstm_detector.threshold:.6f}")
    else:
        print(f"\n[A.2] Generating {T_train}s of clean training data...")
        process.set_seed(config.seed + 1000)
        sim_train = process.simulate(T_train, seed=config.seed + 1000)

        ekf_train = ExtendedKalmanFilter(sys_cfg, ekf_cfg)
        ekf_train.set_noise_covariances(Q_hat, R)
        ekf_train_results = ekf_train.run_batch(sim_train['y_noisy'],
                                                 sim_train['u'])
        clean_innovations = ekf_train_results['std_innovation']

        # --- Step 3: Train LSTM ---
        print(f"\n[A.3] Training LSTM autoencoder...")
        lstm_detector = LSTMDetector(N, config.lstm)
        train_history = lstm_detector.train(clean_innovations, verbose=verbose)
        print(f"  Final train loss: {train_history['train_losses'][-1]:.6f}")
        print(f"  Threshold: {train_history['threshold']:.6f}")

        # Save LSTM model
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state': lstm_detector.model.state_dict(),
            'threshold': lstm_detector.threshold,
            'config': config.lstm,
        }, lstm_ckpt_path)

    # --- Step 4: Evaluate detection (Table 6, skip if already done) ---
    table6_path = RESULTS_DIR / "table6_detection.json"
    if table6_path.exists() and lstm_ckpt_path.exists():
        print(f"\n[A.4] table6_detection.json found — skipping detection evaluation.")
        with open(table6_path) as f:
            table6 = json.load(f)
        return {
            'lstm_detector': lstm_detector,
            'calib': calib,
            'table6': table6,
        }

    print(f"\n[A.4] Evaluating detection performance...")
    table6 = {}

    # 4a. FPR on clean data
    print("  Evaluating FPR on clean data...")
    sim_clean = process.simulate(T_eval, seed=config.seed + 50000)
    ekf_clean = ExtendedKalmanFilter(sys_cfg, ekf_cfg)
    ekf_clean.set_noise_covariances(Q_hat, R)
    ekf_clean_results = ekf_clean.run_batch(sim_clean['y_noisy'],
                                              sim_clean['u'])

    cusum_clean = CUSUMDetector(N, config.cusum)
    cusum_clean_res = cusum_clean.run_batch(
        ekf_clean_results['std_innovation'])

    iswt_clean = ISWTDetector(N, iswt_cfg)
    iswt_clean_res = iswt_clean.run_batch(
        ekf_clean_results['std_innovation'])

    lstm_clean_res = lstm_detector.run_batch(
        ekf_clean_results['std_innovation'])

    W = iswt_cfg.W
    table6['fpr'] = {
        'cusum_only': float(np.mean(
            np.any(cusum_clean_res['alarm'], axis=1)[W:])),
        'iswt_only': float(np.mean(iswt_clean_res['alarm'][W:])),
        'lstm_only': float(np.mean(lstm_clean_res['alarm'][W:])),
        'cusum_iswt': float(np.mean(
            (np.any(cusum_clean_res['alarm'], axis=1)
             | iswt_clean_res['alarm'])[W:])),
        'full_pipeline': float(np.mean(
            (np.any(cusum_clean_res['alarm'], axis=1)
             | iswt_clean_res['alarm']
             | lstm_clean_res['alarm'])[W:])),
    }

    # 4b. TPR on faulted data (no TCA evasion)
    print("  Evaluating TPR on faulted data...")
    for fault_mult in config.s1_fault_magnitudes:
        sim_fault = process.simulate(T_eval, fault_config={
            'sensor_idx': [5],
            'fault_start': 0,
            'fault_magnitude': fault_mult,
        }, seed=config.seed + 60000)

        ekf_fault = ExtendedKalmanFilter(sys_cfg, ekf_cfg)
        ekf_fault.set_noise_covariances(Q_hat, R)
        ekf_fault_results = ekf_fault.run_batch(
            sim_fault['y_faulted'], sim_fault['u'])

        cusum_fault = CUSUMDetector(N, config.cusum)
        cusum_fault_res = cusum_fault.run_batch(
            ekf_fault_results['std_innovation'])

        iswt_fault = ISWTDetector(N, iswt_cfg)
        iswt_fault_res = iswt_fault.run_batch(
            ekf_fault_results['std_innovation'])

        lstm_fault_res = lstm_detector.run_batch(
            ekf_fault_results['std_innovation'])

        key = f"tpr_{fault_mult}sigma"
        table6[key] = {
            'cusum_only': float(np.mean(
                np.any(cusum_fault_res['alarm'], axis=1)[W:])),
            'iswt_only': float(np.mean(iswt_fault_res['alarm'][W:])),
            'lstm_only': float(np.mean(lstm_fault_res['alarm'][W:])),
            'cusum_iswt': float(np.mean(
                (np.any(cusum_fault_res['alarm'], axis=1)
                 | iswt_fault_res['alarm'])[W:])),
            'full_pipeline': float(np.mean(
                (np.any(cusum_fault_res['alarm'], axis=1)
                 | iswt_fault_res['alarm']
                 | lstm_fault_res['alarm'])[W:])),
        }

    # Print Table 6
    print("\n  TABLE 6: Detection Comparison")
    print("  " + "-" * 65)
    print(f"  {'Metric':<20} {'CUSUM':>8} {'ISWT':>8} {'LSTM':>8} "
          f"{'C+I':>8} {'Full':>8}")
    print("  " + "-" * 65)

    for metric_key, label in [('fpr', 'FPR (clean)')]:
        row = table6[metric_key]
        print(f"  {label:<20} {row['cusum_only']:>8.3f} "
              f"{row['iswt_only']:>8.3f} {row['lstm_only']:>8.3f} "
              f"{row['cusum_iswt']:>8.3f} {row['full_pipeline']:>8.3f}")

    for fault_mult in config.s1_fault_magnitudes:
        key = f"tpr_{fault_mult}sigma"
        row = table6[key]
        label = f"TPR ({fault_mult}σ fault)"
        print(f"  {label:<20} {row['cusum_only']:>8.3f} "
              f"{row['iswt_only']:>8.3f} {row['lstm_only']:>8.3f} "
              f"{row['cusum_iswt']:>8.3f} {row['full_pipeline']:>8.3f}")

    # Save Table 6
    with open(RESULTS_DIR / "table6_detection.json", 'w') as f:
        json.dump(table6, f, indent=2)

    return {
        'lstm_detector': lstm_detector,
        'calib': calib,
        'table6': table6,
    }


# ======================================================================
# Phase B: Adversarial PGD on LSTM (Table 7)
# ======================================================================

def run_phase_b(config: ExperimentConfig,
                lstm_detector: LSTMDetector,
                calib: dict,
                verbose: bool = True) -> dict:
    """Compare TCA evasion with and without LSTM in the defense.

    Returns:
        Dictionary with table7 results.
    """
    print("\n" + "=" * 70)
    print("PHASE B: Adversarial PGD Attack on LSTM (Table 7)")
    print("=" * 70)

    sys_cfg = config.system
    N = sys_cfg.n_sensors
    ekf_cfg = calib['ekf_config']
    iswt_cfg = calib['iswt_config']
    Q_hat = calib['Q']
    R = calib['R']
    W = iswt_cfg.W

    # --- Mode overrides ---
    if FAST_MODE:
        T_eval       = _FAST_T_EVAL
        fault_mults  = _FAST_FAULT_MULTS
        eps_ratios   = _FAST_EPS_RATIOS
        k_pgd        = _FAST_K_PGD
        print(f"  [FAST_MODE] T_eval={T_eval}, K_pgd={k_pgd}, "
              f"faults={fault_mults}, eps={eps_ratios}")
    else:
        T_eval       = _GPU_T_EVAL
        fault_mults  = _GPU_FAULT_MULTS
        eps_ratios   = _GPU_EPS_RATIOS
        k_pgd        = _GPU_K_PGD
        print(f"  [GPU_MODE] T_eval={T_eval}, K_pgd={k_pgd}, "
              f"faults={fault_mults}, eps={eps_ratios}, "
              f"device={DEVICE}")

    process = TwoTankProcess(sys_cfg)
    table7 = {}

    for fault_mult in fault_mults:
        for eps_ratio in eps_ratios:
            # Wall-clock budget check
            if time_budget_remaining() < 60:
                print("  [BUDGET] 2-hour budget nearly exhausted — skipping remaining Phase B configs.")
                break

            key = f"{fault_mult}sigma_eps{eps_ratio}"
            print(f"\n  --- Fault={fault_mult}σ, ε/σ={eps_ratio} "
                  f"(budget left: {time_budget_remaining()/60:.1f} min) ---")

            # Generate faulted data
            sim = process.simulate(T_eval, fault_config={
                'sensor_idx': [5],
                'fault_start': 0,
                'fault_magnitude': fault_mult,
            }, seed=config.seed + int(fault_mult * 1000))

            Y_faulted = sim['y_faulted']
            U = sim['u']
            epsilon = eps_ratio * sys_cfg.sigma

            # Override K based on mode
            tca_cfg = config.tca
            from dataclasses import replace
            tca_cfg = replace(tca_cfg, K=k_pgd)

            tca = TargetedConsistencyAttack(
                sys_cfg, ekf_cfg, config.cusum, iswt_cfg, tca_cfg)

            # --- TCA without LSTM (existing) ---
            print("    Running TCA (CUSUM+ISWT)...")
            try:
                result_std = tca.run_whitebox(
                    Y_faulted, U, list(range(N)), [5],
                    epsilon, verbose=False)
                delta_std = result_std['delta']
                sds_std = result_std['sds_final']
            except Exception as e:
                print(f"    TCA failed: {e}")
                delta_std = np.zeros_like(Y_faulted)
                sds_std = 0.0

            # Evaluate standard TCA against full pipeline
            Y_std = Y_faulted + delta_std
            ekf_std = ExtendedKalmanFilter(sys_cfg, ekf_cfg)
            ekf_std.set_noise_covariances(Q_hat, R)
            ekf_std_res = ekf_std.run_batch(Y_std, U)

            cusum_std = CUSUMDetector(N, config.cusum)
            cusum_std_res = cusum_std.run_batch(
                ekf_std_res['std_innovation'])
            iswt_std = ISWTDetector(N, iswt_cfg)
            iswt_std_res = iswt_std.run_batch(
                ekf_std_res['std_innovation'])
            lstm_std_res = lstm_detector.run_batch(
                ekf_std_res['std_innovation'])

            cusum_alarm_std = np.any(cusum_std_res['alarm'], axis=1)
            combined_std = cusum_alarm_std | iswt_std_res['alarm']
            full_std = combined_std | lstm_std_res['alarm']

            # --- TCA with LSTM (new) ---
            print("    Running TCA-Neural (CUSUM+ISWT+LSTM)...")
            lstm_model = lstm_detector.get_model()
            lstm_model._threshold = lstm_detector.threshold

            try:
                result_neural = tca.run_whitebox_neural(
                    Y_faulted, U, list(range(N)), [5],
                    epsilon, lstm_model=lstm_model,
                    lstm_config=config.lstm, verbose=False)
                delta_neural = result_neural['delta']
                sds_neural = result_neural['sds_final']
            except Exception as e:
                print(f"    TCA-Neural failed: {e}")
                delta_neural = np.zeros_like(Y_faulted)
                sds_neural = 0.0

            # Evaluate neural TCA against full pipeline
            Y_neural = Y_faulted + delta_neural
            ekf_neural = ExtendedKalmanFilter(sys_cfg, ekf_cfg)
            ekf_neural.set_noise_covariances(Q_hat, R)
            ekf_neural_res = ekf_neural.run_batch(Y_neural, U)

            cusum_neural = CUSUMDetector(N, config.cusum)
            cusum_neural_res = cusum_neural.run_batch(
                ekf_neural_res['std_innovation'])
            iswt_neural = ISWTDetector(N, iswt_cfg)
            iswt_neural_res = iswt_neural.run_batch(
                ekf_neural_res['std_innovation'])
            lstm_neural_res = lstm_detector.run_batch(
                ekf_neural_res['std_innovation'])

            cusum_alarm_neural = np.any(cusum_neural_res['alarm'], axis=1)
            combined_neural = cusum_alarm_neural | iswt_neural_res['alarm']
            full_neural = combined_neural | lstm_neural_res['alarm']

            table7[key] = {
                'fault_mult': fault_mult,
                'eps_ratio': eps_ratio,
                'tca_standard': {
                    'sds': sds_std,
                    'cusum_iswt_tpr': float(np.mean(combined_std[W:])),
                    'full_tpr': float(np.mean(full_std[W:])),
                    'lstm_tpr': float(np.mean(lstm_std_res['alarm'][W:])),
                },
                'tca_neural': {
                    'sds': sds_neural,
                    'cusum_iswt_tpr': float(np.mean(combined_neural[W:])),
                    'full_tpr': float(np.mean(full_neural[W:])),
                    'lstm_tpr': float(np.mean(
                        lstm_neural_res['alarm'][W:])),
                },
            }

            print(f"    Standard TCA: SDS={sds_std:.4f}, "
                  f"C+I TPR={np.mean(combined_std[W:]):.3f}, "
                  f"Full TPR={np.mean(full_std[W:]):.3f}")
            print(f"    Neural  TCA:  SDS={sds_neural:.4f}, "
                  f"C+I TPR={np.mean(combined_neural[W:]):.3f}, "
                  f"Full TPR={np.mean(full_neural[W:]):.3f}")

    # Save Table 7
    with open(RESULTS_DIR / "table7_adversarial_lstm.json", 'w') as f:
        json.dump(table7, f, indent=2)

    return {'table7': table7}


# ======================================================================
# Phase C: GAN vs PGD Comparison (Table 8)
# ======================================================================

def run_phase_c(config: ExperimentConfig,
                lstm_detector: LSTMDetector,
                calib: dict,
                verbose: bool = True) -> dict:
    """Train GAN and compare with PGD.

    Returns:
        Dictionary with table8 results and GAN trainer.
    """
    print("\n" + "=" * 70)
    print("PHASE C: GAN vs PGD Comparison (Table 8)")
    print("=" * 70)

    sys_cfg = config.system
    N = sys_cfg.n_sensors
    ekf_cfg = calib['ekf_config']
    iswt_cfg = calib['iswt_config']
    Q_hat = calib['Q']
    R = calib['R']
    W = iswt_cfg.W

    # --- Mode overrides ---
    if FAST_MODE:
        T_eval         = _FAST_T_EVAL
        fault_mults_c  = _FAST_FAULT_MULTS
        eps_ratios_c   = _FAST_EPS_RATIOS
        n_gan_sessions = _FAST_GAN_SESSIONS
        k_pgd_c        = _FAST_K_PGD
        print(f"  [FAST_MODE] T_eval={T_eval}, K_pgd={k_pgd_c}, "
              f"GAN sessions={n_gan_sessions}")
    else:
        T_eval         = _GPU_T_EVAL
        fault_mults_c  = _GPU_FAULT_MULTS
        eps_ratios_c   = _GPU_EPS_RATIOS
        n_gan_sessions = _GPU_GAN_SESSIONS
        k_pgd_c        = _GPU_K_PGD
        print(f"  [GPU_MODE] T_eval={T_eval}, K_pgd={k_pgd_c}, "
              f"GAN sessions={n_gan_sessions}, device={DEVICE}")

    # --- Step 1: Train GAN ---
    print("\n[C.1] Training GAN evasion generator...")
    lstm_model = lstm_detector.get_model()
    lstm_model._threshold = lstm_detector.threshold

    gan_trainer = train_evasion_gan(
        sys_config=sys_cfg,
        ekf_config=ekf_cfg,
        cusum_config=config.cusum,
        iswt_config=iswt_cfg,
        gan_config=config.gan,
        lstm_model=lstm_model,
        lstm_config=config.lstm,
        n_training_sessions=n_gan_sessions,
        seed=config.seed,
        verbose=verbose,
    )

    # Save GAN model
    torch.save({
        'generator_state': gan_trainer.G.state_dict(),
        'discriminator_state': gan_trainer.D.state_dict(),
        'training_history': gan_trainer.training_history,
        'config': config.gan,
    }, RESULTS_DIR / "gan_model.pt")

    # --- Step 2: Compare GAN vs PGD ---
    print("\n[C.2] Comparing GAN vs PGD...")
    table8 = {}
    process = TwoTankProcess(sys_cfg)
    gan_seq_len = config.gan.seq_len

    for fault_mult in fault_mults_c:
        for eps_ratio in eps_ratios_c:
            # Wall-clock budget check
            if time_budget_remaining() < 60:
                print("  [BUDGET] 2-hour budget nearly exhausted — skipping remaining Phase C configs.")
                break

            key = f"{fault_mult}sigma_eps{eps_ratio}"
            print(f"\n  --- Fault={fault_mult}sigma, eps/sigma={eps_ratio} "
                  f"(budget left: {time_budget_remaining()/60:.1f} min) ---")

            epsilon = eps_ratio * sys_cfg.sigma

            # Generate evaluation data
            sim = process.simulate(T_eval, fault_config={
                'sensor_idx': [5],
                'fault_start': 0,
                'fault_magnitude': fault_mult,
            }, seed=config.seed + int(fault_mult * 2000))

            Y_faulted = sim['y_faulted']
            U = sim['u']

            # --- PGD baseline ---
            print("    Running PGD...")
            t_pgd_start = time.time()

            tca_cfg_c = config.tca
            from dataclasses import replace
            tca_cfg_c = replace(tca_cfg_c, K=k_pgd_c)
            tca = TargetedConsistencyAttack(
                sys_cfg, ekf_cfg, config.cusum, iswt_cfg, tca_cfg_c)
            try:
                result_pgd = tca.run_whitebox(
                    Y_faulted, U, list(range(N)), [5],
                    epsilon, verbose=False)
                delta_pgd = result_pgd['delta']
                sds_pgd = result_pgd['sds_final']
            except Exception:
                delta_pgd = np.zeros_like(Y_faulted)
                sds_pgd = 0.0

            t_pgd = time.time() - t_pgd_start

            # Evaluate PGD
            pgd_metrics = _evaluate_perturbation(
                Y_faulted, U, delta_pgd, sys_cfg, ekf_cfg,
                config.cusum, iswt_cfg, lstm_detector, Q_hat, R, W)

            # --- GAN ---
            print("    Running GAN...")
            t_gan_start = time.time()

            # Generate perturbation (single forward pass)
            delta_gan_raw = gan_trainer.generate_perturbation(
                eps_ratio, fault_mult, n_samples=1)
            delta_gan_seq = delta_gan_raw[0]  # (gan_seq_len, N)

            # Tile GAN output to match evaluation length
            n_tiles = (T_eval // gan_seq_len) + 1
            delta_gan_full = np.tile(delta_gan_seq, (n_tiles, 1))[:T_eval]

            t_gan = time.time() - t_gan_start

            # Evaluate GAN
            gan_metrics = _evaluate_perturbation(
                Y_faulted, U, delta_gan_full, sys_cfg, ekf_cfg,
                config.cusum, iswt_cfg, lstm_detector, Q_hat, R, W)

            table8[key] = {
                'fault_mult': fault_mult,
                'eps_ratio': eps_ratio,
                'pgd': {
                    'sds': sds_pgd,
                    'cusum_iswt_tpr': pgd_metrics['cusum_iswt_tpr'],
                    'full_tpr': pgd_metrics['full_tpr'],
                    'lstm_tpr': pgd_metrics['lstm_tpr'],
                    'time_seconds': t_pgd,
                },
                'gan': {
                    'cusum_iswt_tpr': gan_metrics['cusum_iswt_tpr'],
                    'full_tpr': gan_metrics['full_tpr'],
                    'lstm_tpr': gan_metrics['lstm_tpr'],
                    'time_seconds': t_gan,
                },
                'speedup': t_pgd / max(t_gan, 1e-6),
            }

            print(f"    PGD: SDS={sds_pgd:.4f}, "
                  f"Full TPR={pgd_metrics['full_tpr']:.3f}, "
                  f"time={t_pgd:.2f}s")
            print(f"    GAN: "
                  f"Full TPR={gan_metrics['full_tpr']:.3f}, "
                  f"time={t_gan:.4f}s, "
                  f"speedup={t_pgd/max(t_gan,1e-6):.0f}x")

    # Save Table 8
    with open(RESULTS_DIR / "table8_gan_vs_pgd.json", 'w') as f:
        json.dump(table8, f, indent=2)

    # Save GAN training history for figures
    with open(RESULTS_DIR / "gan_training_history.json", 'w') as f:
        json.dump(gan_trainer.training_history, f, indent=2)

    return {'table8': table8, 'gan_trainer': gan_trainer}


def _evaluate_perturbation(Y_faulted, U, delta, sys_cfg, ekf_cfg,
                           cusum_cfg, iswt_cfg, lstm_detector,
                           Q_hat, R, W):
    """Evaluate a perturbation against the full detection pipeline.

    Returns dict with TPR metrics.
    """
    N = sys_cfg.n_sensors
    Y_attacked = Y_faulted + delta

    ekf = ExtendedKalmanFilter(sys_cfg, ekf_cfg)
    ekf.set_noise_covariances(Q_hat, R)
    ekf_res = ekf.run_batch(Y_attacked, U)

    cusum = CUSUMDetector(N, cusum_cfg)
    cusum_res = cusum.run_batch(ekf_res['std_innovation'])

    iswt = ISWTDetector(N, iswt_cfg)
    iswt_res = iswt.run_batch(ekf_res['std_innovation'])

    lstm_res = lstm_detector.run_batch(ekf_res['std_innovation'])

    cusum_alarm = np.any(cusum_res['alarm'], axis=1)
    combined = cusum_alarm | iswt_res['alarm']
    full = combined | lstm_res['alarm']

    return {
        'cusum_tpr': float(np.mean(cusum_alarm[W:])),
        'iswt_tpr': float(np.mean(iswt_res['alarm'][W:])),
        'lstm_tpr': float(np.mean(lstm_res['alarm'][W:])),
        'cusum_iswt_tpr': float(np.mean(combined[W:])),
        'full_tpr': float(np.mean(full[W:])),
    }


# ======================================================================
# Main
# ======================================================================

def main():
    global _EXPERIMENT_START
    print("=" * 70)
    print("S3 Neural Attack/Defense Evaluation")
    print(f"  Device  : {DEVICE}")
    print(f"  Mode    : {'FAST (smoke-test)' if FAST_MODE else 'FULL (GPU-optimised)'}")
    print(f"  Budget  : {WALL_CLOCK_BUDGET_SEC / 60:.0f} minutes wall-clock")
    print("=" * 70)

    config = ExperimentConfig()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    _EXPERIMENT_START = t_start  # Arm the wall-clock budget

    # Phase A: Train LSTM, establish detection baselines
    phase_a = run_phase_a(config, verbose=True)
    lstm_detector = phase_a['lstm_detector']
    calib = phase_a['calib']

    # Phase B: Adversarial PGD attacks on LSTM
    phase_b = run_phase_b(config, lstm_detector, calib, verbose=True)

    # Phase C: GAN vs PGD comparison
    phase_c = run_phase_c(config, lstm_detector, calib, verbose=True)

    elapsed = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"All S3 experiments complete in {elapsed / 60:.1f} minutes "
          f"(budget used: {elapsed / WALL_CLOCK_BUDGET_SEC * 100:.1f}%)")
    print(f"Results saved to {RESULTS_DIR}")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
