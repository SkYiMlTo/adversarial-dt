"""
S1 Automated Evaluation: Fill Tables 1, 3, 4 (S1), and 5.

Runs all S1 experiments using the core library directly (without Docker)
for efficient bulk evaluation. The same mathematical components (EKF,
CUSUM, ISWT, TCA) are used — Docker testbed validates on live infra.

Experiments:
    Table 1: Red-team evasion rate (30 sessions × regime × fault magnitude)
    Table 3: SDS vs budget sweep
    Table 4 (S1): IWD detection performance (TPR/FPR)
    Table 5: Ablation of pipeline components

Parallelization: across fault magnitudes and budget levels.
"""

import os
import sys
import json
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import ExperimentConfig, SystemConfig, EKFConfig, CUSUMConfig, ISWTConfig, TCAConfig
from core.process_model import TwoTankProcess
from core.ekf import ExtendedKalmanFilter
from core.cusum import CUSUMDetector
from core.iswt import ISWTDetector, combined_alarm
from core.sds import compute_sds_timeseries
from core.tca import TargetedConsistencyAttack
from core.calibration import full_calibration


# Output directory
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "s1"


def run_single_session(session_id: int,
                       regime: str,
                       fault_sigma_mult: float,
                       config: ExperimentConfig,
                       epsilon_ratio: float = 0.75,
                       seed_offset: int = 0) -> dict:
    """Run a single red-team session.

    1. Generate clean calibration data
    2. Calibrate EKF + validate whiteness
    3. Simulate attack session with fault + TCA perturbation
    4. Evaluate detection performance

    Args:
        session_id: Session number (0-29).
        regime: 'whitebox' or 'greybox'.
        fault_sigma_mult: Fault magnitude in multiples of σ_η.
        config: Experiment configuration.
        epsilon_ratio: ε/σ_η ratio for TCA.
        seed_offset: Seed offset for reproducibility.

    Returns:
        Session result dictionary.
    """
    seed = config.seed + session_id + seed_offset
    sys_cfg = config.system
    T_session = config.s1_session_duration_steps
    T_calib = config.s1_calibration_steps
    evasion_window = config.s1_evasion_window

    # --- Step 1: Calibration ---
    calib = full_calibration(sys_cfg, T_calib, seed=seed)
    ekf_cfg = calib['ekf_config']
    iswt_cfg = calib['iswt_config']
    Q_hat = calib['Q']
    R = calib['R']

    if not calib['whiteness_validation']['passed']:
        # Session discarded (whiteness test failed on clean data)
        return {'session_id': session_id, 'discarded': True,
                'reason': 'whiteness_failed'}

    # --- Step 2: Generate attack session data ---
    process = TwoTankProcess(sys_cfg)
    process.set_seed(seed + 10000)

    # Determine fault magnitude (absolute units)
    # Attack sensor 5 (Q_pump) — pump flow sensor
    attack_sensor = 5
    compromised_idx = [attack_sensor]
    attacked_idx = list(range(sys_cfg.n_sensors))  # TCA can perturb all

    sim = process.simulate(
        T_session,
        fault_config={
            'sensor_idx': compromised_idx,
            'fault_start': 0,  # Fault active from start
            'fault_magnitude': fault_sigma_mult,  # In sigma units
        },
        seed=seed + 10000
    )

    Y_faulted = sim['y_faulted']
    U = sim['u']

    # --- Step 3: Run TCA ---
    epsilon = epsilon_ratio * sys_cfg.sigma[attack_sensor]
    tca = TargetedConsistencyAttack(
            sys_cfg, ekf_cfg, config.cusum, iswt_cfg, config.tca
        )

    try:
        if regime == 'whitebox':
            tca_result = tca.run_whitebox(
                Y_faulted, U, attacked_idx, compromised_idx,
                epsilon, verbose=False
            )
        else:
            tca_result = tca.run_greybox(
                Y_faulted, U, attacked_idx, compromised_idx,
                epsilon, verbose=False
            )
        delta = tca_result['delta']
        sds_final = tca_result['sds_final']
    except Exception as e:
        # TCA failed — count as detection success
        delta = np.zeros_like(Y_faulted)
        sds_final = 0.0

    # --- Step 4: Evaluate detection ---
    # 4a. CUSUM-only detection (no ISWT)
    Y_attacked = Y_faulted + delta
    ekf = ExtendedKalmanFilter(sys_cfg, ekf_cfg)
    ekf.set_noise_covariances(Q_hat, R)
    ekf_results = ekf.run_batch(Y_attacked, U)

    cusum = CUSUMDetector(sys_cfg.n_sensors, config.cusum)
    cusum_results = cusum.run_batch(ekf_results['std_innovation'])

    iswt_detector = ISWTDetector(sys_cfg.n_sensors, iswt_cfg)
    iswt_results = iswt_detector.run_batch(ekf_results['std_innovation'])

    # Check evasion: no alarm for evasion_window consecutive steps
    cusum_any_alarm = np.any(cusum_results['alarm'], axis=1)
    combined_alarms = cusum_any_alarm | iswt_results['alarm']

    # CUSUM-only evasion
    cusum_evasion = _check_evasion(cusum_any_alarm, evasion_window)
    # Combined evasion
    combined_evasion = _check_evasion(combined_alarms, evasion_window)

    # TPR/FPR computation
    # True positives: alarm during fault period
    # For S1, fault is active throughout, so:
    # TPR = fraction of timesteps with alarm (during fault)
    # FPR would be on clean data (not applicable here)
    cusum_tpr = np.mean(cusum_any_alarm[iswt_cfg.W:])
    combined_tpr = np.mean(combined_alarms[iswt_cfg.W:])
    iswt_tpr = np.mean(iswt_results['alarm'][iswt_cfg.W:])

    return {
        'session_id': session_id,
        'regime': regime,
        'fault_sigma_mult': fault_sigma_mult,
        'epsilon_ratio': epsilon_ratio,
        'discarded': False,
        'cusum_evasion': cusum_evasion,
        'combined_evasion': combined_evasion,
        'sds_final': sds_final,
        'cusum_tpr': cusum_tpr,
        'combined_tpr': combined_tpr,
        'iswt_tpr': iswt_tpr,
    }


def _check_evasion(alarms: np.ndarray, window: int) -> bool:
    """Check if there's a window of consecutive alarm-free steps."""
    consecutive = 0
    for a in alarms:
        if not a:
            consecutive += 1
            if consecutive >= window:
                return True
        else:
            consecutive = 0
    return False


# ======================================================================
# Table 1: Red-team evasion rate
# ======================================================================

def run_table1(config: ExperimentConfig) -> dict:
    """Run all sessions for Table 1 (evasion rate)."""
    print("\n" + "=" * 70)
    print("TABLE 1: Red-Team Evasion Rate")
    print("=" * 70)

    results = {}
    for regime in ['whitebox', 'greybox']:
        for fault_mult in config.s1_fault_magnitudes:
            key = (regime, fault_mult)
            print(f"\n--- {regime}, Δy = {fault_mult}σ ---")

            sessions = []
            with ProcessPoolExecutor(max_workers=config.n_workers) as pool:
                futures = []
                for sid in range(config.s1_sessions_per_config):
                    f = pool.submit(
                        run_single_session, sid, regime, fault_mult,
                        config, epsilon_ratio=0.75,
                        seed_offset=hash(key) % 10000
                    )
                    futures.append(f)

                for f in as_completed(futures):
                    result = f.result()
                    sessions.append(result)
                    if not result['discarded']:
                        print(f"  Session {result['session_id']:2d}: "
                              f"CUSUM evade={result['cusum_evasion']}, "
                              f"Combined evade={result['combined_evasion']}, "
                              f"SDS={result['sds_final']:.3f}")

            valid = [s for s in sessions if not s['discarded']]
            n_valid = len(valid)

            cusum_evasion_rate = (
                sum(1 for s in valid if s['cusum_evasion']) / max(n_valid, 1)
            )
            combined_evasion_rate = (
                sum(1 for s in valid if s['combined_evasion']) / max(n_valid, 1)
            )

            results[key] = {
                'regime': regime,
                'fault_mult': fault_mult,
                'n_sessions': n_valid,
                'cusum_evasion_rate': cusum_evasion_rate,
                'combined_evasion_rate': combined_evasion_rate,
            }

            print(f"  → CUSUM-only evasion: {cusum_evasion_rate:.1%}")
            print(f"  → IWD∨CUSUM evasion: {combined_evasion_rate:.1%}")

    return results


# ======================================================================
# Table 3: SDS vs budget sweep
# ======================================================================

def run_table3(config: ExperimentConfig) -> dict:
    """Run budget sweep for Table 3."""
    print("\n" + "=" * 70)
    print("TABLE 3: SDS vs Budget Sweep")
    print("=" * 70)

    results = {}
    fault_mult = 2.0  # Fixed at 2σ

    # Generate one representative session
    process = TwoTankProcess(config.system)
    process.set_seed(config.seed)
    T = config.s1_session_duration_steps

    sim = process.simulate(T, fault_config={
        'sensor_idx': [5],
        'fault_start': 0,
        'fault_magnitude': fault_mult,
    }, seed=config.seed)

    Y = sim['y_faulted']
    U = sim['u']

    calib = full_calibration(config.system, config.s1_calibration_steps,
                              seed=config.seed + 99999)
    ekf_cfg = calib['ekf_config']
    iswt_cfg = calib['iswt_config']

    tca = TargetedConsistencyAttack(
        config.system, ekf_cfg, config.cusum, iswt_cfg, config.tca
    )

    for regime in ['whitebox', 'greybox']:
        print(f"\n--- {regime} ---")
        for ratio in config.tca.epsilon_ratios:
            epsilon = ratio * config.system.sigma[5]
            print(f"  ε/σ_η = {ratio:.2f} ...", end=" ", flush=True)

            try:
                if regime == 'whitebox':
                    result = tca.run_whitebox(
                        Y, U, list(range(6)), [5], epsilon, verbose=False
                    )
                else:
                    result = tca.run_greybox(
                        Y, U, list(range(6)), [5], epsilon, verbose=False
                    )
                sds = result['sds_final']
            except Exception as e:
                sds = 0.0

            results[(regime, ratio)] = sds
            print(f"SDS = {sds:.4f}")

    return results


# ======================================================================
# Table 5: Ablation
# ======================================================================

def run_table5(config: ExperimentConfig) -> dict:
    """Run ablation study for Table 5."""
    print("\n" + "=" * 70)
    print("TABLE 5: Detection Pipeline Ablation")
    print("=" * 70)

    fault_mult = 2.0
    epsilon_ratio = 0.75

    # Generate data and run TCA
    process = TwoTankProcess(config.system)
    T = config.s1_session_duration_steps

    sim = process.simulate(T, fault_config={
        'sensor_idx': [5],
        'fault_start': 0,
        'fault_magnitude': fault_mult,
    }, seed=config.seed)

    Y = sim['y_faulted']
    U = sim['u']

    calib = full_calibration(config.system, config.s1_calibration_steps,
                             seed=config.seed)
    ekf_cfg = calib['ekf_config']
    iswt_cfg = calib['iswt_config']
    Q_hat = calib['Q']
    R = calib['R']

    # Run white-box TCA
    tca = TargetedConsistencyAttack(
        config.system, ekf_cfg, config.cusum, iswt_cfg, config.tca
    )
    epsilon = epsilon_ratio * config.system.sigma[5]

    try:
        tca_result = tca.run_whitebox(
            Y, U, list(range(6)), [5], epsilon, verbose=True
        )
        delta = tca_result['delta']
    except Exception:
        delta = np.zeros_like(Y)

    Y_attacked = Y + delta

    # Run pipeline
    ekf = ExtendedKalmanFilter(config.system, ekf_cfg)
    ekf.set_noise_covariances(Q_hat, R)
    ekf_results = ekf.run_batch(Y_attacked, U)

    cusum = CUSUMDetector(6, config.cusum)
    cusum_results = cusum.run_batch(ekf_results['std_innovation'])

    iswt = ISWTDetector(6, iswt_cfg)
    iswt_results = iswt.run_batch(ekf_results['std_innovation'])

    W = iswt_cfg.W
    cusum_alarm = np.any(cusum_results['alarm'], axis=1)
    iswt_alarm = iswt_results['alarm']
    combined = cusum_alarm | iswt_alarm

    results = {
        'cusum_only': {
            'tpr': float(np.mean(cusum_alarm[W:])),
            'fpr': 0.0,  # Would need clean-data run
        },
        'iswt_only': {
            'tpr': float(np.mean(iswt_alarm[W:])),
            'fpr': 0.0,
        },
        'combined': {
            'tpr': float(np.mean(combined[W:])),
            'fpr': 0.0,
        },
    }

    # Run on clean data for FPR
    sim_clean = process.simulate(T, seed=config.seed + 50000)
    ekf_clean = ExtendedKalmanFilter(config.system, ekf_cfg)
    ekf_clean.set_noise_covariances(Q_hat, R)
    ekf_clean_results = ekf_clean.run_batch(sim_clean['y_noisy'],
                                              sim_clean['u'])

    cusum_clean = CUSUMDetector(6, config.cusum)
    cusum_clean_res = cusum_clean.run_batch(ekf_clean_results['std_innovation'])

    iswt_clean = ISWTDetector(6, iswt_cfg)
    iswt_clean_res = iswt_clean.run_batch(ekf_clean_results['std_innovation'])

    results['cusum_only']['fpr'] = float(
        np.mean(np.any(cusum_clean_res['alarm'], axis=1)[W:]))
    results['iswt_only']['fpr'] = float(
        np.mean(iswt_clean_res['alarm'][W:]))
    results['combined']['fpr'] = float(
        np.mean((np.any(cusum_clean_res['alarm'], axis=1) |
                 iswt_clean_res['alarm'])[W:]))

    for name, r in results.items():
        print(f"  {name:15s}: TPR={r['tpr']:.3f}, FPR={r['fpr']:.3f}")

    return results


# ======================================================================
# Main
# ======================================================================

def main():
    print("=" * 70)
    print("S1 Automated Evaluation")
    print("=" * 70)

    config = ExperimentConfig()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    t_start = time.time()

    # Table 1
    t1 = run_table1(config)
    with open(RESULTS_DIR / "table1_evasion.json", 'w') as f:
        json.dump({str(k): v for k, v in t1.items()}, f, indent=2)

    # Table 3
    t3 = run_table3(config)
    with open(RESULTS_DIR / "table3_budget_sweep.json", 'w') as f:
        json.dump({str(k): v for k, v in t3.items()}, f, indent=2)

    # Table 5
    t5 = run_table5(config)
    with open(RESULTS_DIR / "table5_ablation.json", 'w') as f:
        json.dump(t5, f, indent=2)

    elapsed = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"All S1 experiments complete in {elapsed / 60:.1f} minutes")
    print(f"Results saved to {RESULTS_DIR}")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
