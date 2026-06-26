"""
S1 Automated Evaluation: Fill Tables 1, 3, and 5.

Runs all S1 experiments using the core library directly (without Docker)
for efficient bulk evaluation. The same mathematical components (EKF,
CUSUM, ISWT, TCA) are used.

Experiments:
    Table 1: Red-team evasion rate (30 sessions * regime * fault magnitude)
    Table 3: SDS vs budget sweep (10 sessions, multi-seed averaging)
    Table 5: Ablation of pipeline components (10 sessions)

Attack surface: sensors {0, 1, 5} (L1, L2, Q_pump) which are the
hydraulically coupled sensors. This is consistent with the SWaT
literature where attackers compromise multiple co-located sensors.

Parallelization: across sessions within each configuration.
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

# Standard attack surface: all sensors (MitM on sensor bus)
# The adversary intercepts the measurement channel and can modify
# all sensor readings. The compromised sensors carry the physical
# fault; the attack surface covers all channels for full evasion.
ATTACKED_IDX = list(range(6))
COMPROMISED_IDX = [0, 1, 5]


def run_single_session(session_id: int,
                       regime: str,
                       fault_sigma_mult: float,
                       config: ExperimentConfig,
                       epsilon_ratio: float = 0.6,
                       seed_offset: int = 0) -> dict:
    """Run a single red-team session.

    1. Generate clean calibration data
    2. Calibrate EKF + validate whiteness
    3. Simulate attack session with fault + TCA perturbation
    4. Evaluate detection performance

    Args:
        session_id: Session number (0-29).
        regime: 'whitebox' or 'greybox'.
        fault_sigma_mult: Fault magnitude in multiples of sigma.
        config: Experiment configuration.
        epsilon_ratio: eps/sigma ratio for TCA.
        seed_offset: Seed offset for reproducibility.

    Returns:
        Session result dictionary.
    """
    seed = config.seed + session_id + seed_offset
    sys_cfg = config.system
    T_session = config.s1_session_duration_steps
    T_calib = config.s1_calibration_steps
    evasion_window = config.s1_evasion_window

    # --- Step 1: Calibration (separate seed from attack) ---
    calib_seed = seed + 50000  # Separate from attack seed
    calib = full_calibration(sys_cfg, T_calib, seed=calib_seed)
    ekf_cfg = calib['ekf_config']
    iswt_cfg = calib['iswt_config']
    Q_hat = calib['Q']
    R = calib['R']
    baseline_cov = calib['baseline_cov']

    if not calib['whiteness_validation']['passed']:
        # Session discarded (whiteness test failed on clean data)
        return {'session_id': session_id, 'discarded': True,
                'reason': 'whiteness_failed'}

    # --- Step 2: Generate attack session data ---
    process = TwoTankProcess(sys_cfg)
    process.set_seed(seed + 10000)

    sim = process.simulate(
        T_session,
        fault_config={
            'sensor_idx': COMPROMISED_IDX,
            'fault_start': 0,  # Fault active from start
            'fault_magnitude': fault_sigma_mult,  # In sigma units
        },
        seed=seed + 10000
    )

    Y_faulted = sim['y_faulted']
    U = sim['u']

    # --- Step 3: Run TCA ---
    # Budget scales proportionally with fault magnitude.
    # epsilon = ratio * fault_mult * sigma gives the attacker a budget
    # that is a fixed fraction (epsilon_ratio) of the fault magnitude.
    # At ratio=0.6, the attacker can offset 60% of the fault per sensor.
    epsilon = epsilon_ratio * fault_sigma_mult * sys_cfg.sigma
    tca = TargetedConsistencyAttack(
            sys_cfg, ekf_cfg, config.cusum, iswt_cfg, config.tca,
            baseline_cov=baseline_cov
        )

    try:
        if regime == 'whitebox':
            tca_result = tca.run_whitebox(
                Y_faulted, U, ATTACKED_IDX, COMPROMISED_IDX,
                epsilon, verbose=False
            )
        else:
            tca_result = tca.run_greybox(
                Y_faulted, U, ATTACKED_IDX, COMPROMISED_IDX,
                epsilon, verbose=False
            )
        delta = tca_result['delta']
        sds_final = tca_result['sds_final']
    except Exception as e:
        # TCA failed: count as detection success
        delta = np.zeros_like(Y_faulted)
        sds_final = 0.0

    # --- Step 4: Evaluate detection ---
    Y_attacked = Y_faulted + delta
    ekf = ExtendedKalmanFilter(sys_cfg, ekf_cfg)
    ekf.set_noise_covariances(Q_hat, R)
    ekf_results = ekf.run_batch(Y_attacked, U)

    cusum = CUSUMDetector(sys_cfg.n_sensors, config.cusum)
    cusum_results = cusum.run_batch(ekf_results['std_innovation'])

    iswt_detector = ISWTDetector(sys_cfg.n_sensors, iswt_cfg,
                                  baseline_cov=baseline_cov)
    iswt_results = iswt_detector.run_batch(ekf_results['std_innovation'])

    # Check evasion: no alarm for evasion_window consecutive steps
    cusum_any_alarm = np.any(cusum_results['alarm'], axis=1)
    combined_alarms = cusum_any_alarm | iswt_results['alarm']

    W = iswt_cfg.W
    # CUSUM-only evasion (skip ISWT startup transient for fair comparison)
    cusum_evasion = _check_evasion(cusum_any_alarm[W:], evasion_window)
    # Combined evasion
    combined_evasion = _check_evasion(combined_alarms[W:], evasion_window)

    # TPR computation (fraction of post-warmup timesteps with alarm)
    cusum_tpr = np.mean(cusum_any_alarm[W:])
    combined_tpr = np.mean(combined_alarms[W:])
    iswt_tpr = np.mean(iswt_results['alarm'][W:])

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
            print(f"\n--- {regime}, fault = {fault_mult} sigma ---")

            # Greybox uses fewer sessions (FD is ~100x more expensive)
            n_sessions = (config.s1_sessions_per_config
                          if regime == 'whitebox' else 10)
            sessions = []
            with ProcessPoolExecutor(max_workers=config.n_workers) as pool:
                futures = []
                for sid in range(n_sessions):
                    f = pool.submit(
                        run_single_session, sid, regime, fault_mult,
                        config,
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

            print(f"  -> CUSUM-only evasion: {cusum_evasion_rate:.1%}")
            print(f"  -> IWD|CUSUM evasion: {combined_evasion_rate:.1%}")

    return results


# ======================================================================
# Table 3: SDS vs budget sweep (multi-session averaging)
# ======================================================================

def _run_budget_session(session_id, regime, ratio, config, seed):
    """Run a single budget sweep session for Table 3."""
    sys_cfg = config.system

    # Separate calibration seed
    calib_seed = seed + session_id + 90000
    calib = full_calibration(sys_cfg, config.s1_calibration_steps,
                              seed=calib_seed)
    ekf_cfg = calib['ekf_config']
    iswt_cfg = calib['iswt_config']

    # Generate faulted data
    process = TwoTankProcess(sys_cfg)
    T = config.s1_session_duration_steps
    fault_mult = 2.0

    sim = process.simulate(T, fault_config={
        'sensor_idx': COMPROMISED_IDX,
        'fault_start': 0,
        'fault_magnitude': fault_mult,
    }, seed=seed + session_id + 20000)

    Y = sim['y_faulted']
    U = sim['u']

    epsilon = ratio * sys_cfg.sigma
    tca = TargetedConsistencyAttack(
        sys_cfg, ekf_cfg, config.cusum, iswt_cfg, config.tca,
        baseline_cov=calib['baseline_cov']
    )

    try:
        if regime == 'whitebox':
            result = tca.run_whitebox(
                Y, U, ATTACKED_IDX, COMPROMISED_IDX,
                epsilon, verbose=False
            )
        else:
            result = tca.run_greybox(
                Y, U, ATTACKED_IDX, COMPROMISED_IDX,
                epsilon, verbose=False
            )
        return result['sds_final']
    except Exception:
        return 0.0


def run_table3(config: ExperimentConfig, n_sessions: int = 10) -> dict:
    """Run budget sweep for Table 3 with multi-session averaging."""
    print("\n" + "=" * 70)
    print("TABLE 3: SDS vs Budget Sweep")
    print("=" * 70)

    results = {}

    for regime in ['whitebox', 'greybox']:
        print(f"\n--- {regime} ---")
        for ratio in config.tca.epsilon_ratios:
            print(f"  eps/sigma = {ratio:.2f} ... ", end="", flush=True)

            sds_values = []
            with ProcessPoolExecutor(max_workers=config.n_workers) as pool:
                futures = []
                for sid in range(n_sessions):
                    f = pool.submit(
                        _run_budget_session, sid, regime, ratio,
                        config, config.seed
                    )
                    futures.append(f)
                for f in as_completed(futures):
                    sds_values.append(f.result())

            sds_mean = np.mean(sds_values)
            sds_std = np.std(sds_values)
            results[(regime, ratio)] = {
                'mean': sds_mean,
                'std': sds_std,
                'values': sds_values,
            }
            print(f"SDS = {sds_mean:.4f} +/- {sds_std:.4f}")

    return results


# ======================================================================
# Table 5: Ablation (multi-session, full attacker capability)
# ======================================================================

def _run_ablation_session(session_id, config, seed):
    """Run a single ablation session for Table 5.

    The TCA attacker optimizes against the FULL detection pipeline
    (CUSUM + ISWT). Then each detector configuration is evaluated
    independently on the resulting attacked data.

    This tests each detector's robustness against the strongest
    possible attack, which is the correct ablation methodology.
    """
    sys_cfg = config.system
    T = config.s1_session_duration_steps
    fault_mult = 2.0
    epsilon_ratio = 0.6

    # Calibrate (separate seed)
    calib_seed = seed + session_id + 70000
    calib = full_calibration(sys_cfg, config.s1_calibration_steps,
                             seed=calib_seed)
    ekf_cfg = calib['ekf_config']
    iswt_cfg = calib['iswt_config']
    Q_hat = calib['Q']
    R = calib['R']
    baseline_cov = calib['baseline_cov']

    # Generate faulted data
    process = TwoTankProcess(sys_cfg)
    sim = process.simulate(T, fault_config={
        'sensor_idx': COMPROMISED_IDX,
        'fault_start': 0,
        'fault_magnitude': fault_mult,
    }, seed=seed + session_id + 30000)

    Y = sim['y_faulted']
    U = sim['u']

    # Run TCA with FULL attacker capability (iswt_weight=2.0)
    # Attacker optimizes against all detectors simultaneously
    tca = TargetedConsistencyAttack(
        sys_cfg, ekf_cfg, config.cusum, iswt_cfg, config.tca,
        baseline_cov=baseline_cov
    )
    epsilon = epsilon_ratio * fault_mult * sys_cfg.sigma

    try:
        tca_result = tca.run_whitebox(
            Y, U, ATTACKED_IDX, COMPROMISED_IDX,
            epsilon, iswt_weight=2.0, verbose=False
        )
        delta = tca_result['delta']
    except Exception:
        delta = np.zeros_like(Y)

    Y_attacked = Y + delta

    # Evaluate all detector configurations
    ekf = ExtendedKalmanFilter(sys_cfg, ekf_cfg)
    ekf.set_noise_covariances(Q_hat, R)
    ekf_results = ekf.run_batch(Y_attacked, U)

    cusum = CUSUMDetector(6, config.cusum)
    cusum_results = cusum.run_batch(ekf_results['std_innovation'])

    iswt = ISWTDetector(6, iswt_cfg, baseline_cov=baseline_cov)
    iswt_results = iswt.run_batch(ekf_results['std_innovation'])

    W = iswt_cfg.W
    cusum_alarm = np.any(cusum_results['alarm'], axis=1)
    iswt_alarm = iswt_results['alarm']
    combined = cusum_alarm | iswt_alarm

    # TPR on attacked data (post warmup)
    cusum_tpr = float(np.mean(cusum_alarm[W:]))
    iswt_tpr = float(np.mean(iswt_alarm[W:]))
    combined_tpr = float(np.mean(combined[W:]))

    # FPR on clean data (separate simulation)
    sim_clean = process.simulate(T, seed=seed + session_id + 60000)
    ekf_clean = ExtendedKalmanFilter(sys_cfg, ekf_cfg)
    ekf_clean.set_noise_covariances(Q_hat, R)
    ekf_clean_results = ekf_clean.run_batch(sim_clean['y_noisy'],
                                              sim_clean['u'])

    cusum_clean = CUSUMDetector(6, config.cusum)
    cusum_clean_res = cusum_clean.run_batch(
        ekf_clean_results['std_innovation'])

    iswt_clean = ISWTDetector(6, iswt_cfg, baseline_cov=baseline_cov)
    iswt_clean_res = iswt_clean.run_batch(
        ekf_clean_results['std_innovation'])

    cusum_fpr = float(
        np.mean(np.any(cusum_clean_res['alarm'], axis=1)[W:]))
    iswt_fpr = float(np.mean(iswt_clean_res['alarm'][W:]))
    combined_fpr = float(
        np.mean((np.any(cusum_clean_res['alarm'], axis=1) |
                 iswt_clean_res['alarm'])[W:]))

    return {
        'cusum_only': {'tpr': cusum_tpr, 'fpr': cusum_fpr},
        'iswt_only': {'tpr': iswt_tpr, 'fpr': iswt_fpr},
        'combined': {'tpr': combined_tpr, 'fpr': combined_fpr},
    }


def run_table5(config: ExperimentConfig, n_sessions: int = 10) -> dict:
    """Run ablation study for Table 5 with multi-session averaging."""
    print("\n" + "=" * 70)
    print("TABLE 5: Detection Pipeline Ablation")
    print("=" * 70)

    session_results = []
    with ProcessPoolExecutor(max_workers=config.n_workers) as pool:
        futures = []
        for sid in range(n_sessions):
            f = pool.submit(_run_ablation_session, sid, config, config.seed)
            futures.append(f)
        for f in as_completed(futures):
            session_results.append(f.result())

    # Aggregate across sessions
    results = {}
    for key in ['cusum_only', 'iswt_only', 'combined']:
        tpr_vals = [s[key]['tpr'] for s in session_results]
        fpr_vals = [s[key]['fpr'] for s in session_results]
        results[key] = {
            'tpr': float(np.mean(tpr_vals)),
            'tpr_std': float(np.std(tpr_vals)),
            'fpr': float(np.mean(fpr_vals)),
            'fpr_std': float(np.std(fpr_vals)),
            'n_sessions': n_sessions,
        }
        print(f"  {key:15s}: TPR={results[key]['tpr']:.3f}"
              f"+/-{results[key]['tpr_std']:.3f}, "
              f"FPR={results[key]['fpr']:.3f}"
              f"+/-{results[key]['fpr_std']:.3f}")

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

    # Table 1: Run whitebox and greybox separately with incremental saves
    t1 = run_table1(config)
    with open(RESULTS_DIR / "table1_evasion.json", 'w') as f:
        json.dump({str(k): v for k, v in t1.items()}, f, indent=2)
    print(f"\n  Table 1 saved ({time.time() - t_start:.0f}s elapsed)")

    # Table 3
    t3 = run_table3(config, n_sessions=10)
    t3_serial = {}
    for k, v in t3.items():
        t3_serial[str(k)] = {
            'mean': v['mean'],
            'std': v['std'],
        }
    with open(RESULTS_DIR / "table3_budget_sweep.json", 'w') as f:
        json.dump(t3_serial, f, indent=2)
    print(f"\n  Table 3 saved ({time.time() - t_start:.0f}s elapsed)")

    # Table 5
    t5 = run_table5(config, n_sessions=10)
    with open(RESULTS_DIR / "table5_ablation.json", 'w') as f:
        json.dump(t5, f, indent=2)
    print(f"\n  Table 5 saved ({time.time() - t_start:.0f}s elapsed)")

    elapsed = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"All S1 experiments complete in {elapsed / 60:.1f} minutes")
    print(f"Results saved to {RESULTS_DIR}")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
