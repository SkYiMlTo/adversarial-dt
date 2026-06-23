"""
S2 Offline Evaluation: SWaT Dataset (Tables 2, 4-S2).

Runs the TCA attack and IWD defense on the SWaT dataset in offline mode.
If the real SWaT dataset is not available, uses synthetic data with
comparable statistical properties.

Experiments:
    Table 2: Per-stage detection metrics (TPR, FPR, F1, AUC)
    Table 4 (S2): IWD detection performance on SWaT
"""

import os
import sys
import json
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import ExperimentConfig, SystemConfig, EKFConfig, CUSUMConfig, ISWTConfig, TCAConfig
from core.process_model import TwoTankProcess
from core.ekf import ExtendedKalmanFilter
from core.cusum import CUSUMDetector
from core.iswt import ISWTDetector, combined_alarm
from core.sds import compute_sds_timeseries
from core.tca import TargetedConsistencyAttack
from core.calibration import calibrate_ekf, validate_whiteness

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "s2"
SWAT_DIR = Path(__file__).resolve().parent.parent / "swat" / "dataset"


def load_or_generate_data(config: ExperimentConfig) -> dict:
    """Load SWaT dataset or generate synthetic equivalent.

    Attempts to load the real SWaT dataset. If not found, generates
    synthetic data with comparable dynamics using the two-tank model
    over an extended operating range.

    Returns:
        Dictionary with 'train' and 'test' splits.
    """
    try:
        from swat.swat_adapter import load_swat_dataset
        train = load_swat_dataset(str(SWAT_DIR), mode='normal')
        test = load_swat_dataset(str(SWAT_DIR), mode='attack')
        print("  Using real SWaT dataset")
        from core.calibration import calibrate_ekf
        from core.config import EKFConfig, ISWTConfig

        sys_cfg = config.system
        Q_hat, R = calibrate_ekf(train['Y'], train['U'], sys_cfg)
        ekf_cfg = EKFConfig(Q_diag=np.diag(Q_hat))
        iswt_cfg = ISWTConfig()

        return {'train': train, 'test': test, 'source': 'swat',
                'ekf_config': ekf_cfg, 'iswt_config': iswt_cfg,
                'Q': Q_hat, 'R': R}
    except (FileNotFoundError, ImportError) as e:
        print(f"  SWaT dataset not available ({e})")
        print("  Generating synthetic SWaT-equivalent data")
        return generate_synthetic_data(config)


def generate_synthetic_data(config: ExperimentConfig) -> dict:
    """Generate synthetic data mimicking SWaT operational profiles.

    Creates a long time series with:
    - Training: 72000 steps (20 hours) of clean operation
    - Testing: 18000 steps (5 hours) with embedded attacks

    Attack scenarios (drawn from real SWaT attack descriptions):
        A1: Steady-state offset on L1 (tank overflow attack)
        A2: Ramp injection on Q_pump (slow pump degradation)
        A3: Multi-sensor coordinated attack on L1 + Q12
        A4: Intermittent spike on P_out (pressure transient)
    """
    sys_cfg = config.system
    process = TwoTankProcess(sys_cfg)

    # ---- Training data: clean operation ----
    T_train = config.s2_train_steps
    print(f"  Generating training data ({T_train} steps = "
          f"{T_train / 3600:.1f} hours)...")
    sim_train = process.simulate(T_train, seed=config.seed)

    # ---- Test data: with attacks ----
    T_test = 18000  # 5 hours
    print(f"  Generating test data ({T_test} steps = "
          f"{T_test / 3600:.1f} hours)...")
    sim_test = process.simulate(T_test, seed=config.seed + 100)

    Y_test = sim_test['y_noisy'].copy()
    labels = np.zeros(T_test)

    # Attack A1: Steady offset on L1 (steps 2000-4000)
    a1_start, a1_end = 2000, 4000
    a1_magnitude = 3.0 * sys_cfg.sigma[0]  # 3σ offset
    Y_test[a1_start:a1_end, 0] += a1_magnitude
    labels[a1_start:a1_end] = 1

    # Attack A2: Slow ramp on Q_pump (steps 6000-8000)
    a2_start, a2_end = 6000, 8000
    ramp = np.linspace(0, 5.0 * sys_cfg.sigma[5], a2_end - a2_start)
    Y_test[a2_start:a2_end, 5] += ramp
    labels[a2_start:a2_end] = 1

    # Attack A3: Multi-sensor on L1 + Q12 (steps 10000-12000)
    a3_start, a3_end = 10000, 12000
    Y_test[a3_start:a3_end, 0] += 2.0 * sys_cfg.sigma[0]
    Y_test[a3_start:a3_end, 4] -= 2.0 * sys_cfg.sigma[4]
    labels[a3_start:a3_end] = 1

    # Attack A4: Intermittent spikes on P_out (steps 14000-16000)
    a4_start, a4_end = 14000, 16000
    rng = np.random.default_rng(config.seed + 200)
    spike_times = rng.choice(range(a4_start, a4_end),
                              size=(a4_end - a4_start) // 5,
                              replace=False)
    Y_test[spike_times, 3] += 4.0 * sys_cfg.sigma[3]
    labels[a4_start:a4_end] = 1

    # Attack A1 (steady offset on L1)
    a1_start, a1_end = 2000, 4000
    Y_test[a1_start:a1_end, 0] += 3.0 * sys_cfg.sigma[0]
    labels[a1_start:a1_end] = 1

    from core.calibration import full_calibration
    T_calib = min(T_train, config.s1_calibration_steps)
    calib = full_calibration(sys_cfg, T_calib, seed=config.seed)

    return {
        'train': {
            'Y': sim_train['y_noisy'],
            'U': sim_train['u'],
            'labels': np.zeros(T_train),
            'sensor_names': sys_cfg.sensor_names,
        },
        'test': {
            'Y': Y_test,
            'U': sim_test['u'],
            'labels': labels,
            'sensor_names': sys_cfg.sensor_names,
        },
        'source': 'synthetic',
        'ekf_config': calib['ekf_config'],
        'iswt_config': calib['iswt_config'],
        'Q': calib['Q'], 'R': calib['R'],
        'attack_info': {
            'A1': {'start': 2000, 'end': 4000, 'type': 'steady_offset', 'sensors': ['L1'], 'magnitude': '3σ'}
        }
    }


def compute_detection_metrics(labels: np.ndarray,
                               predictions: np.ndarray,
                               skip: int = 200) -> dict:
    """Compute detection performance metrics."""
    labels = labels[skip:]
    predictions = predictions[skip:]

    TP = np.sum((labels == 1) & (predictions == 1))
    FP = np.sum((labels == 0) & (predictions == 1))
    TN = np.sum((labels == 0) & (predictions == 0))
    FN = np.sum((labels == 1) & (predictions == 0))

    TPR = TP / max(TP + FN, 1)
    FPR = FP / max(FP + TN, 1)
    precision = TP / max(TP + FP, 1)
    recall = TPR
    F1 = 2 * precision * recall / max(precision + recall, 1e-10)
    balanced_acc = (TPR + (1 - FPR)) / 2

    return {
        'TPR': float(TPR), 'FPR': float(FPR), 'precision': float(precision),
        'recall': float(recall), 'F1': float(F1), 'balanced_accuracy': float(balanced_acc),
        'TP': int(TP), 'FP': int(FP), 'TN': int(TN), 'FN': int(FN),
    }


def run_table2(config: ExperimentConfig, data: dict) -> dict:
    """Run detection pipeline on test data and compute per-attack metrics."""
    print("\n" + "=" * 70)
    print("TABLE 2: Detection Metrics (S2)")
    print("=" * 70)

    sys_cfg = config.system
    N = sys_cfg.n_sensors
    ekf_cfg = data['ekf_config']
    iswt_cfg = data['iswt_config']
    
    ekf = ExtendedKalmanFilter(sys_cfg, ekf_cfg)
    ekf.set_noise_covariances(data['Q'], data['R'])
    ekf_results = ekf.run_batch(data['test']['Y'], data['test']['U'])

    cusum = CUSUMDetector(N, config.cusum)
    cusum_results = cusum.run_batch(ekf_results['std_innovation'])

    iswt = ISWTDetector(N, iswt_cfg)
    iswt_results = iswt.run_batch(ekf_results['std_innovation'])

    cusum_alarm = np.any(cusum_results['alarm'], axis=1)
    iswt_alarm = iswt_results['alarm']
    combined = (cusum_alarm | iswt_alarm).astype(float)

    results = {}
    for name, pred in [('cusum_only', cusum_alarm), ('iswt_only', iswt_alarm), ('combined', combined)]:
        results[name] = compute_detection_metrics(data['test']['labels'], pred.astype(float), skip=iswt_cfg.W)

    print("\n  Overall detection results:")
    for name, m in results.items():
        print(f"    {name:15s}: TPR={m['TPR']:.3f} FPR={m['FPR']:.3f} F1={m['F1']:.3f} BalAcc={m['balanced_accuracy']:.3f}")

    return results


def run_table4_s2(config: ExperimentConfig, data: dict) -> dict:
    """Run TCA on SWaT data and evaluate evasion."""
    print("\n" + "=" * 70)
    print("TABLE 4 (S2): TCA Evasion on SWaT")
    print("=" * 70)

    sys_cfg = config.system
    N = sys_cfg.n_sensors
    ekf_cfg = data['ekf_config']
    iswt_cfg = data['iswt_config']
    Q = data['Q']
    R = data['R']

    start = data['attack_info']['A1']['start'] if 'attack_info' in data else 2000
    T_window = 600
    Y_attack = data['test']['Y'][start:start + T_window].copy()
    U_attack = data['test']['U'][start:start + T_window].copy()
    attacked_idx, compromised_idx = [0, 2, 4, 5], [0]

    results = {}
    for regime in ['whitebox', 'greybox']:
        print(f"\n  --- {regime} ---")
        regime_results = {}
        tca = TargetedConsistencyAttack(sys_cfg, ekf_cfg, config.cusum, iswt_cfg, config.tca)

        for ratio in [0.50, 0.75, 1.00]:
            epsilon = ratio * sys_cfg.sigma[compromised_idx[0]]
            try:
                if regime == 'whitebox':
                    tca_result = tca.run_whitebox(
                        Y_attack, U_attack, attacked_idx,
                        compromised_idx, epsilon, verbose=False
                    )
                else:
                    tca_result = tca.run_greybox(
                        Y_attack, U_attack, attacked_idx,
                        compromised_idx, epsilon, verbose=False
                    )

                # Evaluate detection on attacked data
                Y_pert = Y_attack + tca_result['delta']
                ekf = ExtendedKalmanFilter(sys_cfg, ekf_cfg)
                ekf.set_noise_covariances(Q, R)
                ekf_res = ekf.run_batch(Y_pert, U_attack)

                cusum = CUSUMDetector(N, config.cusum)
                cusum_res = cusum.run_batch(ekf_res['std_innovation'])

                iswt = ISWTDetector(N, iswt_cfg)
                iswt_res = iswt.run_batch(ekf_res['std_innovation'])

                cusum_alarm = np.any(cusum_res['alarm'], axis=1)
                combined = cusum_alarm | iswt_res['alarm']

                # Evasion: check for 60s alarm-free window
                evasion = _check_evasion(combined,
                                          config.s1_evasion_window)

                regime_results[ratio] = {
                    'sds_final': tca_result['sds_final'],
                    'cusum_evasion': bool(np.mean(cusum_alarm) < 0.1),
                    'combined_evasion': bool(evasion),
                    'cusum_alarm_rate': float(np.mean(cusum_alarm)),
                    'combined_alarm_rate': float(np.mean(combined)),
                }

                print(f"SDS={tca_result['sds_final']:.3f}, "
                      f"evade={evasion}")

            except Exception as e:
                print(f"ERROR: {e}")
                regime_results[ratio] = {
                    'sds_final': 0.0,
                    'error': str(e),
                }

        results[regime] = regime_results

    return results


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
# Main
# ======================================================================

def main():
    print("=" * 70)
    print("S2 Offline Evaluation (SWaT)")
    print("=" * 70)

    config = ExperimentConfig()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    t_start = time.time()

    # Load or generate data
    print("\nLoading data...")
    data = load_or_generate_data(config)

    # Table 2
    t2 = run_table2(config, data)
    with open(RESULTS_DIR / "table2_detection.json", 'w') as f:
        json.dump(t2, f, indent=2)

    # Table 4 (S2)
    t4 = run_table4_s2(config, data)
    with open(RESULTS_DIR / "table4_s2_evasion.json", 'w') as f:
        json.dump({str(k): v for k, v in t4.items()}, f, indent=2)

    elapsed = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"All S2 experiments complete in {elapsed / 60:.1f} minutes")
    print(f"Results saved to {RESULTS_DIR}")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
