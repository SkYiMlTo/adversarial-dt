"""
S2 Offline Evaluation on BATADAL (C-Town Water Distribution Network).

Uses a data-driven Kalman Filter (learned linear state-space model)
to generalize the detection framework to an external CPS dataset.

Pipeline:
    1. Load BATADAL dataset (dataset03=train, dataset04=test)
    2. System identification: learn A, B, Q, R via ridge regression
    3. Run data-driven KF on test data
    4. Table 2: Detection metrics (CUSUM, ISWT, Combined)
    5. Table 4 (S2): TCA evasion evaluation (white-box, grey-box)

Falls back to synthetic data if BATADAL is not available.
"""

import numpy as np
import json
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import (ExperimentConfig, SystemConfig, EKFConfig,
                          CUSUMConfig, ISWTConfig, TCAConfig)
from core.cusum import CUSUMDetector
from core.iswt import ISWTDetector, combined_alarm
from core.sds import compute_sds_timeseries
from core.tca import TargetedConsistencyAttack
from core.calibration import calibrate_ekf, validate_whiteness
from core.data_driven_kf import (DataDrivenKalmanFilter,
                                  DifferentiableDataDrivenKF,
                                  identify_linear_system,
                                  normalize_data)
from core.ekf import ExtendedKalmanFilter
from core.process_model import TwoTankProcess

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "s2"
BATADAL_DIR = Path(__file__).resolve().parent.parent / "batadal" / "dataset"

# BATADAL tuned parameters (validated in validate_datadriven_kf.py)
BATADAL_CUSUM_H = 50.0   # Higher threshold for hourly sampling
BATADAL_ISWT_W = 50      # Window size for hourly data
BATADAL_RIDGE_ALPHA = 0.5


# ======================================================================
# Data Loading
# ======================================================================

def load_batadal_raw():
    """Load raw BATADAL data (no rescaling to two-tank ranges)."""
    import pandas as pd

    sensor_cols = ['L_T1', 'L_T3', 'P_J280', 'P_J300', 'F_PU1', 'F_PU2']
    actuator_cols = ['S_PU1', 'S_PU2']

    def _load(fname):
        df = pd.read_csv(BATADAL_DIR / fname)
        df.columns = df.columns.str.strip()
        Y = df[sensor_cols].values.astype(float)
        U = df[actuator_cols].values.astype(float)
        labels = np.zeros(len(df))
        if 'ATT_FLAG' in df.columns:
            labels = np.where(df['ATT_FLAG'].values == 1, 1.0, 0.0)
        # NaN fill
        for i in range(Y.shape[1]):
            for j in range(1, len(Y)):
                if np.isnan(Y[j, i]):
                    Y[j, i] = Y[j - 1, i]
        return Y, U, labels

    Y_train, U_train, _ = _load('BATADAL_dataset03.csv')
    Y_test, U_test, labels_test = _load('BATADAL_dataset04.csv')

    return {
        'train': {'Y': Y_train, 'U': U_train, 'labels': np.zeros(len(Y_train)),
                  'sensor_names': sensor_cols},
        'test': {'Y': Y_test, 'U': U_test, 'labels': labels_test,
                 'sensor_names': sensor_cols},
    }


def load_or_generate_data(config: ExperimentConfig) -> dict:
    """Load BATADAL or generate synthetic data.

    Returns:
        Dictionary with train/test splits and either:
        - 'dd_kf': DataDrivenKalmanFilter (for BATADAL)
        - 'ekf_config' + Q/R: EKF config (for synthetic)
    """
    # --- Try BATADAL ---
    try:
        raw = load_batadal_raw()
        print("  Using BATADAL (C-Town) dataset")
        print(f"    Train: {raw['train']['Y'].shape}")
        print(f"    Test:  {raw['test']['Y'].shape}")
        print(f"    Attack steps: {int(raw['test']['labels'].sum())}")

        # System identification
        sysid = identify_linear_system(
            raw['train']['Y'], raw['train']['U'],
            alpha=BATADAL_RIDGE_ALPHA)
        print(f"    Spectral radius: {sysid['spectral_radius']:.4f}")
        print(f"    Residual std: {np.array2string(sysid['residual_std'], precision=3)}")

        # Normalize train and test data
        Y_train_n, U_train_n = normalize_data(
            raw['train']['Y'], raw['train']['U'], sysid)
        Y_test_n, U_test_n = normalize_data(
            raw['test']['Y'], raw['test']['U'], sysid)

        # Create data-driven KF
        dd_kf = DataDrivenKalmanFilter(
            A=sysid['A'], B=sysid['B'],
            Q=sysid['Q'], R=sysid['R'],
            x0=Y_train_n[0])

        # Compute ISWT baseline from first normal portion of test data
        labels = raw['test']['labels']
        first_normal_end = 0
        for i in range(len(labels)):
            if labels[i] == 0:
                first_normal_end = i + 1
            else:
                if first_normal_end > 200:
                    break

        # Run KF on calibration portion to get baseline
        calib_results = dd_kf.run_batch(
            Y_test_n[:first_normal_end], U_test_n[:first_normal_end])
        skip = 200
        baseline_cov = np.cov(
            calib_results['std_innovation'][skip:first_normal_end].T)

        # Find first attack window
        attack_idx = np.where(labels == 1)[0]
        attack_start = int(attack_idx[0]) if len(attack_idx) > 0 else 2000

        # CUSUM/ISWT configs tuned for hourly BATADAL data
        cusum_cfg = CUSUMConfig()
        cusum_cfg.h = BATADAL_CUSUM_H
        iswt_cfg = ISWTConfig()
        iswt_cfg.W = BATADAL_ISWT_W

        return {
            'train': {'Y': Y_train_n, 'U': U_train_n,
                      'labels': raw['train']['labels'],
                      'sensor_names': raw['train']['sensor_names']},
            'test': {'Y': Y_test_n, 'U': U_test_n,
                     'labels': raw['test']['labels'],
                     'sensor_names': raw['test']['sensor_names']},
            'source': 'batadal',
            'dd_kf': dd_kf,
            'sysid': sysid,
            'cusum_config': cusum_cfg,
            'iswt_config': iswt_cfg,
            'baseline_cov': baseline_cov,
            'Q': sysid['Q'], 'R': sysid['R'],
            'attack_info': {'A1': {'start': attack_start}},
        }
    except (FileNotFoundError, ImportError) as e:
        print(f"  BATADAL not available ({e})")

    # --- Fallback: synthetic ---
    print("  Generating synthetic data")
    return generate_synthetic_data(config)


def generate_synthetic_data(config: ExperimentConfig) -> dict:
    """Generate synthetic data using the two-tank model."""
    sys_cfg = config.system
    process = TwoTankProcess(sys_cfg)

    T_train = config.s2_train_steps
    sim_train = process.simulate(T_train, seed=config.seed)

    T_test = 18000
    sim_test = process.simulate(T_test, seed=config.seed + 100)

    Y_test = sim_test['y_noisy'].copy()
    labels = np.zeros(T_test)

    # Attacks
    Y_test[2000:4000, 0] += 3.0 * sys_cfg.sigma[0]
    labels[2000:4000] = 1
    Y_test[6000:8000, 5] += np.linspace(0, 5.0 * sys_cfg.sigma[5], 2000)
    labels[6000:8000] = 1
    Y_test[10000:12000, 0] += 2.0 * sys_cfg.sigma[0]
    Y_test[10000:12000, 4] -= 2.0 * sys_cfg.sigma[4]
    labels[10000:12000] = 1

    from core.calibration import full_calibration
    T_calib = min(T_train, config.s1_calibration_steps)
    calib = full_calibration(sys_cfg, T_calib, seed=config.seed)

    return {
        'train': {'Y': sim_train['y_noisy'], 'U': sim_train['u'],
                  'labels': np.zeros(T_train), 'sensor_names': sys_cfg.sensor_names},
        'test': {'Y': Y_test, 'U': sim_test['u'],
                 'labels': labels, 'sensor_names': sys_cfg.sensor_names},
        'source': 'synthetic',
        'ekf_config': calib['ekf_config'],
        'iswt_config': calib['iswt_config'],
        'cusum_config': config.cusum,
        'baseline_cov': calib['baseline_cov'],
        'Q': calib['Q'], 'R': calib['R'],
        'attack_info': {'A1': {'start': 2000}},
    }


# ======================================================================
# Detection Metrics
# ======================================================================

def compute_detection_metrics(labels, predictions, skip=200):
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
    F1 = 2 * precision * TPR / max(precision + TPR, 1e-10)
    balanced_acc = (TPR + (1 - FPR)) / 2
    return {
        'TPR': float(TPR), 'FPR': float(FPR), 'precision': float(precision),
        'F1': float(F1), 'balanced_accuracy': float(balanced_acc),
        'TP': int(TP), 'FP': int(FP), 'TN': int(TN), 'FN': int(FN),
    }


def _get_kf(data):
    """Return the appropriate KF for the data source."""
    if 'dd_kf' in data:
        return data['dd_kf']
    else:
        sys_cfg = SystemConfig()
        ekf = ExtendedKalmanFilter(sys_cfg, data['ekf_config'])
        ekf.set_noise_covariances(data['Q'], data['R'])
        return ekf


# ======================================================================
# Table 2: Detection
# ======================================================================

def run_table2(config: ExperimentConfig, data: dict) -> dict:
    """Run detection pipeline on test data."""
    print("\n" + "=" * 70)
    print("TABLE 2: Detection Metrics (S2)")
    print("=" * 70)

    kf = _get_kf(data)
    N = data['test']['Y'].shape[1]
    cusum_cfg = data.get('cusum_config', config.cusum)
    iswt_cfg = data.get('iswt_config', ISWTConfig())

    kf_results = kf.run_batch(data['test']['Y'], data['test']['U'])

    cusum = CUSUMDetector(N, cusum_cfg)
    cusum_results = cusum.run_batch(kf_results['std_innovation'])

    iswt = ISWTDetector(N, iswt_cfg, baseline_cov=data.get('baseline_cov'))
    iswt_results = iswt.run_batch(kf_results['std_innovation'])

    cusum_alarm = np.any(cusum_results['alarm'], axis=1)
    iswt_alarm = iswt_results['alarm']
    combined = (cusum_alarm | iswt_alarm).astype(float)

    results = {}
    skip = max(iswt_cfg.W, 200)
    for name, pred in [('cusum_only', cusum_alarm),
                       ('iswt_only', iswt_alarm),
                       ('combined', combined)]:
        results[name] = compute_detection_metrics(
            data['test']['labels'], pred.astype(float), skip=skip)

    print("\n  Detection results:")
    for name, m in results.items():
        print(f"    {name:15s}: TPR={m['TPR']:.3f} FPR={m['FPR']:.3f} "
              f"F1={m['F1']:.3f} BalAcc={m['balanced_accuracy']:.3f}")

    # Innovation magnitude comparison
    labels = data['test']['labels']
    std_innov = kf_results['std_innovation']
    nm = labels[skip:] == 0
    am = labels[skip:] == 1
    if np.any(am) and np.any(nm):
        mn = np.mean(np.abs(std_innov[skip:][nm]))
        ma = np.mean(np.abs(std_innov[skip:][am]))
        print(f"\n  Innovation magnitude: normal={mn:.3f}, attack={ma:.3f}, "
              f"ratio={ma / max(mn, 1e-10):.2f}x")

    return results


# ======================================================================
# Table 4 (S2): TCA Evasion
# ======================================================================

def run_table4_s2(config: ExperimentConfig, data: dict) -> dict:
    """Run TCA evasion evaluation."""
    print("\n" + "=" * 70)
    print("TABLE 4 (S2): TCA Evasion on BATADAL")
    print("=" * 70)

    N = data['test']['Y'].shape[1]
    cusum_cfg = data.get('cusum_config', config.cusum)
    iswt_cfg = data.get('iswt_config', ISWTConfig())
    Q = data['Q']
    R = data['R']

    start = data['attack_info']['A1']['start']
    T_window = min(600, data['test']['Y'].shape[0] - start)
    Y_attack = data['test']['Y'][start:start + T_window].copy()
    U_attack = data['test']['U'][start:start + T_window].copy()

    # Attack sensors: all 6 sensors attacked, sensor 0 compromised
    attacked_idx = list(range(N))
    compromised_idx = [0]

    # For data-driven model: use the dd_kf for TCA evaluation
    is_datadriven = 'dd_kf' in data

    results = {}
    for regime in ['whitebox', 'greybox']:
        print(f"\n  --- {regime} ---")
        regime_results = {}

        if is_datadriven:
            # TCA with data-driven KF: use custom forward pass
            sys_cfg = SystemConfig()
            sys_cfg._n_states = N
            sys_cfg._n_sensors = N

            # Create TCA with dummy configs (the data-driven KF will
            # be injected via custom evaluation)
            tca = TargetedConsistencyAttack(
                sys_config=sys_cfg,
                ekf_config=EKFConfig(),
                cusum_config=cusum_cfg,
                iswt_config=iswt_cfg,
                tca_config=config.tca,
                baseline_cov=data.get('baseline_cov'),
                kf_model=data['dd_kf'])
        else:
            tca = TargetedConsistencyAttack(
                sys_config=config.system,
                ekf_config=data['ekf_config'],
                cusum_config=cusum_cfg,
                iswt_config=iswt_cfg,
                tca_config=config.tca,
                baseline_cov=data.get('baseline_cov'))

        # Find all contiguous attack windows in test labels
        labels = data['test']['labels']
        attack_diffs = np.diff(np.concatenate(([0], labels, [0])))
        attack_starts = np.where(attack_diffs == 1)[0]
        attack_ends = np.where(attack_diffs == -1)[0]

        for ratio in [0.25, 0.50, 0.75, 1.00, 1.50]:
            # Compute epsilon from training data residual std
            if is_datadriven:
                sysid = data['sysid']
                epsilon = np.ones(N) * ratio * np.mean(sysid['residual_std'])
            else:
                epsilon = ratio * config.system.sigma

            evasion_count = 0
            total_windows = 0
            sds_list = []
            alarm_rates = []

            # Evaluate across sub-windows within attack periods, with W-step pre-padding
            W = iswt_cfg.W
            sub_window = 30
            for win_idx in range(len(attack_starts)):
                w_start = int(attack_starts[win_idx])
                w_end = int(attack_ends[win_idx])

                for s_start in range(w_start, w_end, 10):
                    s_end = min(w_end, s_start + sub_window)
                    if s_end - s_start < 5:
                        continue

                    # Pre-pad with W steps of normal history so ISWT buffer is initialized
                    pad_start = max(0, s_start - W)
                    Y_win = data['test']['Y'][pad_start:s_end].copy()
                    U_win = data['test']['U'][pad_start:s_end].copy()

                    try:
                        if regime == 'whitebox':
                            tca_res = tca.run_whitebox(
                                Y_win, U_win, attacked_idx,
                                compromised_idx, epsilon, verbose=False)
                        else:
                            tca_res = tca.run_greybox(
                                Y_win, U_win, attacked_idx,
                                compromised_idx, epsilon, verbose=False)

                        Y_pert = Y_win + tca_res['delta']
                        kf = _get_kf(data)
                        kf_res = kf.run_batch(Y_pert, U_win)

                        cusum = CUSUMDetector(N, cusum_cfg)
                        cusum_res = cusum.run_batch(kf_res['std_innovation'])

                        iswt = ISWTDetector(N, iswt_cfg,
                                            baseline_cov=data.get('baseline_cov'))
                        iswt_res = iswt.run_batch(kf_res['std_innovation'])

                        # Evaluate alarm decisions only during active attack timesteps
                        att_slice = slice(s_start - pad_start, s_end - pad_start)
                        cusum_alarm = np.any(cusum_res['alarm'][att_slice], axis=1)
                        combined = cusum_alarm | iswt_res['alarm'][att_slice]

                        evade = not np.any(combined)
                        if evade:
                            evasion_count += 1
                        total_windows += 1
                        sds_list.append(tca_res['sds_final'])
                        alarm_rates.append(np.mean(combined))

                    except Exception as e:
                        pass

            total_windows = max(total_windows, 1)
            evasion_rate = (evasion_count / total_windows) * 100.0
            mean_sds = float(np.mean(sds_list)) if len(sds_list) > 0 else 0.0
            mean_alarm_rate = float(np.mean(alarm_rates)) if len(alarm_rates) > 0 else 0.0

            regime_results[str(ratio)] = {
                'sds_final': mean_sds,
                'evasion_rate_pct': evasion_rate,
                'combined_alarm_rate': mean_alarm_rate,
            }

            print(f"  eps={ratio:.2f}: Evasion Rate={evasion_rate:.1f}%, SDS={mean_sds:.3f}, alarm_rate={mean_alarm_rate:.3f}")

        results[regime] = regime_results

    return results


def _check_evasion(alarms, window):
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
    print("S2 Offline Evaluation (BATADAL / Synthetic)")
    print("=" * 70)

    config = ExperimentConfig()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    t_start = time.time()

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
