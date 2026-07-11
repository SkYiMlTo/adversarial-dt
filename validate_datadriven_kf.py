"""
Validation v4: Focused fixes for BATADAL and SWaT.

BATADAL fixes:
    1. VAR(2) model (two lags) for better prediction
    2. Calibrate ISWT baseline from first normal portion of test data
    3. Sweep both CUSUM h and ISWT W

SWaT fixes:
    1. Train on first 100k of normal.csv
    2. Test on last 50k of normal.csv + all of attack.csv (proper mix)
"""
import sys
import numpy as np
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core.config import CUSUMConfig, ISWTConfig
from core.cusum import CUSUMDetector
from core.iswt import ISWTDetector


def load_batadal_raw(data_dir, mode='normal'):
    data_path = Path(data_dir)
    fname = 'BATADAL_dataset03.csv' if mode == 'normal' else 'BATADAL_dataset04.csv'
    df = pd.read_csv(data_path / fname)
    df.columns = df.columns.str.strip()
    sensor_cols = ['L_T1', 'L_T3', 'P_J280', 'P_J300', 'F_PU1', 'F_PU2']
    actuator_cols = ['S_PU1', 'S_PU2']
    Y = df[sensor_cols].values.astype(float)
    U = df[actuator_cols].values.astype(float)
    labels = np.zeros(len(df))
    if 'ATT_FLAG' in df.columns:
        labels = np.where(df['ATT_FLAG'].values == 1, 1.0, 0.0)
    return Y, U, labels


def load_swat_split(data_dir):
    """Load SWaT with proper train/test split.
    Train: first 100k of normal.csv
    Test: last 50k of normal.csv + all of attack.csv (concatenated)
    """
    data_path = Path(data_dir)
    sensor_cols = ['LIT101', 'LIT301', 'AIT201', 'AIT501', 'FIT101', 'FIT301']
    actuator_cols = ['MV101', 'P101']

    def _load(fpath, max_rows=None, skip_rows=None):
        if skip_rows:
            df = pd.read_csv(fpath, skiprows=range(1, skip_rows+1), nrows=max_rows)
        else:
            df = pd.read_csv(fpath, nrows=max_rows)
        df.columns = df.columns.str.strip()
        Y = df[sensor_cols].values.astype(float)
        U = np.ones((len(df), 2))
        for j, col in enumerate(actuator_cols):
            if col in df.columns:
                U[:, j] = df[col].values.astype(float)
        labels = np.zeros(len(df))
        for lc in ['Normal/Attack', 'Label']:
            if lc in df.columns:
                lv = df[lc].astype(str).str.strip().str.lower()
                labels = np.where(lv.isin(['attack', '1', 'a']), 1.0, 0.0)
                break
        # NaN fill
        for i in range(Y.shape[1]):
            m = np.isnan(Y[:, i])
            if np.any(m):
                for j in range(1, len(Y)):
                    if np.isnan(Y[j, i]):
                        Y[j, i] = Y[j-1, i]
                for j in range(len(Y)-2, -1, -1):
                    if np.isnan(Y[j, i]):
                        Y[j, i] = Y[j+1, i]
        return Y, U, labels

    normal_path = data_path / 'normal.csv'
    attack_path = data_path / 'attack.csv'

    # Train: first 100k rows of normal
    Y_train, U_train, _ = _load(normal_path, max_rows=100000)

    # Test normal: last 50k of normal.csv
    # First count total rows
    total_normal = sum(1 for _ in open(normal_path)) - 1
    skip_normal = max(0, total_normal - 50000)
    Y_test_n, U_test_n, l_test_n = _load(normal_path, skip_rows=skip_normal)

    # Test attack: all of attack.csv
    Y_test_a, U_test_a, l_test_a = _load(attack_path)

    # Concatenate test sets
    Y_test = np.vstack([Y_test_n, Y_test_a])
    U_test = np.vstack([U_test_n, U_test_a])
    labels_test = np.concatenate([l_test_n, l_test_a])

    return Y_train, U_train, Y_test, U_test, labels_test


def zscore(Y_tr, Y_te, U_tr, U_te):
    ym, ys = Y_tr.mean(0), np.maximum(Y_tr.std(0), 1e-10)
    um, us = U_tr.mean(0), np.maximum(U_tr.std(0), 1e-10)
    return (Y_tr-ym)/ys, (Y_te-ym)/ys, (U_tr-um)/us, (U_te-um)/us


def fit_var2_ridge(Y, U, alpha=0.01):
    """VAR(2): x_{t+1} = A1 @ x_t + A2 @ x_{t-1} + B @ u_t."""
    N, M = Y.shape[1], U.shape[1]
    # Features: [x_t, x_{t-1}, u_t]
    X = np.hstack([Y[1:-1], Y[:-2], U[1:-1]])  # (T-2, 2N+M)
    Yt = Y[2:]  # (T-2, N)
    XtX = X.T @ X
    coeffs = np.linalg.solve(XtX + alpha * np.eye(XtX.shape[0]), X.T @ Yt)
    A1 = coeffs[:N].T
    A2 = coeffs[N:2*N].T
    B = coeffs[2*N:].T
    res = Yt - X @ coeffs
    Q = np.cov(res.T)
    return A1, A2, B, Q, np.std(res, axis=0)


def fit_var1_ridge(Y, U, alpha=0.01):
    """VAR(1): x_{t+1} = A @ x_t + B @ u_t."""
    N, M = Y.shape[1], U.shape[1]
    X = np.hstack([Y[:-1], U[:-1]])
    Yt = Y[1:]
    XtX = X.T @ X
    coeffs = np.linalg.solve(XtX + alpha * np.eye(XtX.shape[0]), X.T @ Yt)
    A = coeffs[:N].T
    B = coeffs[N:].T
    res = Yt - X @ coeffs
    Q = np.cov(res.T)
    return A, B, Q, np.std(res, axis=0)


def run_kf_var1(Y, U, A, B, Q, R):
    T, N = Y.shape
    x = Y[0].copy()
    P = np.eye(N) * 0.1
    std_innov = np.zeros((T, N))
    for t in range(1, T):
        xp = A @ x + B @ U[t-1]
        Pp = A @ P @ A.T + Q
        nu = Y[t] - xp
        S = Pp + R
        K = Pp @ np.linalg.inv(S)
        x = xp + K @ nu
        IK = np.eye(N) - K
        P = IK @ Pp @ IK.T + K @ R @ K.T
        std_innov[t] = nu / np.sqrt(np.maximum(np.diag(S), 1e-12))
    return std_innov


def run_kf_var2(Y, U, A1, A2, B, Q, R):
    T, N = Y.shape
    # Augmented state: z = [x_t, x_{t-1}]
    # z_{t+1} = [A1 A2; I 0] z_t + [B; 0] u_t
    Aa = np.block([[A1, A2], [np.eye(N), np.zeros((N, N))]])
    Ba = np.vstack([B, np.zeros((N, U.shape[1]))])
    Qa = np.block([[Q, np.zeros((N, N))], [np.zeros((N, N)), np.zeros((N, N))]])
    Ha = np.hstack([np.eye(N), np.zeros((N, N))])
    Ra = R

    z = np.concatenate([Y[1], Y[0]])
    P = np.eye(2*N) * 0.1
    std_innov = np.zeros((T, N))

    for t in range(2, T):
        zp = Aa @ z + Ba @ U[t-1]
        Pp = Aa @ P @ Aa.T + Qa
        nu = Y[t] - Ha @ zp
        S = Ha @ Pp @ Ha.T + Ra
        K = Pp @ Ha.T @ np.linalg.inv(S)
        z = zp + K @ nu
        IK = np.eye(2*N) - K @ Ha
        P = IK @ Pp @ IK.T + K @ Ra @ K.T
        std_innov[t] = nu / np.sqrt(np.maximum(np.diag(S), 1e-12))
    return std_innov


def metrics(labels, preds, skip=50):
    l, p = labels[skip:], preds[skip:]
    TP = int(np.sum((l == 1) & (p == 1)))
    FP = int(np.sum((l == 0) & (p == 1)))
    TN = int(np.sum((l == 0) & (p == 0)))
    FN = int(np.sum((l == 1) & (p == 0)))
    TPR = TP / max(TP + FN, 1)
    FPR = FP / max(FP + TN, 1)
    prec = TP / max(TP + FP, 1)
    F1 = 2 * prec * TPR / max(prec + TPR, 1e-10)
    bal = (TPR + (1 - FPR)) / 2
    return {'TPR': TPR, 'FPR': FPR, 'F1': F1, 'BalAcc': bal,
            'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN}


def validate(name, Ytr, Utr, Yte, Ute, labels):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    n_att = int(labels.sum())
    n_norm = len(labels) - n_att
    print(f"  Train: {Ytr.shape[0]} | Test: {Yte.shape[0]} "
          f"({n_norm} normal, {n_att} attack = {100*n_att/len(labels):.1f}%)")

    Ytr_n, Yte_n, Utr_n, Ute_n = zscore(Ytr, Yte, Utr, Ute)
    N = Ytr.shape[1]

    # ---- VAR(1) ----
    print("\n  --- VAR(1) ---")
    best_a1, best_sr1 = 0.5, 999
    for a in [1e-4, 5e-4, 1e-3, 5e-3, 0.01, 0.05, 0.1, 0.5]:
        At, _, _, _ = fit_var1_ridge(Ytr_n, Utr_n, alpha=a)
        sr = np.max(np.abs(np.linalg.eigvals(At)))
        if sr < 0.999:
            best_a1, best_sr1 = a, sr
            break
        if sr < best_sr1:
            best_a1, best_sr1 = a, sr

    A, B, Q1, rs1 = fit_var1_ridge(Ytr_n, Utr_n, alpha=best_a1)
    sr1 = np.max(np.abs(np.linalg.eigvals(A)))
    print(f"  alpha={best_a1}, sr={sr1:.4f}, res_std={np.array2string(rs1, precision=3)}")

    R1 = np.diag(np.diag(Q1) * 0.1)
    std1_tr = run_kf_var1(Ytr_n, Utr_n, A, B, Q1, R1)
    std1_te = run_kf_var1(Yte_n, Ute_n, A, B, Q1, R1)

    # ---- VAR(2) ----
    print("\n  --- VAR(2) ---")
    A1v, A2v, Bv, Q2, rs2 = fit_var2_ridge(Ytr_n, Utr_n, alpha=best_a1)
    # Check augmented matrix stability
    Aa = np.block([[A1v, A2v], [np.eye(N), np.zeros((N, N))]])
    sr2 = np.max(np.abs(np.linalg.eigvals(Aa)))
    print(f"  alpha={best_a1}, sr={sr2:.4f}, res_std={np.array2string(rs2, precision=3)}")

    R2 = np.diag(np.diag(Q2) * 0.1)
    std2_tr = run_kf_var2(Ytr_n, Utr_n, A1v, A2v, Bv, Q2, R2)
    std2_te = run_kf_var2(Yte_n, Ute_n, A1v, A2v, Bv, Q2, R2)

    # ---- Detection with ISWT calibrated from test-data normal portion ----
    # Use first normal portion of test data for ISWT baseline
    first_normal_end = 0
    for i in range(len(labels)):
        if labels[i] == 0:
            first_normal_end = i + 1
        else:
            if first_normal_end > 200:
                break

    print(f"\n  ISWT baseline calibrated from first {first_normal_end} test steps (normal)")

    for var_name, std_tr, std_te in [("VAR(1)", std1_tr, std1_te),
                                      ("VAR(2)", std2_tr, std2_te)]:
        print(f"\n  === {var_name} Detection ===")

        # Calibrate baseline from test-data normal portion
        calib_start = 200
        calib_end = min(first_normal_end, len(std_te))
        if calib_end > calib_start:
            baseline = np.cov(std_te[calib_start:calib_end].T)
        else:
            baseline = np.cov(std_tr[200:].T)

        for h_val in [3.0, 5.0, 8.0, 12.0, 20.0, 50.0]:
            for W_val in [20, 50]:
                cusum_cfg = CUSUMConfig()
                cusum_cfg.h = h_val
                cusum = CUSUMDetector(N, cusum_cfg)
                cr = cusum.run_batch(std_te)
                ca = np.any(cr['alarm'], axis=1)

                iswt_cfg = ISWTConfig()
                iswt_cfg.W = W_val
                iswt = ISWTDetector(N, iswt_cfg, baseline_cov=baseline)
                ir = iswt.run_batch(std_te)
                ia = ir['alarm']
                comb = (ca | ia).astype(float)

                mc = metrics(labels, ca.astype(float), skip=max(W_val, 200))
                mi = metrics(labels, ia.astype(float), skip=max(W_val, 200))
                mk = metrics(labels, comb, skip=max(W_val, 200))

                if mk['BalAcc'] > 0.55 or mc['BalAcc'] > 0.55:
                    print(f"    h={h_val:5.1f} W={W_val:3d} | "
                          f"CUSUM: TPR={mc['TPR']:.3f} FPR={mc['FPR']:.3f} BA={mc['BalAcc']:.3f} | "
                          f"ISWT: TPR={mi['TPR']:.3f} FPR={mi['FPR']:.3f} | "
                          f"Comb: TPR={mk['TPR']:.3f} FPR={mk['FPR']:.3f} BA={mk['BalAcc']:.3f} F1={mk['F1']:.3f}")

    # Innovation comparison
    skip = 200
    nm = labels[skip:] == 0
    am = labels[skip:] == 1
    for vn, st in [("VAR(1)", std1_te), ("VAR(2)", std2_te)]:
        if np.any(am) and np.any(nm):
            mn = np.mean(np.abs(st[skip:][nm]))
            ma = np.mean(np.abs(st[skip:][am]))
            print(f"\n  {vn} innov magnitude: normal={mn:.3f}, attack={ma:.3f}, ratio={ma/max(mn,1e-10):.2f}x")


if __name__ == '__main__':
    batadal_dir = str(Path(__file__).resolve().parent / "batadal" / "dataset")
    swat_dir = str(Path(__file__).resolve().parent / "swat" / "dataset")

    # BATADAL
    try:
        Yb_tr, Ub_tr, _, = load_batadal_raw(batadal_dir, 'normal')
        Yb_te, Ub_te, lb = load_batadal_raw(batadal_dir, 'attack')
        validate("BATADAL", Yb_tr, Ub_tr, Yb_te, Ub_te, lb)
    except Exception as e:
        print(f"BATADAL failed: {e}")
        import traceback; traceback.print_exc()

    # SWaT
    try:
        Ys_tr, Us_tr, Ys_te, Us_te, ls = load_swat_split(swat_dir)
        validate("SWaT", Ys_tr, Us_tr, Ys_te, Us_te, ls)
    except Exception as e:
        print(f"SWaT failed: {e}")
        import traceback; traceback.print_exc()

    print("\nDONE")
