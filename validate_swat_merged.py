"""
Quick SWaT validation using merged.csv properly.

merged.csv: 1,441,719 rows = 1,387,098 Normal + 54,621 Attack
Train: rows 0-100,000 (clean normal data)
Test:  rows 1,337,098 - 1,441,719 (last 50k normal + all 54k attack)
"""
import sys
import numpy as np
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core.config import CUSUMConfig, ISWTConfig
from core.cusum import CUSUMDetector
from core.iswt import ISWTDetector

SWAT_DIR = Path(__file__).resolve().parent / "swat" / "dataset"
SENSOR_COLS = ['LIT101', 'LIT301', 'AIT201', 'AIT501', 'FIT101', 'FIT301']
ACTUATOR_COLS = ['MV101', 'P101']


def load_chunk(fpath, skip=0, nrows=None):
    if skip > 0:
        df = pd.read_csv(fpath, skiprows=range(1, skip+1), nrows=nrows)
    else:
        df = pd.read_csv(fpath, nrows=nrows)
    df.columns = df.columns.str.strip()
    Y = df[SENSOR_COLS].values.astype(float)
    U = np.ones((len(df), 2))
    for j, col in enumerate(ACTUATOR_COLS):
        if col in df.columns:
            U[:, j] = df[col].values.astype(float)
    labels = np.zeros(len(df))
    if 'Normal/Attack' in df.columns:
        lv = df['Normal/Attack'].astype(str).str.strip().str.lower()
        labels = np.where(lv.isin(['attack', '1', 'a']), 1.0, 0.0)
    # NaN fill
    for i in range(Y.shape[1]):
        for j in range(1, len(Y)):
            if np.isnan(Y[j, i]):
                Y[j, i] = Y[j-1, i]
        for j in range(len(Y)-2, -1, -1):
            if np.isnan(Y[j, i]):
                Y[j, i] = Y[j+1, i]
    return Y, U, labels


def fit_ridge(Y, U, alpha=0.01):
    N, M = Y.shape[1], U.shape[1]
    X = np.hstack([Y[:-1], U[:-1]])
    Yt = Y[1:]
    XtX = X.T @ X
    coeffs = np.linalg.solve(XtX + alpha * np.eye(XtX.shape[0]), X.T @ Yt)
    A, B = coeffs[:N].T, coeffs[N:].T
    res = Yt - X @ coeffs
    Q = np.cov(res.T)
    return A, B, Q, np.std(res, axis=0)


def run_kf(Y, U, A, B, Q, R):
    T, N = Y.shape
    x = Y[0].copy()
    P = np.eye(N) * 0.1
    std_innov = np.zeros((T, N))
    for t in range(1, T):
        xp = A @ x + B @ U[t-1]
        Pp = A @ P @ A.T + Q
        nu = Y[t] - xp
        S = Pp + R
        Sd = np.diag(S)
        K = Pp @ np.linalg.inv(S)
        x = xp + K @ nu
        IK = np.eye(N) - K
        P = IK @ Pp @ IK.T + K @ R @ K.T
        std_innov[t] = nu / np.sqrt(np.maximum(Sd, 1e-12))
    return std_innov


def metrics(labels, preds, skip=200):
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


if __name__ == '__main__':
    merged = SWAT_DIR / 'merged.csv'
    print("Loading SWaT merged.csv...")

    # Train: first 100k rows
    print("  Loading training data (first 100k rows)...")
    Y_train, U_train, _ = load_chunk(merged, skip=0, nrows=100000)
    print(f"  Train shape: {Y_train.shape}")

    # Test: last ~105k rows (50k normal + 54,621 attack)
    # Total rows = 1,441,719. Skip header + first 1,337,098 data rows.
    skip_test = 1337098
    print(f"  Loading test data (skip {skip_test}, rest of file)...")
    Y_test, U_test, labels = load_chunk(merged, skip=skip_test)
    n_att = int(labels.sum())
    n_norm = len(labels) - n_att
    print(f"  Test shape: {Y_test.shape}")
    print(f"  Labels: {n_norm} normal, {n_att} attack ({100*n_att/len(labels):.1f}%)")

    # Z-score normalize
    ym, ys = Y_train.mean(0), np.maximum(Y_train.std(0), 1e-10)
    um, us = U_train.mean(0), np.maximum(U_train.std(0), 1e-10)
    Ytr = (Y_train - ym) / ys
    Yte = (Y_test - ym) / ys
    Utr = (U_train - um) / us
    Ute = (U_test - um) / us

    # Try multiple alphas
    print("\n  Fitting VAR(1) models...")
    for alpha in [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]:
        A, B, Q, res_std = fit_ridge(Ytr, Utr, alpha=alpha)
        sr = np.max(np.abs(np.linalg.eigvals(A)))
        print(f"    alpha={alpha:6.3f}: sr={sr:.6f} res_std_mean={res_std.mean():.4f}")

    # Use best stable alpha
    print("\n  Running detection with multiple alphas...")
    for alpha in [0.5, 1.0, 2.0, 5.0]:
        A, B, Q, res_std = fit_ridge(Ytr, Utr, alpha=alpha)
        sr = np.max(np.abs(np.linalg.eigvals(A)))
        if sr >= 1.0:
            continue
        R = np.diag(np.diag(Q) * 0.1)

        # Run KF on test
        std_te = run_kf(Yte, Ute, A, B, Q, R)

        # Check for NaN
        nan_count = np.sum(np.isnan(std_te))
        if nan_count > 0:
            print(f"    alpha={alpha}: {nan_count} NaN values, skipping")
            continue

        # Calibrate ISWT from first normal portion of test
        first_norm_end = 0
        for i in range(len(labels)):
            if labels[i] == 0:
                first_norm_end = i + 1
            else:
                if first_norm_end > 500:
                    break
        baseline = np.cov(std_te[200:min(first_norm_end, len(std_te))].T)

        N = 6
        print(f"\n    alpha={alpha}, sr={sr:.4f}, baseline from {first_norm_end} normal steps:")
        for h in [3.0, 5.0, 8.0, 12.0, 20.0]:
            for W in [20, 50]:
                cusum_cfg = CUSUMConfig()
                cusum_cfg.h = h
                cusum = CUSUMDetector(N, cusum_cfg)
                cr = cusum.run_batch(std_te)
                ca = np.any(cr['alarm'], axis=1)

                iswt_cfg = ISWTConfig()
                iswt_cfg.W = W
                iswt = ISWTDetector(N, iswt_cfg, baseline_cov=baseline)
                ir = iswt.run_batch(std_te)
                ia = ir['alarm']
                comb = (ca | ia).astype(float)

                mc = metrics(labels, ca.astype(float), skip=max(W, 200))
                mi = metrics(labels, ia.astype(float), skip=max(W, 200))
                mk = metrics(labels, comb, skip=max(W, 200))

                if mc['BalAcc'] > 0.55 or mk['BalAcc'] > 0.55:
                    print(f"      h={h:5.1f} W={W:3d} | "
                          f"CUSUM: TPR={mc['TPR']:.3f} FPR={mc['FPR']:.3f} BA={mc['BalAcc']:.3f} | "
                          f"ISWT: TPR={mi['TPR']:.3f} FPR={mi['FPR']:.3f} BA={mi['BalAcc']:.3f} | "
                          f"Comb: TPR={mk['TPR']:.3f} FPR={mk['FPR']:.3f} BA={mk['BalAcc']:.3f} F1={mk['F1']:.3f}")

        # Innovation comparison
        nm = labels[200:] == 0
        am = labels[200:] == 1
        if np.any(am) and np.any(nm):
            mn = np.mean(np.abs(std_te[200:][nm]))
            ma = np.mean(np.abs(std_te[200:][am]))
            print(f"      Innov magnitude: normal={mn:.3f}, attack={ma:.3f}, ratio={ma/max(mn,1e-10):.2f}x")

    print("\nDONE")
