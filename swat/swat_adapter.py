"""
SWaT Dataset Adapter.

Provides loading and preprocessing of the Secure Water Treatment (SWaT)
dataset for offline evaluation (S2 protocol, Sec. 4.5).

The SWaT dataset must be obtained from iTrust:
    https://itrust.sutd.edu.sg/itrust-labs_datasets/

Expected file: SWaT_Dataset_Normal_v1.xlsx (or .csv export)

The adapter selects a subset of sensors relevant to our two-tank
analogy and normalizes them for EKF ingestion.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import SystemConfig

# SWaT sensor mapping:
# We select sensors from Stage 1 (raw water intake) and Stage 3 (ultrafiltration)
# that form a hydraulically coherent subsystem analogous to our two-tank model.
#
# Selected 6 sensors (|A| = 4 attacked + 2 witnesses):
#   LIT101  — Raw water tank level      → analogous to L1
#   LIT301  — UF feed tank level        → analogous to L2
#   AIT201  — Water quality (pH) proxy  → analogous to P_in
#   AIT501  — RO permeate quality       → analogous to P_out
#   FIT101  — Raw water inlet flow      → analogous to Q12
#   FIT301  — UF feed flow              → analogous to Q_pump
#
# Actuator inputs:
#   MV101   — Raw water intake valve
#   P101    — Raw water pump status

SWAT_SENSOR_COLS = ['LIT101', 'LIT301', 'AIT201', 'AIT501', 'FIT101', 'FIT301']
SWAT_ACTUATOR_COLS = ['MV101', 'P101']

# Sensor full-scale ranges for normalization
SWAT_RANGES = {
    'LIT101': (0.0, 1200.0),    # mm
    'LIT301': (0.0, 1200.0),    # mm
    'AIT201': (0.0, 14.0),      # pH
    'AIT501': (0.0, 500.0),     # μS/cm
    'FIT101': (0.0, 5.0),       # m³/h
    'FIT301': (0.0, 5.0),       # m³/h
}


def load_swat_dataset(data_dir: str,
                      mode: str = 'normal',
                      max_rows: Optional[int] = None) -> dict:
    """Load and preprocess the SWaT dataset.

    Args:
        data_dir: Path to directory containing SWaT CSV files.
        mode: 'normal' for training data, 'attack' for attack data.
        max_rows: Maximum number of rows to load (for debugging).

    Returns:
        Dictionary with:
            - 'Y': (T, 6) normalized sensor measurements
            - 'U': (T, 2) actuator states
            - 'labels': (T,) attack labels (0=normal, 1=attack)
            - 'timestamps': timestamps
            - 'raw_df': raw DataFrame
    """
    data_path = Path(data_dir)

    # Try different file name conventions
    candidates = [
        f'SWaT_Dataset_{mode.capitalize()}_v1.csv',
        f'SWaT_Dataset_{mode.capitalize()}_v0.csv',
        f'swat_{mode}.csv',
        f'{mode}.csv',
    ]

    df = None
    for candidate in candidates:
        fpath = data_path / candidate
        if fpath.exists():
            df = pd.read_csv(fpath, nrows=max_rows)
            break

    # Also try .xlsx
    if df is None:
        for candidate in candidates:
            fpath = data_path / candidate.replace('.csv', '.xlsx')
            if fpath.exists():
                df = pd.read_excel(fpath, nrows=max_rows)
                break

    if df is None:
        raise FileNotFoundError(
            f"SWaT dataset not found in {data_dir}. "
            f"Expected one of: {candidates}. "
            f"Download from https://itrust.sutd.edu.sg/itrust-labs_datasets/"
        )

    # Clean column names (strip whitespace)
    df.columns = df.columns.str.strip()

    # Extract sensor columns
    available_sensors = [c for c in SWAT_SENSOR_COLS if c in df.columns]
    if len(available_sensors) < 6:
        raise ValueError(
            f"Missing sensors: expected {SWAT_SENSOR_COLS}, "
            f"found {available_sensors}"
        )

    Y_raw = df[available_sensors].values.astype(float)

    # Handle NaN by forward-filling
    for i in range(Y_raw.shape[1]):
        mask = np.isnan(Y_raw[:, i])
        if np.any(mask):
            # Forward fill
            for j in range(1, len(Y_raw)):
                if np.isnan(Y_raw[j, i]):
                    Y_raw[j, i] = Y_raw[j - 1, i]
            # Backward fill remaining
            for j in range(len(Y_raw) - 2, -1, -1):
                if np.isnan(Y_raw[j, i]):
                    Y_raw[j, i] = Y_raw[j + 1, i]

    # Normalize to unit range
    Y_norm = np.zeros_like(Y_raw)
    for i, col in enumerate(available_sensors):
        lo, hi = SWAT_RANGES[col]
        Y_norm[:, i] = (Y_raw[:, i] - lo) / (hi - lo + 1e-12)
        Y_norm[:, i] = np.clip(Y_norm[:, i], 0.0, 1.0)

    # Map to our state-space scale
    sys_cfg = SystemConfig()
    Y_scaled = np.zeros_like(Y_norm)
    for i in range(6):
        lo, hi = sys_cfg.sensor_ranges[i]
        Y_scaled[:, i] = lo + Y_norm[:, i] * (hi - lo)

    # Extract actuator columns
    U = np.ones((len(df), 2))
    if 'P101' in df.columns:
        U[:, 0] = df['P101'].values.astype(float)
    if 'MV101' in df.columns:
        valve_raw = df['MV101'].values.astype(float)
        U[:, 1] = valve_raw / 2.0  # Binary → 0.0/0.5

    # Extract labels
    labels = np.zeros(len(df))
    label_col = None
    for candidate_col in ['Normal/Attack', 'Label', 'label', 'Attack']:
        if candidate_col in df.columns:
            label_col = candidate_col
            break

    if label_col is not None:
        label_vals = df[label_col].astype(str).str.strip().str.lower()
        labels = np.where(label_vals.isin(['attack', '1', 'a']), 1.0, 0.0)

    # Timestamps
    timestamps = None
    for ts_col in ['Timestamp', 'timestamp', 'Time', 'time']:
        if ts_col in df.columns:
            timestamps = pd.to_datetime(df[ts_col].astype(str).str.strip(), format='mixed', dayfirst=True)
            break

    return {
        'Y': Y_scaled,
        'U': U,
        'labels': labels,
        'timestamps': timestamps,
        'raw_df': df,
        'sensor_names': available_sensors,
    }


def generate_synthetic_swat(n_steps: int = 72000,
                            seed: int = 42) -> dict:
    """Generate synthetic SWaT-like data for testing without the real dataset.

    Produces data with similar statistical characteristics to real SWaT:
    - Periodic pump cycling (L1 oscillates between setpoints)
    - Slow drift in quality sensors
    - Gaussian noise at levels matching industrial sensors

    Args:
        n_steps: Number of timesteps (72000 = 20 hours at 1 Hz).
        seed: Random seed.

    Returns:
        Same format as load_swat_dataset.
    """
    from core.process_model import TwoTankProcess

    sys_cfg = SystemConfig()
    process = TwoTankProcess(sys_cfg)
    process.set_seed(seed)

    sim = process.simulate(n_steps, seed=seed)

    # Create attack labels: 10% of data is under attack
    labels = np.zeros(n_steps)
    attack_start = int(0.7 * n_steps)
    attack_end = int(0.8 * n_steps)
    labels[attack_start:attack_end] = 1.0

    # Apply a fault during attack period
    Y = sim['y_noisy'].copy()
    fault_magnitude = 3.0 * sys_cfg.sigma
    Y[attack_start:attack_end] += fault_magnitude

    return {
        'Y': Y,
        'U': sim['u'],
        'labels': labels,
        'timestamps': None,
        'raw_df': None,
        'sensor_names': sys_cfg.sensor_names,
    }
