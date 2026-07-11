"""
BATADAL Dataset Adapter.

Provides loading and preprocessing of the BATADAL (BATtle of the Attack
Detection ALgorithms) dataset for offline evaluation (S2 protocol).

The BATADAL dataset is freely available from https://batadal.net and
published in Taormina et al. (2018), ASCE JWRPM.

The C-Town water distribution network has 7 tanks, 11 pumps, and 1 valve.
We select sensors from two hydraulically connected tanks (T1 and T3) that
form a subsystem analogous to our two-tank testbed model:

    L_T1    -> Tank 1 water level     (analogous to L1)
    L_T3    -> Tank 3 water level     (analogous to L2)
    P_J280  -> Junction 280 pressure  (analogous to P_in)
    P_J300  -> Junction 300 pressure  (analogous to P_out)
    F_PU1   -> Pump 1 flow            (analogous to Q12)
    F_PU2   -> Pump 2 flow            (analogous to Q_pump)

Actuator inputs:
    S_PU1   -> Pump 1 status (on/off)
    S_PU2   -> Pump 2 status (on/off)

These sensors are hydraulically coupled: PU1 and PU2 feed T1, which
drains to T3 via gravity. The mass-balance dynamics are directly
compatible with the two-tank EKF process model.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import SystemConfig

# BATADAL sensor mapping to two-tank state vector
BATADAL_SENSOR_COLS = ['L_T1', 'L_T3', 'P_J280', 'P_J300', 'F_PU1', 'F_PU2']
BATADAL_ACTUATOR_COLS = ['S_PU1', 'S_PU2']

# Sensor full-scale ranges (from BATADAL documentation and data inspection)
BATADAL_RANGES = {
    'L_T1':   (0.0, 7.0),      # meters
    'L_T3':   (0.0, 7.0),      # meters
    'P_J280': (0.0, 5.0),      # meters of head
    'P_J300': (0.0, 40.0),     # meters of head
    'F_PU1':  (0.0, 120.0),    # L/s
    'F_PU2':  (0.0, 120.0),    # L/s
}


def load_batadal_dataset(data_dir: str,
                         mode: str = 'normal',
                         max_rows: Optional[int] = None) -> dict:
    """Load and preprocess the BATADAL dataset.

    Args:
        data_dir: Path to directory containing BATADAL CSV files.
        mode: 'normal' for training data (dataset03), 'attack' for
              attack data (dataset04).
        max_rows: Maximum number of rows to load.

    Returns:
        Dictionary with:
            - 'Y': (T, 6) normalized sensor measurements
            - 'U': (T, 2) actuator states
            - 'labels': (T,) attack labels (0=normal, 1=attack)
            - 'timestamps': timestamps
            - 'sensor_names': list of sensor names
    """
    data_path = Path(data_dir)

    if mode == 'normal':
        candidates = ['BATADAL_dataset03.csv', 'training_dataset_1.csv']
    else:
        candidates = ['BATADAL_dataset04.csv', 'training_dataset_2.csv']

    df = None
    for candidate in candidates:
        fpath = data_path / candidate
        if fpath.exists():
            df = pd.read_csv(fpath, nrows=max_rows)
            break

    if df is None:
        raise FileNotFoundError(
            f"BATADAL dataset not found in {data_dir}. "
            f"Expected one of: {candidates}. "
            f"Download from https://batadal.net/data.html"
        )

    # Clean column names (strip whitespace)
    df.columns = df.columns.str.strip()

    # Verify sensor columns exist
    available_sensors = [c for c in BATADAL_SENSOR_COLS if c in df.columns]
    if len(available_sensors) < 6:
        raise ValueError(
            f"Missing sensors: expected {BATADAL_SENSOR_COLS}, "
            f"found {available_sensors}"
        )

    Y_raw = df[available_sensors].values.astype(float)

    # Handle NaN by forward-filling then backward-filling
    for i in range(Y_raw.shape[1]):
        mask = np.isnan(Y_raw[:, i])
        if np.any(mask):
            for j in range(1, len(Y_raw)):
                if np.isnan(Y_raw[j, i]):
                    Y_raw[j, i] = Y_raw[j - 1, i]
            for j in range(len(Y_raw) - 2, -1, -1):
                if np.isnan(Y_raw[j, i]):
                    Y_raw[j, i] = Y_raw[j + 1, i]

    # Normalize to unit range
    Y_norm = np.zeros_like(Y_raw)
    for i, col in enumerate(available_sensors):
        lo, hi = BATADAL_RANGES[col]
        Y_norm[:, i] = (Y_raw[:, i] - lo) / (hi - lo + 1e-12)
        Y_norm[:, i] = np.clip(Y_norm[:, i], 0.0, 1.0)

    # Map to our two-tank state-space scale
    sys_cfg = SystemConfig()
    Y_scaled = np.zeros_like(Y_norm)
    for i in range(6):
        lo, hi = sys_cfg.sensor_ranges[i]
        Y_scaled[:, i] = lo + Y_norm[:, i] * (hi - lo)

    # Extract actuator columns
    U = np.ones((len(df), 2))
    for j, col in enumerate(BATADAL_ACTUATOR_COLS):
        if col in df.columns:
            U[:, j] = df[col].values.astype(float)

    # Extract labels
    labels = np.zeros(len(df))
    if 'ATT_FLAG' in df.columns:
        att_vals = df['ATT_FLAG'].values
        # ATT_FLAG: 1 = attack, 0 or -999 = normal/unknown
        labels = np.where(att_vals == 1, 1.0, 0.0)

    # Timestamps
    timestamps = None
    if 'DATETIME' in df.columns:
        timestamps = df['DATETIME'].values

    return {
        'Y': Y_scaled,
        'U': U,
        'labels': labels,
        'timestamps': timestamps,
        'sensor_names': available_sensors,
    }
