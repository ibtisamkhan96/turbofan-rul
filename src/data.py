"""Data preparation for NASA C-MAPSS turbofan RUL (FD001).

Loads the run-to-failure sensor logs, builds the Remaining Useful Life (RUL) label, keeps the
informative sensors, scales them, and windows each engine's history into fixed-length sequences
for an LSTM. Public-domain NASA data.
"""

import numpy as np
import pandas as pd

COLS = ["unit", "cycle", "op1", "op2", "op3"] + [f"sensor{i}" for i in range(1, 22)]
# Sensors that actually vary in FD001 (the rest are constant and carry no information).
FEATURES = ["sensor2", "sensor3", "sensor4", "sensor7", "sensor8", "sensor9", "sensor11",
            "sensor12", "sensor13", "sensor14", "sensor15", "sensor17", "sensor20", "sensor21"]
RUL_CAP = 125   # piecewise-linear RUL: degradation is only detectable late, so cap early life
WINDOW = 30


def _read(path):
    return pd.read_csv(path, sep=r"\s+", header=None, names=COLS)


def load(data_dir="data"):
    train = _read(f"{data_dir}/train_FD001.txt")
    test = _read(f"{data_dir}/test_FD001.txt")
    y_test = pd.read_csv(f"{data_dir}/RUL_FD001.txt", header=None).values.ravel().astype(float)
    return train, test, y_test


def add_rul(df):
    """Training RUL = cycles remaining until failure, capped."""
    last = df.groupby("unit")["cycle"].transform("max")
    df = df.copy()
    df["RUL"] = (last - df["cycle"]).clip(upper=RUL_CAP).astype(float)
    return df


def fit_scaler(train):
    mn = train[FEATURES].min()
    mx = train[FEATURES].max()
    return {"min": mn, "range": (mx - mn).replace(0, 1.0)}


def scale(df, scaler):
    df = df.copy()
    df[FEATURES] = (df[FEATURES] - scaler["min"]) / scaler["range"]
    return df


def train_sequences(df, window=WINDOW):
    """Sliding windows over each engine; target is the RUL at the window's last cycle."""
    Xs, ys = [], []
    for _, g in df.groupby("unit"):
        arr = g[FEATURES].values
        rul = g["RUL"].values
        for i in range(len(g) - window + 1):
            Xs.append(arr[i:i + window])
            ys.append(rul[i + window - 1])
    return np.asarray(Xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def test_sequences(df, window=WINDOW):
    """One window per engine: its last `window` cycles (front-padded if shorter)."""
    Xs = []
    for _, g in df.groupby("unit"):
        arr = g[FEATURES].values
        if len(arr) >= window:
            Xs.append(arr[-window:])
        else:
            pad = np.repeat(arr[:1], window - len(arr), axis=0)
            Xs.append(np.vstack([pad, arr]))
    return np.asarray(Xs, dtype=np.float32)
