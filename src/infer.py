"""Load the trained LSTM and predict Remaining Useful Life for an engine's recent sensor window."""

import json

import numpy as np
import pandas as pd
import torch

from .data import load, FEATURES, WINDOW, RUL_CAP
from .model import LSTMRegressor

MODELS_DIR = "models"


def load_model(models_dir=MODELS_DIR):
    scaler = json.load(open(f"{models_dir}/scaler.json"))
    model = LSTMRegressor(len(scaler["features"]), hidden=96)
    model.load_state_dict(torch.load(f"{models_dir}/lstm_rul.pt", map_location="cpu"))
    model.eval()
    return model, scaler


def _window(engine_df, scaler):
    """Scale and take the last WINDOW cycles of one engine (front-pad if short)."""
    feats = scaler["features"]
    mn = pd.Series(scaler["min"]); rng = pd.Series(scaler["range"])
    x = (engine_df[feats] - mn) / rng
    arr = x.values.astype(np.float32)
    w = scaler["window"]
    if len(arr) >= w:
        arr = arr[-w:]
    else:
        arr = np.vstack([np.repeat(arr[:1], w - len(arr), axis=0), arr])
    return arr


def predict_rul(engine_df, model=None, scaler=None):
    if model is None:
        model, scaler = load_model()
    arr = _window(engine_df, scaler)
    with torch.no_grad():
        rul = float(model(torch.tensor(arr[None, ...])).item()) * scaler["rul_cap"]
    return max(0.0, round(rul, 1))


if __name__ == "__main__":
    model, scaler = load_model()
    _, test_df, y_true = load("data")
    print("Engine  predicted_RUL  true_RUL")
    for unit in [1, 2, 3, 24, 50, 100]:
        g = test_df[test_df["unit"] == unit]
        pred = predict_rul(g, model, scaler)
        print(f"  {unit:3d}      {pred:6.1f}        {y_true[unit-1]:5.0f}")
