"""Train and evaluate the LSTM RUL model on C-MAPSS FD001.

Reports test RMSE and the NASA asymmetric score (which penalises late predictions, the dangerous
ones, more than early ones). Saves the model, the scaler, metrics, and diagnostic plots.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .data import load, add_rul, fit_scaler, scale, train_sequences, test_sequences, FEATURES, RUL_CAP, WINDOW
from .model import LSTMRegressor

MODELS_DIR = "models"
REPORTS_DIR = "reports"
torch.manual_seed(42)
np.random.seed(42)


def nasa_score(pred, true):
    d = pred - true
    return float(np.sum(np.where(d < 0, np.exp(-d / 13.0) - 1, np.exp(d / 10.0) - 1)))


def main(epochs=40, batch=256, lr=1e-3, data_dir="data"):
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_df, test_df, y_test = load(data_dir)
    train_df = add_rul(train_df)
    scaler = fit_scaler(train_df)
    train_df = scale(train_df, scaler)
    test_df = scale(test_df, scaler)

    X, y = train_sequences(train_df, WINDOW)
    # Hold out 15 engines' windows as validation (group-aware split by engine id).
    units = train_df["unit"].values
    rng = np.random.default_rng(42)
    val_units = set(rng.choice(np.arange(1, 101), size=15, replace=False))
    # rebuild masks per window: recompute which unit each window came from
    mask_val, idx = [], 0
    for u, g in train_df.groupby("unit"):
        n = len(g) - WINDOW + 1
        mask_val.extend([u in val_units] * max(n, 0))
    mask_val = np.asarray(mask_val)
    Xtr, ytr = X[~mask_val], y[~mask_val]
    Xva, yva = X[mask_val], y[mask_val]
    print(f"train windows {len(Xtr)}, val windows {len(Xva)}, features {len(FEATURES)}")

    # Train on RUL normalised to [0, 1] so the MSE is well-scaled (otherwise the optimiser
    # collapses to predicting the mean). Predictions are scaled back to cycles for every metric.
    tl = DataLoader(TensorDataset(torch.tensor(Xtr), torch.tensor(ytr / RUL_CAP)), batch_size=batch, shuffle=True)
    model = LSTMRegressor(len(FEATURES), hidden=96).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = torch.nn.MSELoss()

    history = []
    Xva_t = torch.tensor(Xva).to(device)
    for ep in range(1, epochs + 1):
        model.train()
        tot = 0.0
        for xb, yb in tl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(xb)
        model.eval()
        with torch.no_grad():
            vp = model(Xva_t).cpu().numpy() * RUL_CAP
        vrmse = float(np.sqrt(np.mean((vp - yva) ** 2)))
        history.append({"epoch": ep, "train_mse": tot / len(Xtr), "val_rmse": vrmse})
        if ep % 5 == 0 or ep == 1:
            print(f"  epoch {ep:02d}  train_mse {tot/len(Xtr):.2f}  val_rmse {vrmse:.2f}")

    # ---- Test: one window per engine, compare to the held-out true RUL ----
    Xte = test_sequences(test_df, WINDOW)
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(Xte).to(device)).cpu().numpy() * RUL_CAP
    pred = np.clip(pred, 0, None)
    true_clip = np.clip(y_test, 0, RUL_CAP)
    rmse = float(np.sqrt(np.mean((pred - true_clip) ** 2)))
    score = nasa_score(pred, true_clip)
    print(f"TEST RMSE {rmse:.2f} cycles  |  NASA score {score:.0f}  (100 engines)")

    # ---- Plots ----
    plt.figure(figsize=(6, 4))
    plt.plot([h["epoch"] for h in history], [h["val_rmse"] for h in history], color="#3b6fd4")
    plt.xlabel("epoch"); plt.ylabel("validation RMSE"); plt.title("Validation RMSE"); plt.tight_layout()
    plt.savefig(f"{REPORTS_DIR}/val_rmse.png", dpi=120); plt.close()

    order = np.argsort(true_clip)
    plt.figure(figsize=(7, 4))
    plt.plot(true_clip[order], label="true RUL", color="#222")
    plt.plot(pred[order], label="predicted RUL", color="#d63b35", alpha=0.8)
    plt.xlabel("engine (sorted by true RUL)"); plt.ylabel("RUL (cycles)")
    plt.legend(); plt.title("Predicted vs true RUL on the test engines"); plt.tight_layout()
    plt.savefig(f"{REPORTS_DIR}/pred_vs_true.png", dpi=120); plt.close()

    # ---- Persist ----
    torch.save(model.state_dict(), f"{MODELS_DIR}/lstm_rul.pt")
    scaler_save = {"min": scaler["min"].to_dict(), "range": scaler["range"].to_dict(),
                   "features": FEATURES, "window": WINDOW, "rul_cap": RUL_CAP}
    json.dump(scaler_save, open(f"{MODELS_DIR}/scaler.json", "w"), indent=2)
    metrics = {"test_rmse": round(rmse, 2), "nasa_score": round(score, 0),
               "val_rmse_final": round(history[-1]["val_rmse"], 2), "epochs": epochs,
               "n_features": len(FEATURES), "window": WINDOW}
    json.dump(metrics, open(f"{REPORTS_DIR}/metrics.json", "w"), indent=2)
    print("Saved model, scaler, metrics, plots.")
    return metrics


if __name__ == "__main__":
    main()
