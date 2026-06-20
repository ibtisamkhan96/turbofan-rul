# Turbofan Remaining Useful Life (Deep Learning)

A PyTorch **LSTM** that predicts the **Remaining Useful Life (RUL)** of jet engines from multivariate sensor time-series, on NASA's public-domain **C-MAPSS FD001** run-to-failure dataset (100 engines, 21 sensors). This is the deep-learning, sensor-time-series side of predictive maintenance: how many cycles a machine has left before it fails.

## Result (verified on the held-out test set)

- **Test RMSE: 13.3 cycles** across 100 engines (a strong result; common LSTM baselines on FD001 sit around 16 to 18).
- **NASA asymmetric score: 284.** This score penalises late predictions, the dangerous ones, more heavily than early ones.
- The model is most accurate near end of life, which is where a maintenance decision actually gets made. See `reports/pred_vs_true.png`.

Example predictions vs the held-out truth:

| Engine | Predicted RUL | True RUL |
|--------|---------------|----------|
| 24 | 22.9 | 20 |
| 50 | 87.5 | 79 |
| 100 | 21.3 | 20 |

## Method

- **RUL label:** cycles remaining until failure, **capped at 125** (piecewise-linear RUL). Degradation only becomes visible late in life, so a cap stops the model chasing meaningless large early-life values, the standard C-MAPSS practice.
- **Sensors:** the 14 informative sensors are kept; the 7 that are constant in FD001 are dropped.
- **Sequences:** each engine's history is windowed into length-30 sliding sequences; the model sees the last 30 cycles and predicts the RUL at the most recent one.
- **Model:** a 2-layer LSTM (hidden 96) into a small MLP head, MSE loss, Adam, gradient clipping.
- **Key training detail:** the RUL target is normalised to [0, 1] before the loss. Without it, the 0-to-125 scale makes the MSE huge and the optimiser collapses to predicting the mean (RMSE around 41). Normalising fixes it and the RMSE drops to 13.

## Run it

```bash
pip install -r requirements.txt
python -m src.train     # trains the LSTM, evaluates on the test engines, saves the model + plots
python -m src.infer     # predict RUL for a few test engines vs the true labels
```

The trained model is committed, so `infer` runs without retraining.

## How it pairs with the predictive-maintenance project

The classification project answers "is this machine about to fail, and how." This one answers "how long has it got." Together they cover both halves of condition-based maintenance: anomaly and failure detection on tabular sensor snapshots, and remaining-life forecasting on sensor time-series.

## Honest notes

- FD001 is a single operating condition with a single fault mode, the easiest C-MAPSS subset. FD002 to FD004 add operating conditions and fault modes and are harder; the same pipeline extends to them with condition-aware normalisation.
- RUL is reported in cycles, capped at 125, so very healthy engines all read near the cap, which is expected and intended.

## Stack

Python, PyTorch, NumPy, Pandas, Matplotlib. Data: NASA C-MAPSS Turbofan Engine Degradation (FD001), public domain.

---

*Built by Ibtisam Ahmed Khan, materials engineer and data and AI practitioner. [linkedin.com/in/ibtisam-ahmed-khan](https://linkedin.com/in/ibtisam-ahmed-khan)*
