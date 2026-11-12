"""
dataset.py
One shared way to load, split, scale and window the processed SCATS data,
so the LSTM, GRU and Regression Tree are trained and tested on identical data.

Typical use (inside a model's training script):

    from data_processing.dataset import load_series, fit_minmax, make_windows
    df = load_series("data/processed/scats_15min.csv")
    scaler = load_scaler("data/processed/scaler.json")
    X_tr, y_tr, meta_tr = make_windows(df, "train", scaler, lags=12)
    X_va, y_va, meta_va = make_windows(df, "val", scaler, lags=12)
    X_te, y_te, meta_te = make_windows(df, "test", scaler, lags=12)
    # LSTM/GRU: X.reshape(len(X), lags, 1). Regression tree: use X as is.
    # Back to vehicles per 15 min: inverse_minmax(pred, scaler)

Leakage rules this module enforces:
  * Splits are by calendar day, identical for every site (no shuffling
    before splitting).
  * The scaler is fitted on training targets only.
  * A window belongs to the split of its TARGET time. Its input lags may
    reach back into the previous split (e.g. the first validation window
    looks at the last 3 hours of training). That is past data, available
    at prediction time, so it is not leakage.
  * Windows whose inputs or target contain a missing value are dropped
    rather than filled, so no value is invented from the future.
"""

import json

import numpy as np
import pandas as pd

# 1 Oct 2006 is a Sunday. Train = first 22 days (3 full weeks + Sunday),
# validation = Mon-Thu, test = Fri-Tue (includes a weekend).
SPLIT_DAYS = {
    "train": (pd.Timestamp("2006-10-01"), pd.Timestamp("2006-10-22")),
    "val": (pd.Timestamp("2006-10-23"), pd.Timestamp("2006-10-26")),
    "test": (pd.Timestamp("2006-10-27"), pd.Timestamp("2006-10-31")),
}


def split_of(datetimes):
    """Label each timestamp 'train', 'val' or 'test' by its calendar day."""
    days = pd.to_datetime(pd.Series(datetimes)).dt.normalize()
    labels = pd.Series("unassigned", index=days.index)
    for name, (start, end) in SPLIT_DAYS.items():
        labels[(days >= start) & (days <= end)] = name
    return labels.values


def load_series(path):
    """Read scats_15min.csv with proper dtypes, sorted by site then time."""
    df = pd.read_csv(path, parse_dates=["datetime"])
    df["scats_site"] = df["scats_site"].astype(int)
    return df.sort_values(["scats_site", "datetime"]).reset_index(drop=True)


def fit_minmax(train_values):
    """Global min-max parameters from TRAINING volumes only (NaN ignored)."""
    v = pd.Series(train_values).dropna()
    return {"method": "minmax", "fitted_on": "train", "min": float(v.min()), "max": float(v.max())}


def load_scaler(path):
    with open(path) as f:
        return json.load(f)


def apply_minmax(values, scaler):
    return (np.asarray(values, dtype=float) - scaler["min"]) / (scaler["max"] - scaler["min"])


def inverse_minmax(values, scaler):
    return np.asarray(values, dtype=float) * (scaler["max"] - scaler["min"]) + scaler["min"]


def make_windows(df, split, scaler, lags=12, horizon=1, sites=None):
    """Sliding windows of `lags` past steps -> value `horizon` steps ahead.

    Returns X (n, lags), y (n,), both scaled, and a meta DataFrame with the
    site and target datetime of every window (for plots and per-site metrics).
    """
    xs, ys, meta = [], [], []
    for site, g in df.groupby("scats_site"):
        if sites is not None and site not in sites:
            continue
        g = g.set_index("datetime").asfreq("15min")  # keep gaps visible as NaN
        vals = apply_minmax(g["traffic_volume_15min"].values, scaler)
        times = g.index
        target_split = split_of(times)
        for t in range(lags + horizon - 1, len(vals)):
            if target_split[t] != split:
                continue
            window = vals[t - horizon - lags + 1: t - horizon + 1]
            if np.isnan(window).any() or np.isnan(vals[t]):
                continue
            xs.append(window)
            ys.append(vals[t])
            meta.append((site, times[t]))
    meta = pd.DataFrame(meta, columns=["scats_site", "datetime"])
    return np.array(xs), np.array(ys), meta
