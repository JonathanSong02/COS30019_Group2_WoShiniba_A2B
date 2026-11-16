"""Regression Tree traffic-flow forecasting for COS30019 Assignment 2B (Darren).

The third ML model, compared against the LSTM and GRU under IDENTICAL
conditions: same shared preprocessing (data_processing.dataset), same calendar
train/val/test split, same train-only min-max scaler, same sliding windows
(the previous `lookback` 15-minute counts predict the count `horizon` minutes
after the last observation) and the same metrics (evaluation.evaluate).

How the model works
-------------------
A regression tree splits the input space with yes/no questions such as
"was the count 15 min ago (lag_1) above 0.42?". Each leaf stores the mean
target of the training windows that reach it, and that mean is the forecast.
max_depth and min_samples_leaf control how finely it splits; they are chosen
on the VALIDATION period only (grid search). The test period is never used
for any choice.

Saved files (saved_models/, same naming as the GRU):
    rt_site_<site>_h<h>_s<lookback>.json                  metadata + metrics
    rt_site_<site>_h<h>_s<lookback>_tree.json             the trained tree
    rt_site_<site>_h<h>_s<lookback>_test_predictions.csv  test predictions

The tree is stored as plain arrays (feature, threshold, children, leaf value),
so PREDICTION needs no scikit-learn and no pickle files, and works the same on
every teammate's machine. scikit-learn is only needed for TRAINING.

Usage:
    python -m models.regression_tree_model train --site 2000 --horizon 15
    python -m models.regression_tree_model train-all --horizons 15 30 60
    python -m models.regression_tree_model predict --site 2000 --time 2006-10-27T08:00
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path

import numpy as np

from data_processing.dataset import (SPLIT_DAYS, apply_minmax, inverse_minmax,
                                     load_scaler, load_series, make_windows)
from evaluation.evaluate import calculate_mae, calculate_nrmse, calculate_rmse

INTERVAL_MINUTES = 15
SUPPORTED_HORIZONS = {15, 30, 60}
MODEL_TYPE = "RegressionTree"
DEFAULT_CSV = Path("data/processed/scats_15min.csv")
DEFAULT_MODELS = Path("saved_models")


@dataclass
class TreeSettings:
    site: str
    horizon_minutes: int = 15
    lookback: int = 8  # same as GRU lookback / LSTM sequence_length
    max_depth_grid: list = field(default_factory=lambda: [3, 4, 5, 6, 8, 10, 12, None])
    min_samples_leaf_grid: list = field(default_factory=lambda: [1, 5, 10, 20, 50])
    seed: int = 42

    def validate(self):
        validate_horizon(self.horizon_minutes)
        if type(self.lookback) is not int or self.lookback < 1:
            raise ValueError("lookback must be a positive integer")
        if not self.max_depth_grid or not self.min_samples_leaf_grid:
            raise ValueError("hyper-parameter grids must not be empty")


def validate_horizon(horizon_minutes: int) -> int:
    if type(horizon_minutes) is not int or horizon_minutes not in SUPPORTED_HORIZONS:
        raise ValueError("horizon_minutes must be 15, 30 or 60")
    return horizon_minutes // INTERVAL_MINUTES


def _prefix(site, horizon_minutes, lookback):
    return f"rt_site_{site}_h{horizon_minutes}_s{lookback}"


def _metrics(actual, predicted):
    """Same metric definitions as the GRU/LSTM (NRMSE = RMSE / actual range)."""
    return {
        "count": len(actual),
        "mae": calculate_mae(actual, predicted),
        "rmse": calculate_rmse(actual, predicted),
        "nrmse": calculate_nrmse(actual, predicted, normalisation="range"),
        "nrmse_normalisation": "actual_range",
    }


def _shared_splits(processed_csv, site, settings, scaler_path=None):
    """Exactly the windows the GRU and LSTM use (data_processing.make_windows)."""
    processed_csv = Path(processed_csv)
    config_file = processed_csv.parent / "split_config.json"
    if config_file.is_file():
        expected = {k: [str(a.date()), str(b.date())] for k, (a, b) in SPLIT_DAYS.items()}
        if json.loads(config_file.read_text(encoding="utf-8")) != expected:
            raise ValueError("split_config.json disagrees with data_processing.dataset.SPLIT_DAYS")
    scaler = load_scaler(scaler_path or processed_csv.parent / "scaler.json")
    if scaler.get("method") != "minmax" or scaler.get("fitted_on") != "train":
        raise ValueError("Shared scaler must be the train-only minmax scaler")
    df = load_series(processed_csv)
    site_num = int(site)
    site_df = df[df["scats_site"] == site_num]
    if site_df.empty:
        raise ValueError(f"No SCATS data for site {site_num}")
    splits = {}
    for name in ("train", "val", "test"):
        x, y, meta = make_windows(site_df, name, scaler, lags=settings.lookback,
                                  horizon=settings.horizon_minutes // INTERVAL_MINUTES,
                                  sites={site_num})
        if len(x) == 0:
            raise ValueError(f"No complete {name} windows for SCATS site {site_num}")
        splits[name] = (x, y, list(meta["datetime"]))
    return splits, scaler


def _export_tree(model):
    """sklearn tree -> plain JSON-serialisable arrays."""
    t = model.tree_
    return {
        "children_left": t.children_left.tolist(),
        "children_right": t.children_right.tolist(),
        "feature": t.feature.tolist(),
        "threshold": t.threshold.tolist(),
        "value": t.value[:, 0, 0].tolist(),
        "n_features": int(model.n_features_in_),
    }


def predict_with_tree(tree, x):
    """Walk the exported tree for each row of x (scaled lags). Matches sklearn:
    go left when feature value <= threshold."""
    x = np.atleast_2d(np.asarray(x, dtype=float))
    if x.shape[1] != tree["n_features"]:
        raise ValueError(f"Expected {tree['n_features']} input values, got {x.shape[1]}")
    left, right = tree["children_left"], tree["children_right"]
    feature, threshold, value = tree["feature"], tree["threshold"], tree["value"]
    out = np.empty(len(x))
    for i, row in enumerate(x):
        node = 0
        while left[node] != -1:
            node = left[node] if row[feature[node]] <= threshold[node] else right[node]
        out[i] = value[node]
    return out


def train_regression_tree(processed_csv: str | Path, output_folder: str | Path,
                          settings: TreeSettings, scaler_path: str | Path | None = None):
    """Tune on validation, fit on train, evaluate on held-out test, save files."""
    from sklearn.tree import DecisionTreeRegressor  # training-only dependency

    settings.validate()
    settings.site = str(int(settings.site))
    splits, scaler = _shared_splits(processed_csv, settings.site, settings, scaler_path)
    (x_tr, y_tr, _), (x_va, y_va, _) = splits["train"], splits["val"]

    # 1) Hyper-parameter search on the validation period only.
    started = time.perf_counter()
    search = []
    for depth in settings.max_depth_grid:
        for leaf in settings.min_samples_leaf_grid:
            m = DecisionTreeRegressor(max_depth=depth, min_samples_leaf=leaf,
                                      random_state=settings.seed).fit(x_tr, y_tr)
            val_mse = float(np.mean((m.predict(x_va) - y_va) ** 2))
            search.append({"max_depth": depth, "min_samples_leaf": leaf,
                           "validation_mse_scaled": val_mse})
    best = min(search, key=lambda r: r["validation_mse_scaled"])
    search_seconds = time.perf_counter() - started

    # 2) Final model: best settings, fitted on the training period.
    started = time.perf_counter()
    model = DecisionTreeRegressor(max_depth=best["max_depth"],
                                  min_samples_leaf=best["min_samples_leaf"],
                                  random_state=settings.seed).fit(x_tr, y_tr)
    training_seconds = time.perf_counter() - started
    tree = _export_tree(model)

    def evaluate(name):
        x, y, times = splits[name]
        t0 = time.perf_counter()
        scaled = predict_with_tree(tree, x)
        seconds = time.perf_counter() - t0
        predicted = np.maximum(0.0, inverse_minmax(scaled, scaler))
        actual = inverse_minmax(y, scaler)
        metrics = _metrics(actual, predicted)
        metrics["prediction_time_seconds_total"] = seconds
        metrics["prediction_time_seconds_per_sample"] = seconds / len(y)
        return metrics, list(zip(times, actual.tolist(), predicted.tolist()))

    validation_metrics, _ = evaluate("val")
    test_metrics, predictions = evaluate("test")

    out = Path(output_folder)
    out.mkdir(parents=True, exist_ok=True)
    prefix = _prefix(settings.site, settings.horizon_minutes, settings.lookback)
    tree_file = out / f"{prefix}_tree.json"
    metadata_file = out / f"{prefix}.json"
    prediction_file = out / f"{prefix}_test_predictions.csv"
    tree_file.write_text(json.dumps(tree), encoding="utf-8")
    with prediction_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["scats_site", "target_datetime", "actual_vehicles_15min",
                         "predicted_vehicles_15min"])
        writer.writerows((settings.site, t.isoformat(), a, p) for t, a, p in predictions)
    importances = {f"lag_{settings.lookback - i}": float(v)  # lag_1 = most recent count
                   for i, v in enumerate(model.feature_importances_)}
    metadata = {
        "model_type": MODEL_TYPE, "scats_site": settings.site,
        "settings": asdict(settings), "scaler": scaler,
        "split_method": "shared_calendar_days",
        "selected_hyperparameters": {"max_depth": best["max_depth"],
                                     "min_samples_leaf": best["min_samples_leaf"]},
        "hyperparameter_search": search,
        "tree_depth": int(model.get_depth()), "tree_leaves": int(model.get_n_leaves()),
        "feature_importance": importances,
        "training_samples": len(y_tr), "validation_samples": len(y_va),
        "test_samples": len(splits["test"][1]),
        "training_time_seconds": training_seconds,
        "hyperparameter_search_seconds": search_seconds,
        "best_validation_mse_scaled": best["validation_mse_scaled"],
        "validation_metrics": validation_metrics, "test_metrics": test_metrics,
        "prediction_semantics": "as_of_time + horizon_minutes",
        "output_units": "vehicles per 15-minute interval",
        "note": "Hyper-parameters chosen on validation only; test is held out.",
    }
    metadata_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {"tree_path": str(tree_file), "metadata_path": str(metadata_file),
            "test_predictions_path": str(prediction_file),
            "training_time_seconds": training_seconds,
            "selected_hyperparameters": metadata["selected_hyperparameters"],
            "validation": validation_metrics, "test": test_metrics}


@lru_cache(maxsize=4)
def _observations(processed_csv: str, mtime: float):
    """{site: {timestamp: count}} for non-missing values; cached per file version."""
    df = load_series(processed_csv).dropna(subset=["traffic_volume_15min"])
    return {site: dict(zip(g["datetime"].tolist(), g["traffic_volume_15min"]))
            for site, g in df.groupby("scats_site")}


@lru_cache(maxsize=256)
def _load_model(metadata_path: str, tree_path: str, mtime: float):
    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    tree = json.loads(Path(tree_path).read_text(encoding="utf-8"))
    return metadata, tree


def predict_regression_tree(scats_site: str | int, as_of_time: str | datetime,
                            horizon_minutes: int, processed_csv: str | Path = DEFAULT_CSV,
                            model_dir: str | Path = DEFAULT_MODELS, sequence_length: int = 8,
                            scaler_path: str | Path | None = None) -> float:
    """Predict the 15-minute count at as_of_time + horizon (same contract as predict_gru)."""
    validate_horizon(horizon_minutes)
    site = str(int(str(scats_site).strip()))
    root = Path(model_dir)
    prefix = _prefix(site, horizon_minutes, sequence_length)
    metadata_file, tree_file = root / f"{prefix}.json", root / f"{prefix}_tree.json"
    if not metadata_file.exists() or not tree_file.exists():
        raise FileNotFoundError(
            f"No Regression Tree trained for site={site}, horizon={horizon_minutes}, "
            f"sequence={sequence_length}")
    metadata, tree = _load_model(str(metadata_file), str(tree_file),
                                 max(metadata_file.stat().st_mtime, tree_file.stat().st_mtime))
    if (metadata["model_type"] != MODEL_TYPE or metadata["scats_site"] != site
            or metadata["settings"]["horizon_minutes"] != horizon_minutes
            or metadata["settings"]["lookback"] != sequence_length):
        raise ValueError("Saved Regression Tree does not match the requested settings")
    as_of = datetime.fromisoformat(as_of_time) if isinstance(as_of_time, str) else as_of_time
    if (not isinstance(as_of, datetime) or as_of.tzinfo is not None or as_of.minute % 15
            or as_of.second or as_of.microsecond):
        raise ValueError("as_of_time must be a 15-minute aligned naive local datetime")
    scaler = metadata["scaler"]
    if scaler_path is not None and load_scaler(scaler_path) != scaler:
        raise ValueError("Provided scaler differs from the scaler used for training")
    path = Path(processed_csv)
    observations = _observations(str(path), path.stat().st_mtime).get(int(site), {})
    required = [as_of - timedelta(minutes=INTERVAL_MINUTES * i)
                for i in reversed(range(sequence_length))]
    if any(t not in observations for t in required):
        raise ValueError("Insufficient nonmissing historical observations for this time")
    x = apply_minmax([observations[t] for t in required], scaler)
    output = float(inverse_minmax(predict_with_tree(tree, x), scaler)[0])
    return max(0.0, output)


def train_all(processed_csv, output_folder, horizons=(15,), lookback=8, sites=None):
    """Train every SCATS site for each horizon; returns (results, failures)."""
    df = load_series(processed_csv)
    sites = sorted(sites or df["scats_site"].unique())
    results, failures = [], []
    for h in horizons:
        for site in sites:
            try:
                r = train_regression_tree(processed_csv, output_folder,
                                          TreeSettings(site=str(site), horizon_minutes=h,
                                                       lookback=lookback))
                results.append((site, h, r["test"]))
                print(f"site {site} h{h}: test RMSE {r['test']['rmse']:.2f}, "
                      f"MAE {r['test']['mae']:.2f}, params {r['selected_hyperparameters']}")
            except Exception as exc:  # keep going, report at the end
                failures.append((site, h, str(exc)))
                print(f"site {site} h{h}: FAILED - {exc}")
    return results, failures


def main():
    parser = argparse.ArgumentParser(description="Train or predict an A2B Regression Tree")
    parser.add_argument("command", choices=["train", "train-all", "predict"])
    parser.add_argument("--csv", default=str(DEFAULT_CSV))
    parser.add_argument("--site")
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--horizons", type=int, nargs="+", default=[15])
    parser.add_argument("--models-dir", default=str(DEFAULT_MODELS))
    parser.add_argument("--lookback", "--sequence-length", dest="lookback", type=int, default=8)
    parser.add_argument("--time", help="Last observed interval, e.g. 2006-10-27T08:00")
    args = parser.parse_args()
    if args.command == "train":
        if not args.site:
            parser.error("train requires --site")
        result = train_regression_tree(args.csv, args.models_dir,
                                       TreeSettings(site=args.site, horizon_minutes=args.horizon,
                                                    lookback=args.lookback))
        print(json.dumps(result, indent=2))
    elif args.command == "train-all":
        _, failures = train_all(args.csv, args.models_dir, args.horizons, args.lookback)
        print(f"Done. {len(failures)} failure(s).")
    else:
        if not args.site or not args.time:
            parser.error("predict requires --site and --time")
        value = predict_regression_tree(args.site, args.time, args.horizon, args.csv,
                                        args.models_dir, sequence_length=args.lookback)
        print(json.dumps({"scats_site": args.site, "as_of": args.time,
                          "horizon_minutes": args.horizon,
                          "predicted_vehicles_per_15min": value}, indent=2))


if __name__ == "__main__":
    main()
