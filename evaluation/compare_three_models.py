"""Matched three-model evaluation across the SCATS network at 15, 30 and 60 minutes."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from evaluation.evaluate import evaluate_predictions

SITES = sorted({
    int(Path(p).name.split("_")[2])
    for p in Path("saved_models").glob("rt_site_*_h15_s8.json")
})
HORIZONS = (15, 30, 60)
MODEL_NAMES = ("GRU", "LSTM", "Regression Tree")
OUTPUT = Path("results/model_comparison/three_model")


def read_predictions(path: Path, model: str) -> dict:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No test predictions found in {path}")
    if model == "LSTM":
        timestamp_col, actual_col, pred_col = "datetime", "actual_traffic_volume", "predicted_traffic_volume"
    else:
        timestamp_col, actual_col, pred_col = "target_datetime", "actual_vehicles_15min", "predicted_vehicles_15min"
    return {
        "timestamps": [r[timestamp_col] for r in rows],
        "actual": [float(r[actual_col]) for r in rows],
        "predicted": [float(r[pred_col]) for r in rows],
    }


def paths_for(site: int, horizon: int) -> dict[str, Path]:
    result = {
        "GRU": Path(f"saved_models/gru_site_{site}_h{horizon}_s8_test_predictions.csv"),
        "Regression Tree": Path(f"saved_models/rt_site_{site}_h{horizon}_s8_test_predictions.csv"),
    }
    candidates = []
    if site == 2000:
        candidates.append(Path(f"results/lstm_site2000_comparison/site_{site}_h{horizon}_s8/predictions.csv"))
    # Full-network LSTM runs are stored per horizon (15, 30 and 60 minutes).
    candidates.append(Path(f"results/lstm/full_network_h{horizon}_s8/site_{site}_h{horizon}_s8/predictions.csv"))
    if horizon == 15:
        candidates.append(Path(f"results/lstm/final_two_sites_h15_s8/site_{site}_h15_s8/predictions.csv"))
    candidates.append(Path(f"results/lstm/site_{site}_h{horizon}_s8/predictions.csv"))
    result["LSTM"] = next((p for p in candidates if p.exists()), candidates[-1])
    return result


def metrics_for(site: int, horizon: int, model: str, pred: dict) -> dict:
    m = evaluate_predictions(pred["actual"], pred["predicted"], normalisation="range")
    model_path = Path("saved_models") / f"{'rt' if model == 'Regression Tree' else model.lower()}_site_{site}_h{horizon}_s8.json"
    metadata = json.loads(model_path.read_text(encoding="utf-8"))
    if model == "LSTM":
        # The LSTM model metadata stores training time; evaluation metrics beside its CSV store inference time.
        training_seconds = float(metadata["training_time_seconds"])
        prediction_seconds = None
        pred_path = paths_for(site, horizon)["LSTM"]
        metrics_path = pred_path.parent / "metrics.json"
        if metrics_path.exists():
            evaluation = json.loads(metrics_path.read_text(encoding="utf-8"))
            prediction_seconds = evaluation.get("prediction_time_seconds")
    else:
        training_seconds = float(metadata["training_time_seconds"])
        prediction_seconds = metadata.get("test_metrics", {}).get("prediction_time_seconds_total")
    model_selection_seconds = metadata.get("hyperparameter_search_seconds") if model == "Regression Tree" else None
    return {
        "model": model,
        "scats_site": site,
        "horizon_minutes": horizon,
        "sequence_length": 8,
        "sample_count": int(m["count"]),
        "mae": float(m["mae"]),
        "rmse": float(m["rmse"]),
        "nrmse": float(m["nrmse"]),
        "nrmse_percent": float(m["nrmse"]) * 100.0,
        "nrmse_normalisation": "site_test_actual_range",
        "training_time_seconds": training_seconds,
        "model_selection_seconds": model_selection_seconds,
        "prediction_time_seconds": prediction_seconds,
        "prediction_time_per_sample_seconds": (
            prediction_seconds / int(m["count"]) if prediction_seconds is not None else None
        ),
    }


def compare() -> dict:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = []
    common_sites = {}
    site2000 = {}

    for horizon in HORIZONS:
        sites_for_horizon = []
        for site in SITES:
            paths = paths_for(site, horizon)
            if not all(path.exists() for path in paths.values()):
                continue
            preds = {name: read_predictions(paths[name], name) for name in MODEL_NAMES}
            reference = preds["GRU"]
            for name in ("LSTM", "Regression Tree"):
                other = preds[name]
                if reference["timestamps"] != other["timestamps"]:
                    raise ValueError(f"Timestamp mismatch at site {site}, horizon {horizon}, model {name}")
                if len(reference["actual"]) != len(other["actual"]):
                    raise ValueError(f"Sample-count mismatch at site {site}, horizon {horizon}, model {name}")
                if any(abs(a-b) > 1e-4 for a,b in zip(reference["actual"], other["actual"])):
                    raise ValueError(f"Actual-value mismatch at site {site}, horizon {horizon}, model {name}")
            sites_for_horizon.append(site)
            for name in MODEL_NAMES:
                row = metrics_for(site, horizon, name, preds[name])
                rows.append(row)
                if site == 2000:
                    site2000[(horizon,name)] = row
        common_sites[str(horizon)] = sites_for_horizon

    if not rows:
        raise RuntimeError("No common model test predictions were found")

    with (OUTPUT / "per_site_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "comparison": "GRU vs LSTM vs Regression Tree",
        "sequence_length": 8,
        "common_sites_by_horizon": common_sites,
        "matched_test_predictions_by_horizon": {},
        "nrmse_method": "Each site's RMSE divided by the actual-value range for that site's full test split; network averages are macro averages across sites.",
        "macro_average_metrics_by_horizon": {},
        "lower_rmse_site_wins_by_horizon": {},
        "site_2000_metrics": {},
    }

    for horizon in HORIZONS:
        site_ids = common_sites[str(horizon)]
        summary["matched_test_predictions_by_horizon"][str(horizon)] = sum(
            next(r["sample_count"] for r in rows if r["scats_site"] == site and r["horizon_minutes"] == horizon and r["model"] == "GRU")
            for site in site_ids
        )
        summary["macro_average_metrics_by_horizon"][str(horizon)] = {}
        wins = {name: 0 for name in MODEL_NAMES}
        for site in site_ids:
            per_site = {name: next(r for r in rows if r["scats_site"] == site and r["horizon_minutes"] == horizon and r["model"] == name) for name in MODEL_NAMES}
            winner = min(MODEL_NAMES, key=lambda name: per_site[name]["rmse"])
            wins[winner] += 1
        summary["lower_rmse_site_wins_by_horizon"][str(horizon)] = wins
        for name in MODEL_NAMES:
            selected = [r for r in rows if r["horizon_minutes"] == horizon and r["model"] == name]
            if selected:
                summary["macro_average_metrics_by_horizon"][str(horizon)][name] = {
                    metric: sum(r[metric] for r in selected) / len(selected)
                    for metric in ("mae", "rmse", "nrmse", "nrmse_percent", "training_time_seconds")
                }
                available_selection_times = [r["model_selection_seconds"] for r in selected if r["model_selection_seconds"] is not None]
                summary["macro_average_metrics_by_horizon"][str(horizon)][name]["model_selection_seconds"] = (
                    sum(available_selection_times) / len(available_selection_times) if available_selection_times else None
                )
                available_prediction_times = [r["prediction_time_per_sample_seconds"] for r in selected if r["prediction_time_per_sample_seconds"] is not None]
                summary["macro_average_metrics_by_horizon"][str(horizon)][name]["prediction_time_per_sample_seconds"] = (
                    sum(available_prediction_times) / len(available_prediction_times) if available_prediction_times else None
                )
        summary["site_2000_metrics"][str(horizon)] = {
            name: site2000[(horizon,name)] for name in MODEL_NAMES if (horizon,name) in site2000
        }

    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Network-wide 15-minute macro average across the sites with all three models.
    horizon = 15
    network = summary["macro_average_metrics_by_horizon"][str(horizon)]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, metric, title in zip(axes, ("mae", "rmse", "nrmse_percent"), ("Macro mean MAE", "Macro mean RMSE", "Macro mean NRMSE (%)")):
        vals = [network[name][metric] for name in MODEL_NAMES]
        bars = ax.bar(["GRU", "LSTM", "Tree"], vals, color=["#4472C4", "#ED7D31", "#70AD47"])
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)
    fig.suptitle(f"Three-model comparison · 15 min · lookback 8 · {len(common_sites['15'])} shared sites")
    fig.tight_layout()
    fig.savefig(OUTPUT / "network_h15_macro_comparison.png", dpi=180)
    plt.close(fig)

    # Network-wide macro averages for every forecast horizon.
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, metric, title in zip(axes, ("mae", "rmse", "nrmse_percent"), ("Macro mean MAE", "Macro mean RMSE", "Macro mean NRMSE (%)")):
        for name, color in zip(MODEL_NAMES, ("#4472C4", "#ED7D31", "#70AD47")):
            values = [summary["macro_average_metrics_by_horizon"][str(h)].get(name, {}).get(metric) for h in HORIZONS]
            ax.plot(HORIZONS, values, marker="o", label=name, color=color)
        ax.set_title(title)
        ax.set_xlabel("Horizon (minutes)")
        ax.set_xticks(HORIZONS)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Vehicles / 15 min")
    axes[0].legend()
    fig.suptitle(f"Three-model comparison · all horizons · lookback 8 · {len(common_sites['60'])} shared sites")
    fig.tight_layout()
    fig.savefig(OUTPUT / "network_all_horizons_comparison.png", dpi=180)
    plt.close(fig)

    # Site 2000 is the matched reference for all three forecast horizons.
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, metric, title in zip(axes, ("mae", "rmse", "nrmse_percent"), ("MAE", "RMSE", "NRMSE (%)")):
        for name, color in zip(MODEL_NAMES, ("#4472C4", "#ED7D31", "#70AD47")):
            values = [site2000[(h,name)][metric] for h in HORIZONS]
            ax.plot(HORIZONS, values, marker="o", label=name, color=color)
        ax.set_title(title)
        ax.set_xlabel("Horizon (minutes)")
        ax.set_xticks(HORIZONS)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Vehicles / 15 min")
    axes[0].legend()
    fig.suptitle("Three-model comparison · site 2000 · lookback 8")
    fig.tight_layout()
    fig.savefig(OUTPUT / "site_2000_horizons_comparison.png", dpi=180)
    plt.close(fig)
    return summary


if __name__ == "__main__":
    print(json.dumps(compare(), indent=2))
