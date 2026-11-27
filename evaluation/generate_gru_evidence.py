"""Produce reproducible COS30019 A2B GRU report evidence from saved checkpoints.

Run from the project root:
    python -m evaluation.generate_gru_evidence

Does not retrain or modify any existing trained models or teammate files.
Reads saved_models/gru_site_*_h*_s*.json and their held-out CSVs.
"""
from __future__ import annotations

import csv
import json
import math
import re
import statistics
from pathlib import Path

MODEL_PATTERN = re.compile(r"^gru_site_(\d+)_h(\d+)_s(\d+)\.json$")
MODEL_DIR = Path("saved_models")
OUTPUT = Path("results/gru")


def save_csv(path: Path, rows: list[dict], columns: list[str]):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def load_json_models():
    models = []
    for path in sorted(MODEL_DIR.glob("gru_site_*_h*_s*.json")):
        name = MODEL_PATTERN.fullmatch(path.name)
        if not name:
            continue
        meta = json.loads(path.read_text(encoding="utf-8"))
        if meta.get("model_type") != "GRU":
            raise ValueError(f"Unexpected model type in {path}")
        site, horizon, lookback = map(int, name.groups())
        if (str(meta["scats_site"]), int(meta["settings"]["horizon_minutes"]), int(meta["settings"]["lookback"])) != (str(site), horizon, lookback):
            raise ValueError(f"Filename and metadata disagree: {path}")
        met = meta["test_metrics"]
        if met.get("nrmse_normalisation") != "actual_range":
            raise ValueError(f"Unexpected NRMSE definition in {path}")
        models.append({
            "site": site,
            "horizon_min": horizon,
            "lookback": lookback,
            "test_samples": int(met["count"]),
            "rmse_vehicles_per_15min": float(met["rmse"]),
            "mae_vehicles_per_15min": float(met["mae"]),
            "range_nrmse_pct": float(met["nrmse"]) * 100,
            "training_time_s": float(meta["training_time_seconds"]),
            "metadata_file": str(path).replace("\\", "/"),
            "_meta": meta,
        })
    if not models:
        raise FileNotFoundError("No GRU metadata in saved_models/. Run from the project root.")
    return models


def plot_all(models):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for report graphs: python -m pip install matplotlib") from exc

    horizons = sorted((m for m in models if m["site"] == 2000 and m["lookback"] == 8), key=lambda m: m["horizon_min"])
    if len(horizons) < 2:
        print("Skipping horizon graph: not enough site-2000 horizon checkpoints.")
        return

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot([m["horizon_min"] for m in horizons], [m["rmse_vehicles_per_15min"] for m in horizons], marker="o")
    ax.set_xticks([m["horizon_min"] for m in horizons])
    ax.set_xlabel("Forecast horizon (minutes)")
    ax.set_ylabel("Test RMSE (vehicles / 15 min)")
    ax.set_title("GRU — SCATS 2000: forecast horizon versus test error")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUTPUT / "site_2000_horizon_rmse.png", dpi=160)
    plt.close(fig)

    test_path = MODEL_DIR / "gru_site_2000_h15_s8_test_predictions.csv"
    if test_path.exists():
        with test_path.open(encoding="utf-8-sig", newline="") as f:
            predictions = list(csv.DictReader(f))
        first_day = predictions[0]["target_datetime"][:10]
        day = [p for p in predictions if p["target_datetime"].startswith(first_day)]
        if day:
            fig, ax = plt.subplots(figsize=(11, 4))
            ax.plot(range(len(day)), [float(p["actual_vehicles_15min"]) for p in day], label="Actual")
            ax.plot(range(len(day)), [float(p["predicted_vehicles_15min"]) for p in day], label="GRU predicted")
            ax.set_xlabel(f"15-minute intervals on {first_day} (held-out test set)")
            ax.set_ylabel("Traffic volume (vehicles / 15 min)")
            ax.set_title("GRU — SCATS 2000: actual versus predicted traffic (15-min horizon)")
            ax.legend()
            ax.grid(alpha=0.2)
            fig.tight_layout()
            fig.savefig(OUTPUT / "site_2000_actual_vs_predicted.png", dpi=160)
            plt.close(fig)

    baseline = next((m for m in horizons if m["horizon_min"] == 15), None)
    if baseline:
        history = baseline["_meta"].get("loss_history", [])
        if history:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.plot([x["epoch"] for x in history], [x["training_mse_scaled"] for x in history], label="Training MSE")
            ax.plot([x["epoch"] for x in history], [x["validation_mse_scaled"] for x in history], label="Validation MSE")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("MSE (scaled target)")
            ax.set_title("GRU — SCATS 2000: training versus validation loss")
            ax.legend()
            ax.grid(alpha=0.2)
            fig.tight_layout()
            fig.savefig(OUTPUT / "site_2000_training_validation_loss.png", dpi=160)
            plt.close(fig)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    models = load_json_models()
    metrics = ["site", "horizon_min", "lookback", "test_samples", "rmse_vehicles_per_15min", "mae_vehicles_per_15min", "range_nrmse_pct", "training_time_s", "metadata_file"]

    rows = [{k: m[k] for k in metrics} for m in models if m["horizon_min"] == 15 and m["lookback"] == 8]
    rows.sort(key=lambda r: r["site"])
    save_csv(OUTPUT / "all_sites_15min_gru_metrics.csv", rows, metrics)

    site_2000 = [{k: m[k] for k in metrics} for m in models if m["site"] == 2000 and m["lookback"] == 8]
    site_2000.sort(key=lambda r: r["horizon_min"])
    save_csv(OUTPUT / "site_2000_horizon_comparison.csv", site_2000, metrics)

    plot_all(models)

    all_sites = [m["site"] for m in rows]
    note = f"""# GRU evaluation evidence — COS30019 Assignment 2B

Source: saved GRU metadata and held-out test prediction CSVs from this project; no models were retrained.

- GRU site coverage at **15-minute horizon / 8-step lookback:** {len(all_sites)} model checkpoints.
- Site 2000 has horizon data for: {', '.join(str(r['horizon_min']) for r in site_2000)} minutes.
- Error units: RMSE and MAE are **vehicles per 15-minute target interval**.
- NRMSE (%) = RMSE / (max(actual test) - min(actual test)) × 100, computed per model/site; **not classification accuracy**.
- Models use shared chronological training/validation/test splits, training-only scaler, saved seed and settings; compare matched targets when reviewing cross-model results.
- Models for different sites have differing held-out sample counts; do not claim pooled network accuracy from a simple unweighted average of site errors.
- Horizon results describe SCATS 2000 only; they do not establish whether GRU outperforms LSTM or Regression Tree.
- These files do not verify route quality, map connectivity, or physical validity of traffic-flow-to-speed conversion.

Evidence files:

- `all_sites_15min_gru_metrics.csv`: held-out results per available SCATS site.
- `site_2000_horizon_comparison.csv`: test metrics and elapsed training time across available horizons.
- `site_2000_horizon_rmse.png`: effect of forecast horizon on test RMSE.
- `site_2000_actual_vs_predicted.png`: first held-out test day actual/predicted series for site 2000.
- `site_2000_training_validation_loss.png`: training and validation scaled MSE by epoch for site 2000.
"""
    (OUTPUT / "README_evidence.md").write_text(note, encoding="utf-8")
    print(f"Created {OUTPUT}/")
    print(f"15-min / lookback-8 models: {len(all_sites)}")
    for row in site_2000:
        print(f"Site 2000, {row['horizon_min']} min: RMSE={row['rmse_vehicles_per_15min']:.2f}, MAE={row['mae_vehicles_per_15min']:.2f}, NRMSE={row['range_nrmse_pct']:.2f}%, n={row['test_samples']}")


if __name__ == "__main__":
    main()
