"""Regression Tree extras for the model comparison (Darren, Task 5).

Adds what only the Regression Tree can show, next to Anson's matched
three-model comparison (evaluation/compare_three_models.py):

1. Feature importance: which of the 8 past readings each tree relies on,
   summarised over all 40 sites for every horizon.
2. Model size: saved file size and number of learned parameters for GRU,
   LSTM and Regression Tree (15 min, every site that has all three).
3. Peak vs off-peak for the Regression Tree at site 2000, with exactly the
   same periods and NRMSE as evaluation/analyze_peak_offpeak.py, merged with
   Anson's GRU/LSTM rows into one three-model table.
4. A naive "last value" baseline on the same test windows, to show how much
   the tree adds over simply repeating the latest reading.

Run from the project root:
    python -m evaluation.regression_tree_extras
Outputs go to results/model_comparison/regression_tree/.
"""
from __future__ import annotations

import ast
import csv
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from evaluation.analyze_peak_offpeak import is_peak_period

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "saved_models"
OUTPUT = ROOT / "results" / "model_comparison" / "regression_tree"
PROCESSED_CSV = ROOT / "data" / "processed" / "scats_15min.csv"
ANSON_PEAK = ROOT / "results" / "model_comparison" / "peak_offpeak" / "peak_offpeak_metrics.csv"
HORIZONS = (15, 30, 60)
LOOKBACK = 8
LAGS = [f"lag_{i}" for i in range(1, LOOKBACK + 1)]   # lag_1 = most recent reading
COLOURS = {"GRU": "#2a78d6", "LSTM": "#eb6834", "Regression Tree": "#1baf7a"}
HIDDEN_SIZE = 32   # GRU and LSTM settings in their saved metadata (input 1, one layer)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _field(value):
    """Metadata fields may be dicts or their Python repr as a string."""
    return ast.literal_eval(value) if isinstance(value, str) else value


def rt_sites(horizon: int, models_dir: Path = MODELS_DIR) -> list[int]:
    return sorted(int(p.name.split("_")[2]) for p in models_dir.glob(f"rt_site_*_h{horizon}_s{LOOKBACK}.json"))


def rt_metadata(site: int, horizon: int, models_dir: Path = MODELS_DIR) -> dict:
    return json.loads((models_dir / f"rt_site_{site}_h{horizon}_s{LOOKBACK}.json").read_text(encoding="utf-8"))


def recurrent_parameters(cell_gates: int, hidden: int = HIDDEN_SIZE, inputs: int = 1) -> int:
    """PyTorch GRU (3 gates) / LSTM (4 gates), one layer, plus the Linear(hidden, 1) head."""
    per_gate = hidden * inputs + hidden * hidden + 2 * hidden   # W_ih, W_hh, b_ih, b_hh
    return cell_gates * per_gate + hidden + 1


# --------------------------------------------------------------------------
# 1. feature importance
# --------------------------------------------------------------------------
def feature_importance(models_dir: Path = MODELS_DIR) -> list[dict]:
    rows = []
    for h in HORIZONS:
        for site in rt_sites(h, models_dir):
            imp = _field(rt_metadata(site, h, models_dir)["feature_importance"])
            total = sum(imp.values()) or 1.0
            rows.append({"scats_site": site, "horizon_minutes": h,
                         **{lag: float(imp.get(lag, 0.0)) / total for lag in LAGS}})
    return rows


def importance_summary(rows: list[dict]) -> dict:
    out = {}
    for h in HORIZONS:
        sel = [r for r in rows if r["horizon_minutes"] == h]
        if not sel:
            continue
        out[str(h)] = {
            "sites": len(sel),
            "mean_importance": {lag: statistics.fmean(r[lag] for r in sel) for lag in LAGS},
            "sites_where_lag_1_is_top": sum(max(LAGS, key=lambda lag: r[lag]) == "lag_1" for r in sel),
        }
    return out


# --------------------------------------------------------------------------
# 2. model size
# --------------------------------------------------------------------------
def model_sizes(models_dir: Path = MODELS_DIR, horizon: int = 15) -> list[dict]:
    rows = []
    for site in rt_sites(horizon, models_dir):
        files = {
            "GRU": models_dir / f"gru_site_{site}_h{horizon}_s{LOOKBACK}.pt",
            "LSTM": models_dir / f"lstm_site_{site}_h{horizon}_s{LOOKBACK}.pt",
            "Regression Tree": models_dir / f"rt_site_{site}_h{horizon}_s{LOOKBACK}_tree.json",
        }
        if not all(f.exists() for f in files.values()):
            continue
        leaves = int(rt_metadata(site, horizon, models_dir)["tree_leaves"])
        params = {"GRU": recurrent_parameters(3), "LSTM": recurrent_parameters(4),
                  # each split stores a feature and a threshold, each leaf a value
                  "Regression Tree": 2 * (leaves - 1) + leaves}
        for model, path in files.items():
            rows.append({"model": model, "scats_site": site, "horizon_minutes": horizon,
                         "file": path.name, "file_kb": round(path.stat().st_size / 1024, 2),
                         "learned_parameters": params[model],
                         "tree_leaves": leaves if model == "Regression Tree" else None})
    return rows


def size_summary(rows: list[dict]) -> dict:
    out = {}
    for model in COLOURS:
        sel = [r for r in rows if r["model"] == model]
        if sel:
            out[model] = {"sites": len(sel),
                          "median_file_kb": statistics.median(r["file_kb"] for r in sel),
                          "median_learned_parameters": statistics.median(r["learned_parameters"] for r in sel)}
    return out


# --------------------------------------------------------------------------
# 3. peak vs off-peak for the Regression Tree (same method as Anson's script)
# --------------------------------------------------------------------------
def rt_peak_offpeak(site: int = 2000, models_dir: Path = MODELS_DIR) -> list[dict]:
    rows = []
    for h in HORIZONS:
        path = models_dir / f"rt_site_{site}_h{h}_s{LOOKBACK}_test_predictions.csv"
        with path.open(newline="", encoding="utf-8") as f:
            data = [(r["target_datetime"], float(r["actual_vehicles_15min"]), float(r["predicted_vehicles_15min"]))
                    for r in csv.DictReader(f)]
        denominator = max(a for _, a, _ in data) - min(a for _, a, _ in data)
        for period, keep in (("Peak", True), ("Off-Peak", False)):
            sel = [(a, p) for t, a, p in data if is_peak_period(t) == keep]
            errors = [p - a for a, p in sel]
            rmse = (sum(e * e for e in errors) / len(errors)) ** 0.5
            rows.append({"model": "Regression Tree", "scats_site": site, "horizon_minutes": h,
                         "period": period, "sample_count": len(sel),
                         "mae": sum(abs(e) for e in errors) / len(errors), "rmse": rmse,
                         "nrmse": rmse / denominator, "nrmse_percent": rmse / denominator * 100,
                         "nrmse_normalisation": "full_test_set_range",
                         "nrmse_denominator": denominator})
    return rows


def three_model_peak_offpeak(rt_rows: list[dict], anson_csv: Path = ANSON_PEAK) -> list[dict]:
    rows = []
    if anson_csv.exists():
        with anson_csv.open(newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rows.append({**r, "horizon_minutes": int(r["horizon_minutes"]), "sample_count": int(r["sample_count"]),
                             **{k: float(r[k]) for k in ("mae", "rmse", "nrmse", "nrmse_percent", "nrmse_denominator")}})
    rows += rt_rows
    order = {m: i for i, m in enumerate(COLOURS)}
    return sorted(rows, key=lambda r: (r["horizon_minutes"], r["period"] != "Peak", order[r["model"]]))


# --------------------------------------------------------------------------
# 4. naive "last value" baseline on the same test windows
# --------------------------------------------------------------------------
def persistence_baseline(models_dir: Path = MODELS_DIR, processed_csv: Path = PROCESSED_CSV) -> list[dict]:
    """Predict that traffic at the target time equals the last observed reading.

    Uses the Regression Tree's own test windows, so the comparison is like for
    like: same targets, same actual values, same 480 points per site.
    """
    observed: dict[tuple[int, str], float] = {}
    with processed_csv.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["traffic_volume_15min"] not in ("", "nan", "NaN"):
                observed[(int(r["scats_site"]), r["datetime"])] = float(r["traffic_volume_15min"])
    from datetime import datetime, timedelta
    rows = []
    for h in HORIZONS:
        for site in rt_sites(h, models_dir):
            path = models_dir / f"rt_site_{site}_h{h}_s{LOOKBACK}_test_predictions.csv"
            with path.open(newline="", encoding="utf-8") as f:
                data = list(csv.DictReader(f))
            naive, tree = [], []
            for r in data:
                last = datetime.fromisoformat(r["target_datetime"]) - timedelta(minutes=h)
                value = observed.get((site, last.strftime("%Y-%m-%d %H:%M")))
                if value is None:
                    continue
                actual = float(r["actual_vehicles_15min"])
                naive.append(value - actual)
                tree.append(float(r["predicted_vehicles_15min"]) - actual)
            rmse = lambda e: (sum(x * x for x in e) / len(e)) ** 0.5
            rows.append({"scats_site": site, "horizon_minutes": h, "sample_count": len(naive),
                         "last_value_rmse": rmse(naive), "regression_tree_rmse": rmse(tree),
                         "improvement_percent": (1 - rmse(tree) / rmse(naive)) * 100})
    return rows


def baseline_summary(rows: list[dict]) -> dict:
    out = {}
    for h in HORIZONS:
        sel = [r for r in rows if r["horizon_minutes"] == h]
        if sel:
            out[str(h)] = {"sites": len(sel),
                           "mean_last_value_rmse": statistics.fmean(r["last_value_rmse"] for r in sel),
                           "mean_regression_tree_rmse": statistics.fmean(r["regression_tree_rmse"] for r in sel),
                           "sites_where_tree_beats_last_value": sum(r["regression_tree_rmse"] < r["last_value_rmse"] for r in sel)}
    return out


# --------------------------------------------------------------------------
# charts for the report
# --------------------------------------------------------------------------
def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e6e5e1")
    ax.set_axisbelow(True)


def plot_importance(summary: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    x = list(range(1, LOOKBACK + 1))
    for h, colour in zip(HORIZONS, ("#1baf7a", "#2a78d6", "#6250d6")):
        if str(h) in summary:
            y = [summary[str(h)]["mean_importance"][f"lag_{i}"] * 100 for i in x]
            ax.plot(x, y, marker="o", color=colour, linewidth=2, label=f"{h} min ahead")
    ax.set_xticks(x, [f"{15 * i} min\nago" for i in x])
    ax.set_ylabel("Share of the tree's importance (%)")
    ax.set_title("Regression Tree: which past readings matter (mean of 40 sites)", loc="left")
    ax.legend(frameon=False)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_peak(rows: list[dict], path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    models = list(COLOURS)
    for ax, h in zip(axes, HORIZONS):
        for j, period in enumerate(("Peak", "Off-Peak")):
            for i, m in enumerate(models):
                r = next((r for r in rows if r["model"] == m and r["horizon_minutes"] == h and r["period"] == period), None)
                if r:
                    ax.bar(j * 4 + i, r["rmse"], color=COLOURS[m], width=0.9, label=m if (j == 0 and h == 15) else None)
        ax.set_xticks([1, 5], ["Peak", "Off-peak"])
        ax.set_title(f"{h} min ahead", loc="left")
        _style(ax)
    axes[0].set_ylabel("RMSE (vehicles / 15 min)")
    fig.legend(loc="upper right", frameon=False, ncol=3)
    fig.suptitle("Site 2000: peak (weekday 7-9 am, 4-6 pm) vs off-peak", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=180)
    plt.close(fig)


# --------------------------------------------------------------------------
def _write_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(output: Path = OUTPUT, models_dir: Path = MODELS_DIR) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    imp_rows = feature_importance(models_dir)
    size_rows = model_sizes(models_dir)
    peak_rows = three_model_peak_offpeak(rt_peak_offpeak(models_dir=models_dir))
    base_rows = persistence_baseline(models_dir)
    summary = {
        "feature_importance": importance_summary(imp_rows),
        "model_size_h15": size_summary(size_rows),
        "peak_offpeak_site_2000": peak_rows,
        "last_value_baseline": baseline_summary(base_rows),
        "notes": {
            "feature_importance": "sklearn feature_importances_, normalised to sum to 1 per site; lag_1 = most recent 15-minute reading",
            "learned_parameters": f"GRU/LSTM: one layer, hidden size {HIDDEN_SIZE}, input 1, plus Linear(hidden, 1). "
                                  "Regression Tree: a feature and a threshold per split plus a value per leaf.",
            "last_value_baseline": "Naive forecast = last observed reading, on the Regression Tree's own test windows (27-31 Oct).",
            "peak_offpeak": "Same periods and NRMSE as evaluation/analyze_peak_offpeak.py (Anson); GRU/LSTM rows copied from its CSV.",
        },
    }
    _write_csv(imp_rows, output / "feature_importance_by_site.csv")
    _write_csv(size_rows, output / "model_size_by_site.csv")
    _write_csv(peak_rows, output / "peak_offpeak_three_models.csv")
    _write_csv(base_rows, output / "last_value_baseline_by_site.csv")
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot_importance(summary["feature_importance"], output / "feature_importance.png")
    plot_peak(peak_rows, output / "peak_offpeak_three_models.png")
    return summary


if __name__ == "__main__":
    s = run()
    for h, v in s["feature_importance"].items():
        top = max(v["mean_importance"], key=v["mean_importance"].get)
        print(f"{h} min: most important = {top} ({v['mean_importance'][top]:.0%}); "
              f"lag_1 top at {v['sites_where_lag_1_is_top']}/{v['sites']} sites")
    for m, v in s["model_size_h15"].items():
        print(f"{m}: median {v['median_file_kb']} KB, {v['median_learned_parameters']} learned parameters")
    for h, v in s["last_value_baseline"].items():
        print(f"{h} min: last-value RMSE {v['mean_last_value_rmse']:.1f} vs tree {v['mean_regression_tree_rmse']:.1f}; "
              f"tree better at {v['sites_where_tree_beats_last_value']}/{v['sites']} sites")
    print(f"Saved to {OUTPUT}")
