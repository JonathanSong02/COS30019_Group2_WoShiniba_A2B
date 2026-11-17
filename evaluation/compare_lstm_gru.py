"""Compare LSTM and GRU using identical SCATS test observations."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from evaluation.evaluate import evaluate_predictions


HORIZONS = [15, 30, 60]
SITE = 2000
SEQUENCE_LENGTH = 8


def load_predictions(
    csv_path: str | Path,
    *,
    timestamp_column: str,
    actual_column: str,
    predicted_column: str,
) -> dict:
    """Load timestamps, actual values and predictions from a CSV."""

    with Path(csv_path).open(
        "r",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        rows = list(csv.DictReader(csv_file))

    if not rows:
        raise ValueError(f"No prediction rows found in {csv_path}")

    return {
        "timestamps": [
            row[timestamp_column] for row in rows
        ],
        "actual": [
            float(row[actual_column]) for row in rows
        ],
        "predicted": [
            float(row[predicted_column]) for row in rows
        ],
    }


def compare_lstm_gru(
    saved_models_dir: str | Path = "saved_models",
    lstm_results_dir: str | Path = (
        "results/lstm_site2000_comparison"
    ),
    output_dir: str | Path = (
        "results/model_comparison/site_2000_lstm_vs_gru"
    ),
) -> dict:
    """Compare LSTM and GRU fairly across three horizons."""

    saved_models_dir = Path(saved_models_dir)
    lstm_results_dir = Path(lstm_results_dir)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    comparison_rows = []
    horizon_summary = {}

    for horizon in HORIZONS:
        gru_path = saved_models_dir / (
            f"gru_site_{SITE}_h{horizon}_s8_test_predictions.csv"
        )

        lstm_path = (
            lstm_results_dir
            / f"site_{SITE}_h{horizon}_s8"
            / "predictions.csv"
        )

        gru = load_predictions(
            gru_path,
            timestamp_column="target_datetime",
            actual_column="actual_vehicles_15min",
            predicted_column="predicted_vehicles_15min",
        )

        lstm = load_predictions(
            lstm_path,
            timestamp_column="datetime",
            actual_column="actual_traffic_volume",
            predicted_column="predicted_traffic_volume",
        )

        if gru["timestamps"] != lstm["timestamps"]:
            raise ValueError(
                f"GRU and LSTM timestamps differ for horizon {horizon}"
            )

        if len(gru["actual"]) != len(lstm["actual"]):
            raise ValueError(
                f"GRU and LSTM sample counts differ for horizon {horizon}"
            )

        actual_mismatches = sum(
            abs(gru_value - lstm_value) > 0.000001
            for gru_value, lstm_value in zip(
                gru["actual"],
                lstm["actual"],
            )
        )

        if actual_mismatches:
            raise ValueError(
                f"GRU and LSTM actual values differ for "
                f"horizon {horizon}"
            )

        gru_metrics = evaluate_predictions(
            actual=gru["actual"],
            predicted=gru["predicted"],
            normalisation="range",
        )

        lstm_metrics = evaluate_predictions(
            actual=lstm["actual"],
            predicted=lstm["predicted"],
            normalisation="range",
        )

        for model_name, metrics in (
            ("GRU", gru_metrics),
            ("LSTM", lstm_metrics),
        ):
            comparison_rows.append(
                {
                    "model": model_name,
                    "scats_site": SITE,
                    "sequence_length": SEQUENCE_LENGTH,
                    "horizon_minutes": horizon,
                    "sample_count": metrics["count"],
                    "mae": metrics["mae"],
                    "rmse": metrics["rmse"],
                    "nrmse": metrics["nrmse"],
                    "nrmse_percent": metrics["nrmse"] * 100.0,
                }
            )

        best_model = (
            "LSTM"
            if lstm_metrics["rmse"] < gru_metrics["rmse"]
            else "GRU"
        )

        rmse_difference_percent = (
            (
                lstm_metrics["rmse"]
                - gru_metrics["rmse"]
            )
            / gru_metrics["rmse"]
            * 100.0
        )

        horizon_summary[str(horizon)] = {
            "best_model_by_rmse": best_model,
            "gru_rmse": gru_metrics["rmse"],
            "lstm_rmse": lstm_metrics["rmse"],
            "lstm_rmse_difference_vs_gru_percent": (
                rmse_difference_percent
            ),
        }

    comparison_csv_path = destination / "model_comparison.csv"
    comparison_json_path = destination / "comparison_summary.json"
    comparison_plot_path = destination / "model_comparison.png"

    with comparison_csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        fieldnames = list(comparison_rows[0].keys())
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(comparison_rows)

    comparison_json_path.write_text(
        json.dumps(
            {
                "site": SITE,
                "sequence_length": SEQUENCE_LENGTH,
                "test_samples_per_configuration": 480,
                "horizons": horizon_summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    def values_for(model: str, metric: str) -> list[float]:
        values = []

        for horizon in HORIZONS:
            matching_row = next(
                row
                for row in comparison_rows
                if row["model"] == model
                and row["horizon_minutes"] == horizon
            )
            values.append(float(matching_row[metric]))

        return values

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5),
    )

    graph_metrics = [
        ("mae", "MAE"),
        ("rmse", "RMSE"),
        ("nrmse_percent", "NRMSE (%)"),
    ]

    positions = list(range(len(HORIZONS)))
    width = 0.36

    for axis, (metric, title) in zip(axes, graph_metrics):
        gru_values = values_for("GRU", metric)
        lstm_values = values_for("LSTM", metric)

        gru_bars = axis.bar(
            [position - width / 2 for position in positions],
            gru_values,
            width,
            label="GRU",
        )

        lstm_bars = axis.bar(
            [position + width / 2 for position in positions],
            lstm_values,
            width,
            label="LSTM",
        )

        axis.set_title(title)
        axis.set_xlabel("Prediction Horizon")
        axis.set_xticks(
            positions,
            [f"{horizon} min" for horizon in HORIZONS],
        )
        axis.grid(axis="y", alpha=0.3)
        axis.bar_label(
            gru_bars,
            fmt="%.2f",
            padding=3,
            fontsize=8,
        )
        axis.bar_label(
            lstm_bars,
            fmt="%.2f",
            padding=3,
            fontsize=8,
        )

    axes[0].set_ylabel("Prediction Error")
    axes[0].legend()

    figure.suptitle(
        "GRU vs LSTM — SCATS Site 2000, Sequence Length 8"
    )
    figure.tight_layout()
    figure.savefig(comparison_plot_path, dpi=200)
    plt.close(figure)

    return {
        "comparison_csv": str(comparison_csv_path),
        "comparison_summary": str(comparison_json_path),
        "comparison_plot": str(comparison_plot_path),
    }


if __name__ == "__main__":
    result = compare_lstm_gru()
    print(json.dumps(result, indent=2))