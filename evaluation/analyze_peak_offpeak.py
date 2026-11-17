"""Compare GRU and LSTM during peak and off-peak periods."""

from __future__ import annotations

import csv
import json
from datetime import datetime, time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from evaluation.compare_lstm_gru import load_predictions
from evaluation.evaluate import evaluate_predictions


HORIZONS = [15, 30, 60]
MODELS = ["GRU", "LSTM"]


def is_peak_period(timestamp_text: str) -> bool:
    """Return whether a timestamp is within a weekday peak period."""

    timestamp = datetime.fromisoformat(timestamp_text)

    if timestamp.weekday() >= 5:
        return False

    current_time = timestamp.time()

    morning_peak = time(7, 0) <= current_time < time(9, 0)
    evening_peak = time(16, 0) <= current_time < time(18, 0)

    return morning_peak or evening_peak


def analyse_peak_offpeak(
    output_dir: str | Path = (
        "results/model_comparison/peak_offpeak"
    ),
) -> dict:
    """Evaluate GRU and LSTM for peak and off-peak observations."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    result_rows = []

    for horizon in HORIZONS:
        prediction_sets = {
            "GRU": load_predictions(
                (
                    "saved_models/"
                    f"gru_site_2000_h{horizon}_s8_"
                    "test_predictions.csv"
                ),
                timestamp_column="target_datetime",
                actual_column="actual_vehicles_15min",
                predicted_column="predicted_vehicles_15min",
            ),
            "LSTM": load_predictions(
                (
                    "results/lstm_site2000_comparison/"
                    f"site_2000_h{horizon}_s8/"
                    "predictions.csv"
                ),
                timestamp_column="datetime",
                actual_column="actual_traffic_volume",
                predicted_column="predicted_traffic_volume",
            ),
        }

        if (
            prediction_sets["GRU"]["timestamps"]
            != prediction_sets["LSTM"]["timestamps"]
        ):
            raise ValueError(
                f"Model timestamps differ at horizon {horizon}"
            )

        full_test_actual = prediction_sets["GRU"]["actual"]
        full_test_range = max(full_test_actual) - min(full_test_actual)
        if full_test_range <= 0.0:
            raise ValueError(
                f"Full test-set actual range is zero at horizon {horizon}"
            )

        for model_name, predictions in prediction_sets.items():
            for period_name, peak_required in (
                ("Peak", True),
                ("Off-Peak", False),
            ):
                selected_actual = []
                selected_predicted = []

                for timestamp, actual, predicted in zip(
                    predictions["timestamps"],
                    predictions["actual"],
                    predictions["predicted"],
                ):
                    if is_peak_period(timestamp) == peak_required:
                        selected_actual.append(actual)
                        selected_predicted.append(predicted)

                metrics = evaluate_predictions(
                    actual=selected_actual,
                    predicted=selected_predicted,
                    normalisation="range",
                )

                common_nrmse = metrics["rmse"] / full_test_range
                result_rows.append(
                    {
                        "model": model_name,
                        "scats_site": 2000,
                        "horizon_minutes": horizon,
                        "period": period_name,
                        "sample_count": metrics["count"],
                        "mae": metrics["mae"],
                        "rmse": metrics["rmse"],
                        "nrmse": common_nrmse,
                        "nrmse_percent": common_nrmse * 100.0,
                        "nrmse_normalisation": "full_test_set_range",
                        "nrmse_denominator": full_test_range,
                    }
                )

    csv_path = destination / "peak_offpeak_metrics.csv"
    json_path = destination / "peak_offpeak_summary.json"
    graph_path = destination / "peak_offpeak_rmse.png"

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=list(result_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(result_rows)

    json_path.write_text(
        json.dumps(
            {
                "definition": {
                    "weekday_only": True,
                    "morning_peak": "07:00-09:00",
                    "evening_peak": "16:00-18:00",
                    "off_peak": "All remaining observations",
                    "nrmse": "Subgroup RMSE divided by the actual-value range across all 480 test observations for the same horizon.",
                },
                "results": result_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    positions = list(range(len(HORIZONS)))
    width = 0.2

    figure, axis = plt.subplots(figsize=(10, 6))

    combinations = [
        ("GRU", "Peak"),
        ("GRU", "Off-Peak"),
        ("LSTM", "Peak"),
        ("LSTM", "Off-Peak"),
    ]

    offsets = [-1.5, -0.5, 0.5, 1.5]

    for offset, (model_name, period_name) in zip(
        offsets,
        combinations,
    ):
        values = [
            next(
                row["rmse"]
                for row in result_rows
                if row["model"] == model_name
                and row["period"] == period_name
                and row["horizon_minutes"] == horizon
            )
            for horizon in HORIZONS
        ]

        axis.bar(
            [
                position + offset * width
                for position in positions
            ],
            values,
            width,
            label=f"{model_name} {period_name}",
        )

    axis.set_title(
        "Peak vs Off-Peak Prediction RMSE — SCATS Site 2000"
    )
    axis.set_xlabel("Prediction Horizon")
    axis.set_ylabel("RMSE")
    axis.set_xticks(
        positions,
        [f"{horizon} min" for horizon in HORIZONS],
    )
    axis.grid(axis="y", alpha=0.3)
    axis.legend()

    figure.tight_layout()
    figure.savefig(graph_path, dpi=200)
    plt.close(figure)

    return {
        "metrics_csv": str(csv_path),
        "summary_json": str(json_path),
        "comparison_graph": str(graph_path),
    }


if __name__ == "__main__":
    result = analyse_peak_offpeak()
    print(json.dumps(result, indent=2))