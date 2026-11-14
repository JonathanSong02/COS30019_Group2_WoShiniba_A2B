"""Evaluate a saved LSTM model and generate report evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluation.evaluate import (
    evaluate_predictions,
    plot_actual_vs_predicted,
    plot_loss_history,
)
from models.lstm_model import predict_lstm_split


def evaluate_saved_lstm(
    site: int,
    horizon_minutes: int,
    sequence_length: int,
    processed_csv: str | Path = "data/processed/scats_15min.csv",
    scaler_path: str | Path = "data/processed/scaler.json",
    model_dir: str | Path = "saved_models",
    output_dir: str | Path = "results/lstm",
):
    """Evaluate one saved LSTM configuration on the test split."""

    prediction_result = predict_lstm_split(
        scats_site=site,
        horizon_minutes=horizon_minutes,
        processed_csv=processed_csv,
        scaler_path=scaler_path,
        model_dir=model_dir,
        sequence_length=sequence_length,
        split="test",
    )

    metrics = evaluate_predictions(
        actual=prediction_result["actual"],
        predicted=prediction_result["predicted"],
        normalisation="range",
    )

    model_metadata = prediction_result["model_metadata"]

    metrics.update(
        {
            "model": "LSTM",
            "scats_site": site,
            "horizon_minutes": horizon_minutes,
            "sequence_length": sequence_length,
            "training_time_seconds": model_metadata[
                "training_time_seconds"
            ],
            "prediction_time_seconds": prediction_result[
                "prediction_time_seconds"
            ],
            "prediction_time_per_sample_seconds": prediction_result[
                "prediction_time_per_sample_seconds"
            ],
        }
    )

    experiment_name = (
        f"site_{site}_h{horizon_minutes}_s{sequence_length}"
    )

    destination = Path(output_dir) / experiment_name
    destination.mkdir(parents=True, exist_ok=True)

    metrics_path = destination / "metrics.json"
    prediction_plot_path = destination / "actual_vs_predicted.png"
    loss_plot_path = destination / "training_validation_loss.png"

    metrics_path.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    timestamps = prediction_result["metadata"]["datetime"]

    plot_actual_vs_predicted(
        actual=prediction_result["actual"],
        predicted=prediction_result["predicted"],
        timestamps=timestamps,
        output_path=prediction_plot_path,
        title=(
            f"LSTM Actual vs Predicted — Site {site}, "
            f"{horizon_minutes}-Minute Horizon, "
            f"Sequence {sequence_length}"
        ),
    )

    plot_loss_history(
        loss_history=model_metadata["loss_history"],
        output_path=loss_plot_path,
        title=(
            f"LSTM Loss — Site {site}, "
            f"{horizon_minutes}-Minute Horizon, "
            f"Sequence {sequence_length}"
        ),
    )

    return {
        "metrics": metrics,
        "metrics_path": str(metrics_path),
        "prediction_plot_path": str(prediction_plot_path),
        "loss_plot_path": str(loss_plot_path),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a saved COS30019 LSTM model"
    )

    parser.add_argument("--site", type=int, required=True)
    parser.add_argument(
        "--horizon",
        type=int,
        choices=[15, 30, 60],
        required=True,
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        choices=[4, 8],
        required=True,
    )
    parser.add_argument(
        "--models-dir",
        default="saved_models",
    )
    parser.add_argument(
        "--output-dir",
        default="results/lstm",
    )

    arguments = parser.parse_args()

    result = evaluate_saved_lstm(
        site=arguments.site,
        horizon_minutes=arguments.horizon,
        sequence_length=arguments.sequence_length,
        model_dir=arguments.models_dir,
        output_dir=arguments.output_dir,
    )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()