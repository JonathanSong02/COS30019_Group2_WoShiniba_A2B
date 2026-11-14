"""Run LSTM sequence-length and prediction-horizon experiments."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from evaluation.run_lstm_evaluation import evaluate_saved_lstm
from models.lstm_model import LSTMSettings, train_lstm


DEFAULT_SITES = [2825, 3812, 4043]
DEFAULT_SEQUENCE_LENGTHS = [4, 8]
DEFAULT_HORIZONS = [15, 30, 60]


def run_lstm_experiments(
    sites,
    sequence_lengths,
    horizons,
    epochs=30,
    model_dir="saved_models",
    results_dir="results/lstm",
):
    """Train and evaluate every requested LSTM configuration."""

    all_results = []

    for site in sites:
        for sequence_length in sequence_lengths:
            for horizon_minutes in horizons:
                experiment_name = (
                    f"site_{site}_h{horizon_minutes}_"
                    f"s{sequence_length}"
                )

                print()
                print("=" * 70)
                print(f"Starting {experiment_name}")
                print("=" * 70)

                settings = LSTMSettings(
                    sequence_length=sequence_length,
                    horizon_minutes=horizon_minutes,
                    hidden_size=32,
                    num_layers=1,
                    dropout=0.2,
                    epochs=epochs,
                    batch_size=64,
                    learning_rate=0.001,
                    patience=6,
                    seed=42,
                )

                training_result = train_lstm(
                    processed_csv="data/processed/scats_15min.csv",
                    scaler_path="data/processed/scaler.json",
                    output_folder=model_dir,
                    scats_site=site,
                    settings=settings,
                )

                evaluation_result = evaluate_saved_lstm(
                    site=site,
                    horizon_minutes=horizon_minutes,
                    sequence_length=sequence_length,
                    processed_csv="data/processed/scats_15min.csv",
                    scaler_path="data/processed/scaler.json",
                    model_dir=model_dir,
                    output_dir=results_dir,
                )

                metrics = evaluation_result["metrics"]

                result_row = {
                    "model": "LSTM",
                    "scats_site": site,
                    "sequence_length": sequence_length,
                    "horizon_minutes": horizon_minutes,
                    "sample_count": metrics["count"],
                    "mae": metrics["mae"],
                    "rmse": metrics["rmse"],
                    "nrmse": metrics["nrmse"],
                    "nrmse_normalisation": metrics[
                        "nrmse_normalisation"
                    ],
                    "training_time_seconds": training_result[
                        "training_time_seconds"
                    ],
                    "prediction_time_seconds": metrics[
                        "prediction_time_seconds"
                    ],
                    "prediction_time_per_sample_seconds": metrics[
                        "prediction_time_per_sample_seconds"
                    ],
                    "best_validation_mse_scaled": training_result[
                        "best_validation_mse_scaled"
                    ],
                }

                all_results.append(result_row)

                print(json.dumps(result_row, indent=2))

    results_directory = Path(results_dir)
    results_directory.mkdir(parents=True, exist_ok=True)

    summary_path = (
        results_directory / "lstm_experiment_summary.csv"
    )

    if all_results:
        with summary_path.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as output_file:
            writer = csv.DictWriter(
                output_file,
                fieldnames=list(all_results[0].keys()),
            )
            writer.writeheader()
            writer.writerows(all_results)

    print()
    print(f"Experiment summary saved to {summary_path}")

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Run COS30019 LSTM experiments"
    )

    parser.add_argument(
        "--sites",
        nargs="+",
        type=int,
        default=DEFAULT_SITES,
    )

    parser.add_argument(
        "--sequence-lengths",
        nargs="+",
        type=int,
        choices=[4, 8],
        default=DEFAULT_SEQUENCE_LENGTHS,
    )

    parser.add_argument(
        "--horizons",
        nargs="+",
        type=int,
        choices=[15, 30, 60],
        default=DEFAULT_HORIZONS,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--models-dir",
        default="saved_models",
    )

    parser.add_argument(
        "--results-dir",
        default="results/lstm",
    )

    arguments = parser.parse_args()

    run_lstm_experiments(
        sites=arguments.sites,
        sequence_lengths=arguments.sequence_lengths,
        horizons=arguments.horizons,
        epochs=arguments.epochs,
        model_dir=arguments.models_dir,
        results_dir=arguments.results_dir,
    )


if __name__ == "__main__":
    main()