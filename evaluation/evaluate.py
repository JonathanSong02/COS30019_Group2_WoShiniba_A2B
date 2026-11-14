"""Reusable evaluation tools for the Assignment 2B ML models."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np


def _validated_arrays(
    actual: Sequence[float],
    predicted: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Convert and validate actual and predicted values."""

    actual_array = np.asarray(actual, dtype=float).reshape(-1)
    predicted_array = np.asarray(predicted, dtype=float).reshape(-1)

    if len(actual_array) == 0:
        raise ValueError("actual and predicted values cannot be empty")

    if len(actual_array) != len(predicted_array):
        raise ValueError(
            "actual and predicted values must have the same length"
        )

    if not np.isfinite(actual_array).all():
        raise ValueError("actual values must all be finite")

    if not np.isfinite(predicted_array).all():
        raise ValueError("predicted values must all be finite")

    return actual_array, predicted_array


def calculate_mae(
    actual: Sequence[float],
    predicted: Sequence[float],
) -> float:
    """Calculate mean absolute error."""

    actual_array, predicted_array = _validated_arrays(
        actual,
        predicted,
    )

    return float(
        np.mean(np.abs(actual_array - predicted_array))
    )


def calculate_rmse(
    actual: Sequence[float],
    predicted: Sequence[float],
) -> float:
    """Calculate root mean squared error."""

    actual_array, predicted_array = _validated_arrays(
        actual,
        predicted,
    )

    squared_errors = np.square(actual_array - predicted_array)

    return float(math.sqrt(np.mean(squared_errors)))


def calculate_nrmse(
    actual: Sequence[float],
    predicted: Sequence[float],
    normalisation: str = "range",
) -> float:
    """Calculate NRMSE using the actual-value range or mean."""

    actual_array, predicted_array = _validated_arrays(
        actual,
        predicted,
    )

    rmse = calculate_rmse(actual_array, predicted_array)

    if normalisation == "range":
        denominator = float(
            np.max(actual_array) - np.min(actual_array)
        )
    elif normalisation == "mean":
        denominator = abs(float(np.mean(actual_array)))
    else:
        raise ValueError(
            "normalisation must be either 'range' or 'mean'"
        )

    if denominator == 0.0:
        raise ValueError(
            "NRMSE cannot be calculated with a zero denominator"
        )

    return rmse / denominator


def evaluate_predictions(
    actual: Sequence[float],
    predicted: Sequence[float],
    normalisation: str = "range",
) -> dict[str, float | int | str]:
    """Calculate the required Assignment 2B prediction metrics."""

    actual_array, predicted_array = _validated_arrays(
        actual,
        predicted,
    )

    return {
        "count": len(actual_array),
        "mae": calculate_mae(actual_array, predicted_array),
        "rmse": calculate_rmse(actual_array, predicted_array),
        "nrmse": calculate_nrmse(
            actual_array,
            predicted_array,
            normalisation=normalisation,
        ),
        "nrmse_normalisation": normalisation,
    }


def time_prediction(
    prediction_function: Callable[..., Any],
    *args,
    **kwargs,
) -> tuple[Any, float]:
    """Run a prediction function and measure execution time."""

    start_time = time.perf_counter()

    result = prediction_function(*args, **kwargs)

    elapsed_seconds = time.perf_counter() - start_time

    return result, elapsed_seconds


def plot_actual_vs_predicted(
    actual: Sequence[float],
    predicted: Sequence[float],
    output_path: str | Path,
    title: str = "Actual vs Predicted Traffic Flow",
    timestamps: Sequence[Any] | None = None,
) -> str:
    """Save an actual-versus-predicted traffic graph."""

    import matplotlib.pyplot as plt

    actual_array, predicted_array = _validated_arrays(
        actual,
        predicted,
    )

    if timestamps is None:
        x_values = np.arange(len(actual_array))
        x_label = "Sample"
    else:
        if len(timestamps) != len(actual_array):
            raise ValueError(
                "timestamps must have the same length as actual values"
            )

        x_values = timestamps
        x_label = "Time"

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(12, 5))

    axis.plot(
        x_values,
        actual_array,
        label="Actual traffic",
        linewidth=1.8,
    )

    axis.plot(
        x_values,
        predicted_array,
        label="Predicted traffic",
        linewidth=1.5,
    )

    axis.set_title(title)
    axis.set_xlabel(x_label)
    axis.set_ylabel("Vehicles per 15-minute interval")
    axis.grid(True, alpha=0.3)
    axis.legend()

    figure.tight_layout()
    figure.savefig(destination, dpi=150)
    plt.close(figure)

    return str(destination)

def plot_loss_history(
    loss_history: Sequence[dict[str, float | int]],
    output_path: str | Path,
    title: str = "LSTM Training and Validation Loss",
) -> str:
    """Save training and validation loss across epochs."""

    import matplotlib.pyplot as plt

    if not loss_history:
        raise ValueError("loss_history cannot be empty")

    epochs = []
    training_losses = []
    validation_losses = []

    for entry in loss_history:
        required = {
            "epoch",
            "training_mse_scaled",
            "validation_mse_scaled",
        }

        if not required.issubset(entry):
            raise ValueError(
                "Each loss entry requires epoch, "
                "training_mse_scaled and validation_mse_scaled"
            )

        epochs.append(int(entry["epoch"]))
        training_losses.append(
            float(entry["training_mse_scaled"])
        )
        validation_losses.append(
            float(entry["validation_mse_scaled"])
        )

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(9, 5))

    axis.plot(
        epochs,
        training_losses,
        marker="o",
        label="Training loss",
    )

    axis.plot(
        epochs,
        validation_losses,
        marker="o",
        label="Validation loss",
    )

    axis.set_title(title)
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Mean squared error (scaled)")
    axis.grid(True, alpha=0.3)
    axis.legend()

    figure.tight_layout()
    figure.savefig(destination, dpi=150)
    plt.close(figure)

    return str(destination)