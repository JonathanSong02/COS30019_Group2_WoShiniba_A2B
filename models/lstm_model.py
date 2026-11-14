"""PyTorch LSTM model for SCATS traffic-flow prediction."""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from data_processing.dataset import (
    apply_minmax,
    inverse_minmax,
    load_scaler,
    load_series,
    make_windows,
)


INTERVAL_MINUTES = 15
VALID_HORIZONS = {15, 30, 60}
VALID_SEQUENCE_LENGTHS = {4, 8}


@dataclass
class LSTMSettings:
    """Training and model settings for the LSTM predictor."""

    sequence_length: int = 8
    horizon_minutes: int = 15
    hidden_size: int = 32
    num_layers: int = 1
    dropout: float = 0.2
    epochs: int = 30
    batch_size: int = 64
    learning_rate: float = 0.001
    patience: int = 6
    seed: int = 42

    def validate(self) -> None:
        if self.sequence_length not in VALID_SEQUENCE_LENGTHS:
            raise ValueError("sequence_length must be 4 or 8")

        if self.horizon_minutes not in VALID_HORIZONS:
            raise ValueError("horizon_minutes must be 15, 30 or 60")

        if self.hidden_size < 1:
            raise ValueError("hidden_size must be positive")

        if self.num_layers < 1:
            raise ValueError("num_layers must be positive")

        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be between 0 and 1")
        
        if self.epochs < 1:
            raise ValueError("epochs must be positive")

        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")

        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")

        if self.patience < 1:
            raise ValueError("patience must be positive")

    @property
    def horizon_steps(self) -> int:
        """Convert minutes into 15-minute dataset steps."""
        return self.horizon_minutes // INTERVAL_MINUTES


class LSTMTrafficModel(nn.Module):
    """LSTM model that predicts one traffic-flow value."""

    def __init__(
        self,
        hidden_size: int = 32,
        num_layers: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        if hidden_size < 1:
            raise ValueError("hidden_size must be positive")

        if num_layers < 1:
            raise ValueError("num_layers must be positive")

        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be between 0 and 1")

        recurrent_dropout = dropout if num_layers > 1 else 0.0

        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=recurrent_dropout,
        )

        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return one prediction for each input sequence."""

        if x.ndim != 3:
            raise ValueError(
                "LSTM input must have shape "
                "[number_of_samples, sequence_length, 1]"
            )

        if x.shape[2] != 1:
            raise ValueError("LSTM input must contain exactly one feature")

        states, _ = self.lstm(x)
        final_state = states[:, -1, :]
        final_state = self.dropout(final_state)

        return self.output(final_state).squeeze(-1)
    
def prepare_lstm_data(
    processed_csv: str | Path,
    scaler_path: str | Path,
    scats_site: int,
    settings: LSTMSettings,
):
    """Load identical train, validation and test windows for one SCATS site."""

    settings.validate()

    dataframe = load_series(processed_csv)
    scaler = load_scaler(scaler_path)

    site = int(scats_site)
    prepared = {}

    for split in ("train", "val", "test"):
        x_values, y_values, metadata = make_windows(
            dataframe,
            split=split,
            scaler=scaler,
            lags=settings.sequence_length,
            horizon=settings.horizon_steps,
            sites={site},
        )

        if len(x_values) == 0:
            raise ValueError(
                f"No usable {split} samples were found for SCATS site {site}"
            )

        # make_windows returns [samples, sequence_length].
        # PyTorch LSTM requires [samples, sequence_length, features].
        x_tensor = torch.tensor(
            x_values,
            dtype=torch.float32,
        ).unsqueeze(-1)

        y_tensor = torch.tensor(
            y_values,
            dtype=torch.float32,
        )

        prepared[split] = {
            "x": x_tensor,
            "y": y_tensor,
            "metadata": metadata,
        }

    return prepared, scaler

def train_lstm(
    processed_csv: str | Path,
    scaler_path: str | Path,
    output_folder: str | Path,
    scats_site: int,
    settings: LSTMSettings,
):
    """Train one site-specific and horizon-specific LSTM model."""

    settings.validate()

    torch.manual_seed(settings.seed)

    prepared, scaler = prepare_lstm_data(
        processed_csv=processed_csv,
        scaler_path=scaler_path,
        scats_site=scats_site,
        settings=settings,
    )

    train_x = prepared["train"]["x"]
    train_y = prepared["train"]["y"]
    validation_x = prepared["val"]["x"]
    validation_y = prepared["val"]["y"]

    training_dataset = TensorDataset(train_x, train_y)
    training_loader = DataLoader(
        training_dataset,
        batch_size=settings.batch_size,
        shuffle=True,
    )

    model = LSTMTrafficModel(
        hidden_size=settings.hidden_size,
        num_layers=settings.num_layers,
        dropout=settings.dropout,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=settings.learning_rate,
    )

    loss_function = nn.MSELoss()

    best_validation_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0
    loss_history = []

    training_start = time.perf_counter()

    for epoch in range(1, settings.epochs + 1):
        model.train()

        total_training_loss = 0.0

        for batch_x, batch_y in training_loader:
            optimizer.zero_grad()

            predictions = model(batch_x)
            loss = loss_function(predictions, batch_y)

            loss.backward()
            optimizer.step()

            total_training_loss += loss.item() * len(batch_x)

        average_training_loss = (
            total_training_loss / len(training_dataset)
        )

        model.eval()

        with torch.no_grad():
            validation_predictions = model(validation_x)
            validation_loss = loss_function(
                validation_predictions,
                validation_y,
            ).item()

        loss_history.append(
            {
                "epoch": epoch,
                "training_mse_scaled": average_training_loss,
                "validation_mse_scaled": validation_loss,
            }
        )

        print(
            f"Epoch {epoch:03d} | "
            f"training loss: {average_training_loss:.6f} | "
            f"validation loss: {validation_loss:.6f}"
        )

        if validation_loss < best_validation_loss - 1e-8:
            best_validation_loss = validation_loss
            epochs_without_improvement = 0

            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= settings.patience:
            print("Early stopping activated.")
            break

    training_time_seconds = time.perf_counter() - training_start

    if best_state is None:
        raise RuntimeError("LSTM training did not produce a valid model")

    model.load_state_dict(best_state)
    model.eval()

    output_directory = Path(output_folder)
    output_directory.mkdir(parents=True, exist_ok=True)

    model_name = (
        f"lstm_site_{int(scats_site)}_"
        f"h{settings.horizon_minutes}_"
        f"s{settings.sequence_length}"
    )

    weights_path = output_directory / f"{model_name}.pt"
    metadata_path = output_directory / f"{model_name}.json"

    torch.save(best_state, weights_path)

    metadata = {
        "model_type": "LSTM",
        "scats_site": int(scats_site),
        "settings": asdict(settings),
        "scaler": scaler,
        "training_samples": len(prepared["train"]["x"]),
        "validation_samples": len(prepared["val"]["x"]),
        "test_samples": len(prepared["test"]["x"]),
        "best_validation_mse_scaled": best_validation_loss,
        "training_time_seconds": training_time_seconds,
        "loss_history": loss_history,
        "output_units": "vehicles per 15-minute interval",
        "prediction_semantics": "as_of_time + horizon_minutes",
    }

    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    return {
        "model": model,
        "weights_path": str(weights_path),
        "metadata_path": str(metadata_path),
        "training_time_seconds": training_time_seconds,
        "best_validation_mse_scaled": best_validation_loss,
        "loss_history": loss_history,
    }
    
def load_lstm(
    weights_path: str | Path,
    metadata_path: str | Path,
):
    """Load a saved LSTM model and its metadata."""

    metadata = json.loads(
        Path(metadata_path).read_text(encoding="utf-8")
    )

    settings = LSTMSettings(**metadata["settings"])
    settings.validate()

    model = LSTMTrafficModel(
        hidden_size=settings.hidden_size,
        num_layers=settings.num_layers,
        dropout=settings.dropout,
    )

    state = torch.load(
        weights_path,
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(state)
    model.eval()

    return model, metadata

def predict_lstm(
    scats_site: int,
    as_of_time: str | datetime,
    horizon_minutes: int,
    processed_csv: str | Path,
    scaler_path: str | Path,
    model_dir: str | Path = "saved_models",
    sequence_length: int = 8,
) -> float:
    """Predict traffic flow at as_of_time plus the selected horizon."""

    settings = LSTMSettings(
        sequence_length=sequence_length,
        horizon_minutes=horizon_minutes,
    )
    settings.validate()

    site = int(scats_site)

    if isinstance(as_of_time, str):
        prediction_origin = datetime.fromisoformat(as_of_time)
    elif isinstance(as_of_time, datetime):
        prediction_origin = as_of_time
    else:
        raise TypeError("as_of_time must be an ISO datetime string or datetime")

    if prediction_origin.tzinfo is not None:
        raise ValueError("as_of_time must not contain a timezone offset")

    if (
        prediction_origin.minute % INTERVAL_MINUTES != 0
        or prediction_origin.second != 0
        or prediction_origin.microsecond != 0
    ):
        raise ValueError("as_of_time must be aligned to a 15-minute interval")

    model_name = (
        f"lstm_site_{site}_"
        f"h{horizon_minutes}_"
        f"s{sequence_length}"
    )

    model_directory = Path(model_dir)
    weights_path = model_directory / f"{model_name}.pt"
    metadata_path = model_directory / f"{model_name}.json"

    if not weights_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            f"No trained LSTM model exists for site {site}, "
            f"horizon {horizon_minutes} and sequence length "
            f"{sequence_length}"
        )

    model, metadata = load_lstm(
        weights_path=weights_path,
        metadata_path=metadata_path,
    )
    
    if metadata["scats_site"] != site:
        raise ValueError("Saved model SCATS site does not match")

    saved_settings = metadata["settings"]

    if saved_settings["horizon_minutes"] != horizon_minutes:
        raise ValueError("Saved model horizon does not match the request")

    if saved_settings["sequence_length"] != sequence_length:
        raise ValueError("Saved model sequence length does not match")

    dataframe = load_series(processed_csv)
    site_data = dataframe[dataframe["scats_site"] == site]

    if site_data.empty:
        raise ValueError(f"No traffic records exist for SCATS site {site}")

    observations = {
        timestamp.to_pydatetime(): float(value)
        for timestamp, value in zip(
            site_data["datetime"],
            site_data["traffic_volume_15min"],
        )
    }

    required_times = [
        prediction_origin
        - timedelta(
            minutes=INTERVAL_MINUTES
            * (sequence_length - 1 - index)
        )
        for index in range(sequence_length)
    ]

    historical_values = []

    for timestamp in required_times:
        if timestamp not in observations:
            raise ValueError(
                f"Missing traffic observation at {timestamp.isoformat()}"
            )

        value = observations[timestamp]

        if not math.isfinite(value):
            raise ValueError(
                f"Invalid traffic observation at {timestamp.isoformat()}"
            )

        historical_values.append(value)

    scaler = load_scaler(scaler_path)
    scaled_values = apply_minmax(historical_values, scaler)

    model_input = torch.tensor(
        scaled_values,
        dtype=torch.float32,
    ).reshape(1, sequence_length, 1)

    prediction_start = time.perf_counter()

    model.eval()

    with torch.no_grad():
        scaled_prediction = model(model_input).item()

    prediction_time_seconds = (
        time.perf_counter() - prediction_start
    )

    prediction = float(
        inverse_minmax([scaled_prediction], scaler)[0]
    )
    
    if not math.isfinite(prediction):
        raise RuntimeError("LSTM produced a non-finite prediction")

    # Traffic volume cannot be negative.
    prediction = max(0.0, prediction)

    print(
        f"LSTM prediction completed in "
        f"{prediction_time_seconds:.6f} seconds"
    )

    return prediction

def predict_lstm_split(
    scats_site: int,
    horizon_minutes: int,
    processed_csv: str | Path,
    scaler_path: str | Path,
    model_dir: str | Path = "saved_models",
    sequence_length: int = 8,
    split: str = "test",
):
    """Predict every sample in a dataset split."""

    if split not in {"train", "val", "test"}:
        raise ValueError("split must be train, val or test")

    settings = LSTMSettings(
        sequence_length=sequence_length,
        horizon_minutes=horizon_minutes,
    )
    settings.validate()

    site = int(scats_site)

    model_name = (
        f"lstm_site_{site}_"
        f"h{horizon_minutes}_"
        f"s{sequence_length}"
    )

    model_directory = Path(model_dir)
    weights_path = model_directory / f"{model_name}.pt"
    metadata_path = model_directory / f"{model_name}.json"

    if not weights_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            f"No saved model exists for {model_name}"
        )

    model, model_metadata = load_lstm(
        weights_path=weights_path,
        metadata_path=metadata_path,
    )
    
    saved_settings = model_metadata["settings"]

    if model_metadata["scats_site"] != site:
        raise ValueError("Saved model SCATS site does not match")

    if saved_settings["horizon_minutes"] != horizon_minutes:
        raise ValueError("Saved model horizon does not match")

    if saved_settings["sequence_length"] != sequence_length:
        raise ValueError("Saved model sequence length does not match")

    prepared, scaler = prepare_lstm_data(
        processed_csv=processed_csv,
        scaler_path=scaler_path,
        scats_site=site,
        settings=settings,
    )

    inputs = prepared[split]["x"]
    targets = prepared[split]["y"]

    prediction_start = time.perf_counter()

    model.eval()

    with torch.no_grad():
        scaled_predictions = model(inputs).cpu().numpy()

    prediction_time_seconds = (
        time.perf_counter() - prediction_start
    )

    actual_values = inverse_minmax(
        targets.cpu().numpy(),
        scaler,
    )

    predicted_values = inverse_minmax(
        scaled_predictions,
        scaler,
    )
    
    if not all(
        math.isfinite(float(value))
        for value in predicted_values
    ):
        raise RuntimeError("LSTM produced non-finite predictions")

    # Traffic volume cannot be negative.
    predicted_values = predicted_values.clip(min=0.0)

    return {
        "actual": actual_values,
        "predicted": predicted_values,
        "metadata": prepared[split]["metadata"],
        "model_metadata": model_metadata,
        "prediction_time_seconds": prediction_time_seconds,
        "prediction_time_per_sample_seconds": (
            prediction_time_seconds / len(predicted_values)
        ),
        "sample_count": len(predicted_values),
        "split": split,
    }