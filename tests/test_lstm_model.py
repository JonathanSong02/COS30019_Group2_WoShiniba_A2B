"""Automated tests for the PyTorch LSTM traffic model."""

import math
import os
import tempfile
import unittest
from pathlib import Path

import torch

from models.lstm_model import (
    LSTMSettings,
    LSTMTrafficModel,
    load_lstm,
    predict_lstm,
    prepare_lstm_data,
    train_lstm,
)


PROCESSED_CSV = Path("data/processed/scats_15min.csv")
SCALER_PATH = Path("data/processed/scaler.json")
TEST_SITE = 970


class LSTMArchitectureTests(unittest.TestCase):
    def test_prediction_output_shape(self):
        model = LSTMTrafficModel(
            hidden_size=8,
            num_layers=1,
            dropout=0.0,
        )

        model_input = torch.rand(5, 8, 1)
        output = model(model_input)

        self.assertEqual(output.shape, torch.Size([5]))

    def test_invalid_settings_are_rejected(self):
        invalid_sequence = LSTMSettings(sequence_length=3)
        invalid_horizon = LSTMSettings(horizon_minutes=45)

        with self.assertRaises(ValueError):
            invalid_sequence.validate()

        with self.assertRaises(ValueError):
            invalid_horizon.validate()


@unittest.skipUnless(
    PROCESSED_CSV.exists() and SCALER_PATH.exists(),
    "Processed SCATS data is required",
)
class LSTMDataAndTrainingTests(unittest.TestCase):
    def test_shared_dataset_shapes_and_horizon(self):
        settings = LSTMSettings(
            sequence_length=4,
            horizon_minutes=60,
        )

        prepared, scaler = prepare_lstm_data(
            processed_csv=PROCESSED_CSV,
            scaler_path=SCALER_PATH,
            scats_site=TEST_SITE,
            settings=settings,
        )

        self.assertEqual(settings.horizon_steps, 4)
        self.assertEqual(prepared["train"]["x"].shape[1:], (4, 1))
        self.assertEqual(prepared["val"]["x"].shape[1:], (4, 1))
        self.assertEqual(prepared["test"]["x"].shape[1:], (4, 1))
        self.assertEqual(scaler["fitted_on"], "train")

    def test_training_loading_and_prediction(self):
        settings = LSTMSettings(
            sequence_length=8,
            horizon_minutes=15,
            hidden_size=4,
            num_layers=1,
            dropout=0.0,
            epochs=1,
            batch_size=128,
            learning_rate=0.001,
            patience=1,
            seed=42,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            result = train_lstm(
                processed_csv=PROCESSED_CSV,
                scaler_path=SCALER_PATH,
                output_folder=temporary_directory,
                scats_site=TEST_SITE,
                settings=settings,
            )

            weights_path = Path(result["weights_path"])
            metadata_path = Path(result["metadata_path"])

            self.assertTrue(weights_path.exists())
            self.assertTrue(metadata_path.exists())

            loaded_model, metadata = load_lstm(
                weights_path=weights_path,
                metadata_path=metadata_path,
            )

            self.assertIsInstance(loaded_model, LSTMTrafficModel)
            self.assertEqual(metadata["model_type"], "LSTM")
            self.assertEqual(metadata["scats_site"], TEST_SITE)

            prediction = predict_lstm(
                scats_site=TEST_SITE,
                as_of_time="2006-10-27T12:00:00",
                horizon_minutes=15,
                processed_csv=PROCESSED_CSV,
                scaler_path=SCALER_PATH,
                model_dir=temporary_directory,
                sequence_length=8,
            )

            self.assertTrue(math.isfinite(prediction))
            self.assertGreaterEqual(prediction, 0.0)


if __name__ == "__main__":
    unittest.main()