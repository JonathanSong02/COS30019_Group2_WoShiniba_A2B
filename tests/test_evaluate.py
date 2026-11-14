"""Tests for the shared ML evaluation functions."""

import math
import time
import unittest
import tempfile
from pathlib import Path

from evaluation.evaluate import (
    calculate_mae,
    calculate_nrmse,
    calculate_rmse,
    evaluate_predictions,
    plot_actual_vs_predicted,
    plot_loss_history,
    time_prediction,
)


class EvaluationMetricTests(unittest.TestCase):
    def test_mae_and_rmse(self):
        actual = [1.0, 2.0, 3.0]
        predicted = [1.0, 3.0, 2.0]

        self.assertAlmostEqual(
            calculate_mae(actual, predicted),
            2.0 / 3.0,
        )

        self.assertAlmostEqual(
            calculate_rmse(actual, predicted),
            math.sqrt(2.0 / 3.0),
        )

    def test_range_normalised_rmse(self):
        actual = [1.0, 2.0, 3.0]
        predicted = [1.0, 3.0, 2.0]

        expected = math.sqrt(2.0 / 3.0) / 2.0

        self.assertAlmostEqual(
            calculate_nrmse(actual, predicted, "range"),
            expected,
        )

    def test_evaluation_rejects_invalid_values(self):
        with self.assertRaises(ValueError):
            evaluate_predictions([], [])

        with self.assertRaises(ValueError):
            evaluate_predictions([1.0, 2.0], [1.0])

        with self.assertRaises(ValueError):
            evaluate_predictions([1.0, float("nan")], [1.0, 2.0])

    def test_prediction_timing(self):
        def example_prediction(value):
            time.sleep(0.001)
            return value * 2

        result, elapsed = time_prediction(
            example_prediction,
            5,
        )

        self.assertEqual(result, 10)
        self.assertGreater(elapsed, 0.0)

class EvaluationPlotTests(unittest.TestCase):
    def test_actual_vs_predicted_plot_is_created(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = (
                Path(temporary_directory)
                / "actual_vs_predicted.png"
            )

            result = plot_actual_vs_predicted(
                actual=[100, 120, 140, 130],
                predicted=[105, 115, 138, 134],
                output_path=output_path,
                title="Test Traffic Prediction",
            )

            self.assertEqual(Path(result), output_path)
            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 0)

    def test_loss_plot_is_created(self):
        history = [
            {
                "epoch": 1,
                "training_mse_scaled": 0.20,
                "validation_mse_scaled": 0.25,
            },
            {
                "epoch": 2,
                "training_mse_scaled": 0.12,
                "validation_mse_scaled": 0.15,
            },
        ]

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = (
                Path(temporary_directory)
                / "loss_history.png"
            )

            result = plot_loss_history(
                loss_history=history,
                output_path=output_path,
            )

            self.assertEqual(Path(result), output_path)
            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 0)

if __name__ == "__main__":
    unittest.main()