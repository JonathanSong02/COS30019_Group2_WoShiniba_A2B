"""Tests for LSTM evaluation evidence export."""

import csv
import tempfile
import unittest
from pathlib import Path

from evaluation.run_lstm_evaluation import write_prediction_csv


class LSTMPredictionCSVTests(unittest.TestCase):
    def test_prediction_csv_export(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = (
                Path(temporary_directory) / "predictions.csv"
            )

            result = write_prediction_csv(
                output_path=output_path,
                site=2000,
                horizon_minutes=15,
                sequence_length=8,
                timestamps=[
                    "2006-10-27T00:00:00",
                    "2006-10-27T00:15:00",
                ],
                actual=[100.0, 120.0],
                predicted=[110.0, 115.0],
            )

            self.assertEqual(result, output_path)
            self.assertTrue(output_path.exists())

            with output_path.open(
                "r",
                newline="",
                encoding="utf-8",
            ) as csv_file:
                rows = list(csv.DictReader(csv_file))

            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["scats_site"], "2000")
            self.assertEqual(rows[0]["horizon_minutes"], "15")
            self.assertEqual(rows[0]["sequence_length"], "8")
            self.assertEqual(
                float(rows[0]["absolute_error"]),
                10.0,
            )


if __name__ == "__main__":
    unittest.main()