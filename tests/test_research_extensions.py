"""Tests for the full-horizon three-model comparison."""
import json
import unittest
from pathlib import Path

from gui import comparison_view as cv

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results" / "model_comparison" / "three_model" / "summary.json"


class FullHorizonComparisonTests(unittest.TestCase):
    def setUp(self):
        self.summary = json.loads(SUMMARY.read_text(encoding="utf-8"))

    def test_all_three_models_cover_all_40_sites_at_every_horizon(self):
        for horizon in ("15", "30", "60"):
            self.assertEqual(len(self.summary["common_sites_by_horizon"][horizon]), 40)
            self.assertEqual(set(self.summary["macro_average_metrics_by_horizon"][horizon]),
                             {"GRU", "LSTM", "Regression Tree"})

    def test_error_grows_with_horizon_for_every_model(self):
        macro = self.summary["macro_average_metrics_by_horizon"]
        for model in ("GRU", "LSTM", "Regression Tree"):
            rmse = [macro[h][model]["rmse"] for h in ("15", "30", "60")]
            self.assertEqual(rmse, sorted(rmse))

    def test_gui_finds_full_network_lstm_predictions(self):
        files = cv.prediction_files(970, 30)
        self.assertTrue(files["LSTM"].is_file())
        self.assertIn("full_network_h30_s8", str(files["LSTM"]))


if __name__ == "__main__":
    unittest.main()
