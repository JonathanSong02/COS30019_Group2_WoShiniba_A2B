"""Tests for models.regression_tree_model (Darren).

Uses the real processed SCATS files; run data_processing.scats_preprocess first.
Training tests write to a temporary folder so saved_models/ is not touched.
"""
import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from models.regression_tree_model import (TreeSettings, predict_regression_tree,
                                          predict_with_tree, train_regression_tree,
                                          validate_horizon)

CSV = Path("data") / "processed" / "scats_15min.csv"
SAVED = Path("saved_models")
WHEN = "2006-10-27T08:00:00"
SMALL_GRID = dict(max_depth_grid=[4, 8], min_samples_leaf_grid=[5, 20])

try:
    import sklearn  # noqa: F401  (only training needs it)
    HAVE_SKLEARN = True
except ImportError:
    HAVE_SKLEARN = False


@unittest.skipUnless(CSV.exists(), "run data_processing.scats_preprocess first")
@unittest.skipUnless(HAVE_SKLEARN, "scikit-learn is needed for training tests")
class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.result = train_regression_tree(CSV, cls.tmp.name,
                                           TreeSettings(site="2000", **SMALL_GRID))
        cls.meta = json.loads(Path(cls.result["metadata_path"]).read_text())

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_files_saved_with_gru_style_names(self):
        for key in ("tree_path", "metadata_path", "test_predictions_path"):
            self.assertTrue(Path(self.result[key]).exists())
        self.assertTrue(self.result["metadata_path"].endswith("rt_site_2000_h15_s8.json"))

    def test_same_test_windows_as_gru(self):
        # Same shared windows -> same number of held-out test samples as the GRU.
        gru = SAVED / "gru_site_2000_h15_s8.json"
        if not gru.exists():
            self.skipTest("GRU results not available")
        self.assertEqual(self.meta["test_samples"], json.loads(gru.read_text())["test_samples"])

    def test_hyperparameters_chosen_from_validation_search(self):
        best = min(self.meta["hyperparameter_search"], key=lambda r: r["validation_mse_scaled"])
        self.assertEqual(self.meta["selected_hyperparameters"],
                         {"max_depth": best["max_depth"], "min_samples_leaf": best["min_samples_leaf"]})

    def test_exported_tree_matches_sklearn(self):
        from sklearn.tree import DecisionTreeRegressor
        rng = np.random.default_rng(0)
        x, y = rng.random((300, 8)), rng.random(300)
        model = DecisionTreeRegressor(max_depth=6, random_state=0).fit(x, y)
        from models.regression_tree_model import _export_tree
        np.testing.assert_allclose(predict_with_tree(_export_tree(model), x), model.predict(x))

    def test_metrics_reasonable(self):
        test = self.result["test"]
        self.assertEqual(test["count"], self.meta["test_samples"])
        self.assertGreater(test["rmse"], 0)
        self.assertLess(test["nrmse"], 0.2)

    def test_prediction_matches_saved_model(self):
        value = predict_regression_tree(2000, WHEN, 15, CSV, model_dir=self.tmp.name)
        self.assertGreaterEqual(value, 0)


@unittest.skipUnless(CSV.exists(), "run data_processing.scats_preprocess first")
class PredictionInterfaceTests(unittest.TestCase):
    def setUp(self):
        if not (SAVED / "rt_site_2000_h15_s8_tree.json").exists():
            self.skipTest("saved Regression Tree models not available")

    def test_predicts_a_vehicle_count(self):
        value = predict_regression_tree("2000", WHEN, 15, CSV, model_dir=SAVED)
        self.assertTrue(np.isfinite(value) and value >= 0)

    def test_previously_broken_sites_now_work(self):
        for site in (3685, 4264):
            with self.subTest(site=site):
                self.assertGreaterEqual(predict_regression_tree(site, WHEN, 15, CSV, model_dir=SAVED), 0)

    def test_unknown_site_raises(self):
        with self.assertRaises(FileNotFoundError):
            predict_regression_tree(9999, WHEN, 15, CSV, model_dir=SAVED)

    def test_misaligned_time_raises(self):
        with self.assertRaises(ValueError):
            predict_regression_tree(2000, "2006-10-27T08:07:00", 15, CSV, model_dir=SAVED)

    def test_missing_history_raises(self):
        # 2000 has whole missing days; the first lookback steps of October have no history.
        with self.assertRaises(ValueError):
            predict_regression_tree(2000, "2006-10-01T00:30:00", 15, CSV, model_dir=SAVED)

    def test_unsupported_horizon_raises(self):
        with self.assertRaises(ValueError):
            validate_horizon(45)

    def test_main_py_uses_regression_tree(self):
        from main import predict_traffic
        direct = predict_regression_tree(2000, WHEN, 15, CSV, model_dir=SAVED)
        via_main = predict_traffic(2000, WHEN, "regression_tree", 15, 8, CSV, SAVED)
        self.assertAlmostEqual(direct, via_main)


if __name__ == "__main__":
    unittest.main()
