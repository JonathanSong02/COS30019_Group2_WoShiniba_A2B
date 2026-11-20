"""Regression tests: Gradio defaults must come from config.json."""
import unittest
from unittest.mock import patch
from gui.app import gui_config_defaults, build_app


class GUIConfigurationTests(unittest.TestCase):
    def test_original_defaults(self):
        self.assertEqual(gui_config_defaults({"default_model": "GRU", "prediction_horizon_minutes": 15}), ("GRU", 15))

    def test_regression_tree_30_minutes(self):
        self.assertEqual(gui_config_defaults({"default_model": "Regression Tree", "prediction_horizon_minutes": 30}), ("Regression Tree", 30))

    def test_model_identifiers_and_case(self):
        self.assertEqual(gui_config_defaults({"default_model": "regression_tree", "prediction_horizon_minutes": "60"}), ("Regression Tree", 60))
        self.assertEqual(gui_config_defaults({"default_model": "lstm", "prediction_horizon_minutes": 15}), ("LSTM", 15))

    def test_invalid_entries_fall_back_safely(self):
        self.assertEqual(gui_config_defaults({"default_model": "invalid", "prediction_horizon_minutes": "bad"}), ("GRU", 15))
        self.assertEqual(gui_config_defaults({"default_model": False, "prediction_horizon_minutes": True}), ("GRU", 15))

    def test_both_tabs_respond_to_real_config(self):
        config = {"default_model": "Regression Tree", "prediction_horizon_minutes": 30, "top_k_routes": 5}
        with patch("gui.app.backend.load_config", return_value=config):
            ui = build_app()
        by_label = {
            block.label: block.value for block in ui.blocks.values()
            if getattr(block, "label", None) in (
                "ML model", "Forecast horizon (minutes)",
                "Prediction model", "Prediction horizon (min)")
        }
        self.assertEqual(by_label, {
            "ML model": "Regression Tree",
            "Forecast horizon (minutes)": 30,
            "Prediction model": "Regression Tree",
            "Prediction horizon (min)": 30,
        })


if __name__ == "__main__":
    unittest.main()
