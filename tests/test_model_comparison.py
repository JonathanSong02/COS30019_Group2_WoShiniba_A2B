"""Tests for Task 5: Regression Tree extras, the Model comparison tab and route comparison."""
import html
import json
import re
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from evaluation import regression_tree_extras as extras
from gui import comparison_view as cv
from gui.app import run_actual_vs_predicted, run_route_comparison


def _specs(iframe: str) -> list:
    doc = html.unescape(re.search(r'srcdoc="(.*?)" style=', iframe, re.S).group(1))
    return json.loads(re.search(r"const SPECS = (\[.*?\]);\n", doc, re.S).group(1).replace("<\\/", "</"))


class RegressionTreeExtrasTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.summary = extras.run(output=Path(cls.tmp.name))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_feature_importance_covers_all_sites_and_horizons(self):
        for h in ("15", "30", "60"):
            imp = self.summary["feature_importance"][h]
            self.assertEqual(imp["sites"], 40)
            self.assertAlmostEqual(sum(imp["mean_importance"].values()), 1.0, places=6)
            self.assertEqual(max(imp["mean_importance"], key=imp["mean_importance"].get), "lag_1")

    def test_parameter_counts_match_pytorch_formula(self):
        # GRU(1->32): 3*(32*1 + 32*32 + 2*32) + Linear(32->1) 33 = 3393; LSTM: 4 gates = 4513
        self.assertEqual(extras.recurrent_parameters(3), 3393)
        self.assertEqual(extras.recurrent_parameters(4), 4513)
        sizes = self.summary["model_size_h15"]
        self.assertEqual(set(sizes), {"GRU", "LSTM", "Regression Tree"})
        self.assertEqual(sizes["GRU"]["sites"], 40)

    def test_peak_offpeak_has_all_three_models_and_matches_anson(self):
        rows = self.summary["peak_offpeak_site_2000"]
        self.assertEqual({r["model"] for r in rows}, {"GRU", "LSTM", "Regression Tree"})
        self.assertEqual(len(rows), 3 * 3 * 2)                   # models x horizons x periods
        rt = [r for r in rows if r["model"] == "Regression Tree"]
        self.assertEqual(sum(r["sample_count"] for r in rt if r["horizon_minutes"] == 15), 480)

    def test_last_value_baseline_uses_the_same_480_points(self):
        base = extras.persistence_baseline()
        r = next(r for r in base if r["scats_site"] == 2000 and r["horizon_minutes"] == 15)
        self.assertEqual(r["sample_count"], 480)
        self.assertLess(r["regression_tree_rmse"], r["last_value_rmse"])


class ComparisonTabTests(unittest.TestCase):
    def setUp(self):
        self.results = cv.load_results()

    def test_results_table_rows_and_fair_training_time(self):
        rows = cv.summary_rows(self.results)
        self.assertEqual([r[0] for r in rows][:3], ["GRU", "LSTM", "Regression Tree"])
        rt = rows[2]
        self.assertGreater(rt[5], 0.1)   # includes the ~0.2 s hyper-parameter search
        self.assertIn("baseline", rows[-1][0])

    def test_headline_names_the_best_model_from_the_numbers(self):
        text = cv.headline(self.results)
        macro = self.results["three_model"]["macro_average_metrics_by_horizon"]["15"]
        best = min(macro, key=lambda m: macro[m]["rmse"])
        self.assertTrue(text.startswith(f"**{best}**"))

    def test_overview_charts(self):
        specs = cv.overview_specs(self.results)
        self.assertEqual([s["type"] for s in specs], ["hbars", "lines", "dots", "groups", "lines"])
        dots = specs[2]
        self.assertEqual(len(dots["rows"]), 40)
        self.assertEqual([s["name"] for s in dots["series"]], ["GRU", "LSTM", "Regression Tree"])
        self.assertIsNotNone(specs[0]["ref"])                  # last-reading baseline line
        self.assertEqual(_specs(cv.build_iframe(specs))[0]["title"], specs[0]["title"])

    def test_missing_results_give_a_clear_message(self):
        with tempfile.TemporaryDirectory() as tmp:
            empty = cv.load_results(Path(tmp), Path(tmp))
        self.assertIn("not found", cv.headline(empty))
        self.assertEqual(cv.overview_specs(empty), [])

    def test_actual_vs_predicted_site_2000_has_all_models(self):
        status, chart = run_actual_vs_predicted("2000", 15, "2006-10-30")
        spec = _specs(chart)[0]
        self.assertEqual([s["name"] for s in spec["series"]], ["Actual", "GRU", "LSTM", "Regression Tree"])
        self.assertEqual(len(spec["x"]), 96)                   # one test day of 15-minute steps
        self.assertEqual(status, "")

    def test_actual_vs_predicted_reports_missing_model_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec, notes = cv.actual_vs_predicted_spec(970, 15, "2006-10-30", models_dir=Path(tmp),
                                                      lstm_site_2000_dir=Path(tmp))
        self.assertIsNone(spec)
        self.assertEqual(len(notes), 3)


class RouteComparisonTests(unittest.TestCase):
    def _route(self, sites, minutes, rank=1):
        return {"rank": rank, "scats_sites": sites, "travel_time_minutes": minutes, "travel_time_seconds": minutes * 60}

    def test_rows_flag_routes_that_differ_from_regression_tree(self):
        results = {"GRU": [self._route([1, 2, 3], 10.0)], "LSTM": [self._route([1, 4, 3], 11.0)],
                   "Regression Tree": [self._route([1, 2, 3], 9.5)]}
        rows, by_model, notes = cv.compare_routes(1, 3, datetime(2006, 10, 30, 20), route_fn=lambda m: {"routes": results[m]})
        same = {r[0]: r[4] for r in rows}
        self.assertEqual(same, {"GRU": "yes", "LSTM": "no", "Regression Tree": "—"})
        best = cv.best_routes_for_map(by_model)
        self.assertEqual([b["colour"] for b in best], [cv.COLOURS[m] for m in cv.MODELS])
        self.assertEqual(best[1]["label"], "LSTM best route")

    def test_a_failing_model_is_reported_not_hidden(self):
        def route_fn(m):
            if m == "LSTM":
                raise FileNotFoundError("Missing LSTM checkpoint")
            return {"routes": [self._route([1, 3], 5.0)]}
        rows, by_model, notes = cv.compare_routes(1, 3, datetime(2006, 10, 30, 20), route_fn=route_fn)
        self.assertNotIn("LSTM", by_model)
        self.assertIn("LSTM: Missing LSTM checkpoint", notes)

    def test_gui_callback_real_regression_tree_routes(self):
        status, rows, map_html = run_route_comparison("2000", "3002", "2006-10-30T20:00:00",
                                                      "Clamp to 1500 (research sensitivity only)")
        self.assertIn("Best route under each model", status)
        self.assertIn("Regression Tree", {r[0] for r in rows})
        self.assertIn("Regression Tree best route", html.unescape(map_html))


if __name__ == "__main__":
    unittest.main()
