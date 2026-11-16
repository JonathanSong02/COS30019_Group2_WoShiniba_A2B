"""Tests for the Traffic prediction tab: coverage status, chart data and callback."""
import html
import json
import re
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from gui import prediction_view as pv
from gui.app import DEFAULT_PREDICTION_TIME, run_prediction_view
from main import DEFAULT_CSV, DEFAULT_MODELS


def _payload(iframe: str) -> dict:
    doc = html.unescape(re.search(r'srcdoc="(.*?)" style=', iframe, re.S).group(1))
    return json.loads(re.search(r"const D = (\{.*?\});\n", doc, re.S).group(1).replace("<\\/", "</"))


class CoverageTests(unittest.TestCase):
    def test_counts_come_from_files_on_disk(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("gru_site_2000_h15_s8.pt", "gru_site_970_h15_s8.pt",
                         "rt_site_2000_h30_s8.json", "rt_site_2000_h30_s8_tree.json",
                         "rt_site_2000_h30_s8_test_predictions.csv", "lstm_site_2000_h15_s4.pt"):
                Path(tmp, name).write_text("x")
            cov = pv.model_coverage(tmp)
        self.assertEqual(cov["GRU"][15], {2000, 970})
        self.assertEqual(cov["Regression Tree"][30], {2000})   # _tree/_test files not double counted
        self.assertEqual(cov["LSTM"][15], set())               # other lookback ignored

    def test_status_table_lists_all_models_and_horizons(self):
        cov = pv.model_coverage(DEFAULT_MODELS)
        table = pv.coverage_markdown(cov, 40)
        for label in ("GRU", "LSTM", "Regression Tree"):
            self.assertIn(f"| {label} |", table)
        for h in (15, 30, 60):
            self.assertIn(f"{h} min", table)
        self.assertEqual(len(cov["Regression Tree"][15]), 40)
        self.assertIn("✅ 40 / 40", table)


class ChartDataTests(unittest.TestCase):
    def test_default_time_has_complete_data_for_every_site(self):
        when = datetime.fromisoformat(DEFAULT_PREDICTION_TIME)
        self.assertGreaterEqual(when.date().isoformat(), "2006-10-29")   # outside 24-28 Oct gaps
        for site in (2000, 3002, 3127, 4262, 4263, 4812, 970):
            values = [v for _, v in pv.observed_series(site, when.replace(hour=4), when.replace(hour=10))]
            self.assertTrue(all(v is not None for v in values), site)

    def test_missing_observations_stay_missing(self):
        # Site 3002 has no data on 24 Oct; the chart must show a gap, not a zero.
        values = pv.observed_series(3002, datetime(2006, 10, 24, 8), datetime(2006, 10, 24, 9))
        self.assertTrue(all(v is None for _, v in values))

    def test_payload_has_history_window_actual_and_predictions(self):
        when = datetime(2006, 10, 30, 8, 0)
        results = [{"model": "GRU", "value": 210.0, "note": ""},
                   {"model": "LSTM", "value": None, "note": "No saved LSTM model"}]
        p = pv.build_chart_payload(2000, when, 30, 8, results, DEFAULT_CSV, selected_model="GRU")
        self.assertEqual(p["target"], "2006-10-30T08:30")
        self.assertEqual(p["windowStart"], "2006-10-30T06:15")       # 8 inputs ending at 08:00
        self.assertEqual(p["points"][0]["t"], "2006-10-30T04:00")
        self.assertEqual(p["points"][-1]["t"], "2006-10-30T09:30")
        self.assertIsNotNone(p["actual"])
        self.assertEqual([x["model"] for x in p["predictions"]], ["GRU"])   # no invented LSTM value
        self.assertTrue(p["predictions"][0]["selected"])

    def test_table_error_is_prediction_minus_actual(self):
        rows = pv.results_table([{"model": "GRU", "value": 210.0, "note": ""}], 200.0)
        self.assertEqual(rows[0][3], 10.0)

    def test_missing_model_is_reported_not_run(self):
        calls = []
        out = pv.predict_models(4264, datetime(2006, 10, 30, 8), 15, 8, ["LSTM"],
                                lambda *a, **k: calls.append(a) or 1.0,
                                {"LSTM": {15: {2000}}})
        self.assertEqual(calls, [])
        self.assertIsNone(out[0]["value"])
        self.assertIn("No saved LSTM model", out[0]["note"])


class CallbackTests(unittest.TestCase):
    @patch("gui.app.backend.predict_traffic", return_value=250.0)
    def test_compare_all_runs_every_available_model(self, mock_predict):
        status, volume, details, chart, table = run_prediction_view(
            "2000", DEFAULT_PREDICTION_TIME, "GRU", 15, 8, True)
        self.assertIn("successful", status)
        self.assertEqual(volume, 250.0)
        self.assertEqual([r[0] for r in table], ["GRU", "LSTM", "Regression Tree"])
        self.assertEqual(mock_predict.call_count, 3)      # site 2000 has all three at 15 min
        self.assertIn("actual_vehicles_per_15min", details)
        self.assertEqual(len(_payload(chart)["predictions"]), 3)

    @patch("gui.app.backend.predict_traffic", return_value=250.0)
    def test_single_model_mode(self, mock_predict):
        *_, chart, table = run_prediction_view("2000", DEFAULT_PREDICTION_TIME, "Regression Tree", 15, 8, False)
        self.assertEqual([r[0] for r in table], ["Regression Tree"])
        self.assertEqual(mock_predict.call_count, 1)

    @patch("gui.app.backend.predict_traffic", return_value=250.0)
    def test_comparison_is_off_unless_chosen(self, mock_predict):
        *_, chart, table = run_prediction_view("2000", DEFAULT_PREDICTION_TIME, "GRU", 15, 8)
        self.assertEqual([r[0] for r in table], ["GRU"])
        self.assertEqual(mock_predict.call_count, 1)

    def test_chart_has_zoomed_detail_panel_with_actual_on_top(self):
        p = pv.build_chart_payload(2000, datetime(2006, 10, 30, 8), 15, 8,
                                   [{"model": "GRU", "value": 1300.0, "note": ""}], DEFAULT_CSV)
        doc = pv.build_chart_document(p)
        self.assertIn("Forecast detail", doc)
        # the actual-value ring is drawn after the prediction dots, so it is never hidden
        self.assertLess(doc.index('el("circle", {cx: tx + dxOf(i)'), doc.index('el("ellipse"'))

    def test_invalid_input_returns_no_chart(self):
        status, volume, details, chart, table = run_prediction_view(
            "2000", "not-a-date", "GRU", 15, 8, True)
        self.assertIn("unavailable", status)
        self.assertEqual((chart, table), ("", []))

    def test_real_regression_tree_prediction_in_chart(self):
        status, volume, details, chart, table = run_prediction_view(
            "2000", DEFAULT_PREDICTION_TIME, "Regression Tree", 15, 8, False)
        self.assertIn("successful", status)
        self.assertGreater(volume, 0)
        self.assertEqual(_payload(chart)["predictions"][0]["model"], "Regression Tree")


if __name__ == "__main__":
    unittest.main()
