"""GUI callback tests: no browser required, external routing dependencies mocked."""
import unittest
from unittest.mock import patch

from gui.app import _site_id, run_prediction, run_routes


class GUIBackendTests(unittest.TestCase):
    def test_site_validation(self):
        self.assertEqual(_site_id("2000"), 2000)
        for bad in ("", "abc", "-3", "2.5", True):
            with self.subTest(value=bad), self.assertRaises(ValueError):
                _site_id(bad)

    @patch("gui.app.backend.predict_traffic", return_value=864.778846)
    def test_real_prediction_callback_contract(self, mock_predict):
        status, prediction, details = run_prediction(
            "2000", "2006-10-20T10:00:00", "GRU", 30, 8)
        self.assertIn("successful", status)
        self.assertAlmostEqual(prediction, 864.78)
        self.assertEqual(details["forecast_target_interval"], "2006-10-20T10:30:00")
        self.assertIn("NOT", details["units_note"])
        mock_predict.assert_called_once()

    def test_invalid_date_is_explained(self):
        status, prediction, details = run_prediction(
            "2000", "not-a-date", "GRU", 30, 8)
        self.assertIn("unavailable", status)
        self.assertIsNone(prediction)
        self.assertEqual(details, {})

    def test_gui_uses_darren_road_graph_by_default(self):
        from gui.app import ROADS_DEFAULT
        self.assertEqual(ROADS_DEFAULT.name, "scats_road_edges.csv")
        self.assertTrue(ROADS_DEFAULT.is_file())

    def test_partial_route_json_is_serialisable_by_gradio(self):
        # Tests the browser output contract, not just Python route calculation.
        import gradio as gr
        from gui import app as gui_module
        from main import DEFAULT_ROADS
        # Deliberately simulate two unavailable checkpoints instead of relying
        # on the current saved_models/ folder. This test must continue to work
        # after sites 3685 and 4264 are trained.
        real_predict = gui_module.backend.predict_traffic

        def missing_two_sites(site, as_of, **kwargs):
            if int(site) in (3685, 4264):
                raise FileNotFoundError(f"Simulated missing checkpoint for site {site}")
            return real_predict(site, as_of, **kwargs)

        with patch("gui.app.backend.predict_traffic", side_effect=missing_two_sites):
            status, routes, details, map_html = run_routes(
                "2000", "3002", "2006-10-20T10:00:00", "GRU", 15,
                8, 5, str(DEFAULT_ROADS), "origin",
                "Clamp to 1500 (research sensitivity only)", "exclude",
            )
        self.assertIn("PARTIAL-NETWORK", status)
        self.assertEqual(len(routes), 5)
        self.assertIn("3685", details["assumptions"]["unavailable_sites"])
        self.assertIn("4264", details["assumptions"]["unavailable_sites"])
        gr.JSON().postprocess(details)  # Previously raised TypeError: Dict key must be str
        gr.Dataframe(headers=gui_module.ROUTE_HEADERS).postprocess(routes)
        # Map shows the same five routes, and the excluded sites greyed out.
        self.assertIn("<iframe", map_html)
        self.assertEqual(map_html.count("&quot;rank&quot;:"), 5)
        gr.HTML().postprocess(map_html)

    def test_missing_verified_graph_error(self):
        status, routes, details, map_html = run_routes(
            "2000", "3002", "2006-10-20T10:00:00", "GRU", 15,
            8, 5, "missing_verified_links_for_test.csv", "origin",
            "Reject values above 1500 veh/hour")
        self.assertIn("unavailable", status)
        self.assertIn("Missing verified road links", status)
        self.assertEqual(routes, [])
        self.assertEqual(details, {})
        # The map still renders the sites so the tab is never blank.
        self.assertIn("<iframe", map_html)


if __name__ == "__main__":
    unittest.main()
