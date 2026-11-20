"""Integration contract tests using clearly synthetic graph and fake predictor."""
import tempfile
import unittest
from pathlib import Path

from main import (DEFAULT_SITES, compute_traffic_routes, load_site_approach_counts,
                  load_site_coordinates, load_verified_road_links, predict_traffic,
                  valid_as_of)


class MainIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.coords = {101: (-37.80, 145.0), 102: (-37.81, 145.0),
                       103: (-37.82, 145.0), 105: (-37.83, 145.0)}
        # Distinct directed routes, all synthetic and only for interface tests.
        self.links = [(101, 102, 0.5), (102, 105, 0.5),
                      (101, 103, 0.6), (103, 105, 0.6)]
        self.when = '2006-10-20T10:00:00'
        # One approach per synthetic site: per-approach flow equals the site
        # total, so the original contract tests keep their meaning.
        self.approaches = {101: 1, 102: 1, 103: 1, 105: 1}

    def _routes(self, *args, **kwargs):
        kwargs.setdefault('approach_counts', self.approaches)
        return compute_traffic_routes(*args, **kwargs)

    def test_cheapest_route_ranking(self):
        r = self._routes(101, 105, self.when, self.links, self.coords,
                                   predictor=lambda site, t: 50.0)
        self.assertEqual(r['route_count'], 2)
        self.assertEqual(r['routes'][0]['scats_sites'], [101, 102, 105])
        self.assertLess(r['routes'][0]['travel_time_seconds'],
                        r['routes'][1]['travel_time_seconds'])

    def test_congestion_reranks_routes(self):
        base = self._routes(101, 105, self.when, self.links, self.coords,
                                      predictor=lambda site, t: 50.0)
        congested = self._routes(101, 105, self.when, self.links, self.coords,
                                           predictor=lambda site, t: 370.0 if site == 102 else 50.0)
        # Note: 370*4=1480 veh/hr, under capacity, but very slow.
        self.assertEqual(base['routes'][0]['scats_sites'], [101, 102, 105])
        self.assertEqual(congested['routes'][0]['scats_sites'], [101, 103, 105])

    def test_missing_site_prediction_propagates(self):
        def missing(site, when):
            raise FileNotFoundError(f'No model for site {site}')
        with self.assertRaisesRegex(FileNotFoundError, 'No model'):
            self._routes(101, 105, self.when, self.links, self.coords, missing)

    def test_missing_sites_can_be_excluded_with_clear_warning(self):
        def predictor(site, when):
            if site == 103:
                raise FileNotFoundError("No model for site 103")
            return 50.0
        result = self._routes(
            101, 105, self.when, self.links, self.coords, predictor,
            missing_site_policy="exclude",
        )
        self.assertEqual(result["route_count"], 1)
        self.assertEqual(result["routes"][0]["scats_sites"], [101, 102, 105])
        self.assertEqual(result["assumptions"]["excluded_road_link_count"], 2)
        self.assertIn(103, result["assumptions"]["unavailable_sites"])
        self.assertTrue(result["assumptions"]["network_incomplete"])

    def test_selected_destination_without_model_is_rejected_in_partial_mode(self):
        def predictor(site, when):
            if site == 105:
                raise FileNotFoundError("Missing checkpoint")
            return 50.0
        with self.assertRaisesRegex(ValueError, "Selected origin/destination"):
            self._routes(101, 105, self.when, self.links, self.coords,
                                   predictor, missing_site_policy="exclude")

    def test_invalid_missing_site_policy_rejected(self):
        with self.assertRaisesRegex(ValueError, "missing_site_policy"):
            self._routes(101, 105, self.when, self.links, self.coords,
                                   predictor=lambda site, when: 50.0,
                                   missing_site_policy="fill-with-zero")

    def test_partial_mode_excludes_missing_history_without_imputation(self):
        def predictor(site, when):
            if site == 103:
                raise ValueError("Insufficient nonmissing historical observations for this time")
            return 50.0
        result = self._routes(101, 105, self.when, self.links, self.coords,
                                        predictor, missing_site_policy="exclude")
        self.assertEqual(result["route_count"], 1)

    def test_overcapacity_rejected_default(self):
        with self.assertRaisesRegex(ValueError, 'exceeding'):
            self._routes(101, 105, self.when, self.links, self.coords,
                                   predictor=lambda s,t: 600.0)

    def test_capacity_clamp_explicit(self):
        r = self._routes(101, 105, self.when, self.links, self.coords,
                                   predictor=lambda s,t: 600.0, capacity_policy='clamp')
        self.assertTrue(r['assumptions']['capacity_clamped_sites'])

    def test_no_inferred_reverse_link(self):
        r = self._routes(105, 101, self.when, self.links, self.coords,
                                   predictor=lambda s,t: 50.0)
        self.assertEqual(r['routes'], [])

    def test_time_validation(self):
        with self.assertRaises(ValueError):
            valid_as_of('2006-10-20T10:03:00')

    def test_invalid_road_links_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)/'roads.csv'
            p.write_text('from_site,to_site,distance_km\n101,102,0.5\n101,102,0.6\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'repeated'):
                load_verified_road_links(p)

    def test_load_site_coordinates(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)/'sites.csv'
            p.write_text('scats_site,latitude,longitude\n101,-37.8,145.0\n102,-37.9,145.1\n', encoding='utf-8')
            self.assertEqual(len(load_site_coordinates(p)), 2)

    def test_missing_regression_tree_not_silent(self):
        # Regression Tree is integrated; a missing checkpoint must still fail loudly.
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(FileNotFoundError):
                predict_traffic(2000, self.when, 'regression_tree', model_dir=folder)


    # --- Per-approach flow conversion (agreed with Dr. Joel, 19 Nov) ---
    def test_per_approach_divides_site_total_by_approach_count(self):
        four = {s: 4 for s in self.coords}
        r = compute_traffic_routes(101, 105, self.when, self.links, self.coords,
                                   predictor=lambda s, t: 500.0, approach_counts=four)
        self.assertEqual(r['route_count'], 2)          # 2000 veh/h site total, 500 per approach
        self.assertEqual(r['assumptions']['flow_conversion'], 'per_approach')
        rec = {x['site']: x for x in r['assumptions']['site_flows']}[101]
        self.assertEqual(rec['predicted_vehicles_per_15min'], 500.0)   # model output unchanged
        self.assertEqual(rec['site_total_veh_per_hour'], 2000.0)
        self.assertEqual(rec['approaches'], 4)
        self.assertEqual(rec['flow_used_veh_per_hour'], 500.0)
        self.assertFalse(r['assumptions']['capacity_clamped_sites'])

    def test_site_total_keeps_old_behaviour(self):
        with self.assertRaisesRegex(ValueError, 'site-total'):
            compute_traffic_routes(101, 105, self.when, self.links, self.coords,
                                   predictor=lambda s, t: 500.0, flow_conversion='site_total')

    def test_per_approach_flow_above_capacity_still_rejected(self):
        four = {s: 4 for s in self.coords}
        with self.assertRaisesRegex(ValueError, 'per-approach vehicles/hour'):
            compute_traffic_routes(101, 105, self.when, self.links, self.coords,
                                   predictor=lambda s, t: 1600.0, approach_counts=four)

    def test_per_approach_clamp_is_labelled_experimental(self):
        four = {s: 4 for s in self.coords}
        r = compute_traffic_routes(101, 105, self.when, self.links, self.coords,
                                   predictor=lambda s, t: 1600.0, approach_counts=four,
                                   capacity_policy='clamp')
        self.assertTrue(r['assumptions']['capacity_clamped_sites'])
        self.assertIn('EXPERIMENTAL', r['assumptions']['capacity_policy_note'])
        rec = r['assumptions']['site_flows'][0]
        self.assertTrue(rec['clamped'])
        self.assertEqual(rec['flow_used_veh_per_hour'], 1500.0)

    def test_missing_approach_count_is_an_error_not_a_guess(self):
        with self.assertRaisesRegex(KeyError, 'approach count'):
            compute_traffic_routes(101, 105, self.when, self.links, self.coords,
                                   predictor=lambda s, t: 50.0, approach_counts={101: 4})

    def test_invalid_flow_conversion_rejected(self):
        with self.assertRaisesRegex(ValueError, 'flow_conversion'):
            self._routes(101, 105, self.when, self.links, self.coords,
                         predictor=lambda s, t: 50.0, flow_conversion='average')

    def test_real_site_approach_counts(self):
        counts = load_site_approach_counts(DEFAULT_SITES)
        self.assertEqual(len(counts), 40)
        self.assertTrue(all(n >= 1 for n in counts.values()))
        self.assertEqual(counts[3685], 2)   # broken Warrigal Rd S detector dropped
        self.assertEqual(counts[4264], 3)   # broken Glenferrie Rd S detector dropped

if __name__ == '__main__':
    unittest.main()
