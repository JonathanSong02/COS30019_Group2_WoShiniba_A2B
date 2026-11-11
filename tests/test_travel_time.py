"""Unit tests grounded in supplied traffic-flow conversion PDF."""
import math
import unittest
from routing.travel_time import (
    speed_from_hourly_flow, edge_travel_time_seconds,
    hourly_volume_from_four_quarters, weights_from_road_links,
)
from routing.scats_adapter import build_weighted_scats_graph
from routing.top_k_routes import find_top_k_routes

class TravelTimeTests(unittest.TestCase):
    def test_free_flow_speed_is_capped_at_60(self):
        for q in (0, 100, 351):
            self.assertAlmostEqual(speed_from_hourly_flow(q), 60)

    def test_speed_slows_as_flow_increases(self):
        self.assertGreater(speed_from_hourly_flow(600), speed_from_hourly_flow(1000))
        self.assertGreater(speed_from_hourly_flow(1000), speed_from_hourly_flow(1490))

    def test_at_capacity_close_to_32(self):
        self.assertAlmostEqual(speed_from_hourly_flow(1500), 32, delta=0.1)

    def test_time_at_speed_limit_and_intersection_delay(self):
        self.assertAlmostEqual(edge_travel_time_seconds(1, 100), 90)

    def test_high_flow_takes_longer(self):
        self.assertGreater(edge_travel_time_seconds(1, 1200),
                           edge_travel_time_seconds(1, 100))

    def test_four_quarters_to_hour(self):
        self.assertEqual(hourly_volume_from_four_quarters([100, 110, 120, 130]), 460)
        with self.assertRaises(ValueError):
            hourly_volume_from_four_quarters([100, 120])

    def test_invalid_flow_rejected(self):
        for q in (-1, 1501, float('nan'), float('inf'), True):
            with self.subTest(q=q), self.assertRaises(ValueError):
                speed_from_hourly_flow(q)

    def test_missing_predictions_rejected(self):
        with self.assertRaises(KeyError):
            weights_from_road_links([(10, 20, 1.0)], {})

    def test_origin_vs_destination_convention_is_explicit(self):
        links = [(10, 20, 1.0)]
        counts = {10: 100, 20: 1300}
        orig = weights_from_road_links(links, counts, flow_site='origin')[0][2]
        dest = weights_from_road_links(links, counts, flow_site='destination')[0][2]
        self.assertLess(orig, dest)

    def test_integration_route_changes_with_traffic(self):
        # Synthetic graph only, NOT real SCATS connections or measured data.
        sites = {10: (-37.8, 145.0), 20: (-37.8, 145.01), 30: (-37.81, 145.0), 40: (-37.81, 145.01)}
        links = [(10, 20, 1.0), (20, 40, 1.0),
                 (10, 30, 1.3), (30, 40, 1.3)]
        low = {10: 100, 20: 100, 30: 100}
        heavy = {10: 100, 20: 1490, 30: 100}
        graph_low = build_weighted_scats_graph(sites, weights_from_road_links(links, low))
        graph_heavy = build_weighted_scats_graph(sites, weights_from_road_links(links, heavy))
        route_low = find_top_k_routes(graph_low, 10, 40, 2)
        route_heavy = find_top_k_routes(graph_heavy, 10, 40, 2)
        self.assertEqual(route_low[0][0], [10, 20, 40])
        self.assertEqual(route_heavy[0][0], [10, 30, 40])

if __name__ == '__main__':
    unittest.main()
