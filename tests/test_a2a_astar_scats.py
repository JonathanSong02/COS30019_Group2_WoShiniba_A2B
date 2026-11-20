"""Verify original A2A A* against A2B A* first route on A2B SCATS graphs.

These are *synthetic*, within-capacity traffic inputs for a deterministic
algorithm-integration check, NOT actual trained-model predictions.
No changes are made to production routing or model code.
"""
import csv
import math
from pathlib import Path
import unittest

from algorithms.astar import search as astar_search
from routing.scats_adapter import build_weighted_scats_graph
from routing.top_k_routes import find_top_k_routes
from routing.travel_time import weights_from_road_links

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class A2A_AStar_SCATS_IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with (PROJECT_ROOT / "data/processed/scats_sites.csv").open(
            newline="", encoding="utf-8-sig"
        ) as file:
            cls.coords = {
                int(row["scats_site"]):
                (float(row["latitude"]), float(row["longitude"]))
                for row in csv.DictReader(file)
            }
        with (PROJECT_ROOT / "data/processed/scats_road_edges.csv").open(
            newline="", encoding="utf-8-sig"
        ) as file:
            cls.roads = [
                (int(row["from_site"]), int(row["to_site"]),
                 float(row["distance_km"]))
                for row in csv.DictReader(file)
            ]
        # A deterministic, invented traffic scenario for integration testing.
        # Every input is below the assignment's 1500 veh/h capacity.
        cls.hourly_flows = {site: 300.0 + site % 600 for site in cls.coords}

    def _check_shortest_route(self, origin, destination, convention):
        weighted_edges = weights_from_road_links(
            self.roads, self.hourly_flows, flow_site=convention
        )
        graph = build_weighted_scats_graph(self.coords, weighted_edges)
        graph.origin = origin
        graph.destinations = {destination}
        a2a_goal, created = astar_search(graph)
        self.assertIsNotNone(a2a_goal)
        self.assertGreater(created, 0)
        self.assertEqual(a2a_goal.path()[0], origin)
        self.assertEqual(a2a_goal.path()[-1], destination)
        self.assertEqual(len(a2a_goal.path()), len(set(a2a_goal.path())))
        routes = find_top_k_routes(graph, origin, destination, k=5)
        self.assertGreater(len(routes), 0)
        self.assertTrue(math.isclose(
            a2a_goal.path_cost, routes[0][1], rel_tol=1e-9, abs_tol=1e-7
        ), msg=f"A* {a2a_goal.path_cost} vs A2B A* {routes[0][1]}")

    def test_a2a_astar_matches_a2b_first_route_using_origin_flow(self):
        for origin, destination in [(2000, 3002), (3002, 2000), (970, 4812)]:
            with self.subTest(origin=origin, destination=destination):
                self._check_shortest_route(origin, destination, "origin")

    def test_a2a_astar_matches_a2b_first_route_using_destination_flow(self):
        for origin, destination in [(2000, 3002), (3002, 2000), (970, 4812)]:
            with self.subTest(origin=origin, destination=destination):
                self._check_shortest_route(origin, destination, "destination")


if __name__ == "__main__":
    unittest.main()
