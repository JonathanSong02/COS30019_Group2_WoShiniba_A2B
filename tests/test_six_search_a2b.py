"""Independent checks of all six A2A methods on A2B weighted travel-time graphs."""
import itertools
import math
import random
import unittest

from graph import Graph
from routing.top_k_routes import METHODS, search_routes, find_top_k_routes


def graph_from_edges(nodes, edges):
    graph = Graph()
    for site in nodes:
        graph.add_node(site, site * 0.7, (site * 5) % 7)
    for u, v, w in edges:
        graph.add_edge(u, v, w)
    graph.finalise()
    return graph


def exhaustive_routes(graph, start, end):
    """Tiny-graph brute-force oracle; different from all six search methods."""
    results = []
    def enumerate_paths(path, cost):
        if path[-1] == end:
            results.append((list(path), cost))
            return
        for next_node, weight in graph.neighbours(path[-1]):
            if next_node not in path:
                enumerate_paths(path + (next_node,), cost + weight)
    enumerate_paths((start,), 0.0)
    return sorted(results, key=lambda r: (r[1], len(r[0]), tuple(r[0])))


class SixA2ASearchIntegrationTests(unittest.TestCase):
    def test_all_six_find_valid_directed_loopless_routes(self):
        g = graph_from_edges([1, 2, 3, 4, 5], [
            (1, 2, 1), (1, 3, 4), (1, 5, 10), (2, 3, 1),
            (2, 4, 1), (2, 5, 3), (3, 4, 1), (3, 5, 3), (4, 5, 2)])
        for name in METHODS:
            with self.subTest(method=name):
                routes, stats = search_routes(g, 1, 5, 5, name)
                self.assertTrue(routes)
                self.assertLessEqual(len(routes), 5)
                self.assertGreater(stats['expanded'], 0)
                self.assertEqual(len({tuple(p) for p, _ in routes}), len(routes))
                for path, cost in routes:
                    self.assertEqual((path[0], path[-1]), (1, 5))
                    self.assertEqual(len(path), len(set(path)))
                    self.assertAlmostEqual(cost, sum(dict(g.neighbours(u))[v] for u, v in zip(path, path[1:])))

    def test_bfs_and_cus1_minimise_hops_not_travel_time(self):
        g = graph_from_edges([1, 2, 3], [(1, 2, 1), (2, 3, 1), (1, 3, 30)])
        for method in ('BFS', 'CUS1'):
            self.assertEqual(find_top_k_routes(g, 1, 3, k=1, algorithm=method), [([1, 3], 30.0)])
        for method in ('AS', 'CUS2'):
            self.assertEqual(find_top_k_routes(g, 1, 3, k=1, algorithm=method), [([1, 2, 3], 2.0)])

    def test_gbfs_is_greedy_not_cost_optimal(self):
        g = graph_from_edges([1, 2, 3, 4], [(1, 2, 1), (2, 4, 1), (1, 4, 40)])
        # A direct goal is preferred by greedy h=0 even though its cost is higher.
        self.assertEqual(find_top_k_routes(g, 1, 4, 1, 'GBFS'), [([1, 4], 40)])

    def test_a_star_and_weighted_ida_star_match_bruteforce_k_on_random_dags(self):
        for seed in range(20):
            rng = random.Random(seed)
            nodes = list(range(1, 8))
            edges = [(u, v, rng.choice([0, 1, 2, 3, 4, 8]))
                     for u in nodes for v in nodes if u < v and rng.random() < 0.45]
            g = graph_from_edges(nodes, edges)
            truth = exhaustive_routes(g, 1, 7)[:5]
            for method in ('AS', 'CUS2'):
                with self.subTest(seed=seed, method=method):
                    routes, stats = search_routes(g, 1, 7, 5, method)
                    self.assertEqual(routes, truth)
                    self.assertTrue(stats['cost_optimal'])

    def test_a_star_and_weighted_ida_star_on_cyclic_graph(self):
        g = graph_from_edges([1, 2, 3, 4], [
            (1, 2, 1), (2, 1, 1), (1, 3, 4), (2, 3, 1),
            (3, 2, 1), (2, 4, 3), (3, 4, 2), (1, 4, 10)])
        expected = exhaustive_routes(g, 1, 4)[:5]
        for method in ('AS', 'CUS2'):
            self.assertEqual(find_top_k_routes(g, 1, 4, 5, method), expected)

    def test_all_six_handle_unreachable_and_same_origin(self):
        g = graph_from_edges([1, 2, 3], [(1, 2, 5)])
        for name in METHODS:
            with self.subTest(method=name):
                self.assertEqual(find_top_k_routes(g, 1, 3, algorithm=name), [])
                self.assertEqual(find_top_k_routes(g, 2, 2, algorithm=name), [([2], 0.0)])

    def test_invalid_search_algorithm_and_costs_rejected(self):
        g = graph_from_edges([1, 2], [(1, 2, 1)])
        with self.assertRaisesRegex(ValueError, 'Unknown search algorithm'):
            find_top_k_routes(g, 1, 2, algorithm='YEN')
        for bad in (float('nan'), -1, float('inf')):
            g = graph_from_edges([1, 2], [(1, 2, bad)])
            for name in METHODS:
                with self.subTest(bad=bad, algorithm=name), self.assertRaises(ValueError):
                    find_top_k_routes(g, 1, 2, algorithm=name)

    def test_gui_exposes_all_six_algorithms_in_both_route_tabs(self):
        from gui.app import build_app
        app = build_app()
        selectors = [component['props'] for component in app.get_config_file()['components']
                     if component['props'].get('label') == 'A2A search algorithm']
        self.assertEqual(len(selectors), 2)
        for selector in selectors:
            self.assertEqual([v for _, v in selector['choices']], list(METHODS))
            self.assertEqual(selector['value'], 'AS')

    def test_configuration_search_default_validation(self):
        import json
        import tempfile
        from pathlib import Path
        import main
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.json'
            path.write_text(json.dumps({'default_search_algorithm': 'CUS2'}), encoding='utf-8')
            self.assertEqual(main.load_config(path)['default_search_algorithm'], 'CUS2')
            path.write_text(json.dumps({'default_search_algorithm': 'YEN'}), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'default_search_algorithm'):
                main.load_config(path)

    def test_main_integration_selects_algorithm(self):
        import main
        coords = {1: (-37.9, 145.0), 2: (-37.9, 145.01), 3: (-37.91, 145.01)}
        links = [(1, 2, 1), (2, 3, 1), (1, 3, 2)]
        for name in METHODS:
            with self.subTest(name=name):
                result = main.compute_traffic_routes(
                    1, 3, '2006-10-30T20:00:00', links, coords,
                    predictor=lambda site, time: 100.0,
                    approach_counts={1: 1, 2: 1, 3: 1},
                    search_algorithm=name, k=2)
                self.assertEqual(result['assumptions']['search_algorithm'], name)
                self.assertEqual(result['assumptions']['search_cost_optimal'], name in ('AS', 'CUS2'))
                self.assertGreater(len(result['routes']), 0)


if __name__ == '__main__':
    unittest.main()
