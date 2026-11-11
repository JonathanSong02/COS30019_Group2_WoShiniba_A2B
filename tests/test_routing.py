"""Early routing unit tests; do not mistake these for final A2B's 15 tests."""
import unittest
from graph import Graph, load_graph
from routing.top_k_routes import find_top_k_routes
from routing.scats_adapter import build_weighted_scats_graph

def make_graph(edges):
    graph = Graph()
    for node_id in (1, 2, 3, 4, 5):
        graph.add_node(node_id, node_id, 0)
    for source, destination, cost in edges:
        graph.add_edge(source, destination, cost)
    graph.finalise()
    return graph

class RoutingTest(unittest.TestCase):
    def test_valid_route(self):
        g = make_graph([(1, 2, 2), (2, 5, 3), (1, 5, 9)])
        self.assertEqual(find_top_k_routes(g, 1, 5)[0], ([1, 2, 5], 5))

    def test_unreachable(self):
        self.assertEqual(find_top_k_routes(make_graph([(1, 2, 2)]), 1, 5), [])

    def test_origin_equals_destination(self):
        self.assertEqual(find_top_k_routes(make_graph([]), 1, 1), [([1], 0.0)])

    def test_top_five_loopless_ordered(self):
        g = make_graph([(1, 2, 1), (1, 3, 2), (1, 4, 4), (1, 5, 8),
                        (2, 3, 1), (2, 4, 2), (2, 5, 9),
                        (3, 4, 1), (3, 5, 4), (4, 5, 2), (4, 1, 1)])
        routes = find_top_k_routes(g, 1, 5)
        self.assertEqual(len(routes), 5)
        self.assertEqual(routes[0], ([1, 2, 4, 5], 5.0))
        self.assertEqual(len({tuple(p) for p, _ in routes}), 5)
        self.assertEqual([c for _, c in routes], sorted(c for _, c in routes))
        self.assertTrue(all(len(p) == len(set(p)) for p, _ in routes))

    def test_invalid_inputs(self):
        g = make_graph([(1, 2, 1)])
        with self.assertRaises(ValueError):
            find_top_k_routes(g, 1, 999)
        with self.assertRaises(ValueError):
            find_top_k_routes(g, 1, 2, k=6)
        g.add_edge(2, 5, -3)
        with self.assertRaises(ValueError):
            find_top_k_routes(g, 1, 5)

    def test_predicted_cost_changes_route(self):
        g = make_graph([(1, 2, 5), (2, 5, 5), (1, 3, 6), (3, 5, 6)])
        self.assertEqual(find_top_k_routes(g, 1, 5)[0][0], [1, 2, 5])
        g = make_graph([(1, 2, 15), (2, 5, 5), (1, 3, 6), (3, 5, 6)])
        self.assertEqual(find_top_k_routes(g, 1, 5)[0][0], [1, 3, 5])

    def test_scats_adapter_not_guessing_roads(self):
        g = build_weighted_scats_graph(
            {2000: (-37.80, 145.03), 2001: (-37.81, 145.04)},
            [(2000, 2001, 44.0)]
        )
        self.assertEqual(g.neighbours(2000), [(2001, 44.0)])
        self.assertEqual(find_top_k_routes(g, 2000, 2001), [([2000, 2001], 44.0)])
        self.assertEqual(find_top_k_routes(g, 2001, 2000), [])

    def test_a2a_graph_compatibility(self):
        g = load_graph('PathFinder-test.txt')
        self.assertEqual(find_top_k_routes(g, 2, 5)[0], ([2, 3, 5], 10.0))

if __name__ == '__main__':
    unittest.main()
