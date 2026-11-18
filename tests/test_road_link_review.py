"""Offline tests for data_processing/review_road_links.py (no internet: the servers are faked)."""
import re
import unittest

from data_processing import review_road_links as rv
from data_processing.build_road_graph import ROAD_LINKS, haversine_km

ROAD_NAMES = sorted({n for pair in rv.SNAP_TO_ROAD.values() for n in pair})


def fake_server(stretch: float = 1.0):
    """OSRM: route length = straight line x stretch on 'Test Road'.
    OSM map: one node 10 m north of the box centre on every snapping road."""
    calls = []

    def fetch(url):
        calls.append(url)
        if "/route/v1/driving/" in url:
            lon1, lat1, lon2, lat2 = map(float, re.search(r"driving/([-\d.]+),([-\d.]+);([-\d.]+),([-\d.]+)", url).groups())
            metres = haversine_km(lat1, lon1, lat2, lon2) * 1000 * stretch
            return {"code": "Ok", "routes": [{"distance": metres, "duration": metres / 10,
                                              "legs": [{"steps": [{"name": "Test Road", "distance": metres}]}]}]}
        w, s, e, n = map(float, re.search(r"bbox=([-\d.]+),([-\d.]+),([-\d.]+),([-\d.]+)", url).groups())
        lat, lon = (s + n) / 2 + 0.00009, (w + e) / 2
        elements = [{"type": "node", "id": 1, "lat": lat, "lon": lon}]
        elements += [{"type": "way", "id": 10 + i, "nodes": [1], "tags": {"highway": "primary", "name": name}}
                     for i, name in enumerate(ROAD_NAMES)]
        return {"elements": elements}

    fetch.calls = calls
    return fetch


class ReviewLogicTests(unittest.TestCase):
    def test_main_roads_keeps_roads_with_15_percent_or_more(self):
        steps = [{"name": "Warrigal Road", "distance": 800}, {"name": "Toorak Road", "distance": 140},
                 {"name": "Hyslop Street", "distance": 50}]
        self.assertEqual(rv.main_roads(steps), "Warrigal Rd")
        steps[1]["distance"] = 300
        self.assertEqual(rv.main_roads(steps), "Warrigal Rd+Toorak Rd")

    def test_classify_by_distance_ratio(self):
        self.assertEqual(rv.classify(1, 2, 1.05)[0], "VERIFIED")
        self.assertEqual(rv.classify(1, 2, 1.3)[0], "DISTANCE_SHORT")
        self.assertEqual(rv.classify(1, 2, 0.7)[0], "DISTANCE_LONG")
        self.assertEqual(rv.classify(2827, 2825, 1.6)[0], "WRONG_ROAD_FIXED")   # either order

    def test_snapping_picks_nearest_node_of_the_named_road(self):
        point, metres = rv.nearest_node_on_road((-37.8, 145.0), "Bulleen Road", fetch=fake_server())
        self.assertAlmostEqual(metres, 10, delta=1)
        with self.assertRaises(ValueError):
            rv.nearest_node_on_road((-37.8, 145.0), "No Such Road", fetch=fake_server())

    def test_osrm_error_is_raised_not_hidden(self):
        with self.assertRaises(ValueError):
            rv.route((-37.8, 145.0), (-37.81, 145.0), fetch=lambda url: {"code": "NoRoute"})


class ServerFallbackTests(unittest.TestCase):
    def test_second_server_used_when_first_drops_the_connection(self):
        good = fake_server()

        def fetch(url):
            if url.startswith(rv.OSRM_SERVERS[0]):
                raise ConnectionResetError("forcibly closed by the remote host")
            return good(url)
        r = rv.route((-37.8, 145.0), (-37.81, 145.0), fetch=fetch)
        self.assertEqual(r["server"], rv.OSRM_SERVERS[1])
        self.assertAlmostEqual(r["km"], 1.112, places=2)

    def test_clear_error_when_no_server_reachable(self):
        def fetch(url):
            raise ConnectionResetError("forcibly closed by the remote host")
        with self.assertRaises(ConnectionError) as ctx:
            rv.route((-37.8, 145.0), (-37.81, 145.0), fetch=fetch)
        self.assertIn("No OSRM server reachable", str(ctx.exception))


class FullReviewOfflineTests(unittest.TestCase):
    def test_every_link_reviewed_both_ways_with_snapping_where_listed(self):
        fetch = fake_server(stretch=1.0)
        rows, cache = rv.review(fetch=fetch, pause=0)
        self.assertEqual(len(rows), len(ROAD_LINKS))
        routes = [u for u in fetch.calls if "osrm" in u]
        self.assertEqual(len(routes), 2 * len(ROAD_LINKS))
        snapped = {frozenset((r["site_a"], r["site_b"])) for r in rows if r["endpoint_method"] == "snapped_to_named_road"}
        self.assertEqual(snapped, {frozenset(k) for k in rv.SNAP_TO_ROAD})
        direct = [r for r in rows if r["link_type"] == "direct" and r["endpoint_method"] != "snapped_to_named_road"]
        self.assertTrue(all(r["status"] == "VERIFIED" for r in direct))   # route = straight line here

    def test_cache_means_no_second_round_of_requests(self):
        fetch = fake_server()
        _, cache = rv.review(fetch=fetch, pause=0)
        n = len(fetch.calls)
        rv.review(fetch=fetch, cache=cache, pause=0)
        self.assertEqual(len(fetch.calls), n)

    def test_long_detours_are_flagged(self):
        rows, _ = rv.review(fetch=fake_server(stretch=1.5), pause=0)
        noted = {frozenset(k) for k in rv.NOTES}
        plain = [r for r in rows if frozenset((r["site_a"], r["site_b"])) not in noted and r["link_type"] == "direct"]
        self.assertTrue(all(r["status"] == "DISTANCE_SHORT" for r in plain))


if __name__ == "__main__":
    unittest.main()
