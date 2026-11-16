"""Tests for the OpenStreetMap route view (gui/route_map.py). No browser needed."""
import html
import json
import re
import unittest

from gui import route_map
from main import DEFAULT_ROADS, DEFAULT_SITES, load_site_coordinates, load_verified_road_links


def _payload_from_iframe(iframe: str) -> dict:
    doc = html.unescape(re.search(r'srcdoc="(.*?)" style=', iframe, re.S).group(1))
    data = re.search(r"const D = (\{.*?\});\n", doc, re.S).group(1)
    return json.loads(data.replace("<\\/", "</"))


class CoordinateOffsetTests(unittest.TestCase):
    def test_measured_offset_is_documented_and_plausible(self):
        offset = route_map.load_offset()
        # SCATS points sit ~200 m south-west of the OSM intersections.
        self.assertAlmostEqual(offset["dlat"], 0.0015, delta=0.0003)
        self.assertAlmostEqual(offset["dlon"], 0.0013, delta=0.0003)
        self.assertGreaterEqual(offset["n_samples"], 10)
        self.assertLess(offset["mean_error_after_m"], 15)
        self.assertGreater(offset["mean_error_before_m"], 150)

    def test_same_shift_applied_to_every_site(self):
        coords = load_site_coordinates(DEFAULT_SITES)
        fixed = route_map.corrected_coords(coords, {"dlat": 0.001, "dlon": -0.002})
        self.assertEqual(set(fixed), set(coords))
        for site, (lat, lon) in coords.items():
            self.assertAlmostEqual(fixed[site][0] - lat, 0.001)
            self.assertAlmostEqual(fixed[site][1] - lon, -0.002)

    def test_missing_offset_file_means_no_shift(self):
        self.assertEqual(route_map.load_offset("no_such_offset.json")["dlat"], 0.0)

    def test_offset_does_not_change_routing_coordinates(self):
        before = load_site_coordinates(DEFAULT_SITES)
        route_map.build_map_payload(before, [])
        self.assertEqual(before, load_site_coordinates(DEFAULT_SITES))


class MapPayloadTests(unittest.TestCase):
    def setUp(self):
        self.coords = load_site_coordinates(DEFAULT_SITES)
        self.links = load_verified_road_links(DEFAULT_ROADS)
        self.routes = [
            {"rank": 1, "scats_sites": [2000, 3685, 970], "travel_time_minutes": 4.5},
            {"rank": 2, "scats_sites": [2000, 3682], "travel_time_minutes": 6.25},
        ]

    def test_all_sites_and_undirected_links_drawn(self):
        p = route_map.build_map_payload(self.coords, self.links, offset={"dlat": 0, "dlon": 0})
        self.assertEqual(len(p["sites"]), 40)
        undirected = {(min(u, v), max(u, v)) for u, v, _ in self.links}
        self.assertEqual(len(p["links"]), len(undirected))

    def test_routes_coloured_by_rank_with_travel_time(self):
        p = route_map.build_map_payload(self.coords, self.links, self.routes,
                                        origin=2000, destination=970)
        self.assertEqual([r["rank"] for r in p["routes"]], [1, 2])
        self.assertNotEqual(p["routes"][0]["colour"], p["routes"][1]["colour"])
        self.assertEqual(p["routes"][1]["minutes"], 6.25)
        self.assertEqual(len(p["routes"][0]["points"]), 3)
        roles = {s["id"]: s["role"] for s in p["sites"]}
        self.assertEqual((roles[2000], roles[970]), ("origin", "destination"))

    def test_hover_shows_predicted_flow_and_flags(self):
        p = route_map.build_map_payload(self.coords, self.links,
                                        site_flows_15min={2000: 210.04},
                                        unavailable_sites=[4264], clamped_sites=[2000])
        site = {s["id"]: s for s in p["sites"]}
        self.assertEqual(site[2000]["flow15"], 210.0)
        self.assertEqual(site[2000]["flowHour"], 840)
        self.assertTrue(site[2000]["clamped"])
        self.assertTrue(site[4264]["unavailable"])
        self.assertIsNone(site[970]["flow15"])

    def test_site_names_from_approaches(self):
        names = route_map.load_site_names(DEFAULT_SITES)
        self.assertTrue(names[2000].startswith("Warrigal Rd / "))
        self.assertIn("Warrigal Rd", names[3682])

    def test_iframe_round_trip_and_escaping(self):
        p = route_map.build_map_payload(self.coords, self.links, self.routes,
                                        model_label="GRU </script><b>")
        iframe = route_map.build_map_iframe(p)
        self.assertTrue(iframe.startswith("<iframe"))
        self.assertNotIn("<script", iframe)  # page is escaped inside srcdoc
        self.assertEqual(_payload_from_iframe(iframe)["model"], "GRU </script><b>")
        self.assertIn("tile.openstreetmap.org", route_map.build_map_document(p))

    def test_map_and_satellite_backgrounds_switchable(self):
        doc = route_map.build_map_document(route_map.build_map_payload(self.coords, self.links))
        self.assertIn("World_Imagery", doc)          # satellite tiles (no API key)
        self.assertIn("L.control.layers", doc)      # the switcher
        self.assertIn('"Satellite": satelliteWithRoads', doc)
        self.assertIn("Esri", doc)                   # required attribution


if __name__ == "__main__":
    unittest.main()
