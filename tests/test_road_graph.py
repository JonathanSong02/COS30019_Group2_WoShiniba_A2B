import csv
import unittest
from pathlib import Path

from data_processing.build_road_graph import (
    ROAD_LINKS, build_edges, haversine_km, load_sites, validate)

ROOT = Path(__file__).resolve().parents[1]
SITES = ROOT / "data" / "processed" / "scats_sites.csv"
EDGES = ROOT / "data" / "processed" / "scats_road_edges.csv"


class TestRoadGraph(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sites = load_sites(SITES)
        cls.rows = build_edges(cls.sites)

    def test_haversine_known_distance(self):
        # 0.01 deg latitude is about 1.11 km
        self.assertAlmostEqual(haversine_km(-37.8, 145.0, -37.81, 145.0), 1.112, places=2)

    def test_graph_valid_connected_two_way(self):
        validate(self.sites, self.rows)  # raises on any problem

    def test_every_link_written_both_ways(self):
        self.assertEqual(len(self.rows), 2 * len(ROAD_LINKS))

    def test_no_proximity_only_shortcut(self):
        # 2000 and 3685 are neighbours on Warrigal Rd; 2000 and 4273 are not road-adjacent
        pairs = {(r["from_site"], r["to_site"]) for r in self.rows}
        self.assertIn((2000, 3685), pairs)
        self.assertNotIn((2000, 4273), pairs)

    def test_committed_csv_matches_generator(self):
        with open(EDGES, newline="") as f:
            reader = csv.DictReader(f)
            self.assertEqual(reader.fieldnames, ["from_site", "to_site", "distance_km"])
            saved = [(int(r["from_site"]), int(r["to_site"]), float(r["distance_km"])) for r in reader]
        built = [(r["from_site"], r["to_site"], r["distance_km"]) for r in self.rows]
        self.assertEqual(saved, built)


if __name__ == "__main__":
    unittest.main()
