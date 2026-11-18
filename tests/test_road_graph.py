import csv
import unittest
from pathlib import Path

from data_processing.build_road_graph import (
    ROAD_LINKS, build_edges, haversine_km, load_sites, validate)

ROOT = Path(__file__).resolve().parents[1]
SITES = ROOT / "data" / "processed" / "scats_sites.csv"
EDGES = ROOT / "data" / "processed" / "scats_road_edges.csv"
REVIEW = ROOT / "data" / "processed" / "scats_road_edges_review.csv"


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

    def test_reviewed_eastern_fwy_shortcut_uses_measured_distance(self):
        # OSM review: no freeway link between the Burke Rd and Bulleen Rd
        # interchanges, so 2825-2827 uses the measured local-street distance.
        row = next(r for r in self.rows if (r["from_site"], r["to_site"]) == (2825, 2827))
        self.assertEqual(row["link_type"], "measured")
        self.assertAlmostEqual(row["distance_km"], 2.258)
        self.assertNotIn("Eastern Fwy", row["road"])

    def test_review_covers_every_link(self):
        with open(REVIEW, newline="", encoding="utf-8") as f:
            review = list(csv.DictReader(f))
        reviewed = {frozenset((int(r["site_a"]), int(r["site_b"]))) for r in review}
        self.assertEqual(reviewed, {frozenset((a, b)) for a, b, *_ in ROAD_LINKS})
        self.assertTrue(all(r["status"] for r in review))
        by_pair = {frozenset((int(r["site_a"]), int(r["site_b"]))): r for r in review}
        for a, b, *_ in ROAD_LINKS:   # graph distance recorded in the review matches the graph
            built = next(x["distance_km"] for x in self.rows if (x["from_site"], x["to_site"]) == (a, b))
            self.assertAlmostEqual(float(by_pair[frozenset((a, b))]["graph_km_now"]), built)


if __name__ == "__main__":
    unittest.main()
