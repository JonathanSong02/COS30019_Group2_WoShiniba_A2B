"""Check every SCATS road link against OpenStreetMap driving routes (Task 7).

For each link in the road graph (build_road_graph.ROAD_LINKS) this asks the
public OSRM routing service, which routes on OpenStreetMap road data, for the
driving route between the two sites, in both directions. It then compares:

  * the roads the route uses  vs  the road our link claims (e.g. "Warrigal Rd")
  * the route's length        vs  our straight-line / L-shaped distance

and writes one row per link to data/processed/scats_road_edges_review.csv.

Site positions are the SCATS coordinates shifted by the measured offset
(data/processed/scats_coordinate_offset.json), so they sit on the real
intersections. OSRM moves ("snaps") each point to the nearest road; next to a
freeway that can be the freeway itself, which gives a long detour that says
nothing about our link. For the links listed in SNAP_TO_ROAD, each end is
first moved onto the named road using OpenStreetMap's own map data (the
nearest node of a road with that name), and the route is requested between
those points.

Run from the project root (needs internet, takes about 3 minutes because the
public servers allow about one request per second):
    python -m data_processing.review_road_links
Raw answers are cached in data/processed/road_link_review_osrm.json, so a
second run is instant; use --refresh to ask the servers again.

Services used (free, no API key):
  OSRM servers      https://router.project-osrm.org, then
                    https://routing.openstreetmap.de/routed-car   (route/v1/driving)
  OSM editing API   https://api.openstreetmap.org     (api/0.6/map, for snapping)
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

from data_processing.build_road_graph import (
    MEASURED_KM, ROAD_LINKS, build_edges, haversine_km, l_shape_km, load_sites)

ROOT = Path(__file__).resolve().parents[1]
SITES = ROOT / "data" / "processed" / "scats_sites.csv"
OFFSET = ROOT / "data" / "processed" / "scats_coordinate_offset.json"
REVIEW = ROOT / "data" / "processed" / "scats_road_edges_review.csv"
CACHE = ROOT / "data" / "processed" / "road_link_review_osrm.json"

# OSRM servers, tried in order: the OSRM project's demo server, then the FOSSGIS
# server that openstreetmap.org's own "Directions" uses. Both route cars on
# OpenStreetMap data with the same OSRM software.
OSRM_SERVERS = ["https://router.project-osrm.org", "https://routing.openstreetmap.de/routed-car"]
OSRM = "{server}/route/v1/driving/{lon1},{lat1};{lon2},{lat2}?overview=false&steps=true{extra}"
RETRIES = 3
OSM_MAP = "https://api.openstreetmap.org/api/0.6/map.json?bbox={w},{s},{e},{n}"
USER_AGENT = "COS30019-A2B-TBRGS-road-link-review (student project)"
REQUEST_GAP_S = 1.1       # public servers: about one request per second
VERIFIED_RANGE = (0.8, 1.2)
MAIN_ROAD_SHARE = 0.15    # a road is listed if it carries >= 15% of the route

# Links checked with both ends moved onto the named road first (OSM road names).
# These are the sites next to a freeway, plus Camberwell Junction, where the
# plain check snapped onto the wrong road.
SNAP_TO_ROAD = {
    (2825, 2827): ("Burke Road", "Bulleen Road"),
    (2827, 4051): ("Bulleen Road", "Bulleen Road"),
    (2820, 2825): ("Princess Street", "Burke Road"),
    (2846, 4043): ("High Street", "Burke Road"),
    (970, 2846): ("High Street Road", "High Street"),
    (3812, 4040): ("Camberwell Road", "Camberwell Road"),
}

# Explanations for links whose result needs more than the status word.
NOTES = {
    (2825, 2827): ("WRONG_ROAD_FIXED", "Labelled Eastern Fwy, but OSM has no freeway connection between the Burke Rd "
                   "and Bulleen Rd interchanges; the shortest drive uses local streets (The Boulevard, Aquila St). "
                   "Graph now uses the measured distance (MEASURED_KM in build_road_graph.py)."),
    (2827, 4051): ("VERIFIED", "Whole route on Bulleen Rd once both ends are snapped to Bulleen Rd; unsnapped, the "
                   "router started on the Eastern Fwy and reported a 5.7-9.8 km detour."),
    (2820, 2825): ("DISTANCE_SHORT", "Connection confirmed via Chandler Hwy, Eastern Fwy and the Burke Rd ramps; "
                   "the ramps make the drive longer than the straight line."),
    (2000, 3812): ("DISTANCE_LONG", "Route confirmed on Camberwell Rd and Toorak Rd; the L-shaped estimate "
                   "overstates it because Camberwell Rd runs diagonally."),
    (3812, 4043): ("VERIFIED", "Shortest drive uses Seymour Gv and Burke Rd rather than the labelled Camberwell Rd "
                   "+ Toorak Rd; the distance agrees with the estimate."),
    (4270, 4273): ("VERIFIED", "Router prefers Auburn Rd or Tooronga Rd + Riversdale Rd over the labelled "
                   "Glenferrie Rd + Toorak Rd; all are about 2.9-3.0 km, matching the estimate."),
    (3812, 4040): ("VERIFIED", "Whole route on Camberwell Rd after snapping both ends to Camberwell Rd."),
}


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
def corrected_sites() -> dict[int, tuple[float, float]]:
    off = json.loads(OFFSET.read_text(encoding="utf-8"))
    return {s: (lat + off["dlat"], lon + off["dlon"]) for s, (lat, lon) in load_sites(SITES).items()}


def http_json(url: str) -> dict:
    """GET a JSON answer, retrying dropped connections (public servers drop some)."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(1, RETRIES + 1):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, ConnectionError, TimeoutError) as exc:
            if attempt == RETRIES:
                raise
            print(f"  connection problem ({exc}); retrying in {2 * attempt} s")
            time.sleep(2 * attempt)


def short_road(name: str) -> str:
    for long, short in ((" Road", " Rd"), (" Street", " St"), (" Freeway", " Fwy"), (" Highway", " Hwy")):
        if name.endswith(long):
            return name[: -len(long)] + short
    return name


def main_roads(steps: list[dict]) -> str:
    """Roads carrying at least MAIN_ROAD_SHARE of the route, longest first."""
    by_name: dict[str, float] = {}
    for st in steps:
        name = st.get("name") or st.get("ref") or "(unnamed)"
        by_name[name] = by_name.get(name, 0.0) + float(st.get("distance", 0.0))
    total = sum(by_name.values()) or 1.0
    ranked = sorted(by_name.items(), key=lambda kv: -kv[1])
    return "+".join(short_road(n) for n, d in ranked if d / total >= MAIN_ROAD_SHARE)


def nearest_node_on_road(point: tuple[float, float], road: str, fetch=http_json, box_deg: float = 0.004):
    """Nearest OSM node of a highway named `road` around `point` -> ((lat, lon), metres away)."""
    lat, lon = point
    data = fetch(OSM_MAP.format(w=lon - box_deg, s=lat - box_deg, e=lon + box_deg, n=lat + box_deg))
    nodes = {e["id"]: (e["lat"], e["lon"]) for e in data["elements"] if e["type"] == "node"}
    best, best_m = None, math.inf
    for e in data["elements"]:
        if e["type"] == "way" and e.get("tags", {}).get("highway") and e["tags"].get("name") == road:
            for n in e["nodes"]:
                if n in nodes:
                    m = haversine_km(lat, lon, *nodes[n]) * 1000
                    if m < best_m:
                        best, best_m = nodes[n], m
    if best is None:
        raise ValueError(f"No OSM road named {road!r} near {point}")
    return best, best_m


def route(a: tuple[float, float], b: tuple[float, float], fetch=http_json, snapped: bool = False) -> dict:
    """OSRM driving route a -> b: {'km', 'minutes', 'roads', 'server'}.

    Tries each server in OSRM_SERVERS until one answers; a server that answers
    with an error (e.g. no route) is reported, not skipped.
    """
    extra = "&radiuses=5;5" if snapped else ""   # snapped points are already on the road
    last_error = None
    for server in OSRM_SERVERS:
        try:
            data = fetch(OSRM.format(server=server, lat1=a[0], lon1=a[1], lat2=b[0], lon2=b[1], extra=extra))
        except (urllib.error.URLError, ConnectionError, TimeoutError) as exc:
            last_error = exc
            print(f"  {server} unreachable ({exc}); trying the next server")
            continue
        if data.get("code") != "Ok":
            raise ValueError(f"OSRM answered {data.get('code')}: {data.get('message', '')}")
        r = data["routes"][0]
        return {"km": round(r["distance"] / 1000, 3), "minutes": round(r["duration"] / 60, 2),
                "roads": main_roads(r["legs"][0]["steps"]), "server": server}
    raise ConnectionError(f"No OSRM server reachable: {last_error}. Check your internet connection, "
                          "or whether antivirus/firewall software blocks Python.")


# --------------------------------------------------------------------------
# review
# --------------------------------------------------------------------------
def classify(a: int, b: int, ratio: float) -> tuple[str, str]:
    if (a, b) in NOTES or (b, a) in NOTES:
        return NOTES.get((a, b)) or NOTES[(b, a)]
    low, high = VERIFIED_RANGE
    if low <= ratio <= high:
        return "VERIFIED", ""
    if ratio > high:
        return "DISTANCE_SHORT", "Route confirmed on the named road; the road curves or jogs, so the straight-line estimate is short."
    return "DISTANCE_LONG", "Road distance is shorter than the estimate."


def review(fetch=http_json, cache: dict | None = None, pause: float = REQUEST_GAP_S) -> tuple[list[dict], dict]:
    """Check every link; returns (review rows, raw answers keyed by request)."""
    cache = {} if cache is None else cache
    sites = corrected_sites()
    graph_km = {(r["from_site"], r["to_site"]): r["distance_km"] for r in build_edges(load_sites(SITES))}

    def cached(key, fn):
        if key not in cache:
            cache[key] = fn()
            time.sleep(pause)
        return cache[key]

    rows = []
    for i, (a, b, road, kind, conf) in enumerate(ROAD_LINKS, 1):
        print(f"[{i}/{len(ROAD_LINKS)}] {a}-{b} {road}")
        (la, oa), (lb, ob) = sites[a], sites[b]
        # the estimate our method gives (for the measured link: its straight line before the review)
        estimate = l_shape_km(la, oa, lb, ob) if kind == "junction" else haversine_km(la, oa, lb, ob)
        snap_roads = SNAP_TO_ROAD.get((a, b)) or tuple(reversed(SNAP_TO_ROAD.get((b, a), ()))) or None
        pa, pb = sites[a], sites[b]
        if snap_roads:
            pa = tuple(cached(f"snap {a} {snap_roads[0]}", lambda: nearest_node_on_road(sites[a], snap_roads[0], fetch))[0])
            pb = tuple(cached(f"snap {b} {snap_roads[1]}", lambda: nearest_node_on_road(sites[b], snap_roads[1], fetch))[0])
        ab = cached(f"route {a}->{b}", lambda: route(pa, pb, fetch, bool(snap_roads)))
        ba = cached(f"route {b}->{a}", lambda: route(pb, pa, fetch, bool(snap_roads)))
        # the shorter direction: the longer one is often a turn-ban or one-way detour
        osm_km = min(ab["km"], ba["km"])
        ratio = osm_km / estimate
        status, note = classify(a, b, ratio)
        rows.append({
            "site_a": a, "site_b": b, "road": road, "link_type": kind, "confidence": conf,
            "estimate_km_before_review": round(estimate, 3), "graph_km_now": graph_km[(a, b)],
            "osm_km_a_to_b": ab["km"], "osm_km_b_to_a": ba["km"], "osm_km_shorter": osm_km,
            "osm_over_estimate": round(ratio, 2), "osm_roads_a_to_b": ab["roads"], "osm_roads_b_to_a": ba["roads"],
            "endpoint_method": "snapped_to_named_road" if snap_roads else "corrected_site_coordinates",
            "router": ab.get("server", ""),
            "status": status, "notes": note,
        })
    return rows, cache


def summary(rows: list[dict]) -> str:
    counts = Counter(r["status"] for r in rows)
    ratios = [r["osm_over_estimate"] for r in rows if r["status"] != "WRONG_ROAD_FIXED"]
    lines = [f"{len(rows)} links: " + ", ".join(f"{k} {v}" for k, v in counts.most_common()),
             f"OSM / estimate over {len(ratios)} links: median {statistics.median(ratios):.3f}, "
             f"mean {statistics.fmean(ratios):.3f}, within 10%: {sum(0.9 <= x <= 1.1 for x in ratios)}, "
             f"within 20%: {sum(0.8 <= x <= 1.2 for x in ratios)}"]
    lines += [f"  {r['site_a']}-{r['site_b']} {r['status']}: estimate {r['estimate_km_before_review']} km, "
              f"OSM {r['osm_km_shorter']} km ({r['road']})" for r in rows if r["status"] != "VERIFIED"]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--refresh", action="store_true", help="ignore the cache and ask the servers again")
    args = ap.parse_args()
    cache = {} if args.refresh or not CACHE.is_file() else json.loads(CACHE.read_text(encoding="utf-8"))
    try:
        rows, cache = review(cache=cache)
    finally:   # keep whatever was fetched, even if a request fails part-way
        CACHE.write_text(json.dumps(cache, indent=1), encoding="utf-8")
    with REVIEW.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(summary(rows))
    print(f"wrote {REVIEW.relative_to(ROOT)} and {CACHE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
