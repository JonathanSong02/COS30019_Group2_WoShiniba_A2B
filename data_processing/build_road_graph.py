"""Build the Boroondara SCATS road graph (Darren).

Output: data/processed/scats_road_edges.csv  ->  from_site,to_site,distance_km
        data/processed/scats_road_edges_detailed.csv (adds road, link type, confidence)

How the edges were chosen
-------------------------
Edges are NOT based on proximity. Each link below was taken from the road
names in the SCATS "Location" field (e.g. "WARRIGAL_RD N of HIGH STREET_RD")
and checked against the VicRoads Boroondara map (IBM data_Bor.pdf):

* "direct"   - two sites that are next to each other on the same road, with
               no other site from our 40-site dataset between them.
               Distance = great-circle (haversine) distance between the sites.
* "junction" - the road leaving a site meets another road at an intersection
               that is NOT in our dataset (one unsignalised/unsampled junction
               only), then reaches the next site. Distance = |north-south| +
               |east-west| km (an L-shaped estimate for two near-perpendicular
               roads).

Every road is treated as two-way, so each link is written in both directions.

Limitations (mention in the report)
* Distances are straight-line / L-shaped estimates, not measured along curves.
* SCATS coordinates are offset slightly from the true map position; the offset
  is roughly the same for all sites, so distances are not materially affected.
* Links marked confidence "medium" go through freeways or road continuations
  that should be double-checked on a street map.

Run:  python -m data_processing.build_road_graph
"""

import argparse
import csv
import math
from collections import defaultdict, deque
from pathlib import Path

# (from_site, to_site, road, link_type, confidence)
ROAD_LINKS = [
    # Burke Rd, north -> south
    (2825, 4030, "Burke Rd", "direct", "high"),
    (4030, 4032, "Burke Rd", "direct", "high"),
    (4032, 4034, "Burke Rd", "direct", "high"),
    (4034, 4035, "Burke Rd", "direct", "high"),
    (4035, 3120, "Burke Rd", "direct", "high"),
    (3120, 4040, "Burke Rd", "direct", "high"),
    (4040, 4043, "Burke Rd", "direct", "high"),
    # Union Rd / Warrigal Rd, north -> south
    (2200, 3126, "Union Rd / Warrigal Rd", "direct", "high"),
    (3126, 3682, "Warrigal Rd", "direct", "high"),
    (3682, 2000, "Warrigal Rd", "direct", "high"),
    (2000, 3685, "Warrigal Rd", "direct", "high"),
    (3685, 970, "Warrigal Rd", "direct", "high"),
    # Canterbury Rd, west -> east
    (3120, 3122, "Canterbury Rd", "direct", "high"),
    (3122, 3127, "Canterbury Rd", "direct", "high"),
    (3127, 3126, "Canterbury Rd", "direct", "high"),
    # Balwyn Rd, north -> south
    (3180, 4057, "Balwyn Rd", "direct", "high"),
    (4057, 4063, "Balwyn Rd", "direct", "high"),
    (4063, 3127, "Balwyn Rd", "direct", "high"),
    # Cotham Rd / Whitehorse Rd / Maroondah Hwy, west -> east
    (3662, 4324, "Cotham Rd", "direct", "high"),
    (4324, 4034, "Cotham Rd", "direct", "high"),
    (4034, 4063, "Whitehorse Rd", "direct", "high"),
    (4063, 2200, "Whitehorse Rd / Maroondah Hwy", "direct", "high"),
    # Harp Rd / Belmore Rd
    (4321, 4032, "Harp Rd", "direct", "high"),
    (4032, 4057, "Belmore Rd", "direct", "high"),
    # High St (Kew) / Doncaster Rd, south-west -> north-east
    (3001, 3662, "High St (Kew)", "direct", "high"),
    (3662, 4335, "High St (Kew)", "direct", "high"),
    (4335, 4321, "High St (Kew)", "direct", "high"),
    (4321, 4030, "High St (Kew)", "direct", "high"),
    (4030, 4051, "Doncaster Rd", "direct", "high"),
    (4051, 3180, "Doncaster Rd", "direct", "high"),
    # Bulleen Rd
    (4051, 2827, "Bulleen Rd", "direct", "high"),
    # Eastern Freeway (via interchanges / ramps)
    (2820, 2825, "Eastern Fwy", "direct", "medium"),
    (2825, 2827, "Eastern Fwy", "direct", "medium"),
    # Princess St, Denmark St, Power St, Church St
    (2820, 3662, "Princess St", "direct", "high"),
    (3662, 3002, "Denmark St", "direct", "high"),
    (3002, 4263, "Power St", "direct", "high"),
    (3001, 4262, "Church St", "direct", "high"),
    # Victoria St / Barkers Rd, west -> east
    (4821, 3001, "Victoria St / Barkers Rd", "direct", "high"),
    (3001, 3002, "Barkers Rd", "direct", "high"),
    (3002, 4035, "Barkers Rd", "direct", "high"),
    # Burwood Rd, west -> east (meets Camberwell Junction)
    (4262, 4263, "Burwood Rd", "direct", "high"),
    (4263, 4264, "Burwood Rd", "direct", "high"),
    (4264, 4266, "Burwood Rd", "direct", "high"),
    (4266, 4040, "Burwood Rd", "direct", "high"),
    # Glenferrie Rd, north -> south
    (4324, 4264, "Glenferrie Rd", "direct", "high"),
    (4264, 4270, "Glenferrie Rd", "direct", "high"),
    # Swan St / Riversdale Rd, west -> east
    (4812, 4270, "Swan St / Riversdale Rd", "direct", "medium"),
    (4270, 4272, "Riversdale Rd", "direct", "high"),
    (4272, 4040, "Riversdale Rd", "direct", "high"),
    (4040, 3804, "Riversdale Rd", "direct", "high"),
    (3804, 3682, "Riversdale Rd", "direct", "high"),
    # Trafalgar Rd, Camberwell Rd, Tooronga Rd
    (3804, 3812, "Trafalgar Rd", "direct", "high"),
    (4040, 3812, "Camberwell Rd", "direct", "high"),
    (4272, 4273, "Tooronga Rd", "direct", "high"),
    # Toorak Rd, west -> east
    (4273, 4043, "Toorak Rd", "direct", "high"),
    (4043, 2000, "Toorak Rd", "direct", "high"),
    # High St (Glen Iris) / High Street Rd
    (2846, 970, "High St / High Street Rd", "direct", "medium"),
    # Links through one junction that is not in the dataset
    (4043, 2846, "Burke Rd + High St (Glen Iris)", "junction", "medium"),
    (3812, 4043, "Camberwell Rd + Toorak Rd", "junction", "medium"),
    (3812, 2000, "Camberwell Rd + Toorak Rd", "junction", "medium"),
    (3120, 4266, "Rathmines Rd + Auburn Rd", "junction", "medium"),
    (4266, 4270, "Auburn Rd + Riversdale Rd", "junction", "medium"),
    (4266, 4272, "Auburn Rd + Riversdale Rd", "junction", "medium"),
    (4263, 4270, "Power St + Riversdale Rd", "junction", "medium"),
    (4263, 4812, "Power St + Swan St", "junction", "medium"),
    (4262, 4821, "Bridge Rd + Burnley St", "junction", "medium"),
    (4821, 4812, "Burnley St + Swan St", "junction", "medium"),
    (4270, 4273, "Glenferrie Rd + Toorak Rd", "junction", "medium"),
]

EARTH_RADIUS_KM = 6371.0088


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def l_shape_km(lat1, lon1, lat2, lon2):
    ns = haversine_km(lat1, lon1, lat2, lon1)
    ew = haversine_km(lat2, lon1, lat2, lon2)
    return ns + ew


def load_sites(path):
    sites = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            sites[int(row["scats_site"])] = (float(row["latitude"]), float(row["longitude"]))
    return sites


def build_edges(sites, links=ROAD_LINKS):
    rows = []
    seen = set()
    for a, b, road, kind, conf in links:
        if a not in sites or b not in sites:
            raise ValueError(f"Unknown SCATS site in link {a}-{b}")
        if a == b or frozenset((a, b)) in seen:
            raise ValueError(f"Self-loop or duplicate link {a}-{b}")
        seen.add(frozenset((a, b)))
        (la, oa), (lb, ob) = sites[a], sites[b]
        dist = haversine_km(la, oa, lb, ob) if kind == "direct" else l_shape_km(la, oa, lb, ob)
        for s, t in ((a, b), (b, a)):  # two-way roads
            rows.append({"from_site": s, "to_site": t, "distance_km": round(dist, 3),
                         "road": road, "link_type": kind, "confidence": conf})
    rows.sort(key=lambda r: (r["from_site"], r["to_site"]))
    return rows


def validate(sites, rows, min_km=0.2, max_km=5.0):
    adj = defaultdict(set)
    for r in rows:
        adj[r["from_site"]].add(r["to_site"])
        if not (min_km <= r["distance_km"] <= max_km):
            raise ValueError(f"Implausible distance {r}")
    isolated = set(sites) - set(adj)
    if isolated:
        raise ValueError(f"Sites with no road connection: {sorted(isolated)}")
    for s in adj:  # every edge has a reverse edge
        for t in adj[s]:
            if s not in adj[t]:
                raise ValueError(f"Missing reverse edge {t}->{s}")
    start = next(iter(sites))
    seen, q = {start}, deque([start])
    while q:
        for n in adj[q.popleft()]:
            if n not in seen:
                seen.add(n)
                q.append(n)
    if seen != set(sites):
        raise ValueError(f"Graph not connected; unreachable: {sorted(set(sites) - seen)}")
    return adj


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sites", default="data/processed/scats_sites.csv")
    ap.add_argument("--out", default="data/processed/scats_road_edges.csv")
    args = ap.parse_args()

    sites = load_sites(args.sites)
    rows = build_edges(sites)
    adj = validate(sites, rows)

    out = Path(args.out)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["from_site", "to_site", "distance_km"], extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    detailed = out.with_name(out.stem + "_detailed.csv")
    with open(detailed, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    d = [r["distance_km"] for r in rows]
    print(f"{len(sites)} sites, {len(rows)} directed edges ({len(rows)//2} two-way roads)")
    print(f"degree min/max: {min(map(len, adj.values()))}/{max(map(len, adj.values()))}")
    print(f"distance km min/mean/max: {min(d):.3f}/{sum(d)/len(d):.3f}/{max(d):.3f}")
    print(f"wrote {out} and {detailed}")


if __name__ == "__main__":
    main()
