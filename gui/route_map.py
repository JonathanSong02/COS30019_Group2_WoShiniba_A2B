"""Interactive OpenStreetMap view for TBRGS route guidance.

Builds a self-contained Leaflet page (no API key, no extra Python packages)
showing the 40 SCATS sites, the map-reviewed road links and the top-k routes.
The GUI embeds it in an <iframe srcdoc=...> so Leaflet's scripts run inside
Gradio; the same HTML can be written to disk for report screenshots.

Coordinates: the VicRoads SCATS latitude/longitude values sit a roughly constant
distance away from the true intersections on OpenStreetMap. The measured shift
is stored in data/processed/scats_coordinate_offset.json and applied to every
site before drawing. It is a display correction only: routing and distances
still use the original processed coordinates.

Limitations (documented in the GUI and report):
- routes are drawn as straight lines between intersections, not along streets;
- the map background and the Leaflet library load from the internet
  (route calculation itself works offline).
"""
from __future__ import annotations

import csv
import html
import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OFFSET_FILE = ROOT / "data" / "processed" / "scats_coordinate_offset.json"
DEFAULT_SITES_FILE = ROOT / "data" / "processed" / "scats_sites.csv"

LEAFLET_VERSION = "1.9.4"
# Colour-blind-safe, high-contrast route colours (rank 1 first).
ROUTE_COLOURS = ["#d7191c", "#2c7bb6", "#1a9641", "#7b3294", "#fdae61"]
MAP_HEIGHT_PX = 560


# --------------------------------------------------------------------------
# Coordinate correction
# --------------------------------------------------------------------------
def load_offset(path: str | Path = DEFAULT_OFFSET_FILE) -> dict:
    """Return {'dlat': float, 'dlon': float, ...}; zero shift if file missing."""
    path = Path(path)
    if not path.is_file():
        return {"dlat": 0.0, "dlon": 0.0, "note": "No offset file; raw SCATS coordinates shown"}
    data = json.loads(path.read_text(encoding="utf-8"))
    dlat, dlon = float(data["dlat"]), float(data["dlon"])
    if abs(dlat) > 0.01 or abs(dlon) > 0.01:  # > ~1 km would be a bad measurement
        raise ValueError(f"Implausible SCATS coordinate offset: {dlat}, {dlon}")
    return {**data, "dlat": dlat, "dlon": dlon}


def corrected_coords(site_coords: Mapping[int, tuple[float, float]],
                     offset: Mapping[str, float]) -> dict[int, tuple[float, float]]:
    """Apply one constant (dlat, dlon) shift to every SCATS site."""
    dlat, dlon = float(offset.get("dlat", 0.0)), float(offset.get("dlon", 0.0))
    return {int(s): (lat + dlat, lon + dlon) for s, (lat, lon) in site_coords.items()}


# --------------------------------------------------------------------------
# Site labels
# --------------------------------------------------------------------------
def _pretty_road(token: str) -> str:
    return token.strip().replace("_", " ").title()


def load_site_names(path: str | Path = DEFAULT_SITES_FILE) -> dict[int, str]:
    """Derive 'Warrigal Rd / Toorak Rd'-style names from the approach list.

    Each approach reads like "TOORAK_RD W of WARRIGAL_RD"; the two road names
    mentioned most often across a site's approaches name the intersection.
    """
    names: dict[int, str] = {}
    path = Path(path)
    if not path.is_file():
        return names
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            counts: Counter[str] = Counter()
            for approach in str(row.get("approaches", "")).split(";"):
                parts = re.split(r"\s+(?:N|S|E|W|NE|NW|SE|SW)\s+of\s+", approach.strip(), flags=re.I)
                for part in parts:
                    if part.strip():
                        counts[_pretty_road(part)] += 1
            names[int(row["scats_site"])] = " / ".join(r for r, _ in counts.most_common(2))
    return names


# --------------------------------------------------------------------------
# Map payload + HTML
# --------------------------------------------------------------------------
def build_map_payload(
    site_coords: Mapping[int, tuple[float, float]],
    road_links: Iterable[tuple[int, int, float]],
    routes: Sequence[Mapping] = (),
    *,
    site_flows_15min: Mapping[int, float] | None = None,
    site_names: Mapping[int, str] | None = None,
    unavailable_sites: Iterable[int] = (),
    clamped_sites: Iterable[int] = (),
    origin: int | None = None,
    destination: int | None = None,
    offset: Mapping[str, float] | None = None,
    model_label: str = "",
) -> dict:
    """Return the JSON-serialisable data the Leaflet page draws."""
    offset = dict(offset if offset is not None else load_offset())
    coords = corrected_coords(site_coords, offset)
    flows = {int(k): float(v) for k, v in (site_flows_15min or {}).items()}
    names = {int(k): v for k, v in (site_names or {}).items()}
    unavailable = {int(s) for s in unavailable_sites}
    clamped = {int(s) for s in clamped_sites}

    sites = []
    for site, (lat, lon) in sorted(coords.items()):
        sites.append({
            "id": site, "lat": round(lat, 6), "lon": round(lon, 6),
            "name": names.get(site, ""),
            "flow15": round(flows[site], 1) if site in flows else None,
            "flowHour": round(flows[site] * 4, 0) if site in flows else None,
            "unavailable": site in unavailable,
            "clamped": site in clamped,
            "role": ("origin" if site == origin else
                     "destination" if site == destination else ""),
        })

    seen: set[tuple[int, int]] = set()
    links = []
    for u, v, dist in road_links:
        u, v = int(u), int(v)
        key = (min(u, v), max(u, v))
        if key in seen or u not in coords or v not in coords:
            continue
        seen.add(key)
        links.append({"a": coords[u], "b": coords[v], "label": f"{u} ↔ {v} · {float(dist):.2f} km"})

    drawn_routes = []
    for i, route in enumerate(routes[: len(ROUTE_COLOURS)]):
        path = [int(s) for s in route["scats_sites"]]
        if any(s not in coords for s in path):
            continue
        drawn_routes.append({
            "rank": int(route.get("rank", i + 1)),
            "colour": ROUTE_COLOURS[i],
            "minutes": round(float(route["travel_time_minutes"]), 2),
            "path": path,
            "points": [coords[s] for s in path],
        })

    return {
        "sites": sites, "links": links, "routes": drawn_routes,
        "offset": {"dlat": offset.get("dlat", 0.0), "dlon": offset.get("dlon", 0.0)},
        "model": model_label,
    }


_PAGE = """<!DOCTYPE html>
<html><head><meta charset="utf-8">
<link rel="stylesheet" href="https://unpkg.com/leaflet@__LV__/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@__LV__/dist/leaflet.js"></script>
<style>
 html,body,#map{height:100%;margin:0;font-family:system-ui,Segoe UI,Arial,sans-serif}
 .legend{background:#fff;padding:8px 10px;border-radius:6px;box-shadow:0 1px 4px rgba(0,0,0,.3);font-size:12px;line-height:1.5}
 .legend .sw{display:inline-block;width:18px;height:4px;margin-right:6px;vertical-align:middle;border-radius:2px}
 .offline{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;padding:20px;text-align:center;color:#555;font-size:14px}
</style></head><body>
<div id="map"><div class="offline">Map needs an internet connection (OpenStreetMap tiles and Leaflet).<br>Routes are still listed in the table.</div></div>
<script>
const D = __DATA__;
if (typeof L === "undefined") { /* offline: leave the message */ } else {
document.getElementById("map").innerHTML = "";
const map = L.map("map", {preferCanvas: false});
const streetMap = L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
  maxZoom: 19, attribution: "&copy; OpenStreetMap contributors"}).addTo(map);
const satellite = L.tileLayer(
  "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", {
  maxZoom: 19, attribution: "Imagery &copy; Esri, Maxar, Earthstar Geographics, and the GIS User Community"});
const roadNames = L.tileLayer(
  "https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Transportation/MapServer/tile/{z}/{y}/{x}", {
  maxZoom: 19, attribution: "Labels &copy; Esri"});
const satelliteWithRoads = L.layerGroup([satellite, roadNames]);

const linkLayer = L.layerGroup().addTo(map);
D.links.forEach(l => L.polyline([l.a, l.b], {color: "#6b6b6b", weight: 2, opacity: 0.45, dashArray: "4 4"})
  .bindTooltip(l.label, {sticky: true}).addTo(linkLayer));

const routeLayer = L.layerGroup().addTo(map);
[...D.routes].reverse().forEach(r => {
  const w = r.rank === 1 ? 7 : 5;
  L.polyline(r.points, {color: "#ffffff", weight: w + 3, opacity: 0.9}).addTo(routeLayer);
  L.polyline(r.points, {color: r.colour, weight: w, opacity: 0.95})
    .bindTooltip(`<b>Route ${r.rank}</b>: ${r.minutes.toFixed(2)} min<br>${r.path.join(" → ")}`, {sticky: true})
    .addTo(routeLayer);
});

const siteLayer = L.layerGroup().addTo(map);
D.sites.forEach(s => {
  let fill = "#ffffff", stroke = "#333", radius = 6;
  if (s.unavailable) { fill = "#bdbdbd"; stroke = "#777"; }
  if (s.role === "origin") { fill = "#1a9641"; radius = 9; }
  if (s.role === "destination") { fill = "#000000"; radius = 9; }
  let tip = `<b>SCATS ${s.id}</b>` + (s.name ? `<br>${s.name}` : "");
  if (s.flow15 !== null) tip += `<br>Predicted: ${s.flow15} veh / 15 min (≈${s.flowHour} veh/h)`;
  if (s.clamped) tip += `<br><i>Clamped to 1,500 veh/h</i>`;
  if (s.unavailable) tip += `<br><i>No forecast available (excluded)</i>`;
  if (s.role) tip += `<br><b>${s.role === "origin" ? "Origin" : "Destination"}</b>`;
  L.circleMarker([s.lat, s.lon], {radius, color: stroke, weight: 2, fillColor: fill, fillOpacity: 1})
    .bindTooltip(tip).addTo(siteLayer);
  if (s.role) L.marker([s.lat, s.lon], {opacity: 0}).bindTooltip(String(s.id),
    {permanent: true, direction: "top", offset: [0, -6]}).addTo(siteLayer);
});

const legend = L.control({position: "bottomright"});
legend.onAdd = () => {
  const div = L.DomUtil.create("div", "legend");
  let h = D.model ? `<b>${D.model}</b><br>` : "";
  D.routes.forEach(r => h += `<span class="sw" style="background:${r.colour}"></span>Route ${r.rank} · ${r.minutes.toFixed(1)} min<br>`);
  h += `<span class="sw" style="background:#6b6b6b"></span>Road link (map-reviewed)<br>`;
  h += `<span style="font-size:11px;color:#666">Sites shifted ${(D.offset.dlat*111320).toFixed(0)} m N, `
     + `${(D.offset.dlon*111320*Math.cos(37.82*Math.PI/180)).toFixed(0)} m E to match OSM.<br>`
     + `Routes drawn as straight lines between intersections.</span>`;
  div.innerHTML = h; return div;
};
legend.addTo(map);

// Top-right switcher: Map / Satellite background, and show/hide each layer.
L.control.layers(
  {"Map (OpenStreetMap)": streetMap, "Satellite": satelliteWithRoads},
  {"Top routes": routeLayer, "SCATS sites": siteLayer, "Road links": linkLayer},
  {position: "topright", collapsed: false}).addTo(map);

const pts = D.routes.length ? D.routes.flatMap(r => r.points) : D.sites.map(s => [s.lat, s.lon]);
map.fitBounds(L.latLngBounds(pts).pad(0.15));
}
</script></body></html>"""


def build_map_document(payload: Mapping) -> str:
    """Full standalone HTML page for the payload."""
    data = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    return _PAGE.replace("__LV__", LEAFLET_VERSION).replace("__DATA__", data)


def build_map_iframe(payload: Mapping, height: int = MAP_HEIGHT_PX) -> str:
    """HTML snippet for gr.HTML: the page inside an iframe so its scripts run."""
    doc = html.escape(build_map_document(payload), quote=True)
    return (f'<iframe title="TBRGS route map" srcdoc="{doc}" '
            f'style="width:100%;height:{int(height)}px;border:1px solid #ccc;border-radius:8px" '
            'loading="eager" referrerpolicy="no-referrer"></iframe>')


def save_map_html(payload: Mapping, path: str | Path) -> Path:
    """Write a standalone map page (for report screenshots)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_map_document(payload), encoding="utf-8")
    return path
