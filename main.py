"""COS30019 A2B entry point: real traffic prediction and SCATS route integration.

The prediction command works with existing GRU checkpoints now. The route
command REQUIRES Darren's verified directed road links. It never creates roads
from geographic proximity and never substitutes synthetic traffic silently.

Traffic input to the travel-time model must be VEHICLES PER HOUR. A single
15-minute GRU forecast is NOT an hourly forecast; the optional steady-hour
assumption (4x) is an explicitly disclosed approximation for prototyping.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

from routing.scats_adapter import build_weighted_scats_graph
from routing.top_k_routes import find_top_k_routes
from routing.travel_time import (
    CAPACITY_VEH_PER_HOUR,
    weights_from_road_links,
)

ROOT = Path(__file__).resolve().parent
DEFAULT_CSV = ROOT / "data" / "processed" / "scats_15min.csv"
DEFAULT_SITES = ROOT / "data" / "processed" / "scats_sites.csv"
DEFAULT_ROADS = ROOT / "data" / "processed" / "scats_road_edges.csv"  # Darren's map-reviewed graph
DEFAULT_MODELS = ROOT / "saved_models"


def load_config(path: str | Path = ROOT / "config.json") -> dict:
    """Read the group's default settings without modifying its config file."""
    with Path(path).open(encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("default_model", "gru").lower() not in {"gru", "lstm", "regression_tree"}:
        raise ValueError("config.json default_model must be GRU, LSTM or Regression_Tree")
    if config.get("volume_used_for_edge", "origin") not in {"origin", "destination"}:
        raise ValueError("config.json volume_used_for_edge must be origin or destination")
    return config


def valid_as_of(value: str | datetime) -> datetime:
    try:
        when = datetime.fromisoformat(value) if isinstance(value, str) else value
    except ValueError as exc:
        raise ValueError("--time must be ISO format, e.g. 2006-10-20T10:00:00") from exc
    if (not isinstance(when, datetime) or when.tzinfo is not None or
            when.minute % 15 != 0 or when.second != 0 or when.microsecond != 0):
        raise ValueError("--time must be a local naive datetime aligned to 15-minute intervals")
    return when


def predict_traffic(
    scats_site: int | str,
    as_of_time: str | datetime,
    model_name: str = "gru",
    horizon_minutes: int = 15,
    sequence_length: int = 8,
    processed_csv: str | Path = DEFAULT_CSV,
    model_dir: str | Path = DEFAULT_MODELS,
    scaler_path: str | Path | None = None,
) -> float:
    """Return predicted traffic COUNT in a future 15-minute interval.

    Input time is the LAST observed interval, not the forecast target time.
    No fallback to an untrained model or a guessed traffic volume.
    """
    as_of = valid_as_of(as_of_time)
    if horizon_minutes not in (15, 30, 60) or type(horizon_minutes) is not int:
        raise ValueError("horizon_minutes must be 15, 30 or 60")
    if type(sequence_length) is not int or sequence_length < 1:
        raise ValueError("sequence_length must be a positive integer")
    name = model_name.strip().lower()
    if name == "gru":
        from models.gru_model import predict_gru
        output = predict_gru(
            scats_site, as_of, horizon_minutes, processed_csv,
            model_dir=model_dir, sequence_length=sequence_length,
            scaler_path=scaler_path,
        )
    elif name == "lstm":
        from models.lstm_model import predict_lstm
        output = predict_lstm(
            int(scats_site), as_of, horizon_minutes, processed_csv,
            scaler_path=scaler_path or Path(processed_csv).parent / "scaler.json",
            model_dir=model_dir, sequence_length=sequence_length,
        )
    elif name in ("regression_tree", "tree", "regression-tree"):
        from models.regression_tree_model import predict_regression_tree
        output = predict_regression_tree(
            scats_site, as_of, horizon_minutes, processed_csv,
            model_dir=model_dir, sequence_length=sequence_length,
            scaler_path=scaler_path,
        )
    else:
        raise ValueError("model_name must be gru, lstm or regression_tree")
    result = float(output)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"Invalid predicted traffic flow for SCATS site {scats_site}")
    return result


def load_site_coordinates(csv_path: str | Path) -> dict[int, tuple[float, float]]:
    """Read site metadata; coordinates are NOT road connectivity evidence."""
    coords: dict[int, tuple[float, float]] = {}
    with Path(csv_path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not {"scats_site", "latitude", "longitude"}.issubset(reader.fieldnames or []):
            raise ValueError("Site CSV needs scats_site,latitude,longitude")
        for row in reader:
            site = int(row["scats_site"])
            lat, lon = float(row["latitude"]), float(row["longitude"])
            if not math.isfinite(lat) or not math.isfinite(lon) or not -90 <= lat <= 90 or not -180 <= lon <= 180:
                raise ValueError(f"Invalid coordinates for site {site}")
            if site in coords:
                raise ValueError(f"Duplicate site metadata: {site}")
            coords[site] = (lat, lon)
    if not coords:
        raise ValueError("Site metadata is empty")
    return coords


def load_verified_road_links(csv_path: str | Path) -> list[tuple[int, int, float]]:
    """Read manually/externally verified *directed* connections and distances.

    Column names: from_site,to_site,distance_km. No automatic reverse edges.
    """
    links = []
    seen = set()
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing verified road links CSV: {path}. Darren must supply "
            "from_site,to_site,distance_km for actual directed roads."
        )
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not {"from_site", "to_site", "distance_km"}.issubset(reader.fieldnames or []):
            raise ValueError("Road CSV needs from_site,to_site,distance_km")
        for index, row in enumerate(reader, start=2):
            try:
                u, v, distance = int(row["from_site"]), int(row["to_site"]), float(row["distance_km"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid road row {index}") from exc
            if u == v or (u, v) in seen or not math.isfinite(distance) or distance <= 0:
                raise ValueError(f"Invalid or repeated directed road {u}->{v} at row {index}")
            seen.add((u, v))
            links.append((u, v, distance))
    if not links:
        raise ValueError("Road links CSV is empty")
    return links


def compute_traffic_routes(
    origin: int,
    destination: int,
    as_of_time: str | datetime,
    verified_road_links: list[tuple[int, int, float]],
    site_coords: dict[int, tuple[float, float]],
    predictor: Callable[[int, datetime], float],
    *,
    k: int = 5,
    flow_site: str = "origin",
    hourly_policy: str = "steady_4x",
    capacity_policy: str = "reject",
    intersection_delay_seconds: float = 30.0,
    missing_site_policy: str = "reject",
) -> dict:
    """Convert per-site predicted 15-min counts to route costs, then rank routes.

    The steady_4x policy assumes all four future 15-min counts equal the one
    predicted interval. This is NOT a genuine hourly forecast. It is explicit.
    'reject' is the default when >1500 veh/hour; optional clamp is for a
    clearly labelled sensitivity/demo scenario only, never an observed value.
    """
    as_of = valid_as_of(as_of_time)
    if hourly_policy != "steady_4x":
        raise ValueError("Only explicit steady_4x hourly approximation is currently supported")
    if capacity_policy not in {"reject", "clamp"}:
        raise ValueError("capacity_policy must be reject or clamp")
    if missing_site_policy not in {"reject", "exclude"}:
        raise ValueError("missing_site_policy must be reject or exclude")
    if flow_site not in {"origin", "destination"}:
        raise ValueError("flow_site must be origin or destination")
    if origin not in site_coords or destination not in site_coords:
        raise ValueError("Origin/destination must be known SCATS site IDs")
    if not 1 <= k <= 5:
        raise ValueError("k must be from 1 to 5")
    for u, v, dist in verified_road_links:
        if u not in site_coords or v not in site_coords:
            raise ValueError(f"Unknown SCATS site in verified road link: {u}->{v}")
        if type(dist) not in (int, float) or not math.isfinite(dist) or dist <= 0:
            raise ValueError(f"Invalid distance for {u}->{v}")
    # The full graph can include sites with no trained model (3685, 4264).
    # Keep the default strict, but permit an EXPLICIT partial-network experiment.
    # In partial mode, remove every edge incident to an unavailable site; never
    # fabricate predictions or use a misleading zero for missing observations.
    required_sites = (sorted({s for u, v, _ in verified_road_links for s in (u, v)})
                      if missing_site_policy == "exclude" else
                      sorted({u if flow_site == "origin" else v
                              for u, v, _ in verified_road_links}))
    hourly = {}
    clamped_sites = []
    excluded_sites = {}
    for site in required_sites:
        try:
            volume_15min = float(predictor(site, as_of))
        except FileNotFoundError as exc:
            if missing_site_policy != "exclude":
                raise
            excluded_sites[site] = str(exc)
            continue
        except ValueError as exc:
            # Missing historical observations may make a valid trained model
            # unavailable for a particular requested time.
            if (missing_site_policy != "exclude" or
                    "Insufficient nonmissing historical observations" not in str(exc)):
                raise
            excluded_sites[site] = str(exc)
            continue
        if not math.isfinite(volume_15min) or volume_15min < 0:
            raise ValueError(f"Invalid 15-minute prediction for site {site}")
        hourly_estimate = volume_15min * 4
        if hourly_estimate > CAPACITY_VEH_PER_HOUR:
            if capacity_policy == "reject":
                raise ValueError(
                    f"SCATS {site} predicted {hourly_estimate:.1f} vehicles/hour "
                    f"under steady_4x, exceeding the lecturer's "
                    f"{CAPACITY_VEH_PER_HOUR:.0f} veh/hour assumption. "
                    "Ask tutor for a documented policy; optional "
                    "--capacity-policy clamp is an explicit approximation."
                )
            clamped_sites.append(site)
            hourly_estimate = CAPACITY_VEH_PER_HOUR
        hourly[site] = hourly_estimate
    usable_links = ([ (u, v, d) for u, v, d in verified_road_links
                       if u not in excluded_sites and v not in excluded_sites]
                    if missing_site_policy == "exclude" else verified_road_links)
    if missing_site_policy == "exclude" and (origin in excluded_sites or destination in excluded_sites):
        raise ValueError("Selected origin/destination has no usable prediction. "
                         "Choose another site or a date with complete historical data.")
    edges = weights_from_road_links(
        usable_links, hourly,
        flow_site=flow_site, intersection_delay_s=intersection_delay_seconds,
    )
    graph = build_weighted_scats_graph(site_coords, edges)
    routes = find_top_k_routes(graph, origin, destination, k=k)
    return {
        "origin": origin,
        "destination": destination,
        "as_of": as_of.isoformat(),
        "route_count": len(routes),
        "routes": [
            {"rank": i + 1, "scats_sites": path,
             "travel_time_seconds": round(total_s, 3),
             "travel_time_minutes": round(total_s / 60.0, 3)}
            for i, (path, total_s) in enumerate(routes)
        ],
        "assumptions": {
            "flow_site": flow_site,
            "hourly_conversion": "4 x one predicted 15-minute count (steady-flow assumption)",
            "capacity_policy": capacity_policy,
            "capacity_clamped_sites": clamped_sites,
            "intersection_delay_seconds": intersection_delay_seconds,
            "road_links": "Caller-supplied map-reviewed directed edges; no roads inferred at runtime",
            "road_distances": "Estimated straight-line or L-shaped distances, not measured road distance",
            "missing_site_policy": missing_site_policy,
            "unavailable_sites": excluded_sites,
            "source_road_link_count": len(verified_road_links),
            "usable_road_link_count": len(usable_links),
            "excluded_road_link_count": len(verified_road_links) - len(usable_links),
            "network_incomplete": bool(excluded_sites),
            "route_scope": ("PARTIAL model-covered graph; not globally optimal on full road network"
                            if excluded_sites else "All supplied graph links eligible"),
        },
    }


def main(argv=None) -> int:
    defaults = load_config()
    parser = argparse.ArgumentParser(description="COS30019 A2B - traffic forecasting and routing")
    sub = parser.add_subparsers(dest="command", required=True)
    def common(p):
        p.add_argument("--model", default=str(defaults.get("default_model", "gru")).lower(), choices=["gru", "lstm", "regression_tree"])
        p.add_argument("--time", required=True, help="Last observed 15-min timestamp, e.g. 2006-10-20T10:00:00")
        p.add_argument("--horizon", type=int, choices=[15, 30, 60], default=defaults.get("prediction_horizon_minutes", 15))
        p.add_argument("--sequence-length", type=int, default=8)
        p.add_argument("--csv", type=Path, default=DEFAULT_CSV)
        p.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS)
        p.add_argument("--scaler", type=Path, default=None)

    p_pred = sub.add_parser("predict", help="Predict real 15-minute traffic for a trained SCATS site")
    common(p_pred)
    p_pred.add_argument("--site", required=True, type=int)

    p_route = sub.add_parser("route", help="Route using VERIFIED roads and available per-site model checkpoints")
    common(p_route)
    p_route.add_argument("--origin", required=True, type=int)
    p_route.add_argument("--destination", required=True, type=int)
    p_route.add_argument("--roads", type=Path, default=DEFAULT_ROADS, help="Map-reviewed directed road CSV (default: Darren's scats_road_edges.csv)")
    p_route.add_argument("--sites", type=Path, default=DEFAULT_SITES)
    p_route.add_argument("--top-k", type=int, default=defaults.get("top_k_routes", 5))
    p_route.add_argument("--flow-site", choices=["origin", "destination"], default=defaults.get("volume_used_for_edge", "origin"))
    p_route.add_argument("--hourly-policy", choices=["steady_4x"], required=True)
    p_route.add_argument("--capacity-policy", choices=["reject", "clamp"], default="reject")
    p_route.add_argument("--missing-site-policy", choices=["reject", "exclude"], default="reject",
                         help="Exclude unavailable model sites for explicitly partial-network development tests")
    p_route.add_argument("--intersection-delay", type=float, default=defaults.get("controlled_intersection_delay_seconds", 30.0))
    args = parser.parse_args(argv)
    try:
        when = valid_as_of(args.time)
        predict_args = dict(model_name=args.model, horizon_minutes=args.horizon,
                            sequence_length=args.sequence_length,
                            processed_csv=args.csv, model_dir=args.models_dir,
                            scaler_path=args.scaler)
        if args.command == "predict":
            y = predict_traffic(args.site, when, **predict_args)
            result = {"model": args.model.upper(), "site": args.site,
                      "as_of": when.isoformat(),
                      "target_time": (when + timedelta(minutes=args.horizon)).isoformat(),
                      "horizon_minutes": args.horizon,
                      "predicted_vehicles_per_15min": round(y, 4),
                      "note": "This is ONE future 15-minute interval, not an hourly count."}
        else:
            road_links = load_verified_road_links(args.roads)
            coords = load_site_coordinates(args.sites)
            result = compute_traffic_routes(
                args.origin, args.destination, when, road_links, coords,
                predictor=lambda site, as_of: predict_traffic(site, as_of, **predict_args),
                k=args.top_k, flow_site=args.flow_site,
                hourly_policy=args.hourly_policy,
                capacity_policy=args.capacity_policy,
                intersection_delay_seconds=args.intersection_delay,
                missing_site_policy=args.missing_site_policy,
            )
            result["model"] = args.model.upper()
            result["horizon_minutes"] = args.horizon
        print(json.dumps(result, indent=2))
        return 0
    except (FileNotFoundError, ValueError, KeyError, NotImplementedError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
