"""Gradio interface for COS30019 A2B.

Run from the project root:
    python -m gui.app

The GUI delegates all predictions and route calculations to main.py. It does not
train models, infer road connections, or invent missing model checkpoints.
"""
from __future__ import annotations

import json
import math
from datetime import timedelta
from pathlib import Path

import gradio as gr

import main as backend
from gui import comparison_view, prediction_view, route_map

ROOT = Path(__file__).resolve().parents[1]
ROADS_DEFAULT = backend.DEFAULT_ROADS
ROUTE_HEADERS = ["Rank", "SCATS route", "Travel time (min)", "Travel time (sec)"]
VALID_MODELS = {"GRU": "gru", "LSTM": "lstm", "Regression Tree": "regression_tree"}
# Default forecast time: Mon 30 Oct 2006, 08:00. It is in the test split (data the
# models never saw), every site has complete data that day (24-28 Oct has gaps
# at five sites), and it is the morning peak, so the chart has a clear shape.
DEFAULT_PREDICTION_TIME = "2006-10-30T08:00:00"
PREDICTION_HEADERS = prediction_view.TABLE_HEADERS
# Route comparison default: Mon 30 Oct 2006 20:00 (test split) with Reject.
# Routing uses per-approach flow (site total / approach count), which is under
# 1,500 veh/h in 99.9% of observed intervals. The exceptions are the morning
# and evening peaks at single-approach sites, mainly 2825 (Burke Rd S of the
# Eastern Fwy, over 1,500 at 08:00-08:30 on most weekdays), where Reject stops.
DEFAULT_COMPARISON_TIME = "2006-10-30T20:00:00"


def _site_id(value) -> int:
    if isinstance(value, bool) or not str(value).strip().isdigit():
        raise ValueError("SCATS site must be a positive integer site ID")
    site = int(str(value).strip())
    if site < 1:
        raise ValueError("SCATS site must be positive")
    return site


def _safe_site(value):
    try:
        return _site_id(value)
    except ValueError:
        return None


def known_sites() -> list[str]:
    """Get SCATS IDs from processed metadata, not from location TFM identifiers."""
    if not backend.DEFAULT_SITES.is_file():
        return ["2000"]
    try:
        sites = backend.load_site_coordinates(backend.DEFAULT_SITES)
        return [str(x) for x in sorted(sites)]
    except (OSError, ValueError, KeyError):
        return ["2000"]


def route_map_html(road_file=None, routes=(), **payload_kwargs) -> str:
    """Interactive OpenStreetMap view (iframe HTML). Never raises: the map is
    supporting visualisation, so a map problem must not hide route results."""
    try:
        coords = backend.load_site_coordinates(backend.DEFAULT_SITES)
        try:
            links = backend.load_verified_road_links(str(road_file or ROADS_DEFAULT))
        except (FileNotFoundError, OSError, ValueError):
            links = []
        payload = route_map.build_map_payload(
            coords, links, routes, site_names=route_map.load_site_names(backend.DEFAULT_SITES),
            **payload_kwargs,
        )
        return route_map.build_map_iframe(payload)
    except (FileNotFoundError, OSError, ValueError, KeyError, TypeError) as exc:
        return f"<p><b>Map unavailable:</b> {exc}</p>"


def run_prediction(site, as_of_time, model, horizon_minutes, sequence_length):
    """GUI callback. Return (status, predicted count, structured details)."""
    try:
        site_id = _site_id(site)
        when = backend.valid_as_of(str(as_of_time).strip())
        selected = VALID_MODELS[model]
        horizon = int(horizon_minutes)
        lookback = int(sequence_length)
        volume = backend.predict_traffic(
            scats_site=site_id,
            as_of_time=when,
            model_name=selected,
            horizon_minutes=horizon,
            sequence_length=lookback,
        )
        result = {
            "model": model,
            "scats_site": site_id,
            "last_observed_interval": when.isoformat(),
            "forecast_target_interval": (when + timedelta(minutes=horizon)).isoformat(),
            "horizon_minutes": horizon,
            "sequence_length": lookback,
            "predicted_vehicles_per_15min": round(volume, 4),
            "units_note": "One forecast 15-minute interval, NOT vehicles per hour",
        }
        return (f"✅ **Prediction successful** — {model} checkpoint loaded for SCATS site {site_id}.", round(volume, 2), result)
    except (FileNotFoundError, OSError, ValueError, TypeError, KeyError, NotImplementedError) as exc:
        return (f"⚠️ **Prediction unavailable:** {exc}", None, {})


def run_prediction_view(site, as_of_time, model, horizon_minutes, sequence_length,
                        compare_all=False):
    """Traffic prediction tab callback.

    Returns (status, predicted count, details JSON, chart HTML, model table).
    The selected model goes through run_prediction unchanged; with compare_all
    the other models are run on the same input for the chart and table.
    """
    status, volume, details = run_prediction(site, as_of_time, model, horizon_minutes, sequence_length)
    try:
        site_id = _site_id(site)
        when = backend.valid_as_of(str(as_of_time).strip())
        horizon, lookback = int(horizon_minutes), int(sequence_length)
    except (ValueError, TypeError):
        return status, volume, details, "", []
    coverage = prediction_view.model_coverage(backend.DEFAULT_MODELS)
    labels = list(VALID_MODELS) if compare_all else [model]
    others = [m for m in labels if m != model]
    results = []
    if volume is not None:
        results.append({"model": model, "value": details["predicted_vehicles_per_15min"], "note": ""})
    else:
        results.append({"model": model, "value": None,
                        "note": status.replace("⚠️ **Prediction unavailable:** ", "")})
    results += prediction_view.predict_models(
        site_id, when, horizon, lookback, others, backend.predict_traffic, coverage)
    results.sort(key=lambda r: list(VALID_MODELS).index(r["model"]))
    try:
        names = route_map.load_site_names(backend.DEFAULT_SITES)
        payload = prediction_view.build_chart_payload(
            site_id, when, horizon, lookback, results, backend.DEFAULT_CSV,
            site_name=names.get(site_id, ""), selected_model=model)
        chart = prediction_view.build_chart_iframe(payload)
        table = prediction_view.results_table(results, payload["actual"])
    except (FileNotFoundError, OSError, ValueError, KeyError) as exc:
        return status, volume, details, f"<p><b>Chart unavailable:</b> {exc}</p>", []
    if volume is not None and payload["actual"] is not None:
        details = {**details, "actual_vehicles_per_15min": payload["actual"],
                   "error_vehicles_per_15min": round(volume - payload["actual"], 2)}
    return status, volume, details, chart, table


def comparison_overview():
    """Static part of the Model comparison tab: (headline, results table, charts)."""
    results = comparison_view.load_results()
    return (comparison_view.headline(results), comparison_view.summary_rows(results),
            comparison_view.build_iframe(comparison_view.overview_specs(results), initial_height=1500))


def run_actual_vs_predicted(site, horizon_minutes, day):
    """Test-day chart for one site: actual traffic and each model's forecast."""
    try:
        site_id, horizon = _site_id(site), int(horizon_minutes)
    except (ValueError, TypeError) as exc:
        return f"⚠️ {exc}", ""
    spec, notes = comparison_view.actual_vs_predicted_spec(site_id, horizon, str(day))
    if spec is None:
        return "⚠️ No saved test predictions for this choice. " + "; ".join(notes), ""
    status = "Not shown: " + "; ".join(notes) if notes else ""
    return status, comparison_view.build_iframe([spec], initial_height=420)


def run_route_comparison(origin, destination, as_of_time, capacity_policy, route_count=5, road_file=None, search_algorithm="AS"):
    """Same trip and time under GRU, LSTM and Regression Tree (15 min ahead).

    Returns (status, table rows, map HTML).
    """
    try:
        start, end = _site_id(origin), _site_id(destination)
        when = backend.valid_as_of(str(as_of_time).strip())
        links = backend.load_verified_road_links(str(road_file or ROADS_DEFAULT))
        coords = backend.load_site_coordinates(backend.DEFAULT_SITES)
    except (FileNotFoundError, OSError, ValueError, TypeError, KeyError) as exc:
        return f"⚠️ **Route comparison unavailable:** {exc}", [], route_map_html(road_file)
    policy = "reject" if str(capacity_policy).startswith("Reject") else "clamp"
    flow_site = backend.load_config().get("volume_used_for_edge", "origin")

    def route_fn(label):
        return backend.compute_traffic_routes(
            origin=start, destination=end, as_of_time=when,
            verified_road_links=links, site_coords=coords,
            predictor=lambda site, t: backend.predict_traffic(
                site, t, model_name=VALID_MODELS[label], horizon_minutes=15, sequence_length=8),
            k=int(route_count), flow_site=flow_site, hourly_policy="steady_4x",
            capacity_policy=policy, missing_site_policy="reject",
            search_algorithm=search_algorithm,
        )

    rows, by_model, notes = comparison_view.compare_routes(start, end, when, route_fn=route_fn, k=int(route_count))
    best = {m: r[0] for m, r in by_model.items() if r}
    if best:
        parts = [f"{m} {r['travel_time_minutes']:.1f} min" for m, r in best.items()]
        same = len({tuple(r["scats_sites"]) for r in best.values()}) == 1
        status = (f"**Best route under each model** using {search_algorithm} search "
                  f"({when:%a %d %b %H:%M}, 15 min ahead): " + ", ".join(parts) + ". "
                  + ("All models choose the **same** best route." if same and len(best) > 1 else
                     "The models choose **different** best routes; the map shows each in its model's colour."
                     if len(best) > 1 else ""))
    else:
        status = "⚠️ **No model produced routes.**"
    if notes:
        status += "\n\n⚠️ Not available: " + "; ".join(notes)
    if search_algorithm not in {"AS", "CUS2"}:
        status += "\n\n⚠️ This algorithm does not guarantee the fastest travel-time routes."
    if policy == "clamp":
        status += "\n\nExperimental Clamp is on: sites forecast above 1,500 veh/h per approach are capped, which hides differences between models there."
    map_html = route_map_html(road_file, comparison_view.best_routes_for_map(by_model),
                              origin=start, destination=end,
                              model_label=f"Best route per model · {when:%a %d %b %H:%M}")
    return status, rows, map_html


def run_routes(origin, destination, as_of_time, model, horizon_minutes,
               sequence_length, route_count, road_file, flow_site, capacity_policy, missing_site_policy="reject", search_algorithm="AS"):
    """Route callback. Requires verified road connections and per-site models.

    Returns (status, route rows, details JSON, map HTML).
    """
    start = end = None
    try:
        start, end = _site_id(origin), _site_id(destination)
        when = backend.valid_as_of(str(as_of_time).strip())
        links = backend.load_verified_road_links(str(road_file).strip())
        coords = backend.load_site_coordinates(backend.DEFAULT_SITES)
        selected_model = VALID_MODELS[model]
        horizon, lookback = int(horizon_minutes), int(sequence_length)
        policy = "reject" if str(capacity_policy).startswith("Reject") else "clamp"
        site_flows: dict[int, float] = {}

        def predictor(site, as_of):
            # Same call as before; also remember each site's forecast so the
            # map can show it on hover (no extra model runs).
            value = backend.predict_traffic(
                site, as_of, model_name=selected_model,
                horizon_minutes=horizon, sequence_length=lookback,
            )
            site_flows[int(site)] = float(value)
            return value

        result = backend.compute_traffic_routes(
            origin=start, destination=end, as_of_time=when,
            verified_road_links=links, site_coords=coords,
            predictor=predictor,
            k=int(route_count), flow_site=str(flow_site).lower(),
            hourly_policy="steady_4x", capacity_policy=policy,
            missing_site_policy=missing_site_policy, search_algorithm=search_algorithm,
        )
        rows = [
            [route["rank"], " → ".join(str(s) for s in route["scats_sites"]),
             route["travel_time_minutes"], route["travel_time_seconds"]]
            for route in result["routes"]
        ]
        note = ("⚠️ **Experimental Clamp:** per-approach flow above 1,500 veh/h was capped. "
                "Do not report those values as measured traffic. " if result["assumptions"]["capacity_clamped_sites"] else "")
        partial = result["assumptions"]["network_incomplete"]
        if partial:
            unavailable = sorted(result["assumptions"]["unavailable_sites"])
            prefix = ("⚠️ **PARTIAL-NETWORK DEVELOPMENT RESULT — not a verified best route on the "
                      f"full 40-site graph.** Excluded forecast-unavailable SCATS sites: "
                      f"{', '.join(map(str, unavailable))}. "
                      f"Excluded {result['assumptions']['excluded_road_link_count']} road-link rows. ")
        else:
            prefix = "✅ "
        method_note = ("Cost-optimal top routes" if result["assumptions"]["search_cost_optimal"]
                       else "Method-discovered routes — NOT guaranteed fastest")
        status = (prefix + f"**Computed {len(rows)} candidate route(s)** using {model} forecasts "
                  f"and {search_algorithm} search ({method_note}; "
                  f"{result['assumptions']['search_nodes_expanded']} expanded). "
                  "Hour flow assumes four identical future 15-minute counts (4× approximation), "
                  "divided by each site's number of approaches (flow on one road). " + note)
        if not rows:
            status = "⚠️ **No route found** between these verified SCATS intersections. " + status
        # Backend uses integer SCATS IDs as dictionary keys, which is valid
        # Python but not accepted by gr.JSON's strict JSON serialiser.
        # Convert the GUI payload only; preserve the backend's integer-keyed
        # results for routing and tests.
        display_result = {
            **result,
            "assumptions": {
                **result["assumptions"],
                "unavailable_sites": {
                    str(site): reason
                    for site, reason in result["assumptions"]["unavailable_sites"].items()
                },
            },
        }
        map_html = route_map_html(
            road_file, result["routes"],
            site_flows_15min=site_flows,
            unavailable_sites=result["assumptions"]["unavailable_sites"].keys(),
            clamped_sites=result["assumptions"]["capacity_clamped_sites"],
            site_flows_per_approach={r["site"]: r["flow_used_veh_per_hour"]
                                     for r in result["assumptions"]["site_flows"]
                                     if r["approaches"] is not None},
            origin=start, destination=end,
            model_label=f"{model} · +{horizon} min after {when:%a %d %b %Y %H:%M}",
        )
        return status, rows, display_result, map_html
    except (FileNotFoundError, OSError, ValueError, TypeError, KeyError, NotImplementedError) as exc:
        return (f"⚠️ **Route calculation unavailable:** {exc}", [], {},
                route_map_html(road_file, origin=start, destination=end))



def gui_config_defaults(config: dict) -> tuple[str, int]:
    """Use safe GUI defaults from config.json; fall back when invalid.

    Model values may use display labels ("Regression Tree") or their backend
    identifiers ("regression_tree"). The prediction and route tabs share
    these defaults. Model coverage still depends on the saved checkpoints.
    """
    configured_model = config.get("default_model", "GRU")
    model = "GRU"
    if isinstance(configured_model, str):
        candidate = configured_model.strip().casefold()
        for label, backend_name in VALID_MODELS.items():
            if candidate in (label.casefold(), backend_name.casefold()):
                model = label
                break

    configured_horizon = config.get("prediction_horizon_minutes", 15)
    try:
        # bool is an int subclass, but True/False are not sensible horizons.
        if isinstance(configured_horizon, bool):
            raise ValueError("Boolean is not a horizon")
        horizon = int(configured_horizon)
    except (TypeError, ValueError, OverflowError):
        horizon = 15
    if horizon not in (15, 30, 60):
        horizon = 15
    return model, horizon


def build_app() -> gr.Blocks:
    defaults = backend.load_config()
    default_model, default_horizon = gui_config_defaults(defaults)
    sites = known_sites()
    start_site = "2000" if "2000" in sites else sites[0]
    dest_site = "3002" if "3002" in sites else sites[-1]

    with gr.Blocks(title="TBRGS · COS30019 Assignment 2B") as app:
        gr.Markdown(
            "# Traffic-Based Route Guidance System (TBRGS)\n"
            "**COS30019 · Group 2 · Assignment 2B**  |  Boroondara SCATS data · October 2006"
        )
        coverage = prediction_view.model_coverage(backend.DEFAULT_MODELS)
        with gr.Accordion("Model coverage: SCATS sites with a trained model (8-step lookback)", open=True):
            gr.Markdown(
                prediction_view.coverage_markdown(coverage, len(sites)) + "\n\n"
                "Route guidance needs a forecast for every site on the network, so full-network "
                "routes need ✅ for the chosen model and horizon. The road graph is Darren's "
                "**map-reviewed, estimated-distance** network; it has not been independently "
                "ground-truthed."
            )
        with gr.Tabs():
            with gr.Tab("Traffic prediction"):
                gr.Markdown("### Predict traffic for one future 15-minute interval")
                with gr.Row():
                    with gr.Column(scale=1):
                        site = gr.Dropdown(label="SCATS site ID", choices=sites,
                                           value=start_site, allow_custom_value=True)
                        time = gr.Textbox(label="Last observed date and time (ISO local)",
                                          value=DEFAULT_PREDICTION_TIME,
                                          info="Must be a 15-minute boundary. 27-31 Oct is the test set "
                                               "(unseen by the models); 24-28 Oct has gaps at five sites.")
                        model = gr.Dropdown(label="ML model", choices=list(VALID_MODELS), value=default_model)
                        horizon = gr.Dropdown(label="Forecast horizon (minutes)",
                                              choices=[15, 30, 60], value=default_horizon,
                                              info="Regression Tree: all sites at 15/30/60 min. GRU/LSTM: see Model coverage above")
                        seq = gr.Number(label="Historical lookback (15-minute intervals)",
                                        value=8, precision=0, minimum=1, maximum=96)
                        compare = gr.Checkbox(label="Compare with the other two models (side by side on the chart)",
                                              value=False)
                        submit = gr.Button("Predict traffic", variant="primary")
                    with gr.Column(scale=2):
                        status = gr.Markdown("Select a trained model and click **Predict traffic**.")
                        volume = gr.Number(label="Predicted vehicles / 15 minutes (selected model)",
                                           interactive=False)
                        chart = gr.HTML(label="Forecast in context")
                        model_table = gr.Dataframe(
                            headers=PREDICTION_HEADERS, datatype=["str", "number", "number", "number", "str"],
                            value=[], interactive=False, label="Models at the forecast time")
                        with gr.Accordion("Prediction details", open=False):
                            result_json = gr.JSON(label="Details")
                submit.click(
                    fn=run_prediction_view, inputs=[site, time, model, horizon, seq, compare],
                    outputs=[status, volume, result_json, chart, model_table],
                )
            with gr.Tab("Route guidance"):
                gr.Markdown(
                    "### Up to five traffic-based candidate routes\n"
                    "Uses Darren's map-reviewed Boroondara graph with estimated link distances. "
                    "If model coverage is incomplete, optional partial-network testing excludes "
                    "the affected sites and cannot guarantee global top-five routes."
                )
                with gr.Row():
                    with gr.Column():
                        origin = gr.Dropdown(label="Origin SCATS site", choices=sites,
                                             value=start_site, allow_custom_value=True)
                        destination = gr.Dropdown(label="Destination SCATS site", choices=sites,
                                                  value=dest_site, allow_custom_value=True)
                        route_time = gr.Textbox(label="Last observed date/time", value=DEFAULT_COMPARISON_TIME,
                                                info="Held-out test example. Choose 15, 30 or 60 minutes ahead; full-network GRU/LSTM coverage is currently available at 15 minutes.")
                        route_model = gr.Dropdown(label="Prediction model", choices=list(VALID_MODELS), value=default_model)
                        route_horizon = gr.Dropdown(label="Prediction horizon (min)", choices=[15, 30, 60], value=default_horizon)
                        search_method = gr.Dropdown(
                            label="A2A search algorithm", choices=["DFS", "BFS", "GBFS", "AS", "CUS1", "CUS2"],
                            value=defaults.get("default_search_algorithm", "AS"),
                            info="AS/A* and CUS2/weighted IDA* guarantee fastest top-five; others do not")
                        route_seq = gr.Number(label="Historical lookback", value=8, precision=0, minimum=1, maximum=96)
                        k = gr.Slider(label="Maximum routes", minimum=1, maximum=5, step=1, value=defaults.get("top_k_routes", 5))
                        roads = gr.Textbox(label="SCATS directed road CSV path (map-reviewed)", value=str(ROADS_DEFAULT),
                                           info="CSV columns: from_site,to_site,distance_km")
                        flow_site = gr.Radio(label="Edge flow taken from (origin = conversion PDF)",
                                             choices=["origin", "destination"], value=defaults.get("volume_used_for_edge", "origin"))
                        capacity = gr.Radio(label="Over-capacity policy", choices=[
                            "Reject values above 1500 veh/hour",
                            "Clamp to 1500 (experimental)"],
                            value="Reject values above 1500 veh/hour")
                        missing_mode = gr.Radio(
                            label="Missing model forecasts",
                            choices=[("Stop if any required model is unavailable", "reject"),
                                     ("Exclude unavailable sites (partial-network test only)", "exclude")],
                            value="reject",
                        )
                        route_btn = gr.Button("Calculate routes", variant="primary")
                    with gr.Column(scale=2):
                        route_status = gr.Markdown("Awaiting verified road links and relevant saved model files.")
                        route_map_view = gr.HTML(
                            value=route_map_html(origin=_safe_site(start_site), destination=_safe_site(dest_site)),
                            label="Route map (OpenStreetMap)")
                        route_table = gr.Dataframe(headers=ROUTE_HEADERS, datatype=["number", "str", "number", "number"],
                                                    value=[], interactive=False, label="Route ranking")
                        with gr.Accordion("Route calculation details and assumptions", open=False):
                            route_json = gr.JSON(label="Details")
                route_btn.click(
                    fn=run_routes,
                    inputs=[origin, destination, route_time, route_model, route_horizon,
                            route_seq, k, roads, flow_site, capacity, missing_mode, search_method],
                    outputs=[route_status, route_table, route_json, route_map_view],
                )
                gr.Markdown(
                    "**Six A2A search algorithms:** AS (A*) and CUS2 (weighted IDA*) give the "
                    "lowest-cost paths; DFS, BFS, GBFS and CUS1 (IDDFS) find valid alternatives "
                    "but cannot guarantee lowest travel time.\n\n"
                    "**Modelling limitations:** `steady_4x` converts one 15-minute model forecast "
                    "to an estimated hourly flow; this is *not* a true four-period forecast. "
                    "The provided flow-to-speed formula assumes at most 1,500 vehicles/hour on one road, "
                    "so each SCATS site total is divided by its number of approaches first "
                    "(this assumes traffic splits evenly across the approaches). "
                    "Map-reviewed road links still have estimated distances and some medium-confidence "
                    "connections; partial-network results are for integration tests only.\n\n"
                    "**Map:** SCATS coordinates are shifted by one measured offset (≈168 m north, "
                    "≈115 m east; see `data/processed/scats_coordinate_offset.json`) so sites sit on "
                    "their OpenStreetMap intersections. Routes are drawn as straight lines between "
                    "intersections, not along street curves. The map background needs internet; "
                    "route calculation itself works offline."
                )
            with gr.Tab("Model comparison"):
                headline_md, summary_table, overview_html = comparison_overview()
                gr.Markdown("### GRU vs LSTM vs Regression Tree\n" + headline_md)
                gr.Markdown(
                    "All numbers are on the same held-out test data (27–31 Oct 2006, 480 points per site) "
                    "with an 8-reading lookback. Sources: `results/model_comparison/three_model/` "
                    "(Anson's matched comparison) and `results/model_comparison/regression_tree/` "
                    "(Regression Tree extras and the last-reading baseline).")
                gr.HTML(overview_html)
                gr.Markdown("#### Results table: network average at 15 min ahead\n"
                            "Training time is per site (Regression Tree includes its hyper-parameter search). "
                            "Model file and learned parameters are medians over sites.")
                gr.Dataframe(headers=comparison_view.SUMMARY_HEADERS, value=summary_table, interactive=False,
                             datatype=["str", "number", "number", "number", "number", "number", "number", "number"])

                gr.Markdown("#### Actual vs predicted on a test day")
                with gr.Row():
                    cmp_site = gr.Dropdown(label="SCATS site", choices=sites, value=start_site,
                                           allow_custom_value=True, scale=1)
                    cmp_horizon = gr.Dropdown(label="Minutes ahead", choices=[15, 30, 60], value=15, scale=1,
                                              info="30/60 min: all three models only at site 2000")
                    cmp_day = gr.Dropdown(label="Test day", choices=list(comparison_view.TEST_DAYS),
                                          value="2006-10-30", scale=1)
                    cmp_btn = gr.Button("Show", variant="primary", scale=0)
                cmp_status = gr.Markdown()
                cmp_chart = gr.HTML()
                cmp_btn.click(fn=run_actual_vs_predicted, inputs=[cmp_site, cmp_horizon, cmp_day],
                              outputs=[cmp_status, cmp_chart])

                gr.Markdown("#### Routes under each model\n"
                            "The same trip and time with GRU, LSTM and Regression Tree forecasts "
                            "(15 min ahead, full 40-site network).")
                with gr.Row():
                    with gr.Column(scale=1):
                        rc_origin = gr.Dropdown(label="Origin SCATS site", choices=sites, value=start_site,
                                                allow_custom_value=True)
                        rc_dest = gr.Dropdown(label="Destination SCATS site", choices=sites, value=dest_site,
                                              allow_custom_value=True)
                        rc_time = gr.Textbox(label="Last observed date/time", value=DEFAULT_COMPARISON_TIME,
                                             info="Default Mon 30 Oct 20:00. At 08:00-08:30 site 2825 exceeds 1,500 veh/h per approach, so Reject stops")
                        rc_capacity = gr.Radio(label="Over-capacity policy", choices=[
                            "Reject values above 1500 veh/hour", "Clamp to 1500 (experimental)"],
                            value="Reject values above 1500 veh/hour")
                        rc_algorithm = gr.Dropdown(
                            label="A2A search algorithm",
                            choices=["DFS", "BFS", "GBFS", "AS", "CUS1", "CUS2"],
                            value=defaults.get("default_search_algorithm", "AS"),
                            info="AS / CUS2 optimise travel time; other methods may not")
                        rc_btn = gr.Button("Compare routes", variant="primary")
                    with gr.Column(scale=2):
                        rc_status = gr.Markdown("Pick a trip and click **Compare routes**.")
                        rc_map = gr.HTML()
                        rc_table = gr.Dataframe(headers=comparison_view.ROUTE_HEADERS, value=[], interactive=False,
                                                datatype=["str", "number", "str", "number", "str"],
                                                label="Top routes under each model")
                rc_btn.click(fn=run_route_comparison, inputs=[rc_origin, rc_dest, rc_time, rc_capacity,
                                                               gr.State(5), gr.State(None), rc_algorithm],
                             outputs=[rc_status, rc_table, rc_map])
    return app


if __name__ == "__main__":
    build_app().launch(server_name="127.0.0.1", share=False, show_error=False)
