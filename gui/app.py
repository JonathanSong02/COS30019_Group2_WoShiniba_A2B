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

ROOT = Path(__file__).resolve().parents[1]
ROADS_DEFAULT = backend.DEFAULT_ROADS
ROUTE_HEADERS = ["Rank", "SCATS route", "Travel time (min)", "Travel time (sec)"]
VALID_MODELS = {"GRU": "gru", "LSTM": "lstm", "Regression Tree": "regression_tree"}


def _site_id(value) -> int:
    if isinstance(value, bool) or not str(value).strip().isdigit():
        raise ValueError("SCATS site must be a positive integer site ID")
    site = int(str(value).strip())
    if site < 1:
        raise ValueError("SCATS site must be positive")
    return site


def known_sites() -> list[str]:
    """Get SCATS IDs from processed metadata, not from location TFM identifiers."""
    if not backend.DEFAULT_SITES.is_file():
        return ["2000"]
    try:
        sites = backend.load_site_coordinates(backend.DEFAULT_SITES)
        return [str(x) for x in sorted(sites)]
    except (OSError, ValueError, KeyError):
        return ["2000"]


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


def run_routes(origin, destination, as_of_time, model, horizon_minutes,
               sequence_length, route_count, road_file, flow_site, capacity_policy, missing_site_policy="reject"):
    """Route callback. Requires verified road connections and per-site models."""
    try:
        start, end = _site_id(origin), _site_id(destination)
        when = backend.valid_as_of(str(as_of_time).strip())
        links = backend.load_verified_road_links(str(road_file).strip())
        coords = backend.load_site_coordinates(backend.DEFAULT_SITES)
        selected_model = VALID_MODELS[model]
        horizon, lookback = int(horizon_minutes), int(sequence_length)
        policy = "reject" if str(capacity_policy).startswith("Reject") else "clamp"
        result = backend.compute_traffic_routes(
            origin=start, destination=end, as_of_time=when,
            verified_road_links=links, site_coords=coords,
            predictor=lambda site, as_of: backend.predict_traffic(
                site, as_of, model_name=selected_model,
                horizon_minutes=horizon, sequence_length=lookback,
            ),
            k=int(route_count), flow_site=str(flow_site).lower(),
            hourly_policy="steady_4x", capacity_policy=policy,
            missing_site_policy=missing_site_policy,
        )
        rows = [
            [route["rank"], " → ".join(str(s) for s in route["scats_sites"]),
             route["travel_time_minutes"], route["travel_time_seconds"]]
            for route in result["routes"]
        ]
        note = ("⚠️ **Sensitivity scenario:** Flow above capacity was clamped. "
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
        status = (prefix + f"**Computed {len(rows)} candidate route(s)** using {model} forecasts. "
                  "Hour flow assumes four identical future 15-minute counts (4× approximation). " + note)
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
        return status, rows, display_result
    except (FileNotFoundError, OSError, ValueError, TypeError, KeyError, NotImplementedError) as exc:
        return f"⚠️ **Route calculation unavailable:** {exc}", [], {}


def build_app() -> gr.Blocks:
    defaults = backend.load_config()
    sites = known_sites()
    start_site = "2000" if "2000" in sites else sites[0]
    dest_site = "3002" if "3002" in sites else sites[-1]

    with gr.Blocks(title="TBRGS · COS30019 Assignment 2B") as app:
        gr.Markdown(
            "# Traffic-Based Route Guidance System (TBRGS)\n"
            "**COS30019 · Group 2 · Assignment 2B**  |  Boroondara SCATS data · October 2006"
        )
        available = len(list(backend.DEFAULT_MODELS.glob("gru_site_*_h15_s8.pt")))
        gr.Markdown(
            f"**Development status:** GRU checkpoints available for **{available} of {len(sites)}** "
            "SCATS sites at 15-minute horizon (8-step lookback). Other horizons/models require "
            "their own checkpoints. Road graph is Darren's **map-reviewed, estimated-distance** "
            "network; it has not been independently ground-truthed."
        )
        with gr.Tabs():
            with gr.Tab("Traffic prediction"):
                gr.Markdown("### Predict traffic for one future 15-minute interval")
                with gr.Row():
                    with gr.Column():
                        site = gr.Dropdown(label="SCATS site ID", choices=sites,
                                           value=start_site, allow_custom_value=True)
                        time = gr.Textbox(label="Last observed date and time (ISO local)",
                                          value="2006-10-20T10:00:00",
                                          info="Example: 2006-10-20T10:00:00. Must be a 15-minute boundary.")
                        model = gr.Dropdown(label="ML model", choices=list(VALID_MODELS), value="GRU")
                        horizon = gr.Dropdown(label="Forecast horizon (minutes)",
                                              choices=[15, 30, 60], value=30)
                        seq = gr.Number(label="Historical lookback (15-minute intervals)",
                                        value=8, precision=0, minimum=1, maximum=96)
                        submit = gr.Button("Predict traffic", variant="primary")
                    with gr.Column():
                        status = gr.Markdown("Select a trained model and click **Predict traffic**.")
                        volume = gr.Number(label="Predicted vehicles / 15 minutes", interactive=False)
                        result_json = gr.JSON(label="Prediction details")
                submit.click(
                    fn=run_prediction, inputs=[site, time, model, horizon, seq],
                    outputs=[status, volume, result_json],
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
                        route_time = gr.Textbox(label="Last observed date/time", value="2006-10-20T10:00:00")
                        route_model = gr.Dropdown(label="Prediction model", choices=list(VALID_MODELS), value="GRU")
                        route_horizon = gr.Dropdown(label="Prediction horizon (min)", choices=[15, 30, 60], value=15)
                        route_seq = gr.Number(label="Historical lookback", value=8, precision=0, minimum=1, maximum=96)
                        k = gr.Slider(label="Maximum routes", minimum=1, maximum=5, step=1, value=defaults.get("top_k_routes", 5))
                        roads = gr.Textbox(label="SCATS directed road CSV path (map-reviewed)", value=str(ROADS_DEFAULT),
                                           info="CSV columns: from_site,to_site,distance_km")
                        flow_site = gr.Radio(label="SCATS flow convention (confirm with tutor)",
                                             choices=["origin", "destination"], value=defaults.get("volume_used_for_edge", "origin"))
                        capacity = gr.Radio(label="Over-capacity policy", choices=[
                            "Reject values above 1500 veh/hour",
                            "Clamp to 1500 (research sensitivity only)"],
                            value="Reject values above 1500 veh/hour")
                        missing_mode = gr.Radio(
                            label="Missing model forecasts",
                            choices=[("Stop if any required model is unavailable", "reject"),
                                     ("Exclude unavailable sites (partial-network test only)", "exclude")],
                            value="reject",
                        )
                        route_btn = gr.Button("Calculate routes", variant="primary")
                    with gr.Column():
                        route_status = gr.Markdown("Awaiting verified road links and relevant saved model files.")
                        route_table = gr.Dataframe(headers=ROUTE_HEADERS, datatype=["number", "str", "number", "number"],
                                                    value=[], interactive=False, label="Route ranking")
                        route_json = gr.JSON(label="Route calculation details and assumptions")
                route_btn.click(
                    fn=run_routes,
                    inputs=[origin, destination, route_time, route_model, route_horizon,
                            route_seq, k, roads, flow_site, capacity, missing_mode],
                    outputs=[route_status, route_table, route_json],
                )
                gr.Markdown(
                    "**Modelling limitations:** `steady_4x` converts one 15-minute model forecast "
                    "to an estimated hourly flow; this is *not* a true four-period forecast. "
                    "The provided flow-to-speed formula assumes at most 1,500 vehicles/hour. "
                    "Map-reviewed road links still have estimated distances and some medium-confidence "
                    "connections; partial-network results are for integration tests only."
                )
    return app


if __name__ == "__main__":
    build_app().launch(server_name="127.0.0.1", share=False, show_error=False)
