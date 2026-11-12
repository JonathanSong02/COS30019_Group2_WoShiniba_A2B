# COS30019 Assignment 2B — Shared ML and Routing Contract (proposed)

**Before merging code, Jonathan, Darren and Anson should approve this contract.**

## 1. Processed data (Darren)

Path: `data/processed/scats_15min.csv` (name can be agreed).

Required columns in strict long format:

| scats_site | datetime | traffic_volume_15min |
|---|---|---|
| 2000 | 2006-10-01T08:00:00 | 126 |
| 2000 | 2006-10-01T08:15:00 | 134 |

**The rows above are fabricated format examples, not observations from the supplied workbook.**

- `scats_site`: SCATS site identifier as a string; not the VicRoads traffic counter `TFM_ID`.
- `datetime`: original observation time, local naive ISO 8601, aligned to 15-minute increments.
- `traffic_volume_15min`: observed nonnegative count within the 15-minute interval.
- One row per `(scats_site, datetime)`; duplicate handling is Darren's responsibility.
- Document how missing values, site-level gaps and any negative/corrupt data are handled.

The uploaded legacy `.xls` workbook could not be inspected using the available spreadsheet reader. Its actual worksheet layout and count direction/summing rules still require confirmation by Darren. Do not assume the example columns are present in the original.

## 2. Prediction (Jonathan GRU, Anson LSTM, Darren Regression Tree)

Baseline `models/gru_model.py` exposes:

`predict_gru(scats_site, as_of_time, horizon_minutes, processed_csv, model_dir) -> float`

Output is *vehicles in a predicted 15-minute interval*, not veh/hour. `as_of_time` is the time of the latest **observed** traffic sample; the target is `as_of_time + horizon_minutes`. Required historical observations must be available. Do not silently use future observations.

Train on chronological splits. Fit scaling parameters using training observations only. For research compare models on identical target timestamps/sites and use shared output units.

Model artifacts are site-specific and horizon-specific for this first **baseline**. It is not yet a full-network traffic predictor. For production, decide whether to train all needed sites or develop a multi-site model. The model must never silently invent predictions for sites/horizons without weights.

## 3. Traffic cost (Darren and Jonathan)

Before `routing/travel_time.py`, convert the relevant 15-minute predictions to an **hourly flow** based on a clearly defined aggregation policy. Four 15-minute forecasts may be summed for a one-hour window; multiplying one sample by four is only an approximation and must be labeled as such. Follow the lecturer's quadratic speed model, 60 km/h cap and 30-second intersection delay.

**Conflict to clarify with Dr. Joel:** dedicated conversion PDF says source-site volume (site A), but the main A2B PDF says destination-site volume (site B). `config.json` currently uses `origin` provisionally.

## 4. Routing (Jonathan and Darren)

Graph nodes are verified SCATS site IDs. Edges must represent actual road connectivity (not simply geographic proximity). Distances should ideally follow connected road segments, not a straight line unless the approximation is acknowledged. `routing.top_k_routes` ranks up to 5 unique simple routes from travel-time edge costs; do not use raw A2A weights as traffic-based travel times.

## 5. Before GUI integration

1. Darren delivers a processed file and identifies its timestamps and units.
2. The three members agree on target-time semantics and available date range.
3. Each member supplies a callable predictor with the same units.
4. Darren supplies a valid SCATS adjacency list and verified conversion function.
5. Jonathan connects predictors to travel-time costs, then top-k search.

**Don't submit the lecturer's example scripts.** Retain this own implementation, clearly document any adapted ideas and cite them.
