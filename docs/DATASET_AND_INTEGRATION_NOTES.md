# Review of user-supplied A2B files — 2026-11-12

## Input file roles

1. `Scats Data October 2006.xls`: primary historical 15-minute SCATS traffic-volume data for training and evaluation. The file is legacy Excel .xls. **The content of its sheets has not been programmatically validated in this starter.** Darren must export/parse it and document column names, missing values, site coverage, and train/validation/test split before GRU/LSTM training.
2. `SCATSSiteListingSpreadsheet_VicRoads.xls`: intended SCATS intersection site metadata; its column names and coverage have not yet been validated.
3. `Traffic_Count_Locations_with_LONG_LAT.csv`: a *different* Victoria traffic-count location dataset with columns X,Y,TFM_ID,TFM_DESC,...,AADT_ALLVE. It has 57,947 records, and a location with TFM_ID=3002 is **not** the SCATS 3002 example in the assignment. Never equate TFM_ID with SCATS site ID without verified mapping.
4. `IBM data_Bor.pdf`: Boroondara sketch/map with numbered SCATS intersections; not a machine-readable proof of road connections.
5. `Traffic Flow to Travel Time Conversion v1.0.pdf`: official flow–speed formula.

## Formula (provided PDF)

- `flow = -1.4648375 * speed^2 + 93.75 * speed`
- `flow` in vehicles per hour, `speed` in kilometres per hour.
- Choose the **high-speed / uncongested (green) root** under the assignment's under-capacity assumption.
- Cap speed at 60 km/hour; the graph's threshold is approximately 351.585 vehicles/hour.
- Model assumes capacity of 1500 vehicles/hour, at approximately 32 km/hour.
- Edge time (seconds) = distance_km / speed_kmh * 3600 + 30 seconds intersection delay.
- Conversion PDF says traffic on link A->B is determined using the **origin A** site flow. Main A2B brief instead mentions **destination B**. Ask tutor. `weights_from_road_links(..., flow_site='origin'|'destination')` supports both explicitly.
- Values above 1500 vehicles/hour raise `ValueError`; they require a documented strategy, not silent clamping to made-up speeds.

## Interface contract (prototype, to agree among members)

1. Darren supplies `verified_road_links` as `(origin_SCATS, destination_SCATS, road_distance_km)`, plus `site_coords` mapping SCATS to `(lat, lon)`.
2. ML modules supply one agreed output quantity: **hourly flow in vehicles/hour**, per chosen site and prediction timestamp. For 15-minute forecasts, construct an hourly rate consistently, e.g. four consecutive counts sum to an hour. Do not casually multiply one quarter-hour prediction by four unless explicitly treating it as a constant-rate estimate.
3. Jonathan calls `weights_from_road_links(links, predicted_hourly_flows, flow_site='origin')` and feeds the resulting weights to `build_weighted_scats_graph(site_coords, edges_seconds)` then `find_top_k_routes(graph, origin, destination, k=5)`.
4. The returned route cost is in **seconds**, not graph-coordinate distance or number of vehicles.
5. Route travel-time prediction is currently a single fixed-time *snapshot* of edge costs. A time-dependent route requires predictions aligned with each edge's expected arrival time; implement and report separately if attempted.
6. Dates/times are **dataset-relative** until a forecasting strategy for arbitrary later dates is defined. Do not present 2026 predictions as actual forecasts supported by October 2006 alone.

## What is NOT complete

No trained GRU/LSTM/Regression Tree, no genuine machine-readable SCATS road connectivity, no dataset-derived routing results, no live GUI, and no 15 A2B-specific system-level tests. The tests supplied use *synthetic* road links and traffic counts to check numerical and software integration correctness.
