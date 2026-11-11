# COS30019 Group 2 — Assignment 2B routing + traffic conversion starter

This is a **routing foundation only**, built from your uploaded A2A source code.
It is **not a complete Assignment 2B submission**: it has no real SCATS dataset,
traffic prediction models, traffic-flow conversion, model evaluation, or A2B GUI.

## What is retained unchanged from A2A
`graph.py`, `node.py`, `search.py`, `visualize_app.py`, `visit_all.py`,
`algorithms/`, `testcases/`, and `run_15_testcases.py`.

## What is added
- `routing/top_k_routes.py` — Yen's K shortest simple paths using Dijkstra;
  works with A2A's `Graph.neighbours()` interface, returns at most five paths
  ordered by ascending cost.
- `routing/scats_adapter.py` — accepts *already verified* directed SCATS road
  edges with travel times; it does not create roads from nearby coordinates.
- `demo_routing.py` — demonstration with A2A graph weights (not real traffic times).
- `tests/test_routing.py` — eight prototype routing tests.

## Run
From this directory:

```bash
python run_15_testcases.py
python -m unittest discover -s tests -v
python demo_routing.py PathFinder-test.txt 2 5
```

These route examples have arbitrary A2A costs. **Do not label them predicted
travel times** until Darren's conversion and real data are connected.

## Integration contract (draft — agree with the group)
- Darren provides SCATS sites and verified directed roads, including travel times
  in *seconds*. Use `build_weighted_scats_graph(site_coords, road_edges_seconds)`.
  Latitude/longitude is stored for future use, but `graph.distance()` is
  Euclidean and not valid for computing geographic metres.
- Jonathan calls `find_top_k_routes(graph, origin, destination, k=5)`.
  Result: `[(path_site_ids, total_seconds), ...]`.
- ML group agrees on `predict_traffic(site, when, model, horizon_minutes)`;
  the history/features needed by GRU/LSTM still need explicit design.
- Graph edge costs must be recomputed from predictions and the lecturer's
  traffic-flow-to-travel-time conversion. Do not mix 15-minute vehicle counts
  with hourly counts, or degrees of latitude with kilometres.
- The A2A `astar.py` heuristic is tailored to its XY coordinate system;
  do not use it unchanged with raw SCATS coordinates and time weights.

## Remaining work (not implemented)
1. Inspect and clean SCATS traffic dataset (Darren).
2. Verify SCATS road connectivity and geographic distance (Darren).
3. Apply provided traffic-to-travel-time formula and units (Darren).
4. Build GRU (Jonathan), LSTM (Anson), Regression Tree (Darren).
5. Evaluate all models fairly, using chronological validation/testing (Anson).
6. Connect inference, conversion, and route search in `main.py` (Jonathan).
7. Build date/time/model/route GUI (Jonathan, Anson assisting with plots).
8. Produce 15 A2B-specific documented tests and experiment results (everyone).

Note: this demonstration does not prove A2B GUI input validation or actual
routing changes from predicted traffic. Test fixtures simulate weights only.


## Additional uploaded dataset review (2026-11-12)
- `routing/travel_time.py` implements the supplied under-capacity traffic flow-to-speed relation, 60 km/h cap and intersection delay.
- `tests/test_travel_time.py` validates conversion and demonstrates a synthetic traffic-caused route change.
- `config.json` contains explicit provisional flow-site choice.
- `docs/DATASET_AND_INTEGRATION_NOTES.md` describes uploaded data sources, unresolved ID mapping and flow-site contradiction.
- Run `python -m unittest discover -s tests -v` to run original routing and additional conversion tests.
- The .xls sources have not been parsed; models are NOT trained; road connectivity is NOT mapped.
