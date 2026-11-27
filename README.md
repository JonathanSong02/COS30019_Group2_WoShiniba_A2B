# COS30019 Assignment 2B – Traffic-Based Route Guidance System (Group 2)

TBRGS predicts traffic flow at 40 Boroondara SCATS sites (VicRoads, October 2006)
with **GRU**, **LSTM** and **Regression Tree** models, converts the forecasts into
road travel times, and finds up to five routes between two SCATS sites using the
six search methods from Assignment 2A (A* by default).

> Times are historical (October 2006 test days). Results are assignment simulations,
> not live navigation advice.

## 1. Install

Python 3.12 on Windows 11, from the project root (PowerShell):

```powershell
python -m pip install -r requirements.txt
```

## 2. Run

```powershell
python -m gui.app          # GUI: Traffic Prediction, Route Guidance, Model Comparison tabs
```

Open the local address printed in the terminal (usually http://127.0.0.1:7860).
Example route: origin `2000`, destination `3002`, time `2006-10-30T20:00`, model `GRU`,
horizon `15`, search `A*`, up to `5` routes.

Command-line equivalents:

```powershell
python main.py predict --site 2000 --model gru --time 2006-10-30T20:00:00 --horizon 15
python main.py route --origin 2000 --destination 3002 --model gru --time 2006-10-30T20:00:00 --horizon 15 --hourly-policy steady_4x --search-algorithm AS
```

Defaults (model, horizon, number of routes, speed limit, intersection delay, flow
convention, search method) are stored in `config.json`.

## 3. Test

```powershell
python -m unittest discover -s tests -v   # 166 unit tests
python run_a2b_testcases.py               # 28 documented A2B cases -> A2B_Test_Results.csv
python run_15_testcases.py                # 15 original A2A cases x 6 methods -> A2A_15_Test_Results.csv
```

## 4. Reproduce models and results

```powershell
python -m data_processing.scats_preprocess                    # raw .xls -> data/processed (needs data/raw)
python -m models.train_all_gru --horizons 15 30 60            # GRU, all 40 sites (skips existing)
python -m evaluation.run_lstm_experiments --sequence-lengths 8 --horizons 30 --sites <ids> --results-dir results/lstm/full_network_h30_s8
python -m models.regression_tree_model                        # Regression Trees
python -m evaluation.compare_three_models                     # matched 40-site comparison, 15/30/60 min
python -m evaluation.evaluate_search_methods --sample 30      # research: six search methods, 60 trips
```

## 5. Project layout

| Folder / file | Purpose |
|---|---|
| `data_processing/` | SCATS cleaning, 15-minute grid, chronological split, road-graph building |
| `data/processed/` | Processed series, train-only scaler, SCATS sites and directed road edges |
| `models/` | GRU, LSTM and Regression Tree training and prediction |
| `saved_models/` | Trained models for 40 sites × 3 models × 15/30/60 minutes |
| `routing/` | Flow → travel-time conversion, SCATS graph adapter, top-k route search |
| `main.py` | Prediction and routing API and command line |
| `gui/` | Gradio interface and route map |
| `evaluation/`, `results/` | Metrics, model comparisons, research experiments and their outputs |
| `tests/`, `run_a2b_testcases.py` | Unit tests and the 28 documented acceptance cases |
| `graph.py`, `node.py`, `search.py`, `algorithms/`, `testcases/`, `run_15_testcases.py` | Assignment 2A search code reused by A2B routing |
| `visit_all.py`, `visualize_app.py`, `PathFinder-test.txt` | Assignment 2A extras (not used by A2B) |
| `docs/` | Technical notes (data contract, road links, search integration, test cases) |

## 6. Key assumptions

- Each model predicts one 15-minute count; hourly flow = 4 × that count, divided by the
  site's number of approach detectors (`flow_conversion: per_approach`).
- Speed comes from `flow = -1.46484375·speed² + 93.75·speed` (under-capacity branch),
  capped at 60 km/h; link time = distance / speed + 30 s intersection delay.
- Edge A → B uses the flow at the starting site A, as in the conversion document
  (`volume_used_for_edge: origin`; `destination` is available as an option).
- Sites predicted above 1,500 vehicles/hour per approach lose their outgoing links
  (**Reject**, default). **Clamp** is an experimental sensitivity option.
- A* and IDA* return the five fastest loop-free routes; DFS, BFS, GBFS and IDDFS return
  five valid routes in their own search order. See `docs/SIX_A2A_SEARCH_INTEGRATION.md`.
- Road distances were reviewed against OpenStreetMap/OSRM but are estimates, not surveyed
  measurements. See `docs/SCATS_ROAD_LINKS.md`.
