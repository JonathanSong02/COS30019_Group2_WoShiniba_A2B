# COS30019 Assignment 2B — Group 2 TBRGS (current implementation)

> Updated 21 November 2026 after the six-method routing integration, retaining Darren's dataset and map updates. This is the **current** quick-start guide, replacing the original routing-only development notes. The project's GRU, LSTM, Regression Tree, Gradio GUI, 40-site road graph and prediction-driven routing are implemented. This does not mean that all assumptions are independently validated or the report is ready for submission.

## Start here (PowerShell, from the project root)

```powershell
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python run_a2b_testcases.py
python -m evaluation.evaluate_search_methods
python -m gui.app
```

The interface normally starts at a local Gradio URL printed in the terminal. In the Route Guidance tab try origin `2000`, destination `3002`, last-observed time `2006-10-30T20:00:00`, model `GRU`, horizon `15`, lookback `8`, up to `5` routes. The time is historical (held-out test period), **not a current traffic prediction**.

Equivalent command-line prediction and routing demonstrations:

```powershell
python main.py predict --site 2000 --model gru --time 2006-10-30T20:00:00 --horizon 15
python main.py route --origin 2000 --destination 3002 --model gru --time 2006-10-30T20:00:00 --horizon 15 --hourly-policy steady_4x --search-algorithm AS
```

## Implemented modules

- `data_processing/` — cleaning, per-site 15-minute grid, chronological train/validation/test split and road-graph preparation.
- `models/` and `saved_models/` — saved LSTM, GRU and Regression Tree checkpoints and prediction interfaces; 40 processed sites have 15-minute saved models for the three methods. Other horizon coverage is more limited.
- `evaluation/`, `results/` — error metrics (MAE, RMSE, NRMSE), matched three-model comparisons, peak/off-peak and route comparison evidence.
- `main.py`, `routing/` — converts forecast flow to road travel time and obtains up to five ranked, directed routes on the 40-site network using the selected A2A search strategy.
- `gui/` — Gradio Traffic Prediction, Route Guidance and Model Comparison tabs; interactive visualisations.
- `tests/`, `run_a2b_testcases.py`, `A2B_Test_Results.csv` — the updated unittest suite and 26 recorded A2B acceptance scenarios passed in the review environment on 20 November 2026. Results must also be verified on the team's Windows machines.
- Six A2A strategies (DFS, BFS, GBFS, AS/A*, CUS1/IDDFS, CUS2/weighted IDA*) now power TBRGS routing. AS is the default for globally lowest-cost top five. DFS/BFS/GBFS/CUS1 are for comparative route exploration, not globally optimal travel time. See `docs/SIX_A2A_SEARCH_INTEGRATION.md`.

## Traffic and distance assumptions

The model predicts ONE future **15-minute** site count. The `steady_4x` approximation multiplies it by four to estimate hourly site flow (it is not a four-step hourly forecast). `flow_conversion: per_approach` divides by `n_core_approaches` in `scats_sites.csv` to estimate average flow per approach before applying the lecturer's under-capacity flow-to-speed curve. This assumes equal traffic split across approaches. The `site_total` comparison option applies the older whole-site flow.

Default capacity policy is **Reject** above 1,500 vehicles/hour per approach. **Clamp** is an explicit experimental sensitivity scenario, not the normal result. `volume_used_for_edge: origin` is the default because the conversion PDF specifies starting-site flow; the assignment brief separately refers to destination flow. Record and justify the chosen convention in the report. Darren's data-processing notes state that Dr. Joel accepted using the start or end site consistently on 19 November; keep documentary evidence of that consultation.

The provided graph uses 40 SCATS sites and 136 directed links (68 bidirectional pairs). Darren conducted an OpenStreetMap/OSRM review, but road assumptions, direction restrictions, road-length approximations and skipped unsampled intersections remain limitations. Route times are simulated estimates for an October 2006 teaching exercise, not real-world navigation advice.

## Before submission

Follow `docs/SUBMISSION_CHECKLIST.md`. Report: 8–10 pages excluding cover and TOC; signed member page, contribution details, testing, research, features/bugs, acknowledgements and references. Submit the official `COS30019 - Who did What.docx` as required by Canvas. **Updated Canvas due date: 29 November 2026 at 23:59.**

Older READMEs (`README_GRU_NEXT.md`, `README_GUI_INSTALL.md`) describe development milestones, not the current system state.
