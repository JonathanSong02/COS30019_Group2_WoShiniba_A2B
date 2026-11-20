# TBRGS `main.py` Integration — Group 2 (current status, 20 Nov 2026)

`main.py` integrates trained **GRU, LSTM and Regression Tree** prediction functions with 40-site Boroondara SCATS routing. The saved models are historical October 2006 models; their inputs are the *last observed* 15-minute interval. The target is `as_of + horizon_minutes`. Do not infer live/2026 predictions.

## API and CLI

- `predict_traffic(site, as_of_time, model_name, horizon_minutes, sequence_length=8, ...)` returns one future quarter-hour vehicle count. Model name: `gru`, `lstm`, `regression_tree`.
- `load_verified_road_links(data/processed/scats_road_edges.csv)` reads 136 directed links; no links are invented from coordinate proximity at runtime.
- `compute_traffic_routes(origin, destination, as_of_time, links, coords, predictor, ...)` predicts required sites, converts flows to travel times, builds the shared A2A graph adapter, and returns up to five distinct ranked paths with estimated seconds/minutes.
- `config.json` controls defaults for model, forecast horizon, route count, intersection delay, flow-site convention and flow conversion.

```powershell
python main.py predict --site 2000 --model gru --time 2006-10-30T20:00:00 --horizon 15
python main.py route --model gru --origin 2000 --destination 3002 --time 2006-10-30T20:00:00 --horizon 15 --hourly-policy steady_4x
```

The route CLI requires `--hourly-policy steady_4x` to make its single-forecast hourly approximation explicit. Optional `--capacity-policy clamp` is EXPERIMENTAL. `--flow-conversion site_total` reproduces the old whole-site interpretation for sensitivity comparison. The default is `per_approach`.

## Formula and limitations

- Predicted hourly **site total** = 4 × predicted vehicles/15 min (steady-flow approximation).
- Predicted hourly **approach average** = site total / `n_core_approaches`. This per-approach average is used in the lecturer's `flow = -1.4648375 × speed² + 93.75 × speed` curve, with speed capped at 60 km/h and traffic normally assumed below 1,500 vehicles/hour.
- Travel time for a link = distance/speed + a 30-second controlled-intersection delay. Distances and road topology are map-reviewed research approximations, not independently verified road geometry.
- Default edge flow comes from the **origin** site, matching the dedicated conversion PDF. The A2B brief uses the destination site; document the choice and the team's tutor consultation.
- Unknown sites, unsupported timestamps, missing model checkpoints and over-capacity forecasts are reported as errors, not silently replaced with invented traffic values.

## Evidence and testing

```powershell
python -m unittest discover -s tests -v
python run_a2b_testcases.py
python -m gui.app
```

On 20 Nov 2026 in the review environment: **143/143** unit tests and **26/26** documented end-to-end acceptance scenarios passed. An example GRU route (2000→3002, last observed 30 Oct 2006 20:00, strict Reject, per-approach) returned 5 routes; fastest was **12.216 minutes** via `2000→4043→4273→4270→4263→3002` with no capacity clamping. These are assignment simulation outputs, not field-measured travel times.

See `README_A2B_START_HERE.md` and `docs/SUBMISSION_CHECKLIST.md` before submitting.
