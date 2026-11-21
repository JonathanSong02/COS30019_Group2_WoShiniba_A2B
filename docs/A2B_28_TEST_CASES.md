# Assignment 2B — 28 documented acceptance cases

Generated from `python run_a2b_testcases.py` on the corrected version. These are the exact recorded inputs, expected outputs and actual results; see CSV for machine-readable data.

## TC01 — Processed data is a complete 15-minute grid
- Area: Data processing
- Input: data/processed/scats_15min.csv
- Expected: 119040 rows (40 sites x 2976 steps), 40 sites, 0 duplicates
- Actual: 119040 rows, 40 sites, 0 duplicates
- Status: **PASS**

## TC02 — A missing day stays missing (not filled with zeros)
- Area: Data processing
- Input: Site 3002, 24 Oct 2006
- Expected: 96 blank intervals, 0 zeros
- Actual: 96 intervals: 96 blank, 0 zero
- Status: **PASS**

## TC03 — Sites 3685 and 4264 have validation and test data after the detector fix
- Area: Data processing
- Input: Sites 3685, 4264
- Expected: more than 0 readings in val (23-26 Oct) and test (27-31 Oct) for both
- Actual: 3685 val: 384, 3685 test: 480, 4264 val: 384, 4264 test: 480
- Status: **PASS**

## TC04 — Chronological split with no overlap
- Area: Data processing
- Input: data/processed/split_config.json
- Expected: train before val before test, no shared days
- Actual: train 2006-10-01..2006-10-22, val 2006-10-23..2006-10-26, test 2006-10-27..2006-10-31
- Status: **PASS**

## TC05 — Regression Tree prediction through main.py equals its evaluated value
- Area: Prediction
- Input: Site 2000, last observed 2006-10-30T08:00:00, +15 min
- Expected: 1374.40 vehicles / 15 min (saved test prediction)
- Actual: 1374.40 (saved evaluation value 1374.40)
- Status: **PASS**

## TC06 — GRU prediction through main.py equals its evaluated value
- Area: Prediction
- Input: Site 2000, last observed 2006-10-30T08:00:00, +15 min
- Expected: 1329.93 vehicles / 15 min (saved test prediction)
- Actual: 1329.93 (saved evaluation value 1329.93)
- Status: **PASS**

## TC07 — LSTM prediction through main.py equals its evaluated value
- Area: Prediction
- Input: Site 2000, last observed 2006-10-30T08:00:00, +15 min
- Expected: 1364.02 vehicles / 15 min (saved test prediction)
- Actual: 1364.02 (saved evaluation value 1364.02)
- Status: **PASS**

## TC08 — 60-minute horizon works and matches its evaluated value
- Area: Prediction
- Input: Regression Tree, site 2000, last observed 2006-10-30T08:00:00, +60 min
- Expected: same as saved test prediction for 09:00
- Actual: 1142.87 (saved evaluation value 1142.87)
- Status: **PASS**

## TC09 — Unknown SCATS site is rejected
- Area: Input validation
- Input: Site 9999
- Expected: error, no prediction
- Actual: rejected: FileNotFoundError: No Regression Tree trained for site=9999, horizon=15, sequence=8
- Status: **PASS**

## TC10 — Time not on a 15-minute boundary is rejected
- Area: Input validation
- Input: Site 2000, 2006-10-30T08:07
- Expected: error, no prediction
- Actual: rejected: ValueError: --time must be a local naive datetime aligned to 15-minute intervals
- Status: **PASS**

## TC11 — Missing history is reported, not filled with a made-up value
- Area: Input validation
- Input: Site 3002, 2006-10-24T10:00 (day with no data)
- Expected: error, no prediction
- Actual: rejected: ValueError: Insufficient nonmissing historical observations for this time
- Status: **PASS**

## TC12 — Low flow drives at the 60 km/h speed limit
- Area: Travel time
- Input: 300 veh/h, 1 km edge
- Expected: 60 km/h; 60 s driving + 30 s intersection delay = 90 s
- Actual: 60.00 km/h, 90.0 s
- Status: **PASS**

## TC13 — At capacity the speed is about 32 km/h
- Area: Travel time
- Input: 1500 veh/h, 1 km edge
- Expected: about 32 km/h; about 142.3 s
- Actual: 32.07 km/h, 142.3 s
- Status: **PASS**

## TC14 — Medium flow uses the under-capacity (green) branch
- Area: Travel time
- Input: 1000 veh/h
- Expected: about 50.48 km/h (between 32 and 60)
- Actual: 50.48 km/h
- Status: **PASS**

## TC15 — Over-capacity forecast: Reject excludes affected edges; Clamp is experimental
- Area: Travel time
- Input: 2000 -> 3002, Regression Tree, 2006-10-30T08:00:00 (morning peak)
- Expected: Reject: 5 feasible routes with 2825 edges excluded; Clamp: 5 routes with only 2825 capped
- Actual: Reject -> 5 feasible routes; excluded site 2825 and 3 affected links; experimental Clamp -> 5 routes, capped sites [2825]
- Status: **PASS**

## TC16 — Per-approach flow is the site total divided by the approach count
- Area: Travel time
- Input: 2000 -> 3002, Regression Tree, 2006-10-30T20:00:00; site 2000 (4 approaches)
- Expected: prediction unchanged; site total = 4 x prediction; flow used = site total / 4
- Actual: predicted 525.70 / 15 min, site total 2102.8 veh/h, 4 approaches, flow used 525.7 veh/h
- Status: **PASS**

## TC17 — Per-approach conversion brings observed flows under the 1500 veh/h capacity
- Area: Travel time
- Input: All non-missing intervals in data/processed/scats_15min.csv
- Expected: site totals over 1500 in more than 40% of intervals; per-approach in less than 0.1%
- Actual: 115584 intervals: site total over 1500 in 43.0%, per-approach in 0.08% (88 intervals)
- Status: **PASS**

## TC18 — Road graph is complete, two-way and connected
- Area: Road graph
- Input: data/processed/scats_road_edges.csv
- Expected: 40 sites, 68 two-way roads, connected
- Actual: 40 sites, 68 two-way roads, connected
- Status: **PASS**

## TC19 — Link fixed by the OpenStreetMap review uses its measured distance
- Area: Road graph
- Input: 2825 <-> 2827
- Expected: 2.258 km both ways (not the 1.451 km straight line)
- Actual: 2825->2827 2.258 km, 2827->2825 2.258 km
- Status: **PASS**

## TC20 — Brief's example trip returns 5 valid routes, fastest first
- Area: Routing
- Input: 2000 -> 3002, Regression Tree, 2006-10-30T20:00:00, Reject
- Expected: 5 routes; ascending times; start 2000, end 3002; no repeated site; every hop is a road
- Actual: 5 routes, 12.28-12.47 min, best 2000-4043-4273-4270-4263-3002
- Status: **PASS**

## TC21 — Predicted traffic changes the best route
- Area: Routing
- Input: 2000 -> 3002, Regression Tree, 03:00 vs 20:00
- Expected: different best routes; the 03:00 trip is faster
- Actual: 03:00 best 2000-3812-4040-4266-4264-4263-3002 (12.1 min); 20:00 best 2000-4043-4273-4270-4263-3002 (12.3 min)
- Status: **PASS**

## TC22 — Origin equal to destination
- Area: Routing
- Input: 2000 -> 2000
- Expected: one route [2000], 0 s
- Actual: [{'rank': 1, 'scats_sites': [2000], 'travel_time_seconds': 0.0, 'travel_time_minutes': 0.0}]
- Status: **PASS**

## TC23 — Unknown origin is rejected
- Area: Routing
- Input: 9999 -> 3002
- Expected: error, no routes
- Actual: rejected: ValueError: Origin/destination must be known SCATS site IDs
- Status: **PASS**

## TC24 — All three models give full-network routes for the same trip
- Area: Routing
- Input: 2000 -> 3002, 2006-10-30T20:00:00, Reject
- Expected: 5 routes for each model on the full 40-site network, no site excluded or capped
- Actual: gru: 5 routes, best 12.22 min via 2000-4043-4273-4270-4263-3002; lstm: 5 routes, best 12.28 min via 2000-4043-4273-4270-4263-3002; regression_tree: 5 routes, best 12.28 min via 2000-4043-4273-4270-4263-3002
- Status: **PASS**

## TC25 — Prediction tab explains a badly formatted date
- Area: GUI
- Input: '30/10/2006 8am'
- Expected: message containing 'unavailable', no number
- Actual: status: ⚠️ **Prediction unavailable:** --time must be ISO format, e.g. 2006-10; value: None
- Status: **PASS**

## TC26 — Route guidance tab returns the route table and the map
- Area: GUI
- Input: 2000 -> 3002, Regression Tree, 2006-10-30T20:00:00, Reject
- Expected: 5 table rows and a map
- Actual: 5 table rows, map present, best 2000 → 4043 → 4273 → 4270 → 4263 → 3002
- Status: **PASS**

## TC27 — Remote over-capacity site does not cancel a feasible selected journey
- Area: Routing
- Input: Synthetic four-site graph; site 3 over capacity; trip 1 -> 4
- Expected: 1 -> 2 -> 4 remains valid; only edge from site 3 excluded with warning
- Actual: valid path [1, 2, 4]; capacity excluded [3], 1 edge
- Status: **PASS**

## TC28 — Missing numeric detector value is not summed as a complete site total
- Area: Data processing
- Input: Synthetic site 99, approaches A=8 and B=missing at the same interval
- Expected: Site total is missing, never a partial count of 8
- Actual: partly missing detector value produces blank site-total
- Status: **PASS**
