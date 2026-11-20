# SCATS Boroondara Road Graph — Current Status (20 Nov 2026)

The active TBRGS network is `data/processed/scats_road_edges.csv`, not the earlier prototype `scats_road_links.csv`. The network contains **40 SCATS sites and 136 directed edges**, corresponding to 68 pairs represented in both directions; the test suite confirms graph connectivity. The connected pairs were checked against named roads and Darren's OpenStreetMap/OSRM review. The details and corrections are recorded in `data/processed/scats_road_edges_review.csv`, `data/processed/scats_road_edges_detailed.csv`, and `data/processed/road_link_review_osrm.json`.

**Accuracy boundary:** Even after review, the graph is a teaching approximation. Some distances are road-routing estimates while others may still be geometric approximations. Network links may skip unmodelled intersections. Bidirectional representation assumes road travel is possible in both directions and does not prove all turn restrictions. The GUI draws simplified links rather than full road-centreline polylines. Do not claim the graph is a fully validated real-world road network.

Road CSV format:

```csv
from_site,to_site,distance_km
```

`main.py` reads this file via `load_verified_road_links()` and computes edge costs by predicting per-site counts (GRU / LSTM / Regression Tree), estimating average **per-approach** hourly flow with the explicit `steady_4x` assumption, and applying the supplied flow-to-speed conversion plus 30 seconds per controlled intersection. `capacity_policy=reject` is the default; `clamp` is experimental.

Example:

```powershell
python main.py route --origin 2000 --destination 3002 --model gru --time 2006-10-30T20:00:00 --horizon 15 --hourly-policy steady_4x
```

On the uploaded historical data this returned five candidate routes, fastest **12.216 minutes** via 2000→4043→4273→4270→4263→3002 under strict Reject with no clamping. Consult `data_processing/README.md` for traffic count and approach-count construction; see `README_A2B_START_HERE.md` for GUI/testing instructions. Confirm graph accuracy and coordinate offsets before claiming real-world journey-time performance.
