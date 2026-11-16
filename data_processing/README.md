# SCATS data processing (Darren)

## Setup and run

```
pip install pandas numpy xlrd>=2.0.1
python -m data_processing.scats_preprocess --input "data/raw/Scats Data October 2006.xls"
python -m unittest discover -s tests -v
```

Put the raw .xls in `data/raw/` (not committed). Outputs go to `data/processed/`.

## Output: `scats_15min.csv`

| column | meaning |
|---|---|
| `scats_site` | SCATS site number as int (raw file stores it as text with a leading zero, e.g. `0970` -> `970`) |
| `datetime` | **start** of the 15-min interval, `YYYY-MM-DD HH:MM` (raw `V00` -> `00:00`, `V95` -> `23:45`) |
| `traffic_volume_15min` | vehicles in that 15 min, summed over all core approaches of the site; **blank = missing** |

The file is a complete grid of 40 sites x 31 days x 96 steps = 119,040 rows, so consecutive rows of a site are always exactly 15 min apart. 2.9% of values are blank (missing). Don't fill them with 0.

## Cleaning decisions
- **Approach key = Location + detector number.** Site 4335 has two different detectors with the same Location name, different coordinates and different counts. They look like duplicates but aren't, so both are kept.
- **Incomplete site-days are blank, not partial sums.** When one approach of a site didn't report, summing the rest would look like a fake traffic drop.
- **Detectors that fail for good are dropped for the whole month.** An approach only counts if it reports at least once in every split (train, val, test). Warrigal Rd S at 3685 and Glenferrie Rd S at 4264 stop on 23 Oct and never come back. Keeping them blanked all of 23-31 Oct (the whole val and test period), so GRU training failed for those sites. Their totals now use the remaining 2 (3685) and 3 (4264) approaches all month, so the series is consistent and every site has val and test data. Only these two sites changed, and the train scaler (max 1409) is unchanged.
- **Rare approaches are ignored.** Only approaches present on at least 50% of a site's days count. This only affects 3001 (Church St, 2 of 31 days).
- **Missing days.** The VicRoads notes say poor data was removed for some sites. Five sites (3002, 3127, 4262, 4263, 4812) all miss 24 to 28 Oct. Full list is in `data_quality_report.md`.
- **Coordinates.** 31 rows have lat/lon = 0 and are ignored. Site coordinates in `scats_sites.csv` are the median of the remaining rows.

## Split (chronological, same for every site)

| split | days | weekdays |
|---|---|---|
| train | 1 to 22 Oct (22 days) | 3 full weeks + Sun |
| val | 23 to 26 Oct (4 days) | Mon to Thu |
| test | 27 to 31 Oct (5 days) | Fri to Tue, includes a weekend |

## Leakage rules (enforced in `dataset.py`, checked in `tests/test_preprocessing.py`)
- Min-max scaler is fitted on **training values only** and saved to `scaler.json` (min 0, max 1409). Use `inverse_minmax` to get vehicles back before computing RMSE.
- A window belongs to the split of its **target**. Its lag inputs may reach back into the previous split, because that is past data.
- Windows that touch a missing value are dropped. Nothing is interpolated.
- No shuffling before splitting. Shuffling the training windows afterwards is fine.

## Using it in your model

```python
from data_processing.dataset import load_series, load_scaler, make_windows, inverse_minmax
df = load_series("data/processed/scats_15min.csv")
scaler = load_scaler("data/processed/scaler.json")
X_tr, y_tr, meta_tr = make_windows(df, "train", scaler, lags=12)   # (82956, 12)
X_va, y_va, meta_va = make_windows(df, "val", scaler, lags=12)     # (13056, 12)
X_te, y_te, meta_te = make_windows(df, "test", scaler, lags=12)    # (17208, 12)
X_tr = X_tr.reshape(len(X_tr), 12, 1)   # LSTM/GRU input shape, as in the hyswin example
```

`meta_*` gives the site and target time of each window, for per-site metrics and plots. All three models should use the same `lags` and `horizon` so the comparison is fair.

## Open issue: site totals vs the 1500 veh/h capacity
Summed site totals converted to an hourly rate exceed the conversion PDF's 1500 veh/h capacity **42% of the time** (median 1307, max 5389 veh/h). `routing/travel_time.py` raises `ValueError` above 1500. Per-approach hourly volumes exceed 1500 only 0.65% of the time (max 2345). The group needs to decide which volume feeds the travel-time formula: the approach volume (`scats_15min_by_approach.csv`), or the site total divided by its number of approaches.

## Road graph: `scats_road_edges.csv`

```
python -m data_processing.build_road_graph
python -m unittest tests.test_road_graph -v
```

Columns: `from_site,to_site,distance_km`. There are 136 directed edges (68 two-way roads) covering all 40 sites, and the graph is fully connected.

- **Edges follow real roads, not proximity.** Each link comes from the road names in the SCATS `Location` field and was checked against the VicRoads Boroondara map (`IBM data_Bor.pdf`). The full list, with road names, is `ROAD_LINKS` in `build_road_graph.py`.
- **All roads are two-way**, so every link appears in both directions.
- **Distance:** `direct` links (two sites next to each other on one road) use the haversine straight-line distance. `junction` links (the route turns once at an intersection that isn't in our dataset) use an L-shaped north-south plus east-west estimate. Distances are approximations, not measured along road curves.
- `scats_road_edges_detailed.csv` adds `road`, `link_type` and `confidence`. The 14 `medium` links (orange in `docs/scats_road_graph.png`) are freeway or road-continuation links: Eastern Fwy, Swan St -> Riversdale Rd, the High St (Glen Iris) links, and the junction links. Check these on a street map if they matter for a demo route.
- **Coordinate offset (measured):** SCATS coordinates sit about 200 m south-west of the real intersections. Comparing 14 signalised intersections with OpenStreetMap (the node shared by the two named roads) gave a near-constant shift of +0.001505° lat / +0.001305° lon (about 168 m north, 115 m east). Applying the median shift cuts the mean position error from 202.5 m to 3.9 m (max 13.4 m). The values are in `data/processed/scats_coordinate_offset.json`, per-site measurements in `scats_coordinate_offset_samples.csv`. The shift is used only by the GUI map (`gui/route_map.py`); because it is the same for every site, routing distances are unaffected and still use the original coordinates.
