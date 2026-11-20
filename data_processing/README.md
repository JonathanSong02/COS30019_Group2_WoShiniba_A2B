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

## Flow conversion: per-approach flow for the 1500 veh/h formula (Task 6)
**Decision (19 Nov, accepted by Dr. Joel):** before the flow-to-speed formula, each site's hourly flow is divided by its number of approaches. The model predictions themselves are not changed.

```
site_total_veh_per_hour = 4 x predicted 15-minute count        (steady_4x)
flow_used_veh_per_hour  = site_total_veh_per_hour / n_core_approaches
```

`n_core_approaches` comes from `scats_sites.csv`. It is the number of approach detectors whose counts were summed into `traffic_volume_15min` (see Cleaning decisions), so the division gives the average flow on one approach road. Sites 3685 and 4264 use their reduced counts (2 and 3) after the broken detectors were dropped, which matches how their totals were built. Counts across the 40 sites: 22 have 4 approaches, 11 have 3, 4 have 2, 2 have 1 and 1 has 6.

**Why:** a SCATS site volume is the total over all roads into the intersection, but the conversion PDF's formula describes one road, whose maximum flow is 93.75^2 / (4 x 1.4648) = 1500 veh/h. On the observed October data (115,584 non-missing 15-minute intervals):

| Flow fed to the formula | Intervals over 1500 veh/h | Max veh/h |
| --- | --- | --- |
| Site total (old behaviour) | 43.0% | 5,389 |
| Site total / approach count (new default) | 0.08% (88 intervals) | 1,820 |
| Individual approach detectors (`scats_15min_by_approach.csv`) | 0.65% | 2,345 |

With site totals, Reject failed almost all day and Clamp set most sites to the same speed. With per-approach flow, almost every time of day routes under Reject. The 88 intervals that remain over 1500 are morning and evening peaks at three sites: 2825 (Burke Rd S of Eastern Fwy, 1 approach, 60 intervals, mostly 08:00-08:30), 4262 (Bridge Rd SW of Burwood Rd, 1 approach, 17) and 3001 (3 approaches, 11). At those times Reject stops, as the brief's under-capacity assumption requires, for example 2000 -> 3002 at Mon 30 Oct 08:00.

**Settings:** `flow_conversion` in `config.json` and `main.py route --flow-conversion` choose `per_approach` (default) or `site_total` (old behaviour, kept for before/after comparison). `capacity_policy` stays `reject` by default. `clamp` is an experimental option that caps the per-approach flow at 1500 and is labelled as such in the GUI and in the route output. Every route result lists, per site, the predicted 15-minute count, the hourly site total, the approach count and the flow used (`assumptions.site_flows`).

**Edge flow site:** the flow on an edge A -> B comes from its starting site A, as the conversion PDF specifies, and this is applied to every edge (`volume_used_for_edge: origin`). The A2B brief mentions the destination. Dr. Joel accepted either one if it is used consistently and justified.

**Assumptions and limitations:**
- Traffic splits evenly across a site's approaches. In reality the main road carries more than the side road, so the busier approach is underestimated and the quieter one overestimated.
- One flow per site is used for every road leaving it, though each edge is one road in one direction. The approach detectors are not matched to individual road links.
- Detector counts include turning traffic and both lanes of an approach. The 1500 veh/h capacity is the PDF's simplified value, not a per-lane capacity.
- The 2 single-approach sites (2825, 4262) are unchanged by the division, so they are the ones that can still exceed 1500.
- The hourly flow is still 4 x one 15-minute forecast (steady_4x), not a forecast of four separate intervals.

## Road graph: `scats_road_edges.csv`

```
python -m data_processing.build_road_graph
python -m unittest tests.test_road_graph -v
```

Columns: `from_site,to_site,distance_km`. There are 136 directed edges (68 two-way roads) covering all 40 sites, and the graph is fully connected.

- **Edges follow real roads, not proximity.** Each link comes from the road names in the SCATS `Location` field and was checked against the VicRoads Boroondara map (`IBM data_Bor.pdf`). The full list, with road names, is `ROAD_LINKS` in `build_road_graph.py`.
- **All roads are two-way**, so every link appears in both directions.
- **Distance:** `direct` links (two sites next to each other on one road) use the haversine straight-line distance. `junction` links (the route turns once at an intersection that isn't in our dataset) use an L-shaped north-south plus east-west estimate. Distances are approximations, not measured along road curves.
- `scats_road_edges_detailed.csv` adds `road`, `link_type` and `confidence`. The 15 `medium` links (orange in `docs/scats_road_graph.png`) are freeway or road-continuation links: Eastern Fwy, Swan St -> Riversdale Rd, the High St (Glen Iris) links, and the junction links. Check these on a street map if they matter for a demo route.
- **Coordinate offset (measured):** SCATS coordinates sit about 200 m south-west of the real intersections. Comparing 14 signalised intersections with OpenStreetMap (the node shared by the two named roads) gave a near-constant shift of +0.001505° lat / +0.001305° lon (about 168 m north, 115 m east). Applying the median shift cuts the mean position error from 202.5 m to 3.9 m (max 13.4 m). The values are in `data/processed/scats_coordinate_offset.json`, per-site measurements in `scats_coordinate_offset_samples.csv`. The shift is used only by the GUI map (`gui/route_map.py`); because it is the same for every site, routing distances are unaffected and still use the original coordinates.

## Road link review against OpenStreetMap (Task 7)

Run: `python -m data_processing.review_road_links` (needs internet, about 3 minutes; tests: `python -m unittest tests.test_road_link_review`).

For every link, the script asks the OSRM routing service (`router.project-osrm.org`, which finds driving routes on OpenStreetMap road data) for the driving route between the two sites, in both directions, and compares the roads the route uses and its length with our estimate. Site positions are the offset-corrected coordinates. OSRM moves each point onto the nearest road; next to a freeway that can be the freeway itself, so for the six links in `SNAP_TO_ROAD` both ends are first moved onto the link's named road using OpenStreetMap's map data (the nearest node of that road, 2-49 m from the corrected site). OSRM's raw answers are cached in `data/processed/road_link_review_osrm.json`. Results per link: `data/processed/scats_road_edges_review.csv`.

| Result | Links | Meaning |
|---|---|---|
| `VERIFIED` | 61 | The route uses the named roads and the road distance is within 20% of our estimate |
| `DISTANCE_SHORT` | 5 | Right road, but it curves or jogs, so our straight line is 20-32% short: Eastern Fwy 2820-2825 (ramps), Princess St 2820-3662, Cotham Rd 3662-4324, Burke Rd 4030-4032 (Kew Junction), Burwood Rd 4262-4263 |
| `DISTANCE_LONG` | 1 | 2000-3812 (Camberwell Rd + Toorak Rd): the L-shaped estimate is 4.54 km, the road 3.57 km, because Camberwell Rd runs diagonally |
| `WRONG_ROAD_FIXED` | 1 | 2825-2827 was labelled Eastern Fwy, but there is no freeway connection between the Burke Rd and Bulleen Rd interchanges. The shortest drive uses local streets (The Boulevard, Aquila St), 2.258 km instead of the 1.451 km straight line. The link is now `measured` with that distance (`MEASURED_KM` in `build_road_graph.py`) |

Over the other 67 links, the road distance (shorter direction) is a median **1.007x** our estimate (mean 1.033x): 59 links are within 10% and 62 within 20%. The shorter direction is used because the longer one is often a detour forced by turn bans or one-way snapping, not the road length itself.

**Distance limitations (for the report):**

- Distances are straight-line (`direct`) or L-shaped (`junction`) estimates, not lengths along the road, except the one `measured` link. They are about right on average (median 1.007x) but up to 32% short on curving roads and 21% long on one diagonal road.
- Only the wrong link was changed, so the routes and comparisons from earlier tasks still hold. Using the OSM distances for every link would be a possible improvement; the values are in the review file (`osm_km_shorter`).
- Every road is treated as two-way with the same length both ways. OSM shows that turn bans make some trips longer in one direction (for example Riversdale Rd 4040 to 3804: 0.63 km one way, 1.47 km back), which the graph does not model.
- Freeway links (2820-2825, 2846-4043 via Monash Fwy ramps) use the same 60 km/h speed cap and flow formula as arterial roads.
- The check relies on OSM data and one router; it confirms that a drivable connection exists and how long it is, not traffic rules such as lane closures in October 2006.

