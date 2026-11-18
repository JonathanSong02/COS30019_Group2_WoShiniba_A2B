# Group 2 · TBRGS Gradio GUI patch

This patch adds `gui/app.py`, `gui/__init__.py`, and `tests/test_gui_backend.py` to your existing A2B project. It does not change `main.py`, A2A algorithms, Darren's preprocessing, Anson's LSTM, or existing model checkpoints.

## Install

From `C:\COS30019_Group2_WoShiniba_A2B`, extract the patch and retain the folder structure. Your `python -m pip install gradio` already succeeded.

## Run

```powershell
python -m gui.app
```

Open the local address printed in the terminal (typically `http://127.0.0.1:7860`). The GUI is local only (`share=False`).

Under **Traffic prediction**, use GRU / site `2000` / last observed `2006-10-20T10:00:00` / horizon `30` / lookback `8`. The output should be approximately `864.78 vehicles per 15 minutes` if your checkpoint files match the previously tested run.

Under **Route guidance**, the system expects a verified `data/processed/scats_road_links.csv` with `from_site,to_site,distance_km`. Until that exists and the required site-specific models are trained, this tab will return a clear error instead of producing synthetic routes.

## Tests

```powershell
python -m unittest discover -s tests -v
```

The patch adds four GUI backend tests. If the existing 54 are unchanged, you should see 58 total.

## Limitations

- Route guidance shows an interactive OpenStreetMap view (Leaflet, no API key): the 40 sites, the road links and the top-5 routes in different colours. Hover a site for its predicted flow, a route for its travel time. Sites are shifted by the measured SCATS offset (`data/processed/scats_coordinate_offset.json`). Routes are straight lines between intersections, and the map background needs internet (routing works offline).
- Model coverage is shown at the top of the GUI, counted from the files in `saved_models/`. Regression Tree: all 40 sites at 15/30/60 min. GRU: 38 sites at 15 min (no 3685, 4264), site 2000 only at 30/60. LSTM: site 2000 only.
- Model comparison tab: GRU vs LSTM vs Regression Tree from `results/model_comparison/` (Anson's matched three-model comparison plus `python -m evaluation.regression_tree_extras`): network and per-site error charts, site 2000 by horizon, peak vs off-peak, Regression Tree feature importance, a results table with training time and model size, an actual-vs-predicted chart for any site and test day, and the same trip routed under each model. Route comparison defaults to Clamp at Mon 30 Oct 20:00, because from 07:00 to 18:00 most site totals exceed 1,500 veh/h. `python -m evaluation.compare_routes` saves route comparisons for the report.
- Traffic prediction shows a chart of the forecast in context: observed traffic for the 4 hours before (model input window shaded), what actually happened afterwards, and each model's prediction at the forecast time, with a table of prediction, actual and error. Default time is Mon 30 Oct 2006 08:00 (test split, complete data at every site).
- GRU output is one future 15-minute interval, not an hourly vehicle count.
- The provisional `steady_4x` hourly approximation and origin/destination flow convention must be confirmed/documented for final evaluation.
- `Gradio` is not added to the existing UTF-16 encoded `requirements.txt` in this patch. For packaging, ensure your final dependency file is UTF-8 and contains a compatible Gradio version.
