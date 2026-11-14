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

- No map display yet: route table is only activated when the verified road network exists.
- Only site 2000 GRU models were previously confirmed; LSTM and Regression Tree may be unavailable.
- GRU output is one future 15-minute interval, not an hourly vehicle count.
- The provisional `steady_4x` hourly approximation and origin/destination flow convention must be confirmed/documented for final evaluation.
- `Gradio` is not added to the existing UTF-16 encoded `requirements.txt` in this patch. For packaging, ensure your final dependency file is UTF-8 and contains a compatible Gradio version.
