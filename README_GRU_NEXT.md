# COS30019 Assignment 2B — GRU update (Jonathan)

## What was changed

- Missing SCATS volume cells (`''`, `NaN`, `NA`, `null`) are **skipped**, never treated as zero. Invalid nonmissing values still raise an error.
- The real `data/processed/scats_15min.csv` now uses exactly the shared `data_processing.dataset.make_windows()` logic, fixed calendar-day split and training-only min–max scaler from `data/processed/scaler.json`. Its `split_config.json` is checked against `SPLIT_DAYS`.
- Training supports the agreed 15-, 30-, and 60-minute horizons and configurable sequence lengths (4 or 8 are the planned research settings).
- Checkpoints are separated by site, horizon and lookback: `gru_site_{site}_h{minutes}_s{lookback}.pt`, preventing overwrites across experiments.
- Saves per-epoch training/validation losses, train/test metrics (MAE, RMSE, actual-range NRMSE), elapsed time, and held-out test predictions. The test set is not used to choose the best epoch.
- Prediction validates time alignment, loads the matching trained checkpoint and requires actual, contiguous historical observations.
- Existing unit-test API (`TrainingSettings`, `train_gru`, `predict_gru`) remains compatible; synthetic CSVs without a shared scaler use a clearly marked standalone test-only fallback.

## Install without overwriting Darren's or Anson's work

1. Make a backup of your current `models/gru_model.py`.
2. Copy this ZIP's `models/gru_model.py` into your current A2B project, replacing **only** that file.
3. Copy `tests/test_gru_real_data.py` into your project's `tests/` folder.
4. Leave Darren's `data_processing/`, Anson's `models/lstm_model.py`, and other teammates' files untouched.
5. From the project root run:

```powershell
python -m unittest discover -s tests -v
```

## Train on REAL SCATS traffic (2006)

From your project root, with `data/processed/scats_15min.csv` and `data/processed/scaler.json` available:

```powershell
python -m models.gru_model train --csv data/processed/scats_15min.csv --site 2000 --horizon 15 --lookback 8 --epochs 30
```

This creates:

- `saved_models/gru_site_2000_h15_s8.pt`
- `saved_models/gru_site_2000_h15_s8.json`
- `saved_models/gru_site_2000_h15_s8_test_predictions.csv`

Predict using the historical inputs available at 12:00 noon on 27 October 2006:

```powershell
python -m models.gru_model predict --csv data/processed/scats_15min.csv --site 2000 --horizon 15 --lookback 8 --time 2006-10-27T12:00:00
```

Experiment with 60 minutes ahead and 4 past observations (a separate checkpoint):

```powershell
python -m models.gru_model train --csv data/processed/scats_15min.csv --site 2000 --horizon 60 --lookback 4 --epochs 30
```

## Verified local results in this update

- Full A2B latest-folder suite: **46/46 tests passed** (44 existing + 2 new).
- Real-data GRU site 2000, horizon 15 min, lookback 8, max epochs 30, seed 42:
  - Validation: MAE 42.8065; RMSE 58.6327; NRMSE 0.04324 on 384 windows.
  - Test: MAE 42.8621; RMSE 55.5030; NRMSE 0.04078 on 480 windows.
  - Example 2006-10-27 12:00:00 prediction for 12:15: 881.06 vehicles per 15 minutes.
- A separate 60-minute / 4-step configuration also trained and predicted successfully (smoke test, 2 epochs).

These are initial single-site baseline results, **not** a full model comparison or finished TBRGS. Validate the results on your machine before using them in the report. Values can vary by software and hardware. Training one model per site, sequence length and horizon does not yet provide network-wide traffic forecasting. The test CSV contains held-out results for comparisons, not live forecasts.

## Critical integration notes

- The model predicts one **15-minute volume**, not an hourly rate. Convert/aggregate as required by the traffic-to-travel-time equation and document the chosen conversion.
- The UI date/time is an **as-of** observation time. The forecast refers to `as_of + horizon_minutes`. It cannot predict arbitrary 2026 dates with no observed history.
- Align training parameters and evaluation sample sets with Anson's LSTM before claiming one model is better.
- The final report must acknowledge tools/resources used in development, as required by the assignment.
