# A2B — Jonathan's next steps (GRU baseline)

The **GRU source code exists**, but this ZIP does **not** contain a cleaned real dataset or a real-data-trained model.

## 1. Ask Darren to provide

`data/processed/scats_15min.csv` with three long-format columns:
`scats_site, datetime, traffic_volume_15min`.

See `docs/SHARED_DATA_CONTRACT.md` for timestamp semantics, leakage rules and edge cost integration notes.

## 2. Install for GRU training

```shell
python -m pip install torch
```

## 3. Run GRU-related tests with *synthetic* time series

From the extracted project folder:
```shell
python -m unittest tests.test_gru_model -v
```

## 4. Train on Darren's *real processed* data

```shell
python -m models.gru_model train --csv data/processed/scats_15min.csv --site 2000 --horizon 15 --epochs 30
```

The command writes a `.pt` model and `.json` metrics/metadata to `saved_models/`.
The sample site 2000 must actually exist in Darren's dataset.

## 5. Forecast with known history (only after training)

```shell
python -m models.gru_model predict --csv data/processed/scats_15min.csv --site 2000 --horizon 15 --time 2006-10-15T10:00:00
```

The timestamp requires contiguous historical observations. It is an example,
not a guaranteed valid time within the supplied workbook. Arbitrary 2026 dates cannot be predicted with this historical one-step interface alone.

## 6. Research extensions (shared with Anson)

Try `--horizon 15`, `--horizon 30`, `--horizon 60`; and `--lookback 4` / `--lookback 8`.
Keep the test period untouched when choosing hyperparameters. Compare predictions with Anson's evaluation code using the same timestamps and NRMSE definition.

**Not yet completed:** real `.xls` extraction, multi-site deployment, model evaluation against the actual October 2006 data, prediction-to-hourly-flow mapping, live SCATS graph, GUI/main.py integration, robust 15-test-case suite.
