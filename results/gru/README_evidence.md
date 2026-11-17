# GRU evaluation evidence — COS30019 Assignment 2B

Source: saved GRU metadata and held-out test prediction CSVs from this project; no models were retrained.

- GRU site coverage at **15-minute horizon / 8-step lookback:** 38 model checkpoints.
- Site 2000 has horizon data for: 15, 30, 60 minutes.
- Error units: RMSE and MAE are **vehicles per 15-minute target interval**.
- NRMSE (%) = RMSE / (max(actual test) - min(actual test)) × 100, computed per model/site; **not classification accuracy**.
- Models use shared chronological training/validation/test splits, training-only scaler, saved seed and settings; compare matched targets when reviewing cross-model results.
- Models for different sites have differing held-out sample counts; do not claim pooled network accuracy from a simple unweighted average of site errors.
- Horizon results describe SCATS 2000 only; they do not establish whether GRU outperforms LSTM or Regression Tree.
- These files do not verify route quality, map connectivity, or physical validity of traffic-flow-to-speed conversion.

Evidence files:

- `all_sites_15min_gru_metrics.csv`: held-out results per available SCATS site.
- `site_2000_horizon_comparison.csv`: test metrics and elapsed training time across available horizons.
- `site_2000_horizon_rmse.png`: effect of forecast horizon on test RMSE.
- `site_2000_actual_vs_predicted.png`: first held-out test day actual/predicted series for site 2000.
- `site_2000_training_validation_loss.png`: training and validation scaled MSE by epoch for site 2000.
