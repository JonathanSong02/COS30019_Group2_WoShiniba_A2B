"""Train site-specific GRU checkpoints for every processed SCATS site.

Usage (from the project root):
    python -m models.train_all_gru                   # 15-minute horizon (default)
    python -m models.train_all_gru --horizons 30 60    # longer horizons

Existing checkpoints are skipped, so rerunning the script never overwrites
an evaluated model.
"""

import argparse
import csv
from pathlib import Path

from models.gru_model import TrainingSettings, train_gru

CSV_PATH = Path("data/processed/scats_15min.csv")
MODEL_DIR = Path("saved_models")
LOOKBACK = 8
EPOCHS = 30


def processed_sites():
    """Return every SCATS site present in the processed 15-minute grid."""
    with CSV_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return sorted({
            row["scats_site"].strip()
            for row in reader
            if row["scats_site"].strip()
        }, key=int)


def train_horizon(sites, horizon):
    """Train one GRU per site for a single forecast horizon."""
    successful = 0
    failed = []
    for index, site in enumerate(sites, start=1):
        checkpoint = MODEL_DIR / f"gru_site_{site}_h{horizon}_s{LOOKBACK}.pt"
        if checkpoint.exists() and checkpoint.with_suffix(".json").exists():
            print(f"[h{horizon} {index}/{len(sites)}] Site {site}: already trained")
            successful += 1
            continue

        print(f"[h{horizon} {index}/{len(sites)}] Training site {site}")
        try:
            result = train_gru(
                CSV_PATH,
                MODEL_DIR,
                TrainingSettings(site=site, horizon_minutes=horizon,
                                 lookback=LOOKBACK, epochs=EPOCHS),
            )
            successful += 1
            print("Test RMSE:", result["test"]["rmse"])
        except Exception as exc:  # report and continue with the next site
            failed.append((site, str(exc)))
            print(f"Failed site {site}: {exc}")

    print(f"\nHorizon {horizon}: available models {successful}/{len(sites)}")
    for site, error in failed:
        print(f"Site {site}: {error}")


def main():
    parser = argparse.ArgumentParser(description="Train GRU models for all SCATS sites")
    parser.add_argument("--horizons", nargs="+", type=int, choices=[15, 30, 60], default=[15])
    args = parser.parse_args()

    sites = processed_sites()
    print(f"Found {len(sites)} SCATS sites")
    for horizon in args.horizons:
        train_horizon(sites, horizon)


if __name__ == "__main__":
    main()
