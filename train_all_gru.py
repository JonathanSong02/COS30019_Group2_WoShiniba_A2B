
import csv
from pathlib import Path

from models.gru_model import TrainingSettings, train_gru

CSV_PATH = Path("data/processed/scats_15min.csv")
MODEL_DIR = Path("saved_models")

HORIZON = 15
LOOKBACK = 8
EPOCHS = 30

with CSV_PATH.open("r", encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)
    sites = sorted({
        row["scats_site"].strip()
        for row in reader
        if row["scats_site"].strip()
    }, key=int)

print(f"Found {len(sites)} SCATS sites")

successful = 0
failed = []

for index, site in enumerate(sites, start=1):
    checkpoint = MODEL_DIR / (
        f"gru_site_{site}_h{HORIZON}_s{LOOKBACK}.pt"
    )
    metadata = checkpoint.with_suffix(".json")

    if checkpoint.exists() and metadata.exists():
        print(f"[{index}/{len(sites)}] Site {site}: already trained")
        successful += 1
        continue

    print(f"[{index}/{len(sites)}] Training site {site}")

    try:
        result = train_gru(
            CSV_PATH,
            MODEL_DIR,
            TrainingSettings(
                site=site,
                horizon_minutes=HORIZON,
                lookback=LOOKBACK,
                epochs=EPOCHS,
            ),
        )

        successful += 1
        print("Test RMSE:", result["test"]["rmse"])

    except Exception as exc:
        failed.append((site, str(exc)))
        print(f"Failed site {site}: {exc}")

print(f"\nAvailable models: {successful}/{len(sites)}")
for site, error in failed:
    print(f"Site {site}: {error}")
