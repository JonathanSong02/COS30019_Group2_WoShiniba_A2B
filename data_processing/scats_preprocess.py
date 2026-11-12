"""
scats_preprocess.py
Turn the VicRoads 'Scats Data October 2006.xls' into clean, model-ready files.

Raw format (sheet "Data", header on the 2nd row):
    one row = one detector approach (e.g. "WARRIGAL_RD N of HIGH STREET_RD")
              at one SCATS site on one day,
    V00..V95 = vehicle counts for the 96 quarter-hours of that day
              (V00 = 00:00-00:15, ..., V95 = 23:45-24:00).

Outputs (written to --out, default data/processed/):
    scats_15min.csv              scats_site, datetime, traffic_volume_15min
                                 (site total = sum of its approaches; a full
                                 40-site x 2976-step grid, blank = missing)
    scats_15min_by_approach.csv  same, kept per approach (direction-level)
    scats_sites.csv              site id, lat, lon, approach names
    split_config.json            chronological train/val/test cut-offs
    scaler.json                  min-max parameters fitted on TRAIN ONLY
    data_quality_report.md       numbers for the report's data section

Run from the project root:
    python -m data_processing.scats_preprocess --input "data/raw/Scats Data October 2006.xls"
"""

import argparse
import json
import os

import pandas as pd

from data_processing.dataset import SPLIT_DAYS, fit_minmax, split_of

INTERVALS_PER_DAY = 96
VOLUME_COLS = [f"V{i:02d}" for i in range(INTERVALS_PER_DAY)]
# An approach is "core" for a site if it reports on at least this share of the
# site's days. Approaches seen on only a few days (e.g. CHURCH_ST at 3001,
# 2 of 31 days) are ignored so they don't make the site total jump.
CORE_APPROACH_SHARE = 0.5


def load_raw(path):
    """Read the Data sheet and normalise ids, dates and approach keys."""
    df = pd.read_excel(path, sheet_name="Data", header=1).copy()
    df["scats_site"] = df["SCATS Number"].astype(int)
    # Dates are stored as e.g. 2006-10-01 00:15:00; only the day is meaningful.
    df["day"] = pd.to_datetime(df["Date"]).dt.normalize()
    # Site 4335 has two different detectors that share one Location name
    # (different coordinates and counts), so key approaches by Location AND
    # VicRoads' internal detector location number, not by name alone.
    df["approach"] = df["Location"].str.strip() + " #" + df["VR Internal Loc"].astype(str)
    return df


def to_long(df):
    """Wide V00..V95 rows -> one row per (site, approach, 15-min timestamp)."""
    long = df.melt(id_vars=["scats_site", "approach", "day"], value_vars=VOLUME_COLS,
                   var_name="slot", value_name="volume")
    slot = long["slot"].str[1:].astype(int)
    long["datetime"] = long["day"] + pd.to_timedelta(slot * 15, unit="min")
    return long.drop(columns="slot")


def core_approaches(df):
    """{site: set of approaches present on >= CORE_APPROACH_SHARE of its days}."""
    days_per_site = df.groupby("scats_site")["day"].nunique()
    days_per_appr = df.groupby(["scats_site", "approach"])["day"].nunique().reset_index()
    days_per_appr["share"] = days_per_appr["day"] / days_per_appr["scats_site"].map(days_per_site)
    core = days_per_appr[days_per_appr["share"] >= CORE_APPROACH_SHARE]
    return core.groupby("scats_site")["approach"].apply(set).to_dict()


def site_totals(long, core, full_index):
    """Sum core approaches per site/timestamp; incomplete site-days become NaN.

    Summing only the approaches that happened to report would make a
    missing detector look like a sudden traffic drop, so a site-day counts
    only when every core approach of that site reported.
    """
    is_core = [a in core[s] for s, a in zip(long["scats_site"], long["approach"])]
    agg = long[is_core].groupby(["scats_site", "datetime"]).agg(
        traffic_volume_15min=("volume", "sum"), n_reporting=("approach", "nunique"))
    expected = agg.index.get_level_values("scats_site").map({s: len(a) for s, a in core.items()})
    agg.loc[agg["n_reporting"].values != expected.values, "traffic_volume_15min"] = float("nan")
    out = agg[["traffic_volume_15min"]].reindex(full_index)  # whole missing days -> NaN
    return out.reset_index()


def site_table(df, core):
    """One row per site: coordinates (zeros dropped) and core approach names."""
    rows = []
    for site, g in df.groupby("scats_site"):
        ok = g[(g["NB_LATITUDE"] != 0) & (g["NB_LONGITUDE"] != 0)]
        rows.append({
            "scats_site": site,
            "latitude": round(ok["NB_LATITUDE"].median(), 6),
            "longitude": round(ok["NB_LONGITUDE"].median(), 6),
            "n_core_approaches": len(core[site]),
            "approaches": "; ".join(sorted(a.rsplit(" #", 1)[0] for a in core[site])),
        })
    return pd.DataFrame(rows)


def quality_report(df, totals, core, sites):
    """Markdown summary of the cleaning decisions and their effect."""
    days = df.groupby(["scats_site", "approach", "day"]).size()
    missing = totals.groupby("scats_site")["traffic_volume_15min"].apply(lambda s: int(s.isna().sum()))
    totals = totals.assign(split=split_of(totals["datetime"]))
    lines = [
        "# SCATS October 2006 - data quality report", "",
        f"- Raw rows (approach-days): {len(df)}",
        f"- SCATS sites: {df['scats_site'].nunique()}; distinct approaches: {df['approach'].nunique()}",
        f"- Date range: {df['day'].min().date()} to {df['day'].max().date()} "
        f"({df['day'].nunique()} days, {INTERVALS_PER_DAY} x 15-min intervals per day)",
        f"- Duplicate (site, approach, day) rows: {int((days > 1).sum())}",
        f"- Rows with zero coordinates: {int((df['NB_LATITUDE'] == 0).sum())} (ignored for site coordinates)",
        f"- Negative or missing raw counts: {int(df[VOLUME_COLS].isna().sum().sum() + (df[VOLUME_COLS] < 0).sum().sum())}",
        f"- Max 15-min count at one approach: {int(df[VOLUME_COLS].max().max())}",
        f"- Max 15-min site total: {int(totals['traffic_volume_15min'].max())}",
        f"- Site-level grid: {len(totals)} rows, of which {int(totals['traffic_volume_15min'].isna().sum())} missing "
        f"({totals['traffic_volume_15min'].isna().mean():.1%})", "",
        "## Missing 15-min steps per split", "",
        "| split | rows | missing |", "|---|---|---|",
    ]
    for name, g in totals.groupby("split", sort=False):
        lines.append(f"| {name} | {len(g)} | {int(g['traffic_volume_15min'].isna().sum())} |")
    lines += ["", "## Sites with missing data", "", "| site | missing 15-min steps | missing days |", "|---|---|---|"]
    for site, n in missing[missing > 0].items():
        lines.append(f"| {site} | {n} | {n / INTERVALS_PER_DAY:.0f} |")
    lines += ["", "## Ignored non-core approaches", ""]
    all_appr = df.groupby("scats_site")["approach"].apply(set)
    for site, appr in all_appr.items():
        extra = appr - core[site]
        if extra:
            lines.append(f"- {site}: {', '.join(sorted(extra))}")
    lines += ["", f"Sites table: {len(sites)} sites with coordinates."]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--input", default=os.path.join("data", "raw", "Scats Data October 2006.xls"))
    parser.add_argument("--out", default=os.path.join("data", "processed"))
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    df = load_raw(args.input)
    long = to_long(df)
    core = core_approaches(df)

    all_times = pd.date_range(df["day"].min(), df["day"].max() + pd.Timedelta(minutes=15 * 95), freq="15min")
    full_index = pd.MultiIndex.from_product([sorted(df["scats_site"].unique()), all_times],
                                            names=["scats_site", "datetime"])
    totals = site_totals(long, core, full_index)
    sites = site_table(df, core)

    # Normalisation parameters come from the training period only.
    train_values = totals.loc[split_of(totals["datetime"]) == "train", "traffic_volume_15min"]
    scaler = fit_minmax(train_values)

    fmt = "%Y-%m-%d %H:%M"
    totals.to_csv(os.path.join(args.out, "scats_15min.csv"), index=False, date_format=fmt)
    (long.sort_values(["scats_site", "approach", "datetime"])
         [["scats_site", "approach", "datetime", "volume"]]
         .rename(columns={"volume": "traffic_volume_15min"})
         .to_csv(os.path.join(args.out, "scats_15min_by_approach.csv"), index=False, date_format=fmt))
    sites.to_csv(os.path.join(args.out, "scats_sites.csv"), index=False)
    with open(os.path.join(args.out, "split_config.json"), "w") as f:
        json.dump({k: [str(a.date()), str(b.date())] for k, (a, b) in SPLIT_DAYS.items()}, f, indent=2)
    with open(os.path.join(args.out, "scaler.json"), "w") as f:
        json.dump(scaler, f, indent=2)
    with open(os.path.join(args.out, "data_quality_report.md"), "w") as f:
        f.write(quality_report(df, totals, core, sites))
    print(f"Wrote {len(totals)} site rows for {len(sites)} sites to {args.out}")


if __name__ == "__main__":
    main()
