"""Route comparison across models for the report (Darren, Task 5).

Runs the same trips and times under GRU, LSTM and Regression Tree (15 min
ahead, full 40-site network) through the same code as the GUI's Model
comparison tab, and saves:
  results/model_comparison/routes/route_comparison.csv   every top-5 route
  results/model_comparison/routes/best_route_summary.csv  best route per model
  results/model_comparison/routes/map_<origin>_<dest>_<time>.html  map of each
      model's best route (open in a browser for screenshots)

Needs PyTorch for GRU and LSTM, so run it where the GUI runs:
    python -m evaluation.compare_routes
    python -m evaluation.compare_routes --trip 2000 3002 --time 2006-10-30T20:00:00 --policy clamp
"""
from __future__ import annotations

import argparse
import csv
import html
import re
from pathlib import Path

from gui import comparison_view
from gui.app import run_route_comparison

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "model_comparison" / "routes"
# The brief's example trip plus two cross-network trips; an evening and an
# early-morning time in the test split. Routing uses per-approach flow, so
# Reject (the default) works at both; the experimental Clamp is optional.
DEFAULT_TRIPS = [(2000, 3002), (970, 4821), (2827, 4273)]
DEFAULT_TIMES = ["2006-10-30T20:00:00", "2006-10-31T06:00:00"]


def run(trips=DEFAULT_TRIPS, times=DEFAULT_TIMES, policy: str = "reject", output: Path = OUTPUT) -> list[list]:
    output.mkdir(parents=True, exist_ok=True)
    label = "Clamp to 1500 (experimental)" if policy == "clamp" else "Reject values above 1500 veh/hour"
    all_rows, best_rows = [], []
    for origin, dest in trips:
        for t in times:
            status, rows, map_html = run_route_comparison(str(origin), str(dest), t, label)
            for r in rows:
                all_rows.append([origin, dest, t, policy] + r)
            best = {m: next((r for r in rows if r[0] == m), None) for m in comparison_view.MODELS}
            rt_best = best["Regression Tree"][2] if best["Regression Tree"] else None
            for model, r in best.items():
                same = ("" if r is None or rt_best is None or model == "Regression Tree"
                        else "yes" if r[2] == rt_best else "no")
                best_rows.append([origin, dest, t, policy, model, r[2] if r else "", r[3] if r else "",
                                  same, "" if r else "no routes"])
            # save the map itself (not the iframe wrapper) for screenshots
            doc = re.search(r'srcdoc="(.*?)" style=', map_html, re.S)
            if doc:
                name = f"map_{origin}_{dest}_{t[:16].replace(':', '').replace('-', '')}.html"
                (output / name).write_text(html.unescape(doc.group(1)), encoding="utf-8")
            print(f"{origin} -> {dest} at {t}: {status.splitlines()[0]}")
    with (output / "route_comparison.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["origin", "destination", "as_of", "capacity_policy"] + comparison_view.ROUTE_HEADERS)
        w.writerows(all_rows)
    with (output / "best_route_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["origin", "destination", "as_of", "capacity_policy", "model",
                    "best_route", "travel_time_min", "same_best_route_as_regression_tree", "note"])
        w.writerows(best_rows)
    print(f"Saved to {output}")
    return all_rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--trip", nargs=2, type=int, action="append", metavar=("ORIGIN", "DEST"))
    ap.add_argument("--time", action="append", help="last observed time, e.g. 2006-10-30T20:00:00")
    ap.add_argument("--policy", choices=["clamp", "reject"], default="reject")
    a = ap.parse_args()
    run([tuple(t) for t in a.trip] if a.trip else DEFAULT_TRIPS, a.time or DEFAULT_TIMES, a.policy)
