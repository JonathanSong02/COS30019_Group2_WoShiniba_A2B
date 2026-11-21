"""Documented A2B test cases: run end-to-end scenarios and record the results.

Same idea as run_15_testcases.py (A2A): each case states the scenario, the
input, the expected result and how it was checked; the script runs it on the
real system (processed data, saved models, road graph, routing, GUI callbacks)
and records the actual result as PASS / FAIL / ERROR.

Run from the project root (GRU and LSTM cases need PyTorch):
    python run_a2b_testcases.py
Results: A2B_Test_Results.csv (one row per case) and a summary printed below.

Expected values come from three places, named in each case's "remarks":
  * the assignment documents (e.g. the flow-to-speed PDF: 60 km/h, 32 km/h)
  * facts about the data or graph that we checked by hand (e.g. 40 sites)
  * the models' own saved evaluation files: the system's prediction must equal
    the value the model produced when it was evaluated (consistency check)
"""
from __future__ import annotations

import csv
import json
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "A2B_Test_Results.csv"
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "saved_models"
FORECAST_TIME = "2006-10-30T08:00:00"      # Mon 30 Oct, test split, complete data at every site
ROUTE_TIME = "2006-10-30T20:00:00"         # evening, test split: every per-approach flow is under 1500 veh/h
PEAK_TIME = "2006-10-30T08:00:00"          # morning peak: single-approach site 2825 is over 1500 veh/h


@dataclass
class Case:
    test_id: str
    area: str
    owner: str          # who built the part being tested
    scenario: str
    input: str
    expected: str
    check: Callable[[], tuple[str, bool]]   # returns (actual result as text, passed?)
    remarks: str = ""


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _backend():
    import main
    return main


def _saved_prediction(path: Path, target: str) -> float:
    """A model's own saved test prediction for one target time."""
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            t = row.get("target_datetime") or row.get("datetime")
            if t.startswith(target):
                return float(row.get("predicted_vehicles_15min") or row["predicted_traffic_volume"])
    raise KeyError(f"{target} not in {path.name}")


def _rejected(fn) -> tuple[str, bool]:
    """Expect fn() to raise a clear error (and so return no made-up value)."""
    try:
        value = fn()
    except (ValueError, FileNotFoundError, KeyError) as exc:
        return f"rejected: {type(exc).__name__}: {str(exc).splitlines()[0][:90]}", True
    return f"accepted, returned {value!r}", False


def _predict(site, when, model, horizon=15):
    return _backend().predict_traffic(site, when, model_name=model, horizon_minutes=horizon)


def _prediction_matches_evaluation(model: str, saved: Path, horizon: int = 15) -> tuple[str, bool]:
    from datetime import datetime, timedelta
    target = (datetime.fromisoformat(FORECAST_TIME) + timedelta(minutes=horizon)).strftime("%Y-%m-%dT%H:%M")
    expected = _saved_prediction(saved, target)
    got = _predict(2000, FORECAST_TIME, model, horizon)
    return f"{got:.2f} (saved evaluation value {expected:.2f})", abs(got - expected) <= 0.5


def _routes(origin, dest, when, model="regression_tree", policy="reject", conversion="per_approach"):
    b = _backend()
    links = b.load_verified_road_links(b.DEFAULT_ROADS)
    coords = b.load_site_coordinates(b.DEFAULT_SITES)
    return b.compute_traffic_routes(
        origin, dest, when, links, coords,
        lambda s, t: b.predict_traffic(s, t, model_name=model), capacity_policy=policy,
        flow_conversion=conversion)


def _path(route) -> str:
    return "-".join(map(str, route["scats_sites"]))


# --------------------------------------------------------------------------
# A. data processing
# --------------------------------------------------------------------------
def tc_grid():
    rows = list(csv.DictReader((PROCESSED / "scats_15min.csv").open(newline="", encoding="utf-8")))
    sites = {r["scats_site"] for r in rows}
    dupes = len(rows) - len({(r["scats_site"], r["datetime"]) for r in rows})
    return f"{len(rows)} rows, {len(sites)} sites, {dupes} duplicates", len(rows) == 119040 and len(sites) == 40 and dupes == 0


def tc_missing_not_zero():
    rows = [r for r in csv.DictReader((PROCESSED / "scats_15min.csv").open(newline="", encoding="utf-8"))
            if r["scats_site"] == "3002" and r["datetime"].startswith("2006-10-24")]
    blank = sum(r["traffic_volume_15min"] in ("", "nan", "NaN") for r in rows)
    zero = sum(r["traffic_volume_15min"] in ("0", "0.0") for r in rows)
    return f"{len(rows)} intervals: {blank} blank, {zero} zero", len(rows) == 96 and blank == 96


def tc_fixed_sites():
    split = json.loads((PROCESSED / "split_config.json").read_text(encoding="utf-8"))
    counts = {}
    for r in csv.DictReader((PROCESSED / "scats_15min.csv").open(newline="", encoding="utf-8")):
        if r["scats_site"] in ("3685", "4264") and r["traffic_volume_15min"] not in ("", "nan", "NaN"):
            day = r["datetime"][:10]
            part = "val" if split["val"][0] <= day <= split["val"][1] else "test" if day >= split["test"][0] else "train"
            counts[(r["scats_site"], part)] = counts.get((r["scats_site"], part), 0) + 1
    text = ", ".join(f"{s} {p}: {counts.get((s, p), 0)}" for s in ("3685", "4264") for p in ("val", "test"))
    return text, all(counts.get((s, p), 0) > 0 for s in ("3685", "4264") for p in ("val", "test"))


def tc_split():
    s = json.loads((PROCESSED / "split_config.json").read_text(encoding="utf-8"))
    ok = s["train"][1] < s["val"][0] <= s["val"][1] < s["test"][0] <= s["test"][1]
    return f"train {s['train'][0]}..{s['train'][1]}, val {s['val'][0]}..{s['val'][1]}, test {s['test'][0]}..{s['test'][1]}", ok


# --------------------------------------------------------------------------
# C. travel time (flow-to-speed PDF)
# --------------------------------------------------------------------------
def tc_free_flow():
    from routing.travel_time import edge_travel_time_seconds, speed_from_hourly_flow
    v, t = speed_from_hourly_flow(300), edge_travel_time_seconds(1.0, 300)
    return f"{v:.2f} km/h, {t:.1f} s", abs(v - 60) < 1e-6 and abs(t - 90) < 1e-6


def tc_capacity():
    from routing.travel_time import edge_travel_time_seconds, speed_from_hourly_flow
    v, t = speed_from_hourly_flow(1500), edge_travel_time_seconds(1.0, 1500)
    return f"{v:.2f} km/h, {t:.1f} s", abs(v - 32) < 0.1 and abs(t - 142.3) < 0.5


def tc_mid_flow():
    from routing.travel_time import speed_from_hourly_flow
    v = speed_from_hourly_flow(1000)
    return f"{v:.2f} km/h", 32 < v < 60 and abs(v - 50.48) < 0.05


def tc_over_capacity():
    strict = _routes(2000, 3002, PEAK_TIME, policy="reject")
    experiment = _routes(2000, 3002, PEAK_TIME, policy="clamp")
    excluded = strict["assumptions"]["capacity_excluded_sites"]
    nlinks = strict["assumptions"]["capacity_excluded_link_count"]
    clamped = experiment["assumptions"]["capacity_clamped_sites"]
    ok = (len(strict["routes"]) == 5 and 2825 in excluded and nlinks > 0
          and not strict["assumptions"]["capacity_clamped_sites"]
          and strict["assumptions"]["network_incomplete"]
          and len(experiment["routes"]) == 5 and clamped == [2825]
          and experiment["assumptions"]["capacity_excluded_sites"] == {})
    return (f"Reject -> {len(strict['routes'])} feasible routes; excluded site 2825 "
            f"and {nlinks} affected links; experimental Clamp -> "
            f"{len(experiment['routes'])} routes, capped sites {clamped}", ok)


def tc_unrelated_capacity():
    from main import compute_traffic_routes
    coords = {1: (0, 0), 2: (0, 1), 3: (1, 1), 4: (1, 2)}
    roads = [(1, 2, 1.0), (2, 4, 1.0), (1, 3, 2.0), (3, 4, 2.0)]
    result = compute_traffic_routes(1, 4, ROUTE_TIME, roads, coords,
            predictor=lambda s, t: 600.0 if s == 3 else 50.0,
            approach_counts={s: 1 for s in coords})
    a = result["assumptions"]
    ok = (result["routes"][0]["scats_sites"] == [1, 2, 4]
          and set(a["capacity_excluded_sites"]) == {3}
          and a["capacity_excluded_link_count"] == 1
          and a["network_incomplete"])
    return (f"valid path {result['routes'][0]['scats_sites']}; capacity excluded "
            f"{list(a['capacity_excluded_sites'])}, {a['capacity_excluded_link_count']} edge", ok)


def tc_missing_detector_numeric():
    import pandas as pd
    from data_processing.scats_preprocess import site_totals
    instant = pd.Timestamp("2006-10-01T00:00:00")
    detector_rows = pd.DataFrame({
        "scats_site": [99, 99], "approach": ["A", "B"],
        "datetime": [instant, instant], "volume": [8.0, float("nan")],
    })
    grid = pd.MultiIndex.from_tuples([(99, instant)], names=["scats_site", "datetime"])
    result = site_totals(detector_rows, {99: {"A", "B"}}, grid)
    value = result.loc[0, "traffic_volume_15min"]
    return ("partly missing detector value produces blank site-total" if pd.isna(value)
            else f"incorrect partial sum {value}", bool(pd.isna(value)))


def tc_per_approach_flow():
    from main import load_site_approach_counts, predict_traffic
    r = _routes(2000, 3002, ROUTE_TIME)
    rec = {x["site"]: x for x in r["assumptions"]["site_flows"]}[2000]
    raw = predict_traffic(2000, ROUTE_TIME, model_name="regression_tree")
    n = load_site_approach_counts()[2000]
    ok = (r["assumptions"]["flow_conversion"] == "per_approach" and rec["approaches"] == n == 4
          and abs(rec["predicted_vehicles_per_15min"] - raw) < 1e-3
          and abs(rec["site_total_veh_per_hour"] - 4 * raw) < 0.01
          and abs(rec["flow_used_veh_per_hour"] - 4 * raw / n) < 0.01)
    return (f"predicted {rec['predicted_vehicles_per_15min']:.2f} / 15 min, site total "
            f"{rec['site_total_veh_per_hour']:.1f} veh/h, {rec['approaches']} approaches, "
            f"flow used {rec['flow_used_veh_per_hour']:.1f} veh/h"), ok


def tc_flow_conversion_on_observed_data():
    from main import load_site_approach_counts
    n = load_site_approach_counts()
    total = over_site = over_approach = 0
    for row in csv.DictReader((PROCESSED / "scats_15min.csv").open(newline="", encoding="utf-8")):
        v = row["traffic_volume_15min"]
        if v in ("", "nan", "NaN"):
            continue
        hourly = float(v) * 4
        total += 1
        over_site += hourly > 1500
        over_approach += hourly / n[int(row["scats_site"])] > 1500
    a, b = 100 * over_site / total, 100 * over_approach / total
    return (f"{total} intervals: site total over 1500 in {a:.1f}%, per-approach in {b:.2f}% "
            f"({over_approach} intervals)"), a > 40 and b < 0.1


# --------------------------------------------------------------------------
# D. road graph
# --------------------------------------------------------------------------
def tc_graph():
    from data_processing.build_road_graph import build_edges, load_sites, validate
    sites = load_sites(PROCESSED / "scats_sites.csv")
    rows = build_edges(sites)
    validate(sites, rows)    # raises if disconnected, one-way or implausible
    return f"{len(sites)} sites, {len(rows) // 2} two-way roads, connected", len(sites) == 40 and len(rows) == 136


def tc_reviewed_link():
    b = _backend()
    d = {(u, v): km for u, v, km in b.load_verified_road_links(b.DEFAULT_ROADS)}
    return f"2825->2827 {d[(2825, 2827)]} km, 2827->2825 {d[(2827, 2825)]} km", d[(2825, 2827)] == d[(2827, 2825)] == 2.258


# --------------------------------------------------------------------------
# E. routing
# --------------------------------------------------------------------------
def tc_brief_example():
    b = _backend()
    edges = {(u, v) for u, v, _ in b.load_verified_road_links(b.DEFAULT_ROADS)}
    routes = _routes(2000, 3002, ROUTE_TIME)["routes"]
    times = [r["travel_time_minutes"] for r in routes]
    ok = (len(routes) == 5 and times == sorted(times)
          and all(r["scats_sites"][0] == 2000 and r["scats_sites"][-1] == 3002 for r in routes)
          and all(len(set(r["scats_sites"])) == len(r["scats_sites"]) for r in routes)
          and all((u, v) in edges for r in routes for u, v in zip(r["scats_sites"], r["scats_sites"][1:])))
    return f"{len(routes)} routes, {times[0]:.2f}-{times[-1]:.2f} min, best {_path(routes[0])}", ok


def tc_traffic_changes_route():
    night = _routes(2000, 3002, "2006-10-31T03:00:00")["routes"][0]
    evening = _routes(2000, 3002, ROUTE_TIME)["routes"][0]
    return (f"03:00 best {_path(night)} ({night['travel_time_minutes']:.1f} min); "
            f"20:00 best {_path(evening)} ({evening['travel_time_minutes']:.1f} min)",
            _path(night) != _path(evening) and night["travel_time_minutes"] < evening["travel_time_minutes"])


def tc_origin_is_destination():
    routes = _routes(2000, 2000, ROUTE_TIME)["routes"]
    return f"{routes}", len(routes) == 1 and routes[0]["scats_sites"] == [2000] and routes[0]["travel_time_seconds"] == 0


def tc_three_models_route():
    results = {m: _routes(2000, 3002, ROUTE_TIME, model=m) for m in ("gru", "lstm", "regression_tree")}
    text = "; ".join(f"{m}: {len(r['routes'])} routes, best {r['routes'][0]['travel_time_minutes']:.2f} min "
                     f"via {_path(r['routes'][0])}" for m, r in results.items() if r["routes"])
    ok = all(len(r["routes"]) == 5 and not r["assumptions"]["network_incomplete"]
             and not r["assumptions"]["capacity_clamped_sites"] for r in results.values())
    return text, ok


# --------------------------------------------------------------------------
# F. GUI
# --------------------------------------------------------------------------
def tc_gui_bad_date():
    from gui.app import run_prediction
    status, value, details = run_prediction("2000", "30/10/2006 8am", "Regression Tree", 15, 8)
    return f"status: {status[:70]}; value: {value}", "unavailable" in status and value is None and details == {}


def tc_gui_routes_and_map():
    from gui.app import run_routes
    status, rows, details, map_html = run_routes(
        "2000", "3002", ROUTE_TIME, "Regression Tree", 15, 8, 5, str(_backend().DEFAULT_ROADS), "origin",
        "Reject values above 1500 veh/hour")
    return (f"{len(rows)} table rows, map {'present' if '<iframe' in map_html else 'missing'}, "
            f"best {rows[0][1] if rows else '-'}"), len(rows) == 5 and "<iframe" in map_html


# --------------------------------------------------------------------------
CASES = [
    # A. data processing
    Case("TC01", "Data processing", "Darren", "Processed data is a complete 15-minute grid",
         "data/processed/scats_15min.csv", "119040 rows (40 sites x 2976 steps), 40 sites, 0 duplicates", tc_grid,
         "31 days x 96 intervals = 2976 per site"),
    Case("TC02", "Data processing", "Darren", "A missing day stays missing (not filled with zeros)",
         "Site 3002, 24 Oct 2006", "96 blank intervals, 0 zeros", tc_missing_not_zero,
         "Detectors at 3002 did not report on 24 Oct"),
    Case("TC03", "Data processing", "Darren", "Sites 3685 and 4264 have validation and test data after the detector fix",
         "Sites 3685, 4264", "more than 0 readings in val (23-26 Oct) and test (27-31 Oct) for both", tc_fixed_sites,
         "Before the fix both had none, so their models could not be evaluated"),
    Case("TC04", "Data processing", "Darren", "Chronological split with no overlap",
         "data/processed/split_config.json", "train before val before test, no shared days", tc_split),
    # B. predictions
    Case("TC05", "Prediction", "Darren", "Regression Tree prediction through main.py equals its evaluated value",
         f"Site 2000, last observed {FORECAST_TIME}, +15 min", "1374.40 vehicles / 15 min (saved test prediction)",
         lambda: _prediction_matches_evaluation("regression_tree", MODELS / "rt_site_2000_h15_s8_test_predictions.csv"),
         "Expected value from the model's own saved test predictions; tolerance 0.5"),
    Case("TC06", "Prediction", "Jonathan", "GRU prediction through main.py equals its evaluated value",
         f"Site 2000, last observed {FORECAST_TIME}, +15 min", "1329.93 vehicles / 15 min (saved test prediction)",
         lambda: _prediction_matches_evaluation("gru", MODELS / "gru_site_2000_h15_s8_test_predictions.csv"),
         "Needs PyTorch; tolerance 0.5"),
    Case("TC07", "Prediction", "Anson", "LSTM prediction through main.py equals its evaluated value",
         f"Site 2000, last observed {FORECAST_TIME}, +15 min", "1364.02 vehicles / 15 min (saved test prediction)",
         lambda: _prediction_matches_evaluation(
             "lstm", ROOT / "results" / "lstm_site2000_comparison" / "site_2000_h15_s8" / "predictions.csv"),
         "Needs PyTorch; tolerance 0.5"),
    Case("TC08", "Prediction", "Darren", "60-minute horizon works and matches its evaluated value",
         f"Regression Tree, site 2000, last observed {FORECAST_TIME}, +60 min", "same as saved test prediction for 09:00",
         lambda: _prediction_matches_evaluation("regression_tree", MODELS / "rt_site_2000_h60_s8_test_predictions.csv", 60)),
    Case("TC09", "Input validation", "Team", "Unknown SCATS site is rejected",
         "Site 9999", "error, no prediction", lambda: _rejected(lambda: _predict(9999, FORECAST_TIME, "regression_tree"))),
    Case("TC10", "Input validation", "Team", "Time not on a 15-minute boundary is rejected",
         "Site 2000, 2006-10-30T08:07", "error, no prediction",
         lambda: _rejected(lambda: _predict(2000, "2006-10-30T08:07:00", "regression_tree"))),
    Case("TC11", "Input validation", "Team", "Missing history is reported, not filled with a made-up value",
         "Site 3002, 2006-10-24T10:00 (day with no data)", "error, no prediction",
         lambda: _rejected(lambda: _predict(3002, "2006-10-24T10:00:00", "regression_tree"))),
    # C. travel time
    Case("TC12", "Travel time", "Jonathan", "Low flow drives at the 60 km/h speed limit",
         "300 veh/h, 1 km edge", "60 km/h; 60 s driving + 30 s intersection delay = 90 s", tc_free_flow,
         "Flow-to-speed PDF: at or below 351 veh/h the speed is capped at 60 km/h"),
    Case("TC13", "Travel time", "Jonathan", "At capacity the speed is about 32 km/h",
         "1500 veh/h, 1 km edge", "about 32 km/h; about 142.3 s", tc_capacity, "Flow-to-speed PDF turning point (1500, 32)"),
    Case("TC14", "Travel time", "Jonathan", "Medium flow uses the under-capacity (green) branch",
         "1000 veh/h", "about 50.48 km/h (between 32 and 60)", tc_mid_flow,
         "v = (-B - sqrt(B^2 + 4Aq)) / 2A with A = -1.4648, B = 93.75"),
    Case("TC15", "Travel time", "Team", "Over-capacity forecast: Reject excludes affected edges; Clamp is experimental",
         f"2000 -> 3002, Regression Tree, {PEAK_TIME} (morning peak)",
         "Reject: 5 feasible routes with 2825 edges excluded; Clamp: 5 routes with only 2825 capped", tc_over_capacity,
         "2825 (Burke Rd S of Eastern Fwy) has 1 approach detector, so its per-approach flow is the "
         "site total; it exceeds 1500 veh/h at 08:00-08:30. Reject is the default, as the brief assumes "
         "roads are under capacity"),
    Case("TC16", "Travel time", "Darren", "Per-approach flow is the site total divided by the approach count",
         f"2000 -> 3002, Regression Tree, {ROUTE_TIME}; site 2000 (4 approaches)",
         "prediction unchanged; site total = 4 x prediction; flow used = site total / 4",
         tc_per_approach_flow, "Task 6; values read from the route result's assumptions.site_flows"),
    Case("TC17", "Travel time", "Darren", "Per-approach conversion brings observed flows under the 1500 veh/h capacity",
         "All non-missing intervals in data/processed/scats_15min.csv",
         "site totals over 1500 in more than 40% of intervals; per-approach in less than 0.1%",
         tc_flow_conversion_on_observed_data,
         "Evidence for the per-approach choice: 43.0% vs 0.08% (88 intervals at sites 2825, 4262, 3001)"),
    # D. road graph
    Case("TC18", "Road graph", "Darren", "Road graph is complete, two-way and connected",
         "data/processed/scats_road_edges.csv", "40 sites, 68 two-way roads, connected", tc_graph),
    Case("TC19", "Road graph", "Darren", "Link fixed by the OpenStreetMap review uses its measured distance",
         "2825 <-> 2827", "2.258 km both ways (not the 1.451 km straight line)", tc_reviewed_link,
         "See data/processed/scats_road_edges_review.csv"),
    # E. routing
    Case("TC20", "Routing", "Team", "Brief's example trip returns 5 valid routes, fastest first",
         f"2000 -> 3002, Regression Tree, {ROUTE_TIME}, Reject",
         "5 routes; ascending times; start 2000, end 3002; no repeated site; every hop is a road", tc_brief_example),
    Case("TC21", "Routing", "Team", "Predicted traffic changes the best route",
         "2000 -> 3002, Regression Tree, 03:00 vs 20:00",
         "different best routes; the 03:00 trip is faster", tc_traffic_changes_route,
         "03:00: via 3812 (12.1 min); 20:00: via 4043 (12.3 min), per-approach flow under Reject"),
    Case("TC22", "Routing", "Team", "Origin equal to destination",
         "2000 -> 2000", "one route [2000], 0 s", tc_origin_is_destination),
    Case("TC23", "Routing", "Team", "Unknown origin is rejected",
         "9999 -> 3002", "error, no routes", lambda: _rejected(lambda: _routes(9999, 3002, ROUTE_TIME))),
    Case("TC24", "Routing", "Team", "All three models give full-network routes for the same trip",
         f"2000 -> 3002, {ROUTE_TIME}, Reject",
         "5 routes for each model on the full 40-site network, no site excluded or capped",
         tc_three_models_route, "Needs PyTorch; best times are recorded in the actual result"),
    # F. GUI
    Case("TC25", "GUI", "Jonathan", "Prediction tab explains a badly formatted date",
         "'30/10/2006 8am'", "message containing 'unavailable', no number", tc_gui_bad_date),
    Case("TC26", "GUI", "Darren", "Route guidance tab returns the route table and the map",
         f"2000 -> 3002, Regression Tree, {ROUTE_TIME}, Reject", "5 table rows and a map", tc_gui_routes_and_map),
    Case("TC27", "Routing", "Jonathan", "Remote over-capacity site does not cancel a feasible selected journey",
         "Synthetic four-site graph; site 3 over capacity; trip 1 -> 4",
         "1 -> 2 -> 4 remains valid; only edge from site 3 excluded with warning",
         tc_unrelated_capacity, "Synthetic regression test for selective strict capacity handling"),
    Case("TC28", "Data processing", "Jonathan", "Missing numeric detector value is not summed as a complete site total",
         "Synthetic site 99, approaches A=8 and B=missing at the same interval",
         "Site total is missing, never a partial count of 8",
         tc_missing_detector_numeric, "Reproduces and checks the detector NaN robustness fix"),
]


def run(cases=CASES, out: Path = RESULTS) -> list[dict]:
    results = []
    for c in cases:
        try:
            actual, passed = c.check()
            status = "PASS" if passed else "FAIL"
        except Exception as exc:   # an unexpected crash is a result too
            actual, status = f"{type(exc).__name__}: {str(exc).splitlines()[0][:120] if str(exc) else ''}", "ERROR"
            if "torch" in actual:
                actual += " (PyTorch not installed)"
            else:
                traceback.print_exc()
        results.append({"test_id": c.test_id, "area": c.area, "owner": c.owner, "scenario": c.scenario,
                         "input": c.input, "expected": c.expected, "actual": actual, "status": status,
                         "remarks": c.remarks})
        print(f"{c.test_id} {status:5} {c.scenario}")
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0]))
        w.writeheader()
        w.writerows(results)
    passed = sum(r["status"] == "PASS" for r in results)
    print(f"\n{len(results)} test cases: {passed} PASS, "
          f"{sum(r['status'] == 'FAIL' for r in results)} FAIL, {sum(r['status'] == 'ERROR' for r in results)} ERROR")
    print(f"Saved to: {out.name}")
    return results


if __name__ == "__main__":
    run()
