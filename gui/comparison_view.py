"""Model comparison tab: GRU vs LSTM vs Regression Tree.

Reads results that are already in the repo; it does not retrain or re-score
anything:
  * results/model_comparison/three_model/   Anson's matched comparison
        (summary.json, per_site_metrics.csv)
  * results/model_comparison/regression_tree/   Darren's extras
        (summary.json: feature importance, model size, peak/off-peak with all
        three models, last-value baseline; last_value_baseline_by_site.csv)
  * saved_models/*_test_predictions.csv and results/lstm_site2000_comparison/
        for the actual-vs-predicted chart.

Charts are drawn as SVG in a self-contained page inside an <iframe srcdoc>
(offline, hover tooltips), like the other tabs. The route comparison runs the
same trip under each model with the existing route backend.
"""
from __future__ import annotations

import csv
import html
import json
from datetime import datetime
from pathlib import Path
from typing import Callable, Mapping

ROOT = Path(__file__).resolve().parents[1]
THREE_MODEL_DIR = ROOT / "results" / "model_comparison" / "three_model"
EXTRAS_DIR = ROOT / "results" / "model_comparison" / "regression_tree"
MODELS_DIR = ROOT / "saved_models"
LSTM_SITE_2000_DIR = ROOT / "results" / "lstm_site2000_comparison"

MODELS = ("GRU", "LSTM", "Regression Tree")
BACKEND_NAMES = {"GRU": "gru", "LSTM": "lstm", "Regression Tree": "regression_tree"}
COLOURS = {"GRU": "#2a78d6", "LSTM": "#eb6834", "Regression Tree": "#1baf7a"}
INK = "#3d3c39"
BASELINE_COLOUR = "#8a8984"
TEST_DAYS = ("2006-10-27", "2006-10-28", "2006-10-29", "2006-10-30", "2006-10-31")
SUMMARY_HEADERS = ["Model", "MAE", "RMSE", "NRMSE (%)", "Sites with lowest RMSE",
                   "Training time per site (s)", "Model file (KB)", "Learned parameters"]
ROUTE_HEADERS = ["Model", "Rank", "SCATS route", "Travel time (min)", "Also in Regression Tree top 5?"]


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_results(three_model_dir: Path = THREE_MODEL_DIR, extras_dir: Path = EXTRAS_DIR) -> dict:
    """Everything the tab shows, from the committed result files."""
    per_site = [{**r, "scats_site": int(r["scats_site"]), "horizon_minutes": int(r["horizon_minutes"]),
                 "rmse": float(r["rmse"]), "mae": float(r["mae"])}
                for r in _read_csv(three_model_dir / "per_site_metrics.csv")]
    baseline = [{**r, "scats_site": int(r["scats_site"]), "horizon_minutes": int(r["horizon_minutes"]),
                 "last_value_rmse": float(r["last_value_rmse"])}
                for r in _read_csv(extras_dir / "last_value_baseline_by_site.csv")]
    return {"three_model": _read_json(three_model_dir / "summary.json"), "per_site": per_site,
            "extras": _read_json(extras_dir / "summary.json"), "baseline": baseline}


# --------------------------------------------------------------------------
# results table
# --------------------------------------------------------------------------
def summary_rows(results: Mapping) -> list[list]:
    """Network averages at 15 min plus cost columns, one row per model."""
    tm = results["three_model"]
    macro = tm.get("macro_average_metrics_by_horizon", {}).get("15", {})
    wins = tm.get("lower_rmse_site_wins_by_horizon", {}).get("15", {})
    sizes = results["extras"].get("model_size_h15", {})
    rows = []
    for m in MODELS:
        if m not in macro:
            continue
        v = macro[m]
        train = v.get("training_time_seconds")
        if train is not None and v.get("model_selection_seconds"):
            train += v["model_selection_seconds"]   # Regression Tree: include its search
        rows.append([m, round(v["mae"], 1), round(v["rmse"], 1), round(v["nrmse_percent"], 2),
                     wins.get(m), None if train is None else round(train, 3),
                     sizes.get(m, {}).get("median_file_kb"),
                     sizes.get(m, {}).get("median_learned_parameters")])
    base = results["extras"].get("last_value_baseline", {}).get("15")
    if base:
        rows.append(["Repeat last reading (baseline)", None, round(base["mean_last_value_rmse"], 1),
                     None, None, None, None, None])
    return rows


def headline(results: Mapping) -> str:
    """One-paragraph answer to 'which model is better and why', from the numbers."""
    macro = results["three_model"].get("macro_average_metrics_by_horizon", {}).get("15", {})
    wins = results["three_model"].get("lower_rmse_site_wins_by_horizon", {}).get("15", {})
    if not macro:
        return ("⚠️ Three-model results not found. Run `python -m evaluation.compare_three_models` "
                "(Anson) and `python -m evaluation.regression_tree_extras` (Darren).")
    best = min(macro, key=lambda m: macro[m]["rmse"])
    n_sites = sum(wins.values()) if wins else 0
    text = (f"**{best}** has the lowest average error across the network at 15 minutes ahead "
            f"(RMSE {macro[best]['rmse']:.1f} vehicles per 15 min) and is the most accurate model at "
            f"**{wins.get(best, 0)} of {n_sites}** sites.")
    if "Regression Tree" in macro and "GRU" in macro:
        rt, gru = macro["Regression Tree"], macro["GRU"]
        if gru.get("training_time_seconds") and rt.get("training_time_seconds"):
            # count the tree's hyper-parameter search too, so the speed claim is fair
            rt_total = rt["training_time_seconds"] + (rt.get("model_selection_seconds") or 0)
            text += (f" The Regression Tree takes {rt_total:.2f} s per site including its hyper-parameter "
                     f"search, about {gru['training_time_seconds'] / rt_total:.0f}× faster than the GRU "
                     f"({gru['training_time_seconds']:.1f} s).")
    base = results["extras"].get("last_value_baseline", {}).get("15")
    if base:
        text += (f" Simply repeating the last reading scores RMSE {base['mean_last_value_rmse']:.1f}, "
                 "so every model is judged against that floor.")
    return text


# --------------------------------------------------------------------------
# chart specs (drawn by the page's JavaScript)
# --------------------------------------------------------------------------
def overview_specs(results: Mapping) -> list[dict]:
    tm, extras = results["three_model"], results["extras"]
    specs = []
    macro = tm.get("macro_average_metrics_by_horizon", {}).get("15", {})
    base15 = extras.get("last_value_baseline", {}).get("15")
    if macro:
        specs.append({
            "type": "hbars", "title": "Network average error, 15 min ahead",
            "subtitle": f"RMSE averaged over {len(tm.get('common_sites_by_horizon', {}).get('15', []))} sites, lower is better",
            "unit": "RMSE (vehicles / 15 min)",
            "bars": [{"label": m, "value": round(macro[m]["rmse"], 2), "colour": COLOURS[m]} for m in MODELS if m in macro],
            "ref": ({"label": "repeat last reading", "value": round(base15["mean_last_value_rmse"], 2)} if base15 else None),
        })
    s2000 = tm.get("site_2000_metrics", {})
    if s2000:
        horizons = sorted(int(h) for h in s2000)
        series = [{"name": m, "colour": COLOURS[m],
                   "values": [round(s2000[str(h)][m]["rmse"], 2) if m in s2000[str(h)] else None for h in horizons]}
                  for m in MODELS]
        base = {(r["scats_site"], r["horizon_minutes"]): r["last_value_rmse"] for r in results["baseline"]}
        if all((2000, h) in base for h in horizons):
            series.append({"name": "Repeat last reading", "colour": BASELINE_COLOUR, "dashed": True,
                           "values": [round(base[(2000, h)], 2) for h in horizons]})
        specs.append({
            "type": "lines", "title": "Site 2000: error grows with the forecast horizon",
            "subtitle": "The only site with all three models at 15, 30 and 60 min",
            "x": [f"{h} min" for h in horizons], "unit": "RMSE (vehicles / 15 min)", "series": series,
        })
    per_site = [r for r in results["per_site"] if r["horizon_minutes"] == 15]
    if per_site:
        sites = sorted({r["scats_site"] for r in per_site},
                       key=lambda s: next((r["rmse"] for r in per_site if r["scats_site"] == s and r["model"] == "Regression Tree"), 0))
        lookup = {(r["scats_site"], r["model"]): r["rmse"] for r in per_site}
        specs.append({
            "type": "dots", "wide": True, "title": "Error at every site, 15 min ahead",
            "subtitle": "One row per SCATS site, sorted by Regression Tree error; busier sites have larger errors",
            "unit": "RMSE (vehicles / 15 min)", "rows": [f"SCATS {s}" for s in sites],
            "series": [{"name": m, "colour": COLOURS[m],
                        "values": [round(lookup[(s, m)], 1) if (s, m) in lookup else None for s in sites]} for m in MODELS],
        })
    peak = extras.get("peak_offpeak_site_2000", [])
    if peak:
        groups = []
        for h in (15, 30, 60):
            for period in ("Peak", "Off-Peak"):
                groups.append({"label": f"{h} min · {period.lower()}",
                               "values": {r["model"]: round(r["rmse"], 1) for r in peak
                                          if r["horizon_minutes"] == h and r["period"] == period}})
        specs.append({
            "type": "groups", "title": "Site 2000: busy vs quiet times",
            "subtitle": "Peak = weekdays 7–9 am and 4–6 pm; every model is less accurate at peak",
            "unit": "RMSE (vehicles / 15 min)", "series": [{"name": m, "colour": COLOURS[m]} for m in MODELS],
            "groups": groups,
        })
    imp = extras.get("feature_importance", {})
    if imp:
        lags = [f"lag_{i}" for i in range(1, 9)]
        specs.append({
            "type": "lines", "title": "What the Regression Tree looks at",
            "subtitle": "Share of each tree's importance by past reading (mean of 40 sites); the newest reading dominates",
            "x": [str(15 * i) for i in range(1, 9)], "unit": "Share of importance (%) · x: minutes before the forecast",
            "series": [{"name": f"{h} min ahead", "colour": c,
                        "values": [round(imp[h]["mean_importance"][lag] * 100, 1) for lag in lags]}
                       for h, c in (("15", "#1baf7a"), ("30", "#2a78d6"), ("60", "#6250d6")) if h in imp],
        })
    return specs


def prediction_files(site: int, horizon: int, models_dir: Path = MODELS_DIR,
                     lstm_site_2000_dir: Path = LSTM_SITE_2000_DIR) -> dict[str, Path]:
    files = {"GRU": models_dir / f"gru_site_{site}_h{horizon}_s8_test_predictions.csv",
             "Regression Tree": models_dir / f"rt_site_{site}_h{horizon}_s8_test_predictions.csv"}
    lstm = models_dir / f"lstm_site_{site}_h{horizon}_s8_test_predictions.csv"
    if not lstm.is_file() and site == 2000:
        lstm = lstm_site_2000_dir / f"site_2000_h{horizon}_s8" / "predictions.csv"
    files["LSTM"] = lstm
    return {m: files[m] for m in MODELS}


def actual_vs_predicted_spec(site: int, horizon: int, day: str, **dirs) -> tuple[dict | None, list[str]]:
    """Test-day line chart: actual traffic and each model's forecast. Returns (spec, notes)."""
    series, actual, notes, times = [], None, [], None
    for model, path in prediction_files(site, horizon, **dirs).items():
        rows = _read_csv(path)
        if not rows:
            notes.append(f"{model}: no saved test predictions for site {site} at {horizon} min")
            continue
        t_col = "target_datetime" if "target_datetime" in rows[0] else "datetime"
        a_col = "actual_vehicles_15min" if "actual_vehicles_15min" in rows[0] else "actual_traffic_volume"
        p_col = "predicted_vehicles_15min" if "predicted_vehicles_15min" in rows[0] else "predicted_traffic_volume"
        day_rows = [r for r in rows if r[t_col].startswith(day)]
        if not day_rows:
            notes.append(f"{model}: no test predictions on {day}")
            continue
        these_times = [r[t_col][11:16] for r in day_rows]
        if times is None:
            times = these_times
            actual = [round(float(r[a_col]), 1) for r in day_rows]
        elif these_times != times:
            notes.append(f"{model}: different time points, not shown")
            continue
        series.append({"name": model, "colour": COLOURS[model],
                       "values": [round(float(r[p_col]), 1) for r in day_rows]})
    if times is None:
        return None, notes
    errors = []
    for s in series:
        e = [p - a for p, a in zip(s["values"], actual)]
        errors.append(f"{s['name']} RMSE {(sum(x * x for x in e) / len(e)) ** 0.5:.1f}")
    weekday = datetime.fromisoformat(day).strftime("%a %d %b %Y")
    return {
        "type": "lines", "wide": True, "points": False, "height": 320,
        "title": f"SCATS {site}: actual vs predicted, {weekday}, {horizon} min ahead",
        "subtitle": "Test data the models never saw · " + " · ".join(errors) + " (this day only)",
        "x": times, "xEvery": 8, "unit": "Vehicles / 15 min",
        "series": [{"name": "Actual", "colour": INK, "width": 2.5, "values": actual}] + series,
    }, notes


# --------------------------------------------------------------------------
# route comparison
# --------------------------------------------------------------------------
def compare_routes(origin: int, destination: int, as_of: datetime, *, route_fn: Callable,
                   models=MODELS, k: int = 5) -> tuple[list[list], dict, list[str]]:
    """Run the same trip under each model.

    route_fn(model_label) -> result dict from main.compute_traffic_routes.
    Returns (table rows, {model: routes}, notes).
    """
    by_model, notes = {}, []
    for m in models:
        try:
            by_model[m] = route_fn(m)["routes"][:k]
        except (FileNotFoundError, OSError, ValueError, TypeError, KeyError, NotImplementedError, ImportError) as exc:
            notes.append(f"{m}: {str(exc).splitlines()[0][:160]}")
    reference = {tuple(r["scats_sites"]) for r in by_model.get("Regression Tree", [])}
    rows = []
    for m, routes in by_model.items():
        for r in routes:
            same = ("—" if m == "Regression Tree" or not reference else
                    "yes" if tuple(r["scats_sites"]) in reference else "no")
            rows.append([m, r["rank"], " → ".join(map(str, r["scats_sites"])),
                         round(float(r["travel_time_minutes"]), 2), same])
    return rows, by_model, notes


def best_routes_for_map(by_model: Mapping[str, list]) -> list[dict]:
    """Each model's fastest route, coloured by model, for gui.route_map."""
    out = []
    for m in MODELS:
        if by_model.get(m):
            r = by_model[m][0]
            out.append({**r, "rank": len(out) + 1, "colour": COLOURS[m],
                        "label": f"{m} best route"})
    return out


# --------------------------------------------------------------------------
# page
# --------------------------------------------------------------------------
_PAGE = r"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><style>
 :root{--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#8a8984;--grid:#e6e5e1}
 html,body{margin:0;background:var(--surface);font-family:system-ui,Segoe UI,Arial,sans-serif;color:var(--ink)}
 .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:14px;padding:12px}
 .card{border:1px solid var(--grid);border-radius:8px;padding:10px 12px;background:#fff;min-width:0}
 .card.wide{grid-column:1/-1}
 .title{font-size:14px;font-weight:600}.sub{font-size:12px;color:var(--ink2);margin:2px 0 6px}
 .legend{display:flex;flex-wrap:wrap;gap:12px;font-size:12px;color:var(--ink2);margin-bottom:4px}
 .legend span{display:inline-flex;align-items:center;gap:5px}
 .sw{display:inline-block;width:10px;height:10px;border-radius:50%}
 .ln{display:inline-block;width:16px;border-top:2px solid}
 svg{display:block;width:100%}svg text{font-size:11px;fill:var(--ink2)}
 .empty{padding:24px;color:var(--ink2);font-size:13px}
 #tip{position:fixed;pointer-events:none;background:#fff;border:1px solid #d8d7d2;border-radius:6px;
      box-shadow:0 2px 8px #0002;padding:6px 8px;font-size:12px;line-height:1.45;display:none;white-space:nowrap;z-index:9}
 #tip i{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px}
</style></head><body><div class="grid" id="grid"></div><div id="tip"></div>
<script>
const SPECS = __DATA__;
const NS = "http://www.w3.org/2000/svg", tip = document.getElementById("tip");
const el = (p, n, a) => { const e = document.createElementNS(NS, n); for (const k in a) e.setAttribute(k, a[k]); p.appendChild(e); return e; };
const txt = (p, x, y, s, a = {}) => { const t = el(p, "text", {x, y, ...a}); t.textContent = s; return t; };
const fmt = v => v === null || v === undefined ? "–" : (Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(1));
function niceMax(v) { const s = [1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10]; const p = Math.pow(10, Math.floor(Math.log10(v || 1)));
  for (const k of s) if (k * p >= v) return k * p; return 10 * p; }
function ticks(max) { const st = [1, 2, 2.5, 5, 10].map(k => k * Math.pow(10, Math.floor(Math.log10(max / 5)))).find(x => max / x <= 6); const out = []; for (let v = 0; v <= max + 1e-9; v += st) out.push(+v.toFixed(6)); return out; }
function showTip(ev, html) { tip.innerHTML = html; tip.style.display = "block";
  tip.style.left = Math.min(ev.clientX + 12, innerWidth - tip.offsetWidth - 8) + "px"; tip.style.top = (ev.clientY + 12) + "px"; }
const hideTip = () => tip.style.display = "none";
const dot = c => `<i style="background:${c}"></i>`;

function legend(card, items) { const d = document.createElement("div"); d.className = "legend";
  d.innerHTML = items.map(s => s.dashed || s.line ? `<span><i class="ln" style="border-color:${s.colour};${s.dashed ? "border-top-style:dashed" : ""}"></i>${s.name}</span>`
                                         : `<span><i class="sw" style="background:${s.colour}"></i>${s.name}</span>`).join("");
  card.appendChild(d); }

function hbars(card, s) {
  const W = card.clientWidth - 24, rowH = 34, m = {l: 120, r: 46, t: 20, b: 30}, H = m.t + m.b + rowH * s.bars.length;
  const svg = el(card, "svg", {height: H}); const max = niceMax(Math.max(...s.bars.map(b => b.value), s.ref ? s.ref.value : 0) * 1.05);
  const x = v => m.l + v / max * (W - m.l - m.r);
  ticks(max).forEach(v => { el(svg, "line", {x1: x(v), x2: x(v), y1: m.t, y2: H - m.b, stroke: "var(--grid)"}); txt(svg, x(v), H - m.b + 15, v, {"text-anchor": "middle"}); });
  txt(svg, (m.l + W - m.r) / 2, H - 2, s.unit, {"text-anchor": "middle"});
  s.bars.forEach((b, i) => { const y = m.t + i * rowH + 7;
    txt(svg, m.l - 8, y + 14, b.label, {"text-anchor": "end", style: "fill:var(--ink)"});
    const r = el(svg, "rect", {x: m.l, y, width: Math.max(1, x(b.value) - m.l), height: rowH - 14, rx: 4, fill: b.colour});
    txt(svg, x(b.value) + 6, y + 14, fmt(b.value), {style: "fill:var(--ink)"});
    r.addEventListener("mousemove", e => showTip(e, `${dot(b.colour)}<b>${b.label}</b><br>${s.unit}: ${b.value}`)); r.addEventListener("mouseleave", hideTip); });
  if (s.ref) { const rx = x(s.ref.value);
    el(svg, "line", {x1: rx, x2: rx, y1: m.t, y2: H - m.b, stroke: "var(--muted)", "stroke-width": 2, "stroke-dasharray": "4 3"});
    txt(svg, rx, m.t - 7, s.ref.label + " " + fmt(s.ref.value), {"text-anchor": "middle", style: "fill:var(--ink2)"}); }
}

function lines(card, s) {
  legend(card, s.series.map(x => ({...x, line: true})));
  const W = card.clientWidth - 24, H = s.height || 260, m = {l: 46, r: 16, t: 10, b: 38};
  const svg = el(card, "svg", {height: H}); const n = s.x.length;
  const vals = s.series.flatMap(x => x.values).filter(v => v !== null); const max = niceMax(Math.max(...vals) * 1.08);
  const x = i => m.l + (n === 1 ? 0.5 : i / (n - 1)) * (W - m.l - m.r), y = v => H - m.b - v / max * (H - m.t - m.b);
  ticks(max).forEach(v => { el(svg, "line", {x1: m.l, x2: W - m.r, y1: y(v), y2: y(v), stroke: "var(--grid)"}); txt(svg, m.l - 6, y(v) + 4, v, {"text-anchor": "end"}); });
  const every = s.xEvery || 1;
  s.x.forEach((lab, i) => { if (i % every === 0) txt(svg, x(i), H - m.b + 15, lab,
    {"text-anchor": i === 0 ? "start" : (i === n - 1 ? "end" : "middle")}); });
  txt(svg, m.l, H - 3, s.unit, {});
  s.series.forEach(se => { let d = "", pen = false;
    se.values.forEach((v, i) => { if (v === null) { pen = false; return; } d += (pen ? "L" : "M") + x(i).toFixed(1) + "," + y(v).toFixed(1); pen = true; });
    el(svg, "path", {d, fill: "none", stroke: se.colour, "stroke-width": se.width || 2, "stroke-dasharray": se.dashed ? "5 4" : "", "stroke-linejoin": "round"});
    if (s.points !== false) se.values.forEach((v, i) => v !== null && el(svg, "circle", {cx: x(i), cy: y(v), r: 4, fill: se.colour, stroke: "#fff", "stroke-width": 2})); });
  const cross = el(svg, "line", {y1: m.t, y2: H - m.b, stroke: "var(--ink2)", opacity: 0});
  const hit = el(svg, "rect", {x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, fill: "transparent"});
  hit.addEventListener("mousemove", e => { const r = svg.getBoundingClientRect(); const mx = e.clientX - r.left;
    let i = Math.round((mx - m.l) / (W - m.l - m.r) * (n - 1)); i = Math.max(0, Math.min(n - 1, i));
    cross.setAttribute("x1", x(i)); cross.setAttribute("x2", x(i)); cross.setAttribute("opacity", 0.4);
    showTip(e, `<b>${s.x[i]}</b>` + s.series.map(se => `<br>${dot(se.colour)}${se.name}: ${fmt(se.values[i])}`).join("")); });
  hit.addEventListener("mouseleave", () => { hideTip(); cross.setAttribute("opacity", 0); });
}

function dots(card, s) {
  legend(card, s.series);
  const W = card.clientWidth - 24, rowH = 15, m = {l: 78, r: 16, t: 6, b: 32}, H = m.t + m.b + rowH * s.rows.length;
  const svg = el(card, "svg", {height: H});
  const max = niceMax(Math.max(...s.series.flatMap(x => x.values).filter(v => v !== null)) * 1.05);
  const x = v => m.l + v / max * (W - m.l - m.r);
  ticks(max).forEach(v => { el(svg, "line", {x1: x(v), x2: x(v), y1: m.t, y2: H - m.b, stroke: "var(--grid)"}); txt(svg, x(v), H - m.b + 15, v, {"text-anchor": "middle"}); });
  txt(svg, (m.l + W - m.r) / 2, H - 2, s.unit, {"text-anchor": "middle"});
  s.rows.forEach((lab, i) => { const yy = m.t + i * rowH + rowH / 2;
    const vs = s.series.map(se => se.values[i]).filter(v => v !== null);
    if (i % 2) el(svg, "rect", {x: m.l, y: yy - rowH / 2, width: W - m.l - m.r, height: rowH, fill: "#f6f6f4"});
    txt(svg, m.l - 6, yy + 4, lab, {"text-anchor": "end"});
    if (vs.length) el(svg, "line", {x1: x(Math.min(...vs)), x2: x(Math.max(...vs)), y1: yy, y2: yy, stroke: "#c9c8c3", "stroke-width": 2});
    s.series.forEach(se => { const v = se.values[i]; if (v === null) return;
      const c = el(svg, "circle", {cx: x(v), cy: yy, r: 4.5, fill: se.colour, stroke: "#fff", "stroke-width": 1.5}); });
    const hit = el(svg, "rect", {x: 0, y: yy - rowH / 2, width: W, height: rowH, fill: "transparent"});
    hit.addEventListener("mousemove", e => showTip(e, `<b>${lab}</b>` + s.series.map(se => `<br>${dot(se.colour)}${se.name}: ${fmt(se.values[i])}`).join("")));
    hit.addEventListener("mouseleave", hideTip); });
}

function groups(card, s) {
  legend(card, s.series);
  const W = card.clientWidth - 24, H = s.height || 270, m = {l: 46, r: 10, t: 10, b: 50};
  const svg = el(card, "svg", {height: H}); const g = s.groups.length, k = s.series.length;
  const max = niceMax(Math.max(...s.groups.flatMap(gr => Object.values(gr.values))) * 1.08);
  const gw = (W - m.l - m.r) / g, bw = Math.min(22, (gw - 12) / k), y = v => H - m.b - v / max * (H - m.t - m.b);
  ticks(max).forEach(v => { el(svg, "line", {x1: m.l, x2: W - m.r, y1: y(v), y2: y(v), stroke: "var(--grid)"}); txt(svg, m.l - 6, y(v) + 4, v, {"text-anchor": "end"}); });
  txt(svg, m.l, H - 3, s.unit, {});
  s.groups.forEach((gr, gi) => { const x0 = m.l + gi * gw + (gw - bw * k - 2 * (k - 1)) / 2;
    gr.label.split(" · ").forEach((part, li) =>
      txt(svg, m.l + gi * gw + gw / 2, H - m.b + 15 + li * 13, part, {"text-anchor": "middle"}));
    s.series.forEach((se, si) => { const v = gr.values[se.name]; if (v === undefined) return;
      const r = el(svg, "rect", {x: x0 + si * (bw + 2), y: y(v), width: bw, height: Math.max(1, H - m.b - y(v)), rx: 3, fill: se.colour});
      r.addEventListener("mousemove", e => showTip(e, `<b>${gr.label}</b><br>${dot(se.colour)}${se.name}: ${fmt(v)}`)); r.addEventListener("mouseleave", hideTip); }); });
}

const DRAW = {hbars, lines, dots, groups};
function render() {
  const grid = document.getElementById("grid"); grid.innerHTML = "";
  if (!SPECS.length) { grid.innerHTML = '<div class="empty">No results to show yet.</div>'; }
  // lay out every card first, then draw, so each chart measures its final width
  const cards = SPECS.map(s => { const card = document.createElement("div"); card.className = "card" + (s.wide ? " wide" : "");
    card.innerHTML = `<div class="title"></div><div class="sub"></div>`;
    card.querySelector(".title").textContent = s.title; card.querySelector(".sub").textContent = s.subtitle || "";
    grid.appendChild(card); return card; });
  SPECS.forEach((s, i) => DRAW[s.type](cards[i], s));
  // fit the iframe to the charts (documentElement.scrollHeight never shrinks below the frame itself)
  try { if (window.frameElement) window.frameElement.style.height = (grid.getBoundingClientRect().height + 8) + "px"; } catch (e) {}
}
render(); let rt; addEventListener("resize", () => { clearTimeout(rt); rt = setTimeout(render, 150); });
</script></body></html>"""


def build_page(specs: list[dict]) -> str:
    data = json.dumps(specs, ensure_ascii=False).replace("</", "<\\/")
    return _PAGE.replace("__DATA__", data)


def build_iframe(specs: list[dict], initial_height: int = 600) -> str:
    doc = html.escape(build_page(specs), quote=True)
    return (f'<iframe title="Model comparison charts" srcdoc="{doc}" '
            f'style="width:100%;height:{int(initial_height)}px;border:0"></iframe>')
