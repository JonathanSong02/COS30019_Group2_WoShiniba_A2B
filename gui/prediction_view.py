"""Traffic prediction tab helpers: model coverage and a forecast-in-context chart.

The chart shows, for one SCATS site:
  * observed traffic for the hours before the forecast (the model's input
    window is shaded),
  * what actually happened afterwards (dashed), when October 2006 has it,
  * each model's prediction as a marker at the forecast target time.

It is drawn as a small self-contained SVG page with a hover tooltip, embedded
in an <iframe srcdoc=...> like the route map. No internet or extra Python
packages are needed.
"""
from __future__ import annotations

import html
import json
import re
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Callable, Iterable, Mapping

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = ROOT / "data" / "processed" / "scats_15min.csv"
DEFAULT_MODELS = ROOT / "saved_models"

HORIZONS = (15, 30, 60)
# Display name -> (backend name, saved-model filename pattern)
MODELS = {
    "GRU": ("gru", "gru_site_{site}_h{h}_s{s}.pt"),
    "LSTM": ("lstm", "lstm_site_{site}_h{h}_s{s}.pt"),
    "Regression Tree": ("regression_tree", "rt_site_{site}_h{h}_s{s}.json"),
}
# Fixed categorical colours (colour follows the model, never its rank).
MODEL_COLOURS = {"GRU": "#2a78d6", "LSTM": "#eb6834", "Regression Tree": "#1baf7a"}
CHART_HEIGHT_PX = 380
HISTORY_HOURS = 4
AFTER_HOURS = 1
TABLE_HEADERS = ["Model", "Predicted (veh / 15 min)", "Actual (veh / 15 min)", "Error", "Note"]


# --------------------------------------------------------------------------
# Model coverage (status line)
# --------------------------------------------------------------------------
def model_coverage(model_dir: str | Path = DEFAULT_MODELS,
                   sequence_length: int = 8) -> dict[str, dict[int, set[int]]]:
    """{model: {horizon: {sites with a saved model}}} from the files on disk."""
    model_dir = Path(model_dir)
    coverage: dict[str, dict[int, set[int]]] = {}
    for label, (_, pattern) in MODELS.items():
        prefix, suffix = pattern.split("{site}")
        coverage[label] = {}
        for h in HORIZONS:
            tail = suffix.format(h=h, s=sequence_length)
            regex = re.compile(re.escape(prefix) + r"(\d+)" + re.escape(tail) + "$")
            sites = {int(m.group(1)) for f in model_dir.glob(prefix + "*" + tail)
                     if (m := regex.match(f.name))}
            coverage[label][h] = sites
    return coverage


def coverage_markdown(coverage: Mapping[str, Mapping[int, set[int]]], total_sites: int) -> str:
    """Status table: how many sites each model can forecast at each horizon."""
    lines = ["| Model | " + " | ".join(f"{h} min" for h in HORIZONS) + " |",
             "|---|" + "---|" * len(HORIZONS)]
    for label in MODELS:
        cells = []
        for h in HORIZONS:
            n = len(coverage.get(label, {}).get(h, ()))
            mark = "✅" if n == total_sites else ("⚠️" if n else "❌")
            cells.append(f"{mark} {n} / {total_sites}")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def has_model(coverage, label: str, site: int, horizon: int) -> bool:
    return site in coverage.get(label, {}).get(horizon, set())


# --------------------------------------------------------------------------
# Observed data
# --------------------------------------------------------------------------
@lru_cache(maxsize=4)
def _load_series(csv_path: str, mtime: float) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["datetime"])
    return df.set_index(["scats_site", "datetime"]).sort_index()


def observed_series(site: int, start: datetime, end: datetime,
                    csv_path: str | Path = DEFAULT_CSV) -> list[tuple[datetime, float | None]]:
    """Observed 15-minute counts from start to end inclusive (None where missing)."""
    csv_path = Path(csv_path)
    data = _load_series(str(csv_path), csv_path.stat().st_mtime)
    if site not in data.index.get_level_values(0):
        raise ValueError(f"SCATS site {site} is not in the processed data")
    values = data.loc[site]["traffic_volume_15min"]
    out = []
    t = start
    while t <= end:
        v = values.get(pd.Timestamp(t))
        out.append((t, None if v is None or pd.isna(v) else float(v)))
        t += timedelta(minutes=15)
    return out


# --------------------------------------------------------------------------
# Predictions for one or more models
# --------------------------------------------------------------------------
def _short_reason(exc: Exception) -> str:
    text = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
    return text if len(text) <= 140 else text[:137] + "..."


def predict_models(
    site: int, as_of: datetime, horizon: int, sequence_length: int,
    labels: Iterable[str], predictor: Callable[..., float],
    coverage: Mapping[str, Mapping[int, set[int]]],
) -> list[dict]:
    """Run each requested model; never invent a value for a missing model."""
    results = []
    for label in labels:
        backend_name = MODELS[label][0]
        if sequence_length == 8 and not has_model(coverage, label, site, horizon):
            results.append({"model": label, "value": None,
                            "note": f"No saved {label} model for site {site} at {horizon} min"})
            continue
        try:
            value = predictor(site, as_of, model_name=backend_name,
                              horizon_minutes=horizon, sequence_length=sequence_length)
            results.append({"model": label, "value": float(value), "note": ""})
        except (FileNotFoundError, OSError, ValueError, TypeError, KeyError,
                NotImplementedError, ImportError) as exc:
            results.append({"model": label, "value": None, "note": _short_reason(exc)})
    return results


def results_table(results: list[dict], actual: float | None) -> list[list]:
    rows = []
    for r in results:
        value = r["value"]
        error = (round(value - actual, 1) if value is not None and actual is not None else None)
        rows.append([r["model"],
                     None if value is None else round(value, 1),
                     None if actual is None else round(actual, 1),
                     error,
                     r["note"] or ("Actual not in data" if actual is None and value is not None else "")])
    return rows


# --------------------------------------------------------------------------
# Chart
# --------------------------------------------------------------------------
def build_chart_payload(site: int, as_of: datetime, horizon: int, sequence_length: int,
                        results: list[dict], csv_path: str | Path = DEFAULT_CSV,
                        site_name: str = "", selected_model: str = "") -> dict:
    target = as_of + timedelta(minutes=horizon)
    start = as_of - timedelta(hours=HISTORY_HOURS)
    end = target + timedelta(hours=AFTER_HOURS)
    series = observed_series(site, start, end, csv_path)
    actual = dict(series).get(target)
    fmt = "%Y-%m-%dT%H:%M"
    return {
        "site": site, "siteName": site_name,
        "asOf": as_of.strftime(fmt), "target": target.strftime(fmt),
        "horizon": horizon,
        "windowStart": (as_of - timedelta(minutes=15 * (sequence_length - 1))).strftime(fmt),
        "points": [{"t": t.strftime(fmt), "v": v} for t, v in series],
        "actual": actual,
        "predictions": [
            {"model": r["model"], "value": r["value"], "colour": MODEL_COLOURS[r["model"]],
             "selected": r["model"] == selected_model}
            for r in results if r["value"] is not None
        ],
    }


_PAGE = r"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><style>
 :root{--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#8a8984;--grid:#e6e5e1;
       --observed:#3d3c39;--window:#efeee9}
 html,body{margin:0;overflow:hidden;background:var(--surface);font-family:system-ui,Segoe UI,Arial,sans-serif;color:var(--ink)}
 .wrap{padding:10px 12px 6px}
 .title{font-size:14px;font-weight:600}
 .sub{font-size:12px;color:var(--ink2);margin-top:2px}
 .legend{display:flex;flex-wrap:wrap;gap:14px;font-size:12px;color:var(--ink2);margin:8px 0 2px}
 .legend span{display:inline-flex;align-items:center;gap:6px}
 .key{display:inline-block;width:18px;height:0;border-top:2px solid var(--observed)}
 .key.dash{border-top-style:dashed;border-color:var(--muted)}
 .dot{width:10px;height:10px;border-radius:50%;border:2px solid var(--surface);box-shadow:0 0 0 1px #0002}
 .ring{width:10px;height:10px;border-radius:50%;border:2px solid var(--ink);background:var(--surface)}
 .win{width:14px;height:10px;background:var(--window);border:1px solid var(--grid)}
 svg text{font-size:11px;fill:var(--ink2)}
 #tip{position:fixed;pointer-events:none;background:#fff;border:1px solid #d8d7d2;border-radius:6px;
      box-shadow:0 2px 8px #0002;padding:6px 8px;font-size:12px;line-height:1.45;display:none;white-space:nowrap}
 #tip .sw{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px}
</style></head><body><div class="wrap">
<div class="title" id="title"></div><div class="sub" id="sub"></div>
<div class="legend" id="legend"></div>
<svg id="chart" width="100%" height="__H__"></svg></div><div id="tip"></div>
<script>
const D = __DATA__;
const svg = document.getElementById("chart"), tip = document.getElementById("tip");
const NS = "http://www.w3.org/2000/svg";
const el = (n, a, p = svg) => { const e = document.createElementNS(NS, n);
  for (const k in a) e.setAttribute(k, a[k]); p.appendChild(e); return e; };
const T = s => new Date(s + ":00");
const hm = d => d.toTimeString().slice(0, 5);
const day = d => d.toLocaleDateString("en-AU", {weekday: "short", day: "numeric", month: "short"});

document.getElementById("title").textContent =
  `SCATS ${D.site}${D.siteName ? " · " + D.siteName : ""} — vehicles per 15 minutes`;
document.getElementById("sub").textContent =
  `Last observed ${day(T(D.asOf))} ${hm(T(D.asOf))} · forecast for ${hm(T(D.target))} (+${D.horizon} min)`;
let lg = `<span><i class="key"></i>Observed</span><span><i class="key dash"></i>Actual after forecast time</span>`
       + `<span><i class="win"></i>Model input window</span>`;
D.predictions.forEach(p => lg += `<span><i class="dot" style="background:${p.colour}"></i>${p.model}${p.selected ? " (selected)" : ""}</span>`);
if (D.actual !== null) lg += `<span><i class="ring"></i>Actual at forecast time</span>`;
document.getElementById("legend").innerHTML = lg;

function draw() {
  svg.innerHTML = "";
  const W = svg.clientWidth || 700, H = __H__;
  const detail = D.predictions.length > 0, PW = detail ? Math.min(230, W * 0.3) : 0;
  const m = {l: 46, r: 16 + (detail ? PW + 24 : 0), t: 10, b: 34};
  const pts = D.points.map(p => ({t: T(p.t), v: p.v}));
  const t0 = pts[0].t, t1 = pts[pts.length - 1].t;
  const vals = pts.map(p => p.v).concat(D.predictions.map(p => p.value)).filter(v => v !== null);
  if (D.actual !== null) vals.push(D.actual);
  let vmax = Math.max(10, ...vals) * 1.12;
  const step = [10, 20, 25, 50, 100, 200, 250, 500].find(s => vmax / s <= 6) || 1000;
  vmax = Math.ceil(vmax / step) * step;
  const x = t => m.l + (t - t0) / (t1 - t0) * (W - m.l - m.r);
  const y = v => H - m.b - v / vmax * (H - m.t - m.b);

  // input window + grid + axes
  el("rect", {x: x(T(D.windowStart)), y: m.t, width: Math.max(2, x(T(D.asOf)) - x(T(D.windowStart))),
              height: H - m.t - m.b, fill: "var(--window)"});
  for (let v = 0; v <= vmax; v += step) {
    el("line", {x1: m.l, x2: W - m.r, y1: y(v), y2: y(v), stroke: "var(--grid)", "stroke-width": 1});
    el("text", {x: m.l - 6, y: y(v) + 4, "text-anchor": "end"}).textContent = v;
  }
  const firstHour = new Date(t0); if (firstHour.getMinutes()) firstHour.setHours(firstHour.getHours() + 1, 0, 0, 0);
  for (let t = firstHour; t <= t1; t = new Date(t.getTime() + 3600e3)) {
    el("text", {x: x(t), y: H - m.b + 16, "text-anchor": "middle"}).textContent = hm(t);
  }
  el("line", {x1: x(T(D.asOf)), x2: x(T(D.asOf)), y1: m.t, y2: H - m.b, stroke: "var(--muted)",
              "stroke-width": 1, "stroke-dasharray": "2 3"});
  el("text", {x: x(T(D.asOf)) - 4, y: m.t + 11, "text-anchor": "end"}).textContent = "last observed";

  // lines: observed up to as_of (solid), actual afterwards (dashed); gaps stay gaps
  const path = (sel, dash, colour) => {
    let d = "", pen = false;
    pts.forEach(p => { if (!sel(p) || p.v === null) { pen = false; return; }
      d += (pen ? "L" : "M") + x(p.t).toFixed(1) + "," + y(p.v).toFixed(1); pen = true; });
    if (d) el("path", {d, fill: "none", stroke: colour, "stroke-width": 2,
                       "stroke-dasharray": dash, "stroke-linejoin": "round", "stroke-linecap": "round"});
  };
  const asOf = T(D.asOf);
  path(p => p.t <= asOf, "", "var(--observed)");
  path(p => p.t >= asOf, "5 4", "var(--muted)");

  // Predictions at the forecast time. With several models the dots are spread
  // sideways so none hides another; a thin line in the model's colour runs from
  // each dot to the actual value, so its length is the error.
  const tx = x(T(D.target)), n = D.predictions.length, gap = 14;
  const dxOf = i => (i - (n - 1) / 2) * gap;
  if (D.actual !== null)
    D.predictions.forEach((p, i) => el("line", {x1: tx + dxOf(i), x2: tx + dxOf(i), y1: y(p.value), y2: y(D.actual),
      stroke: p.colour, "stroke-width": 2, opacity: 0.55}));
  D.predictions.forEach((p, i) =>
    el("circle", {cx: tx + dxOf(i), cy: y(p.value), r: p.selected ? 6 : 5, fill: p.colour,
                  stroke: "var(--surface)", "stroke-width": 2}));
  // Actual value drawn LAST (on top) as a hollow ring wide enough to span the
  // dots, so it is never covered; a dot inside the ring means an accurate forecast.
  if (D.actual !== null) {
    const rx = Math.max(8, (n - 1) / 2 * gap + 9);
    el("ellipse", {cx: tx, cy: y(D.actual), rx, ry: 8, fill: "none", stroke: "var(--ink)", "stroke-width": 2});
    el("text", {x: tx + rx + 6, y: y(D.actual) + 4, "text-anchor": "start",
                style: "fill:var(--ink);font-weight:600"}).textContent = `actual ${D.actual.toFixed(0)}`;
  }

  // Forecast detail panel (right): the same forecast-time values on a zoomed
  // scale, one column per model, so errors of a few vehicles are readable.
  if (detail) {
    const px0 = W - PW - 4, px1 = W - 8, top = m.t + 18, bot = H - m.b;
    const vs = D.predictions.map(p => p.value).concat(D.actual !== null ? [D.actual] : []);
    let lo = Math.min(...vs), hi = Math.max(...vs);
    const pad = Math.max(10, (hi - lo) * 0.35); lo -= pad; hi += pad;
    const zy = v => bot - (v - lo) / (hi - lo) * (bot - top);
    el("rect", {x: px0 - 10, y: m.t, width: PW + 14, height: H - m.t - m.b + 2, rx: 6,
                fill: "none", stroke: "var(--grid)"});
    el("text", {x: px0, y: m.t + 12, style: "fill:var(--ink);font-weight:600"}).textContent =
      `Forecast detail · ${hm(T(D.target))}`;
    const zstep = [1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500].find(s => (hi - lo) / s <= 4) || 1000;
    for (let v = Math.ceil(lo / zstep) * zstep; v <= hi; v += zstep) {
      el("line", {x1: px0 + 30, x2: px1, y1: zy(v), y2: zy(v), stroke: "var(--grid)"});
      el("text", {x: px0 + 26, y: zy(v) + 4, "text-anchor": "end"}).textContent = v;
    }
    const cols = D.predictions.length, cw = (px1 - px0 - 36) / cols;
    if (D.actual !== null) {
      el("line", {x1: px0 + 30, x2: px1, y1: zy(D.actual), y2: zy(D.actual), stroke: "var(--ink)", "stroke-width": 2});
      el("text", {x: px1, y: zy(D.actual) - 5, "text-anchor": "end",
                  style: "fill:var(--ink);font-weight:600"}).textContent = `actual ${D.actual.toFixed(0)}`;
    }
    D.predictions.forEach((p, i) => {
      const cx = px0 + 36 + cw * (i + 0.5);
      if (D.actual !== null)
        el("line", {x1: cx, x2: cx, y1: zy(p.value), y2: zy(D.actual), stroke: p.colour, "stroke-width": 2, opacity: 0.6});
      el("circle", {cx, cy: zy(p.value), r: p.selected ? 6 : 5, fill: p.colour, stroke: "var(--surface)", "stroke-width": 2});
      const err = D.actual !== null ? ` (${p.value - D.actual >= 0 ? "+" : ""}${(p.value - D.actual).toFixed(0)})` : "";
      const above = D.actual === null || p.value >= D.actual;
      el("text", {x: cx, y: zy(p.value) + (above ? -10 : 18), "text-anchor": "middle"}).textContent =
        p.value.toFixed(0) + err;
      el("text", {x: cx, y: bot + 16, "text-anchor": "middle"}).textContent =
        p.model === "Regression Tree" ? "Reg. Tree" : p.model;
    });
  }

  // hover: crosshair snapped to the nearest 15-minute step
  const cross = el("line", {y1: m.t, y2: H - m.b, stroke: "var(--ink2)", "stroke-width": 1, opacity: 0});
  const hit = el("rect", {x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, fill: "transparent"});
  hit.addEventListener("mousemove", ev => {
    const r = svg.getBoundingClientRect(), mx = ev.clientX - r.left;
    let best = pts[0]; pts.forEach(p => { if (Math.abs(x(p.t) - mx) < Math.abs(x(best.t) - mx)) best = p; });
    cross.setAttribute("x1", x(best.t)); cross.setAttribute("x2", x(best.t)); cross.setAttribute("opacity", 0.5);
    let h = `<b>${day(best.t)} ${hm(best.t)}</b><br>`;
    h += best.v === null ? "No observation (missing)" : `${best.t > asOf ? "Actual" : "Observed"}: ${best.v.toFixed(0)}`;
    if (best.t.getTime() === T(D.target).getTime())
      D.predictions.forEach(p => h += `<br><i class="sw" style="background:${p.colour}"></i>${p.model}: ${p.value.toFixed(1)}`
        + (D.actual !== null ? ` (${p.value - D.actual >= 0 ? "+" : ""}${(p.value - D.actual).toFixed(1)})` : ""));
    if (best.t >= T(D.windowStart) && best.t <= asOf) h += `<br><span style="color:#8a8984">model input</span>`;
    tip.innerHTML = h; tip.style.display = "block";
    tip.style.left = Math.min(ev.clientX + 12, innerWidth - tip.offsetWidth - 8) + "px";
    tip.style.top = (ev.clientY + 12) + "px";
  });
  hit.addEventListener("mouseleave", () => { tip.style.display = "none"; cross.setAttribute("opacity", 0); });
}
draw(); addEventListener("resize", draw);
</script></body></html>"""


def build_chart_document(payload: Mapping) -> str:
    data = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    return _PAGE.replace("__DATA__", data).replace("__H__", str(CHART_HEIGHT_PX))


def build_chart_iframe(payload: Mapping) -> str:
    doc = html.escape(build_chart_document(payload), quote=True)
    height = CHART_HEIGHT_PX + 110  # title, subtitle and legend above the plot
    return (f'<iframe title="Traffic forecast chart" srcdoc="{doc}" '
            f'style="width:100%;height:{height}px;border:1px solid #ddd;border-radius:8px"></iframe>')
