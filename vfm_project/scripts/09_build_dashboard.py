"""Step 09: Build a self-contained interactive HTML dashboard for PhysicsVFM."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Dict, List

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vfm.config import load_config


RATE_PHASES = (
    ("oil_rate_stbd", "Oil (STB/d)"),
    ("water_rate_stbd", "Water (STB/d)"),
    ("gas_rate_mscfd", "Gas (Mscf/d)"),
    ("liquid_rate_stbd", "Liquid (STB/d)"),
)

MODEL_ORDER = (
    "PhysicsVFM",
    "Choke_only",
    "IPR_only",
    "VLP_only",
    "B0_Persistence",
    "B1_LinearInterp",
    "HybridVFM",
)


def _compute_benchmark_summary(bench_df: pd.DataFrame) -> Dict[str, List[Dict[str, Any]]]:
    """Compute per-mode MAE and MAPE summaries matching scripts/08_benchmark.py."""
    summary: Dict[str, List[Dict[str, Any]]] = {}
    for cv_mode, grp in bench_df.groupby("cv_mode"):
        rows: List[Dict[str, Any]] = []
        present = [m for m in MODEL_ORDER if m in set(grp["model"].unique())]
        for model in present:
            sub = grp[grp["model"] == model]
            rec: Dict[str, Any] = {"model": model, "n_tests": int(len(sub))}
            for col, _ in RATE_PHASES:
                t = sub[f"true_{col}"].to_numpy(dtype="float64")
                p = sub[f"pred_{col}"].to_numpy(dtype="float64")
                valid = np.isfinite(t) & np.isfinite(p)
                if np.any(valid):
                    err = np.abs(p[valid] - t[valid])
                    mae = float(np.mean(err))
                    mape = float(np.mean(err / np.maximum(np.abs(t[valid]), 1.0)) * 100.0)
                else:
                    mae, mape = 0.0, 0.0
                rec[f"{col}_mae"] = round(mae, 2)
                rec[f"{col}_mape"] = round(mape, 2)
            rows.append(rec)
        summary[str(cv_mode)] = rows
    return summary


def build_payload(cfg) -> Dict[str, Any]:
    """Assemble compact JSON data payload from verified pipeline outputs."""
    cache_dir = cfg.resolve_path("cache_dir")
    outputs_dir = cfg.resolve_path("outputs_dir")

    daily = pd.read_csv(outputs_dir / "daily_allocated_rates_physics.csv")
    daily["date_dt"] = pd.to_datetime(daily["date"], utc=True)
    daily["date_str"] = daily["date_dt"].dt.strftime("%Y-%m-%d")
    daily = daily.sort_values(["well_name", "date_dt"]).reset_index(drop=True)

    wt = pd.read_csv(cache_dir / "well_tests_harmonized.csv")
    wt["t_start"] = pd.to_datetime(wt["test_start"], utc=True)
    wt["t_end"] = pd.to_datetime(wt["test_end"], utc=True)
    wt["t_mid_str"] = (wt["t_start"] + (wt["t_end"] - wt["t_start"]) / 2).dt.strftime("%Y-%m-%d")
    wt = wt.sort_values(["well_name", "t_start"]).reset_index(drop=True)

    res_p = pd.read_csv(outputs_dir / "reservoir_params.csv")
    chk_p = pd.read_csv(outputs_dir / "choke_vlp_params.csv")
    bench_df = pd.read_csv(outputs_dir / "benchmark_detail.csv")

    wells = sorted(daily["well_name"].unique().tolist())

    per_well: Dict[str, Any] = {}
    for w in wells:
        dw = daily[daily["well_name"] == w]
        ww = wt[wt["well_name"] == w]
        per_well[w] = {
            "dates": dw["date_str"].tolist(),
            "alloc": {
                "oil": np.round(dw["oil_rate_alloc_stbd"], 2).tolist(),
                "water": np.round(dw["water_rate_alloc_stbd"], 2).tolist(),
                "liquid": np.round(dw["liquid_rate_alloc_stbd"], 2).tolist(),
                "gas": np.round(dw["gas_rate_alloc_mscfd"], 2).tolist(),
                "total_gas": np.round(dw["total_gas_rate_alloc_mscfd"], 2).tolist(),
            },
            "vfm": {
                "oil": np.round(dw["oil_rate_vfm_stbd"], 2).tolist(),
                "water": np.round(dw["water_rate_vfm_stbd"], 2).tolist(),
                "liquid": np.round(dw["liquid_rate_vfm_stbd"], 2).tolist(),
                "gas": np.round(dw["gas_rate_vfm_mscfd"], 2).tolist(),
                "total_gas": np.round(dw["total_gas_vfm_mscfd"], 2).tolist(),
            },
            "gas_lift": np.round(dw["daily_gas_lift_mscfd"], 2).tolist(),
            "uptime_pct": np.round(dw["uptime_frac"] * 100.0, 1).tolist(),
            "completeness_pct": np.round(dw["data_completeness"] * 100.0, 1).tolist(),
            "shutin_dates": dw.loc[dw["uptime_frac"] < 0.99, "date_str"].tolist(),
            "imputed_dates": dw.loc[dw["is_imputed_day"].astype(bool), "date_str"].tolist(),
            "tests": {
                "dates": ww["t_mid_str"].tolist(),
                "oil": np.round(ww["oil_rate_stbd"], 1).tolist(),
                "water": np.round(ww["water_rate_stbd"], 1).tolist(),
                "liquid": np.round(ww["liquid_rate_stbd"], 1).tolist(),
                "gas": np.round(ww["gas_rate_mscfd"], 1).tolist(),
                "total_gas": np.round(ww["total_gas_rate_mscfd"], 1).tolist(),
                "wc_pct": np.round(ww["water_cut_pct"], 2).tolist(),
            },
        }

    # Field cumulative KPIs
    cum_oil_mmstb = float(daily["oil_rate_alloc_stbd"].sum() / 1e6)
    cum_wat_mmstb = float(daily["water_rate_alloc_stbd"].sum() / 1e6)
    cum_gas_bscf = float(daily["gas_rate_alloc_mscfd"].sum() / 1e6)
    cum_tot_gas_bscf = float(daily["total_gas_rate_alloc_mscfd"].sum() / 1e6)

    # Calibrated parameter table
    params_merged = res_p.merge(chk_p, on="well_name", how="left")
    param_rows: List[Dict[str, Any]] = []
    for _, r in params_merged.sort_values("well_name").iterrows():
        pbu_str = (
            f"{r['buildup_pr_datum_psia']:.1f} ({r['buildup_duration_hr']:.1f}h)"
            if pd.notna(r["buildup_pr_datum_psia"])
            else "Joint Fit"
        )
        param_rows.append({
            "well_name": str(r["well_name"]),
            "pbu_anchor": pbu_str,
            "J": round(float(r["J_stbd_psi"]), 3),
            "Pr0": round(float(r["Pr0_datum_psia"]), 1),
            "decline": round(float(r["decline_psi_per_day"]), 3),
            "ipr_mape": round(float(r["loto_mape_pct"]), 2),
            "cd": round(float(r["choke_cd"]), 4),
            "choke_mape": round(float(r["choke_loto_mape_pct"]), 2),
            "slip_k": round(float(r["vlp_slip_k"]), 2),
            "vlp_mape": round(float(r["vlp_loto_mape_pct"]), 2),
        })

    return {
        "wells": wells,
        "per_well": per_well,
        "benchmark": _compute_benchmark_summary(bench_df),
        "params_table": param_rows,
        "kpis": {
            "n_wells": len(wells),
            "n_days": int(daily["date_str"].nunique()),
            "n_tests": int(len(wt)),
            "cum_oil_mmstb": round(cum_oil_mmstb, 2),
            "cum_wat_mmstb": round(cum_wat_mmstb, 2),
            "cum_gas_bscf": round(cum_gas_bscf, 2),
            "cum_tot_gas_bscf": round(cum_tot_gas_bscf, 2),
        },
    }


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>PhysicsVFM Interactive Dashboard</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    :root {
      --bg: #0b1120;
      --card: #111a2e;
      --border: #1e293b;
      --text: #e2e8f0;
      --muted: #94a3b8;
      --accent: #38bdf8;
      --good: #34d399;
      --warn: #fbbf24;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      padding: 24px;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: var(--bg);
      color: var(--text);
    }
    .container { max-width: 1440px; margin: 0 auto; }
    header {
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      align-items: center;
      gap: 16px;
      margin-bottom: 20px;
      padding-bottom: 16px;
      border-bottom: 1px solid var(--border);
    }
    h1 { margin: 0; font-size: 1.55rem; letter-spacing: -0.01em; }
    .subtitle { color: var(--muted); font-size: 0.9rem; margin-top: 4px; }
    .controls {
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      align-items: center;
    }
    label { font-size: 0.85rem; color: var(--muted); display: flex; align-items: center; gap: 8px; }
    select {
      background: var(--card);
      color: var(--text);
      border: 1px solid #334155;
      border-radius: 8px;
      padding: 8px 12px;
      font-size: 0.9rem;
      cursor: pointer;
    }
    .kpi-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
      gap: 14px;
      margin-bottom: 20px;
    }
    .kpi-card {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 14px 16px;
    }
    .kpi-label { font-size: 0.78rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.04em; }
    .kpi-val { font-size: 1.45rem; font-weight: 700; margin-top: 6px; color: var(--text); }
    .kpi-sub { font-size: 0.8rem; color: var(--good); margin-top: 4px; }
    .grid-2 {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(600px, 1fr));
      gap: 16px;
      margin-bottom: 16px;
    }
    .panel {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 16px;
    }
    .panel-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 10px;
      margin-bottom: 8px;
    }
    .panel-title { font-size: 1rem; font-weight: 600; }
    .chart { width: 100%; height: 390px; }
    table { width: 100%; border-collapse: collapse; font-size: 0.86rem; }
    th, td {
      padding: 9px 10px;
      text-align: right;
      border-bottom: 1px solid var(--border);
    }
    th { color: var(--muted); font-weight: 600; background: #0f172a; }
    th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) { text-align: left; }
    tr:hover td { background: rgba(56, 189, 248, 0.06); }
    .pill {
      display: inline-block;
      padding: 2px 8px;
      border-radius: 999px;
      font-size: 0.75rem;
      background: rgba(56, 189, 248, 0.14);
      color: var(--accent);
    }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div>
        <h1>PhysicsVFM — Intelligent Well Production &amp; Benchmark Dashboard</h1>
        <div class="subtitle">Mechanistic IPR + Multiphase Choke + 2-Segment Tubing VLP (WLS Reconciled) &bull; Jan–Sep 2025</div>
      </div>
      <div class="controls">
        <label>Well:
          <select id="wellSelect"></select>
        </label>
        <label>Daily Series:
          <select id="seriesMode">
            <option value="alloc" selected>Allocated (Test-Anchored)</option>
            <option value="vfm">Raw PhysicsVFM (Unanchored)</option>
          </select>
        </label>
      </div>
    </header>

    <section class="kpi-grid" id="kpiGrid"></section>

    <div class="grid-2">
      <div class="panel">
        <div class="panel-header">
          <span class="panel-title" id="liqTitle">Oil, Water &amp; Liquid Rates vs. Well Tests</span>
          <span class="pill">Shaded = Shut-in / Imputed Outage</span>
        </div>
        <div id="liqChart" class="chart"></div>
      </div>
      <div class="panel">
        <div class="panel-header">
          <span class="panel-title" id="gasTitle">Formation Gas, Gas Lift, Total Gas &amp; Uptime</span>
        </div>
        <div id="gasChart" class="chart"></div>
      </div>
    </div>

    <div class="grid-2">
      <div class="panel">
        <div class="panel-header">
          <span class="panel-title">Field-Wide Stacked Daily Allocated Oil (All 10 Wells)</span>
        </div>
        <div id="fieldChart" class="chart"></div>
      </div>
      <div class="panel">
        <div class="panel-header">
          <span class="panel-title">Model Benchmark Comparison</span>
          <div class="controls">
            <select id="cvSelect">
              <option value="walk_forward" selected>Walk-Forward (Causal, N=75)</option>
              <option value="loto">Leave-One-Test-Out (N=85)</option>
            </select>
            <select id="metricSelect">
              <option value="mape" selected>MAPE (%)</option>
              <option value="mae">MAE (STB/d or Mscf/d)</option>
            </select>
          </div>
        </div>
        <div id="benchChart" class="chart"></div>
      </div>
    </div>

    <div class="panel">
      <div class="panel-header">
        <span class="panel-title">Calibrated Per-Well Physics Parameters &amp; Single-Estimator LOTO Accuracy</span>
      </div>
      <div style="overflow-x:auto;">
        <table id="paramsTable">
          <thead>
            <tr>
              <th>Well</th>
              <th>PBU Anchor (psia)</th>
              <th>PI J (stb/d/psi)</th>
              <th>Pr0 Datum (psia)</th>
              <th>Decline (psi/d)</th>
              <th>IPR LOTO MAPE</th>
              <th>Choke Cd</th>
              <th>Choke LOTO MAPE</th>
              <th>VLP slip_k</th>
              <th>VLP LOTO MAPE</th>
            </tr>
          </thead>
          <tbody></tbody>
        </table>
      </div>
    </div>
  </div>

  <script>
    const DATA = __PAYLOAD_JSON__;

    const baseLayout = {
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(15,23,42,0.55)',
      font: { color: '#e2e8f0', size: 11.5 },
      margin: { l: 55, r: 52, t: 18, b: 42 },
      legend: { orientation: 'h', y: -0.18, x: 0 },
      xaxis: { gridcolor: '#1e293b', zeroline: false },
      yaxis: { gridcolor: '#1e293b', zeroline: false }
    };

    function renderKPIs() {
      const wf = DATA.benchmark.walk_forward || [];
      const pv = wf.find(r => r.model === 'PhysicsVFM') || {};
      const b0 = wf.find(r => r.model === 'B0_Persistence') || {};
      const hy = wf.find(r => r.model === 'HybridVFM') || {};
      const k = DATA.kpis;
      const cards = [
        { label: 'Cumulative Field Oil', val: `${k.cum_oil_mmstb.toFixed(2)} MMSTB`, sub: `${k.n_wells} wells • ${k.n_days} days` },
        { label: 'Cumulative Water / Gas', val: `${k.cum_wat_mmstb.toFixed(2)} MMSTB`, sub: `Formation Gas: ${k.cum_gas_bscf.toFixed(2)} Bscf` },
        { label: 'Walk-Forward Liquid MAPE', val: `${(pv.liquid_rate_stbd_mape ?? 0).toFixed(2)}%`, sub: `MAE ${pv.liquid_rate_stbd_mae} vs B0 ${b0.liquid_rate_stbd_mae} / Hybrid ${hy.liquid_rate_stbd_mae}` },
        { label: 'Walk-Forward Oil MAPE', val: `${(pv.oil_rate_stbd_mape ?? 0).toFixed(2)}%`, sub: `MAE ${pv.oil_rate_stbd_mae} vs B0 ${b0.oil_rate_stbd_mae} / Hybrid ${hy.oil_rate_stbd_mae}` },
        { label: 'Walk-Forward Gas MAPE', val: `${(pv.gas_rate_mscfd_mape ?? 0).toFixed(2)}%`, sub: `Calibrated on ${k.n_tests} well tests` }
      ];
      document.getElementById('kpiGrid').innerHTML = cards.map(c => `
        <div class="kpi-card">
          <div class="kpi-label">${c.label}</div>
          <div class="kpi-val">${c.val}</div>
          <div class="kpi-sub">${c.sub}</div>
        </div>
      `).join('');
    }

    function buildShapes(shutinDates, imputedDates) {
      const shapes = [];
      for (const d of shutinDates) {
        shapes.push({
          type: 'rect', xref: 'x', yref: 'paper',
          x0: d, x1: d, y0: 0, y1: 1,
          line: { color: 'rgba(248,113,113,0.45)', width: 2 },
          fillcolor: 'rgba(248,113,113,0.22)'
        });
      }
      for (const d of imputedDates) {
        shapes.push({
          type: 'rect', xref: 'x', yref: 'paper',
          x0: d, x1: d, y0: 0, y1: 1,
          line: { color: 'rgba(251,191,36,0.65)', width: 2, dash: 'dot' }
        });
      }
      return shapes;
    }

    function renderWellCharts() {
      const well = document.getElementById('wellSelect').value;
      const mode = document.getElementById('seriesMode').value;
      const w = DATA.per_well[well];
      const s = w[mode];
      const t = w.tests;
      const shapes = buildShapes(w.shutin_dates, w.imputed_dates);

      document.getElementById('liqTitle').textContent = `${well} — Oil, Water & Liquid Rates vs. Well Tests`;
      document.getElementById('gasTitle').textContent = `${well} — Gas Balance & Operational Uptime`;

      const liqTraces = [
        { x: w.dates, y: s.liquid, name: 'Liquid Daily', mode: 'lines', line: { color: '#38bdf8', width: 2 } },
        { x: w.dates, y: s.oil, name: 'Oil Daily', mode: 'lines', line: { color: '#34d399', width: 2 } },
        { x: w.dates, y: s.water, name: 'Water Daily', mode: 'lines', line: { color: '#818cf8', width: 2 } },
        { x: t.dates, y: t.liquid, name: 'Test Liquid', mode: 'markers', marker: { color: '#bae6fd', size: 8, symbol: 'diamond', line: { color: '#0284c7', width: 1 } } },
        { x: t.dates, y: t.oil, name: 'Test Oil', mode: 'markers', marker: { color: '#a7f3d0', size: 8, symbol: 'circle', line: { color: '#059669', width: 1 } } },
        { x: t.dates, y: t.water, name: 'Test Water', mode: 'markers', marker: { color: '#c7d2fe', size: 8, symbol: 'square', line: { color: '#4f46e5', width: 1 } } }
      ];
      Plotly.newPlot('liqChart', liqTraces, {
        ...baseLayout,
        yaxis: { ...baseLayout.yaxis, title: 'Rate (STB/d)' },
        shapes
      }, { responsive: true });

      const gasTraces = [
        { x: w.dates, y: s.total_gas, name: 'Total Gas Daily', mode: 'lines', line: { color: '#f472b6', width: 2 } },
        { x: w.dates, y: w.gas_lift, name: 'Gas Lift Daily', mode: 'lines', line: { color: '#fbbf24', width: 1.6, dash: 'dot' } },
        { x: w.dates, y: s.gas, name: 'Formation Gas Daily', mode: 'lines', line: { color: '#22d3ee', width: 2 } },
        { x: t.dates, y: t.total_gas, name: 'Test Total Gas', mode: 'markers', marker: { color: '#fbcfe8', size: 7, symbol: 'diamond' } },
        { x: t.dates, y: t.gas, name: 'Test Form. Gas', mode: 'markers', marker: { color: '#a5f3fc', size: 7, symbol: 'circle' } },
        { x: w.dates, y: w.uptime_pct, name: 'Uptime (%)', mode: 'lines', yaxis: 'y2', line: { color: 'rgba(148,163,184,0.45)', width: 1.3 } }
      ];
      Plotly.newPlot('gasChart', gasTraces, {
        ...baseLayout,
        yaxis: { ...baseLayout.yaxis, title: 'Gas Rate (Mscf/d)' },
        yaxis2: { title: 'Uptime (%)', overlaying: 'y', side: 'right', range: [0, 108], showgrid: false },
        shapes
      }, { responsive: true });
    }

    function renderFieldStacked() {
      const palette = ['#34d399', '#38bdf8', '#818cf8', '#f472b6', '#fbbf24', '#fb7185', '#2dd4bf', '#a78bfa', '#60a5fa', '#facc15'];
      const traces = DATA.wells.map((well, i) => ({
        x: DATA.per_well[well].dates,
        y: DATA.per_well[well].alloc.oil,
        name: well,
        stackgroup: 'oil',
        mode: 'lines',
        line: { width: 0.6, color: palette[i % palette.length] }
      }));
      Plotly.newPlot('fieldChart', traces, {
        ...baseLayout,
        yaxis: { ...baseLayout.yaxis, title: 'Total Field Oil (STB/d)' }
      }, { responsive: true });
    }

    function renderBenchmark() {
      const cv = document.getElementById('cvSelect').value;
      const metric = document.getElementById('metricSelect').value;
      const rows = DATA.benchmark[cv] || [];
      const models = rows.map(r => r.model);
      const phases = [
        { key: 'oil_rate_stbd', name: 'Oil', color: '#34d399' },
        { key: 'water_rate_stbd', name: 'Water', color: '#818cf8' },
        { key: 'gas_rate_mscfd', name: 'Formation Gas', color: '#22d3ee' },
        { key: 'liquid_rate_stbd', name: 'Total Liquid', color: '#38bdf8' }
      ];
      const traces = phases.map(p => ({
        x: models,
        y: rows.map(r => r[`${p.key}_${metric}`]),
        name: p.name,
        type: 'bar',
        marker: { color: p.color },
        text: rows.map(r => r[`${p.key}_${metric}`].toFixed(metric === 'mape' ? 2 : 1)),
        textposition: 'auto'
      }));
      Plotly.newPlot('benchChart', traces, {
        ...baseLayout,
        barmode: 'group',
        yaxis: { ...baseLayout.yaxis, title: metric === 'mape' ? 'MAPE (%)' : 'MAE (STB/d or Mscf/d)' }
      }, { responsive: true });
    }

    function renderParamsTable() {
      const tbody = document.querySelector('#paramsTable tbody');
      tbody.innerHTML = DATA.params_table.map(r => `
        <tr>
          <td><strong>${r.well_name}</strong></td>
          <td>${r.pbu_anchor}</td>
          <td>${r.J.toFixed(3)}</td>
          <td>${r.Pr0.toFixed(1)}</td>
          <td>${r.decline.toFixed(3)}</td>
          <td>${r.ipr_mape.toFixed(2)}%</td>
          <td>${r.cd.toFixed(4)}</td>
          <td>${r.choke_mape.toFixed(2)}%</td>
          <td>${r.slip_k.toFixed(2)}</td>
          <td>${r.vlp_mape.toFixed(2)}%</td>
        </tr>
      `).join('');
    }

    function init() {
      const sel = document.getElementById('wellSelect');
      sel.innerHTML = DATA.wells.map(w => `<option value="${w}">${w}</option>`).join('');
      sel.addEventListener('change', renderWellCharts);
      document.getElementById('seriesMode').addEventListener('change', renderWellCharts);
      document.getElementById('cvSelect').addEventListener('change', renderBenchmark);
      document.getElementById('metricSelect').addEventListener('change', renderBenchmark);

      renderKPIs();
      renderWellCharts();
      renderFieldStacked();
      renderBenchmark();
      renderParamsTable();
    }

    init();
  </script>
</body>
</html>
"""


def main() -> None:
    cfg = load_config()
    outputs_dir = cfg.resolve_path("outputs_dir")
    outputs_dir.mkdir(parents=True, exist_ok=True)

    payload = build_payload(cfg)
    html = HTML_TEMPLATE.replace("__PAYLOAD_JSON__", json.dumps(payload, separators=(",", ":")))

    out_path = outputs_dir / "vfm_dashboard.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"Dashboard generated -> {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
