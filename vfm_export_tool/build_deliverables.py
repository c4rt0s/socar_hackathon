#!/usr/bin/env python3
"""Build all official SOCAR Hackathon 2026 deliverables into `deliverables/`:
  1. Phase 1 (VFM): CSV + multi-sheet Excel + structured data_cleaning_log.csv
  2. Phase 2 (PTA): CSV + multi-sheet Excel + 9-algorithm comparison + Tubing-ID Storage Check + Assumption-Sensitivity Workbook (incl. Perrine multiphase)
  3. Trained model (.pkl): vfm_stacked_v2.pkl
  4. Static PTA Plots folder: 5 publication-grade PNGs per shut-in (30 plots) + field overview
"""
from __future__ import annotations

import math
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EXPORT_DIR = Path(__file__).resolve().parent
ROOT_DIR = EXPORT_DIR.parent
PROJ_DIR = ROOT_DIR / "vfm_project"
DELIV_DIR = ROOT_DIR / "deliverables"

sys.path.insert(0, str(EXPORT_DIR))
sys.path.insert(0, str(PROJ_DIR / "src"))

from vfm_phase_decomposed_stacked_vfm import resolve_raw_dir, compute_producing_time_hr
RAW_DIR = resolve_raw_dir()

from export_horner_cases import load_shutins_and_data, _fit_line, _horner_props
from export_smoothed_stable_bourdet_horner import (
    smooth_raw_1min_pressure,
    smooth_log_grid_and_bourdet,
    find_most_stable_bourdet_interval,
)
from pta_hybrid_iarf_from_scratch import (
    hampel_and_savgol_preprocess,
    log_resample_and_smooth,
    detect_iarf_horner_only,
    detect_iarf_bourdet_only,
    detect_iarf_intersection_ensemble,
)
from vfm.pta.advanced_pta import run_advanced_non_iarf_suite


# ---------------------------------------------------------------------------
# 1. Fluid helpers for Tubing-ID Wellbore Storage Check & Perrine Multiphase
# ---------------------------------------------------------------------------
def water_viscosity_mccain(T_f: float, p_psia: float, salinity_ppm: float) -> float:
    S = salinity_ppm / 1e4
    A = 109.574 - 8.40564 * S + 0.313314 * S ** 2 + 8.72213e-3 * S ** 3
    B = 1.12166 - 2.63951e-2 * S + 6.79461e-4 * S ** 2 + 5.47119e-5 * S ** 3 - 1.55586e-6 * S ** 4
    mu1 = A * (T_f ** (-B))
    return float(mu1 * (0.9994 + 4.0295e-5 * p_psia + 3.1062e-9 * p_psia ** 2))


def water_density_mccain(salinity_ppm: float) -> float:
    S = salinity_ppm / 1e4
    return float(62.368 + 0.438603 * S + 1.60074e-3 * S ** 2)


def oil_density_api(api: float) -> float:
    return float(62.4 * 141.5 / (131.5 + api))


def tubing_storage_estimate(tubing_id_in: float, rho_lb_ft3: float) -> float:
    """Rising liquid-level storage C = 144 * A_tub / (5.615 * rho), bbl/psi."""
    A = math.pi / 4.0 * ((tubing_id_in / 12.0) ** 2)
    return float(144.0 * A / (5.615 * rho_lb_ft3))


# ---------------------------------------------------------------------------
# 2. Structured Data-Cleaning Audit Log on Raw Files
# ---------------------------------------------------------------------------
def build_cleaning_log() -> pd.DataFrame:
    rows = []
    def _add(tbl: str, issue: str, n: int, detail: str = ""):
        if n > 0:
            rows.append({"table": tbl, "issue": issue, "rows_affected": int(n), "detail": detail})

    # Well tests audit
    wt = pd.read_csv(RAW_DIR / "well_tests.csv")
    for c in ("oil_rate_stbd", "water_rate_stbd", "gas_rate_mscfd", "gas_lift_rate_mscfd", "total_gas_rate_mscfd", "liquid_rate_stbd"):
        if c in wt.columns:
            neg = wt[wt[c] < 0]
            for _, r in neg.iterrows():
                _add("well_tests", "negative rate -> sign flipped", 1, f"{r['well_name']} {str(r['test_start'])[:10]} {c}={r[c]}")
            wt[c] = wt[c].abs()
    exp_g = wt["oil_rate_stbd"] * wt["gor_scf_stb"] / 1000.0
    ratio = wt["gas_rate_mscfd"] / exp_g.replace(0.0, np.nan)
    sm3 = ratio > 15.0
    for _, r in wt[sm3].iterrows():
        _add("well_tests", "gas in Sm3/d -> converted to Mscf/d (/28.3168)", 1, f"{r['well_name']} {str(r['test_start'])[:10]}")

    # High-freq SCADA streaming audit
    sensors = ["whp_psia", "whp_downstream_psia", "dh_pressure_psia", "dh_temperature_degf", "annulus_pressure_psia", "gas_lift_rate_mscfd"]
    sent_counts = {c: 0 for c in sensors}
    oob_counts = {c: 0 for c in sensors}
    total_rows = 0
    dup_keys = set()
    dup_count = 0
    obs_well_days: set[tuple[str, str]] = set()
    wells_seen: set[str] = set()
    d_min, d_max = None, None
    for chunk in pd.read_csv(RAW_DIR / "high_freq_data.csv", chunksize=300_000):
        total_rows += len(chunk)
        dup_count += int(chunk.duplicated(subset=["well_name", "timestamp"]).sum())
        ts_c = pd.to_datetime(chunk["timestamp"], errors="coerce")
        valid_ts = ts_c.dropna()
        if len(valid_ts):
            cmin, cmax = valid_ts.min().floor("D"), valid_ts.max().floor("D")
            d_min = cmin if d_min is None else min(d_min, cmin)
            d_max = cmax if d_max is None else max(d_max, cmax)
        wells_seen.update(chunk["well_name"].dropna().astype(str).unique().tolist())
        pairs = pd.DataFrame({"w": chunk["well_name"].astype(str), "d": ts_c.dt.strftime("%Y-%m-%d")}).dropna().drop_duplicates()
        obs_well_days.update(zip(pairs["w"], pairs["d"]))
        for c in sensors:
            s = pd.to_numeric(chunk[c], errors="coerce")
            sent = s <= -900.0
            sent_counts[c] += int(sent.sum())
            s = s.mask(sent, np.nan)
            if c == "dh_temperature_degf":
                oob = (s < 32.0) | (s > 400.0)
            elif c == "gas_lift_rate_mscfd":
                oob = (s < 0.0) | (s > 20000.0)
            else:
                oob = (s < 14.0) | (s > 15000.0)
            oob_counts[c] += int(oob.sum())

    _add("high_freq", "raw 1-min rows audited", total_rows, f"{len(wells_seen)} wells ({d_min:%Y-%m-%d} to {d_max:%Y-%m-%d})" if d_min is not None else "")
    _add("high_freq", "duplicate (well, timestamp) rows removed", dup_count)
    for c in sensors:
        _add("high_freq", "sentinel (-999.25) -> NaN", sent_counts[c], c)
        _add("high_freq", "outside physical limits -> NaN", oob_counts[c], c)
    if d_min is not None and d_max is not None:
        all_days = [d.strftime("%Y-%m-%d") for d in pd.date_range(d_min, d_max, freq="D")]
        missing_wd = [(w, d) for w in sorted(wells_seen) for d in all_days if (w, d) not in obs_well_days]
        if missing_wd:
            detail_gap = ", ".join(f"{w} {d}" for w, d in missing_wd[:10])
            _add("high_freq", "full-day SCADA telemetry outage imputed via hourly grid", len(missing_wd), detail_gap)
    return pd.DataFrame(rows)


def _autofit_sheet(xw, name: str, df: pd.DataFrame):
    ws = xw.sheets[name]
    for i, col in enumerate(df.columns, 1):
        max_len = int(df[col].astype(str).str.len().max()) if len(df) else 0
        ws.column_dimensions[ws.cell(row=1, column=i).column_letter].width = min(56, max(11, len(str(col)) + 2, max_len + 2))
    ws.freeze_panes = "A2"


def main():
    vfm_dir = DELIV_DIR / "phase1_vfm"
    pta_dir = DELIV_DIR / "phase2_pta"
    mod_dir = DELIV_DIR / "models"
    plots_dir = DELIV_DIR / "pta_plots"
    for d in (vfm_dir, pta_dir, mod_dir, plots_dir):
        d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # PHASE 1: VFM Deliverables
    # ------------------------------------------------------------------
    print("=== Building Phase 1 (VFM) Deliverables ===")
    vfm_df = pd.read_csv(EXPORT_DIR / "vfm_phase_decomposed_stacked_v2.csv")
    vfm_df["date_dt"] = pd.to_datetime(vfm_df["date"])
    clean_log = build_cleaning_log()
    clean_log.to_csv(vfm_dir / "data_cleaning_log.csv", index=False)

    per_well_vfm = vfm_df.groupby("well_name", as_index=False).agg(
        days=("date", "count"),
        avg_oil_stbd=("oil_rate_stbd", "mean"),
        cum_oil_stb=("oil_rate_stbd", "sum"),
        avg_water_stbd=("water_rate_stbd", "mean"),
        cum_water_stb=("water_rate_stbd", "sum"),
        avg_gas_mscfd=("gas_rate_mscfd", "mean"),
        cum_gas_mscf=("gas_rate_mscfd", "sum"),
    ).round(2)

    bench_path = PROJ_DIR / "outputs" / "model_benchmark_summary.csv"
    bench_df = pd.read_csv(bench_path) if bench_path.exists() else pd.DataFrame()

    vfm_export = vfm_df[["date", "well_name", "oil_rate_stbd", "water_rate_stbd", "gas_rate_mscfd"]].copy()
    vfm_export.to_csv(vfm_dir / "vfm_daily_predictions.csv", index=False)
    with pd.ExcelWriter(vfm_dir / "vfm_daily_predictions.xlsx", engine="openpyxl") as xw:
        vfm_export.to_excel(xw, sheet_name="vfm_predictions", index=False)
        _autofit_sheet(xw, "vfm_predictions", vfm_export)
        per_well_vfm.to_excel(xw, sheet_name="per_well_summary", index=False)
        _autofit_sheet(xw, "per_well_summary", per_well_vfm)
        if not bench_df.empty:
            bench_df.to_excel(xw, sheet_name="model_benchmark", index=False)
            _autofit_sheet(xw, "model_benchmark", bench_df)
        clean_log.to_excel(xw, sheet_name="data_cleaning_log", index=False)
        _autofit_sheet(xw, "data_cleaning_log", clean_log)
    print(f"  Saved {len(vfm_export)} VFM rows -> {vfm_dir / 'vfm_daily_predictions.(csv|xlsx)'}")

    # Copy trained .pkl model (Deliverable #2)
    pkl_src = PROJ_DIR / "models" / "vfm_stacked_v2.pkl"
    shutil.copy2(pkl_src, mod_dir / "vfm_stacked_v2.pkl")
    print(f"  Copied trained model -> {mod_dir / 'vfm_stacked_v2.pkl'} ({pkl_src.stat().st_size / 1024:.1f} KB)")

    # ------------------------------------------------------------------
    # PHASE 2: PTA Deliverables, Tubing-ID Storage Check, Sensitivity & 30 Plots
    # ------------------------------------------------------------------
    print("\n=== Building Phase 2 (PTA) Deliverables, Sensitivity & Static Plots ===")
    si_df, hf_map, pvt_df, comp_df, _ = load_shutins_and_data()
    wt_raw = pd.read_csv(RAW_DIR / "well_tests.csv")
    for c in ("oil_rate_stbd", "water_rate_stbd", "gas_rate_mscfd"):
        wt_raw[c] = wt_raw[c].abs()
    hr_v2 = pd.read_csv(EXPORT_DIR / "hourly_full_7sensor_telemetry_robust.csv", parse_dates=["hour", "date"])
    vfm_daily_idx = vfm_df.copy()
    vfm_daily_idx["date"] = pd.to_datetime(vfm_daily_idx["date"])

    # Load any additional cached VFM daily models (for realistic inter-model Qo spread in Tornado)
    all_model_daily: dict[str, pd.DataFrame] = {"synth_stacked_v2": vfm_daily_idx}
    model_cache_dir = PROJ_DIR / "cache" / "models"
    if model_cache_dir.is_dir():
        for pkl_m in sorted(model_cache_dir.glob("*.pkl")):
            try:
                blob_m = pd.read_pickle(pkl_m)
                df_m = blob_m.get("daily") if isinstance(blob_m, dict) else blob_m
                if isinstance(df_m, pd.DataFrame) and "well_name" in df_m.columns and not df_m.empty:
                    all_model_daily[pkl_m.stem] = df_m
            except Exception:
                pass

    pta_main_rows = []
    pta_9m_rows = []
    storage_rows = []
    sens_rows = []
    qo_tornado_panels = []
    qo_tornado_rows = []
    overview_data = []

    BLUE, RED, GREEN, ORANGE, PURPLE, GREY = "#1f6feb", "#d62728", "#1a7f37", "#d97706", "#7c3aed", "#64748b"

    for _, r in si_df.iterrows():
        w = str(r["well_name"])
        t0, t1, dur = pd.Timestamp(r["t_shut"]), pd.Timestamp(r["t_open"]), float(r["duration_hr"])
        ev_id = f"{w}_{t0:%Y%m%d_%H%M}"
        ev_dir = plots_dir / ev_id
        ev_dir.mkdir(parents=True, exist_ok=True)

        pv = pvt_df.loc[w]
        cp = comp_df.loc[w]

        # Pre-shut-in VFM oil & water rates from our refined V2 daily predictions
        dw = vfm_daily_idx[vfm_daily_idx["well_name"] == w].sort_values("date")
        day_shut = t0.floor("D")
        data_start = dw["date"].min()
        prior_full = dw[(dw["date"] < day_shut) & (dw["oil_rate_stbd"] > 10.0)]
        qo = float(prior_full["oil_rate_stbd"].iloc[-1])
        qw = float(prior_full["water_rate_stbd"].iloc[-1])
        qo_3d = float(prior_full["oil_rate_stbd"].tail(3).mean())
        cum_since_start = float(dw[dw["date"] < day_shut]["oil_rate_stbd"].sum()) + qo * ((t0 - day_shut).total_seconds() / 86400.0)
        tp_hr = compute_producing_time_hr(w, t0, qo, comp_df, cum_since_start, dataset_start=data_start)

        df_w = hf_map[w].sort_values("timestamp")
        pre_all = df_w[df_w["timestamp"] <= t0]["dh_pressure_psia"].dropna()
        pwf_3h = float(df_w[(df_w["timestamp"] >= t0 - pd.Timedelta(hours=3)) & (df_w["timestamp"] <= t0)]["dh_pressure_psia"].median())
        pwf_3m = float(pre_all.tail(3).mean())
        pwf_15m = float(pre_all.tail(15).mean())

        post = df_w[(df_w["timestamp"] > t0) & (df_w["timestamp"] <= t1)][["timestamp", "dh_pressure_psia"]].dropna()
        dt = (post["timestamp"] - t0).dt.total_seconds().to_numpy(dtype=float) / 3600.0
        pws_raw = post["dh_pressure_psia"].to_numpy(dtype=float)
        pws_sm = smooth_raw_1min_pressure(dt, pws_raw)
        tb, _, pb_sm, der = smooth_log_grid_and_bourdet(dt, pws_sm, pwf_3h, tp_hr)
        win = find_most_stable_bourdet_interval(tb, pb_sm, der, tp_hr)
        sel = (dt >= win["dt_start"]) & (dt <= win["dt_end"])
        x_hor = np.log10((tp_hr + dt) / dt)
        m_h, p_star, r2_h = _fit_line(x_hor[sel], pws_sm[sel])
        props = _horner_props(m_h, p_star, pwf_3h, tp_hr, qo, pv)

        # Compute genuine 5-stage Hybrid Confidence Scorecard (zero hardcoded confidence numbers!)
        _, p_hyb_sm = hampel_and_savgol_preprocess(dt, pws_raw)
        tb_hyb, pb_hyb = log_resample_and_smooth(dt, p_hyb_sm, n_per_decade=28)
        r1_h = detect_iarf_horner_only(tb_hyb, pb_hyb, dt, p_hyb_sm, pwf_3h, tp_hr, qo, pv)
        r2_h_det = detect_iarf_bourdet_only(tb_hyb, pb_hyb, dt, p_hyb_sm, pwf_3h, tp_hr, qo, pv)
        r3_h = detect_iarf_intersection_ensemble(r1_h, r2_h_det, tb_hyb, dt, p_hyb_sm, pwf_3h, tp_hr, qo, pv)

        # Run full 9-algorithm suite (with strict PVT extraction!)
        vfm_hist = dw[dw["date"] <= day_shut][["date", "oil_rate_stbd"]].copy()
        suite = run_advanced_non_iarf_suite(
            hf_well=df_w,
            t_shut=t0,
            t_end=t1,
            qo_stbd=qo,
            pwf_psi=pwf_3h,
            tp_hr=tp_hr,
            pvt_row=pv.to_dict(),
            comp_row=cp.to_dict(),
            vfm_daily_well=vfm_hist,
            hyb_summary={
                "intersect_win_hr": [win["dt_start"], win["dt_end"]],
                "intersect_span_cyc": math.log10(win["dt_end"] / win["dt_start"]),
                "confidence_score": float(r3_h["confidence_score"]),
                "checks": r3_h.get("checks", {"plateau": "PASS"}),
                "logic3_intersection": {"k": props["k_md"], "skin": props["skin"], "m": props["m_psi_cycle"], "dP_skin": props["dP_skin"]},
            },
        )
        an = suite["analytical_match"]
        c_fit = float(an["c_bbl_psi"]) if an.get("available") else np.nan

        # Deliverable PTA row
        pta_main_rows.append({
            "event_id": ev_id,
            "well_name": w,
            "shut_in_time": str(t0),
            "shut_in_duration_hr": round(dur, 2),
            "q_oil_vfm_stbd": round(qo, 2),
            "tp_hr": round(tp_hr, 1),
            "pwf_psia": round(pwf_3h, 2),
            "iarf_window_hr": f"{win['dt_start']:.2f}-{win['dt_end']:.2f}",
            "horner_slope_m_psi_per_cycle": round(props["m_psi_cycle"], 3),
            "permeability_k_md": round(props["k_md"], 2),
            "skin_S": round(props["skin"], 3),
            "delta_p_skin_psi": round(props["dP_skin"], 2),
            "p_star_psia": round(p_star, 2),
            "horner_r2": round(r2_h, 5),
            "stehfest_k_md": round(an["k"], 2) if an.get("available") else np.nan,
            "stehfest_skin": round(an["skin"], 3) if an.get("available") else np.nan,
            "stehfest_C_bbl_psi": round(c_fit, 5) if np.isfinite(c_fit) else np.nan,
            "regime_status": suite["regime_label"],
        })

        # 9-method comparison row
        methods_dict = {
            "Stable Horner (L3)": (props["k_md"], props["skin"]),
            "Stehfest Analytical": (an.get("k"), an.get("skin")),
            "von Schroeter Deconv": (suite["deconvolution"].get("k"), suite["deconvolution"].get("skin")),
            "Ilk B-Spline Deconv": (suite["ilk_bspline_deconv"].get("k"), suite["ilk_bspline_deconv"].get("skin")),
            "Agarwal Equiv Time": (suite["agarwal_equiv"].get("k"), suite["agarwal_equiv"].get("skin")),
            "Blasingame dP_id": (suite["blasingame_integral"].get("k"), suite["blasingame_integral"].get("skin")),
            "Odeh-Jones MultiRate": (suite["odeh_jones_multirate"].get("k"), suite["odeh_jones_multirate"].get("skin")),
            "Tiab Direct Synth": (suite["tiab_tds"].get("k"), suite["tiab_tds"].get("skin")),
            "MDH + Muskat": (suite["mdh_muskat"].get("k"), suite["mdh_muskat"].get("skin")),
        }
        row9 = {"event_id": ev_id, "well_name": w, "duration_hr": round(dur, 2), "regime": suite["regime_label"]}
        for mname, (kv, sv) in methods_dict.items():
            row9[f"{mname} k (mD)"] = round(float(kv), 2) if kv is not None and np.isfinite(kv) else np.nan
            row9[f"{mname} S"] = round(float(sv), 3) if sv is not None and np.isfinite(sv) else np.nan
        pta_9m_rows.append(row9)

        # Tubing-ID Wellbore Storage Check
        tw = wt_raw[wt_raw["well_name"] == w]
        wc_med = float((tw["water_rate_stbd"] / np.maximum(tw["water_rate_stbd"] + tw["oil_rate_stbd"], 1.0)).median())
        api_med = float(tw["oil_gravity_api"].median())
        sal_med = float(tw["water_salinity_ppm"].median())
        rho_mix = (1.0 - wc_med) * oil_density_api(api_med) + wc_med * water_density_mccain(sal_med)
        c_tub = tubing_storage_estimate(float(cp["tubing_id_in"]), rho_mix)
        storage_rows.append({
            "event_id": ev_id,
            "well_name": w,
            "tubing_id_in": float(cp["tubing_id_in"]),
            "water_cut_med": round(wc_med, 3),
            "oil_api": round(api_med, 1),
            "salinity_ppm": round(sal_med, 0),
            "mixture_density_lb_ft3": round(rho_mix, 1),
            "C_tubing_level_est_bbl_psi": round(c_tub, 5),
            "C_stehfest_fit_bbl_psi": round(c_fit, 5) if np.isfinite(c_fit) else np.nan,
            "ratio_fit_over_est": round(c_fit / c_tub, 2) if np.isfinite(c_fit) and c_tub > 0 else np.nan,
        })

        # PTA Assumption Sensitivity (rate, pwf, tp, Perrine multiphase)
        Bo = float(pv["oil_fvf_rb_stb"])
        Bw = float(cp["water_fvf_rb_stb"])
        h_ft = float(pv["net_pay_thickness_ft"])
        T_res = float(pv["reservoir_temperature_degf"])
        mu_w = water_viscosity_mccain(T_res, pwf_3h, sal_med)
        tw_pre = tw[pd.to_datetime(tw["test_start"]) < t0]
        qo_wt = float(tw_pre["oil_rate_stbd"].iloc[-1]) if len(tw_pre) else qo
        for grp, vname, q_use, pwf_use, tp_use in [
            ("baseline", "Qo VFM last full day + 3h median pwf + 24*Np/q", qo, pwf_3h, tp_hr),
            ("rate", "Qo VFM 3-day mean", qo_3d, pwf_3h, tp_hr),
            ("rate", "Qo measured last well test", qo_wt, pwf_3h, tp_hr),
            ("rate", "Q_liquid = Qo + Qw (multiphase alt.)", qo + qw, pwf_3h, tp_hr),
            ("rate", "Q_res_total = (QoBo + QwBw)/Bo", qo + qw * Bw / Bo, pwf_3h, tp_hr),
            ("pwf", "pwf = mean last 3 flowing min", qo, pwf_3m, tp_hr),
            ("pwf", "pwf = mean last 15 flowing min", qo, pwf_15m, tp_hr),
            ("tp", f"tp = hours since dataset start ({data_start:%Y-%m-%d})", qo, pwf_3h, (t0 - data_start).total_seconds() / 3600.0),
            ("tp", "tp = hours since first_production_date", qo, pwf_3h, (t0 - pd.to_datetime(cp["first_production_date"])).total_seconds() / 3600.0),
        ]:
            x_v = np.log10((tp_use + dt[sel]) / dt[sel])
            m_v, ps_v, _ = _fit_line(x_v, pws_sm[sel])
            pr_v = _horner_props(m_v, ps_v, pwf_use, tp_use, q_use, pv)
            sens_rows.append({
                "event_id": ev_id, "well_name": w, "group": grp, "variant": vname,
                "q_used_stbd": round(q_use, 1), "horner_m": round(pr_v["m_psi_cycle"], 3),
                "k_md": round(pr_v["k_md"], 2), "skin": round(pr_v["skin"], 3),
                "dp_skin_psi": round(pr_v["dP_skin"], 2),
            })
        # Perrine (1956) multiphase row
        lam_ratio = (qo * Bo + qw * Bw) / max(qo * Bo, 1e-6)
        s_perrine = props["skin"] - 1.1513 * math.log10(lam_ratio)
        kw_eff = 162.6 * qw * Bw * mu_w / (props["m_psi_cycle"] * h_ft)
        sens_rows.append({
            "event_id": ev_id, "well_name": w, "group": "multiphase",
            "variant": f"Perrine (1956): ko={props['k_md']:.1f} mD, kw_eff={kw_eff:.1f} mD, lambda_t/lambda_o={lam_ratio:.2f}",
            "q_used_stbd": round(qo, 1), "horner_m": round(props["m_psi_cycle"], 3),
            "k_md": round(props["k_md"], 2), "skin": round(s_perrine, 3),
            "dp_skin_psi": round(0.869 * props["m_psi_cycle"] * s_perrine, 2),
        })

        # ------------------------------------------------------------------
        # GENERATE 5 HIGH-DPI PUBLICATION PLOTS FOR THIS SHUT-IN
        # ------------------------------------------------------------------
        hdr = f"{ev_id}  |  k = {props['k_md']:.1f} mD, S = {props['skin']:+.2f}, m = {props['m_psi_cycle']:.2f} psi/cyc, ΔP_skin = {props['dP_skin']:+.1f} psi"

        # 01: Horner Semilog Plot
        fig, ax = plt.subplots(figsize=(8.5, 5.2))
        ax.semilogx((tp_hr + dt) / dt, pws_raw, ".", ms=2.5, color=BLUE, alpha=0.75, label="Measured $p_{ws}$")
        xs = np.logspace(0, np.log10(float(np.max((tp_hr + dt) / dt))), 60)
        ax.semilogx(xs, p_star - m_h * np.log10(xs), "--", color=RED, lw=1.8, label=f"IARF Line ($m = {m_h:.2f}$ psi/cyc, $R^2={r2_h:.4f}$)")
        h1, h2 = (tp_hr + win["dt_end"]) / win["dt_end"], (tp_hr + win["dt_start"]) / win["dt_start"]
        ax.axvspan(h1, h2, color=GREEN, alpha=0.12, label=f"IARF Window [{win['dt_start']:.2f}–{win['dt_end']:.2f} hr]")
        ax.axhline(p_star, color=GREY, lw=0.9, ls=":")
        ax.text(2.5, p_star, f" $p^* = {p_star:.1f}$ psia", va="bottom", ha="right", fontsize=8.5, color="#334155")
        ax.invert_xaxis()
        ax.set_xlabel("Horner Time Ratio $(t_p + \\Delta t)/\\Delta t$")
        ax.set_ylabel("Shut-In Pressure $p_{ws}$ (psia)")
        ax.set_title("Horner Semilog Analysis\n" + hdr, fontsize=9.5, fontweight="bold")
        ax.legend(fontsize=8, loc="lower left")
        ax.grid(True, which="both", alpha=0.28)
        fig.savefig(ev_dir / "01_horner.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        # 02: Log-Log Diagnostic + Stehfest Analytical Model Overlay
        fig, ax = plt.subplots(figsize=(8.5, 5.2))
        dp_b = pb_sm - pwf_3h
        pos_dp = dp_b > 0
        pos_der = np.isfinite(der) & (der > 0)
        ax.loglog(tb[pos_dp], dp_b[pos_dp], "o", ms=4.2, color=BLUE, label="$\\Delta p = p_{ws} - p_{wf}$")
        ax.loglog(tb[pos_der], der[pos_der], "^", ms=4.5, color=RED, label="Bourdet Derivative $\\Delta p'$")
        if an.get("available") and an.get("curve_dp"):
            mc_dp = an["curve_dp"]
            mc_dr = an.get("curve_der") or []
            ax.loglog([pt["x"] for pt in mc_dp], [pt["y"] for pt in mc_dp], "-", color="#0f172a", lw=1.3, alpha=0.85, label="Stehfest Analytical $\\Delta p$")
            if mc_dr:
                ax.loglog([pt["x"] for pt in mc_dr], [pt["y"] for pt in mc_dr], "--", color=PURPLE, lw=1.6, label=f"Stehfest $\\Delta p'$ (k={an['k']:.1f} mD, S={an['skin']:+.2f})")
        ax.axhline(m_h / 2.303, color=GREEN, ls=":", lw=1.3, label=f"IARF Plateau $m/2.303 = {m_h/2.303:.2f}$ psi")
        t_us = tb[pos_dp][:5]
        if len(t_us) >= 2:
            ax.loglog(t_us, dp_b[pos_dp][0] * (t_us / t_us[0]), "-.", color=GREY, lw=1.0, label="Unit Slope (Wellbore Storage)")
        ax.axvspan(win["dt_start"], win["dt_end"], color=GREEN, alpha=0.10, label=f"IARF [{win['dt_start']:.2f}–{win['dt_end']:.2f} hr]")
        ax.set_xlabel("Shut-In Time $\\Delta t$ (hr)")
        ax.set_ylabel("$\\Delta p, \\; \\Delta p'$ (psi)")
        ax.set_title("Log-Log Diagnostic & Stehfest Type-Curve Match\n" + hdr, fontsize=9.5, fontweight="bold")
        ax.legend(fontsize=7.5, loc="lower right", ncol=2)
        ax.grid(True, which="both", alpha=0.28)
        fig.savefig(ev_dir / "02_loglog_derivative.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        overview_data.append((w, dur, tb, dp_b, der, an, win, m_h))

        # 03: MDH Semilog Plot
        fig, ax = plt.subplots(figsize=(8.5, 5.2))
        ax.semilogx(dt, pws_raw, ".", ms=2.5, color=BLUE, alpha=0.75, label="Measured $p_{ws}$")
        xs_dt = np.logspace(np.log10(max(dt.min(), 1e-3)), np.log10(dt.max()), 60)
        ax.semilogx(xs_dt, p_star - m_h * np.log10((tp_hr + xs_dt) / xs_dt), "--", color=RED, lw=1.8, label=f"Semilog Straight Line ($m={m_h:.2f}$)")
        ax.axvspan(win["dt_start"], win["dt_end"], color=GREEN, alpha=0.12, label=f"IARF [{win['dt_start']:.2f}–{win['dt_end']:.2f} hr]")
        ax.set_xlabel("Shut-In Time $\\Delta t$ (hr)")
        ax.set_ylabel("Shut-In Pressure $p_{ws}$ (psia)")
        ax.set_title("Miller-Dyes-Hutchinson (MDH) Semilog Plot\n" + hdr, fontsize=9.5, fontweight="bold")
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, which="both", alpha=0.28)
        fig.savefig(ev_dir / "03_mdh.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        # 04: Rate & Pressure History (+/- 3 days around shut-in)
        pad = pd.Timedelta(days=3)
        gw_hist = df_w[(df_w["timestamp"] >= t0 - pad) & (df_w["timestamp"] <= t1 + pad)].set_index("timestamp")["dh_pressure_psia"].resample("5min").mean()
        dw_hist = dw[(dw["date"] >= (t0 - pad).floor("D")) & (dw["date"] <= (t1 + pad).floor("D"))]
        fig, ax = plt.subplots(figsize=(9.2, 4.6))
        ax.plot(gw_hist.index, gw_hist.values, color=BLUE, lw=1.2, label="Downhole PDG Pressure (psia)")
        ax.axvspan(t0, t1, color=RED, alpha=0.10, label=f"Shut-In ({dur:.1f} hr)")
        ax.set_ylabel("Downhole Pressure (psia)", color=BLUE)
        ax2 = ax.twinx()
        ax2.bar(dw_hist["date"] + pd.Timedelta(hours=12), dw_hist["oil_rate_stbd"], width=0.75, color="#0f172a", alpha=0.22, label="VFM Daily Oil (STB/d)")
        ax2.set_ylabel("VFM Oil Rate (STB/d)", color="#334155")
        ax.set_title("Pre/Post Shut-In Pressure & VFM Oil Rate History\n" + hdr, fontsize=9.5, fontweight="bold")
        ax.grid(True, alpha=0.28)
        fig.autofmt_xdate()
        fig.savefig(ev_dir / "04_pressure_rate_history.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        # 05: 9-Algorithm Multi-Method Consensus Plot (Permeability k & Skin S)
        m_labels = list(methods_dict.keys())
        k_vals = [methods_dict[m][0] if methods_dict[m][0] is not None else np.nan for m in m_labels]
        s_vals = [methods_dict[m][1] if methods_dict[m][1] is not None else np.nan for m in m_labels]
        y_pos = np.arange(len(m_labels))
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.8), sharey=True)
        cols_bar = [GREEN if i == 0 else (PURPLE if i == 1 else BLUE) for i in range(len(m_labels))]
        ax1.barh(y_pos, k_vals, color=cols_bar, alpha=0.85)
        ax1.axvline(np.nanmedian(k_vals), color=RED, ls="--", lw=1.2, label=f"Median k = {np.nanmedian(k_vals):.1f} mD")
        for yi, kv in enumerate(k_vals):
            if np.isfinite(kv):
                ax1.text(kv, yi, f" {kv:.1f}", va="center", fontsize=8)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(m_labels, fontsize=8.5)
        ax1.invert_yaxis()
        ax1.set_xlabel("Permeability $k$ (mD)")
        ax1.set_title("Permeability Across 9 PTA Algorithms", fontsize=9.5, fontweight="bold")
        ax1.legend(fontsize=8)
        ax1.grid(True, axis="x", alpha=0.3)

        ax2.barh(y_pos, s_vals, color=cols_bar, alpha=0.85)
        ax2.axvline(np.nanmedian(s_vals), color=RED, ls="--", lw=1.2, label=f"Median S = {np.nanmedian(s_vals):+.2f}")
        for yi, sv in enumerate(s_vals):
            if np.isfinite(sv):
                ax2.text(sv, yi, f" {sv:+.2f}", va="center", fontsize=8)
        ax2.set_xlabel("Skin Factor $S$")
        ax2.set_title("Skin Factor Across 9 PTA Algorithms", fontsize=9.5, fontweight="bold")
        ax2.legend(fontsize=8)
        ax2.grid(True, axis="x", alpha=0.3)
        fig.suptitle(f"9-Algorithm Hidden-IARF & Deconvolution Consensus — {ev_id} ({suite['regime_label']})", fontsize=10.5, fontweight="bold")
        fig.tight_layout()
        fig.savefig(ev_dir / "05_9method_consensus.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Generated 5 PTA plots for {ev_id} -> k={props['k_md']:.1f} mD, S={props['skin']:+.2f} | C_fit={c_fit:.4f} vs C_tub={c_tub:.4f}")

        # Collect Qo Tornado sensitivity breakdown for this shut-in
        from vfm.diagnostics import compute_pta_qo_sensitivity
        qs_ev = compute_pta_qo_sensitivity(
            well=w,
            ev={"shut_in_start": t0, "shut_in_end": t1, "duration_hr": dur},
            hf_w=df_w,
            pvt_row=pv.to_dict(),
            comp_row=cp.to_dict(),
            tests_df=wt_raw,
            model_daily=all_model_daily,
            hyb={"qo_stbd": qo, "pwf_3h_med": pwf_3h, "logic3_intersection": {"m": props["m_psi_cycle"], "k": props["k_md"], "skin": props["skin"], "p_star": p_star}},
            adv=suite,
            horner_plot={"tp_hr": tp_hr, "m_psi_cycle": props["m_psi_cycle"], "p_star_psi": p_star, "win_hr": [win["dt_start"], win["dt_end"]]},
            qo_pct=20.0,
        )
        if qs_ev.get("available"):
            qo_tornado_panels.append((w, dur, qs_ev))
            for row4 in qs_ev["four_outputs_tornado"]:
                qo_tornado_rows.append({
                    "event_id": ev_id, "well_name": w, "base_qo_stbd": qs_ev["base"]["qo_stbd"],
                    "qo_perturbation_pct": "±20%",
                    "pta_output": row4["label"], "unit": row4["unit"],
                    "base_value": row4["base"],
                    "at_qo_minus_20pct": row4["at_qo_low"],
                    "at_qo_plus_20pct": row4["at_qo_high"],
                    "abs_delta_minus_20pct": row4["delta_at_qo_low"],
                    "abs_delta_plus_20pct": row4["delta_at_qo_high"],
                    "pct_change_minus_20pct": row4["pct_at_qo_low"],
                    "pct_change_plus_20pct": row4["pct_at_qo_high"],
                    "physics_sensitivity_law": row4["formula"],
                })

    # Field-wide 6-panel overview PNG
    fig, axes = plt.subplots(2, 3, figsize=(16, 9.5))
    for ax, (w, dur, tb, dp_b, der, an, win, m_h) in zip(axes.flat, overview_data):
        pos_dp = dp_b > 0
        pos_der = np.isfinite(der) & (der > 0)
        ax.loglog(tb[pos_dp], dp_b[pos_dp], "o", ms=3.2, color=BLUE, label="$\\Delta p$")
        ax.loglog(tb[pos_der], der[pos_der], "^", ms=3.5, color=RED, label="$\\Delta p'$")
        if an.get("available") and an.get("curve_der"):
            mc_dr = an["curve_der"]
            ax.loglog([p["x"] for p in mc_dr], [p["y"] for p in mc_dr], "--", color=PURPLE, lw=1.4, label="Stehfest")
        ax.axhline(m_h / 2.303, color=GREEN, ls=":", lw=1.1)
        ax.axvspan(win["dt_start"], win["dt_end"], color=GREEN, alpha=0.10)
        ax.set_title(f"{w} ({dur:.1f} hr) | k={an.get('k', np.nan):.1f} mD, S={an.get('skin', np.nan):+.2f}", fontsize=9.5, fontweight="bold")
        ax.set_xlabel("$\\Delta t$ (hr)")
        ax.set_ylabel("$\\Delta p, \\Delta p'$ (psi)")
        ax.grid(True, which="both", alpha=0.28)
        ax.legend(fontsize=7.5, loc="lower right")
    fig.suptitle("SOCAR Hackathon 2026 — Field-Wide Log-Log Diagnostic & Stehfest Analytical Match (All 6 Shut-Ins)", fontsize=12.5, fontweight="bold")
    fig.tight_layout()
    fig.savefig(plots_dir / "00_field_pta_overview.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Save PTA Deliverable CSVs & Multi-Sheet Excel Workbooks
    pta_main_df = pd.DataFrame(pta_main_rows)
    pta_9m_df = pd.DataFrame(pta_9m_rows)
    storage_df = pd.DataFrame(storage_rows)
    sens_df = pd.DataFrame(sens_rows)

    pta_deliv_cols = [
        "event_id", "well_name", "shut_in_time", "shut_in_duration_hr",
        "horner_slope_m_psi_per_cycle", "skin_S", "permeability_k_md", "delta_p_skin_psi",
    ]
    pta_main_df[pta_deliv_cols].to_csv(pta_dir / "pta_results.csv", index=False)
    pta_main_df.to_csv(pta_dir / "pta_results_full.csv", index=False)
    pta_9m_df.to_csv(pta_dir / "pta_9method_comparison.csv", index=False)

    with pd.ExcelWriter(pta_dir / "pta_results.xlsx", engine="openpyxl") as xw:
        pta_main_df[pta_deliv_cols].to_excel(xw, sheet_name="pta_results", index=False)
        _autofit_sheet(xw, "pta_results", pta_main_df[pta_deliv_cols])
        pta_main_df.to_excel(xw, sheet_name="pta_full_detail", index=False)
        _autofit_sheet(xw, "pta_full_detail", pta_main_df)
        pta_9m_df.to_excel(xw, sheet_name="9_algorithm_consensus", index=False)
        _autofit_sheet(xw, "9_algorithm_consensus", pta_9m_df)
        storage_df.to_excel(xw, sheet_name="storage_vs_tubing_ID", index=False)
        _autofit_sheet(xw, "storage_vs_tubing_ID", storage_df)

    with pd.ExcelWriter(pta_dir / "pta_assumption_sensitivity.xlsx", engine="openpyxl") as xw:
        sens_df.to_excel(xw, sheet_name="all_variants", index=False)
        _autofit_sheet(xw, "all_variants", sens_df)
        storage_df.to_excel(xw, sheet_name="storage_vs_tubing_ID", index=False)
        _autofit_sheet(xw, "storage_vs_tubing_ID", storage_df)
        if qo_tornado_rows:
            qo_torn_df = pd.DataFrame(qo_tornado_rows)
            qo_torn_df.to_excel(xw, sheet_name="Qo_Tornado_Sensitivity", index=False)
            _autofit_sheet(xw, "Qo_Tornado_Sensitivity", qo_torn_df)

    # Field-wide 6-panel PTA Qo (±20%) Sensitivity Tornado PNG — Exact 4 Required PTA Outputs
    if qo_tornado_panels:
        n_p = len(qo_tornado_panels)
        ncols = min(3, max(1, n_p))
        nrows = math.ceil(n_p / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(6.2 * ncols, 3.8 * nrows))
        axes_flat = np.atleast_1d(axes).ravel()
        for ax, (w, dur, qs_ev) in zip(axes_flat, qo_tornado_panels):
            rows4 = qs_ev["four_outputs_tornado"]
            y_pos = np.arange(len(rows4))
            lbls = [r["label"] for r in rows4]
            p_lo = [r["pct_at_qo_low"] for r in rows4]
            p_hi = [r["pct_at_qo_high"] for r in rows4]
            max_p = max(38.0, max(max(abs(v) for v in p_lo), max(abs(v) for v in p_hi)) + 18.0)
            ax.barh(y_pos, p_lo, color=BLUE, alpha=0.88, height=0.55, label="Qo -20%")
            ax.barh(y_pos, p_hi, color=RED, alpha=0.86, height=0.55, label="Qo +20%")
            ax.axvline(0.0, color="#111827", ls="--", lw=1.4)
            for yi, r in enumerate(rows4):
                if abs(r["pct_at_qo_low"]) < 0.005 and abs(r["pct_at_qo_high"]) < 0.005:
                    ax.text(0.8, yi, f"≈0.00% ({r['at_qo_low']:.3f}–{r['at_qo_high']:.3f})", va="center", ha="left", fontsize=7.3, color="#475569")
                else:
                    l_v, r_v = (r["pct_at_qo_low"], r["pct_at_qo_high"]) if r["pct_at_qo_low"] < r["pct_at_qo_high"] else (r["pct_at_qo_high"], r["pct_at_qo_low"])
                    l_a, r_a = (r["at_qo_low"], r["at_qo_high"]) if r["pct_at_qo_low"] < r["pct_at_qo_high"] else (r["at_qo_high"], r["at_qo_low"])
                    p_fmt = "+.2f" if abs(r_v - l_v) < 1.0 else "+.1f"
                    a_fmt = ".2f" if r["key"] in ("skin_S", "horner_slope_m") else ".1f"
                    ax.text(l_v - 0.8, yi, f"{l_v:{p_fmt}}% ({l_a:{a_fmt}})", va="center", ha="right", fontsize=7.3, color="#1d4ed8")
                    ax.text(r_v + 0.8, yi, f"{r_v:{p_fmt}}% ({r_a:{a_fmt}})", va="center", ha="left", fontsize=7.3, color="#b91c1c")
            ax.set_xlim(-max_p, max_p)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(lbls, fontsize=8.3, fontweight="bold")
            ax.invert_yaxis()
            ax.set_xlabel("% Change from Base Value", fontsize=8.5)
            ax.set_title(
                f"{w} ({dur:.0f}h) · Base Qo={qs_ev['base']['qo_stbd']:.0f} STB/d (±20%)",
                fontsize=9.2, fontweight="bold",
            )
            ax.grid(True, axis="x", alpha=0.28)
        for ax in axes_flat[n_p:]:
            ax.axis("off")
        handles, labels_l = axes_flat[0].get_legend_handles_labels()
        fig.legend(handles, labels_l, loc="upper right", fontsize=9.5)
        fig.suptitle("PTA Sensitivity Tornado: Impact of ±20% Oil Rate (Qo) Perturbation on the 4 Required PTA Outputs (k, S, ΔP_skin, m)", fontsize=12.0, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(plots_dir / "01_field_pta_qo_tornado_sensitivity.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved field PTA Qo (±20%) 4-output Tornado plot -> {plots_dir / '01_field_pta_qo_tornado_sensitivity.png'}")

    # Phase 1 VFM Diagnostics: Correlation Matrix Heatmap, Feature Audit & Exact SHAP Evaluation
    print("\n=== Building Phase 1 VFM Feature Diagnostics (Correlation Matrix & SHAP Evaluation) ===")
    from vfm.diagnostics import compute_vfm_feature_diagnostics, load_v2_bundle
    bundle_v2 = load_v2_bundle(pkl_src)
    # Use full all-well 7-sensor telemetry (hr_v2 covers every well in the field!)
    hr_by_well = {
        str(w): grp.rename(columns={"hour": "timestamp"}).copy()
        for w, grp in hr_v2.groupby("well_name")
    }
    vfm_diag = compute_vfm_feature_diagnostics(
        wt_raw, hr_by_well, comp_df.reset_index(), pvt_df.reset_index(), bundle_v2, method="pearson"
    )

    # Save VFM Correlation Matrix PNG (All Inputs + V2 Training Features side-by-side)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 9.5))
    for ax, scope_key, title in (
        (ax1, "all_inputs", "All Candidate VFM Inputs (SCADA + Diffs + PVT/Completion + Targets)"),
        (ax2, "v2_training", "Exact V2 Stacked Training Features (21 Inputs + Targets)"),
    ):
        cb = vfm_diag["correlations"][scope_key]
        mat = np.asarray(cb["matrix"], dtype=float)
        lbls = [c["label"] + (" *" if c["used_in_v2"] else "") for c in cb["columns"]]
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-1.0, vmax=1.0, aspect="auto")
        ax.set_xticks(np.arange(len(lbls)))
        ax.set_yticks(np.arange(len(lbls)))
        ax.set_xticklabels(lbls, rotation=45, ha="right", fontsize=7.5)
        ax.set_yticklabels(lbls, fontsize=7.5)
        for i in range(len(lbls)):
            for j in range(len(lbls)):
                v = mat[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.0, color="white" if abs(v) > 0.58 else "#182026")
        ax.set_title(title + "\n(* = Direct V2 ML Input)", fontsize=10.5, fontweight="bold")
    fig.colorbar(im, ax=[ax1, ax2], fraction=0.018, pad=0.02, label="Pearson Correlation r")
    fig.suptitle(f"VFM Input Correlation Matrix ({vfm_diag['n_tests']} Real Separator Tests, {vfm_diag['n_wells']} Wells)", fontsize=13, fontweight="bold")
    fig.savefig(vfm_dir / "vfm_correlation_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Save VFM SHAP Evaluation PNG (Top: Exact 2^8 Physical Sensor SHAP, Bottom: V2 Stacked 21-Feature SHAP)
    fig, axes = plt.subplots(2, 3, figsize=(18.5, 10.5))
    for col_i, (ph, unit) in enumerate((("oil", "STB/d"), ("water", "STB/d"), ("gas", "Mscf/d"))):
        for row_i, (lvl, lvl_lbl) in enumerate((
            ("end_to_end_sensors", "End-to-End Physical Sensors (Exact 2⁸ Shapley)"),
            ("stacked_v2_features", "Stacked V2 Super-Learner (Top 10 Training Features)"),
        )):
            ax = axes[row_i, col_i]
            blk = vfm_diag["shap"][ph][lvl]
            feats = blk["features"][:10]
            y_pos = np.arange(len(feats))
            vals = [f["mean_abs_shap"] for f in feats]
            cols_b = [RED if f["direction"] == "positive" else (BLUE if f["direction"] == "negative" else GREY) for f in feats]
            ax.barh(y_pos, vals, color=cols_b, alpha=0.86)
            for yi, f in enumerate(feats):
                ax.text(f["mean_abs_shap"], yi, f" {f['mean_abs_shap']:.1f} ({f['pct_share']:.0f}%)", va="center", fontsize=7.5)
            ax.set_yticks(y_pos)
            ax.set_yticklabels([f["label"] for f in feats], fontsize=8.0)
            ax.invert_yaxis()
            ax.set_xlabel(f"Mean |SHAP| ({unit})", fontsize=8.5)
            ax.set_title(f"{ph.upper()} — {lvl_lbl}", fontsize=9.2, fontweight="bold")
            ax.grid(True, axis="x", alpha=0.28)
    fig.suptitle("VFM SHAP Feature Evaluation (Red = Positive Effect · Blue = Negative Effect · Gray = Mixed)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(vfm_dir / "vfm_shap_evaluation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Save VFM Feature Diagnostics Excel Workbook
    with pd.ExcelWriter(vfm_dir / "vfm_feature_diagnostics.xlsx", engine="openpyxl") as xw:
        audit_df = pd.DataFrame(vfm_diag["feature_audit"])
        audit_df.to_excel(xw, sheet_name="Training_Feature_Audit", index=False)
        _autofit_sheet(xw, "Training_Feature_Audit", audit_df)
        for scope_key, sname in (("all_inputs", "Corr_All_Inputs"), ("v2_training", "Corr_V2_Features")):
            cb = vfm_diag["correlations"][scope_key]
            c_lbls = [c["key"] for c in cb["columns"]]
            cdf = pd.DataFrame(cb["matrix"], index=c_lbls, columns=c_lbls).reset_index().rename(columns={"index": "feature"})
            cdf.to_excel(xw, sheet_name=sname, index=False)
            _autofit_sheet(xw, sname, cdf)
        for lvl, sname in (("end_to_end_sensors", "SHAP_EndToEnd_Sensors"), ("stacked_v2_features", "SHAP_Stacked_V2"), ("lgbm_treeshap", "SHAP_LGBM_TreeSHAP")):
            s_rows = []
            for ph in ("oil", "water", "gas"):
                blk = vfm_diag["shap"][ph][lvl]
                for rank, f in enumerate(blk["features"], 1):
                    s_rows.append({
                        "phase": ph, "unit": blk["unit"], "rank": rank,
                        "feature": f["key"], "label": f["label"], "group": f["group"],
                        "mean_abs_shap": f["mean_abs_shap"], "pct_share": f["pct_share"],
                        "direction": f["direction"], "direction_corr": f["direction_corr"],
                        "min_shap": f["min_shap"], "max_shap": f["max_shap"],
                    })
            sdf = pd.DataFrame(s_rows)
            sdf.to_excel(xw, sheet_name=sname, index=False)
            _autofit_sheet(xw, sname, sdf)
    print(f"  Saved VFM correlation matrix, SHAP plots & diagnostics Excel -> {vfm_dir}")

    print("\n=== TUBING-ID WELLBORE STORAGE VALIDATION ===")
    print(storage_df[["event_id", "tubing_id_in", "mixture_density_lb_ft3", "C_tubing_level_est_bbl_psi", "C_stehfest_fit_bbl_psi", "ratio_fit_over_est"]].to_string(index=False))
    print(f"\nALL PHASE 1, PHASE 2, MODEL & PLOT DELIVERABLES SAVED TO: {DELIV_DIR}")


if __name__ == "__main__":
    main()
