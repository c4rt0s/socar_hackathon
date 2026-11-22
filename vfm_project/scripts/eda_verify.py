"""Empirical EDA and physical verification across cached HF telemetry and well tests."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Dict, List

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vfm.config import load_config
from vfm.io import load_frame, load_static_tables

SENSOR_COLS = [
    "whp_psia",
    "whp_downstream_psia",
    "dh_pressure_psia",
    "dh_temperature_degf",
    "annulus_pressure_psia",
    "gas_lift_rate_mscfd",
]


def _pearson(x: pd.Series, y: pd.Series) -> float:
    mask = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    xs, ys = x[mask].to_numpy(dtype=float), y[mask].to_numpy(dtype=float)
    if np.std(xs) == 0 or np.std(ys) == 0:
        return float("nan")
    return float(np.corrcoef(xs, ys)[0, 1])


def _spearman(x: pd.Series, y: pd.Series) -> float:
    mask = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    return _pearson(x[mask].rank(), y[mask].rank())


def _linear_slope_per_30d(t: pd.Series, y: pd.Series) -> tuple[float, float, float]:
    """Return (slope_per_30d, start_val, end_val) from OLS over timestamps."""
    mask = t.notna() & y.notna() & np.isfinite(y)
    if mask.sum() < 2:
        return float("nan"), float("nan"), float("nan")
    t_days = (t[mask] - t[mask].min()).dt.total_seconds().to_numpy(dtype=float) / 86400.0
    yv = y[mask].to_numpy(dtype=float)
    if np.std(t_days) == 0:
        return float("nan"), float(yv[0]), float(yv[-1])
    slope_day, intercept = np.polyfit(t_days, yv, 1)
    return float(slope_day * 30.0), float(yv[0]), float(yv[-1])


def estimate_hydrostatic_grad_psi_ft(api: float, wc_frac: float, salinity_ppm: float) -> float:
    """Approximate wellbore liquid gradient (psi/ft) from oil API, water cut, salinity."""
    sg_oil = 141.5 / (131.5 + max(api, 5.0))
    sg_water = 1.0 + 0.7e-6 * max(salinity_ppm, 0.0)
    sg_mix = (1.0 - wc_frac) * sg_oil + wc_frac * sg_water
    return 0.433 * sg_mix


def main() -> None:
    cfg = load_config()
    cache_dir = cfg.resolve_path("cache_dir")
    eda_dir = cfg.resolve_path("outputs_dir") / "eda"
    eda_dir.mkdir(parents=True, exist_ok=True)

    completion, pvt, choke_events = load_static_tables(
        cfg.resolve_path("well_completion_csv"),
        cfg.resolve_path("reservoir_pvt_csv"),
        cfg.resolve_path("choke_events_csv"),
        schema=cfg.schema,
    )
    wt = pd.read_csv(cache_dir / "well_tests_harmonized.csv")
    wt["test_start"] = pd.to_datetime(wt["test_start"], utc=True)
    wt["test_end"] = pd.to_datetime(wt["test_end"], utc=True)

    wells = sorted(completion["well_name"].unique())
    hf_by_well: Dict[str, pd.DataFrame] = {}
    for w in wells:
        df = load_frame(cache_dir / "wells" / w)
        df = df.sort_values("timestamp").reset_index(drop=True)
        hf_by_well[w] = df

    # Precompute per-well median fluid gradient from tests
    well_grad = {}
    for w in wells:
        sub = wt[wt["well_name"] == w]
        api = float(sub["oil_gravity_api"].median())
        wc = float(sub["water_cut_pct"].median()) / 100.0
        sal = float(sub["water_salinity_ppm"].median())
        well_grad[w] = estimate_hydrostatic_grad_psi_ft(api, wc, sal)

    # ------------------------------------------------------------------
    # 1. HF vs Well-Test Sensor Alignment & 7. Steady Coverage per Test
    # ------------------------------------------------------------------
    align_rows: List[Dict[str, Any]] = []
    cov_rows: List[Dict[str, Any]] = []

    for idx, row in wt.iterrows():
        w = row["well_name"]
        t0, t1 = row["test_start"], row["test_end"]
        hf = hf_by_well[w]
        win = hf[(hf["timestamp"] >= t0) & (hf["timestamp"] <= t1)]
        steady = win[win["is_steady_flowing"]] if len(win) else win

        n_win = int(len(win))
        n_steady = int(len(steady))
        n_shut = int(win["is_shut_in"].sum()) if n_win else 0
        n_trans = int(win["is_transient"].sum()) if n_win else 0
        expected_mins = int(round((t1 - t0).total_seconds() / 60.0)) + 1
        cov = float(n_steady / max(n_win, 1))

        cov_rows.append({
            "well_name": w,
            "test_start": t0,
            "test_end": t1,
            "duration_hours": round((t1 - t0).total_seconds() / 3600.0, 2),
            "expected_rows_inclusive": expected_mins,
            "n_window_rows": n_win,
            "n_steady_rows": n_steady,
            "n_shut_in_rows": n_shut,
            "n_transient_rows": n_trans,
            "steady_coverage": round(cov, 4),
            "passes_qc_70pct": bool(n_steady > 0 and cov >= 0.70),
        })

        rec: Dict[str, Any] = {
            "well_name": w,
            "test_start": t0,
            "test_end": t1,
            "n_window_rows": n_win,
            "n_steady_rows": n_steady,
            "steady_coverage": round(cov, 4),
        }
        src = steady if n_steady > 0 else win
        for col in SENSOR_COLS:
            wt_val = float(row[col])
            hf_mean_all = float(win[col].mean()) if n_win else np.nan
            hf_med_all = float(win[col].median()) if n_win else np.nan
            hf_mean_st = float(src[col].mean()) if len(src) else np.nan
            hf_med_st = float(src[col].median()) if len(src) else np.nan
            rec[f"{col}_wt"] = wt_val
            rec[f"{col}_hf_mean_all"] = round(hf_mean_all, 4)
            rec[f"{col}_hf_med_all"] = round(hf_med_all, 4)
            rec[f"{col}_hf_mean_steady"] = round(hf_mean_st, 4)
            rec[f"{col}_hf_med_steady"] = round(hf_med_st, 4)
            rec[f"{col}_diff_mean_all"] = round(hf_mean_all - wt_val, 4)
            rec[f"{col}_diff_med_steady"] = round(hf_med_st - wt_val, 4)
        align_rows.append(rec)

    align_df = pd.DataFrame(align_rows)
    cov_df = pd.DataFrame(cov_rows)
    align_df.to_csv(eda_dir / "sensor_alignment_per_test.csv", index=False)
    cov_df.to_csv(eda_dir / "steady_coverage_per_test.csv", index=False)

    align_sum_rows = []
    for col in SENSOR_COLS:
        d_all = align_df[f"{col}_diff_mean_all"]
        d_st = align_df[f"{col}_diff_med_steady"]
        align_sum_rows.append({
            "sensor": col,
            "n_tests": int(d_all.notna().sum()),
            "mae_hf_mean_all_vs_wt": round(float(d_all.abs().mean()), 4),
            "max_abs_hf_mean_all_vs_wt": round(float(d_all.abs().max()), 4),
            "bias_hf_mean_all_vs_wt": round(float(d_all.mean()), 4),
            "mae_hf_med_steady_vs_wt": round(float(d_st.abs().mean()), 4),
            "max_abs_hf_med_steady_vs_wt": round(float(d_st.abs().max()), 4),
            "corr_wt_vs_hf_mean_all": round(_pearson(align_df[f"{col}_wt"], align_df[f"{col}_hf_mean_all"]), 6),
        })
    align_sum_df = pd.DataFrame(align_sum_rows)
    align_sum_df.to_csv(eda_dir / "sensor_alignment_summary.csv", index=False)

    # Check if a +/- hour shift improves or degrades whp_psia alignment (timezone sanity check)
    tz_rows = []
    for shift_h in (-6, -4, -1, 0, 1, 4, 6):
        errs = []
        delta = pd.Timedelta(hours=shift_h)
        for _, row in wt.iterrows():
            hf = hf_by_well[row["well_name"]]
            win = hf[(hf["timestamp"] >= row["test_start"] + delta) & (hf["timestamp"] <= row["test_end"] + delta)]
            if len(win):
                errs.append(abs(float(win["whp_psia"].mean()) - float(row["whp_psia"])))
        tz_rows.append({
            "shift_hours": shift_h,
            "whp_mae_psia": round(float(np.mean(errs)), 4) if errs else np.nan,
            "whp_max_err_psia": round(float(np.max(errs)), 4) if errs else np.nan,
        })
    pd.DataFrame(tz_rows).to_csv(eda_dir / "timezone_shift_check.csv", index=False)

    # ------------------------------------------------------------------
    # 2. Pwf vs Pb per well & 3. Choke Ratio Regime & 8. Gas Lift vs Annulus
    # ------------------------------------------------------------------
    pwf_rows = []
    choke_rows = []
    gl_rows = []

    for w in wells:
        hf = hf_by_well[w]
        comp_r = completion[completion["well_name"] == w].iloc[0]
        pvt_r = pvt[pvt["well_name"] == w].iloc[0]

        pb = float(pvt_r["bubble_point_pressure_psia"])
        gauge_tvd = float(comp_r["gauge_tvd_ft"])
        datum_tvd = float(pvt_r["datum_tvd_ft"])
        grad = well_grad[w]
        datum_offset = (datum_tvd - gauge_tvd) * grad

        flowing = hf[~hf["is_shut_in"]]
        steady = hf[hf["is_steady_flowing"]]
        shut_in = hf[hf["is_shut_in"] & (hf["choke_opening_pct"] <= 0.5)]

        dh_all = hf["dh_pressure_psia"].dropna()
        dh_flow = flowing["dh_pressure_psia"].dropna()
        dh_flow_datum = dh_flow + datum_offset

        pwf_rows.append({
            "well_name": w,
            "bubble_point_psia": pb,
            "gauge_tvd_ft": gauge_tvd,
            "datum_tvd_ft": datum_tvd,
            "est_fluid_grad_psi_ft": round(grad, 4),
            "datum_offset_psi": round(datum_offset, 2),
            "flowing_dh_min_psia": round(float(dh_flow.min()), 2),
            "flowing_dh_p01_psia": round(float(dh_flow.quantile(0.01)), 2),
            "flowing_dh_p05_psia": round(float(dh_flow.quantile(0.05)), 2),
            "flowing_dh_median_psia": round(float(dh_flow.median()), 2),
            "flowing_dh_p95_psia": round(float(dh_flow.quantile(0.95)), 2),
            "flowing_dh_max_psia": round(float(dh_flow.max()), 2),
            "all_dh_max_psia": round(float(dh_all.max()), 2),
            "shut_in_dh_max_psia": round(float(shut_in["dh_pressure_psia"].max()), 2) if len(shut_in) else np.nan,
            "flowing_datum_min_psia": round(float(dh_flow_datum.min()), 2),
            "flowing_datum_median_psia": round(float(dh_flow_datum.median()), 2),
            "min_margin_above_pb_gauge_psi": round(float(dh_flow.min() - pb), 2),
            "min_margin_above_pb_datum_psi": round(float(dh_flow_datum.min() - pb), 2),
            "pct_flowing_below_pb": round(float((dh_flow < pb).mean() * 100.0), 4),
        })

        # Choke ratio regime (steady flowing)
        cr = steady["choke_pressure_ratio"].dropna()
        chk_pct = steady["choke_opening_pct"].dropna()
        dp_chk = steady["dp_choke_psi"].dropna()
        choke_rows.append({
            "well_name": w,
            "n_steady_mins": int(len(steady)),
            "choke_pct_min": round(float(chk_pct.min()), 2),
            "choke_pct_median": round(float(chk_pct.median()), 2),
            "choke_pct_max": round(float(chk_pct.max()), 2),
            "dp_choke_min_psi": round(float(dp_chk.min()), 2),
            "dp_choke_median_psi": round(float(dp_chk.median()), 2),
            "dp_choke_max_psi": round(float(dp_chk.max()), 2),
            "ratio_min": round(float(cr.min()), 4),
            "ratio_p05": round(float(cr.quantile(0.05)), 4),
            "ratio_median": round(float(cr.median()), 4),
            "ratio_p95": round(float(cr.quantile(0.95)), 4),
            "ratio_max": round(float(cr.max()), 4),
            "pct_subcritical_gt_055": round(float((cr > 0.55).mean() * 100.0), 2),
            "pct_subcritical_gt_053": round(float((cr > 0.53).mean() * 100.0), 2),
            "pct_critical_le_055": round(float((cr <= 0.55).mean() * 100.0), 2),
        })

        # Gas lift vs annulus pressure (steady flowing)
        gl = steady["gas_lift_rate_mscfd"]
        dp_ann_whp = steady["annulus_pressure_psia"] - steady["whp_psia"]
        # Gas density / hydrostatic proxy in annulus (sg=0.65 -> ~0.015-0.025 psi/ft at ~1000 psi)
        gl_tvd = float(comp_r["gl_valve_tvd_ft"])
        port_64 = float(comp_r["gl_valve_port_64ths"])
        tub_grad_est = (steady["dh_pressure_psia"] - steady["whp_psia"]) / gauge_tvd
        p_tub_valve_est = steady["whp_psia"] + tub_grad_est * gl_tvd
        # Static gas column approx: P_ann_valve ~ P_ann * exp(0.01875 * sg * tvd / (z * T_avg))
        # Use T_avg ~ 140 F (600 R), z ~ 0.9, sg = 0.65 -> factor ~ 0.01875*0.65/(0.9*600) = 2.257e-5 / ft
        p_ann_valve_est = steady["annulus_pressure_psia"] * np.exp(2.26e-5 * gl_tvd)
        dp_valve_est = p_ann_valve_est - p_tub_valve_est

        gl_rows.append({
            "well_name": w,
            "gl_valve_tvd_ft": gl_tvd,
            "gl_valve_port_64ths": port_64,
            "gl_rate_min_mscfd": round(float(gl.min()), 2),
            "gl_rate_median_mscfd": round(float(gl.median()), 2),
            "gl_rate_max_mscfd": round(float(gl.max()), 2),
            "annulus_p_median_psia": round(float(steady["annulus_pressure_psia"].median()), 2),
            "dp_annulus_minus_whp_median_psi": round(float(dp_ann_whp.median()), 2),
            "dp_valve_est_median_psi": round(float(dp_valve_est.median()), 2),
            "pct_dp_valve_est_positive": round(float((dp_valve_est > 0).mean() * 100.0), 2),
            "corr_gl_vs_dp_ann_whp": round(_pearson(gl, dp_ann_whp), 4),
            "corr_gl_vs_annulus_p": round(_pearson(gl, steady["annulus_pressure_psia"]), 4),
            "corr_gl_vs_sqrt_dp_valve": round(_pearson(gl, np.sqrt(dp_valve_est.clip(lower=0.0))), 4),
            "whp_std_15m_median": round(float(steady["whp_std_15m"].median()), 3),
            "whp_std_15m_p95": round(float(steady["whp_std_15m"].quantile(0.95)), 3),
            "annulus_std_15m_median": round(float(steady["annulus_std_15m"].median()), 3),
            "annulus_std_15m_p95": round(float(steady["annulus_std_15m"].quantile(0.95)), 3),
        })

    pd.DataFrame(pwf_rows).to_csv(eda_dir / "pwf_vs_pb_per_well.csv", index=False)
    pd.DataFrame(choke_rows).to_csv(eda_dir / "choke_regime_per_well.csv", index=False)
    pd.DataFrame(gl_rows).to_csv(eda_dir / "gas_lift_annulus_per_well.csv", index=False)

    # ------------------------------------------------------------------
    # 4. GOR vs Rs & Gas Mass Balance
    # ------------------------------------------------------------------
    gor_test_rows = []
    gor_well_rows = []
    wt_m = wt.merge(completion[["well_name", "solution_gor_at_pb_scf_stb", "choke_max_size_64ths"]], on="well_name", how="left")
    wt_m["expected_form_gas_mscfd"] = wt_m["oil_rate_stbd"] * wt_m["gor_scf_stb"] / 1000.0
    wt_m["expected_total_gas_mscfd"] = wt_m["expected_form_gas_mscfd"] + wt_m["gas_lift_rate_mscfd"]
    wt_m["gas_balance_err_mscfd"] = wt_m["total_gas_rate_mscfd"] - wt_m["expected_total_gas_mscfd"]
    wt_m["gor_minus_rs"] = wt_m["gor_scf_stb"] - wt_m["solution_gor_at_pb_scf_stb"]
    wt_m["gor_to_rs_ratio"] = wt_m["gor_scf_stb"] / wt_m["solution_gor_at_pb_scf_stb"]
    wt_m[[
        "well_name", "test_start", "oil_rate_stbd", "gor_scf_stb", "solution_gor_at_pb_scf_stb",
        "gor_minus_rs", "gor_to_rs_ratio", "gas_rate_mscfd", "expected_form_gas_mscfd",
        "gas_lift_rate_mscfd", "total_gas_rate_mscfd", "expected_total_gas_mscfd", "gas_balance_err_mscfd"
    ]].to_csv(eda_dir / "gor_gas_balance_per_test.csv", index=False)

    for w, grp in wt_m.groupby("well_name"):
        rs = float(grp["solution_gor_at_pb_scf_stb"].iloc[0])
        slope_gor, _, _ = _linear_slope_per_30d(grp["test_start"], grp["gor_scf_stb"])
        gor_well_rows.append({
            "well_name": w,
            "n_tests": int(len(grp)),
            "rs_pb_scf_stb": rs,
            "gor_mean_scf_stb": round(float(grp["gor_scf_stb"].mean()), 2),
            "gor_median_scf_stb": round(float(grp["gor_scf_stb"].median()), 2),
            "gor_std_scf_stb": round(float(grp["gor_scf_stb"].std()), 2),
            "gor_min_scf_stb": round(float(grp["gor_scf_stb"].min()), 2),
            "gor_max_scf_stb": round(float(grp["gor_scf_stb"].max()), 2),
            "mean_gor_to_rs_ratio": round(float(grp["gor_to_rs_ratio"].mean()), 4),
            "gor_slope_scf_stb_per_30d": round(slope_gor, 2),
            "max_abs_gas_balance_err_mscfd": round(float(grp["gas_balance_err_mscfd"].abs().max()), 3),
        })
    pd.DataFrame(gor_well_rows).to_csv(eda_dir / "gor_vs_rs_per_well.csv", index=False)

    # ------------------------------------------------------------------
    # 5. Temporal Trends & 6. Physical Correlations
    # ------------------------------------------------------------------
    # Attach steady HF features to each test for physical correlations & PI proxy
    test_feat_rows = []
    for _, row in wt.iterrows():
        w = row["well_name"]
        hf = hf_by_well[w]
        t0, t1 = row["test_start"], row["test_end"]
        win = hf[(hf["timestamp"] >= t0) & (hf["timestamp"] <= t1)]
        steady = win[win["is_steady_flowing"]]
        src = steady if len(steady) > 0 else win

        choke_pct = float(src["choke_opening_pct"].median())
        choke_max_64 = float(src["choke_max_size_64ths"].iloc[0])
        d_inch = (choke_pct / 100.0) * (choke_max_64 / 64.0)
        dp_choke = float(src["dp_choke_psi"].median())
        whp = float(src["whp_psia"].median())
        dh_p = float(src["dh_pressure_psia"].median())
        dp_tub = float(src["dp_tubing_psi"].median())
        gl_rate = float(src["gas_lift_rate_mscfd"].median())

        test_feat_rows.append({
            **row.to_dict(),
            "choke_opening_pct": choke_pct,
            "choke_d_in": d_inch,
            "d1_sqrt_dp_choke": d_inch * np.sqrt(max(dp_choke, 0.0)),
            "d2_sqrt_dp_choke": (d_inch ** 2) * np.sqrt(max(dp_choke, 0.0)),
            "d2_sqrt_whp_dp_choke": (d_inch ** 2) * np.sqrt(max(whp * dp_choke, 0.0)),
            "choke_flow_proxy_hf": float(src["choke_flow_proxy"].median()),
            "dp_choke_psi": dp_choke,
            "dp_tubing_psi": dp_tub,
            "hf_whp_psia": whp,
            "hf_dh_pressure_psia": dh_p,
            "hf_gl_rate_mscfd": gl_rate,
        })
    tf_df = pd.DataFrame(test_feat_rows)

    trend_rows = []
    corr_rows = []

    for w in wells:
        sub = tf_df[tf_df["well_name"] == w].sort_values("test_start").reset_index(drop=True)
        hf = hf_by_well[w]
        steady = hf[hf["is_steady_flowing"]]
        shut_in = hf[hf["is_shut_in"] & (hf["choke_opening_pct"] <= 0.5)]

        # Shut-in max buildup P (if any)
        p_shutin_max = float(shut_in["dh_pressure_psia"].max()) if len(shut_in) else np.nan
        # Proxy Pr: if shut-in exists, use shut-in max as rough Pr anchor; else max(dh_p) + 500
        # Also compute implied PI proxy from linear regression of q_liq vs dh_p: q = J*Pr - J*Pwf -> slope = -J
        dh_slope_hf, dh_start_hf, dh_end_hf = _linear_slope_per_30d(steady["timestamp"], steady["dh_pressure_psia"])
        wc_slope, wc_start, wc_end = _linear_slope_per_30d(sub["test_start"], sub["water_cut_pct"])
        qliq_slope, qliq_start, qliq_end = _linear_slope_per_30d(sub["test_start"], sub["liquid_rate_stbd"])
        qoil_slope, qoil_start, qoil_end = _linear_slope_per_30d(sub["test_start"], sub["oil_rate_stbd"])
        whp_slope, _, _ = _linear_slope_per_30d(steady["timestamp"], steady["whp_psia"])

        # Implied PI proxy if Pr anchor exists: q_liq / (p_shutin_max - dh_p)
        if np.isfinite(p_shutin_max):
            dd_proxy = (p_shutin_max - sub["hf_dh_pressure_psia"]).clip(lower=10.0)
            pi_proxy = sub["liquid_rate_stbd"] / dd_proxy
            pi_slope, pi_start, pi_end = _linear_slope_per_30d(sub["test_start"], pi_proxy)
            pi_mean = float(pi_proxy.mean())
        else:
            # Use nominal Pr proxy = max test dh_p + 400 psi just for relative trend check
            pr_nom = float(sub["hf_dh_pressure_psia"].max()) + 400.0
            dd_proxy = (pr_nom - sub["hf_dh_pressure_psia"]).clip(lower=10.0)
            pi_proxy = sub["liquid_rate_stbd"] / dd_proxy
            pi_slope, pi_start, pi_end = _linear_slope_per_30d(sub["test_start"], pi_proxy)
            pi_mean = float(pi_proxy.mean())

        trend_rows.append({
            "well_name": w,
            "n_tests": int(len(sub)),
            "wc_start_pct": round(wc_start, 2),
            "wc_end_pct": round(wc_end, 2),
            "wc_change_pct_pts": round(wc_end - wc_start, 2),
            "wc_slope_pct_per_30d": round(wc_slope, 3),
            "dh_p_start_hf_psia": round(dh_start_hf, 1),
            "dh_p_end_hf_psia": round(dh_end_hf, 1),
            "dh_p_slope_psi_per_30d": round(dh_slope_hf, 2),
            "whp_slope_psi_per_30d": round(whp_slope, 2),
            "qliq_start_stbd": round(qliq_start, 1),
            "qliq_end_stbd": round(qliq_end, 1),
            "qliq_slope_stbd_per_30d": round(qliq_slope, 2),
            "qoil_start_stbd": round(qoil_start, 1),
            "qoil_end_stbd": round(qoil_end, 1),
            "qoil_slope_stbd_per_30d": round(qoil_slope, 2),
            "shut_in_max_dh_p_psia": round(p_shutin_max, 2) if np.isfinite(p_shutin_max) else np.nan,
            "pi_proxy_mean_stbd_psi": round(pi_mean, 3),
            "pi_proxy_slope_per_30d": round(pi_slope, 4),
        })

        # Physical correlations against liquid_rate_stbd
        q = sub["liquid_rate_stbd"]
        corr_rows.append({
            "well_name": w,
            "n_tests": int(len(sub)),
            "r_qliq_vs_d1_sqrt_dp_choke": round(_pearson(q, sub["d1_sqrt_dp_choke"]), 4),
            "r_qliq_vs_d2_sqrt_dp_choke": round(_pearson(q, sub["d2_sqrt_dp_choke"]), 4),
            "r_qliq_vs_d2_sqrt_whp_dp": round(_pearson(q, sub["d2_sqrt_whp_dp_choke"]), 4),
            "rho_qliq_vs_d2_sqrt_dp_choke": round(_spearman(q, sub["d2_sqrt_dp_choke"]), 4),
            "r_qliq_vs_choke_flow_proxy": round(_pearson(q, sub["choke_flow_proxy_hf"]), 4),
            "r_qliq_vs_dh_pressure": round(_pearson(q, sub["hf_dh_pressure_psia"]), 4),
            "r_qliq_vs_drawdown_proxy": round(_pearson(q, dd_proxy), 4),
            "rho_qliq_vs_drawdown_proxy": round(_spearman(q, dd_proxy), 4),
            "r_qliq_vs_dp_tubing": round(_pearson(q, sub["dp_tubing_psi"]), 4),
            "rho_qliq_vs_dp_tubing": round(_spearman(q, sub["dp_tubing_psi"]), 4),
            "r_qliq_vs_whp": round(_pearson(q, sub["hf_whp_psia"]), 4),
            "r_qliq_vs_choke_pct": round(_pearson(q, sub["choke_opening_pct"]), 4),
            "r_qliq_vs_gl_rate": round(_pearson(q, sub["hf_gl_rate_mscfd"]), 4),
        })

    pd.DataFrame(trend_rows).to_csv(eda_dir / "temporal_trends_per_well.csv", index=False)
    pd.DataFrame(corr_rows).to_csv(eda_dir / "physical_correlations_per_well.csv", index=False)

    # Print concise summary to stdout
    print("=== 1. SENSOR ALIGNMENT SUMMARY ===")
    print(align_sum_df.to_string(index=False))
    print("\n=== TIMEZONE SHIFT CHECK (WHP MAE) ===")
    print(pd.DataFrame(tz_rows).to_string(index=False))
    print("\n=== 2. PWF vs PB PER WELL ===")
    print(pd.DataFrame(pwf_rows)[["well_name", "bubble_point_psia", "flowing_dh_min_psia", "flowing_dh_median_psia", "flowing_dh_max_psia", "shut_in_dh_max_psia", "datum_offset_psi", "min_margin_above_pb_gauge_psi"]].to_string(index=False))
    print("\n=== 3. CHOKE REGIME PER WELL ===")
    print(pd.DataFrame(choke_rows)[["well_name", "choke_pct_min", "choke_pct_max", "ratio_p05", "ratio_median", "ratio_p95", "pct_subcritical_gt_055", "pct_critical_le_055"]].to_string(index=False))
    print("\n=== 4. GOR vs RS PER WELL ===")
    print(pd.DataFrame(gor_well_rows).to_string(index=False))
    print("\n=== 5. TEMPORAL TRENDS PER WELL ===")
    print(pd.DataFrame(trend_rows)[["well_name", "wc_start_pct", "wc_end_pct", "wc_slope_pct_per_30d", "dh_p_start_hf_psia", "dh_p_end_hf_psia", "dh_p_slope_psi_per_30d", "qliq_start_stbd", "qliq_end_stbd", "shut_in_max_dh_p_psia"]].to_string(index=False))
    print("\n=== 6. PHYSICAL CORRELATIONS PER WELL ===")
    print(pd.DataFrame(corr_rows)[["well_name", "r_qliq_vs_d2_sqrt_dp_choke", "r_qliq_vs_dh_pressure", "r_qliq_vs_drawdown_proxy", "r_qliq_vs_dp_tubing", "r_qliq_vs_choke_pct"]].to_string(index=False))
    print("\n=== 7. STEADY COVERAGE TESTS < 0.99 ===")
    low_cov = cov_df[cov_df["steady_coverage"] < 0.99]
    print(f"Total tests: {len(cov_df)}, passing >=0.70: {int(cov_df['passes_qc_70pct'].sum())}, <0.99 steady coverage: {len(low_cov)}")
    if not low_cov.empty:
        print(low_cov.to_string(index=False))
    print("\n=== 8. GAS LIFT vs ANNULUS PER WELL ===")
    print(pd.DataFrame(gl_rows)[["well_name", "gl_valve_port_64ths", "gl_rate_median_mscfd", "dp_annulus_minus_whp_median_psi", "dp_valve_est_median_psi", "corr_gl_vs_dp_ann_whp", "corr_gl_vs_annulus_p", "whp_std_15m_p95", "annulus_std_15m_p95"]].to_string(index=False))


if __name__ == "__main__":
    main()
