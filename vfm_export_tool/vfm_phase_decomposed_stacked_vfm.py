#!/usr/bin/env python3
"""Refined Phase-Decomposed Stacked Grey-Box VFM with 5 Targeted Adjustments:

  1. Well-Specific SCADA Operating Envelope + Manifold-vs-Separator Backpressure
     Counterfactuals in the Few-Shot Synthetic Generator (eliminating synthetic
     choke overshoot and explicitly teaching the -21 psi test-separator shift).
  2. Volumetric Material-Balance Depletion Rate (dPr/dt from `reservoir_pvt.csv`
     drainage area A, h, phi, ct) + Observed Shut-In Pressure Anchoring for P_r(t).
  3. Robust Per-Hour & Per-Test Median Aggregation (with per-block MAD outlier
     rejection) instead of raw arithmetic mean on 1-minute SCADA data.
  4. [Kept standard choke scaling — Adjustment 4 intentionally omitted per request.]
  5. Soft Phase-Sum Coherence Reconciliation: trains a 4th stacked super-learner
     for Total Liquid Rate Q_liq = f_liq(X) and reconciles independent (Q_o, Q_w)
     using an OOF-tuned reconciliation factor lambda_rec in [0, 1].
  6. Controlled Real-vs-Synthetic Total Weight Budget (total synthetic weight
     normalized to a calibrated fraction of total real test weight) + OOF-tuned
     SVR/MLP regularization per phase.
"""

from __future__ import annotations

import json
import math
import os
import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.optimize import minimize
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

warnings.filterwarnings("ignore")

OUT_DIR = Path(__file__).resolve().parent
REPO_ROOT = OUT_DIR.parent


def resolve_raw_dir(cli_path: str | Path | None = None) -> Path:
    """Resolve raw dataset directory dynamically without hardcoded user paths."""
    candidates: list[Path] = []
    if cli_path:
        candidates.append(Path(cli_path))
    env_dir = os.environ.get("VFM_DATA_DIR")
    if env_dir:
        candidates.append(Path(env_dir))
    cfg_path = REPO_ROOT / "vfm_project" / "configs" / "default.json"
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            rd = (cfg.get("paths") or {}).get("raw_data_dir")
            if rd:
                p = Path(rd)
                candidates.append(p if p.is_absolute() else (cfg_path.parent.parent / p))
        except Exception:
            pass
    candidates.extend([
        REPO_ROOT / "data" / "raw",
        REPO_ROOT / "vfm_project" / "data" / "raw",
        Path.home() / "Desktop" / "SOCAR HACKATHON",
    ])
    for cand in candidates:
        if cand.exists() and (cand / "well_tests.csv").exists():
            return cand
    raise FileNotFoundError(
        "Raw dataset folder not found. Pass --data <path> or set VFM_DATA_DIR to the folder containing well_tests.csv."
    )


def find_cum_oil_column(comp: pd.DataFrame) -> str | None:
    """Detect initial cumulative-oil column dynamically (regardless of dataset reference year)."""
    for col in comp.columns:
        low = str(col).lower()
        if "cum_oil" in low or "cumulative_oil" in low:
            return str(col)
    return None


def compute_producing_time_hr(
    well: str,
    t_shut: pd.Timestamp,
    qo_stbd: float,
    comp: pd.DataFrame,
    cum_since_start_stb: float,
    dataset_start: pd.Timestamp | None = None,
) -> float:
    """Compute Horner producing time t_p (hr) strictly from dataset tables (no hardcoded constants)."""
    qo = max(float(qo_stbd), 1.0)
    cum_col = find_cum_oil_column(comp)
    if cum_col is not None and well in comp.index:
        val = pd.to_numeric(comp.at[well, cum_col], errors="coerce")
        if np.isfinite(val) and float(val) > 0.0:
            return float(24.0 * (float(val) + max(cum_since_start_stb, 0.0)) / qo)
    if "first_production_date" in comp.columns and well in comp.index:
        fp = pd.to_datetime(comp.at[well, "first_production_date"], errors="coerce")
        if pd.notna(fp):
            t0_n = t_shut.tz_localize(None) if t_shut.tzinfo is not None else t_shut
            fp_n = fp.tz_localize(None) if fp.tzinfo is not None else fp
            hrs = (t0_n - fp_n).total_seconds() / 3600.0
            if hrs > 24.0:
                return float(hrs)
    if dataset_start is not None:
        t0_n = t_shut.tz_localize(None) if t_shut.tzinfo is not None else t_shut
        ds_n = dataset_start.tz_localize(None) if dataset_start.tzinfo is not None else dataset_start
        return float(max((t0_n - ds_n).total_seconds() / 3600.0, 24.0))
    return float(max(24.0 * cum_since_start_stb / qo, 24.0))


R_CRIT = 0.55
SENSOR_COLS = [
    "whp_psia",
    "whp_downstream_psia",
    "dh_pressure_psia",
    "dh_temperature_degf",
    "annulus_pressure_psia",
    "gas_lift_rate_mscfd",
]


# ==============================================================================
# 1. Adjustment 3: Robust Median 1-Minute SCADA Streaming (Exact Windows + Hourly)
# ==============================================================================

def load_harmonized_well_tests(raw_dir: Path | None = None, ref_date: pd.Timestamp | None = None) -> pd.DataFrame:
    """Load well_tests.csv, repair sign-flips & Sm3/d -> Mscf/d unit mismatch."""
    raw_dir = resolve_raw_dir(raw_dir)
    wt = pd.read_csv(raw_dir / "well_tests.csv")
    for col in (
        "gas_lift_rate_mscfd",
        "oil_rate_stbd",
        "water_rate_stbd",
        "gas_rate_mscfd",
        "total_gas_rate_mscfd",
        "liquid_rate_stbd",
    ):
        if col in wt.columns:
            wt[col] = wt[col].abs()

    if {"oil_rate_stbd", "gor_scf_stb", "gas_rate_mscfd"} <= set(wt.columns):
        expected_gas = (wt["oil_rate_stbd"] * wt["gor_scf_stb"]) / 1000.0
        ratio = wt["gas_rate_mscfd"] / expected_gas.replace(0.0, np.nan)
        for w, grp in wt.groupby("well_name"):
            if float(ratio.loc[grp.index].median()) > 15.0:
                m = wt["well_name"] == w
                wt.loc[m, "gas_rate_mscfd"] = wt.loc[m, "gas_rate_mscfd"] / 28.3168466
                if "total_gas_rate_mscfd" in wt.columns:
                    wt.loc[m, "total_gas_rate_mscfd"] = wt.loc[m, "total_gas_rate_mscfd"] / 28.3168466

    wt["test_start"] = pd.to_datetime(wt["test_start"])
    wt["test_end"] = pd.to_datetime(wt["test_end"])
    wt["test_mid"] = wt["test_start"] + (wt["test_end"] - wt["test_start"]) / 2
    t_ref = pd.Timestamp(ref_date) if ref_date is not None else wt["test_start"].min().floor("D")
    wt["days_since_start"] = (wt["test_mid"] - t_ref).dt.total_seconds() / 86400.0
    return wt.sort_values(["well_name", "test_start"]).reset_index(drop=True)


def _robust_trimmed_median(arr: np.ndarray) -> float:
    """Reject 5-MAD outliers and return median of clean readings."""
    v = arr[np.isfinite(arr)]
    if len(v) == 0:
        return np.nan
    med = float(np.median(v))
    if len(v) < 5:
        return med
    mad = float(np.median(np.abs(v - med)))
    if mad > 1e-6:
        clean = v[np.abs(v - med) <= 5.0 * 1.4826 * mad]
        if len(clean) > 0:
            return float(np.median(clean))
    return med


def stream_raw_scada_robust_median(
    wt: pd.DataFrame, ce: pd.DataFrame, raw_dir: Path | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict], pd.Timestamp]:
    """Stream raw high_freq_data.csv using robust per-hour & per-test medians (Adjustment 3),
    plus extract per-well SCADA operating envelopes & shut-in pressures (Adjustments 1 & 2).
    """
    raw_dir = resolve_raw_dir(raw_dir)
    wt_cache = OUT_DIR / "well_tests_exact_7sensor_scada_robust.csv"
    hr_cache = OUT_DIR / "hourly_full_7sensor_telemetry_robust.csv"
    env_cache = OUT_DIR / "well_scada_envelopes_and_shutin_pr.csv"

    if wt_cache.exists() and hr_cache.exists() and env_cache.exists():
        print("Loading cached robust-median SCADA test windows, hourly telemetry & envelopes...")
        wt_scada = pd.read_csv(wt_cache, parse_dates=["test_start", "test_end", "test_mid"])
        hr_df = pd.read_csv(hr_cache, parse_dates=["hour", "date"])
        env_df = pd.read_csv(env_cache).set_index("well_name")
        ref_date = pd.Timestamp(min(hr_df["hour"].min(), wt_scada["test_start"].min())).floor("D")
        wt_scada["days_since_start"] = (wt_scada["test_mid"] - ref_date).dt.total_seconds() / 86400.0
        hr_df["days_since_start"] = (hr_df["hour"] - ref_date).dt.total_seconds() / 86400.0 + (0.5 / 24.0)
        return wt_scada, hr_df, env_df.to_dict(orient="index"), ref_date

    print("Streaming raw high_freq_data.csv (all 7 sensors) with robust per-hour & per-test medians...")
    ce_sorted = ce.sort_values(["timestamp", "well_name"]).copy()

    wt_windows_by_well: dict[str, list[tuple[int, pd.Timestamp, pd.Timestamp]]] = {}
    for idx, r in wt.iterrows():
        wt_windows_by_well.setdefault(r["well_name"], []).append((idx, r["test_start"], r["test_end"]))

    # Collect per-test raw values for exact robust median over [test_start, test_end]
    wt_vals: dict[int, dict[str, list[np.ndarray]]] = {
        idx: {c: [] for c in SENSOR_COLS + ["choke_opening_pct"]} for idx in wt.index
    }

    # Collect hourly chunks (each chunk covers contiguous time, so we aggregate per-hour medians)
    hourly_parts = []
    shutin_pmax_ts: dict[str, list[tuple[pd.Timestamp, float]]] = {w: [] for w in ce["well_name"].unique()}

    usecols = ["timestamp", "well_name"] + SENSOR_COLS
    for chunk in pd.read_csv(raw_dir / "high_freq_data.csv", usecols=usecols, chunksize=600_000):
        chunk["timestamp"] = pd.to_datetime(chunk["timestamp"])
        for col in SENSOR_COLS:
            chunk[col] = chunk[col].replace([-999.25, -999.0], np.nan)
            if col == "gas_lift_rate_mscfd":
                chunk.loc[chunk[col] < 0, col] = 0.0
            elif col == "dh_temperature_degf":
                chunk.loc[(chunk[col] < 40.0) | (chunk[col] > 400.0), col] = np.nan
            else:
                chunk.loc[chunk[col] <= 0.0, col] = np.nan

        merged_wells = []
        for w, g in chunk.groupby("well_name"):
            g = g.sort_values("timestamp")
            ce_w = ce_sorted[ce_sorted["well_name"] == w][["timestamp", "choke_opening_pct"]].sort_values("timestamp")
            m = pd.merge_asof(g, ce_w, on="timestamp", direction="backward")
            m["choke_opening_pct"] = m["choke_opening_pct"].bfill().fillna(0.0)

            # Record shut-in high pressures for P_r(t) anchoring (Adjustment 2)
            si_sub = m[m["choke_opening_pct"] <= 0.5]
            if len(si_sub) >= 60:
                p98 = float(si_sub["dh_pressure_psia"].quantile(0.98))
                if np.isfinite(p98) and p98 > 500.0:
                    shutin_pmax_ts.setdefault(w, []).append((pd.Timestamp(si_sub["timestamp"].iloc[-1]), p98))

            # Collect exact test-window samples
            if w in wt_windows_by_well:
                tmin, tmax = m["timestamp"].iloc[0], m["timestamp"].iloc[-1]
                for idx, t0, t1 in wt_windows_by_well[w]:
                    if tmax < t0 or tmin > t1:
                        continue
                    sub = m[(m["timestamp"] >= t0) & (m["timestamp"] <= t1) & (m["choke_opening_pct"] > 0.5)]
                    if sub.empty:
                        continue
                    for c in SENSOR_COLS + ["choke_opening_pct"]:
                        v = sub[c].dropna().to_numpy(dtype=float)
                        if len(v) > 0:
                            wt_vals[idx][c].append(v)
            merged_wells.append(m)

        m_all = pd.concat(merged_wells, ignore_index=True)
        m_all["hour"] = m_all["timestamp"].dt.floor("h")
        m_all["is_flowing"] = (m_all["choke_opening_pct"] > 0.5).astype(int)

        tot_grp = m_all.groupby(["well_name", "hour"], as_index=False).agg(
            n_total=("is_flowing", "count"),
            n_flowing=("is_flowing", "sum"),
        )
        fl = m_all[m_all["is_flowing"] == 1]
        # Per-hour MEDIAN (robust against 1-minute SCADA spikes & transients!)
        med_cols = SENSOR_COLS + ["choke_opening_pct"]
        fl_grp = fl.groupby(["well_name", "hour"], as_index=False)[med_cols].median()
        hourly_parts.append(pd.merge(tot_grp, fl_grp, on=["well_name", "hour"], how="left"))

    # Combine hourly chunks and derive dataset time span & reference date strictly from data
    hr_full = pd.concat(hourly_parts, ignore_index=True)
    agg_ops = {"n_total": "sum", "n_flowing": "sum"}
    for c in SENSOR_COLS + ["choke_opening_pct"]:
        agg_ops[c] = "median"
    hr_agg = hr_full.groupby(["well_name", "hour"], as_index=False).agg(agg_ops)

    t_start_grid = pd.Timestamp(hr_agg["hour"].min()).floor("D")
    t_end_grid = pd.Timestamp(hr_agg["hour"].max()).floor("h")
    ref_date = pd.Timestamp(min(t_start_grid, wt["test_start"].min().floor("D")))

    # Finalize exact robust median well-test sensors
    wt_out = wt.copy()
    wt_out["days_since_start"] = (wt_out["test_mid"] - ref_date).dt.total_seconds() / 86400.0
    for idx in wt_out.index:
        for c in SENSOR_COLS + ["choke_opening_pct"]:
            if wt_vals[idx][c]:
                all_v = np.concatenate(wt_vals[idx][c])
                val = _robust_trimmed_median(all_v)
                if np.isfinite(val):
                    wt_out.at[idx, c] = val
    wt_out.to_csv(wt_cache, index=False)

    all_hours = pd.date_range(t_start_grid, t_end_grid, freq="h")
    wells = sorted(set(hr_agg["well_name"].unique()) | set(ce["well_name"].unique()))
    grid = pd.MultiIndex.from_product([wells, all_hours], names=["well_name", "hour"]).to_frame(index=False)
    hr = pd.merge(grid, hr_agg, on=["well_name", "hour"], how="left")

    imputed_wells = []
    envelopes = {}
    for w in wells:
        sub = hr[hr["well_name"] == w].sort_values("hour").copy()
        ce_w = ce_sorted[ce_sorted["well_name"] == w][["timestamp", "choke_opening_pct"]].sort_values("timestamp")
        sub_choke = pd.merge_asof(
            sub[["hour"]].rename(columns={"hour": "timestamp"}), ce_w, on="timestamp", direction="backward"
        )["choke_opening_pct"].bfill().fillna(0.0).to_numpy()

        missing_scada = sub["n_total"].fillna(0) == 0
        sub["choke_opening_pct"] = np.where(sub["choke_opening_pct"].isna(), sub_choke, sub["choke_opening_pct"])
        sub["uptime_frac"] = np.where(
            missing_scada,
            np.where(sub["choke_opening_pct"] > 0.5, 1.0, 0.0),
            np.clip(sub["n_flowing"].fillna(0) / np.maximum(sub["n_total"].fillna(1), 1), 0.0, 1.0),
        )
        # Apply gentle 5-hour rolling median despike for extreme isolated hourly glitches before linear imputation
        for c in SENSOR_COLS:
            s = sub[c]
            med5 = s.rolling(5, center=True, min_periods=2).median()
            mad5 = (s - med5).abs().rolling(5, center=True, min_periods=2).median()
            floor = 10.0 if "pressure" in c or "whp" in c else (1.5 if "temp" in c else 35.0)
            spike = (s - med5).abs() > np.maximum(6.0 * 1.4826 * mad5, floor)
            s = s.mask(spike, np.nan)
            sub[c] = s.interpolate(method="linear").bfill().ffill()

        # Extract real SCADA operating envelope (flowing hours only) for Adjustment 1 & shut-in P_r for Adjustment 2
        fl_sub = sub[sub["choke_opening_pct"] > 0.5]
        si_pts = shutin_pmax_ts.get(w, [])
        if si_pts:
            best_ts, si_pr = max(si_pts, key=lambda x: x[1])
            si_day = float((best_ts - ref_date).total_seconds() / 86400.0)
        else:
            si_pr, si_day = np.nan, np.nan

        envelopes[w] = {
            "choke_p02": float(fl_sub["choke_opening_pct"].quantile(0.02)),
            "choke_p98": float(fl_sub["choke_opening_pct"].quantile(0.98)),
            "ds_p02": float(fl_sub["whp_downstream_psia"].quantile(0.02)),
            "ds_p98": float(fl_sub["whp_downstream_psia"].quantile(0.98)),
            "gl_p05": float(fl_sub["gas_lift_rate_mscfd"].quantile(0.05)),
            "gl_p95": float(fl_sub["gas_lift_rate_mscfd"].quantile(0.95)),
            "shutin_pmax_psi": si_pr,
            "shutin_pmax_day": si_day,
        }
        imputed_wells.append(sub)

    hr_clean = pd.concat(imputed_wells, ignore_index=True)
    hr_clean["date"] = hr_clean["hour"].dt.floor("D")
    hr_clean["days_since_start"] = (hr_clean["hour"] - ref_date).dt.total_seconds() / 86400.0 + (0.5 / 24.0)
    keep_cols = ["well_name", "hour", "date", "days_since_start", "uptime_frac", "choke_opening_pct"] + SENSOR_COLS
    hr_clean = hr_clean[keep_cols]
    hr_clean.to_csv(hr_cache, index=False)

    env_df = pd.DataFrame.from_dict(envelopes, orient="index")
    env_df.index.name = "well_name"
    env_df.to_csv(env_cache)
    print(f"Saved robust-median SCADA ({len(wt_out)} tests, {len(hr_clean)} well-hours, envelopes for {len(envelopes)} wells).")
    return wt_out, hr_clean, envelopes, ref_date


# ==============================================================================
# 2. Adjustment 2: Volumetric Material-Balance Depletion & Shut-In P_r Anchoring
# ==============================================================================

def choke_bean_diameter_in(choke_pct: np.ndarray | float, max_64ths: np.ndarray | float = 128.0) -> np.ndarray:
    c = np.clip(np.asarray(choke_pct, dtype=float) / 100.0, 0.0, 1.0)
    d_max = np.asarray(max_64ths, dtype=float) / 64.0
    return d_max * np.where(c <= 0.005, 0.0, 0.15 * c + 0.85 * (c ** 1.15))


def effective_choke_dp(whp: np.ndarray | float, ds: np.ndarray | float, r_crit: float = R_CRIT) -> np.ndarray:
    w = np.asarray(whp, dtype=float)
    d = np.asarray(ds, dtype=float)
    return np.maximum(w - np.maximum(d, r_crit * w), 0.0)


def calibrate_well_anchors(
    wt: pd.DataFrame, comp: pd.DataFrame, pvt: pd.DataFrame, envelopes: dict[str, dict]
) -> dict[str, dict]:
    """Calibrate per-well anchors using volumetric material-balance dPr/dt & shut-in P_r (Adjustment 2)."""
    wells = sorted(comp.index)
    anchors: dict[str, dict] = {}

    # Data-driven field priors (no hardcoded dataset slopes or hydrostatic gradients)
    wc_slopes: list[float] = []
    hydro_grads: list[float] = []
    jt_slopes: list[float] = []
    for w in wells:
        sub = wt[wt["well_name"] == w].sort_values("days_since_start")
        if len(sub) >= 3:
            t_w = sub["days_since_start"].to_numpy()
            wc_w = np.clip(sub["water_cut_pct"].to_numpy() / 100.0, 0.01, 0.99)
            wc_slopes.append(float(np.polyfit(t_w, np.log(wc_w / (1.0 - wc_w)), 1)[0]))
            dhp_w = sub["dh_pressure_psia"].to_numpy(dtype=float)
            dht_w = sub["dh_temperature_degf"].to_numpy(dtype=float)
            if np.std(dhp_w) > 15.0 and np.std(dht_w) > 0.05:
                slp_jt = float(np.polyfit(dhp_w, dht_w, 1)[0])
                if -0.015 <= slp_jt <= 0.0:
                    jt_slopes.append(slp_jt)
        si_p_w = envelopes.get(w, {}).get("shutin_pmax_psi", np.nan)
        if np.isfinite(si_p_w) and w in pvt.index:
            dtvd = float(pvt.at[w, "datum_tvd_ft"])
            if dtvd > 0:
                hydro_grads.append(float(si_p_w) / dtvd)

    global_wc_slope = float(np.median(wc_slopes)) if wc_slopes else 0.0
    if hydro_grads:
        field_hydro_grad = float(np.median(hydro_grads))
    elif "oil_gravity_api" in wt.columns:
        sg_oil = 141.5 / (131.5 + float(wt["oil_gravity_api"].median()))
        wc_f = float(np.clip(wt["water_cut_pct"].median() / 100.0, 0.0, 1.0)) if "water_cut_pct" in wt.columns else 0.3
        field_hydro_grad = 0.433 * ((1.0 - wc_f) * sg_oil + wc_f * 1.05)
    else:
        field_hydro_grad = 0.433 * 0.90  # standard oilfield freshwater/oil gradient fallback
    field_jt_slope = float(np.clip(np.median(jt_slopes), -0.006, -0.0005)) if jt_slopes else 0.0

    for idx, w in enumerate(wells):
        sub = wt[wt["well_name"] == w].sort_values("days_since_start")
        c_row = comp.loc[w]
        p_row = pvt.loc[w]
        env = envelopes[w]

        t = sub["days_since_start"].to_numpy()
        q_liq = sub["liquid_rate_stbd"].to_numpy(dtype=float)
        q_oil = sub["oil_rate_stbd"].to_numpy(dtype=float)
        q_wat = sub["water_rate_stbd"].to_numpy(dtype=float)
        dhp = sub["dh_pressure_psia"].to_numpy(dtype=float)
        dht = sub["dh_temperature_degf"].to_numpy(dtype=float)
        whp = sub["whp_psia"].to_numpy(dtype=float)
        ds = sub["whp_downstream_psia"].to_numpy(dtype=float)
        ann = sub["annulus_pressure_psia"].to_numpy(dtype=float)
        gl = sub["gas_lift_rate_mscfd"].to_numpy(dtype=float)
        choke = sub["choke_opening_pct"].to_numpy(dtype=float)

        # 1) Choke orifice calibration (unchanged — Adjustment 4 intentionally omitted)
        d_in = choke_bean_diameter_in(choke, float(c_row["choke_max_size_64ths"]))
        area = np.pi / 4.0 * (d_in ** 2)
        dp_eff = effective_choke_dp(whp, ds)
        cap = np.maximum(area * np.sqrt(np.maximum(dp_eff, 0.0)), 1e-4)
        cd_med = float(np.median(q_liq / cap))

        # 2) Adjustment 2: Volumetric Material-Balance Depletion Rate dPr/dt (psi/day)
        #    Under active waterflood/aquifer support (water cut rising 20%-55%), net voidage
        #    depletion scales with q_res / (7758 * A * h * phi * ct) (~3% net voidage imbalance),
        #    producing physically distinct per-well depletion rates instead of hitting a flat clamp.
        A_acres = float(p_row["estimated_drainage_area_acres"])
        h_ft = float(p_row["net_pay_thickness_ft"])
        phi = float(p_row["porosity_frac"])
        ct = float(p_row["total_compressibility_1_psi"])
        bo = float(p_row["oil_fvf_rb_stb"])
        bw = float(c_row["water_fvf_rb_stb"])
        q_res_rbd = float(np.median(q_oil) * bo + np.median(q_wat) * bw)
        pore_vol_rb = 7758.0 * A_acres * h_ft * phi
        dpr_dt_pss = -0.025 * q_res_rbd / max(pore_vol_rb * ct, 1e-3)
        dpr_dt = float(np.clip(dpr_dt_pss, -0.45, -0.03))

        # Anchor Pr(0) using observed shut-in PBU maximum pressure when available;
        # for wells without shut-ins (e.g. GL-07), calibrate Pr(0) via regularized IPR fit
        si_pr = env.get("shutin_pmax_psi", np.nan)
        si_day = env.get("shutin_pmax_day", np.nan)
        pr_hydro = field_hydro_grad * float(p_row["datum_tvd_ft"])
        pb = float(p_row["bubble_point_pressure_psia"])
        if np.isfinite(si_pr) and np.isfinite(si_day) and si_pr > float(np.max(dhp)) + 30.0:
            # Back-project observed shut-in static pressure to Jan 1, 2025 (+20 psi margin for finite shut-in)
            pr0 = max(float(si_pr) + 20.0 - dpr_dt * float(si_day), float(np.max(dhp)) + 150.0)
        else:
            pr0_prior = max(float(np.max(dhp)) + 280.0, pr_hydro)
            pr_grid = np.linspace(float(np.max(dhp)) + 100.0, max(float(np.max(dhp)) + 750.0, pr_hydro + 250.0), 45)
            best_pr0, best_obj = pr0_prior, 1e18
            for cand_pr0 in pr_grid:
                pr_c = cand_pr0 + dpr_dt * t
                dd_c = np.where(
                    dhp >= pb,
                    np.maximum(pr_c - dhp, 20.0),
                    np.maximum(pr_c - pb, 0.0) + (pb / 1.8) * (1.0 - 0.2 * (dhp / pb) - 0.8 * ((dhp / pb) ** 2)),
                )
                pi_c = float(np.median(q_liq / np.maximum(dd_c, 10.0)))
                rel_err = float(np.mean(((pi_c * dd_c - q_liq) / np.maximum(q_liq, 10.0)) ** 2))
                prior_pen = 0.002 * (((cand_pr0 - pr0_prior) / 300.0) ** 2)
                if rel_err + prior_pen < best_obj:
                    best_obj = rel_err + prior_pen
                    best_pr0 = float(cand_pr0)
            pr0 = best_pr0

        pr_t = pr0 + dpr_dt * t
        eff_dd = np.where(
            dhp >= pb,
            np.maximum(pr_t - dhp, 20.0),
            np.maximum(pr_t - pb, 0.0) + (pb / 1.8) * (1.0 - 0.2 * (dhp / pb) - 0.8 * ((dhp / pb) ** 2)),
        )
        pi_med = float(np.median(q_liq / np.maximum(eff_dd, 10.0)))

        # 3) Two-Segment Completion VLP calibration (with standardized scales so Joule-Thomson T_dh contributes cleanly)
        gauge_tvd = float(c_row["gauge_tvd_ft"])
        glv_tvd = float(c_row["gl_valve_tvd_ft"])
        tub_id = float(c_row["tubing_id_in"])
        tub_area = math.pi / 4.0 * (tub_id ** 2)

        grad_eff = (dhp - whp) / gauge_tvd
        grad_ref = float(np.median(grad_eff))
        gl_ref = float(np.median(gl))
        dht_ref = float(np.median(dht))
        ann_ref = float(np.median(ann))
        vlp_base = float(np.median(q_liq))

        A_vlp_raw = np.column_stack([
            (grad_eff - grad_ref) * gauge_tvd,
            gl - gl_ref,
            ann - ann_ref,
            (dht - dht_ref) * 10.0,
        ])
        vlp_scales = np.maximum(np.std(A_vlp_raw, axis=0), np.array([15.0, 25.0, 10.0, 0.5]))
        A_vlp_std = A_vlp_raw / vlp_scales
        reg_vlp = Ridge(alpha=3.0).fit(A_vlp_std, q_liq - vlp_base)
        vlp_coefs = (reg_vlp.coef_ / vlp_scales).astype(float)

        q_c_hat = cd_med * cap
        q_i_hat = pi_med * eff_dd
        q_v_hat = vlp_base + A_vlp_raw @ vlp_coefs
        var_c = max(float(np.mean(((q_c_hat - q_liq) / q_liq) ** 2)), 2.5e-4)
        var_i = max(float(np.mean(((q_i_hat - q_liq) / q_liq) ** 2)), 2.5e-4)
        var_v = max(float(np.mean(((q_v_hat - q_liq) / q_liq) ** 2)), 2.5e-4)
        inv_sum = (1.0 / var_c) + (1.0 / var_i) + (1.0 / var_v)
        w_ch_raw = (1.0 / var_c) / inv_sum
        w_ip_raw = (1.0 / var_i) / inv_sum
        w_vl_raw = (1.0 / var_v) / inv_sum
        # Cap choke weight at 0.45 so subcritical manifold-vs-separator P_ds shifts stay balanced with IPR & VLP
        if w_ch_raw > 0.45:
            rem = 1.0 - 0.45
            other = max(w_ip_raw + w_vl_raw, 1e-6)
            w_ch_raw = 0.45
            w_ip_raw = rem * (w_ip_raw / other)
            w_vl_raw = rem * (w_vl_raw / other)

        # 4) Smooth regularized Water Cut & GOR trajectories
        wc = np.clip(sub["water_cut_pct"].to_numpy(dtype=float) / 100.0, 0.005, 0.995)
        logit_wc = np.log(wc / (1.0 - wc))
        t_mean = float(np.mean(t))
        if len(t) >= 2:
            raw_slope = float(np.polyfit(t - t_mean, logit_wc, 1)[0])
            wc_b1 = 0.70 * raw_slope + 0.30 * global_wc_slope
        else:
            wc_b1 = global_wc_slope
        wc_b0 = float(np.median(logit_wc - wc_b1 * (t - t_mean)))

        gor_tests = sub["gor_scf_stb"].to_numpy(dtype=float)
        rs = float(c_row["solution_gor_at_pb_scf_stb"])
        gor_med = float(0.90 * np.median(gor_tests) + 0.10 * rs)
        gor_slope = float(np.polyfit(t - t_mean, gor_tests, 1)[0]) * 0.25 if len(t) >= 3 else 0.0

        anchors[w] = {
            "well_idx": idx,
            "cd": cd_med,
            "pi": pi_med,
            "pr0": pr0,
            "dpr_dt": dpr_dt,
            "gauge_tvd": gauge_tvd,
            "glv_tvd": glv_tvd,
            "tub_id": tub_id,
            "tub_area": tub_area,
            "grad_ref": grad_ref,
            "gl_ref": gl_ref,
            "dht_ref": dht_ref,
            "ann_ref": ann_ref,
            "vlp_base": vlp_base,
            "vlp_coefs": vlp_coefs,
            "w_choke": w_ch_raw,
            "w_ipr": w_ip_raw,
            "w_vlp": w_vl_raw,
            "wc_t_mean": t_mean,
            "wc_b0": wc_b0,
            "wc_b1": wc_b1,
            "gor_med": gor_med,
            "gor_slope": gor_slope,
            "jt_slope": field_jt_slope,
        }
    return anchors


def build_rich_feature_matrix(
    df: pd.DataFrame, comp: pd.DataFrame, pvt: pd.DataFrame, anchors: dict[str, dict]
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Vectorized builder of 20 physics + 7-sensor + two-segment completion features."""
    wells = df["well_name"].to_numpy()
    choke = df["choke_opening_pct"].to_numpy(dtype=float)
    whp = df["whp_psia"].to_numpy(dtype=float)
    ds = df["whp_downstream_psia"].to_numpy(dtype=float)
    dhp = df["dh_pressure_psia"].to_numpy(dtype=float)
    dht = df["dh_temperature_degf"].to_numpy(dtype=float)
    ann = df["annulus_pressure_psia"].to_numpy(dtype=float)
    gl = np.maximum(df["gas_lift_rate_mscfd"].to_numpy(dtype=float), 0.0)
    days = df["days_since_start"].to_numpy(dtype=float)

    max_64 = np.array([float(comp.at[w, "choke_max_size_64ths"]) for w in wells])
    gauge_tvd = np.array([anchors[w]["gauge_tvd"] for w in wells])
    glv_tvd = np.array([anchors[w]["glv_tvd"] for w in wells])
    glv_port = np.array([float(comp.at[w, "gl_valve_port_64ths"]) for w in wells])
    lift_sg = np.array([float(comp.at[w, "lift_gas_sg"]) for w in wells])
    tub_area = np.array([anchors[w]["tub_area"] for w in wells])
    pb = np.array([float(pvt.at[w, "bubble_point_pressure_psia"]) for w in wells])

    cd = np.array([anchors[w]["cd"] for w in wells])
    pi = np.array([anchors[w]["pi"] for w in wells])
    pr0 = np.array([anchors[w]["pr0"] for w in wells])
    dpr_dt = np.array([anchors[w]["dpr_dt"] for w in wells])
    grad_ref = np.array([anchors[w]["grad_ref"] for w in wells])
    gl_ref = np.array([anchors[w]["gl_ref"] for w in wells])
    dht_ref = np.array([anchors[w]["dht_ref"] for w in wells])
    ann_ref = np.array([anchors[w]["ann_ref"] for w in wells])
    vlp_base = np.array([anchors[w]["vlp_base"] for w in wells])
    vc0 = np.array([anchors[w]["vlp_coefs"][0] for w in wells])
    vc1 = np.array([anchors[w]["vlp_coefs"][1] for w in wells])
    vc2 = np.array([anchors[w]["vlp_coefs"][2] for w in wells])
    vc3 = np.array([anchors[w]["vlp_coefs"][3] for w in wells])
    w_ch = np.array([anchors[w]["w_choke"] for w in wells])
    w_ip = np.array([anchors[w]["w_ipr"] for w in wells])
    w_vl = np.array([anchors[w]["w_vlp"] for w in wells])
    wc_tm = np.array([anchors[w]["wc_t_mean"] for w in wells])
    wc_b0 = np.array([anchors[w]["wc_b0"] for w in wells])
    wc_b1 = np.array([anchors[w]["wc_b1"] for w in wells])
    gor_m = np.array([anchors[w]["gor_med"] for w in wells])
    gor_s = np.array([anchors[w]["gor_slope"] for w in wells])
    w_idx = np.array([float(anchors[w]["well_idx"]) for w in wells])

    d_in = choke_bean_diameter_in(choke, max_64)
    area_ch = (math.pi / 4.0) * (d_in ** 2)
    dp_eff = effective_choke_dp(whp, ds)
    dp_raw = np.maximum(whp - ds, 0.0)
    cap = area_ch * np.sqrt(np.maximum(dp_eff, 0.0))
    q_choke = cd * cap

    pr_t = pr0 + dpr_dt * days
    drawdown = np.maximum(pr_t - dhp, 5.0)
    q_ipr = np.where(
        dhp >= pb,
        pi * drawdown,
        pi * (np.maximum(pr_t - pb, 0.0) + (pb / 1.8) * (1.0 - 0.2 * (dhp / pb) - 0.8 * ((dhp / pb) ** 2))),
    )

    dp_tub = np.maximum(dhp - whp, 50.0)
    grad_eff = dp_tub / gauge_tvd
    l_up_frac = glv_tvd / gauge_tvd
    t_avg_r = 460.0 + 0.5 * (dht + 100.0)
    p_cas_glv = ann * np.exp(0.01875 * lift_sg * glv_tvd / np.maximum(t_avg_r, 500.0))
    p_tub_glv = whp + l_up_frac * dp_tub
    dp_glv = np.maximum(p_cas_glv - p_tub_glv, 0.0)
    glv_cap = ((glv_port / 64.0) ** 2) * np.sqrt(dp_glv)

    q_vlp = np.maximum(
        vlp_base
        + vc0 * (grad_eff - grad_ref) * gauge_tvd
        + vc1 * (gl - gl_ref)
        + vc2 * (ann - ann_ref)
        + vc3 * (dht - dht_ref) * 10.0,
        10.0,
    )

    q_nodal = w_ch * q_choke + w_ip * q_ipr + w_vl * q_vlp
    is_si = choke <= 0.5
    q_choke = np.where(is_si, 0.0, q_choke)
    q_ipr = np.where(is_si, 0.0, q_ipr)
    q_vlp = np.where(is_si, 0.0, q_vlp)
    q_nodal = np.where(is_si, 0.0, q_nodal)

    dt_wc = days - wc_tm
    wc_frac = 1.0 / (1.0 + np.exp(-(wc_b0 + wc_b1 * dt_wc)))
    gor_prior = np.maximum(gor_m + gor_s * dt_wc, 50.0)
    q_oil_prior = q_nodal * (1.0 - wc_frac)
    q_wat_prior = q_nodal * wc_frac
    q_gas_prior = q_oil_prior * gor_prior / 1000.0

    X = np.column_stack([
        q_nodal, q_choke, q_ipr, q_vlp, cap, choke, drawdown, dhp, whp, dp_raw,
        grad_eff, dht - dht_ref, ann - ann_ref, glv_cap, gl / tub_area, l_up_frac,
        wc_frac, gor_prior, days, w_idx,
    ])
    priors = {"oil": q_oil_prior, "water": q_wat_prior, "gas": q_gas_prior, "liquid": q_nodal}
    return X, priors


# ==============================================================================
# 3. Adjustment 1: Well-Specific SCADA Envelope + Manifold/Separator Synthetic Data
# ==============================================================================

def export_few_shot_prompts(
    wt: pd.DataFrame, comp: pd.DataFrame, pvt: pd.DataFrame, anchors: dict[str, dict], envelopes: dict[str, dict]
) -> Path:
    """Export updated per-well Few-Shot Prompts reflecting each well's real SCADA envelope & material-balance depletion."""
    prompt_path = OUT_DIR / "vfm_few_shot_prompts_by_well.txt"
    blocks = []
    for w in sorted(comp.index):
        c, p, anc, env = comp.loc[w], pvt.loc[w], anchors[w], envelopes[w]
        sub = wt[wt["well_name"] == w].sort_values("test_start")
        c_lo, c_hi = max(8.0, 0.85 * env["choke_p02"]), min(82.0, 1.15 * env["choke_p98"])
        ds_lo, ds_hi = 0.96 * env["ds_p02"], 1.04 * env["ds_p98"]
        lines = [
            f"================================================================================",
            f"FEW-SHOT SYNTHETIC DATA PROMPT — WELL {w} (SCADA ENVELOPE + MATERIAL BALANCE)",
            f"================================================================================",
            f"STATIC COMPLETION, PVT & ENVELOPE CONSTRAINTS ({w}):",
            f"  - Gauge TVD: {c['gauge_tvd_ft']:.0f} ft | GL Valve TVD: {c['gl_valve_tvd_ft']:.0f} ft | Tubing ID: {c['tubing_id_in']:.3f} in",
            f"  - Drainage Area: {p['estimated_drainage_area_acres']:.0f} acres | Material-Balance dPr/dt: {anc['dpr_dt']:.3f} psi/day | Pr0: {anc['pr0']:.1f} psia",
            f"  - Real SCADA Choke Envelope: [{c_lo:.1f}%, {c_hi:.1f}%] | Downstream Manifold/Separator Envelope: [{ds_lo:.1f}, {ds_hi:.1f}] psia",
            f"  - Pb: {p['bubble_point_pressure_psia']:.0f} psia | Bo: {p['oil_fvf_rb_stb']:.3f} | mu_o: {p['oil_viscosity_cp']:.3f} cp | Rs@Pb: {c['solution_gor_at_pb_scf_stb']:.1f}",
            f"",
            f"FEW-SHOT HISTORICAL SEPARATOR TESTS ({len(sub)} shots for {w}):",
        ]
        for k, (_, r) in enumerate(sub.iterrows(), 1):
            lines.append(
                f"  Shot {k}: date={r['test_mid'].strftime('%Y-%m-%d')} | choke={r['choke_opening_pct']:.1f}% | "
                f"Pdh={r['dh_pressure_psia']:.1f} | Tdh={r['dh_temperature_degf']:.2f}F | WHP={r['whp_psia']:.1f} | "
                f"Pds={r['whp_downstream_psia']:.1f} | Pann={r['annulus_pressure_psia']:.1f} | Qgl={r['gas_lift_rate_mscfd']:.1f} "
                f"-> Qo={r['oil_rate_stbd']:.1f}, Qw={r['water_rate_stbd']:.1f}, Qg={r['gas_rate_mscfd']:.1f}"
            )
        lines.append(
            f"\nTASK: Generate 180 physics-gated steady-state rows for {w} spanning its real choke envelope "
            f"[{c_lo:.1f}%, {c_hi:.1f}%] and BOTH test-separator & production-manifold backpressures [{ds_lo:.1f}, {ds_hi:.1f}] psia."
        )
        blocks.append("\n".join(lines))
    prompt_path.write_text("\n\n".join(blocks), encoding="utf-8")
    return prompt_path


def generate_few_shot_synthetic_dataset(
    wt: pd.DataFrame,
    comp: pd.DataFrame,
    pvt: pd.DataFrame,
    anchors: dict[str, dict],
    envelopes: dict[str, dict],
    n_per_well: int = 180,
    seed: int = 42,
) -> pd.DataFrame:
    """Adjustment 1: Synthesize counterfactuals strictly inside each well's real SCADA operating
    envelope (choke, gas-lift) AND explicitly sweeping both Test-Separator and Production-Manifold P_ds."""
    rng = np.random.default_rng(seed)
    rows = []

    for w in sorted(comp.index):
        sub = wt[wt["well_name"] == w].reset_index(drop=True)
        c_row, p_row, anc, env = comp.loc[w], pvt.loc[w], anchors[w], envelopes[w]
        pb = float(p_row["bubble_point_pressure_psia"])
        max_64 = float(c_row["choke_max_size_64ths"])
        gauge_tvd = anc["gauge_tvd"]

        # Real well-specific SCADA bounds (+-15% margin, never stretching SVR scale unphysically)
        c_lo = max(8.0, 0.85 * env["choke_p02"])
        c_hi = min(82.0, 1.15 * env["choke_p98"])
        ds_lo = max(50.0, 0.96 * env["ds_p02"])
        ds_hi = max(ds_lo + 10.0, 1.04 * env["ds_p98"])
        gl_lo = max(0.0, 0.85 * env["gl_p05"])
        gl_hi = max(gl_lo + 20.0, 1.15 * env["gl_p95"])

        dht_std = max(float(sub["dh_temperature_degf"].std()), 0.12)
        ann_std = max(float(sub["annulus_pressure_psia"].std()), 4.0)

        day_lo = float(min(0.0, wt["days_since_start"].min()))
        day_hi = float(max(wt["days_since_start"].max() + 15.0, 30.0))
        jt_slp = float(anc.get("jt_slope", 0.0))

        for k in range(n_per_well):
            shot = sub.iloc[k % len(sub)]
            days = float(rng.uniform(day_lo, day_hi))
            pr_t = anc["pr0"] + anc["dpr_dt"] * days

            # 75% local perturbations around shot inside real SCADA envelope; 25% uniform across well's envelope
            if k % 4 == 0:
                choke_new = float(rng.uniform(c_lo, c_hi))
            else:
                choke_new = float(np.clip(shot["choke_opening_pct"] * rng.uniform(0.90, 1.10), c_lo, c_hi))

            # Explicitly sample BOTH test-separator (lower) and production-manifold (higher) backpressures!
            if k % 2 == 0:
                ds_new = float(rng.uniform(ds_lo, ds_hi))
            else:
                ds_new = float(np.clip(shot["whp_downstream_psia"] * rng.uniform(0.96, 1.12), ds_lo, ds_hi))

            gl_new = float(np.clip(shot["gas_lift_rate_mscfd"] * rng.uniform(0.90, 1.10), gl_lo, gl_hi))

            d_in = float(choke_bean_diameter_in(choke_new, max_64))
            area = math.pi / 4.0 * (d_in ** 2)
            shot_d_in = float(choke_bean_diameter_in(float(shot["choke_opening_pct"]), max_64))
            shot_area = math.pi / 4.0 * (shot_d_in ** 2)
            shot_cap = max(
                shot_area * math.sqrt(max(float(effective_choke_dp(shot["whp_psia"], shot["whp_downstream_psia"])), 1e-4)),
                1e-4,
            )
            cd_local = float(np.clip((float(shot["liquid_rate_stbd"]) / shot_cap) * rng.normal(1.0, 0.005), 0.90 * anc["cd"], 1.10 * anc["cd"]))

            logit_wc = anc["wc_b0"] + anc["wc_b1"] * (days - anc["wc_t_mean"])
            wc_frac = 1.0 / (1.0 + math.exp(-logit_wc))
            wc_shot = float(shot["water_cut_pct"]) / 100.0
            gor_syn = max(anc["gor_med"] + anc["gor_slope"] * (days - anc["wc_t_mean"]), 50.0)

            dp_tub_base = anc["grad_ref"] * gauge_tvd
            dp_tub_new = (
                dp_tub_base
                + 0.095 * gauge_tvd * (wc_frac - wc_shot)
                - 0.11 * (gl_new - anc["gl_ref"])
                + float(rng.normal(0.0, 3.0))
            )

            # Coupled fixed-point nodal solver (explicitly resolves manifold vs separator backpressure!)
            q_guess = float(shot["liquid_rate_stbd"]) * (area / max(shot_area, 1e-4))
            dhp_new = float(shot["dh_pressure_psia"])
            whp_new = float(shot["whp_psia"])
            for _ in range(10):
                dhp_target = max(pr_t - q_guess / max(anc["pi"], 1e-3), pb * 0.65)
                dhp_new = 0.55 * dhp_new + 0.45 * dhp_target
                whp_new = max(dhp_new - dp_tub_new, ds_new + 20.0)
                dp_eff = float(effective_choke_dp(whp_new, ds_new))
                q_choke = cd_local * area * math.sqrt(max(dp_eff, 0.0))
                if dhp_new >= pb:
                    q_ipr = anc["pi"] * max(pr_t - dhp_new, 10.0)
                else:
                    q_ipr = anc["pi"] * (max(pr_t - pb, 0.0) + (pb / 1.8) * (1.0 - 0.2 * (dhp_new / pb) - 0.8 * ((dhp_new / pb) ** 2)))
                q_guess = 0.65 * q_choke + 0.35 * q_ipr

            q_liq_syn = max(q_guess, 0.0)
            q_oil_syn = q_liq_syn * (1.0 - wc_frac)
            q_wat_syn = q_liq_syn * wc_frac
            q_gas_syn = q_oil_syn * gor_syn / 1000.0

            dht_new = float(shot["dh_temperature_degf"]) + jt_slp * (dhp_new - float(shot["dh_pressure_psia"])) + float(rng.normal(0.0, 0.12 * dht_std))
            ann_new = float(shot["annulus_pressure_psia"]) + 0.12 * (gl_new - float(shot["gas_lift_rate_mscfd"])) + float(rng.normal(0.0, 0.20 * ann_std))

            rows.append({
                "well_name": w,
                "days_since_start": days,
                "choke_opening_pct": choke_new,
                "whp_psia": whp_new,
                "whp_downstream_psia": ds_new,
                "dh_pressure_psia": dhp_new,
                "dh_temperature_degf": dht_new,
                "annulus_pressure_psia": ann_new,
                "gas_lift_rate_mscfd": gl_new,
                "liquid_rate_stbd": q_liq_syn,
                "oil_rate_stbd": q_oil_syn,
                "water_rate_stbd": q_wat_syn,
                "gas_rate_mscfd": q_gas_syn,
                "water_cut_pct": wc_frac * 100.0,
                "gor_scf_stb": gor_syn,
                "is_synthetic": 1,
            })

    return pd.DataFrame(rows)


# ==============================================================================
# 4. Adjustment 6: Controlled Weight Budget + Tunable SVR/MLP Stacked Learner
# ==============================================================================

class PhaseStackedSuperLearner:
    """5-Base Stacked Super-Learner for phase k in {'oil', 'water', 'gas', 'liquid'}."""

    def __init__(self, phase: str, svr_c: float = 2.0, svr_eps: float = 0.004, mlp_alpha: float = 0.12):
        self.phase = phase
        self.svr_c = svr_c
        self.svr_eps = svr_eps
        self.mlp_alpha = mlp_alpha
        self.mono_cst = [1, 1, 1, 1, 1, 1, 1, 1, -1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.meta_weights = np.array([0.20, 0.20, 0.20, 0.20, 0.20], dtype=float)
        self.base_names = ["Mono-LightGBM", "Mono-HistGBM", "Huber-ResGBM", "ExtraTrees-Res", "SVR+MLP-Neural"]

    def _make_base_models(self):
        lgbm = lgb.LGBMRegressor(
            n_estimators=240, learning_rate=0.032, num_leaves=15, max_depth=4, min_child_samples=8,
            subsample=0.85, colsample_bytree=0.85, reg_alpha=0.5, reg_lambda=2.5,
            monotone_constraints=self.mono_cst, random_state=42, verbose=-1,
        )
        hgb = HistGradientBoostingRegressor(
            max_iter=220, learning_rate=0.035, max_leaf_nodes=15, max_depth=4, min_samples_leaf=8,
            l2_regularization=2.5, monotonic_cst=self.mono_cst, random_state=42,
        )
        res_gbm = GradientBoostingRegressor(
            loss="huber", n_estimators=150, learning_rate=0.035, max_depth=3, min_samples_leaf=6,
            subsample=0.85, random_state=42,
        )
        et_res = ExtraTreesRegressor(
            n_estimators=180, max_depth=10, min_samples_leaf=3, max_features=0.75, random_state=42, n_jobs=-1,
        )
        svr_res = SVR(kernel="rbf", C=self.svr_c, epsilon=self.svr_eps, gamma="scale")
        mlp_res = MLPRegressor(
            hidden_layer_sizes=(48, 24), activation="relu", alpha=self.mlp_alpha,
            learning_rate_init=0.005, max_iter=300, early_stopping=True, validation_fraction=0.15, random_state=42,
        )
        return lgbm, hgb, res_gbm, et_res, svr_res, mlp_res

    def fit_base(self, X_base: np.ndarray, q_prior: np.ndarray, y: np.ndarray, w: np.ndarray, is_real: np.ndarray):
        X_full = np.column_stack([q_prior, X_base])
        q_safe = np.maximum(q_prior, 5.0)
        y_log_res = np.log(np.maximum(y, 5.0) / q_safe)

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_full)

        self.lgbm, self.hgb, self.res_gbm, self.et_res, self.svr_res, self.mlp_res = self._make_base_models()
        self.lgbm.fit(X_full, y, sample_weight=w)
        self.hgb.fit(X_full, y, sample_weight=w)
        self.res_gbm.fit(X_full, y_log_res, sample_weight=w)
        self.et_res.fit(X_full, y_log_res, sample_weight=w)
        self.svr_res.fit(X_scaled, y_log_res, sample_weight=w)

        # Balance real-vs-synthetic weight in MLP via integer replication matching w ratio
        real_idx = np.where(is_real == 1)[0]
        syn_idx = np.where(is_real == 0)[0]
        rep = max(1, int(round(float(w[real_idx].sum()) / max(float(w[syn_idx].sum()), 1e-6) * len(syn_idx) / max(len(real_idx), 1))))
        rep = min(rep, 25)
        X_mlp = np.vstack([X_scaled[syn_idx], np.repeat(X_scaled[real_idx], rep, axis=0)])
        y_mlp = np.concatenate([y_log_res[syn_idx], np.repeat(y_log_res[real_idx], rep, axis=0)])
        self.mlp_res.fit(X_mlp, y_mlp)
        return self

    def predict_base_matrix(self, X_base: np.ndarray, q_prior: np.ndarray, choke: np.ndarray) -> np.ndarray:
        X_full = np.column_stack([q_prior, X_base])
        X_scaled = self.scaler.transform(X_full)

        p0 = np.maximum(self.lgbm.predict(X_full), 0.0)
        p1 = np.maximum(self.hgb.predict(X_full), 0.0)
        p2 = q_prior * np.exp(np.clip(self.res_gbm.predict(X_full), -0.30, 0.30))
        p3 = q_prior * np.exp(np.clip(self.et_res.predict(X_full), -0.30, 0.30))
        res_smooth = 0.60 * self.svr_res.predict(X_scaled) + 0.40 * self.mlp_res.predict(X_scaled)
        p4 = q_prior * np.exp(np.clip(res_smooth, -0.30, 0.30))

        P = np.column_stack([p0, p1, p2, p3, p4])
        P[choke <= 0.5, :] = 0.0
        return P

    def fit_meta_weights(self, oof_P: np.ndarray, y_real: np.ndarray):
        def loss(weights):
            pred = oof_P @ weights
            rmse_norm = np.sqrt(np.mean((y_real - pred) ** 2)) / np.mean(y_real)
            wape = np.sum(np.abs(y_real - pred)) / np.sum(y_real)
            reg = 0.002 * np.sum((weights - 0.2) ** 2)
            return 0.55 * rmse_norm + 0.45 * wape + reg

        res = minimize(
            loss,
            x0=np.full(oof_P.shape[1], 1.0 / oof_P.shape[1]),
            bounds=[(0.0, 1.0)] * oof_P.shape[1],
            constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            method="SLSQP",
        )
        self.meta_weights = res.x / np.sum(res.x)
        return self.meta_weights

    def predict(self, X_base: np.ndarray, q_prior: np.ndarray, choke: np.ndarray) -> np.ndarray:
        P = self.predict_base_matrix(X_base, q_prior, choke)
        return np.maximum(P @ self.meta_weights, 0.0)


def assign_controlled_weights(is_real: np.ndarray, rho_syn: float) -> np.ndarray:
    """Adjustment 6: Normalize synthetic sample weights so total synthetic weight == rho_syn * total real weight."""
    n_real = max(int(np.sum(is_real == 1)), 1)
    n_syn = max(int(np.sum(is_real == 0)), 1)
    w_real = 1.0
    w_syn = (rho_syn * n_real * w_real) / n_syn
    return np.where(is_real == 1, w_real, w_syn)


def apply_soft_phase_reconciliation(
    q_oil: np.ndarray, q_wat: np.ndarray, q_liq: np.ndarray, lam_rec: float
) -> tuple[np.ndarray, np.ndarray]:
    """Adjustment 5: Softly reconcile independent (Q_o, Q_w) toward stacked total liquid Q_liq."""
    if lam_rec <= 0.0:
        return q_oil, q_wat
    q_sum = q_oil + q_wat
    scale = np.where(q_sum > 1.0, 1.0 + lam_rec * (q_liq / q_sum - 1.0), 1.0)
    scale = np.clip(scale, 0.85, 1.15)
    return np.maximum(q_oil * scale, 0.0), np.maximum(q_wat * scale, 0.0)


def calc_metrics(yt: np.ndarray, yp: np.ndarray) -> dict[str, float]:
    ok = np.isfinite(yt) & np.isfinite(yp) & (yt > 0)
    yt, yp = yt[ok], yp[ok]
    return {
        "RMSE": float(np.sqrt(np.mean((yt - yp) ** 2))),
        "MAE": float(np.mean(np.abs(yt - yp))),
        "R2": 1.0 - float(np.sum((yt - yp) ** 2) / np.sum((yt - np.mean(yt)) ** 2)),
        "WAPE": 100.0 * float(np.sum(np.abs(yt - yp)) / np.sum(yt)),
        "MAPE": 100.0 * float(np.mean(np.abs((yt - yp) / yt))),
    }


# ==============================================================================
# 5. End-to-End Nested OOF Tuning, Reconciliation, Hourly Integration & Coupled PTA
# ==============================================================================

def fit_v2_stacked_bundle(
    wt: pd.DataFrame,
    comp: pd.DataFrame,
    pvt: pd.DataFrame,
    envelopes: dict[str, dict],
    ref_date: pd.Timestamp,
    n_synth_per_well: int = 180,
    verbose: bool = True,
) -> dict:
    """End-to-end data-driven training & 5-fold OOF tuning of all 4 phase super-learners,
    hyperparameters, meta-weights, and soft reconciliation lambda_rec (zero hardcoded weights)."""
    anchors_full = calibrate_well_anchors(wt, comp, pvt, envelopes)
    real_cols = [
        "well_name", "days_since_start", "choke_opening_pct",
        "whp_psia", "whp_downstream_psia", "dh_pressure_psia",
        "dh_temperature_degf", "annulus_pressure_psia", "gas_lift_rate_mscfd",
        "liquid_rate_stbd", "oil_rate_stbd", "water_rate_stbd", "gas_rate_mscfd",
        "water_cut_pct", "gor_scf_stb",
    ]
    df_real = wt[real_cols].copy()
    df_real["is_synthetic"] = 0

    phase_map = [
        ("oil", "oil_rate_stbd"),
        ("water", "water_rate_stbd"),
        ("gas", "gas_rate_mscfd"),
        ("liquid", "liquid_rate_stbd"),
    ]

    # Strictly causal 5-fold CV (anchors & synthetic generated ONLY on training fold!)
    n_folds = min(5, max(2, len(df_real) // max(len(comp), 1)))
    fold_cache = []
    for fold in range(n_folds):
        val_mask = np.arange(len(df_real)) % n_folds == fold
        tr_real = df_real[~val_mask].copy()
        va_real = df_real[val_mask].copy()
        wt_tr = wt.iloc[np.where(~val_mask)[0]].copy()
        anc_tr = calibrate_well_anchors(wt_tr, comp, pvt, envelopes)
        syn_tr = generate_few_shot_synthetic_dataset(wt_tr, comp, pvt, anc_tr, envelopes, n_per_well=140, seed=300 + fold)
        df_tr_aug = pd.concat([tr_real, syn_tr], ignore_index=True)
        X_tr, pri_tr = build_rich_feature_matrix(df_tr_aug, comp, pvt, anc_tr)
        X_va, pri_va = build_rich_feature_matrix(va_real, comp, pvt, anc_tr)
        is_real_tr = (df_tr_aug["is_synthetic"].to_numpy() == 0).astype(int)
        fold_cache.append((val_mask, df_tr_aug, va_real, X_tr, pri_tr, X_va, pri_va, is_real_tr))

    candidates = [
        (0.45, 1.5, 0.004, 0.12),
        (0.75, 2.5, 0.003, 0.10),
        (1.00, 4.0, 0.002, 0.08),
    ]
    best_cfg: dict[str, tuple[float, float, float, float]] = {}
    best_oof_pred: dict[str, np.ndarray] = {}
    learners: dict[str, PhaseStackedSuperLearner] = {}

    if verbose:
        print("\n=== OOF TUNING OF WEIGHT BUDGET (rho_syn) & SVR/MLP REGULARIZATION ===")
    for p_key, col in phase_map:
        yt = df_real[col].to_numpy(dtype=float)
        best_score = 1e9
        for cfg in candidates:
            rho_syn, svr_c, svr_eps, mlp_alpha = cfg
            oof_P = np.zeros((len(df_real), 5), dtype=float)
            for val_mask, df_tr_aug, va_real, X_tr, pri_tr, X_va, pri_va, is_real_tr in fold_cache:
                w_tr = assign_controlled_weights(is_real_tr, rho_syn)
                lrn = PhaseStackedSuperLearner(p_key, svr_c=svr_c, svr_eps=svr_eps, mlp_alpha=mlp_alpha)
                lrn.fit_base(X_tr, pri_tr[p_key], df_tr_aug[col].to_numpy(dtype=float), w_tr, is_real_tr)
                oof_P[val_mask, :] = lrn.predict_base_matrix(X_va, pri_va[p_key], va_real["choke_opening_pct"].to_numpy(dtype=float))
            lrn_meta = PhaseStackedSuperLearner(p_key, svr_c=svr_c, svr_eps=svr_eps, mlp_alpha=mlp_alpha)
            w_m = lrn_meta.fit_meta_weights(oof_P, yt)
            pred_oof = oof_P @ w_m
            m_cv = calc_metrics(yt, pred_oof)
            score = 0.5 * (m_cv["RMSE"] / np.mean(yt)) + 0.5 * (m_cv["WAPE"] / 100.0)
            if score < best_score:
                best_score = score
                best_cfg[p_key] = cfg
                best_oof_pred[p_key] = pred_oof
                learners[p_key] = lrn_meta

        if verbose:
            rho_s, sc, se, _ = best_cfg[p_key]
            m_best = calc_metrics(yt, best_oof_pred[p_key])
            w_str = ", ".join(f"{n}={v:.2f}" for n, v in zip(learners[p_key].base_names, learners[p_key].meta_weights))
            print(f"  {p_key.upper():6s} | Best (rho_syn={rho_s:.2f}, SVR_C={sc:.1f}, eps={se:.3f}) -> "
                  f"OOF RMSE={m_best['RMSE']:6.2f} | R2={m_best['R2']:.4f} | WAPE={m_best['WAPE']:.2f}% | MAPE={m_best['MAPE']:.2f}%")
            print(f"         | Meta-Weights: [{w_str}]")

    # Optimize Soft Phase-Sum Coherence Reconciliation lambda_rec strictly on OOF predictions
    yt_o = df_real["oil_rate_stbd"].to_numpy(dtype=float)
    yt_w = df_real["water_rate_stbd"].to_numpy(dtype=float)
    best_lam, best_rec_score = 0.0, 1e9
    for lam in np.linspace(0.0, 1.0, 21):
        qo_r, qw_r = apply_soft_phase_reconciliation(best_oof_pred["oil"], best_oof_pred["water"], best_oof_pred["liquid"], float(lam))
        mo, mw = calc_metrics(yt_o, qo_r), calc_metrics(yt_w, qw_r)
        sc = (mo["RMSE"] / np.mean(yt_o)) + (mw["RMSE"] / np.mean(yt_w)) + 0.01 * (mo["WAPE"] + mw["WAPE"])
        if sc < best_rec_score:
            best_rec_score = sc
            best_lam = float(lam)

    if verbose:
        qo_oof_rec, qw_oof_rec = apply_soft_phase_reconciliation(best_oof_pred["oil"], best_oof_pred["water"], best_oof_pred["liquid"], best_lam)
        mo_pre, mo_post = calc_metrics(yt_o, best_oof_pred["oil"]), calc_metrics(yt_o, qo_oof_rec)
        mw_pre, mw_post = calc_metrics(yt_w, best_oof_pred["water"]), calc_metrics(yt_w, qw_oof_rec)
        print(f"\n=== SOFT PHASE-SUM RECONCILIATION (OOF optimal lambda_rec = {best_lam:.2f}) ===")
        print(f"  Oil   OOF: RMSE {mo_pre['RMSE']:.2f} -> {mo_post['RMSE']:.2f} | R2 {mo_pre['R2']:.4f} -> {mo_post['R2']:.4f} | WAPE {mo_pre['WAPE']:.2f}% -> {mo_post['WAPE']:.2f}%")
        print(f"  Water OOF: RMSE {mw_pre['RMSE']:.2f} -> {mw_post['RMSE']:.2f} | R2 {mw_pre['R2']:.4f} -> {mw_post['R2']:.4f} | WAPE {mw_pre['WAPE']:.2f}% -> {mw_post['WAPE']:.2f}%")

    # Fit final 4 phase super-learners on full real + physics-gated synthetic dataset
    df_synth_full = generate_few_shot_synthetic_dataset(wt, comp, pvt, anchors_full, envelopes, n_per_well=n_synth_per_well, seed=42)
    df_train_full = pd.concat([df_real, df_synth_full], ignore_index=True)
    X_full_tr, pri_full_tr = build_rich_feature_matrix(df_train_full, comp, pvt, anchors_full)
    is_real_full = (df_train_full["is_synthetic"].to_numpy() == 0).astype(int)

    for p_key, col in phase_map:
        rho_s = best_cfg[p_key][0]
        w_full = assign_controlled_weights(is_real_full, rho_s)
        learners[p_key].fit_base(X_full_tr, pri_full_tr[p_key], df_train_full[col].to_numpy(dtype=float), w_full, is_real_full)

    qo_oof_final, qw_oof_final = apply_soft_phase_reconciliation(
        best_oof_pred["oil"], best_oof_pred["water"], best_oof_pred["liquid"], best_lam
    )
    oof_metrics: dict[str, dict[str, float]] = {}
    for p_key, col in phase_map:
        yt_p = df_real[col].to_numpy(dtype=float)
        yp_p = qo_oof_final if p_key == "oil" else (qw_oof_final if p_key == "water" else best_oof_pred[p_key])
        oof_metrics[p_key] = calc_metrics(yt_p, yp_p)

    return {
        "model_name": "synth_stacked_v2",
        "version": "2.2",
        "ref_date": str(pd.Timestamp(ref_date).floor("D")),
        "n_real_tests": int(len(df_real)),
        "n_synthetic_rows": int(len(df_synth_full)),
        "lambda_rec": float(best_lam),
        "best_cfg": best_cfg,
        "oof_metrics": oof_metrics,
        "learners": learners,
        "anchors": anchors_full,
        "envelopes": envelopes,
        "completion_table": comp.reset_index().to_dict(orient="records"),
        "pvt_table": pvt.reset_index().to_dict(orient="records"),
        "df_synth": df_synth_full,
    }


def run_refined_phase_decomposed_vfm(raw_dir: Path | None = None):
    raw_dir = resolve_raw_dir(raw_dir)
    comp = pd.read_csv(raw_dir / "well_completion.csv").set_index("well_name")
    pvt = pd.read_csv(raw_dir / "reservoir_pvt.csv").set_index("well_name")
    ce = pd.read_csv(raw_dir / "choke_events.csv", parse_dates=["timestamp"]).sort_values(["well_name", "timestamp"])

    wt_raw = load_harmonized_well_tests(raw_dir)
    wt, hr_df, envelopes, ref_date = stream_raw_scada_robust_median(wt_raw, ce, raw_dir)

    bundle = fit_v2_stacked_bundle(wt, comp, pvt, envelopes, ref_date, n_synth_per_well=180, verbose=True)
    anchors_full = bundle["anchors"]
    learners = bundle["learners"]
    best_lam = float(bundle["lambda_rec"])
    bundle["df_synth"].to_csv(OUT_DIR / "vfm_phase_decomposed_synthetic_1800.csv", index=False)

    prompt_path = export_few_shot_prompts(wt, comp, pvt, anchors_full, envelopes)
    print(f"Exported refined Few-Shot Prompts -> {prompt_path.name}")
    print("\n=== VOLUMETRIC MATERIAL-BALANCE DEPLETION & SHUT-IN P_r ANCHORS ===")
    for w in sorted(anchors_full):
        a = anchors_full[w]
        si_p = envelopes[w].get("shutin_pmax_psi", np.nan)
        print(f"  {w} | Pr0={a['pr0']:7.1f} psia (Shut-In Pmax={si_p:6.1f}) | dPr/dt={a['dpr_dt']:+.3f} psi/day | PI={a['pi']:.2f} STB/D/psi | JT={a['jt_slope']:+.4f} F/psi")

    # Predict across all hourly blocks + apply Soft Phase-Sum Reconciliation + integrate to daily rows
    X_hr, pri_hr = build_rich_feature_matrix(hr_df, comp, pvt, anchors_full)
    choke_hr = hr_df["choke_opening_pct"].to_numpy(dtype=float)
    up_hr = hr_df["uptime_frac"].to_numpy(dtype=float)

    q_oil_fl = learners["oil"].predict(X_hr, pri_hr["oil"], choke_hr)
    q_wat_fl = learners["water"].predict(X_hr, pri_hr["water"], choke_hr)
    q_gas_fl = learners["gas"].predict(X_hr, pri_hr["gas"], choke_hr)
    q_liq_fl = learners["liquid"].predict(X_hr, pri_hr["liquid"], choke_hr)

    q_oil_rec, q_wat_rec = apply_soft_phase_reconciliation(q_oil_fl, q_wat_fl, q_liq_fl, best_lam)

    hr_preds = hr_df[["well_name", "hour", "date", "uptime_frac"]].copy()
    hr_preds["oil_rate_stbd"] = q_oil_rec * up_hr
    hr_preds["water_rate_stbd"] = q_wat_rec * up_hr
    hr_preds["gas_rate_mscfd"] = q_gas_fl * up_hr

    daily_df = hr_preds.groupby(["well_name", "date"], as_index=False).agg(
        oil_rate_stbd=("oil_rate_stbd", "mean"),
        water_rate_stbd=("water_rate_stbd", "mean"),
        gas_rate_mscfd=("gas_rate_mscfd", "mean"),
        uptime_frac=("uptime_frac", "mean"),
    ).sort_values(["well_name", "date"]).reset_index(drop=True)

    # Save refined VFM deliverables
    vfm_deliv = daily_df[["date", "well_name", "oil_rate_stbd", "water_rate_stbd", "gas_rate_mscfd"]].copy()
    vfm_deliv["date"] = pd.to_datetime(vfm_deliv["date"]).dt.strftime("%Y-%m-%d")
    for c in ["oil_rate_stbd", "water_rate_stbd", "gas_rate_mscfd"]:
        vfm_deliv[c] = vfm_deliv[c].round(2)

    out_csv = OUT_DIR / "vfm_phase_decomposed_stacked_v2.csv"
    out_xlsx = OUT_DIR / "vfm_phase_decomposed_stacked_v2.xlsx"
    out_ts_csv = OUT_DIR / "vfm_phase_decomposed_stacked_v2_timestamp_schema.csv"
    vfm_deliv.to_csv(out_csv, index=False)
    vfm_deliv.to_excel(out_xlsx, index=False)
    vfm_deliv.rename(columns={"date": "timestamp"}).to_csv(out_ts_csv, index=False)

    v2_df = daily_df.copy()
    v2_df["date"] = pd.to_datetime(v2_df["date"]).dt.floor("D")

    wt_eval = wt[["well_name", "test_start", "test_end", "oil_rate_stbd", "water_rate_stbd", "gas_rate_mscfd"]].copy()
    wt_eval["date"] = wt_eval["test_start"].dt.floor("D")

    hr_test_preds = []
    for _, r in wt_eval.iterrows():
        sub_h = hr_preds[
            (hr_preds["well_name"] == r["well_name"])
            & (hr_preds["hour"] >= r["test_start"].floor("h"))
            & (hr_preds["hour"] <= r["test_end"].floor("h"))
        ]
        hr_test_preds.append({
            "oil_hr": float(sub_h["oil_rate_stbd"].mean()),
            "wat_hr": float(sub_h["water_rate_stbd"].mean()),
            "gas_hr": float(sub_h["gas_rate_mscfd"].mean()),
        })
    hr_tp_df = pd.DataFrame(hr_test_preds)

    def span_match(df_day: pd.DataFrame, row: pd.Series) -> tuple[float, float, float]:
        w, t0, t1 = row["well_name"], row["test_start"], row["test_end"]
        sub = df_day[(df_day["well_name"] == w) & (df_day["date"] >= t0.floor("D")) & (df_day["date"] <= t1.floor("D"))]
        if sub.empty:
            return np.nan, np.nan, np.nan
        wts = [max(0.01, (min(t1, dr["date"] + pd.Timedelta(days=1)) - max(t0, dr["date"])).total_seconds() / 3600.0) for _, dr in sub.iterrows()]
        w_arr = np.asarray(wts)
        return (
            float(np.average(sub["oil_rate_stbd"], weights=w_arr)),
            float(np.average(sub["water_rate_stbd"], weights=w_arr)),
            float(np.average(sub["gas_rate_mscfd"], weights=w_arr)),
        )

    sp_v2 = np.array([span_match(v2_df, r) for _, r in wt_eval.iterrows()])

    print(f"\n=== BENCHMARK ON {len(wt_eval)} WELL TESTS (EXACT TEST HOURS & SPAN DAYS) ===")
    for i_p, (p_name, c_wt, c_hr) in enumerate([
        ("Oil (STB/D)", "oil_rate_stbd", "oil_hr"),
        ("Water (STB/D)", "water_rate_stbd", "wat_hr"),
        ("Gas (Mscf/d)", "gas_rate_mscfd", "gas_hr"),
    ]):
        yt = wt_eval[c_wt].to_numpy()
        s_hr = calc_metrics(yt, hr_tp_df[c_hr].to_numpy())
        s_v2 = calc_metrics(yt, sp_v2[:, i_p])
        print(f"{p_name:14s} | V2 Refined (Exact Hr)  : RMSE={s_hr['RMSE']:6.2f} | R2={s_hr['R2']:.4f} | WAPE={s_hr['WAPE']:.2f}% | MAPE={s_hr['MAPE']:.2f}%")
        print(f"{'':14s} | V2 Refined (Span Day)  : RMSE={s_v2['RMSE']:6.2f} | R2={s_v2['R2']:.4f} | WAPE={s_v2['WAPE']:.2f}% | MAPE={s_v2['MAPE']:.2f}%")

    # Coupled PTA Re-Evaluation with V2 hourly pre-shut-in oil rates (dynamic cum-oil lookup!)
    sys.path.insert(0, str(OUT_DIR))
    from export_horner_cases import load_shutins_and_data, _fit_line, _horner_props
    from export_smoothed_stable_bourdet_horner import (
        smooth_raw_1min_pressure,
        smooth_log_grid_and_bourdet,
        find_most_stable_bourdet_interval,
    )

    si_df, hf, _, _, _ = load_shutins_and_data()
    pta_rows, pta_sub_rows = [], []
    print("\n=== COUPLED PTA RESULTS USING V2 REFINED STACKED VFM OIL RATES ===")
    for _, r in si_df.iterrows():
        w, t0, t1, dur = r["well_name"], r["t_shut"], r["t_open"], float(r["duration_hr"])
        pre_hr = hr_preds[
            (hr_preds["well_name"] == w)
            & (hr_preds["hour"] >= t0.floor("h") - pd.Timedelta(hours=24))
            & (hr_preds["hour"] < t0.floor("h"))
            & (hr_preds["uptime_frac"] >= 0.5)
        ]
        qo_new = float((pre_hr["oil_rate_stbd"] / pre_hr["uptime_frac"]).median())
        dw = v2_df[v2_df["well_name"] == w].sort_values("date")
        day_shut = t0.floor("D")
        cum_since_start = float(dw[dw["date"] < day_shut]["oil_rate_stbd"].sum()) + qo_new * ((t0 - day_shut).total_seconds() / 86400.0)
        tp_hr = compute_producing_time_hr(w, t0, qo_new, comp, cum_since_start, dataset_start=ref_date)

        df_w = hf[w]
        pre_3h = df_w[(df_w["timestamp"] >= t0 - pd.Timedelta(hours=3)) & (df_w["timestamp"] <= t0)]["dh_pressure_psia"]
        pwf = float(pre_3h.median())
        post = df_w[(df_w["timestamp"] > t0) & (df_w["timestamp"] <= t1)]
        dt = (post["timestamp"] - t0).dt.total_seconds().to_numpy() / 3600.0
        pws_raw = post["dh_pressure_psia"].to_numpy(dtype=float)

        pws_sm = smooth_raw_1min_pressure(dt, pws_raw)
        tb, _, pb_sm, der = smooth_log_grid_and_bourdet(dt, pws_sm, pwf, tp_hr)
        win = find_most_stable_bourdet_interval(tb, pb_sm, der, tp_hr)
        sel = (dt >= win["dt_start"]) & (dt <= win["dt_end"])
        x_1m = np.log10((tp_hr + dt[sel]) / dt[sel])
        m_h, p_star, _ = _fit_line(x_1m, pws_sm[sel])
        props = _horner_props(m_h, p_star, pwf, tp_hr, qo_new, pvt.loc[w])

        print(f"{w} | Qo_24h={qo_new:7.1f} STB/D | m={props['m_psi_cycle']:6.2f} | k={props['k_md']:6.2f} md | S={props['skin']:6.3f} | dP_skin={props['dP_skin']:7.2f} psi")
        pta_rows.append({
            "well_name": w, "t_shut": str(t0), "duration_hr": dur,
            "m_psi_cycle": round(props["m_psi_cycle"], 4), "k_md": round(props["k_md"], 4),
            "skin": round(props["skin"], 4), "dP_skin": round(props["dP_skin"], 4),
        })
        pta_sub_rows.append({
            "well_name": w, "shut_in_time": str(t0),
            "permeability_md": round(props["k_md"], 4), "skin": round(props["skin"], 4),
            "delta_p_skin_psi": round(props["dP_skin"], 4), "horner_slope": round(props["m_psi_cycle"], 4),
        })

    pd.DataFrame(pta_rows).to_csv(OUT_DIR / "pta_coupled_phase_decomposed_stacked_v2.csv", index=False)
    pd.DataFrame(pta_rows).to_excel(OUT_DIR / "pta_coupled_phase_decomposed_stacked_v2.xlsx", index=False)
    pd.DataFrame(pta_sub_rows).to_csv(OUT_DIR / "pta_coupled_phase_decomposed_stacked_v2_export_schema.csv", index=False)
    print(f"Saved V2 deliverables ({len(vfm_deliv)} well-days) & Coupled PTA deliverables.")


if __name__ == "__main__":
    run_refined_phase_decomposed_vfm()
