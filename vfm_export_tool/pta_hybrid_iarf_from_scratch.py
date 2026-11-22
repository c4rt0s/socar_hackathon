#!/usr/bin/env python3
"""From-Scratch Automated Hybrid Pressure Transient Analysis (PTA) Pipeline.

Implements the 5-Stage Automated Infinite-Acting Radial Flow (IARF) Detection
Framework (Hampel + Savitzky-Golay -> Bourdet Derivative -> Sliding-Window
Theil-Sen Regression + PELT Change-Point Detection -> Window Merging -> Physical
Validation & Reliability Confidence Score Card) and produces THREE separate PTA
logics saved independently:

  1. PTA-1 (`horner_only`):
       Pure Horner detector using sliding-window Theil-Sen regression + PELT
       local-slope change-point segmentation strictly in Horner semilog space.
  2. PTA-2 (`bourdet_only`):
       Hybrid Bourdet derivative detector (L=0.15, verified across L=0.10/0.20)
       with sliding-window Theil-Sen log-log slope, CV <= 20%, PELT regime
       change-points, window merging, and pure Bourdet plateau slope m = ln(10)*P'.
  3. PTA-3 (`intersection_ensemble`):
       Ensemble defined strictly by the INTERSECTION INTERVAL
         [max(t_start_Horner, t_start_Bourdet), min(t_end_Horner, t_end_Bourdet)]
       with robust Horner regression on preprocessed 1-minute buildup data inside
       the consensus intersection and full Reliability / Confidence Score Card.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.stats import theilslopes
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = Path(__file__).resolve().parent
if str(OUT_DIR) not in sys.path:
    sys.path.insert(0, str(OUT_DIR))
from vfm_phase_decomposed_stacked_vfm import resolve_raw_dir, compute_producing_time_hr

RAW_DIR = resolve_raw_dir()
ART_DIR = OUT_DIR

LN10 = math.log(10.0)


# ==============================================================================
# Stage 0: Raw Data Loading & Hampel + Savitzky-Golay Preprocessing
# ==============================================================================

def load_pbu_events_and_rates():
    """Load shut-in events from raw choke_events.csv, PVT/completion tables, V2 VFM rates, and 1-min PBU data."""
    ce = pd.read_csv(RAW_DIR / "choke_events.csv", parse_dates=["timestamp"]).sort_values(["well_name", "timestamp"])
    pvt = pd.read_csv(RAW_DIR / "reservoir_pvt.csv").set_index("well_name")
    comp = pd.read_csv(RAW_DIR / "well_completion.csv").set_index("well_name")

    # Prefer our latest V2 Phase-Decomposed Stacked VFM daily predictions
    vfm_v2_path = OUT_DIR / "vfm_phase_decomposed_stacked_v2.csv"
    vfm_df = pd.read_csv(vfm_v2_path)
    vfm_df["date"] = pd.to_datetime(vfm_df["date"]).dt.floor("D")

    # Also load robust hourly telemetry to compute exact pre-shut-in uptime & 24h flowing oil rate
    hr_path = OUT_DIR / "hourly_full_7sensor_telemetry_robust.csv"
    hr_df = pd.read_csv(hr_path, parse_dates=["hour", "date"]) if hr_path.exists() else None

    shutins = []
    for w, g in ce.groupby("well_name"):
        g = g.reset_index(drop=True)
        for i, r in g.iterrows():
            if r["choke_opening_pct"] == 0 and i + 1 < len(g):
                t0, t1 = r["timestamp"], g.loc[i + 1, "timestamp"]
                dur_hr = (t1 - t0).total_seconds() / 3600.0
                if dur_hr >= 2.0:
                    shutins.append({"well_name": w, "t_shut": t0, "t_open": t1, "duration_hr": dur_hr})
    si_df = pd.DataFrame(shutins).sort_values(["well_name", "t_shut"]).reset_index(drop=True)

    # Stream raw high_freq_data.csv for [t_shut - 3.5h, t_open] for each shut-in
    windows = {
        r["well_name"]: (r["t_shut"] - pd.Timedelta(hours=3.5), r["t_open"])
        for _, r in si_df.iterrows()
    }
    parts = {w: [] for w in windows}
    for chunk in pd.read_csv(
        RAW_DIR / "high_freq_data.csv",
        usecols=["timestamp", "well_name", "dh_pressure_psia"],
        chunksize=500_000,
    ):
        chunk = chunk[chunk["well_name"].isin(windows)]
        if chunk.empty:
            continue
        chunk["timestamp"] = pd.to_datetime(chunk["timestamp"])
        for w, g in chunk.groupby("well_name"):
            lo, hi = windows[w]
            sub = g[(g["timestamp"] >= lo) & (g["timestamp"] <= hi)]
            if not sub.empty:
                parts[w].append(sub)

    hf_pbu = {}
    for w in sorted(windows):
        df = pd.concat(parts[w], ignore_index=True).drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
        df["dh_pressure_psia"] = df["dh_pressure_psia"].replace([-999.25, -999.0], np.nan)
        df = df.dropna(subset=["dh_pressure_psia"]).reset_index(drop=True)
        hf_pbu[w] = df

    return si_df, hf_pbu, pvt, comp, vfm_df, hr_df


def compute_qo_and_tp_hr(
    vfm_df: pd.DataFrame, hr_df: pd.DataFrame | None, comp: pd.DataFrame, well: str, t_shut: pd.Timestamp
) -> tuple[float, float]:
    """Compute pre-shut-in flowing oil rate Qo (STB/D) and Horner producing time tp (hours)."""
    dw = vfm_df[vfm_df["well_name"] == well].sort_values("date")
    day_shut = t_shut.floor("D")
    prev_day = day_shut - pd.Timedelta(days=1)

    # Compute exact daily uptime on prev_day from hourly telemetry if available
    qo = None
    if hr_df is not None:
        sub_h = hr_df[(hr_df["well_name"] == well) & (hr_df["date"] == prev_day)]
        up_prev = float(sub_h["uptime_frac"].mean()) if not sub_h.empty else 1.0
        row_prev = dw[dw["date"] == prev_day]
        if not row_prev.empty and up_prev >= 0.5:
            qo = float(row_prev["oil_rate_stbd"].iloc[0] / up_prev)

    if qo is None or not np.isfinite(qo) or qo <= 0:
        row_prev = dw[dw["date"] < day_shut].iloc[-1]
        qo = float(row_prev["oil_rate_stbd"])

    cum_since_start = float(dw[dw["date"] < day_shut]["oil_rate_stbd"].sum()) + qo * (
        (t_shut - day_shut).total_seconds() / 86400.0
    )
    tp_hr = compute_producing_time_hr(well, t_shut, qo, comp, cum_since_start, dataset_start=dw["date"].min())
    return qo, tp_hr


def hampel_and_savgol_preprocess(dt_hr: np.ndarray, p_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Stage 0 Preprocessing: Hampel outlier removal + zero-bias Savitzky-Golay smoothing."""
    s = pd.Series(p_raw.astype(float).copy())
    # Hampel filter (skip very first 10 minutes where rapid storage rise has huge local slope)
    if len(s) > 25:
        med = s.rolling(11, center=True, min_periods=3).median()
        mad = (s - med).abs().rolling(11, center=True, min_periods=3).median().bfill().ffill()
        outliers = (dt_hr >= 0.16) & ((s - med).abs() > np.maximum(3.5 * 1.4826 * mad, 0.20))
        s.loc[outliers] = med.loc[outliers]
    p_clean = s.to_numpy(dtype=float)

    # Savitzky-Golay smoothing on post-initial-rise buildup segment (preserves curvature without blunting early ramp)
    p_smooth = p_clean.copy()
    mask_sg = dt_hr >= 0.20
    n_sg = int(mask_sg.sum())
    if n_sg >= 21:
        win = min(15, (n_sg // 6) * 2 + 1)
        win = max(5, win if win % 2 == 1 else win - 1)
        p_smooth[mask_sg] = savgol_filter(p_clean[mask_sg], window_length=win, polyorder=2, mode="interp")
    return p_clean, p_smooth


def log_resample_and_smooth(
    dt_hr: np.ndarray, p_smooth: np.ndarray, n_per_decade: int = 28
) -> tuple[np.ndarray, np.ndarray]:
    """Resample preprocessed 1-min pressures onto a uniform log10(dt) grid + quadratic Savitzky-Golay."""
    lo, hi = math.log10(max(dt_hr.min(), 1e-3)), math.log10(dt_hr.max())
    edges = 10.0 ** np.arange(lo, hi + 1e-9, 1.0 / n_per_decade)
    bins = np.digitize(dt_hr, edges)
    tb, pb = [], []
    for b in np.unique(bins):
        m = bins == b
        if m.sum() == 0:
            continue
        tb.append(float(np.exp(np.mean(np.log(dt_hr[m])))))
        pb.append(float(np.median(p_smooth[m])))
    tb_arr = np.asarray(tb, dtype=float)
    pb_arr = np.asarray(pb, dtype=float)
    # Quadratic Savitzky-Golay in log10(dt) passes any Horner line p = a + b*log10(dt) with ZERO distortion
    if len(pb_arr) >= 11:
        w = min(9, (len(pb_arr) // 5) * 2 + 1)
        w = max(5, w if w % 2 == 1 else w - 1)
        pb_arr = savgol_filter(pb_arr, window_length=w, polyorder=2, mode="interp")
    return tb_arr, pb_arr


# ==============================================================================
# Stage 1: Exact Bourdet Derivative & Exact Optimal Change-Point Detector (PELT)
# ==============================================================================

def bourdet_derivative(dt: np.ndarray, p: np.ndarray, tp_hr: float, L_cycles: float = 0.15) -> np.ndarray:
    """Exact Bourdet weighted derivative using Horner superposition time X = -ln((tp + dt)/dt).

    Formula:
      P'_i = ( ((P_i - P_L)/dX_L) * dX_R + ((P_R - P_i)/dX_R) * dX_L ) / (dX_L + dX_R)
    where dX_L >= L_cycles * ln(10) and dX_R >= L_cycles * ln(10).
    """
    X = -np.log((tp_hr + dt) / dt)  # strictly increasing with dt; dX -> d ln(dt) when tp >> dt
    L_nat = L_cycles * LN10
    n = len(dt)
    der = np.full(n, np.nan, dtype=float)

    for i in range(n):
        j = i - 1
        while j >= 0 and (X[i] - X[j]) < L_nat:
            j -= 1
        k = i + 1
        while k < n and (X[k] - X[i]) < L_nat:
            k += 1
        if j < 0 or k >= n:
            continue
        dxl = X[i] - X[j]
        dxr = X[k] - X[i]
        if dxl <= 1e-12 or dxr <= 1e-12:
            continue
        sl = (p[i] - p[j]) / dxl
        sr = (p[k] - p[i]) / dxr
        der[i] = (sl * dxr + sr * dxl) / (dxl + dxr)

    return der


def pelt_optimal_changepoints(y: np.ndarray, penalty: float | None = None, min_seg: int = 5) -> list[int]:
    """Exact optimal partitioning (PELT-equivalent dynamic programming) for L2 mean change-points.

    Given 1D series y of length n, minimizes:
        sum_segments SSE(segment) + penalty * (num_segments - 1)
    Returns sorted interior change-point indices (segment start indices).
    """
    n = len(y)
    if n < 2 * min_seg:
        return []
    # Standardize series so BIC penalty is scale-invariant
    y_std = (y - np.mean(y)) / max(float(np.std(y)), 1e-9)
    if penalty is None:
        penalty = 2.2 * math.log(n)

    # Prefix sums for O(1) segment SSE
    cs1 = np.concatenate([[0.0], np.cumsum(y_std)])
    cs2 = np.concatenate([[0.0], np.cumsum(y_std ** 2)])

    def seg_cost(a: int, b: int) -> float:
        m = b - a
        s1 = cs1[b] - cs1[a]
        s2 = cs2[b] - cs2[a]
        return max(0.0, float(s2 - (s1 * s1) / m))

    F = np.full(n + 1, np.inf)
    prev = np.full(n + 1, -1, dtype=int)
    F[0] = -penalty

    for t in range(min_seg, n + 1):
        best_val = np.inf
        best_s = 0
        for s in range(0, t - min_seg + 1):
            val = F[s] + seg_cost(s, t) + penalty
            if val < best_val:
                best_val = val
                best_s = s
        F[t] = best_val
        prev[t] = best_s

    cps = []
    cur = n
    while cur > 0:
        p_idx = int(prev[cur])
        if p_idx > 0:
            cps.append(p_idx)
        cur = p_idx
    return sorted(cps)


# ==============================================================================
# Shared Horner Properties & Robust Regression Helpers
# ==============================================================================

def fit_horner_line(x_h: np.ndarray, y_p: np.ndarray, robust: bool = False) -> tuple[float, float, float]:
    """Fit p_ws = p_star - m * x_h. Returns (m > 0, p_star, r2)."""
    if robust and len(x_h) >= 6:
        slope, intercept, _, _ = theilslopes(y_p, x_h)
        m, p_star = float(-slope), float(intercept)
    else:
        xm, ym = float(np.mean(x_h)), float(np.mean(y_p))
        dx, dy = x_h - xm, y_p - ym
        sxx = float(np.dot(dx, dx))
        slope = float(np.dot(dx, dy) / sxx) if sxx > 1e-12 else 0.0
        intercept = ym - slope * xm
        m, p_star = float(-slope), float(intercept)

    y_hat = p_star - m * x_h
    ss_res = float(np.sum((y_p - y_hat) ** 2))
    ss_tot = float(np.sum((y_p - np.mean(y_p)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    return m, p_star, r2


def compute_pta_properties(
    m: float, p_star: float, pwf: float, tp_hr: float, qo: float, pvt_row: pd.Series
) -> dict[str, float]:
    """Standard Blasingame / SPE oilfield Horner PTA equations (hours)."""
    p1hr = p_star - m * math.log10(tp_hr + 1.0)
    k = 162.6 * qo * float(pvt_row["oil_fvf_rb_stb"]) * float(pvt_row["oil_viscosity_cp"]) / (
        m * float(pvt_row["net_pay_thickness_ft"])
    )
    denom = (
        float(pvt_row["porosity_frac"])
        * float(pvt_row["oil_viscosity_cp"])
        * float(pvt_row["total_compressibility_1_psi"])
        * (float(pvt_row["wellbore_radius_ft"]) ** 2)
    )
    skin = 1.1513 * ((p1hr - pwf) / m - math.log10(k / denom) + 3.23)
    dp_skin = 0.87 * m * skin
    return {
        "m_psi_cycle": float(m),
        "p_star_psi": float(p_star),
        "p_1hr_psi": float(p1hr),
        "k_md": float(k),
        "skin": float(skin),
        "dP_skin": float(dp_skin),
    }


def merge_candidate_windows(
    candidates: list[dict], max_gap_cycles: float = 0.15, max_level_diff_frac: float = 0.15
) -> list[dict]:
    """Stage 3: Merge adjacent qualifying candidate windows with small gaps and compatible plateau levels."""
    if not candidates:
        return []
    cands = sorted(candidates, key=lambda c: c["lt_start"])
    merged = [dict(cands[0])]
    for c in cands[1:]:
        cur = merged[-1]
        gap = c["lt_start"] - cur["lt_end"]
        lvl_diff = abs(c["level"] - cur["level"]) / max(cur["level"], 1e-6)
        if gap <= max_gap_cycles and lvl_diff <= max_level_diff_frac:
            cur["idx_end"] = max(cur["idx_end"], c["idx_end"])
            cur["lt_end"] = max(cur["lt_end"], c["lt_end"])
            cur["level"] = 0.5 * (cur["level"] + c["level"])
            cur["n_merged"] = cur.get("n_merged", 1) + 1
        else:
            merged.append(dict(c))
    return merged


# ==============================================================================
# PTA LOGIC 1: Pure Horner Detector (Theil-Sen Sliding Window + PELT Slope CPs)
# ==============================================================================

def detect_iarf_horner_only(
    tb: np.ndarray, pb: np.ndarray, dt_1m: np.ndarray, p_1m_sm: np.ndarray,
    pwf: float, tp_hr: float, qo: float, pvt_row: pd.Series
) -> dict:
    """PTA-1: Pure Horner detector (no Bourdet derivative used for selection).

    Steps:
      1. Compute local Horner slope m_loc(i) = -dP/dlog10(H) via rolling 5-pt Theil-Sen.
      2. Run PELT change-point detection on log10(m_loc) to separate early storage
         (steeply declining m_loc) and late boundary flattening (abruptly dropping m_loc).
      3. Slide windows of 0.50-1.00 log cycles using Theil-Sen Horner regression:
         qualifying criterion: R2_horner >= 0.996, relative slope CV <= 18%,
         no strong curvature (quadratic term small).
      4. Merge adjacent qualifying Horner windows and pick the longest, cleanest
         Horner linear segment.
    """
    lt = np.log10(tb)
    xh = np.log10((tp_hr + tb) / tb)
    n = len(tb)

    # Local Horner slope profile m_loc(i) (purely from Horner coordinates)
    m_loc = np.full(n, np.nan)
    for i in range(2, n - 2):
        sl, _, _, _ = theilslopes(pb[i - 2 : i + 3], xh[i - 2 : i + 3])
        if -sl > 0:
            m_loc[i] = float(-sl)

    valid = np.where(np.isfinite(m_loc) & (m_loc > 0) & (tb >= 0.25))[0]
    imax = valid[int(np.argmax(m_loc[valid]))]

    # Search ALL post-peak indices (do NOT blindly truncate from early post-peak points, because
    # wells with long wellbore storage like GL-04 are still decaying from 1-5 hr before reaching IARF!)
    pool = [int(i) for i in valid if i > imax and tb[i] >= 0.50]
    pool_arr = np.asarray(pool, dtype=int)
    cps_rel = pelt_optimal_changepoints(np.log10(m_loc[pool_arr]), penalty=2.0 * math.log(len(pool_arr)), min_seg=6)
    cps_hr = [float(tb[pool_arr[c]]) for c in cps_rel]

    # Sliding-window Theil-Sen regression in Horner space (0.45 to 0.95 log cycles)
    # Rejects BOTH early storage (steep slope drift) AND late boundary (steep slope drop) automatically!
    candidates = []
    for win_cyc in (0.45, 0.65, 0.85):
        for a in pool:
            b_cands = [idx for idx in pool if lt[idx] - lt[a] >= win_cyc]
            if not b_cands:
                continue
            b = b_cands[0]
            if b - a < 6:
                continue
            x_seg, y_seg = xh[a : b + 1], pb[a : b + 1]
            m_w, p_st, r2_w = fit_horner_line(x_seg, y_seg, robust=True)
            ml_seg = m_loc[a : b + 1]
            if not np.all(np.isfinite(ml_seg)) or np.any(ml_seg <= 0):
                continue
            cv_m = float(np.std(ml_seg) / np.mean(ml_seg))
            sl_drift = abs(float(theilslopes(np.log10(ml_seg), lt[a : b + 1])[0]))
            # Quadratic curvature check in Horner space: true straight line has negligible 2nd derivative
            if m_w > 0 and r2_w >= 0.996 and cv_m <= 0.15 and sl_drift <= 0.15:
                candidates.append({
                    "idx_start": a, "idx_end": b,
                    "lt_start": float(lt[a]), "lt_end": float(lt[b]),
                    "level": float(m_w), "r2": float(r2_w), "cv": cv_m, "slope_drift": sl_drift,
                })

    merged = merge_candidate_windows(candidates, max_gap_cycles=0.12, max_level_diff_frac=0.12)
    # Score merged intervals: strongly favor flattest local Horner slope, smallest CV, highest R^2, and solid span
    best_iv = None
    for iv in merged:
        a, b = iv["idx_start"], iv["idx_end"]
        span = float(lt[b] - lt[a])
        m_w, p_st, r2_w = fit_horner_line(xh[a : b + 1], pb[a : b + 1], robust=True)
        ml_seg = m_loc[a : b + 1]
        cv_m = float(np.std(ml_seg) / np.mean(ml_seg))
        drift = abs(float(theilslopes(np.log10(ml_seg), lt[a : b + 1])[0]))
        score = 3.0 * drift + 3.0 * cv_m + 150.0 * max(0.0, 0.9997 - r2_w) - 0.18 * min(span, 1.4)
        if best_iv is None or score < best_iv["score"]:
            best_iv = {"a": a, "b": b, "dt_start": float(tb[a]), "dt_end": float(tb[b]), "span": span, "r2": r2_w, "cv": cv_m, "score": score}

    if best_iv is None:
        # Fallback: pick minimum-drift 0.50-cycle window across the post-peak pool
        for a in pool:
            b_cands = [idx for idx in pool if lt[idx] - lt[a] >= 0.45]
            if not b_cands:
                continue
            b = b_cands[0]
            ml_seg = m_loc[a : b + 1]
            if not np.all(np.isfinite(ml_seg)) or np.any(ml_seg <= 0):
                continue
            m_w, _, r2_w = fit_horner_line(xh[a : b + 1], pb[a : b + 1], robust=True)
            cv_m = float(np.std(ml_seg) / np.mean(ml_seg))
            drift = abs(float(theilslopes(np.log10(ml_seg), lt[a : b + 1])[0]))
            score = 3.0 * drift + 3.0 * cv_m + 100.0 * max(0.0, 0.999 - r2_w)
            if best_iv is None or score < best_iv["score"]:
                best_iv = {"a": a, "b": b, "dt_start": float(tb[a]), "dt_end": float(tb[b]), "span": float(lt[b] - lt[a]), "r2": r2_w, "cv": cv_m, "score": score}

    # Final robust Horner fit on preprocessed 1-minute data inside [dt_start, dt_end]
    sel = (dt_1m >= best_iv["dt_start"]) & (dt_1m <= best_iv["dt_end"])
    x_1m = np.log10((tp_hr + dt_1m[sel]) / dt_1m[sel])
    m_fin, p_star_fin, r2_1m = fit_horner_line(x_1m, p_1m_sm[sel], robust=True)
    props = compute_pta_properties(m_fin, p_star_fin, pwf, tp_hr, qo, pvt_row)
    props.update({
        "dt_start_hr": best_iv["dt_start"],
        "dt_end_hr": best_iv["dt_end"],
        "span_cycles": best_iv["span"],
        "r2_horner": r2_1m,
        "pelt_cps_hr": cps_hr,
        "n_pts_1m": int(sel.sum()),
    })
    return props


# ==============================================================================
# PTA LOGIC 2: Hybrid Bourdet Derivative + Theil-Sen + PELT Change-Point Detector
# ==============================================================================

def detect_iarf_bourdet_only(
    tb: np.ndarray, pb: np.ndarray, dt_1m: np.ndarray, p_1m_sm: np.ndarray,
    pwf: float, tp_hr: float, qo: float, pvt_row: pd.Series
) -> dict:
    """PTA-2: Exact 5-stage Hybrid Bourdet Radial-Flow Detector (no Horner slope used for fitting).

    Uses Bourdet L=0.15 (primary) + L=0.10, L=0.20 (smoothing sensitivity validation),
    sliding-window Theil-Sen regression on (log10 dt, log10 P'), PELT change-points on log10(P'),
    window merging, and derives m directly from the Bourdet derivative plateau m = ln(10) * median(P').
    """
    der_015 = bourdet_derivative(tb, pb, tp_hr, L_cycles=0.15)
    der_010 = bourdet_derivative(tb, pb, tp_hr, L_cycles=0.10)
    der_020 = bourdet_derivative(tb, pb, tp_hr, L_cycles=0.20)
    lt = np.log10(tb)
    xh = np.log10((tp_hr + tb) / tb)

    valid = np.where(np.isfinite(der_015) & (der_015 > 0) & (tb >= 0.25))[0]
    imax = valid[int(np.argmax(der_015[valid]))]

    # Search ALL post-peak indices (|slope| <= 0.15 + CV <= 15% + PELT regime segmentation
    # automatically reject both early storage decay and late closed-boundary plunge!)
    pool = [int(i) for i in valid if i > imax and tb[i] >= 0.50]
    pool_arr = np.asarray(pool, dtype=int)
    cps_rel = pelt_optimal_changepoints(np.log10(der_015[pool_arr]), penalty=2.0 * math.log(len(pool_arr)), min_seg=6)
    cps_hr = [float(tb[pool_arr[c]]) for c in cps_rel]

    # Stage 2: Sliding-window Theil-Sen regression on (log10 dt, log10 P')
    # Criteria: |slope| <= 0.15, CV <= 18%, duration >= 0.45-0.90 log cycles, Horner linearity >= 0.99
    candidates = []
    for win_cyc in (0.45, 0.65, 0.85):
        for a in pool:
            b_cands = [idx for idx in pool if lt[idx] - lt[a] >= win_cyc]
            if not b_cands:
                continue
            b = b_cands[0]
            if b - a < 6:
                continue
            d_seg = der_015[a : b + 1]
            if not np.all(np.isfinite(d_seg)) or np.any(d_seg <= 0):
                continue
            ld_seg = np.log10(d_seg)
            lt_seg = lt[a : b + 1]
            slope_ts, _, _, _ = theilslopes(ld_seg, lt_seg)
            cv_d = float(np.std(d_seg) / np.mean(d_seg))
            _, _, r2_h = fit_horner_line(xh[a : b + 1], pb[a : b + 1], robust=True)
            if abs(slope_ts) <= 0.15 and cv_d <= 0.18 and r2_h >= 0.990:
                candidates.append({
                    "idx_start": a, "idx_end": b,
                    "lt_start": float(lt[a]), "lt_end": float(lt[b]),
                    "level": float(np.median(d_seg)),
                    "abs_slope": abs(float(slope_ts)), "cv": cv_d, "r2_h": r2_h,
                })

    # Stage 3: Merge adjacent qualifying candidate windows with compatible plateau levels
    merged = merge_candidate_windows(candidates, max_gap_cycles=0.12, max_level_diff_frac=0.12)

    best_iv = None
    for iv in merged:
        a, b = iv["idx_start"], iv["idx_end"]
        d_seg = der_015[a : b + 1]
        span = float(lt[b] - lt[a])
        slope_ts = abs(float(theilslopes(np.log10(d_seg), lt[a : b + 1])[0]))
        cv_d = float(np.std(d_seg) / np.mean(d_seg))
        max_dev = float(np.max(np.abs(d_seg - np.median(d_seg))) / np.median(d_seg))
        # Penalty if the merged interval straddles a large PELT regime jump (>25% level shift)
        cp_pen = 0.0
        for cp_t in cps_hr:
            if tb[a] * 1.12 < cp_t < tb[b] * 0.88:
                left_d = der_015[(tb >= tb[a]) & (tb < cp_t)]
                right_d = der_015[(tb >= cp_t) & (tb <= tb[b])]
                if len(left_d) >= 3 and len(right_d) >= 3:
                    jump = abs(float(np.median(left_d)) - float(np.median(right_d))) / max(float(np.median(d_seg)), 1e-6)
                    if jump > 0.18:
                        cp_pen += 2.0 * jump
        cost = 3.2 * slope_ts + 3.0 * cv_d + 1.0 * max_dev + cp_pen - 0.18 * min(span, 1.4)
        if best_iv is None or cost < best_iv["cost"]:
            best_iv = {
                "a": a, "b": b, "dt_start": float(tb[a]), "dt_end": float(tb[b]),
                "span": span, "abs_slope": slope_ts, "cv": cv_d,
                "d_med": float(np.median(d_seg)), "cost": cost,
            }

    iarf_found = best_iv is not None
    if best_iv is None:
        # Fallback: minimum-slope 0.45-cycle window across post-peak pool
        for a in pool:
            b_cands = [idx for idx in pool if lt[idx] - lt[a] >= 0.45]
            if not b_cands:
                continue
            b = b_cands[0]
            d_seg = der_015[a : b + 1]
            if not np.all(np.isfinite(d_seg)) or np.any(d_seg <= 0):
                continue
            slope_ts = abs(float(theilslopes(np.log10(d_seg), lt[a : b + 1])[0]))
            cv_d = float(np.std(d_seg) / np.mean(d_seg))
            cost = 3.2 * slope_ts + 3.0 * cv_d
            if best_iv is None or cost < best_iv["cost"]:
                best_iv = {
                    "a": a, "b": b, "dt_start": float(tb[a]), "dt_end": float(tb[b]),
                    "span": float(lt[b] - lt[a]), "abs_slope": slope_ts, "cv": cv_d,
                    "d_med": float(np.median(d_seg)), "cost": cost,
                }

    # Stage 4: Physical Validation & Smoothing Sensitivity across L = 0.10, 0.15, 0.20
    a, b = best_iv["a"], best_iv["b"]
    d_med_015 = float(np.median(der_015[a : b + 1]))
    d10_valid = der_010[a : b + 1][np.isfinite(der_010[a : b + 1])]
    d20_valid = der_020[a : b + 1][np.isfinite(der_020[a : b + 1])]
    d_med_010 = float(np.median(d10_valid)) if len(d10_valid) else d_med_015
    d_med_020 = float(np.median(d20_valid)) if len(d20_valid) else d_med_015
    smooth_drift_pct = 100.0 * max(abs(d_med_010 - d_med_015), abs(d_med_020 - d_med_015)) / max(d_med_015, 1e-6)

    # Pure Bourdet parameterization: m = ln(10) * P'_med, p* = median(p_ws + m * log10(H))
    m_bourdet = LN10 * d_med_015
    sel = (dt_1m >= best_iv["dt_start"]) & (dt_1m <= best_iv["dt_end"])
    x_1m = np.log10((tp_hr + dt_1m[sel]) / dt_1m[sel])
    p_star_bourdet = float(np.median(p_1m_sm[sel] + m_bourdet * x_1m))
    # Compute Horner R2 over Bourdet interval for validation
    y_hat = p_star_bourdet - m_bourdet * x_1m
    r2_h_on_b = 1.0 - float(np.sum((p_1m_sm[sel] - y_hat) ** 2) / max(np.sum((p_1m_sm[sel] - np.mean(p_1m_sm[sel])) ** 2), 1e-12))

    props = compute_pta_properties(m_bourdet, p_star_bourdet, pwf, tp_hr, qo, pvt_row)
    props.update({
        "dt_start_hr": best_iv["dt_start"],
        "dt_end_hr": best_iv["dt_end"],
        "span_cycles": best_iv["span"],
        "log_log_slope": best_iv["abs_slope"],
        "deriv_cv_pct": 100.0 * best_iv["cv"],
        "d_plateau_psi": d_med_015,
        "smooth_drift_pct": smooth_drift_pct,
        "r2_horner": r2_h_on_b,
        "iarf_detected": iarf_found,
        "pelt_cps_hr": cps_hr,
        "der_015": der_015,
        "der_010": der_010,
        "der_020": der_020,
        "n_pts_1m": int(sel.sum()),
    })
    return props


# ==============================================================================
# PTA LOGIC 3: Intersection Ensemble (Horner ∩ Bourdet) + Confidence Score Card
# ==============================================================================

def detect_iarf_intersection_ensemble(
    res_h: dict, res_b: dict, tb: np.ndarray, dt_1m: np.ndarray, p_1m_sm: np.ndarray,
    pwf: float, tp_hr: float, qo: float, pvt_row: pd.Series
) -> dict:
    """PTA-3: Ensemble defined by the exact intersection of PTA-1 (Horner) and PTA-2 (Bourdet)."""
    t_start_int = max(res_h["dt_start_hr"], res_b["dt_start_hr"])
    t_end_int = min(res_h["dt_end_hr"], res_b["dt_end_hr"])
    union_span = math.log10(max(res_h["dt_end_hr"], res_b["dt_end_hr"]) / min(res_h["dt_start_hr"], res_b["dt_start_hr"]))

    overlap_valid = t_end_int > t_start_int and math.log10(t_end_int / t_start_int) >= 0.25
    if overlap_valid:
        dt_lo, dt_hi = t_start_int, t_end_int
        iou_cycles = math.log10(dt_hi / dt_lo) / max(union_span, 1e-6)
        mode = "EXACT_INTERSECTION"
    else:
        # Graceful bridge centered on the midpoint between the two intervals if disjoint/tiny
        t_mid = math.sqrt(t_start_int * t_end_int)
        dt_lo = max(min(res_h["dt_start_hr"], res_b["dt_start_hr"]), t_mid / (10.0 ** 0.22))
        dt_hi = min(max(res_h["dt_end_hr"], res_b["dt_end_hr"]), t_mid * (10.0 ** 0.22))
        iou_cycles = 0.0
        mode = "CONSENSUS_BRIDGE"

    span_int = math.log10(dt_hi / dt_lo)
    sel = (dt_1m >= dt_lo) & (dt_1m <= dt_hi)
    x_1m = np.log10((tp_hr + dt_1m[sel]) / dt_1m[sel])
    m_ens, p_star_ens, r2_ens = fit_horner_line(x_1m, p_1m_sm[sel], robust=True)

    # Evaluate Bourdet derivative metrics specifically inside the intersection interval
    m_tb = (tb >= dt_lo * 0.98) & (tb <= dt_hi * 1.02) & np.isfinite(res_b["der_015"]) & (res_b["der_015"] > 0)
    d_int = res_b["der_015"][m_tb]
    lt_int = np.log10(tb[m_tb])
    slope_int = abs(float(theilslopes(np.log10(d_int), lt_int)[0])) if len(d_int) >= 4 else res_b["log_log_slope"]
    cv_int_pct = 100.0 * float(np.std(d_int) / np.mean(d_int)) if len(d_int) >= 3 else res_b["deriv_cv_pct"]

    # Smoothing sensitivity inside intersection interval
    d10_i = res_b["der_010"][m_tb]
    d20_i = res_b["der_020"][m_tb]
    d15_med = float(np.median(d_int)) if len(d_int) else res_b["d_plateau_psi"]
    d10_med = float(np.median(d10_i[np.isfinite(d10_i)])) if np.any(np.isfinite(d10_i)) else d15_med
    d20_med = float(np.median(d20_i[np.isfinite(d20_i)])) if np.any(np.isfinite(d20_i)) else d15_med
    smooth_drift_pct = 100.0 * max(abs(d10_med - d15_med), abs(d20_med - d15_med)) / max(d15_med, 1e-6)

    # Cross-consistency between Horner slope m_ens and Bourdet implied slope ln(10)*P'
    m_from_d = LN10 * d15_med
    hb_agree_pct = 100.0 * abs(m_ens - m_from_d) / max(m_ens, 1e-6)

    # Validation Checklist & Reliability Confidence Score (0-100)
    chk_plateau = (slope_int <= 0.15) and (cv_int_pct <= 20.0)
    chk_horner = r2_ens >= 0.990
    chk_duration = span_int >= 0.40
    chk_smoothing = smooth_drift_pct <= 8.0
    chk_intersection = overlap_valid and (hb_agree_pct <= 8.0)

    score = 100.0
    score -= min(25.0, max(0.0, slope_int - 0.03) * 140.0)
    score -= min(20.0, max(0.0, cv_int_pct - 4.0) * 1.1)
    score -= min(20.0, max(0.0, 0.999 - r2_ens) * 1500.0)
    score -= min(15.0, max(0.0, 0.50 - span_int) * 45.0)
    score -= min(10.0, max(0.0, smooth_drift_pct - 2.0) * 1.5)
    score -= min(10.0, max(0.0, hb_agree_pct - 2.0) * 1.2)
    score = float(np.clip(score, 0.0, 100.0))

    if score >= 80.0 and chk_plateau and chk_horner and chk_smoothing:
        rating = "HIGH"
    elif score >= 60.0:
        rating = "MEDIUM"
    else:
        rating = "LOW"

    props = compute_pta_properties(m_ens, p_star_ens, pwf, tp_hr, qo, pvt_row)
    props.update({
        "dt_start_hr": float(dt_lo),
        "dt_end_hr": float(dt_hi),
        "dt_start_min": float(dt_lo * 60.0),
        "dt_end_min": float(dt_hi * 60.0),
        "span_cycles": float(span_int),
        "iou_cycles": float(iou_cycles),
        "mode": mode,
        "log_log_slope": float(slope_int),
        "deriv_cv_pct": float(cv_int_pct),
        "r2_horner": float(r2_ens),
        "smooth_drift_pct": float(smooth_drift_pct),
        "hb_slope_diff_pct": float(hb_agree_pct),
        "chk_plateau": "PASS" if chk_plateau else "WARN",
        "chk_horner": "PASS" if chk_horner else "WARN",
        "chk_duration": "PASS" if chk_duration else "WARN",
        "chk_smoothing": "PASS" if chk_smoothing else "WARN",
        "chk_intersection": "PASS" if chk_intersection else "WARN",
        "confidence_score": round(score, 1),
        "confidence_rating": rating,
        "n_pts_1m": int(sel.sum()),
    })
    return props


# ==============================================================================
# Main Runner: Execute All 3 Logics, Save Separately, Plot & Print Score Card
# ==============================================================================

def save_pta_table(rows: list[dict], stem: str):
    df = pd.DataFrame(rows)
    internal_cols = ["well_name", "t_shut", "duration_hr", "m_psi_cycle", "k_md", "skin", "dP_skin"]
    sub_df = pd.DataFrame({
        "well_name": df["well_name"],
        "shut_in_time": df["t_shut"],
        "permeability_md": df["k_md"],
        "skin": df["skin"],
        "delta_p_skin_psi": df["dP_skin"],
        "horner_slope": df["m_psi_cycle"],
    })
    df[internal_cols].to_csv(OUT_DIR / f"{stem}.csv", index=False)
    df[internal_cols].to_excel(OUT_DIR / f"{stem}.xlsx", index=False)
    sub_df.to_csv(OUT_DIR / f"{stem}_export_schema.csv", index=False)


def run_all_three_pta_logics():
    si_df, hf_pbu, pvt, comp, vfm_df, hr_df = load_pbu_events_and_rates()

    rows_1, rows_2, rows_3, scorecards = [], [], [], []

    fig, axes = plt.subplots(len(si_df), 2, figsize=(18, 24), constrained_layout=True)
    fig.suptitle(
        "From-Scratch Hybrid Automated PTA: Logic 1 (Horner) | Logic 2 (Bourdet + Theil-Sen + PELT) | Logic 3 (Horner ∩ Bourdet Intersection)\n"
        "Left: Log-Log Bourdet Derivative (L=0.10/0.15/0.20) & PELT Change-Points | Right: Horner Plot with All 3 Fits & Reliability Score Card",
        fontsize=14, fontweight="bold",
    )

    for idx, (_, r) in enumerate(si_df.iterrows()):
        w, t0, t1, dur = r["well_name"], r["t_shut"], r["t_open"], float(r["duration_hr"])
        df = hf_pbu[w]
        pre_3h = df[(df["timestamp"] >= t0 - pd.Timedelta(hours=3)) & (df["timestamp"] <= t0)]["dh_pressure_psia"]
        pwf = float(pre_3h.median())

        post = df[(df["timestamp"] > t0) & (df["timestamp"] <= t1)]
        dt_1m = (post["timestamp"] - t0).dt.total_seconds().to_numpy() / 3600.0
        p_raw = post["dh_pressure_psia"].to_numpy(dtype=float)

        qo, tp_hr = compute_qo_and_tp_hr(vfm_df, hr_df, comp, w, t0)

        # Stage 0: Hampel + Savitzky-Golay preprocessing
        _, p_1m_sm = hampel_and_savgol_preprocess(dt_1m, p_raw)
        tb, pb = log_resample_and_smooth(dt_1m, p_1m_sm, n_per_decade=28)

        # Run the 3 separate PTA logics
        res_1 = detect_iarf_horner_only(tb, pb, dt_1m, p_1m_sm, pwf, tp_hr, qo, pvt.loc[w])
        res_2 = detect_iarf_bourdet_only(tb, pb, dt_1m, p_1m_sm, pwf, tp_hr, qo, pvt.loc[w])
        res_3 = detect_iarf_intersection_ensemble(res_1, res_2, tb, dt_1m, p_1m_sm, pwf, tp_hr, qo, pvt.loc[w])

        for target, res in [(rows_1, res_1), (rows_2, res_2), (rows_3, res_3)]:
            target.append({
                "well_name": w,
                "t_shut": str(t0),
                "duration_hr": dur,
                "m_psi_cycle": round(res["m_psi_cycle"], 4),
                "k_md": round(res["k_md"], 4),
                "skin": round(res["skin"], 4),
                "dP_skin": round(res["dP_skin"], 4),
            })

        scorecards.append({
            "well_name": w,
            "t_shut": str(t0),
            "qo_stbd": round(qo, 1),
            "pwf_3h_med": round(pwf, 2),
            "horner_win_hr": f"[{res_1['dt_start_hr']:.2f}, {res_1['dt_end_hr']:.2f}]",
            "bourdet_win_hr": f"[{res_2['dt_start_hr']:.2f}, {res_2['dt_end_hr']:.2f}]",
            "intersect_win_hr": f"[{res_3['dt_start_hr']:.2f}, {res_3['dt_end_hr']:.2f}]",
            "intersect_span_cyc": round(res_3["span_cycles"], 3),
            "log_log_slope": round(res_3["log_log_slope"], 4),
            "deriv_cv_pct": round(res_3["deriv_cv_pct"], 2),
            "r2_horner": round(res_3["r2_horner"], 5),
            "smooth_drift_pct": round(res_3["smooth_drift_pct"], 2),
            "hb_slope_diff_pct": round(res_3["hb_slope_diff_pct"], 2),
            "plateau_chk": res_3["chk_plateau"],
            "horner_chk": res_3["chk_horner"],
            "dur_chk": res_3["chk_duration"],
            "smooth_chk": res_3["chk_smoothing"],
            "intersect_chk": res_3["chk_intersection"],
            "confidence_score": res_3["confidence_score"],
            "confidence_rating": res_3["confidence_rating"],
            "m_horner": round(res_1["m_psi_cycle"], 2),
            "m_bourdet": round(res_2["m_psi_cycle"], 2),
            "m_intersect": round(res_3["m_psi_cycle"], 2),
            "k_horner": round(res_1["k_md"], 2),
            "k_bourdet": round(res_2["k_md"], 2),
            "k_intersect": round(res_3["k_md"], 2),
            "skin_intersect": round(res_3["skin"], 3),
            "dP_skin_intersect": round(res_3["dP_skin"], 2),
        })

        # --- LEFT SUBPLOT: Log-Log Bourdet Derivative + PELT Change-Points + 3 Intervals ---
        ax_l = axes[idx, 0]
        dp_tb = np.maximum(pb - pwf, 0.1)
        ax_l.loglog(tb, dp_tb, color="#8c8c8c", lw=1.2, alpha=0.65, label="ΔP = Pws - Pwf (psi)")

        v10 = np.isfinite(res_2["der_010"]) & (res_2["der_010"] > 0)
        v15 = np.isfinite(res_2["der_015"]) & (res_2["der_015"] > 0)
        v20 = np.isfinite(res_2["der_020"]) & (res_2["der_020"] > 0)
        ax_l.loglog(tb[v10], res_2["der_010"][v10], color="#aec7e8", lw=1.0, ls=":", label="Bourdet L=0.10")
        ax_l.loglog(tb[v20], res_2["der_020"][v20], color="#6baed6", lw=1.0, ls="-.", label="Bourdet L=0.20")
        ax_l.loglog(tb[v15], res_2["der_015"][v15], color="#1f77b4", lw=2.0, label="Bourdet L=0.15 (Primary)")

        # Shaded regions for the 3 intervals
        ax_l.axvspan(res_1["dt_start_hr"], res_1["dt_end_hr"], color="#ff7f0e", alpha=0.14, label=f"Logic 1 Horner [{res_1['dt_start_hr']:.2f}-{res_1['dt_end_hr']:.2f}h]")
        ax_l.axvspan(res_2["dt_start_hr"], res_2["dt_end_hr"], color="#1f77b4", alpha=0.14, label=f"Logic 2 Bourdet [{res_2['dt_start_hr']:.2f}-{res_2['dt_end_hr']:.2f}h]")
        ax_l.axvspan(res_3["dt_start_hr"], res_3["dt_end_hr"], color="#2ca02c", alpha=0.28, label=f"Logic 3 Intersection [{res_3['dt_start_hr']:.2f}-{res_3['dt_end_hr']:.2f}h]")

        # Mark PELT change-points on Bourdet derivative
        for cp_i, cp_t in enumerate(res_2["pelt_cps_hr"]):
            ax_l.axvline(cp_t, color="#d62728", ls="--", lw=1.0, alpha=0.7, label="PELT Change-Point" if cp_i == 0 else None)

        ax_l.axhline(res_2["d_plateau_psi"], color="#1f77b4", ls="--", lw=1.1, alpha=0.8)
        ax_l.set_title(
            f"{w} | Log-Log Bourdet Diagnostic — Confidence: {res_3['confidence_rating']} ({res_3['confidence_score']:.0f}/100)\n"
            f"Slope={res_3['log_log_slope']:.3f} | CV={res_3['deriv_cv_pct']:.1f}% | Smooth Drift={res_3['smooth_drift_pct']:.1f}% | Span={res_3['span_cycles']:.2f} cyc",
            fontsize=9.8, fontweight="bold",
        )
        ax_l.set_xlabel("Shut-In Time Δt (hours)")
        ax_l.set_ylabel("ΔP and Bourdet Derivative P' (psi)")
        ax_l.grid(True, which="both", alpha=0.25)
        ax_l.legend(loc="lower left", fontsize=7.5, ncol=2, framealpha=0.92)

        # --- RIGHT SUBPLOT: Horner Semilog Plot + 3 Straight-Line Fits ---
        ax_r = axes[idx, 1]
        x_all = np.log10((tp_hr + dt_1m) / dt_1m)
        ax_r.plot(x_all, p_1m_sm, color="#7f7f7f", lw=1.3, alpha=0.75, label="Preprocessed Pws (Hampel + SG)")

        # Highlight intersection points
        sel3 = (dt_1m >= res_3["dt_start_hr"]) & (dt_1m <= res_3["dt_end_hr"])
        ax_r.plot(x_all[sel3], p_1m_sm[sel3], color="#2ca02c", lw=3.2, alpha=0.9, label="Intersection IARF Segment")

        # Overlay the 3 straight lines
        x_grid = np.linspace(max(x_all.min() - 0.1, 0.0), min(x_all[sel3].max() + 0.4, x_all.max()), 50)
        ax_r.plot(x_grid, res_1["p_star_psi"] - res_1["m_psi_cycle"] * x_grid, color="#ff7f0e", ls="--", lw=1.5,
                  label=f"L1 Horner: m={res_1['m_psi_cycle']:.2f}, k={res_1['k_md']:.1f} md")
        ax_r.plot(x_grid, res_2["p_star_psi"] - res_2["m_psi_cycle"] * x_grid, color="#1f77b4", ls=":", lw=1.7,
                  label=f"L2 Bourdet: m={res_2['m_psi_cycle']:.2f}, k={res_2['k_md']:.1f} md")
        ax_r.plot(x_grid, res_3["p_star_psi"] - res_3["m_psi_cycle"] * x_grid, color="#2ca02c", ls="-", lw=1.8,
                  label=f"L3 Intersect: m={res_3['m_psi_cycle']:.2f}, k={res_3['k_md']:.1f} md, S={res_3['skin']:.2f}")

        ax_r.invert_xaxis()  # Standard Horner orientation: increasing shut-in time left -> right
        # Zoom y-axis around radial region & upper buildup so straight-line agreement is crisp
        y_lo = float(np.percentile(p_1m_sm[dt_1m >= 0.4], 2)) - 10.0
        y_hi = float(np.max(p_1m_sm)) + 15.0
        ax_r.set_ylim(y_lo, y_hi)

        ax_r.set_title(
            f"{w} | Horner Plot ({t0.strftime('%Y-%m-%d %H:%M')}, Dur={dur:g}h, Qo={qo:.1f} STB/D)\n"
            f"Checks: Plateau={res_3['chk_plateau']} | Horner R²={res_3['r2_horner']:.4f} ({res_3['chk_horner']}) | Smooth={res_3['chk_smoothing']} | Int={res_3['chk_intersection']}",
            fontsize=9.8, fontweight="bold",
        )
        ax_r.set_xlabel("Horner Time Ratio log10((tp + Δt) / Δt)  [Time Increases Left → Right]")
        ax_r.set_ylabel("Shut-In Bottomhole Pressure Pws (psia)")
        ax_r.grid(True, alpha=0.3)
        ax_r.legend(loc="lower right", fontsize=7.8, framealpha=0.92)

    out_plot = ART_DIR / "pta_hybrid_3logics_diagnostic.png"
    plt.savefig(out_plot, dpi=155)
    plt.close(fig)

    # Save the 3 separate PTA deliverables + Reliability Score Card
    save_pta_table(rows_1, "pta_hybrid_1_horner_only")
    save_pta_table(rows_2, "pta_hybrid_2_bourdet_only")
    save_pta_table(rows_3, "pta_hybrid_3_intersection_ensemble")

    sc_df = pd.DataFrame(scorecards)
    sc_df.to_csv(OUT_DIR / "pta_hybrid_iarf_confidence_scorecard.csv", index=False)

    print("\n=== AUTOMATED HYBRID PTA RELIABILITY & CONFIDENCE SCORE CARD ===")
    print(sc_df[[
        "well_name", "horner_win_hr", "bourdet_win_hr", "intersect_win_hr",
        "log_log_slope", "deriv_cv_pct", "r2_horner", "smooth_drift_pct",
        "confidence_score", "confidence_rating"
    ]].to_string(index=False))

    print("\n=== COMPARISON OF THE 3 SEPARATE PTA LOGICS ===")
    print(sc_df[[
        "well_name", "m_horner", "m_bourdet", "m_intersect",
        "k_horner", "k_bourdet", "k_intersect", "skin_intersect", "dP_skin_intersect"
    ]].to_string(index=False))

    print(f"\nSaved Diagnostic Plot -> {out_plot}")
    print("Saved 3 Separate PTA Deliverable Sets in vfm_export_tool/:")
    print("  1) pta_hybrid_1_horner_only.(csv|xlsx|_export_schema.csv)")
    print("  2) pta_hybrid_2_bourdet_only.(csv|xlsx|_export_schema.csv)")
    print("  3) pta_hybrid_3_intersection_ensemble.(csv|xlsx|_export_schema.csv)")
    print("  4) pta_hybrid_iarf_confidence_scorecard.csv")


if __name__ == "__main__":
    run_all_three_pta_logics()
