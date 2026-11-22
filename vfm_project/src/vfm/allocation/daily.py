"""Daily aggregation of 1-minute PhysicsVFM predictions and well-test anchoring.

Moved from scripts/07 so the batch script and the dashboard share one implementation.
Nothing here depends on specific wells or dates: the time axis for interpolation is the
first day of the supplied date grid.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd


def _days_since(ts: pd.Series, origin: pd.Timestamp) -> np.ndarray:
    return (ts - origin).dt.total_seconds().to_numpy(dtype="float64") / 86400.0


def calendar_date_grid(hf_by_well) -> pd.DatetimeIndex:
    """UTC daily grid covering every well's telemetry."""
    lo = min(pd.to_datetime(df["timestamp"], utc=True).min() for df in hf_by_well.values()).floor("D")
    hi = max(pd.to_datetime(df["timestamp"], utc=True).max() for df in hf_by_well.values()).floor("D")
    return pd.date_range(lo, hi, freq="D", tz="UTC")


def aggregate_well_daily_physics(
    well: str,
    hf_well: pd.DataFrame,
    preds: pd.DataFrame,
    date_grid: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Aggregate 1-minute PhysicsVFM predictions for one well onto a calendar-complete UTC daily
    grid. Daily rates are 24-hour averages (shut-in minutes count as zero); `uptime_frac`
    separates operating time from SCADA `data_completeness`. Full-outage days are linearly
    imputed from neighbouring days and flagged `is_imputed_day`.
    """
    origin = date_grid[0]
    ts = pd.to_datetime(hf_well["timestamp"], utc=True)
    dates = ts.dt.floor("D")

    is_si = preds["is_shut_in"].to_numpy(dtype=bool)
    q_liq = preds["liquid_rate_stbd"].to_numpy(dtype="float64")
    q_oil = preds["oil_rate_stbd"].to_numpy(dtype="float64")
    q_wat = preds["water_rate_stbd"].to_numpy(dtype="float64")
    q_gas = preds["gas_rate_mscfd"].to_numpy(dtype="float64")
    q_gtot = preds["total_gas_rate_mscfd"].to_numpy(dtype="float64")
    q_gl_raw = hf_well["gas_lift_rate_mscfd"].to_numpy(dtype="float64")
    q_gl_eff = np.where(is_si, 0.0, np.where(np.isfinite(q_gl_raw), np.maximum(q_gl_raw, 0.0), np.nan))

    valid_rate = is_si | np.isfinite(q_liq)
    work = pd.DataFrame(
        {
            "date": dates,
            "is_shut_in": is_si,
            "is_flowing": ~is_si,
            "is_valid": valid_rate,
            "is_sensor_missing": preds["is_sensor_missing"].to_numpy(dtype=bool),
            "is_reconcile_outlier": preds["is_reconcile_outlier"].to_numpy(dtype=bool),
            "oil_rate_vfm_stbd": np.where(valid_rate, q_oil, np.nan),
            "water_rate_vfm_stbd": np.where(valid_rate, q_wat, np.nan),
            "gas_rate_vfm_mscfd": np.where(valid_rate, q_gas, np.nan),
            "liquid_rate_vfm_stbd": np.where(valid_rate, q_liq, np.nan),
            "daily_gas_lift_mscfd": np.where(valid_rate, q_gl_eff, np.nan),
            "total_gas_vfm_mscfd": np.where(valid_rate, q_gtot, np.nan),
        }
    )
    for col in ("q_liq_ipr", "q_liq_choke", "q_liq_vlp"):
        if col in preds.columns:  # per-estimator daily means (flowing minutes only), for diagnostics
            work[col] = np.where(~is_si, preds[col].to_numpy(dtype="float64"), np.nan)
    for col in ("is_transient", "is_ewma_transient"):
        if col in preds.columns:
            work[col] = preds[col].to_numpy(dtype=bool) & ~is_si

    agg = dict(
        n_observed_minutes=("is_valid", "count"),
        n_valid_minutes=("is_valid", "sum"),
        flowing_minutes=("is_flowing", "sum"),
        shut_in_minutes=("is_shut_in", "sum"),
        n_sensor_missing_minutes=("is_sensor_missing", "sum"),
        n_reconcile_outlier_minutes=("is_reconcile_outlier", "sum"),
        oil_rate_vfm_stbd=("oil_rate_vfm_stbd", "mean"),
        water_rate_vfm_stbd=("water_rate_vfm_stbd", "mean"),
        gas_rate_vfm_mscfd=("gas_rate_vfm_mscfd", "mean"),
        liquid_rate_vfm_stbd=("liquid_rate_vfm_stbd", "mean"),
        daily_gas_lift_mscfd=("daily_gas_lift_mscfd", "mean"),
        total_gas_vfm_mscfd=("total_gas_vfm_mscfd", "mean"),
    )
    for col in ("q_liq_ipr", "q_liq_choke", "q_liq_vlp"):
        if col in work.columns:
            agg[f"{col}_flowing_mean"] = (col, "mean")
    if "is_transient" in work.columns:
        agg["n_transient_minutes"] = ("is_transient", "sum")
    if "is_ewma_transient" in work.columns:
        agg["n_ewma_transient_minutes"] = ("is_ewma_transient", "sum")
    daily = work.groupby("date", as_index=True).agg(**agg).reindex(date_grid)

    daily.index.name = "date"
    daily = daily.reset_index()
    daily.insert(0, "well_name", well)
    daily["n_expected_minutes"] = 1440

    count_cols = [
        "n_observed_minutes", "n_valid_minutes", "flowing_minutes", "shut_in_minutes",
        "n_sensor_missing_minutes", "n_reconcile_outlier_minutes",
    ] + [c for c in ("n_transient_minutes", "n_ewma_transient_minutes") if c in daily.columns]
    for col in count_cols:
        daily[col] = daily[col].fillna(0).astype(int)

    daily["is_imputed_day"] = daily["n_valid_minutes"] == 0
    daily["data_completeness"] = (daily["n_valid_minutes"] / 1440.0).round(4)
    daily["uptime_frac"] = np.where(
        daily["n_observed_minutes"] > 0,
        daily["flowing_minutes"] / np.maximum(daily["n_observed_minutes"], 1),
        np.nan,
    )

    interp_cols = ["uptime_frac", "oil_rate_vfm_stbd", "water_rate_vfm_stbd", "gas_rate_vfm_mscfd",
                   "daily_gas_lift_mscfd"]
    t_days = _days_since(daily["date"], origin)
    for col in interp_cols:
        vals = daily[col].to_numpy(dtype="float64")
        fin = np.isfinite(vals)
        if fin.any() and not fin.all():
            daily[col] = np.interp(t_days, t_days[fin], vals[fin])
        elif not fin.any():
            daily[col] = 0.0

    full_si = (~daily["is_imputed_day"]) & (daily["flowing_minutes"] == 0)
    for col in ("oil_rate_vfm_stbd", "water_rate_vfm_stbd", "gas_rate_vfm_mscfd", "daily_gas_lift_mscfd"):
        daily[col] = np.where(full_si, 0.0, np.clip(daily[col].to_numpy(dtype="float64"), 0.0, None))

    daily["uptime_frac"] = np.round(np.clip(daily["uptime_frac"].to_numpy(dtype="float64"), 0.0, 1.0), 4)
    daily["liquid_rate_vfm_stbd"] = daily["oil_rate_vfm_stbd"] + daily["water_rate_vfm_stbd"]
    daily["total_gas_vfm_mscfd"] = np.where(full_si, 0.0, daily["gas_rate_vfm_mscfd"] + daily["daily_gas_lift_mscfd"])
    return daily


def compute_exact_window_vfm_refs(
    well: str,
    hf_well: pd.DataFrame,
    preds: pd.DataFrame,
    well_tests_sub: pd.DataFrame,
) -> pd.DataFrame:
    """Unanchored PhysicsVFM phase rates over each exact [test_start, test_end] window."""
    ts = pd.to_datetime(hf_well["timestamp"], utc=True)
    rows: List[dict] = []
    for _, test in well_tests_sub.sort_values("test_start").iterrows():
        t0 = pd.Timestamp(test["test_start"])
        t1 = pd.Timestamp(test["test_end"])
        in_win = (ts >= t0) & (ts <= t1)
        flowing = in_win & (~preds["is_shut_in"]) & np.isfinite(preds["liquid_rate_stbd"])
        steady = flowing & (~preds["is_transient"])
        src_mask = steady if steady.any() else flowing
        if not src_mask.any():
            continue
        rows.append({
            "well_name": well, "test_start": t0, "test_end": t1, "t_mid": t0 + (t1 - t0) / 2,
            "oil_rate_stbd": float(test["oil_rate_stbd"]),
            "water_rate_stbd": float(test["water_rate_stbd"]),
            "gas_rate_mscfd": float(test["gas_rate_mscfd"]),
            "oil_rate_vfm_ref": float(preds.loc[src_mask, "oil_rate_stbd"].mean()),
            "water_rate_vfm_ref": float(preds.loc[src_mask, "water_rate_stbd"].mean()),
            "gas_rate_vfm_ref": float(preds.loc[src_mask, "gas_rate_mscfd"].mean()),
        })
    return pd.DataFrame(rows)


def anchor_daily_exact_window(
    daily_vfm: pd.DataFrame,
    test_refs: pd.DataFrame,
    min_multiplier: float = 0.75,
    max_multiplier: float = 1.35,
    origin: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """
    Anchor daily VFM rates to well tests with piecewise-linear multipliers (test / VFM)
    evaluated at test midpoints and interpolated to each day's noon; clipped to
    [min_multiplier, max_multiplier], held constant outside the first/last test.
    """
    out = daily_vfm.copy()
    origin = origin if origin is not None else pd.Timestamp(out["date"].min())
    out["oil_rate_alloc_stbd"] = out["oil_rate_vfm_stbd"]
    out["water_rate_alloc_stbd"] = out["water_rate_vfm_stbd"]
    out["gas_rate_alloc_mscfd"] = out["gas_rate_vfm_mscfd"]

    pairs = (
        ("oil_rate_stbd", "oil_rate_vfm_ref", "oil_rate_vfm_stbd", "oil_rate_alloc_stbd"),
        ("water_rate_stbd", "water_rate_vfm_ref", "water_rate_vfm_stbd", "water_rate_alloc_stbd"),
        ("gas_rate_mscfd", "gas_rate_vfm_ref", "gas_rate_vfm_mscfd", "gas_rate_alloc_mscfd"),
    )
    for well, w_idx in out.groupby("well_name").groups.items():
        w_refs = test_refs[test_refs["well_name"] == well].sort_values("t_mid") if len(test_refs) else test_refs
        if w_refs.empty:
            continue
        x_days = _days_since(out.loc[w_idx, "date"] + pd.Timedelta(hours=12), origin)
        t_days = _days_since(w_refs["t_mid"], origin)
        for wt_col, ref_col, vfm_col, alloc_col in pairs:
            denom = np.maximum(w_refs[ref_col].to_numpy(dtype="float64"), 1.0)
            ratios = np.clip(w_refs[wt_col].to_numpy(dtype="float64") / denom, min_multiplier, max_multiplier)
            valid = np.isfinite(ratios)
            if not valid.any():
                continue
            mult = np.interp(x_days, t_days[valid], ratios[valid])
            out.loc[w_idx, alloc_col] = np.clip(out.loc[w_idx, vfm_col].to_numpy(dtype="float64") * mult, 0.0, None)

    full_si = (~out["is_imputed_day"]) & (out["flowing_minutes"] == 0)
    for col in ("oil_rate_alloc_stbd", "water_rate_alloc_stbd", "gas_rate_alloc_mscfd"):
        out[col] = np.where(full_si, 0.0, np.clip(out[col].to_numpy(dtype="float64"), 0.0, None))
    out["liquid_rate_alloc_stbd"] = out["oil_rate_alloc_stbd"] + out["water_rate_alloc_stbd"]
    out["total_gas_rate_alloc_mscfd"] = np.where(full_si, 0.0, out["gas_rate_alloc_mscfd"] + out["daily_gas_lift_mscfd"])
    return out
