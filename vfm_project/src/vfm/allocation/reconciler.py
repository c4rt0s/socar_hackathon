"""Daily rate integration, well-test anchoring, and manifold reconciliation (legacy API).

Backported fixes from docs/code_review.md:
  * calendar-complete daily grid (full SCADA-outage days are kept and flagged, not dropped);
  * `data_completeness` (valid minutes / 1440) is separate from `uptime_frac`
    (flowing / observed minutes); NaN predictions are not counted as valid;
  * well tests are anchored over their exact [test_start, test_end] window at the test
    midpoint (59 of 85 tests span midnight), not on `test_start.floor("D")`;
  * `total_gas_rate_alloc_mscfd` is produced;
  * manifold reconciliation groups by date *and* manifold, never gives flow to shut-in wells,
    and keeps the liquid total consistent.
The physics pipeline (scripts 05-08, dashboard) uses vfm.allocation.daily, which works on
1-minute predictions; these functions serve the legacy script 04 and daily-level callers.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd


ALLOC_PHASE_COLS: Sequence[str] = ("oil_rate_stbd", "water_rate_stbd", "gas_rate_mscfd")


def aggregate_daily_vfm(
    minute_preds: pd.DataFrame,
    date_grid: Optional[pd.DatetimeIndex] = None,
) -> pd.DataFrame:
    """1-minute predictions (timestamp, well_name, is_shut_in, rate columns) -> calendar-complete
    daily rates. Daily rates are 24-hour averages over valid minutes (shut-in minutes count as 0)."""
    df = minute_preds.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["date"] = df["timestamp"].dt.floor("D")
    si = df["is_shut_in"].astype(bool)
    liq = df["liquid_rate_stbd"] if "liquid_rate_stbd" in df else df["oil_rate_stbd"] + df["water_rate_stbd"]
    df["_valid"] = si | np.isfinite(liq.to_numpy(dtype="float64"))
    df["_flowing"] = ~si
    for col in ("oil_rate_stbd", "water_rate_stbd", "gas_rate_mscfd", "total_gas_rate_mscfd", "liquid_rate_stbd"):
        if col in df:
            df[col] = np.where(si, 0.0, df[col].to_numpy(dtype="float64"))
            df[col] = df[col].where(df["_valid"])
    if date_grid is None:
        date_grid = pd.date_range(df["date"].min(), df["date"].max(), freq="D", tz="UTC")
    parts = []
    for well, g in df.groupby("well_name"):
        d = g.groupby("date").agg(
            n_observed_minutes=("timestamp", "count"),
            n_valid_minutes=("_valid", "sum"),
            flowing_minutes=("_flowing", "sum"),
            oil_rate_vfm_stbd=("oil_rate_stbd", "mean"),
            water_rate_vfm_stbd=("water_rate_stbd", "mean"),
            gas_rate_vfm_mscfd=("gas_rate_mscfd", "mean"),
            total_gas_vfm_mscfd=("total_gas_rate_mscfd", "mean") if "total_gas_rate_mscfd" in g else ("gas_rate_mscfd", "mean"),
        ).reindex(date_grid)
        d.index.name = "date"
        d = d.reset_index()
        d.insert(0, "well_name", well)
        for c in ("n_observed_minutes", "n_valid_minutes", "flowing_minutes"):
            d[c] = d[c].fillna(0).astype(int)
        d["is_imputed_day"] = d["n_valid_minutes"] == 0
        d["data_completeness"] = (d["n_valid_minutes"] / 1440.0).round(4)
        d["uptime_frac"] = np.where(d["n_observed_minutes"] > 0,
                                    d["flowing_minutes"] / d["n_observed_minutes"].clip(lower=1), np.nan)
        x = (d["date"] - date_grid[0]).dt.days.to_numpy(dtype="float64")
        for c in ("oil_rate_vfm_stbd", "water_rate_vfm_stbd", "gas_rate_vfm_mscfd", "total_gas_vfm_mscfd", "uptime_frac"):
            v = d[c].to_numpy(dtype="float64")
            ok = np.isfinite(v)
            if ok.any() and not ok.all():
                d[c] = np.interp(x, x[ok], v[ok])  # outage days imputed and flagged
        d["liquid_rate_vfm_stbd"] = d["oil_rate_vfm_stbd"] + d["water_rate_vfm_stbd"]
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


def _window_average(daily_w: pd.DataFrame, col: str, t0: pd.Timestamp, t1: pd.Timestamp) -> float:
    """Time-weighted average of daily values over [t0, t1] (hours of overlap per day)."""
    day0 = pd.to_datetime(daily_w["date"], utc=True)
    ov = (np.minimum(day0 + pd.Timedelta(days=1), t1) - np.maximum(day0, t0)).dt.total_seconds().clip(lower=0)
    w = ov.to_numpy(dtype="float64")
    v = daily_w[col].to_numpy(dtype="float64")
    ok = (w > 0) & np.isfinite(v)
    return float(np.sum(w[ok] * v[ok]) / np.sum(w[ok])) if ok.any() else float("nan")


def anchor_daily_to_well_tests(
    daily_vfm: pd.DataFrame,
    well_tests_df: pd.DataFrame,
    max_multiplier: float = 1.5,
    min_multiplier: float = 0.65,
    default_test_hours: float = 24.0,
) -> pd.DataFrame:
    """
    Calibrate daily VFM rates against well tests. For each test the VFM reference is the
    time-weighted daily VFM over the exact [test_start, test_end] window; test/VFM ratios
    are placed at the test midpoint and interpolated (held constant outside) to each day's noon.
    """
    out = daily_vfm.copy()
    out["date"] = pd.to_datetime(out["date"], utc=True)
    wt = well_tests_df.copy()
    wt["test_start"] = pd.to_datetime(wt["test_start"], utc=True)
    wt["test_end"] = (pd.to_datetime(wt["test_end"], utc=True) if "test_end" in wt
                      else wt["test_start"] + pd.Timedelta(hours=default_test_hours))
    origin = out["date"].min()

    pairs = (("oil_rate_stbd", "oil_rate_vfm_stbd", "oil_rate_alloc_stbd"),
             ("water_rate_stbd", "water_rate_vfm_stbd", "water_rate_alloc_stbd"),
             ("gas_rate_mscfd", "gas_rate_vfm_mscfd", "gas_rate_alloc_mscfd"))
    for _, _, alloc_col in pairs:
        out[alloc_col] = out[alloc_col.replace("_alloc_", "_vfm_")]

    for well, w_idx in out.groupby("well_name").groups.items():
        w_tests = wt[wt["well_name"] == well].sort_values("test_start")
        if w_tests.empty:
            continue
        dw = out.loc[w_idx]
        x_days = ((dw["date"] + pd.Timedelta(hours=12)) - origin).dt.total_seconds().to_numpy() / 86400.0
        t_mid = w_tests["test_start"] + (w_tests["test_end"] - w_tests["test_start"]) / 2
        t_days = (t_mid - origin).dt.total_seconds().to_numpy() / 86400.0
        for wt_col, vfm_col, alloc_col in pairs:
            ref = np.array([_window_average(dw, vfm_col, a, b) for a, b in zip(w_tests["test_start"], w_tests["test_end"])])
            ratios = np.clip(w_tests[wt_col].to_numpy(dtype="float64") / np.maximum(ref, 1.0), min_multiplier, max_multiplier)
            valid = np.isfinite(ratios)
            if not valid.any():
                continue
            mult = np.interp(x_days, t_days[valid], ratios[valid])
            out.loc[w_idx, alloc_col] = np.clip(dw[vfm_col].to_numpy(dtype="float64") * mult, 0.0, None)

    out["liquid_rate_alloc_stbd"] = out["oil_rate_alloc_stbd"] + out["water_rate_alloc_stbd"]
    gl = out["total_gas_vfm_mscfd"] - out["gas_rate_vfm_mscfd"] if "total_gas_vfm_mscfd" in out else 0.0
    out["total_gas_rate_alloc_mscfd"] = out["gas_rate_alloc_mscfd"] + np.clip(gl, 0.0, None)
    return out


def reconcile_to_manifold_totals(
    daily_alloc: pd.DataFrame,
    manifold_totals: Optional[pd.DataFrame],
    variance_weights: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    If commingled separator/manifold daily totals are available, adjust well allocations so
    that sum_w q_alloc = Q_measured per (date[, production_manifold]). The correction is
    distributed in proportion to `variance_weights` (default: the allocated rate), only over
    wells that flowed (q > 0); the result is non-negative and sums exactly to the total.
    """
    if manifold_totals is None or manifold_totals.empty:
        return daily_alloc
    keys = ["date"] + (["production_manifold"] if "production_manifold" in manifold_totals
                       and "production_manifold" in daily_alloc else [])
    out = daily_alloc.merge(manifold_totals, on=keys, how="left", suffixes=("", "_meas"))
    for col, meas_col in (("oil_rate_alloc_stbd", "oil_total_meas_stbd"),
                          ("water_rate_alloc_stbd", "water_total_meas_stbd"),
                          ("gas_rate_alloc_mscfd", "gas_total_meas_mscfd")):
        if meas_col not in out.columns:
            continue
        for _, idx in out.groupby(keys).groups.items():
            q_meas = out.loc[idx, meas_col].iloc[0]
            if not np.isfinite(q_meas):
                continue
            q = out.loc[idx, col].to_numpy(dtype="float64")
            flowing = q > 0
            if not flowing.any():
                continue
            w = (variance_weights.reindex(out.loc[idx, "well_name"]).fillna(1.0).to_numpy(dtype="float64")
                 if variance_weights is not None else q.copy())
            w = np.where(flowing, np.maximum(w, 0.0), 0.0)
            adj = q.copy()
            for _ in range(20):  # redistribute among wells that stay positive
                active = flowing & (adj > 0) if (adj > 0).any() else flowing
                gap = q_meas - adj.sum()
                if abs(gap) < 1e-9 or w[active].sum() <= 0:
                    break
                adj[active] += w[active] / w[active].sum() * gap
                adj = np.clip(adj, 0.0, None)
            out.loc[idx, col] = adj
    out["liquid_rate_alloc_stbd"] = out["oil_rate_alloc_stbd"] + out["water_rate_alloc_stbd"]
    if "total_gas_rate_alloc_mscfd" in out and "total_gas_vfm_mscfd" in out:
        out["total_gas_rate_alloc_mscfd"] = out["gas_rate_alloc_mscfd"] + np.clip(
            out["total_gas_vfm_mscfd"] - out["gas_rate_vfm_mscfd"], 0.0, None)
    return out
