"""Telemetry cleaning: deduplication, sentinels, bounds, Hampel despiking, safe gap filling.

Backported fixes from docs/code_review.md:
  * explicit per-sensor MAD floors (the old floor of 1 % of the median meant a ~180 psi
    threshold on a 3,000 psi gauge, i.e. no despiking at all);
  * gas-lift despiking only on positive readings, so genuine steps to 0.0 (shut-ins, lift
    stops) are never treated as spikes;
  * timestamp-aware interpolation: a NaN is only filled when the surrounding valid samples
    are at most `max_fill_gap_min` apart in time, so multi-hour SCADA outages are never bridged;
  * duplicate timestamps keep the row with the most valid readings.
The Hampel window stays centred (7 minutes look-ahead): this is offline data cleaning, and a
trailing window would flag every genuine pressure step at shut-in/restart as a spike.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from vfm.io.loaders import HF_NUMERIC_COLS

DEFAULT_MAD_FLOORS: Dict[str, float] = {
    "whp_psia": 5.0,
    "whp_downstream_psia": 3.0,
    "dh_pressure_psia": 5.0,
    "dh_temperature_degf": 0.5,
    "annulus_pressure_psia": 5.0,
    "gas_lift_rate_mscfd": 25.0,
}
POSITIVE_ONLY_HAMPEL = ("gas_lift_rate_mscfd",)


def _hampel_mask(s: pd.Series, window: int, n_sigma: float, floor: float) -> pd.Series:
    med = s.rolling(window=window, center=True, min_periods=5).median()
    mad = (s - med).abs().rolling(window=window, center=True, min_periods=5).median()
    scale = np.maximum(1.4826 * mad, floor)
    return ((s - med).abs() > n_sigma * scale).fillna(False)


def time_aware_fill(s: pd.Series, ts: pd.Series, max_gap_min: float) -> pd.Series:
    """Linear-in-time interpolation of interior NaNs whose neighbouring valid samples are at
    most `max_gap_min` minutes apart. Larger gaps stay NaN."""
    valid = s.notna()
    if valid.all() or not valid.any():
        return s
    t = pd.Series(ts.to_numpy(), index=s.index)
    prev_t = t.where(valid).ffill()
    next_t = t.where(valid).bfill()
    span_min = (next_t - prev_t).dt.total_seconds() / 60.0
    fillable = (~valid) & span_min.notna() & (span_min <= max_gap_min)
    if not fillable.any():
        return s
    x = (t - t.iloc[0]).dt.total_seconds().to_numpy(dtype="float64")
    filled = np.interp(x, x[valid.to_numpy()], s[valid].to_numpy(dtype="float64"))
    return s.where(~fillable, pd.Series(filled, index=s.index))


def clean_well_telemetry(
    df: pd.DataFrame,
    sentinel_values: Iterable[float] = (-999.25, -999.0),
    physical_bounds: Mapping[str, Tuple[float, float]] | None = None,
    duplicate_policy: str = "keep_first",
    hampel_window: int = 15,
    hampel_n_sigma: float = 6.0,
    interp_limit: int = 5,
    mad_floors: Optional[Mapping[str, float]] = None,
    max_fill_gap_min: Optional[float] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Clean single-well 1-minute telemetry:
      1. UTC timestamps, chronological order, one row per timestamp (best row kept).
      2. Sentinels and out-of-bounds readings -> NaN.
      3. Centred Hampel despiking with per-sensor MAD floors (gas lift: positive readings only).
      4. Fill only short gaps in time (default `interp_limit` minutes).
    """
    out = df.copy()
    raw_count = int(len(out))
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    sent_set = set(float(v) for v in sentinel_values)
    present = [c for c in HF_NUMERIC_COLS if c in out.columns]

    # duplicates: prefer the row with the most valid (non-NaN, non-sentinel) readings
    quality = sum((out[c].notna() & ~out[c].isin(sent_set)).astype(int) for c in present) if present else 0
    out["_q"] = quality
    out = out.sort_values(["timestamp", "_q"], ascending=[True, False], kind="mergesort")
    keep_opt = "first" if duplicate_policy in ("keep_first", "keep_best") else "last"
    dup_mask = out.duplicated(subset=["timestamp"], keep=keep_opt)
    dup_count = int(dup_mask.sum())
    out = out.loc[~dup_mask].drop(columns="_q").reset_index(drop=True)

    floors = {**DEFAULT_MAD_FLOORS, **(mad_floors or {})}
    bounds = physical_bounds or {}
    gap_min = float(max_fill_gap_min if max_fill_gap_min is not None else interp_limit)
    sentinel_counts: Dict[str, int] = {}
    oob_counts: Dict[str, int] = {}
    spike_counts: Dict[str, int] = {}
    filled_counts: Dict[str, int] = {}

    for col in present:
        s = out[col].astype("float64")
        s_mask = s.isin(sent_set)
        sentinel_counts[col] = int(s_mask.sum())
        s = s.mask(s_mask, np.nan)
        if col in bounds:
            lo, hi = bounds[col]
            b_mask = (s < lo) | (s > hi)
            oob_counts[col] = int(b_mask.sum())
            s = s.mask(b_mask, np.nan)
        else:
            oob_counts[col] = 0
        floor = float(floors.get(col, 0.5))
        if col in POSITIVE_ONLY_HAMPEL:
            pos = s.where(s > 0)
            sp_mask = _hampel_mask(pos, hampel_window, hampel_n_sigma, floor) & (s > 0)
        else:
            sp_mask = _hampel_mask(s, hampel_window, hampel_n_sigma, floor)
        spike_counts[col] = int(sp_mask.sum())
        s = s.mask(sp_mask, np.nan)
        n_nan_before = int(s.isna().sum())
        s = time_aware_fill(s, out["timestamp"], gap_min)
        filled_counts[col] = n_nan_before - int(s.isna().sum())
        out[col] = s

    audit = {
        "raw_rows": raw_count,
        "deduped_rows": int(len(out)),
        "duplicates_removed": dup_count,
        "sentinels_masked": sentinel_counts,
        "out_of_bounds_masked": oob_counts,
        "hampel_spikes_masked": spike_counts,
        "short_gaps_filled": filled_counts,
        "max_fill_gap_min": gap_min,
        "mad_floors": floors,
    }
    return out, audit
