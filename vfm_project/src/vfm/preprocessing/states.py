"""Operational state segmentation: SHUT_IN vs FLOWING (steady / transient / flow stall).

Backported fixes from docs/code_review.md:
  * shut-in is defined only by the choke (`choke_opening_pct <= threshold`); a low choke
    differential (`dp_choke <= dp_stall_psi`, e.g. GL-03 at 50 % choke on 2025-09-06) is a
    separate `is_flow_stall` flag and counts as transient, never as shut-in;
  * no backward fill of choke events: telemetry before the first event has an unknown
    choke (`is_state_unknown`) instead of borrowing a future setting;
  * lift-gas meter NaNs never affect the flowing/shut-in state;
  * multi-well frames are handled per well.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _classify_one(out: pd.DataFrame, ev: pd.DataFrame, choke_closed_pct_threshold: float,
                  dp_choke_shut_in_psi: float, stabilization_buffer_min: int) -> pd.DataFrame:
    out = out.sort_values("timestamp").reset_index(drop=True)
    if ev.empty:
        out["choke_opening_pct"] = np.nan
        out["minutes_since_choke_change"] = np.inf
    else:
        ev = ev.sort_values("timestamp").rename(columns={"timestamp": "choke_event_ts"})
        merged = pd.merge_asof(out[["timestamp"]], ev[["choke_event_ts", "choke_opening_pct"]],
                               left_on="timestamp", right_on="choke_event_ts", direction="backward")
        out["choke_opening_pct"] = merged["choke_opening_pct"].to_numpy()  # NaN before first event
        dt_min = (merged["timestamp"] - merged["choke_event_ts"]).dt.total_seconds() / 60.0
        out["minutes_since_choke_change"] = dt_min.fillna(np.inf).to_numpy()

    choke = out["choke_opening_pct"]
    out["is_state_unknown"] = choke.isna()
    out["is_shut_in"] = choke.notna() & (choke <= choke_closed_pct_threshold)
    dp_choke = out["whp_psia"] - out["whp_downstream_psia"]
    out["is_flow_stall"] = (~out["is_shut_in"]) & dp_choke.notna() & (dp_choke <= dp_choke_shut_in_psi)
    recent_change = out["minutes_since_choke_change"] < stabilization_buffer_min
    out["is_transient"] = (~out["is_shut_in"]) & (recent_change | out["is_flow_stall"])
    out["is_steady_flowing"] = (~out["is_shut_in"]) & (~out["is_transient"]) & (~out["is_state_unknown"])
    return out


def classify_well_states(
    df: pd.DataFrame,
    choke_events: pd.DataFrame,
    choke_closed_pct_threshold: float = 0.5,
    dp_choke_shut_in_psi: float = 2.0,
    stabilization_buffer_min: int = 30,
) -> pd.DataFrame:
    """
    Attach the active choke opening (backward as-of join only) and label each 1-minute row:
      - is_shut_in:        choke <= threshold (the only shut-in criterion)
      - is_flow_stall:     choke open but dp_choke <= dp_choke_shut_in_psi (kept as transient)
      - is_transient:      open and (within `stabilization_buffer_min` of a choke event or stalled)
      - is_steady_flowing: open, not transient, choke state known
    `dp_choke_shut_in_psi` keeps its historical name for config compatibility.
    """
    parts = []
    for well, grp in df.groupby("well_name", sort=False):
        ev = choke_events[choke_events["well_name"] == well]
        parts.append(_classify_one(grp, ev, choke_closed_pct_threshold, dp_choke_shut_in_psi, stabilization_buffer_min))
    return pd.concat(parts, ignore_index=True) if parts else df.copy()
