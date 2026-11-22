"""Step 07: Fit PhysicsVFM across all wells and produce calendar-complete daily allocated rates.

Uses the dataset config ($VFM_CONFIG or configs/default.json). This is the uncoupled
baseline; the PTA-coupled variant is produced by the dashboard workbench (vfm.workbench).
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Dict, List
import time

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vfm.allocation.daily import (
    aggregate_well_daily_physics,
    anchor_daily_exact_window,
    calendar_date_grid,
    compute_exact_window_vfm_refs,
)
from vfm.config import load_config
from vfm.io import load_frame, load_static_tables
from vfm.modeling.physics_vfm import physics_vfm_from_config


RATE_COLS = [
    "oil_rate_vfm_stbd", "water_rate_vfm_stbd", "gas_rate_vfm_mscfd", "liquid_rate_vfm_stbd",
    "total_gas_vfm_mscfd", "oil_rate_alloc_stbd", "water_rate_alloc_stbd", "gas_rate_alloc_mscfd",
    "liquid_rate_alloc_stbd", "total_gas_rate_alloc_mscfd",
]


def main() -> None:
    t_start = time.perf_counter()
    cfg = load_config()
    cache_dir = cfg.resolve_path("cache_dir")
    outputs_dir = cfg.resolve_path("outputs_dir")
    outputs_dir.mkdir(parents=True, exist_ok=True)

    comp, pvt, _ = load_static_tables(
        cfg.resolve_path("well_completion_csv"),
        cfg.resolve_path("reservoir_pvt_csv"),
        cfg.resolve_path("choke_events_csv"),
        schema=cfg.schema,
    )
    wt_clean = load_frame(cache_dir / "well_tests_harmonized")
    wt_clean["test_start"] = pd.to_datetime(wt_clean["test_start"], utc=True)
    wt_clean["test_end"] = pd.to_datetime(wt_clean["test_end"], utc=True)

    wells = sorted(wt_clean["well_name"].unique())
    hf_cache: Dict[str, pd.DataFrame] = {w: load_frame(cache_dir / "wells" / w) for w in wells}

    model = physics_vfm_from_config(cfg).fit(wt_clean, hf_cache, comp, pvt)
    print("\n=== PhysicsVFM Calibrated Parameters & Inverse-Variance WLS Weights ===")
    print(model.weights_df_.to_string(index=False))

    date_grid = calendar_date_grid(hf_cache)
    daily_parts: List[pd.DataFrame] = []
    ref_parts: List[pd.DataFrame] = []
    for well in wells:
        hf_w = hf_cache[well]
        preds = model.predict(hf_w)
        daily_parts.append(aggregate_well_daily_physics(well, hf_w, preds, date_grid))
        ref_parts.append(compute_exact_window_vfm_refs(well, hf_w, preds, wt_clean[wt_clean["well_name"] == well]))

    daily_vfm = pd.concat(daily_parts, ignore_index=True).sort_values(["well_name", "date"]).reset_index(drop=True)
    test_refs = pd.concat(ref_parts, ignore_index=True)
    daily_alloc = anchor_daily_exact_window(daily_vfm, test_refs)

    out_csv = outputs_dir / "daily_allocated_rates_physics.csv"
    daily_alloc.to_csv(out_csv, index=False)

    n_nan = int(daily_alloc[RATE_COLS].isna().sum().sum())
    min_val = float(daily_alloc[RATE_COLS].min().min())
    full_si_rows = daily_alloc[(~daily_alloc["is_imputed_day"]) & (daily_alloc["flowing_minutes"] == 0)]
    max_si_rate = float(full_si_rows[RATE_COLS].max().max()) if not full_si_rows.empty else 0.0
    elapsed = time.perf_counter() - t_start
    print(f"\nWrote {len(daily_alloc)} daily records ({len(wells)} wells x {len(date_grid)} days) -> {out_csv} in {elapsed:.1f}s")
    print(f"Sanity: NaNs={n_nan}, min_rate={min_val:.6f}, full_shutin_days={len(full_si_rows)}, max_rate_on_full_shutin={max_si_rate:.6f}")
    if not full_si_rows.empty:
        print("Full shut-in days:", full_si_rows[["well_name", "date"]].astype(str).values.tolist())


if __name__ == "__main__":
    main()
