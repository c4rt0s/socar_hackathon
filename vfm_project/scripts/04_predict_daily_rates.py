"""Step 04: Score 1-minute telemetry and produce daily well-test-anchored allocated rates."""

from __future__ import annotations

from pathlib import Path
import pickle
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vfm.allocation import aggregate_daily_vfm, anchor_daily_to_well_tests
from vfm.config import load_config
from vfm.io import load_frame


def main() -> None:
    cfg = load_config()
    cache_dir = cfg.resolve_path("cache_dir")
    wells_cache = cache_dir / "wells"
    models_dir = cfg.resolve_path("models_dir")
    outputs_dir = cfg.resolve_path("outputs_dir")
    outputs_dir.mkdir(parents=True, exist_ok=True)

    with (models_dir / "hybrid_vfm.pkl").open("rb") as f:
        model = pickle.load(f)

    wt_clean = load_frame(cache_dir / "well_tests_harmonized")
    daily_parts = []

    for well in sorted(wt_clean["well_name"].unique()):
        hf_well = load_frame(wells_cache / well)
        preds = model.predict(hf_well)
        minute_df = pd.concat(
            [hf_well[["well_name", "timestamp", "is_shut_in"]], preds],
            axis=1,
        )
        daily_parts.append(aggregate_daily_vfm(minute_df))

    daily_vfm = pd.concat(daily_parts, ignore_index=True).sort_values(["well_name", "date"])
    daily_alloc = anchor_daily_to_well_tests(daily_vfm, wt_clean)

    out_csv = outputs_dir / "daily_allocated_rates.csv"
    daily_alloc.to_csv(out_csv, index=False)
    print(f"Wrote {len(daily_alloc)} daily well records -> {out_csv}")


if __name__ == "__main__":
    main()
