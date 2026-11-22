"""Step 02: Build steady-state aligned well-test training dataset."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vfm.config import load_config
from vfm.features import aggregate_well_test_windows
from vfm.io import load_frame, save_frame


def main() -> None:
    cfg = load_config()
    cache_dir = cfg.resolve_path("cache_dir")
    wells_cache = cache_dir / "wells"

    wt_clean = load_frame(cache_dir / "well_tests_harmonized")
    aligned_parts = []
    for well in sorted(wt_clean["well_name"].unique()):
        hf_well = load_frame(wells_cache / well)
        aligned_well = aggregate_well_test_windows(
            hf_well,
            wt_clean,
            min_steady_coverage=cfg.well_tests["min_steady_coverage"],
        )
        aligned_parts.append(aligned_well)

    dataset = pd.concat(aligned_parts, ignore_index=True).sort_values(["well_name", "test_start"])
    out_path = save_frame(dataset, cache_dir / "well_test_features")
    dataset.to_csv(cache_dir / "well_test_features.csv", index=False)

    n_qc = int(dataset["passes_qc"].sum())
    print(f"Aligned {len(dataset)} well tests ({n_qc} passed QC) -> {out_path}")


if __name__ == "__main__":
    main()
