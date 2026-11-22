"""Step 01: Audit raw CSVs, clean telemetry, classify states, engineer features, and cache per well."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vfm.config import load_config
from vfm.io import (
    iter_high_freq_chunks,
    load_and_harmonize_well_tests,
    load_static_tables,
    save_frame,
)
from vfm.physics import add_physics_features
from vfm.preprocessing import clean_well_telemetry, classify_well_states


def main() -> None:
    cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else None)
    cache_dir = cfg.resolve_path("cache_dir")
    wells_cache = cache_dir / "wells"
    outputs_dir = cfg.resolve_path("outputs_dir")
    wells_cache.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    completion, pvt, choke_events = load_static_tables(
        cfg.resolve_path("well_completion_csv"),
        cfg.resolve_path("reservoir_pvt_csv"),
        cfg.resolve_path("choke_events_csv"),
        schema=cfg.schema,
    )
    wt_clean, wt_audit = load_and_harmonize_well_tests(
        cfg.resolve_path("well_tests_csv"),
        sm3_per_mscf=cfg.well_tests["sm3_per_mscf"],
        unit_mismatch_ratio_threshold=cfg.well_tests["unit_mismatch_ratio_threshold"],
        column_map=cfg.schema.get("well_tests"),
    )
    save_frame(wt_clean, cache_dir / "well_tests_harmonized")
    wt_clean.to_csv(cache_dir / "well_tests_harmonized.csv", index=False)

    print("Partitioning high_freq_data.csv by well_name...")
    per_well_parts: dict[str, list[pd.DataFrame]] = defaultdict(list)
    for chunk in iter_high_freq_chunks(cfg.resolve_path("high_freq_csv"), chunksize=500_000,
                                       column_map=cfg.schema.get("high_freq")):
        for well, grp in chunk.groupby("well_name", sort=False):
            per_well_parts[str(well)].append(grp)

    well_audits: dict[str, dict] = {}
    for well in sorted(per_well_parts):
        raw_df = pd.concat(per_well_parts.pop(well), ignore_index=True)
        cleaned_df, c_audit = clean_well_telemetry(
            raw_df,
            sentinel_values=cfg.cleaning["sentinel_values"],
            physical_bounds={k: tuple(v) for k, v in cfg.cleaning["physical_bounds"].items()},
            duplicate_policy=cfg.cleaning["duplicate_policy"],
            hampel_window=cfg.cleaning["hampel_window_min"],
            hampel_n_sigma=cfg.cleaning["hampel_n_sigma"],
        )
        states_df = classify_well_states(
            cleaned_df,
            choke_events=choke_events,
            choke_closed_pct_threshold=cfg.states["choke_closed_pct_threshold"],
            dp_choke_shut_in_psi=cfg.states["dp_choke_shut_in_psi"],
            stabilization_buffer_min=cfg.states["stabilization_buffer_min"],
        )
        feat_df = add_physics_features(
            states_df,
            completion=completion,
            pvt=pvt,
            rolling_windows_min=cfg.well_tests["rolling_feature_windows_min"],
        )
        saved_path = save_frame(feat_df, wells_cache / well)
        c_audit["shut_in_minutes"] = int(feat_df["is_shut_in"].sum())
        c_audit["transient_minutes"] = int(feat_df["is_transient"].sum())
        c_audit["steady_flowing_minutes"] = int(feat_df["is_steady_flowing"].sum())
        c_audit["cached_file"] = str(saved_path)
        well_audits[well] = c_audit
        print(
            f"  {well}: raw={c_audit['raw_rows']:,} -> dedup={c_audit['deduped_rows']:,} "
            f"(dups={c_audit['duplicates_removed']}, shut_in={c_audit['shut_in_minutes']:,})"
        )

    audit_report = {"well_tests": wt_audit, "high_freq_wells": well_audits}
    report_path = outputs_dir / "qc_audit_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(audit_report, f, indent=2)
    print(f"Saved QC audit report to {report_path}")


if __name__ == "__main__":
    main()
