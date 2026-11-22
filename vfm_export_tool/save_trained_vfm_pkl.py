#!/usr/bin/env python3
"""Train and serialize the V2 Phase-Decomposed Stacked Grey-Box VFM model to .pkl format
(Deliverable #2) with 100% data-driven 5-fold OOF tuning of all base weights and reconciliation
parameters (zero hardcoded paths, dates, or meta-weights)."""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
import numpy as np
import pandas as pd

EXPORT_DIR = Path(__file__).resolve().parent
PROJ_MODELS = EXPORT_DIR.parent / "vfm_project" / "models"
PROJ_MODELS.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(EXPORT_DIR))
from vfm_phase_decomposed_stacked_vfm import (
    resolve_raw_dir,
    load_harmonized_well_tests,
    stream_raw_scada_robust_median,
    fit_v2_stacked_bundle,
    build_rich_feature_matrix,
    apply_soft_phase_reconciliation,
)


def main(data_dir: Path | str | None = None):
    raw_dir = resolve_raw_dir(data_dir)
    comp = pd.read_csv(raw_dir / "well_completion.csv").set_index("well_name")
    pvt = pd.read_csv(raw_dir / "reservoir_pvt.csv").set_index("well_name")
    ce = pd.read_csv(raw_dir / "choke_events.csv", parse_dates=["timestamp"]).sort_values(["well_name", "timestamp"])

    wt_raw = load_harmonized_well_tests(raw_dir)
    wt, _, envelopes, ref_date = stream_raw_scada_robust_median(wt_raw, ce, raw_dir)

    fitted = fit_v2_stacked_bundle(wt, comp, pvt, envelopes, ref_date, n_synth_per_well=180, verbose=True)
    artifact = {k: v for k, v in fitted.items() if k != "df_synth"}

    p_export = EXPORT_DIR / "vfm_stacked_v2.pkl"
    p_proj = PROJ_MODELS / "vfm_stacked_v2.pkl"
    for p in (p_export, p_proj):
        with open(p, "wb") as f:
            pickle.dump(artifact, f)
        print(f"Saved trained VFM model pickle -> {p} ({p.stat().st_size / 1024:.1f} KB)")

    # Verify round-trip load and inference on all real well tests
    with open(p_proj, "rb") as f:
        loaded = pickle.load(f)
    X_chk, pri_chk = build_rich_feature_matrix(wt, comp, pvt, loaded["anchors"])
    choke = wt["choke_opening_pct"].to_numpy(float)
    qo = loaded["learners"]["oil"].predict(X_chk, pri_chk["oil"], choke)
    qw = loaded["learners"]["water"].predict(X_chk, pri_chk["water"], choke)
    ql = loaded["learners"]["liquid"].predict(X_chk, pri_chk["liquid"], choke)
    qo_r, _ = apply_soft_phase_reconciliation(qo, qw, ql, float(loaded["lambda_rec"]))
    wape_o = 100.0 * np.sum(np.abs(qo_r - wt["oil_rate_stbd"])) / np.sum(wt["oil_rate_stbd"])
    print(f"Round-trip .pkl inference verified! {len(wt)}-test Oil WAPE = {wape_o:.2f}% (ref_date={loaded['ref_date']}, lambda_rec={loaded['lambda_rec']:.2f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=None, help="Optional path to raw dataset folder")
    args = ap.parse_args()
    main(args.data)
