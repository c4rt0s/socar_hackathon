"""Step 05: Estimate reservoir pressure build-ups, Pr(t) decline, and PI (IPR) per well."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vfm.config import load_config
from vfm.io import load_frame, load_static_tables
from vfm.physics.reservoir import fit_pr_and_pi


def main() -> None:
    cfg = load_config()
    cache_dir = cfg.resolve_path("cache_dir")
    wells_cache = cache_dir / "wells"
    outputs_dir = cfg.resolve_path("outputs_dir")
    outputs_dir.mkdir(parents=True, exist_ok=True)

    wt = load_frame(cache_dir / "well_tests_harmonized")
    completion, pvt, _ = load_static_tables(
        cfg.resolve_path("well_completion_csv"),
        cfg.resolve_path("reservoir_pvt_csv"),
        cfg.resolve_path("choke_events_csv"),
        schema=cfg.schema,
    )

    hf_cache_by_well: dict[str, pd.DataFrame] = {}
    for well in sorted(completion["well_name"].unique()):
        hf_cache_by_well[well] = load_frame(wells_cache / well)

    fitted = fit_pr_and_pi(
        well_tests=wt,
        hf_cache_by_well=hf_cache_by_well,
        pvt=pvt,
        completion=completion,
    )

    rows = []
    for well in sorted(fitted):
        p = fitted[well]
        b0 = p["buildup_anchors"][0] if p["buildup_anchors"] else {}
        rows.append(
            {
                "well_name": well,
                "has_pbu_anchor": bool(p["has_pbu_anchor"]),
                "pr_prior_source": p["pr_prior_source"],
                "pr_gradient_prior_psi_ft": round(float(p["pr_gradient_prior_psi_ft"]), 4)
                if p["pr_gradient_prior_psi_ft"] is not None else np.nan,
                "Pr0_gradient_psi_ft": round(float(p["Pr0_datum_psia"]) / float(p["datum_tvd_ft"]), 4),
                "n_tests": int(p["n_tests"]),
                "n_buildups": int(p["n_buildups"]),
                "J_stbd_psi": round(float(p["J"]), 6),
                "Pr0_datum_psia": round(float(p["Pr0_datum_psia"]), 3),
                "decline_psi_per_day": round(float(p["decline_psi_per_day"]), 6),
                "datum_tvd_ft": float(p["datum_tvd_ft"]),
                "gauge_tvd_ft": float(p["gauge_tvd_ft"]),
                "bubble_point_psia": float(p["bubble_point_psia"]),
                "buildup_start": str(b0["shut_in_start"]) if b0 else "",
                "buildup_duration_hr": round(float(b0["duration_hr"]), 2) if b0 else np.nan,
                "buildup_pr_gauge_psia": round(float(b0["pr_gauge_psia"]), 3) if b0 else np.nan,
                "buildup_pr_datum_psia": round(float(b0["pr_datum_psia"]), 3) if b0 else np.nan,
                "buildup_p_star_gauge_psia": round(float(b0["p_star_gauge_psia"]), 3) if b0 else np.nan,
                "buildup_p_end_gauge_psia": round(float(b0["p_end_gauge_psia"]), 3) if b0 else np.nan,
                "buildup_kh_md_ft": round(float(b0["kh_md_ft"]), 2) if b0 else np.nan,
                "buildup_k_md": round(float(b0["k_md"]), 2) if b0 else np.nan,
                "buildup_skin": round(float(b0["skin"]), 3) if b0 else np.nan,
                "buildup_t_pss_hr": round(float(b0["t_pss_hr"]), 2) if b0 else np.nan,
                "buildup_quality_flag": str(b0["quality_flag"]) if b0 else "NO_SHUTIN",
                "in_sample_mae_stbd": round(float(p["in_sample_mae_stbd"]), 3),
                "in_sample_rmse_stbd": round(float(p["in_sample_rmse_stbd"]), 3),
                "loto_mae_stbd": round(float(p["loto_mae_stbd"]), 3),
                "loto_rmse_stbd": round(float(p["loto_rmse_stbd"]), 3),
                "loto_mape_pct": round(float(p["loto_mape_pct"]), 3),
                "sigma_ipr": round(float(p["sigma_ipr"]), 3),
            }
        )

    out_df = pd.DataFrame(rows)
    out_path = outputs_dir / "reservoir_params.csv"
    out_df.to_csv(out_path, index=False)

    print(f"Saved reservoir parameters to {out_path}")
    print(
        out_df[
            [
                "well_name",
                "has_pbu_anchor",
                "pr_prior_source",
                "Pr0_gradient_psi_ft",
                "J_stbd_psi",
                "Pr0_datum_psia",
                "decline_psi_per_day",
                "buildup_pr_datum_psia",
                "buildup_p_star_gauge_psia",
                "loto_mae_stbd",
                "loto_rmse_stbd",
                "loto_mape_pct",
            ]
        ].to_string(index=False)
    )
    print(
        f"\nMean LOTO MAE: {out_df['loto_mae_stbd'].mean():.2f} stb/d | "
        f"Mean LOTO RMSE: {out_df['loto_rmse_stbd'].mean():.2f} stb/d | "
        f"Mean LOTO MAPE: {out_df['loto_mape_pct'].mean():.2f}%"
    )


if __name__ == "__main__":
    main()
