"""Align high-frequency telemetry windows with well-test intervals."""

from __future__ import annotations

from typing import List, Sequence

import numpy as np
import pandas as pd


AGG_FEATURE_COLS: Sequence[str] = (
    "whp_psia",
    "whp_downstream_psia",
    "dh_pressure_psia",
    "dh_temperature_degf",
    "annulus_pressure_psia",
    "gas_lift_rate_mscfd",
    "choke_opening_pct",
    "dp_choke_psi",
    "choke_pressure_ratio",
    "is_subcritical_choke",
    "choke_flow_proxy",
    "choke_area_x_whp",
    "dp_tubing_psi",
    "tubing_grad_psi_ft",
    "dh_minus_pb_psi",
    "dp_gl_annulus_whp_psi",
    "gl_valve_submergence_ratio",
    "gl_to_choke_proxy",
    "dh_temp_minus_res_degf",
    "well_age_years",
    "tubing_id_sq_in2",
    "whp_std_15m",
    "annulus_std_15m",
    "dh_p_mean_15m",
    "whp_std_60m",
    "annulus_std_60m",
    "dh_p_mean_60m",
    "net_pay_thickness_ft",
    "porosity_frac",
    "oil_fvf_rb_stb",
    "oil_viscosity_cp",
    "solution_gor_at_pb_scf_stb",
    "gl_valve_port_64ths",
    "cum_oil_at_2025_01_01_stb",
)


def aggregate_well_test_windows(
    hf_features_df: pd.DataFrame,
    well_tests_df: pd.DataFrame,
    min_steady_coverage: float = 0.70,
) -> pd.DataFrame:
    """
    Aggregate steady-flowing high-frequency features over each [test_start, test_end]
    window for a single well (or all wells if combined) and join well-test ground truth.
    """
    rows: List[dict] = []
    wells_present = set(hf_features_df["well_name"].unique())
    wt_sub = well_tests_df[well_tests_df["well_name"].isin(wells_present)].copy()

    for _, test in wt_sub.iterrows():
        well = test["well_name"]
        t0, t1 = test["test_start"], test["test_end"]
        w_df = hf_features_df[
            (hf_features_df["well_name"] == well)
            & (hf_features_df["timestamp"] >= t0)
            & (hf_features_df["timestamp"] <= t1)
        ]
        n_total = len(w_df)
        steady = w_df[w_df["is_steady_flowing"]] if n_total else w_df
        n_steady = len(steady)
        coverage = float(n_steady / max(n_total, 1))

        rec = {
            "well_name": well,
            "test_start": t0,
            "test_end": t1,
            "n_window_rows": int(n_total),
            "n_steady_rows": int(n_steady),
            "steady_coverage": round(coverage, 4),
            "passes_qc": bool(n_steady > 0 and coverage >= min_steady_coverage),
            # Ground-truth well test targets (post-harmonization)
            "oil_rate_stbd": float(test["oil_rate_stbd"]),
            "water_rate_stbd": float(test["water_rate_stbd"]),
            "liquid_rate_stbd": float(test["liquid_rate_stbd"]),
            "gas_rate_mscfd": float(test["gas_rate_mscfd"]),
            "total_gas_rate_mscfd": float(test["total_gas_rate_mscfd"]),
            "water_cut_frac": float(test["water_cut_pct"]) / 100.0,
            "gor_scf_stb": float(test["gor_scf_stb"]),
        }

        source_df = steady if n_steady > 0 else w_df
        for col in AGG_FEATURE_COLS:
            if col not in source_df.columns or source_df.empty:
                rec[col] = np.nan
            else:
                rec[col] = float(source_df[col].median())

        rows.append(rec)

    return pd.DataFrame(rows)
