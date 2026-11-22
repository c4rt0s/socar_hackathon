"""Physics-informed feature engineering for gas-lift Virtual Flow Metering."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def add_physics_features(
    df: pd.DataFrame,
    completion: pd.DataFrame,
    pvt: pd.DataFrame,
    rolling_windows_min: Sequence[int] = (15, 60),
) -> pd.DataFrame:
    """
    Enrich 1-minute well telemetry with static completion/PVT properties and
    mechanistic hydraulic features across choke, tubing, and gas-lift annulus.
    """
    out = df.copy()

    # Join static well metadata (dynamically include any cum_oil* column rather than a year-specific name)
    base_comp_cols = [
        "well_name",
        "gauge_tvd_ft",
        "gl_valve_tvd_ft",
        "gl_valve_port_64ths",
        "tubing_id_in",
        "solution_gor_at_pb_scf_stb",
        "choke_max_size_64ths",
        "first_production_date",
    ]
    cum_oil_cols = [c for c in completion.columns if str(c).lower().startswith("cum_oil")]
    comp_cols = [c for c in base_comp_cols + cum_oil_cols if c in completion.columns]
    pvt_cols = [
        c for c in (
            "well_name",
            "net_pay_thickness_ft",
            "porosity_frac",
            "oil_fvf_rb_stb",
            "oil_viscosity_cp",
            "bubble_point_pressure_psia",
            "reservoir_temperature_degf",
        )
        if c in pvt.columns
    ]
    out = out.merge(completion[comp_cols], on="well_name", how="left")
    out = out.merge(pvt[pvt_cols], on="well_name", how="left")

    # 1. Choke hydraulics
    whp = out["whp_psia"].clip(lower=1e-3)
    dp_choke = (out["whp_psia"] - out["whp_downstream_psia"]).clip(lower=0.0)
    choke_frac = (out["choke_opening_pct"].fillna(0.0) / 100.0).clip(0.0, 1.0)
    choke_area = choke_frac ** 2
    choke_ratio = (out["whp_downstream_psia"] / whp).clip(0.0, 2.0)

    out["dp_choke_psi"] = dp_choke
    out["choke_pressure_ratio"] = choke_ratio
    out["is_subcritical_choke"] = (choke_ratio > 0.55).astype("float64")
    # Gilbert-style multiphase choke proxy (area * sqrt(dp))
    out["choke_flow_proxy"] = (choke_frac ** 1.89) * np.sqrt(dp_choke)
    out["choke_area_x_whp"] = choke_area * whp

    # 2. Tubing hydrostatic & friction lift proxy
    dp_tubing = (out["dh_pressure_psia"] - out["whp_psia"]).clip(lower=0.0)
    out["dp_tubing_psi"] = dp_tubing
    out["tubing_grad_psi_ft"] = dp_tubing / out["gauge_tvd_ft"].clip(lower=1.0)
    out["dh_minus_pb_psi"] = out["dh_pressure_psia"] - out["bubble_point_pressure_psia"]

    # 3. Gas-lift injection & annulus coupling
    out["dp_gl_annulus_whp_psi"] = out["annulus_pressure_psia"] - out["whp_psia"]
    out["gl_valve_submergence_ratio"] = out["gl_valve_tvd_ft"] / out["gauge_tvd_ft"].clip(lower=1.0)
    out["gl_to_choke_proxy"] = out["gas_lift_rate_mscfd"] / (out["choke_flow_proxy"] + 1e-2)

    # 4. Thermal & maturity proxies
    out["dh_temp_minus_res_degf"] = out["dh_temperature_degf"] - out["reservoir_temperature_degf"]
    well_age_days = (out["timestamp"] - out["first_production_date"]).dt.total_seconds() / 86400.0
    out["well_age_years"] = well_age_days / 365.25
    out["tubing_id_sq_in2"] = out["tubing_id_in"] ** 2

    # 5. Rolling stability / slugging features (turbulence & casing heading)
    for w in rolling_windows_min:
        out[f"whp_std_{w}m"] = out["whp_psia"].rolling(w, min_periods=3).std().fillna(0.0)
        out[f"annulus_std_{w}m"] = out["annulus_pressure_psia"].rolling(w, min_periods=3).std().fillna(0.0)
        out[f"dh_p_mean_{w}m"] = out["dh_pressure_psia"].rolling(w, min_periods=3).mean()

    return out
