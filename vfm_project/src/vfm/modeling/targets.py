"""Decomposed physical rate targets and mass-balanced reconstruction."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def decompose_well_test_targets(wt_df: pd.DataFrame) -> pd.DataFrame:
    """Extract physically bounded targets (q_liq, wc_frac, gor_form_scf_stb)."""
    q_liq = (wt_df["oil_rate_stbd"] + wt_df["water_rate_stbd"]).clip(lower=0.0)
    wc = (wt_df["water_rate_stbd"] / q_liq.replace(0.0, np.nan)).fillna(0.0).clip(0.0, 1.0)
    q_oil = wt_df["oil_rate_stbd"].clip(lower=1e-6)
    gor = (wt_df["gas_rate_mscfd"].clip(lower=0.0) * 1000.0) / q_oil
    return pd.DataFrame({
        "q_liq_stbd": q_liq,
        "water_cut_frac": wc,
        "gor_form_scf_stb": gor,
    }, index=wt_df.index)


def reconstruct_phase_rates(
    q_liq_stbd: np.ndarray,
    water_cut_frac: np.ndarray,
    gor_form_scf_stb: np.ndarray,
    gas_lift_rate_mscfd: np.ndarray,
    is_shut_in: np.ndarray | None = None,
) -> Dict[str, np.ndarray]:
    """
    Reconstruct non-negative phase rates (qo, qw, qg_form, qg_total) from
    decomposed targets, zeroing shut-in intervals identically.
    """
    liq = np.clip(np.asarray(q_liq_stbd, dtype="float64"), 0.0, None)
    wc = np.clip(np.asarray(water_cut_frac, dtype="float64"), 0.0, 1.0)
    gor = np.clip(np.asarray(gor_form_scf_stb, dtype="float64"), 0.0, None)
    gl = np.clip(np.asarray(gas_lift_rate_mscfd, dtype="float64"), 0.0, None)

    q_oil = liq * (1.0 - wc)
    q_water = liq * wc
    q_gas_form = (q_oil * gor) / 1000.0
    q_gas_total = q_gas_form + gl

    if is_shut_in is not None:
        active = (~np.asarray(is_shut_in, dtype=bool)).astype("float64")
        liq = liq * active
        q_oil = q_oil * active
        q_water = q_water * active
        q_gas_form = q_gas_form * active
        q_gas_total = q_gas_total * active

    return {
        "liquid_rate_stbd": liq,
        "oil_rate_stbd": q_oil,
        "water_rate_stbd": q_water,
        "gas_rate_mscfd": q_gas_form,
        "total_gas_rate_mscfd": q_gas_total,
        "water_cut_pct": wc * 100.0,
    }
