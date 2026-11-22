"""Step 06: Calibrate per-well Choke & 2-segment VLP models with strict LOTO validation."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vfm.config import load_config
from vfm.io import load_frame, load_static_tables
from vfm.physics.choke import (
    _extract_test_choke_pct,
    calibrate_choke,
    predict_choke_q_liq,
)
from vfm.physics.vlp import (
    calibrate_vlp,
    predict_vlp_q_liq,
    vlp_pressure_drop,
)


def main() -> None:
    cfg = load_config()
    cache_dir = cfg.resolve_path("cache_dir")
    wells_cache = cache_dir / "wells"
    outputs_dir = cfg.resolve_path("outputs_dir")
    outputs_dir.mkdir(parents=True, exist_ok=True)

    wt = load_frame(cache_dir / "well_tests_harmonized").copy()
    wt["test_start"] = pd.to_datetime(wt["test_start"], utc=True)
    wt["test_end"] = pd.to_datetime(wt["test_end"], utc=True)
    wt = wt.sort_values(["well_name", "test_start"]).reset_index(drop=True)

    completion, pvt, _ = load_static_tables(
        cfg.resolve_path("well_completion_csv"),
        cfg.resolve_path("reservoir_pvt_csv"),
        cfg.resolve_path("choke_events_csv"),
        schema=cfg.schema,
    )

    hf_cache_by_well: Dict[str, pd.DataFrame] = {}
    for well in sorted(completion["well_name"].unique()):
        hf_cache_by_well[well] = load_frame(wells_cache / well)

    # dataset assumptions from the config (critical pressure ratio, wellhead temperature)
    ck = {"r_crit": float(cfg.assumptions.get("choke_critical_pressure_ratio", 0.55)),
          "wh_temp_degf": float(cfg.assumptions.get("wellhead_temp_degf", 120.0))}
    vk = {"wh_temp_degf": ck["wh_temp_degf"]}

    # 1. Calibrate full-dataset per-well Choke & VLP parameters (with internal LOTO uncertainties)
    choke_params = calibrate_choke(
        well_tests=wt,
        hf_cache_by_well=hf_cache_by_well,
        completion=completion,
        pvt=pvt,
        **ck,
    )
    vlp_params = calibrate_vlp(
        well_tests=wt,
        completion=completion,
        pvt=pvt,
        **vk,
    )

    # 2. Compute explicit per-test LOTO error table & linear-diameter vs linear-area comparison
    loto_rows: List[dict] = []
    comp_map = completion.set_index("well_name").to_dict(orient="index")
    pvt_map = pvt.set_index("well_name").to_dict(orient="index")

    per_well_lin_area_stats: Dict[str, dict] = {}
    for well in sorted(comp_map.keys()):
        w_wt = wt[wt["well_name"] == well].sort_values("test_start").reset_index(drop=True)
        if w_wt.empty:
            continue
        w_comp = completion[completion["well_name"] == well]
        w_pvt = pvt[pvt["well_name"] == well]
        choke_pct = _extract_test_choke_pct(w_wt, hf_cache_by_well.get(well), max_ts=None)
        w_wt = w_wt.copy()
        w_wt["choke_opening_pct"] = choke_pct

        # Equivalent opening % if choke_opening_pct had been linear in area rather than diameter
        w_wt_area = w_wt.copy()
        w_wt_area["choke_opening_pct"] = 100.0 * np.sqrt(np.clip(choke_pct, 0.0, 100.0) / 100.0)

        rs_pb = float(comp_map[well]["solution_gor_at_pb_scf_stb"])
        n_tests = len(w_wt)
        area_apes = []
        area_aes = []

        for k in range(n_tests):
            tr_wt = w_wt.drop(index=k).reset_index(drop=True)
            tr_wt_area = w_wt_area.drop(index=k).reset_index(drop=True)
            te_row = w_wt.iloc[k]
            te_area_pct = float(w_wt_area.iloc[k]["choke_opening_pct"])

            # Fit strictly on n - 1 training tests
            c_fold = calibrate_choke(tr_wt, hf_cache_by_well=None, completion=w_comp, pvt=w_pvt, compute_loto=False, **ck)[well]
            c_area_fold = calibrate_choke(tr_wt_area, hf_cache_by_well=None, completion=w_comp, pvt=w_pvt, compute_loto=False, **ck)[well]
            v_fold = calibrate_vlp(tr_wt, completion=w_comp, pvt=w_pvt, compute_loto=False, **vk)[well]

            wc_k = float(te_row["water_cut_pct"]) / 100.0
            whp_k = float(te_row["whp_psia"])
            whp_ds_k = float(te_row["whp_downstream_psia"])
            dhp_k = float(te_row["dh_pressure_psia"])
            dht_k = float(te_row["dh_temperature_degf"])
            qgl_k = float(te_row["gas_lift_rate_mscfd"])
            api_k = float(te_row["oil_gravity_api"])
            gsg_k = float(te_row["gas_sg"])
            sal_k = float(te_row["water_salinity_ppm"])
            q_true = float(te_row["liquid_rate_stbd"])

            q_choke_k = float(
                predict_choke_q_liq(
                    c_fold,
                    choke_opening_pct=float(te_row["choke_opening_pct"]),
                    whp_psia=whp_k,
                    whp_downstream_psia=whp_ds_k,
                    gas_lift_rate_mscfd=qgl_k,
                    wc_frac=wc_k,
                    gor_scf_stb=rs_pb,
                    oil_api=api_k,
                    gas_sg=gsg_k,
                    water_salinity_ppm=sal_k,
                )
            )
            q_area_k = float(
                predict_choke_q_liq(
                    c_area_fold,
                    choke_opening_pct=te_area_pct,
                    whp_psia=whp_k,
                    whp_downstream_psia=whp_ds_k,
                    gas_lift_rate_mscfd=qgl_k,
                    wc_frac=wc_k,
                    gor_scf_stb=rs_pb,
                    oil_api=api_k,
                    gas_sg=gsg_k,
                    water_salinity_ppm=sal_k,
                )
            )

            q_vlp_k = float(
                predict_vlp_q_liq(
                    v_fold,
                    whp_psia=whp_k,
                    dh_pressure_psia=dhp_k,
                    gas_lift_rate_mscfd=qgl_k,
                    wc_frac=wc_k,
                    gor_scf_stb=rs_pb,
                    res_temp_degf=dht_k,
                    oil_api=api_k,
                    gas_sg=gsg_k,
                    water_salinity_ppm=sal_k,
                )
            )
            dp_vlp_pred_k = float(
                vlp_pressure_drop(
                    q_liq_stbd=q_true,
                    wc_frac=wc_k,
                    whp_psia=whp_k,
                    gas_lift_rate_mscfd=qgl_k,
                    gor_scf_stb=rs_pb,
                    oil_api=api_k,
                    gas_sg=gsg_k,
                    lift_gas_sg=float(v_fold["lift_gas_sg"]),
                    gauge_tvd_ft=float(v_fold["gauge_tvd_ft"]),
                    gl_valve_tvd_ft=float(v_fold["gl_valve_tvd_ft"]),
                    tubing_id_in=float(v_fold["tubing_id_in"]),
                    res_temp_degf=dht_k,
                    bo=float(v_fold["oil_fvf_rb_stb"]),
                    bw=float(v_fold["water_fvf_rb_stb"]),
                    mu_oil_cp=float(v_fold["oil_viscosity_cp"]),
                    water_salinity_ppm=sal_k,
                    wh_temp_degf=float(v_fold["wh_temp_degf"]),
                    bubble_point_psia=float(v_fold["bubble_point_psia"]),
                    c_slip=float(v_fold["c_slip"]),
                    f_mult=float(v_fold["f_mult"]),
                    slip_k=float(v_fold["slip_k"]),
                    n_steps=int(v_fold["n_steps"]),
                )
            )

            choke_err = q_choke_k - q_true
            choke_ape = abs(choke_err) / max(q_true, 1.0) * 100.0
            area_err = q_area_k - q_true
            area_ape = abs(area_err) / max(q_true, 1.0) * 100.0
            area_aes.append(abs(area_err))
            area_apes.append(area_ape)

            vlp_err = q_vlp_k - q_true if np.isfinite(q_vlp_k) else np.nan
            vlp_ape = abs(vlp_err) / max(q_true, 1.0) * 100.0 if np.isfinite(vlp_err) else np.nan

            loto_rows.append(
                {
                    "well_name": well,
                    "test_idx": int(k),
                    "test_start": str(te_row["test_start"]),
                    "test_end": str(te_row["test_end"]),
                    "choke_opening_pct": round(float(te_row["choke_opening_pct"]), 2),
                    "whp_psia": round(whp_k, 3),
                    "whp_downstream_psia": round(whp_ds_k, 3),
                    "choke_pressure_ratio": round(whp_ds_k / max(whp_k, 1e-6), 4),
                    "dh_pressure_psia": round(dhp_k, 3),
                    "dp_vlp_meas_psi": round(dhp_k - whp_k, 3),
                    "gas_lift_rate_mscfd": round(qgl_k, 2),
                    "water_cut_frac": round(wc_k, 4),
                    "liquid_rate_stbd": round(q_true, 2),
                    "choke_cd_fold": round(float(c_fold["cd"]), 5),
                    "q_liq_choke_loto": round(q_choke_k, 2),
                    "choke_err_stbd": round(choke_err, 2),
                    "choke_abs_err_stbd": round(abs(choke_err), 2),
                    "choke_ape_pct": round(choke_ape, 3),
                    "q_liq_choke_lin_area_loto": round(q_area_k, 2),
                    "choke_lin_area_ape_pct": round(area_ape, 3),
                    "vlp_slip_k_fold": round(float(v_fold["slip_k"]), 4),
                    "dp_vlp_pred_loto_psi": round(dp_vlp_pred_k, 3),
                    "q_liq_vlp_loto": round(q_vlp_k, 2) if np.isfinite(q_vlp_k) else np.nan,
                    "vlp_err_stbd": round(vlp_err, 2) if np.isfinite(vlp_err) else np.nan,
                    "vlp_abs_err_stbd": round(abs(vlp_err), 2) if np.isfinite(vlp_err) else np.nan,
                    "vlp_ape_pct": round(vlp_ape, 3) if np.isfinite(vlp_ape) else np.nan,
                }
            )

        per_well_lin_area_stats[well] = {
            "mae": float(np.mean(area_aes)),
            "mape": float(np.mean(area_apes)),
        }

    # 3. Save summary params CSV & per-test LOTO error CSV
    param_rows: List[dict] = []
    for well in sorted(comp_map.keys()):
        cp = choke_params[well]
        vp = vlp_params[well]
        la = per_well_lin_area_stats.get(well, {"mae": np.nan, "mape": np.nan})
        param_rows.append(
            {
                "well_name": well,
                "n_tests": int(cp["n_tests"]),
                "choke_max_size_64ths": float(cp["choke_max_size_64ths"]),
                "tubing_id_in": float(cp["tubing_id_in"]),
                "gauge_tvd_ft": float(vp["gauge_tvd_ft"]),
                "gl_valve_tvd_ft": float(vp["gl_valve_tvd_ft"]),
                "solution_gor_scf_stb": float(cp["solution_gor_scf_stb"]),
                "choke_cd": round(float(cp["cd"]), 5),
                "choke_cd_std": round(float(cp["cd_std"]), 5) if np.isfinite(cp["cd_std"]) else np.nan,
                "choke_r_crit": float(cp["r_crit"]),
                "choke_in_sample_mae_stbd": round(float(cp["in_sample_mae_stbd"]), 3),
                "choke_loto_mae_stbd": round(float(cp["loto_mae_stbd"]), 3),
                "choke_loto_rmse_stbd": round(float(cp["loto_rmse_stbd"]), 3),
                "choke_loto_mape_pct": round(float(cp["loto_mape_pct"]), 3),
                "sigma_choke": round(float(cp["sigma_choke"]), 3),
                "choke_lin_area_loto_mae_stbd": round(float(la["mae"]), 3),
                "choke_lin_area_loto_mape_pct": round(float(la["mape"]), 3),
                "vlp_c_slip": float(vp["c_slip"]),
                "vlp_slip_k": round(float(vp["slip_k"]), 4),
                "vlp_f_mult": float(vp["f_mult"]),
                "vlp_dp_in_sample_mae_psi": round(float(vp["dp_in_sample_mae_psi"]), 3),
                "vlp_in_sample_mae_stbd": round(float(vp["in_sample_mae_stbd"]), 3),
                "vlp_loto_mae_stbd": round(float(vp["loto_mae_stbd"]), 3),
                "vlp_loto_rmse_stbd": round(float(vp["loto_rmse_stbd"]), 3),
                "vlp_loto_mape_pct": round(float(vp["loto_mape_pct"]), 3),
                "vlp_n_valid_loto": int(vp["n_valid_loto"]),
                "sigma_vlp": round(float(vp["sigma_vlp"]), 3),
            }
        )

    params_df = pd.DataFrame(param_rows)
    errors_df = pd.DataFrame(loto_rows)

    params_path = outputs_dir / "choke_vlp_params.csv"
    errors_path = outputs_dir / "choke_vlp_loto_errors.csv"
    params_df.to_csv(params_path, index=False)
    errors_df.to_csv(errors_path, index=False)

    print(f"Saved choke & VLP parameters -> {params_path}")
    print(f"Saved LOTO per-test errors    -> {errors_path} ({len(errors_df)} tests)")
    print(
        params_df[
            [
                "well_name",
                "choke_cd",
                "choke_loto_mae_stbd",
                "choke_loto_mape_pct",
                "choke_lin_area_loto_mape_pct",
                "vlp_slip_k",
                "vlp_dp_in_sample_mae_psi",
                "vlp_loto_mae_stbd",
                "vlp_loto_mape_pct",
            ]
        ].to_string(index=False)
    )
    print(
        f"\nChoke Linear-Diameter Mean LOTO: MAE={params_df['choke_loto_mae_stbd'].mean():.2f} STB/d, "
        f"RMSE={params_df['choke_loto_rmse_stbd'].mean():.2f} STB/d, "
        f"MAPE={params_df['choke_loto_mape_pct'].mean():.2f}% "
        f"(vs Linear-Area MAPE={params_df['choke_lin_area_loto_mape_pct'].mean():.2f}%)"
    )
    print(
        f"VLP 2-Segment Mechanistic Mean LOTO: MAE={params_df['vlp_loto_mae_stbd'].mean():.2f} STB/d, "
        f"RMSE={params_df['vlp_loto_rmse_stbd'].mean():.2f} STB/d, "
        f"MAPE={params_df['vlp_loto_mape_pct'].mean():.2f}% "
        f"(valid={int(params_df['vlp_n_valid_loto'].sum())}/{int(params_df['n_tests'].sum())})"
    )


if __name__ == "__main__":
    main()
