"""Step 08: Independent chronological walk-forward and LOTO benchmark suite."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vfm.config import load_config
from vfm.features import aggregate_well_test_windows
from vfm.io import load_frame, load_static_tables
from vfm.modeling.hybrid_model import HybridVFM
from vfm.modeling.physics_vfm import PhysicsVFM, field_logit_wc_slope
from vfm.physics.reservoir import DEFAULT_REF_TIMESTAMP
from vfm.modeling.targets import reconstruct_phase_rates
from vfm.modeling.validation import RATE_EVAL_COLS, compute_regression_metrics
from vfm.physics.reservoir import fit_pr_and_pi


def collect_buildup_gradients(
    wt_df: pd.DataFrame,
    hf_cache: Dict[str, pd.DataFrame],
    completion: pd.DataFrame,
    pvt: pd.DataFrame,
) -> pd.DataFrame:
    """All build-up Pr/datum-TVD gradients with their shut-in end times (one pass)."""
    params = fit_pr_and_pi(wt_df, hf_cache, pvt, completion)
    rows = [
        {"well_name": w, "shut_in_end": pd.Timestamp(b["shut_in_end"]), "grad": b["pr_gradient_psi_ft"]}
        for w, p in params.items() for b in p["buildup_anchors"]
    ]
    return pd.DataFrame(rows, columns=["well_name", "shut_in_end", "grad"])


def field_gradient_prior(pbu_grads: pd.DataFrame, max_time: Optional[pd.Timestamp]) -> Optional[float]:
    """Median gradient of build-ups completed by `max_time` (all of them if None)."""
    sub = pbu_grads if max_time is None else pbu_grads[pbu_grads["shut_in_end"] <= max_time]
    return float(sub["grad"].median()) if len(sub) else None


MODELS_WF = ("B0_Persistence", "HybridVFM", "PhysicsVFM", "IPR_only", "Choke_only", "VLP_only")
MODELS_LOTO = ("B0_Persistence", "B1_LinearInterp", "HybridVFM", "PhysicsVFM", "IPR_only", "Choke_only", "VLP_only")


def _extract_test_window_hf(
    hf_well: pd.DataFrame,
    t_start: pd.Timestamp,
    t_end: pd.Timestamp,
) -> pd.DataFrame:
    ts = hf_well["timestamp"]
    win = hf_well[(ts >= t_start) & (ts <= t_end)].copy()
    if "is_steady_flowing" in win.columns and win["is_steady_flowing"].any():
        win = win[win["is_steady_flowing"]].copy()
    return win


def _summarize_physics_predictions(pred_df: pd.DataFrame, hf_win: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Aggregate 1-minute PhysicsVFM + single-estimator predictions over a test window."""
    out: Dict[str, Dict[str, float]] = {}

    # Full reconciled PhysicsVFM
    out["PhysicsVFM"] = {
        col: float(np.nanmean(pred_df[col].to_numpy(dtype="float64")))
        for col in RATE_EVAL_COLS
    }

    # Single-estimator phase rates using the same causal WC(t) & GOR_hat
    wc = pred_df["water_cut_pct"].to_numpy(dtype="float64") / 100.0
    gor = pred_df["gor_scf_stb"].to_numpy(dtype="float64")
    q_gl = hf_win["gas_lift_rate_mscfd"].to_numpy(dtype="float64")
    shut_in = pred_df["is_shut_in"].to_numpy(dtype=bool) if "is_shut_in" in pred_df else None

    for model_name, q_col in (("IPR_only", "q_liq_ipr"), ("Choke_only", "q_liq_choke"), ("VLP_only", "q_liq_vlp")):
        q_est = pred_df[q_col].to_numpy(dtype="float64")
        valid = np.isfinite(q_est)
        if not np.any(valid):
            out[model_name] = {col: np.nan for col in RATE_EVAL_COLS}
            continue
        rec = reconstruct_phase_rates(
            q_liq_stbd=q_est[valid],
            water_cut_frac=wc[valid],
            gor_form_scf_stb=gor[valid],
            gas_lift_rate_mscfd=q_gl[valid],
            is_shut_in=shut_in[valid] if shut_in is not None else None,
        )
        out[model_name] = {
            col: float(np.nanmean(rec[col]))
            for col in RATE_EVAL_COLS
        }

    return out


def run_walk_forward_benchmark(
    wt_df: pd.DataFrame,
    wt_features_df: pd.DataFrame,
    hf_cache: Dict[str, pd.DataFrame],
    completion: pd.DataFrame,
    pvt: pd.DataFrame,
    pbu_grads: pd.DataFrame,
) -> pd.DataFrame:
    """
    Strict chronological walk-forward validation per well:
    For test k >= 2 of well w (at t_start), train ONLY on tests with test_end <= t_start
    and HF telemetry with timestamp <= t_start.
    """
    rows: List[Dict[str, Any]] = []
    wt_sorted = wt_df.sort_values(["well_name", "test_start"]).reset_index(drop=True)

    # Group by test_start across wells so HybridVFM is trained once per causal cutoff
    for well, w_grp in wt_sorted.groupby("well_name", sort=True):
        w_grp = w_grp.sort_values("test_start").reset_index(drop=True)
        for k in range(1, len(w_grp)):
            test_row = w_grp.iloc[k]
            prev_row = w_grp.iloc[k - 1]
            t_cut = pd.Timestamp(test_row["test_start"])
            t_end = pd.Timestamp(test_row["test_end"])

            # 1. B0: Causal persistence
            b0_pred = {col: float(prev_row[col]) for col in RATE_EVAL_COLS}

            # 2. HybridVFM: train on all tests across all wells with test_end <= t_cut
            train_ml = wt_features_df[wt_features_df["test_end"] <= t_cut]
            test_ml = wt_features_df[
                (wt_features_df["well_name"] == well) & (wt_features_df["test_start"] == t_cut)
            ]
            if len(train_ml) >= 3 and len(test_ml) == 1:
                ml_pred_df = HybridVFM(backend="auto").fit(train_ml).predict(test_ml)
                ml_pred = {col: float(ml_pred_df[col].iloc[0]) for col in RATE_EVAL_COLS}
            else:
                ml_pred = {col: np.nan for col in RATE_EVAL_COLS}

            # 3. PhysicsVFM + single estimators: strictly causal max_time = t_cut
            wt_causal = wt_sorted[(wt_sorted["well_name"] == well) & (wt_sorted["test_end"] <= t_cut)]
            comp_w = completion[completion["well_name"] == well]
            pvt_w = pvt[pvt["well_name"] == well]
            hf_w = hf_cache[well]

            # Field Pr prior only from build-ups (any well) completed before the cutoff
            # Field water-cut slope only from tests (any well) ending before the cutoff
            phys = PhysicsVFM(**PHYS_KW, pr_gradient_prior_psi_ft=field_gradient_prior(pbu_grads, t_cut),
                              wc_global_slope=field_logit_wc_slope(wt_sorted, DEFAULT_REF_TIMESTAMP, t_cut)).fit(
                well_tests_df=wt_causal,
                hf_cache_by_well={well: hf_w},
                completion_df=comp_w,
                pvt_df=pvt_w,
                max_time=t_cut,
            )
            hf_win = _extract_test_window_hf(hf_w, t_cut, t_end)
            phys_pred_df = phys.predict(hf_win)
            phys_preds = _summarize_physics_predictions(phys_pred_df, hf_win)

            model_preds = {
                "B0_Persistence": b0_pred,
                "HybridVFM": ml_pred,
                **phys_preds,
            }
            for m_name, preds in model_preds.items():
                rec: Dict[str, Any] = {
                    "cv_mode": "walk_forward",
                    "model": m_name,
                    "well_name": well,
                    "test_idx": k + 1,
                    "test_start": t_cut,
                    "test_end": t_end,
                    "n_train_tests_well": k,
                }
                for col in RATE_EVAL_COLS:
                    rec[f"true_{col}"] = float(test_row[col])
                    rec[f"pred_{col}"] = float(preds[col])
                rows.append(rec)

    return pd.DataFrame(rows)


def run_loto_benchmark(
    wt_df: pd.DataFrame,
    wt_features_df: pd.DataFrame,
    hf_cache: Dict[str, pd.DataFrame],
    completion: pd.DataFrame,
    pvt: pd.DataFrame,
    pbu_grads: pd.DataFrame,
) -> pd.DataFrame:
    """
    Leave-One-Test-Out (LOTO) validation across all tests per well.
    Includes non-causal B1 linear interpolation reference.
    """
    rows: List[Dict[str, Any]] = []
    wt_sorted = wt_df.sort_values(["well_name", "test_start"]).reset_index(drop=True)

    for well, w_grp in wt_sorted.groupby("well_name", sort=True):
        w_grp = w_grp.sort_values("test_start").reset_index(drop=True)
        t_mids = (
            w_grp["test_start"] + (w_grp["test_end"] - w_grp["test_start"]) / 2
        ).astype("int64") / 1e9

        comp_w = completion[completion["well_name"] == well]
        pvt_w = pvt[pvt["well_name"] == well]
        hf_w = hf_cache[well]

        for k in range(len(w_grp)):
            test_row = w_grp.iloc[k]
            t_start = pd.Timestamp(test_row["test_start"])
            t_end = pd.Timestamp(test_row["test_end"])

            # 1. B0: Nearest prior test (or next test for k == 0)
            b0_idx = k - 1 if k > 0 else 1
            b0_pred = {col: float(w_grp.iloc[b0_idx][col]) for col in RATE_EVAL_COLS}

            # 2. B1: Linear interpolation between neighbouring tests (clamped at boundaries)
            if 0 < k < len(w_grp) - 1:
                t_prev = float(t_mids.iloc[k - 1])
                t_next = float(t_mids.iloc[k + 1])
                t_cur = float(t_mids.iloc[k])
                frac = (t_cur - t_prev) / max(t_next - t_prev, 1.0)
                b1_pred = {
                    col: float(w_grp.iloc[k - 1][col] + frac * (w_grp.iloc[k + 1][col] - w_grp.iloc[k - 1][col]))
                    for col in RATE_EVAL_COLS
                }
            else:
                b1_idx = 1 if k == 0 else len(w_grp) - 2
                b1_pred = {col: float(w_grp.iloc[b1_idx][col]) for col in RATE_EVAL_COLS}

            # 3. HybridVFM: leave out this test
            train_ml = wt_features_df[
                ~((wt_features_df["well_name"] == well) & (wt_features_df["test_start"] == t_start))
            ]
            test_ml = wt_features_df[
                (wt_features_df["well_name"] == well) & (wt_features_df["test_start"] == t_start)
            ]
            ml_pred_df = HybridVFM(backend="auto").fit(train_ml).predict(test_ml)
            ml_pred = {col: float(ml_pred_df[col].iloc[0]) for col in RATE_EVAL_COLS}

            # 4. PhysicsVFM + single estimators: leave out this test for this well
            wt_loto = w_grp.drop(index=k).copy()
            wt_all_but_k = wt_sorted[~((wt_sorted["well_name"] == well) & (wt_sorted["test_start"] == t_start))]
            phys = PhysicsVFM(**PHYS_KW, pr_gradient_prior_psi_ft=field_gradient_prior(pbu_grads, None),
                              wc_global_slope=field_logit_wc_slope(wt_all_but_k, DEFAULT_REF_TIMESTAMP, None)).fit(
                well_tests_df=wt_loto,
                hf_cache_by_well={well: hf_w},
                completion_df=comp_w,
                pvt_df=pvt_w,
                max_time=None,
            )
            hf_win = _extract_test_window_hf(hf_w, t_start, t_end)
            phys_pred_df = phys.predict(hf_win)
            phys_preds = _summarize_physics_predictions(phys_pred_df, hf_win)

            model_preds = {
                "B0_Persistence": b0_pred,
                "B1_LinearInterp": b1_pred,
                "HybridVFM": ml_pred,
                **phys_preds,
            }
            for m_name, preds in model_preds.items():
                rec: Dict[str, Any] = {
                    "cv_mode": "loto",
                    "model": m_name,
                    "well_name": well,
                    "test_idx": k + 1,
                    "test_start": t_start,
                    "test_end": t_end,
                    "n_train_tests_well": len(w_grp) - 1,
                }
                for col in RATE_EVAL_COLS:
                    rec[f"true_{col}"] = float(test_row[col])
                    rec[f"pred_{col}"] = float(preds[col])
                rows.append(rec)

    return pd.DataFrame(rows)


def _compute_summary_table(detail_df: pd.DataFrame, cv_mode: str, model_order: tuple[str, ...]) -> pd.DataFrame:
    sub_cv = detail_df[detail_df["cv_mode"] == cv_mode]
    summary_rows: List[Dict[str, Any]] = []
    for m_name in model_order:
        sub = sub_cv[sub_cv["model"] == m_name]
        row: Dict[str, Any] = {"cv_mode": cv_mode, "model": m_name, "n_tests": int(len(sub))}
        for col in RATE_EVAL_COLS:
            y_t = sub[f"true_{col}"].to_numpy(dtype="float64")
            y_p = sub[f"pred_{col}"].to_numpy(dtype="float64")
            valid = np.isfinite(y_t) & np.isfinite(y_p)
            if np.any(valid):
                metrics = compute_regression_metrics(y_t[valid], y_p[valid])
                row[f"{col}_mae"] = metrics["mae"]
                row[f"{col}_mape"] = metrics["mape_pct"]
                row[f"{col}_valid_n"] = int(np.sum(valid))
            else:
                row[f"{col}_mae"] = np.nan
                row[f"{col}_mape"] = np.nan
                row[f"{col}_valid_n"] = 0
        summary_rows.append(row)
    return pd.DataFrame(summary_rows)


def _format_markdown_table(summary_df: pd.DataFrame) -> str:
    headers = [
        "Model",
        "N",
        "Oil MAE",
        "Oil MAPE%",
        "Water MAE",
        "Water MAPE%",
        "Gas MAE",
        "Gas MAPE%",
        "Liquid MAE",
        "Liquid MAPE%",
    ]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for _, r in summary_df.iterrows():
        lines.append(
            f"| {r['model']} | {int(r['n_tests'])} | "
            f"{r['oil_rate_stbd_mae']:.2f} | {r['oil_rate_stbd_mape']:.2f}% | "
            f"{r['water_rate_stbd_mae']:.2f} | {r['water_rate_stbd_mape']:.2f}% | "
            f"{r['gas_rate_mscfd_mae']:.2f} | {r['gas_rate_mscfd_mape']:.2f}% | "
            f"{r['liquid_rate_stbd_mae']:.2f} | {r['liquid_rate_stbd_mape']:.2f}% |"
        )
    return "\n".join(lines)


PHYS_KW: Dict[str, Any] = {}  # dataset assumptions from the config (set in main)


def main() -> None:
    cfg = load_config()
    from vfm.modeling.physics_vfm import physics_kwargs_from_config
    PHYS_KW.update(physics_kwargs_from_config(cfg))
    cache_dir = cfg.resolve_path("cache_dir")
    raw_dir = cfg.resolve_path("raw_data_dir")
    out_dir = cfg.resolve_path("outputs_dir")
    out_dir.mkdir(parents=True, exist_ok=True)

    wt_clean = load_frame(cache_dir / "well_tests_harmonized")
    completion, pvt, _ = load_static_tables(
        raw_dir / "well_completion.csv",
        raw_dir / "reservoir_pvt.csv",
        raw_dir / "choke_events.csv",
        schema=cfg.schema,
    )

    hf_cache: Dict[str, pd.DataFrame] = {}
    wt_feat_parts: List[pd.DataFrame] = []
    for well in sorted(wt_clean["well_name"].unique()):
        hf_w = load_frame(cache_dir / "wells" / well)
        hf_cache[well] = hf_w
        wt_feat_parts.append(
            aggregate_well_test_windows(
                hf_w,
                wt_clean,
                min_steady_coverage=cfg.well_tests["min_steady_coverage"],
            )
        )
    wt_features_df = pd.concat(wt_feat_parts, ignore_index=True).sort_values(["well_name", "test_start"])

    pbu_grads = collect_buildup_gradients(wt_clean, hf_cache, completion, pvt)
    print(f"Build-up Pr gradients (psi/ft): {pbu_grads.round(4).to_dict(orient='records')}")
    print("Running strict chronological walk-forward benchmark...")
    wf_df = run_walk_forward_benchmark(wt_clean, wt_features_df, hf_cache, completion, pvt, pbu_grads)
    print("Running leave-one-test-out (LOTO) benchmark...")
    loto_df = run_loto_benchmark(wt_clean, wt_features_df, hf_cache, completion, pvt, pbu_grads)

    detail_df = pd.concat([wf_df, loto_df], ignore_index=True)
    detail_path = out_dir / "benchmark_detail.csv"
    detail_df.to_csv(detail_path, index=False)

    wf_summary = _compute_summary_table(detail_df, "walk_forward", MODELS_WF)
    loto_summary = _compute_summary_table(detail_df, "loto", MODELS_LOTO)

    # Per-well LOTO summary for PhysicsVFM vs HybridVFM vs B0
    well_rows: List[str] = []
    for well in sorted(wt_clean["well_name"].unique()):
        sub_w = detail_df[(detail_df["cv_mode"] == "loto") & (detail_df["well_name"] == well)]
        parts = [well]
        for m in ("B0_Persistence", "B1_LinearInterp", "HybridVFM", "PhysicsVFM"):
            sm = sub_w[sub_w["model"] == m]
            m_oil = compute_regression_metrics(sm["true_oil_rate_stbd"].to_numpy(), sm["pred_oil_rate_stbd"].to_numpy())
            m_liq = compute_regression_metrics(sm["true_liquid_rate_stbd"].to_numpy(), sm["pred_liquid_rate_stbd"].to_numpy())
            parts.append(f"{m_oil['mae']:.1f} ({m_oil['mape_pct']:.2f}%)")
            parts.append(f"{m_liq['mae']:.1f} ({m_liq['mape_pct']:.2f}%)")
        well_rows.append("| " + " | ".join(parts) + " |")

    md_lines = [
        "# Independent VFM Benchmark Report",
        "",
        "## Methodology & Anti-Leakage Guarantees",
        "- **Chronological Walk-Forward (`walk_forward`, N=75):** For each well and test $k \\ge 2$ starting at $t_\\text{start}$, all models are trained/calibrated strictly on well tests with `test_end <= t_start` and high-frequency telemetry (including PBU shut-ins) with `timestamp <= t_start`. Zero future data is visible.",
        "- **Leave-One-Test-Out (`loto`, N=85):** For each test $k$ of well $w$, that test is held out from per-well calibration (`PhysicsVFM`, `IPR_only`, `Choke_only`, `VLP_only`) and pooled training (`HybridVFM`).",
        "- **Baselines:** `B0_Persistence` carries the previous test forward (using the second test for $k=1$ in LOTO); `B1_LinearInterp` linearly interpolates between neighbouring test midpoints (non-causal LOTO reference, clamped at boundaries).",
        "- **Metrics:** `MAE` in `STB/d` (oil, water, liquid) or `Mscf/d` (formation gas); `MAPE` computed as `mean(|pred - true| / max(|true|, 1.0)) * 100%`.",
        "",
        "## 1. Chronological Walk-Forward Results (Strict Causal, N=75)",
        "",
        _format_markdown_table(wf_summary),
        "",
        "## 2. Leave-One-Test-Out (LOTO) Results (All 85 Tests)",
        "",
        _format_markdown_table(loto_summary),
        "",
        "## 3. Per-Well LOTO Comparison: Oil & Liquid MAE (MAPE%)",
        "",
        "| Well | B0 Oil | B0 Liq | B1 Oil | B1 Liq | HybridVFM Oil | HybridVFM Liq | PhysicsVFM Oil | PhysicsVFM Liq |",
        "|---|---|---|---|---|---|---|---|---|",
        *well_rows,
        "",
    ]

    md_path = out_dir / "benchmark.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    print("\n=== Walk-Forward Summary ===")
    print(wf_summary.to_string(index=False))
    print("\n=== LOTO Summary ===")
    print(loto_summary.to_string(index=False))
    print(f"\nSaved benchmark report -> {md_path}")
    print(f"Saved detailed results -> {detail_path}")


if __name__ == "__main__":
    main()
