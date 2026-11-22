"""Leakage-free validation utilities (chronological split & GroupKFold by well)."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from vfm.modeling.hybrid_model import HybridVFM


RATE_EVAL_COLS = ("oil_rate_stbd", "water_rate_stbd", "gas_rate_mscfd", "liquid_rate_stbd")


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute MAE, RMSE, MAPE (%), and R2."""
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    denom = np.maximum(np.abs(y_true), 1.0)
    mape = float(np.mean(np.abs(err) / denom) * 100.0)
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    return {"mae": round(mae, 2), "rmse": round(rmse, 2), "mape_pct": round(mape, 2), "r2": round(r2, 4)}


def chronological_holdout_eval(
    dataset: pd.DataFrame,
    cutoff_utc: str = "2025-07-01",
    backend: str = "auto",
) -> Tuple[Dict[str, Dict[str, float]], pd.DataFrame]:
    """Train on tests before cutoff_utc and evaluate on tests on/after cutoff_utc."""
    cutoff = pd.Timestamp(cutoff_utc, tz="UTC")
    train_df = dataset[dataset["test_start"] < cutoff].copy()
    test_df = dataset[dataset["test_start"] >= cutoff].copy()
    if train_df.empty or test_df.empty:
        raise ValueError("Chronological split produced empty train or test set.")

    model = HybridVFM(backend=backend).fit(train_df)
    preds = model.predict(test_df)

    metrics = {
        col: compute_regression_metrics(test_df[col].to_numpy(), preds[col].to_numpy())
        for col in RATE_EVAL_COLS
    }
    eval_out = test_df[["well_name", "test_start", "test_end", *RATE_EVAL_COLS]].copy()
    for col in RATE_EVAL_COLS:
        eval_out[f"pred_{col}"] = preds[col].round(2)
    return metrics, eval_out


def leave_one_well_out_eval(
    dataset: pd.DataFrame,
    backend: str = "auto",
) -> Tuple[Dict[str, Dict[str, float]], pd.DataFrame]:
    """Leave-one-well-out cross-validation for unseen-well generalization."""
    oof_parts: List[pd.DataFrame] = []
    for well in sorted(dataset["well_name"].unique()):
        train_df = dataset[dataset["well_name"] != well]
        val_df = dataset[dataset["well_name"] == well]
        preds = HybridVFM(backend=backend).fit(train_df).predict(val_df)
        part = val_df[["well_name", "test_start", *RATE_EVAL_COLS]].copy()
        for col in RATE_EVAL_COLS:
            part[f"pred_{col}"] = preds[col].round(2)
        oof_parts.append(part)

    oof = pd.concat(oof_parts, ignore_index=True)
    metrics = {
        col: compute_regression_metrics(oof[col].to_numpy(), oof[f"pred_{col}"].to_numpy())
        for col in RATE_EVAL_COLS
    }
    return metrics, oof
