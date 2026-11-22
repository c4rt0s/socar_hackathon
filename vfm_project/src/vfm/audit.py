"""Multi-CSV VFM evaluation, schema audit, pairwise diff, and grader metric backcalculator."""

from __future__ import annotations

import base64
import io
import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

PHASES: Tuple[Tuple[str, str, str], ...] = (
    ("oil", "oil_rate_stbd", "STB/d"),
    ("water", "water_rate_stbd", "STB/d"),
    ("gas", "gas_rate_mscfd", "Mscf/d"),
)

_COL_ALIASES: Dict[str, Tuple[str, ...]] = {
    "date": ("date", "timestamp", "time", "datetime", "day"),
    "well_name": ("well_name", "well", "wellname"),
    "oil_rate_stbd": ("oil_rate_stbd", "oil_rate_alloc_stbd", "oil_rate_vfm_stbd", "oil", "qo", "pred_oil_rate_stbd"),
    "water_rate_stbd": ("water_rate_stbd", "water_rate_alloc_stbd", "water_rate_vfm_stbd", "water", "qw", "pred_water_rate_stbd"),
    "gas_rate_mscfd": ("gas_rate_mscfd", "gas_rate_alloc_mscfd", "gas_rate_vfm_mscfd", "gas", "qg", "pred_gas_rate_mscfd"),
}


def _find_col(cols: Sequence[str], candidates: Tuple[str, ...]) -> Optional[str]:
    low = {c.lower().strip(): c for c in cols}
    for cand in candidates:
        if cand in low:
            return low[cand]
    return None


def parse_submission_bytes(raw: bytes, filename: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Load an uploaded CSV or Excel VFM submission into canonical columns + format audit."""
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else "csv"
    if ext in ("xlsx", "xls"):
        df_raw = pd.read_excel(io.BytesIO(raw))
    else:
        df_raw = pd.read_csv(io.BytesIO(raw))

    orig_cols = list(df_raw.columns)
    mapped: Dict[str, str] = {}
    missing: List[str] = []
    for canon, cands in _COL_ALIASES.items():
        found = _find_col(orig_cols, cands)
        if found is None:
            missing.append(canon)
        else:
            mapped[canon] = found
    if missing:
        raise ValueError(f"{filename}: missing required columns {missing} (found {orig_cols})")

    out = pd.DataFrame({
        "date": pd.to_datetime(df_raw[mapped["date"]], utc=True, errors="coerce").dt.strftime("%Y-%m-%d"),
        "well_name": df_raw[mapped["well_name"]].astype(str).str.strip(),
        "oil_rate_stbd": pd.to_numeric(df_raw[mapped["oil_rate_stbd"]], errors="coerce"),
        "water_rate_stbd": pd.to_numeric(df_raw[mapped["water_rate_stbd"]], errors="coerce"),
        "gas_rate_mscfd": pd.to_numeric(df_raw[mapped["gas_rate_mscfd"]], errors="coerce"),
    })
    out["liquid_rate_stbd"] = out["oil_rate_stbd"] + out["water_rate_stbd"]

    n_rows = int(len(out))
    n_bad_dates = int(out["date"].isna().sum())
    n_dups = int(out.duplicated(subset=["well_name", "date"]).sum())
    n_nan = int(out[["oil_rate_stbd", "water_rate_stbd", "gas_rate_mscfd"]].isna().any(axis=1).sum())
    n_neg = int((out[["oil_rate_stbd", "water_rate_stbd", "gas_rate_mscfd"]] < 0).any(axis=1).sum())
    valid_dates = out["date"].dropna()
    audit = {
        "filename": filename,
        "column_map": mapped,
        "date_column_exact": mapped["date"] == "date",
        "n_rows": n_rows,
        "n_wells": int(out["well_name"].nunique()),
        "date_min": str(valid_dates.min()) if len(valid_dates) else None,
        "date_max": str(valid_dates.max()) if len(valid_dates) else None,
        "n_bad_dates": n_bad_dates,
        "n_duplicate_well_date": n_dups,
        "n_nan_rows": n_nan,
        "n_negative_rows": n_neg,
    }
    clean = out.dropna(subset=["date"]).drop_duplicates(subset=["well_name", "date"], keep="last")
    return clean.sort_values(["well_name", "date"]).reset_index(drop=True), audit


def decode_uploaded_file(spec: Mapping[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Decode base64-encoded browser upload."""
    name = str(spec.get("name") or "upload.csv")
    b64 = str(spec.get("content_b64") or "")
    if "," in b64 and b64.startswith("data:"):
        b64 = b64.split(",", 1)[1]
    raw = base64.b64decode(b64)
    return parse_submission_bytes(raw, name)


def compute_error_metrics(y_true: np.ndarray, y_pred: np.ndarray, wells: np.ndarray) -> Dict[str, Optional[float]]:
    """Compute full suite of candidate regression metrics for one phase."""
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    if not ok.any():
        return {k: None for k in ("n", "mae", "rmse", "mape", "smape", "wape", "nrmse", "r2_pooled", "r2_per_well_mean", "bias_pct")}
    yt, yp, ww = y_true[ok], y_pred[ok], wells[ok]
    err = yp - yt
    abs_err = np.abs(err)
    mae = float(np.mean(abs_err))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mape = float(np.mean(abs_err / np.maximum(np.abs(yt), 1.0)) * 100.0)
    smape = float(np.mean(2.0 * abs_err / np.maximum(np.abs(yt) + np.abs(yp), 1.0)) * 100.0)
    sum_true = float(np.sum(np.abs(yt)))
    wape = float(np.sum(abs_err) / max(sum_true, 1.0) * 100.0)
    mean_true = float(np.mean(np.abs(yt)))
    nrmse = float(rmse / max(mean_true, 1.0) * 100.0)
    bias_pct = float(np.sum(err) / max(sum_true, 1.0) * 100.0)
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    r2_pooled = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else None

    well_r2s: List[float] = []
    for w in np.unique(ww):
        m = ww == w
        if m.sum() < 2:
            continue
        wt_y, wt_p = yt[m], yp[m]
        st = float(np.sum((wt_y - np.mean(wt_y)) ** 2))
        if st > 1e-9:
            well_r2s.append(float(1.0 - np.sum((wt_p - wt_y) ** 2) / st))
    r2_well_mean = float(np.mean(well_r2s)) if well_r2s else None

    return {
        "n": int(ok.sum()),
        "mae": round(mae, 3),
        "rmse": round(rmse, 3),
        "mape": round(mape, 3),
        "smape": round(smape, 3),
        "wape": round(wape, 3),
        "nrmse": round(nrmse, 3),
        "r2_pooled": round(r2_pooled, 5) if r2_pooled is not None else None,
        "r2_per_well_mean": round(r2_well_mean, 5) if r2_well_mean is not None else None,
        "bias_pct": round(bias_pct, 3),
    }


def build_test_reference(well_tests: pd.DataFrame, daily_meta: Optional[pd.DataFrame] = None,
                         q3_cutoff: Optional[str] = None) -> pd.DataFrame:
    """Build daily truth rows from well tests (plus optional uptime & lift-gas context)."""
    wt = well_tests.copy()
    wt["test_start"] = pd.to_datetime(wt["test_start"], utc=True)
    wt["test_end"] = pd.to_datetime(wt["test_end"], utc=True)
    if q3_cutoff:
        wt = wt[wt["test_start"] >= pd.Timestamp(q3_cutoff, tz="UTC")]
    wt["date"] = (wt["test_start"] + (wt["test_end"] - wt["test_start"]) / 2).dt.strftime("%Y-%m-%d")
    cols = ["well_name", "date", "oil_rate_stbd", "water_rate_stbd", "gas_rate_mscfd", "liquid_rate_stbd"]
    ref = wt[cols].groupby(["well_name", "date"], as_index=False).mean()
    if daily_meta is not None and not daily_meta.empty:
        dm = daily_meta.copy()
        dm["date"] = pd.to_datetime(dm["date"], utc=True).dt.strftime("%Y-%m-%d")
        keep = [c for c in ("well_name", "date", "uptime_frac", "daily_gas_lift_mscfd", "total_gas_vfm_mscfd") if c in dm.columns]
        ref = ref.merge(dm[keep], on=["well_name", "date"], how="left")
    return ref


def evaluate_submission(
    sub_df: pd.DataFrame,
    ref_df: pd.DataFrame,
    audit: Dict[str, Any],
    reported_score: Optional[float] = None,
    daily_meta: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Evaluate one submission against `ref_df`, returning per-phase, per-well, and format checks."""
    merged = ref_df.merge(sub_df, on=["well_name", "date"], how="inner", suffixes=("_ref", "_sub"))
    n_matched = int(len(merged))
    n_ref = int(len(ref_df))

    metrics: Dict[str, Dict[str, Optional[float]]] = {}
    wells_arr = merged["well_name"].to_numpy()
    for phase, col, _ in PHASES + (("liquid", "liquid_rate_stbd", "STB/d"),):
        metrics[phase] = compute_error_metrics(
            merged[f"{col}_ref"].to_numpy(dtype="float64"),
            merged[f"{col}_sub"].to_numpy(dtype="float64"),
            wells_arr,
        )

    # Check shut-in day zeroing and total-gas vs formation-gas hypotheses
    shutin_check: Dict[str, Any] = {}
    if daily_meta is not None and not daily_meta.empty:
        dm = daily_meta.copy()
        dm["date"] = pd.to_datetime(dm["date"], utc=True).dt.strftime("%Y-%m-%d")
        dm_m = dm.merge(sub_df, on=["well_name", "date"], how="inner")
        if "flowing_minutes" in dm_m.columns and "is_imputed_day" in dm_m.columns:
            full_si = (~dm_m["is_imputed_day"]) & (dm_m["flowing_minutes"] == 0)
            shutin_check["n_full_shutin_days"] = int(full_si.sum())
            shutin_check["n_nonzero_shutin_days"] = int((full_si & (dm_m["oil_rate_stbd"] > 1.0)).sum()) if full_si.any() else 0

    gas_total_alt = None
    if "daily_gas_lift_mscfd" in merged.columns and merged["daily_gas_lift_mscfd"].notna().any():
        gl = merged["daily_gas_lift_mscfd"].fillna(0.0).to_numpy(dtype="float64")
        gas_total_alt = compute_error_metrics(
            merged["gas_rate_mscfd_ref"].to_numpy(dtype="float64") + gl,
            merged["gas_rate_mscfd_sub"].to_numpy(dtype="float64"),
            wells_arr,
        )

    uptime_adj_oil = None
    if "uptime_frac" in merged.columns and merged["uptime_frac"].notna().any():
        up = np.clip(merged["uptime_frac"].fillna(1.0).to_numpy(dtype="float64"), 0.05, 1.0)
        uptime_adj_oil = compute_error_metrics(
            merged["oil_rate_stbd_ref"].to_numpy(dtype="float64"),
            merged["oil_rate_stbd_sub"].to_numpy(dtype="float64") / up,
            wells_arr,
        )

    per_well: List[Dict[str, Any]] = []
    for w, g in merged.groupby("well_name"):
        w_row: Dict[str, Any] = {"well_name": w, "n": int(len(g))}
        for phase, col, _ in PHASES + (("liquid", "liquid_rate_stbd", "STB/d"),):
            yt = g[f"{col}_ref"].to_numpy(dtype="float64")
            yp = g[f"{col}_sub"].to_numpy(dtype="float64")
            w_row[f"{phase}_mape"] = round(float(np.mean(np.abs(yp - yt) / np.maximum(np.abs(yt), 1.0)) * 100.0), 2)
            w_row[f"{phase}_mae"] = round(float(np.mean(np.abs(yp - yt))), 2)
            w_row[f"{phase}_bias_pct"] = round(float(np.sum(yp - yt) / max(float(np.sum(np.abs(yt))), 1.0) * 100.0), 2)
        per_well.append(w_row)

    # Composite summaries
    owg_mape = [metrics[p]["mape"] for p in ("oil", "water", "gas") if metrics[p]["mape"] is not None]
    owg_r2p = [metrics[p]["r2_pooled"] for p in ("oil", "water", "gas") if metrics[p]["r2_pooled"] is not None]
    owg_r2w = [metrics[p]["r2_per_well_mean"] for p in ("oil", "water", "gas") if metrics[p]["r2_per_well_mean"] is not None]
    return {
        "name": audit["filename"],
        "reported_score": reported_score,
        "audit": audit,
        "n_ref": n_ref,
        "n_matched": n_matched,
        "coverage_pct": round(100.0 * n_matched / max(n_ref, 1), 1),
        "metrics": metrics,
        "mean_owg_mape": round(float(np.mean(owg_mape)), 3) if owg_mape else None,
        "mean_owg_r2_pooled": round(float(np.mean(owg_r2p)), 5) if owg_r2p else None,
        "mean_owg_r2_per_well": round(float(np.mean(owg_r2w)), 5) if owg_r2w else None,
        "shutin_check": shutin_check,
        "gas_if_scored_against_total_gas": gas_total_alt,
        "oil_if_uptime_adjusted": uptime_adj_oil,
        "per_well": sorted(per_well, key=lambda r: -(r.get("oil_mape") or 0.0)),
    }


def pairwise_comparison(named_dfs: List[Tuple[str, pd.DataFrame]]) -> List[Dict[str, Any]]:
    """Compare all pairs of submissions to highlight which phases and wells diverge most."""
    out: List[Dict[str, Any]] = []
    for i in range(len(named_dfs)):
        for j in range(i + 1, len(named_dfs)):
            na, dfa = named_dfs[i]
            nb, dfb = named_dfs[j]
            m = dfa.merge(dfb, on=["well_name", "date"], suffixes=("_a", "_b"))
            if m.empty:
                continue
            item: Dict[str, Any] = {"file_a": na, "file_b": nb, "n_common_rows": int(len(m)), "phases": {}}
            for phase, col, _ in PHASES:
                va = m[f"{col}_a"].to_numpy(dtype="float64")
                vb = m[f"{col}_b"].to_numpy(dtype="float64")
                diff = vb - va
                item["phases"][phase] = {
                    "mae": round(float(np.mean(np.abs(diff))), 3),
                    "mape_pct": round(float(np.mean(np.abs(diff) / np.maximum(np.abs(va), 1.0)) * 100.0), 3),
                    "mean_diff": round(float(np.mean(diff)), 3),
                }
            # Top 3 diverging wells by oil MAE
            w_diffs = []
            for w, g in m.groupby("well_name"):
                d_o = np.mean(np.abs(g["oil_rate_stbd_b"].to_numpy(float) - g["oil_rate_stbd_a"].to_numpy(float)))
                d_g = np.mean(np.abs(g["gas_rate_mscfd_b"].to_numpy(float) - g["gas_rate_mscfd_a"].to_numpy(float)))
                w_diffs.append({"well_name": w, "oil_mae_diff": round(float(d_o), 2), "gas_mae_diff": round(float(d_g), 2)})
            w_diffs.sort(key=lambda r: -r["oil_mae_diff"])
            item["top_diverging_wells"] = w_diffs[:5]
            out.append(item)
    return out


def backcalculate_metrics(evals: List[Dict[str, Any]], max_score: float = 30.0) -> Dict[str, Any]:
    """Backcalculate candidate grader formulas from reported scores vs observed reference metrics."""
    phase_weights = {
        "Equal (O, W, G)": (1 / 3, 1 / 3, 1 / 3),
        "50/25/25 (Oil heavy)": (0.5, 0.25, 0.25),
        "Oil only": (1.0, 0.0, 0.0),
        "Gas only": (0.0, 0.0, 1.0),
    }

    def _wmetric(ev: Dict[str, Any], key: str, weights: Tuple[float, float, float]) -> Optional[float]:
        vals = [ev["metrics"][p].get(key) for p in ("oil", "water", "gas")]
        if any(v is None for v in vals):
            return None
        return sum(w * float(v) for w, v in zip(weights, vals))

    # Candidate formulas: name -> callable(ev) -> predicted score in [0, max_score]
    candidates: List[Tuple[str, Any]] = []
    for w_name, ws in phase_weights.items():
        for cap in (15.0, 20.0, 25.0, 30.0, 50.0, 100.0):
            candidates.append((
                f"Linear MAPE (cap={cap:g}%, {w_name})",
                lambda ev, w=ws, c=cap: _clip_score(max_score * (1.0 - (_wmetric(ev, "mape", w) or 0.0) / c), max_score),
            ))
        candidates.append((
            f"R2 pooled × {max_score:g} ({w_name})",
            lambda ev, w=ws: _clip_score(max_score * (_wmetric(ev, "r2_pooled", w) or 0.0), max_score),
        ))
        candidates.append((
            f"R2 per-well × {max_score:g} ({w_name})",
            lambda ev, w=ws: _clip_score(max_score * (_wmetric(ev, "r2_per_well_mean", w) or 0.0), max_score),
        ))
        candidates.append((
            f"Inverse MAPE: {max_score:g} / (1 + MAPE/10%, {w_name})",
            lambda ev, w=ws: _clip_score(max_score / (1.0 + (_wmetric(ev, "mape", w) or 0.0) / 10.0), max_score),
        ))
        candidates.append((
            f"Linear WAPE (cap=25%, {w_name})",
            lambda ev, w=ws: _clip_score(max_score * (1.0 - (_wmetric(ev, "wape", w) or 0.0) / 25.0), max_score),
        ))
        candidates.append((
            f"Linear NRMSE (cap=30%, {w_name})",
            lambda ev, w=ws: _clip_score(max_score * (1.0 - (_wmetric(ev, "nrmse", w) or 0.0) / 30.0), max_score),
        ))

    scored = [ev for ev in evals if ev.get("reported_score") is not None and math.isfinite(ev["reported_score"])]
    formula_rows: List[Dict[str, Any]] = []
    for fname, fn in candidates:
        preds: List[float] = []
        ok = True
        for ev in evals:
            try:
                p = fn(ev)
            except Exception:
                p = None
            if p is None or not math.isfinite(p):
                ok = False
                break
            preds.append(round(float(p), 2))
        if not ok:
            continue
        row: Dict[str, Any] = {"formula": fname, "predicted_scores": preds}
        if scored:
            y_rep = np.array([float(ev["reported_score"]) for ev in scored], dtype="float64")
            y_prd = np.array([float(fn(ev)) for ev in scored], dtype="float64")
            row["mae_vs_reported"] = round(float(np.mean(np.abs(y_prd - y_rep))), 3)
            if len(scored) >= 2 and np.std(y_rep) > 1e-9 and np.std(y_prd) > 1e-9:
                row["corr"] = round(float(np.corrcoef(y_rep, y_prd)[0, 1]), 4)
            else:
                row["corr"] = None
        else:
            row["mae_vs_reported"] = None
            row["corr"] = None
        formula_rows.append(row)

    if scored:
        formula_rows.sort(key=lambda r: (r["mae_vs_reported"] if r["mae_vs_reported"] is not None else 1e9))

    # Also compute implied hidden-test error for any single reported score (e.g. 20/30 -> ratio = 0.667)
    implied: List[Dict[str, Any]] = []
    for ev in scored:
        s = float(ev["reported_score"])
        frac = min(max(s / max(max_score, 1e-6), 0.0), 1.0)
        implied.append({
            "name": ev["name"],
            "reported_score": s,
            "score_pct": round(frac * 100.0, 1),
            "implied_r2": round(frac, 4),
            "implied_mape_if_cap30": round((1.0 - frac) * 30.0, 2),
            "implied_mape_if_cap25": round((1.0 - frac) * 25.0, 2),
            "implied_mape_if_cap20": round((1.0 - frac) * 20.0, 2),
            "implied_mape_if_inv10": round((1.0 / max(frac, 1e-6) - 1.0) * 10.0, 2) if frac > 0 else None,
        })

    return {
        "max_score": max_score,
        "n_scored": len(scored),
        "submission_names": [ev["name"] for ev in evals],
        "formulas": formula_rows[:20],
        "implied_targets": implied,
    }


def _clip_score(v: float, max_score: float) -> float:
    return float(min(max(v, 0.0), max_score))
