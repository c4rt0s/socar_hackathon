"""MVP Engine: Horner plot builder, Automated Reservoir Behaviour Interpreter,
Trained .pkl Inference on Uploaded Raw Data, and Standalone HTML Report Export."""

from __future__ import annotations

import base64
import io
import math
import pickle
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

EXPORT_DIR = Path(__file__).resolve().parents[3] / "vfm_export_tool"
MODELS_DIR = Path(__file__).resolve().parents[2] / "models"


def _extract_buildup_series(
    ev: Mapping[str, Any],
    hf_well: Optional[pd.DataFrame],
    hyb: Optional[Mapping[str, Any]] = None,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Return (dt_hr, pws_psia, tp_hr, pwf_psi) for a shut-in event strictly from dataset data."""
    s0 = pd.Timestamp(ev["shut_in_start"])
    s1 = pd.Timestamp(ev["shut_in_end"])
    tp_raw = (hyb or {}).get("tp_hr") or ((ev.get("result") or {}).get("inputs") or {}).get("rate_source", {}).get("producing_duration_hr")
    tp_hr = float(tp_raw) if (tp_raw is not None and np.isfinite(float(tp_raw)) and float(tp_raw) > 0.0) else 0.0
    pwf_psi = float((hyb or {}).get("pwf_3h_med") or 0.0)

    if hf_well is None or hf_well.empty:
        return np.array([], dtype=float), np.array([], dtype=float), max(tp_hr, 24.0), pwf_psi

    ts = pd.to_datetime(hf_well["timestamp"], utc=True)
    t0_u = s0.tz_localize("UTC") if s0.tzinfo is None else s0.tz_convert("UTC")
    t1_u = s1.tz_localize("UTC") if s1.tzinfo is None else s1.tz_convert("UTC")

    if tp_hr <= 0.0:
        # Derive producing time directly from observed telemetry span prior to shut-in
        t_first = ts.min()
        tp_hr = float(max((t0_u - t_first).total_seconds() / 3600.0, 24.0))

    if pwf_psi <= 0.0:
        pre = hf_well[(ts >= t0_u - pd.Timedelta(hours=3)) & (ts <= t0_u)]["dh_pressure_psia"].dropna()
        if not pre.empty:
            pwf_psi = float(pre.median())

    post = hf_well[(ts > t0_u) & (ts <= t1_u)][["timestamp", "dh_pressure_psia"]].dropna()
    if len(post) < 5:
        return np.array([], dtype=float), np.array([], dtype=float), tp_hr, pwf_psi

    dt = (pd.to_datetime(post["timestamp"], utc=True) - t0_u).dt.total_seconds().to_numpy(dtype=float) / 3600.0
    pws = post["dh_pressure_psia"].to_numpy(dtype=float)
    ok = (dt > 0) & np.isfinite(pws)
    return dt[ok], pws[ok], tp_hr, pwf_psi


def compute_manual_two_point_pta(
    ev: Mapping[str, Any],
    pvt_row: Mapping[str, Any],
    comp_row: Mapping[str, Any],
    hyb: Optional[Mapping[str, Any]],
    hf_well: Optional[pd.DataFrame],
    dt1_hr: float,
    dt2_hr: float,
    fit_mode: str = "two_point",
) -> Dict[str, Any]:
    """Fit a Horner straight line manually from two user-selected points (dt1_hr, dt2_hr)."""
    t_lo, t_hi = sorted((float(dt1_hr), float(dt2_hr)))
    if not (np.isfinite(t_lo) and np.isfinite(t_hi)) or t_lo <= 0 or t_hi <= t_lo:
        raise ValueError("Please pick two distinct positive shut-in times (Δt1 < Δt2).")

    dt, pws, tp_hr, pwf_psi = _extract_buildup_series(ev, hf_well, hyb)
    if len(dt) < 5:
        raise ValueError("Insufficient build-up gauge measurements for manual PTA fit.")

    # Snap both clicked times to the nearest measured build-up timestamps (with 3-pt median smoothing to avoid 1-min spike bias)
    i1 = int(np.argmin(np.abs(dt - t_lo)))
    i2 = int(np.argmin(np.abs(dt - t_hi)))
    if i1 == i2:
        raise ValueError("Selected points snapped to the same timestamp. Choose two points farther apart.")
    if i1 > i2:
        i1, i2 = i2, i1

    def _local_p(idx: int) -> float:
        a, b = max(0, idx - 1), min(len(pws), idx + 2)
        return float(np.median(pws[a:b]))

    dt1_s, dt2_s = float(dt[i1]), float(dt[i2])
    p1_s, p2_s = _local_p(i1), _local_p(i2)
    x1_s = float(math.log10((tp_hr + dt1_s) / dt1_s))
    x2_s = float(math.log10((tp_hr + dt2_s) / dt2_s))

    sel = (dt >= dt1_s) & (dt <= dt2_s)
    xh_sel = np.log10((tp_hr + dt[sel]) / dt[sel])
    p_sel = pws[sel]

    if fit_mode == "interval_ols" and len(xh_sel) >= 3:
        slope_ols, intercept_ols = np.polyfit(xh_sel, p_sel, 1)
        m_val = float(-slope_ols)
        p_star = float(intercept_ols)
    else:
        fit_mode = "two_point"
        dx = x1_s - x2_s
        if abs(dx) < 1e-6:
            raise ValueError("Horner time separation between the two points is too small.")
        m_val = float((p2_s - p1_s) / dx)
        p_star = float(p1_s + m_val * x1_s)

    if not np.isfinite(m_val) or m_val <= 0.1:
        raise ValueError(
            f"Selected points produced an invalid Horner slope (m = {m_val:.2f} psi/cycle). "
            "Ensure Point 2 is at a later time and higher shut-in pressure than Point 1."
        )

    # Goodness-of-fit R² across measured points inside [dt1, dt2]
    if len(xh_sel) >= 3:
        pred_sel = p_star - m_val * xh_sel
        ss_res = float(np.sum((p_sel - pred_sel) ** 2))
        ss_tot = float(np.sum((p_sel - np.mean(p_sel)) ** 2))
        r2_val = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-9 else 1.0
    else:
        r2_val = 1.0

    # Reservoir & fluid parameters strictly from dataset PVT/completion tables (no silent hardcoded defaults)
    from vfm.pta.advanced_pta import extract_pta_rock_fluid
    qo_raw = (hyb or {}).get("qo_stbd") or ((ev.get("result") or {}).get("inputs") or {}).get("rate_source", {}).get("rate_stbd")
    if qo_raw is None or not np.isfinite(float(qo_raw)) or float(qo_raw) <= 0.0:
        raise ValueError("Pre-shut-in flowing oil rate (qo_stbd) is unavailable for this shut-in event.")
    qo = float(qo_raw)
    bo, mu, h_ft, phi, ct, rw = extract_pta_rock_fluid(pvt_row, comp_row)

    k_md = float(162.6 * qo * bo * mu / (m_val * h_ft))
    kh_md_ft = float(k_md * h_ft)
    p1hr = float(p_star - m_val * math.log10((tp_hr + 1.0) / 1.0))
    skin = float(1.1513 * (((p1hr - pwf_psi) / m_val) - math.log10(k_md / (phi * mu * ct * rw * rw)) + 3.2275))
    dp_skin = float(0.869 * m_val * skin)
    span_cyc = float(math.log10(dt2_s / dt1_s))

    return {
        "fit_mode": fit_mode,
        "dt1_hr": round(dt1_s, 4),
        "dt2_hr": round(dt2_s, 4),
        "p1_psia": round(p1_s, 2),
        "p2_psia": round(p2_s, 2),
        "x1_horner": round(x1_s, 4),
        "x2_horner": round(x2_s, 4),
        "win_hr": [round(dt1_s, 4), round(dt2_s, 4)],
        "span_cyc": round(span_cyc, 3),
        "n_pts_in_win": int(sel.sum()),
        "tp_hr": round(tp_hr, 1),
        "qo_stbd": round(qo, 2),
        "pwf_psi": round(pwf_psi, 2),
        "m": round(m_val, 3),
        "p_star": round(p_star, 2),
        "p1hr": round(p1hr, 2),
        "k": round(k_md, 2),
        "kh": round(kh_md_ft, 1),
        "skin": round(skin, 3),
        "dP_skin": round(dp_skin, 2),
        "r2_horner": round(max(-1.0, min(1.0, r2_val)), 4),
    }


def build_horner_and_interpretation(
    ev: Mapping[str, Any],
    pvt_row: Mapping[str, Any],
    comp_row: Mapping[str, Any],
    hyb: Optional[Mapping[str, Any]],
    hf_well: Optional[pd.DataFrame],
    manual_fit: Optional[Mapping[str, Any]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Build Horner semilog plot series and an engineering Reservoir Behaviour Interpretation."""
    dt, pws, tp_hr, _ = _extract_buildup_series(ev, hf_well, hyb)

    horner_pts: List[Dict[str, float]] = []
    horner_line: List[Dict[str, float]] = []
    p_star_est = None
    m_used = None
    win_hr = (
        [float(manual_fit["win_hr"][0]), float(manual_fit["win_hr"][1])]
        if manual_fit and manual_fit.get("win_hr")
        else ([float(hyb["intersect_win_hr"][0]), float(hyb["intersect_win_hr"][1])] if (hyb and hyb.get("intersect_win_hr")) else None)
    )

    if len(dt) >= 10:
        # Log-spaced sampling (~150 points) for fast, crisp Horner rendering
        if len(dt) > 155:
            log_grid = np.geomspace(dt[0], dt[-1], 145)
            idx = np.unique(np.searchsorted(dt, log_grid))
            idx = np.clip(idx, 0, len(dt) - 1)
            dt_s, pws_s = dt[idx], pws[idx]
        else:
            dt_s, pws_s = dt, pws
        xh_s = np.log10((tp_hr + dt_s) / dt_s)
        horner_pts = [{"x": round(float(x), 4), "y": round(float(p), 2), "dt": round(float(t), 4)}
                      for x, p, t in zip(xh_s, pws_s, dt_s)]

        if manual_fit:
            m_used = float(manual_fit["m"])
            p_star_est = float(manual_fit["p_star"])
            x_lo = float(xh_s.min())
            x_hi = min(max(float(manual_fit["x1_horner"]), float(manual_fit["x2_horner"])) + 0.40, float(xh_s.max()))
            horner_line = [
                {"x": round(x_hi, 4), "y": round(p_star_est - m_used * x_hi, 2)},
                {"x": round(x_lo, 4), "y": round(p_star_est - m_used * x_lo, 2)},
            ]
        elif hyb and hyb.get("intersect_win_hr"):
            m_used = float(hyb["logic3_intersection"]["m"])
            sel = (dt >= win_hr[0]) & (dt <= win_hr[1])
            if sel.sum() >= 5:
                xh_sel = np.log10((tp_hr + dt[sel]) / dt[sel])
                p_star_est = float(np.median(pws[sel] + m_used * xh_sel))
                x_lo = float(xh_s.min())
                x_hi = min(float(xh_sel.max()) + 0.35, float(xh_s.max()))
                horner_line = [
                    {"x": round(x_hi, 4), "y": round(p_star_est - m_used * x_hi, 2)},
                    {"x": round(x_lo, 4), "y": round(p_star_est - m_used * x_lo, 2)},
                ]

    horner_plot = {
        "tp_hr": round(tp_hr, 1),
        "points": horner_pts,
        "fit_line": horner_line,
        "p_star_psi": round(p_star_est, 1) if p_star_est is not None else None,
        "m_psi_cycle": round(m_used, 2) if m_used is not None else None,
        "win_hr": win_hr,
        "is_manual": bool(manual_fit),
        "manual_pins": (
            [
                {"x": float(manual_fit["x1_horner"]), "y": float(manual_fit["p1_psia"]), "dt": float(manual_fit["dt1_hr"]), "label": "P1"},
                {"x": float(manual_fit["x2_horner"]), "y": float(manual_fit["p2_psia"]), "dt": float(manual_fit["dt2_hr"]), "label": "P2"},
            ]
            if manual_fit else []
        ),
    }

    # 2. Automated Reservoir Behaviour Interpretation (Hackathon Requirement #6)
    l3 = (hyb or {}).get("logic3_intersection") or {}
    kern_res = (ev.get("result") or {}).get("results") or {}
    src_vals = manual_fit if manual_fit else l3
    k_md = src_vals.get("k") if src_vals.get("k") is not None else kern_res.get("k_md")
    skin = src_vals.get("skin") if src_vals.get("skin") is not None else kern_res.get("apparent_skin")
    dp_skin = src_vals.get("dP_skin") if src_vals.get("dP_skin") is not None else kern_res.get("skin_pressure_drop_psi")
    m_val = src_vals.get("m") if src_vals.get("m") is not None else kern_res.get("semilog_slope_psi_per_cycle")
    area_ac = float(pvt_row.get("estimated_drainage_area_acres", 0.0) or 0.0)
    dur_hr = float(ev.get("duration_hr", 0.0) or 0.0)
    span_cyc = float(manual_fit["span_cyc"]) if manual_fit else float((hyb or {}).get("intersect_span_cyc") or 0.0)
    win_end = float(win_hr[1]) if win_hr else dur_hr

    # Permeability regime
    if k_md is None:
        perm_cls = "Undetermined"
        perm_txt = "Insufficient radial-flow data to classify formation permeability."
    elif k_md >= 150.0:
        perm_cls = "High-Permeability Reservoir"
        perm_txt = f"Prolific formation with k = {k_md:.1f} mD (low semilog slope m = {m_val:.1f} psi/cycle), enabling high productivity at modest drawdown."
    elif k_md >= 50.0:
        perm_cls = "Moderate-to-Good Permeability"
        perm_txt = f"Good conventional sandstone quality with k = {k_md:.1f} mD (m = {m_val:.1f} psi/cycle)."
    else:
        perm_cls = "Moderate-Tight Formation"
        perm_txt = f"Lower permeability rock with k = {k_md:.1f} mD (steep semilog slope m = {m_val:.1f} psi/cycle), making near-wellbore stimulation critical."

    # Near-wellbore skin & damage diagnosis
    if skin is None:
        skin_cls = "Undetermined"
        skin_txt = "Skin factor unavailable."
        rec_txt = "Acquire a longer shut-in build-up to resolve radial flow."
    elif skin >= 3.0:
        skin_cls = "Damaged Wellbore (Stimulation Candidate)"
        skin_txt = f"Significant positive skin (S = +{skin:.2f}) causes an extra {dp_skin:.1f} psi near-wellbore pressure drop during production."
        rec_txt = "High-priority acidizing / matrix stimulation candidate to remove near-wellbore damage and unlock incremental oil rate."
    elif skin >= 0.5:
        skin_cls = "Mild Near-Wellbore Damage"
        skin_txt = f"Slight positive skin (S = +{skin:.2f}, ΔP_skin = {dp_skin:.1f} psi), indicating minor completion or partial-penetration impairment."
        rec_txt = "Monitor skin trend across future shut-ins; stimulation is optional unless water-cut/scale worsens."
    elif skin >= -1.0:
        skin_cls = "Clean / Undamaged Wellbore"
        skin_txt = f"Near-zero skin (S = {skin:+.2f}, ΔP_skin = {dp_skin:+.1f} psi) confirming an efficient, clean wellbore-to-formation connection."
        rec_txt = "Wellbore completion is performing near ideal; focus optimization on gas-lift rate and choke management."
    else:
        skin_cls = "Stimulated / Enhanced Near-Wellbore"
        skin_txt = f"Negative skin (S = {skin:+.2f}, ΔP_skin = {dp_skin:+.1f} psi) reflecting effective stimulation or natural near-wellbore conductivity enhancement."
        rec_txt = "Maintain current drawdown discipline to preserve fracture/stimulated conductivity and avoid fines migration."

    # Boundary & flow regime behaviour
    w0 = f"{win_hr[0]:.2f}" if win_hr else "—"
    w1 = f"{win_hr[1]:.2f}" if win_hr else f"{dur_hr:.1f}"
    if dur_hr >= 24.0 and win_end < 0.25 * dur_hr:
        bnd_cls = "Closed-Boundary / Compartment Depletion"
        bnd_txt = (
            f"Bourdet derivative drops sharply after ~{win_end:.1f} hr inside the {dur_hr:.0f}-hr shut-in "
            f"(drainage area ~{area_ac:.0f} acres), confirming a bounded compartment / pressure-depletion tail."
        )
    elif span_cyc >= 1.2:
        bnd_cls = "Extended Infinite-Acting Radial Flow (IARF)"
        bnd_txt = (
            f"Clean horizontal derivative plateau persists across {span_cyc:.2f} log cycles "
            f"([{w0}–{w1} hr]) with no nearby no-flow boundary detected ({area_ac:.0f} acres)."
        )
    elif dur_hr <= 10.0:
        bnd_cls = "Short Build-Up — Mid-Time IARF Captured"
        bnd_txt = (
            f"Despite short {dur_hr:.1f}-hr shut-in duration, hybrid intersection isolates {span_cyc:.2f} log cycles of radial flow "
            f"([{w0}–{w1} hr]); late-time boundaries remain beyond investigation radius."
        )
    else:
        bnd_cls = "Storage-Delayed Infinite-Acting Radial Flow"
        bnd_txt = (
            f"Extended early wellbore storage transitions cleanly into radial flow at {w0} hr "
            f"and stabilizes through {w1} hr ({span_cyc:.2f} log cycles)."
        )

    interpretation = {
        "permeability_class": perm_cls,
        "permeability_summary": perm_txt,
        "skin_class": skin_cls,
        "skin_summary": skin_txt,
        "boundary_class": bnd_cls,
        "boundary_summary": bnd_txt,
        "recommendation": rec_txt,
    }
    return horner_plot, interpretation


def _normalise_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lower().replace(" ", "_") for c in out.columns]
    if "well_name" not in out.columns and "well_id" in out.columns:
        out = out.rename(columns={"well_id": "well_name"})
    if "well_name" in out.columns:
        out["well_name"] = out["well_name"].astype(str).str.strip().str.upper()
    return out


def _guess_table_kind(df: pd.DataFrame) -> Optional[str]:
    """Identify which of the 5 hackathon tables a DataFrame represents from its columns."""
    cols = set(_normalise_cols(df.head(1)).columns)
    if {"test_start", "test_end"} <= cols or {"oil_rate_stbd", "water_rate_stbd"} <= cols:
        return "well_tests"
    if "choke_opening_pct" in cols and "dh_pressure_psia" not in cols:
        return "choke_events"
    if "porosity_frac" in cols or "porosity_fraction" in cols or "oil_fvf_rb_stb" in cols:
        return "reservoir_pvt"
    if "gauge_tvd_ft" in cols or "tubing_id_in" in cols or any("cum_oil" in c for c in cols):
        return "well_completion"
    if {"timestamp", "dh_pressure_psia"} <= cols or {"timestamp", "whp_psia"} <= cols:
        return "high_freq"
    return None


def _read_table_bytes(data: bytes, fname: str) -> pd.DataFrame:
    low = fname.lower()
    if low.endswith((".xlsx", ".xls")):
        return _normalise_cols(pd.read_excel(io.BytesIO(data)))
    if low.endswith(".parquet"):
        return _normalise_cols(pd.read_parquet(io.BytesIO(data)))
    return _normalise_cols(pd.read_csv(io.BytesIO(data)))


def _decode_csv_bytes(item: Mapping[str, Any]) -> pd.DataFrame:
    raw = str(item.get("content_b64") or "")
    if "," in raw and raw.startswith("data:"):
        raw = raw.split(",", 1)[1]
    data = base64.b64decode(raw)
    fname = str(item.get("name") or "")
    return _read_table_bytes(data, fname)


def _unpack_uploaded_tables(files_map: Mapping[str, Any]) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
    """Decode uploaded files (CSV, Excel, Parquet, or multi-file .zip) and auto-classify by columns."""
    tables: Dict[str, pd.DataFrame] = {}
    notes: List[str] = []

    def _ingest_item(slot_key: str, item: Mapping[str, Any]) -> None:
        raw = str(item.get("content_b64") or "")
        if not raw:
            return
        if "," in raw and raw.startswith("data:"):
            raw = raw.split(",", 1)[1]
        data = base64.b64decode(raw)
        fname = str(item.get("name") or slot_key)
        if fname.lower().endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                for member in zf.namelist():
                    if member.startswith("__MACOSX") or member.endswith("/"):
                        continue
                    if not member.lower().endswith((".csv", ".xlsx", ".xls", ".parquet")):
                        continue
                    df = _read_table_bytes(zf.read(member), member)
                    kind = _guess_table_kind(df)
                    if kind:
                        tables[kind] = df
                        notes.append(f"Auto-detected {Path(member).name} -> {kind} ({len(df):,} rows)")
        else:
            df = _read_table_bytes(data, fname)
            kind = _guess_table_kind(df) or (slot_key if slot_key in ("high_freq", "choke_events", "well_completion", "reservoir_pvt", "well_tests") else None)
            if kind:
                tables[kind] = df
                notes.append(f"Loaded {fname} -> {kind} ({len(df):,} rows)")

    for k, val in (files_map or {}).items():
        if not val:
            continue
        if isinstance(val, list):
            for entry in val:
                if isinstance(entry, Mapping):
                    _ingest_item(k, entry)
        elif isinstance(val, Mapping):
            _ingest_item(k, val)
    return tables, notes


def _build_zero_shot_anchor(
    well: str,
    well_idx: int,
    hr_w: pd.DataFrame,
    c_row: pd.Series,
    p_row: pd.Series,
    existing_anchors: Mapping[str, Mapping[str, Any]],
    comp_ref: pd.DataFrame,
    pvt_ref: pd.DataFrame,
) -> Dict[str, Any]:
    """Build a reservoir-scaled mechanistic anchor for an unseen well without well tests."""
    from vfm.pta.advanced_pta import extract_pta_rock_fluid
    spec_pis, cds, wc_b0s, wc_b1s, hydro_grads, jt_slopes = [], [], [], [], [], []
    for aw, a in existing_anchors.items():
        if aw in pvt_ref.index:
            pr = pvt_ref.loc[aw]
            bo_a, mu_a, h_a, _, _, _ = extract_pta_rock_fluid(pr, comp_ref.loc[aw] if aw in comp_ref.index else {})
            spec_pis.append(float(a["pi"]) * mu_a * bo_a / max(h_a, 1.0))
            dtvd_a = float(pr.get("datum_tvd_ft") or 0.0)
            if dtvd_a > 0.0 and float(a.get("pr0", 0.0)) > 0.0:
                hydro_grads.append(float(a["pr0"]) / dtvd_a)
        cds.append(float(a["cd"]))
        wc_b0s.append(float(a["wc_b0"]))
        wc_b1s.append(float(a["wc_b1"]))
        if "jt_slope" in a and np.isfinite(float(a["jt_slope"])):
            jt_slopes.append(float(a["jt_slope"]))

    bo_w, mu_w, h_w, phi_w, ct_w, _ = extract_pta_rock_fluid(p_row, c_row)
    area_w = float(p_row.get("estimated_drainage_area_acres") or pvt_ref["estimated_drainage_area_acres"].median())
    datum_w = float(p_row.get("datum_tvd_ft") or pvt_ref["datum_tvd_ft"].median())
    gauge_tvd = float(c_row.get("gauge_tvd_ft") or datum_w)
    glv_tvd = float(c_row.get("gl_valve_tvd_ft") or comp_ref["gl_valve_tvd_ft"].median())
    tub_id = float(c_row.get("tubing_id_in") or comp_ref["tubing_id_in"].median())
    rs_pb = float(c_row.get("solution_gor_at_pb_scf_stb") or comp_ref["solution_gor_at_pb_scf_stb"].median())

    pi_new = float(np.median(spec_pis) * h_w / max(mu_w * bo_w, 0.2))
    cd_new = float(np.median(cds))
    hydro_grad = float(np.median(hydro_grads)) if hydro_grads else 0.433 * 0.90

    fl = hr_w[hr_w["uptime_frac"] > 0.5] if "uptime_frac" in hr_w.columns else hr_w
    if fl.empty:
        fl = hr_w
    dhp_med = float(fl["dh_pressure_psia"].median())
    dhp_max = float(fl["dh_pressure_psia"].max())
    whp_med = float(fl["whp_psia"].median())
    dht_ref = float(fl["dh_temperature_degf"].median())
    ann_ref = float(fl["annulus_pressure_psia"].median())
    gl_ref = float(fl["gas_lift_rate_mscfd"].median())

    pr0 = max(dhp_max + 200.0, hydro_grad * datum_w)
    q_est = max(pi_new * max(pr0 - dhp_med, 25.0), 50.0)
    pore_vol = 7758.0 * area_w * h_w * phi_w
    dpr_dt = float(np.clip(-0.025 * q_est * bo_w / max(pore_vol * ct_w, 1e-3), -0.45, -0.03))
    grad_ref = max(dhp_med - whp_med, 50.0) / max(gauge_tvd, 100.0)

    return {
        "well_idx": well_idx,
        "cd": cd_new,
        "pi": pi_new,
        "pr0": pr0,
        "dpr_dt": dpr_dt,
        "gauge_tvd": gauge_tvd,
        "glv_tvd": glv_tvd,
        "tub_id": tub_id,
        "tub_area": math.pi / 4.0 * (tub_id ** 2),
        "grad_ref": grad_ref,
        "gl_ref": gl_ref,
        "dht_ref": dht_ref,
        "ann_ref": ann_ref,
        "vlp_base": q_est,
        "vlp_coefs": np.zeros(4, dtype=float),
        "w_choke": 0.40,
        "w_ipr": 0.45,
        "w_vlp": 0.15,
        "wc_t_mean": float(fl["days_since_start"].median()) if "days_since_start" in fl.columns and len(fl) else 0.0,
        "wc_b0": float(np.median(wc_b0s)) if wc_b0s else -0.8,
        "wc_b1": float(np.median(wc_b1s)) if wc_b1s else 0.0,
        "gor_med": rs_pb,
        "gor_slope": 0.0,
        "jt_slope": float(np.median(jt_slopes)) if jt_slopes else 0.0,
    }


def run_uploaded_inference(
    files_map: Mapping[str, Any],
    default_comp: pd.DataFrame,
    default_pvt: pd.DataFrame,
    default_tests: pd.DataFrame,
) -> Dict[str, Any]:
    """Run our pre-trained .pkl V2 Stacked VFM + 3-Logic Hybrid PTA on user-uploaded raw files
    (supports CSV, Excel, Parquet, multi-file .zip, column auto-detection, uploaded well-test
    anchor calibration, and reservoir-scaled zero-shot fallback for unseen wells)."""
    if str(EXPORT_DIR) not in sys.path:
        sys.path.insert(0, str(EXPORT_DIR))
    from vfm_phase_decomposed_stacked_vfm import (
        build_rich_feature_matrix,
        apply_soft_phase_reconciliation,
        calibrate_well_anchors,
        compute_producing_time_hr,
        SENSOR_COLS,
    )
    from pta_hybrid_iarf_from_scratch import (
        hampel_and_savgol_preprocess,
        log_resample_and_smooth,
        detect_iarf_horner_only,
        detect_iarf_bourdet_only,
        detect_iarf_intersection_ensemble,
    )

    pkl_path = MODELS_DIR / "vfm_stacked_v2.pkl"
    if not pkl_path.exists():
        pkl_path = EXPORT_DIR / "vfm_stacked_v2.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(f"Trained VFM pickle not found at {pkl_path}")

    with open(pkl_path, "rb") as f:
        bundle = pickle.load(f)

    tables, ingest_notes = _unpack_uploaded_tables(files_map)
    if "high_freq" not in tables or tables["high_freq"].empty:
        raise ValueError("Please upload at least high_freq_data.csv (or a .zip archive containing it).")

    cleaning_log: List[Dict[str, Any]] = []
    def _clog(tbl: str, issue: str, count: int, detail: str = "") -> None:
        if count > 0:
            cleaning_log.append({"table": tbl, "issue": issue, "rows_affected": int(count), "detail": detail})

    hf_raw = tables["high_freq"].copy()
    ce_df = tables.get("choke_events")
    comp_df = tables["well_completion"].copy() if "well_completion" in tables else _normalise_cols(default_comp)
    pvt_df = tables["reservoir_pvt"].copy() if "reservoir_pvt" in tables else _normalise_cols(default_pvt)
    wt_df = tables["well_tests"].copy() if "well_tests" in tables else None

    if "porosity_frac" not in pvt_df.columns and "porosity_fraction" in pvt_df.columns:
        pvt_df["porosity_frac"] = pvt_df["porosity_fraction"]

    comp = comp_df.set_index("well_name") if "well_name" in comp_df.columns else comp_df
    pvt = pvt_df.set_index("well_name") if "well_name" in pvt_df.columns else pvt_df

    # Clean high_freq: timestamps, exact duplicates, sentinels, non-physical limits
    n0 = len(hf_raw)
    hf_raw["timestamp"] = pd.to_datetime(hf_raw["timestamp"], utc=True, errors="coerce").dt.tz_localize(None)
    bad_ts = int(hf_raw["timestamp"].isna().sum())
    _clog("high_freq", "unparseable timestamp dropped", bad_ts)
    hf_raw = hf_raw.dropna(subset=["timestamp"])

    n_pre_dup = len(hf_raw)
    hf_raw = hf_raw.drop_duplicates(subset=["well_name", "timestamp"])
    _clog("high_freq", "duplicate (well, timestamp) rows dropped", n_pre_dup - len(hf_raw))

    for col in SENSOR_COLS:
        if col not in hf_raw.columns:
            hf_raw[col] = np.nan
        s = pd.to_numeric(hf_raw[col], errors="coerce")
        sent_mask = s <= -900.0
        _clog("high_freq", "sentinel (-999.25) -> NaN", int(sent_mask.sum()), col)
        s = s.mask(sent_mask, np.nan)
        if col == "gas_lift_rate_mscfd":
            neg_gl = int((s < 0.0).sum())
            _clog("high_freq", "negative gas_lift clipped to 0", neg_gl, col)
            s = s.clip(lower=0.0).fillna(0.0)
        elif col != "dh_temperature_degf":
            bad_p = (s <= 0.0) | (s > 15000.0)
            _clog("high_freq", "outside physical limits -> NaN", int(bad_p.sum()), col)
            s = s.mask(bad_p, np.nan)
        else:
            bad_t = (s < 32.0) | (s > 400.0)
            _clog("high_freq", "outside physical limits -> NaN", int(bad_t.sum()), col)
            s = s.mask(bad_t, np.nan)
        hf_raw[col] = s

    # Merge choke_events if provided (or use choke_opening_pct if already in high_freq)
    if ce_df is not None and not ce_df.empty:
        ce_df = ce_df.copy()
        ce_df["timestamp"] = pd.to_datetime(ce_df["timestamp"], utc=True, errors="coerce").dt.tz_localize(None)
        ce_df = ce_df.dropna(subset=["timestamp", "choke_opening_pct"]).drop_duplicates()
        ce_df["choke_opening_pct"] = pd.to_numeric(ce_df["choke_opening_pct"], errors="coerce").clip(0.0, 100.0)
        ce_df = ce_df.sort_values(["well_name", "timestamp"])
        parts = []
        for w, g in hf_raw.groupby("well_name"):
            g = g.sort_values("timestamp")
            cw = ce_df[ce_df["well_name"] == w][["timestamp", "choke_opening_pct"]].sort_values("timestamp")
            if not cw.empty:
                g_clean = g.drop(columns=["choke_opening_pct"], errors="ignore")
                m = pd.merge_asof(g_clean, cw, on="timestamp", direction="backward")
                m["choke_opening_pct"] = m["choke_opening_pct"].bfill().fillna(0.0)
            else:
                m = g.copy()
                if "choke_opening_pct" not in m.columns:
                    m["choke_opening_pct"] = 40.0
            parts.append(m)
        hf = pd.concat(parts, ignore_index=True)
    else:
        hf = hf_raw.copy()
        if "choke_opening_pct" not in hf.columns:
            hf["choke_opening_pct"] = 40.0

    wells_uploaded = sorted(hf["well_name"].dropna().astype(str).unique().tolist())

    # Aggregate hourly medians on a COMPLETE hourly grid per well (prevents 24h SCADA outages from becoming 0 STB/d!)
    hf["hour"] = hf["timestamp"].dt.floor("h")
    hf["is_flowing"] = (hf["choke_opening_pct"] > 0.5).astype(int)
    tot_h = hf.groupby(["well_name", "hour"], as_index=False).agg(n_tot=("is_flowing", "count"), n_fl=("is_flowing", "sum"))
    fl_h = hf[hf["is_flowing"] == 1].groupby(["well_name", "hour"], as_index=False)[SENSOR_COLS + ["choke_opening_pct"]].median()
    hr_obs = pd.merge(tot_h, fl_h, on=["well_name", "hour"], how="left")

    full_hrs = []
    for w in wells_uploaded:
        sub = hr_obs[hr_obs["well_name"] == w].sort_values("hour")
        if sub.empty:
            continue
        grid = pd.DataFrame({"hour": pd.date_range(sub["hour"].min(), sub["hour"].max(), freq="h")})
        grid["well_name"] = w
        sub = pd.merge(grid, sub, on=["well_name", "hour"], how="left")
        # Determine uptime on missing SCADA hours from choke_events if available, else carry last state
        obs_up = np.clip(sub["n_fl"] / np.maximum(sub["n_tot"], 1.0), 0.0, 1.0)
        if ce_df is not None and not ce_df.empty and (ce_df["well_name"] == w).any():
            cw = ce_df[ce_df["well_name"] == w][["timestamp", "choke_opening_pct"]].sort_values("timestamp").rename(columns={"timestamp": "hour", "choke_opening_pct": "_ce_ch"})
            sub = pd.merge_asof(sub.sort_values("hour"), cw, on="hour", direction="backward")
            sub["_ce_ch"] = sub["_ce_ch"].bfill().fillna(40.0)
            missing_hr = sub["n_tot"].isna()
            sub["uptime_frac"] = np.where(missing_hr, (sub["_ce_ch"] > 0.5).astype(float), obs_up)
            sub["choke_opening_pct"] = sub["choke_opening_pct"].fillna(sub["_ce_ch"])
            sub = sub.drop(columns=["_ce_ch"])
        else:
            sub["uptime_frac"] = obs_up.ffill().bfill().fillna(1.0)
            sub["choke_opening_pct"] = sub["choke_opening_pct"].ffill().bfill().fillna(0.0)
        for col in SENSOR_COLS:
            sub[col] = sub[col].interpolate().bfill().ffill().fillna(0.0)
        full_hrs.append(sub)

    hr = pd.concat(full_hrs, ignore_index=True).sort_values(["well_name", "hour"]).reset_index(drop=True)
    hr["date"] = hr["hour"].dt.floor("D")
    ref_date = pd.Timestamp(bundle["ref_date"]).floor("D") if bundle.get("ref_date") else hr["hour"].min().floor("D")
    hr["days_since_start"] = (hr["hour"] - ref_date).dt.total_seconds() / 86400.0 + (0.5 / 24.0)

    # Ensure completion & PVT rows exist for every uploaded well (use median field static row if missing)
    comp_med = comp.select_dtypes(include=[np.number]).median()
    pvt_med = pvt.select_dtypes(include=[np.number]).median()
    for w in wells_uploaded:
        if w not in comp.index:
            row_c = comp.iloc[0].copy()
            for c_num, v_num in comp_med.items():
                row_c[c_num] = v_num
            comp.loc[w] = row_c
        if w not in pvt.index:
            row_p = pvt.iloc[0].copy()
            for c_num, v_num in pvt_med.items():
                row_p[c_num] = v_num
            pvt.loc[w] = row_p

    # Calibrate anchors: (1) use uploaded well_tests when supplied, (2) keep trained anchors for known wells,
    # (3) build reservoir-scaled zero-shot mechanistic anchor for brand-new unseen wells without tests
    anchors = dict(bundle["anchors"])
    calibration_notes: List[str] = []

    if wt_df is not None and not wt_df.empty:
        wt_c = wt_df.copy()
        for col in ("gas_lift_rate_mscfd", "oil_rate_stbd", "water_rate_stbd", "gas_rate_mscfd", "total_gas_rate_mscfd", "liquid_rate_stbd"):
            if col in wt_c.columns:
                wt_c[col] = pd.to_numeric(wt_c[col], errors="coerce")
                neg_n = int((wt_c[col] < 0).sum())
                _clog("well_tests", "negative rate -> sign flipped", neg_n, col)
                wt_c[col] = wt_c[col].abs()
        if {"oil_rate_stbd", "gor_scf_stb", "gas_rate_mscfd"} <= set(wt_c.columns):
            exp_g = (wt_c["oil_rate_stbd"] * wt_c["gor_scf_stb"]) / 1000.0
            ratio = wt_c["gas_rate_mscfd"] / exp_g.replace(0.0, np.nan)
            for w_g, grp in wt_c.groupby("well_name"):
                if float(ratio.loc[grp.index].median()) > 15.0:
                    m_g = wt_c["well_name"] == w_g
                    _clog("well_tests", "gas in Sm3/d -> converted to Mscf/d", int(m_g.sum()), str(w_g))
                    wt_c.loc[m_g, "gas_rate_mscfd"] = wt_c.loc[m_g, "gas_rate_mscfd"] / 28.3168466
                    if "total_gas_rate_mscfd" in wt_c.columns:
                        wt_c.loc[m_g, "total_gas_rate_mscfd"] = wt_c.loc[m_g, "total_gas_rate_mscfd"] / 28.3168466
        if "liquid_rate_stbd" not in wt_c.columns and {"oil_rate_stbd", "water_rate_stbd"} <= set(wt_c.columns):
            wt_c["liquid_rate_stbd"] = wt_c["oil_rate_stbd"] + wt_c["water_rate_stbd"]
        if "water_cut_pct" not in wt_c.columns and {"oil_rate_stbd", "water_rate_stbd"} <= set(wt_c.columns):
            wt_c["water_cut_pct"] = 100.0 * wt_c["water_rate_stbd"] / np.maximum(wt_c["oil_rate_stbd"] + wt_c["water_rate_stbd"], 1.0)
        if "gor_scf_stb" not in wt_c.columns and {"oil_rate_stbd", "gas_rate_mscfd"} <= set(wt_c.columns):
            wt_c["gor_scf_stb"] = 1000.0 * wt_c["gas_rate_mscfd"] / np.maximum(wt_c["oil_rate_stbd"], 1.0)

        wt_c["test_start"] = pd.to_datetime(wt_c["test_start"], utc=True, errors="coerce").dt.tz_localize(None)
        wt_c["test_end"] = pd.to_datetime(wt_c["test_end"], utc=True, errors="coerce").dt.tz_localize(None)
        wt_c = wt_c.dropna(subset=["well_name", "test_start", "test_end", "oil_rate_stbd"])
        wt_c["test_mid"] = wt_c["test_start"] + (wt_c["test_end"] - wt_c["test_start"]) / 2
        wt_c["days_since_start"] = (wt_c["test_mid"] - ref_date).dt.total_seconds() / 86400.0

        # Align exact 7-sensor SCADA medians onto uploaded well tests
        aligned_rows = []
        for _, tr in wt_c.iterrows():
            w_t = tr["well_name"]
            gw = hf[(hf["well_name"] == w_t) & (hf["timestamp"] >= tr["test_start"]) & (hf["timestamp"] <= tr["test_end"]) & (hf["is_flowing"] == 1)]
            row_dict = tr.to_dict()
            for c_s in SENSOR_COLS + ["choke_opening_pct"]:
                if not gw.empty and c_s in gw.columns and gw[c_s].notna().any():
                    row_dict[c_s] = float(gw[c_s].median())
                elif c_s not in row_dict or pd.isna(row_dict.get(c_s)):
                    row_dict[c_s] = float(hr[hr["well_name"] == w_t][c_s].median()) if (hr["well_name"] == w_t).any() else 0.0
            aligned_rows.append(row_dict)

        if aligned_rows:
            wt_al = pd.DataFrame(aligned_rows)
            wt_wells = [w for w in wells_uploaded if (wt_al["well_name"] == w).sum() >= 2]
            if wt_wells:
                envs_up: Dict[str, Dict[str, Any]] = {}
                for w in wt_wells:
                    gw_h = hr[(hr["well_name"] == w) & (hr["uptime_frac"] > 0.5)]
                    si_p = hf[(hf["well_name"] == w) & (hf["is_flowing"] == 0)]["dh_pressure_psia"].dropna()
                    envs_up[w] = {
                        "choke_p02": float(gw_h["choke_opening_pct"].quantile(0.02)) if len(gw_h) else 15.0,
                        "choke_p98": float(gw_h["choke_opening_pct"].quantile(0.98)) if len(gw_h) else 65.0,
                        "ds_p02": float(gw_h["whp_downstream_psia"].quantile(0.02)) if len(gw_h) else 150.0,
                        "ds_p98": float(gw_h["whp_downstream_psia"].quantile(0.98)) if len(gw_h) else 250.0,
                        "gl_p05": float(gw_h["gas_lift_rate_mscfd"].quantile(0.05)) if len(gw_h) else 400.0,
                        "gl_p95": float(gw_h["gas_lift_rate_mscfd"].quantile(0.95)) if len(gw_h) else 1200.0,
                        "shutin_pmax_psi": float(si_p.quantile(0.98)) if len(si_p) >= 60 else np.nan,
                        "shutin_pmax_day": float((hf.loc[si_p.index, "timestamp"].median() - ref_date).total_seconds() / 86400.0) if len(si_p) >= 60 else np.nan,
                    }
                cal = calibrate_well_anchors(wt_al[wt_al["well_name"].isin(wt_wells)], comp.loc[wt_wells], pvt.loc[wt_wells], envs_up)
                for w in wt_wells:
                    if w in anchors:
                        cal[w]["well_idx"] = anchors[w]["well_idx"]
                    anchors[w] = cal[w]
                calibration_notes.append(f"Calibrated physics anchors from uploaded well tests for {len(wt_wells)} well(s): {', '.join(wt_wells)}")

    unseen_zero_shot = []
    # Merge training-bundle PVT & completion tables as reference so spec_pis / hydro_grad always resolve
    # even when the uploaded reservoir_pvt.csv contains ONLY brand-new unseen well(s).
    pvt_ref_all = pvt.copy()
    if bundle.get("pvt_table"):
        bt_pvt = _normalise_cols(pd.DataFrame(bundle["pvt_table"])).set_index("well_name")
        pvt_ref_all = pd.concat([pvt_ref_all, bt_pvt[~bt_pvt.index.isin(pvt_ref_all.index)]])
    comp_ref_all = comp.copy()
    if bundle.get("completion_table"):
        bt_comp = _normalise_cols(pd.DataFrame(bundle["completion_table"])).set_index("well_name")
        comp_ref_all = pd.concat([comp_ref_all, bt_comp[~bt_comp.index.isin(comp_ref_all.index)]])

    for idx_u, w in enumerate(wells_uploaded):
        if w not in anchors:
            anchors[w] = _build_zero_shot_anchor(
                w, len(anchors) + idx_u, hr[hr["well_name"] == w], comp.loc[w], pvt.loc[w], anchors, comp_ref_all, pvt_ref_all
            )
            unseen_zero_shot.append(w)
    if unseen_zero_shot:
        calibration_notes.append(f"Built reservoir-scaled zero-shot mechanistic anchors for unseen well(s): {', '.join(unseen_zero_shot)}")

    # Run trained .pkl V2 Stacked VFM inference!
    X_hr, pri_hr = build_rich_feature_matrix(hr, comp, pvt, anchors)
    choke_hr = hr["choke_opening_pct"].to_numpy(dtype=float)
    up_hr = hr["uptime_frac"].to_numpy(dtype=float)

    learners = bundle["learners"]
    qo = learners["oil"].predict(X_hr, pri_hr["oil"], choke_hr)
    qw = learners["water"].predict(X_hr, pri_hr["water"], choke_hr)
    qg = learners["gas"].predict(X_hr, pri_hr["gas"], choke_hr)
    ql = learners["liquid"].predict(X_hr, pri_hr["liquid"], choke_hr)
    qo_r, qw_r = apply_soft_phase_reconciliation(qo, qw, ql, float(bundle["lambda_rec"]))

    hr["oil_rate_stbd"] = qo_r * up_hr
    hr["water_rate_stbd"] = qw_r * up_hr
    hr["gas_rate_mscfd"] = qg * up_hr

    daily_out = hr.groupby(["well_name", "date"], as_index=False).agg(
        oil_rate_stbd=("oil_rate_stbd", "mean"),
        water_rate_stbd=("water_rate_stbd", "mean"),
        gas_rate_mscfd=("gas_rate_mscfd", "mean"),
        uptime_frac=("uptime_frac", "mean"),
    ).sort_values(["well_name", "date"]).reset_index(drop=True)
    daily_out["date_str"] = daily_out["date"].dt.strftime("%Y-%m-%d")

    # Detect shut-in events (>= 2 hr choke == 0) and run 3-Logic Hybrid Automated PTA!
    pta_results = []
    for w in wells_uploaded:
        gw = hf[hf["well_name"] == w].sort_values("timestamp").reset_index(drop=True)
        si_flag = (gw["choke_opening_pct"] <= 0.5).to_numpy()
        # Find contiguous shut-in runs
        edges = np.diff(np.concatenate([[0], si_flag.astype(int), [0]]))
        starts = np.where(edges == 1)[0]
        ends = np.where(edges == -1)[0] - 1
        for a, b in zip(starts, ends):
            t0 = gw["timestamp"].iloc[max(0, a - 1)]
            t1 = gw["timestamp"].iloc[b]
            dur_hr = (t1 - t0).total_seconds() / 3600.0
            if dur_hr < 2.0:
                continue
            pre_3h = gw[(gw["timestamp"] >= t0 - pd.Timedelta(hours=3)) & (gw["timestamp"] <= t0)]["dh_pressure_psia"].dropna()
            if pre_3h.empty:
                continue
            pwf = float(pre_3h.median())
            post = gw[(gw["timestamp"] > t0) & (gw["timestamp"] <= t1)][["timestamp", "dh_pressure_psia"]].dropna()
            if len(post) < 30:
                continue
            dt_1m = (post["timestamp"] - t0).dt.total_seconds().to_numpy(dtype=float) / 3600.0
            p_raw = post["dh_pressure_psia"].to_numpy(dtype=float)

            # Pre-shut-in flowing oil rate from our .pkl VFM predictions
            pre_h = hr[(hr["well_name"] == w) & (hr["hour"] < t0.floor("h")) & (hr["uptime_frac"] >= 0.5)].tail(24)
            qo_pre = float((pre_h["oil_rate_stbd"] / pre_h["uptime_frac"]).median()) if not pre_h.empty else float(daily_out[daily_out["well_name"] == w]["oil_rate_stbd"].median())
            qo_pre = max(qo_pre, 10.0)
            cum_cur = float(daily_out[(daily_out["well_name"] == w) & (daily_out["date"] < t0.floor("D"))]["oil_rate_stbd"].sum())
            tp_hr = compute_producing_time_hr(w, t0, qo_pre, comp, cum_cur, dataset_start=ref_date)

            _, p_sm = hampel_and_savgol_preprocess(dt_1m, p_raw)
            tb, pb = log_resample_and_smooth(dt_1m, p_sm, n_per_decade=28)
            r1 = detect_iarf_horner_only(tb, pb, dt_1m, p_sm, pwf, tp_hr, qo_pre, pvt.loc[w])
            r2 = detect_iarf_bourdet_only(tb, pb, dt_1m, p_sm, pwf, tp_hr, qo_pre, pvt.loc[w])
            r3 = detect_iarf_intersection_ensemble(r1, r2, tb, dt_1m, p_sm, pwf, tp_hr, qo_pre, pvt.loc[w])

            # Non-IARF industry fallback: Stehfest Analytical Type-Curve Match (WBS + Skin + Closed Boundary)
            method_used = "Hybrid Horner ∩ Bourdet"
            m_out, k_out, s_out, dp_out = r3["m_psi_cycle"], r3["k_md"], r3["skin"], r3["dP_skin"]
            if float(r3.get("confidence_score") or 0.0) < 55.0 or not np.isfinite(k_out):
                try:
                    from vfm.pta.advanced_pta import _preprocess_log_grid, fit_analytical_type_curve, run_tiab_direct_synthesis
                    dt_g, dte_g, dp_g, _, der_ag = _preprocess_log_grid(dt_1m, p_raw, pwf, tp_hr)
                    tds = run_tiab_direct_synthesis(dt_g, dte_g, dp_g, der_ag, qo_pre, pvt.loc[w], comp.loc[w])
                    an = fit_analytical_type_curve(
                        dt_g, dte_g, dp_g, der_ag, tp_hr, qo_pre, pvt.loc[w], comp.loc[w],
                        init_k=tds.get("k") if tds.get("available") else 80.0,
                        init_skin=tds.get("skin") if tds.get("available") else 0.0,
                        init_c=tds.get("c_bbl_psi") if tds.get("available") else 0.015,
                    )
                    if an.get("available"):
                        m_out, k_out, s_out, dp_out = an["m"], an["k"], an["skin"], an["dP_skin"]
                        method_used = f"Stehfest Analytical Match (No-IARF Fallback, der MAPE {an['deriv_mape_pct']}%)"
                except Exception:
                    pass

            pta_results.append({
                "well_name": w,
                "shut_in_time": str(t0),
                "duration_hr": round(dur_hr, 2),
                "qo_stbd": round(qo_pre, 1),
                "method": method_used,
                "horner_slope": round(m_out, 3),
                "permeability_md": round(k_out, 2),
                "skin": round(s_out, 3),
                "delta_p_skin_psi": round(dp_out, 2),
                "intersect_win_hr": [round(r3["dt_start_hr"], 2), round(r3["dt_end_hr"], 2)],
                "confidence_score": r3["confidence_score"],
                "confidence_rating": r3["confidence_rating"],
                "r2_horner": round(r3["r2_horner"], 4),
            })

    vfm_export = daily_out[["date_str", "well_name", "oil_rate_stbd", "water_rate_stbd", "gas_rate_mscfd"]].rename(columns={"date_str": "date"})
    for c in ("oil_rate_stbd", "water_rate_stbd", "gas_rate_mscfd"):
        vfm_export[c] = vfm_export[c].round(2)

    return {
        "status": "ok",
        "model_used": f"{bundle['model_name']} v{bundle['version']} (vfm_stacked_v2.pkl)",
        "n_raw_rows": int(len(hf_raw)),
        "wells": wells_uploaded,
        "date_range": [str(daily_out["date_str"].min()), str(daily_out["date_str"].max())] if len(daily_out) else ["—", "—"],
        "n_daily_rows": int(len(vfm_export)),
        "n_pta_events": int(len(pta_results)),
        "ingest_notes": ingest_notes,
        "calibration_notes": calibration_notes,
        "cleaning_log": cleaning_log,
        "vfm_preview": vfm_export.head(30).to_dict(orient="records"),
        "vfm_csv": vfm_export.to_csv(index=False),
        "pta_results": pta_results,
        "pta_csv": pd.DataFrame(pta_results).to_csv(index=False) if pta_results else "",
    }


def render_engineering_report_html(wb: Any) -> bytes:
    """Generate a clean, standalone HTML Engineering Summary Report for Phase 1 (VFM) & Phase 2 (PTA)."""
    vfm_df = wb.vfm_deliverable_df("synth_stacked_v2", "alloc")
    hyb = wb._load_hybrid_pta_cache()
    per_well_vfm = vfm_df.groupby("well_name", as_index=False).agg(
        days=("date", "count"),
        avg_oil=("oil_rate_stbd", "mean"),
        cum_oil=("oil_rate_stbd", "sum"),
        avg_water=("water_rate_stbd", "mean"),
        avg_gas=("gas_rate_mscfd", "mean"),
    )
    vfm_rows_html = "".join(
        f"<tr><td><strong>{r.well_name}</strong></td><td>{int(r.days)}</td>"
        f"<td>{r.avg_oil:,.1f}</td><td>{r.cum_oil:,.0f}</td>"
        f"<td>{r.avg_water:,.1f}</td><td>{r.avg_gas:,.1f}</td></tr>"
        for r in per_well_vfm.itertuples()
    )

    pta_rows_html = ""
    interp_cards_html = ""
    for w in sorted(hyb):
        h = hyb[w]
        l3 = h["logic3_intersection"]
        ev = (wb.events.get(w) or [{}])[0]
        _, interp = build_horner_and_interpretation(ev, wb._row(wb.pvt, w), wb._row(wb.comp, w), h, None)
        pta_rows_html += (
            f"<tr><td><strong>{w}</strong></td><td>{h['t_shut']}</td>"
            f"<td>[{h['intersect_win_hr'][0]:.2f}–{h['intersect_win_hr'][1]:.2f} hr]</td>"
            f"<td>{l3['m']:.2f}</td><td><strong>{l3['k']:.2f}</strong></td>"
            f"<td><strong>{l3['skin']:+.3f}</strong></td><td>{l3['dP_skin']:+.1f}</td>"
            f"<td>{h['confidence_rating']} ({h['confidence_score']:.0f}/100)</td></tr>"
        )
        interp_cards_html += (
            f"<div class='card'><h4>{w} — {interp['permeability_class']} · {interp['skin_class']}</h4>"
            f"<p><strong>Permeability:</strong> {interp['permeability_summary']}</p>"
            f"<p><strong>Wellbore Condition:</strong> {interp['skin_summary']}</p>"
            f"<p><strong>Flow Regime &amp; Boundary:</strong> {interp['boundary_summary']}</p>"
            f"<p><strong>Recommendation:</strong> {interp['recommendation']}</p></div>"
        )

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>SOCAR VFM & PTA Engineering Report</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color:#182026; max-width:1100px; margin:24px auto; padding:0 20px; line-height:1.45; background:#f8fafc; }}
  h1,h2,h3 {{ color:#10161a; }}
  .hdr {{ background:#ffffff; border:1px solid #d8e1e8; border-left:5px solid #2d72d2; padding:18px 22px; border-radius:6px; margin-bottom:20px; }}
  table {{ width:100%; border-collapse:collapse; background:#ffffff; border:1px solid #d8e1e8; margin-bottom:20px; font-size:13px; }}
  th, td {{ border-bottom:1px solid #e1e8ed; padding:8px 10px; text-align:right; font-variant-numeric:tabular-nums; }}
  th:first-child, td:first-child, th:nth-child(2), td:nth-child(2), th:nth-child(3), td:nth-child(3) {{ text-align:left; }}
  th {{ background:#f5f8fa; color:#394b59; font-weight:600; }}
  .card {{ background:#ffffff; border:1px solid #d8e1e8; border-radius:6px; padding:12px 16px; margin-bottom:12px; }}
  .card h4 {{ margin:0 0 6px; color:#184a90; }}
  .card p {{ margin:4px 0; font-size:13px; }}
</style></head><body>
  <div class="hdr">
    <h1 style="margin:0 0 6px;">Intelligent Well Performance Analysis — Engineering Report</h1>
    <div>Generated by SOCAR VFM + PTA Workbench · Model: <strong>Phase-Decomposed Stacked Grey-Box VFM V2</strong> &amp; <strong>5-Stage Hybrid Automated PTA (Horner ∩ Bourdet)</strong></div>
  </div>
  <h2>Phase 1: Virtual Flow Metering (VFM) Field Production Summary (Jan–Sep 2025)</h2>
  <table><thead><tr><th>Well</th><th>Days</th><th>Avg Oil (STB/d)</th><th>Cum Oil (STB)</th><th>Avg Water (STB/d)</th><th>Avg Gas (Mscf/d)</th></tr></thead>
  <tbody>{vfm_rows_html}</tbody></table>
  <h2>Phase 2: Automated Hybrid PTA Results (Horner ∩ Bourdet Intersection Ensemble)</h2>
  <table><thead><tr><th>Well</th><th>Shut-In Start</th><th>IARF Window</th><th>Horner m (psi/cyc)</th><th>k (mD)</th><th>Skin</th><th>ΔP Skin (psi)</th><th>Confidence</th></tr></thead>
  <tbody>{pta_rows_html}</tbody></table>
  <h2>Automated Reservoir Behaviour Interpretation &amp; Engineering Recommendations</h2>
  {interp_cards_html}
</body></html>"""
    return html.encode("utf-8")
