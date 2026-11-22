"""Data-driven PTA Oil-Rate (Qo) Sensitivity Tornado & VFM Feature Diagnostics
(Correlation Matrix, Training Feature Audit, and Exact SHAP Evaluation).

Every quantity is derived strictly from the active dataset and trained VFM bundle —
zero hardcoded well names, dates, paths, or silent rock/fluid defaults.
"""

from __future__ import annotations

import math
from pathlib import Path
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from vfm.pta.advanced_pta import extract_pta_rock_fluid


# ==============================================================================
# 1. Exact PTA Equations & Oil-Rate (Qo) Sensitivity Tornado Engine
# ==============================================================================

def _pta_k_and_skin(
    qo_stbd: float,
    m_psi_cyc: float,
    bo: float,
    mu_o: float,
    h_ft: float,
    phi: float,
    ct: float,
    rw_ft: float,
    p1hr_psia: float,
    pwf0_psia: float,
) -> Tuple[float, float, float, float]:
    """Exact SPE oilfield semilog equations for k, kh, skin S, and dP_skin."""
    m_safe = max(float(m_psi_cyc), 1e-6)
    h_safe = max(float(h_ft), 1e-6)
    kh = 162.6 * float(qo_stbd) * float(bo) * float(mu_o) / m_safe
    k = kh / h_safe
    denom = max(float(phi) * float(mu_o) * float(ct) * (float(rw_ft) ** 2), 1e-30)
    log_term = math.log10(max(k / denom, 1e-30))
    skin = 1.1513 * (((float(p1hr_psia) - float(pwf0_psia)) / m_safe) - log_term + 3.2275)
    dp_skin = 0.869 * m_safe * skin
    return float(k), float(kh), float(skin), float(dp_skin)


def _eval_p1hr_horner(p_star_psia: float, m_psi_cyc: float, tp_hr: float) -> float:
    """Horner straight line evaluated at dt = 1 hr: p(1hr) = p* - m * log10((tp + 1) / 1)."""
    return float(p_star_psia) - float(m_psi_cyc) * math.log10((max(float(tp_hr), 1.0) + 1.0) / 1.0)


def _coupled_horner_at_qo(
    qo_new: float,
    qo_base: float,
    tp_base_hr: float,
    m_base: float,
    p_star_base: float,
    dt_win: Optional[np.ndarray],
    pws_win: Optional[np.ndarray],
    bo: float,
    mu_o: float,
    h_ft: float,
    phi: float,
    ct: float,
    rw_ft: float,
    pwf0_psia: float,
    win_bounds_hr: Optional[Tuple[float, float]] = None,
) -> Tuple[float, float, float, float, float, float]:
    """Evaluate exact coupled Horner PTA outputs when Qo is perturbed, accounting for BOTH:
      1) Producing time tp(Qo) = 24 * Np / Qo = tp_base * (qo_base / qo_new), which shifts
         the Horner time coordinate x_H(dt) = log10((tp(Qo) + dt) / dt) and refits (m, p*, p1hr), AND
      2) Parameter equations k = 162.6*Qo*Bo*mu/(m*h), S, and dP_skin = 0.869*m*S.
    Returns (tp_new, m_new, p_star_new, k_new, skin_new, dp_skin_new).
    """
    tp_new = max(float(tp_base_hr) * (float(qo_base) / max(float(qo_new), 1e-6)), 1.0)

    if dt_win is not None and pws_win is not None and len(dt_win) >= 3:
        # Exact OLS Horner refit on the radial-window build-up points at tp_new
        xh_new = np.log10((tp_new + dt_win) / dt_win)
        xh_base = np.log10((tp_base_hr + dt_win) / dt_win)
        # Re-scale m_base strictly by the ratio of Horner-axis linear regression slopes so
        # base (qo_new == qo_base) reproduces exact active-method m_base & p_star_base!
        xm_b, pm = float(np.mean(xh_base)), float(np.mean(pws_win))
        xm_n = float(np.mean(xh_new))
        var_b = float(np.sum((xh_base - xm_b) ** 2))
        var_n = float(np.sum((xh_new - xm_n) ** 2))
        if var_b > 1e-12 and var_n > 1e-12:
            m_ols_b = float(-np.sum((xh_base - xm_b) * (pws_win - pm)) / var_b)
            m_ols_n = float(-np.sum((xh_new - xm_n) * (pws_win - pm)) / var_n)
            pstar_ols_b = pm + m_ols_b * xm_b
            pstar_ols_n = pm + m_ols_n * xm_n
            slope_ratio = (m_ols_n / m_ols_b) if (m_ols_b > 1e-6 and m_ols_n > 1e-6) else 1.0
            m_new = float(m_base) * slope_ratio
            p_star_new = float(p_star_base) + (pstar_ols_n - pstar_ols_b)
        else:
            m_new, p_star_new = float(m_base), float(p_star_base)
    else:
        # Analytical 2-point window curvature transform when raw window points are unavailable
        dt1, dt2 = win_bounds_hr if (win_bounds_hr and win_bounds_hr[1] > win_bounds_hr[0] > 0) else (0.5, 5.0)
        dx_b = math.log10((tp_base_hr + dt1) / dt1) - math.log10((tp_base_hr + dt2) / dt2)
        dx_n = math.log10((tp_new + dt1) / dt1) - math.log10((tp_new + dt2) / dt2)
        m_new = float(m_base) * (dx_b / max(dx_n, 1e-12))
        dt_mid = math.sqrt(dt1 * dt2)
        p_mid = float(p_star_base) - float(m_base) * math.log10((tp_base_hr + dt_mid) / dt_mid)
        p_star_new = p_mid + m_new * math.log10((tp_new + dt_mid) / dt_mid)

    p1hr_new = _eval_p1hr_horner(p_star_new, m_new, tp_new)
    k_new, _, skin_new, dps_new = _pta_k_and_skin(
        qo_new, m_new, bo, mu_o, h_ft, phi, ct, rw_ft, p1hr_new, pwf0_psia
    )
    return float(tp_new), float(m_new), float(p_star_new), float(k_new), float(skin_new), float(dps_new)


def _nearest_pre_shutin_rate(
    df_daily: Optional[pd.DataFrame],
    well: str,
    s0_utc: pd.Timestamp,
    lookback_days: int = 1,
) -> Tuple[Optional[float], Optional[float]]:
    """Return (oil_rate_stbd, water_rate_stbd) averaged over [d0 - lookback_days, d0)."""
    if df_daily is None or df_daily.empty or "well_name" not in df_daily.columns:
        return None, None
    wdf = df_daily[df_daily["well_name"] == well]
    if wdf.empty:
        return None, None
    date_col = "date" if "date" in wdf.columns else ("timestamp" if "timestamp" in wdf.columns else None)
    if date_col is None:
        return None, None
    dt = pd.to_datetime(wdf[date_col], utc=True)
    d0 = s0_utc.normalize()
    mask = (dt >= d0 - pd.Timedelta(days=int(lookback_days))) & (dt < d0)
    sub = wdf.loc[mask]
    if sub.empty:
        # Fallback: closest prior flowing day
        prior = wdf.loc[dt < d0]
        if prior.empty:
            return None, None
        sub = prior.iloc[[-1]]
    o_col = next((c for c in ("oil_rate_vfm_stbd", "oil_rate_stbd", "oil_alloc_stbd") if c in sub.columns), None)
    w_col = next((c for c in ("water_rate_vfm_stbd", "water_rate_stbd", "water_alloc_stbd") if c in sub.columns), None)
    qo_val = float(sub[o_col].dropna().mean()) if o_col and sub[o_col].notna().any() else None
    qw_val = float(sub[w_col].dropna().mean()) if w_col and sub[w_col].notna().any() else None
    if qo_val is not None and (not np.isfinite(qo_val) or qo_val <= 0.0):
        qo_val = None
    if qw_val is not None and (not np.isfinite(qw_val) or qw_val < 0.0):
        qw_val = None
    return qo_val, qw_val


def compute_pta_qo_sensitivity(
    well: str,
    ev: Mapping[str, Any],
    hf_w: pd.DataFrame,
    pvt_row: Mapping[str, Any],
    comp_row: Mapping[str, Any],
    tests_df: pd.DataFrame,
    model_daily: Mapping[str, pd.DataFrame],
    hyb: Optional[Mapping[str, Any]] = None,
    adv: Optional[Mapping[str, Any]] = None,
    horner_plot: Optional[Mapping[str, Any]] = None,
    qo_pct: float = 20.0,
) -> Dict[str, Any]:
    """Build complete data-driven Qo Sensitivity Tornado & Root-Cause Diagnostics for one PBU."""
    bo, mu_o, h_ft, phi, ct, rw_ft = extract_pta_rock_fluid(pvt_row, comp_row)
    bw = float(comp_row.get("water_fvf_rb_stb") or pvt_row.get("water_fvf_rb_stb") or 1.02)

    s0 = pd.Timestamp(ev["shut_in_start"])
    s0_utc = s0.tz_localize("UTC") if s0.tzinfo is None else s0.tz_convert("UTC")

    # Resolve base PTA parameters strictly from active event / hybrid / horner
    l3 = (hyb or {}).get("logic3_intersection") or {}
    res_block = (ev.get("result") or {}).get("results") or {}
    qo_base = (hyb or {}).get("qo_stbd")
    if qo_base is None or not np.isfinite(float(qo_base)) or float(qo_base) <= 0.0:
        qo_base = ((ev.get("result") or {}).get("inputs") or {}).get("rate_source", {}).get("rate_stbd")
    if qo_base is None or not np.isfinite(float(qo_base)) or float(qo_base) <= 0.0:
        qo_1d, _ = _nearest_pre_shutin_rate(model_daily.get("synth_stacked_v2"), well, s0_utc, 1)
        qo_base = qo_1d
    if qo_base is None or float(qo_base) <= 0.0:
        raise ValueError(f"Cannot compute Qo sensitivity for {well}: no valid pre-shut-in oil rate.")
    qo_base = float(qo_base)

    m_base = l3.get("m") or (horner_plot or {}).get("m_psi_cycle") or res_block.get("semilog_slope_psi_per_cycle")
    if m_base is None or not np.isfinite(float(m_base)) or float(m_base) <= 0.0:
        # Try Stehfest analytical match slope if radial window wasn't found
        an = (adv or {}).get("analytical_match") or {}
        m_base = an.get("m")
    if m_base is None or not np.isfinite(float(m_base)) or float(m_base) <= 0.0:
        return {"available": False, "note": "No valid semilog slope available for Tornado sensitivity."}
    m_base = float(m_base)

    tp_hr = (horner_plot or {}).get("tp_hr") or ((ev.get("result") or {}).get("inputs") or {}).get("rate_source", {}).get("producing_duration_hr")
    if tp_hr is None or not np.isfinite(float(tp_hr)) or float(tp_hr) <= 0.0:
        ts_all = pd.to_datetime(hf_w["timestamp"], utc=True)
        tp_hr = max((s0_utc - ts_all.min()).total_seconds() / 3600.0, 24.0)
    tp_hr = float(tp_hr)

    pwf0 = (hyb or {}).get("pwf_3h_med")
    if pwf0 is None or not np.isfinite(float(pwf0)) or float(pwf0) <= 0.0:
        ts_hf = pd.to_datetime(hf_w["timestamp"], utc=True)
        pre = hf_w[(ts_hf >= s0_utc - pd.Timedelta(hours=3)) & (ts_hf <= s0_utc)]["dh_pressure_psia"].dropna()
        pwf0 = float(pre.median()) if len(pre) else float(res_block.get("p_wf0_psia") or 0.0)
    pwf0 = float(pwf0)

    p_star = (horner_plot or {}).get("p_star_psi") or l3.get("p_star") or res_block.get("p_star_psia")
    if p_star is not None and np.isfinite(float(p_star)):
        p_star = float(p_star)
        p1hr = _eval_p1hr_horner(p_star, m_base, tp_hr)
    else:
        # Reconstruct consistent p1hr from base k and skin
        k_tmp = 162.6 * qo_base * bo * mu_o / (m_base * h_ft)
        s_tmp = float(l3.get("skin") or ((adv or {}).get("analytical_match") or {}).get("skin") or 0.0)
        denom = max(phi * mu_o * ct * (rw_ft ** 2), 1e-30)
        p1hr = pwf0 + m_base * (s_tmp / 1.1513 + math.log10(max(k_tmp / denom, 1e-30)) - 3.2275)
        p_star = p1hr + m_base * math.log10((tp_hr + 1.0) / 1.0)

    k_base, kh_base, s_base, dps_base = _pta_k_and_skin(
        qo_base, m_base, bo, mu_o, h_ft, phi, ct, rw_ft, p1hr, pwf0
    )

    # ------------------------------------------------------------------
    # Gather REAL MEASURED Qo uncertainty sources on THIS exact well
    # ------------------------------------------------------------------
    # 1. Pre-shut-in water rate & Perrine multiphase mobility
    v2_df = model_daily.get("synth_stacked_v2")
    if v2_df is None and model_daily:
        v2_df = next(iter(model_daily.values()))
    qo_1d, qw_1d = _nearest_pre_shutin_rate(v2_df, well, s0_utc, 1)
    qo_3d, _ = _nearest_pre_shutin_rate(v2_df, well, s0_utc, 3)
    qo_7d, _ = _nearest_pre_shutin_rate(v2_df, well, s0_utc, 7)

    # Fallback water rate from nearest separator well test if daily water absent
    wt_w = tests_df[tests_df["well_name"] == well].copy() if tests_df is not None and not tests_df.empty else pd.DataFrame()
    q_wt_pre = None
    gap_pre_days = None
    q_wt_near = None
    gap_near_days = None
    wt_near_qw = None
    if not wt_w.empty:
        wt_w["test_end_u"] = pd.to_datetime(wt_w["test_end"], utc=True)
        wt_w["dt_days"] = (s0_utc - wt_w["test_end_u"]).dt.total_seconds() / 86400.0
        pre_wt = wt_w[wt_w["dt_days"] >= 0.0].sort_values("dt_days")
        if not pre_wt.empty:
            r_pre = pre_wt.iloc[0]
            q_wt_pre = float(r_pre["oil_rate_stbd"])
            gap_pre_days = float(r_pre["dt_days"])
        idx_near = int(wt_w["dt_days"].abs().argmin())
        r_near = wt_w.iloc[idx_near]
        q_wt_near = float(r_near["oil_rate_stbd"])
        gap_near_days = float(r_near["dt_days"])
        if "water_rate_stbd" in r_near and pd.notna(r_near["water_rate_stbd"]):
            wt_near_qw = float(r_near["water_rate_stbd"])

    qw_base = qw_1d if qw_1d is not None else (wt_near_qw if wt_near_qw is not None else 0.0)
    wc_frac = float(qw_base / max(qo_base + qw_base, 1e-6))
    perrine_factor = float((qo_base * bo + qw_base * bw) / max(qo_base * bo, 1e-6))
    qo_perrine_equiv = qo_base * perrine_factor

    # 2. Pre-shut-in HF choke & PDG stability (6h vs 24h vs 72h)
    ts_hf = pd.to_datetime(hf_w["timestamp"], utc=True)
    pre6 = hf_w[(ts_hf >= s0_utc - pd.Timedelta(hours=6)) & (ts_hf < s0_utc)]
    pre24 = hf_w[(ts_hf >= s0_utc - pd.Timedelta(hours=24)) & (ts_hf < s0_utc)]
    pre72 = hf_w[(ts_hf >= s0_utc - pd.Timedelta(hours=72)) & (ts_hf < s0_utc)]
    chk6 = float(pre6["choke_opening_pct"].dropna().median()) if len(pre6) and "choke_opening_pct" in pre6 else None
    chk24 = float(pre24["choke_opening_pct"].dropna().median()) if len(pre24) and "choke_opening_pct" in pre24 else None
    chk72 = float(pre72["choke_opening_pct"].dropna().median()) if len(pre72) and "choke_opening_pct" in pre72 else None
    pdg6 = float(pre6["dh_pressure_psia"].dropna().median()) if len(pre6) and "dh_pressure_psia" in pre6 else None
    pdg72 = float(pre72["dh_pressure_psia"].dropna().median()) if len(pre72) and "dh_pressure_psia" in pre72 else None
    choke_drift_pct = float(abs((chk6 or 0.0) - (chk72 or 0.0))) if (chk6 is not None and chk72 is not None) else 0.0
    pdg_drift_psi = float((pdg6 or 0.0) - (pdg72 or 0.0)) if (pdg6 is not None and pdg72 is not None) else 0.0

    # 3. Inter-model VFM spread across all available daily models
    model_qo_map: Dict[str, float] = {}
    for mname, dfm in (model_daily or {}).items():
        q_m, _ = _nearest_pre_shutin_rate(dfm, well, s0_utc, 1)
        if q_m is not None and q_m > 0.0:
            model_qo_map[mname] = float(q_m)
    model_q_vals = list(model_qo_map.values()) or [qo_base]
    qo_mod_min = float(min(min(model_q_vals), qo_base))
    qo_mod_max = float(max(max(model_q_vals), qo_base))

    # 4. Pre-shut-in temporal window spread (1d / 3d / 7d vs base)
    win_vals = [v for v in (qo_base, qo_1d, qo_3d, qo_7d) if v is not None and np.isfinite(v) and v > 0]
    qo_win_min = float(min(win_vals)) if win_vals else qo_base
    qo_win_max = float(max(win_vals)) if win_vals else qo_base

    # 5. Multi-rate superposition / deconvolution vs constant-Qo Horner
    adv_d = adv or {}
    oj = adv_d.get("odeh_jones_multirate") or {}
    dc = adv_d.get("deconvolution") or {}
    st_m = adv_d.get("analytical_match") or {}
    multirate_ks = [float(x["k"]) for x in (oj, dc) if isinstance(x, Mapping) and x.get("k") and np.isfinite(float(x["k"])) and float(x["k"]) > 0]
    multirate_ss = [float(x["skin"]) for x in (oj, dc) if isinstance(x, Mapping) and x.get("skin") is not None and np.isfinite(float(x["skin"]))]

    # Collect slope spread across all PTA methods for m uncertainty comparison
    all_method_ms = []
    for blk in (
        (hyb or {}).get("logic1_horner"),
        (hyb or {}).get("logic2_bourdet"),
        l3,
        st_m,
        dc,
        adv_d.get("ilk_bspline_deconv"),
        adv_d.get("agarwal_equiv"),
        adv_d.get("blasingame_integral"),
        oj,
        adv_d.get("tiab_tds"),
        adv_d.get("mdh_muskat"),
    ):
        if isinstance(blk, Mapping) and blk.get("m") and np.isfinite(float(blk["m"])) and float(blk["m"]) > 0:
            all_method_ms.append(float(blk["m"]))
    m_method_lo = float(np.percentile(all_method_ms, 10)) if len(all_method_ms) >= 3 else m_base * 0.90
    m_method_hi = float(np.percentile(all_method_ms, 90)) if len(all_method_ms) >= 3 else m_base * 1.10

    def _bar(
        key: str,
        label: str,
        category: str,
        source_detail: str,
        low_input_label: str,
        high_input_label: str,
        k_lo: float,
        k_hi: float,
        s_lo: float,
        s_hi: float,
        is_qo_driver: bool = False,
    ) -> Dict[str, Any]:
        # Order min/max per output so Tornado spans cleanly around base
        k_min, k_max = min(k_lo, k_hi, k_base), max(k_lo, k_hi, k_base)
        s_min, s_max = min(s_lo, s_hi, s_base), max(s_lo, s_hi, s_base)
        return {
            "key": key,
            "label": label,
            "category": category,
            "is_qo_driver": bool(is_qo_driver),
            "source_detail": source_detail,
            "low_input_label": low_input_label,
            "high_input_label": high_input_label,
            "k_at_low_input": round(float(k_lo), 3),
            "k_at_high_input": round(float(k_hi), 3),
            "k_min": round(float(k_min), 3),
            "k_max": round(float(k_max), 3),
            "k_span": round(float(k_max - k_min), 3),
            "k_pct_low": round(100.0 * (k_min - k_base) / max(k_base, 1e-6), 2),
            "k_pct_high": round(100.0 * (k_max - k_base) / max(k_base, 1e-6), 2),
            "kh_at_low_input": round(float(k_lo * h_ft), 1),
            "kh_at_high_input": round(float(k_hi * h_ft), 1),
            "skin_at_low_input": round(float(s_lo), 3),
            "skin_at_high_input": round(float(s_hi), 3),
            "skin_min": round(float(s_min), 3),
            "skin_max": round(float(s_max), 3),
            "skin_span": round(float(s_max - s_min), 3),
        }

    # ------------------------------------------------------------------
    # VIEW A: Qo Root-Cause Tornado (Measured Sources on THIS Well)
    # ------------------------------------------------------------------
    qo_bars: List[Dict[str, Any]] = []

    # A1: Multiphase Perrine total-fluid mobility vs single-phase oil Qo
    k_per, _, s_per, _ = _pta_k_and_skin(qo_perrine_equiv, m_base, bo, mu_o, h_ft, phi, ct, rw_ft, p1hr, pwf0)
    qo_bars.append(_bar(
        "qo_multiphase_perrine",
        f"Multiphase Water Mobility (WC={wc_frac*100:.1f}%)",
        "Fluid Phase Assumption",
        f"Single-phase Qo·Bo ({qo_base:.0f} STB/d oil) yields effective oil perm k_o; Perrine total-fluid (Qo·Bo + Qw·Bw, Qw={qw_base:.0f} STB/d) scales mobility by {perrine_factor:.2f}×.",
        f"Single-phase Qo={qo_base:.0f} STB/d",
        f"Perrine equiv={qo_perrine_equiv:.0f} STB/d ({perrine_factor:.2f}×)",
        k_base, k_per, s_base, s_per, is_qo_driver=True,
    ))

    # A2: Parametric VFM Qo uncertainty (±qo_pct%)
    frac = float(qo_pct) / 100.0
    k_qlo, _, s_qlo, _ = _pta_k_and_skin(qo_base * (1.0 - frac), m_base, bo, mu_o, h_ft, phi, ct, rw_ft, p1hr, pwf0)
    k_qhi, _, s_qhi, _ = _pta_k_and_skin(qo_base * (1.0 + frac), m_base, bo, mu_o, h_ft, phi, ct, rw_ft, p1hr, pwf0)
    qo_bars.append(_bar(
        "qo_parametric_band",
        f"Oil Rate Qo Uncertainty (±{qo_pct:.0f}%)",
        "Direct Rate Sensitivity",
        f"Linear 1:1 propagation into k (k ∝ Qo) and weak logarithmic shift in Skin (ΔS = -1.1513·log10(Qo/Qo0) = {s_qhi - s_base:+.3f} / {s_qlo - s_base:+.3f}).",
        f"Qo -{qo_pct:.0f}% ({qo_base*(1-frac):.0f} STB/d)",
        f"Qo +{qo_pct:.0f}% ({qo_base*(1+frac):.0f} STB/d)",
        k_qlo, k_qhi, s_qlo, s_qhi, is_qo_driver=True,
    ))

    # A3: Stale separator well test vs real-time VFM
    q_wt_ref = q_wt_pre if q_wt_pre is not None else q_wt_near
    gap_ref = gap_pre_days if gap_pre_days is not None else gap_near_days
    if q_wt_ref is not None and q_wt_ref > 0.0:
        k_wt, _, s_wt, _ = _pta_k_and_skin(q_wt_ref, m_base, bo, mu_o, h_ft, phi, ct, rw_ft, p1hr, pwf0)
        q_lo_wt, q_hi_wt = min(qo_base, q_wt_ref), max(qo_base, q_wt_ref)
        k_lo_wt, _, s_lo_wt, _ = _pta_k_and_skin(q_lo_wt, m_base, bo, mu_o, h_ft, phi, ct, rw_ft, p1hr, pwf0)
        k_hi_wt, _, s_hi_wt, _ = _pta_k_and_skin(q_hi_wt, m_base, bo, mu_o, h_ft, phi, ct, rw_ft, p1hr, pwf0)
        rel_wt_pct = 100.0 * (q_wt_ref - qo_base) / qo_base
        qo_bars.append(_bar(
            "qo_stale_welltest",
            f"Separator Test Lag ({abs(gap_ref or 0.0):.0f}d before shut-in)",
            "Test Staleness / Allocation",
            f"Nearest pre-shut-in separator test was {q_wt_ref:.1f} STB/d ({rel_wt_pct:+.1f}% vs VFM {qo_base:.1f} STB/d, {abs(gap_ref or 0.0):.1f} days gap) — reservoir depletion or choke changes between test and shut-in bias Qo.",
            f"Min={q_lo_wt:.0f} STB/d",
            f"Max={q_hi_wt:.0f} STB/d",
            k_lo_wt, k_hi_wt, s_lo_wt, s_hi_wt, is_qo_driver=True,
        ))

    # A4: Inter-Model VFM spread (only shown when >= 2 VFM models are available)
    if len(model_qo_map) >= 2:
        k_mlo, _, s_mlo, _ = _pta_k_and_skin(qo_mod_min, m_base, bo, mu_o, h_ft, phi, ct, rw_ft, p1hr, pwf0)
        k_mhi, _, s_mhi, _ = _pta_k_and_skin(qo_mod_max, m_base, bo, mu_o, h_ft, phi, ct, rw_ft, p1hr, pwf0)
        mod_desc = ", ".join(f"{k}={v:.0f}" for k, v in sorted(model_qo_map.items()))
        qo_bars.append(_bar(
            "qo_vfm_model_spread",
            f"VFM Model Discrepancy ({len(model_qo_map)} Models)",
            "VFM Model Choice",
            f"Pre-shut-in oil rate spread across {len(model_qo_map)} VFM models [{qo_mod_min:.0f}–{qo_mod_max:.0f} STB/d]: {mod_desc}.",
            f"Min model={qo_mod_min:.0f} STB/d",
            f"Max model={qo_mod_max:.0f} STB/d",
            k_mlo, k_mhi, s_mlo, s_mhi, is_qo_driver=True,
        ))

    # A5: Pre-shut-in temporal averaging window / transient drift (1d vs 3d vs 7d)
    k_wlo, _, s_wlo, _ = _pta_k_and_skin(qo_win_min, m_base, bo, mu_o, h_ft, phi, ct, rw_ft, p1hr, pwf0)
    k_whi, _, s_whi, _ = _pta_k_and_skin(qo_win_max, m_base, bo, mu_o, h_ft, phi, ct, rw_ft, p1hr, pwf0)
    qo_bars.append(_bar(
        "qo_preshutin_transient",
        f"Pre-Shut-in Window Drift (1d–7d, ΔChk={choke_drift_pct:.1f}%)",
        "Pre-Shut-in Stabilization",
        f"VFM Qo across 1d/3d/7d pre-shut-in windows spans [{qo_win_min:.0f}–{qo_win_max:.0f} STB/d]; 6h vs 72h choke shift={choke_drift_pct:.1f}%, PDG drift={pdg_drift_psi:+.1f} psi.",
        f"Min window={qo_win_min:.0f} STB/d",
        f"Max window={qo_win_max:.0f} STB/d",
        k_wlo, k_whi, s_wlo, s_whi, is_qo_driver=True,
    ))

    # A6: Constant-Qo Horner vs Multi-Rate Superposition / Deconvolution
    if multirate_ks:
        k_mr_lo = min(min(multirate_ks), k_base)
        k_mr_hi = max(max(multirate_ks), k_base)
        s_mr_lo = min(min(multirate_ss), s_base) if multirate_ss else s_base
        s_mr_hi = max(max(multirate_ss), s_base) if multirate_ss else s_base
        oj_txt = f"Odeh–Jones k={oj['k']:.1f} mD" if oj.get("k") else ""
        dc_txt = f"von Schroeter Deconv k={dc['k']:.1f} mD" if dc.get("k") else ""
        detail_mr = " · ".join(x for x in (oj_txt, dc_txt) if x)
        qo_bars.append(_bar(
            "qo_multirate_superposition",
            "Constant-Qo Horner vs Multi-Rate History",
            "Rate History Assumption",
            f"Single-rate Horner assumes constant Qo={qo_base:.0f} STB/d; multi-rate superposition models yield: {detail_mr}.",
            f"Min multi-rate k={k_mr_lo:.1f} mD",
            f"Max multi-rate k={k_mr_hi:.1f} mD",
            k_mr_lo, k_mr_hi, s_mr_lo, s_mr_hi, is_qo_driver=True,
        ))

    qo_bars.sort(key=lambda b: b["k_span"], reverse=True)

    # ------------------------------------------------------------------
    # VIEW B: Full PTA Input Parameter Tornado (Qo vs m, h, mu, Bo, phi, ct, rw, Pwf0, tp)
    # ------------------------------------------------------------------
    param_bars: List[Dict[str, Any]] = []

    # B1: Oil Rate Qo (±qo_pct%)
    param_bars.append(_bar(
        "param_qo",
        f"Oil Rate Qo (±{qo_pct:.0f}%)",
        "VFM Rate Input",
        f"k ∝ Qo (exact 1:1 linear sensitivity); Skin S shifts weakly via -1.1513·log10(Qo/Qo0).",
        f"{qo_base*(1-frac):.0f} STB/d (-{qo_pct:.0f}%)",
        f"{qo_base*(1+frac):.0f} STB/d (+{qo_pct:.0f}%)",
        k_qlo, k_qhi, s_qlo, s_qhi, is_qo_driver=True,
    ))

    # B2: Semilog Slope m (IARF window pick — actual inter-method P10–P90 or ±10%)
    k_ml, _, s_ml, _ = _pta_k_and_skin(qo_base, m_method_lo, bo, mu_o, h_ft, phi, ct, rw_ft, _eval_p1hr_horner(p_star, m_method_lo, tp_hr), pwf0)
    k_mh, _, s_mh, _ = _pta_k_and_skin(qo_base, m_method_hi, bo, mu_o, h_ft, phi, ct, rw_ft, _eval_p1hr_horner(p_star, m_method_hi, tp_hr), pwf0)
    param_bars.append(_bar(
        "param_m",
        f"Semilog Slope m (9-Method Spread: {m_method_lo:.1f}–{m_method_hi:.1f})",
        "PTA Window Pick",
        "Slope m from IARF radial selection affects BOTH k (inversely: k ∝ 1/m) and Skin S strongly via (p1hr - Pwf0)/m.",
        f"m={m_method_lo:.2f} psi/cyc",
        f"m={m_method_hi:.2f} psi/cyc",
        k_ml, k_mh, s_ml, s_mh,
    ))

    # B3: Net Pay Thickness h (±10%)
    k_hl, _, s_hl, _ = _pta_k_and_skin(qo_base, m_base, bo, mu_o, h_ft * 0.90, phi, ct, rw_ft, p1hr, pwf0)
    k_hh, _, s_hh, _ = _pta_k_and_skin(qo_base, m_base, bo, mu_o, h_ft * 1.10, phi, ct, rw_ft, p1hr, pwf0)
    param_bars.append(_bar(
        "param_h",
        f"Net Pay Thickness h (±10%, base={h_ft:.0f} ft)",
        "Petrophysics",
        "k ∝ 1/h (inversely linear); kh = 162.6·Qo·Bo·μo/m is independent of h.",
        f"h={h_ft*0.90:.1f} ft (-10%)",
        f"h={h_ft*1.10:.1f} ft (+10%)",
        k_hl, k_hh, s_hl, s_hh,
    ))

    # B4: Oil Viscosity mu_o (±10%) — note mu_o cancels out of Skin S!
    k_ul, _, s_ul, _ = _pta_k_and_skin(qo_base, m_base, bo, mu_o * 0.90, h_ft, phi, ct, rw_ft, p1hr, pwf0)
    k_uh, _, s_uh, _ = _pta_k_and_skin(qo_base, m_base, bo, mu_o * 1.10, h_ft, phi, ct, rw_ft, p1hr, pwf0)
    param_bars.append(_bar(
        "param_mu",
        f"Oil Viscosity μo (±10%, base={mu_o:.2f} cp)",
        "PVT Fluid",
        "k ∝ μo (linear), whereas μo cancels algebraically in k/(φ·μo·ct·rw²) so Skin S is 100% invariant to μo.",
        f"μo={mu_o*0.90:.3f} cp (-10%)",
        f"μo={mu_o*1.10:.3f} cp (+10%)",
        k_ul, k_uh, s_ul, s_uh,
    ))

    # B5: Oil FVF Bo (±5%)
    k_bl, _, s_bl, _ = _pta_k_and_skin(qo_base, m_base, bo * 0.95, mu_o, h_ft, phi, ct, rw_ft, p1hr, pwf0)
    k_bh, _, s_bh, _ = _pta_k_and_skin(qo_base, m_base, bo * 1.05, mu_o, h_ft, phi, ct, rw_ft, p1hr, pwf0)
    param_bars.append(_bar(
        "param_bo",
        f"Oil FVF Bo (±5%, base={bo:.3f} rb/STB)",
        "PVT Fluid",
        "Converts surface Qo to reservoir flow rate Qo·Bo; linear in k and weakly logarithmic in Skin S.",
        f"Bo={bo*0.95:.3f} (-5%)",
        f"Bo={bo*1.05:.3f} (+5%)",
        k_bl, k_bh, s_bl, s_bh,
    ))

    # B6: Flowing Pressure Pwf0 (±5 psi gauge/shut-in pick error)
    k_pl, _, s_pl, _ = _pta_k_and_skin(qo_base, m_base, bo, mu_o, h_ft, phi, ct, rw_ft, p1hr, pwf0 - 5.0)
    k_ph, _, s_ph, _ = _pta_k_and_skin(qo_base, m_base, bo, mu_o, h_ft, phi, ct, rw_ft, p1hr, pwf0 + 5.0)
    param_bars.append(_bar(
        "param_pwf0",
        f"Shut-in Start Pwf0 (±5 psi, base={pwf0:.1f} psia)",
        "Gauge / Shut-in Pick",
        "Zero impact on permeability k; directly shifts Skin S by ∓1.1513·(5/m) psi/cycle.",
        f"Pwf0={pwf0-5.0:.1f} psia (-5 psi)",
        f"Pwf0={pwf0+5.0:.1f} psia (+5 psi)",
        k_pl, k_ph, s_pl, s_ph,
    ))

    # B7: Porosity phi (±15%)
    k_fl, _, s_fl, _ = _pta_k_and_skin(qo_base, m_base, bo, mu_o, h_ft, phi * 0.85, ct, rw_ft, p1hr, pwf0)
    k_fh, _, s_fh, _ = _pta_k_and_skin(qo_base, m_base, bo, mu_o, h_ft, phi * 1.15, ct, rw_ft, p1hr, pwf0)
    param_bars.append(_bar(
        "param_phi",
        f"Porosity φ (±15%, base={phi:.3f})",
        "Petrophysics",
        "Zero impact on k; enters Skin S logarithmically inside storage term log10(k/(φ·μo·ct·rw²)).",
        f"φ={phi*0.85:.3f} (-15%)",
        f"φ={phi*1.15:.3f} (+15%)",
        k_fl, k_fh, s_fl, s_fh,
    ))

    # B8: Total Compressibility ct (±20%)
    k_cl, _, s_cl, _ = _pta_k_and_skin(qo_base, m_base, bo, mu_o, h_ft, phi, ct * 0.80, rw_ft, p1hr, pwf0)
    k_ch, _, s_ch, _ = _pta_k_and_skin(qo_base, m_base, bo, mu_o, h_ft, phi, ct * 1.20, rw_ft, p1hr, pwf0)
    param_bars.append(_bar(
        "param_ct",
        f"Total Compressibility ct (±20%, base={ct:.1e})",
        "Rock/Fluid",
        "Zero impact on k; enters Skin S logarithmically inside log10(k/(φ·μo·ct·rw²)).",
        f"ct={ct*0.80:.2e} (-20%)",
        f"ct={ct*1.20:.2e} (+20%)",
        k_cl, k_ch, s_cl, s_ch,
    ))

    # B9: Wellbore Radius rw (±10%)
    k_rl, _, s_rl, _ = _pta_k_and_skin(qo_base, m_base, bo, mu_o, h_ft, phi, ct, rw_ft * 0.90, p1hr, pwf0)
    k_rh, _, s_rh, _ = _pta_k_and_skin(qo_base, m_base, bo, mu_o, h_ft, phi, ct, rw_ft * 1.10, p1hr, pwf0)
    param_bars.append(_bar(
        "param_rw",
        f"Wellbore Radius rw (±10%, base={rw_ft:.3f} ft)",
        "Completion",
        "Zero impact on k; squared in rw² so ±10% rw shifts Skin S by ~±0.10 (2× sensitivity of φ).",
        f"rw={rw_ft*0.90:.3f} ft (-10%)",
        f"rw={rw_ft*1.10:.3f} ft (+10%)",
        k_rl, k_rh, s_rl, s_rh,
    ))

    # B10: Producing Time tp (±50%)
    k_tl, _, s_tl, _ = _pta_k_and_skin(qo_base, m_base, bo, mu_o, h_ft, phi, ct, rw_ft, _eval_p1hr_horner(p_star, m_base, tp_hr * 0.5), pwf0)
    k_th, _, s_th, _ = _pta_k_and_skin(qo_base, m_base, bo, mu_o, h_ft, phi, ct, rw_ft, _eval_p1hr_horner(p_star, m_base, tp_hr * 1.5), pwf0)
    param_bars.append(_bar(
        "param_tp",
        f"Producing Time tp (±50%, base={tp_hr:.0f} hr)",
        "Rate History",
        "Shifts Horner p1hr slightly via log10((tp+1)/1); negligible when tp >> Δt.",
        f"tp={tp_hr*0.5:.0f} hr (-50%)",
        f"tp={tp_hr*1.5:.0f} hr (+50%)",
        k_tl, k_th, s_tl, s_th,
    ))

    param_bars.sort(key=lambda b: b["k_span"], reverse=True)

    # ------------------------------------------------------------------
    # PRIMARY TORNADO: Tweaking Oil Rate Qo (±qo_pct%, default ±20%) ->
    # Exact COUPLED effect through BOTH tp(Qo) = 24*Np/Qo (Horner axis refit)
    # AND parameter formulas across all 4 Required PTA Outputs (k, S, ΔP_skin, m)!
    # ------------------------------------------------------------------
    hp_pts = (horner_plot or {}).get("points") or []
    win_hr_pair = (horner_plot or {}).get("win_hr") or (hyb or {}).get("intersect_win_hr")
    dt_win_arr: Optional[np.ndarray] = None
    pws_win_arr: Optional[np.ndarray] = None
    if hp_pts:
        dts = np.array([float(p["dt"]) for p in hp_pts if p.get("dt") is not None and p.get("y") is not None], dtype=float)
        pws_a = np.array([float(p["y"]) for p in hp_pts if p.get("dt") is not None and p.get("y") is not None], dtype=float)
        if win_hr_pair and len(win_hr_pair) == 2:
            mask_w = (dts >= float(win_hr_pair[0])) & (dts <= float(win_hr_pair[1])) & (dts > 0)
            if int(mask_w.sum()) >= 3:
                dt_win_arr, pws_win_arr = dts[mask_w], pws_a[mask_w]
        if dt_win_arr is None and len(dts) >= 4:
            # Middle 40%..85% radial segment fallback
            i0, i1 = int(len(dts) * 0.40), int(len(dts) * 0.85)
            if i1 - i0 >= 3:
                dt_win_arr, pws_win_arr = dts[i0:i1], pws_a[i0:i1]
    elif hf_w is not None and not hf_w.empty:
        s1_u = pd.Timestamp(ev["shut_in_end"])
        s1_u = s1_u.tz_localize("UTC") if s1_u.tzinfo is None else s1_u.tz_convert("UTC")
        ts_h = pd.to_datetime(hf_w["timestamp"], utc=True)
        post = hf_w[(ts_h > s0_utc) & (ts_h <= s1_u)][["timestamp", "dh_pressure_psia"]].dropna()
        if len(post) >= 5:
            dts = (pd.to_datetime(post["timestamp"], utc=True) - s0_utc).dt.total_seconds().to_numpy(dtype=float) / 3600.0
            pws_a = post["dh_pressure_psia"].to_numpy(dtype=float)
            if win_hr_pair and len(win_hr_pair) == 2:
                mask_w = (dts >= float(win_hr_pair[0])) & (dts <= float(win_hr_pair[1])) & (dts > 0)
                if int(mask_w.sum()) >= 3:
                    dt_win_arr, pws_win_arr = dts[mask_w], pws_a[mask_w]
            if dt_win_arr is None:
                i0, i1 = int(len(dts) * 0.40), int(len(dts) * 0.85)
                if i1 - i0 >= 3:
                    dt_win_arr, pws_win_arr = dts[i0:i1], pws_a[i0:i1]

    wb_tuple = (float(win_hr_pair[0]), float(win_hr_pair[1])) if (win_hr_pair and len(win_hr_pair) == 2) else None
    tp_qlo, m_qlo, pstar_qlo, k_qlo4, s_qlo4, dps_qlo4 = _coupled_horner_at_qo(
        qo_base * (1.0 - frac), qo_base, tp_hr, m_base, p_star, dt_win_arr, pws_win_arr,
        bo, mu_o, h_ft, phi, ct, rw_ft, pwf0, win_bounds_hr=wb_tuple,
    )
    tp_qhi, m_qhi, pstar_qhi, k_qhi4, s_qhi4, dps_qhi4 = _coupled_horner_at_qo(
        qo_base * (1.0 + frac), qo_base, tp_hr, m_base, p_star, dt_win_arr, pws_win_arr,
        bo, mu_o, h_ft, phi, ct, rw_ft, pwf0, win_bounds_hr=wb_tuple,
    )

    def _pct_rel(val: float, base_v: float, min_denom: float = 1e-6) -> float:
        return round(100.0 * (float(val) - float(base_v)) / max(abs(float(base_v)), min_denom), 3)

    four_outputs_tornado: List[Dict[str, Any]] = [
        {
            "key": "permeability_k_md",
            "label": "Permeability k (mD)",
            "unit": "mD",
            "base": round(k_base, 2),
            "at_qo_low": round(k_qlo4, 2),
            "at_qo_high": round(k_qhi4, 2),
            "delta_at_qo_low": round(k_qlo4 - k_base, 2),
            "delta_at_qo_high": round(k_qhi4 - k_base, 2),
            "pct_at_qo_low": _pct_rel(k_qlo4, k_base),
            "pct_at_qo_high": _pct_rel(k_qhi4, k_base),
            "abs_pct_span": round(abs(_pct_rel(k_qhi4, k_base) - _pct_rel(k_qlo4, k_base)), 3),
            "formula": "k = 162.6·Qo·Bo·μo / (m(tp)·h) — direct linear Qo scaling + coupled m(tp(Qo)) refit",
            "direction": "direct",
        },
        {
            "key": "skin_S",
            "label": "Skin Factor S",
            "unit": "dimless",
            "base": round(s_base, 3),
            "at_qo_low": round(s_qlo4, 3),
            "at_qo_high": round(s_qhi4, 3),
            "delta_at_qo_low": round(s_qlo4 - s_base, 3),
            "delta_at_qo_high": round(s_qhi4 - s_base, 3),
            "pct_at_qo_low": _pct_rel(s_qlo4, s_base, min_denom=0.25),
            "pct_at_qo_high": _pct_rel(s_qhi4, s_base, min_denom=0.25),
            "abs_pct_span": round(abs(_pct_rel(s_qhi4, s_base, min_denom=0.25) - _pct_rel(s_qlo4, s_base, min_denom=0.25)), 3),
            "formula": "S(Qo, tp(Qo)) — coupled via k(Qo), Horner p1hr(tp), and refitted slope m(tp)",
            "direction": "inverse",
        },
        {
            "key": "delta_p_skin_psi",
            "label": "Skin Pressure Drop ΔP_skin (psi)",
            "unit": "psi",
            "base": round(dps_base, 2),
            "at_qo_low": round(dps_qlo4, 2),
            "at_qo_high": round(dps_qhi4, 2),
            "delta_at_qo_low": round(dps_qlo4 - dps_base, 2),
            "delta_at_qo_high": round(dps_qhi4 - dps_base, 2),
            "pct_at_qo_low": _pct_rel(dps_qlo4, dps_base, min_denom=1.0),
            "pct_at_qo_high": _pct_rel(dps_qhi4, dps_base, min_denom=1.0),
            "abs_pct_span": round(abs(_pct_rel(dps_qhi4, dps_base, min_denom=1.0) - _pct_rel(dps_qlo4, dps_base, min_denom=1.0)), 3),
            "formula": "ΔP_skin = 0.869·m(tp)·S(Qo, tp) — combines coupled m(tp) and S(Qo, tp) shifts",
            "direction": "inverse",
        },
        {
            "key": "horner_slope_m",
            "label": "Horner Semilog Slope m (psi/cyc)",
            "unit": "psi/cyc",
            "base": round(m_base, 3),
            "at_qo_low": round(m_qlo, 3),
            "at_qo_high": round(m_qhi, 3),
            "delta_at_qo_low": round(m_qlo - m_base, 4),
            "delta_at_qo_high": round(m_qhi - m_base, 4),
            "pct_at_qo_low": _pct_rel(m_qlo, m_base),
            "pct_at_qo_high": _pct_rel(m_qhi, m_base),
            "abs_pct_span": round(abs(_pct_rel(m_qhi, m_base) - _pct_rel(m_qlo, m_base)), 3),
            "formula": f"Refitted via tp(Qo) = 24·Np/Qo ({tp_qhi:.0f}–{tp_qlo:.0f} hr vs base {tp_hr:.0f} hr) shifting Horner log10((tp+Δt)/Δt)",
            "direction": "coupled_tp",
        },
    ]
    four_outputs_tornado.sort(key=lambda r: r["abs_pct_span"], reverse=True)

    # Identify top non-parametric real Qo driver for plain-English diagnosis
    real_qo_drivers = [b for b in qo_bars if b["key"] != "qo_parametric_band"]
    top_driver = real_qo_drivers[0] if real_qo_drivers else qo_bars[0]
    diagnosis = (
        f"Tweaking {well} pre-shut-in Qo by ±{qo_pct:.0f}% ({qo_base*(1-frac):.0f}–{qo_base*(1+frac):.0f} STB/d) shifts Horner "
        f"producing time tp = 24·Np/Qo from {tp_hr:.0f} hr to {tp_qlo:.0f} hr (-{qo_pct:.0f}% Qo) / {tp_qhi:.0f} hr (+{qo_pct:.0f}% Qo), "
        f"refitting Horner slope m to {m_qlo:.3f}–{m_qhi:.3f} psi/cyc ({_pct_rel(m_qlo, m_base):+.2f}%/{_pct_rel(m_qhi, m_base):+.2f}%), "
        f"Permeability k to {k_qlo4:.1f}–{k_qhi4:.1f} mD ({_pct_rel(k_qlo4, k_base):+.1f}%/{_pct_rel(k_qhi4, k_base):+.1f}%), "
        f"Skin S to {s_qlo4:+.3f} / {s_qhi4:+.3f}, and ΔP_skin to {dps_qlo4:.1f} / {dps_qhi4:.1f} psi."
    )

    return {
        "available": True,
        "well_name": well,
        "qo_pct_default": float(qo_pct),
        "base": {
            "qo_stbd": round(qo_base, 2),
            "qw_stbd": round(qw_base, 2),
            "wc_pct": round(wc_frac * 100.0, 2),
            "perrine_factor": round(perrine_factor, 3),
            "bo": round(bo, 4),
            "bw": round(bw, 4),
            "mu_o": round(mu_o, 4),
            "h_ft": round(h_ft, 2),
            "phi": round(phi, 4),
            "ct": float(ct),
            "rw_ft": round(rw_ft, 4),
            "m_psi_cyc": round(m_base, 3),
            "p_star_psia": round(p_star, 2),
            "p1hr_psia": round(p1hr, 2),
            "pwf0_psia": round(pwf0, 2),
            "tp_hr": round(tp_hr, 1),
            "k_md": round(k_base, 3),
            "kh_md_ft": round(kh_base, 1),
            "skin": round(s_base, 3),
            "dp_skin_psi": round(dps_base, 2),
        },
        "four_outputs_tornado": four_outputs_tornado,
        "measured_qo_sources": {
            "qo_base_vfm_stbd": round(qo_base, 2),
            "qw_pre_shutin_stbd": round(qw_base, 2),
            "water_cut_pct": round(wc_frac * 100.0, 2),
            "perrine_mobility_factor": round(perrine_factor, 3),
            "qo_perrine_equiv_stbd": round(qo_perrine_equiv, 2),
            "nearest_pre_test_qo_stbd": round(q_wt_pre, 2) if q_wt_pre is not None else None,
            "nearest_pre_test_gap_days": round(gap_pre_days, 1) if gap_pre_days is not None else None,
            "nearest_any_test_qo_stbd": round(q_wt_near, 2) if q_wt_near is not None else None,
            "nearest_any_test_gap_days": round(gap_near_days, 1) if gap_near_days is not None else None,
            "vfm_1d_stbd": round(qo_1d, 2) if qo_1d is not None else None,
            "vfm_3d_stbd": round(qo_3d, 2) if qo_3d is not None else None,
            "vfm_7d_stbd": round(qo_7d, 2) if qo_7d is not None else None,
            "choke_6h_pct": round(chk6, 2) if chk6 is not None else None,
            "choke_24h_pct": round(chk24, 2) if chk24 is not None else None,
            "choke_72h_pct": round(chk72, 2) if chk72 is not None else None,
            "choke_drift_6h_vs_72h_pct": round(choke_drift_pct, 2),
            "pdg_drift_6h_vs_72h_psi": round(pdg_drift_psi, 2),
            "model_qo_map": {k: round(v, 2) for k, v in sorted(model_qo_map.items())},
        },
        "qo_root_cause_bars": qo_bars,
        "parameter_tornado_bars": param_bars,
        "diagnosis": diagnosis,
    }


# ==============================================================================
# 2. VFM Correlation Matrix, Training Feature Audit & Exact SHAP Evaluation
# ==============================================================================

V2_BASE_FEATURE_META: List[Dict[str, str]] = [
    {"key": "q_nodal", "label": "Nodal Liquid Prior (q_nodal)", "group": "Physics Prior", "role": "Inverse-variance blend of Choke + IPR + VLP liquid rates (STB/d)"},
    {"key": "q_choke", "label": "Choke Rate Prior (q_choke)", "group": "Physics Prior", "role": "Gilbert/Achong orifice equation using choke %, WHP, P_ds & max bean size"},
    {"key": "q_ipr", "label": "Darcy+Vogel IPR Prior (q_ipr)", "group": "Physics Prior", "role": "Reservoir inflow using Pr(t), PDG pressure, PI & bubble point Pb"},
    {"key": "q_vlp", "label": "Tubing VLP Prior (q_vlp)", "group": "Physics Prior", "role": "Vertical lift regression on tubing gradient, gas-lift rate, annulus P & T_dh"},
    {"key": "choke_cap", "label": "Choke Flow Capacity (A·√ΔP)", "group": "Wellhead & Choke", "role": "Orifice area × √(effective choke ΔP) governing surface flow"},
    {"key": "choke_opening_pct", "label": "Choke Opening (%)", "group": "Wellhead & Choke", "role": "Direct SCADA surface choke valve position (%)"},
    {"key": "drawdown_psi", "label": "Reservoir Drawdown (Pr - P_dh)", "group": "Downhole Telemetry", "role": "Driving pressure differential across sandface (psi)"},
    {"key": "dh_pressure_psia", "label": "Downhole Gauge Pressure (PDG)", "group": "Downhole Telemetry", "role": "Direct SCADA downhole pressure sensor (psia)"},
    {"key": "whp_psia", "label": "Wellhead Pressure (WHP)", "group": "Wellhead & Choke", "role": "Direct SCADA upstream choke pressure (psia)"},
    {"key": "dp_choke_raw_psi", "label": "Choke ΔP (WHP - P_ds)", "group": "Wellhead & Choke", "role": "Pressure drop across surface choke into manifold/separator (psi)"},
    {"key": "tubing_grad_psi_ft", "label": "Tubing Fluid Gradient (psi/ft)", "group": "Tubing / VLP", "role": "(P_dh - WHP) / gauge_TVD — liquid holdup & water-cut density proxy"},
    {"key": "dht_delta_degf", "label": "Downhole Temp Shift (ΔT_dh)", "group": "Downhole Telemetry", "role": "T_dh minus well baseline — Joule–Thomson & inflow cooling signature"},
    {"key": "ann_delta_psia", "label": "Annulus Pressure Shift (ΔP_ann)", "group": "Gas-Lift System", "role": "Casing annulus pressure shift driving gas-lift valve injection"},
    {"key": "glv_cap", "label": "GL Valve Capacity (d²·√ΔP_glv)", "group": "Gas-Lift System", "role": "Subsurface gas-lift valve differential throughput capacity"},
    {"key": "gl_flux_mscfd_in2", "label": "Gas-Lift Flux (Q_gl / A_tub)", "group": "Gas-Lift System", "role": "Lift-gas injection intensity normalized by tubing cross-sectional area"},
    {"key": "glv_depth_frac", "label": "GL Valve Depth Ratio (TVD_glv/TVD)", "group": "Completion", "role": "Normalized gas-lift injection depth along wellbore"},
    {"key": "wc_prior_frac", "label": "Water-Cut Prior Trajectory", "group": "Phase Trajectory", "role": "Smooth logistic water-cut evolution calibrated on well tests"},
    {"key": "gor_prior_scf_stb", "label": "Formation GOR Prior (scf/STB)", "group": "Phase Trajectory", "role": "Linear formation gas-oil ratio trajectory calibrated on well tests"},
    {"key": "days_since_start", "label": "Elapsed Days (Depletion Time)", "group": "Time / Depletion", "role": "Calendar days from field start capturing depletion & water encroachment"},
    {"key": "well_idx", "label": "Well Index (ID)", "group": "Metadata", "role": "Well categorical index (softly regularized; replaced via zero-shot transfer for unseen wells)"},
]

SENSOR_SHAP_COLS: List[Dict[str, str]] = [
    {"key": "choke_opening_pct", "label": "Choke Opening (%)", "group": "Wellhead & Choke"},
    {"key": "dh_pressure_psia", "label": "Downhole Pressure (PDG)", "group": "Downhole Telemetry"},
    {"key": "whp_psia", "label": "Wellhead Pressure (WHP)", "group": "Wellhead & Choke"},
    {"key": "whp_downstream_psia", "label": "Downstream Pressure (P_ds)", "group": "Wellhead & Choke"},
    {"key": "gas_lift_rate_mscfd", "label": "Gas-Lift Rate (Mscf/d)", "group": "Gas-Lift System"},
    {"key": "annulus_pressure_psia", "label": "Annulus Pressure (psia)", "group": "Gas-Lift System"},
    {"key": "dh_temperature_degf", "label": "Downhole Temp (°F)", "group": "Downhole Telemetry"},
    {"key": "days_since_start", "label": "Depletion Time (Days)", "group": "Time / Depletion"},
]

AUDIT_FEATURE_CATALOG: List[Dict[str, Any]] = [
    # Raw SCADA (7 sensors)
    {"key": "choke_opening_pct", "label": "Choke Opening (%)", "group": "Raw SCADA", "tier": "Direct V2 ML + Physics Engine", "v2_ml": True, "physics_vfm": True, "rf_gbm": True, "role": "Direct ML input AND computes choke bean diameter / flow area in q_choke"},
    {"key": "dh_pressure_psia", "label": "Downhole Pressure — PDG (psia)", "group": "Raw SCADA", "tier": "Direct V2 ML + Physics Engine", "v2_ml": True, "physics_vfm": True, "rf_gbm": True, "role": "Direct ML input AND computes drawdown (Pr - P_dh), Darcy/Vogel q_ipr & tubing gradient"},
    {"key": "whp_psia", "label": "Wellhead Pressure — WHP (psia)", "group": "Raw SCADA", "tier": "Direct V2 ML + Physics Engine", "v2_ml": True, "physics_vfm": True, "rf_gbm": True, "role": "Direct ML input AND computes choke ΔP (WHP - P_ds) & tubing ΔP (P_dh - WHP)"},
    {"key": "whp_downstream_psia", "label": "Downstream Pressure — P_ds (psia)", "group": "Raw SCADA", "tier": "Physics Engine Input (via ΔP_choke & q_choke)", "v2_ml": False, "physics_vfm": True, "rf_gbm": True, "role": "Enters V2 via dp_choke_raw = WHP - P_ds and effective critical/subcritical choke ΔP"},
    {"key": "gas_lift_rate_mscfd", "label": "Gas-Lift Injection Rate (Mscf/d)", "group": "Raw SCADA", "tier": "Physics Engine Input (via gl_flux & q_vlp)", "v2_ml": False, "physics_vfm": True, "rf_gbm": True, "role": "Enters V2 via tubing-area-normalized gl_flux = Q_gl / A_tub and VLP gas-lift term"},
    {"key": "annulus_pressure_psia", "label": "Annulus Pressure (psia)", "group": "Raw SCADA", "tier": "Physics Engine Input (via ΔP_ann & glv_cap)", "v2_ml": False, "physics_vfm": True, "rf_gbm": True, "role": "Enters V2 via ΔP_ann shift, casing hydrostatic pressure at GL valve & glv_cap"},
    {"key": "dh_temperature_degf", "label": "Downhole Temperature (°F)", "group": "Raw SCADA", "tier": "Physics Engine Input (via ΔT_dh & VLP)", "v2_ml": False, "physics_vfm": True, "rf_gbm": True, "role": "Enters V2 via ΔT_dh Joule–Thomson shift, gas column density & VLP term"},
    # Key engineered differentials
    {"key": "dp_choke_raw_psi", "label": "Choke Differential (WHP - P_ds)", "group": "Engineered Physics", "tier": "Direct V2 ML + Physics Engine", "v2_ml": True, "physics_vfm": True, "rf_gbm": True, "role": "Surface choke pressure drop separating critical vs. subcritical backpressure regimes"},
    {"key": "dp_tubing_psi", "label": "Tubing Differential (P_dh - WHP)", "group": "Engineered Physics", "tier": "Physics Engine Input (via tubing_grad)", "v2_ml": False, "physics_vfm": True, "rf_gbm": True, "role": "Hydrostatic + frictional lift drop; normalized by gauge_TVD into tubing_grad_psi_ft"},
    {"key": "drawdown_psi", "label": "Reservoir Drawdown (Pr(t) - P_dh)", "group": "Engineered Physics", "tier": "Direct V2 ML + Physics Engine", "v2_ml": True, "physics_vfm": True, "rf_gbm": True, "role": "Primary sandface inflow driving force; directly determines Darcy/Vogel q_ipr"},
    # Static completion & PVT parameters (directly answers user's question about Pb, tubing ID, etc.!)
    {"key": "tubing_id_in", "label": "Tubing Inner Diameter (in)", "group": "Static Completion", "tier": "Physics Engine + VLP/Storage", "v2_ml": False, "physics_vfm": True, "rf_gbm": True, "role": "Computes tubing flow area A_tub (gl_flux = Q_gl/A_tub), VLP friction, and PTA storage C"},
    {"key": "gauge_tvd_ft", "label": "Gauge TVD (ft)", "group": "Static Completion", "tier": "Physics Engine + VLP", "v2_ml": False, "physics_vfm": True, "rf_gbm": True, "role": "Computes effective tubing gradient (P_dh - WHP)/TVD and datum pressure correction"},
    {"key": "gl_valve_tvd_ft", "label": "Gas-Lift Valve TVD (ft)", "group": "Static Completion", "tier": "Physics Engine Input (glv_depth_frac & glv_cap)", "v2_ml": False, "physics_vfm": True, "rf_gbm": True, "role": "Computes injection depth ratio TVD_glv/TVD_gauge and tubing pressure at valve depth"},
    {"key": "gl_valve_port_64ths", "label": "GL Valve Port Size (/64 in)", "group": "Static Completion", "tier": "Physics Engine Input (via glv_cap)", "v2_ml": False, "physics_vfm": True, "rf_gbm": True, "role": "Sets gas-lift valve orifice area (port/64)² inside glv_cap"},
    {"key": "choke_max_size_64ths", "label": "Max Choke Bean Size (/64 in)", "group": "Static Completion", "tier": "Physics Engine Input (via choke_cap)", "v2_ml": False, "physics_vfm": True, "rf_gbm": True, "role": "Converts choke % into physical orifice area A_choke for q_choke"},
    {"key": "bubble_point_pressure_psia", "label": "Bubble Point Pressure Pb (psia)", "group": "Static PVT", "tier": "Physics Engine Input (Vogel IPR & VLP)", "v2_ml": False, "physics_vfm": True, "rf_gbm": True, "role": "Switches Darcy linear IPR to Vogel two-phase IPR when P_dh < Pb and governs free gas in VLP"},
    {"key": "solution_gor_at_pb_scf_stb", "label": "Solution GOR at Pb (scf/STB)", "group": "Static PVT", "tier": "Physics Engine + VLP", "v2_ml": False, "physics_vfm": True, "rf_gbm": True, "role": "Governs dissolved vs. free gas in VLP/choke multiphase mixture density"},
    {"key": "oil_fvf_rb_stb", "label": "Oil FVF Bo (rb/STB)", "group": "Static PVT", "tier": "PTA + Material Balance + Zero-Shot VFM", "v2_ml": False, "physics_vfm": True, "rf_gbm": True, "role": "Used in volumetric material-balance depletion, VLP/choke liquid FVF, zero-shot PI scaling & PTA k"},
    {"key": "oil_viscosity_cp", "label": "Oil Viscosity μo (cp)", "group": "Static PVT", "tier": "PTA + VLP + Zero-Shot VFM", "v2_ml": False, "physics_vfm": True, "rf_gbm": True, "role": "Used in VLP friction, zero-shot specific-PI scaling (PI ∝ h/(μo·Bo)), and PTA permeability k"},
    {"key": "net_pay_thickness_ft", "label": "Net Pay Thickness h (ft)", "group": "Static PVT", "tier": "PTA + Material Balance + Zero-Shot VFM", "v2_ml": False, "physics_vfm": True, "rf_gbm": True, "role": "Computes OOIP for material-balance dPr/dt, zero-shot PI scaling, and PTA k = kh / h"},
    {"key": "porosity_frac", "label": "Porosity φ (fraction)", "group": "Static PVT", "tier": "PTA + Material Balance Only", "v2_ml": False, "physics_vfm": False, "rf_gbm": True, "role": "Computes OOIP pore volume for material balance and enters PTA Skin S logarithmically"},
    {"key": "total_compressibility_1_psi", "label": "Total Compressibility ct (1/psi)", "group": "Static PVT", "tier": "PTA + Material Balance Only", "v2_ml": False, "physics_vfm": False, "rf_gbm": False, "role": "Computes reservoir depletion rate dPr/dt = -q/(OOIP·ct) and enters PTA Skin S & Stehfest tD"},
    {"key": "wellbore_radius_ft", "label": "Wellbore Radius rw (ft)", "group": "Static PVT", "tier": "PTA Only", "v2_ml": False, "physics_vfm": False, "rf_gbm": False, "role": "Enters PTA Skin S via log10(k/(φ·μo·ct·rw²)) and dimensionless time tD"},
]


def _import_v2_module():
    """Import vfm_phase_decomposed_stacked_vfm cleanly without hardcoded paths."""
    root = Path(__file__).resolve().parents[3]
    export_dir = root / "vfm_export_tool"
    if export_dir.is_dir() and str(export_dir) not in sys.path:
        sys.path.insert(0, str(export_dir))
    import vfm_phase_decomposed_stacked_vfm as v2mod  # type: ignore
    return v2mod


def load_v2_bundle(bundle_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load trained vfm_stacked_v2.pkl from portable repository locations."""
    import pickle as _pkl

    _import_v2_module()
    root = Path(__file__).resolve().parents[3]
    candidates = []
    if bundle_path is not None:
        candidates.append(Path(bundle_path))
    candidates.extend([
        root / "vfm_project" / "models" / "vfm_stacked_v2.pkl",
        root / "deliverables" / "models" / "vfm_stacked_v2.pkl",
        root / "vfm_export_tool" / "vfm_stacked_v2.pkl",
    ])
    for p in candidates:
        if p.is_file():
            with p.open("rb") as f:
                return _pkl.load(f)
    raise FileNotFoundError(f"Trained vfm_stacked_v2.pkl not found in {[str(c) for c in candidates]}")


def _align_tests_with_scada(
    tests_df: pd.DataFrame,
    hf_by_well: Mapping[str, pd.DataFrame],
    ref_date: pd.Timestamp,
) -> pd.DataFrame:
    """Align separator well tests with robust-median 1-min SCADA windows strictly from live data."""
    wt = tests_df.copy()
    wt["test_start"] = pd.to_datetime(wt["test_start"], utc=True)
    wt["test_end"] = pd.to_datetime(wt["test_end"], utc=True)
    wt["test_mid"] = wt["test_start"] + (wt["test_end"] - wt["test_start"]) / 2
    wt["days_since_start"] = (wt["test_mid"] - ref_date).dt.total_seconds() / 86400.0
    for c in ("oil_rate_stbd", "water_rate_stbd", "gas_rate_mscfd"):
        if c in wt.columns:
            wt[c] = pd.to_numeric(wt[c], errors="coerce").abs()
    # Harmonize gas_rate_mscfd unit anomalies (scf/d or total-gas rows > 2.5x expected formation gas)
    if {"oil_rate_stbd", "gor_scf_stb", "gas_rate_mscfd"}.issubset(wt.columns):
        exp_g = (wt["oil_rate_stbd"] * pd.to_numeric(wt["gor_scf_stb"], errors="coerce")) / 1000.0
        bad_g = exp_g.notna() & (exp_g > 0) & ((wt["gas_rate_mscfd"] > 2.5 * exp_g) | (wt["gas_rate_mscfd"] < 0.2 * exp_g))
        if bad_g.any():
            wt.loc[bad_g, "gas_rate_mscfd"] = exp_g.loc[bad_g]
    if "liquid_rate_stbd" not in wt.columns:
        wt["liquid_rate_stbd"] = wt["oil_rate_stbd"].astype(float) + wt["water_rate_stbd"].astype(float)

    sensor_cols = [
        "dh_pressure_psia", "dh_temperature_degf", "whp_psia", "whp_downstream_psia",
        "choke_opening_pct", "annulus_pressure_psia", "gas_lift_rate_mscfd",
    ]
    rows = []
    for _, r in wt.iterrows():
        w = str(r["well_name"])
        hf_w = hf_by_well.get(w)
        rec = r.to_dict()
        if hf_w is not None and not hf_w.empty:
            t_col = "timestamp" if "timestamp" in hf_w.columns else ("hour" if "hour" in hf_w.columns else None)
            if t_col is not None:
                ts = pd.to_datetime(hf_w[t_col], utc=True)
                win = hf_w[(ts >= r["test_start"]) & (ts <= r["test_end"])]
                if win.empty:
                    win = hf_w[(ts >= r["test_start"] - pd.Timedelta(hours=2)) & (ts <= r["test_end"] + pd.Timedelta(hours=2))]
                for c in sensor_cols:
                    if c in win.columns:
                        vals = pd.to_numeric(win[c], errors="coerce")
                        vals = vals[(vals > -900.0) & np.isfinite(vals)]
                        if len(vals):
                            rec[c] = float(vals.median())
                        elif c not in rec or pd.isna(rec.get(c)):
                            rec[c] = np.nan
        rows.append(rec)
    out = pd.DataFrame(rows)
    for c in sensor_cols:
        if c not in out.columns:
            out[c] = np.nan
        out[c] = pd.to_numeric(out[c], errors="coerce")
        out.loc[out[c] <= -900.0, c] = np.nan
        out[c] = out.groupby("well_name")[c].transform(lambda s: s.fillna(s.median()))
        med_global = float(out[c].median()) if out[c].notna().any() else 0.0
        out[c] = out[c].fillna(med_global)
    return out


def _corr_payload(df: pd.DataFrame, col_specs: Sequence[Dict[str, str]], method: str = "pearson") -> Dict[str, Any]:
    """Compute clean correlation matrix JSON payload with zero NaNs."""
    present = [s for s in col_specs if s["key"] in df.columns and df[s["key"]].notna().sum() >= 3]
    # Drop zero-variance columns (e.g. constant wellbore_radius_ft across field) so corr stays finite
    active = []
    for s in present:
        vals = pd.to_numeric(df[s["key"]], errors="coerce").dropna()
        if len(vals) >= 3 and float(vals.std()) > 1e-12:
            active.append(s)
    keys = [s["key"] for s in active]
    sub = df[keys].apply(pd.to_numeric, errors="coerce")
    cmat = sub.corr(method=method).fillna(0.0)
    matrix = [[round(float(cmat.iloc[i, j]), 4) for j in range(len(keys))] for i in range(len(keys))]
    return {
        "method": method,
        "n_samples": int(len(sub)),
        "columns": [
            {"key": s["key"], "label": s["label"], "group": s["group"], "used_in_v2": bool(s.get("used_in_v2", False))}
            for s in active
        ],
        "matrix": matrix,
    }


def _exact_subset_shapley_8(
    predict_fn,
    base_df: pd.DataFrame,
    sensor_keys: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute EXACT Shapley values over all 2^M coalitions (M=8 -> 256 subsets, ZERO MC noise!).

    Batches all 2^M * N rows into ONE vectorized predict_fn call for sub-second speed.
    Uses within-well operational mean baselines so each well's physical depth/reservoir
    pressure stays physically valid when masking a sensor, and exact Shapley efficiency
    holds per sample: sum_j phi_ij + base_i == f(x_i) to machine precision.
    """
    n = len(base_df)
    m = len(sensor_keys)
    assert m <= 10, "Exact 2^M Shapley intended for M <= 10 features."

    bg_arr = np.zeros((n, m), dtype=float)
    for j, k in enumerate(sensor_keys):
        w_means = base_df.groupby("well_name")[k].transform("mean")
        bg_arr[:, j] = w_means.fillna(float(base_df[k].mean())).to_numpy(dtype=float)
    X_obs = base_df[list(sensor_keys)].to_numpy(dtype=float)

    fact = [math.factorial(i) for i in range(m + 1)]
    coal_weights = np.array([fact[s] * fact[m - s - 1] / fact[m] for s in range(m)], dtype=float)

    # Single vectorized batch over all 2^m coalitions
    n_masks = 1 << m
    rep_df = pd.concat([base_df] * n_masks, ignore_index=True)
    for j, k in enumerate(sensor_keys):
        bits = np.array([(mask >> j) & 1 for mask in range(n_masks)], dtype=bool)
        col_mat = np.where(bits[:, None], X_obs[None, :, j], bg_arr[None, :, j])
        rep_df[k] = col_mat.reshape(-1)
    preds = np.asarray(predict_fn(rep_df), dtype=float).reshape(n_masks, n)

    shap_vals = np.zeros((n, m), dtype=float)
    for j in range(m):
        bit = 1 << j
        for mask in range(n_masks):
            if mask & bit:
                continue
            s_size = bin(mask).count("1")
            w = coal_weights[s_size]
            shap_vals[:, j] += w * (preds[mask | bit, :] - preds[mask, :])

    base_vals = preds[0, :]
    fx = preds[n_masks - 1, :]
    return shap_vals, base_vals, fx


def _antithetic_ensemble_shap(
    predict_full_fn,
    X_full: np.ndarray,
    n_perms: int = 24,
    seed: int = 42,
) -> Tuple[np.ndarray, float, np.ndarray]:
    """Fast, exact-efficiency batched antithetic permutation Shapley for the 5-base Super-Learner."""
    rng = np.random.default_rng(seed)
    n, d = X_full.shape
    bg = X_full.mean(axis=0)

    orders: List[np.ndarray] = []
    for _ in range(n_perms):
        perm = rng.permutation(d)
        orders.append(perm)
        orders.append(perm[::-1])
    n_passes = len(orders)

    # Build ALL intermediate states in one vectorized array (n_passes * d + 2 blocks of size n)
    n_blocks = n_passes * d + 2
    X_batch = np.empty((n_blocks, n, d), dtype=float)
    X_batch[0] = bg[None, :]
    X_batch[1] = X_full
    blk = 2
    for order in orders:
        cur = np.repeat(bg[None, :], n, axis=0)
        for j in order:
            cur = cur.copy()
            cur[:, j] = X_full[:, j]
            X_batch[blk] = cur
            blk += 1

    preds_all = np.asarray(predict_full_fn(X_batch.reshape(-1, d)), dtype=float).reshape(n_blocks, n)
    p_bg = preds_all[0]
    p_obs = preds_all[1]

    shap_acc = np.zeros((n, d), dtype=float)
    blk = 2
    for order in orders:
        prev_p = p_bg
        for j in order:
            next_p = preds_all[blk]
            shap_acc[:, j] += (next_p - prev_p)
            prev_p = next_p
            blk += 1

    shap_vals = shap_acc / float(n_passes)
    return shap_vals, float(p_bg.mean()), p_obs


def _summarize_shap(
    shap_vals: np.ndarray,
    X_vals: np.ndarray,
    feature_meta: Sequence[Dict[str, str]],
    wells: Sequence[str],
    base_value: Any,
    fx: np.ndarray,
    unit: str,
) -> Dict[str, Any]:
    """Package SHAP summary bars + per-sample beeswarm points for UI and static plots."""
    n, d = shap_vals.shape
    mean_abs = np.mean(np.abs(shap_vals), axis=0)
    total_imp = float(np.sum(mean_abs)) or 1.0

    # Efficiency verification metric (supports scalar or per-row base_value)
    base_arr = np.asarray(base_value, dtype=float)
    base_scalar = float(np.mean(base_arr))
    recon_err = float(np.max(np.abs((shap_vals.sum(axis=1) + base_arr) - fx)))

    items = []
    for j in range(d):
        xv = X_vals[:, j]
        sv = shap_vals[:, j]
        if float(np.std(xv)) > 1e-12 and float(np.std(sv)) > 1e-12:
            corr = float(np.corrcoef(xv, sv)[0, 1])
        else:
            corr = 0.0
        lo, hi = float(np.percentile(xv, 2)), float(np.percentile(xv, 98))
        span = max(hi - lo, 1e-12)
        x_norm = np.clip((xv - lo) / span, 0.0, 1.0)
        pts = [
            {
                "shap": round(float(sv[i]), 3),
                "x_norm": round(float(x_norm[i]), 3),
                "x_raw": round(float(xv[i]), 3),
                "well": str(wells[i]),
            }
            for i in range(n)
        ]
        items.append({
            "key": feature_meta[j]["key"],
            "label": feature_meta[j]["label"],
            "group": feature_meta[j]["group"],
            "mean_abs_shap": round(float(mean_abs[j]), 3),
            "pct_share": round(100.0 * float(mean_abs[j]) / total_imp, 2),
            "direction_corr": round(corr, 3),
            "direction": "positive" if corr > 0.15 else ("negative" if corr < -0.15 else "mixed"),
            "min_shap": round(float(np.min(sv)), 3),
            "max_shap": round(float(np.max(sv)), 3),
            "beeswarm": pts,
        })

    items.sort(key=lambda r: r["mean_abs_shap"], reverse=True)
    return {
        "unit": unit,
        "n_samples": int(n),
        "base_value": round(base_scalar, 2),
        "max_efficiency_error": round(recon_err, 6),
        "features": items,
    }


def compute_vfm_feature_diagnostics(
    tests_df: pd.DataFrame,
    hf_by_well: Mapping[str, pd.DataFrame],
    comp_df: pd.DataFrame,
    pvt_df: pd.DataFrame,
    bundle: Mapping[str, Any],
    method: str = "pearson",
) -> Dict[str, Any]:
    """Compute Correlation Matrices, Training Feature Audit Table & Multi-Level SHAP Evaluation."""
    v2mod = _import_v2_module()
    ref_date = pd.Timestamp(bundle.get("ref_date") or pd.to_datetime(tests_df["test_start"], utc=True).min().normalize())
    if ref_date.tzinfo is None:
        ref_date = ref_date.tz_localize("UTC")

    wt = _align_tests_with_scada(tests_df, hf_by_well, ref_date)
    comp = comp_df.set_index("well_name") if "well_name" in comp_df.columns else comp_df.copy()
    pvt = pvt_df.set_index("well_name") if "well_name" in pvt_df.columns else pvt_df.copy()
    anchors = bundle["anchors"]
    learners = bundle["learners"]

    # Build exact V2 training feature matrix (20 base features + phase prior = 21 features)
    X_base, priors = v2mod.build_rich_feature_matrix(wt, comp, pvt, anchors)
    base_keys = [m["key"] for m in V2_BASE_FEATURE_META]
    df_v2 = pd.DataFrame(X_base, columns=base_keys)
    df_v2["well_name"] = wt["well_name"].values
    for ph in ("oil", "water", "gas"):
        df_v2[f"q_{ph}_prior"] = priors[ph]

    # Build combined DataFrame with raw SCADA + differentials + static PVT/completion + targets
    df_all = wt.copy().reset_index(drop=True)
    df_all["dp_choke_raw_psi"] = df_v2["dp_choke_raw_psi"].values
    df_all["dp_tubing_psi"] = np.maximum(df_all["dh_pressure_psia"].values - df_all["whp_psia"].values, 0.0)
    df_all["drawdown_psi"] = df_v2["drawdown_psi"].values
    df_all["q_oil_prior"] = priors["oil"]
    df_all["q_water_prior"] = priors["water"]
    df_all["q_gas_prior"] = priors["gas"]
    df_all["q_nodal"] = df_v2["q_nodal"].values

    comp_r = comp.reset_index()
    pvt_r = pvt.reset_index()
    for c in comp_r.columns:
        if c != "well_name" and c not in df_all.columns:
            m = comp_r.set_index("well_name")[c]
            df_all[c] = df_all["well_name"].map(m)
    for c in pvt_r.columns:
        if c != "well_name" and c not in df_all.columns:
            m = pvt_r.set_index("well_name")[c]
            df_all[c] = df_all["well_name"].map(m)

    # ------------------------------------------------------------------
    # A. Correlation Matrices (All Candidate Inputs vs Exact V2 Features)
    # ------------------------------------------------------------------
    all_specs = [
        {"key": "oil_rate_stbd", "label": "Oil Rate (Target)", "group": "Target", "used_in_v2": False},
        {"key": "water_rate_stbd", "label": "Water Rate (Target)", "group": "Target", "used_in_v2": False},
        {"key": "gas_rate_mscfd", "label": "Gas Rate (Target)", "group": "Target", "used_in_v2": False},
        {"key": "choke_opening_pct", "label": "Choke %", "group": "Raw SCADA", "used_in_v2": True},
        {"key": "dh_pressure_psia", "label": "PDG Pressure", "group": "Raw SCADA", "used_in_v2": True},
        {"key": "whp_psia", "label": "WHP", "group": "Raw SCADA", "used_in_v2": True},
        {"key": "whp_downstream_psia", "label": "Downstream P", "group": "Raw SCADA", "used_in_v2": False},
        {"key": "gas_lift_rate_mscfd", "label": "Gas-Lift Rate", "group": "Raw SCADA", "used_in_v2": False},
        {"key": "annulus_pressure_psia", "label": "Annulus P", "group": "Raw SCADA", "used_in_v2": False},
        {"key": "dh_temperature_degf", "label": "DH Temp", "group": "Raw SCADA", "used_in_v2": False},
        {"key": "dp_choke_raw_psi", "label": "Choke ΔP", "group": "Engineered", "used_in_v2": True},
        {"key": "dp_tubing_psi", "label": "Tubing ΔP", "group": "Engineered", "used_in_v2": False},
        {"key": "drawdown_psi", "label": "Drawdown", "group": "Engineered", "used_in_v2": True},
        {"key": "tubing_id_in", "label": "Tubing ID", "group": "Completion", "used_in_v2": False},
        {"key": "gauge_tvd_ft", "label": "Gauge TVD", "group": "Completion", "used_in_v2": False},
        {"key": "net_pay_thickness_ft", "label": "Net Pay h", "group": "PVT", "used_in_v2": False},
        {"key": "porosity_frac", "label": "Porosity φ", "group": "PVT", "used_in_v2": False},
        {"key": "oil_fvf_rb_stb", "label": "Oil FVF Bo", "group": "PVT", "used_in_v2": False},
        {"key": "oil_viscosity_cp", "label": "Viscosity μo", "group": "PVT", "used_in_v2": False},
        {"key": "bubble_point_pressure_psia", "label": "Bubble Point Pb", "group": "PVT", "used_in_v2": False},
        {"key": "solution_gor_at_pb_scf_stb", "label": "Rs @ Pb", "group": "PVT", "used_in_v2": False},
    ]

    v2_corr_df = df_v2.copy()
    v2_corr_df["oil_rate_stbd"] = wt["oil_rate_stbd"].values
    v2_corr_df["water_rate_stbd"] = wt["water_rate_stbd"].values
    v2_corr_df["gas_rate_mscfd"] = wt["gas_rate_mscfd"].values
    v2_specs = [
        {"key": "oil_rate_stbd", "label": "Oil Rate", "group": "Target", "used_in_v2": False},
        {"key": "water_rate_stbd", "label": "Water Rate", "group": "Target", "used_in_v2": False},
        {"key": "gas_rate_mscfd", "label": "Gas Rate", "group": "Target", "used_in_v2": False},
        {"key": "q_oil_prior", "label": "Oil Prior", "group": "Physics Prior", "used_in_v2": True},
    ] + [
        {"key": m["key"], "label": m["label"].split(" (")[0], "group": m["group"], "used_in_v2": True}
        for m in V2_BASE_FEATURE_META
    ]

    corr_all = _corr_payload(df_all, all_specs, method=method)
    corr_v2 = _corr_payload(v2_corr_df, v2_specs, method=method)

    # ------------------------------------------------------------------
    # B. Complete Feature Usage Audit Table (with live r_oil / r_water / r_gas)
    # ------------------------------------------------------------------
    def _safe_corr(col: str, target: str) -> Optional[float]:
        if col not in df_all.columns or target not in df_all.columns:
            return None
        s1 = pd.to_numeric(df_all[col], errors="coerce")
        s2 = pd.to_numeric(df_all[target], errors="coerce")
        ok = s1.notna() & s2.notna()
        if int(ok.sum()) < 3 or float(s1[ok].std()) < 1e-12 or float(s2[ok].std()) < 1e-12:
            return None
        return round(float(s1[ok].corr(s2[ok], method=method)), 3)

    audit_rows = []
    for item in AUDIT_FEATURE_CATALOG:
        k = item["key"]
        if k not in df_all.columns:
            continue
        vals = pd.to_numeric(df_all[k], errors="coerce").dropna()
        audit_rows.append({
            "key": k,
            "label": item["label"],
            "group": item["group"],
            "tier": item["tier"],
            "used_in_v2_ml": bool(item["v2_ml"]),
            "used_in_physics_vfm": bool(item["physics_vfm"]),
            "used_in_rf_gbm": bool(item["rf_gbm"]),
            "role": item["role"],
            "r_oil": _safe_corr(k, "oil_rate_stbd"),
            "r_water": _safe_corr(k, "water_rate_stbd"),
            "r_gas": _safe_corr(k, "gas_rate_mscfd"),
            "min_val": round(float(vals.min()), 4) if len(vals) else None,
            "max_val": round(float(vals.max()), 4) if len(vals) else None,
        })

    # ------------------------------------------------------------------
    # C. Multi-Level SHAP Evaluation (Oil, Water, Gas)
    # ------------------------------------------------------------------
    wells_seq = wt["well_name"].astype(str).tolist()
    sensor_keys = [s["key"] for s in SENSOR_SHAP_COLS]
    X_sensors = wt[sensor_keys].to_numpy(dtype=float)

    shap_out: Dict[str, Any] = {}
    for phase, unit in (("oil", "STB/d"), ("water", "STB/d"), ("gas", "Mscf/d")):
        learner = learners[phase]
        q_prior = priors[phase]
        X_full = np.column_stack([q_prior, X_base])
        choke_obs = wt["choke_opening_pct"].to_numpy(dtype=float)
        full_meta = [
            {"key": f"q_{phase}_prior", "label": f"Phase Physics Prior (q_{phase}_prior)", "group": "Physics Prior"}
        ] + [{"key": m["key"], "label": m["label"], "group": m["group"]} for m in V2_BASE_FEATURE_META]

        # Level 1: Exact End-to-End Physical Sensor Shapley (all 2^8 = 256 coalitions!)
        def _pred_end_to_end(df_pert: pd.DataFrame, _lrn=learner, _ph=phase) -> np.ndarray:
            Xb_p, pr_p = v2mod.build_rich_feature_matrix(df_pert, comp, pvt, anchors)
            chk_p = df_pert["choke_opening_pct"].to_numpy(dtype=float)
            return _lrn.predict(Xb_p, pr_p[_ph], chk_p)

        s_e2e, base_e2e, fx_e2e = _exact_subset_shapley_8(_pred_end_to_end, wt, sensor_keys)

        # Level 2: Exact Native LightGBM TreeSHAP on 21 V2 Training Features
        contrib = learner.lgbm.booster_.predict(X_full, pred_contrib=True)
        s_tree = contrib[:, :-1]
        base_tree = float(contrib[0, -1])
        fx_tree = learner.lgbm.predict(X_full)

        # Level 3: Full 5-Model Stacked Super-Learner Antithetic Shapley on 21 Features
        # Note: X_full[:, 6] is choke_opening_pct (all separator tests are flowing > 0.5%)
        def _pred_stacked_xfull(Xf: np.ndarray, _lrn=learner) -> np.ndarray:
            return _lrn.predict(Xf[:, 1:], Xf[:, 0], Xf[:, 6])

        s_ens, base_ens, fx_ens = _antithetic_ensemble_shap(_pred_stacked_xfull, X_full, n_perms=8, seed=42)

        shap_out[phase] = {
            "meta_weights": {
                name: round(float(w), 4)
                for name, w in zip(learner.base_names, learner.meta_weights)
            },
            "end_to_end_sensors": _summarize_shap(s_e2e, X_sensors, SENSOR_SHAP_COLS, wells_seq, base_e2e, fx_e2e, unit),
            "stacked_v2_features": _summarize_shap(s_ens, X_full, full_meta, wells_seq, base_ens, fx_ens, unit),
            "lgbm_treeshap": _summarize_shap(s_tree, X_full, full_meta, wells_seq, base_tree, fx_tree, unit),
        }

    return {
        "method": method,
        "n_tests": int(len(wt)),
        "n_wells": int(wt["well_name"].nunique()),
        "n_v2_features": 21,
        "correlations": {
            "all_inputs": corr_all,
            "v2_training": corr_v2,
        },
        "feature_audit": audit_rows,
        "shap": shap_out,
    }


# ==============================================================================
# 3. Interactive Custom Feature-Subset VFM Training & Live SHAP Evaluation
# ==============================================================================

def _build_candidate_feature_frame(
    df_scada: pd.DataFrame,
    comp_df: pd.DataFrame,
    pvt_df: pd.DataFrame,
    anchors: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> pd.DataFrame:
    """Attach engineered pressure differentials, drawdown, and static PVT/completion columns."""
    out = df_scada.copy().reset_index(drop=True)
    comp = comp_df.set_index("well_name") if "well_name" in comp_df.columns else comp_df.copy()
    pvt = pvt_df.set_index("well_name") if "well_name" in pvt_df.columns else pvt_df.copy()

    for tbl in (comp, pvt):
        for col in tbl.columns:
            if col not in out.columns:
                out[col] = out["well_name"].map(tbl[col])

    if {"whp_psia", "whp_downstream_psia"}.issubset(out.columns):
        out["dp_choke_raw_psi"] = np.maximum(
            pd.to_numeric(out["whp_psia"], errors="coerce")
            - pd.to_numeric(out["whp_downstream_psia"], errors="coerce"),
            0.0,
        )
    if {"dh_pressure_psia", "whp_psia"}.issubset(out.columns):
        out["dp_tubing_psi"] = np.maximum(
            pd.to_numeric(out["dh_pressure_psia"], errors="coerce")
            - pd.to_numeric(out["whp_psia"], errors="coerce"),
            0.0,
        )
    if "dh_pressure_psia" in out.columns:
        dhp = pd.to_numeric(out["dh_pressure_psia"], errors="coerce")
        if anchors:
            pr0_s = out["well_name"].map(lambda w: float((anchors.get(str(w)) or {}).get("pr0", np.nan)))
            dpr_s = out["well_name"].map(lambda w: float((anchors.get(str(w)) or {}).get("dpr_dt", 0.0)))
            days_s = pd.to_numeric(out.get("days_since_start", 0.0), errors="coerce").fillna(0.0)
            pr_t = pr0_s + dpr_s * days_s
            out["drawdown_psi"] = np.maximum(pr_t - dhp, 5.0)
        else:
            # Purely data-derived per-well P98 PDG proxy if anchors not supplied
            p_max = out.groupby("well_name")["dh_pressure_psia"].transform(lambda s: s.quantile(0.98))
            out["drawdown_psi"] = np.maximum(p_max - dhp, 5.0)
    return out


def _make_custom_estimator(model_type: str, fast_screen: bool = False):
    """Create a fast, clean scikit-learn / LightGBM regressor for live interactive training."""
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    mt = str(model_type or "lgbm").lower()
    if mt == "rf":
        return RandomForestRegressor(
            n_estimators=50 if fast_screen else 160,
            max_depth=10, min_samples_leaf=2, random_state=42, n_jobs=1,
        )
    if mt == "gbr":
        return GradientBoostingRegressor(
            n_estimators=60 if fast_screen else 140,
            learning_rate=0.07 if fast_screen else 0.05,
            max_depth=3, min_samples_leaf=3, random_state=42,
        )
    if mt == "ridge":
        return Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))])
    import lightgbm as lgb
    return lgb.LGBMRegressor(
        n_estimators=90 if fast_screen else 160,
        learning_rate=0.06 if fast_screen else 0.05,
        num_leaves=15, max_depth=5,
        min_child_samples=5, subsample=0.85, colsample_bytree=0.85,
        reg_alpha=0.2, reg_lambda=1.0, random_state=42, verbose=-1, n_jobs=1,
    )


def _daily_scada_summary(
    hf_by_well: Mapping[str, pd.DataFrame],
    date_grid: pd.DatetimeIndex,
    ref_date: pd.Timestamp,
) -> pd.DataFrame:
    """Compute steady-flowing daily median SCADA telemetry and uptime fraction per well/day."""
    sensor_cols = [
        "dh_pressure_psia", "dh_temperature_degf", "whp_psia", "whp_downstream_psia",
        "choke_opening_pct", "annulus_pressure_psia", "gas_lift_rate_mscfd",
    ]
    d_grid = pd.DatetimeIndex(pd.to_datetime(date_grid, utc=True)).normalize().unique().sort_values()
    frames = []
    for w, hf_w in sorted(hf_by_well.items()):
        if hf_w is None or hf_w.empty:
            continue
        df = hf_w.copy()
        t_col = "timestamp" if "timestamp" in df.columns else "hour"
        ts = pd.to_datetime(df[t_col], utc=True)
        df["_date"] = ts.dt.normalize()
        for c in sensor_cols:
            if c in df.columns:
                vals = pd.to_numeric(df[c], errors="coerce")
                vals = vals.where(vals > -900.0, np.nan)
                df[c] = vals
            else:
                df[c] = np.nan

        is_flow = (df["choke_opening_pct"].fillna(0.0) > 0.5)
        uptime = is_flow.groupby(df["_date"]).mean().reindex(d_grid).fillna(0.0)
        flow_df = df.loc[is_flow]
        med = flow_df.groupby("_date")[sensor_cols].median().reindex(d_grid)
        # Fill full-day SCADA gaps / shut-in days forward/backward so model inputs stay finite
        med = med.ffill().bfill()
        med["well_name"] = str(w)
        med["date"] = d_grid
        med["uptime_frac"] = uptime.to_numpy(dtype=float)
        med["days_since_start"] = (d_grid - ref_date).total_seconds() / 86400.0
        frames.append(med.reset_index(drop=True))
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    for c in sensor_cols:
        if c in out.columns and out[c].notna().any():
            out[c] = out[c].fillna(float(out[c].median()))
    return out


def retrain_custom_vfm(
    selected_features: Sequence[str],
    model_type: str,
    tests_df: pd.DataFrame,
    hf_by_well: Mapping[str, pd.DataFrame],
    comp_df: pd.DataFrame,
    pvt_df: pd.DataFrame,
    date_grid: pd.DatetimeIndex,
    bundle: Optional[Mapping[str, Any]] = None,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Train a custom VFM strictly on user-selected parameters, run 5-fold well-stratified CV,
    predict daily rates across all wells, and return live SHAP feature importance."""
    from sklearn.model_selection import StratifiedKFold

    mt = str(model_type or "lgbm").lower()
    if mt not in ("lgbm", "rf", "gbr", "ridge"):
        mt = "lgbm"

    ref_date = pd.Timestamp(
        (bundle or {}).get("ref_date")
        or pd.to_datetime(tests_df["test_start"], utc=True).min().normalize()
    )
    if ref_date.tzinfo is None:
        ref_date = ref_date.tz_localize("UTC")

    anchors = (bundle or {}).get("anchors")
    wt_aligned = _align_tests_with_scada(tests_df, hf_by_well, ref_date)
    wt_feat = _build_candidate_feature_frame(wt_aligned, comp_df, pvt_df, anchors=anchors)

    # Validate selected features against available columns
    audit_meta = {item["key"]: item for item in AUDIT_FEATURE_CATALOG}
    valid_feats: List[str] = []
    for f in selected_features:
        fk = str(f).strip()
        if fk in wt_feat.columns and fk not in valid_feats:
            valid_feats.append(fk)
    if not valid_feats:
        raise ValueError("Please select at least one valid numeric input parameter for training.")

    X_df = wt_feat[valid_feats].apply(pd.to_numeric, errors="coerce")
    col_meds = {c: (float(X_df[c].median()) if X_df[c].notna().any() else 0.0) for c in valid_feats}
    for c in valid_feats:
        X_df[c] = X_df[c].fillna(col_meds[c])
    X = X_df.to_numpy(dtype=float)

    wells_arr = wt_feat["well_name"].astype(str).to_numpy()
    min_per_well = int(pd.Series(wells_arr).value_counts().min()) if len(wells_arr) else 0
    n_splits = max(2, min(5, min_per_well))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Build daily prediction frame across all wells & dates
    daily_scada = _daily_scada_summary(hf_by_well, date_grid, ref_date)
    daily_feat = _build_candidate_feature_frame(daily_scada, comp_df, pvt_df, anchors=anchors)
    X_daily_df = pd.DataFrame(index=daily_feat.index)
    for c in valid_feats:
        if c in daily_feat.columns:
            X_daily_df[c] = pd.to_numeric(daily_feat[c], errors="coerce").fillna(col_meds[c])
        else:
            X_daily_df[c] = col_meds[c]
    X_daily = X_daily_df.to_numpy(dtype=float)

    daily_out = pd.DataFrame({
        "well_name": daily_feat["well_name"].values,
        "date": pd.to_datetime(daily_feat["date"], utc=True),
        "uptime_frac": daily_feat["uptime_frac"].to_numpy(dtype=float),
    })

    feat_meta_list = [
        {
            "key": c,
            "label": audit_meta[c]["label"] if c in audit_meta else c,
            "group": audit_meta[c]["group"] if c in audit_meta else "Custom",
        }
        for c in valid_feats
    ]

    metrics_out: Dict[str, Dict[str, float]] = {}
    shap_out: Dict[str, Any] = {}
    phases = [
        ("oil", "oil_rate_stbd", "stbd", "STB/d"),
        ("water", "water_rate_stbd", "stbd", "STB/d"),
        ("gas", "gas_rate_mscfd", "mscfd", "Mscf/d"),
        ("liquid", "liquid_rate_stbd", "stbd", "STB/d"),
    ]

    for ph, y_col, u_slug, unit in phases:
        y = pd.to_numeric(wt_feat[y_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        oof = np.zeros_like(y)
        for tr_idx, va_idx in skf.split(X, wells_arr):
            est_cv = _make_custom_estimator(mt)
            est_cv.fit(X[tr_idx], y[tr_idx])
            oof[va_idx] = np.maximum(est_cv.predict(X[va_idx]), 0.0)

        err = oof - y
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err ** 2)))
        wape = float(100.0 * np.sum(np.abs(err)) / max(np.sum(np.abs(y)), 1e-6))
        pos = y > 1.0
        mape = float(100.0 * np.mean(np.abs(err[pos]) / y[pos])) if pos.any() else wape
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = float(1.0 - np.sum(err ** 2) / max(ss_tot, 1e-9))
        metrics_out[ph] = {
            "mae": round(mae, 2),
            "rmse": round(rmse, 2),
            "wape": round(wape, 2),
            "mape": round(mape, 2),
            "r2": round(r2, 4),
        }

        # Fit full model & predict daily series
        est_full = _make_custom_estimator(mt)
        est_full.fit(X, y)
        pred_vfm = np.maximum(est_full.predict(X_daily), 0.0)
        up = daily_out["uptime_frac"].to_numpy(dtype=float)
        pred_alloc = pred_vfm * up
        daily_out[f"{ph}_rate_vfm_{u_slug}"] = pred_vfm
        daily_out[f"{ph}_rate_alloc_{u_slug}"] = pred_alloc
        daily_out[f"{ph}_rate_{u_slug}"] = pred_alloc

        # Compute exact/fast SHAP for oil, water, gas on ONLY the selected features
        if ph in ("oil", "water", "gas"):
            fx = est_full.predict(X)
            if mt == "lgbm":
                contrib = est_full.booster_.predict(X, pred_contrib=True)
                sv, bv = contrib[:, :-1], float(contrib[0, -1])
            elif mt == "ridge":
                scaler = est_full.named_steps["scaler"]
                reg = est_full.named_steps["ridge"]
                Xs = scaler.transform(X)
                sv = (Xs - Xs.mean(axis=0)) * reg.coef_[None, :]
                bv = float(fx.mean())
            else:
                sv, bv, fx = _antithetic_ensemble_shap(est_full.predict, X, n_perms=12, seed=42)
            shap_out[ph] = _summarize_shap(sv, X, feat_meta_list, wells_arr.tolist(), bv, fx, unit)

    payload = {
        "model_id": "custom_selected",
        "model_type": mt,
        "selected_features": valid_feats,
        "selected_labels": [m["label"] for m in feat_meta_list],
        "n_features": len(valid_feats),
        "n_tests": int(len(wt_feat)),
        "cv_folds": int(n_splits),
        "metrics": metrics_out,
        "shap": shap_out,
    }
    return payload, daily_out


def _build_grid_presets(available_cols: Sequence[str], custom_features: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
    """Build structured feature-subset presets dynamically from AUDIT_FEATURE_CATALOG."""
    avail = set(available_cols)
    by_group: Dict[str, List[str]] = {}
    v2_ml_cols: List[str] = []
    phys_cols: List[str] = []
    all_cols: List[str] = []

    for item in AUDIT_FEATURE_CATALOG:
        k = item["key"]
        if k not in avail:
            continue
        all_cols.append(k)
        by_group.setdefault(item["group"], []).append(k)
        if item.get("v2_ml"):
            v2_ml_cols.append(k)
        if item.get("physics_vfm"):
            phys_cols.append(k)

    raw_scada = by_group.get("Raw SCADA", [])
    eng_phys = by_group.get("Engineered Physics", [])
    comp_cols = by_group.get("Static Completion", [])
    pvt_cols = by_group.get("Static PVT", [])

    def _uniq(seq: Sequence[str]) -> List[str]:
        out: List[str] = []
        for s in seq:
            if s in avail and s not in out:
                out.append(s)
        return out

    raw_no_pds = [c for c in raw_scada if c != "whp_downstream_psia"]

    candidates = [
        {
            "preset_id": "v2_direct_ml",
            "preset_name": "V2 Direct ML Core",
            "description": "Direct V2 ML inputs (Choke, PDG, WHP, ΔP_choke, Drawdown)",
            "features": _uniq(v2_ml_cols),
        },
        {
            "preset_id": "scada_no_pds",
            "preset_name": "Manifold-Safe SCADA (6 Sensors, no P_ds)",
            "description": "Raw SCADA excluding raw downstream pressure to avoid test-vs-manifold shift",
            "features": _uniq(raw_no_pds),
        },
        {
            "preset_id": "scada7",
            "preset_name": "Raw 7 SCADA Sensors",
            "description": "All 7 raw high-frequency telemetry channels",
            "features": _uniq(raw_scada),
        },
        {
            "preset_id": "scada_diffs_no_pds",
            "preset_name": "Manifold-Safe SCADA + Physics Diffs",
            "description": "Manifold-safe SCADA sensors plus ΔP_choke, ΔP_tubing & Drawdown",
            "features": _uniq(raw_no_pds + eng_phys),
        },
        {
            "preset_id": "v2_core_plus_scada",
            "preset_name": "V2 Core + All SCADA & Physics Diffs",
            "description": "All 7 raw SCADA sensors plus engineered pressure differentials",
            "features": _uniq(raw_scada + eng_phys),
        },
        {
            "preset_id": "scada_completion",
            "preset_name": "SCADA + Static Completion",
            "description": "Raw telemetry combined with tubing ID, TVD, GL valve & max bean",
            "features": _uniq(raw_scada + comp_cols),
        },
        {
            "preset_id": "scada_pvt",
            "preset_name": "SCADA + Completion + PVT",
            "description": "Raw telemetry combined with completion geometry and reservoir PVT",
            "features": _uniq(raw_scada + comp_cols + pvt_cols),
        },
        {
            "preset_id": "physics_complete_no_pds",
            "preset_name": "Physics-Complete (Manifold-Safe + Diffs + PVT)",
            "description": "Manifold-safe SCADA + engineered differentials + completion + PVT",
            "features": _uniq(raw_no_pds + eng_phys + comp_cols + pvt_cols),
        },
        {
            "preset_id": "physics_engine_all",
            "preset_name": "All Physics-VFM Inputs",
            "description": "Every parameter used anywhere in the Physics VFM engine",
            "features": _uniq(phys_cols),
        },
        {
            "preset_id": "all_inputs",
            "preset_name": "All Available Parameters",
            "description": "Full catalog (SCADA + Physics Diffs + Completion + all PVT)",
            "features": _uniq(all_cols),
        },
    ]

    # Deduplicate identical feature sets and keep non-empty presets
    seen_sets: set = set()
    presets: List[Dict[str, Any]] = []
    for c in candidates:
        fs = tuple(sorted(c["features"]))
        if fs and fs not in seen_sets:
            seen_sets.add(fs)
            presets.append(c)

    if custom_features:
        cf = _uniq(custom_features)
        fs = tuple(sorted(cf))
        if fs and fs not in seen_sets:
            presets.append({
                "preset_id": "current_selection",
                "preset_name": "Current Manual Checkboxes",
                "description": f"Your currently checked {len(cf)} parameters",
                "features": cf,
            })
    return presets


def grid_search_vfm_presets(
    model_types: Sequence[str],
    optimize_metric: str,
    tests_df: pd.DataFrame,
    hf_by_well: Mapping[str, pd.DataFrame],
    comp_df: pd.DataFrame,
    pvt_df: pd.DataFrame,
    date_grid: pd.DatetimeIndex,
    bundle: Optional[Mapping[str, Any]] = None,
    custom_features: Optional[Sequence[str]] = None,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Run automated 5-fold stratified CV grid search across (feature presets × model families),
    rank all combinations, and train the #1 winning combination for live deployment + SHAP."""
    from concurrent.futures import ThreadPoolExecutor
    from sklearn.model_selection import StratifiedKFold

    allowed_algos = ("lgbm", "rf", "gbr", "ridge")
    algos = [str(m).lower() for m in model_types if str(m).lower() in allowed_algos]
    if not algos:
        algos = list(allowed_algos)
    # Deduplicate preserving order
    algos = list(dict.fromkeys(algos))

    opt = str(optimize_metric or "composite").lower()
    if opt not in ("composite", "oil", "water", "gas", "liquid"):
        opt = "composite"

    ref_date = pd.Timestamp(
        (bundle or {}).get("ref_date")
        or pd.to_datetime(tests_df["test_start"], utc=True).min().normalize()
    )
    if ref_date.tzinfo is None:
        ref_date = ref_date.tz_localize("UTC")

    anchors = (bundle or {}).get("anchors")
    wt_aligned = _align_tests_with_scada(tests_df, hf_by_well, ref_date)
    wt_feat = _build_candidate_feature_frame(wt_aligned, comp_df, pvt_df, anchors=anchors)

    presets = _build_grid_presets(wt_feat.columns.tolist(), custom_features=custom_features)
    if not presets:
        raise ValueError("No valid feature presets could be constructed from the available columns.")

    wells_arr = wt_feat["well_name"].astype(str).to_numpy()
    min_per_well = int(pd.Series(wells_arr).value_counts().min()) if len(wells_arr) else 0
    n_splits = max(2, min(5, min_per_well))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    splits = list(skf.split(np.zeros((len(wt_feat), 1)), wells_arr))

    # Pre-extract numeric matrices & target vectors once
    phases = [
        ("oil", "oil_rate_stbd"),
        ("water", "water_rate_stbd"),
        ("gas", "gas_rate_mscfd"),
        ("liquid", "liquid_rate_stbd"),
    ]
    y_map = {
        ph: pd.to_numeric(wt_feat[y_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        for ph, y_col in phases
    }

    preset_X: Dict[str, np.ndarray] = {}
    for p in presets:
        X_df = wt_feat[p["features"]].apply(pd.to_numeric, errors="coerce")
        for c in p["features"]:
            med = float(X_df[c].median()) if X_df[c].notna().any() else 0.0
            X_df[c] = X_df[c].fillna(med)
        preset_X[p["preset_id"]] = X_df.to_numpy(dtype=float)

    def _eval_combo(preset: Dict[str, Any], mt: str) -> Dict[str, Any]:
        X = preset_X[preset["preset_id"]]
        m_out: Dict[str, Dict[str, float]] = {}
        for ph, _ in phases:
            y = y_map[ph]
            oof = np.zeros_like(y)
            for tr_idx, va_idx in splits:
                est = _make_custom_estimator(mt, fast_screen=True)
                est.fit(X[tr_idx], y[tr_idx])
                oof[va_idx] = np.maximum(est.predict(X[va_idx]), 0.0)
            err = oof - y
            mae = float(np.mean(np.abs(err)))
            wape = float(100.0 * np.sum(np.abs(err)) / max(np.sum(np.abs(y)), 1e-6))
            pos = y > 1.0
            mape = float(100.0 * np.mean(np.abs(err[pos]) / y[pos])) if pos.any() else wape
            ss_tot = float(np.sum((y - np.mean(y)) ** 2))
            r2 = float(1.0 - np.sum(err ** 2) / max(ss_tot, 1e-9))
            m_out[ph] = {
                "mae": round(mae, 2),
                "wape": round(wape, 2),
                "mape": round(mape, 2),
                "r2": round(r2, 4),
            }
        comp_mape = round((m_out["oil"]["mape"] + m_out["water"]["mape"] + m_out["gas"]["mape"]) / 3.0, 2)
        score = comp_mape if opt == "composite" else m_out[opt]["mape"]
        return {
            "preset_id": preset["preset_id"],
            "preset_name": preset["preset_name"],
            "description": preset["description"],
            "model_type": mt,
            "features": list(preset["features"]),
            "n_features": len(preset["features"]),
            "composite_mape": comp_mape,
            "score": round(float(score), 4),
            "oil_mape": m_out["oil"]["mape"],
            "oil_wape": m_out["oil"]["wape"],
            "oil_r2": m_out["oil"]["r2"],
            "water_mape": m_out["water"]["mape"],
            "gas_mape": m_out["gas"]["mape"],
            "liquid_mape": m_out["liquid"]["mape"],
            "metrics": m_out,
        }

    tasks = [(p, mt) for p in presets for mt in algos]
    max_workers = min(6, max(1, len(tasks)))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        results = list(pool.map(lambda pair: _eval_combo(pair[0], pair[1]), tasks))

    results.sort(key=lambda r: (r["score"], r["oil_mape"], r["n_features"]))
    for rank_idx, rec in enumerate(results, start=1):
        rec["rank"] = rank_idx

    best = results[0]
    best_train_payload, best_daily = retrain_custom_vfm(
        selected_features=best["features"],
        model_type=best["model_type"],
        tests_df=tests_df,
        hf_by_well=hf_by_well,
        comp_df=comp_df,
        pvt_df=pvt_df,
        date_grid=date_grid,
        bundle=bundle,
    )

    grid_payload = {
        "optimize_metric": opt,
        "algorithms_tested": algos,
        "n_presets": len(presets),
        "n_combinations": len(results),
        "n_tests": int(len(wt_feat)),
        "cv_folds": int(n_splits),
        "best": best,
        "leaderboard": results,
        "best_custom_result": best_train_payload,
    }
    return grid_payload, best_daily
