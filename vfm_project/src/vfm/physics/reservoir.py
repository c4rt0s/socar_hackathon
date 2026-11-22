"""Reservoir engineering, Pressure Build-Up (PBU), and IPR physics module."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


ArrayLike = Union[float, int, np.ndarray, pd.Series]

# Reference epoch for time-linear reservoir pressure depletion Pr(t) = Pr0 - a * t_days
DEFAULT_REF_TIMESTAMP = pd.Timestamp("2025-01-01 00:00:00+00:00")

# Radial-flow acceptance for the Bourdet window (heuristics, not physical laws):
# derivative scatter (std/median), |d log10(deriv) / d log10(dt)|, and window extent.
RADIAL_MAX_CV = 0.10
RADIAL_MAX_TREND_PER_CYCLE = 0.08
RADIAL_MIN_CYCLES = 0.35

# Nominal Pr / datum-TVD gradient, used only when no build-up is available anywhere. It is
# not data-based: in this field the measured build-up wells sit at 0.375-0.402 psi/ft.
DEFAULT_PR_GRADIENT_PRIOR_PSI_FT = 0.41


def _to_numpy(x: ArrayLike) -> Tuple[np.ndarray, bool]:
    """Convert scalar or array-like to float64 ndarray and track scalar input."""
    is_scalar = np.ndim(x) == 0
    arr = np.asarray(x, dtype="float64")
    return arr, is_scalar


def compute_wellbore_fluid_gradient(
    api: ArrayLike,
    wc_frac: ArrayLike,
    gor_scf_stb: ArrayLike = 0.0,
    gas_sg: ArrayLike = 0.7,
    bo: ArrayLike = 1.14,
    bw: ArrayLike = 1.02,
    salinity_ppm: ArrayLike = 100_000.0,
) -> Union[float, np.ndarray]:
    """
    Compute bottomhole liquid mixture hydrostatic pressure gradient (psi/ft).

    At downhole gauge/datum pressures above bubble point, solution gas is
    dissolved in the oil phase. Oil specific gravity is computed from API
    gravity, brine specific gravity from dissolved salinity, and in-situ
    densities are weighted by surface water cut (converted via FVF or direct
    volume fraction proxy).
    """
    api_arr, s_api = _to_numpy(api)
    wc_arr, s_wc = _to_numpy(wc_frac)
    gor_arr, s_gor = _to_numpy(gor_scf_stb)
    gsg_arr, s_gsg = _to_numpy(gas_sg)
    bo_arr, s_bo = _to_numpy(bo)
    bw_arr, s_bw = _to_numpy(bw)
    sal_arr, s_sal = _to_numpy(salinity_ppm)

    wc_clipped = np.clip(wc_arr, 0.0, 1.0)
    gamma_o = 141.5 / np.clip(131.5 + api_arr, 10.0, None)
    gamma_w = np.clip(1.0 + 0.7e-6 * np.clip(sal_arr, 0.0, 350_000.0), 1.0, 1.25)

    # 1 stb = 5.61458 ft3; 1 scf air = 0.0763 lb -> 0.0763 / 5.61458 = 0.01359 lb/ft3 per scf/stb
    rho_o_lb_ft3 = (62.428 * gamma_o + 0.01359 * np.clip(gor_arr, 0.0, None) * gsg_arr) / np.clip(bo_arr, 0.5, 5.0)
    rho_w_lb_ft3 = (62.428 * gamma_w) / np.clip(bw_arr, 0.8, 2.0)

    rho_liq_lb_ft3 = (1.0 - wc_clipped) * rho_o_lb_ft3 + wc_clipped * rho_w_lb_ft3
    grad_psi_ft = rho_liq_lb_ft3 / 144.0

    all_scalar = s_api and s_wc and s_gor and s_gsg and s_bo and s_bw and s_sal
    return float(grad_psi_ft) if all_scalar else grad_psi_ft


def gauge_to_datum_pressure(
    p_gauge_psia: ArrayLike,
    gauge_tvd_ft: ArrayLike,
    datum_tvd_ft: ArrayLike,
    fluid_grad_psi_ft: ArrayLike,
) -> Union[float, np.ndarray]:
    """
    Correct downhole gauge pressure from gauge TVD to reservoir datum TVD.

    P_datum = P_gauge + fluid_grad_psi_ft * (datum_tvd_ft - gauge_tvd_ft)
    """
    p_arr, s_p = _to_numpy(p_gauge_psia)
    g_arr, s_g = _to_numpy(gauge_tvd_ft)
    d_arr, s_d = _to_numpy(datum_tvd_ft)
    gr_arr, s_gr = _to_numpy(fluid_grad_psi_ft)

    p_datum = p_arr + gr_arr * (d_arr - g_arr)
    all_scalar = s_p and s_g and s_d and s_gr
    return float(p_datum) if all_scalar else p_datum


def ipr_rate(
    J: ArrayLike,
    Pr: ArrayLike,
    Pwf: ArrayLike,
    Pb: Optional[ArrayLike] = None,
) -> Union[float, np.ndarray]:
    """
    Compute liquid inflow rate (stb/d) via undersaturated linear PI above Pb
    with C1-continuous composite Vogel transition below Pb.

    For Pr > Pb:
      - Pwf >= Pb: q = J * max(Pr - Pwf, 0)
      - Pwf <  Pb: q = qb + qmax_v * [1 - 0.2*(Pwf/Pb) - 0.8*(Pwf/Pb)^2]
        where qb = J*(Pr - Pb) and qmax_v = J*Pb / 1.8 ensures exact continuity
        of both q(Pb) and dq/dPwf(Pb) = -J.

    For Pr <= Pb (saturated reservoir):
      - q = (J*Pr / 1.8) * [1 - 0.2*(Pwf/Pr) - 0.8*(Pwf/Pr)^2] for Pwf < Pr, else 0.
    """
    j_arr, s_j = _to_numpy(J)
    pr_arr, s_pr = _to_numpy(Pr)
    pwf_arr, s_pwf = _to_numpy(Pwf)
    is_scalar = s_j and s_pr and s_pwf and (Pb is None or np.ndim(Pb) == 0)

    if Pb is None:
        q = np.where(
            np.isfinite(pwf_arr) & np.isfinite(pr_arr) & np.isfinite(j_arr),
            np.maximum(j_arr, 0.0) * np.maximum(pr_arr - pwf_arr, 0.0),
            np.nan,
        )
        return float(q) if is_scalar else q

    pb_arr, _ = _to_numpy(Pb)
    j_pos = np.maximum(j_arr, 0.0)
    pwf_pos = np.maximum(pwf_arr, 0.0)
    valid = np.isfinite(pwf_arr) & np.isfinite(pr_arr) & np.isfinite(j_arr) & np.isfinite(pb_arr)

    # Broadcast shapes cleanly
    j_b, pr_b, pwf_b, pb_b, valid_b = np.broadcast_arrays(j_pos, pr_arr, pwf_pos, pb_arr, valid)
    q = np.full(pwf_b.shape, np.nan, dtype="float64")

    # Above or equal to Pr -> zero inflow
    zero_mask = valid_b & (pwf_b >= pr_b)
    q[zero_mask] = 0.0

    # Undersaturated reservoir (Pr > Pb)
    lin_mask = valid_b & (pwf_b < pr_b) & (pr_b > pb_b) & (pwf_b >= pb_b)
    q[lin_mask] = j_b[lin_mask] * (pr_b[lin_mask] - pwf_b[lin_mask])

    pb_safe = np.where(pb_b > 1e-6, pb_b, 1.0)
    comp_vogel_mask = valid_b & (pwf_b < pr_b) & (pr_b > pb_b) & (pwf_b < pb_b)
    if np.any(comp_vogel_mask):
        qb = j_b[comp_vogel_mask] * (pr_b[comp_vogel_mask] - pb_b[comp_vogel_mask])
        qmax_v = (j_b[comp_vogel_mask] * pb_b[comp_vogel_mask]) / 1.8
        r = pwf_b[comp_vogel_mask] / pb_safe[comp_vogel_mask]
        q[comp_vogel_mask] = qb + qmax_v * (1.0 - 0.2 * r - 0.8 * (r ** 2))

    # Saturated reservoir (Pr <= Pb)
    sat_mask = valid_b & (pwf_b < pr_b) & (pr_b <= pb_b)
    if np.any(sat_mask):
        pr_safe = np.where(pr_b > 1e-6, pr_b, 1.0)
        qmax_s = (j_b[sat_mask] * pr_b[sat_mask]) / 1.8
        r_s = pwf_b[sat_mask] / pr_safe[sat_mask]
        q[sat_mask] = qmax_s * (1.0 - 0.2 * r_s - 0.8 * (r_s ** 2))

    return float(q.item()) if is_scalar else q


def compute_bourdet_derivative(
    dt_hr: np.ndarray,
    dp_psi: np.ndarray,
    smoothing_l: float = 0.20,
) -> np.ndarray:
    """
    Compute the Bourdet logarithmic pressure derivative d(dp)/d(ln dt).

    Uses symmetric log-time bracketing with minimum separation L in ln(dt)
    to suppress 1-minute gauge noise while preserving storage and radial trends.
    """
    n = len(dt_hr)
    deriv = np.full(n, np.nan, dtype="float64")
    if n < 5:
        return deriv

    valid = np.isfinite(dt_hr) & (dt_hr > 0.0) & np.isfinite(dp_psi)
    if np.sum(valid) < 5:
        return deriv

    x = np.log(dt_hr[valid])
    y = dp_psi[valid]
    m = len(x)
    d_valid = np.full(m, np.nan, dtype="float64")

    for i in range(m):
        left_idx = np.where(x[:i] <= x[i] - smoothing_l)[0]
        right_idx = np.where(x[i + 1 :] >= x[i] + smoothing_l)[0]
        if len(left_idx) > 0 and len(right_idx) > 0:
            il = int(left_idx[-1])
            ir = int(i + 1 + right_idx[0])
            dx1 = x[i] - x[il]
            dx2 = x[ir] - x[i]
            dp1 = y[i] - y[il]
            dp2 = y[ir] - y[i]
            d_valid[i] = ((dp1 / dx1) * dx2 + (dp2 / dx2) * dx1) / (dx1 + dx2)

    deriv[valid] = d_valid
    return deriv


def _extract_row_dict(row: Union[pd.Series, Mapping[str, Any]]) -> Dict[str, Any]:
    if isinstance(row, pd.Series):
        return row.to_dict()
    return dict(row)


def estimate_pr_from_buildup(
    hf_well_df: pd.DataFrame,
    shut_in_start: pd.Timestamp,
    shut_in_end: pd.Timestamp,
    prior_rate_stbd: float,
    pvt_row: Union[pd.Series, Mapping[str, Any]],
    comp_row: Union[pd.Series, Mapping[str, Any]],
    fluid_grad_psi_ft: Optional[float] = None,
    wc_frac: float = 0.0,
    oil_api: float = 32.0,
    gas_sg: float = 0.75,
    salinity_ppm: float = 100_000.0,
    producing_time_hr: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Analyze a shut-in pressure build-up transient using Bourdet derivative and
    Horner / Matthews-Brons-Hazebroek (MBH) analysis.

    Returns a dictionary containing gauge and datum average reservoir pressure
    (`pr_gauge_psia`, `pr_datum_psia`), Horner extrapolated `p_star_gauge_psia`,
    final stabilized `p_end_gauge_psia`, diagnostic `kh_md_ft`, `k_md`, `skin`,
    `t_pss_hr`, and `quality_flag`.
    """
    ts0 = pd.Timestamp(shut_in_start)
    ts1 = pd.Timestamp(shut_in_end)
    pvt_d = _extract_row_dict(pvt_row)
    comp_d = _extract_row_dict(comp_row)

    df = hf_well_df if hf_well_df["timestamp"].is_monotonic_increasing else hf_well_df.sort_values("timestamp")
    pre = df[(df["timestamp"] >= ts0 - pd.Timedelta("30min")) & (df["timestamp"] <= ts0)]
    si = df[(df["timestamp"] > ts0) & (df["timestamp"] <= ts1)]

    if si.empty or len(si) < 15:
        return {
            "shut_in_start": ts0,
            "shut_in_end": ts1,
            "duration_hr": 0.0,
            "pr_gauge_psia": np.nan,
            "pr_datum_psia": np.nan,
            "p_star_gauge_psia": np.nan,
            "p_end_gauge_psia": np.nan,
            "p_wf0_gauge_psia": np.nan,
            "dp_radial_psi": np.nan,
            "kh_md_ft": np.nan,
            "k_md": np.nan,
            "skin": np.nan,
            "t_pss_hr": np.nan,
            "quality_flag": "INSUFFICIENT_DATA",
        }

    dt_hr = (si["timestamp"] - ts0).dt.total_seconds().to_numpy(dtype="float64") / 3600.0
    p_si = si["dh_pressure_psia"].to_numpy(dtype="float64")
    valid = np.isfinite(dt_hr) & (dt_hr > 0.0) & np.isfinite(p_si)
    dt_hr, p_si = dt_hr[valid], p_si[valid]

    if len(dt_hr) < 15:
        return {
            "shut_in_start": ts0,
            "shut_in_end": ts1,
            "duration_hr": 0.0,
            "pr_gauge_psia": np.nan,
            "pr_datum_psia": np.nan,
            "p_star_gauge_psia": np.nan,
            "p_end_gauge_psia": np.nan,
            "p_wf0_gauge_psia": np.nan,
            "dp_radial_psi": np.nan,
            "kh_md_ft": np.nan,
            "k_md": np.nan,
            "skin": np.nan,
            "t_pss_hr": np.nan,
            "quality_flag": "INSUFFICIENT_VALID_PRESSURE",
        }

    pre_valid = pre["dh_pressure_psia"].dropna()
    pwf0 = float(pre_valid.median()) if len(pre_valid) > 0 else float(p_si[0])
    dp = p_si - pwf0
    dur_hr = float(dt_hr[-1])
    p_end = float(np.median(p_si[dt_hr >= max(0.0, dur_hr - 0.5)]))

    # Fluid gradient for gauge -> datum correction
    if fluid_grad_psi_ft is None:
        fluid_grad = float(
            compute_wellbore_fluid_gradient(
                api=oil_api,
                wc_frac=wc_frac,
                gor_scf_stb=float(comp_d.get("solution_gor_at_pb_scf_stb", 0.0)),
                gas_sg=gas_sg,
                bo=float(pvt_d["oil_fvf_rb_stb"]),
                bw=float(comp_d.get("water_fvf_rb_stb", 1.02)),
                salinity_ppm=salinity_ppm,
            )
        )
    else:
        fluid_grad = float(fluid_grad_psi_ft)

    gauge_tvd = float(comp_d["gauge_tvd_ft"])
    datum_tvd = float(pvt_d["datum_tvd_ft"])

    # Compute Bourdet derivative
    deriv = compute_bourdet_derivative(dt_hr, dp, smoothing_l=0.20)

    # Identify wellbore storage hump peak (< 2h) to avoid early-time storage distortion
    early_mask = (dt_hr <= min(2.0, 0.3 * dur_hr)) & np.isfinite(deriv)
    if np.any(early_mask):
        early_indices = np.where(early_mask)[0]
        peak_idx = int(early_indices[int(np.argmax(deriv[early_mask]))])
        t_min_rad = max(1.5 if dur_hr >= 5.0 else 0.8, 2.5 * float(dt_hr[peak_idx]))
        t_min_rad = min(t_min_rad, 0.45 * dur_hr)
    else:
        t_min_rad = min(1.5, 0.35 * dur_hr)

    cand_mask = (
        (dt_hr >= t_min_rad)
        & (dt_hr <= 0.78 * dur_hr)
        & np.isfinite(deriv)
        & (deriv > 0.0)
    )
    cand_idx = np.where(cand_mask)[0]

    # Select the flattest log-time window long enough to qualify as a radial plateau
    # (span >= RADIAL_MIN_CYCLES log10 cycles); acceptance is checked further below.
    min_span_ln = RADIAL_MIN_CYCLES * np.log(10.0)
    best_cv = np.inf
    best_win = None
    if len(cand_idx) >= 10:
        ln_t = np.log(dt_hr)
        for i in cand_idx:
            win = cand_idx[(ln_t[cand_idx] >= ln_t[i]) & (ln_t[cand_idx] <= ln_t[i] + min_span_ln + 0.25)]
            if len(win) >= 12 and (ln_t[win[-1]] - ln_t[win[0]]) >= min_span_ln:
                d_win = deriv[win]
                med_d = float(np.median(d_win))
                if med_d > 0.0:
                    cv = float(np.std(d_win) / med_d)
                    if cv < best_cv:
                        best_cv = cv
                        best_win = win

    if best_win is None or len(best_win) < 5:
        if len(cand_idx) >= 5:
            best_win = cand_idx
            best_cv = float(np.std(deriv[best_win]) / max(float(np.median(deriv[best_win])), 1e-6))
        else:
            # Very short shut-in fallback
            pr_datum = gauge_to_datum_pressure(p_end, gauge_tvd, datum_tvd, fluid_grad)
            return {
                "shut_in_start": ts0,
                "shut_in_end": ts1,
                "duration_hr": dur_hr,
                "pr_gauge_psia": p_end,
                "pr_datum_psia": pr_datum,
                "p_star_gauge_psia": p_end,
                "p_end_gauge_psia": p_end,
                "p_wf0_gauge_psia": pwf0,
                "dp_radial_psi": np.nan,
                "kh_md_ft": np.nan,
                "k_md": np.nan,
                "skin": np.nan,
                "t_pss_hr": np.nan,
                "quality_flag": "SHORT_FALLBACK",
            }

    dp_rad = float(np.median(deriv[best_win]))

    # Radial-flow credibility: the flattest window is only radial flow if it is genuinely
    # flat (low scatter), has no systematic trend, and spans enough log time. A steadily
    # falling derivative (boundary/support or still-ending storage) has a "flattest" window
    # too, so without these tests kh/skin would be reported for non-radial responses.
    lt_win = np.log10(dt_hr[best_win])
    lz_win = np.log10(np.maximum(deriv[best_win], 1e-12))
    rad_span_cycles = float(np.ptp(lt_win)) if len(best_win) > 1 else 0.0
    rad_trend = float(np.polyfit(lt_win, lz_win, 1)[0]) if rad_span_cycles > 0.0 else np.inf
    is_radial = bool(
        best_cv <= RADIAL_MAX_CV
        and abs(rad_trend) <= RADIAL_MAX_TREND_PER_CYCLE
        and rad_span_cycles >= RADIAL_MIN_CYCLES
    )

    Bo = float(pvt_d["oil_fvf_rb_stb"])
    Bw = float(comp_d.get("water_fvf_rb_stb", 1.02))
    mu_o = float(pvt_d["oil_viscosity_cp"])
    h = float(pvt_d["net_pay_thickness_ft"])
    phi = float(pvt_d["porosity_frac"])
    ct = float(pvt_d["total_compressibility_1_psi"])
    rw = float(pvt_d["wellbore_radius_ft"])
    area_ft2 = float(pvt_d["estimated_drainage_area_acres"]) * 43560.0

    # Producing time tp for the Horner plot. Callers should pass the real producing time
    # (fit_pr_and_pi uses first_production_date); fall back to first_production_date or
    # telemetry span dynamically (no hardcoded calendar epoch).
    tp_is_assumed = producing_time_hr is None or producing_time_hr <= 0.0
    if tp_is_assumed:
        fpd = comp_d.get("first_production_date")
        if fpd is not None and pd.notna(fpd):
            tp_hr = max(24.0, (ts0 - pd.Timestamp(fpd, tz="UTC")).total_seconds() / 3600.0)
        else:
            hf_ts_min = pd.to_datetime(hf_df["timestamp"], utc=True).min()
            tp_hr = max(24.0, (ts0 - hf_ts_min).total_seconds() / 3600.0)
    else:
        tp_hr = float(producing_time_hr)

    if not is_radial:
        # No credible radial flow: kh, k, skin, p* and t_pss are not defined. The last
        # shut-in pressure is a lower bound on average reservoir pressure.
        pr_datum = float(gauge_to_datum_pressure(p_end, gauge_tvd, datum_tvd, fluid_grad))
        return {
            "shut_in_start": ts0,
            "shut_in_end": ts1,
            "duration_hr": dur_hr,
            "pr_gauge_psia": p_end,
            "pr_datum_psia": pr_datum,
            "p_star_gauge_psia": np.nan,
            "p_end_gauge_psia": p_end,
            "p_wf0_gauge_psia": pwf0,
            "dp_radial_psi": np.nan,
            "horner_slope_psi_cycle": np.nan,
            "kh_md_ft": np.nan,
            "k_md": np.nan,
            "skin": np.nan,
            "t_pss_hr": np.nan,
            "rad_cv": float(best_cv),
            "rad_trend_per_cycle": rad_trend,
            "rad_span_cycles": rad_span_cycles,
            "producing_time_hr": tp_hr,
            "producing_time_assumed": tp_is_assumed,
            "quality_flag": "NO_RADIAL_P_END",
        }

    horner_x = np.log10((tp_hr + dt_hr[best_win]) / dt_hr[best_win])
    slope_h, p_star = np.polyfit(horner_x, p_si[best_win], 1)
    m_horner = abs(float(slope_h))
    p_star_val = float(p_star)
    p_1hr = p_star_val - m_horner * np.log10(tp_hr + 1.0)

    b_mix = (1.0 - np.clip(wc_frac, 0.0, 1.0)) * Bo + np.clip(wc_frac, 0.0, 1.0) * Bw
    kh_md_ft = float(70.6 * max(prior_rate_stbd, 1.0) * b_mix * mu_o / max(dp_rad, 1e-6))
    k_md = kh_md_ft / max(h, 1.0)

    skin = float(
        1.151
        * (
            (p_1hr - pwf0) / max(m_horner, 1e-6)
            - np.log10(max(k_md / max(phi * mu_o * ct * (rw ** 2), 1e-12), 1e-6))
            + 3.23
        )
    )
    t_pss_hr = float(0.1 * phi * mu_o * ct * area_ft2 / max(0.0002637 * k_md, 1e-12))

    # Check if late-time pressure has reached boundary stabilization
    late_mask = (dt_hr >= 0.70 * dur_hr) & np.isfinite(deriv)
    late_deriv = float(np.median(deriv[late_mask])) if np.any(late_mask) else dp_rad

    if dur_hr >= 1.5 * t_pss_hr and late_deriv < 0.35 * dp_rad:
        pr_gauge = p_end
        quality_flag = "STABILIZED_PSS"
    elif dur_hr >= 3.0:
        # Dietz / MBH average reservoir pressure from radial flow window (circular drainage, CA = 31.62)
        ca_dietz = 31.62
        arg = (phi * mu_o * ct * area_ft2) / (ca_dietz * 0.0002637 * k_md * dt_hr[best_win])
        pr_mbh_pts = p_si[best_win] + dp_rad * np.log(np.maximum(arg, 1.0))
        pr_gauge = float(np.median(pr_mbh_pts))
        quality_flag = "RADIAL_MBH"
    else:
        pr_gauge = max(p_end, p_star_val) if dur_hr < 3.0 else p_end
        quality_flag = "RADIAL_FALLBACK"

    pr_datum = float(gauge_to_datum_pressure(pr_gauge, gauge_tvd, datum_tvd, fluid_grad))

    return {
        "shut_in_start": ts0,
        "shut_in_end": ts1,
        "duration_hr": dur_hr,
        "pr_gauge_psia": float(pr_gauge),
        "pr_datum_psia": pr_datum,
        "p_star_gauge_psia": p_star_val,
        "p_end_gauge_psia": p_end,
        "p_wf0_gauge_psia": pwf0,
        "dp_radial_psi": dp_rad,
        "horner_slope_psi_cycle": abs(float(slope_h)),
        "kh_md_ft": kh_md_ft,
        "k_md": k_md,
        "skin": skin,
        "t_pss_hr": t_pss_hr,
        "rad_cv": float(best_cv),
        "rad_trend_per_cycle": rad_trend,
        "rad_span_cycles": rad_span_cycles,
        "producing_time_hr": tp_hr,
        "producing_time_assumed": tp_is_assumed,
        "quality_flag": quality_flag,
    }


def detect_shutin_buildups(
    hf_well_df: pd.DataFrame,
    min_duration_hr: float = 2.0,
    choke_closed_pct: float = 0.5,
    max_time: Optional[pd.Timestamp] = None,
) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Detect true shut-in build-up intervals in 1-minute well telemetry.

    Requires `choke_opening_pct <= choke_closed_pct` (if available) in addition to
    `is_shut_in` so brief surface dp_choke sensor artifacts (like GL-03 at 50% choke)
    are never misclassified as build-ups.
    """
    df = hf_well_df if hf_well_df["timestamp"].is_monotonic_increasing else hf_well_df.sort_values("timestamp")
    if max_time is not None:
        df = df[df["timestamp"] <= pd.Timestamp(max_time)]
    if df.empty or len(df) < 30:
        return []

    if "choke_opening_pct" in df.columns:
        choke_closed = df["choke_opening_pct"].fillna(100.0) <= choke_closed_pct
        if "is_shut_in" in df.columns:
            si_mask = choke_closed & df["is_shut_in"].astype(bool)
        else:
            si_mask = choke_closed
    elif "is_shut_in" in df.columns:
        si_mask = df["is_shut_in"].astype(bool)
    else:
        return []

    arr = si_mask.to_numpy(dtype=bool)
    padded = np.concatenate([[False], arr, [False]])
    starts = np.where(~padded[:-1] & padded[1:])[0]
    ends = np.where(padded[:-1] & ~padded[1:])[0]

    intervals: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    ts = df["timestamp"].reset_index(drop=True)
    for s_idx, e_idx in zip(starts, ends):
        t_start = pd.Timestamp(ts.iloc[s_idx])
        t_end = pd.Timestamp(ts.iloc[e_idx - 1]) + pd.Timedelta("1min")
        dur_hr = (t_end - t_start).total_seconds() / 3600.0
        if dur_hr >= min_duration_hr:
            intervals.append((t_start, t_end))
    return intervals


def _fit_anchored_ipr(
    t_days: np.ndarray,
    pwf_datum: np.ndarray,
    q_liq: np.ndarray,
    anchor_times_days: np.ndarray,
    anchor_pr_datum: np.ndarray,
    j_mult: Optional[np.ndarray] = None,
    j_fixed: Optional[float] = None,
) -> Tuple[float, float, float]:
    """
    Fit Pr(t) = Pr0 - a * t and productivity index J anchored to PBU estimates.

    `j_mult` (per test, default 1) scales J in time: q = J * m(t) * (Pr(t) - Pwf).
    """
    m_arr = np.ones(len(q_liq), dtype="float64") if j_mult is None else np.asarray(j_mult, dtype="float64")
    t_ref = float(np.mean(anchor_times_days))
    pr_ref = float(np.mean(anchor_pr_datum))

    best_loss = np.inf
    best_params: Optional[Tuple[float, float, float]] = None

    # Coarse + fine 1D search over decline rate a >= 0; J has exact closed form for fixed a
    for a_grid in (np.linspace(0.0, 1.5, 301),):
        for a in a_grid:
            pr_t = pr_ref - a * (t_days - t_ref)
            dd = pr_t - pwf_datum
            if np.any(dd <= 5.0):
                continue
            de = dd * m_arr
            j_val = float(j_fixed) if j_fixed else float(np.sum(q_liq * de) / max(np.sum(de ** 2), 1e-9))
            if not (0.05 <= j_val <= 120.0):
                continue
            resid = q_liq - j_val * de
            loss = float(np.sum(resid ** 2))
            if len(anchor_times_days) > 1:
                anchor_pred = pr_ref - a * (anchor_times_days - t_ref)
                loss += float(np.sum(((anchor_pred - anchor_pr_datum) * j_val) ** 2))
            if loss < best_loss:
                best_loss = loss
                best_params = (float(pr_ref + a * t_ref), float(a), j_val)

    if best_params is not None:
        # Fine local polish around best a
        a0 = best_params[1]
        for a in np.linspace(max(0.0, a0 - 0.02), min(1.5, a0 + 0.02), 81):
            pr_t = pr_ref - a * (t_days - t_ref)
            dd = pr_t - pwf_datum
            if np.any(dd <= 5.0):
                continue
            de = dd * m_arr
            j_val = float(j_fixed) if j_fixed else float(np.sum(q_liq * de) / max(np.sum(de ** 2), 1e-9))
            if not (0.05 <= j_val <= 120.0):
                continue
            resid = q_liq - j_val * de
            loss = float(np.sum(resid ** 2))
            if len(anchor_times_days) > 1:
                anchor_pred = pr_ref - a * (anchor_times_days - t_ref)
                loss += float(np.sum(((anchor_pred - anchor_pr_datum) * j_val) ** 2))
            if loss < best_loss:
                best_loss = loss
                best_params = (float(pr_ref + a * t_ref), float(a), j_val)
        return best_params

    # Fallback if all drawdowns were too tight
    de = np.maximum(pr_ref - pwf_datum, 10.0) * m_arr
    j_val = float(j_fixed) if j_fixed else float(np.clip(np.sum(q_liq * de) / max(np.sum(de ** 2), 1e-9), 0.05, 120.0))
    return pr_ref, 0.0, j_val


def _fit_joint_unanchored_ipr(
    t_days: np.ndarray,
    pwf_datum: np.ndarray,
    q_liq: np.ndarray,
    datum_tvd_ft: float,
    grad_prior_psi_ft: float = None,
    j_mult: Optional[np.ndarray] = None,
    j_fixed: Optional[float] = None,
) -> Tuple[float, float, float]:
    """
    Jointly estimate (Pr0_datum, a, J) from well tests without a PBU anchor.

    Uses bounded optimization over J > 0, a >= 0, Pr(t) > Pwf(t), with a gentle prior on
    Pr0 / datum_tvd (`grad_prior_psi_ft`, normally the field build-up gradient) to
    regularize collinear (t, Pwf) cases.
    """
    g = DEFAULT_PR_GRADIENT_PRIOR_PSI_FT if grad_prior_psi_ft is None else float(grad_prior_psi_ft)
    m_arr = np.ones(len(t_days), dtype="float64") if j_mult is None else np.asarray(j_mult, dtype="float64")
    if len(t_days) == 0:
        return g * datum_tvd_ft, 0.15, 2.0

    if len(t_days) == 1:
        a_prior = 0.15
        pr0 = max(float(pwf_datum[0]) + 150.0 + a_prior * float(t_days[0]), g * datum_tvd_ft)
        pr_t = pr0 - a_prior * float(t_days[0])
        if j_fixed:  # Pr from the single test given J
            return float(pwf_datum[0] + q_liq[0] / (m_arr[0] * j_fixed) + a_prior * float(t_days[0])), a_prior, float(j_fixed)
        j_val = float(np.clip(q_liq[0] / (m_arr[0] * max(pr_t - float(pwf_datum[0]), 10.0)), 0.05, 120.0))
        return pr0, a_prior, j_val

    pr0_target = g * datum_tvd_ft
    sigma_pr0 = 0.05 * datum_tvd_ft
    sigma_q = max(50.0, 0.03 * float(np.mean(q_liq)))
    w_p = 1.0 / (sigma_pr0 ** 2)
    w_a = 1.0 / (0.50 ** 2)
    n = len(t_days)

    best_loss = np.inf
    best_params: Optional[Tuple[float, float, float]] = None

    def _eval_j(j_val: float) -> Optional[Tuple[float, float, float, float]]:
        y = q_liq / (j_val * m_arr) + pwf_datum
        w_d = (j_val * m_arr / sigma_q) ** 2  # per-test weight in pressure space
        x_mat = np.column_stack([np.ones(n), -t_days])
        a_mat = (x_mat.T * w_d) @ x_mat + np.diag([w_p, w_a])
        b_vec = (x_mat.T * w_d) @ y + np.array([w_p * pr0_target, 0.0])
        theta = np.linalg.solve(a_mat, b_vec)
        pr0, a_val = float(theta[0]), float(theta[1])
        if a_val < 0.0 or a_val > 1.5:
            a_val = float(np.clip(a_val, 0.0, 1.5))
            pr0 = float((np.sum(w_d * (y + a_val * t_days)) + w_p * pr0_target) / (np.sum(w_d) + w_p))
        pr_t = pr0 - a_val * t_days
        if np.any(pr_t <= pwf_datum + 5.0):
            return None
        q_hat = j_val * m_arr * (pr_t - pwf_datum)
        loss = (
            float(np.sum(((q_liq - q_hat) / sigma_q) ** 2))
            + ((pr0 - pr0_target) / sigma_pr0) ** 2
            + (a_val / 0.50) ** 2
        )
        return loss, pr0, a_val, j_val

    if j_fixed:
        res = _eval_j(float(j_fixed))
        if res is not None:
            return res[1], res[2], res[3]
    for j_val in (np.geomspace(0.15, 80.0, 260) if not j_fixed else []):
        res = _eval_j(float(j_val))
        if res is not None and res[0] < best_loss:
            best_loss = res[0]
            best_params = (res[1], res[2], res[3])

    if best_params is not None and not j_fixed:
        j0 = best_params[2]
        for j_val in np.linspace(max(0.10, 0.85 * j0), min(100.0, 1.15 * j0), 120):
            res = _eval_j(float(j_val))
            if res is not None and res[0] < best_loss:
                best_loss = res[0]
                best_params = (res[1], res[2], res[3])
        return best_params

    # Safety fallback
    pr0 = max(float(np.max(pwf_datum)) + 200.0, pr0_target)
    de = np.maximum(pr0 - pwf_datum, 20.0) * m_arr
    j_val = float(j_fixed) if j_fixed else float(np.clip(np.sum(q_liq * de) / max(np.sum(de ** 2), 1e-9), 0.1, 100.0))
    return pr0, 0.0, j_val


def fit_pr_and_pi(
    well_tests: pd.DataFrame,
    hf_cache_by_well: Mapping[str, pd.DataFrame],
    pvt: pd.DataFrame,
    completion: pd.DataFrame,
    max_time: Optional[pd.Timestamp] = None,
    ref_timestamp: Optional[pd.Timestamp] = None,
    pr_gradient_prior_psi_ft: Optional[float] = None,
    external_anchors: Optional[Mapping[str, Sequence[Mapping[str, Any]]]] = None,
    j_multipliers: Optional[Mapping[str, Callable[[np.ndarray], np.ndarray]]] = None,
    fixed_j: Optional[Mapping[str, float]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Fit reservoir pressure decline Pr(t) = Pr0 - a * t_days and productivity index J
    across all wells in `well_tests`.

    Coupling hooks (both optional, per well):
      * `external_anchors[well]`: list of {"timestamp", "pr_datum_psia", ...} that REPLACE the
        internal build-up detection for that well (an empty list means "no pressure anchor").
      * `j_multipliers[well]`: m(t_days) scaling J in time (skin/kh changes). It is applied in
        calibration as well as prediction, so J refers to the reference well condition.

    Wells without a build-up use a prior on Pr0 / datum_tvd: `pr_gradient_prior_psi_ft`
    if given, else the median gradient of the build-ups found in this call, else
    `DEFAULT_PR_GRADIENT_PRIOR_PSI_FT`.

    When `max_time` is provided, strictly excludes any well tests ending after
    `max_time` and any high-frequency telemetry (including shut-in PBUs) after
    `max_time`, preventing future leakage during chronological walk-forward CV.
    """
    wt = well_tests.copy()
    wt["test_start"] = pd.to_datetime(wt["test_start"], utc=True)
    wt["test_end"] = pd.to_datetime(wt["test_end"], utc=True)
    if max_time is not None:
        mt = pd.Timestamp(max_time)
        wt = wt[wt["test_end"] <= mt].copy()

    if ref_timestamp is None:
        ref_candidates: List[pd.Timestamp] = []
        if len(wt) > 0:
            ref_candidates.append(pd.Timestamp(wt["test_start"].min()).normalize())
        for hf_w in hf_cache_by_well.values():
            if hf_w is not None and len(hf_w) > 0 and "timestamp" in hf_w.columns:
                ts_min = pd.to_datetime(hf_w["timestamp"], utc=True).min()
                if pd.notna(ts_min):
                    ref_candidates.append(pd.Timestamp(ts_min).normalize())
        ref_timestamp = min(ref_candidates) if ref_candidates else DEFAULT_REF_TIMESTAMP

    results: Dict[str, Dict[str, Any]] = {}
    pvt_by_well = pvt.set_index("well_name")
    comp_by_well = completion.set_index("well_name")
    prepared: List[Dict[str, Any]] = []

    wells = sorted(set(wt["well_name"].unique()) | set(comp_by_well.index.unique()))
    for well in wells:
        if well not in pvt_by_well.index or well not in comp_by_well.index:
            continue

        p_row = pvt_by_well.loc[well]
        c_row = comp_by_well.loc[well]
        sub = wt[wt["well_name"] == well].sort_values("test_start").copy()
        if sub.empty:
            continue

        gauge_tvd = float(c_row["gauge_tvd_ft"])
        datum_tvd = float(p_row["datum_tvd_ft"])
        pb_psia = float(p_row["bubble_point_pressure_psia"])
        bo = float(p_row["oil_fvf_rb_stb"])
        bw = float(c_row["water_fvf_rb_stb"])
        rs_pb = float(c_row["solution_gor_at_pb_scf_stb"])

        wc_frac = np.clip(sub["water_cut_pct"].to_numpy(dtype="float64") / 100.0, 0.0, 1.0)
        api_arr = sub["oil_gravity_api"].to_numpy(dtype="float64")
        gsg_arr = sub["gas_sg"].to_numpy(dtype="float64")
        sal_arr = sub["water_salinity_ppm"].to_numpy(dtype="float64")

        grads = compute_wellbore_fluid_gradient(
            api=api_arr,
            wc_frac=wc_frac,
            gor_scf_stb=rs_pb,
            gas_sg=gsg_arr,
            bo=bo,
            bw=bw,
            salinity_ppm=sal_arr,
        )
        pwf_datum = gauge_to_datum_pressure(
            sub["dh_pressure_psia"].to_numpy(dtype="float64"),
            gauge_tvd,
            datum_tvd,
            grads,
        )
        t_mid = sub["test_start"] + (sub["test_end"] - sub["test_start"]) / 2
        t_days = (t_mid - ref_timestamp).dt.total_seconds().to_numpy(dtype="float64") / 86400.0
        q_liq = sub["liquid_rate_stbd"].to_numpy(dtype="float64")

        # Detect shut-in build-ups strictly <= max_time (or take externally supplied anchors)
        buildups: List[Dict[str, Any]] = []
        hf_df = hf_cache_by_well.get(well)
        if external_anchors is not None and well in external_anchors:
            for anc in external_anchors[well]:
                a_ts = pd.Timestamp(anc["timestamp"])
                a_ts = a_ts.tz_localize("UTC") if a_ts.tzinfo is None else a_ts.tz_convert("UTC")
                if max_time is not None and a_ts > pd.Timestamp(max_time):
                    continue  # causal guard also applies to external anchors
                pr_d = float(anc["pr_datum_psia"])
                buildups.append({**dict(anc), "t_days": (a_ts - ref_timestamp).total_seconds() / 86400.0,
                                 "pr_datum_psia": pr_d, "pr_gradient_psi_ft": pr_d / datum_tvd,
                                 "quality_flag": anc.get("quality_flag", "EXTERNAL_ANCHOR"),
                                 "source": anc.get("source", "external")})
        elif hf_df is not None and not hf_df.empty:
            intervals = detect_shutin_buildups(hf_df, min_duration_hr=2.0, max_time=max_time)
            for s_start, s_end in intervals:
                prior = sub[sub["test_end"] <= s_start]
                if prior.empty:
                    prior_row = sub.iloc[0]
                else:
                    prior_row = prior.iloc[-1]
                t_pbu_d = (s_start - ref_timestamp).total_seconds() / 86400.0
                tp_hr = None
                first_prod = c_row.get("first_production_date", None)
                if first_prod is not None and pd.notna(first_prod):
                    fp_ts = pd.Timestamp(first_prod)
                    fp_ts = fp_ts.tz_localize("UTC") if fp_ts.tzinfo is None else fp_ts.tz_convert("UTC")
                    tp_hr = max((pd.Timestamp(s_start) - fp_ts).total_seconds() / 3600.0, 0.0) or None
                wc_pbu = float(np.interp(t_pbu_d, t_days, wc_frac))
                grad_pbu = float(np.interp(t_pbu_d, t_days, grads))
                res_pbu = estimate_pr_from_buildup(
                    hf_well_df=hf_df if max_time is None else hf_df[hf_df["timestamp"] <= pd.Timestamp(max_time)],
                    shut_in_start=s_start,
                    shut_in_end=s_end,
                    prior_rate_stbd=float(prior_row["liquid_rate_stbd"]),
                    pvt_row=p_row,
                    comp_row=c_row,
                    fluid_grad_psi_ft=grad_pbu,
                    wc_frac=wc_pbu,
                    oil_api=float(prior_row["oil_gravity_api"]),
                    gas_sg=float(prior_row["gas_sg"]),
                    salinity_ppm=float(prior_row["water_salinity_ppm"]),
                    producing_time_hr=tp_hr,
                )
                if np.isfinite(res_pbu["pr_datum_psia"]):
                    res_pbu["t_days"] = float(t_pbu_d)
                    res_pbu["pr_gradient_psi_ft"] = float(res_pbu["pr_datum_psia"]) / datum_tvd
                    buildups.append(res_pbu)

        prepared.append({
            "well": well, "sub": sub, "gauge_tvd": gauge_tvd, "datum_tvd": datum_tvd,
            "pb_psia": pb_psia, "bo": bo, "bw": bw, "rs_pb": rs_pb, "wc_frac": wc_frac,
            "api_arr": api_arr, "gsg_arr": gsg_arr, "sal_arr": sal_arr, "pwf_datum": pwf_datum,
            "t_days": t_days, "q_liq": q_liq, "buildups": buildups,
        })

    # Reservoir-pressure prior for wells without a build-up: the median Pr/datum-TVD gradient
    # of the build-ups found in this call (all detected <= max_time, so causal). A caller that
    # fits one well at a time passes it explicitly; the nominal default is the last resort.
    field_grads = [b["pr_gradient_psi_ft"] for p in prepared for b in p["buildups"]]
    if pr_gradient_prior_psi_ft is not None and np.isfinite(pr_gradient_prior_psi_ft):
        grad_prior, prior_source = float(pr_gradient_prior_psi_ft), "caller"
    elif field_grads:
        grad_prior, prior_source = float(np.median(field_grads)), f"field_buildups(n={len(field_grads)})"
    else:
        grad_prior, prior_source = DEFAULT_PR_GRADIENT_PRIOR_PSI_FT, "nominal_default"

    for prep in prepared:
        well, sub, buildups = prep["well"], prep["sub"], prep["buildups"]
        gauge_tvd, datum_tvd, pb_psia = prep["gauge_tvd"], prep["datum_tvd"], prep["pb_psia"]
        bo, bw, rs_pb, wc_frac = prep["bo"], prep["bw"], prep["rs_pb"], prep["wc_frac"]
        api_arr, gsg_arr, sal_arr = prep["api_arr"], prep["gsg_arr"], prep["sal_arr"]
        pwf_datum, t_days, q_liq = prep["pwf_datum"], prep["t_days"], prep["q_liq"]
        j_fn = (j_multipliers or {}).get(well)
        m_arr = np.asarray(j_fn(t_days), dtype="float64") if j_fn is not None else np.ones(len(t_days))
        j_fix = (fixed_j or {}).get(well)

        has_pbu = len(buildups) > 0
        if has_pbu:
            anchor_t = np.array([b["t_days"] for b in buildups], dtype="float64")
            anchor_pr = np.array([b["pr_datum_psia"] for b in buildups], dtype="float64")
            pr0, decline_a, j_val = _fit_anchored_ipr(t_days, pwf_datum, q_liq, anchor_t, anchor_pr, m_arr, j_fix)
        else:
            pr0, decline_a, j_val = _fit_joint_unanchored_ipr(t_days, pwf_datum, q_liq, datum_tvd, grad_prior, m_arr, j_fix)

        # In-sample and Leave-One-Test-Out (LOTO) validation
        pr_in = pr0 - decline_a * t_days
        q_in = ipr_rate(j_val * m_arr, pr_in, pwf_datum, Pb=pb_psia)
        in_mae = float(np.mean(np.abs(q_liq - q_in)))
        in_rmse = float(np.sqrt(np.mean((q_liq - q_in) ** 2)))

        n_tests = len(t_days)
        if n_tests >= 3:
            loto_errs = []
            for k in range(n_tests):
                m = np.ones(n_tests, dtype=bool)
                m[k] = False
                if has_pbu:
                    pr0_k, a_k, j_k = _fit_anchored_ipr(t_days[m], pwf_datum[m], q_liq[m], anchor_t, anchor_pr, m_arr[m], j_fix)
                else:
                    pr0_k, a_k, j_k = _fit_joint_unanchored_ipr(t_days[m], pwf_datum[m], q_liq[m], datum_tvd, grad_prior, m_arr[m], j_fix)
                q_pred_k = float(ipr_rate(j_k * m_arr[k], pr0_k - a_k * t_days[k], pwf_datum[k], Pb=pb_psia))
                loto_errs.append(float(q_liq[k]) - q_pred_k)
            loto_arr = np.asarray(loto_errs, dtype="float64")
            loto_mae = float(np.mean(np.abs(loto_arr)))
            loto_rmse = float(np.sqrt(np.mean(loto_arr ** 2)))
            loto_mape = float(np.mean(np.abs(loto_arr) / np.maximum(q_liq, 1.0)) * 100.0)
        else:
            loto_mae = in_mae
            loto_rmse = max(in_rmse, 0.05 * float(np.mean(q_liq)))
            loto_mape = float(np.mean(np.abs(q_liq - q_in) / np.maximum(q_liq, 1.0)) * 100.0)

        sigma_ipr = max(loto_rmse, 1e-3)

        results[well] = {
            "well_name": well,
            "J": float(j_val),
            "Pr0_datum_psia": float(pr0),
            "decline_psi_per_day": float(decline_a),
            "ref_timestamp": ref_timestamp,
            "datum_tvd_ft": datum_tvd,
            "gauge_tvd_ft": gauge_tvd,
            "bubble_point_psia": pb_psia,
            "oil_fvf_rb_stb": bo,
            "water_fvf_rb_stb": bw,
            "solution_gor_scf_stb": rs_pb,
            "default_api": float(np.median(api_arr)),
            "default_gas_sg": float(np.median(gsg_arr)),
            "default_salinity_ppm": float(np.median(sal_arr)),
            "default_wc_frac": float(np.median(wc_frac)),
            "has_pbu_anchor": has_pbu,
            "pr_gradient_prior_psi_ft": None if has_pbu else grad_prior,
            "pr_prior_source": ("external_anchor" if external_anchors is not None and well in external_anchors
                                and has_pbu else "buildup_anchor") if has_pbu else prior_source,
            "j_multiplier": j_fn,
            "j_source": "fixed (external PI)" if j_fix else "fitted to well tests",
            "n_tests": int(n_tests),
            "n_buildups": int(len(buildups)),
            "buildup_anchors": buildups,
            "in_sample_mae_stbd": in_mae,
            "in_sample_rmse_stbd": in_rmse,
            "loto_mae_stbd": loto_mae,
            "loto_rmse_stbd": loto_rmse,
            "loto_mape_pct": loto_mape,
            "sigma_ipr": sigma_ipr,
        }

    return results


def predict_ipr_q_liq(
    well_params: Mapping[str, Any],
    timestamps: Union[pd.Series, pd.DatetimeIndex, Sequence[pd.Timestamp]],
    dh_pressure_psia: ArrayLike,
    wc_frac: Optional[ArrayLike] = None,
    oil_api: Optional[ArrayLike] = None,
    gas_sg: Optional[ArrayLike] = None,
    salinity_ppm: Optional[ArrayLike] = None,
) -> np.ndarray:
    """
    Predict 1-minute or test-level liquid rate (stb/d) from fitted IPR parameters.

    Returns `np.nan` wherever `dh_pressure_psia` is non-finite.
    """
    ts = pd.DatetimeIndex(pd.to_datetime(timestamps, utc=True))
    ref_ts = pd.Timestamp(well_params.get("ref_timestamp", DEFAULT_REF_TIMESTAMP))
    t_days = (ts - ref_ts).total_seconds().to_numpy(dtype="float64") / 86400.0

    pr_datum = (
        float(well_params["Pr0_datum_psia"])
        - float(well_params["decline_psi_per_day"]) * t_days
    )

    wc_val = well_params.get("default_wc_frac", 0.3) if wc_frac is None else wc_frac
    api_val = well_params.get("default_api", 32.0) if oil_api is None else oil_api
    gsg_val = well_params.get("default_gas_sg", 0.75) if gas_sg is None else gas_sg
    sal_val = well_params.get("default_salinity_ppm", 100_000.0) if salinity_ppm is None else salinity_ppm

    grad = compute_wellbore_fluid_gradient(
        api=api_val,
        wc_frac=wc_val,
        gor_scf_stb=float(well_params.get("solution_gor_scf_stb", 0.0)),
        gas_sg=gsg_val,
        bo=float(well_params.get("oil_fvf_rb_stb", 1.14)),
        bw=float(well_params.get("water_fvf_rb_stb", 1.02)),
        salinity_ppm=sal_val,
    )
    pwf_datum = gauge_to_datum_pressure(
        dh_pressure_psia,
        float(well_params["gauge_tvd_ft"]),
        float(well_params["datum_tvd_ft"]),
        grad,
    )
    j_fn = well_params.get("j_multiplier")
    j_eff = float(well_params["J"]) * (np.asarray(j_fn(t_days), dtype="float64") if j_fn is not None else 1.0)
    q = ipr_rate(
        j_eff,
        pr_datum,
        pwf_datum,
        Pb=float(well_params["bubble_point_psia"]),
    )
    return np.asarray(q, dtype="float64")
