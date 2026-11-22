"""Globally Recognized Industry PTA Methods for Rough / Ambiguous / Non-IARF Build-Ups.

Implements the four canonical SPE / KAPPA Saphir / IHS WellTest methods used when
Infinite-Acting Radial Flow (IARF) is obscured by wellbore storage, short shut-in
duration, variable rate history, or early closed-boundary depletion:

1. Nonlinear Analytical Model Matching (Bourdet-Gringarten C_D*e^(2S) + Closed
   Circular Boundary via Stehfest 12-Term Laplace Inversion, SPE-12777 / Bourdet 2002):
   - Fits exact coupled (k, S, C, r_e) directly to BOTH Delta-P and Bourdet Derivative
     with full build-up superposition and effective-radius negative-skin handling.
   - Works even when ZERO flat radial-flow plateau exists, returning fitted type curves,
     true radial derivative asymptote (70.6*q*B*mu/kh), and Jacobian parameter uncertainty.

2. Regularized Pressure-Rate Deconvolution (von Schroeter / Levitan / Ilk, SPE-77688 /
   SPE-90680):
   - Combines continuous multi-rate VFM production history q(t) with PDG shut-in &
     flowing pressure to recover the constant-rate unit-drawdown response Delta-P_u(tau)
     and deconvolved Bourdet derivative extending well past the shut-in duration.

3. Agarwal Equivalent Time (SPE-9289) + Tiab's Direct Synthesis (TDS, SPE-26138):
   - Superposition-corrected equivalent time dt_e = tp*dt/(tp+dt) and direct analytical
     wellbore-storage (C, C_D), hump-peak, and 1.5-log-cycle end-of-storage diagnostics.

4. Regime Triage & Automated Method Recommendation:
   - Classifies each build-up (Clean IARF / Boundary-Masked IARF / Storage-Dominated
     No-IARF) and recommends the primary physical interpretation method accordingly.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import BSpline
from scipy.optimize import least_squares, lsq_linear, minimize_scalar
from scipy.signal import savgol_filter
from scipy.special import i0e, i1e, k0e, k1e
from scipy.stats import theilslopes

# ---------------------------------------------------------------------------
# 1. Stehfest 12-Term Laplace Inversion Coefficients (Exact Double Precision)
# ---------------------------------------------------------------------------
def _stehfest_weights(n: int = 12) -> np.ndarray:
    assert n % 2 == 0
    v = np.zeros(n, dtype=np.float64)
    half = n // 2
    for i in range(1, n + 1):
        s = 0.0
        for k in range((i + 1) // 2, min(i, half) + 1):
            num = (k ** half) * math.factorial(2 * k)
            den = (
                math.factorial(half - k)
                * math.factorial(k)
                * math.factorial(k - 1)
                * math.factorial(i - k)
                * math.factorial(2 * k - i)
            )
            s += num / den
        v[i - 1] = ((-1) ** (i + half)) * s
    return v


_STEHFEST_N = 12
_STEHFEST_V = _stehfest_weights(_STEHFEST_N)
_STEHFEST_K = np.arange(1, _STEHFEST_N + 1, dtype=np.float64) * math.log(2.0)


def _laplace_p_wd(
    s: np.ndarray,
    c_d: float,
    skin: float,
    re_d: float,
) -> np.ndarray:
    """Exact Laplace-space wellbore pressure p_bar_wD(s) for WBS + Skin + Circular Closed Boundary.

    Uses exponentially scaled modified Bessel functions (k0e, k1e, i0e, i1e) so
    large arguments (infinite-acting limit) never overflow.
    For negative skin (S < 0), uses the exact effective wellbore radius transform
    r_wa = r_w * exp(-S) (handled by caller scaling t_D, C_D, r_eD with S_eff = 0).
    """
    u = np.sqrt(np.maximum(s, 1e-30))
    z = re_d * u

    k0_u = k0e(u)
    k1_u = k1e(u)
    i0_u = i0e(u)
    i1_u = i1e(u)

    # Boundary ratio alpha = exp(-2*(z - u)) * K1e(z) / I1e(z) -> 0 smoothly for large z
    exp_arg = -2.0 * (z - u)
    alpha = np.zeros_like(u)
    mask = (exp_arg > -600.0) & (re_d < 1e7)
    if np.any(mask):
        zm = z[mask]
        i1_z = np.maximum(i1e(zm), 1e-300)
        alpha[mask] = np.exp(exp_arg[mask]) * (k1e(zm) / i1_z)

    num = k0_u + alpha * i0_u
    den = u * np.maximum(k1_u - alpha * i1_u, 1e-300)
    p_no_wbs = (num / den + skin) / np.maximum(s, 1e-30)

    # Wellbore storage coupling in Laplace space (Agarwal, Al-Hussainy & Ramey 1970)
    return p_no_wbs / (1.0 + c_d * (s * s) * p_no_wbs)


def stehfest_pwd(
    t_d: np.ndarray,
    c_d: float,
    skin: float,
    re_d: float,
) -> np.ndarray:
    """Evaluate dimensionless wellbore pressure p_wD(t_D) via 12-term Stehfest inversion."""
    t_d = np.asarray(t_d, dtype=np.float64)
    # Effective wellbore radius transform for stimulated wells (S < 0) guarantees positivity for all s > 0
    if skin < 0.0:
        scale = math.exp(2.0 * skin)
        t_eff = np.maximum(t_d * scale, 1e-12)
        cd_eff = c_d * scale
        red_eff = max(re_d * math.exp(skin), 2.5)
        s_eff = 0.0
    else:
        t_eff = np.maximum(t_d, 1e-12)
        cd_eff = c_d
        red_eff = max(re_d, 2.5)
        s_eff = skin

    # Matrix of Laplace variables s_{i,j} = (j * ln 2) / t_i
    s_mat = _STEHFEST_K[None, :] / t_eff[:, None]
    f_mat = _laplace_p_wd(s_mat, cd_eff, s_eff, red_eff)
    pwd = (math.log(2.0) / t_eff) * (f_mat @ _STEHFEST_V)
    return np.maximum(pwd, 0.0)


def extract_pta_rock_fluid(
    pvt_row: Mapping[str, Any], comp_row: Optional[Mapping[str, Any]] = None
) -> Tuple[float, float, float, float, float, float]:
    """Strictly extract required reservoir PVT & wellbore parameters from dataset tables.

    Raises ValueError if any required rock/fluid property is missing or non-positive,
    preventing silent hardcoded fallbacks that could mask schema mismatches.
    Returns (bo, mu, h_ft, phi, ct, rw).
    """
    comp_row = {} if comp_row is None else comp_row

    def _req(name: str, aliases: Tuple[str, ...], sources: Tuple[Mapping[str, Any], ...]) -> float:
        for src in sources:
            for k in aliases:
                if k in src and src[k] is not None:
                    try:
                        val = float(src[k])
                    except (TypeError, ValueError):
                        continue
                    if np.isfinite(val) and val > 0.0:
                        return val
        raise ValueError(f"Required PVT/completion parameter '{name}' (aliases={aliases}) is missing or non-positive.")

    bo = _req("oil_fvf_rb_stb", ("oil_fvf_rb_stb", "bo"), (pvt_row,))
    mu = _req("oil_viscosity_cp", ("oil_viscosity_cp", "mu_o", "viscosity_cp"), (pvt_row,))
    h_ft = _req("net_pay_thickness_ft", ("net_pay_thickness_ft", "contributing_thickness_ft", "h_ft"), (pvt_row,))
    phi = _req("porosity_frac", ("porosity_frac", "porosity_fraction", "porosity"), (pvt_row,))
    ct = _req("total_compressibility_1_psi", ("total_compressibility_1_psi", "total_compressibility_per_psi", "ct"), (pvt_row,))
    rw = _req("wellbore_radius_ft", ("wellbore_radius_ft", "rw_ft"), (pvt_row, comp_row))
    return bo, mu, h_ft, phi, ct, rw


def analytical_buildup_dp_and_deriv(
    dt_hr: np.ndarray,
    tp_hr: float,
    qo_stbd: float,
    k_md: float,
    skin: float,
    c_bbl_psi: float,
    re_ft: float,
    pvt_row: Mapping[str, Any],
    comp_row: Mapping[str, Any],
    bourdet_L: float = 0.12,
    use_agarwal: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute exact analytical build-up Delta-P(dt) and Bourdet derivative P'(dt)."""
    bo, mu, h_ft, phi, ct, rw = extract_pta_rock_fluid(pvt_row, comp_row)

    dt = np.asarray(dt_hr, dtype=np.float64)
    t_coeff = 0.0002637 * k_md / (phi * mu * ct * rw * rw)
    p_coeff = 141.2 * qo_stbd * bo * mu / (k_md * h_ft)
    c_d = 0.8936 * c_bbl_psi / (phi * ct * h_ft * rw * rw)
    re_d = max(re_ft / rw, 20.0)

    # Build-up superposition: dp_bu(dt) = p_coeff * [p_wD(dt_D) - (p_wD(tp_D + dt_D) - p_wD(tp_D))]
    # Avoid giant-number Stehfest cancellation when tp_D >> t_pss_D = 0.25 * re_D^2:
    # In exact PSS, p_wD(tp_D + dt_D) - p_wD(tp_D) == 2 * dt_D / re_D^2 identically!
    tp_d = max(tp_hr * t_coeff, 1.0)
    dt_d = dt * t_coeff
    p_dt = stehfest_pwd(dt_d, c_d, skin, re_d)
    t_pss_d = 0.25 * re_d * re_d
    if re_d < 1e6 and tp_d >= t_pss_d:
        # Exact PSS closed-boundary superposition (zero floating-point cancellation!)
        superpos_diff = 2.0 * dt_d / (re_d * re_d)
    elif re_d >= 1e6 or tp_d > 50.0 * np.max(dt_d):
        # Exact infinite-acting radial superposition: p_D(tp+dt) - p_D(tp) = 0.5 * ln(1 + dt/tp)
        # (smoothly blended if tp_d is approaching PSS)
        if re_d < 1e6 and tp_d > 0.05 * re_d * re_d:
            p_tp = float(stehfest_pwd(np.array([tp_d]), c_d, skin, re_d)[0])
            p_tp_dt = stehfest_pwd(tp_d + dt_d, c_d, skin, re_d)
            superpos_diff = np.maximum(p_tp_dt - p_tp, 0.0)
        else:
            superpos_diff = 0.5 * np.log1p(dt_d / tp_d)
    else:
        p_tp = float(stehfest_pwd(np.array([tp_d]), c_d, skin, re_d)[0])
        p_tp_dt = stehfest_pwd(tp_d + dt_d, c_d, skin, re_d)
        superpos_diff = np.maximum(p_tp_dt - p_tp, 0.0)
    dp_model = np.maximum(p_coeff * (p_dt - superpos_diff), 1e-6)

    # Exact smooth derivative wrt Agarwal equivalent time (or elapsed time)
    t_axis = (tp_hr * dt) / (tp_hr + dt) if use_agarwal else dt
    deriv_model = compute_bourdet_derivative(t_axis, dp_model, L=bourdet_L)
    return dp_model, deriv_model


def compute_bourdet_derivative(t: np.ndarray, dp: np.ndarray, L: float = 0.15) -> np.ndarray:
    """Standard Bourdet weighted logarithmic derivative d(dp)/d(ln t)."""
    t = np.asarray(t, dtype=np.float64)
    dp = np.asarray(dp, dtype=np.float64)
    n = len(t)
    if n < 3:
        return np.full(n, np.nan)
    ln_t = np.log(np.maximum(t, 1e-12))
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        j_l = i - 1
        while j_l > 0 and (ln_t[i] - ln_t[j_l]) < L:
            j_l -= 1
        j_r = i + 1
        while j_r < n - 1 and (ln_t[j_r] - ln_t[i]) < L:
            j_r += 1
        has_l = j_l >= 0 and (ln_t[i] - ln_t[j_l]) > 1e-8
        has_r = j_r < n and (ln_t[j_r] - ln_t[i]) > 1e-8
        if has_l and has_r:
            dx_l = ln_t[i] - ln_t[j_l]
            dx_r = ln_t[j_r] - ln_t[i]
            m_l = (dp[i] - dp[j_l]) / dx_l
            m_r = (dp[j_r] - dp[i]) / dx_r
            out[i] = (m_l * dx_r + m_r * dx_l) / (dx_l + dx_r)
        elif has_r:
            out[i] = (dp[j_r] - dp[i]) / (ln_t[j_r] - ln_t[i])
        elif has_l:
            out[i] = (dp[i] - dp[j_l]) / (ln_t[i] - ln_t[j_l])
    return out


def _preprocess_log_grid(
    dt_1m: np.ndarray,
    p_raw: np.ndarray,
    pwf_psi: float,
    tp_hr: float,
    n_per_decade: int = 24,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Hampel+SavGol clean and log-space resample build-up for Agarwal Bourdet analysis."""
    dt = np.asarray(dt_1m, dtype=np.float64)
    p = np.asarray(p_raw, dtype=np.float64)
    ok = (dt > 0) & np.isfinite(p)
    dt, p = dt[ok], p[ok]
    if len(dt) < 8:
        raise ValueError("Need at least 8 valid build-up points.")

    # Light Hampel outlier cleaning + gentle Savitzky-Golay smoothing
    s = pd.Series(p)
    med = s.rolling(7, center=True, min_periods=1).median()
    mad = (s - med).abs().rolling(7, center=True, min_periods=1).median()
    sigma = np.maximum(1.4826 * mad, 0.25)
    p_clean = np.where((s - med).abs() > 3.5 * sigma, med, s).astype(np.float64)
    if len(p_clean) >= 11:
        win = min(11, len(p_clean) if len(p_clean) % 2 == 1 else len(p_clean) - 1)
        p_sm = savgol_filter(p_clean, win, 2)
    else:
        p_sm = p_clean

    # Log-spaced resampling
    t_lo = max(float(dt[0]), 1.0 / 60.0)
    t_hi = float(dt[-1])
    decades = max(0.3, math.log10(t_hi / t_lo))
    n_pts = int(np.clip(round(decades * n_per_decade), 18, 120))
    t_grid = np.geomspace(t_lo, t_hi, n_pts)
    p_grid = np.interp(np.log(t_grid), np.log(dt), p_sm)
    dp_grid = np.maximum(p_grid - pwf_psi, 0.1)

    # Agarwal equivalent time dt_e = tp * dt / (tp + dt)
    dte_grid = (tp_hr * t_grid) / (tp_hr + t_grid)
    der_raw = compute_bourdet_derivative(t_grid, dp_grid, L=0.15)
    der_ag = compute_bourdet_derivative(dte_grid, dp_grid, L=0.15)
    return t_grid, dte_grid, dp_grid, der_raw, der_ag


# ---------------------------------------------------------------------------
# 2. Method C: Agarwal Equivalent Time + Tiab's Direct Synthesis (TDS, SPE-26138)
# ---------------------------------------------------------------------------
def run_tiab_direct_synthesis(
    dt_grid: np.ndarray,
    dte_grid: np.ndarray,
    dp_grid: np.ndarray,
    der_ag: np.ndarray,
    qo_stbd: float,
    pvt_row: Mapping[str, Any],
    comp_row: Mapping[str, Any],
) -> Dict[str, Any]:
    """Tiab's Direct Synthesis (SPE-26138) + Agarwal Equivalent-Time diagnostics."""
    bo, mu, h_ft, phi, ct, rw = extract_pta_rock_fluid(pvt_row, comp_row)

    # 1. Early-time wellbore storage coefficient C (bbl/psi) from earliest points
    n_early = max(2, min(6, len(dt_grid) // 6))
    c_est = float(np.median((qo_stbd * bo / 24.0) * (dt_grid[:n_early] / np.maximum(dp_grid[:n_early], 0.5))))
    c_est = float(np.clip(c_est, 1e-4, 5.0))
    c_d = 0.8936 * c_est / (phi * ct * h_ft * rw * rw)

    # 2. Storage hump peak (t_x, dp_x, p'_x) on Agarwal Bourdet derivative
    valid_der = np.isfinite(der_ag) & (der_ag > 0)
    if valid_der.sum() < 5:
        return {"available": False, "note": "Insufficient positive Bourdet derivative points."}

    idx_v = np.flatnonzero(valid_der)
    # Search peak in first 60% of log-time span (storage hump precedes radial flow)
    n_search = max(3, int(0.65 * len(idx_v)))
    peak_local = int(np.argmax(der_ag[idx_v[:n_search]]))
    ipk = int(idx_v[peak_local])
    t_peak = float(dte_grid[ipk])
    dp_peak = float(dp_grid[ipk])
    der_peak = float(der_ag[ipk])

    # 3. Minimum/flattest post-peak derivative estimate for radial asymptote P'_r
    post_idx = idx_v[idx_v >= ipk]
    if len(post_idx) >= 5:
        # Exclude extreme late-time boundary collapse (<30% of mid-time median)
        mid_slice = post_idx[max(1, len(post_idx) // 5): max(2, int(0.75 * len(post_idx)))]
        # Find local minimum absolute slope on log(P') vs log(t_e)
        lt = np.log10(dte_grid[post_idx])
        ld = np.log10(der_ag[post_idx])
        slopes = np.abs(np.gradient(ld, lt))
        # Prefer points after peak where slope is closest to zero and derivative is positive
        cand = np.arange(max(1, len(post_idx) // 6), max(2, int(0.85 * len(post_idx))))
        best_j = int(cand[int(np.argmin(slopes[cand]))])
        i_rad = int(post_idx[best_j])
        der_rad = float(der_ag[i_rad])
        t_rad = float(dte_grid[i_rad])
        dp_rad = float(dp_grid[i_rad])
    else:
        # Pure storage-dominated fallback using hump height ratio (Tiab 1993)
        der_rad = max(0.35 * der_peak, 1.0)
        t_rad = float(dte_grid[idx_v[-1]])
        dp_rad = float(dp_grid[idx_v[-1]])

    # 4. Tiab Direct Synthesis permeability & skin from radial derivative level
    k_tds = float(70.6 * qo_stbd * bo * mu / (der_rad * h_ft))
    m_tds = float(2.302585 * der_rad)
    skin_tds = float(
        0.5 * ((dp_rad / der_rad) - math.log(max(1e-6, k_tds * t_rad / (phi * mu * ct * rw * rw))) + 7.4316)
    )
    dp_skin_tds = float(0.869 * m_tds * skin_tds)
    cd_e2s = float(c_d * math.exp(min(40.0, max(-20.0, 2.0 * skin_tds))))

    # Classical 1.5 log-cycle rule: end of wellbore storage occurs ~ 1.5 log cycles after unit-slope departure
    t_end_wbs_est = float(min(t_peak * 15.0, float(dt_grid[-1]) * 10.0))

    return {
        "available": True,
        "method": "Tiab Direct Synthesis (TDS) + Agarwal Δt_e",
        "c_bbl_psi": round(c_est, 5),
        "c_d": round(c_d, 1),
        "cd_e2s": round(cd_e2s, 2) if cd_e2s < 1e9 else float(f"{cd_e2s:.3e}"),
        "t_peak_hr": round(t_peak, 3),
        "der_peak_psi": round(der_peak, 2),
        "der_radial_est_psi": round(der_rad, 2),
        "t_end_wbs_est_hr": round(t_end_wbs_est, 2),
        "m": round(m_tds, 3),
        "k": round(k_tds, 2),
        "kh": round(k_tds * h_ft, 1),
        "skin": round(skin_tds, 3),
        "dP_skin": round(dp_skin_tds, 2),
    }


# ---------------------------------------------------------------------------
# 3. Method A: Nonlinear Analytical Model Matching (Stehfest WBS+Skin+Boundary)
# ---------------------------------------------------------------------------
def fit_analytical_type_curve(
    dt_grid: np.ndarray,
    dte_grid: np.ndarray,
    dp_grid: np.ndarray,
    der_ag: np.ndarray,
    tp_hr: float,
    qo_stbd: float,
    pvt_row: Mapping[str, Any],
    comp_row: Mapping[str, Any],
    init_k: Optional[float] = None,
    init_skin: Optional[float] = None,
    init_c: Optional[float] = None,
) -> Dict[str, Any]:
    """Fit exact Stehfest Laplace-space (WBS + Skin + Circular Closed Boundary) model
    simultaneously to Delta-P and Agarwal Bourdet Derivative.

    Works accurately even when NO flat radial-flow plateau exists in the shut-in!
    """
    bo, mu, h_ft, phi, ct, _ = extract_pta_rock_fluid(pvt_row, comp_row)
    area_raw = pvt_row.get("estimated_drainage_area_acres")
    if area_raw is not None and np.isfinite(float(area_raw)) and float(area_raw) > 0.0:
        re_prior = max(250.0, math.sqrt(float(area_raw) * 43560.0 / math.pi))
    else:
        k_est = float(init_k) if (init_k is not None and np.isfinite(init_k) and init_k > 0.0) else 50.0
        ri_est = 0.029 * math.sqrt(k_est * max(float(dt_grid[-1]), 1.0) / (phi * mu * ct))
        re_prior = max(3.0 * ri_est, 1000.0)

    valid_dp = np.isfinite(dp_grid) & (dp_grid > 0.5)
    valid_der = np.isfinite(der_ag) & (der_ag > 0.2)
    # Trim extreme 1st/last edge points from derivative matching (Bourdet edge bias)
    edge_mask = np.ones(len(dt_grid), dtype=bool)
    if len(dt_grid) > 10:
        edge_mask[0] = False
        edge_mask[-1] = False
    v_der_fit = valid_der & edge_mask

    # Smart initial guesses from TDS / drainage area
    k0_val = float(np.clip(init_k or 80.0, 2.0, 1500.0))
    s0_val = float(np.clip(init_skin if init_skin is not None else 1.0, -4.5, 20.0))
    c0_val = float(np.clip(init_c or 0.015, 1e-4, 1.0))

    # Parameters in unconstrained/log space: [log10(k), skin, log10(C), log10(re_ft)]
    x0 = np.array([math.log10(k0_val), s0_val, math.log10(c0_val), math.log10(re_prior)], dtype=np.float64)
    lb = np.array([math.log10(1.0), -5.5, math.log10(1e-5), math.log10(120.0)], dtype=np.float64)
    ub = np.array([math.log10(5000.0), 30.0, math.log10(5.0), math.log10(25000.0)], dtype=np.float64)
    x0 = np.clip(x0, lb + 1e-4, ub - 1e-4)

    log_dp_obs = np.log(dp_grid[valid_dp])
    log_der_obs = np.log(der_ag[v_der_fit])
    w_dp = 1.0 / math.sqrt(max(1, valid_dp.sum()))
    w_der = 1.35 / math.sqrt(max(1, v_der_fit.sum()))  # Slightly higher weight on derivative signature (Bourdet 2002)

    # Weak physical prior on r_e so short shut-ins that never feel boundaries don't drift erratically
    log_re_prior = math.log10(re_prior)

    def _residuals(x: np.ndarray) -> np.ndarray:
        k_try = 10.0 ** float(x[0])
        s_try = float(x[1])
        c_try = 10.0 ** float(x[2])
        re_try = 10.0 ** float(x[3])
        dp_m, der_m = analytical_buildup_dp_and_deriv(
            dt_grid, tp_hr, qo_stbd, k_try, s_try, c_try, re_try, pvt_row, comp_row,
            bourdet_L=0.15, use_agarwal=True,
        )
        r_dp = w_dp * (np.log(np.maximum(dp_m[valid_dp], 1e-4)) - log_dp_obs)
        r_der = w_der * (np.log(np.maximum(der_m[v_der_fit], 1e-4)) - log_der_obs)
        r_re = np.array([0.025 * (float(x[3]) - log_re_prior)], dtype=np.float64)
        return np.concatenate([r_dp, r_der, r_re])

    # Multi-start (3 diverse physical starts so blind No-IARF tests converge globally!)
    best_res = None
    starts = [
        x0,
        np.clip(np.array([math.log10(max(2.0, k0_val * 0.55)), s0_val - 1.5, math.log10(c0_val), log_re_prior]), lb + 1e-4, ub - 1e-4),
        np.clip(np.array([math.log10(min(2500.0, k0_val * 1.80)), s0_val + 2.0, math.log10(c0_val), log_re_prior]), lb + 1e-4, ub - 1e-4),
    ]
    for st in starts:
        res = least_squares(
            _residuals, st, bounds=(lb, ub), method="trf",
            diff_step=1e-4, ftol=1e-6, xtol=1e-6, max_nfev=90,
        )
        if best_res is None or res.cost < best_res.cost:
            best_res = res

    x_opt = best_res.x
    k_fit = float(10.0 ** x_opt[0])
    s_fit = float(x_opt[1])
    c_fit = float(10.0 ** x_opt[2])
    re_fit = float(10.0 ** x_opt[3])

    # Compute parameter uncertainties via Gauss-Newton covariance J^T J
    k_std_pct, s_std = None, None
    try:
        J = best_res.jac
        dof = max(1, len(best_res.fun) - len(x_opt))
        s2 = float(np.sum(best_res.fun ** 2)) / dof
        cov = np.linalg.pinv(J.T @ J) * s2
        se_logk = math.sqrt(max(0.0, float(cov[0, 0])))
        k_std_pct = min(99.0, round(math.log(10.0) * se_logk * 100.0, 1))
        s_std = min(20.0, round(math.sqrt(max(0.0, float(cov[1, 1]))), 2))
    except Exception:
        pass

    dp_fit, der_fit = analytical_buildup_dp_and_deriv(
        dt_grid, tp_hr, qo_stbd, k_fit, s_fit, c_fit, re_fit, pvt_row, comp_row,
        bourdet_L=0.15, use_agarwal=True,
    )

    # Fit quality metrics (percentage RMS error in log-space)
    dp_mape = float(np.mean(np.abs(dp_fit[valid_dp] - dp_grid[valid_dp]) / dp_grid[valid_dp]) * 100.0)
    der_mape = float(np.mean(np.abs(der_fit[v_der_fit] - der_ag[v_der_fit]) / der_ag[v_der_fit]) * 100.0)

    # True theoretical infinite-acting radial asymptote P'_radial = 70.6 * q * B * mu / (k * h)
    p_prime_radial = float(70.6 * qo_stbd * bo * mu / (k_fit * h_ft))
    m_fit = float(2.302585 * p_prime_radial)
    dp_skin_fit = float(0.869 * m_fit * s_fit)
    area_fit_ac = float(math.pi * re_fit * re_fit / 43560.0)

    # Build clean chart series for Log-Log overlay
    curve_dp = [{"x": round(float(t), 4), "y": round(float(v), 2)} for t, v in zip(dt_grid, dp_fit) if np.isfinite(v) and v > 0]
    curve_der = [{"x": round(float(t), 4), "y": round(float(v), 2)} for t, v in zip(dt_grid, der_fit) if np.isfinite(v) and v > 0]

    return {
        "available": True,
        "method": "Analytical Type-Curve Match (Stehfest WBS + Skin + Boundary)",
        "m": round(m_fit, 3),
        "k": round(k_fit, 2),
        "k_std_pct": k_std_pct,
        "kh": round(k_fit * h_ft, 1),
        "skin": round(s_fit, 3),
        "skin_std": s_std,
        "dP_skin": round(dp_skin_fit, 2),
        "c_bbl_psi": round(c_fit, 5),
        "re_ft": round(re_fit, 1),
        "drainage_area_acres": round(area_fit_ac, 1),
        "radial_deriv_asymptote_psi": round(p_prime_radial, 2),
        "dp_mape_pct": round(dp_mape, 2),
        "deriv_mape_pct": round(der_mape, 2),
        "curve_dp": curve_dp,
        "curve_der": curve_der,
    }


# ---------------------------------------------------------------------------
# 4. Method B: Regularized Pressure-Rate Deconvolution (von Schroeter, SPE-77688)
# ---------------------------------------------------------------------------
def _eval_log_spline_integral(tau: np.ndarray, knots_ln: np.ndarray, g_knots: np.ndarray) -> np.ndarray:
    """Integrate piecewise-linear-in-log-time derivative g(ln t) = e^z from tau_0 to tau."""
    ln_t = np.log(np.maximum(tau, math.exp(float(knots_ln[0]))))
    # Cumulative trapezoidal integral across knots
    dx = np.diff(knots_ln)
    seg_area = 0.5 * (g_knots[:-1] + g_knots[1:]) * dx
    cum = np.concatenate([[0.0], np.cumsum(seg_area)])

    idx = np.clip(np.searchsorted(knots_ln, ln_t, side="right") - 1, 0, len(knots_ln) - 2)
    x0 = knots_ln[idx]
    h = knots_ln[idx + 1] - x0
    frac = np.clip((ln_t - x0) / np.maximum(h, 1e-12), 0.0, 1.0)
    g0 = g_knots[idx]
    g1 = g_knots[idx + 1]
    g_at = g0 + frac * (g1 - g0)
    partial = 0.5 * (g0 + g_at) * (ln_t - x0)
    # Extrapolate flat beyond last knot if tau > knots_ln[-1]
    over = ln_t > knots_ln[-1]
    res = cum[idx] + partial
    if np.any(over):
        res[over] = cum[-1] + g_knots[-1] * (ln_t[over] - knots_ln[-1])
    return res


def run_vfm_pressure_rate_deconvolution(
    hf_well: pd.DataFrame,
    t_shut: pd.Timestamp,
    t_end: pd.Timestamp,
    qo_pre: float,
    pwf_psi: float,
    pvt_row: Mapping[str, Any],
    comp_row: Mapping[str, Any],
    vfm_daily_well: Optional[pd.DataFrame] = None,
    init_k: float = 80.0,
    lookback_days: int = 21,
) -> Dict[str, Any]:
    """von Schroeter (SPE-77688) Tikhonov-Regularized Pressure-Rate Deconvolution.

    Uses multi-week VFM flowing rate history q(t) + PDG pressure P(t) across both
    flowing periods and the build-up test to reconstruct the equivalent constant-rate
    unit drawdown Delta-P_u(tau) and deconvolved Bourdet derivative g(tau) beyond
    the shut-in duration.
    """
    bo, mu, h_ft, phi, ct, rw = extract_pta_rock_fluid(pvt_row, comp_row)

    ts_all = pd.to_datetime(hf_well["timestamp"], utc=True)
    t0_u = t_shut.tz_localize("UTC") if t_shut.tzinfo is None else t_shut.tz_convert("UTC")
    t1_u = t_end.tz_localize("UTC") if t_end.tzinfo is None else t_end.tz_convert("UTC")
    t_start_u = t0_u - pd.Timedelta(days=lookback_days)

    sub = hf_well[(ts_all >= t_start_u) & (ts_all <= t1_u)].copy()
    sub["_ts"] = pd.to_datetime(sub["timestamp"], utc=True)
    sub = sub.sort_values("_ts").reset_index(drop=True)
    if len(sub) < 60:
        return {"available": False, "note": "Insufficient pre-shut-in telemetry window for deconvolution."}

    # Build hourly rate steps from VFM daily table (or constant pre-shut-in rate scaled by choke)
    choke = pd.to_numeric(sub.get("choke_opening_pct", pd.Series(40.0, index=sub.index)), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    is_si = choke <= 0.5
    if vfm_daily_well is not None and not vfm_daily_well.empty:
        mw = vfm_daily_well.copy()
        mw["dkey"] = pd.to_datetime(mw["date"], utc=True).dt.floor("D")
        mw = mw.drop_duplicates("dkey").set_index("dkey")
        dkeys = sub["_ts"].dt.floor("D")
        up = np.maximum(dkeys.map(mw["uptime_frac"]).to_numpy(dtype=float), 0.05)
        q_col = "oil_rate_alloc_stbd" if "oil_rate_alloc_stbd" in mw.columns else "oil_rate_stbd"
        q_raw = dkeys.map(mw[q_col]).to_numpy(dtype=float) / up
        q_raw = np.where(np.isfinite(q_raw) & (q_raw > 0), q_raw, qo_pre)
    else:
        q_raw = np.full(len(sub), qo_pre, dtype=float)
    q_min = np.where(is_si, 0.0, q_raw)

    # Compress rate schedule into 6-hour flowing blocks + exact shut-in step change
    t_hr_all = (sub["_ts"] - t_start_u).dt.total_seconds().to_numpy(dtype=float) / 3600.0
    t_shut_rel = float((t0_u - t_start_u).total_seconds() / 3600.0)
    t_end_rel = float((t1_u - t_start_u).total_seconds() / 3600.0)
    dur_hr = max(0.5, t_end_rel - t_shut_rel)

    # Pre-shut-in blocks (6-hr chunks) + exact shut-in onset step to q = 0
    step_times: List[float] = [0.0]
    step_rates: List[float] = []
    pre_mask = t_hr_all < t_shut_rel
    block_edges = np.arange(0.0, t_shut_rel, 6.0)
    if len(block_edges) == 0 or block_edges[-1] < t_shut_rel - 0.5:
        block_edges = np.append(block_edges, t_shut_rel)
    else:
        block_edges[-1] = t_shut_rel

    prev_q = qo_pre
    q_steps_t = []
    q_steps_dq = []
    # Initial production start far earlier (pseudo-history at -t_hist) to anchor initial drawdown level
    t_hist_hr = min(2000.0, max(400.0, lookback_days * 24.0 * 2.5))
    q_steps_t.append(-t_hist_hr)
    q_steps_dq.append(qo_pre)
    last_q = qo_pre
    for b0, b1 in zip(block_edges[:-1], block_edges[1:]):
        m = pre_mask & (t_hr_all >= b0) & (t_hr_all < b1)
        if m.any():
            qb = float(np.median(q_min[m]))
            if abs(qb - last_q) > max(15.0, 0.03 * qo_pre):
                q_steps_t.append(float(b0))
                q_steps_dq.append(qb - last_q)
                last_q = qb
    # Exact shut-in step: rate drops from last_q -> 0 at t_shut_rel
    q_steps_t.append(t_shut_rel)
    q_steps_dq.append(-last_q)

    step_t = np.asarray(q_steps_t, dtype=np.float64)
    step_dq = np.asarray(q_steps_dq, dtype=np.float64)

    # Observation points: (a) 45 log-spaced build-up points (high weight) + (b) 40 pre-shut-in flowing points
    post = sub[sub["_ts"] > t0_u][["_ts", "dh_pressure_psia"]].dropna()
    if len(post) < 12:
        return {"available": False, "note": "Insufficient build-up points for deconvolution."}
    dt_post = (post["_ts"] - t0_u).dt.total_seconds().to_numpy(dtype=float) / 3600.0
    p_post = post["dh_pressure_psia"].to_numpy(dtype=float)
    bu_grid_dt = np.geomspace(max(dt_post[0], 1.0 / 60.0), dt_post[-1], min(48, len(dt_post)))
    bu_p = np.interp(np.log(bu_grid_dt), np.log(dt_post), p_post)
    bu_t_rel = t_shut_rel + bu_grid_dt

    pre_fl = sub[(sub["_ts"] <= t0_u) & (~is_si)][["_ts", "dh_pressure_psia"]].dropna()
    if len(pre_fl) >= 10:
        idx_pre = np.linspace(0, len(pre_fl) - 1, min(36, len(pre_fl)), dtype=int)
        pre_t_rel = (pre_fl["_ts"].iloc[idx_pre] - t_start_u).dt.total_seconds().to_numpy(dtype=float) / 3600.0
        pre_p = pre_fl["dh_pressure_psia"].iloc[idx_pre].to_numpy(dtype=float)
    else:
        pre_t_rel = np.array([t_shut_rel], dtype=np.float64)
        pre_p = np.array([pwf_psi], dtype=np.float64)

    obs_t = np.concatenate([pre_t_rel, bu_t_rel])
    obs_p = np.concatenate([pre_p, bu_p])
    # Weight build-up gauge measurements higher (zero tubing multiphase noise)
    obs_w = np.concatenate([np.full(len(pre_t_rel), 0.35), np.full(len(bu_t_rel), 1.50)])

    # Log-time derivative spline knots: 26 knots (~5/decade) from earliest buildup point up to extended horizon
    tau_min = max(0.015, float(bu_grid_dt[0]) * 0.85)
    tau_max = t_hist_hr + t_end_rel
    n_knots = 26
    knots_ln = np.linspace(math.log(tau_min), math.log(tau_max), n_knots)

    # Precompute lag matrices (exact Levitan/von Schroeter build-up Delta-P_ws + pre-shut-in P(t))
    lag_bu = bu_t_rel[:, None] - step_t[None, :]
    act_bu = lag_bu > 1e-4
    lag_0 = t_shut_rel - step_t
    act_0 = lag_0 > 1e-4
    lag_pre = pre_t_rel[:, None] - step_t[None, :]
    act_pre = lag_pre > 1e-4

    bu_dp_obs = np.maximum(bu_p - pwf_psi, 0.1)
    bu_der_obs = compute_bourdet_derivative(bu_grid_dt, bu_dp_obs, L=0.15)
    valid_bder = np.isfinite(bu_der_obs) & (bu_der_obs > 0.2)
    if len(valid_bder) > 6:
        valid_bder[0] = False
        valid_bder[-1] = False

    # Initial guess for unit-rate derivative g_u(tau) seeded from observed build-up derivative / qo_pre
    g_u_rad = max(1e-5, 70.6 * bo * mu / (max(init_k, 5.0) * h_ft))
    obs_g_u = np.where(valid_bder, bu_der_obs / max(qo_pre, 10.0), np.nan)
    ok_g = np.isfinite(obs_g_u) & (obs_g_u > 1e-6)
    if ok_g.sum() >= 5:
        g_seed = np.interp(knots_ln, np.log(bu_grid_dt[ok_g]), obs_g_u[ok_g], left=float(obs_g_u[ok_g][0]), right=g_u_rad)
        z0 = np.log(np.clip(g_seed, 1e-6, 0.8))
    else:
        z0 = np.full(n_knots, math.log(g_u_rad), dtype=np.float64)

    # c0 = Delta_P_u(tau_min) > 0 (early-time pressure drop up to tau_min)
    c0_init = float(np.clip(bu_dp_obs[0] / max(qo_pre, 10.0), 1e-5, 5.0))
    p0_init = float(np.max(bu_p)) + 80.0
    x0 = np.concatenate([[p0_init, c0_init], z0])

    lb = np.concatenate([[float(np.max(bu_p)) - 20.0, 0.0], np.full(n_knots, math.log(1e-6))])
    ub = np.concatenate([[float(np.max(bu_p)) + 2500.0, 10.0], np.full(n_knots, math.log(1.0))])
    x0 = np.clip(x0, lb + 1e-5, ub - 1e-5)

    # von Schroeter / Levitan regularized objective:
    # Strong build-up Delta-P + log-derivative fidelity + gentle pre-shut-in drawdown + smooth curvature
    lam_curv = 0.55
    p_scale = max(25.0, float(np.max(bu_dp_obs)))
    n_bu = max(1, len(bu_dp_obs))
    n_der = max(1, int(valid_bder.sum()))
    n_pre = max(1, len(pre_p))

    def _eval_dpu(lags: np.ndarray, c0_v: float, g_k: np.ndarray) -> np.ndarray:
        return c0_v + _eval_log_spline_integral(lags, knots_ln, g_k)

    def _buildup_and_pre_model(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        p0_v = float(x[0])
        c0_v = float(x[1])
        g_k = np.exp(x[2:])

        drop_0 = 0.0
        for j in range(len(step_t)):
            if act_0[j]:
                drop_0 += float(step_dq[j] * _eval_dpu(np.array([lag_0[j]]), c0_v, g_k)[0])

        drop_bu = np.zeros(len(bu_t_rel), dtype=np.float64)
        for j in range(len(step_t)):
            m = act_bu[:, j]
            if np.any(m):
                drop_bu[m] += step_dq[j] * _eval_dpu(lag_bu[m, j], c0_v, g_k)
        dp_bu_mod = drop_0 - drop_bu

        drop_pre = np.zeros(len(pre_t_rel), dtype=np.float64)
        for j in range(len(step_t)):
            m = act_pre[:, j]
            if np.any(m):
                drop_pre[m] += step_dq[j] * _eval_dpu(lag_pre[m, j], c0_v, g_k)
        p_pre_mod = p0_v - drop_pre
        return dp_bu_mod, p_pre_mod

    def _res_deconv(x: np.ndarray) -> np.ndarray:
        dp_bu_mod, p_pre_mod = _buildup_and_pre_model(x)
        r_bu = (2.2 / math.sqrt(n_bu)) * (dp_bu_mod - bu_dp_obs) / p_scale
        der_bu_mod = compute_bourdet_derivative(bu_grid_dt, dp_bu_mod, L=0.15)
        if np.any(valid_bder):
            r_der = (1.6 / math.sqrt(n_der)) * (np.log(np.maximum(der_bu_mod[valid_bder], 1e-3)) - np.log(bu_der_obs[valid_bder]))
        else:
            r_der = np.array([], dtype=np.float64)
        # Mild pre-shut-in flowing pressure anchor (prevents gauge/rate noise from distorting build-up derivative)
        r_pre = (0.35 / math.sqrt(n_pre)) * (p_pre_mod - pre_p) / max(p_scale, 150.0)
        z = x[2:]
        r_curv = lam_curv * (z[2:] - 2.0 * z[1:-1] + z[:-2])
        return np.concatenate([r_bu, r_der, r_pre, r_curv])

    sol = least_squares(_res_deconv, x0, bounds=(lb, ub), method="trf", diff_step=1e-4, ftol=1e-6, xtol=1e-6, max_nfev=85)
    x_opt = sol.x
    p0_fit = float(x_opt[0])
    c0_fit = float(x_opt[1])
    g_knots = np.exp(x_opt[2:])
    dp_bu_pred, _ = _buildup_and_pre_model(x_opt)
    bu_rmse_psi = float(np.sqrt(np.mean((dp_bu_pred - bu_dp_obs) ** 2)))

    # Reconstruct constant-rate equivalent drawdown Delta-P_eq(tau) = qo_pre * Delta-P_u(tau)
    # and deconvolved Bourdet derivative P'_eq(tau) = qo_pre * g(tau) on dense log grid
    tau_plot = np.geomspace(max(float(bu_grid_dt[0]), 0.025), min(tau_max, max(dur_hr * 6.0, 400.0)), 70)
    g_plot = np.exp(np.interp(np.log(tau_plot), knots_ln, x_opt[2:]))
    dpu_plot = c0_fit + _eval_log_spline_integral(tau_plot, knots_ln, g_knots)
    dp_eq = np.maximum(qo_pre * dpu_plot, 0.5)
    der_eq = np.maximum(qo_pre * g_plot, 0.2)

    # Locate post-storage-hump flattest radial plateau on the deconvolved derivative
    # Skip early WBS peak (max in first 40% of log-time within shut-in)
    in_si = tau_plot <= max(dur_hr, 2.0)
    si_idx = np.flatnonzero(in_si)
    t_after_pk = 0.35
    if len(si_idx) >= 6:
        pk_sub = si_idx[: max(3, int(0.55 * len(si_idx)))]
        t_after_pk = float(tau_plot[pk_sub[int(np.argmax(der_eq[pk_sub]))]]) * 2.2

    rad_mask = (tau_plot >= min(t_after_pk, dur_hr * 0.6)) & (tau_plot <= max(dur_hr * 1.8, 12.0))
    if rad_mask.sum() >= 5:
        r_idx = np.flatnonzero(rad_mask)
        lt = np.log10(tau_plot[r_idx])
        ld = np.log10(der_eq[r_idx])
        slp = np.abs(np.gradient(ld, lt))
        best_order = np.argsort(slp)[:max(3, min(7, len(r_idx) // 3))]
        i_sel = r_idx[best_order]
        g_iarf_psi = float(np.median(der_eq[i_sel]))
        t_ref = float(np.median(tau_plot[i_sel]))
        dp_ref = float(np.median(dp_eq[i_sel]))
    else:
        g_iarf_psi = float(np.median(der_eq))
        t_ref = float(np.median(tau_plot))
        dp_ref = float(np.median(dp_eq))

    k_dec = float(70.6 * qo_pre * bo * mu / (g_iarf_psi * h_ft))
    m_dec = float(2.302585 * g_iarf_psi)
    skin_dec = float(
        0.5 * ((dp_ref / g_iarf_psi) - math.log(max(1e-6, k_dec * t_ref / (phi * mu * ct * rw * rw))) + 7.4316)
    )
    dp_skin_dec = float(0.869 * m_dec * skin_dec)

    curve_der = [{"x": round(float(t), 4), "y": round(float(v), 2)} for t, v in zip(tau_plot, der_eq) if np.isfinite(v) and v > 0]
    curve_dp = [{"x": round(float(t), 4), "y": round(float(v), 2)} for t, v in zip(tau_plot, dp_eq) if np.isfinite(v) and v > 0]

    return {
        "available": True,
        "method": "Pressure-Rate Deconvolution (von Schroeter VFM-Coupled)",
        "p0_psia": round(p0_fit, 1),
        "bu_fit_rmse_psi": round(bu_rmse_psi, 2),
        "horizon_hr": round(float(tau_plot[-1]), 1),
        "n_rate_steps": int(len(step_t)),
        "m": round(m_dec, 3),
        "k": round(k_dec, 2),
        "kh": round(k_dec * h_ft, 1),
        "skin": round(skin_dec, 3),
        "dP_skin": round(dp_skin_dec, 2),
        "curve_der": curve_der,
        "curve_dp": curve_dp,
    }


# ---------------------------------------------------------------------------
# 5. Method D: Agarwal Equivalent-Time Bourdet & Semilog (SPE-9289)
# ---------------------------------------------------------------------------
def run_agarwal_equivalent_time_pta(
    dt_grid: np.ndarray,
    dte_grid: np.ndarray,
    dp_grid: np.ndarray,
    der_ag: np.ndarray,
    pwf_psi: float,
    tp_hr: float,
    qo_stbd: float,
    pvt_row: Mapping[str, Any],
    comp_row: Mapping[str, Any],
    ref_win_hr: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """Agarwal Equivalent-Time (SPE-9289) Bourdet + Theil-Sen Semilog analysis."""
    bo, mu, h_ft, phi, ct, rw = extract_pta_rock_fluid(pvt_row, comp_row)

    ok = np.isfinite(der_ag) & (der_ag > 0.2) & (dte_grid > 0)
    if ok.sum() < 6:
        return {"available": False, "note": "Insufficient Agarwal equivalent-time derivative points."}

    idx_ok = np.flatnonzero(ok)
    # Locate flattest post-hump window on Agarwal equivalent time (or honor reference window if wide)
    n_search = max(3, int(0.55 * len(idx_ok)))
    i_pk = int(idx_ok[int(np.argmax(der_ag[idx_ok[:n_search]]))])
    post = idx_ok[idx_ok > i_pk]
    if len(post) < 5:
        post = idx_ok[max(1, len(idx_ok) // 3):]

    # Sliding-window search (~0.45 log cycles) minimizing |log-slope| + 0.75 * CV on Agarwal derivative
    best_score = 1e18
    best_sel = post
    lt_all = np.log10(dte_grid)
    ld_all = np.log10(np.maximum(der_ag, 1e-6))
    for a in range(len(post)):
        t_a = lt_all[post[a]]
        cand = post[(lt_all[post] >= t_a) & (lt_all[post] <= t_a + 0.55)]
        if len(cand) < 4:
            continue
        # Exclude extreme late boundary collapse (< 40% of window start)
        if der_ag[cand[-1]] < 0.45 * der_ag[cand[0]]:
            continue
        slp = abs(float(np.polyfit(lt_all[cand], ld_all[cand], 1)[0]))
        cv = float(np.std(der_ag[cand]) / max(np.mean(der_ag[cand]), 1e-6))
        sc = slp + 0.8 * cv - 0.08 * (lt_all[cand[-1]] - lt_all[cand[0]])
        if sc < best_score:
            best_score = sc
            best_sel = cand

    if ref_win_hr and len(ref_win_hr) == 2:
        # Check if reference window has enough points and low slope; prefer wider clean segment
        ref_dte_lo = (tp_hr * ref_win_hr[0]) / (tp_hr + ref_win_hr[0])
        ref_dte_hi = (tp_hr * ref_win_hr[1]) / (tp_hr + ref_win_hr[1])
        ref_cand = idx_ok[(dte_grid[idx_ok] >= ref_dte_lo) & (dte_grid[idx_ok] <= ref_dte_hi)]
        if len(ref_cand) >= 5:
            best_sel = ref_cand

    x_ag = np.log10(dte_grid[best_sel])
    y_dp = dp_grid[best_sel]
    res_ts = theilslopes(y_dp, x_ag)
    m_ag = float(res_ts.slope)
    dp_1hr = float(res_ts.intercept)  # at log10(dt_e) = 0 => dt_e = 1 hr
    if not np.isfinite(m_ag) or m_ag <= 0.1:
        m_ag = float(2.302585 * np.median(der_ag[best_sel]))
        dp_1hr = float(np.median(y_dp - m_ag * x_ag))

    pred = dp_1hr + m_ag * x_ag
    ss_res = float(np.sum((y_dp - pred) ** 2))
    ss_tot = float(np.sum((y_dp - np.mean(y_dp)) ** 2))
    r2_ag = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-9 else 1.0

    k_ag = float(162.6 * qo_stbd * bo * mu / (m_ag * h_ft))
    skin_ag = float(1.1513 * ((dp_1hr / m_ag) - math.log10(max(1e-6, k_ag / (phi * mu * ct * rw * rw))) + 3.2275))
    dp_skin_ag = float(0.869 * m_ag * skin_ag)
    p_star_ag = float(pwf_psi + dp_1hr + m_ag * math.log10(tp_hr))

    curve_der = [
        {"x": round(float(te), 4), "y": round(float(dv), 2)}
        for te, dv in zip(dte_grid, der_ag)
        if np.isfinite(dv) and dv > 0
    ]
    return {
        "available": True,
        "method": "Agarwal Equivalent-Time Bourdet & Semilog (SPE-9289)",
        "win_hr": [round(float(dte_grid[best_sel[0]]), 3), round(float(dte_grid[best_sel[-1]]), 3)],
        "m": round(m_ag, 3),
        "p_star": round(p_star_ag, 1),
        "k": round(k_ag, 2),
        "kh": round(k_ag * h_ft, 1),
        "skin": round(skin_ag, 3),
        "dP_skin": round(dp_skin_ag, 2),
        "radial_deriv_asymptote_psi": round(m_ag / 2.302585, 2),
        "r2_semilog": round(max(-1.0, min(1.0, r2_ag)), 4),
        "curve_der": curve_der,
    }


# ---------------------------------------------------------------------------
# 6. Method E: Blasingame Pressure-Integral Derivative (SPE-19099, Noise-Immune)
# ---------------------------------------------------------------------------
def run_blasingame_pressure_integral(
    dt_1m: np.ndarray,
    p_raw: np.ndarray,
    pwf_psi: float,
    qo_stbd: float,
    pvt_row: Mapping[str, Any],
    comp_row: Mapping[str, Any],
) -> Dict[str, Any]:
    """Blasingame-Johnston-Lee (SPE-19099) Pressure-Integral Derivative Delta-P_id = Delta-P - Delta-P_i.

    Acts as a zero-phase physical low-pass filter eliminating gauge noise; in true
    radial flow Delta-P_id converges identically to P'_r = 70.6 * q * B * mu / (k * h).
    """
    bo, mu, h_ft, phi, ct, rw = extract_pta_rock_fluid(pvt_row, comp_row)

    ok = (dt_1m > 0) & np.isfinite(p_raw)
    dt = dt_1m[ok]
    dp = np.maximum(p_raw[ok] - pwf_psi, 0.1)
    if len(dt) < 12:
        return {"available": False, "note": "Insufficient data for Blasingame Pressure Integral."}

    # Exact cumulative trapezoid integral from t=0 (dp(0)=0)
    t_ext = np.concatenate([[0.0], dt])
    dp_ext = np.concatenate([[0.0], dp])
    seg_area = 0.5 * (dp_ext[:-1] + dp_ext[1:]) * np.diff(t_ext)
    cum_int = np.cumsum(seg_area)
    dp_i = cum_int / dt  # Pressure Integral
    dp_id = np.maximum(dp - dp_i, 0.1)  # Pressure-Integral Derivative (= t * d(dp_i)/dt)

    # Log-spaced resampling for plotting & plateau detection
    t_lo = max(float(dt[0]), 1.0 / 60.0)
    t_hi = float(dt[-1])
    n_pts = int(np.clip(round(max(0.5, math.log10(t_hi / t_lo)) * 22), 20, 90))
    t_g = np.geomspace(t_lo, t_hi, n_pts)
    dp_g = np.interp(np.log(t_g), np.log(dt), dp)
    dpi_g = np.interp(np.log(t_g), np.log(dt), dp_i)
    dpid_g = np.interp(np.log(t_g), np.log(dt), dp_id)

    # Second-order derivative of dp_id wrt ln(t) + lag-free Blasingame integral derivative P'_i_corr = dp_id + dp_idd
    dp_idd = compute_bourdet_derivative(t_g, dpid_g, L=0.15)
    # Exact identity: since dp_i + dp_id == dp, d(dp)/d(ln t) == dp_id + d(dp_id)/d(ln t) = dp_id + dp_idd
    # Computing via the integral suppresses 1-min gauge noise while eliminating the 2.7x integral lag!
    dp_id_corr = np.where(np.isfinite(dp_idd), np.maximum(dpid_g + dp_idd, 0.2), dpid_g)

    n_search = max(3, int(0.55 * len(t_g)))
    i_pk = int(np.argmax(dpid_g[:n_search]))
    # Start strictly AFTER the downward inflection of the storage hump (never pick the peak where slope=0!)
    after_pk = np.arange(i_pk + 1, max(i_pk + 3, int(0.85 * len(t_g))))
    if len(after_pk) >= 4:
        rel_after = dp_idd[after_pk] / np.maximum(dpid_g[after_pk], 0.5)
        rel_after = np.where(np.isfinite(rel_after), rel_after, 0.0)
        # Inflection point is the steepest descent (most negative rel_after)
        i_infl = int(after_pk[int(np.argmin(rel_after))])
        post = np.arange(max(i_infl + 1, i_pk + 3), max(i_infl + 4, int(0.88 * len(t_g))))
    else:
        post = np.arange(max(i_pk + 2, len(t_g) // 2), len(t_g))
    if len(post) < 3:
        post = np.arange(max(1, len(t_g) - 6), len(t_g))

    # Locate flattest segment on the noise-filtered integral derivative (avoid late boundary collapse)
    lt_p = np.log10(t_g[post])
    ld_p = np.log10(np.maximum(dp_id_corr[post], 0.2))
    slp_p = np.abs(np.gradient(ld_p, lt_p))
    # Reject tail where boundary drop pulls curve below 50% of early-post level
    early_lvl = float(np.median(dp_id_corr[post[:max(3, len(post) // 3)]]))
    valid_p = dp_id_corr[post] >= 0.45 * early_lvl
    cand = post[valid_p] if valid_p.sum() >= 3 else post
    slp_c = slp_p[valid_p] if valid_p.sum() >= 3 else slp_p

    best_local = int(np.argmin(slp_c))
    best_i = int(cand[best_local])
    lo_idx = max(int(cand[0]), best_i - 3)
    hi_idx = min(int(cand[-1]), best_i + 3)
    sel = np.arange(lo_idx, hi_idx + 1)

    p_pr_blas = float(np.median(dp_id_corr[sel]))
    t_ref = float(np.median(t_g[sel]))
    dp_ref = float(np.median(dp_g[sel]))
    k_bl = float(70.6 * qo_stbd * bo * mu / (p_pr_blas * h_ft))
    m_bl = float(2.302585 * p_pr_blas)
    skin_bl = float(0.5 * ((dp_ref / p_pr_blas) - math.log(max(1e-6, k_bl * t_ref / (phi * mu * ct * rw * rw))) + 7.4316))
    dp_skin_bl = float(0.869 * m_bl * skin_bl)

    curve_dpid = [{"x": round(float(t), 4), "y": round(float(v), 2)} for t, v in zip(t_g, dp_id_corr) if np.isfinite(v) and v > 0]
    curve_dpi = [{"x": round(float(t), 4), "y": round(float(v), 2)} for t, v in zip(t_g, dpi_g) if np.isfinite(v) and v > 0]

    return {
        "available": True,
        "method": "Blasingame Pressure-Integral Derivative ΔP_id (SPE-19099)",
        "win_hr": [round(float(t_g[sel[0]]), 3), round(float(t_g[sel[-1]]), 3)],
        "m": round(m_bl, 3),
        "k": round(k_bl, 2),
        "kh": round(k_bl * h_ft, 1),
        "skin": round(skin_bl, 3),
        "dP_skin": round(dp_skin_bl, 2),
        "radial_deriv_asymptote_psi": round(p_pr_blas, 2),
        "curve_der": curve_dpid,
        "curve_dp": curve_dpi,
    }


# ---------------------------------------------------------------------------
# 7. Method F: Ilk-Valko-Blasingame Convex B-Spline Deconvolution (SPE-102869)
# ---------------------------------------------------------------------------
def run_ilk_bspline_deconvolution(
    hf_well: pd.DataFrame,
    t_shut: pd.Timestamp,
    t_end: pd.Timestamp,
    qo_pre: float,
    pwf_psi: float,
    pvt_row: Mapping[str, Any],
    comp_row: Mapping[str, Any],
    vfm_daily_well: Optional[pd.DataFrame] = None,
    lookback_days: int = 21,
) -> Dict[str, Any]:
    """Ilk-Valko-Blasingame (SPE-102869) Convex Quadratic B-Spline Deconvolution.

    Expands unit-rate derivative directly on non-negative quadratic B-splines B_j(ln tau)
    and solves a strictly convex bounded linear least-squares problem (ZERO local minima!).
    """
    bo, mu, h_ft, phi, ct, rw = extract_pta_rock_fluid(pvt_row, comp_row)

    ts_all = pd.to_datetime(hf_well["timestamp"], utc=True)
    t0_u = t_shut.tz_localize("UTC") if t_shut.tzinfo is None else t_shut.tz_convert("UTC")
    t1_u = t_end.tz_localize("UTC") if t_end.tzinfo is None else t_end.tz_convert("UTC")
    t_start_u = t0_u - pd.Timedelta(days=lookback_days)
    sub = hf_well[(ts_all >= t_start_u) & (ts_all <= t1_u)].copy()
    sub["_ts"] = pd.to_datetime(sub["timestamp"], utc=True)
    sub = sub.sort_values("_ts").reset_index(drop=True)
    post = sub[sub["_ts"] > t0_u][["_ts", "dh_pressure_psia"]].dropna()
    if len(post) < 12:
        return {"available": False, "note": "Insufficient build-up points for B-spline deconvolution."}

    choke = pd.to_numeric(sub.get("choke_opening_pct", pd.Series(40.0, index=sub.index)), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    is_si = choke <= 0.5
    if vfm_daily_well is not None and not vfm_daily_well.empty:
        mw = vfm_daily_well.copy()
        mw["dkey"] = pd.to_datetime(mw["date"], utc=True).dt.floor("D")
        mw = mw.drop_duplicates("dkey").set_index("dkey")
        dkeys = sub["_ts"].dt.floor("D")
        up = np.maximum(dkeys.map(mw["uptime_frac"]).to_numpy(dtype=float), 0.05)
        q_col = "oil_rate_alloc_stbd" if "oil_rate_alloc_stbd" in mw.columns else "oil_rate_stbd"
        q_raw = dkeys.map(mw[q_col]).to_numpy(dtype=float) / up
        q_raw = np.where(np.isfinite(q_raw) & (q_raw > 0), q_raw, qo_pre)
    else:
        q_raw = np.full(len(sub), qo_pre, dtype=float)
    q_min = np.where(is_si, 0.0, q_raw)

    t_hr_all = (sub["_ts"] - t_start_u).dt.total_seconds().to_numpy(dtype=float) / 3600.0
    t_shut_rel = float((t0_u - t_start_u).total_seconds() / 3600.0)
    t_end_rel = float((t1_u - t_start_u).total_seconds() / 3600.0)
    dur_hr = max(0.5, t_end_rel - t_shut_rel)

    # Variable-rate schedule
    t_hist_hr = min(2000.0, max(400.0, lookback_days * 24.0 * 2.5))
    q_steps_t, q_steps_dq = [-t_hist_hr], [qo_pre]
    last_q = qo_pre
    pre_mask = t_hr_all < t_shut_rel
    edges = np.arange(0.0, t_shut_rel, 6.0)
    if len(edges) == 0 or edges[-1] < t_shut_rel - 0.5:
        edges = np.append(edges, t_shut_rel)
    else:
        edges[-1] = t_shut_rel
    for b0, b1 in zip(edges[:-1], edges[1:]):
        m = pre_mask & (t_hr_all >= b0) & (t_hr_all < b1)
        if m.any():
            qb = float(np.median(q_min[m]))
            if abs(qb - last_q) > max(15.0, 0.03 * qo_pre):
                q_steps_t.append(float(b0))
                q_steps_dq.append(qb - last_q)
                last_q = qb
    q_steps_t.append(t_shut_rel)
    q_steps_dq.append(-last_q)
    step_t = np.asarray(q_steps_t, dtype=np.float64)
    step_dq = np.asarray(q_steps_dq, dtype=np.float64)

    dt_post = (post["_ts"] - t0_u).dt.total_seconds().to_numpy(dtype=float) / 3600.0
    p_post = post["dh_pressure_psia"].to_numpy(dtype=float)
    bu_dt = np.geomspace(max(dt_post[0], 1.0 / 60.0), dt_post[-1], min(55, len(dt_post)))
    bu_dp = np.maximum(np.interp(np.log(bu_dt), np.log(dt_post), p_post) - pwf_psi, 0.1)
    bu_der = compute_bourdet_derivative(bu_dt, bu_dp, L=0.15)
    bu_t_rel = t_shut_rel + bu_dt

    # Quadratic B-spline basis (degree k=2) on ln(tau) over [tau_min, tau_max]
    tau_min = max(0.015, float(bu_dt[0]) * 0.85)
    tau_max = t_hist_hr + t_end_rel
    ln_lo, ln_hi = math.log(tau_min), math.log(tau_max)
    n_int = 18
    deg = 2
    inner = np.linspace(ln_lo, ln_hi, n_int)
    t_knots = np.concatenate([np.full(deg, ln_lo), inner, np.full(deg, ln_hi)])
    n_basis = len(t_knots) - deg - 1

    # Precompute dense antiderivative table for each B-spline basis function B_j(sigma)
    sig_dense = np.linspace(ln_lo, ln_hi, 320)
    dsig = sig_dense[1] - sig_dense[0]
    basis_vals = np.zeros((n_basis, len(sig_dense)), dtype=np.float64)
    basis_ints = np.zeros((n_basis, len(sig_dense)), dtype=np.float64)
    for j in range(n_basis):
        coeff = np.zeros(n_basis, dtype=np.float64)
        coeff[j] = 1.0
        sp = BSpline(t_knots, coeff, deg, extrapolate=False)
        bv = np.nan_to_num(sp(sig_dense), nan=0.0)
        basis_vals[j] = bv
        basis_ints[j, 1:] = np.cumsum(0.5 * (bv[:-1] + bv[1:]) * dsig)

    def _eval_int_mat(lags: np.ndarray) -> np.ndarray:
        ls = np.clip(np.log(np.maximum(lags, tau_min)), ln_lo, ln_hi)
        M = np.empty((len(ls), n_basis), dtype=np.float64)
        for j in range(n_basis):
            M[:, j] = np.interp(ls, sig_dense, basis_ints[j])
        return M

    # Build linear design matrix Phi for Delta-P_ws(dt_k) = c0 * last_q + sum_j w_j * Phi_{k, j}
    n_obs = len(bu_dt)
    Phi = np.zeros((n_obs, n_basis + 1), dtype=np.float64)
    # Column 0: c0 contribution = sum(dq_j for t_j < t_shut) - sum(dq_j for t_j < t_k) == last_q!
    Phi[:, 0] = last_q

    # Exact superposition of basis integrals: drop_0 - drop_bu
    act_0 = (t_shut_rel - step_t) > 1e-4
    row_0 = np.zeros(n_basis, dtype=np.float64)
    for m_step in np.flatnonzero(act_0):
        row_0 += step_dq[m_step] * _eval_int_mat(np.array([t_shut_rel - step_t[m_step]]))[0]

    for m_step in range(len(step_t)):
        lags = bu_t_rel - step_t[m_step]
        act = lags > 1e-4
        if np.any(act):
            Phi[act, 1:] -= step_dq[m_step] * _eval_int_mat(lags[act])
    Phi[:, 1:] += row_0[None, :]

    # Derivative-matching rows: d(Delta-P_ws)/d(ln dt) ≈ last_q * g_u(dt)
    valid_d = np.isfinite(bu_der) & (bu_der > 0.2)
    if len(valid_d) > 6:
        valid_d[0] = False
        valid_d[-1] = False
    n_vd = int(valid_d.sum())
    Phi_d = np.zeros((n_vd, n_basis + 1), dtype=np.float64)
    ls_vd = np.clip(np.log(bu_dt[valid_d]), ln_lo, ln_hi)
    for j in range(n_basis):
        Phi_d[:, j + 1] = last_q * np.interp(ls_vd, sig_dense, basis_vals[j])

    # Second-order difference Tikhonov curvature penalty on B-spline weights w_j
    p_scale = max(25.0, float(np.max(bu_dp)))
    A_bu = (1.8 / (math.sqrt(n_obs) * p_scale)) * Phi
    b_bu = (1.8 / (math.sqrt(n_obs) * p_scale)) * bu_dp

    w_der_scale = np.maximum(bu_der[valid_d], 2.0)
    A_der = (1.25 / math.sqrt(max(1, n_vd))) * (Phi_d / w_der_scale[:, None])
    b_der = (1.25 / math.sqrt(max(1, n_vd))) * (bu_der[valid_d] / w_der_scale)

    D2 = np.zeros((n_basis - 2, n_basis + 1), dtype=np.float64)
    for j in range(n_basis - 2):
        D2[j, j + 1] = 1.0
        D2[j, j + 2] = -2.0
        D2[j, j + 3] = 1.0
    # Properly scaled curvature penalty (same order as normalized data residuals!)
    lam_bs = 0.65 * (qo_pre / max(float(np.median(bu_der[valid_d])) if n_vd else p_scale, 5.0)) / math.sqrt(n_basis)
    A_reg = lam_bs * D2
    b_reg = np.zeros(n_basis - 2, dtype=np.float64)

    A_tot = np.vstack([A_bu, A_der, A_reg])
    b_tot = np.concatenate([b_bu, b_der, b_reg])

    # Strictly convex non-negative bounded linear least-squares solve!
    sol = lsq_linear(A_tot, b_tot, bounds=(0.0, np.inf), method="trf")
    c0_fit = float(sol.x[0])
    w_fit = sol.x[1:]
    bu_pred = Phi @ sol.x
    bu_rmse_psi = float(np.sqrt(np.mean((bu_pred - bu_dp) ** 2)))

    # Reconstruct deconvolved constant-rate drawdown & derivative out to 400+ hr
    tau_plot = np.geomspace(max(float(bu_dt[0]), 0.025), min(tau_max, max(dur_hr * 6.0, 400.0)), 70)
    ls_p = np.clip(np.log(tau_plot), ln_lo, ln_hi)
    g_u_plot = np.zeros(len(tau_plot), dtype=np.float64)
    int_u_plot = np.zeros(len(tau_plot), dtype=np.float64)
    for j in range(n_basis):
        g_u_plot += w_fit[j] * np.interp(ls_p, sig_dense, basis_vals[j])
        int_u_plot += w_fit[j] * np.interp(ls_p, sig_dense, basis_ints[j])
    der_eq = np.maximum(qo_pre * g_u_plot, 0.2)
    dp_eq = np.maximum(qo_pre * (c0_fit + int_u_plot), 0.5)

    # Locate flattest post-hump radial plateau on B-spline deconvolved derivative
    in_si = tau_plot <= max(dur_hr, 2.0)
    si_idx = np.flatnonzero(in_si)
    t_after_pk = 0.35
    if len(si_idx) >= 6:
        pk_sub = si_idx[: max(3, int(0.55 * len(si_idx)))]
        t_after_pk = float(tau_plot[pk_sub[int(np.argmax(der_eq[pk_sub]))]]) * 2.0

    rad_mask = (tau_plot >= min(t_after_pk, dur_hr * 0.6)) & (tau_plot <= max(dur_hr * 1.8, 12.0))
    r_idx = np.flatnonzero(rad_mask) if rad_mask.sum() >= 5 else np.arange(len(tau_plot) // 3, len(tau_plot))
    slp = np.abs(np.gradient(np.log10(der_eq[r_idx]), np.log10(tau_plot[r_idx])))
    best_ord = np.argsort(slp)[:max(3, min(7, len(r_idx) // 3))]
    i_sel = r_idx[best_ord]
    g_iarf = float(np.median(der_eq[i_sel]))
    t_ref = float(np.median(tau_plot[i_sel]))
    dp_ref = float(np.median(dp_eq[i_sel]))

    k_ilk = float(70.6 * qo_pre * bo * mu / (g_iarf * h_ft))
    m_ilk = float(2.302585 * g_iarf)
    skin_ilk = float(0.5 * ((dp_ref / g_iarf) - math.log(max(1e-6, k_ilk * t_ref / (phi * mu * ct * rw * rw))) + 7.4316))
    dp_skin_ilk = float(0.869 * m_ilk * skin_ilk)

    curve_der = [{"x": round(float(t), 4), "y": round(float(v), 2)} for t, v in zip(tau_plot, der_eq) if np.isfinite(v) and v > 0]
    curve_dp = [{"x": round(float(t), 4), "y": round(float(v), 2)} for t, v in zip(tau_plot, dp_eq) if np.isfinite(v) and v > 0]

    return {
        "available": True,
        "method": "Ilk-Valkó-Blasingame Convex B-Spline Deconvolution (SPE-102869)",
        "bu_fit_rmse_psi": round(bu_rmse_psi, 2),
        "horizon_hr": round(float(tau_plot[-1]), 1),
        "n_basis": int(n_basis),
        "m": round(m_ilk, 3),
        "k": round(k_ilk, 2),
        "kh": round(k_ilk * h_ft, 1),
        "skin": round(skin_ilk, 3),
        "dP_skin": round(dp_skin_ilk, 2),
        "radial_deriv_asymptote_psi": round(g_iarf, 2),
        "curve_der": curve_der,
        "curve_dp": curve_dp,
    }


# ---------------------------------------------------------------------------
# 8. Method G: Odeh-Jones Multi-Rate Superposition (SPE-1082)
# ---------------------------------------------------------------------------
def run_odeh_jones_multirate_superposition(
    hf_well: pd.DataFrame,
    t_shut: pd.Timestamp,
    t_end: pd.Timestamp,
    qo_pre: float,
    pwf_psi: float,
    tp_hr: float,
    pvt_row: Mapping[str, Any],
    comp_row: Mapping[str, Any],
    vfm_daily_well: Optional[pd.DataFrame] = None,
    ref_win_hr: Optional[List[float]] = None,
    lookback_days: int = 21,
) -> Dict[str, Any]:
    """Odeh-Jones (SPE-1082) Variable-Rate Superposition Build-Up Analysis."""
    bo, mu, h_ft, phi, ct, rw = extract_pta_rock_fluid(pvt_row, comp_row)

    ts_all = pd.to_datetime(hf_well["timestamp"], utc=True)
    t0_u = t_shut.tz_localize("UTC") if t_shut.tzinfo is None else t_shut.tz_convert("UTC")
    t1_u = t_end.tz_localize("UTC") if t_end.tzinfo is None else t_end.tz_convert("UTC")
    t_start_u = t0_u - pd.Timedelta(days=lookback_days)
    sub = hf_well[(ts_all >= t_start_u) & (ts_all <= t1_u)].copy()
    sub["_ts"] = pd.to_datetime(sub["timestamp"], utc=True)
    sub = sub.sort_values("_ts").reset_index(drop=True)
    post = sub[sub["_ts"] > t0_u][["_ts", "dh_pressure_psia"]].dropna()
    if len(post) < 12:
        return {"available": False, "note": "Insufficient build-up points for Odeh-Jones superposition."}

    choke = pd.to_numeric(sub.get("choke_opening_pct", pd.Series(40.0, index=sub.index)), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    is_si = choke <= 0.5
    if vfm_daily_well is not None and not vfm_daily_well.empty:
        mw = vfm_daily_well.copy()
        mw["dkey"] = pd.to_datetime(mw["date"], utc=True).dt.floor("D")
        mw = mw.drop_duplicates("dkey").set_index("dkey")
        dkeys = sub["_ts"].dt.floor("D")
        up = np.maximum(dkeys.map(mw["uptime_frac"]).to_numpy(dtype=float), 0.05)
        q_col = "oil_rate_alloc_stbd" if "oil_rate_alloc_stbd" in mw.columns else "oil_rate_stbd"
        q_raw = dkeys.map(mw[q_col]).to_numpy(dtype=float) / up
        q_raw = np.where(np.isfinite(q_raw) & (q_raw > 0), q_raw, qo_pre)
    else:
        q_raw = np.full(len(sub), qo_pre, dtype=float)
    q_min = np.where(is_si, 0.0, q_raw)

    t_hr_all = (sub["_ts"] - t_start_u).dt.total_seconds().to_numpy(dtype=float) / 3600.0
    t_shut_rel = float((t0_u - t_start_u).total_seconds() / 3600.0)
    pre_mask = t_hr_all < t_shut_rel
    edges = np.arange(0.0, t_shut_rel, 12.0)
    if len(edges) == 0 or edges[-1] < t_shut_rel - 0.5:
        edges = np.append(edges, t_shut_rel)
    else:
        edges[-1] = t_shut_rel

    # Build variable-rate steps anchored by full historical producing duration tp_hr
    step_t_list = [t_shut_rel - tp_hr]
    step_dq_list = [qo_pre]
    last_q = qo_pre
    for b0, b1 in zip(edges[:-1], edges[1:]):
        m = pre_mask & (t_hr_all >= b0) & (t_hr_all < b1)
        if m.any():
            qb = float(np.median(q_min[m]))
            if abs(qb - last_q) > max(10.0, 0.02 * qo_pre):
                step_t_list.append(float(b0))
                step_dq_list.append(qb - last_q)
                last_q = qb
    step_t_list.append(t_shut_rel)
    step_dq_list.append(-last_q)
    st_t = np.asarray(step_t_list, dtype=np.float64)
    st_dq = np.asarray(step_dq_list, dtype=np.float64)
    q_ref = max(last_q, 10.0)

    dt_post = (post["_ts"] - t0_u).dt.total_seconds().to_numpy(dtype=float) / 3600.0
    p_post = post["dh_pressure_psia"].to_numpy(dtype=float)
    t_post_rel = t_shut_rel + dt_post

    # Exact Odeh-Jones superposition time X_OJ(t) = sum_j (dq_j / q_ref) * log10(t - t_j)
    lags = np.maximum(t_post_rel[:, None] - st_t[None, :], 1e-6)
    x_oj = np.sum((st_dq[None, :] / q_ref) * np.log10(lags), axis=1)

    if ref_win_hr and len(ref_win_hr) == 2:
        sel = (dt_post >= ref_win_hr[0]) & (dt_post <= ref_win_hr[1])
    else:
        sel = (dt_post >= max(0.4, dt_post[-1] * 0.12)) & (dt_post <= dt_post[-1] * 0.70)
    if sel.sum() < 5:
        sel = np.arange(len(dt_post)) >= len(dt_post) // 3

    res_ts = theilslopes(p_post[sel], x_oj[sel])
    m_oj = float(-res_ts.slope)
    p_star_oj = float(res_ts.intercept)
    if not np.isfinite(m_oj) or m_oj <= 0.1:
        return {"available": False, "note": "Non-positive Odeh-Jones superposition slope."}

    pred = p_star_oj - m_oj * x_oj[sel]
    ss_res = float(np.sum((p_post[sel] - pred) ** 2))
    ss_tot = float(np.sum((p_post[sel] - np.mean(p_post[sel])) ** 2))
    r2_oj = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-9 else 1.0

    # Evaluate X_OJ at dt = 1 hr for exact multi-rate skin calculation
    lags_1h = np.maximum((t_shut_rel + 1.0) - st_t, 1e-6)
    x_oj_1h = float(np.sum((st_dq / q_ref) * np.log10(lags_1h)))
    p1hr = float(p_star_oj - m_oj * x_oj_1h)

    k_oj = float(162.6 * q_ref * bo * mu / (m_oj * h_ft))
    skin_oj = float(1.1513 * (((p1hr - pwf_psi) / m_oj) - math.log10(max(1e-6, k_oj / (phi * mu * ct * rw * rw))) + 3.2275))
    dp_skin_oj = float(0.869 * m_oj * skin_oj)

    return {
        "available": True,
        "method": "Odeh-Jones Multi-Rate Superposition (SPE-1082)",
        "n_rate_steps": int(len(st_t)),
        "q_ref_stbd": round(q_ref, 1),
        "win_hr": [round(float(dt_post[sel][0]), 2), round(float(dt_post[sel][-1]), 2)],
        "m": round(m_oj, 3),
        "p_star": round(p_star_oj, 1),
        "k": round(k_oj, 2),
        "kh": round(k_oj * h_ft, 1),
        "skin": round(skin_oj, 3),
        "dP_skin": round(dp_skin_oj, 2),
        "radial_deriv_asymptote_psi": round(m_oj / 2.302585, 2),
        "r2_superpos": round(max(-1.0, min(1.0, r2_oj)), 4),
    }


# ---------------------------------------------------------------------------
# 9. Method H: MDH Semilog + Muskat Late-Time Bounded Extrapolation (SPE-950)
# ---------------------------------------------------------------------------
def run_mdh_muskat_analysis(
    dt_1m: np.ndarray,
    p_raw: np.ndarray,
    pwf_psi: float,
    qo_stbd: float,
    pvt_row: Mapping[str, Any],
    comp_row: Mapping[str, Any],
    ref_win_hr: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """Miller-Dyes-Hutchinson (MDH) Semilog + Muskat Late-Time Bounded Compartment Extrapolation."""
    bo, mu, h_ft, phi, ct, rw = extract_pta_rock_fluid(pvt_row, comp_row)

    ok = (dt_1m > 0) & np.isfinite(p_raw)
    dt, pws = dt_1m[ok], p_raw[ok]
    if len(dt) < 12:
        return {"available": False, "note": "Insufficient build-up points for MDH / Muskat."}

    if ref_win_hr and len(ref_win_hr) == 2:
        sel = (dt >= ref_win_hr[0]) & (dt <= ref_win_hr[1])
    else:
        sel = (dt >= max(0.4, dt[-1] * 0.12)) & (dt <= dt[-1] * 0.65)
    if sel.sum() < 5:
        sel = np.arange(len(dt)) >= len(dt) // 3

    x_mdh = np.log10(dt[sel])
    res_ts = theilslopes(pws[sel], x_mdh)
    m_mdh = float(res_ts.slope)
    p1hr = float(res_ts.intercept)
    if not np.isfinite(m_mdh) or m_mdh <= 0.1:
        return {"available": False, "note": "Non-positive MDH slope."}

    k_mdh = float(162.6 * qo_stbd * bo * mu / (m_mdh * h_ft))
    skin_mdh = float(1.1513 * (((p1hr - pwf_psi) / m_mdh) - math.log10(max(1e-6, k_mdh / (phi * mu * ct * rw * rw))) + 3.2275))
    dp_skin_mdh = float(0.869 * m_mdh * skin_mdh)

    # Muskat late-time exponential extrapolation: ln(p_bar - P_ws) = a - beta * dt on final 45% of shut-in
    late_mask = dt >= (0.45 * dt[-1])
    p_bar_muskat, muskat_r2 = None, None
    if late_mask.sum() >= 10:
        dt_l = dt[late_mask]
        p_l = pws[late_mask]
        p_max = float(np.max(p_l))

        def _neg_r2(pbar_try: float) -> float:
            diff = pbar_try - p_l
            if np.any(diff <= 0.05):
                return 1.0
            y = np.log(diff)
            c = np.polyfit(dt_l, y, 1)
            pred = c[0] * dt_l + c[1]
            ss_r = float(np.sum((y - pred) ** 2))
            ss_t = float(np.sum((y - np.mean(y)) ** 2))
            return (ss_r / ss_t) if ss_t > 1e-9 else 1.0

        opt = minimize_scalar(_neg_r2, bounds=(p_max + 0.5, p_max + 450.0), method="bounded")
        if opt.success and opt.fun < 0.25:
            p_bar_muskat = round(float(opt.x), 1)
            muskat_r2 = round(float(1.0 - opt.fun), 4)

    return {
        "available": True,
        "method": "MDH Semilog + Muskat Late-Time Extrapolation (SPE-950)",
        "win_hr": [round(float(dt[sel][0]), 2), round(float(dt[sel][-1]), 2)],
        "m": round(m_mdh, 3),
        "p1hr": round(p1hr, 1),
        "k": round(k_mdh, 2),
        "kh": round(k_mdh * h_ft, 1),
        "skin": round(skin_mdh, 3),
        "dP_skin": round(dp_skin_mdh, 2),
        "radial_deriv_asymptote_psi": round(m_mdh / 2.302585, 2),
        "p_bar_muskat_psia": p_bar_muskat,
        "muskat_r2": muskat_r2,
    }


# ---------------------------------------------------------------------------
# 10. Master Orchestrator + Regime Triage for Any Build-Up (Clean or No-IARF)
# ---------------------------------------------------------------------------
def run_advanced_non_iarf_suite(
    hf_well: pd.DataFrame,
    t_shut: pd.Timestamp,
    t_end: pd.Timestamp,
    qo_stbd: float,
    pwf_psi: float,
    tp_hr: float,
    pvt_row: Mapping[str, Any],
    comp_row: Mapping[str, Any],
    vfm_daily_well: Optional[pd.DataFrame] = None,
    hyb_summary: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Run the complete globally recognized non-IARF & deconvolution PTA suite."""
    ts = pd.to_datetime(hf_well["timestamp"], utc=True)
    t0_u = t_shut.tz_localize("UTC") if t_shut.tzinfo is None else t_shut.tz_convert("UTC")
    t1_u = t_end.tz_localize("UTC") if t_end.tzinfo is None else t_end.tz_convert("UTC")
    post = hf_well[(ts > t0_u) & (ts <= t1_u)][["timestamp", "dh_pressure_psia"]].dropna()
    if len(post) < 10:
        return {"available": False, "note": "Insufficient build-up points."}

    dt_1m = (pd.to_datetime(post["timestamp"], utc=True) - t0_u).dt.total_seconds().to_numpy(dtype=float) / 3600.0
    p_raw = post["dh_pressure_psia"].to_numpy(dtype=float)
    dur_hr = float(dt_1m[-1])

    dt_grid, dte_grid, dp_grid, der_raw, der_ag = _preprocess_log_grid(dt_1m, p_raw, pwf_psi, tp_hr)
    ref_win = (hyb_summary or {}).get("intersect_win_hr")

    # 1. Tiab Direct Synthesis (TDS, SPE-26138)
    tds = run_tiab_direct_synthesis(dt_grid, dte_grid, dp_grid, der_ag, qo_stbd, pvt_row, comp_row)

    # 2. Agarwal Equivalent-Time Bourdet & Semilog (SPE-9289)
    agarwal = run_agarwal_equivalent_time_pta(
        dt_grid, dte_grid, dp_grid, der_ag, pwf_psi, tp_hr, qo_stbd, pvt_row, comp_row, ref_win_hr=ref_win
    )

    # 3. Blasingame Pressure-Integral Derivative Delta-P_id (SPE-19099, Noise-Immune)
    blasingame = run_blasingame_pressure_integral(dt_1m, p_raw, pwf_psi, qo_stbd, pvt_row, comp_row)

    # 4. Stehfest 12-Term Nonlinear Analytical Type-Curve Match (SPE-12777 / Bourdet 2002)
    init_k = (hyb_summary or {}).get("logic3_intersection", {}).get("k") or (tds.get("k") if tds.get("available") else 80.0)
    init_s = (hyb_summary or {}).get("logic3_intersection", {}).get("skin")
    if init_s is None and tds.get("available"):
        init_s = tds.get("skin")
    init_c = tds.get("c_bbl_psi") if tds.get("available") else 0.015

    analytical = fit_analytical_type_curve(
        dt_grid, dte_grid, dp_grid, der_ag, tp_hr, qo_stbd, pvt_row, comp_row,
        init_k=init_k, init_skin=init_s, init_c=init_c,
    )

    # 5. von Schroeter VFM-Coupled Nonlinear Pressure-Rate Deconvolution (SPE-77688)
    try:
        deconv = run_vfm_pressure_rate_deconvolution(
            hf_well, t_shut, t_end, qo_stbd, pwf_psi, pvt_row, comp_row,
            vfm_daily_well=vfm_daily_well, init_k=analytical["k"],
        )
    except Exception as exc:
        deconv = {"available": False, "note": f"von Schroeter deconvolution skipped: {exc}"}

    # 6. Ilk-Valkó-Blasingame Convex B-Spline Deconvolution (SPE-102869)
    try:
        ilk_deconv = run_ilk_bspline_deconvolution(
            hf_well, t_shut, t_end, qo_stbd, pwf_psi, pvt_row, comp_row,
            vfm_daily_well=vfm_daily_well,
        )
    except Exception as exc:
        ilk_deconv = {"available": False, "note": f"Ilk B-spline deconvolution skipped: {exc}"}

    # 7. Odeh-Jones Multi-Rate Superposition (SPE-1082)
    try:
        odeh_jones = run_odeh_jones_multirate_superposition(
            hf_well, t_shut, t_end, qo_stbd, pwf_psi, tp_hr, pvt_row, comp_row,
            vfm_daily_well=vfm_daily_well, ref_win_hr=ref_win,
        )
    except Exception as exc:
        odeh_jones = {"available": False, "note": f"Odeh-Jones skipped: {exc}"}

    # 8. MDH Semilog + Muskat Late-Time Extrapolation (SPE-950 / SPE-1244)
    try:
        mdh_muskat = run_mdh_muskat_analysis(dt_1m, p_raw, pwf_psi, qo_stbd, pvt_row, comp_row, ref_win_hr=ref_win)
    except Exception as exc:
        mdh_muskat = {"available": False, "note": f"MDH/Muskat skipped: {exc}"}

    # 9. Regime Triage
    conf = float((hyb_summary or {}).get("confidence_score") or 0.0)
    span_cyc = float((hyb_summary or {}).get("intersect_span_cyc") or 0.0)
    plat_ok = ((hyb_summary or {}).get("checks") or {}).get("plateau") == "PASS"

    if hyb_summary and conf >= 82.0 and plat_ok and span_cyc >= 0.65:
        regime_status = "CLEAR_IARF"
        regime_label = "Clean Infinite-Acting Radial Flow"
        recommended = "Hybrid Horner ∩ Bourdet (verified by Stehfest, Agarwal & Deconvolution)"
        guidance = (
            f"Clear horizontal Bourdet plateau spans {span_cyc:.2f} log cycles. "
            f"Stehfest Analytical Match (k = {analytical['k']:.1f} ± {analytical['k_std_pct']}% mD, S = {analytical['skin']:+.2f}) "
            f"and Agarwal Δt_e (k = {agarwal.get('k', '—')} mD) confirm the straight-line Horner interpretation."
        )
    elif hyb_summary and conf >= 55.0 and span_cyc >= 0.30:
        regime_status = "MASKED_OR_SHORT_IARF"
        regime_label = "Boundary-Masked / Short IARF Transition"
        recommended = "Stehfest Analytical Match + Ilk/von Schroeter Deconvolution + Agarwal Δt_e"
        guidance = (
            f"Radial plateau is compressed ({span_cyc:.2f} log cycles) by early wellbore storage (C = {analytical['c_bbl_psi']:.4f} bbl/psi) "
            f"or nearby boundary depletion (r_e ≈ {analytical['re_ft']:.0f} ft). "
            f"Select Stehfest Analytical, Deconvolution, Agarwal Δt_e, or Blasingame ΔP_id in the dropdown above to inspect each method directly."
        )
    else:
        regime_status = "NO_CLEAR_IARF"
        regime_label = "No Distinguishable IARF (Storage / Boundary Dominated)"
        recommended = "Stehfest Analytical Type-Curve Match & B-Spline / von Schroeter Deconvolution"
        guidance = (
            f"No flat Bourdet plateau developed within the {dur_hr:.1f}-hr shut-in — do NOT force a raw Horner line. "
            f"Use Stehfest Analytical Matching (k = {analytical['k']:.1f} ± {analytical['k_std_pct']}% mD, S = {analytical['skin']:+.2f}) "
            f"or Convex B-Spline / von Schroeter Deconvolution."
        )

    return {
        "available": True,
        "duration_hr": round(dur_hr, 2),
        "regime_status": regime_status,
        "regime_label": regime_label,
        "recommended_method": recommended,
        "guidance": guidance,
        "analytical_match": analytical,
        "deconvolution": deconv,
        "ilk_bspline_deconv": ilk_deconv,
        "agarwal_equiv": agarwal,
        "blasingame_integral": blasingame,
        "odeh_jones_multirate": odeh_jones,
        "tiab_tds": tds,
        "mdh_muskat": mdh_muskat,
        "agarwal_deriv_pts": agarwal.get("curve_der", []) if agarwal.get("available") else [],
    }
