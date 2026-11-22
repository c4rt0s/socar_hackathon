from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

_EULER_GAMMA = 0.5772156649015329
_ACRE_TO_SQFT = 43560.0
_DARCY_PRESSURE_COEFF = 141.2
_DARCY_TIME_COEFF = 0.0002637


def _e1_series_coeffs(n_terms: int = 30) -> tuple[float, ...]:
    """Precompute (-1)^k / (k * k!) for k = 1..n_terms."""
    coeffs: list[float] = []
    fact = 1.0
    sign = -1.0
    for k in range(1, n_terms + 1):
        fact *= k
        coeffs.append(sign / (k * fact))
        sign = -sign
    return tuple(coeffs)


_E1_COEFFS = _e1_series_coeffs(30)


def _e1(x: np.ndarray) -> np.ndarray:
    """Evaluate the exponential integral E1(x) = -Ei(-x) for positive x."""
    out = np.zeros_like(x, dtype=np.float64)

    small = (x > 0.0) & (x <= 1.0)
    if np.any(small):
        xs = x[small]
        series = np.zeros_like(xs)
        for coeff in reversed(_E1_COEFFS):
            series = series * xs + coeff
        out[small] = -_EULER_GAMMA - np.log(xs) - xs * series

    # Laguerre/Stieltjes contracted continued fraction; strictly positive denominators for x > 1.
    mid = (x > 1.0) & (x <= 750.0)
    if np.any(mid):
        xm = x[mid]
        n_cf = 35
        w = xm + (2 * n_cf + 1)
        for k in range(n_cf, 0, -1):
            w = xm + (2 * k - 1) - (k * k) / w
        out[mid] = np.exp(-xm) / w

    return out


def _validate_pos_finite(name: str, value: float) -> float:
    """Validate that a scalar parameter is finite and strictly positive."""
    val = float(value)
    if not np.isfinite(val) or val <= 0.0:
        raise ValueError(f"{name} must be finite and positive, got {value!r}")
    return val


def radial_unit_response(
    kh_md_ft: float,
    h_ft: float,
    mu_cp: float,
    phi: float,
    ct_per_psi: float,
    rw_ft: float,
    skin: float = 0.0,
    area_acres: float | None = None,
    shape_factor: float = 31.62,
) -> Callable[[np.ndarray], np.ndarray]:
    """Build a constant-rate radial drawdown unit-response function g(tau_hr).

    Parameters use oilfield units: permeability-thickness (md-ft), net pay (ft),
    viscosity (cp), porosity (fraction), total compressibility (1/psi),
    wellbore radius (ft), dimensionless skin, optional drainage area (acres),
    and Dietz shape factor. Returns psi drop per 1 RB/D after tau hours.
    """
    kh = _validate_pos_finite("kh_md_ft", kh_md_ft)
    h = _validate_pos_finite("h_ft", h_ft)
    mu = _validate_pos_finite("mu_cp", mu_cp)
    por = _validate_pos_finite("phi", phi)
    ct = _validate_pos_finite("ct_per_psi", ct_per_psi)
    rw = _validate_pos_finite("rw_ft", rw_ft)

    skin_val = float(skin)
    if not np.isfinite(skin_val):
        raise ValueError(f"skin must be finite, got {skin!r}")

    area_sqft: float | None = None
    tda_factor = 0.0
    pss_const = 0.0
    if area_acres is not None:
        acres = _validate_pos_finite("area_acres", area_acres)
        ca = _validate_pos_finite("shape_factor", shape_factor)
        area_sqft = acres * _ACRE_TO_SQFT
        tda_factor = (rw * rw) / area_sqft
        pss_const = 0.5 * np.log(4.0 * area_sqft / (1.781 * ca * rw * rw)) + skin_val

    k_md = kh / h
    td_factor = _DARCY_TIME_COEFF * k_md / (por * mu * ct * rw * rw)
    p_coeff = _DARCY_PRESSURE_COEFF * mu / kh

    def unit_response(tau_hr: np.ndarray) -> np.ndarray:
        tau = np.asarray(tau_hr, dtype=np.float64)
        out = np.zeros_like(tau, dtype=np.float64)
        pos = tau > 0.0
        if not np.any(pos):
            return out

        t_d = td_factor * tau[pos]
        p_d_inf = 0.5 * _e1(0.25 / t_d) + skin_val

        if area_sqft is None:
            p_d = p_d_inf
        else:
            t_da = t_d * tda_factor
            p_d_pss = 2.0 * np.pi * t_da + pss_const
            weight = np.clip((t_da - 0.05) / 0.10, 0.0, 1.0)
            p_d = (1.0 - weight) * p_d_inf + weight * p_d_pss

        out[pos] = p_coeff * p_d
        return out

    return unit_response


def _solve_bounded_qp(
    h_mat: np.ndarray,
    b_vec: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    x0: np.ndarray,
    max_iter: int,
    tol: float,
) -> tuple[np.ndarray, int, bool]:
    """Minimize 0.5 * x^T H x - b^T x subject to lb <= x <= ub via projected Newton.

    Alternates spectral projected gradient (Barzilai-Borwein) steps to identify
    the active bounds with exact subspace Newton solves on the free variables.
    """
    n = h_mat.shape[0]
    if n == 0:
        return np.empty(0, dtype=np.float64), 0, True

    x = np.clip(x0.astype(np.float64, copy=True), lb, ub)
    b_norm = max(1.0, float(np.linalg.norm(b_vec)))
    tol_abs = tol * b_norm

    prev_x: np.ndarray | None = None
    prev_g: np.ndarray | None = None
    spg_steps = 8

    for it in range(1, max_iter + 1):
        g = h_mat @ x - b_vec
        proj_grad = g.copy()
        at_lb = x <= lb
        at_ub = x >= ub
        proj_grad[at_lb] = np.minimum(0.0, g[at_lb])
        proj_grad[at_ub] = np.maximum(0.0, g[at_ub])
        proj_grad[lb == ub] = 0.0

        if float(np.linalg.norm(proj_grad)) <= tol_abs:
            return x, it - 1, True

        # Phase 1: short Spectral Projected Gradient burst to settle active bounds.
        prev_active = (at_lb & (g >= 0.0)) | (at_ub & (g <= 0.0)) | (lb == ub)
        for _ in range(spg_steps):
            hr = h_mat @ proj_grad
            rhr = float(np.dot(proj_grad, hr))
            rr = float(np.dot(proj_grad, proj_grad))
            if rr <= tol_abs * tol_abs:
                break
            alpha_c = rr / rhr if rhr > 1e-30 * rr else 1.0

            if prev_x is not None and prev_g is not None:
                s_vec = x - prev_x
                y_vec = g - prev_g
                sy = float(np.dot(s_vec, y_vec))
                ss = float(np.dot(s_vec, s_vec))
                alpha = ss / sy if sy > 1e-30 * ss else alpha_c
            else:
                alpha = alpha_c
            alpha = float(np.clip(alpha, 1e-12, 1e12))

            accepted = False
            for _ in range(25):
                x_trial = np.clip(x - alpha * g, lb, ub)
                dx = x_trial - x
                gdx = float(np.dot(g, dx))
                df = gdx + 0.5 * float(np.dot(dx, h_mat @ dx))
                if df <= 1e-4 * gdx or alpha < 1e-15:
                    prev_x = x
                    prev_g = g
                    x = x_trial
                    g = h_mat @ x - b_vec
                    accepted = True
                    break
                alpha *= 0.5

            if not accepted:
                break

            active_now = ((x <= lb) & (g >= 0.0)) | ((x >= ub) & (g <= 0.0)) | (lb == ub)
            if np.array_equal(active_now, prev_active):
                break
            prev_active = active_now
            proj_grad = g.copy()
            proj_grad[x <= lb] = np.minimum(0.0, g[x <= lb])
            proj_grad[x >= ub] = np.maximum(0.0, g[x >= ub])
            proj_grad[lb == ub] = 0.0

        # Phase 2: exact subspace Newton solve on the free face.
        active = ((x <= lb) & (g >= 0.0)) | ((x >= ub) & (g <= 0.0)) | (lb == ub)
        free = ~active
        if np.any(free):
            h_ff = h_mat[np.ix_(free, free)]
            g_f = g[free]
            try:
                d_f = np.linalg.solve(h_ff, -g_f)
            except np.linalg.LinAlgError:
                d_f, _, _, _ = np.linalg.lstsq(h_ff, -g_f, rcond=None)

            if float(np.dot(g_f, d_f)) < 0.0:
                d = np.zeros(n, dtype=np.float64)
                d[free] = d_f
                x_full = x + d
                if np.all((x_full >= lb) & (x_full <= ub)):
                    prev_x = x
                    prev_g = g
                    x = x_full
                else:
                    x_proj = np.clip(x_full, lb, ub)
                    dx = x_proj - x
                    df_proj = float(np.dot(g, dx)) + 0.5 * float(np.dot(dx, h_mat @ dx))
                    if df_proj < 0.0:
                        prev_x = x
                        prev_g = g
                        x = x_proj
                    else:
                        neg = free & (d < 0.0) & np.isfinite(lb)
                        pos = free & (d > 0.0) & np.isfinite(ub)
                        alpha_max = 1.0
                        if np.any(neg):
                            alpha_max = min(alpha_max, float(np.min((lb[neg] - x[neg]) / d[neg])))
                        if np.any(pos):
                            alpha_max = min(alpha_max, float(np.min((ub[pos] - x[pos]) / d[pos])))
                        if alpha_max > 0.0:
                            prev_x = x
                            prev_g = g
                            x = np.clip(x + alpha_max * d, lb, ub)

    g = h_mat @ x - b_vec
    proj_grad = g.copy()
    proj_grad[x <= lb] = np.minimum(0.0, g[x <= lb])
    proj_grad[x >= ub] = np.maximum(0.0, g[x >= ub])
    proj_grad[lb == ub] = 0.0
    converged = float(np.linalg.norm(proj_grad)) <= tol_abs
    return x, max_iter, converged


def deconvolve_rates(
    t_obs_hr: np.ndarray,
    p_obs: np.ndarray,
    t_edges_hr: np.ndarray,
    unit_response: Callable[[np.ndarray], np.ndarray],
    p_init: float | None = None,
    smoothing: float = 1e-2,
    q_lower: float = 0.0,
    q_upper: float | None = None,
    q_prior: np.ndarray | float | None = None,
    prior_weight: float = 0.0,
    obs_weight: np.ndarray | None = None,
    fixed_zero: np.ndarray | None = None,
    max_iter: int = 2000,
    tol: float = 1e-9,
    background_trend: bool = False,
) -> dict[str, Any]:
    """Estimate piecewise-constant well flow rates by pressure-rate deconvolution.

    Solves the bounded, regularized linear least-squares problem for rates q_j on
    [t_edges_hr[j], t_edges_hr[j+1]) and optional unknown initial pressure p_init.

    background_trend=True (requires p_init=None) adds a free linear pressure drift
    p_init + d*t. It absorbs depletion caused by production *before* the first rate cell
    (e.g. years of history before the data window), which would otherwise be attributed to
    the modelled rates. Both unknowns are eliminated by weighted projection.
    """
    t_obs = np.asarray(t_obs_hr, dtype=np.float64)
    p = np.asarray(p_obs, dtype=np.float64)
    t_edges = np.asarray(t_edges_hr, dtype=np.float64)

    if t_obs.ndim != 1 or p.ndim != 1 or t_obs.shape != p.shape or t_obs.size == 0:
        raise ValueError("t_obs_hr and p_obs must be non-empty 1D arrays of matching length")
    if not np.all(np.isfinite(t_obs)):
        raise ValueError("t_obs_hr must contain only finite values")
    if t_edges.ndim != 1 or t_edges.size < 2 or not np.all(np.isfinite(t_edges)):
        raise ValueError("t_edges_hr must be a finite 1D array with at least 2 edges")
    if not np.all(np.diff(t_edges) > 0.0):
        raise ValueError("t_edges_hr must be strictly increasing")
    if smoothing < 0.0 or not np.isfinite(smoothing):
        raise ValueError("smoothing must be finite and non-negative")
    if prior_weight < 0.0 or not np.isfinite(prior_weight):
        raise ValueError("prior_weight must be finite and non-negative")
    if max_iter <= 0 or tol <= 0.0 or not np.isfinite(tol):
        raise ValueError("max_iter and tol must be positive and finite")

    m = t_obs.size
    n = t_edges.size - 1

    valid = np.isfinite(p)
    if not np.any(valid):
        raise ValueError("p_obs contains no finite observations")

    if obs_weight is None:
        w = np.ones(m, dtype=np.float64)
    else:
        w = np.asarray(obs_weight, dtype=np.float64)
        if w.shape != (m,) or not np.all(np.isfinite(w)) or np.any(w < 0.0):
            raise ValueError("obs_weight must be a finite, non-negative array of length len(p_obs)")

    w_valid = w[valid]
    w_sum = float(np.sum(w_valid))
    if w_sum <= 0.0:
        raise ValueError("sum of obs_weight on finite observations must be positive")

    lb_arr = np.broadcast_to(
        np.asarray(-np.inf if q_lower is None else q_lower, dtype=np.float64), (n,)
    ).copy()
    ub_arr = np.broadcast_to(
        np.asarray(np.inf if q_upper is None else q_upper, dtype=np.float64), (n,)
    ).copy()
    if np.any(lb_arr > ub_arr):
        raise ValueError("q_lower must not exceed q_upper")

    if fixed_zero is None:
        shut_in = np.zeros(n, dtype=bool)
    else:
        shut_in = np.asarray(fixed_zero, dtype=bool)
        if shut_in.shape != (n,):
            raise ValueError(f"fixed_zero must have shape ({n},)")

    if prior_weight > 0.0:
        if q_prior is None:
            prior_vec = np.zeros(n, dtype=np.float64)
        else:
            prior_vec = np.broadcast_to(np.asarray(q_prior, dtype=np.float64), (n,)).copy()
            if not np.all(np.isfinite(prior_vec)):
                raise ValueError("q_prior must contain only finite values")
    else:
        prior_vec = np.zeros(n, dtype=np.float64)

    if p_init is not None:
        p_init_fixed = float(p_init)
        if not np.isfinite(p_init_fixed):
            raise ValueError("p_init must be finite when provided")
    else:
        p_init_fixed = None

    # Vectorised convolution matrix G[i, j] = g(t_i - e_j) - g(t_i - e_{j+1}), evaluating g only for tau > 0.
    dt = t_obs[:, None] - t_edges[None, :]
    g_vals = np.zeros_like(dt, dtype=np.float64)
    pos_dt = dt > 0.0
    if np.any(pos_dt):
        g_vals[pos_dt] = unit_response(dt[pos_dt])
    g_all = g_vals[:, :-1] - g_vals[:, 1:]

    g_valid = g_all[valid]
    p_valid = p[valid]
    sqrt_w = np.sqrt(w_valid)
    gw = g_valid * sqrt_w[:, None]

    col_norms = np.linalg.norm(gw, axis=0)
    pos_cols = col_norms > 0.0
    s = float(np.median(col_norms[pos_cols])) if np.any(pos_cols) else 1.0
    s2 = s * s

    basis_q = None
    if p_init_fixed is None and background_trend:
        t_v = t_obs[valid]
        basis = np.column_stack([np.ones_like(t_v), (t_v - t_v.mean()) / max(float(np.ptp(t_v)), 1e-9)])
        basis_q, _ = np.linalg.qr(basis * sqrt_w[:, None])
        gw_perp = gw - basis_q @ (basis_q.T @ gw)
        y = -p_valid * sqrt_w
        y_perp = y - basis_q @ (basis_q.T @ y)
        h_full = gw_perp.T @ gw_perp
        b_full = gw_perp.T @ y_perp
        gw_mean = np.zeros(n, dtype=np.float64)
        p_bar = 0.0
    elif p_init_fixed is None:
        gw_mean = (g_valid.T @ w_valid) / w_sum
        gw_centered = gw - np.outer(sqrt_w, gw_mean)
        h_full = gw_centered.T @ gw_centered
        p_bar = float(np.dot(w_valid, p_valid) / w_sum)
        b_full = g_valid.T @ (w_valid * (p_bar - p_valid))
    else:
        gw_mean = np.zeros(n, dtype=np.float64)
        p_bar = 0.0
        h_full = gw.T @ gw
        b_full = g_valid.T @ (w_valid * (p_init_fixed - p_valid))

    lam_s = smoothing * s2
    if n > 1 and lam_s > 0.0:
        idx = np.arange(n)
        diag_s = np.full(n, 2.0 * lam_s, dtype=np.float64)
        diag_s[0] = lam_s
        diag_s[-1] = lam_s
        h_full[idx, idx] += diag_s
        off = np.arange(n - 1)
        h_full[off, off + 1] -= lam_s
        h_full[off + 1, off] -= lam_s

    lam_p = prior_weight * s2
    if lam_p > 0.0:
        idx = np.arange(n)
        h_full[idx, idx] += lam_p
        b_full += lam_p * prior_vec

    active_vars = ~shut_in
    q = np.zeros(n, dtype=np.float64)
    if np.any(active_vars):
        h_sub = h_full[np.ix_(active_vars, active_vars)]
        b_sub = b_full[active_vars]
        lb_sub = lb_arr[active_vars]
        ub_sub = ub_arr[active_vars]

        try:
            x0 = np.linalg.solve(h_sub, b_sub)
        except np.linalg.LinAlgError:
            x0, _, _, _ = np.linalg.lstsq(h_sub, b_sub, rcond=None)
        x0 = np.clip(x0, lb_sub, ub_sub)

        q_sub, iterations, converged = _solve_bounded_qp(
            h_sub, b_sub, lb_sub, ub_sub, x0, max_iter=max_iter, tol=tol
        )
        q[active_vars] = q_sub
    else:
        iterations = 0
        converged = True

    trend_per_hr = 0.0
    if basis_q is not None:
        t_v = t_obs[valid]
        basis = np.column_stack([np.ones_like(t_v), t_v])
        coef, *_ = np.linalg.lstsq(basis * sqrt_w[:, None], sqrt_w * (p_valid + g_valid @ q), rcond=None)
        p_init_est, trend_per_hr = float(coef[0]), float(coef[1])
        p_fit_valid = p_init_est + trend_per_hr * t_v - g_valid @ q
    elif p_init_fixed is None:
        p_init_est = p_bar + float(np.dot(gw_mean, q))
        p_fit_valid = p_init_est - g_valid @ q
    else:
        p_init_est = p_init_fixed
        p_fit_valid = p_init_est - g_valid @ q

    p_fit = np.full(m, np.nan, dtype=np.float64)
    p_fit[valid] = p_fit_valid

    resid = p_fit_valid - p_valid
    residual_rms = float(np.sqrt(np.mean(resid * resid)))

    obj = float(np.dot(w_valid, resid * resid))
    if n > 1 and lam_s > 0.0:
        dq = np.diff(q)
        obj += lam_s * float(np.dot(dq, dq))
    if lam_p > 0.0:
        dp = q - prior_vec
        obj += lam_p * float(np.dot(dp, dp))

    return {
        "q": q,
        "p_init": float(p_init_est),
        "trend_psi_per_hr": float(trend_per_hr),
        "p_fit": p_fit,
        "residual_rms": residual_rms,
        "iterations": int(iterations),
        "converged": bool(converged),
        "objective": float(obj),
    }


if __name__ == "__main__":
    g_closed = radial_unit_response(
        kh_md_ft=3000.0,
        h_ft=60.0,
        mu_cp=1.2,
        phi=0.2,
        ct_per_psi=1.2e-5,
        rw_ft=0.33,
        skin=2.0,
        area_acres=300.0,
    )
    assert np.all(g_closed(np.array([-1.0, 0.0])) == 0.0)
    tau_grid = np.logspace(-3, 4, 500)
    g_grid = g_closed(tau_grid)
    assert np.all(np.diff(g_grid) >= 0.0)

    g_inf = radial_unit_response(
        kh_md_ft=3000.0,
        h_ft=60.0,
        mu_cp=1.2,
        phi=0.2,
        ct_per_psi=1.2e-5,
        rw_ft=0.33,
        skin=2.0,
        area_acres=None,
    )
    tau_large = np.array([1.0e4])
    td_large = _DARCY_TIME_COEFF * (3000.0 / 60.0) * tau_large / (0.2 * 1.2 * 1.2e-5 * 0.33**2)
    g_asymp = (_DARCY_PRESSURE_COEFF * 1.2 / 3000.0) * (0.5 * (np.log(td_large) + 0.80907) + 2.0)
    rel_asymp_err = float(np.abs(g_inf(tau_large)[0] - g_asymp[0]) / g_asymp[0])
    assert rel_asymp_err < 1e-3

    n_steps = 120
    t_edges = np.arange(n_steps + 1, dtype=np.float64) * 6.0
    q_true = np.empty(n_steps, dtype=np.float64)
    q_true[:20] = 800.0
    q_true[20:30] = 0.0
    q_true[30:55] = 1200.0
    q_true[55:80] = 1500.0
    q_true[80:90] = 0.0
    q_true[90:] = 500.0

    t_obs = np.arange(0.5, 720.5, 0.5, dtype=np.float64)
    dt_mat = t_obs[:, None] - t_edges[None, :]
    g_mat = g_closed(dt_mat)
    g_true = g_mat[:, :-1] - g_mat[:, 1:]

    p_init_true = 4200.0
    rng = np.random.default_rng(0)
    p_obs = p_init_true - g_true @ q_true + rng.normal(0.0, 0.3, size=t_obs.size)

    shut_in_mask = q_true == 0.0
    result = deconvolve_rates(
        t_obs_hr=t_obs,
        p_obs=p_obs,
        t_edges_hr=t_edges,
        unit_response=g_closed,
        p_init=None,
        fixed_zero=shut_in_mask,
    )

    assert result["converged"] is True
    assert np.all(result["q"][shut_in_mask] == 0.0)
    flowing = ~shut_in_mask
    mean_err = abs(float(np.mean(result["q"][flowing])) - float(np.mean(q_true[flowing]))) / float(
        np.mean(q_true[flowing])
    )
    assert mean_err < 0.03
    assert abs(result["p_init"] - p_init_true) < 2.0

    print("deconvolution self-checks passed")
