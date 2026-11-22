"""Independent pressure transient analysis (PTA) cross-check and productivity methods."""

from __future__ import annotations

from fractions import Fraction
from functools import lru_cache
import math
from typing import Any

import numpy as np

DARCY_P_COEFF = 141.2
HORNER_SLOPE_COEFF = 162.6
HALF_P_COEFF = 70.6
TD_TIME_COEFF = 0.0002637
CD_STORAGE_COEFF = 0.8936
EULER_GAMMA = 0.5772156649015329
LN4_MINUS_GAMMA = 0.80907


def _check_scalar_type(val: Any, name: str) -> float | None:
    """Validate scalar numeric type; return finite float or None for bad data."""
    if val is None:
        return None
    if isinstance(val, (bool, str, bytes, dict, list, tuple)):
        raise TypeError(f"{name} must be a real number, got {type(val).__name__}")
    try:
        fval = float(val)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be numeric") from exc
    return fval if math.isfinite(fval) else None


def _clean_float(val: float | None) -> float | None:
    """Return finite Python float or None."""
    if val is None:
        return None
    fval = float(val)
    return fval if math.isfinite(fval) else None


def _to_1d_float_array(seq: Any, name: str) -> np.ndarray:
    """Convert array-like input to 1D float ndarray, raising on wrong container types."""
    if isinstance(seq, (str, bytes, dict)):
        raise TypeError(f"{name} must be a 1D numeric sequence")
    try:
        arr = np.asarray(seq, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must contain numeric values") from exc
    return arr


def _validate_window(dt_window: Any) -> tuple[float, float] | None:
    """Validate two-element positive window (dt_min, dt_max)."""
    if isinstance(dt_window, (str, bytes, dict)):
        raise TypeError("dt_window must be a 2-element sequence")
    try:
        seq = list(dt_window)
    except TypeError as exc:
        raise TypeError("dt_window must be a 2-element sequence") from exc
    if len(seq) != 2:
        return None
    lo = _check_scalar_type(seq[0], "dt_window[0]")
    hi = _check_scalar_type(seq[1], "dt_window[1]")
    if lo is None or hi is None or lo <= 0.0 or hi < lo:
        return None
    return (lo, hi)


def _linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float] | None:
    """Return (slope, intercept, r2) from least-squares y = intercept + slope * x."""
    xm = float(np.mean(x))
    ym = float(np.mean(y))
    dx = x - xm
    dy = y - ym
    ss_xx = float(np.dot(dx, dx))
    if ss_xx <= 0.0:
        return None
    slope = float(np.dot(dx, dy) / ss_xx)
    intercept = ym - slope * xm
    resid = y - (intercept + slope * x)
    ss_res = float(np.dot(resid, resid))
    ss_tot = float(np.dot(dy, dy))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else (1.0 if ss_res == 0.0 else 0.0)
    return slope, intercept, min(1.0, max(0.0, r2))


def _prepare_buildup_points(
    dt_hr: Any,
    p_psi: Any,
    dt_window: Any | None,
    notes: list[str],
) -> tuple[np.ndarray, np.ndarray, int]:
    """Validate and filter positive finite build-up points, optionally within dt_window."""
    dt_arr = _to_1d_float_array(dt_hr, "dt_hr")
    p_arr = _to_1d_float_array(p_psi, "pws_psi")
    if dt_arr.ndim != 1 or p_arr.ndim != 1 or dt_arr.shape[0] != p_arr.shape[0]:
        notes.append("dt_hr and pressure array must be 1D with matching lengths")
        return np.empty(0, dtype=float), np.empty(0, dtype=float), 0
    mask = np.isfinite(dt_arr) & np.isfinite(p_arr) & (dt_arr > 0.0)
    if not np.all(mask):
        notes.append("Non-finite or non-positive dt points excluded")
    if dt_window is not None:
        win = _validate_window(dt_window)
        if win is None:
            notes.append("Invalid dt_window; must be finite with 0 < dt_window[0] <= dt_window[1]")
            return np.empty(0, dtype=float), np.empty(0, dtype=float), 0
        mask &= (dt_arr >= win[0]) & (dt_arr <= win[1])
    dt_sel = dt_arr[mask]
    p_sel = p_arr[mask]
    return dt_sel, p_sel, int(dt_sel.size)


def _compute_kh_k_skin(
    m: float,
    p_1hr: float,
    q_rb: Any,
    mu_cp: Any,
    h_ft: Any,
    phi: Any,
    ct: Any,
    rw_ft: Any,
    pwf_psi: Any,
    notes: list[str],
) -> tuple[float | None, float | None, float | None]:
    """Compute kh, k, and skin from semilog slope m and p_1hr."""
    q = _check_scalar_type(q_rb, "q_rb")
    mu = _check_scalar_type(mu_cp, "mu_cp")
    h = _check_scalar_type(h_ft, "h_ft")
    ph = _check_scalar_type(phi, "phi")
    c_t = _check_scalar_type(ct, "ct")
    rw = _check_scalar_type(rw_ft, "rw_ft")
    pwf = _check_scalar_type(pwf_psi, "pwf_psi")

    if q is None or q <= 0.0 or mu is None or mu <= 0.0:
        notes.append("q_rb and mu_cp must be positive and finite to compute kh")
        return None, None, None
    kh = HORNER_SLOPE_COEFF * q * mu / m
    if h is None or h <= 0.0:
        notes.append("h_ft must be positive and finite to compute k and skin")
        return _clean_float(kh), None, None
    k = kh / h

    bad_skin: list[str] = []
    if ph is None or ph <= 0.0 or ph > 1.0:
        bad_skin.append("phi")
    if c_t is None or c_t <= 0.0:
        bad_skin.append("ct")
    if rw is None or rw <= 0.0:
        bad_skin.append("rw_ft")
    if pwf is None:
        bad_skin.append("pwf_psi")
    if bad_skin:
        notes.append(f"Invalid {', '.join(bad_skin)}; skin cannot be computed")
        return _clean_float(kh), _clean_float(k), None

    denom = ph * mu * c_t * (rw ** 2)
    if denom <= 0.0 or k <= 0.0:
        notes.append("Non-positive diffusivity denominator; skin cannot be computed")
        return _clean_float(kh), _clean_float(k), None
    skin = 1.151 * ((p_1hr - pwf) / m - math.log10(k / denom) + 3.23)
    return _clean_float(kh), _clean_float(k), _clean_float(skin)


def horner_analysis(
    dt_hr: Any,
    pws_psi: Any,
    tp_hr: Any,
    dt_window: Any,
    q_rb: Any,
    mu_cp: Any,
    h_ft: Any,
    phi: Any,
    ct: Any,
    rw_ft: Any,
    pwf_psi: Any,
) -> dict[str, Any]:
    """Perform Horner semilog build-up analysis over dt_window."""
    notes: list[str] = []
    result: dict[str, Any] = {
        "m": None,
        "kh": None,
        "k": None,
        "p_star": None,
        "p_1hr": None,
        "skin": None,
        "r2": None,
        "n_points": 0,
        "notes": notes,
    }
    dt_sel, p_sel, n_pts = _prepare_buildup_points(dt_hr, pws_psi, dt_window, notes)
    result["n_points"] = n_pts
    tp = _check_scalar_type(tp_hr, "tp_hr")
    if tp is None or tp <= 0.0:
        notes.append("tp_hr must be positive and finite for Horner analysis")
        return result
    if n_pts < 4:
        notes.append("At least 4 valid points in dt_window are required")
        return result

    x = np.log10((tp + dt_sel) / dt_sel)
    fit = _linear_fit(x, p_sel)
    if fit is None:
        notes.append("Zero variance in Horner time ratio; cannot fit line")
        return result
    slope, p_star, r2 = fit
    m = -slope
    p_1hr = p_star - m * math.log10(tp + 1.0)
    result["p_star"] = _clean_float(p_star)
    result["p_1hr"] = _clean_float(p_1hr)
    result["r2"] = _clean_float(r2)
    if m <= 0.0:
        notes.append("Horner slope m <= 0; permeability and skin cannot be computed")
        return result

    result["m"] = _clean_float(m)
    kh, k, skin = _compute_kh_k_skin(m, p_1hr, q_rb, mu_cp, h_ft, phi, ct, rw_ft, pwf_psi, notes)
    result["kh"] = kh
    result["k"] = k
    result["skin"] = skin
    return result


def mdh_analysis(
    dt_hr: Any,
    pws_psi: Any,
    dt_window: Any,
    tp_hr: Any,
    q_rb: Any,
    mu_cp: Any,
    h_ft: Any,
    phi: Any,
    ct: Any,
    rw_ft: Any,
    pwf_psi: Any,
) -> dict[str, Any]:
    """Perform Miller-Dyes-Hutchinson (MDH) build-up analysis over dt_window."""
    notes: list[str] = []
    result: dict[str, Any] = {
        "m": None,
        "kh": None,
        "k": None,
        "p_1hr": None,
        "skin": None,
        "r2": None,
        "n_points": 0,
        "valid": False,
        "notes": notes,
    }
    dt_sel, p_sel, n_pts = _prepare_buildup_points(dt_hr, pws_psi, dt_window, notes)
    result["n_points"] = n_pts
    if n_pts < 4:
        notes.append("At least 4 valid points in dt_window are required")
        return result

    win = _validate_window(dt_window)
    tp = _check_scalar_type(tp_hr, "tp_hr")
    valid_assumption = True
    if tp is None or tp <= 0.0:
        valid_assumption = False
        notes.append("tp_hr must be positive and finite to verify MDH dt << tp assumption")
    elif win is not None and win[1] > 0.1 * tp:
        valid_assumption = False
        notes.append("dt_window[1] > 0.1 * tp_hr; MDH assumption (dt << tp) violated")

    x = np.log10(dt_sel)
    fit = _linear_fit(x, p_sel)
    if fit is None:
        notes.append("Zero variance in log10(dt); cannot fit line")
        return result
    m, p_1hr, r2 = fit
    result["p_1hr"] = _clean_float(p_1hr)
    result["r2"] = _clean_float(r2)
    if m <= 0.0:
        notes.append("MDH slope m <= 0; permeability and skin cannot be computed")
        return result

    result["m"] = _clean_float(m)
    kh, k, skin = _compute_kh_k_skin(m, p_1hr, q_rb, mu_cp, h_ft, phi, ct, rw_ft, pwf_psi, notes)
    result["kh"] = kh
    result["k"] = k
    result["skin"] = skin
    result["valid"] = valid_assumption and (kh is not None)
    return result


def productivity_indices(
    q_liquid_stbd: Any,
    q_rb: Any,
    p_avg_psia: Any,
    pwf_psia: Any,
    kh_md_ft: Any = None,
    skin: Any = None,
    mu_cp: Any = None,
    b_liquid: Any = None,
    re_ft: Any = None,
    rw_ft: Any = None,
    dp_skin_psi: Any = None,
    p_avg_source: str = "",
) -> dict[str, Any]:
    """Compute actual, theoretical, and ideal pseudo-steady-state productivity indices."""
    if not isinstance(p_avg_source, str):
        raise TypeError("p_avg_source must be a str")
    notes: list[str] = []
    q_stbd = _check_scalar_type(q_liquid_stbd, "q_liquid_stbd")
    q_res = _check_scalar_type(q_rb, "q_rb")
    p_avg = _check_scalar_type(p_avg_psia, "p_avg_psia")
    pwf = _check_scalar_type(pwf_psia, "pwf_psia")
    kh = _check_scalar_type(kh_md_ft, "kh_md_ft")
    sk = _check_scalar_type(skin, "skin")
    mu = _check_scalar_type(mu_cp, "mu_cp")
    b_liq = _check_scalar_type(b_liquid, "b_liquid")
    re = _check_scalar_type(re_ft, "re_ft")
    rw = _check_scalar_type(rw_ft, "rw_ft")
    dp_skin = _check_scalar_type(dp_skin_psi, "dp_skin_psi")

    dp_drawdown: float | None = None
    pi_actual: float | None = None
    if p_avg is None or pwf is None:
        notes.append("p_avg_psia and pwf_psia must be finite")
    elif p_avg <= pwf:
        notes.append("p_avg_psia <= pwf_psia; drawdown is non-positive, actual PI is None")
    else:
        dp_drawdown = p_avg - pwf
        if q_stbd is None or q_stbd <= 0.0:
            notes.append("q_liquid_stbd must be positive and finite for actual PI")
        else:
            pi_actual = _clean_float(q_stbd / dp_drawdown)

    if b_liq is None and q_res is not None and q_res > 0.0 and q_stbd is not None and q_stbd > 0.0:
        b_liq = q_res / q_stbd
        notes.append("b_liquid inferred from q_rb / q_liquid_stbd")

    pi_theo: float | None = None
    pi_ideal: float | None = None
    has_base = (
        kh is not None and kh > 0.0
        and mu is not None and mu > 0.0
        and b_liq is not None and b_liq > 0.0
        and re is not None and re > 0.0
        and rw is not None and rw > 0.0
    )
    if has_base:
        assert kh is not None and mu is not None and b_liq is not None
        assert re is not None and rw is not None
        if re <= rw:
            notes.append("re_ft <= rw_ft; cannot compute theoretical or ideal PI")
        else:
            ln_re_rw = math.log(re / rw)
            denom_ideal = ln_re_rw - 0.75
            if denom_ideal <= 0.1:
                notes.append("ln(re/rw) - 0.75 <= 0.1; ideal PI guarded as None")
            else:
                pi_ideal = _clean_float(kh / (DARCY_P_COEFF * b_liq * mu * denom_ideal))
            if sk is None:
                notes.append("skin not provided; theoretical PI is None")
            else:
                denom_theo = ln_re_rw - 0.75 + sk
                if denom_theo <= 0.1:
                    notes.append("ln(re/rw) - 0.75 + skin <= 0.1; theoretical PI guarded as None")
                else:
                    pi_theo = _clean_float(kh / (DARCY_P_COEFF * b_liq * mu * denom_theo))
    else:
        notes.append("Insufficient positive reservoir/fluid inputs for theoretical/ideal PI")

    fe: float | None = None
    if dp_skin is not None:
        if dp_drawdown is not None:
            fe = _clean_float((dp_drawdown - dp_skin) / dp_drawdown)
        else:
            notes.append("Positive drawdown required to compute flow efficiency from dp_skin_psi")
    elif pi_actual is not None and pi_ideal is not None and pi_ideal > 0.0:
        fe = _clean_float(pi_actual / pi_ideal)

    dr: float | None = None
    if fe is not None:
        if fe <= 0.0:
            notes.append("flow_efficiency <= 0; damage_ratio cannot be computed")
        else:
            dr = _clean_float(1.0 / fe)

    prod_ratio: float | None = None
    if pi_actual is not None and pi_theo is not None and pi_theo > 0.0:
        prod_ratio = _clean_float(pi_actual / pi_theo)

    return {
        "pi_actual_stbd_psi": pi_actual,
        "pi_theoretical_stbd_psi": pi_theo,
        "pi_ideal_stbd_psi": pi_ideal,
        "flow_efficiency": fe,
        "damage_ratio": dr,
        "productivity_ratio_actual_to_theoretical": prod_ratio,
        "p_avg_source": p_avg_source,
        "notes": notes,
    }


def bessel_k01_scaled(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (exp(x) * K0(x), exp(x) * K1(x)) for x > 0 via vectorised trapezoidal rule."""
    x_arr = np.asarray(x, dtype=float)
    orig_shape = x_arr.shape
    x_flat = x_arr.ravel()
    if x_flat.size == 0:
        return np.empty(orig_shape, dtype=float), np.empty(orig_shape, dtype=float)
    if not np.all(np.isfinite(x_flat) & (x_flat > 0.0)):
        raise ValueError("x must be finite and strictly positive")

    k0 = np.empty_like(x_flat)
    k1 = np.empty_like(x_flat)
    large = x_flat > 50.0
    small = ~large

    if np.any(small):
        xs = x_flat[small]
        min_x = float(np.min(xs))
        t_req = math.acosh(1.0 + 50.0 / min_x)
        t_max = min(t_req, 30.0)
        step = 0.1
        n_pts = max(2, min(301, int(math.ceil(t_max / step)) + 1))
        t = np.arange(n_pts, dtype=float) * step
        w = np.full(n_pts, step, dtype=float)
        w[0] = 0.5 * step
        w[-1] = 0.5 * step
        cosh_t = np.cosh(t)
        cosh_m1 = cosh_t - 1.0
        exp_mat = np.exp(-xs[:, None] * cosh_m1[None, :])
        k0[small] = exp_mat @ w
        k1[small] = exp_mat @ (w * cosh_t)

    if np.any(large):
        # For x > 50, fixed dt=0.1 aliases the sharp peak; use asymptotic series (<1e-16 error).
        xl = x_flat[large]
        pref = np.sqrt(math.pi / (2.0 * xl))
        inv_8x = 1.0 / (8.0 * xl)
        s0 = np.ones_like(xl)
        s1 = np.ones_like(xl)
        t0 = np.ones_like(xl)
        t1 = np.ones_like(xl)
        for j in range(1, 16):
            odd2 = (2 * j - 1) ** 2
            t0 = t0 * (-odd2 / j) * inv_8x
            t1 = t1 * ((4.0 - odd2) / j) * inv_8x
            s0 += t0
            s1 += t1
        k0[large] = pref * s0
        k1[large] = pref * s1

    return k0.reshape(orig_shape), k1.reshape(orig_shape)


@lru_cache(maxsize=16)
def _stehfest_weights(n: int) -> np.ndarray:
    """Compute exact Stehfest inversion weights using fractions.Fraction."""
    if n <= 0 or n % 2 != 0:
        raise ValueError("n_stehfest must be a positive even integer")
    half = n // 2
    weights: list[float] = []
    for i in range(1, n + 1):
        total = Fraction(0, 1)
        k_lo = (i + 1) // 2
        k_hi = min(i, half)
        for k in range(k_lo, k_hi + 1):
            num = (k ** half) * math.factorial(2 * k)
            den = (
                math.factorial(half - k)
                * math.factorial(k)
                * math.factorial(k - 1)
                * math.factorial(i - k)
                * math.factorial(2 * k - i)
            )
            total += Fraction(num, den)
        if (half + i) % 2 != 0:
            total = -total
        weights.append(float(total))
    return np.asarray(weights, dtype=float)


def pwd(tD: np.ndarray, CD: float, skin: float, n_stehfest: int = 12) -> np.ndarray:
    """Dimensionless wellbore pressure with wellbore storage and skin via Stehfest inversion."""
    t_arr = np.asarray(tD, dtype=float)
    orig_shape = t_arr.shape
    t_flat = t_arr.ravel()
    if t_flat.size == 0:
        return np.empty(orig_shape, dtype=float)
    if not np.all(np.isfinite(t_flat) & (t_flat > 0.0)):
        raise ValueError("tD must be finite and strictly positive")
    cd_val = float(CD)
    sk_val = float(skin)
    if not (math.isfinite(cd_val) and cd_val >= 0.0 and math.isfinite(sk_val)):
        raise ValueError("CD must be >= 0 and skin must be finite")

    if sk_val < 0.0:
        scale = math.exp(2.0 * sk_val)
        t_eff = t_flat * scale
        cd_eff = cd_val * scale
        s_eff = 0.0
    else:
        t_eff = t_flat
        cd_eff = cd_val
        s_eff = sk_val

    asym = (t_eff / max(cd_eff, 1.0)) > 1e6
    out = np.empty_like(t_eff)
    if np.any(asym):
        out[asym] = 0.5 * (np.log(t_eff[asym]) + LN4_MINUS_GAMMA) + s_eff

    rem = ~asym
    if np.any(rem):
        v_weights = _stehfest_weights(n_stehfest)
        k_idx = np.arange(1, n_stehfest + 1, dtype=float)
        ln2 = math.log(2.0)
        u = (ln2 / t_eff[rem, None]) * k_idx[None, :]
        sqrt_u = np.sqrt(u)
        k0s, k1s = bessel_k01_scaled(sqrt_u)
        term = k0s + s_eff * sqrt_u * k1s
        u_pbar = term / (sqrt_u * k1s + cd_eff * u * term)
        out[rem] = u_pbar @ (v_weights / k_idx)

    return out.reshape(orig_shape)


def _nelder_mead(
    obj: Any,
    x0: np.ndarray,
    steps: np.ndarray,
    max_iter: int,
    x_tol: float = 1e-4,
    f_tol: float = 1e-6,
) -> tuple[np.ndarray, float, int, bool]:
    """Adaptive Nelder-Mead minimisation (Gao & Han 2012)."""
    n = x0.size
    alpha = 1.0
    gamma = 1.0 + 2.0 / n
    rho = 0.75 - 1.0 / (2.0 * n)
    sigma = 1.0 - 1.0 / n

    simplex = np.tile(x0, (n + 1, 1))
    for i in range(n):
        simplex[i + 1, i] += steps[i]
    f_vals = np.array([float(obj(v)) for v in simplex], dtype=float)

    converged = False
    iters = 0
    for it in range(1, max_iter + 1):
        order = np.argsort(f_vals)
        simplex = simplex[order]
        f_vals = f_vals[order]
        if (
            np.max(np.abs(simplex[1:] - simplex[0])) <= x_tol
            and abs(f_vals[-1] - f_vals[0]) <= f_tol * max(1.0, abs(f_vals[0]))
        ):
            converged = True
            break
        iters = it

        xc = np.mean(simplex[:-1], axis=0)
        xr = xc + alpha * (xc - simplex[-1])
        fr = float(obj(xr))
        if f_vals[0] <= fr < f_vals[-2]:
            simplex[-1], f_vals[-1] = xr, fr
            continue
        if fr < f_vals[0]:
            xe = xc + gamma * (xr - xc)
            fe = float(obj(xe))
            if fe < fr:
                simplex[-1], f_vals[-1] = xe, fe
            else:
                simplex[-1], f_vals[-1] = xr, fr
            continue
        if fr < f_vals[-1]:
            xoc = xc + rho * (xr - xc)
            foc = float(obj(xoc))
            if foc <= fr:
                simplex[-1], f_vals[-1] = xoc, foc
                continue
        else:
            xic = xc + rho * (simplex[-1] - xc)
            fic = float(obj(xic))
            if fic < f_vals[-1]:
                simplex[-1], f_vals[-1] = xic, fic
                continue
        for i in range(1, n + 1):
            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
            f_vals[i] = float(obj(simplex[i]))

    order = np.argsort(f_vals)
    return simplex[order[0]].copy(), float(f_vals[order[0]]), iters, converged


def model_match(
    dt_hr: Any,
    dp_psi: Any,
    tp_hr: Any,
    q_rb: Any,
    mu_cp: Any,
    h_ft: Any,
    phi: Any,
    ct: Any,
    rw_ft: Any,
    k0_md: Any,
    skin0: Any = 0.0,
    c0_bbl_psi: Any = 0.01,
    fit_storage: bool = True,
    max_iter: int = 300,
) -> dict[str, Any]:
    """Fit radial wellbore-storage + skin build-up model via adaptive Nelder-Mead."""
    if not isinstance(fit_storage, bool):
        raise TypeError("fit_storage must be a bool")
    if isinstance(max_iter, bool) or not isinstance(max_iter, int):
        raise TypeError("max_iter must be an int")

    notes: list[str] = []
    empty_res: dict[str, Any] = {
        "k_md": None,
        "kh_md_ft": None,
        "skin": None,
        "c_bbl_psi": None,
        "cd": None,
        "rms_psi": None,
        "iterations": 0,
        "converged": False,
        "dt_model_hr": None,
        "dp_model_psi": None,
        "deriv_model_psi": None,
        "notes": notes,
    }

    dt_valid, dp_valid, n_pts = _prepare_buildup_points(dt_hr, dp_psi, None, notes)
    if n_pts < 2:
        notes.append("At least 2 finite positive dt points are required for model matching")
        return empty_res

    tp = _check_scalar_type(tp_hr, "tp_hr")
    q = _check_scalar_type(q_rb, "q_rb")
    mu = _check_scalar_type(mu_cp, "mu_cp")
    h = _check_scalar_type(h_ft, "h_ft")
    ph = _check_scalar_type(phi, "phi")
    c_t = _check_scalar_type(ct, "ct")
    rw = _check_scalar_type(rw_ft, "rw_ft")
    k0 = _check_scalar_type(k0_md, "k0_md")
    s0 = _check_scalar_type(skin0, "skin0")
    c0 = _check_scalar_type(c0_bbl_psi, "c0_bbl_psi")

    if (
        tp is None or tp <= 0.0
        or q is None or q <= 0.0
        or mu is None or mu <= 0.0
        or h is None or h <= 0.0
        or ph is None or ph <= 0.0 or ph > 1.0
        or c_t is None or c_t <= 0.0
        or rw is None or rw <= 0.0
        or k0 is None or k0 <= 0.0
        or s0 is None
        or (fit_storage and (c0 is None or c0 <= 0.0))
        or max_iter < 1
    ):
        notes.append("Invalid physical parameter or initial guess for model_match")
        return empty_res

    if s0 < -6.0 or s0 > 50.0:
        s0 = min(50.0, max(-6.0, s0))
        notes.append("skin0 clamped to [-6, 50]")

    td_base = TD_TIME_COEFF / (ph * mu * c_t * (rw ** 2))
    cd_base = CD_STORAGE_COEFF / (ph * c_t * h * (rw ** 2))
    p_coeff_base = DARCY_P_COEFF * q * mu / h
    t_eval = np.concatenate((dt_valid, np.array([tp]), tp + dt_valid))

    def _eval_dp(k_val: float, s_val: float, c_val: float, times: np.ndarray) -> np.ndarray:
        td = (k_val * td_base) * times
        cd = c_val * cd_base
        pd = pwd(td, cd, s_val)
        m_len = (times.size - 1) // 2
        return (p_coeff_base / k_val) * (pd[:m_len] + pd[m_len] - pd[m_len + 1 :])

    def _objective(theta: np.ndarray) -> float:
        ln_k = float(theta[0])
        sk = float(theta[1])
        ln_c = float(theta[2]) if fit_storage else math.log(1e-6)

        penalty = 0.0
        if sk < -6.0:
            penalty += 1e6 * (1.0 + (-6.0 - sk) ** 2)
            sk_c = -6.0
        elif sk > 50.0:
            penalty += 1e6 * (1.0 + (sk - 50.0) ** 2)
            sk_c = 50.0
        else:
            sk_c = sk

        if ln_k < -15.0:
            penalty += 1e6 * (1.0 + (-15.0 - ln_k) ** 2)
            ln_k_c = -15.0
        elif ln_k > 15.0:
            penalty += 1e6 * (1.0 + (ln_k - 15.0) ** 2)
            ln_k_c = 15.0
        else:
            ln_k_c = ln_k

        if ln_c < -25.0:
            penalty += 1e6 * (1.0 + (-25.0 - ln_c) ** 2)
            ln_c_c = -25.0
        elif ln_c > 10.0:
            penalty += 1e6 * (1.0 + (ln_c - 10.0) ** 2)
            ln_c_c = 10.0
        else:
            ln_c_c = ln_c

        dp_pred = _eval_dp(math.exp(ln_k_c), sk_c, math.exp(ln_c_c), t_eval)
        diff = dp_pred - dp_valid
        return float(np.dot(diff, diff)) + penalty

    if fit_storage:
        assert c0 is not None
        theta0 = np.array([math.log(k0), s0, math.log(c0)], dtype=float)
        steps1 = np.array([0.5, 1.0, 0.5], dtype=float)
        steps2 = np.array([0.15, 0.3, 0.15], dtype=float)
    else:
        theta0 = np.array([math.log(k0), s0], dtype=float)
        steps1 = np.array([0.5, 1.0], dtype=float)
        steps2 = np.array([0.15, 0.3], dtype=float)

    budget1 = max(1, (2 * max_iter) // 3)
    best_theta, _, it1, conv1 = _nelder_mead(_objective, theta0, steps1, budget1)
    budget2 = max_iter - it1
    if budget2 > 0:
        best_theta, _, it2, conv2 = _nelder_mead(_objective, best_theta, steps2, budget2)
        total_iters = it1 + it2
        converged = conv2
    else:
        total_iters = it1
        converged = conv1

    k_fit = math.exp(float(np.clip(best_theta[0], -15.0, 15.0)))
    s_fit = float(np.clip(best_theta[1], -6.0, 50.0))
    c_fit = math.exp(float(np.clip(best_theta[2], -25.0, 10.0))) if fit_storage else 1e-6
    cd_fit = c_fit * cd_base

    dp_fit = _eval_dp(k_fit, s_fit, c_fit, t_eval)
    rms = float(np.sqrt(np.mean((dp_fit - dp_valid) ** 2)))

    dt_min = float(np.min(dt_valid))
    dt_max = float(np.max(dt_valid))
    if dt_max > dt_min:
        dt_grid = np.geomspace(dt_min, dt_max, 40)
        t_grid_eval = np.concatenate((dt_grid, np.array([tp]), tp + dt_grid))
        dp_grid = _eval_dp(k_fit, s_fit, c_fit, t_grid_eval)
        deriv_grid = np.gradient(dp_grid, np.log(dt_grid))
        dt_list: list[float] | None = [float(v) for v in dt_grid]
        dp_list: list[float] | None = [float(v) for v in dp_grid]
        deriv_list: list[float] | None = [float(v) for v in deriv_grid]
    else:
        notes.append("dt range has zero span; model curve grid omitted")
        dt_list, dp_list, deriv_list = None, None, None

    if not converged:
        notes.append("Nelder-Mead reached max_iter before convergence tolerance")

    return {
        "k_md": _clean_float(k_fit),
        "kh_md_ft": _clean_float(k_fit * h),
        "skin": _clean_float(s_fit),
        "c_bbl_psi": _clean_float(c_fit),
        "cd": _clean_float(cd_fit),
        "rms_psi": _clean_float(rms),
        "iterations": int(total_iters),
        "converged": bool(converged),
        "dt_model_hr": dt_list,
        "dp_model_psi": dp_list,
        "deriv_model_psi": deriv_list,
        "notes": notes,
    }


if __name__ == "__main__":
    k0_1, k1_1 = bessel_k01_scaled(np.array([1.0]))
    assert abs(float(k0_1[0]) * math.exp(-1.0) - 0.42102443824070834) < 1e-9
    assert abs(float(k1_1[0]) * math.exp(-1.0) - 0.6019072301972346) < 1e-9

    pd_radial = float(pwd(np.array([1e6]), 0.0, 0.0)[0])
    assert abs(pd_radial - 0.5 * (math.log(1e6) + 0.80907)) < 2e-3

    pd_stor = float(pwd(np.array([5.0]), 500.0, 0.0)[0])
    assert abs(pd_stor - 0.01) / 0.01 < 0.03

    k_true, h_true, s_true, c_true = 50.0, 40.0, 3.0, 0.0067
    phi_t, mu_t, ct_t, rw_t, q_t, tp_t = 0.2, 1.2, 1.2e-5, 0.354, 1000.0, 200.0
    dt_syn = np.geomspace(3e-4, 72.0, 60)
    td_fac = TD_TIME_COEFF * k_true / (phi_t * mu_t * ct_t * rw_t ** 2)
    cd_true = CD_STORAGE_COEFF * c_true / (phi_t * ct_t * h_true * rw_t ** 2)
    dp_clean = (DARCY_P_COEFF * q_t * mu_t / (k_true * h_true)) * (
        pwd(dt_syn * td_fac, cd_true, s_true)
        + pwd(np.array([tp_t * td_fac]), cd_true, s_true)[0]
        - pwd((tp_t + dt_syn) * td_fac, cd_true, s_true)
    )
    rng = np.random.default_rng(1)
    dp_noisy = dp_clean + rng.normal(0.0, 0.05, size=dt_syn.size)

    match_res = model_match(
        dt_syn, dp_noisy, tp_t, q_t, mu_t, h_true, phi_t, ct_t, rw_t,
        k0_md=20.0, skin0=0.0, c0_bbl_psi=0.02,
    )
    assert abs(match_res["k_md"] - k_true) / k_true < 0.05
    assert abs(match_res["skin"] - s_true) < 0.5
    assert abs(match_res["c_bbl_psi"] - c_true) / c_true < 0.15

    pwf_syn = 3000.0
    horner_res = horner_analysis(
        dt_syn, pwf_syn + dp_clean, tp_t, (5.0, 50.0),
        q_t, mu_t, h_true, phi_t, ct_t, rw_t, pwf_syn,
    )
    assert abs(horner_res["kh"] - k_true * h_true) / (k_true * h_true) < 0.05
    assert abs(horner_res["skin"] - s_true) < 0.6

    pi_bad = productivity_indices(1000.0, 1200.0, 2500.0, 2600.0)
    assert pi_bad["pi_actual_stbd_psi"] is None and len(pi_bad["notes"]) > 0

    print("pta methods self-checks passed")
