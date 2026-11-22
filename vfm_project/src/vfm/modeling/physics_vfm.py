"""Integrated physics-based Virtual Flow Meter (IPR + Choke + VLP WLS reconciler)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from vfm.physics.choke import calibrate_choke, predict_choke_q_liq
from vfm.physics.reservoir import (
    DEFAULT_REF_TIMESTAMP,
    fit_pr_and_pi,
    predict_ipr_q_liq,
)
from vfm.physics.vlp import calibrate_vlp, predict_vlp_q_liq


PHYSICS_PRED_COLS: Tuple[str, ...] = (
    "liquid_rate_stbd",
    "oil_rate_stbd",
    "water_rate_stbd",
    "gas_rate_mscfd",
    "total_gas_rate_mscfd",
    "water_cut_pct",
    "gor_scf_stb",
    "q_liq_ipr",
    "q_liq_choke",
    "q_liq_vlp",
    "wls_sigma_stbd",
    "reconcile_chi2",
    "is_shut_in",
    "is_transient",
    "is_sensor_missing",
    "is_reconcile_outlier",
    "is_ewma_transient",
    "whp_transient_diff",
    "dhp_transient_diff",
)


def physics_kwargs_from_config(cfg: Any) -> Dict[str, Any]:
    """Dataset assumptions for PhysicsVFM (no field priors), for callers that build it directly."""
    kw = physics_vfm_from_config(cfg).__dict__
    keys = ("wh_temp_degf", "choke_r_crit", "choke_closed_pct", "wc_model", "wc_logit_lambda", "whp_transient_psi",
            "dhp_transient_psi", "transient_window_hr", "ewma_fast_min", "ewma_slow_min")
    return {k: kw[k] for k in keys}


def physics_vfm_from_config(cfg: Any, **overrides: Any) -> "PhysicsVFM":
    """Build a PhysicsVFM with the dataset assumptions from a VFMConfig."""
    a = cfg.assumptions
    kwargs: Dict[str, Any] = {
        "ref_timestamp": cfg.reference_timestamp(DEFAULT_REF_TIMESTAMP),
        "wh_temp_degf": float(a.get("wellhead_temp_degf", 120.0)),
        "choke_r_crit": float(a.get("choke_critical_pressure_ratio", 0.55)),
        "choke_closed_pct": float(cfg.states.get("choke_closed_pct_threshold", 0.5)),
        "pr_gradient_prior_psi_ft": a.get("pr_gradient_prior_psi_ft"),
    }
    m = cfg.modeling
    for key in ("wc_model", "wc_logit_lambda", "whp_transient_psi", "dhp_transient_psi", "transient_window_hr",
                "ewma_fast_min", "ewma_slow_min"):
        if key in m:
            kwargs[key] = m[key]
    kwargs.update(overrides)
    return PhysicsVFM(**kwargs)


def _to_utc_timestamp(ts: Any) -> pd.Timestamp:
    """Parse timestamp-like input and normalize to UTC."""
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def _fit_ridge_wc_trend(
    t_days: np.ndarray,
    wc_frac: np.ndarray,
    ridge_lambda: float = 600.0,
    nonneg_slope: bool = True,
    max_slope_per_day: float = 0.003,
) -> Tuple[float, float, float]:
    """
    Fit a regularized linear water-cut trend WC(t) = wc_mean + slope * (t - t_mean).

    Returns `(t_mean, wc_mean, slope_per_day)`.
    """
    valid = np.isfinite(t_days) & np.isfinite(wc_frac)
    t_v = t_days[valid]
    w_v = np.clip(wc_frac[valid], 0.0, 0.99)
    n = len(t_v)
    if n == 0:
        return 0.0, 0.30, 0.0
    t_mean = float(np.mean(t_v))
    w_mean = float(np.mean(w_v))
    if n == 1:
        return t_mean, w_mean, 0.0

    dt = t_v - t_mean
    denom = float(np.sum(dt ** 2)) + float(ridge_lambda)
    if denom <= 1e-12:
        return t_mean, w_mean, 0.0

    slope = float(np.sum(dt * (w_v - w_mean)) / denom)
    lo_slope = 0.0 if nonneg_slope else -max_slope_per_day
    slope = float(np.clip(slope, lo_slope, max_slope_per_day))
    return t_mean, w_mean, slope


WC_EPS = 1e-3


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, WC_EPS, 1.0 - WC_EPS)
    return np.log(p / (1.0 - p))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def field_logit_wc_slope(
    well_tests: pd.DataFrame,
    ref_timestamp: pd.Timestamp,
    max_time: Optional[Any] = None,
    min_tests: int = 3,
) -> Optional[float]:
    """Pooled within-well slope of logit(WC) per day over wells with >= `min_tests` tests
    ending by `max_time` (causal). None if no well qualifies."""
    wt = well_tests.copy()
    wt["test_start"] = pd.to_datetime(wt["test_start"], utc=True)
    wt["test_end"] = pd.to_datetime(wt["test_end"], utc=True)
    if max_time is not None:
        wt = wt[wt["test_end"] <= _to_utc_timestamp(max_time)]
    if wt.empty or "water_cut_pct" not in wt:
        return None
    t = ((wt["test_start"] + (wt["test_end"] - wt["test_start"]) / 2) - ref_timestamp).dt.total_seconds() / 86400.0
    df = pd.DataFrame({"w": wt["well_name"].to_numpy(), "t": t.to_numpy(),
                       "y": _logit(wt["water_cut_pct"].to_numpy(dtype="float64") / 100.0)})
    counts = df.groupby("w")["t"].transform("size")
    df = df[(counts >= min_tests) & np.isfinite(df["y"])]
    if df.empty:
        return None
    dt = df["t"] - df.groupby("w")["t"].transform("mean")
    dy = df["y"] - df.groupby("w")["y"].transform("mean")
    sxx = float((dt * dt).sum())
    return float((dt * dy).sum() / sxx) if sxx > 0 else None


def _fit_logit_wc_trend(
    t_days: np.ndarray,
    wc_frac: np.ndarray,
    global_slope: Optional[float],
    shrink_lambda: float,
    nonneg_slope: bool,
) -> Tuple[float, float, float]:
    """logit(WC) = b0 + b1 (t - t_mean) with b1 shrunk toward the field slope:
    b1 = (Sxy + lambda * b_global) / (Sxx + lambda). Returns (t_mean, b0, b1)."""
    ok = np.isfinite(t_days) & np.isfinite(wc_frac)
    t, y = t_days[ok], _logit(wc_frac[ok])
    if len(t) == 0:
        return 0.0, float(_logit(np.array([0.30]))[0]), float(global_slope or 0.0)
    t_mean, y_mean = float(np.mean(t)), float(np.mean(y))
    dt = t - t_mean
    bg = float(global_slope) if global_slope is not None and np.isfinite(global_slope) else 0.0
    lam = float(shrink_lambda) if global_slope is not None else 0.0
    sxx = float(np.sum(dt * dt))
    b1 = (float(np.sum(dt * (y - y_mean))) + lam * bg) / (sxx + lam) if (sxx + lam) > 0 else bg
    if nonneg_slope:
        b1 = max(b1, 0.0)
    return t_mean, y_mean, float(b1)


def _ewma_transient_diagnostics(
    ts: pd.Series,
    whp: np.ndarray,
    dhp: np.ndarray,
    fast_min: float = 30.0,
    slow_min: float = 180.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Causal, time-aware EWMA deviations |x - EWMA_slow| for WHP and downhole pressure.

    `ewm(times=...)` decays by elapsed time, so a multi-hour SCADA gap effectively resets the
    average instead of carrying stale levels across it. The fast EWMA (30 min) is used as the
    signal to suppress single-minute noise."""
    out = []
    t = pd.DatetimeIndex(pd.to_datetime(ts, utc=True))
    for x in (whp, dhp):
        s = pd.Series(np.asarray(x, dtype="float64"), index=t)
        if s.notna().sum() == 0:
            out.append(np.full(len(s), np.nan))
            continue
        fast = s.ewm(halflife=pd.Timedelta(minutes=fast_min), times=t, ignore_na=True).mean()
        slow = s.ewm(halflife=pd.Timedelta(minutes=slow_min), times=t, ignore_na=True).mean()
        out.append(np.abs(fast.to_numpy() - slow.to_numpy()))
    return out[0], out[1]


def _fit_bayesian_gor(
    gor_tests: np.ndarray,
    rs_pb: float,
    kappa: float = 3.0,
) -> float:
    """
    Estimate undersaturated producing formation GOR via empirical-Bayes shrinkage
    toward solution GOR at bubble point `Rsb`.
    """
    rs_safe = float(max(rs_pb, 0.0))
    valid = np.isfinite(gor_tests) & (gor_tests >= 0.0)
    n = int(np.sum(valid))
    if n == 0:
        return rs_safe
    gor_bar = float(np.mean(gor_tests[valid]))
    gor_shrunk = (n * gor_bar + kappa * rs_safe) / (n + kappa)
    if rs_safe > 0.0:
        gor_shrunk = float(np.clip(gor_shrunk, 0.85 * rs_safe, 1.15 * rs_safe))
    return float(max(gor_shrunk, 0.0))


def _effective_weight(
    params: Optional[Mapping[str, Any]],
    sigma_key: str,
    n_tests: int,
    sigma_floor: float,
    require_loto_if_ge2: bool = True,
) -> Tuple[float, float]:
    """
    Convert LOTO RMSE into inverse-variance WLS weight with relative/absolute floor.

    Returns `(weight, sigma_effective)`.
    """
    if not params or n_tests <= 0:
        return 0.0, np.inf
    if require_loto_if_ge2 and n_tests >= 2 and int(params.get("n_valid_loto", 0)) <= 0:
        return 0.0, np.inf
    sigma_raw = float(params.get(sigma_key, np.inf))
    if not np.isfinite(sigma_raw) or sigma_raw >= 1e5:
        return 0.0, np.inf
    sigma_eff = max(sigma_raw, sigma_floor)
    return 1.0 / (sigma_eff ** 2), sigma_eff


@dataclass
class PhysicsVFM:
    """
    Mechanistic Virtual Flow Meter combining per-well IPR, multiphase choke,
    and two-segment tubing VLP via inverse-variance WLS reconciliation.

    Water cut `WC_w(t)` and formation `GOR_w` are modeled as slow reservoir/fluid
    trends from well tests rather than fitted to noisy instantaneous sensors.
    """

    ref_timestamp: pd.Timestamp = DEFAULT_REF_TIMESTAMP
    wc_ridge_lambda: float = 600.0
    wc_nonneg_slope: bool = True
    gor_shrinkage_kappa: float = 3.0
    min_sigma_rel: float = 0.015
    min_sigma_abs: float = 15.0
    choke_closed_pct: float = 0.5
    dp_stall_psi: float = 2.0
    min_vlp_sens: float = 0.015
    chi2_outlier_threshold: float = 9.21
    # Pr/datum-TVD prior for wells without a build-up. None = median of the build-ups seen
    # in this fit (needs several wells); pass it explicitly when fitting one well at a time.
    pr_gradient_prior_psi_ft: Optional[float] = None
    # Assumptions with no sensor behind them (configurable per dataset)
    wh_temp_degf: float = 120.0
    choke_r_crit: float = 0.55
    # Optional per-well IPR adjustments from the PTA coupling layer (vfm.coupling):
    # {well: {"anchors": [{"timestamp", "pr_datum_psia", ...}] | None, "j_multiplier": callable | None}}.
    # None (default) = the uncoupled baseline.
    ipr_adjustments: Optional[Dict[str, Dict[str, Any]]] = None
    # Water-cut trend: "logit" (bounded, shrunk to the field slope) or "linear" (legacy ridge)
    wc_model: str = "logit"
    wc_logit_lambda: float = 20000.0          # days^2; weight of the field slope in each well's fit
    wc_global_slope: Optional[float] = None   # field slope; None = computed from the fit data (causal)
    # EWMA transient detection (minutes, psi)
    ewma_fast_min: float = 30.0
    ewma_slow_min: float = 180.0
    whp_transient_psi: float = 15.0
    dhp_transient_psi: float = 20.0
    transient_window_hr: float = 24.0
    ipr_params_: Dict[str, Dict[str, Any]] = field(default_factory=dict, init=False)
    choke_params_: Dict[str, Dict[str, Any]] = field(default_factory=dict, init=False)
    vlp_params_: Dict[str, Dict[str, Any]] = field(default_factory=dict, init=False)
    well_models_: Dict[str, Dict[str, Any]] = field(default_factory=dict, init=False)
    weights_df_: pd.DataFrame = field(default_factory=pd.DataFrame, init=False)

    def fit(
        self,
        well_tests_df: pd.DataFrame,
        hf_cache_by_well: Optional[Mapping[str, pd.DataFrame]],
        completion_df: pd.DataFrame,
        pvt_df: pd.DataFrame,
        max_time: Optional[Any] = None,
    ) -> "PhysicsVFM":
        """
        Calibrate per-well IPR, Choke, VLP, slow WC(t)/GOR trends, and WLS weights.

        When `max_time` is provided, strictly filters both `well_tests_df`
        (`test_end <= max_time`) and `hf_cache_by_well` (`timestamp <= max_time`)
        for zero-leakage chronological walk-forward validation.
        """
        wt = well_tests_df.copy()
        wt["test_start"] = pd.to_datetime(wt["test_start"], utc=True)
        wt["test_end"] = pd.to_datetime(wt["test_end"], utc=True)

        max_ts = _to_utc_timestamp(max_time) if max_time is not None else None
        if max_ts is not None:
            wt = wt[wt["test_end"] <= max_ts].copy()

        hf_causal: Dict[str, pd.DataFrame] = {}
        if hf_cache_by_well is not None:
            for well, df in hf_cache_by_well.items():
                if df is None or df.empty:
                    hf_causal[well] = df
                    continue
                if max_ts is not None:
                    ts = pd.to_datetime(df["timestamp"], utc=True)
                    hf_causal[well] = df[ts <= max_ts]
                else:
                    hf_causal[well] = df

        self.ipr_params_ = fit_pr_and_pi(
            well_tests=wt,
            hf_cache_by_well=hf_causal,
            pvt=pvt_df,
            completion=completion_df,
            max_time=max_ts,
            ref_timestamp=self.ref_timestamp,
            pr_gradient_prior_psi_ft=self.pr_gradient_prior_psi_ft,
            external_anchors={w: a["anchors"] for w, a in (self.ipr_adjustments or {}).items()
                              if a.get("anchors") is not None},
            j_multipliers={w: a["j_multiplier"] for w, a in (self.ipr_adjustments or {}).items()
                           if a.get("j_multiplier") is not None},
            fixed_j={w: a["fixed_j"] for w, a in (self.ipr_adjustments or {}).items() if a.get("fixed_j")},
        )
        self.choke_params_ = calibrate_choke(
            well_tests=wt,
            hf_cache_by_well=hf_causal,
            completion=completion_df,
            pvt=pvt_df,
            max_time=max_ts,
            r_crit=self.choke_r_crit,
            wh_temp_degf=self.wh_temp_degf,
            closed_pct_threshold=self.choke_closed_pct,
        )
        self.vlp_params_ = calibrate_vlp(
            well_tests=wt,
            completion=completion_df,
            pvt=pvt_df,
            max_time=max_ts,
            wh_temp_degf=self.wh_temp_degf,
            min_sens_psi_per_stbd=self.min_vlp_sens,
        )

        comp_map = completion_df.set_index("well_name").to_dict(orient="index")
        self.well_models_ = {}
        summary_rows: List[Dict[str, Any]] = []
        self.wc_global_slope_ = (self.wc_global_slope if self.wc_global_slope is not None
                                 else field_logit_wc_slope(wt, self.ref_timestamp, max_ts))

        for well in sorted(comp_map.keys()):
            c_row = comp_map[well]
            rs_pb = float(c_row.get("solution_gor_at_pb_scf_stb", 250.0))
            sub = wt[wt["well_name"] == well].sort_values("test_start")
            n_tests = int(len(sub))

            if n_tests > 0:
                t_mid = sub["test_start"] + (sub["test_end"] - sub["test_start"]) / 2
                t_days = (t_mid - self.ref_timestamp).dt.total_seconds().to_numpy(dtype="float64") / 86400.0
                if "water_cut_pct" in sub.columns:
                    wc_vals = sub["water_cut_pct"].to_numpy(dtype="float64") / 100.0
                else:
                    q_l = np.maximum(sub["oil_rate_stbd"] + sub["water_rate_stbd"], 1e-6)
                    wc_vals = (sub["water_rate_stbd"] / q_l).to_numpy(dtype="float64")
                gor_vals = (
                    sub["gor_scf_stb"].to_numpy(dtype="float64")
                    if "gor_scf_stb" in sub.columns
                    else np.full(n_tests, rs_pb, dtype="float64")
                )
                q_bar = float(np.nanmean(sub["liquid_rate_stbd"]))
                default_api = float(np.nanmedian(sub["oil_gravity_api"])) if "oil_gravity_api" in sub else 35.0
                default_gsg = float(np.nanmedian(sub["gas_sg"])) if "gas_sg" in sub else 0.75
                default_sal = float(np.nanmedian(sub["water_salinity_ppm"])) if "water_salinity_ppm" in sub else 100_000.0
            else:
                t_days = np.array([], dtype="float64")
                wc_vals = np.array([], dtype="float64")
                gor_vals = np.array([], dtype="float64")
                q_bar = 1500.0
                default_api, default_gsg, default_sal = 35.0, 0.75, 100_000.0

            if self.wc_model == "logit":
                wc_t_mean, wc_mean, wc_slope = _fit_logit_wc_trend(
                    t_days, wc_vals, self.wc_global_slope_, self.wc_logit_lambda, self.wc_nonneg_slope)
            else:
                wc_t_mean, wc_mean, wc_slope = _fit_ridge_wc_trend(
                    t_days, wc_vals, ridge_lambda=self.wc_ridge_lambda, nonneg_slope=self.wc_nonneg_slope)
            gor_hat = _fit_bayesian_gor(gor_vals, rs_pb=rs_pb, kappa=self.gor_shrinkage_kappa)

            sigma_floor = max(self.min_sigma_rel * q_bar, self.min_sigma_abs)
            ipr_p = self.ipr_params_.get(well)
            chk_p = self.choke_params_.get(well)
            vlp_p = self.vlp_params_.get(well)

            w_ipr, sig_ipr_eff = _effective_weight(ipr_p, "sigma_ipr", n_tests, sigma_floor, require_loto_if_ge2=False)
            w_chk, sig_chk_eff = _effective_weight(chk_p, "sigma_choke", n_tests, sigma_floor, require_loto_if_ge2=True)
            w_vlp, sig_vlp_eff = _effective_weight(vlp_p, "sigma_vlp", n_tests, sigma_floor, require_loto_if_ge2=True)

            # If all LOTO weights collapsed (e.g. small sample), fall back to finite in-sample sigmas
            if (w_ipr + w_chk + w_vlp) <= 0.0 and n_tests > 0:
                w_ipr, sig_ipr_eff = _effective_weight(ipr_p, "sigma_ipr", n_tests, sigma_floor, require_loto_if_ge2=False)
                w_chk, sig_chk_eff = _effective_weight(chk_p, "sigma_choke", n_tests, sigma_floor, require_loto_if_ge2=False)
                w_vlp, sig_vlp_eff = _effective_weight(vlp_p, "sigma_vlp", n_tests, sigma_floor, require_loto_if_ge2=False)

            w_tot = w_ipr + w_chk + w_vlp
            self.well_models_[well] = {
                "well_name": well,
                "n_tests": n_tests,
                "ipr_params": ipr_p,
                "choke_params": chk_p,
                "vlp_params": vlp_p,
                "w_ipr": w_ipr,
                "w_choke": w_chk,
                "w_vlp": w_vlp,
                "sigma_ipr_eff": sig_ipr_eff,
                "sigma_choke_eff": sig_chk_eff,
                "sigma_vlp_eff": sig_vlp_eff,
                "wc_t_mean": wc_t_mean,
                "wc_mean": wc_mean,
                "wc_slope": wc_slope,
                "wc_model": self.wc_model,
                "gor_hat": gor_hat,
                "rs_pb": rs_pb,
                "default_api": default_api,
                "default_gas_sg": default_gsg,
                "default_salinity_ppm": default_sal,
            }

            if n_tests > 0:
                summary_rows.append({
                    "well_name": well,
                    "n_tests": n_tests,
                    "J": float(ipr_p["J"]) if ipr_p else np.nan,
                    "Pr0_datum_psia": float(ipr_p["Pr0_datum_psia"]) if ipr_p else np.nan,
                    "decline_psi_per_day": float(ipr_p["decline_psi_per_day"]) if ipr_p else np.nan,
                    "cd": float(chk_p["cd"]) if chk_p else np.nan,
                    "slip_k": float(vlp_p["slip_k"]) if vlp_p else np.nan,
                    "wc_mean": wc_mean,
                    "wc_slope_per_day": wc_slope,
                    "gor_hat": gor_hat,
                    "rs_pb": rs_pb,
                    "sigma_ipr": float(ipr_p["sigma_ipr"]) if ipr_p else np.nan,
                    "sigma_choke": float(chk_p["sigma_choke"]) if chk_p else np.nan,
                    "sigma_vlp": float(vlp_p["sigma_vlp"]) if vlp_p else np.nan,
                    "w_norm_ipr": w_ipr / w_tot if w_tot > 0 else 0.0,
                    "w_norm_choke": w_chk / w_tot if w_tot > 0 else 0.0,
                    "w_norm_vlp": w_vlp / w_tot if w_tot > 0 else 0.0,
                })

        self.weights_df_ = pd.DataFrame(summary_rows)
        return self

    def _predict_single_well(self, well: str, df: pd.DataFrame) -> pd.DataFrame:
        """Score a DataFrame belonging to a single well."""
        n = len(df)
        if well not in self.well_models_:
            raise KeyError(f"Well '{well}' was not found in PhysicsVFM fitted models.")
        wm = self.well_models_[well]

        if "timestamp" in df.columns:
            ts = pd.to_datetime(df["timestamp"], utc=True)
        elif "test_start" in df.columns and "test_end" in df.columns:
            t0 = pd.to_datetime(df["test_start"], utc=True)
            t1 = pd.to_datetime(df["test_end"], utc=True)
            ts = t0 + (t1 - t0) / 2
        else:
            raise KeyError("Input DataFrame must contain 'timestamp' or ('test_start', 'test_end').")

        t_days = (ts - self.ref_timestamp).dt.total_seconds().to_numpy(dtype="float64") / 86400.0
        if wm.get("wc_model", "linear") == "logit":
            wc_pred = _sigmoid(wm["wc_mean"] + wm["wc_slope"] * (t_days - wm["wc_t_mean"]))
        else:
            wc_pred = np.clip(wm["wc_mean"] + wm["wc_slope"] * (t_days - wm["wc_t_mean"]), 0.0, 0.99)
        # gor_hat (shrunk test GOR) drives the formation-gas allocation. The choke and VLP
        # estimators were calibrated with GOR = Rs(Pb) (undersaturated flow), so they must be
        # evaluated with the same value or their calibrated Cd/slip would be biased.
        gor_pred = np.full(n, wm["gor_hat"], dtype="float64")
        gor_hydraulic = np.full(n, wm["rs_pb"], dtype="float64")

        has_choke_col = "choke_opening_pct" in df.columns
        chk_arr = df["choke_opening_pct"].to_numpy(dtype="float64") if has_choke_col else np.full(n, np.nan)
        whp_arr = df["whp_psia"].to_numpy(dtype="float64") if "whp_psia" in df.columns else np.full(n, np.nan)
        ds_arr = df["whp_downstream_psia"].to_numpy(dtype="float64") if "whp_downstream_psia" in df.columns else np.full(n, np.nan)
        dhp_arr = df["dh_pressure_psia"].to_numpy(dtype="float64") if "dh_pressure_psia" in df.columns else np.full(n, np.nan)
        qgl_arr = df["gas_lift_rate_mscfd"].to_numpy(dtype="float64") if "gas_lift_rate_mscfd" in df.columns else np.full(n, np.nan)

        api_arr = df["oil_gravity_api"].to_numpy(dtype="float64") if "oil_gravity_api" in df.columns else wm["default_api"]
        gsg_arr = df["gas_sg"].to_numpy(dtype="float64") if "gas_sg" in df.columns else wm["default_gas_sg"]
        sal_arr = df["water_salinity_ppm"].to_numpy(dtype="float64") if "water_salinity_ppm" in df.columns else wm["default_salinity_ppm"]

        # True physical shut-in: closed choke (immune to dp_choke <= 2 psi noise spikes)
        if has_choke_col:
            fallback_si = df["is_shut_in"].to_numpy(dtype=bool) if "is_shut_in" in df.columns else np.zeros(n, dtype=bool)
            is_shut_in = np.where(np.isfinite(chk_arr), chk_arr <= self.choke_closed_pct, fallback_si)
        elif "is_shut_in" in df.columns:
            is_shut_in = df["is_shut_in"].to_numpy(dtype=bool)
        else:
            is_shut_in = np.zeros(n, dtype=bool)

        dp_choke = whp_arr - ds_arr
        is_flow_stall = (~is_shut_in) & np.isfinite(dp_choke) & (dp_choke <= self.dp_stall_psi)
        base_transient = df["is_transient"].to_numpy(dtype=bool) if "is_transient" in df.columns else np.zeros(n, dtype=bool)
        is_transient = (~is_shut_in) & (base_transient | is_flow_stall)

        # Evaluate individual physical estimators (NaN preserved on missing/ill-conditioned inputs)
        if wm["ipr_params"] is not None:
            q_ipr = predict_ipr_q_liq(
                wm["ipr_params"],
                timestamps=ts,
                dh_pressure_psia=dhp_arr,
                wc_frac=wc_pred,
                oil_api=api_arr,
                gas_sg=gsg_arr,
                salinity_ppm=sal_arr,
            )
            q_ipr = np.where(np.isfinite(q_ipr), np.maximum(q_ipr, 0.0), np.nan)
        else:
            q_ipr = np.full(n, np.nan, dtype="float64")

        if wm["choke_params"] is not None:
            q_chk = predict_choke_q_liq(
                wm["choke_params"],
                choke_opening_pct=chk_arr,
                whp_psia=whp_arr,
                whp_downstream_psia=ds_arr,
                gas_lift_rate_mscfd=qgl_arr,
                wc_frac=wc_pred,
                gor_scf_stb=gor_hydraulic,
                oil_api=api_arr,
                gas_sg=gsg_arr,
                water_salinity_ppm=sal_arr,
            )
            q_chk = np.where(np.isfinite(q_chk), np.maximum(q_chk, 0.0), np.nan)
        else:
            q_chk = np.full(n, np.nan, dtype="float64")

        if wm["vlp_params"] is not None:
            q_vlp = predict_vlp_q_liq(
                wm["vlp_params"],
                whp_psia=whp_arr,
                dh_pressure_psia=dhp_arr,
                gas_lift_rate_mscfd=qgl_arr,
                wc_frac=wc_pred,
                gor_scf_stb=gor_hydraulic,
                oil_api=api_arr,
                gas_sg=gsg_arr,
                water_salinity_ppm=sal_arr,
                min_sens_psi_per_stbd=self.min_vlp_sens,
            )
            q_vlp = np.where(np.isfinite(q_vlp), np.maximum(q_vlp, 0.0), np.nan)
        else:
            q_vlp = np.full(n, np.nan, dtype="float64")

        # Inverse-variance WLS reconciliation over active/finite estimators
        w_i, w_c, w_v = wm["w_ipr"], wm["w_choke"], wm["w_vlp"]
        valid_i = (~is_shut_in) & np.isfinite(q_ipr) & (w_i > 0.0)
        valid_c = (~is_shut_in) & np.isfinite(q_chk) & (w_c > 0.0) & (~is_flow_stall)
        valid_v = (~is_shut_in) & np.isfinite(q_vlp) & (w_v > 0.0)

        w_sum = (
            np.where(valid_i, w_i, 0.0)
            + np.where(valid_c, w_c, 0.0)
            + np.where(valid_v, w_v, 0.0)
        )
        wq_sum = (
            np.where(valid_i, w_i * q_ipr, 0.0)
            + np.where(valid_c, w_c * q_chk, 0.0)
            + np.where(valid_v, w_v * q_vlp, 0.0)
        )
        has_valid = w_sum > 0.0
        q_liq = np.where(has_valid, wq_sum / np.where(has_valid, w_sum, 1.0), np.nan)
        q_liq = np.where(np.isfinite(q_liq), np.maximum(q_liq, 0.0), np.nan)
        wls_sigma = np.where(has_valid, 1.0 / np.sqrt(np.where(has_valid, w_sum, 1.0)), np.nan)

        n_est = valid_i.astype(int) + valid_c.astype(int) + valid_v.astype(int)
        chi2_raw = (
            np.where(valid_i, w_i * ((q_ipr - q_liq) ** 2), 0.0)
            + np.where(valid_c, w_c * ((q_chk - q_liq) ** 2), 0.0)
            + np.where(valid_v, w_v * ((q_vlp - q_liq) ** 2), 0.0)
        )
        reconcile_chi2 = np.where(n_est >= 2, chi2_raw, np.where(has_valid, 0.0, np.nan))
        is_sensor_missing = (~is_shut_in) & (~has_valid)
        is_reconcile_outlier = (~is_shut_in) & (n_est >= 2) & np.isfinite(reconcile_chi2) & (reconcile_chi2 > self.chi2_outlier_threshold)

        # EWMA transient diagnostics (causal): unloading / wellbore-storage transients after a
        # choke event show as a gap between the fast and slow pressure averages, and often as
        # estimator disagreement (chi2). Only flagged within `transient_window_hr` of a choke event.
        whp_dev, dhp_dev = _ewma_transient_diagnostics(ts, whp_arr, dhp_arr, self.ewma_fast_min, self.ewma_slow_min)
        hrs_since = (df["minutes_since_choke_change"].to_numpy(dtype="float64") / 60.0
                     if "minutes_since_choke_change" in df.columns else np.full(n, np.inf))
        near_event = hrs_since < self.transient_window_hr
        is_ewma_transient = (~is_shut_in) & near_event & (
            (np.nan_to_num(whp_dev) > self.whp_transient_psi) | (np.nan_to_num(dhp_dev) > self.dhp_transient_psi)
            | is_reconcile_outlier)
        is_transient = is_transient | is_ewma_transient

        # Phase rate reconstruction & physical bounds
        q_oil = q_liq * (1.0 - wc_pred)
        q_water = q_liq * wc_pred
        q_gas_form = (q_oil * gor_pred) / 1000.0
        q_gas_tot = np.where(
            np.isfinite(q_gas_form) & np.isfinite(qgl_arr),
            q_gas_form + np.maximum(qgl_arr, 0.0),
            np.nan,
        )

        # Exact zeroing on shut-in intervals
        q_liq = np.where(is_shut_in, 0.0, q_liq)
        q_oil = np.where(is_shut_in, 0.0, q_oil)
        q_water = np.where(is_shut_in, 0.0, q_water)
        q_gas_form = np.where(is_shut_in, 0.0, q_gas_form)
        q_gas_tot = np.where(is_shut_in, 0.0, q_gas_tot)
        wls_sigma = np.where(is_shut_in, 0.0, wls_sigma)
        reconcile_chi2 = np.where(is_shut_in, 0.0, reconcile_chi2)

        return pd.DataFrame(
            {
                "liquid_rate_stbd": q_liq,
                "oil_rate_stbd": q_oil,
                "water_rate_stbd": q_water,
                "gas_rate_mscfd": q_gas_form,
                "total_gas_rate_mscfd": q_gas_tot,
                "water_cut_pct": wc_pred * 100.0,
                "gor_scf_stb": gor_pred,
                "q_liq_ipr": q_ipr,
                "q_liq_choke": q_chk,
                "q_liq_vlp": q_vlp,
                "wls_sigma_stbd": wls_sigma,
                "reconcile_chi2": reconcile_chi2,
                "is_shut_in": is_shut_in.astype(bool),
                "is_transient": is_transient.astype(bool),
                "is_sensor_missing": is_sensor_missing.astype(bool),
                "is_reconcile_outlier": is_reconcile_outlier.astype(bool),
                "is_ewma_transient": is_ewma_transient.astype(bool),
                "whp_transient_diff": whp_dev,
                "dhp_transient_diff": dhp_dev,
            },
            index=df.index,
        )

    def predict(self, hf_df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict 1-minute (or well-test-level) reconciled phase rates and QC diagnostics.

        Handles single-well or multi-well DataFrames, cleanly zeroes shut-ins,
        reconciles over whichever physics estimators have valid sensors, and
        enforces physical bounds (`q >= 0`, `0 <= WC <= 1`, `total_gas >= q_gl`).
        """
        if not self.well_models_:
            raise RuntimeError("PhysicsVFM must be fitted before calling predict().")

        if hf_df.empty:
            return pd.DataFrame(columns=list(PHYSICS_PRED_COLS), index=hf_df.index)

        if "well_name" not in hf_df.columns:
            if len(self.well_models_) == 1:
                only_well = next(iter(self.well_models_))
                return self._predict_single_well(only_well, hf_df)
            raise KeyError("Input DataFrame must contain 'well_name' column.")

        unique_wells = hf_df["well_name"].unique()
        if len(unique_wells) == 1:
            return self._predict_single_well(str(unique_wells[0]), hf_df)

        parts: List[pd.DataFrame] = []
        for well, grp in hf_df.groupby("well_name", sort=False):
            parts.append(self._predict_single_well(str(well), grp))
        return pd.concat(parts, axis=0).reindex(hf_df.index)
