"""Allocation model zoo: interchangeable rate models with one interface.

Every model follows the same contract so the benchmark, the daily allocation and the dashboard
can swap them freely:

    model = make_allocation_model("gbm")
    model.fit(well_tests, hf_by_well, completion, pvt, max_time=None, context={...})
    preds = model.predict(hf_well, well)     # 1-minute frame, columns OUT_COLUMNS

`fit` must only use well tests with test_end <= max_time and telemetry <= max_time (causal);
`context` carries field-level quantities that are themselves computed causally by the caller
(reservoir-pressure prior, field water-cut slope, PTA events). Rates are surface STB/D and
Mscf/D; shut-in minutes (choke closed) are exactly zero for every model.

Families:
  physics       PhysicsVFM (WLS of IPR/choke/VLP) and its single-estimator variants
  test-based    last test carried forward; linear interpolation between tests (hindcast only)
  regression/ML ridge, gradient boosting, random forest, k-nearest neighbours on choke/pressure
                features (numpy-only, vfm.modeling.ml_numpy), pooled across wells on *relative*
                rates (each well's rates and features divided by its own training means), because
                85 tests are far too few for per-well ML
  hybrid        PhysicsVFM x ML correction factor
  convolution   pressure-rate deconvolution of the downhole gauge with a radial + closed-boundary
                unit response (kh/skin from an accepted PTA, else from the IPR productivity index),
                scale-calibrated to past well tests
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from vfm.allocation.deconvolution import deconvolve_rates, radial_unit_response
from vfm.modeling.ml_numpy import make_model
from vfm.modeling.physics_vfm import (
    PhysicsVFM,
    _fit_bayesian_gor,
    _fit_logit_wc_trend,
    _sigmoid,
    field_logit_wc_slope,
)
from vfm.physics.choke import _effective_choke_dp, choke_bean_diameter_in
from vfm.physics.reservoir import DEFAULT_REF_TIMESTAMP, fit_pr_and_pi

OUT_COLUMNS = ("liquid_rate_stbd", "oil_rate_stbd", "water_rate_stbd", "gas_rate_mscfd",
               "total_gas_rate_mscfd", "water_cut_pct", "gor_scf_stb", "is_shut_in", "is_transient",
               "is_sensor_missing", "is_reconcile_outlier")


def _utc(ts: Any) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def _causal_tests(tests: pd.DataFrame, max_time: Optional[Any]) -> pd.DataFrame:
    wt = tests.copy()
    wt["test_start"] = pd.to_datetime(wt["test_start"], utc=True)
    wt["test_end"] = pd.to_datetime(wt["test_end"], utc=True)
    if max_time is not None:
        wt = wt[wt["test_end"] <= _utc(max_time)]
    return wt.sort_values(["well_name", "test_start"])


def _shut_in(hf: pd.DataFrame, closed_pct: float = 0.5) -> np.ndarray:
    fb = hf["is_shut_in"].fillna(False).to_numpy(dtype=bool) if "is_shut_in" in hf.columns else np.zeros(len(hf), bool)
    if "choke_opening_pct" in hf.columns:
        c = hf["choke_opening_pct"].to_numpy(dtype="float64")
        return np.where(np.isfinite(c), c <= closed_pct, fb)
    return fb


def _days(ts: pd.Series, ref: pd.Timestamp) -> np.ndarray:
    return (pd.to_datetime(ts, utc=True) - ref).dt.total_seconds().to_numpy(dtype="float64") / 86400.0


def _gas_lift(hf: pd.DataFrame) -> np.ndarray:
    """Lift-gas rate (NaN where the meter is missing, as in PhysicsVFM; 0 if the well has no meter)."""
    if "gas_lift_rate_mscfd" not in hf.columns:
        return np.zeros(len(hf))
    return np.maximum(hf["gas_lift_rate_mscfd"].to_numpy(dtype="float64"), 0.0)


def _fnum(row: Mapping[str, Any], key: str, default: float) -> float:
    try:
        v = float(row.get(key, default))
    except (TypeError, ValueError):
        return default
    return v if np.isfinite(v) else default


def _fingerprint(*parts: Any) -> str:
    import hashlib
    return hashlib.sha1(repr(parts).encode()).hexdigest()[:16]


class PhaseSplit:
    """Per-well slow water-cut (logit trend shrunk to the field slope) and Rs-shrunk GOR,
    shared by every liquid-rate model so phase splitting never differs between models."""

    def __init__(self, ref: pd.Timestamp = DEFAULT_REF_TIMESTAMP, lam: float = 20000.0):
        self.ref, self.lam, self.params = ref, lam, {}

    def fit(self, wt: pd.DataFrame, completion: pd.DataFrame, global_slope: Optional[float]) -> "PhaseSplit":
        comp = completion.set_index("well_name")
        for well, sub in wt.groupby("well_name"):
            t = _days(sub["test_start"] + (sub["test_end"] - sub["test_start"]) / 2, self.ref)
            tm, b0, b1 = _fit_logit_wc_trend(t, sub["water_cut_pct"].to_numpy(dtype="float64") / 100.0,
                                             global_slope, self.lam, True)
            rs = _fnum(comp.loc[well], "solution_gor_at_pb_scf_stb", 250.0) if well in comp.index else 250.0
            self.params[well] = {"tm": tm, "b0": b0, "b1": b1,
                                 "gor": _fit_bayesian_gor(sub["gor_scf_stb"].to_numpy(dtype="float64"), rs)}
        return self

    def apply(self, well: str, ts: pd.Series, q_liq: np.ndarray, q_gl: np.ndarray, si: np.ndarray) -> Dict[str, np.ndarray]:
        p = self.params.get(well)
        n = len(q_liq)
        if p is None:
            wc = np.full(n, np.nan)
            gor = np.full(n, np.nan)
        else:
            wc = _sigmoid(p["b0"] + p["b1"] * (_days(ts, self.ref) - p["tm"]))
            gor = np.full(n, p["gor"])
        q = np.where(si, 0.0, np.maximum(q_liq, 0.0))
        oil, wat = q * (1 - wc), q * wc
        gas = oil * gor / 1000.0
        qgl = np.where(si, 0.0, np.maximum(np.asarray(q_gl, dtype="float64"), 0.0))
        return {"liquid_rate_stbd": q, "oil_rate_stbd": oil, "water_rate_stbd": wat, "gas_rate_mscfd": gas,
                "total_gas_rate_mscfd": np.where(si, 0.0, gas + qgl), "water_cut_pct": wc * 100.0, "gor_scf_stb": gor}


def _frame(hf: pd.DataFrame, cols: Dict[str, np.ndarray], si: np.ndarray, extra: Optional[Dict[str, np.ndarray]] = None) -> pd.DataFrame:
    n = len(hf)
    liq = cols["liquid_rate_stbd"]
    out = {**cols, "is_shut_in": si,
           "is_transient": hf["is_transient"].to_numpy(dtype=bool) & ~si if "is_transient" in hf.columns else np.zeros(n, bool),
           "is_sensor_missing": (~si) & ~np.isfinite(liq), "is_reconcile_outlier": np.zeros(n, bool)}
    out.update(extra or {})
    return pd.DataFrame(out, index=hf.index)


class AllocationModel:
    name = "base"
    family = "base"
    description = ""
    causal = True              # False: uses future tests (hindcast only, excluded from walk-forward)
    needs_history = False      # True: predict() needs telemetry from the start of the data
    pooled = False             # True: trained across wells (benchmark passes all wells' tests)

    def fit(self, well_tests, hf_by_well, completion, pvt, max_time=None, context=None):  # pragma: no cover
        raise NotImplementedError

    def predict(self, hf_well: pd.DataFrame, well: str) -> pd.DataFrame:  # pragma: no cover
        raise NotImplementedError

    def info(self) -> Dict[str, Any]:
        return {"name": self.name, "family": self.family, "description": self.description,
                "causal": self.causal, "needs_history": self.needs_history, "pooled": self.pooled}


# ----------------------------------------------------------------------------- physics
class PhysicsModel(AllocationModel):
    family = "physics"

    def __init__(self, name: str = "physics_wls", estimators: Tuple[str, ...] = ("ipr", "choke", "vlp"),
                 physics_kwargs: Optional[Dict[str, Any]] = None):
        self.name = name
        self.estimators = estimators
        self.physics_kwargs = physics_kwargs or {}
        self.description = ("PhysicsVFM: inverse-variance WLS of IPR, choke and VLP" if len(estimators) == 3
                            else f"PhysicsVFM {estimators[0].upper()} estimator only")

    def fit(self, well_tests, hf_by_well, completion, pvt, max_time=None, context=None):
        ctx = context or {}
        # the variants share one fit per fold; the key covers data, priors and settings
        key = ("physics", _fingerprint(sorted(hf_by_well), str(max_time), ctx.get("fold_key"), ctx.get("pr_prior"),
                                       ctx.get("wc_global_slope"), str(ctx.get("ref_timestamp")),
                                       sorted((ctx.get("physics_kwargs") or {}).items()), sorted(self.physics_kwargs.items()),
                                       list(zip(well_tests["well_name"], well_tests["test_start"].astype(str)))))
        cache = ctx.get("physics_cache")
        if cache is not None and key in cache:
            self.model = cache[key]
        else:
            kw = {"pr_gradient_prior_psi_ft": ctx.get("pr_prior"), "wc_global_slope": ctx.get("wc_global_slope"),
                  **(ctx.get("physics_kwargs") or {}), **self.physics_kwargs}
            if ctx.get("ref_timestamp") is not None:
                kw["ref_timestamp"] = ctx["ref_timestamp"]
            self.model = PhysicsVFM(**kw).fit(well_tests, hf_by_well, completion, pvt, max_time=max_time)
            if cache is not None:
                cache[key] = self.model
        return self

    def predict(self, hf_well, well):
        pred = self.model.predict(hf_well.assign(well_name=well) if "well_name" not in hf_well else hf_well)
        if len(self.estimators) == 3:
            return pred
        col = {"ipr": "q_liq_ipr", "choke": "q_liq_choke", "vlp": "q_liq_vlp"}[self.estimators[0]]
        si = pred["is_shut_in"].to_numpy(dtype=bool)
        q = pred[col].to_numpy(dtype="float64")
        wc = pred["water_cut_pct"].to_numpy(dtype="float64") / 100.0
        gor = pred["gor_scf_stb"].to_numpy(dtype="float64")
        q = np.where(si, 0.0, np.maximum(q, 0.0))
        oil = q * (1 - wc)
        gas = oil * gor / 1000.0
        cols = {"liquid_rate_stbd": q, "oil_rate_stbd": oil, "water_rate_stbd": q * wc, "gas_rate_mscfd": gas,
                "total_gas_rate_mscfd": np.where(si, 0.0, gas + _gas_lift(hf_well)),
                "water_cut_pct": wc * 100.0, "gor_scf_stb": gor}
        return _frame(hf_well, cols, si)


# ----------------------------------------------------------------------------- test-based
class TestBasedModel(AllocationModel):
    family = "test-based"

    def __init__(self, mode: str = "step"):
        self.mode = mode
        self.name = "test_step" if mode == "step" else "test_linear"
        self.causal = mode == "step"
        self.description = ("Last well test carried forward (no sensors except the choke for shut-ins)"
                            if mode == "step" else "Linear interpolation between well tests (hindcast, uses future tests)")

    def fit(self, well_tests, hf_by_well, completion, pvt, max_time=None, context=None):
        self.wt = _causal_tests(well_tests, max_time)
        return self

    def predict(self, hf_well, well):
        si = _shut_in(hf_well)
        sub = self.wt[self.wt["well_name"] == well]
        n = len(hf_well)
        cols = {k: np.full(n, np.nan) for k in ("oil", "water", "gas")}
        if len(sub):
            epoch = pd.Timestamp("1970-01-01", tz="UTC")  # resolution-independent seconds
            sub = sub.assign(_tm=(sub["test_start"] + (sub["test_end"] - sub["test_start"]) / 2)).sort_values("_tm")
            tm = (sub["_tm"] - epoch).dt.total_seconds().to_numpy()
            t = (pd.to_datetime(hf_well["timestamp"], utc=True) - epoch).dt.total_seconds().to_numpy()
            for k, c in (("oil", "oil_rate_stbd"), ("water", "water_rate_stbd"), ("gas", "gas_rate_mscfd")):
                v = sub[c].to_numpy(dtype="float64")
                ok = np.isfinite(v)
                if not ok.any():
                    continue
                tv, vv = tm[ok], v[ok]
                if self.mode == "step":
                    # before the first test the first test is used (hindcast only; never in walk-forward,
                    # where every training test ends before the predicted window)
                    idx = np.clip(np.searchsorted(tv, t, side="right") - 1, 0, len(vv) - 1)
                    cols[k] = vv[idx]
                else:
                    cols[k] = np.interp(t, tv, vv)
        oil = np.where(si, 0.0, cols["oil"])
        wat = np.where(si, 0.0, cols["water"])
        gas = np.where(si, 0.0, cols["gas"])
        liq = oil + wat
        qgl = np.where(si, 0.0, _gas_lift(hf_well))
        with np.errstate(invalid="ignore", divide="ignore"):
            out = {"liquid_rate_stbd": liq, "oil_rate_stbd": oil, "water_rate_stbd": wat, "gas_rate_mscfd": gas,
                   "total_gas_rate_mscfd": gas + qgl, "water_cut_pct": np.where(liq > 0, 100 * wat / liq, np.nan),
                   "gor_scf_stb": np.where(oil > 0, 1000 * gas / oil, np.nan)}
        return _frame(hf_well, out, si)


# ----------------------------------------------------------------------------- features
FEATURES = ("choke_pct", "area_in2", "sqrt_dp_choke", "area_sqrt_dp", "area_sqrt_dp_eff", "whp", "dhp",
            "dp_tubing", "gas_lift", "gl_per_area")


def minute_features(hf: pd.DataFrame, comp_row: Mapping[str, Any], r_crit: float = 0.55) -> np.ndarray:
    """Physics-motivated 1-minute features (n, len(FEATURES)); NaN where sensors are missing."""
    g = lambda c: hf[c].to_numpy(dtype="float64") if c in hf.columns else np.full(len(hf), np.nan)
    choke, whp, ds, dhp, gl = g("choke_opening_pct"), g("whp_psia"), g("whp_downstream_psia"), g("dh_pressure_psia"), g("gas_lift_rate_mscfd")
    d_in = np.asarray(choke_bean_diameter_in(np.nan_to_num(choke), _fnum(comp_row, "choke_max_size_64ths", 128.0)))
    area = np.pi / 4 * d_in ** 2
    dp = np.maximum(whp - ds, 0.0)
    with np.errstate(invalid="ignore"):
        dp_eff = _effective_choke_dp(whp, ds, r_crit=r_crit)
        X = np.column_stack([choke, area, np.sqrt(dp), area * np.sqrt(dp), area * np.sqrt(np.maximum(dp_eff, 0.0)),
                             whp, dhp, dhp - whp, gl, gl / np.maximum(area, 1e-3)])
    X[~np.isfinite(choke)] = np.nan
    return X


def _window_mean_features(hf: pd.DataFrame, comp_row, t0, t1) -> np.ndarray:
    ts = pd.to_datetime(hf["timestamp"], utc=True)
    win = hf[(ts >= t0) & (ts <= t1)]
    if "is_steady_flowing" in win.columns and win["is_steady_flowing"].any():
        win = win[win["is_steady_flowing"]]
    if win.empty:
        return np.full(len(FEATURES), np.nan)
    with np.errstate(all="ignore"):
        return np.nanmean(minute_features(win, comp_row), axis=0)


def _smoothed_minute_features(hf: pd.DataFrame, comp_row, window_min: int = 60) -> np.ndarray:
    """Trailing time-based mean of the 1-minute features (causal), so prediction inputs match the
    steady test-window means the models are trained on (noisy minutes bias nonlinear models)."""
    X = minute_features(hf, comp_row)
    ts = pd.DatetimeIndex(pd.to_datetime(hf["timestamp"], utc=True))
    return pd.DataFrame(X, index=ts).rolling(f"{window_min}min", min_periods=1).mean().to_numpy()


class FeatureMLModel(AllocationModel):
    """Pooled relative-rate regression across wells.

    Target log(q_test / q_ref) and features log(x / x_ref), where the references are the well's own
    training means *excluding the row itself* (leave-one-out), so a well's rows are not pulled toward
    1 and one model learns how rate responds to relative choke/pressure changes. Wells need >= 2
    training tests to contribute rows; prediction uses the well's mean over all its training tests."""

    family = "regression/ML"
    pooled = True

    def __init__(self, regressor: str = "gbm", **reg_kwargs):
        self.regressor = regressor
        self.reg_kwargs = reg_kwargs
        self.name = regressor if regressor != "ridge" else "ridge"
        self.description = {"ridge": "Ridge regression on relative choke/pressure features (pooled)",
                            "gbm": "Gradient-boosted trees on relative features (pooled, numpy)",
                            "rf": "Random forest on relative features (pooled, numpy)",
                            "knn": "k-nearest neighbours on relative features (pooled)",
                            "tree": "Single regression tree on relative features"}.get(regressor, regressor)

    def fit(self, well_tests, hf_by_well, completion, pvt, max_time=None, context=None):
        ctx = context or {}
        wt = _causal_tests(well_tests, max_time)
        comp = completion.set_index("well_name")
        self.comp = comp
        self.split = PhaseSplit(ctx.get("ref_timestamp") or DEFAULT_REF_TIMESTAMP).fit(wt, completion, ctx.get("wc_global_slope"))
        rows, ys, wells = [], [], []
        fcache = ctx.get("feature_cache")  # test windows end <= max_time, so cached means stay causal
        for _, r in wt.iterrows():
            hf = hf_by_well.get(r["well_name"])
            if hf is None or r["well_name"] not in comp.index:
                continue
            key = (r["well_name"], str(r["test_start"]))
            if fcache is not None and key in fcache:
                feats = fcache[key]
            else:
                feats = _window_mean_features(hf, comp.loc[r["well_name"]], r["test_start"], r["test_end"])
                if fcache is not None:
                    fcache[key] = feats
            rows.append(feats)
            ys.append(float(r["liquid_rate_stbd"]))
            wells.append(r["well_name"])
        self.ref_q, self.ref_x = {}, {}
        if not rows:
            self.model = None
            return self
        X, y, wells = np.vstack(rows), np.asarray(ys), np.asarray(wells)
        good = np.isfinite(y) & (y > 0)
        X, y, wells = X[good], y[good], wells[good]
        Xr_rows, yr_rows = [], []
        for w in np.unique(wells):
            m = np.where(wells == w)[0]
            self.ref_q[w] = float(np.mean(y[m]))
            with np.errstate(all="ignore"):
                self.ref_x[w] = np.nanmean(X[m], axis=0)
            if len(m) < 2:
                continue
            for i in m:  # leave-one-out references
                others = m[m != i]
                with np.errstate(all="ignore"):
                    rx = np.nanmean(X[others], axis=0)
                Xr_rows.append(self._rel(X[i], rx))
                yr_rows.append(np.log(y[i] / np.mean(y[others])))
        if len(yr_rows) < 4:
            self.model = None
            return self
        Xr, yr = np.vstack(Xr_rows), np.asarray(yr_rows)
        kw = dict(self.reg_kwargs)
        if self.regressor == "gbm":
            kw = {"n_estimators": 150, "learning_rate": 0.05, "max_depth": 2, "min_samples_leaf": 3, **kw}
        elif self.regressor == "rf":
            kw = {"n_estimators": 150, "max_depth": 5, "min_samples_leaf": 3, **kw}
        elif self.regressor == "ridge":
            kw = {"alpha": 1.0, **kw}
        elif self.regressor == "knn":
            kw = {"k": 5, **kw}
        self.model = make_model(self.regressor, **kw).fit(Xr, yr)
        self.n_train = int(len(y))
        return self

    @staticmethod
    def _rel(X: np.ndarray, ref: np.ndarray) -> np.ndarray:
        """log(x / ref) where both are positive; 0 otherwise (e.g. a well without lift gas)."""
        with np.errstate(all="ignore"):
            ok = (X > 0) & (ref > 0)
            return np.where(ok, np.log(np.where(ok, X, 1.0) / np.where(ok, ref, 1.0)), 0.0)

    def predict(self, hf_well, well):
        si = _shut_in(hf_well)
        n = len(hf_well)
        q = np.full(n, np.nan)
        if self.model is not None and well in self.ref_q and well in self.comp.index:
            X = _smoothed_minute_features(hf_well, self.comp.loc[well])
            raw = minute_features(hf_well, self.comp.loc[well])
            bad = (~np.isfinite(raw[:, FEATURES.index("whp")]) | ~np.isfinite(raw[:, FEATURES.index("dhp")])
                   | ~np.isfinite(raw[:, FEATURES.index("choke_pct")]))
            q = np.exp(self.model.predict(self._rel(X, self.ref_x[well]))) * self.ref_q[well]
            q[bad] = np.nan
        return _frame(hf_well, self.split.apply(well, hf_well["timestamp"], q, _gas_lift(hf_well), si), si)


class HybridPhysicsMLModel(AllocationModel):
    """PhysicsVFM liquid rate x a gradient-boosted correction factor learned on relative features
    (target: test liquid / physics liquid over each causal test window, in-sample)."""

    family = "hybrid"
    name = "hybrid_physics_gbm"
    description = "PhysicsVFM liquid x GBM correction factor (shrunk toward 1)"

    def __init__(self, shrink: float = 0.5):
        self.shrink = shrink
        self.physics = PhysicsModel()

    def fit(self, well_tests, hf_by_well, completion, pvt, max_time=None, context=None):
        ctx = context or {}
        self.physics.fit(well_tests, hf_by_well, completion, pvt, max_time, ctx)
        wt = _causal_tests(well_tests, max_time)
        comp = completion.set_index("well_name")
        self.comp = comp
        rows, ys = [], []
        for _, r in wt.iterrows():
            hf = hf_by_well.get(r["well_name"])
            y_t = float(r["liquid_rate_stbd"])
            if hf is None or r["well_name"] not in comp.index or not np.isfinite(y_t) or y_t <= 0:
                continue
            ts = pd.to_datetime(hf["timestamp"], utc=True)
            win = hf[(ts >= r["test_start"]) & (ts <= r["test_end"])]
            if "is_steady_flowing" in win.columns and win["is_steady_flowing"].any():
                win = win[win["is_steady_flowing"]]  # same minutes as the features
            if win.empty:
                continue
            qp = float(np.nanmean(self.physics.predict(win, r["well_name"])["liquid_rate_stbd"]))
            if not np.isfinite(qp) or qp <= 1:
                continue
            rows.append(np.r_[_window_mean_features(hf, comp.loc[r["well_name"]], r["test_start"], r["test_end"]), qp])
            ys.append(y_t / qp)
        self.model = None
        if len(ys) >= 8:
            # note: the correction is learned on in-sample physics fits, so it is shrunk (`shrink`) and clipped
            X = np.vstack(rows)
            self.model = make_model("gbm", n_estimators=80, learning_rate=0.05, max_depth=2, min_samples_leaf=4).fit(X, np.log(ys))
        return self

    def predict(self, hf_well, well):
        pred = self.physics.predict(hf_well, well).copy()
        if self.model is None or well not in self.comp.index:
            return pred
        si = pred["is_shut_in"].to_numpy(dtype=bool)
        X = np.column_stack([_smoothed_minute_features(hf_well, self.comp.loc[well]), pred["liquid_rate_stbd"].to_numpy()])
        z = self.model.predict(X)
        f = np.exp(self.shrink * np.clip(np.where(np.isfinite(z), z, 0.0), -0.3, 0.3))
        for c in ("liquid_rate_stbd", "oil_rate_stbd", "water_rate_stbd", "gas_rate_mscfd"):
            pred[c] = np.where(si, 0.0, pred[c].to_numpy() * f)
        pred["total_gas_rate_mscfd"] = np.where(si, 0.0, pred["gas_rate_mscfd"].to_numpy() + _gas_lift(hf_well))
        pred["is_sensor_missing"] = (~si) & ~np.isfinite(pred["liquid_rate_stbd"].to_numpy())
        return pred


# ----------------------------------------------------------------------------- deconvolution
class DeconvolutionModel(AllocationModel):
    family = "convolution"
    name = "deconvolution"
    description = ("Pressure-rate deconvolution of the downhole gauge (radial + closed boundary), "
                   "kh/skin from an accepted PTA or the IPR, scale-calibrated to past tests")
    needs_history = True

    def __init__(self, cell_hr: float = 6.0, obs_min: int = 30, smoothing: float = 1e-2, prior_weight: float = 0.2):
        # Under pseudo-steady state a constant rate and the background depletion trend both give a
        # linear pressure decline, so pressure alone fixes rate *changes*, not the level. The level
        # comes from a prior: the last well test (causal, carried forward) in reservoir barrels.
        self.cell_hr, self.obs_min, self.smoothing, self.prior_weight = cell_hr, obs_min, smoothing, prior_weight

    def fit(self, well_tests, hf_by_well, completion, pvt, max_time=None, context=None):
        ctx = context or {}
        self.wt = _causal_tests(well_tests, max_time)
        self.comp, self.pvt = completion.set_index("well_name"), pvt.set_index("well_name")
        ref = ctx.get("ref_timestamp") or DEFAULT_REF_TIMESTAMP
        self.split = PhaseSplit(ref).fit(self.wt, completion, ctx.get("wc_global_slope"))
        ipr = fit_pr_and_pi(self.wt, {}, pvt, completion, max_time=max_time, ref_timestamp=ref,
                            pr_gradient_prior_psi_ft=ctx.get("pr_prior"))
        self.response, self.scale, self.source = {}, {}, {}
        events = ctx.get("pta_events") or {}
        for well in self.wt["well_name"].unique():
            if well not in self.pvt.index or well not in self.comp.index:
                continue
            p = self.pvt.loc[well]
            kh, skin, src = None, 0.0, None
            for ev in sorted(events.get(well, []), key=lambda e: e["shut_in_start"]):
                if ev.get("status") == "RADIAL" and (max_time is None or _utc(ev["shut_in_end"]) <= _utc(max_time)):
                    R = ev["result"]["results"]
                    kh, skin, src = R.get("kh_md_ft"), R.get("apparent_skin") or 0.0, f"PTA {ev['event_id']}"
            if kh is None and well in ipr:
                # PSS productivity index -> kh assuming zero skin (J is per surface STB; x B_liquid)
                J = ipr[well]["J"]
                area = float(p.get("estimated_drainage_area_acres", 160.0) or 160.0)
                re = math.sqrt(area * 43560 / math.pi)
                wcs = self.wt[self.wt.well_name == well]["water_cut_pct"] / 100
                bw = _fnum(self.comp.loc[well], "water_fvf_rb_stb", 1.02)
                b = float(np.nanmean((1 - wcs) * float(p["oil_fvf_rb_stb"]) + wcs * bw))
                kh = J * 141.2 * b * float(p["oil_viscosity_cp"]) * (math.log(re / float(p["wellbore_radius_ft"])) - 0.75)
                skin, src = 0.0, "IPR productivity index (skin 0 assumed)"
            if kh is None or not np.isfinite(kh) or kh <= 0:
                continue
            area = p.get("estimated_drainage_area_acres")
            self.response[well] = radial_unit_response(
                float(kh), float(p["net_pay_thickness_ft"]), float(p["oil_viscosity_cp"]), float(p["porosity_frac"]),
                float(p["total_compressibility_1_psi"]), float(p["wellbore_radius_ft"]), float(skin),
                float(area) if area is not None and np.isfinite(area) else None)
            self.source[well] = src
            # scale calibration on past tests (needs telemetry up to max_time)
            hf = hf_by_well.get(well)
            self.scale[well] = 1.0
            if hf is not None and len(self.wt[self.wt.well_name == well]):
                if max_time is not None:
                    hf = hf[pd.to_datetime(hf["timestamp"], utc=True) <= _utc(max_time)]
                q = self._raw_liquid(hf, well)
                ts = pd.to_datetime(hf["timestamp"], utc=True)
                ratios = []
                for _, r in self.wt[self.wt.well_name == well].iterrows():
                    m = ((ts >= r["test_start"]) & (ts <= r["test_end"])).to_numpy()
                    qv = np.nanmean(q[m]) if m.any() else np.nan
                    ratio = float(r["liquid_rate_stbd"]) / qv if np.isfinite(qv) and qv > 1 else np.nan
                    if np.isfinite(ratio) and ratio > 0:
                        ratios.append(ratio)
                if ratios:
                    self.scale[well] = float(np.median(ratios))
        return self

    def _raw_liquid(self, hf: pd.DataFrame, well: str) -> np.ndarray:
        """Deconvolved surface liquid rate at every minute of `hf` (before scale calibration)."""
        n = len(hf)
        if well not in self.response or n < 60 or "dh_pressure_psia" not in hf.columns:
            return np.full(n, np.nan)
        order = np.argsort(pd.to_datetime(hf["timestamp"], utc=True).to_numpy(), kind="stable")
        if not np.all(order == np.arange(n)):
            out = np.empty(n)
            out[order] = self._raw_liquid(hf.iloc[order].reset_index(drop=True), well)
            return out
        ts = pd.to_datetime(hf["timestamp"], utc=True)
        t0 = ts.iloc[0]
        th = (ts - t0).dt.total_seconds().to_numpy() / 3600.0
        si = _shut_in(hf)
        edges = np.arange(0.0, th[-1] + self.cell_hr, self.cell_hr)
        if len(edges) < 3:
            return np.full(n, np.nan)
        cell = np.clip(np.searchsorted(edges, th, side="right") - 1, 0, len(edges) - 2)
        si_frac = np.bincount(cell, weights=si.astype(float), minlength=len(edges) - 1) / np.maximum(np.bincount(cell, minlength=len(edges) - 1), 1)
        obs_bin = np.floor(th * 60 / self.obs_min).astype(int)
        dfp = pd.DataFrame({"b": obs_bin, "t": th, "p": hf["dh_pressure_psia"].to_numpy(dtype="float64")}).dropna()
        g = dfp.groupby("b").mean()
        if len(g) < 10:
            return np.full(n, np.nan)
        p = self.pvt.loc[well]
        wc = self.split.apply(well, hf["timestamp"], np.ones(n), np.zeros(n), np.zeros(n, bool))["water_cut_pct"] / 100.0
        b_liq = (1 - wc) * float(p["oil_fvf_rb_stb"]) + wc * _fnum(self.comp.loc[well], "water_fvf_rb_stb", 1.02)
        # rate prior per cell: the last test *completed* before the cell start, in RB/D
        # (cells before the first test use the first test: hindcast only)
        sub = self.wt[self.wt["well_name"] == well].sort_values("test_end")
        sub = sub[np.isfinite(sub["liquid_rate_stbd"].to_numpy(dtype="float64"))]
        prior = None
        if len(sub) and self.prior_weight > 0:
            tm = (sub["test_end"] - t0).dt.total_seconds().to_numpy() / 3600.0
            ql = sub["liquid_rate_stbd"].to_numpy(dtype="float64")
            idx = np.clip(np.searchsorted(tm, edges[:-1], side="right") - 1, 0, len(ql) - 1)
            b_cell = np.bincount(cell, weights=b_liq, minlength=len(edges) - 1) / np.maximum(np.bincount(cell, minlength=len(edges) - 1), 1)
            prior = np.where(si_frac > 0.5, 0.0, ql[idx] * np.where(b_cell > 0, b_cell, float(p["oil_fvf_rb_stb"])))
        res = deconvolve_rates(g["t"].to_numpy(), g["p"].to_numpy(), edges, self.response[well],
                               smoothing=self.smoothing, fixed_zero=si_frac > 0.5, background_trend=True,
                               q_prior=prior, prior_weight=self.prior_weight if prior is not None else 0.0)
        q_rb = res["q"][cell]
        return np.where(si, 0.0, q_rb / b_liq)

    def predict(self, hf_well, well):
        si = _shut_in(hf_well)
        q = self._raw_liquid(hf_well, well) * self.scale.get(well, 1.0)
        return _frame(hf_well, self.split.apply(well, hf_well["timestamp"], q, _gas_lift(hf_well), si), si)


# ----------------------------------------------------------------------------- synthetic / stacked external
class PrecomputedSyntheticModel(AllocationModel):
    """Loads precomputed daily predictions from our from-scratch Synthetic-Augmented ML pipelines
    in `vfm_export_tool/` and maps them onto 1-minute or daily frames cleanly."""

    family = "synthetic/ML"
    pooled = True

    def __init__(self, name: str, csv_filename: str, description: str):
        self.name = name
        self.csv_filename = csv_filename
        self.description = description
        self._daily_df: Optional[pd.DataFrame] = None

    def csv_path(self) -> Any:
        from pathlib import Path
        root = Path(__file__).resolve().parents[4]  # .../emil
        return root / "vfm_export_tool" / self.csv_filename

    def load_daily_table(self) -> pd.DataFrame:
        if self._daily_df is None:
            p = self.csv_path()
            if not p.exists():
                raise FileNotFoundError(f"Precomputed VFM CSV not found: {p}")
            df = pd.read_csv(p)
            dcol = "date" if "date" in df.columns else "timestamp"
            df["date"] = pd.to_datetime(df[dcol], utc=True).dt.floor("D")
            self._daily_df = df
        return self._daily_df

    def fit(self, well_tests, hf_by_well, completion, pvt, max_time=None, context=None):
        self.load_daily_table()
        return self

    def predict(self, hf_well, well):
        df = self.load_daily_table()
        sub = df[df["well_name"] == well].set_index("date")
        ts = pd.to_datetime(hf_well["timestamp"], utc=True)
        days = ts.dt.floor("D")
        si = _shut_in(hf_well)
        oil = days.map(sub["oil_rate_stbd"]).to_numpy(dtype="float64") if "oil_rate_stbd" in sub else np.full(len(hf_well), np.nan)
        wat = days.map(sub["water_rate_stbd"]).to_numpy(dtype="float64") if "water_rate_stbd" in sub else np.full(len(hf_well), np.nan)
        gas = days.map(sub["gas_rate_mscfd"]).to_numpy(dtype="float64") if "gas_rate_mscfd" in sub else np.full(len(hf_well), np.nan)
        oil = np.where(si, 0.0, np.maximum(oil, 0.0))
        wat = np.where(si, 0.0, np.maximum(wat, 0.0))
        gas = np.where(si, 0.0, np.maximum(gas, 0.0))
        liq = oil + wat
        qgl = np.where(si, 0.0, _gas_lift(hf_well))
        with np.errstate(invalid="ignore", divide="ignore"):
            wc = np.where(liq > 0, 100.0 * wat / liq, np.nan)
            gor = np.where(oil > 0, 1000.0 * gas / oil, np.nan)
        cols = {"liquid_rate_stbd": liq, "oil_rate_stbd": oil, "water_rate_stbd": wat, "gas_rate_mscfd": gas,
                "total_gas_rate_mscfd": np.where(si, 0.0, gas + qgl), "water_cut_pct": wc, "gor_scf_stb": gor}
        return _frame(hf_well, cols, si)


# ----------------------------------------------------------------------------- registry
REGISTRY: Dict[str, Callable[[], AllocationModel]] = {
    "physics_wls": lambda: PhysicsModel("physics_wls"),
    "synth_stacked_v2": lambda: PrecomputedSyntheticModel(
        "synth_stacked_v2",
        "vfm_phase_decomposed_stacked_v2.csv",
        "Phase-Decomposed Stacked Grey-Box VFM V2 (7 sensors, completion hydraulics, 1,800 few-shot synthetic rows, SVR/MLP/GBDT)",
    ),
    "synth_gbm_ensemble": lambda: PrecomputedSyntheticModel(
        "synth_gbm_ensemble",
        "vfm_synthetic_gbm_ensemble.csv",
        "From-Scratch Synthetic-Augmented Monotonic GBM Ensemble (LightGBM + HistGBM + Huber, 1,500 few-shot synthetic rows)",
    ),
    "physics_ipr": lambda: PhysicsModel("physics_ipr", ("ipr",)),
    "physics_choke": lambda: PhysicsModel("physics_choke", ("choke",)),
    "physics_vlp": lambda: PhysicsModel("physics_vlp", ("vlp",)),
    "hybrid_physics_gbm": HybridPhysicsMLModel,
    "deconvolution": DeconvolutionModel,
    "ridge": lambda: FeatureMLModel("ridge"),
    "gbm": lambda: FeatureMLModel("gbm"),
    "rf": lambda: FeatureMLModel("rf"),
    "knn": lambda: FeatureMLModel("knn"),
    "test_step": lambda: TestBasedModel("step"),
    "test_linear": lambda: TestBasedModel("linear"),
}


def make_allocation_model(name: str) -> AllocationModel:
    if name not in REGISTRY:
        raise ValueError(f"unknown allocation model {name!r}; valid: {sorted(REGISTRY)}")
    return REGISTRY[name]()


def list_allocation_models() -> List[Dict[str, Any]]:
    return [make_allocation_model(n).info() for n in REGISTRY]
