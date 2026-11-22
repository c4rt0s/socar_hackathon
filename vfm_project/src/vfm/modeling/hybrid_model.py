"""Hybrid Physics-Baseline + ML Residual Virtual Flow Meter."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

from vfm.modeling.targets import decompose_well_test_targets, reconstruct_phase_rates


DEFAULT_MODEL_FEATURES: Sequence[str] = (
    "whp_psia",
    "whp_downstream_psia",
    "dh_pressure_psia",
    "dh_temperature_degf",
    "annulus_pressure_psia",
    "gas_lift_rate_mscfd",
    "choke_opening_pct",
    "dp_choke_psi",
    "choke_pressure_ratio",
    "choke_flow_proxy",
    "choke_area_x_whp",
    "dp_tubing_psi",
    "tubing_grad_psi_ft",
    "dh_minus_pb_psi",
    "dp_gl_annulus_whp_psi",
    "gl_to_choke_proxy",
    "dh_temp_minus_res_degf",
    "well_age_years",
    "tubing_id_sq_in2",
    "net_pay_thickness_ft",
    "porosity_frac",
    "oil_fvf_rb_stb",
    "oil_viscosity_cp",
    "solution_gor_at_pb_scf_stb",
    "gl_valve_port_64ths",
    "cum_oil_at_2025_01_01_stb",
)


class _NumpyRidge:
    """Zero-dependency standardized Ridge regressor fallback."""

    def __init__(self, alpha: float = 5.0):
        self.alpha = float(alpha)
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_NumpyRidge":
        self.mean_ = np.nanmean(X, axis=0)
        self.std_ = np.where(np.nanstd(X, axis=0) > 1e-8, np.nanstd(X, axis=0), 1.0)
        Xs = np.nan_to_num((X - self.mean_) / self.std_, nan=0.0)
        self.intercept_ = float(np.mean(y))
        yc = y - self.intercept_
        n_feat = Xs.shape[1]
        reg = self.alpha * np.eye(n_feat)
        self.coef_ = np.linalg.solve(Xs.T @ Xs + reg, Xs.T @ yc)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xs = np.nan_to_num((X - self.mean_) / self.std_, nan=0.0)
        return self.intercept_ + Xs @ self.coef_


def _make_regressor(backend: str, random_state: int = 42) -> Any:
    if backend == "lightgbm":
        import lightgbm as lgb

        return lgb.LGBMRegressor(
            n_estimators=180,
            learning_rate=0.04,
            num_leaves=15,
            min_child_samples=4,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=2.0,
            random_state=random_state,
            verbose=-1,
        )
    if backend == "sklearn_hgb":
        from sklearn.ensemble import HistGradientBoostingRegressor

        return HistGradientBoostingRegressor(
            max_iter=200,
            learning_rate=0.05,
            max_depth=4,
            min_samples_leaf=4,
            l2_regularization=2.0,
            random_state=random_state,
        )
    return _NumpyRidge(alpha=5.0)


def detect_best_backend() -> str:
    """Return best available ML backend ('lightgbm', 'sklearn_hgb', or 'numpy_ridge')."""
    for pkg, name in (("lightgbm", "lightgbm"), ("sklearn", "sklearn_hgb")):
        try:
            __import__(pkg)
            return name
        except ImportError:
            continue
    return "numpy_ridge"


@dataclass
class HybridVFM:
    """
    Two-tier Virtual Flow Meter:
      - Predicts decomposed physical quantities (q_liq, water_cut, gor_form).
      - Tier-1 linear physics baseline captures choke/tubing scaling trends.
      - Tier-2 GBDT (if installed) models nonlinear residual corrections.
    """

    feature_cols: List[str] = field(default_factory=lambda: list(DEFAULT_MODEL_FEATURES))
    backend: str = "auto"
    random_state: int = 42

    def __post_init__(self) -> None:
        if self.backend == "auto":
            self.backend = detect_best_backend()
        self._base_liq = _NumpyRidge(alpha=10.0)
        self._res_liq = None
        self._wc_model = None
        self._gor_model = None

    def _prepare_X(self, df: pd.DataFrame) -> np.ndarray:
        return df[self.feature_cols].to_numpy(dtype="float64")

    def fit(self, train_df: pd.DataFrame) -> "HybridVFM":
        X = self._prepare_X(train_df)
        targets = decompose_well_test_targets(train_df)
        y_liq = targets["q_liq_stbd"].to_numpy()
        y_wc = targets["water_cut_frac"].to_numpy()
        y_gor = np.log1p(targets["gor_form_scf_stb"].to_numpy())

        self._base_liq.fit(X, y_liq)
        if self.backend != "numpy_ridge":
            base_pred = self._base_liq.predict(X)
            self._res_liq = _make_regressor(self.backend, self.random_state).fit(X, y_liq - base_pred)
        self._wc_model = _make_regressor(self.backend, self.random_state).fit(X, y_wc)
        self._gor_model = _make_regressor(self.backend, self.random_state).fit(X, y_gor)
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        X = self._prepare_X(df)
        q_liq = self._base_liq.predict(X)
        if self._res_liq is not None:
            q_liq = q_liq + self._res_liq.predict(X)
        wc = np.clip(self._wc_model.predict(X), 0.0, 0.99)
        gor = np.expm1(np.clip(self._gor_model.predict(X), 0.0, 12.0))

        gl = df["gas_lift_rate_mscfd"].fillna(0.0).to_numpy(dtype="float64")
        shut_in = df["is_shut_in"].to_numpy(dtype=bool) if "is_shut_in" in df.columns else None

        rates: Dict[str, np.ndarray] = reconstruct_phase_rates(
            q_liq_stbd=q_liq,
            water_cut_frac=wc,
            gor_form_scf_stb=gor,
            gas_lift_rate_mscfd=gl,
            is_shut_in=shut_in,
        )
        return pd.DataFrame(rates, index=df.index)
