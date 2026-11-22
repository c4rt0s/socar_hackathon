"""Engineering workbench: baseline VFM, PTA events, judgement-driven coupling, JSON payloads.

The dashboard server (dashboard/server.py) is a thin HTTP layer over this class; it can also
be used from a notebook or script:

    wb = Workbench(load_config()); wb.prepare()
    wb.update_settings({...}); wb.well_payload("GL-08")

Everything is dataset-agnostic: wells come from the well-test table, shut-ins from the choke
data, properties from the PVT/completion tables, assumptions from the config.
"""

from __future__ import annotations

import copy
from dataclasses import asdict
import datetime as _dt
import hashlib
import io
import json
import math
from pathlib import Path
import pickle
import threading
import time
from typing import Any, Callable, Dict, List, Mapping, Optional  # noqa: F401

import numpy as np
import pandas as pd

from vfm.allocation.daily import (
    aggregate_well_daily_physics,
    anchor_daily_exact_window,
    calendar_date_grid,
    compute_exact_window_vfm_refs,
)
from vfm.config import VFMConfig, load_config
from vfm.coupling import CouplingSettings, EventDecision, PropertyChange, build_ipr_adjustments, default_decision
from vfm.io import load_frame, load_static_tables
from vfm.modeling.physics_vfm import PhysicsVFM, physics_vfm_from_config
from vfm.physics.reservoir import compute_wellbore_fluid_gradient, gauge_to_datum_pressure
from vfm.pta.runner import RATE_BASES, PtaRunSettings, detect_pta_events, run_pta_event

HF_COLUMNS = (
    "timestamp", "well_name", "whp_psia", "whp_downstream_psia", "dh_pressure_psia", "dh_temperature_degf",
    "annulus_pressure_psia", "gas_lift_rate_mscfd", "choke_opening_pct", "is_shut_in", "is_transient", "is_steady_flowing",
)
CACHE_VERSION = 1


def to_jsonable(obj: Any) -> Any:
    """NaN/inf -> None, numpy/pandas scalars and timestamps -> plain JSON types."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [to_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, (pd.Timestamp, _dt.datetime, _dt.date)):
        return None if pd.isna(obj) else pd.Timestamp(obj).isoformat()
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if callable(obj):
        return None
    return obj


def _series(df: pd.DataFrame, col: str) -> List[Optional[float]]:
    if col not in df.columns:
        return [None] * len(df)
    return [None if not np.isfinite(v) else round(float(v), 4) for v in df[col].to_numpy(dtype="float64")]


class Workbench:
    def __init__(self, cfg: Optional[VFMConfig] = None, progress: Callable[[str], None] = print):
        self.cfg = cfg or load_config()
        self.progress = progress
        self.lock = threading.RLock()
        self.status = "created"
        cache_dir = self.cfg.resolve_path("cache_dir")
        self.cache_dir = cache_dir
        self.comp, self.pvt, self.chokes = load_static_tables(
            self.cfg.resolve_path("well_completion_csv"), self.cfg.resolve_path("reservoir_pvt_csv"),
            self.cfg.resolve_path("choke_events_csv"), schema=self.cfg.schema)
        try:
            wt = load_frame(cache_dir / "well_tests_harmonized")
        except FileNotFoundError as exc:
            raise FileNotFoundError("No prepared cache. Run scripts/01_audit_and_prepare.py for this dataset first.") from exc
        wt["test_start"] = pd.to_datetime(wt["test_start"], utc=True)
        wt["test_end"] = pd.to_datetime(wt["test_end"], utc=True)
        self.tests = wt.sort_values(["well_name", "test_start"]).reset_index(drop=True)
        self.wells: List[str] = sorted(self.tests["well_name"].unique())
        self._hf: Dict[str, pd.DataFrame] = {}
        self.judgement_path = self.cfg.resolve_path("judgement_file")
        self.coupling_cfg = self.cfg.section("coupling")
        self.settings = CouplingSettings(enabled=bool(self.coupling_cfg.get("enabled", False)),
                                         min_j_multiplier=float(self.coupling_cfg.get("min_j_multiplier", 0.25)),
                                         max_j_multiplier=float(self.coupling_cfg.get("max_j_multiplier", 4.0)))
        self.pta_overrides: Dict[str, Dict[str, Any]] = {}
        rb_cfg = str(self.cfg.section("pta").get("rate_basis", "oil"))
        self.pta_rate_basis: str = rb_cfg if rb_cfg in RATE_BASES else "oil"
        self.events: Dict[str, List[Dict[str, Any]]] = {}
        self.notes: List[Dict[str, Any]] = []
        self.coupled_models: Dict[str, PhysicsVFM] = {}
        self.coupled_daily: Optional[pd.DataFrame] = None
        # allocation model zoo (vfm.allocation.models): daily results per model, computed on request
        self.active_model = "physics_wls"
        self.model_daily: Dict[str, pd.DataFrame] = {}
        self.model_status: Dict[str, str] = {}
        self._model_locks: Dict[str, threading.Lock] = {}
        self.manual_pta_fits: Dict[str, Dict[str, Any]] = {}
        self._advanced_pta_cache: Dict[str, Dict[str, Any]] = {}
        self.bench_status = "idle"
        self._sensor_daily: Optional[pd.DataFrame] = None
        self._sensor_hourly: Optional[pd.DataFrame] = None
        self._monitor_cache: Dict[str, pd.DataFrame] = {}
        self._qo_sens_cache: Dict[str, Dict[str, Any]] = {}
        self._vfm_diag_cache: Dict[str, Dict[str, Any]] = {}
        self._custom_vfm_meta: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------ data access ----
    def hf(self, well: str) -> pd.DataFrame:
        if well not in self._hf:
            df = load_frame(self.cache_dir / "wells" / well)
            self._hf[well] = df[[c for c in HF_COLUMNS if c in df.columns]].copy()
        return self._hf[well]

    def _tests(self, well: str) -> pd.DataFrame:
        return self.tests[self.tests["well_name"] == well]

    def _row(self, table: pd.DataFrame, well: str) -> pd.Series:
        return table.set_index("well_name").loc[well]

    # ------------------------------------------------------------------ baseline -------
    def _cache_key(self) -> str:
        h = hashlib.sha256()
        h.update(json.dumps({k: v for k, v in self.cfg.raw.items() if k not in ("dashboard", "coupling", "pta")},
                            sort_keys=True, default=str).encode())
        for p in sorted((self.cache_dir / "wells").glob("*.pkl")) + [self.cache_dir / "well_tests_harmonized.pkl"]:
            if p.exists():
                h.update(f"{p.name}:{p.stat().st_mtime_ns}:{p.stat().st_size}".encode())
        h.update(str(CACHE_VERSION).encode())
        return h.hexdigest()[:16]

    def build_baseline(self, use_cache: bool = True) -> None:
        cache_file = self.cache_dir / "workbench_baseline.pkl"
        key = self._cache_key()
        if use_cache and cache_file.exists():
            try:
                with cache_file.open("rb") as f:
                    blob = pickle.load(f)
                if blob.get("key") == key:
                    self.base_model, self.base_daily = blob["model"], blob["daily"]
                    self.date_grid, self.field_prior = blob["date_grid"], blob["field_prior"]
                    self.progress("Baseline loaded from cache.")
                    return
            except Exception as exc:  # stale/corrupt cache: rebuild
                self.progress(f"Ignoring baseline cache ({exc}).")
        t0 = time.perf_counter()
        self.progress("Fitting baseline PhysicsVFM (uncoupled) ...")
        hf_all = {w: self.hf(w) for w in self.wells}
        self.base_model = physics_vfm_from_config(self.cfg).fit(self.tests, hf_all, self.comp, self.pvt)
        grads = [b["pr_gradient_psi_ft"] for p in self.base_model.ipr_params_.values() for b in p["buildup_anchors"]]
        cfg_prior = self.cfg.assumptions.get("pr_gradient_prior_psi_ft")
        self.field_prior = float(cfg_prior) if cfg_prior is not None else (float(np.median(grads)) if grads else None)
        self.date_grid = calendar_date_grid(hf_all)
        self.base_daily = self._daily_for(self.base_model, self.wells)
        self.progress(f"Baseline ready in {time.perf_counter() - t0:.0f}s.")
        try:
            with cache_file.open("wb") as f:
                pickle.dump({"key": key, "model": self.base_model, "daily": self.base_daily,
                             "date_grid": self.date_grid, "field_prior": self.field_prior}, f)
        except Exception as exc:
            self.progress(f"Could not write baseline cache ({exc}).")

    def _daily_for(self, model: PhysicsVFM, wells: List[str]) -> pd.DataFrame:
        parts, refs = [], []
        for w in wells:
            hf_w = self.hf(w)
            preds = model.predict(hf_w)
            parts.append(aggregate_well_daily_physics(w, hf_w, preds, self.date_grid))
            refs.append(compute_exact_window_vfm_refs(w, hf_w, preds, self._tests(w)))
        daily = pd.concat(parts, ignore_index=True)
        ref_df = pd.concat(refs, ignore_index=True) if refs else pd.DataFrame()
        return anchor_daily_exact_window(daily, ref_df, origin=self.date_grid[0]).sort_values(["well_name", "date"]).reset_index(drop=True)

    # ------------------------------------------------------------------ PTA ------------
    def pta_settings_for(self, eid: str) -> PtaRunSettings:
        st = PtaRunSettings.from_config(self.cfg.section("pta"))
        if self.pta_rate_basis in RATE_BASES:
            st.rate_basis = self.pta_rate_basis
        for k, v in (self.pta_overrides.get(eid) or {}).items():
            if hasattr(st, k):
                setattr(st, k, tuple(v) if k == "manual_radial_dt_hr" and v else v)
        return st

    def run_pta(self, wells: Optional[List[str]] = None, only_event: Optional[str] = None) -> None:
        pta_cfg = self.cfg.section("pta")
        for w in wells or self.wells:
            hf_w = self.hf(w)
            evs = [] if only_event is None else [e for e in self.events.get(w, []) if e["event_id"] != only_event]
            for s0, s1 in detect_pta_events(hf_w, float(pta_cfg.get("min_shut_in_hr", 2.0)),
                                            float(self.cfg.states.get("choke_closed_pct_threshold", 0.5))):
                from vfm.pta.runner import event_id as _eid
                eid = _eid(w, s0)
                if only_event is not None and eid != only_event:
                    continue
                ev = run_pta_event(w, s0, s1, hf_w, self._tests(w), self._row(self.pvt, w), self._row(self.comp, w),
                                   self.pta_settings_for(eid), self.base_daily[self.base_daily["well_name"] == w])
                ev["internal_pbu"] = self._internal_pbu(w, s0)
                evs.append(ev)
            self.events[w] = sorted(evs, key=lambda e: e["shut_in_start"])
            for ev in self.events[w]:
                self.settings.decisions.setdefault(ev["event_id"], default_decision(ev, self.coupling_cfg))

    def _internal_pbu(self, well: str, start: pd.Timestamp) -> Optional[Dict[str, Any]]:
        p = self.base_model.ipr_params_.get(well) or {}
        for b in p.get("buildup_anchors", []):
            if abs((pd.Timestamp(b["shut_in_start"]) - pd.Timestamp(start)).total_seconds()) < 3600:
                return {k: b.get(k) for k in ("quality_flag", "pr_datum_psia", "pr_gauge_psia", "p_star_gauge_psia",
                                              "kh_md_ft", "skin", "k_md")}
        return None

    # ------------------------------------------------------------------ judgement ------
    def load_judgement(self) -> None:
        if not self.judgement_path.exists():
            return
        d = json.loads(self.judgement_path.read_text(encoding="utf-8"))
        self.settings = CouplingSettings.from_dict(d.get("coupling") or {})
        self.pta_overrides = d.get("pta_settings") or {}
        rb = d.get("pta_rate_basis")
        if rb in RATE_BASES:
            self.pta_rate_basis = rb

    def save_judgement(self) -> Path:
        self.judgement_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"saved_utc": pd.Timestamp.now(tz="UTC").isoformat(), "config": str(self.cfg.source_path),
                   "coupling": self.settings.to_dict(), "pta_settings": self.pta_overrides,
                   "pta_rate_basis": self.pta_rate_basis}
        self.judgement_path.write_text(json.dumps(to_jsonable(payload), indent=2), encoding="utf-8")
        return self.judgement_path

    def _flowing_datum_pressure(self) -> Dict[str, pd.Series]:
        out = {}
        for w in self.wells:
            t = self._tests(w)
            pv, cp = self._row(self.pvt, w), self._row(self.comp, w)
            grad = compute_wellbore_fluid_gradient(
                api=t["oil_gravity_api"].to_numpy(float), wc_frac=t["water_cut_pct"].to_numpy(float) / 100.0,
                gor_scf_stb=float(cp["solution_gor_at_pb_scf_stb"]), gas_sg=t["gas_sg"].to_numpy(float),
                bo=float(pv["oil_fvf_rb_stb"]), bw=float(cp["water_fvf_rb_stb"]),
                salinity_ppm=t["water_salinity_ppm"].to_numpy(float))
            pd_ = gauge_to_datum_pressure(t["dh_pressure_psia"].to_numpy(float), float(cp["gauge_tvd_ft"]),
                                          float(pv["datum_tvd_ft"]), grad)
            out[w] = pd.Series(np.asarray(pd_, float), index=pd.DatetimeIndex(t["test_start"]))
        return out

    # ------------------------------------------------------------------ coupled run ----
    def apply(self) -> None:
        """Recompute the coupled allocation from the current judgement (affected wells only)."""
        adjustments, notes = build_ipr_adjustments(
            self.events, self.settings, self.pvt, self.tests, self.base_model.ref_timestamp,
            self._flowing_datum_pressure())
        affected = set(adjustments)
        for w, toggles in self.settings.estimators.items():
            if toggles and not all(toggles.get(k, True) for k in ("ipr", "choke", "vlp")):
                affected.add(w)
        self.coupled_models = {}
        daily = self.base_daily.copy()
        for w in sorted(affected & set(self.wells)):
            model = physics_vfm_from_config(self.cfg, ipr_adjustments={w: adjustments[w]} if w in adjustments else None,
                                            pr_gradient_prior_psi_ft=self.field_prior,
                                            wc_global_slope=getattr(self.base_model, "wc_global_slope_", None))
            model.fit(self._tests(w), {w: self.hf(w)}, self.comp[self.comp["well_name"] == w],
                      self.pvt[self.pvt["well_name"] == w])
            notes.extend(self._apply_estimator_toggles(model, w))
            self.coupled_models[w] = model
            daily_w = self._daily_for(model, [w])
            daily = pd.concat([daily[daily["well_name"] != w], daily_w], ignore_index=True)
        self.coupled_daily = daily.sort_values(["well_name", "date"]).reset_index(drop=True)
        self.adjustments = adjustments
        self.notes = notes
        self._monitor_cache.clear()

    def _apply_estimator_toggles(self, model: PhysicsVFM, well: str) -> List[Dict[str, Any]]:
        tog = self.settings.estimators.get(well) or {}
        if not tog:
            return []
        wm = model.well_models_[well]
        keys = {"ipr": "w_ipr", "choke": "w_choke", "vlp": "w_vlp"}
        remaining = [k for k in keys if tog.get(k, True) and wm[keys[k]] > 0]
        if not remaining:
            return [{"well_name": well, "level": "BLOCK", "message": "All estimators disabled; toggles ignored."}]
        for k, wk in keys.items():
            if not tog.get(k, True):
                wm[wk] = 0.0
        return [{"well_name": well, "level": "INFO", "message": f"Estimators used: {', '.join(remaining)}."}]

    def _load_precomputed_synthetic_models(self) -> None:
        """Load our from-scratch synthetic-augmented VFM models from vfm_export_tool/ into model_daily."""
        from vfm.allocation.models import PrecomputedSyntheticModel, make_allocation_model
        for mname in ("synth_stacked_v2", "synth_gbm_ensemble"):
            try:
                mdl = make_allocation_model(mname)
                if not isinstance(mdl, PrecomputedSyntheticModel) or not mdl.csv_path().exists():
                    continue
                ext = mdl.load_daily_table().copy()
                # Align cleanly onto base_daily (well_name, date) grid
                base = self.base_daily[["well_name", "date", "uptime_frac", "flowing_minutes", "shut_in_minutes", "daily_gas_lift_mscfd"]].copy()
                base["date_key"] = pd.to_datetime(base["date"], utc=True).dt.floor("D")
                ext["date_key"] = pd.to_datetime(ext["date"], utc=True).dt.floor("D")
                m = base.merge(
                    ext[["well_name", "date_key", "oil_rate_stbd", "water_rate_stbd", "gas_rate_mscfd"]],
                    on=["well_name", "date_key"], how="left"
                )
                for col in ("oil_rate_stbd", "water_rate_stbd", "gas_rate_mscfd"):
                    m[col] = m.groupby("well_name")[col].transform(lambda s: s.interpolate().bfill().ffill()).fillna(0.0)
                m["liquid_rate_stbd"] = m["oil_rate_stbd"] + m["water_rate_stbd"]
                out = pd.DataFrame({
                    "well_name": m["well_name"],
                    "date": m["date"],
                    "uptime_frac": m["uptime_frac"],
                    "flowing_minutes": m["flowing_minutes"],
                    "shut_in_minutes": m["shut_in_minutes"],
                    "daily_gas_lift_mscfd": m["daily_gas_lift_mscfd"],
                    "oil_rate_vfm_stbd": m["oil_rate_stbd"],
                    "water_rate_vfm_stbd": m["water_rate_stbd"],
                    "gas_rate_vfm_mscfd": m["gas_rate_mscfd"],
                    "liquid_rate_vfm_stbd": m["liquid_rate_stbd"],
                    "oil_rate_alloc_stbd": m["oil_rate_stbd"],
                    "water_rate_alloc_stbd": m["water_rate_stbd"],
                    "gas_rate_alloc_mscfd": m["gas_rate_mscfd"],
                    "liquid_rate_alloc_stbd": m["liquid_rate_stbd"],
                    "total_gas_vfm_mscfd": m["gas_rate_mscfd"] + m["daily_gas_lift_mscfd"].fillna(0.0),
                    "total_gas_rate_alloc_mscfd": m["gas_rate_mscfd"] + m["daily_gas_lift_mscfd"].fillna(0.0),
                })
                self.model_daily[mname] = out.sort_values(["well_name", "date"]).reset_index(drop=True)
                self.model_status[mname] = "ready"
            except Exception as exc:
                self.progress(f"Could not preload {mname} ({exc}).")

    def _load_hybrid_pta_cache(self) -> Dict[str, Dict[str, Any]]:
        """Load our 3-logic Hybrid Automated PTA results & Confidence Score Card from vfm_export_tool/."""
        if hasattr(self, "_hybrid_pta_by_well") and self._hybrid_pta_by_well is not None:
            return self._hybrid_pta_by_well
        export_dir = Path(__file__).resolve().parents[3] / "vfm_export_tool"
        sc_path = export_dir / "pta_hybrid_iarf_confidence_scorecard.csv"
        p1_path = export_dir / "pta_hybrid_1_horner_only.csv"
        p2_path = export_dir / "pta_hybrid_2_bourdet_only.csv"
        p3_path = export_dir / "pta_hybrid_3_intersection_ensemble.csv"
        by_well: Dict[str, Dict[str, Any]] = {}
        if sc_path.exists() and p1_path.exists() and p2_path.exists() and p3_path.exists():
            sc = pd.read_csv(sc_path).set_index("well_name")
            d1 = pd.read_csv(p1_path).set_index("well_name")
            d2 = pd.read_csv(p2_path).set_index("well_name")
            d3 = pd.read_csv(p3_path).set_index("well_name")
            for w in sc.index:
                s = sc.loc[w]
                r1, r2, r3 = d1.loc[w], d2.loc[w], d3.loc[w]
                # Parse interval strings "[a, b]"
                def _parse_win(txt: str) -> List[float]:
                    parts = str(txt).strip("[] ").split(",")
                    return [float(parts[0]), float(parts[1])] if len(parts) == 2 else [0.0, 0.0]
                h_ft = float(self._row(self.pvt, w).get("net_pay_thickness_ft", 1.0)) if w in set(self.pvt["well_name"]) else 1.0
                by_well[str(w)] = {
                    "well_name": str(w),
                    "t_shut": str(s["t_shut"]),
                    "qo_stbd": float(s["qo_stbd"]),
                    "pwf_3h_med": float(s["pwf_3h_med"]),
                    "horner_win_hr": _parse_win(s["horner_win_hr"]),
                    "bourdet_win_hr": _parse_win(s["bourdet_win_hr"]),
                    "intersect_win_hr": _parse_win(s["intersect_win_hr"]),
                    "intersect_span_cyc": float(s["intersect_span_cyc"]),
                    "log_log_slope": float(s["log_log_slope"]),
                    "deriv_cv_pct": float(s["deriv_cv_pct"]),
                    "r2_horner": float(s["r2_horner"]),
                    "smooth_drift_pct": float(s["smooth_drift_pct"]),
                    "hb_slope_diff_pct": float(s["hb_slope_diff_pct"]),
                    "checks": {
                        "plateau": str(s["plateau_chk"]),
                        "horner": str(s["horner_chk"]),
                        "duration": str(s["dur_chk"]),
                        "smoothing": str(s["smooth_chk"]),
                        "intersection": str(s["intersect_chk"]),
                    },
                    "confidence_score": float(s["confidence_score"]),
                    "confidence_rating": str(s["confidence_rating"]),
                    "logic1_horner": {
                        "m": float(r1["m_psi_cycle"]), "k": float(r1["k_md"]), "kh": float(r1["k_md"]) * h_ft,
                        "skin": float(r1["skin"]), "dP_skin": float(r1["dP_skin"]),
                        "win_hr": _parse_win(s["horner_win_hr"]),
                    },
                    "logic2_bourdet": {
                        "m": float(r2["m_psi_cycle"]), "k": float(r2["k_md"]), "kh": float(r2["k_md"]) * h_ft,
                        "skin": float(r2["skin"]), "dP_skin": float(r2["dP_skin"]),
                        "win_hr": _parse_win(s["bourdet_win_hr"]),
                    },
                    "logic3_intersection": {
                        "m": float(r3["m_psi_cycle"]), "k": float(r3["k_md"]), "kh": float(r3["k_md"]) * h_ft,
                        "skin": float(r3["skin"]), "dP_skin": float(r3["dP_skin"]),
                        "win_hr": _parse_win(s["intersect_win_hr"]),
                    },
                }
        self._hybrid_pta_by_well = by_well
        return by_well

    def prepare(self) -> None:
        with self.lock:
            self.status = "loading baseline"
            self.build_baseline()
            self.load_judgement()
            self.status = "running PTA"
            self.run_pta()
            self.status = "applying judgement"
            self.apply()
            self.model_daily["physics_wls"] = self.base_daily
            self.model_status["physics_wls"] = "ready"
            self._load_precomputed_synthetic_models()
            self._load_cached_models_from_disk()
            if "synth_stacked_v2" in self.model_daily:
                self.active_model = "synth_stacked_v2"
            self._load_hybrid_pta_cache()
            self.status = "ready"

    def _model_cache_path(self, name: str) -> Path:
        return self.cache_dir / "models" / f"{name}.pkl"

    def _load_cached_models_from_disk(self) -> None:
        from vfm.allocation.models import REGISTRY
        mdir = self.cache_dir / "models"
        if not mdir.exists():
            return
        for p in mdir.glob("*.pkl"):
            name = p.stem
            if name in REGISTRY and name not in self.model_daily:
                try:
                    df = pd.read_pickle(p)
                    if isinstance(df, pd.DataFrame) and len(df) > 0:
                        self.model_daily[name] = df
                        self.model_status[name] = "ready (cached)"
                except Exception:
                    pass

    # ------------------------------------------------------------------ model zoo ------
    def _model_context(self) -> Dict[str, Any]:
        events = {}
        if self.settings.enabled:  # only engineer-accepted build-ups reach the deconvolution model
            events = {w: [e for e in evs if (self.settings.decisions.get(e["event_id"]) or EventDecision(use=False)).use]
                      for w, evs in self.events.items()}
        from vfm.modeling.physics_vfm import physics_kwargs_from_config
        return {"pr_prior": self.field_prior, "wc_global_slope": getattr(self.base_model, "wc_global_slope_", None),
                "pta_events": events, "ref_timestamp": self.base_model.ref_timestamp, "physics_cache": {}, "feature_cache": {},
                "physics_kwargs": physics_kwargs_from_config(self.cfg)}

    def compute_model(self, name: str) -> None:
        """Fit `name` on all tests (hindcast) and build its daily allocation; cached on disk."""
        from vfm.allocation.models import make_allocation_model
        if name == "physics_wls":
            self.model_daily[name] = self.base_daily
            self.model_status[name] = "ready"
            return
        if name in ("synth_stacked_v2", "synth_gbm_ensemble"):
            self._load_precomputed_synthetic_models()
            return
        mlock = self._model_locks.setdefault(name, threading.Lock())
        with mlock:
            if name in self.model_daily:
                return
            cp = self._model_cache_path(name)
            if cp.exists():
                try:
                    df = pd.read_pickle(cp)
                    if isinstance(df, pd.DataFrame) and len(df) > 0:
                        self.model_daily[name] = df
                        self.model_status[name] = "ready (cached)"
                        return
                except Exception:
                    pass
            self.model_status[name] = "running"
            try:
                t0 = time.perf_counter()
                model = make_allocation_model(name)
                hf_all = {w: self.hf(w) for w in self.wells}
                model.fit(self.tests, hf_all, self.comp, self.pvt, max_time=None, context=self._model_context())
                parts, refs = [], []
                for w in self.wells:
                    preds = model.predict(hf_all[w], w)
                    parts.append(aggregate_well_daily_physics(w, hf_all[w], preds, self.date_grid))
                    refs.append(compute_exact_window_vfm_refs(w, hf_all[w], preds, self._tests(w)))
                daily = anchor_daily_exact_window(pd.concat(parts, ignore_index=True),
                                                  pd.concat(refs, ignore_index=True), origin=self.date_grid[0])
                daily = daily.sort_values(["well_name", "date"]).reset_index(drop=True)
                self.model_daily[name] = daily
                try:
                    cp.parent.mkdir(parents=True, exist_ok=True)
                    daily.to_pickle(cp)
                except Exception:
                    pass
                self.model_status[name] = f"ready ({time.perf_counter() - t0:.1f}s)"
            except Exception as exc:
                self.model_status[name] = f"error: {type(exc).__name__}: {exc}"

    def set_active_model(self, name: str) -> None:
        from vfm.allocation.models import REGISTRY
        if name == "custom_selected" and name in self.model_daily:
            self.active_model = name
            return
        if name not in REGISTRY:
            raise KeyError(f"unknown model {name}")
        self.active_model = name
        if name not in self.model_daily:
            if self._model_cache_path(name).exists():
                self.compute_model(name)
            elif self.model_status.get(name) != "running":
                self.model_status[name] = "running"
                threading.Thread(target=self.compute_model, args=(name,), daemon=True).start()

    def run_benchmark_async(self, models: Optional[List[str]] = None) -> None:
        import subprocess
        import sys as _sys
        if self.bench_status.startswith("running"):
            return
        script = Path(__file__).resolve().parents[2] / "scripts" / "09_model_benchmark.py"

        def _run():
            self.bench_status = "running"
            env = dict(__import__("os").environ)
            if self.cfg.source_path:
                env["VFM_CONFIG"] = str(self.cfg.source_path)
            proc = subprocess.run([_sys.executable, str(script), *(models or [])], capture_output=True, text=True, env=env)
            self.bench_status = "finished" if proc.returncode == 0 else f"error: {proc.stderr[-400:]}"
        threading.Thread(target=_run, daemon=True).start()

    # ------------------------------------------------------------------ updates --------
    def update_settings(self, payload: Mapping[str, Any]) -> None:
        with self.lock:
            merged = self.settings.to_dict()
            for k, v in payload.items():
                if k == "decisions":
                    for eid, dec in (v or {}).items():
                        merged["decisions"][eid] = {**merged["decisions"].get(eid, {}), **dec}
                elif k == "estimators":
                    for w, tog in (v or {}).items():
                        merged["estimators"][w] = {**merged["estimators"].get(w, {}), **tog}
                else:
                    merged[k] = v
            self.settings = CouplingSettings.from_dict(merged)
            self.apply()
            self.save_judgement()

    def rerun_pta_event(self, eid: str, overrides: Mapping[str, Any]) -> Dict[str, Any]:
        with self.lock:
            well = eid.split("@")[0]
            clean = {k: v for k, v in overrides.items() if k in PtaRunSettings.__dataclass_fields__}
            if clean.get("manual_radial_dt_hr") in ([], None, ""):
                clean["manual_radial_dt_hr"] = None
            self.pta_overrides[eid] = {**self.pta_overrides.get(eid, {}), **clean}
            self.run_pta([well], only_event=eid)
            # re-derive the default decision only if the engineer never touched it
            self.apply()
            self.save_judgement()
            return self.pta_payload(eid)

    def set_pta_rate_basis(self, rate_basis: str) -> None:
        if rate_basis not in RATE_BASES:
            raise ValueError(f"rate_basis must be one of {RATE_BASES}, got {rate_basis!r}")
        with self.lock:
            self.pta_rate_basis = rate_basis
            for eid, ov in list(self.pta_overrides.items()):
                if isinstance(ov, dict) and "rate_basis" in ov:
                    ov.pop("rate_basis", None)
                    if not ov:
                        self.pta_overrides.pop(eid, None)
            self.run_pta()
            self.apply()
            self.save_judgement()

    def vfm_deliverable_df(self, model_name: Optional[str] = None, rate_type: str = "alloc") -> pd.DataFrame:
        """Hackathon Phase 1 VFM deliverable: date, well_name, oil_rate_stbd, water_rate_stbd, gas_rate_mscfd."""
        if rate_type not in ("alloc", "vfm"):
            raise ValueError(f"Unsupported rate_type {rate_type!r}; expected 'alloc' or 'vfm'")
        mname = model_name or self.active_model or "physics_wls"
        with self.lock:
            if mname == "physics_wls":
                df = self.coupled_daily if (self.settings.enabled and self.coupled_daily is not None) else self.base_daily
            else:
                df = self.model_daily.get(mname)
        if df is None:
            self.compute_model(mname)
            with self.lock:
                df = self.model_daily.get(mname)
        if df is None or df.empty:
            return pd.DataFrame(columns=["date", "well_name", "oil_rate_stbd", "water_rate_stbd", "gas_rate_mscfd"])
        ts = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None).dt.strftime("%Y-%m-%d")
        out = pd.DataFrame({
            "date": ts,
            "well_name": df["well_name"].astype(str),
            "oil_rate_stbd": df[f"oil_rate_{rate_type}_stbd"].astype(float),
            "water_rate_stbd": df[f"water_rate_{rate_type}_stbd"].astype(float),
            "gas_rate_mscfd": df[f"gas_rate_{rate_type}_mscfd"].astype(float),
        })
        return out.sort_values(["well_name", "date"]).reset_index(drop=True)

    def pta_deliverable_df(self, source: str = "auto") -> pd.DataFrame:
        """Hackathon Phase 2 PTA deliverable: well_name, t_shut, duration_hr, m_psi_cycle, k_md, skin, dP_skin."""
        adv_source_map = {
            "stehfest_analytical": "analytical_match",
            "von_schroeter_deconv": "deconvolution",
            "ilk_bspline_deconv": "ilk_bspline_deconv",
            "agarwal_equiv": "agarwal_equiv",
            "blasingame_integral": "blasingame_integral",
            "odeh_jones_multirate": "odeh_jones_multirate",
            "tiab_tds": "tiab_tds",
            "mdh_muskat": "mdh_muskat",
        }
        valid_sources = ("auto", "horner", "kernel", "hybrid_intersection", "hybrid_horner", "hybrid_bourdet", *adv_source_map.keys())
        if source not in valid_sources:
            raise ValueError(f"Unsupported PTA source {source!r}; expected one of {valid_sources}")
        with self.lock:
            if source in adv_source_map:
                akey = adv_source_map[source]
                rows_a: List[Dict[str, Any]] = []
                for w in sorted(self.events):
                    for ev in self.events[w]:
                        adv = self._get_advanced_pta(w, ev)
                        an = (adv or {}).get(akey) or {}
                        if not an.get("available"):
                            continue
                        t_shut = pd.to_datetime(ev["shut_in_start"], utc=True).tz_localize(None).strftime("%Y-%m-%d %H:%M:%S")
                        rows_a.append({
                            "well_name": w,
                            "t_shut": t_shut,
                            "duration_hr": round(float(ev["duration_hr"]), 2),
                            "m_psi_cycle": round(float(an["m"]), 4),
                            "k_md": round(float(an["k"]), 4),
                            "skin": round(float(an["skin"]), 4),
                            "dP_skin": round(float(an["dP_skin"]), 4),
                        })
                cols = ["well_name", "t_shut", "duration_hr", "m_psi_cycle", "k_md", "skin", "dP_skin"]
                df_a = pd.DataFrame(rows_a, columns=cols)
                return df_a.sort_values(["well_name", "t_shut"]).reset_index(drop=True) if not df_a.empty else df_a

            if source.startswith("hybrid_"):
                hyb = self._load_hybrid_pta_cache()
                key_map = {
                    "hybrid_horner": "logic1_horner",
                    "hybrid_bourdet": "logic2_bourdet",
                    "hybrid_intersection": "logic3_intersection",
                }
                lkey = key_map[source]
                rows_h: List[Dict[str, Any]] = []
                for w in sorted(self.events):
                    for ev in self.events[w]:
                        h_ev = hyb.get(w)
                        mf = self.manual_pta_fits.get(ev["event_id"]) if source == "hybrid_intersection" else None
                        if not h_ev and not mf:
                            continue
                        lg = mf if mf else h_ev[lkey]
                        t_shut = pd.to_datetime(ev["shut_in_start"], utc=True).tz_localize(None).strftime("%Y-%m-%d %H:%M:%S")
                        rows_h.append({
                            "well_name": w,
                            "t_shut": t_shut,
                            "duration_hr": round(float(ev["duration_hr"]), 2),
                            "m_psi_cycle": round(float(lg["m"]), 4),
                            "k_md": round(float(lg["k"]), 4),
                            "skin": round(float(lg["skin"]), 4),
                            "dP_skin": round(float(lg["dP_skin"]), 4),
                        })
                cols = ["well_name", "t_shut", "duration_hr", "m_psi_cycle", "k_md", "skin", "dP_skin"]
                df_h = pd.DataFrame(rows_h, columns=cols)
                return df_h.sort_values(["well_name", "t_shut"]).reset_index(drop=True) if not df_h.empty else df_h

            rows: List[Dict[str, Any]] = []
            for w in sorted(self.events):
                for ev in self.events[w]:
                    kern = (ev.get("result") or {}).get("results") or {}
                    horn = (ev.get("methods") or {}).get("horner") or {}
                    is_rad = ev.get("status") in ("RADIAL", "RADIAL_FORCED")
                    if source == "kernel" or (source == "auto" and is_rad):
                        m_val = kern.get("semilog_slope_psi_per_cycle")
                        k_val = kern.get("k_md")
                        s_val = kern.get("apparent_skin")
                        dp_val = kern.get("skin_pressure_drop_psi")
                    else:
                        m_val = horn.get("m")
                        k_val = horn.get("k")
                        s_val = horn.get("skin")
                        dp_val = (0.87 * float(m_val) * float(s_val)) if (
                            m_val is not None and s_val is not None and math.isfinite(m_val) and math.isfinite(s_val)
                        ) else None
                    t_shut = pd.to_datetime(ev["shut_in_start"], utc=True).tz_localize(None).strftime("%Y-%m-%d %H:%M:%S")
                    rows.append({
                        "well_name": w,
                        "t_shut": t_shut,
                        "duration_hr": round(float(ev["duration_hr"]), 2),
                        "m_psi_cycle": m_val,
                        "k_md": k_val,
                        "skin": s_val,
                        "dP_skin": dp_val,
                    })
            cols = ["well_name", "t_shut", "duration_hr", "m_psi_cycle", "k_md", "skin", "dP_skin"]
            df = pd.DataFrame(rows, columns=cols)
            return df.sort_values(["well_name", "t_shut"]).reset_index(drop=True) if not df.empty else df

    @staticmethod
    def export_table_bytes(df: pd.DataFrame, fmt: str = "xlsx", sheet_name: str = "Results") -> Tuple[bytes, str, str]:
        """Serialize a deliverable DataFrame to .xlsx or .csv bytes: returns (bytes, content_type, ext)."""
        fmt = fmt.lower().strip()
        if fmt == "csv":
            return df.to_csv(index=False).encode("utf-8"), "text/csv; charset=utf-8", "csv"
        if fmt == "xlsx":
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                df.to_excel(writer, index=False, sheet_name=sheet_name[:31])
            return (buf.getvalue(),
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    "xlsx")
        raise ValueError(f"Unsupported export format {fmt!r}; expected 'xlsx' or 'csv'")

    def export(self, name: str) -> Dict[str, str]:
        with self.lock:
            safe = "".join(c for c in name if c.isalnum() or c in "-_") or pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            out = self.cfg.resolve_path("outputs_dir") / "scenarios" / safe
            out.mkdir(parents=True, exist_ok=True)
            self.coupled_daily.to_csv(out / "daily_allocated_rates_coupled.csv", index=False)
            self.base_daily.to_csv(out / "daily_allocated_rates_baseline.csv", index=False)
            pd.DataFrame(self._event_rows()).to_csv(out / "pta_events.csv", index=False)
            vfm_df = self.vfm_deliverable_df(self.active_model, "alloc")
            pta_df = self.pta_deliverable_df("auto")
            vfm_df.to_csv(out / "vfm_deliverable.csv", index=False)
            pta_df.to_csv(out / "pta_deliverable.csv", index=False)
            (out / "vfm_deliverable.xlsx").write_bytes(self.export_table_bytes(vfm_df, "xlsx", "VFM")[0])
            (out / "pta_deliverable.xlsx").write_bytes(self.export_table_bytes(pta_df, "xlsx", "PTA")[0])
            (out / "judgement.json").write_text(json.dumps(to_jsonable({
                "coupling": self.settings.to_dict(), "pta_settings": self.pta_overrides,
                "pta_rate_basis": self.pta_rate_basis, "notes": self.notes}), indent=2),
                encoding="utf-8")
            return {"folder": str(out)}

    # ------------------------------------------------------------------ payloads -------
    def _event_rows(self) -> List[Dict[str, Any]]:
        rows = []
        hyb = self._load_hybrid_pta_cache()
        for w in self.wells:
            for ev in self.events.get(w, []):
                R, D = ev["result"]["results"], ev["derived"]
                dec = self.settings.decisions.get(ev["event_id"], EventDecision())
                hev = hyb.get(w)
                l3 = (hev or {}).get("logic3_intersection") or {}
                rows.append({
                    "event_id": ev["event_id"], "well_name": w, "shut_in_start": ev["shut_in_start"],
                    "duration_hr": round(ev["duration_hr"], 2), "status": ev["status"],
                    "kh_md_ft": R.get("kh_md_ft") if R.get("kh_md_ft") is not None else l3.get("kh"),
                    "k_md": R.get("k_md") if R.get("k_md") is not None else l3.get("k"),
                    "skin": R.get("apparent_skin") if R.get("apparent_skin") is not None else l3.get("skin"),
                    "hybrid_confidence_rating": (hev or {}).get("confidence_rating"),
                    "hybrid_confidence_score": (hev or {}).get("confidence_score"),
                    "hybrid_k_md": l3.get("k"),
                    "hybrid_skin": l3.get("skin"),
                    "p_star_gauge_psia": R.get("extrapolated_pressure_psi"),
                    "dietz_pbar_datum_psia": D.get("dietz_pbar_datum_psia"), "p_last_datum_psia": D.get("p_last_datum_psia"),
                    "rate_source": ev["settings"]["rate_source"],
                    "rate_basis": ev["settings"].get("rate_basis", self.pta_rate_basis),
                    "use": dec.use, "apply_pressure": dec.apply_pressure, "pressure_choice": dec.pressure_choice,
                    "apply_skin": dec.apply_skin, "apply_kh": dec.apply_kh,
                    "n_block_flags": sum(1 for f in ev["result"]["qc_flags"] if f["severity"] == "BLOCK"),
                })
        return rows

    def status_payload(self) -> Dict[str, Any]:
        return {"status": self.status, "ready": self.status == "ready"}

    def overview_payload(self) -> Dict[str, Any]:
        with self.lock:
            b, c = self.base_daily, self.coupled_daily
            wells = []
            for w in self.wells:
                bw, cw = b[b["well_name"] == w], c[c["well_name"] == w]
                base_oil, coup_oil = float(bw["oil_rate_alloc_stbd"].sum()), float(cw["oil_rate_alloc_stbd"].sum())
                ip_b = self.base_model.ipr_params_.get(w) or {}
                ip_c = (self.coupled_models[w].ipr_params_.get(w) if w in self.coupled_models else ip_b) or {}
                wells.append({
                    "well_name": w, "n_tests": int(len(self._tests(w))),
                    "cum_oil_baseline_stb": base_oil, "cum_oil_coupled_stb": coup_oil,
                    "cum_oil_change_pct": 100.0 * (coup_oil / base_oil - 1.0) if base_oil > 0 else None,
                    "coupled": w in self.coupled_models,
                    "ipr_loto_mae_baseline": ip_b.get("loto_mae_stbd"), "ipr_loto_mae_coupled": ip_c.get("loto_mae_stbd"),
                    "events": [{"event_id": e["event_id"], "status": e["status"], "duration_hr": e["duration_hr"]}
                               for e in self.events.get(w, [])],
                })
            return to_jsonable({
                "status": self.status, "config": str(self.cfg.source_path),
                "period": [self.date_grid[0], self.date_grid[-1]],
                "settings": self.settings.to_dict(), "wells": wells, "events": self._event_rows(),
                "notes": self.notes, "field_pr_gradient_prior_psi_ft": self.field_prior,
                "pta_rate_basis": self.pta_rate_basis,
                "judgement_file": str(self.judgement_path),
            })

    def well_payload(self, well: str) -> Dict[str, Any]:
        with self.lock:
            if well not in self.wells:
                raise KeyError(f"unknown well {well}")
            b = self.base_daily[self.base_daily["well_name"] == well]
            c = self.coupled_daily[self.coupled_daily["well_name"] == well]
            t = self._tests(well)
            ip_b = self.base_model.ipr_params_.get(well) or {}
            model_c = self.coupled_models.get(well, self.base_model)
            ip_c = model_c.ipr_params_.get(well) or {}
            days = (pd.DatetimeIndex(b["date"]) + pd.Timedelta(hours=12) - self.base_model.ref_timestamp).total_seconds() / 86400.0
            pr = lambda ip: [None if not ip else round(float(ip["Pr0_datum_psia"] - ip["decline_psi_per_day"] * d), 2) for d in days]
            j_fn = ip_c.get("j_multiplier")
            jm = [round(float(v), 4) for v in (j_fn(np.asarray(days)) if j_fn is not None else np.ones(len(days)))]
            phase = lambda df, prefix: {k: _series(df, f"{k}_rate_alloc_{'mscfd' if k == 'gas' else 'stbd'}") for k in ("oil", "water", "liquid", "gas")}
            wm_b = self.base_model.well_models_.get(well, {})
            wm_c = model_c.well_models_.get(well, {})
            weights = lambda wm: {k: wm.get(f"w_{k}") for k in ("ipr", "choke", "vlp")}
            return to_jsonable({
                "well_name": well,
                "dates": [d.strftime("%Y-%m-%d") for d in pd.DatetimeIndex(b["date"])],
                "baseline": {**phase(b, "b"), "liquid_vfm": _series(b, "liquid_rate_vfm_stbd"),
                             "q_ipr": _series(b, "q_liq_ipr_flowing_mean"), "q_choke": _series(b, "q_liq_choke_flowing_mean"),
                             "q_vlp": _series(b, "q_liq_vlp_flowing_mean"), "pr_datum": pr(ip_b)},
                "coupled": {**phase(c, "c"), "liquid_vfm": _series(c, "liquid_rate_vfm_stbd"),
                            "q_ipr": _series(c, "q_liq_ipr_flowing_mean"), "q_choke": _series(c, "q_liq_choke_flowing_mean"),
                            "q_vlp": _series(c, "q_liq_vlp_flowing_mean"), "pr_datum": pr(ip_c), "j_multiplier": jm},
                "uptime_frac": _series(b, "uptime_frac"),
                "tests": [{"t_mid": (r.test_start + (r.test_end - r.test_start) / 2).isoformat(), "oil": r.oil_rate_stbd,
                           "water": r.water_rate_stbd, "liquid": r.liquid_rate_stbd, "gas": r.gas_rate_mscfd}
                          for r in t.itertuples()],
                "ipr": {"baseline": {k: ip_b.get(k) for k in ("J", "Pr0_datum_psia", "decline_psi_per_day", "loto_mae_stbd", "pr_prior_source")},
                        "coupled": {k: ip_c.get(k) for k in ("J", "Pr0_datum_psia", "decline_psi_per_day", "loto_mae_stbd", "pr_prior_source")},
                        "anchors_baseline": [{"timestamp": pd.Timestamp(a["shut_in_start"]) if "shut_in_start" in a else None,
                                              "pr_datum_psia": a["pr_datum_psia"], "source": a.get("source", "internal PBU")}
                                             for a in ip_b.get("buildup_anchors", [])],
                        "anchors_coupled": [{"timestamp": a.get("timestamp", a.get("shut_in_start")), "pr_datum_psia": a["pr_datum_psia"],
                                             "source": a.get("source", "internal PBU")} for a in ip_c.get("buildup_anchors", [])]},
                "weights": {"baseline": weights(wm_b), "coupled": weights(wm_c)},
                "estimators": self.settings.estimators.get(well, {"ipr": True, "choke": True, "vlp": True}),
                "condition_timeline": (getattr(self, "adjustments", {}).get(well) or {}).get("condition_timeline", []),
                "property_changes": [asdict(pc) for pc in self.settings.property_changes if pc.well_name == well],
                "events": [e for e in self._event_rows() if e["well_name"] == well],
                "notes": [n for n in self.notes if n.get("well_name") in (well, "*")],
            })

    def pta_payload(self, eid: str) -> Dict[str, Any]:
        with self.lock:
            well = eid.split("@")[0]
            ev = next((e for e in self.events.get(well, []) if e["event_id"] == eid), None)
            if ev is None:
                raise KeyError(f"unknown event {eid}")
            res = ev["result"]
            ser = res["diagnostics"].get("series") or {}
            dec = self.settings.decisions.get(eid, EventDecision())
            return to_jsonable({
                "event_id": eid, "well_name": well, "status": ev["status"],
                "shut_in_start": ev["shut_in_start"], "shut_in_end": ev["shut_in_end"], "duration_hr": ev["duration_hr"],
                "settings": ev["settings"], "rate_note": ev["rate_note"], "datum": ev["datum"],
                "series": {k: ser.get(k) for k in ("dt_hr", "dp_psi", "derivative_psi")},
                "radial_interval": res["diagnostics"].get("radial_interval"),
                "flow_regimes": {k: v for k, v in (res["diagnostics"].get("flow_regimes") or {}).items() if k != "local_slopes"},
                "results": res["results"], "derived": ev["derived"], "internal_pbu": ev.get("internal_pbu"),
                "qc_flags": res["qc_flags"], "decision": asdict(dec),
                "rate_history": {"q_final_reservoir_bbl_day": res["inputs"]["rate_source"].get("q_final_reservoir_bbl_day"),
                                 "producing_duration_hr": res["inputs"]["rate_source"].get("producing_duration_hr"),
                                 "n_periods": len(res["inputs"]["rate_source"].get("periods", []))},
                "methods": ev.get("methods"),
                "hybrid_pta": self._load_hybrid_pta_cache().get(well),
                **self._pta_horner_and_interp(well, ev),
                "link": next((r for r in self._link_rows() if r["event_id"] == eid), None),
            })

    def _get_advanced_pta(self, well: str, ev: Mapping[str, Any]) -> Dict[str, Any]:
        """Compute & cache globally recognized Non-IARF PTA suite (Stehfest + Deconvolution + Tiab TDS)."""
        eid = ev["event_id"]
        if eid in self._advanced_pta_cache:
            return self._advanced_pta_cache[eid]
        try:
            from vfm.pta.advanced_pta import run_advanced_non_iarf_suite
            hyb = self._load_hybrid_pta_cache().get(well)
            qo_raw = (hyb or {}).get("qo_stbd") or ((ev.get("result") or {}).get("inputs") or {}).get("rate_source", {}).get("rate_stbd")
            vfm_d = self.model_daily.get("synth_stacked_v2", self.base_daily)
            vfm_w = vfm_d[vfm_d["well_name"] == well] if vfm_d is not None else None
            if qo_raw is None or not np.isfinite(float(qo_raw)) or float(qo_raw) <= 0.0:
                if vfm_w is not None and not vfm_w.empty:
                    o_col = "oil_rate_vfm_stbd" if "oil_rate_vfm_stbd" in vfm_w.columns else "oil_rate_stbd"
                    qo_raw = float(vfm_w[vfm_w[o_col] > 0][o_col].median())
                else:
                    raise ValueError(f"No valid pre-shut-in oil rate found for {well}")
            qo = float(qo_raw)
            pwf = float((hyb or {}).get("pwf_3h_med") or 0.0)
            hf_w = self.hf(well)
            ts = pd.to_datetime(hf_w["timestamp"], utc=True)
            s0 = pd.Timestamp(ev["shut_in_start"])
            t0_u = s0.tz_localize("UTC") if s0.tzinfo is None else s0.tz_convert("UTC")
            if pwf <= 0.0:
                pre = hf_w[(ts >= t0_u - pd.Timedelta(hours=3)) & (ts <= t0_u)]["dh_pressure_psia"].dropna()
                if pre.empty:
                    raise ValueError(f"No pre-shut-in gauge pressure found for {well}")
                pwf = float(pre.median())
            tp_raw = (hyb or {}).get("tp_hr") or ((ev.get("result") or {}).get("inputs") or {}).get("rate_source", {}).get("producing_duration_hr")
            if tp_raw is not None and np.isfinite(float(tp_raw)) and float(tp_raw) > 0.0:
                tp_hr = float(tp_raw)
            else:
                tp_hr = float(max((t0_u - ts.min()).total_seconds() / 3600.0, 24.0))
            out = run_advanced_non_iarf_suite(
                hf_w,
                pd.Timestamp(ev["shut_in_start"]),
                pd.Timestamp(ev["shut_in_end"]),
                qo, pwf, tp_hr,
                self._row(self.pvt, well),
                self._row(self.comp, well),
                vfm_daily_well=vfm_w,
                hyb_summary=hyb,
            )
            self._advanced_pta_cache[eid] = out
            return out
        except Exception as exc:
            return {"available": False, "note": str(exc)}

    def _pta_horner_and_interp(self, well: str, ev: Mapping[str, Any]) -> Dict[str, Any]:
        from vfm.mvp_engine import build_horner_and_interpretation
        from vfm.diagnostics import compute_pta_qo_sensitivity
        hyb = self._load_hybrid_pta_cache().get(well)
        mf = self.manual_pta_fits.get(ev["event_id"])
        hp, interp = build_horner_and_interpretation(
            ev, self._row(self.pvt, well), self._row(self.comp, well), hyb, self.hf(well), manual_fit=mf
        )
        adv = self._get_advanced_pta(well, ev)
        eid = ev["event_id"]
        if eid not in self._qo_sens_cache:
            try:
                self._qo_sens_cache[eid] = compute_pta_qo_sensitivity(
                    well, ev, self.hf(well),
                    self._row(self.pvt, well), self._row(self.comp, well),
                    self.tests, self.model_daily,
                    hyb=hyb, adv=adv, horner_plot=hp,
                )
            except Exception as exc:
                self._qo_sens_cache[eid] = {"available": False, "note": str(exc)}
        return {
            "horner_plot": hp,
            "interpretation": interp,
            "manual_fit": mf,
            "advanced_pta": adv,
            "qo_sensitivity": self._qo_sens_cache[eid],
        }

    def vfm_diagnostics_payload(self, method: str = "pearson") -> Dict[str, Any]:
        """Return VFM Correlation Matrix, Training Feature Audit & Multi-Level SHAP Evaluation."""
        meth = "spearman" if str(method).lower() == "spearman" else "pearson"
        if meth in self._vfm_diag_cache:
            return to_jsonable(self._vfm_diag_cache[meth])
        from vfm.diagnostics import compute_vfm_feature_diagnostics, load_v2_bundle
        bundle = load_v2_bundle()
        cache_file = self.cache_dir / f"vfm_diagnostics_{meth}.pkl"
        sig = f"{self._cache_key()}:{bundle.get('n_real_tests')}:{bundle.get('ref_date')}:{meth}:v2"
        if cache_file.exists():
            try:
                with cache_file.open("rb") as f:
                    blob = pickle.load(f)
                if blob.get("sig") == sig:
                    self._vfm_diag_cache[meth] = blob["data"]
                    return to_jsonable(blob["data"])
            except Exception:
                pass
        with self.lock:
            hf_map = {w: self.hf(w) for w in self.wells}
            data = compute_vfm_feature_diagnostics(
                self.tests, hf_map, self.comp, self.pvt, bundle, method=meth
            )
            self._vfm_diag_cache[meth] = data
            try:
                with cache_file.open("wb") as f:
                    pickle.dump({"sig": sig, "data": data}, f)
            except Exception:
                pass
            return to_jsonable(data)

    def retrain_custom_vfm_payload(self, body: Mapping[str, Any]) -> Dict[str, Any]:
        """Retrain a custom VFM strictly on user-selected parameters and register it live."""
        from vfm.diagnostics import load_v2_bundle, retrain_custom_vfm
        selected = body.get("features") or []
        model_type = str(body.get("model_type") or "lgbm").lower()
        activate = bool(body.get("activate", True))
        try:
            bundle = load_v2_bundle()
        except Exception:
            bundle = None
        with self.lock:
            hf_map = {w: self.hf(w) for w in self.wells}
            payload, daily_df = retrain_custom_vfm(
                selected_features=selected,
                model_type=model_type,
                tests_df=self.tests,
                hf_by_well=hf_map,
                comp_df=self.comp,
                pvt_df=self.pvt,
                date_grid=self.date_grid,
                bundle=bundle,
            )
            self.model_daily["custom_selected"] = daily_df
            self.model_status["custom_selected"] = f"ready ({model_type}, {payload['n_features']} feats)"
            self._custom_vfm_meta = payload
            if activate:
                self.active_model = "custom_selected"
            return to_jsonable({
                **payload,
                "models_payload": self.models_payload(),
            })

    def grid_search_vfm_payload(self, body: Mapping[str, Any]) -> Dict[str, Any]:
        """Run automated 5-fold stratified CV grid search across feature presets x model families,
        automatically register the winning preset as custom_selected, and return the full ranked leaderboard."""
        from vfm.diagnostics import grid_search_vfm_presets, load_v2_bundle

        model_types = body.get("model_types") or ["lgbm", "rf", "gbr", "ridge"]
        optimize_metric = str(body.get("optimize_metric") or "composite").lower()
        custom_features = body.get("current_features") or None
        activate = bool(body.get("activate", True))
        try:
            bundle = load_v2_bundle()
        except Exception:
            bundle = None
        with self.lock:
            hf_map = {w: self.hf(w) for w in self.wells}
            grid_payload, best_daily = grid_search_vfm_presets(
                model_types=model_types,
                optimize_metric=optimize_metric,
                tests_df=self.tests,
                hf_by_well=hf_map,
                comp_df=self.comp,
                pvt_df=self.pvt,
                date_grid=self.date_grid,
                bundle=bundle,
                custom_features=custom_features,
            )
            best = grid_payload["best"]
            self.model_daily["custom_selected"] = best_daily
            self.model_status["custom_selected"] = (
                f"ready (grid best: {best['model_type']}, {best['n_features']} feats)"
            )
            self._custom_vfm_meta = grid_payload["best_custom_result"]
            if activate:
                self.active_model = "custom_selected"
            return to_jsonable({
                **grid_payload,
                "models_payload": self.models_payload(),
            })

    def manual_pta_fit_payload(self, body: Mapping[str, Any]) -> Dict[str, Any]:
        from vfm.mvp_engine import compute_manual_two_point_pta
        eid = str(body["event"])
        well = eid.split("@")[0]
        with self.lock:
            ev = next((e for e in self.events.get(well, []) if e["event_id"] == eid), None)
            if ev is None:
                raise KeyError(f"unknown event {eid}")
            if body.get("reset"):
                self.manual_pta_fits.pop(eid, None)
            else:
                hyb = self._load_hybrid_pta_cache().get(well)
                mf = compute_manual_two_point_pta(
                    ev,
                    self._row(self.pvt, well),
                    self._row(self.comp, well),
                    hyb,
                    self.hf(well),
                    float(body["dt1_hr"]),
                    float(body["dt2_hr"]),
                    fit_mode=str(body.get("fit_mode") or "two_point"),
                )
                self.manual_pta_fits[eid] = mf
        return self.pta_payload(eid)

    def upload_and_run_payload(self, body: Mapping[str, Any]) -> Dict[str, Any]:
        from vfm.mvp_engine import run_uploaded_inference
        res = run_uploaded_inference(body.get("files") or {}, self.comp, self.pvt, self.tests)
        return to_jsonable(res)

    def engineering_report_html_bytes(self) -> bytes:
        from vfm.mvp_engine import render_engineering_report_html
        with self.lock:
            return render_engineering_report_html(self)

    # ------------------------------------------------------------------ PTA <-> VFM link
    def _link_rows(self) -> List[Dict[str, Any]]:
        """Side-by-side PTA and VFM (IPR) values per build-up: pressure, productivity, kh."""
        rows = []
        for w in self.wells:
            ip_b = self.base_model.ipr_params_.get(w) or {}
            model_c = self.coupled_models.get(w, self.base_model)
            ip_c = model_c.ipr_params_.get(w) or {}
            pv = self._row(self.pvt, w)
            for ev in self.events.get(w, []):
                t_ev = pd.Timestamp(ev["shut_in_start"])
                d = (t_ev - self.base_model.ref_timestamp).total_seconds() / 86400.0
                pr = lambda ip: None if not ip else float(ip["Pr0_datum_psia"] - ip["decline_psi_per_day"] * d)
                jm = ip_c.get("j_multiplier")
                j_b = ip_b.get("J")
                j_c = None if not ip_c else float(ip_c["J"]) * (float(jm(np.array([d]))[0]) if jm else 1.0)
                R, D = ev["result"]["results"], ev["derived"]
                M = ev.get("methods") or {}
                P, H, MM = M.get("productivity") or {}, M.get("horner") or {}, M.get("model_match") or {}
                kh_from_j = None
                skin = R.get("apparent_skin")
                area = pv.get("estimated_drainage_area_acres")
                if j_b and skin is not None and area and np.isfinite(area):
                    re = math.sqrt(float(area) * 43560 / math.pi)
                    b_liq = None  # liquid FVF from the last well test's water cut
                    t_last = self._tests(w)
                    if len(t_last):
                        wc = float(t_last["water_cut_pct"].iloc[-1]) / 100.0
                        b_liq = (1 - wc) * float(pv["oil_fvf_rb_stb"]) + wc * float(self._row(self.comp, w)["water_fvf_rb_stb"])
                    lt = math.log(re / float(pv["wellbore_radius_ft"])) - 0.75 + float(skin)
                    if b_liq and lt > 0.1:
                        kh_from_j = float(j_b) * 141.2 * b_liq * float(pv["oil_viscosity_cp"]) * lt
                rows.append({
                    "event_id": ev["event_id"], "well_name": w, "status": ev["status"],
                    "kh_derivative": R.get("kh_md_ft"), "kh_horner": H.get("kh"), "kh_model": MM.get("kh_md_ft"),
                    "kh_from_vfm_j": kh_from_j,
                    "skin_derivative": skin, "skin_horner": H.get("skin"), "skin_model": MM.get("skin"),
                    "p_star_gauge": R.get("extrapolated_pressure_psi"), "p_star_horner_gauge": H.get("p_star"),
                    "pbar_datum_pta": D.get("dietz_pbar_datum_psia"), "p_last_datum_pta": D.get("p_last_datum_psia"),
                    "pr_datum_vfm_baseline": pr(ip_b), "pr_datum_vfm_coupled": pr(ip_c),
                    "pi_actual_pta": P.get("pi_actual_stbd_psi"), "pi_theoretical_pta": P.get("pi_theoretical_stbd_psi"),
                    "pi_ideal_pta": P.get("pi_ideal_stbd_psi"), "flow_efficiency": P.get("flow_efficiency"),
                    "damage_ratio": P.get("damage_ratio"), "pi_avg_source": P.get("p_avg_source"),
                    "j_vfm_baseline": j_b, "j_vfm_coupled": j_c,
                    "pi_ratio_pta_to_vfm": (P.get("pi_actual_stbd_psi") / j_b) if (P.get("pi_actual_stbd_psi") and j_b) else None,
                })
        return rows

    def link_payload(self) -> Dict[str, Any]:
        with self.lock:
            return to_jsonable({"rows": self._link_rows(), "settings": self.settings.to_dict(),
                                "notes": self.notes, "enabled": self.settings.enabled})

    def _synthetic_benchmark_rows(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Compute summary & detail benchmark rows on the 85 well tests for our precomputed synthetic models."""
        summ_rows: List[Dict[str, Any]] = []
        det_rows: List[Dict[str, Any]] = []
        for mname in ("synth_stacked_v2", "synth_gbm_ensemble"):
            d = self.model_daily.get(mname)
            if d is None or d.empty:
                continue
            d_tmp = d.copy()
            d_tmp["date_key"] = pd.to_datetime(d_tmp["date"], utc=True).dt.floor("D")
            matched = []
            for r in self.tests.itertuples():
                w = r.well_name
                t0, t1 = pd.Timestamp(r.test_start), pd.Timestamp(r.test_end)
                d0, d1 = t0.floor("D"), t1.floor("D")
                sub = d_tmp[(d_tmp["well_name"] == w) & (d_tmp["date_key"] >= d0) & (d_tmp["date_key"] <= d1)]
                if sub.empty:
                    continue
                wts = [max(0.01, (min(t1, dk + pd.Timedelta(days=1)) - max(t0, dk)).total_seconds() / 3600.0)
                       for dk in sub["date_key"]]
                wa = np.asarray(wts, dtype="float64")
                po = float(np.average(sub["oil_rate_alloc_stbd"], weights=wa))
                pw = float(np.average(sub["water_rate_alloc_stbd"], weights=wa))
                pg = float(np.average(sub["gas_rate_alloc_mscfd"], weights=wa))
                pl = po + pw
                row_d = {
                    "well_name": w, "test_start": str(t0),
                    "true_oil_rate_stbd": float(r.oil_rate_stbd), "pred_oil_rate_stbd": po,
                    "true_water_rate_stbd": float(r.water_rate_stbd), "pred_water_rate_stbd": pw,
                    "true_gas_rate_mscfd": float(r.gas_rate_mscfd), "pred_gas_rate_mscfd": pg,
                    "true_liquid_rate_stbd": float(r.liquid_rate_stbd), "pred_liquid_rate_stbd": pl,
                }
                matched.append(row_d)
                for cv_mode in ("walk_forward", "loto"):
                    det_rows.append({"cv_mode": cv_mode, "model": mname, **row_d})
            if not matched:
                continue
            mdf = pd.DataFrame(matched)
            def _mstats(c_true: str, c_pred: str) -> Tuple[float, float, float]:
                yt = mdf[c_true].to_numpy(dtype="float64")
                yp = mdf[c_pred].to_numpy(dtype="float64")
                ok = np.isfinite(yt) & np.isfinite(yp) & (yt > 0)
                yt, yp = yt[ok], yp[ok]
                mae = float(np.mean(np.abs(yp - yt)))
                mape = float(np.mean(np.abs(yp - yt) / yt) * 100.0)
                bias = float(np.mean((yp - yt) / yt) * 100.0)
                return mae, mape, bias
            o_mae, o_mape, o_bias = _mstats("true_oil_rate_stbd", "pred_oil_rate_stbd")
            w_mae, w_mape, w_bias = _mstats("true_water_rate_stbd", "pred_water_rate_stbd")
            g_mae, g_mape, g_bias = _mstats("true_gas_rate_mscfd", "pred_gas_rate_mscfd")
            l_mae, l_mape, l_bias = _mstats("true_liquid_rate_stbd", "pred_liquid_rate_stbd")
            # For synth_stacked_v2, report strictly out-of-sample 5-fold OOF CV metrics from the trained bundle
            # rather than in-sample deployment-refit residuals, keeping benchmark comparisons 100% leakage-free.
            if mname == "synth_stacked_v2":
                try:
                    from vfm.diagnostics import load_v2_bundle
                    _bnd = load_v2_bundle()
                    _oof = _bnd.get("oof_metrics") or {}
                    if {"oil", "water", "gas", "liquid"}.issubset(_oof.keys()):
                        o_mae, o_mape = float(_oof["oil"]["MAE"]), float(_oof["oil"]["MAPE"])
                        w_mae, w_mape = float(_oof["water"]["MAE"]), float(_oof["water"]["MAPE"])
                        g_mae, g_mape = float(_oof["gas"]["MAE"]), float(_oof["gas"]["MAPE"])
                        l_mae, l_mape = float(_oof["liquid"]["MAE"]), float(_oof["liquid"]["MAPE"])
                except Exception:
                    pass
            for cv_mode in ("walk_forward", "loto"):
                summ_rows.append({
                    "cv_mode": cv_mode, "model": mname, "family": "grey-box stacked (5-fold OOF CV)" if mname == "synth_stacked_v2" else "synthetic/ML",
                    "n_tests": int(len(mdf)),
                    "liquid_mae": l_mae, "liquid_mape": l_mape, "liquid_bias_pct": l_bias,
                    "oil_mae": o_mae, "oil_mape": o_mape, "oil_bias_pct": o_bias,
                    "water_mae": w_mae, "water_mape": w_mape, "water_bias_pct": w_bias,
                    "gas_mae": g_mae, "gas_mape": g_mape, "gas_bias_pct": g_bias,
                })
        return summ_rows, det_rows

    # ------------------------------------------------------------------ models payloads
    def models_payload(self) -> Dict[str, Any]:
        from vfm.allocation.models import list_allocation_models
        out_dir = self.cfg.resolve_path("outputs_dir")
        summ = out_dir / "model_benchmark_summary.csv"
        raw_summary = pd.read_csv(summ).to_dict(orient="records") if summ.exists() else []
        model_infos = list_allocation_models()
        fam_map = {info["name"]: info.get("family", "") for info in model_infos}
        summary = []
        for r in raw_summary:
            nr = dict(r)
            nr.setdefault("family", fam_map.get(nr.get("model"), ""))
            if "oil_mae" not in nr and "oil_rate_stbd_mae" in nr:
                nr["oil_mae"] = nr["oil_rate_stbd_mae"]
            if "oil_mape" not in nr and "oil_rate_stbd_mape" in nr:
                nr["oil_mape"] = nr["oil_rate_stbd_mape"]
            if "water_mae" not in nr and "water_rate_stbd_mae" in nr:
                nr["water_mae"] = nr["water_rate_stbd_mae"]
            if "water_mape" not in nr and "water_rate_stbd_mape" in nr:
                nr["water_mape"] = nr["water_rate_stbd_mape"]
            if "gas_mae" not in nr and "gas_rate_mscfd_mae" in nr:
                nr["gas_mae"] = nr["gas_rate_mscfd_mae"]
            if "gas_mape" not in nr and "gas_rate_mscfd_mape" in nr:
                nr["gas_mape"] = nr["gas_rate_mscfd_mape"]
            if "liquid_mae" not in nr and "liquid_rate_stbd_mae" in nr:
                nr["liquid_mae"] = nr["liquid_rate_stbd_mae"]
            if "liquid_mape" not in nr and "liquid_rate_stbd_mape" in nr:
                nr["liquid_mape"] = nr["liquid_rate_stbd_mape"]
            summary.append(nr)
        syn_summ, _ = self._synthetic_benchmark_rows()
        existing_keys = {(r.get("cv_mode"), r.get("model")) for r in summary}
        for sr in syn_summ:
            if (sr["cv_mode"], sr["model"]) not in existing_keys:
                summary.append(sr)
        models = []
        for info in model_infos:
            models.append({**info, "status": self.model_status.get(info["name"], "not computed")})
        if self._custom_vfm_meta is not None and "custom_selected" in self.model_daily:
            cm = self._custom_vfm_meta
            mt_lbl = cm["model_type"].upper()
            models.insert(0, {
                "name": "custom_selected",
                "family": f"custom/{cm['model_type']}",
                "description": f"Custom {mt_lbl} trained on {cm['n_features']} selected parameters",
                "status": self.model_status.get("custom_selected", "ready (custom)"),
            })
            m_cv = cm["metrics"]
            for cv_mode in ("walk_forward", "loto"):
                summary.append({
                    "cv_mode": cv_mode, "model": "custom_selected", "family": f"custom/{cm['model_type']}",
                    "n_tests": cm["n_tests"],
                    "oil_mae": m_cv["oil"]["mae"], "oil_mape": m_cv["oil"]["mape"],
                    "water_mae": m_cv["water"]["mae"], "water_mape": m_cv["water"]["mape"],
                    "gas_mae": m_cv["gas"]["mae"], "gas_mape": m_cv["gas"]["mape"],
                    "liquid_mae": m_cv["liquid"]["mae"], "liquid_mape": m_cv["liquid"]["mape"],
                })
        return to_jsonable({"models": models, "active": self.active_model, "benchmark_summary": summary,
                            "benchmark_status": self.bench_status})

    def model_well_payload(self, well: str, names: List[str]) -> Dict[str, Any]:
        if well not in self.wells:
            raise KeyError(f"unknown well {well}")
        out: Dict[str, Any] = {"well_name": well, "models": {}}
        for n in names:
            d = self.model_daily.get(n)
            if d is None:
                out["models"][n] = {"status": self.model_status.get(n, "not computed")}
                continue
            dw = d[d["well_name"] == well]
            out["dates"] = [x.strftime("%Y-%m-%d") for x in pd.DatetimeIndex(dw["date"])]
            out["models"][n] = {"status": self.model_status.get(n, "ready"),
                                **{f"{p}_vfm": _series(dw, f"{p}_rate_vfm_{'mscfd' if p == 'gas' else 'stbd'}") for p in ("oil", "water", "liquid", "gas")},
                                **{f"{p}_alloc": _series(dw, f"{p}_rate_alloc_{'mscfd' if p == 'gas' else 'stbd'}") for p in ("oil", "water", "liquid", "gas")},
                                "cum_oil_alloc_stb": float(dw["oil_rate_alloc_stbd"].sum())}
        t = self._tests(well)
        out["tests"] = [{"t_mid": (r.test_start + (r.test_end - r.test_start) / 2).isoformat(), "oil": r.oil_rate_stbd,
                         "water": r.water_rate_stbd, "liquid": r.liquid_rate_stbd, "gas": r.gas_rate_mscfd} for r in t.itertuples()]
        return to_jsonable(out)

    def benchmark_payload(self) -> Dict[str, Any]:
        out_dir = self.cfg.resolve_path("outputs_dir")
        det_p = out_dir / "model_benchmark_detail.csv"
        if not det_p.exists():
            return {"available": False, "status": self.bench_status}
        det = pd.read_csv(det_p)
        summary = pd.read_csv(out_dir / "model_benchmark_summary.csv")
        syn_summ, syn_det = self._synthetic_benchmark_rows()
        if syn_summ:
            syn_s_df = pd.DataFrame(syn_summ)
            summary = pd.concat([summary[~summary["model"].isin(syn_s_df["model"])], syn_s_df], ignore_index=True)
        if syn_det:
            syn_d_df = pd.DataFrame(syn_det)
            det = pd.concat([det[~det["model"].isin(syn_d_df["model"])], syn_d_df], ignore_index=True)
        per_well = []
        for (cv, m, w), g in det.groupby(["cv_mode", "model", "well_name"]):
            t, p = g["true_liquid_rate_stbd"].to_numpy(float), g["pred_liquid_rate_stbd"].to_numpy(float)
            ok = np.isfinite(t) & np.isfinite(p)
            per_well.append({"cv_mode": cv, "model": m, "well_name": w,
                             "liquid_mape": float(np.mean(np.abs(p[ok] - t[ok]) / np.maximum(t[ok], 1)) * 100) if ok.any() else None,
                             "oil_mape": float(np.nanmean(np.abs(g["pred_oil_rate_stbd"] - g["true_oil_rate_stbd"]) / g["true_oil_rate_stbd"].clip(lower=1)) * 100)})
        points = det[["cv_mode", "model", "well_name", "test_start", "true_liquid_rate_stbd", "pred_liquid_rate_stbd",
                      "true_oil_rate_stbd", "pred_oil_rate_stbd"]].to_dict(orient="records")
        return to_jsonable({"available": True, "status": self.bench_status, "summary": summary.to_dict(orient="records"),
                            "per_well": per_well, "points": points})

    # ------------------------------------------------------------------ correlations --
    def _sensor_tables(self) -> None:
        if self._sensor_daily is not None:
            return
        cols = ["whp_psia", "whp_downstream_psia", "dh_pressure_psia", "dh_temperature_degf", "gas_lift_rate_mscfd",
                "choke_opening_pct"]
        dparts, hparts = [], []
        for w in self.wells:
            h = self.hf(w)
            full = load_frame(self.cache_dir / "wells" / w)
            extra = [c for c in ("annulus_pressure_psia",) if c in full.columns]
            df = h[[c for c in cols if c in h.columns]].copy()
            for c in extra:
                df[c] = full[c].to_numpy()
            df.index = pd.DatetimeIndex(pd.to_datetime(h["timestamp"], utc=True))
            df["dp_choke_psi"] = df["whp_psia"] - df["whp_downstream_psia"]
            df["dp_tubing_psi"] = df["dh_pressure_psia"] - df["whp_psia"]
            hr = df.resample("1h").mean()
            hr.insert(0, "well_name", w)
            hparts.append(hr.reset_index(names="time"))
            dy = df.resample("1D").mean()
            dy.insert(0, "well_name", w)
            dparts.append(dy.reset_index(names="date"))
        self._sensor_hourly = pd.concat(hparts, ignore_index=True)
        self._sensor_daily = pd.concat(dparts, ignore_index=True)

    def _corr_frame(self, source: str, well: str) -> pd.DataFrame:
        if source == "well_tests":
            df = self.tests.copy()
        elif source == "hourly":
            self._sensor_tables()
            df = self._sensor_hourly.copy()
        else:  # daily: active model allocation + daily sensor means
            self._sensor_tables()
            d = self.model_daily.get(self.active_model, self.base_daily).copy()
            d["date"] = pd.to_datetime(d["date"], utc=True)
            s = self._sensor_daily.copy()
            s["date"] = pd.to_datetime(s["date"], utc=True)
            df = d.merge(s, on=["well_name", "date"], how="left")
            df = df[df["flowing_minutes"] > 0]
        if well and well != "all":
            df = df[df["well_name"] == well]
        num = df.select_dtypes("number")
        keep = [c for c in num.columns if num[c].notna().sum() >= 5 and num[c].nunique(dropna=True) > 1
                and not c.startswith("n_") and c not in ("test_idx",)]
        return df[["well_name"] + keep]

    def correlation_payload(self, source: str = "well_tests", well: str = "all", method: str = "pearson") -> Dict[str, Any]:
        if method not in ("pearson", "spearman", "kendall"):
            raise KeyError(f"method {method}")
        df = self._corr_frame(source, well)
        num = df.drop(columns=["well_name"])
        c = num.corr(method=method, min_periods=5)
        return to_jsonable({"source": source, "well": well, "method": method, "variables": list(c.columns),
                            "matrix": c.to_numpy().tolist(), "n_rows": int(len(num)),
                            "wells": sorted(df["well_name"].unique().tolist())})

    def scatter_payload(self, source: str, well: str, x: str, y: str, max_points: int = 6000) -> Dict[str, Any]:
        df = self._corr_frame(source, well)
        if x not in df.columns or y not in df.columns:
            raise KeyError(f"unknown variable {x if x not in df.columns else y}")
        d = df[["well_name", x, y]].dropna()
        if len(d) > max_points:
            d = d.sample(max_points, random_state=0)
        r = float(d[x].corr(d[y])) if len(d) > 2 else None
        return to_jsonable({"x": x, "y": y, "points": d.rename(columns={x: "x", y: "y"}).to_dict(orient="records"),
                            "pearson_r": r, "n": int(len(d))})

    # ------------------------------------------------------------------ monitoring ----
    _MONITOR_CHANNELS = (
        {"key": "dh_pressure_psia", "label": "Downhole Pressure (DHP)", "unit": "psia", "group": "Pressures", "axis": "y1", "default_on": True},
        {"key": "whp_psia", "label": "Wellhead Pressure (WHP)", "unit": "psia", "group": "Pressures", "axis": "y1", "default_on": True},
        {"key": "whp_downstream_psia", "label": "Downstream Pressure", "unit": "psia", "group": "Pressures", "axis": "y1", "default_on": False},
        {"key": "annulus_pressure_psia", "label": "Annulus Pressure", "unit": "psia", "group": "Pressures", "axis": "y1", "default_on": False},
        {"key": "dp_tubing_psi", "label": "Tubing ΔP (DHP − WHP)", "unit": "psi", "group": "Derived ΔP", "axis": "y1", "default_on": False},
        {"key": "dp_choke_psi", "label": "Choke ΔP (WHP − DS)", "unit": "psi", "group": "Derived ΔP", "axis": "y1", "default_on": False},
        {"key": "oil_rate_vfm_stbd", "label": "Oil Rate (Selected VFM)", "unit": "STB/d", "group": "VFM Rates", "axis": "y2", "default_on": True},
        {"key": "water_rate_vfm_stbd", "label": "Water Rate (Selected VFM)", "unit": "STB/d", "group": "VFM Rates", "axis": "y2", "default_on": False},
        {"key": "liquid_rate_vfm_stbd", "label": "Liquid Rate (Selected VFM)", "unit": "STB/d", "group": "VFM Rates", "axis": "y2", "default_on": False},
        {"key": "gas_rate_vfm_mscfd", "label": "Formation Gas Rate (Selected VFM)", "unit": "Mscf/d", "group": "VFM Rates", "axis": "y4", "default_on": False},
        {"key": "choke_opening_pct", "label": "Choke Opening", "unit": "%", "group": "Control & Lift", "axis": "y3", "default_on": True},
        {"key": "gas_lift_rate_mscfd", "label": "Gas Lift Injection", "unit": "Mscf/d", "group": "Control & Lift", "axis": "y4", "default_on": False},
        {"key": "dh_temperature_degf", "label": "Downhole Temperature", "unit": "°F", "group": "Temperature", "axis": "y4", "default_on": False},
    )

    @staticmethod
    def _envelope_downsample(df: pd.DataFrame, max_pts: int, ref_col: str = "dh_pressure_psia") -> pd.DataFrame:
        """Preserve local extrema (spikes/shut-ins) when downsampling large time-series slices."""
        n = len(df)
        if n <= max_pts or max_pts < 4:
            return df
        n_buckets = max(2, (max_pts - 2) // 2)
        edges = np.linspace(0, n, n_buckets + 1, dtype=int)
        vals = df[ref_col].to_numpy(dtype="float64") if ref_col in df.columns else np.zeros(n)
        chosen: List[int] = [0]
        for b in range(n_buckets):
            lo, hi = int(edges[b]), int(edges[b + 1])
            if hi <= lo:
                continue
            seg = vals[lo:hi]
            finite = np.flatnonzero(np.isfinite(seg))
            if finite.size == 0:
                chosen.append(lo)
                continue
            i_min = lo + int(finite[int(np.argmin(seg[finite]))])
            i_max = lo + int(finite[int(np.argmax(seg[finite]))])
            if i_min <= i_max:
                chosen.extend((i_min, i_max))
            else:
                chosen.extend((i_max, i_min))
        chosen.append(n - 1)
        idx = sorted(set(chosen))
        return df.iloc[idx]

    def _monitor_frame(self, well: str, model_name: Optional[str] = None) -> Tuple[pd.DataFrame, str, bool]:
        """Build/return monitoring frame whose VFM rate channels match ONLY the active/selected model."""
        req_model = model_name or self.active_model or "physics_wls"
        if well not in self._monitor_cache:
            h = self.hf(well)
            full = load_frame(self.cache_dir / "wells" / well)
            base_df = pd.DataFrame({"ts": pd.to_datetime(h["timestamp"], utc=True)})
            for col in ("dh_pressure_psia", "whp_psia", "whp_downstream_psia", "dh_temperature_degf",
                        "gas_lift_rate_mscfd", "choke_opening_pct"):
                if col in h.columns:
                    base_df[col] = pd.to_numeric(h[col], errors="coerce").to_numpy(dtype="float64")
            if "annulus_pressure_psia" in full.columns:
                base_df["annulus_pressure_psia"] = pd.to_numeric(full["annulus_pressure_psia"], errors="coerce").to_numpy(dtype="float64")
            if "whp_psia" in base_df and "whp_downstream_psia" in base_df:
                base_df["dp_choke_psi"] = base_df["whp_psia"] - base_df["whp_downstream_psia"]
            if "dh_pressure_psia" in base_df and "whp_psia" in base_df:
                base_df["dp_tubing_psi"] = base_df["dh_pressure_psia"] - base_df["whp_psia"]
            # Cache physics_wls 1-min fallback columns
            mdl = self.coupled_models.get(well, self.base_model)
            pred = mdl.predict(h)
            base_df["_phys_oil"] = pred["oil_rate_stbd"].to_numpy(dtype="float64")
            base_df["_phys_water"] = pred["water_rate_stbd"].to_numpy(dtype="float64")
            base_df["_phys_liquid"] = pred["liquid_rate_stbd"].to_numpy(dtype="float64")
            base_df["_phys_gas"] = pred["gas_rate_mscfd"].to_numpy(dtype="float64")
            base_df = base_df.sort_values("ts").reset_index(drop=True)
            base_df["_t_ms"] = (base_df["ts"].astype("int64") // 1_000_000).astype("int64")
            self._monitor_cache[well] = base_df

        df = self._monitor_cache[well].copy()
        ready = True
        used_model = req_model
        if req_model == "physics_wls":
            df["oil_rate_vfm_stbd"] = df["_phys_oil"]
            df["water_rate_vfm_stbd"] = df["_phys_water"]
            df["liquid_rate_vfm_stbd"] = df["_phys_liquid"]
            df["gas_rate_vfm_mscfd"] = df["_phys_gas"]
        else:
            md = self.model_daily.get(req_model)
            if md is not None:
                mw = md[md["well_name"] == well].copy()
                mw["dkey"] = pd.to_datetime(mw["date"], utc=True).dt.floor("D")
                mw = mw.set_index("dkey")
                dkeys = df["ts"].dt.floor("D")
                si = (df["choke_opening_pct"].to_numpy(dtype="float64") <= 0.5) if "choke_opening_pct" in df else np.zeros(len(df), bool)
                up = np.maximum(dkeys.map(mw["uptime_frac"]).to_numpy(dtype="float64"), 0.05)
                oil_s = np.where(si, 0.0, dkeys.map(mw["oil_rate_alloc_stbd"]).to_numpy(dtype="float64") / up)
                wat_s = np.where(si, 0.0, dkeys.map(mw["water_rate_alloc_stbd"]).to_numpy(dtype="float64") / up)
                gas_s = np.where(si, 0.0, dkeys.map(mw["gas_rate_alloc_mscfd"]).to_numpy(dtype="float64") / up)
                df["oil_rate_vfm_stbd"] = oil_s
                df["water_rate_vfm_stbd"] = wat_s
                df["liquid_rate_vfm_stbd"] = oil_s + wat_s
                df["gas_rate_vfm_mscfd"] = gas_s
            else:
                ready = False
                used_model = "physics_wls"
                df["oil_rate_vfm_stbd"] = df["_phys_oil"]
                df["water_rate_vfm_stbd"] = df["_phys_water"]
                df["liquid_rate_vfm_stbd"] = df["_phys_liquid"]
                df["gas_rate_vfm_mscfd"] = df["_phys_gas"]
        return df, used_model, ready

    def monitoring_payload(
        self,
        well: str,
        res: str = "auto",
        start_ms: Optional[int] = None,
        end_ms: Optional[int] = None,
        max_pts: int = 1800,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        with self.lock:
            if well not in self.wells:
                raise KeyError(f"unknown well {well}")
            if res not in ("auto", "1m", "1h", "1d"):
                raise KeyError(f"unsupported resolution {res}")
            max_pts = int(min(max(max_pts, 200), 6000))
            df, used_model, model_ready = self._monitor_frame(well, model_name=model)
            t_all = df["_t_ms"].to_numpy()
            bounds_ms = [int(t_all[0]), int(t_all[-1])]
            lo = int(start_ms) if start_ms is not None else bounds_ms[0]
            hi = int(end_ms) if end_ms is not None else bounds_ms[1]
            if hi < lo:
                lo, hi = hi, lo
            sub = df[(t_all >= lo) & (t_all <= hi)]
            if sub.empty:
                sub = df

            span_hr = (int(sub["_t_ms"].iloc[-1]) - int(sub["_t_ms"].iloc[0])) / 3_600_000.0
            n_raw = len(sub)
            value_cols = [c["key"] for c in self._MONITOR_CHANNELS if c["key"] in sub.columns]

            if res == "1d" or (res == "auto" and span_hr > 24.0 * 365.0):
                out_df = sub.set_index("ts")[value_cols].resample("1D").mean().dropna(how="all").reset_index()
                out_df["_t_ms"] = (out_df["ts"].astype("int64") // 1_000_000).astype("int64")
                actual_res = "1d"
            elif res == "1h" or (res == "auto" and n_raw > max_pts and span_hr > 24.0 * 10.0):
                out_df = sub.set_index("ts")[value_cols].resample("1h").mean().dropna(how="all").reset_index()
                out_df["_t_ms"] = (out_df["ts"].astype("int64") // 1_000_000).astype("int64")
                if len(out_df) > max_pts:
                    out_df = self._envelope_downsample(out_df, max_pts)
                    actual_res = "1h (env)"
                else:
                    actual_res = "1h"
            else:
                if n_raw > max_pts:
                    out_df = self._envelope_downsample(sub, max_pts)
                    actual_res = "1m (env)"
                else:
                    out_df = sub
                    actual_res = "1m"

            ov_raw = df.set_index("ts")[["dh_pressure_psia", "choke_opening_pct"]].resample("6h").mean().dropna(how="all").reset_index()
            ov_raw["_t_ms"] = (ov_raw["ts"].astype("int64") // 1_000_000).astype("int64")
            if len(ov_raw) > 420:
                ov_raw = self._envelope_downsample(ov_raw, 420)

            shut_ins = []
            for ev in self.events.get(well, []):
                s0 = int(pd.Timestamp(ev["shut_in_start"]).timestamp() * 1000)
                s1 = int(pd.Timestamp(ev["shut_in_end"]).timestamp() * 1000)
                shut_ins.append({
                    "event_id": ev["event_id"],
                    "start_ms": s0,
                    "end_ms": s1,
                    "duration_hr": round(float(ev["duration_hr"]), 2),
                    "status": ev["status"],
                })

            t_df = self._tests(well)
            tests = []
            for r in t_df.itertuples():
                t_mid = r.test_start + (r.test_end - r.test_start) / 2
                tests.append({
                    "t_mid_ms": int(pd.Timestamp(t_mid).timestamp() * 1000),
                    "oil_rate_vfm_stbd": r.oil_rate_stbd,
                    "water_rate_vfm_stbd": r.water_rate_stbd,
                    "liquid_rate_vfm_stbd": r.liquid_rate_stbd,
                    "gas_rate_vfm_mscfd": r.gas_rate_mscfd,
                    "dh_pressure_psia": getattr(r, "dh_pressure_psia", None),
                    "whp_psia": getattr(r, "whp_psia", None),
                })

            def _vec(frame: pd.DataFrame, col: str) -> List[Optional[float]]:
                arr = frame[col].to_numpy(dtype="float64")
                return [None if not np.isfinite(v) else round(float(v), 3) for v in arr]

            return to_jsonable({
                "well_name": well,
                "vfm_model": used_model,
                "vfm_requested_model": model or self.active_model,
                "vfm_model_ready": model_ready,
                "bounds_ms": bounds_ms,
                "view_ms": [int(out_df["_t_ms"].iloc[0]), int(out_df["_t_ms"].iloc[-1])] if len(out_df) else bounds_ms,
                "actual_res": actual_res,
                "n_raw": int(n_raw),
                "n_points": int(len(out_df)),
                "t_ms": [int(v) for v in out_df["_t_ms"].to_numpy()],
                "series": {c: _vec(out_df, c) for c in value_cols},
                "channels": [c for c in self._MONITOR_CHANNELS if c["key"] in value_cols],
                "overview": {
                    "t_ms": [int(v) for v in ov_raw["_t_ms"].to_numpy()],
                    "dh_pressure_psia": _vec(ov_raw, "dh_pressure_psia"),
                    "choke_opening_pct": _vec(ov_raw, "choke_opening_pct"),
                },
                "shut_ins": shut_ins,
                "tests": tests,
            })

    # ------------------------------------------------------------------ score audit ---
    def audit_payload(self, spec: Mapping[str, Any]) -> Dict[str, Any]:
        """Compare uploaded and built-in VFM CSVs and backcalculate candidate grader metrics."""
        from vfm import audit as A

        eval_target = str(spec.get("eval_target") or "tests_all")
        max_score = float(spec.get("max_score") or 30.0)
        cutoff = str(self.cfg.modeling.get("time_split_cutoff", "2025-07-01")) if eval_target == "tests_q3" else None

        if eval_target == "reference" and spec.get("reference"):
            ref_df, ref_audit = A.decode_uploaded_file(spec["reference"])
            # Attach uptime / lift gas metadata from baseline daily when matching dates exist
            dm = self.base_daily.copy()
            dm["date"] = pd.to_datetime(dm["date"], utc=True).dt.strftime("%Y-%m-%d")
            keep = [c for c in ("well_name", "date", "uptime_frac", "daily_gas_lift_mscfd") if c in dm.columns]
            ref_df = ref_df.merge(dm[keep], on=["well_name", "date"], how="left")
            ref_info = {"source": f"uploaded: {ref_audit['filename']}", "n_rows": int(len(ref_df))}
        else:
            ref_df = A.build_test_reference(self.tests, daily_meta=self.base_daily, q3_cutoff=cutoff)
            ref_info = {
                "source": f"well_tests ({'Q3 holdout >= ' + cutoff if cutoff else 'all'})",
                "n_rows": int(len(ref_df)),
            }

        evals: List[Dict[str, Any]] = []
        named_dfs: List[Tuple[str, pd.DataFrame]] = []
        for item in spec.get("files") or []:
            score_raw = item.get("score")
            rep_score = float(score_raw) if (score_raw is not None and str(score_raw).strip() != "") else None
            builtin = item.get("builtin")
            if builtin:
                parts = str(builtin).split(":", 1)
                mname = parts[0]
                rtype = parts[1] if len(parts) > 1 else "alloc"
                sub_df = self.vfm_deliverable_df(mname, rate_type=rtype)
                sub_df["liquid_rate_stbd"] = sub_df["oil_rate_stbd"] + sub_df["water_rate_stbd"]
                fname = str(item.get("name") or f"{mname}_{rtype}.csv")
                audit_info = {
                    "filename": fname,
                    "column_map": {"date": "date", "well_name": "well_name", "oil_rate_stbd": "oil_rate_stbd",
                                   "water_rate_stbd": "water_rate_stbd", "gas_rate_mscfd": "gas_rate_mscfd"},
                    "date_column_exact": True,
                    "n_rows": int(len(sub_df)),
                    "n_wells": int(sub_df["well_name"].nunique()),
                    "date_min": str(sub_df["date"].min()) if len(sub_df) else None,
                    "date_max": str(sub_df["date"].max()) if len(sub_df) else None,
                    "n_bad_dates": 0, "n_duplicate_well_date": 0, "n_nan_rows": 0, "n_negative_rows": 0,
                }
            else:
                sub_df, audit_info = A.decode_uploaded_file(item)
                fname = audit_info["filename"]
            ev = A.evaluate_submission(sub_df, ref_df, audit_info, reported_score=rep_score, daily_meta=self.base_daily)
            evals.append(ev)
            named_dfs.append((fname, sub_df))

        return to_jsonable({
            "reference": ref_info,
            "evals": evals,
            "pairwise": A.pairwise_comparison(named_dfs),
            "backcalc": A.backcalculate_metrics(evals, max_score=max_score),
        })
