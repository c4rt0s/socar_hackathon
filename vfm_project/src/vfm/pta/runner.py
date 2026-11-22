"""Run the PTA kernel on project data (any well, any dataset that follows the schema).

Inputs per well:
  * 1-minute telemetry with `timestamp`, `dh_pressure_psia` and `choke_opening_pct`
    (shut-ins are detected from the choke, see `reservoir.detect_shutin_buildups`);
  * a rate history: either the VFM daily allocation (default, much denser than tests) or
    the well tests;
  * PVT/completion rows for fluid and rock properties.

Besides the kernel result (kh, skin, p*, QC flags, log-log series) it derives the two
quantities the VFM coupling needs: an average-reservoir-pressure estimate at datum (Dietz
method, only when radial flow is credible) and the last shut-in pressure (a lower bound).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from vfm.physics.reservoir import compute_wellbore_fluid_gradient, detect_shutin_buildups, gauge_to_datum_pressure
from vfm.pta import kernel as K


RATE_SOURCES = ("vfm_daily", "well_tests")
RATE_BASES = ("oil", "total_liquid")


@dataclass
class PtaRunSettings:
    """Per-event analysis settings (engineer-editable)."""

    rate_source: str = "vfm_daily"
    rate_basis: str = "oil"  # 'oil' (single-phase Q_oil) or 'total_liquid' (qo*Bo + qw*Bw)
    prepend_cumulative_history: bool = True
    points_per_cycle: int = 20
    smoothing_L10: float = 0.1
    manual_radial_dt_hr: Optional[Tuple[float, float]] = None
    force_manual_radial: bool = False
    pre_shut_in_context_days: float = 5.0
    dietz_shape_factor: float = 31.62
    # completion-table column holding cumulative oil, and the date that cumulative refers to
    cumulative_oil_column: str = "cum_oil_at_2025_01_01_stb"
    cumulative_oil_date: Optional[str] = "2025-01-01T00:00:00Z"

    @classmethod
    def from_config(cls, pta_cfg: Mapping[str, Any]) -> "PtaRunSettings":
        keys = {f for f in cls.__dataclass_fields__}
        kw = {k: v for k, v in pta_cfg.items() if k in keys}
        return cls(**kw)


def _utc(ts: Any) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def event_id(well: str, start: pd.Timestamp) -> str:
    return f"{well}@{_utc(start).strftime('%Y-%m-%dT%H:%M')}"


def detect_pta_events(hf_well: pd.DataFrame, min_duration_hr: float = 2.0,
                      choke_closed_pct: float = 0.5) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """Shut-in intervals (choke closed) long enough for a build-up."""
    return detect_shutin_buildups(hf_well, min_duration_hr=min_duration_hr, choke_closed_pct=choke_closed_pct)


def _num(row: Mapping[str, Any], key: str, default: Optional[float] = None) -> Optional[float]:
    """Finite float from a row, else `default` (handles missing keys, None and NaN)."""
    try:
        v = row.get(key, default) if hasattr(row, "get") else default
        v = float(v)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


MIN_ONSTREAM_UPTIME = 0.25  # on-stream rate (alloc / uptime) only from days with enough flowing time


def _rate_points_from_daily(daily_w: pd.DataFrame, shut_start: pd.Timestamp, oil_only: bool = False) -> List[K.RatePoint]:
    """Daily allocated rates -> rate points.

    Every whole day before the shut-in day uses its 24-h average (volume-correct, including
    earlier short shut-ins). The shut-in day itself carries the final flowing rate q_N: the
    on-stream rate (alloc / uptime) of the most recent day with uptime >= 25 %, so a shut-in
    shortly after midnight neither zeroes q_N nor blows it up by a tiny uptime."""
    d = daily_w.copy()
    d["date"] = pd.to_datetime(d["date"], utc=True)
    last_day = shut_start.floor("D")
    d = d[d["date"] <= last_day].sort_values("date")
    pts: List[K.RatePoint] = []
    for _, r in d[d["date"] < last_day].iterrows():
        qo, qw = _num(r, "oil_rate_alloc_stbd"), _num(r, "water_rate_alloc_stbd")
        if qo is None or (not oil_only and qw is None):
            continue
        day = pd.Timestamp(r["date"])
        pts.append(K.RatePoint(day.to_pydatetime(), max(qo, 0.0), "stb/day", 0.0 if oil_only else max(qw, 0.0),
                               source_timestamp=day.isoformat()))
    q_n = None
    for _, r in d.iloc[::-1].iterrows():
        up = _num(r, "uptime_frac", 0.0)
        qo, qw = _num(r, "oil_rate_alloc_stbd"), _num(r, "water_rate_alloc_stbd")
        if up is not None and up >= MIN_ONSTREAM_UPTIME and qo is not None and (oil_only or qw is not None):
            q_n = (qo / up, 0.0 if oil_only else (qw / up), pd.Timestamp(r["date"]))
            break
    if q_n is not None:
        note = f"on-stream rate of {q_n[2].date()}"
        if last_day < shut_start:
            pts.append(K.RatePoint(last_day.to_pydatetime(), q_n[0], "stb/day", q_n[1], source_timestamp=note))
        elif pts:  # shut-in exactly at midnight: the previous day carries q_N
            prev = pts[-1]
            pts[-1] = K.RatePoint(prev.timestamp_utc, q_n[0], "stb/day", q_n[1], source_timestamp=note)
    return pts


def _rate_points_from_tests(tests_w: pd.DataFrame, shut_start: pd.Timestamp, oil_only: bool = False) -> List[K.RatePoint]:
    t = tests_w.assign(_ts=pd.to_datetime(tests_w["test_start"], utc=True))
    t = t[t["_ts"] < shut_start].sort_values("_ts")
    pts = []
    for _, r in t.iterrows():
        qo, qw = _num(r, "oil_rate_stbd"), _num(r, "water_rate_stbd", 0.0)
        if qo is None:
            continue
        pts.append(K.RatePoint(r["_ts"].to_pydatetime(), qo, "stb/day", 0.0 if oil_only else qw,
                               source_timestamp=str(r["test_start"])))
    return pts


def _cumulative_prehistory(comp_row: Mapping[str, Any], tests_w: pd.DataFrame, history_start: pd.Timestamp,
                           cum_column: str, cum_date: Optional[str],
                           oil_only: bool = False) -> Tuple[Optional[K.RatePoint], str]:
    """One average-rate point from first production to the date of the cumulative volume.

    Only used when the cumulative is dated and the rate history starts at that date (within
    2 days); otherwise production between the two would be double-counted or missing."""
    fp = comp_row.get("first_production_date")
    cum = _num(comp_row, cum_column)
    if cum is None:
        import re
        for k in (comp_row.keys() if hasattr(comp_row, "keys") else ()):
            if "cum_oil" in str(k).lower():
                cum = _num(comp_row, str(k))
                if cum is not None:
                    m_d = re.search(r"(\d{4})[_-](\d{2})[_-](\d{2})", str(k))
                    if m_d:
                        cum_date = f"{m_d.group(1)}-{m_d.group(2)}-{m_d.group(3)}T00:00:00Z"
                    break
    if fp is None or pd.isna(fp) or cum is None or not cum_date:
        return None, "no first_production_date / dated cumulative: pre-history not added"
    t0, t_cum = _utc(fp), _utc(cum_date)
    if abs((history_start - t_cum).total_seconds()) > 2 * 86400:
        return None, f"rate history starts {history_start.date()}, cumulative is dated {t_cum.date()}: pre-history not added"
    days = (t_cum - t0).total_seconds() / 86400.0
    if days <= 1:
        return None, "first production too close to the cumulative date"
    qo = float(cum) / days
    wor = 0.0
    if not oil_only and not tests_w.empty:
        first = tests_w.assign(_ts=pd.to_datetime(tests_w["test_start"], utc=True)).sort_values("_ts").iloc[0]
        qo_t, qw_t = _num(first, "oil_rate_stbd", 0.0), _num(first, "water_rate_stbd", 0.0)
        wor = qw_t / qo_t if qo_t > 1.0 else 0.0
    return (K.RatePoint(t0.to_pydatetime(), qo, "stb/day", 0.0 if oil_only else qo * wor, source_timestamp="cumulative_average"),
            f"cumulative-average pre-history {qo:.0f} STB/d oil from {t0.date()}")


def physical_inputs(pvt_row: Mapping[str, Any], comp_row: Mapping[str, Any]) -> K.PhysicalInputs:
    """Missing/NaN values stay None: the kernel then blocks only the outputs that need them."""
    src = "reservoir_pvt / well_completion tables"
    return K.PhysicalInputs(
        oil_fvf=_num(pvt_row, "oil_fvf_rb_stb"), water_fvf=_num(comp_row, "water_fvf_rb_stb", 1.02),
        viscosity_cp=_num(pvt_row, "oil_viscosity_cp"),
        total_compressibility_per_psi=_num(pvt_row, "total_compressibility_1_psi"),
        porosity_fraction=_num(pvt_row, "porosity_frac"),
        contributing_thickness_ft=_num(pvt_row, "net_pay_thickness_ft"),
        wellbore_radius_ft=_num(pvt_row, "wellbore_radius_ft"),
        provenance={k: src for k in ("oil_fvf", "water_fvf", "viscosity_cp", "total_compressibility_per_psi",
                                     "porosity_fraction", "contributing_thickness_ft", "wellbore_radius_ft")},
    )


def _datum_gradient(pvt_row, comp_row, tests_w: pd.DataFrame, shut_start: pd.Timestamp) -> Optional[float]:
    """Liquid gradient (psi/ft) for gauge -> datum, from the last test before the shut-in.

    Same convention as the project's internal build-up anchors (flowing-test mixture with
    dissolved gas). A segregated static column could differ by ~0.05 psi/ft; with the gauge a
    few hundred feet above datum that is ~10 psi. Returns None if no usable test exists."""
    if tests_w is None or tests_w.empty:
        return None
    t = tests_w.assign(_ts=pd.to_datetime(tests_w["test_start"], utc=True)).sort_values("_ts")
    t = t.dropna(subset=[c for c in ("oil_gravity_api", "water_cut_pct", "gas_sg", "water_salinity_ppm") if c in t])
    if t.empty:
        return None
    prior = t[t["_ts"] < shut_start]
    r = prior.iloc[-1] if len(prior) else t.iloc[0]
    g = compute_wellbore_fluid_gradient(
        api=_num(r, "oil_gravity_api", 32.0), wc_frac=_num(r, "water_cut_pct", 0.0) / 100.0,
        gor_scf_stb=_num(comp_row, "solution_gor_at_pb_scf_stb", 0.0), gas_sg=_num(r, "gas_sg", 0.75),
        bo=_num(pvt_row, "oil_fvf_rb_stb", 1.2), bw=_num(comp_row, "water_fvf_rb_stb", 1.02),
        salinity_ppm=_num(r, "water_salinity_ppm", 0.0))
    g = float(g)
    return g if math.isfinite(g) and g > 0 else None


def _dietz_average_pressure(res: Dict[str, Any], pvt_row: Mapping[str, Any], shape_factor: float) -> Dict[str, Any]:
    """p-bar = p_ws on the radial line at dt_bar = phi*mu*ct*A / (0.0002637 k C_A) (Dietz).
    Requires accepted radial flow, k and a drainage area; assumes pseudo-steady state was
    reached before the shut-in (long producing history)."""
    R = res["results"]
    out = {"dietz_pbar_gauge_psia": None, "dietz_dt_bar_hr": None, "dietz_tpda": None, "dietz_note": None}
    k, d_r, p_star = R.get("k_md"), R.get("radial_derivative_psi"), R.get("extrapolated_pressure_psi")
    if not all(v is not None and math.isfinite(v) for v in (k, d_r, p_star)) or k <= 0 or d_r <= 0:
        out["dietz_note"] = "no credible radial flow"
        return out
    area_acres = _num(pvt_row, "estimated_drainage_area_acres")
    phi, mu, ct = _num(pvt_row, "porosity_frac"), _num(pvt_row, "oil_viscosity_cp"), _num(pvt_row, "total_compressibility_1_psi")
    if not area_acres or area_acres <= 0 or not phi or not mu or not ct or shape_factor <= 0:
        out["dietz_note"] = "missing drainage area or rock/fluid properties"
        return out
    rs = res["inputs"]["rate_source"]
    periods = [K.RatePeriod(**p) for p in rs.get("periods", [])]
    tp = _num(rs, "producing_duration_hr")
    if not periods or not tp:
        out["dietz_note"] = "no rate history"
        return out
    area_ft2 = area_acres * 43560.0
    # Dietz needs pseudo-steady state before the shut-in: t_pDA = 0.0002637 k tp / (phi mu ct A) >= 0.1
    tpda = 0.0002637 * k * tp / (phi * mu * ct * area_ft2)
    out["dietz_tpda"] = tpda
    if tpda < 0.1:
        out["dietz_note"] = f"pseudo-steady state not reached before shut-in (t_pDA={tpda:.3f} < 0.1)"
        return out
    dt_bar = phi * mu * ct * area_ft2 / (0.0002637 * k * shape_factor)
    x_bar, _, _, _ = K.compute_superposition([dt_bar], periods, tp)
    pbar = float(p_star) + float(d_r) * float(x_bar[0]) if x_bar else float("nan")
    if not math.isfinite(pbar):
        out["dietz_note"] = "superposition failed at dt_bar"
        return out
    out.update(dietz_pbar_gauge_psia=pbar, dietz_dt_bar_hr=dt_bar,
               dietz_note=f"C_A={shape_factor}, A={area_acres:.0f} acres, t_pDA={tpda:.1f}")
    return out


def run_pta_event(
    well: str,
    shut_start: pd.Timestamp,
    shut_end: pd.Timestamp,
    hf_well: pd.DataFrame,
    tests_w: pd.DataFrame,
    pvt_row: Mapping[str, Any],
    comp_row: Mapping[str, Any],
    settings: PtaRunSettings,
    daily_w: Optional[pd.DataFrame] = None,
    rates_override: Optional[List[K.RatePoint]] = None,
) -> Dict[str, Any]:
    """Analyse one build-up and return {kernel result + derived pressures + metadata}.

    `rates_override` replaces the rate history (e.g. an engineer's single equivalent rate for tp hours)."""
    shut_start, shut_end = _utc(shut_start), _utc(shut_end)
    lo = shut_start - pd.Timedelta(days=settings.pre_shut_in_context_days)
    ts = pd.to_datetime(hf_well["timestamp"], utc=True)
    win = hf_well.loc[(ts >= lo) & (ts <= shut_end), ["timestamp", "dh_pressure_psia"]].dropna()
    win = win.assign(_ts=pd.to_datetime(win["timestamp"], utc=True)).sort_values("_ts")
    pressure = [K.PressurePoint(t.to_pydatetime(), float(p), "psia") for t, p in zip(win["_ts"], win["dh_pressure_psia"])]

    oil_only = settings.rate_basis == "oil"
    if rates_override:
        rates = list(rates_override)
        rate_note = "engineer-specified rate history"
    elif settings.rate_source == "vfm_daily" and daily_w is not None and len(daily_w):
        rates = _rate_points_from_daily(daily_w, shut_start, oil_only=oil_only)
        rate_note = "VFM daily allocation (24-h averages; on-stream rate carried into the shut-in day)"
    else:
        rates = _rate_points_from_tests(tests_w, shut_start, oil_only=oil_only)
        rate_note = "well tests (each test rate held until the next test)"
    pre = None
    if settings.prepend_cumulative_history and rates and not rates_override:
        pre, pre_note = _cumulative_prehistory(comp_row, tests_w, _utc(rates[0].timestamp_utc),
                                               settings.cumulative_oil_column, settings.cumulative_oil_date,
                                               oil_only=oil_only)
        rate_note += f"; {pre_note}"
        if pre is not None:
            rates = [pre] + rates
    rate_note += f" [rate basis: {'Q_oil (single-phase oil)' if oil_only else 'Q_total (qo*Bo + qw*Bw)'}]"

    controls = K.AnalysisControls(
        points_per_cycle=int(settings.points_per_cycle), smoothing_L10=float(settings.smoothing_L10),
        manual_radial_dt_hr=tuple(settings.manual_radial_dt_hr) if settings.manual_radial_dt_hr else None,
        manual_radial_force=bool(settings.force_manual_radial and settings.manual_radial_dt_hr),
        history_complete=pre is not None or bool(rates_override),
    )
    res = K.analyze_buildup(pressure, rates, [K.ShutIn(shut_start.to_pydatetime(), shut_end.to_pydatetime())],
                            physical_inputs(pvt_row, comp_row), controls)

    grad = _datum_gradient(pvt_row, comp_row, tests_w, shut_start)
    gauge_tvd, datum_tvd = _num(comp_row, "gauge_tvd_ft"), _num(pvt_row, "datum_tvd_ft")
    can_datum = grad is not None and gauge_tvd is not None and datum_tvd is not None
    to_datum = lambda p: None if (p is None or not can_datum) else float(gauge_to_datum_pressure(p, gauge_tvd, datum_tvd, grad))
    si = res["inputs"]["shut_in_selection"]
    ser = (res.get("diagnostics") or {}).get("series") or {}
    p_last = None
    if ser.get("dp_psi") and si.get("p_wf_psi") is not None:
        p_last = float(si["p_wf_psi"]) + float(ser["dp_psi"][-1])
    dietz = _dietz_average_pressure(res, pvt_row, settings.dietz_shape_factor)
    methods_out = _cross_methods(res, rates, pvt_row, comp_row, dietz, p_last=p_last, settings=settings)
    radial = (res["diagnostics"] or {}).get("radial_interval") or {}
    radial_ok = bool(radial.get("accepted")) and res["results"].get("kh_md_ft") is not None
    blocks = [f["code"] for f in res["qc_flags"] if f["severity"] == "BLOCK"]
    if radial_ok:
        # an engineer-forced window is labelled separately so it is never applied by default
        status = "RADIAL_FORCED" if radial.get("selection") == "manual_forced" else "RADIAL"
    else:
        status = "FAILED" if any(c in blocks for c in ("NO_SHUT_IN", "INSUFFICIENT_BUILDUP_DATA", "NO_VALID_PRESSURE")) else "NO_RADIAL"
    return {
        "event_id": event_id(well, shut_start),
        "well_name": well,
        "shut_in_start": shut_start.isoformat(),
        "shut_in_end": shut_end.isoformat(),
        "duration_hr": (shut_end - shut_start).total_seconds() / 3600.0,
        "status": status,
        "settings": {**asdict(settings), "manual_radial_dt_hr": list(settings.manual_radial_dt_hr) if settings.manual_radial_dt_hr else None},
        "rate_note": rate_note,
        "datum": {"gauge_tvd_ft": gauge_tvd, "datum_tvd_ft": datum_tvd, "fluid_gradient_psi_ft": grad},
        "derived": {
            **dietz,
            "dietz_pbar_datum_psia": to_datum(dietz["dietz_pbar_gauge_psia"]),
            "p_last_gauge_psia": p_last,
            "p_last_datum_psia": to_datum(p_last),
            "p_wf_gauge_psia": si.get("p_wf_psi"),
        },
        "methods": methods_out,
        "result": res,
    }


def _adaptive_horner_window(dt: np.ndarray, pws: np.ndarray, tp_hr: float,
                            min_cycles: float = 0.6, n_per_decade: int = 12
                            ) -> Tuple[Optional[Tuple[float, float]], Optional[float]]:
    """Find the earliest radial straight-line plateau on smoothed Horner local slopes.

    Used as the indicative fallback for Horner/MDH when the strict Bourdet-derivative check
    finds no radial interval, avoiding late-time equilibrium flattening distortion.
    """
    valid = np.isfinite(dt) & np.isfinite(pws) & (dt > 0)
    if valid.sum() < 12 or tp_hr <= 0:
        return None, None
    dt_v, p_v = dt[valid], pws[valid]
    edges = 10.0 ** np.arange(np.floor(np.log10(dt_v.min())), np.log10(dt_v.max()) + 1e-9, 1.0 / n_per_decade)
    idx = np.digitize(dt_v, edges)
    t_bins, p_bins = [], []
    for b in np.unique(idx):
        s = idx == b
        t_bins.append(float(np.exp(np.mean(np.log(dt_v[s])))))
        p_bins.append(float(np.median(p_v[s])))
    t = np.asarray(t_bins, dtype="float64")
    p = np.asarray(p_bins, dtype="float64")
    if len(t) < 8:
        return None, None

    x = np.log10((tp_hr + t) / t)
    half = 3
    slopes = np.full(len(t), np.nan, dtype="float64")
    for i in range(half, len(t) - half):
        xw = x[i - half:i + half + 1]
        pw = p[i - half:i + half + 1]
        if np.ptp(xw) > 1e-9:
            slopes[i] = -float(np.polyfit(xw, pw, 1)[0])

    ok = np.isfinite(slopes) & (t >= 0.25)
    if not ok.any():
        return None, None
    early = ok & (t <= 0.5 * t[-1])
    if early.any():
        early_idxs = np.where(early)[0]
        peak_local = int(np.nanargmax(slopes[early]))
        peak_val = float(slopes[early][peak_local])
        med_val = float(np.nanmedian(slopes[early]))
        if peak_val > 1.15 * med_val:
            ok &= (t >= t[int(early_idxs[peak_local])])
    cand = np.where(ok)[0]
    for tol in (0.12, 0.20, 0.30):
        for i in cand:
            best_j = None
            for j in cand[cand >= i]:
                seg = slopes[i:j + 1]
                if not np.all(np.isfinite(seg)):
                    break
                med = float(np.median(seg))
                if med <= 0 or np.max(np.abs(seg - med)) > tol * med:
                    break
                if np.log10(t[j] / t[i]) >= min_cycles:
                    best_j = j
            if best_j is not None:
                return (float(t[i]), float(t[best_j])), tol
    return None, None


def _cross_methods(res: Dict[str, Any], rates: List[K.RatePoint], pvt_row, comp_row, dietz: Dict[str, Any],
                   p_last: Optional[float], settings: PtaRunSettings) -> Dict[str, Any]:
    """Independent methods on the same build-up: Horner, MDH, nonlinear model match, and
    productivity indices. Derivative-method results stay the primary interpretation."""
    from vfm.pta import methods as M

    out: Dict[str, Any] = {"horner": None, "mdh": None, "model_match": None, "productivity": None, "notes": []}
    ser = (res.get("diagnostics") or {}).get("series") or {}
    si = res["inputs"]["shut_in_selection"]
    rs = res["inputs"]["rate_source"]
    R = res["results"]
    if not ser.get("dt_hr") or si.get("p_wf_psi") is None:
        out["notes"].append("no build-up series")
        return out
    dt = np.asarray(ser["dt_hr"], dtype="float64")
    dp = np.asarray(ser["dp_psi"], dtype="float64")
    pwf = float(si["p_wf_psi"])
    q_rb = _num(rs, "q_final_reservoir_bbl_day")
    tp = _num(R, "equivalent_producing_time_hr") or _num(rs, "producing_duration_hr")
    mu, h, phi = _num(pvt_row, "oil_viscosity_cp"), _num(pvt_row, "net_pay_thickness_ft"), _num(pvt_row, "porosity_frac")
    ct, rw = _num(pvt_row, "total_compressibility_1_psi"), _num(pvt_row, "wellbore_radius_ft")
    if None in (q_rb, tp, mu, h, phi, ct, rw):
        out["notes"].append("missing rate or rock/fluid properties for cross-methods")
        return out
    radial = (res["diagnostics"] or {}).get("radial_interval") or {}
    if radial.get("accepted"):
        win = (float(radial["dt_start_hr"]), float(radial["dt_end_hr"]))
        out["window_source"] = "accepted radial window"
    else:
        win_adapt, adapt_tol = _adaptive_horner_window(dt, pwf + dp, tp)
        if win_adapt is not None:
            win = win_adapt
            out["window_source"] = f"adaptive Horner plateau (tol={adapt_tol:.0%}; no strict derivative window)"
        else:
            win = (float(dt[-1]) / 10 ** 0.5, float(dt[-1]))  # last half cycle: indicative only
            out["window_source"] = "last half log-cycle (no accepted radial window: indicative only)"
    out["horner"] = M.horner_analysis(dt, pwf + dp, tp, win, q_rb, mu, h, phi, ct, rw, pwf)
    out["mdh"] = M.mdh_analysis(dt, pwf + dp, win, tp, q_rb, mu, h, phi, ct, rw, pwf)
    k0 = _num(R, "k_md") or (out["horner"] or {}).get("k") or 50.0
    try:
        mm = M.model_match(dt, dp, tp, q_rb, mu, h, phi, ct, rw, k0_md=float(k0), skin0=float(_num(R, "apparent_skin", 0.0)),
                           c0_bbl_psi=float(_num(R, "storage_bbl_per_psi", 0.01) or 0.01))
        if not radial.get("accepted"):
            mm.setdefault("notes", []).append("No accepted radial flow: the infinite-acting model ignores boundaries; treat as indicative.")
        out["model_match"] = mm
    except Exception as exc:  # never let an optional method break the analysis
        out["notes"].append(f"model match failed: {type(exc).__name__}: {exc}")
    # productivity: surface liquid rate of the last rate point, average pressure Dietz > last shut-in
    q_stb = None
    if rates:
        last = rates[-1]
        q_stb = float(last.oil_surface_rate or 0.0) + float(last.water_surface_rate or 0.0)
    p_avg, src = dietz.get("dietz_pbar_gauge_psia"), "Dietz average pressure"
    if p_avg is None and p_last is not None:
        p_avg, src = p_last, "last shut-in pressure (lower bound: PI is an upper bound)"
    area = _num(pvt_row, "estimated_drainage_area_acres")
    re = math.sqrt(area * 43560.0 / math.pi) if area else None
    b_liq = q_rb / q_stb if q_stb and q_stb > 0 else None
    if q_stb and p_avg is not None:
        out["productivity"] = M.productivity_indices(
            q_stb, q_rb, float(p_avg), pwf, kh_md_ft=_num(R, "kh_md_ft"), skin=_num(R, "apparent_skin"), mu_cp=mu,
            b_liquid=b_liq, re_ft=re, rw_ft=rw, dp_skin_psi=_num(R, "skin_pressure_drop_psi"), p_avg_source=src)
    else:
        out["notes"].append("productivity index needs a final rate and an average-pressure estimate")
    return out
