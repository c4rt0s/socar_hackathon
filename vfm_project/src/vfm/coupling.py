"""PTA <-> VFM coupling with engineer judgement and guardrails.

Direction PTA <- VFM: the PTA runner takes its rate history from the *uncoupled* VFM daily
allocation (no feedback loop).

Direction PTA -> VFM: accepted build-ups can change the IPR estimator in two ways:

  1. Pressure anchor. The build-up's average reservoir pressure at datum (Dietz p-bar, the
     last shut-in pressure, or an engineer value) replaces the internal build-up anchor of
     that well's Pr(t) = Pr0 - a*t.
  2. Well-condition multiplier. Pseudo-steady-state inflow gives
         J = kh / (141.2 * B * mu * (ln(re/rw) - 0.75 + S)),
     so a change from the reference condition (S_ref, kh_ref) to (S(t), kh(t)) scales J by
         m(t) = [kh(t)/kh_ref] * D(S_ref) / D(S(t)),   D(S) = ln(re/rw) - 0.75 + S.
     S(t) is a step function through the enabled observations (PTA skins and engineer-entered
     changes); the first observation is the reference, so one build-up alone changes nothing.
     m(t) enters the IPR calibration as well as the prediction (vfm.physics.reservoir).

Guardrails, so a bad PTA cannot spoil the allocation:
  * coupling is OFF unless `enabled` is set; each event and each property has its own switch;
  * events are only eligible for skin/kh when the PTA found credible radial flow, and for the
    Dietz pressure only when it exists (otherwise the engineer must pick p_last or a value);
  * pressure anchors must exceed the flowing datum pressure of the surrounding tests;
  * m(t) is clipped to [min_j_multiplier, max_j_multiplier] and D(S) must stay > 1;
  * every decision and guardrail hit is reported back (`explanations`) for the dashboard.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

PRESSURE_CHOICES = ("dietz", "p_last", "manual")


def _utc(ts: Any) -> pd.Timestamp:
    """Engineer-entered times may lack a timezone; treat them as UTC like the data."""
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


@dataclass
class EventDecision:
    """Engineer judgement for one PTA event."""

    use: bool = True
    apply_pressure: bool = True
    pressure_choice: str = "dietz"            # dietz | p_last | manual
    manual_pr_datum_psia: Optional[float] = None
    apply_skin: bool = True
    skin_override: Optional[float] = None
    apply_kh: bool = False
    kh_override: Optional[float] = None
    apply_pi: bool = False                    # fix the IPR productivity index J from this build-up
    pi_choice: str = "actual"                 # actual | theoretical | manual
    pi_override: Optional[float] = None       # STB/d/psi
    note: str = ""


@dataclass
class PropertyChange:
    """Engineer-entered well-condition change (e.g. after a workover or scale build-up)."""

    well_name: str
    timestamp: str
    skin: Optional[float] = None
    kh_md_ft: Optional[float] = None
    enabled: bool = True
    note: str = ""


@dataclass
class CouplingSettings:
    enabled: bool = False
    min_j_multiplier: float = 0.25
    max_j_multiplier: float = 4.0
    decisions: Dict[str, EventDecision] = field(default_factory=dict)     # by event_id
    property_changes: List[PropertyChange] = field(default_factory=list)
    # Estimator switches per well {"ipr": bool, "choke": bool, "vlp": bool}; missing = all on.
    estimators: Dict[str, Dict[str, bool]] = field(default_factory=dict)

    # ---- (de)serialisation for the judgement file -------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "CouplingSettings":
        d = dict(d or {})
        decisions = {k: EventDecision(**{kk: vv for kk, vv in v.items() if kk in EventDecision.__dataclass_fields__})
                     for k, v in (d.get("decisions") or {}).items() if isinstance(v, Mapping)}
        changes = [PropertyChange(**{kk: vv for kk, vv in c.items() if kk in PropertyChange.__dataclass_fields__})
                   for c in (d.get("property_changes") or []) if isinstance(c, Mapping) and c.get("well_name")]
        return cls(
            enabled=bool(d.get("enabled", False)),
            min_j_multiplier=float(d.get("min_j_multiplier", 0.25)),
            max_j_multiplier=float(d.get("max_j_multiplier", 4.0)),
            decisions=decisions,
            property_changes=changes,
            estimators={w: {k: bool(v) for k, v in e.items()} for w, e in (d.get("estimators") or {}).items()
                        if isinstance(e, Mapping)},
        )


RADIAL_STATUSES = ("RADIAL", "RADIAL_FORCED")  # forced = engineer-accepted manual window
MAX_PR_GRADIENT_PSI_FT = 0.8                     # anchors above this datum gradient are rejected


def _finite(v: Any) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def default_decision(event: Mapping[str, Any], coupling_cfg: Mapping[str, Any]) -> EventDecision:
    """Conservative default: use only what the PTA can support. Engineer-forced windows
    (RADIAL_FORCED) are never applied by default; the engineer must switch them on."""
    radial = event.get("status") == "RADIAL"
    has_dietz = (event.get("derived") or {}).get("dietz_pbar_datum_psia") is not None
    return EventDecision(
        use=radial,
        apply_pressure=bool(coupling_cfg.get("apply_pressure_default", True)) and has_dietz,
        pressure_choice="dietz" if has_dietz else "p_last",
        apply_skin=bool(coupling_cfg.get("apply_skin_default", True)) and radial,
        apply_kh=bool(coupling_cfg.get("apply_kh_default", False)) and radial,
    )


def drainage_radius_ft(pvt_row: Mapping[str, Any]) -> Tuple[float, str]:
    area = _finite(pvt_row.get("estimated_drainage_area_acres"))
    if area is None or area <= 0:
        return 1000.0, "no drainage area: re = 1000 ft assumed"
    return math.sqrt(area * 43560.0 / math.pi), f"re from {area:.0f} acres"


def _event_pressure(ev: Mapping[str, Any], dec: EventDecision) -> Tuple[Optional[float], str]:
    d = ev.get("derived") or {}
    if dec.pressure_choice == "manual":
        return dec.manual_pr_datum_psia, "engineer value"
    if dec.pressure_choice == "p_last":
        return d.get("p_last_datum_psia"), "last shut-in pressure (lower bound)"
    return d.get("dietz_pbar_datum_psia"), "Dietz average pressure"


def build_ipr_adjustments(
    events_by_well: Mapping[str, List[Mapping[str, Any]]],
    settings: CouplingSettings,
    pvt: pd.DataFrame,
    tests: pd.DataFrame,
    ref_timestamp: pd.Timestamp,
    flowing_datum_pressure: Optional[Mapping[str, pd.Series]] = None,
) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Turn PTA events + judgement into PhysicsVFM.ipr_adjustments.

    Returns (adjustments, explanations). Wells absent from `adjustments` keep the baseline.
    `flowing_datum_pressure[well]` (indexed by test time) is used to sanity-check anchors.
    """
    adjustments: Dict[str, Dict[str, Any]] = {}
    notes: List[Dict[str, Any]] = []
    if not settings.enabled:
        return adjustments, [{"well_name": "*", "level": "INFO", "message": "Coupling disabled: baseline VFM."}]
    ref_timestamp = _utc(ref_timestamp)
    pvt_by = pvt.drop_duplicates("well_name", keep="first").set_index("well_name")
    wells = sorted(set(events_by_well) | {c.well_name for c in settings.property_changes})

    for well in wells:
        evs = sorted(events_by_well.get(well, []), key=lambda e: _utc(e["shut_in_start"]))
        anchors: List[Dict[str, Any]] = []
        observations: List[Dict[str, Any]] = []   # condition timeline (skin / kh)
        datum_tvd = _finite(pvt_by.loc[well].get("datum_tvd_ft")) if well in pvt_by.index else None
        for ev in evs:
            dec = settings.decisions.get(ev["event_id"])
            if dec is None or not dec.use:
                continue
            R = (ev.get("result") or {}).get("results") or {}
            t_ev = _utc(ev["shut_in_start"])
            note = lambda level, msg: notes.append({"well_name": well, "event_id": ev["event_id"], "level": level, "message": msg})
            if dec.apply_pressure:
                raw, how = _event_pressure(ev, dec)
                pr = _finite(raw)
                ok = pr is not None and pr > 0
                if raw is None:
                    note("WARN", f"No pressure available for choice '{dec.pressure_choice}'; not applied.")
                elif not ok:
                    note("BLOCK", f"Pressure value {raw!r} is not a positive finite number; not applied.")
                if ok and datum_tvd and pr > MAX_PR_GRADIENT_PSI_FT * datum_tvd:
                    ok = False
                    note("BLOCK", f"Pressure anchor {pr:.0f} psia exceeds {MAX_PR_GRADIENT_PSI_FT} psi/ft x datum depth "
                                  f"({MAX_PR_GRADIENT_PSI_FT * datum_tvd:.0f} psia); ignored.")
                if ok:
                    pwf = (flowing_datum_pressure or {}).get(well)
                    near = pd.Series(dtype=float)
                    if pwf is not None and len(pwf):
                        idx = pd.DatetimeIndex(pwf.index)
                        idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
                        pwf = pd.Series(np.asarray(pwf, float), index=idx).dropna()
                        near = pwf[(pwf.index >= t_ev - pd.Timedelta(days=45)) & (pwf.index <= t_ev + pd.Timedelta(days=45))]
                    if len(near) and pr <= float(near.max()) + 5.0:
                        ok = False
                        note("BLOCK", f"Pressure anchor {pr:.0f} psia is not above the flowing datum pressure of "
                                      f"nearby tests ({float(near.max()):.0f}); ignored.")
                    elif not len(near):
                        note("WARN", "No well test within 45 days to sanity-check the pressure anchor.")
                if ok:
                    anchors.append({"timestamp": t_ev, "pr_datum_psia": pr, "source": f"PTA {ev['event_id']}",
                                    "quality_flag": f"PTA_{dec.pressure_choice.upper()}"})
                    note("INFO", f"Pr anchor {pr:.1f} psia at datum ({how}).")
            skin = dec.skin_override if dec.skin_override is not None else R.get("apparent_skin")
            kh = dec.kh_override if dec.kh_override is not None else R.get("kh_md_ft")
            radial = ev.get("status") in RADIAL_STATUSES
            if dec.apply_skin or dec.apply_kh:
                if not radial and dec.skin_override is None and dec.kh_override is None:
                    note("WARN", "No credible radial flow: skin/kh not applied (enter an override to force).")
                else:
                    observations.append({"t": t_ev, "skin": skin if dec.apply_skin else None,
                                         "kh": kh if dec.apply_kh else None, "source": f"PTA {ev['event_id']}"})
        for ch in settings.property_changes:
            if ch.well_name == well and ch.enabled and (ch.skin is not None or ch.kh_md_ft is not None):
                observations.append({"t": _utc(ch.timestamp), "skin": ch.skin, "kh": ch.kh_md_ft,
                                     "source": f"engineer {ch.timestamp} {ch.note}".strip()})

        adj: Dict[str, Any] = {}
        if anchors:
            adj["anchors"] = anchors
        # productivity index link: the latest enabled event with apply_pi fixes J (reference condition)
        pi_pick = None
        for ev in evs:
            dec = settings.decisions.get(ev["event_id"])
            if dec is None or not dec.use or not dec.apply_pi:
                continue
            P = ((ev.get("methods") or {}).get("productivity") or {})
            raw = {"manual": dec.pi_override, "theoretical": P.get("pi_theoretical_stbd_psi")}.get(dec.pi_choice, P.get("pi_actual_stbd_psi"))
            val = _finite(raw)
            if val is None or not (0.05 <= val <= 120.0):
                notes.append({"well_name": well, "event_id": ev["event_id"], "level": "BLOCK",
                              "message": f"PI {raw!r} ({dec.pi_choice}) missing or outside 0.05-120 STB/d/psi; not applied."})
                continue
            pi_pick = (val, _utc(ev["shut_in_start"]), ev["event_id"], dec.pi_choice, P.get("p_avg_source", ""))
        if observations and well not in pvt_by.index:
            notes.append({"well_name": well, "level": "WARN",
                          "message": "Well missing from the PVT table: skin/kh changes cannot be converted to J; ignored."})
        if observations and well in pvt_by.index:
            fn, info = _make_multiplier(observations, pvt_by.loc[well], ref_timestamp,
                                        settings.min_j_multiplier, settings.max_j_multiplier)
            notes.extend({"well_name": well, **n} for n in info)
            if fn is not None:
                adj["j_multiplier"] = fn
                adj["condition_timeline"] = [{"timestamp": o["t"].isoformat(), "skin": o["skin"], "kh_md_ft": o["kh"],
                                              "source": o["source"]} for o in sorted(observations, key=lambda o: o["t"])]
        if pi_pick is not None:
            val, t_ev, eid, choice, src = pi_pick
            m_ev = float(adj["j_multiplier"](np.array([(t_ev - ref_timestamp).total_seconds() / 86400.0]))[0]) \
                if "j_multiplier" in adj else 1.0
            adj["fixed_j"] = val / m_ev
            notes.append({"well_name": well, "event_id": eid, "level": "INFO",
                          "message": f"IPR J fixed to {val:.3f} STB/d/psi from PTA ({choice}; {src})"
                                     + (f", reference J {val / m_ev:.3f} after the condition multiplier." if m_ev != 1.0 else ".")})
        if adj:
            adjustments[well] = adj
    return adjustments, notes


def _make_multiplier(obs: List[Dict[str, Any]], pvt_row: Mapping[str, Any], ref_ts: pd.Timestamp,
                     lo: float, hi: float) -> Tuple[Optional[Callable[[np.ndarray], np.ndarray]], List[Dict[str, Any]]]:
    """Step-function J multiplier from the condition timeline (see module docstring).

    Skin and kh are tracked independently: each property's reference is its own first valid
    observation, and its factor stays 1 until a later observation changes it. Invalid values
    (non-finite, kh <= 0, D(S) <= 0.1) are rejected before they can become a reference.
    Observations at the same time are merged (later entries win per property)."""
    info: List[Dict[str, Any]] = []
    ref_ts = _utc(ref_ts)
    rw = _finite(pvt_row.get("wellbore_radius_ft"))
    re, re_note = drainage_radius_ft(pvt_row)
    if rw is None or rw <= 0 or re <= rw:
        info.append({"level": "BLOCK", "message": f"Invalid wellbore/drainage radius (rw={rw}, re={re:.0f} ft); "
                                                  "skin/kh changes ignored."})
        return None, info
    log_term = math.log(re / rw) - 0.75

    merged: Dict[pd.Timestamp, Dict[str, Any]] = {}
    for o in sorted(obs, key=lambda o: o["t"]):
        m = merged.setdefault(o["t"], {"skin": None, "kh": None, "source": []})
        s, k = o.get("skin"), o.get("kh")
        if s is not None:
            sf = _finite(s)
            if sf is None or log_term + sf <= 0.1:
                info.append({"level": "BLOCK", "message": f"Skin {s!r} from {o['source']} is invalid "
                                                          f"(ln(re/rw)-0.75+S must be > 0.1; {re_note}); ignored."})
            else:
                m["skin"] = sf
        if k is not None:
            kf = _finite(k)
            if kf is None or kf <= 0:
                info.append({"level": "BLOCK", "message": f"kh {k!r} from {o['source']} must be a positive number; ignored."})
            else:
                m["kh"] = kf
        m["source"].append(o["source"])
    steps = [(t, v) for t, v in sorted(merged.items()) if v["skin"] is not None or v["kh"] is not None]

    s_ref = next((v["skin"] for _, v in steps if v["skin"] is not None), None)
    k_ref = next((v["kh"] for _, v in steps if v["kh"] is not None), None)
    cur_s, cur_k = s_ref, k_ref
    times, mults = [], []
    for t, v in steps:
        cur_s = v["skin"] if v["skin"] is not None else cur_s
        cur_k = v["kh"] if v["kh"] is not None else cur_k
        r = 1.0
        if s_ref is not None and cur_s is not None:
            r *= (log_term + s_ref) / (log_term + cur_s)
        if k_ref is not None and cur_k is not None:
            r *= cur_k / k_ref
        rc = float(np.clip(r, lo, hi))
        if rc != r:
            info.append({"level": "WARN", "message": f"J multiplier {r:.2f} clipped to {rc:.2f}."})
        times.append((t - ref_ts).total_seconds() / 86400.0)
        mults.append(rc)
    if len(times) < 2 or all(abs(m - 1.0) < 1e-12 for m in mults):
        info.append({"level": "INFO", "message": f"Reference condition skin={s_ref}, kh={k_ref}; no later change, "
                                                 "J unchanged (needs a later PTA or engineer change)."})
        return None, info
    mults_arr = np.asarray(mults)
    times_arr = np.asarray(times)
    info.append({"level": "INFO", "message": "J multiplier steps: " + ", ".join(
        f"{m:.3f} from day {t:.0f}" for t, m in zip(times, mults)) + f" (reference skin {s_ref}, kh {k_ref}; {re_note})."})

    def j_multiplier(t_days: np.ndarray) -> np.ndarray:
        t = np.asarray(t_days, dtype="float64")
        idx = np.searchsorted(times_arr, t, side="right") - 1
        return np.where(idx < 0, mults_arr[0], mults_arr[np.clip(idx, 0, len(mults_arr) - 1)])

    return j_multiplier, info
