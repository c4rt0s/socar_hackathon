#!/usr/bin/env python3
"""
vfm.pta.kernel
==============

Preliminary pressure build-up (PBU) analysis kernel. Standard library only.
This is the single source of the PTA math; `Media/06_Pressure_Buildup_Analysis.py` is a
thin command-line wrapper around it, and `vfm.pta.runner` adapts project data to it.

Implements the "Pressure Build-Up Analysis: Mathematical Handover" workflow:

    clean pressure -> define shut-in -> reservoir-rate history -> superposition X
    -> log resampling -> Bourdet derivative -> radial window -> kh, k, skin, C, CD,
    p*, r_inv, FE -> QC flags with dependency blocking -> JSON result

Scope: vertical well, single-phase (or explicitly assumed single-phase-equivalent),
homogeneous, infinite-acting radial-flow interpretation. Results are preliminary and
non-unique; every result is labelled DRAFT_REQUIRES_TECHNICAL_REVIEW.

Documented numerical choices (the spec requires these to be stated):
  * Internal units: psi, hr, reservoir bbl/day, cp, mD, ft, 1/psi.
  * The Bourdet derivative is taken with respect to the generalized superposition
    coordinate X, and the smoothing window L is measured in X (natural-log units,
    L_ln = L10 * ln 10). Without a usable rate history it falls back to ln(dt).
  * Radial-window extent, sampling gaps and flow-regime slopes use elapsed log10(dt).
  * Radius of investigation uses elapsed shut-in time dt and the coefficient 948.
  * The reference point is the last pressure observation at or before shut-in start.

Usage:
    python 06_Pressure_Buildup_Analysis.py --self-test
    python 06_Pressure_Buildup_Analysis.py --demo storage_skin
    python 06_Pressure_Buildup_Analysis.py --demo multirate --json-out result.json
    python 06_Pressure_Buildup_Analysis.py --input my_test.json --json-out result.json
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import random
import statistics
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from fractions import Fraction
from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

SCHEMA_VERSION = "1.0"
STATUS = "DRAFT_REQUIRES_TECHNICAL_REVIEW"

LN10 = math.log(10.0)
LN2 = math.log(2.0)
EULER_GAMMA = 0.5772156649015329
RADIAL_CONST = math.log(4.0) - EULER_GAMMA  # 0.80907...

# ==============================================================================
# Unit conversions
# ==============================================================================

PSI_PER_BAR = 14.5037738
PSI_PER_KPA = 0.145037738
FT_PER_M = 3.2808399
BBL_PER_M3 = 6.2898108


def bar_to_psi(p_bar: float) -> float:
    return p_bar * PSI_PER_BAR


def m_to_ft(h_m: float) -> float:
    return h_m * FT_PER_M


def m3_to_bbl(q_m3: float) -> float:
    return q_m3 * BBL_PER_M3


def c_to_f(t_c: float) -> float:
    return t_c * 9.0 / 5.0 + 32.0


def radius_ft_from_diameter_in(d_in: float) -> float:
    """Diameter in inches -> radius in feet (halve first, then convert)."""
    return d_in / 2.0 / 12.0


def radius_ft_from_radius_in(r_in: float) -> float:
    return r_in / 12.0


# unit -> (factor to psi, pressure basis)
_PRESSURE_UNITS = {
    "psi": (1.0, "unspecified"), "psia": (1.0, "absolute"), "psig": (1.0, "gauge"),
    "bar": (PSI_PER_BAR, "unspecified"), "bara": (PSI_PER_BAR, "absolute"),
    "barg": (PSI_PER_BAR, "gauge"),
    "kpa": (PSI_PER_KPA, "unspecified"), "kpaa": (PSI_PER_KPA, "absolute"),
    "kpag": (PSI_PER_KPA, "gauge"),
}
# surface-rate unit -> factor to STB/day
_RATE_UNITS = {
    "stb/day": 1.0, "stb/d": 1.0, "bbl/day": 1.0, "bbl/d": 1.0, "bopd": 1.0,
    "m3/day": BBL_PER_M3, "m3/d": BBL_PER_M3, "sm3/day": BBL_PER_M3, "sm3/d": BBL_PER_M3,
}
_BAD_FLAGS = {"bad", "invalid", "reject", "rejected", "error"}
# Explicit null markers used by logging/historian exports (LAS uses -999.25). These are
# compared against the raw value before unit conversion.
DEFAULT_NULL_SENTINELS = (-999.25, -999.0, -9999.0, -99999.0)

# ==============================================================================
# QC flags and dependency blocking
# ==============================================================================

RESULT_KEYS = [
    "radial_derivative_psi", "semilog_slope_psi_per_cycle", "mobility_thickness",
    "kh_md_ft", "k_md", "apparent_skin", "skin_pressure_drop_psi",
    "storage_bbl_per_psi", "dimensionless_storage", "extrapolated_pressure_psi",
    "investigation_radius_ft", "flow_efficiency", "kh_ratio_pta_to_reference",
    "equivalent_producing_time_hr",
]
RADIAL_OUTPUTS = [
    "radial_derivative_psi", "semilog_slope_psi_per_cycle", "mobility_thickness",
    "kh_md_ft", "k_md", "apparent_skin", "skin_pressure_drop_psi",
    "extrapolated_pressure_psi", "investigation_radius_ft", "flow_efficiency",
    "kh_ratio_pta_to_reference",
]
RATE_OUTPUTS = [
    "mobility_thickness", "kh_md_ft", "k_md", "apparent_skin", "skin_pressure_drop_psi",
    "storage_bbl_per_psi", "dimensionless_storage", "extrapolated_pressure_psi",
    "investigation_radius_ft", "flow_efficiency", "kh_ratio_pta_to_reference",
    "equivalent_producing_time_hr", "radial_derivative_psi", "semilog_slope_psi_per_cycle",
]
ALL = ["*"]


@dataclass
class QCFlag:
    severity: str  # INFO | WARN | BLOCK
    code: str
    message: str
    affected_outputs: List[str] = field(default_factory=list)


class QC:
    def __init__(self) -> None:
        self.flags: List[QCFlag] = []

    def add(self, severity: str, code: str, message: str,
            affected: Sequence[str] = ()) -> None:
        self.flags.append(QCFlag(severity, code, message, list(affected)))

    def extend(self, other: "QC") -> None:
        self.flags.extend(other.flags)

    def blocked(self, output: str) -> bool:
        return any(f.severity == "BLOCK" and ("*" in f.affected_outputs
                                              or output in f.affected_outputs)
                   for f in self.flags)

    def has(self, code: str) -> bool:
        return any(f.code == code for f in self.flags)


# ==============================================================================
# Input contracts
# ==============================================================================

Timestamp = Union[str, datetime]


@dataclass
class PressurePoint:
    timestamp_utc: Timestamp
    pressure: float
    pressure_unit: str
    quality_flag: Optional[str] = None


@dataclass
class RatePoint:
    timestamp_utc: Timestamp
    oil_surface_rate: Optional[float]
    rate_unit: str
    water_surface_rate: Optional[float] = None
    source_timestamp: Optional[str] = None
    quality_flag: Optional[str] = None


@dataclass
class ShutIn:
    start_timestamp: Timestamp
    end_timestamp: Optional[Timestamp] = None
    ongoing: bool = False


@dataclass
class PhysicalInputs:
    oil_fvf: Optional[float] = None
    water_fvf: Optional[float] = None
    viscosity_cp: Optional[float] = None
    total_compressibility_per_psi: Optional[float] = None
    porosity_fraction: Optional[float] = None
    contributing_thickness_ft: Optional[float] = None
    wellbore_radius_ft: Optional[float] = None
    log_kh_md_ft: Optional[float] = None
    provenance: Dict[str, str] = field(default_factory=dict)


@dataclass
class RatePeriod:
    start_time_hr: float
    end_time_hr: float
    reservoir_rate_bbl_day: float
    original_rate_source_timestamp: str


@dataclass
class AnalysisControls:
    duplicate_policy: str = "mean"          # mean | first | last | reject
    null_sentinels: Tuple[float, ...] = DEFAULT_NULL_SENTINELS
    spike_threshold_psi: Optional[float] = None  # None disables spike flagging
    frozen_min_points: int = 20
    merge_policy: str = "none"              # none | merge
    max_merge_gap_hr: float = 1.0
    merged_shutin_verified: bool = False
    reference_gap_tolerance_hr: float = 0.25
    rate_rel_tolerance: float = 0.02        # epsilon_q
    rate_floor_bbl_day: float = 1.0         # q_floor
    history_complete: bool = False          # True only if rates cover all production
    points_per_cycle: int = 20
    smoothing_L10: float = 0.1              # Bourdet L, in log10 cycles
    min_coordinate_step: float = 1e-9
    max_gap_cycles: float = 0.5
    radial_min_points: int = 6
    radial_min_cycles: float = 0.5
    radial_max_sigma: float = 0.02          # std dev of log10(derivative)
    radial_max_trend: float = 0.05          # |d log10(deriv) / d log10(dt)|
    radial_lambda: float = 0.01
    # A locally flat storage-hump top can mimic a radial plateau. When the derivative
    # maximum is followed by a fall below hump_drop_ratio * max, automatic windows must
    # start after the maximum (heuristic; spec sec. 9 notes it is not universally reliable).
    exclude_storage_hump: bool = True
    hump_drop_ratio: float = 0.8
    manual_radial_dt_hr: Optional[Tuple[float, float]] = None
    # Engineer override: accept a manual window that fails the checks (reported as WARN).
    manual_radial_force: bool = False
    storage_slope_tolerance: float = 0.1
    storage_min_points: int = 3


# ==============================================================================
# Small helpers
# ==============================================================================

def _finite(v: Any) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _positive(v: Any) -> Optional[float]:
    f = _finite(v)
    return f if f is not None and f > 0 else None


def parse_utc(ts: Timestamp) -> Tuple[datetime, bool]:
    """Return (aware UTC datetime, timezone_was_assumed)."""
    if isinstance(ts, datetime):
        dt = ts
    else:
        s = str(ts).strip()
        if s[-1:] in ("Z", "z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc), True
    return dt.astimezone(timezone.utc), False


def _iso(dt: Optional[datetime]) -> Optional[str]:
    if dt is None:
        return None
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _hours(td: timedelta) -> float:
    return td.total_seconds() / 3600.0


def _median(xs: Sequence[float]) -> float:
    return statistics.median(xs)


def _linfit(x: Sequence[float], y: Sequence[float]) -> Dict[str, float]:
    n = len(x)
    xb = sum(x) / n
    yb = sum(y) / n
    sxx = sum((xi - xb) ** 2 for xi in x)
    slope = sum((xi - xb) * (yi - yb) for xi, yi in zip(x, y)) / sxx
    icpt = yb - slope * xb
    resid = [yi - (icpt + slope * xi) for xi, yi in zip(x, y)]
    s2 = sum(r * r for r in resid) / (n - 2) if n > 2 else float("nan")
    se_slope = math.sqrt(s2 / sxx) if n > 2 else float("nan")
    se_icpt = math.sqrt(s2 * (1.0 / n + xb * xb / sxx)) if n > 2 else float("nan")
    return {"slope": slope, "intercept": icpt, "se_slope": se_slope, "se_intercept": se_icpt}


def sanitize(obj: Any) -> Any:
    """Replace NaN/inf with None recursively so JSON never carries them as values."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize(v) for v in obj]
    return obj


# ==============================================================================
# 4.1 Cleaning
# ==============================================================================

def clean_pressure(points: Sequence[PressurePoint], duplicate_policy: str = "mean",
                   spike_threshold_psi: Optional[float] = None,
                   frozen_min_points: int = 20,
                   null_sentinels: Sequence[float] = DEFAULT_NULL_SENTINELS
                   ) -> Tuple[List[Tuple[datetime, float]], Dict[str, Any], QC]:
    qc = QC()
    discarded: List[Dict[str, Any]] = []
    rows: List[Tuple[datetime, float, int]] = []
    bases = set()
    assumed_tz = 0
    unknown_units = 0
    n_null = 0
    n_nonpos_abs = 0
    for idx, pt in enumerate(points):
        try:
            t, assumed = parse_utc(pt.timestamp_utc)
        except (ValueError, TypeError):
            discarded.append({"index": idx, "reason": "UNPARSEABLE_TIMESTAMP"})
            continue
        assumed_tz += assumed
        unit = (pt.pressure_unit or "").strip().lower()
        if unit not in _PRESSURE_UNITS:
            unknown_units += 1
            discarded.append({"index": idx, "reason": "UNKNOWN_PRESSURE_UNIT",
                              "unit": pt.pressure_unit})
            continue
        if pt.quality_flag and pt.quality_flag.strip().lower() in _BAD_FLAGS:
            discarded.append({"index": idx, "reason": "FLAGGED_BAD"})
            continue
        p = _finite(pt.pressure)
        if p is None:
            discarded.append({"index": idx, "reason": "NON_FINITE_PRESSURE"})
            continue
        if any(p == s for s in null_sentinels):
            n_null += 1
            discarded.append({"index": idx, "reason": "NULL_SENTINEL", "value": p,
                              "timestamp": _iso(t)})
            continue
        factor, basis = _PRESSURE_UNITS[unit]
        if basis == "absolute" and p <= 0:
            n_nonpos_abs += 1
            discarded.append({"index": idx, "reason": "NONPOSITIVE_ABSOLUTE_PRESSURE",
                              "value": p, "timestamp": _iso(t)})
            continue
        bases.add(basis)
        rows.append((t, p * factor, idx))
    rows.sort(key=lambda r: (r[0], r[2]))

    cleaned: List[Tuple[datetime, float]] = []
    dup_groups = 0
    i = 0
    while i < len(rows):
        j = i
        while j + 1 < len(rows) and rows[j + 1][0] == rows[i][0]:
            j += 1
        group = rows[i:j + 1]
        if len(group) == 1:
            cleaned.append((group[0][0], group[0][1]))
        else:
            dup_groups += 1
            vals = [g[1] for g in group]
            if duplicate_policy == "first":
                keep: Optional[float] = vals[0]
            elif duplicate_policy == "last":
                keep = vals[-1]
            elif duplicate_policy == "mean":
                keep = sum(vals) / len(vals)
            elif duplicate_policy == "reject":
                keep = None
            else:
                raise ValueError(f"unknown duplicate_policy {duplicate_policy!r}")
            discarded.append({"indices": [g[2] for g in group],
                              "reason": f"DUPLICATE_TIMESTAMP_{duplicate_policy.upper()}",
                              "timestamp": _iso(group[0][0]), "values_psi": vals})
            if keep is not None:
                cleaned.append((group[0][0], keep))
        i = j + 1

    spikes: List[str] = []
    if spike_threshold_psi is not None and len(cleaned) >= 5:
        for m in range(2, len(cleaned) - 2):
            nb = [cleaned[m + d][1] for d in (-2, -1, 1, 2)]
            if abs(cleaned[m][1] - _median(nb)) > spike_threshold_psi:
                spikes.append(_iso(cleaned[m][0]))  # flagged, never auto-deleted
    frozen_runs = 0
    run = 1
    for m in range(1, len(cleaned) + 1):
        if m < len(cleaned) and cleaned[m][1] == cleaned[m - 1][1]:
            run += 1
        else:
            if run >= frozen_min_points:
                frozen_runs += 1
            run = 1

    if unknown_units:
        sev = "BLOCK" if not cleaned else "WARN"
        qc.add(sev, "PRESSURE_UNIT_UNKNOWN",
               f"{unknown_units} pressure observations had unknown units and were discarded.",
               ALL if sev == "BLOCK" else [])
    if n_null:
        qc.add("WARN", "NULL_SENTINELS_REMOVED",
               f"{n_null} null-marker values ({', '.join(str(s) for s in null_sentinels)}) "
               "removed; see audit for timestamps.")
    if n_nonpos_abs:
        qc.add("WARN", "NONPOSITIVE_ABSOLUTE_PRESSURE",
               f"{n_nonpos_abs} non-positive absolute pressures removed.")
    if not cleaned:
        qc.add("BLOCK", "NO_VALID_PRESSURE", "No valid pressure observations remain.", ALL)
    if "gauge" in bases and "absolute" in bases:
        qc.add("BLOCK", "MIXED_PRESSURE_BASIS",
               "Gauge and absolute pressures are mixed in one stream.", ALL)
    basis = "mixed" if len(bases) > 1 else (next(iter(bases)) if bases else "unspecified")
    if basis == "unspecified":
        qc.add("INFO", "PRESSURE_BASIS_UNSPECIFIED",
               "Pressure basis (gauge/absolute) not stated; differences are unaffected, "
               "absolute outputs (p*) carry the unstated basis.")
    if assumed_tz:
        qc.add("WARN", "TIMEZONE_ASSUMED_UTC",
               f"{assumed_tz} pressure timestamps had no timezone and were treated as UTC.")
    if dup_groups:
        qc.add("INFO", "DUPLICATE_TIMESTAMPS",
               f"{dup_groups} duplicate-timestamp groups resolved with policy "
               f"'{duplicate_policy}'.")
    if spikes:
        qc.add("WARN", "SPIKE_CANDIDATES",
               f"{len(spikes)} spike candidates flagged (kept; a physical transient can "
               "resemble an outlier).")
    if frozen_runs:
        qc.add("WARN", "FROZEN_GAUGE_SUSPECTED",
               f"{frozen_runs} runs of >= {frozen_min_points} identical values "
               "(may also be gauge quantization).")
    audit = {"n_input": len(points), "n_clean": len(cleaned), "discarded": discarded,
             "spike_candidates_utc": spikes, "pressure_basis": basis}
    return cleaned, audit, qc


# ==============================================================================
# 5.1 Rate normalization
# ==============================================================================

def normalize_rates(points: Sequence[RatePoint], pvt: PhysicalInputs
                    ) -> Tuple[List[Tuple[datetime, float, str]], Dict[str, Any], QC]:
    """Surface oil/water -> reservoir-volume rate q_r = qo*Bo + qw*Bw (bbl/day)."""
    qc = QC()
    discarded: List[Dict[str, Any]] = []
    out: List[Tuple[datetime, float, str]] = []
    bo = _positive(pvt.oil_fvf)
    bw = _positive(pvt.water_fvf)
    if bo is None:
        qc.add("BLOCK", "OIL_FVF_MISSING",
               "Oil formation volume factor missing or invalid; reservoir rate undefined.",
               RATE_OUTPUTS)
        return [], {"discarded": discarded}, qc
    any_water = any(rp.water_surface_rate is not None for rp in points)
    unknown_units = 0
    water_fvf_needed = False
    for idx, rp in enumerate(points):
        try:
            t, _ = parse_utc(rp.timestamp_utc)
        except (ValueError, TypeError):
            discarded.append({"index": idx, "reason": "UNPARSEABLE_TIMESTAMP"})
            continue
        unit = (rp.rate_unit or "").strip().lower()
        if unit not in _RATE_UNITS:
            unknown_units += 1
            discarded.append({"index": idx, "reason": "UNKNOWN_RATE_UNIT", "unit": rp.rate_unit})
            continue
        if rp.quality_flag and rp.quality_flag.strip().lower() in _BAD_FLAGS:
            discarded.append({"index": idx, "reason": "FLAGGED_BAD"})
            continue
        f = _RATE_UNITS[unit]
        qo = _finite(rp.oil_surface_rate)
        if qo is None or qo < 0:
            discarded.append({"index": idx, "reason": "MISSING_OR_NEGATIVE_OIL_RATE"})
            continue
        qw_raw = rp.water_surface_rate
        if qw_raw is None:
            if any_water:  # other points report water: unknown here, do not assume zero
                discarded.append({"index": idx, "reason": "WATER_RATE_MISSING"})
                continue
            qw = 0.0
        else:
            qw = _finite(qw_raw)
            if qw is None or qw < 0:
                discarded.append({"index": idx, "reason": "INVALID_WATER_RATE"})
                continue
        if qw > 0 and bw is None:
            water_fvf_needed = True
            discarded.append({"index": idx, "reason": "WATER_FVF_MISSING"})
            continue
        qr = qo * f * bo + (qw * f * (bw or 0.0))
        out.append((t, qr, rp.source_timestamp or _iso(t)))
    out.sort(key=lambda r: r[0])
    dedup: List[Tuple[datetime, float, str]] = []
    for row in out:
        if dedup and dedup[-1][0] == row[0]:
            discarded.append({"timestamp": _iso(row[0]), "reason": "DUPLICATE_RATE_KEEP_LAST"})
            dedup[-1] = row
        else:
            dedup.append(row)
    if unknown_units and not dedup:
        qc.add("BLOCK", "RATE_UNIT_UNKNOWN", "All rate observations had unknown units.",
               RATE_OUTPUTS)
    elif unknown_units:
        qc.add("WARN", "RATE_UNIT_UNKNOWN",
               f"{unknown_units} rate observations with unknown units discarded.")
    if water_fvf_needed:
        qc.add("BLOCK", "WATER_FVF_MISSING",
               "Water is produced but water FVF is missing; affected rates discarded.",
               RATE_OUTPUTS)
    if not any_water:
        qc.add("WARN", "WATER_RATE_NOT_PROVIDED",
               "No water rates supplied; reservoir rate assumes oil only.")
    qc.add("INFO", "SINGLE_PHASE_EQUIVALENT_ASSUMED",
           "q_r = qo*Bo + qw*Bw treats oil-water flow as single-phase-equivalent; "
           "relative permeability and saturation changes are not modelled.")
    return dedup, {"n_input": len(points), "n_valid": len(dedup), "discarded": discarded}, qc


# ==============================================================================
# 4.2 / 4.3 Shut-in definition and reference point
# ==============================================================================

def define_shut_in(candidates: Sequence[ShutIn], merge_policy: str = "none",
                   max_merge_gap_hr: float = 1.0, merged_verified: bool = False,
                   rates: Sequence[Tuple[datetime, float, str]] = (),
                   rate_floor: float = 1.0
                   ) -> Tuple[Optional[Tuple[datetime, Optional[datetime], bool]],
                              List[Dict[str, Any]], List[float], QC]:
    qc = QC()
    if not candidates:
        qc.add("BLOCK", "NO_SHUT_IN", "No shut-in interval supplied.", ALL)
        return None, [], [], qc
    segs: List[Tuple[datetime, Optional[datetime], bool]] = []
    for c in candidates:
        s, _ = parse_utc(c.start_timestamp)
        e = parse_utc(c.end_timestamp)[0] if c.end_timestamp is not None else None
        if e is not None and e <= s:
            qc.add("BLOCK", "SHUT_IN_END_BEFORE_START", "Shut-in end precedes start.", ALL)
            return None, [], [], qc
        segs.append((s, e, bool(c.ongoing or e is None)))
    segs.sort(key=lambda r: r[0])
    seg_info = [{"start_utc": _iso(s), "end_utc": _iso(e), "ongoing": o} for s, e, o in segs]
    if len(segs) == 1:
        return segs[0], seg_info, [], qc
    if merge_policy != "merge":
        qc.add("BLOCK", "SHUT_IN_MULTIPLE_CANDIDATES",
               f"{len(segs)} shut-in candidates supplied and merge_policy='none'; "
               "distinct tests are never merged silently.", ALL)
        return None, seg_info, [], qc
    gaps: List[float] = []
    for (s0, e0, o0), (s1, _, _) in zip(segs, segs[1:]):
        if e0 is None or o0:
            qc.add("BLOCK", "SHUT_IN_OPEN_CANDIDATE_BEFORE_ANOTHER",
                   "An open-ended candidate precedes another candidate.", ALL)
            return None, seg_info, gaps, qc
        gap = _hours(s1 - e0)
        gaps.append(gap)
        if gap > max_merge_gap_hr:
            qc.add("BLOCK", "SHUT_IN_GAP_TOO_LARGE",
                   f"Gap of {gap:.3f} hr between candidates exceeds {max_merge_gap_hr} hr.",
                   ALL)
            return None, seg_info, gaps, qc
        if any(e0 <= t <= s1 and q > rate_floor for t, q, _ in rates):
            qc.add("BLOCK", "SHUT_IN_PRODUCTION_IN_GAP",
                   "Rate data show production between candidates; they are separate tests.",
                   ALL)
            return None, seg_info, gaps, qc
    start = segs[0][0]
    ongoing = any(o for _, _, o in segs)
    end = None if ongoing else max(e for _, e, _ in segs if e is not None)
    if not merged_verified:
        qc.add("WARN", "MERGED_SHUT_IN_UNVERIFIED",
               f"{len(segs)} candidates merged (gaps {['%.3f' % g for g in gaps]} hr); "
               "verify no sandface production resumed.", [])
    return (start, end, ongoing), seg_info, gaps, qc


def select_reference(pressure: Sequence[Tuple[datetime, float]], shut_start: datetime,
                     tolerance_hr: float) -> Tuple[Optional[Tuple[datetime, float]], QC]:
    qc = QC()
    times = [t for t, _ in pressure]
    pos = bisect.bisect_right(times, shut_start) - 1
    if pos < 0:
        qc.add("BLOCK", "MISSING_PRE_SHUT_PRESSURE",
               "No pressure observation at or before shut-in start.", ALL)
        return None, qc
    t_s, pwf = pressure[pos]
    gap = _hours(shut_start - t_s)
    if gap > tolerance_hr:
        qc.add("WARN", "REFERENCE_PRESSURE_GAP",
               f"Last flowing pressure is {gap:.3f} hr before shut-in start; early-time "
               "diagnostics are sensitive to reference-time error.")
    qc.add("INFO", "REFERENCE_FROM_SAMPLED_DATA",
           "t_s is the last sampled flowing pressure, an approximation of sandface shut-in.")
    return (t_s, pwf), qc


# ==============================================================================
# 5.2 / 5.3 Rate periods and equivalent producing time
# ==============================================================================

def build_rate_periods(rates: Sequence[Tuple[datetime, float, str]], shut_in_time: datetime,
                       rel_tol: float = 0.02, q_floor: float = 1.0,
                       history_complete: bool = False
                       ) -> Tuple[List[RatePeriod], Optional[datetime], QC]:
    """Piecewise-constant reservoir rates before shut-in. Times in hr from history start.

    Each observation holds until the next one. History starts at the first known
    observation; earlier rates stay unknown (never zero-filled).
    """
    qc = QC()
    obs = [r for r in rates if r[0] < shut_in_time]
    if not obs:
        qc.add("BLOCK", "NO_RATE_HISTORY", "No rate observations before shut-in.", RATE_OUTPUTS)
        return [], None, qc
    t0 = obs[0][0]
    periods: List[RatePeriod] = []
    for t, q, src in obs:
        start = _hours(t - t0)
        if periods:
            cur = periods[-1].reservoir_rate_bbl_day
            if abs(q - cur) <= rel_tol * max(abs(cur), q_floor):
                continue
            periods[-1].end_time_hr = start
        periods.append(RatePeriod(start, float("nan"), q, src))
    periods[-1].end_time_hr = _hours(shut_in_time - t0)
    if not history_complete:
        qc.add("WARN", "PRODUCING_HISTORY_START_ASSUMED",
               "Producing history assumed to begin at the first rate observation "
               f"({_iso(t0)}); earlier production is unknown.")
    if periods[-1].reservoir_rate_bbl_day <= q_floor:
        qc.add("BLOCK", "FINAL_RATE_ZERO",
               "Last flowing rate before shut-in is zero or missing.", RATE_OUTPUTS)
    return periods, t0, qc


def equivalent_producing_time(periods: Sequence[RatePeriod]) -> Optional[float]:
    qn = periods[-1].reservoir_rate_bbl_day if periods else 0.0
    if qn <= 0:
        return None
    return sum(p.reservoir_rate_bbl_day * (p.end_time_hr - p.start_time_hr)
               for p in periods) / qn


# ==============================================================================
# 6. Superposition coordinate
# ==============================================================================

def compute_superposition(elapsed_hr: Sequence[float], periods: Sequence[RatePeriod],
                          shut_in_time_hr: float
                          ) -> Tuple[List[float], float, List[Tuple[float, float]], QC]:
    """X(dt) = ln dt - sum_i c_i ln(tau_i + dt), c_i = dq_i/q_N, tau_i = t_s - t_{i-1}."""
    qc = QC()
    qn = periods[-1].reservoir_rate_bbl_day
    terms: List[Tuple[float, float]] = []
    prev = 0.0
    for p in periods:
        dq = p.reservoir_rate_bbl_day - prev
        prev = p.reservoir_rate_bbl_day
        if dq == 0.0:
            continue
        tau = shut_in_time_hr - p.start_time_hr
        if tau <= 0:
            qc.add("BLOCK", "SUPERPOSITION_TAU_NONPOSITIVE",
                   "A rate period starts at or after shut-in.", RATE_OUTPUTS)
            return [], 0.0, [], qc
        terms.append((dq / qn, tau))
    csum = sum(c for c, _ in terms)
    if abs(csum - 1.0) > 1e-9:
        qc.add("BLOCK", "SUPERPOSITION_WEIGHTS", f"sum(c_i) = {csum} != 1.", RATE_OUTPUTS)
    x_off = sum(c * math.log(tau) for c, tau in terms)
    xs = [math.log(d) - sum(c * math.log(tau + d) for c, tau in terms) for d in elapsed_hr]
    bad = sum(1 for a, b in zip(xs, xs[1:]) if b <= a)
    if bad:
        qc.add("WARN", "SUPERPOSITION_NON_MONOTONIC",
               f"X is non-increasing at {bad} steps; the derivative is undefined there.")
    return xs, x_off, terms, qc


# ==============================================================================
# 7. Logarithmic resampling
# ==============================================================================

def log_resample(elapsed_hr: Sequence[float], points_per_cycle: int) -> List[int]:
    """Indices of real observations nearest (in log time) to a log10 grid. No interpolation."""
    pos = [(i, math.log10(d)) for i, d in enumerate(elapsed_hr) if d > 0]
    if not pos:
        return []
    logs = [l for _, l in pos]
    lmin, lmax = logs[0], logs[-1]
    n = int(math.floor((lmax - lmin) * points_per_cycle)) + 1
    chosen = set()
    for j in range(n + 1):
        g = lmin + j / points_per_cycle
        k = bisect.bisect_left(logs, g)
        best = min((c for c in (k - 1, k) if 0 <= c < len(logs)),
                   key=lambda c: abs(logs[c] - g))
        chosen.add(pos[best][0])
    return sorted(chosen)


# ==============================================================================
# 8. Bourdet derivative
# ==============================================================================

def bourdet_derivative(x: Sequence[float], y: Sequence[float], log_dt: Sequence[float],
                       smoothing_L10: float, max_gap_cycles: float,
                       min_step: float = 1e-9) -> List[Optional[float]]:
    """Weighted three-point derivative dy/dx with neighbours >= L apart in x.

    Undefined (None) where valid neighbours are unavailable, where x is not strictly
    increasing between neighbours, or where the span crosses a sampling gap.
    """
    L = max(smoothing_L10 * LN10, min_step)
    n = len(x)
    out: List[Optional[float]] = [None] * n
    for i in range(n):
        j = i - 1
        while j >= 0 and x[i] - x[j] < L:
            j -= 1
        k = i + 1
        while k < n and x[k] - x[i] < L:
            k += 1
        if j < 0 or k >= n:
            continue
        if any(x[m + 1] - x[m] <= min_step for m in range(j, k)):
            continue
        if any(log_dt[m + 1] - log_dt[m] > max_gap_cycles for m in range(j, k)):
            continue
        dx1 = x[i] - x[j]
        dx2 = x[k] - x[i]
        d1 = (y[i] - y[j]) / dx1
        d2 = (y[k] - y[i]) / dx2
        out[i] = (d1 * dx2 + d2 * dx1) / (dx1 + dx2)
    return out


# ==============================================================================
# 9. Candidate radial-flow interval
# ==============================================================================

def _window_metrics(log_dt: Sequence[float], deriv: Sequence[Optional[float]],
                    a: int, b: int) -> Dict[str, float]:
    z = [math.log10(deriv[m]) for m in range(a, b + 1)]  # type: ignore[arg-type]
    xs = list(log_dt[a:b + 1])
    n = len(z)
    zb = sum(z) / n
    xb = sum(xs) / n
    sigma = math.sqrt(sum((zi - zb) ** 2 for zi in z) / n)
    sxx = sum((xi - xb) ** 2 for xi in xs)
    trend = (sum((xi - xb) * (zi - zb) for xi, zi in zip(xs, z)) / sxx) if sxx > 0 else 0.0
    gap = max((xs[m + 1] - xs[m] for m in range(n - 1)), default=0.0)
    return {"n_points": n, "log_cycles": xs[-1] - xs[0], "sigma_log10": sigma,
            "trend_per_cycle": trend, "max_gap_cycles": gap}


def _window_ok(m: Dict[str, float], c: AnalysisControls) -> List[str]:
    fails = []
    if m["n_points"] < c.radial_min_points:
        fails.append("too_few_points")
    if m["log_cycles"] < c.radial_min_cycles:
        fails.append("too_short")
    if m["sigma_log10"] > c.radial_max_sigma:
        fails.append("scatter")
    if abs(m["trend_per_cycle"]) > c.radial_max_trend:
        fails.append("trend")
    if m["max_gap_cycles"] > c.max_gap_cycles:
        fails.append("data_gap")
    return fails


def find_radial_interval(dt: Sequence[float], deriv: Sequence[Optional[float]],
                         c: AnalysisControls) -> Tuple[Optional[Dict[str, Any]], QC]:
    qc = QC()
    log_dt = [math.log10(d) for d in dt]
    valid = [d is not None and d > 0 and math.isfinite(d) for d in deriv]
    n = len(dt)

    if c.manual_radial_dt_hr is not None:
        lo, hi = c.manual_radial_dt_hr
        idx = [m for m in range(n) if lo <= dt[m] <= hi]
        if not idx or not all(valid[m] for m in range(idx[0], idx[-1] + 1)):
            qc.add("BLOCK", "MANUAL_RADIAL_INVALID",
                   "Manual radial interval contains no points or undefined derivatives.",
                   RADIAL_OUTPUTS)
            return None, qc
        a, b = idx[0], idx[-1]
        m = _window_metrics(log_dt, deriv, a, b)
        fails = _window_ok(m, c)
        forced = bool(fails) and c.manual_radial_force
        if forced:
            qc.add("WARN", "MANUAL_RADIAL_FORCED",
                   f"Engineer accepted the manual radial interval despite failed checks: {', '.join(fails)}. "
                   "Results depend on that judgement.")
        elif fails:
            qc.add("BLOCK", "MANUAL_RADIAL_FAILED_QC",
                   f"Manual radial interval fails QC: {', '.join(fails)}.", RADIAL_OUTPUTS)
        win = {**m, "a": a, "b": b, "selection": "manual_forced" if forced else "manual",
               "accepted": (not fails) or forced, "dt_start_hr": dt[a], "dt_end_hr": dt[b], "failed_checks": fails}
        return win, qc

    first_start = 0
    hump = None
    if c.exclude_storage_hump and any(valid):
        imax = max((m for m in range(n) if valid[m]), key=lambda m: deriv[m])  # type: ignore
        limit = c.hump_drop_ratio * deriv[imax]  # type: ignore[operator]
        fall = next((m for m in range(imax + 1, n) if valid[m] and deriv[m] < limit), None)
        if fall is not None:
            first_start = fall  # flat hump tops tie at the max, so cut where it has fallen
            hump = {"dt_hr": dt[imax], "derivative_psi": deriv[imax],
                    "search_starts_dt_hr": dt[fall]}
            qc.add("INFO", "STORAGE_HUMP_EXCLUDED",
                   f"Early derivative maximum ({deriv[imax]:.4g} psi at dt = {dt[imax]:.4g} hr) "
                   f"falls below {c.hump_drop_ratio:g}x by dt = {dt[fall]:.4g} hr; radial "
                   "windows are searched only after that.")

    best: Optional[Dict[str, Any]] = None
    for a in range(first_start, n):
        if not valid[a]:
            continue
        for b in range(a + 1, n):
            if not valid[b] or log_dt[b] - log_dt[b - 1] > c.max_gap_cycles:
                break
            m = _window_metrics(log_dt, deriv, a, b)
            if _window_ok(m, c):
                continue
            score = m["sigma_log10"] - c.radial_lambda * m["log_cycles"]
            if best is None or score < best["score"]:
                best = {**m, "a": a, "b": b, "score": score}
    if best is None:
        qc.add("BLOCK", "RADIAL_INTERVAL_NOT_FOUND",
               "No derivative window meets the radial-flow heuristics; kh and skin are "
               "not reported. A manual interval may be supplied (same QC applies).",
               RADIAL_OUTPUTS)
        return None, qc
    best.update({"selection": "automatic", "accepted": True,
                 "dt_start_hr": dt[best["a"]], "dt_end_hr": dt[best["b"]],
                 "failed_checks": [], "excluded_hump": hump})
    qc.add("INFO", "RADIAL_INTERVAL_HEURISTIC",
           "Radial window chosen by configurable heuristics (not physical laws).")
    return best, qc


# ==============================================================================
# 10.7 Wellbore storage
# ==============================================================================

def estimate_storage(dt: Sequence[float], dp: Sequence[float],
                     deriv: Sequence[Optional[float]], q_n: float,
                     slope_tol: float, min_points: int) -> Tuple[Optional[Dict[str, Any]], QC]:
    """Fit C = q_N/24 * dt/dp over the earliest contiguous unit-slope run."""
    qc = QC()
    pts = [(i, dt[i], dp[i]) for i in range(len(dt)) if dt[i] > 0 and dp[i] > 0]
    if len(pts) < 2:
        qc.add("WARN", "STORAGE_NOT_RESOLVED", "Too few early positive points.",
               ["storage_bbl_per_psi", "dimensionless_storage"])
        return None, qc
    run = [pts[0]]
    for prev, cur in zip(pts, pts[1:]):
        slope = math.log(cur[2] / prev[2]) / math.log(cur[1] / prev[1])
        if abs(slope - 1.0) <= slope_tol:
            run.append(cur)
        else:
            break
    if len(run) < min_points:
        qc.add("WARN", "STORAGE_NOT_RESOLVED",
               "No resolved early unit-slope interval; storage is not estimated.",
               ["storage_bbl_per_psi", "dimensionless_storage"])
        return None, qc
    consistency = [deriv[i] / dp_i for i, _, dp_i in run if deriv[i] is not None]
    if consistency and not all(0.7 <= r <= 1.3 for r in consistency):
        qc.add("WARN", "STORAGE_DERIVATIVE_INCONSISTENT",
               "Pressure and derivative do not overlap on the unit-slope run; storage "
               "estimate is suppressed.", ["storage_bbl_per_psi", "dimensionless_storage"])
        return None, qc
    cs = [q_n / 24.0 * d / p for _, d, p in run]
    return {"C_bbl_per_psi": _median(cs), "n_points": len(run),
            "dt_start_hr": run[0][1], "dt_end_hr": run[-1][1]}, qc


# ==============================================================================
# 12. Flow-regime diagnostics (descriptive only)
# ==============================================================================

_SIGNATURES = [(1.0, "UNIT_SLOPE", "possible wellbore storage"),
               (0.5, "HALF_SLOPE", "possible linear-flow-type response"),
               (0.25, "QUARTER_SLOPE", "possible bilinear-flow-type response"),
               (0.0, "ZERO_SLOPE", "possible radial flow"),
               (-0.5, "NEGATIVE_HALF_SLOPE", "possible spherical-flow-type response")]


def flow_regime_diagnostics(dt: Sequence[float], deriv: Sequence[Optional[float]],
                            radial: Optional[Dict[str, Any]], d_r: Optional[float]
                            ) -> Tuple[Dict[str, Any], QC]:
    qc = QC()
    ld = [math.log10(d) for d in dt]
    ok = [d is not None and d > 0 for d in deriv]
    slopes: List[Optional[float]] = [None] * len(dt)
    for i in range(len(dt)):
        if not ok[i]:
            continue
        j = i - 1
        while j >= 0 and ld[i] - ld[j] < 0.1:
            j -= 1
        k = i + 1
        while k < len(dt) and ld[k] - ld[i] < 0.1:
            k += 1
        if j < 0 or k >= len(dt) or not ok[j] or not ok[k]:
            continue
        slopes[i] = (math.log10(deriv[k]) - math.log10(deriv[j])) / (ld[k] - ld[j])  # type: ignore
    labels = []
    for s in slopes:
        lab = None
        if s is not None:
            for v, name, _ in _SIGNATURES:
                if abs(s - v) <= 0.1:
                    lab = name
        labels.append(lab)
    segments = []
    i = 0
    while i < len(labels):
        if labels[i] is None:
            i += 1
            continue
        j = i
        while j + 1 < len(labels) and labels[j + 1] == labels[i]:
            j += 1
        if ld[j] - ld[i] >= 0.3:
            interp = next(t for _, n, t in _SIGNATURES if n == labels[i])
            segments.append({"signature": labels[i], "dt_start_hr": dt[i], "dt_end_hr": dt[j],
                             "possible_interpretation": interp})
        i = j + 1
    late = None
    if radial is not None and d_r:
        after = [m for m in range(radial["b"] + 1, len(dt)) if ok[m]]
        if len(after) >= 3 and ld[after[-1]] - ld[after[0]] >= 0.3:
            ratio = _median([deriv[m] for m in after]) / d_r  # type: ignore
            late = {"late_to_radial_derivative_ratio": ratio,
                    "dt_start_hr": dt[after[0]], "dt_end_hr": dt[after[-1]]}
            if ratio > 1.2:
                qc.add("INFO", "LATE_DERIVATIVE_UPTURN",
                       f"Late derivative is {ratio:.2f}x the radial level (descriptive; not "
                       "proof of a boundary).")
            elif ratio < 0.83:
                qc.add("INFO", "LATE_DERIVATIVE_DOWNTURN",
                       f"Late derivative is {ratio:.2f}x the radial level (support, "
                       "equilibration, mobility or producing-time effects are possible).")
    return {"local_slopes": slopes, "segments": segments, "late_time": late,
            "note": "Signatures are possible interpretations, not classifications."}, qc


# ==============================================================================
# 10. Parameter estimates
# ==============================================================================

def _validate_physical(phys: PhysicalInputs) -> Tuple[Dict[str, Optional[float]], QC]:
    qc = QC()
    vals = {
        "mu": _positive(phys.viscosity_cp), "ct": _positive(phys.total_compressibility_per_psi),
        "phi": _positive(phys.porosity_fraction), "h": _positive(phys.contributing_thickness_ft),
        "rw": _positive(phys.wellbore_radius_ft), "kh_ref": _positive(phys.log_kh_md_ft),
    }
    if vals["phi"] is not None and vals["phi"] >= 1.0:
        vals["phi"] = None
    needs = {
        "mu": ("viscosity_cp", ["kh_md_ft", "k_md", "apparent_skin", "skin_pressure_drop_psi",
                                "investigation_radius_ft", "flow_efficiency",
                                "kh_ratio_pta_to_reference"]),
        "h": ("contributing_thickness_ft", ["k_md", "apparent_skin", "skin_pressure_drop_psi",
                                            "investigation_radius_ft", "flow_efficiency",
                                            "dimensionless_storage"]),
        "phi": ("porosity_fraction", ["apparent_skin", "skin_pressure_drop_psi",
                                      "investigation_radius_ft", "flow_efficiency",
                                      "dimensionless_storage"]),
        "ct": ("total_compressibility_per_psi", ["apparent_skin", "skin_pressure_drop_psi",
                                                 "investigation_radius_ft", "flow_efficiency",
                                                 "dimensionless_storage"]),
        "rw": ("wellbore_radius_ft", ["apparent_skin", "skin_pressure_drop_psi",
                                      "flow_efficiency", "dimensionless_storage"]),
    }
    for key, (name, outs) in needs.items():
        if vals[key] is None:
            qc.add("BLOCK", f"MISSING_{name.upper()}",
                   f"{name} missing or invalid; dependent outputs suppressed (no defaults "
                   "are substituted).", outs)
    if vals["kh_ref"] is None:
        qc.add("INFO", "NO_REFERENCE_KH",
               "No independent kh supplied; kh ratio not reported.")
    missing_src = [f for f in ("oil_fvf", "viscosity_cp", "total_compressibility_per_psi",
                               "porosity_fraction", "contributing_thickness_ft",
                               "wellbore_radius_ft")
                   if getattr(phys, f) is not None and f not in phys.provenance]
    if missing_src:
        qc.add("INFO", "PHYSICAL_INPUT_PROVENANCE_MISSING",
               f"No source recorded for: {', '.join(missing_src)}.")
    return vals, qc


def compute_parameters(d_r: float, q_n: float, pv: Dict[str, Optional[float]],
                       dp_rad: Sequence[float], x_rad: Sequence[float], x_off: float,
                       pwf: float, dt_max: float, storage: Optional[Dict[str, Any]]
                       ) -> Dict[str, Optional[float]]:
    r: Dict[str, Optional[float]] = dict.fromkeys(RESULT_KEYS)
    mu, ct, phi, h, rw = pv["mu"], pv["ct"], pv["phi"], pv["h"], pv["rw"]
    r["radial_derivative_psi"] = d_r
    r["semilog_slope_psi_per_cycle"] = LN10 * d_r
    r["mobility_thickness"] = 70.6 * q_n / d_r
    kh = 70.6 * q_n * mu / d_r if mu else None
    k = kh / h if kh and h else None
    r["kh_md_ft"], r["k_md"] = kh, k
    b = _median([p - d_r * x for p, x in zip(dp_rad, x_rad)])
    p_star = pwf + b
    r["extrapolated_pressure_psi"] = p_star
    if k and phi and mu and ct and rw:
        tdc = math.log(0.0002637 * k / (phi * mu * ct * rw * rw))
        s = 0.5 * _median([p / d_r - (x + x_off) - tdc - RADIAL_CONST
                           for p, x in zip(dp_rad, x_rad)])
        r["apparent_skin"] = s
        r["skin_pressure_drop_psi"] = 2.0 * d_r * s
        denom = p_star - pwf
        if denom > 1e-6:
            r["flow_efficiency"] = (denom - 2.0 * d_r * s) / denom
    if k and phi and mu and ct:
        r["investigation_radius_ft"] = math.sqrt(k * dt_max / (948.0 * phi * mu * ct))
    if storage:
        c_val = storage["C_bbl_per_psi"]
        r["storage_bbl_per_psi"] = c_val
        if phi and ct and h and rw:
            r["dimensionless_storage"] = 0.8936 * c_val / (phi * ct * h * rw * rw)
    if kh and pv["kh_ref"]:
        r["kh_ratio_pta_to_reference"] = kh / pv["kh_ref"]
    return r


# ==============================================================================
# 11. Optional open-interval property aggregation
# ==============================================================================

def union_intervals(intervals: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []
    for top, base in sorted((min(a, b), max(a, b)) for a, b in intervals):
        if out and top <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], base))
        else:
            out.append((top, base))
    return out


def aggregate_open_interval_properties(zones: Sequence[Dict[str, float]],
                                       completions: Sequence[Tuple[float, float]],
                                       c_f: Optional[float] = None,
                                       c_o: Optional[float] = None,
                                       c_w: Optional[float] = None) -> Dict[str, Any]:
    """zones: dicts with top, base, net_thickness_ft, porosity, water_saturation,
    permeability_md (measured depth). completions: (top, base) open intervals."""
    union = union_intervals(completions)
    rows = []
    for z in zones:
        top, base = z["top"], z["base"]
        overlap = sum(max(0.0, min(base, B) - max(top, T)) for T, B in union)
        f = overlap / (base - top) if base > top else 0.0
        rows.append((z, f, f * z["net_thickness_ft"]))
    h = sum(hz for _, _, hz in rows)
    out: Dict[str, Any] = {"open_union": union, "zone_fractions": [f for _, f, _ in rows],
                           "h_ft": h, "porosity": None, "water_saturation": None,
                           "kh_reference_md_ft": None, "ct_per_psi": None,
                           "assumption": "Net thickness scaled by measured-depth overlap "
                                         "fraction (approximation)."}
    if h > 0:
        out["porosity"] = sum(hz * z["porosity"] for z, _, hz in rows) / h
        out["water_saturation"] = sum(hz * z["water_saturation"] for z, _, hz in rows) / h
        out["kh_reference_md_ft"] = sum(hz * z["permeability_md"] for z, _, hz in rows)
        if None not in (c_f, c_o, c_w):
            sw = out["water_saturation"]
            out["ct_per_psi"] = c_f + (1 - sw) * c_o + sw * c_w  # type: ignore[operator]
    return out


# ==============================================================================
# Orchestration
# ==============================================================================

ASSUMPTIONS = [
    "Vertical well, homogeneous, infinite-acting radial flow during the accepted window.",
    "Single-phase or explicitly assumed single-phase-equivalent flow (q_r = qo*Bo + qw*Bw).",
    "Constant fluid and rock properties (linear diffusion; superposition valid).",
    "Skin formula assumes radial flow was reached in each producing period.",
    "p* is the infinite-shut-in extrapolation, not automatically average reservoir pressure.",
    "Radius of investigation uses r = sqrt(k*dt/(948*phi*mu*ct)); other conventions differ.",
    "Results are preliminary and non-unique.",
]
REFERENCES = [
    "Horner (1951), Pressure Build-up in Wells, 3rd World Petroleum Congress.",
    "Agarwal (1980), SPE-9289.",
    "Bourdet, Whittle, Douglas, Pirard (1983), World Oil.",
    "Bourdet, Ayoub, Pirard (1989), SPE-12777.",
    "van Everdingen and Hurst (1949), Trans. AIME.",
    "Stehfest (1970), Algorithm 368, Comm. ACM.",
    "Earlougher (1977), SPE Monograph 5.",
    "Lee (1982), Well Testing, SPE Textbook Series 1.",
]


def _fingerprint(payload: Dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(sanitize(payload), sort_keys=True, default=str)
                          .encode()).hexdigest()[:16]


def analyze_buildup(pressure_points: Sequence[PressurePoint],
                    rate_points: Sequence[RatePoint],
                    shut_in_candidates: Sequence[ShutIn],
                    physical: PhysicalInputs,
                    controls: Optional[AnalysisControls] = None,
                    include_series: bool = True) -> Dict[str, Any]:
    c = controls or AnalysisControls()
    qc = QC()
    results: Dict[str, Optional[float]] = dict.fromkeys(RESULT_KEYS)
    diagnostics: Dict[str, Any] = {
        "derivative_coordinate": "superposition X (natural log); smoothing L in X",
        "radial_interval": None, "storage_interval": None, "radial_line_fit": None,
        "flow_regimes": None, "series": None,
    }
    shut_info: Dict[str, Any] = {}
    rate_info: Dict[str, Any] = {}

    cleaned, p_audit, f = clean_pressure(pressure_points, c.duplicate_policy,
                                         c.spike_threshold_psi, c.frozen_min_points,
                                         c.null_sentinels)
    qc.extend(f)
    rates, r_audit, f = normalize_rates(rate_points, physical)
    qc.extend(f)
    shut, segs, gaps, f = define_shut_in(shut_in_candidates, c.merge_policy,
                                         c.max_merge_gap_hr, c.merged_shutin_verified,
                                         rates, c.rate_floor_bbl_day)
    qc.extend(f)
    pv, f = _validate_physical(physical)
    qc.extend(f)
    shut_info.update({"segments": segs, "gaps_hr": gaps,
                      "pressure_basis": p_audit["pressure_basis"]})

    def finish() -> Dict[str, Any]:
        for key in RESULT_KEYS:
            if qc.blocked(key):
                results[key] = None
        inputs = {
            "units": {"pressure": "psi", "time": "hr", "rate": "reservoir bbl/day",
                      "viscosity": "cp", "permeability": "mD", "length": "ft",
                      "compressibility": "1/psi"},
            "physical_properties": asdict(physical),
            "rate_source": rate_info,
            "shut_in_selection": shut_info,
            "numerical_controls": asdict(c),
        }
        out = {
            "schema_version": SCHEMA_VERSION,
            "status": STATUS,
            "analysis_fingerprint": _fingerprint({
                "p": [(str(p.timestamp_utc), p.pressure, p.pressure_unit, p.quality_flag)
                      for p in pressure_points],
                "r": [asdict(r) for r in rate_points],
                "s": [asdict(s) for s in shut_in_candidates],
                "phys": asdict(physical), "c": asdict(c)}),
            "inputs": inputs,
            "results": results,
            "diagnostics": diagnostics,
            "qc_flags": [asdict(fl) for fl in qc.flags],
            "audit": {"pressure": p_audit, "rates": r_audit},
            "assumptions": ASSUMPTIONS,
            "references": REFERENCES,
        }
        return sanitize(out)

    if shut is None or not cleaned:
        return finish()
    shut_start, shut_end, ongoing = shut
    ref, f = select_reference(cleaned, shut_start, c.reference_gap_tolerance_hr)
    qc.extend(f)
    if ref is None:
        return finish()
    t_s, pwf = ref
    shut_info.update({"start_utc": _iso(shut_start), "end_utc": _iso(shut_end),
                      "ongoing": ongoing, "reference_time_utc": _iso(t_s), "p_wf_psi": pwf})

    obs = [(t, p) for t, p in cleaned if t > t_s and (shut_end is None or t <= shut_end)]
    dt_all = [_hours(t - t_s) for t, _ in obs]
    dp_all = [p - pwf for _, p in obs]
    if len(dt_all) < 5:
        qc.add("BLOCK", "INSUFFICIENT_BUILDUP_DATA",
               f"Only {len(dt_all)} pressure observations in the shut-in.", ALL)
        return finish()

    periods, t0, f = build_rate_periods(rates, t_s, c.rate_rel_tolerance,
                                        c.rate_floor_bbl_day, c.history_complete)
    qc.extend(f)
    rates_ok = bool(periods) and not qc.blocked("mobility_thickness")
    q_n = periods[-1].reservoir_rate_bbl_day if periods else 0.0
    x_off = 0.0
    if rates_ok:
        ts_hr = _hours(t_s - t0)  # type: ignore[operator]
        tp_eq = equivalent_producing_time(periods)
        results["equivalent_producing_time_hr"] = tp_eq
        rate_info.update({
            "history_start_utc": _iso(t0), "q_final_reservoir_bbl_day": q_n,
            "producing_duration_hr": ts_hr,
            "periods": [asdict(p) for p in periods],
        })
        x_all, x_off, terms, f = compute_superposition(dt_all, periods, ts_hr)
        qc.extend(f)
        if not x_all:
            rates_ok = False
        elif ts_hr < dt_all[-1]:
            qc.add("WARN", "SHORT_PRODUCING_HISTORY",
                   f"Producing duration ({ts_hr:.2f} hr) is shorter than the shut-in "
                   f"({dt_all[-1]:.2f} hr); radial-flow assumptions for the drawdown are weak.")
    if not rates_ok:
        x_all = [math.log(d) for d in dt_all]
        x_off = 0.0
        diagnostics["derivative_coordinate"] = "ln(dt) fallback; producing-time effects " \
                                               "are uncorrected"
        qc.add("WARN", "SUPERPOSITION_UNAVAILABLE",
               "No usable rate history: derivative vs ln(dt); rate-scaled outputs blocked.",
               RATE_OUTPUTS)
        qc.flags[-1].severity = "BLOCK"

    sel = log_resample(dt_all, c.points_per_cycle)
    dt = [dt_all[i] for i in sel]
    dp = [dp_all[i] for i in sel]
    xs = [x_all[i] for i in sel]
    ld = [math.log10(d) for d in dt]
    deriv = bourdet_derivative(xs, dp, ld, c.smoothing_L10, c.max_gap_cycles,
                               c.min_coordinate_step)
    gap_list = [(dt[m], dt[m + 1]) for m in range(len(dt) - 1)
                if ld[m + 1] - ld[m] > c.max_gap_cycles]
    if gap_list:
        qc.add("WARN", "SAMPLING_GAPS",
               f"{len(gap_list)} gaps > {c.max_gap_cycles} log cycles; derivative not taken "
               f"across them: {[(round(a, 5), round(b, 5)) for a, b in gap_list]}.")
    nonpos = sum(1 for d in deriv if d is not None and d <= 0)
    if nonpos:
        qc.add("INFO", "NONPOSITIVE_DERIVATIVE",
               f"{nonpos} non-positive derivative values excluded from log analysis.")
    if include_series:
        diagnostics["series"] = {"dt_hr": dt, "dp_psi": dp, "X": xs, "derivative_psi": deriv,
                                 "n_raw": len(dt_all), "n_resampled": len(dt)}

    if rates_ok:
        storage, f = estimate_storage(dt, dp, deriv, q_n, c.storage_slope_tolerance,
                                      c.storage_min_points)
        qc.extend(f)
        diagnostics["storage_interval"] = storage
    else:
        storage = None

    radial, f = find_radial_interval(dt, deriv, c)
    qc.extend(f)
    d_r = None
    if radial is not None:
        diagnostics["radial_interval"] = {k: v for k, v in radial.items() if k not in ("a", "b")}
        a, b = radial["a"], radial["b"]
        d_r = _median([deriv[m] for m in range(a, b + 1)])  # type: ignore[misc]
        if rates_ok and radial["accepted"]:
            dp_rad, x_rad = dp[a:b + 1], xs[a:b + 1]
            results.update(compute_parameters(d_r, q_n, pv, dp_rad, x_rad, x_off, pwf,
                                              dt_all[-1], storage))
            results["equivalent_producing_time_hr"] = equivalent_producing_time(periods)
            fit = _linfit(x_rad, dp_rad)
            diagnostics["radial_line_fit"] = {
                **fit, "p_star_fit_psi": pwf + fit["intercept"],
                "p_star_fit_se_psi": fit["se_intercept"],
                "note": "Least-squares dp vs X over the accepted window."}
            qc.add("INFO", "SKIN_IS_APPARENT",
                   "Skin is apparent: partial penetration, geometry, multiphase mobility and "
                   "rate-history error all contribute.")
    elif storage is not None:
        results["storage_bbl_per_psi"] = storage["C_bbl_per_psi"]
        if pv["phi"] and pv["ct"] and pv["h"] and pv["rw"]:
            results["dimensionless_storage"] = 0.8936 * storage["C_bbl_per_psi"] / (
                pv["phi"] * pv["ct"] * pv["h"] * pv["rw"] ** 2)
    regimes, f = flow_regime_diagnostics(dt, deriv, radial, d_r)
    qc.extend(f)
    diagnostics["flow_regimes"] = regimes
    return finish()


def export_result_json(result: Dict[str, Any], path: str) -> None:
    """Write an existing result object (no re-analysis). NaN/inf are never written."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(sanitize(result), fh, indent=2, allow_nan=False)


def load_input_json(path: str) -> Tuple[List[PressurePoint], List[RatePoint], List[ShutIn],
                                        PhysicalInputs, AnalysisControls]:
    """JSON with keys: pressure[], rates[], shut_ins[], physical{}, controls{} (optional),
    using the dataclass field names above."""
    with open(path, "r", encoding="utf-8") as fh:
        d = json.load(fh)
    ctrl = d.get("controls", {})
    if ctrl.get("manual_radial_dt_hr") is not None:
        ctrl["manual_radial_dt_hr"] = tuple(ctrl["manual_radial_dt_hr"])
    return ([PressurePoint(**p) for p in d["pressure"]], [RatePoint(**r) for r in d["rates"]],
            [ShutIn(**s) for s in d["shut_ins"]], PhysicalInputs(**d["physical"]),
            AnalysisControls(**ctrl))


# ==============================================================================
# 16.2 Analytical radial model for synthetic fixtures
# ==============================================================================

@lru_cache(maxsize=None)
def stehfest_weights(n: int = 12) -> Tuple[float, ...]:
    if n % 2:
        raise ValueError("Stehfest N must be even")
    half = n // 2
    fact = math.factorial
    v = []
    for i in range(1, n + 1):
        s = Fraction(0)
        for k in range((i + 1) // 2, min(i, half) + 1):
            s += Fraction(k ** half * fact(2 * k),
                          fact(half - k) * fact(k) * fact(k - 1) * fact(i - k) * fact(2 * k - i))
        v.append((-1) ** (i + half) * s)
    return tuple(float(x) for x in v)


def stehfest_invert(F, t: float, n: int = 12) -> float:
    a = LN2 / t
    return a * sum(w * F((j + 1) * a) for j, w in enumerate(stehfest_weights(n)))


@lru_cache(maxsize=200000)
def bessel_k01_scaled(x: float) -> Tuple[float, float]:
    """(e^x K0(x), e^x K1(x)) by trapezoidal quadrature of
    e^x K_nu(x) = int_0^inf exp(-x (cosh t - 1)) cosh(nu t) dt.
    The integrand is analytic in |Im t| < pi/2, so the trapezoid rule converges
    exponentially (error ~ exp(-pi^2/h))."""
    if x <= 0:
        raise ValueError("x must be positive")
    h = 0.1
    s0 = 0.5
    s1 = 0.5
    k = 1
    while True:
        t = k * h
        ch = math.cosh(t)
        arg = x * (ch - 1.0)
        if arg > 50.0:
            break
        e = math.exp(-arg)
        s0 += e
        s1 += e * ch
        k += 1
    return h * s0, h * s1


def pwd_laplace(u: float, cd: float, skin: float) -> float:
    """Laplace-domain wellbore pressure with constant storage and skin (van Everdingen-Hurst)."""
    x = math.sqrt(u)
    k0, k1 = bessel_k01_scaled(x)  # common e^x factor cancels in the ratio
    num = k0 + skin * x * k1
    return num / (u * (x * k1 + cd * u * num))


@lru_cache(maxsize=200000)
def pwd(td: float, cd: float, skin: float, n: int = 12) -> float:
    """Wellbore pD(tD) with storage and skin.

    Negative skin makes K0 + S*sqrt(u)*K1 vanish near u = 1/S^2 (a pole that breaks
    Stehfest), so S < 0 uses the effective-wellbore-radius form r_w' = r_w e^{-S}:
    pD(tD, CD, S) = pD(tD e^{2S}, CD e^{2S}, 0), exact in the radial limit.
    """
    if skin < 0:
        f = math.exp(2.0 * skin)
        return stehfest_invert(lambda u: pwd_laplace(u, cd * f, 0.0), td * f, n)
    return stehfest_invert(lambda u: pwd_laplace(u, cd, skin), td, n)


@dataclass
class SyntheticCase:
    name: str = "storage_skin"
    k_md: float = 50.0
    h_ft: float = 40.0
    porosity: float = 0.2
    viscosity_cp: float = 1.2
    ct_per_psi: float = 1.2e-5
    rw_ft: float = 0.354
    skin: float = 3.0
    cd: float = 500.0
    oil_fvf: float = 1.25
    water_fvf: float = 1.02
    p_initial_psi: float = 4500.0
    # (start_hr, oil STB/d, water STB/d), surface rates
    schedule: List[Tuple[float, float, float]] = field(
        default_factory=lambda: [(0.0, 800.0, 0.0)])
    shut_in_time_hr: float = 200.0
    shut_in_duration_hr: float = 72.0
    dt_min_hr: float = 3e-4
    points_per_decade: int = 25
    irregular: bool = False
    noise_psi: float = 0.0
    quantization_psi: float = 0.0
    seed: int = 7

    def storage_bbl_per_psi(self) -> float:
        return self.cd * self.porosity * self.ct_per_psi * self.h_ft * self.rw_ft ** 2 / 0.8936


def generate_synthetic_buildup(case: SyntheticCase):
    """Multi-rate drawdown + build-up via superposition of the storage/skin solution:
    p_i - p(t) = 141.2 mu/(kh) * sum_j dq_j pwD(tD(t - t_j)), including the shut-in step."""
    rng = random.Random(case.seed)
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    steps = []
    prev = 0.0
    for t0, qo, qw in case.schedule:
        qr = qo * case.oil_fvf + qw * case.water_fvf
        steps.append((t0, qr - prev))
        prev = qr
    steps.append((case.shut_in_time_hr, -prev))
    tdc = 0.0002637 * case.k_md / (case.porosity * case.viscosity_cp * case.ct_per_psi *
                                   case.rw_ft ** 2)
    coef = 141.2 * case.viscosity_cp / (case.k_md * case.h_ft)

    def pressure(t: float) -> float:
        s = sum(dq * pwd(tdc * (t - tj), case.cd, case.skin)
                for tj, dq in steps if dq != 0.0 and t > tj)
        return case.p_initial_psi - coef * s

    ts = case.shut_in_time_hr
    rel = [-x for x in (24.0, 6.0, 2.0, 1.0, 0.25) if ts - x > 0] + [0.0]
    lmin = math.log10(case.dt_min_hr)
    lmax = math.log10(case.shut_in_duration_hr)
    n = int(round((lmax - lmin) * case.points_per_decade))
    for i in range(n + 1):
        l = lmin + i / case.points_per_decade
        if case.irregular:
            l += rng.uniform(-0.45, 0.45) / case.points_per_decade
        rel.append(10 ** min(max(l, lmin), lmax))
    pts: List[PressurePoint] = []
    seen = set()
    for r in sorted(rel):
        stamp = base + timedelta(hours=ts + r)
        if stamp in seen:
            continue
        seen.add(stamp)
        t = _hours(stamp - base)
        p = pressure(t)
        if case.noise_psi:
            p += rng.gauss(0.0, case.noise_psi)
        if case.quantization_psi:
            p = round(p / case.quantization_psi) * case.quantization_psi
        pts.append(PressurePoint(_iso(stamp), p, "psia"))
    rates = [RatePoint(_iso(base + timedelta(hours=t0)), qo, "stb/day", qw)
             for t0, qo, qw in case.schedule]
    rates.append(RatePoint(_iso(base + timedelta(hours=ts)), 0.0, "stb/day", 0.0))
    shut = ShutIn(_iso(base + timedelta(hours=ts)),
                  _iso(base + timedelta(hours=ts + case.shut_in_duration_hr)))
    src = "synthetic fixture (known truth)"
    phys = PhysicalInputs(
        oil_fvf=case.oil_fvf, water_fvf=case.water_fvf, viscosity_cp=case.viscosity_cp,
        total_compressibility_per_psi=case.ct_per_psi, porosity_fraction=case.porosity,
        contributing_thickness_ft=case.h_ft, wellbore_radius_ft=case.rw_ft,
        provenance={f: src for f in ("oil_fvf", "water_fvf", "viscosity_cp",
                                     "total_compressibility_per_psi", "porosity_fraction",
                                     "contributing_thickness_ft", "wellbore_radius_ft")})
    truth = {"k_md": case.k_md, "kh_md_ft": case.k_md * case.h_ft, "apparent_skin": case.skin,
             "storage_bbl_per_psi": case.storage_bbl_per_psi(), "dimensionless_storage": case.cd,
             "extrapolated_pressure_psi": case.p_initial_psi}
    return pts, rates, [shut], phys, truth


DEMO_CASES = {
    "storage_skin": SyntheticCase(name="storage_skin"),
    "no_storage": SyntheticCase(name="no_storage", cd=0.0, skin=0.0),
    "multirate": SyntheticCase(name="multirate", skin=-1.0, cd=200.0,
                               schedule=[(0.0, 600.0, 0.0), (60.0, 0.0, 0.0),
                                         (70.0, 900.0, 200.0), (150.0, 700.0, 150.0)]),
    "noisy": SyntheticCase(name="noisy", irregular=True, noise_psi=0.05,
                           quantization_psi=0.01, seed=11),
    "short": SyntheticCase(name="short", shut_in_duration_hr=0.8),
}

# ==============================================================================
# 16. Self-tests (tolerances fixed before running)
# ==============================================================================

TOL = {"k_rel": 0.05, "skin_abs": 0.5, "C_rel": 0.10, "pstar_psi": 2.0}
TOL_NOISY = {"k_rel": 0.10, "skin_abs": 1.0, "pstar_psi": 3.0}


def _close(a, b, tol):
    return a is not None and abs(a - b) <= tol


def _t_conversions():
    assert abs(bar_to_psi(1.0) - 14.5037738) < 1e-12
    assert abs(m_to_ft(1.0) - 3.2808399) < 1e-12
    assert abs(m3_to_bbl(1.0) - 6.2898108) < 1e-12
    assert abs(c_to_f(100.0) - 212.0) < 1e-12
    assert abs(radius_ft_from_diameter_in(8.5) - 0.3541666667) < 1e-9
    assert abs(radius_ft_from_radius_in(4.25) - radius_ft_from_diameter_in(8.5)) < 1e-12


def _t_duplicates():
    t = "2026-01-01T00:00:00Z"
    pts = [PressurePoint(t, 100.0, "psia"), PressurePoint(t, 102.0, "psia"),
           PressurePoint("2026-01-01T00:00:01Z", 101.0, "psia"),
           PressurePoint("2026-01-01T00:00:02Z", 101.0, "psia")]  # repeat value != duplicate
    for pol, exp in (("mean", 101.0), ("first", 100.0), ("last", 102.0)):
        cl, _, _ = clean_pressure(pts, pol)
        assert len(cl) == 3 and cl[0][1] == exp, (pol, cl)
    cl, _, _ = clean_pressure(pts, "reject")
    assert len(cl) == 2
    cl, _, q = clean_pressure([PressurePoint(t, 1.0, "bar")], "mean")
    assert abs(cl[0][1] - PSI_PER_BAR) < 1e-12 and q.has("PRESSURE_BASIS_UNSPECIFIED")
    _, _, q = clean_pressure([PressurePoint(t, 1.0, "furlongs")], "mean")
    assert q.blocked("k_md")
    # LAS-style null markers and impossible absolute pressures are removed and audited
    cl, au, q = clean_pressure([PressurePoint(t, 3000.0, "psia"),
                                PressurePoint("2026-01-01T00:00:01Z", -999.25, "psia"),
                                PressurePoint("2026-01-01T00:00:02Z", -5.0, "psia"),
                                PressurePoint("2026-01-01T00:00:03Z", 3001.0, "psia")])
    assert [p for _, p in cl] == [3000.0, 3001.0] and q.has("NULL_SENTINELS_REMOVED")
    assert {d["reason"] for d in au["discarded"]} == {"NULL_SENTINEL",
                                                      "NONPOSITIVE_ABSOLUTE_PRESSURE"}


def _t_rates_and_periods():
    pvt = PhysicalInputs(oil_fvf=1.2, water_fvf=1.0)
    qr, _, _ = normalize_rates([RatePoint("2026-01-01T00:00:00Z", 100.0, "stb/day", 50.0)], pvt)
    assert abs(qr[0][1] - 170.0) < 1e-9
    q_eq = 100.0 + 50.0 * 1.0 / 1.2
    assert abs(1.2 * q_eq - 170.0) < 1e-9  # q_r = Bo * q_eq,o
    # history starts at first observation (10 h), not zero-filled before it
    rp = [RatePoint("2026-01-01T10:00:00Z", 100.0, "stb/day", 0.0),
          RatePoint("2026-01-01T20:00:00Z", 200.0, "stb/day", 0.0)]
    rates, _, _ = normalize_rates(rp, PhysicalInputs(oil_fvf=1.0))
    per, t0, q = build_rate_periods(rates, parse_utc("2026-01-02T06:00:00Z")[0])
    assert t0 == parse_utc("2026-01-01T10:00:00Z")[0] and per[0].start_time_hr == 0.0
    assert q.has("PRODUCING_HISTORY_START_ASSUMED") and len(per) == 2
    assert abs(equivalent_producing_time(per) - (100 * 10 + 200 * 10) / 200) < 1e-9
    _, _, q = normalize_rates(rp, PhysicalInputs())
    assert q.blocked("kh_md_ft")
    _, _, q = normalize_rates([RatePoint("2026-01-01T00:00:00Z", 1.0, "gal/min")],
                              PhysicalInputs(oil_fvf=1.0))
    assert q.blocked("kh_md_ft")


def _t_superposition_single_rate():
    tp = 50.0
    per = [RatePeriod(0.0, tp, 500.0, "x")]
    dts = [0.01, 0.1, 1.0, 10.0, 100.0]
    xs, xoff, _, _ = compute_superposition(dts + [1e9], per, tp)
    for d, x in zip(dts, xs):
        assert abs(x - math.log(d / (tp + d))) < 1e-12
        ta = tp * d / (tp + d)
        assert abs(x + xoff - math.log(ta)) < 1e-12
    assert abs(xs[-1]) < 1e-6  # X -> 0 as dt -> infinity


def _t_derivative_line():
    x = [0.0, 0.3, 0.35, 0.9, 1.4, 1.5, 2.2, 3.0, 3.1, 4.0]
    y = [3 * xi + 2 for xi in x]
    d = bourdet_derivative(x, y, [i * 0.05 for i in range(len(x))], 0.1, 1.0)
    vals = [v for v in d if v is not None]
    assert vals and all(abs(v - 3.0) < 1e-12 for v in vals)
    assert d[0] is None and d[-1] is None  # undefined, never zero


def _t_storage_hump_not_radial():
    # flat hump top (0.9 cycle at 200 psi), decline, then a shorter true plateau at 60 psi
    dt = [10 ** (-2 + i / 20) for i in range(80)]
    lg = [math.log10(d) for d in dt]
    der: List[Optional[float]] = []
    for l in lg:
        if l < -1.1:
            der.append(200.0)
        elif l < -0.1:
            der.append(200.0 * (60.0 / 200.0) ** ((l + 1.1) / 1.0))
        elif l < 0.6:
            der.append(60.0)
        else:
            der.append(30.0 * (1 + 0.01 * (l - 0.6)) if l < 0.65 else 25.0 / (l + 0.5))
    win, q = find_radial_interval(dt, der, AnalysisControls())
    assert win and abs(_median(der[win["a"]:win["b"] + 1]) - 60.0) < 1e-9, win
    assert q.has("STORAGE_HUMP_EXCLUDED")
    win, _ = find_radial_interval(dt, der, AnalysisControls(exclude_storage_hump=False))
    assert win and abs(_median(der[win["a"]:win["b"] + 1]) - 200.0) < 1e-9  # the failure mode


def _t_interval_union():
    assert union_intervals([(1050, 1080), (1020, 1060), (1100, 1110)]) == \
        [(1020, 1080), (1100, 1110)]
    agg = aggregate_open_interval_properties(
        [{"top": 1000, "base": 1100, "net_thickness_ft": 80, "porosity": 0.2,
          "water_saturation": 0.3, "permeability_md": 100}],
        [(1020, 1060), (1050, 1080)], c_f=4e-6, c_o=1e-5, c_w=3e-6)
    assert abs(agg["zone_fractions"][0] - 0.6) < 1e-12 and abs(agg["h_ft"] - 48) < 1e-9
    assert abs(agg["kh_reference_md_ft"] - 4800) < 1e-9
    assert abs(agg["ct_per_psi"] - (4e-6 + 0.7e-5 + 0.3 * 3e-6)) < 1e-18


def _t_numerics():
    assert abs(sum(w / (i + 1) for i, w in enumerate(stehfest_weights(12))) - 1.0) < 1e-9
    assert abs(stehfest_invert(lambda u: 1.0 / (u + 1.0), 1.0) - math.exp(-1.0)) < 1e-3
    k0, k1 = bessel_k01_scaled(1.0)
    assert abs(k0 * math.exp(-1) - 0.42102443824070834) < 1e-12
    assert abs(k1 * math.exp(-1) - 0.6019072301972346) < 1e-12
    k0, k1 = bessel_k01_scaled(0.01)
    assert abs(k0 * math.exp(-0.01) - 4.721244730161095) < 1e-10
    for td in (1e4, 1e6):
        assert abs(pwd(td, 0.0, 0.0) - 0.5 * (math.log(td) + RADIAL_CONST)) < 2e-3
    assert abs(pwd(1e6, 0.0, 2.0) - pwd(1e6, 0.0, 0.0) - 2.0) < 2e-3


def _run_case(case: SyntheticCase, controls: Optional[AnalysisControls] = None):
    pts, rates, shut, phys, truth = generate_synthetic_buildup(case)
    return analyze_buildup(pts, rates, shut, phys, controls), truth


def _check_recovery(res, truth, tol, check_c=True):
    r = res["results"]
    assert r["k_md"] is not None, res["qc_flags"]
    assert abs(r["k_md"] / truth["k_md"] - 1) <= tol["k_rel"], (r["k_md"], truth["k_md"])
    assert _close(r["apparent_skin"], truth["apparent_skin"], tol["skin_abs"]), \
        (r["apparent_skin"], truth["apparent_skin"])
    assert _close(r["extrapolated_pressure_psi"], truth["extrapolated_pressure_psi"],
                  tol["pstar_psi"]), r["extrapolated_pressure_psi"]
    if check_c:
        assert r["storage_bbl_per_psi"] is not None, res["qc_flags"]
        assert abs(r["storage_bbl_per_psi"] / truth["storage_bbl_per_psi"] - 1) <= tol["C_rel"], \
            (r["storage_bbl_per_psi"], truth["storage_bbl_per_psi"])


def _t_case_storage_skin():
    res, truth = _run_case(DEMO_CASES["storage_skin"])
    _check_recovery(res, truth, TOL)
    json.loads(json.dumps(res, allow_nan=False))
    assert res["status"] == STATUS


def _t_case_no_storage():
    res, truth = _run_case(DEMO_CASES["no_storage"])
    _check_recovery(res, truth, TOL, check_c=False)
    assert res["results"]["storage_bbl_per_psi"] is None


def _t_case_multirate():
    res, truth = _run_case(DEMO_CASES["multirate"])
    _check_recovery(res, truth, TOL)
    assert len(res["inputs"]["rate_source"]["periods"]) == 4  # zero-rate period preserved


def _t_case_noisy_irregular():
    res, truth = _run_case(DEMO_CASES["noisy"], AnalysisControls(smoothing_L10=0.2))
    _check_recovery(res, truth, TOL_NOISY, check_c=False)


def _t_case_short_no_radial():
    res, _ = _run_case(DEMO_CASES["short"])
    r = res["results"]
    assert r["kh_md_ft"] is None and r["apparent_skin"] is None
    assert any(f["code"] == "RADIAL_INTERVAL_NOT_FOUND" for f in res["qc_flags"])
    assert res["diagnostics"]["series"]["derivative_psi"]  # diagnostics survive the block


def _t_blocking():
    pts, rates, shut, phys, _ = generate_synthetic_buildup(DEMO_CASES["storage_skin"])
    # final rate zero: rate-scaled outputs blocked, derivative diagnostics kept
    zero = [RatePoint(r.timestamp_utc, 0.0, r.rate_unit, 0.0) for r in rates]
    res = analyze_buildup(pts, zero, shut, phys)
    assert all(res["results"][k] is None for k in ("kh_md_ft", "k_md", "storage_bbl_per_psi"))
    assert res["diagnostics"]["series"] is not None
    # missing thickness: kh reported, k suppressed
    phys_h = PhysicalInputs(**{**asdict(phys), "contributing_thickness_ft": None})
    res = analyze_buildup(pts, rates, shut, phys_h)
    assert res["results"]["kh_md_ft"] is not None and res["results"]["k_md"] is None
    # two shut-ins are never merged silently
    later = ShutIn(shut[0].end_timestamp,
                   _iso(parse_utc(shut[0].end_timestamp)[0] + timedelta(hours=10)))
    res = analyze_buildup(pts, rates, [shut[0], later], phys)
    assert any(f["code"] == "SHUT_IN_MULTIPLE_CANDIDATES" for f in res["qc_flags"])
    assert all(v is None for v in res["results"].values())
    # merging across a gap with production is refused
    gap_start = parse_utc(shut[0].end_timestamp)[0] + timedelta(minutes=30)
    late2 = ShutIn(_iso(gap_start + timedelta(minutes=20)), _iso(gap_start + timedelta(hours=5)))
    rates2 = rates + [RatePoint(_iso(gap_start), 800.0, "stb/day", 0.0)]
    res = analyze_buildup(pts, rates2, [shut[0], late2], phys,
                          AnalysisControls(merge_policy="merge"))
    assert any(f["code"] == "SHUT_IN_PRODUCTION_IN_GAP" for f in res["qc_flags"])


def _t_export_roundtrip(tmpdir: Optional[str] = None):
    import os
    import tempfile
    res, _ = _run_case(DEMO_CASES["short"])
    fd, path = tempfile.mkstemp(suffix=".json", dir=tmpdir)
    os.close(fd)
    try:
        export_result_json(res, path)
        with open(path, encoding="utf-8") as fh:
            back = json.load(fh)
        assert back == json.loads(json.dumps(res))
    finally:
        os.remove(path)


SELF_TESTS = [
    ("unit conversions / radius vs diameter", _t_conversions),
    ("duplicate handling and pressure units", _t_duplicates),
    ("oil/water rates, missing rates, periods", _t_rates_and_periods),
    ("single-rate superposition identities", _t_superposition_single_rate),
    ("Bourdet derivative of a straight line", _t_derivative_line),
    ("storage-hump top is not taken as radial", _t_storage_hump_not_radial),
    ("interval union and zone aggregation", _t_interval_union),
    ("Stehfest, Bessel K0/K1, line source", _t_numerics),
    ("synthetic: storage + skin", _t_case_storage_skin),
    ("synthetic: no storage, zero skin", _t_case_no_storage),
    ("synthetic: multi-rate with intermediate shut-in", _t_case_multirate),
    ("synthetic: noise + irregular sampling", _t_case_noisy_irregular),
    ("synthetic: short test, no radial flow", _t_case_short_no_radial),
    ("dependency blocking and shut-in merging", _t_blocking),
    ("export round-trip", _t_export_roundtrip),
]


def run_self_tests() -> bool:
    print("Running self-tests...")
    ok = True
    for name, fn in SELF_TESTS:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as e:
            ok = False
            print(f"  FAIL  {name}: {e}")
    print("All self-tests PASSED." if ok else "Some self-tests FAILED.")
    return ok


# ==============================================================================
# CLI
# ==============================================================================

def print_summary(res: Dict[str, Any], truth: Optional[Dict[str, float]] = None) -> None:
    print("=" * 78)
    print(f" Pressure build-up analysis  [{res['status']}]")
    si = res["inputs"]["shut_in_selection"]
    print(f" Shut-in reference: {si.get('reference_time_utc')}  p_wf = "
          f"{si.get('p_wf_psi') if si.get('p_wf_psi') is None else round(si['p_wf_psi'], 2)} psi")
    rad = res["diagnostics"]["radial_interval"]
    if rad:
        print(f" Radial window: {rad['dt_start_hr']:.4g} - {rad['dt_end_hr']:.4g} hr "
              f"({rad['log_cycles']:.2f} cycles, {rad['n_points']} pts, "
              f"sigma={rad['sigma_log10']:.4f}, {rad['selection']})")
    print("-" * 78)
    print(f" {'Result':<36}{'Value':>18}{'Truth':>18}")
    for k, v in res["results"].items():
        t = truth.get(k) if truth else None
        vs = "null" if v is None else f"{v:.6g}"
        ts = "" if t is None else f"{t:.6g}"
        print(f" {k:<36}{vs:>18}{ts:>18}")
    print("-" * 78)
    for f in res["qc_flags"]:
        if f["severity"] != "INFO":
            print(f" [{f['severity']}] {f['code']}: {f['message']}")
    print("=" * 78)


def main() -> int:
    ap = argparse.ArgumentParser(description="Preliminary pressure build-up analysis.")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--demo", choices=sorted(DEMO_CASES))
    ap.add_argument("--input", help="JSON input file (see load_input_json).")
    ap.add_argument("--json-out", help="Write the result object to this path.")
    args = ap.parse_args()
    if args.self_test:
        return 0 if run_self_tests() else 1
    if args.demo:
        case = DEMO_CASES[args.demo]
        ctrl = AnalysisControls(smoothing_L10=0.2) if args.demo == "noisy" else None
        res, truth = _run_case(case, ctrl)
        print_summary(res, truth)
    elif args.input:
        pts, rates, shut, phys, ctrl = load_input_json(args.input)
        res = analyze_buildup(pts, rates, shut, phys, ctrl)
        print_summary(res)
    else:
        ap.print_help()
        return 2
    if args.json_out:
        export_result_json(res, args.json_out)
        print(f" Result written to {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
