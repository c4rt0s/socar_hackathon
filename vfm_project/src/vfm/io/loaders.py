"""Raw CSV loaders, schema validation, and well-test harmonization."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Optional, Tuple

import numpy as np
import pandas as pd


HF_NUMERIC_COLS = (
    "whp_psia",
    "whp_downstream_psia",
    "dh_pressure_psia",
    "dh_temperature_degf",
    "annulus_pressure_psia",
    "gas_lift_rate_mscfd",
)

SIGN_CHECK_COLS = (
    "gas_lift_rate_mscfd",
    "oil_rate_stbd",
    "water_rate_stbd",
    "gas_rate_mscfd",
    "total_gas_rate_mscfd",
    "liquid_rate_stbd",
)


REQUIRED_COLUMNS: Dict[str, Tuple[str, ...]] = {
    "high_freq": ("timestamp", "well_name", "whp_psia", "whp_downstream_psia", "dh_pressure_psia",
                  "gas_lift_rate_mscfd"),
    # Core identity/rate columns; the physics calibration additionally reads the test-averaged
    # sensors (whp, whp_downstream, dh_pressure, gas_lift_rate) and fluid properties.
    "well_tests": ("well_name", "test_start", "test_end", "oil_rate_stbd", "water_rate_stbd",
                   "gas_rate_mscfd", "gor_scf_stb"),
    "choke_events": ("timestamp", "well_name", "choke_opening_pct"),
    "well_completion": ("well_name", "gauge_tvd_ft", "gl_valve_tvd_ft", "tubing_id_in", "lift_gas_sg",
                        "solution_gor_at_pb_scf_stb", "water_fvf_rb_stb", "choke_max_size_64ths"),
    "reservoir_pvt": ("well_name", "net_pay_thickness_ft", "porosity_frac", "total_compressibility_1_psi",
                      "wellbore_radius_ft", "oil_fvf_rb_stb", "oil_viscosity_cp",
                      "bubble_point_pressure_psia", "reservoir_temperature_degf", "datum_tvd_ft"),
}


def apply_column_map(df: pd.DataFrame, column_map: Optional[Mapping[str, str]], table: str) -> pd.DataFrame:
    """Rename source columns to canonical names ({canonical: source}) and check required ones."""
    if column_map:
        rename = {src: canon for canon, src in column_map.items() if not canon.startswith("_") and src in df.columns}
        df = df.rename(columns=rename)
    missing = [c for c in REQUIRED_COLUMNS.get(table, ()) if c not in df.columns]
    if missing:
        raise KeyError(
            f"{table}: missing required columns {missing}. Map them in the config 'schema.{table}' "
            f"section as {{canonical_name: your_column_name}}."
        )
    return df


def load_static_tables(
    completion_path: Path,
    pvt_path: Path,
    choke_events_path: Path,
    schema: Optional[Mapping[str, Mapping[str, str]]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load completion, PVT, and choke event tables with parsed timestamps."""
    schema = schema or {}
    completion = apply_column_map(pd.read_csv(completion_path), schema.get("well_completion"), "well_completion")
    if "first_production_date" in completion.columns:
        completion["first_production_date"] = pd.to_datetime(completion["first_production_date"], utc=True)

    pvt = apply_column_map(pd.read_csv(pvt_path), schema.get("reservoir_pvt"), "reservoir_pvt")

    choke_events = apply_column_map(pd.read_csv(choke_events_path), schema.get("choke_events"), "choke_events")
    choke_events["timestamp"] = pd.to_datetime(choke_events["timestamp"], utc=True)
    choke_events = choke_events.sort_values(["well_name", "timestamp"]).reset_index(drop=True)
    return completion, pvt, choke_events


def load_and_harmonize_well_tests(
    well_tests_path: Path,
    sm3_per_mscf: float = 28.3168466,
    unit_mismatch_ratio_threshold: float = 15.0,
    column_map: Optional[Mapping[str, str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load well_tests.csv, repair sign-flipped rate/sensor entries, and normalize
    wells whose gas_rate/total_gas_rate were logged in Sm3/d instead of Mscf/d.
    """
    wt = apply_column_map(pd.read_csv(well_tests_path), column_map, "well_tests")
    wt["test_start"] = pd.to_datetime(wt["test_start"], utc=True)
    wt["test_end"] = pd.to_datetime(wt["test_end"], utc=True)

    audit: Dict[str, Any] = {"raw_rows": int(len(wt)), "sign_fixes": [], "unit_conversions": []}

    # 1. Repair sign-flipped positive physical measurements
    for col in SIGN_CHECK_COLS:
        if col not in wt.columns:
            continue
        neg_mask = wt[col] < 0
        if neg_mask.any():
            for idx in wt.index[neg_mask]:
                audit["sign_fixes"].append({
                    "well_name": wt.at[idx, "well_name"],
                    "test_start": str(wt.at[idx, "test_start"]),
                    "column": col,
                    "raw_value": float(wt.at[idx, col]),
                    "fixed_value": float(abs(wt.at[idx, col])),
                })
            wt.loc[neg_mask, col] = wt.loc[neg_mask, col].abs()

    # 2. Detect & fix per-well Sm3/d vs Mscf/d gas unit mismatches using GOR identity
    expected_form_gas_mscfd = (wt["oil_rate_stbd"] * wt["gor_scf_stb"]) / 1000.0
    ratio = wt["gas_rate_mscfd"] / expected_form_gas_mscfd.replace(0.0, np.nan)
    wt["_gas_ratio"] = ratio

    for well_name, grp in wt.groupby("well_name"):
        med_ratio = float(grp["_gas_ratio"].median())
        if np.isfinite(med_ratio) and med_ratio > unit_mismatch_ratio_threshold:
            mask = wt["well_name"] == well_name
            wt.loc[mask, "gas_rate_mscfd"] = wt.loc[mask, "gas_rate_mscfd"] / sm3_per_mscf
            wt.loc[mask, "total_gas_rate_mscfd"] = wt.loc[mask, "total_gas_rate_mscfd"] / sm3_per_mscf
            audit["unit_conversions"].append({
                "well_name": well_name,
                "median_raw_to_gor_ratio": round(med_ratio, 4),
                "divisor_applied": sm3_per_mscf,
                "columns": ["gas_rate_mscfd", "total_gas_rate_mscfd"],
            })

    wt = wt.drop(columns=["_gas_ratio"]).sort_values(["well_name", "test_start"]).reset_index(drop=True)
    return wt, audit


def iter_high_freq_chunks(
    high_freq_csv: Path,
    chunksize: int = 500_000,
    column_map: Optional[Mapping[str, str]] = None,
) -> Iterator[pd.DataFrame]:
    """Stream raw high-frequency telemetry CSV in memory-bounded chunks (canonical names)."""
    cmap = {k: v for k, v in (column_map or {}).items() if not k.startswith("_")}
    src_name = lambda canon: cmap.get(canon, canon)
    dtypes = {src_name("well_name"): "string", **{src_name(c): "float64" for c in HF_NUMERIC_COLS}}
    for chunk in pd.read_csv(high_freq_csv, chunksize=chunksize, dtype=dtypes):
        chunk = apply_column_map(chunk, cmap, "high_freq")
        for col in HF_NUMERIC_COLS:  # optional sensors (e.g. annulus, temperature) may be absent
            if col not in chunk.columns:
                chunk[col] = float("nan")
        yield chunk


def _has_parquet_engine() -> bool:
    for mod in ("pyarrow", "fastparquet"):
        try:
            __import__(mod)
            return True
        except ImportError:
            continue
    return False


def save_frame(df: pd.DataFrame, stem_path: Path) -> Path:
    """Persist DataFrame as Parquet if pyarrow/fastparquet exists, else Pickle."""
    stem_path.parent.mkdir(parents=True, exist_ok=True)
    if _has_parquet_engine():
        out = stem_path.with_suffix(".parquet")
        df.to_parquet(out, index=False)
        return out
    out = stem_path.with_suffix(".pkl")
    df.to_pickle(out)
    return out


def load_frame(stem_path: Path) -> pd.DataFrame:
    """Load DataFrame persisted via save_frame (.parquet preferred over .pkl)."""
    pq = stem_path.with_suffix(".parquet")
    if pq.exists() and _has_parquet_engine():
        return pd.read_parquet(pq)
    pkl = stem_path.with_suffix(".pkl")
    if pkl.exists():
        return pd.read_pickle(pkl)
    raise FileNotFoundError(f"Neither {pq} nor {pkl} exists.")

