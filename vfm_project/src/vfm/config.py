"""Configuration loading and project path resolution.

`configs/default.json` holds every setting. A dataset-specific config only needs the keys
it changes: `load_config(path)` deep-merges it over the defaults. The config path can also
come from the `VFM_CONFIG` environment variable.
"""

from __future__ import annotations

from dataclasses import dataclass
import copy
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.json"
RAW_DATA_KEYS = ("high_freq_csv", "well_tests_csv", "choke_events_csv", "well_completion_csv", "reservoir_pvt_csv")


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge `override` into a copy of `base` (dicts merge, everything else replaces)."""
    out = copy.deepcopy(base)
    for key, val in override.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


@dataclass(frozen=True)
class VFMConfig:
    """Typed wrapper around the merged JSON configuration."""

    raw: Dict[str, Any]
    project_root: Path = PROJECT_ROOT
    source_path: Optional[Path] = None

    def _data_dir_candidates(self) -> list[Path]:
        """Folders searched for the raw CSVs when the configured path does not exist.

        `VFM_DATA_DIR` (env var) wins; then the usual hackathon layouts next to / inside the project."""
        cands: list[Path] = []
        env = os.environ.get("VFM_DATA_DIR")
        if env:
            cands.append(Path(env))
        root = self.project_root
        cands += [root.parent / "data" / "raw", root.parent / "data", root / "data" / "raw", root / "data",
                  root.parent, root.parent / "pta_project", Path.home() / "Desktop" / "SOCAR HACKATHON"]
        return cands

    def resolve_path(self, key: str) -> Path:
        """Resolve a configured path relative to the directory of the config's project root.

        Raw-data CSVs fall back to `VFM_DATA_DIR` and common data folders (see
        `_data_dir_candidates`) so the project runs wherever the five hackathon CSVs are placed."""
        p = Path(self.raw["paths"][key])
        resolved = p if p.is_absolute() else (self.project_root / p).resolve()
        env = os.environ.get("VFM_DATA_DIR")
        if key == "raw_data_dir" and env:
            return Path(env).resolve()
        if key in RAW_DATA_KEYS:
            if env and (Path(env) / resolved.name).exists():
                return (Path(env) / resolved.name).resolve()
            if not resolved.exists():
                for d in self._data_dir_candidates():
                    alt = d / resolved.name
                    if alt.exists():
                        return alt.resolve()
        return resolved

    def section(self, name: str) -> Dict[str, Any]:
        return self.raw.get(name, {})

    @property
    def cleaning(self) -> Dict[str, Any]:
        return self.raw["cleaning"]

    @property
    def states(self) -> Dict[str, Any]:
        return self.raw["states"]

    @property
    def well_tests(self) -> Dict[str, Any]:
        return self.raw["well_tests"]

    @property
    def modeling(self) -> Dict[str, Any]:
        return self.raw["modeling"]

    @property
    def schema(self) -> Dict[str, Dict[str, str]]:
        """Per-file column maps {canonical_name: source_column_name}."""
        return self.raw.get("schema", {})

    @property
    def assumptions(self) -> Dict[str, Any]:
        return self.raw.get("assumptions", {})

    def reference_timestamp(self, fallback: Optional[pd.Timestamp] = None) -> pd.Timestamp:
        """Time origin for Pr(t) = Pr0 - a*t. "auto" uses `fallback` (first data day)."""
        val = self.modeling.get("reference_timestamp", "auto")
        if val in (None, "", "auto"):
            if fallback is None:
                raise ValueError("reference_timestamp is 'auto' but no fallback timestamp was given")
            return pd.Timestamp(fallback).tz_convert("UTC").floor("D") if pd.Timestamp(fallback).tzinfo \
                else pd.Timestamp(fallback).tz_localize("UTC").floor("D")
        ts = pd.Timestamp(val)
        return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def load_config(config_path: str | Path | None = None) -> VFMConfig:
    """Load defaults, then deep-merge a dataset config (argument or $VFM_CONFIG) over them."""
    with DEFAULT_CONFIG_PATH.open("r", encoding="utf-8-sig") as f:
        data = json.load(f)
    path = config_path or os.environ.get("VFM_CONFIG")
    source = DEFAULT_CONFIG_PATH
    if path:
        p = Path(path).resolve()
        if p != DEFAULT_CONFIG_PATH.resolve():
            with p.open("r", encoding="utf-8-sig") as f:  # tolerate a BOM (Notepad/PowerShell)
                data = deep_merge(data, json.load(f))
            source = p
    return VFMConfig(raw=data, project_root=PROJECT_ROOT, source_path=source)
