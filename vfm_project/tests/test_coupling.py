"""Tests for the PTA -> VFM coupling layer (guardrails and J-multiplier physics)."""

from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vfm.config import deep_merge  # noqa: E402
from vfm.coupling import (  # noqa: E402
    CouplingSettings,
    EventDecision,
    PropertyChange,
    build_ipr_adjustments,
    default_decision,
)
from vfm.io.loaders import apply_column_map  # noqa: E402

REF = pd.Timestamp("2025-01-01", tz="UTC")
PVT = pd.DataFrame({"well_name": ["W"], "wellbore_radius_ft": [0.33], "estimated_drainage_area_acres": [160.0]})


def _event(status="RADIAL", skin=2.0, kh=3000.0, dietz=3300.0, p_last=3250.0, start="2025-03-01T00:00:00+00:00"):
    return {"event_id": f"W@{start[:16]}", "well_name": "W", "shut_in_start": start, "status": status,
            "derived": {"dietz_pbar_datum_psia": dietz, "p_last_datum_psia": p_last},
            "result": {"results": {"apparent_skin": skin, "kh_md_ft": kh}}}


class TestCoupling(unittest.TestCase):
    def test_disabled_means_no_adjustment(self) -> None:
        ev = _event()
        st = CouplingSettings(enabled=False, decisions={ev["event_id"]: EventDecision()})
        adj, notes = build_ipr_adjustments({"W": [ev]}, st, PVT, pd.DataFrame(), REF)
        self.assertEqual(adj, {})
        self.assertIn("disabled", notes[0]["message"])

    def test_default_decision_is_conservative(self) -> None:
        cfg = {"apply_pressure_default": True, "apply_skin_default": True}
        bad = default_decision(_event(status="NO_RADIAL", dietz=None), cfg)
        self.assertFalse(bad.use)
        self.assertFalse(bad.apply_pressure)
        self.assertFalse(bad.apply_skin)
        good = default_decision(_event(), cfg)
        self.assertTrue(good.use and good.apply_pressure and good.apply_skin)
        self.assertEqual(good.pressure_choice, "dietz")

    def test_skin_change_scales_j_by_pss_ratio(self) -> None:
        ev1, ev2 = _event(skin=0.0), _event(skin=5.0, start="2025-06-01T00:00:00+00:00")
        st = CouplingSettings(enabled=True, decisions={ev1["event_id"]: EventDecision(apply_pressure=False),
                                                        ev2["event_id"]: EventDecision(apply_pressure=False)})
        adj, _ = build_ipr_adjustments({"W": [ev1, ev2]}, st, PVT, pd.DataFrame(), REF)
        fn = adj["W"]["j_multiplier"]
        re = math.sqrt(160 * 43560 / math.pi)
        d0, d5 = math.log(re / 0.33) - 0.75, math.log(re / 0.33) - 0.75 + 5.0
        t_before, t_after = (pd.Timestamp("2025-04-01", tz="UTC") - REF).days, (pd.Timestamp("2025-07-01", tz="UTC") - REF).days
        np.testing.assert_allclose(fn(np.array([t_before, t_after])), [1.0, d0 / d5], rtol=1e-9)

    def test_single_observation_changes_nothing(self) -> None:
        ev = _event()
        st = CouplingSettings(enabled=True, decisions={ev["event_id"]: EventDecision(apply_pressure=False)})
        adj, notes = build_ipr_adjustments({"W": [ev]}, st, PVT, pd.DataFrame(), REF)
        self.assertNotIn("W", adj)
        self.assertTrue(any("reference" in n["message"].lower() for n in notes))

    def test_skin_and_kh_changes_are_independent(self) -> None:
        # PTA gives skin only; a later engineer change gives kh only: the kh change must not be lost
        ev = _event(skin=2.0, kh=3000.0)
        st = CouplingSettings(enabled=True, decisions={ev["event_id"]: EventDecision(apply_pressure=False, apply_kh=True)},
                              property_changes=[PropertyChange("W", "2025-06-01", kh_md_ft=1500.0)])
        adj, _ = build_ipr_adjustments({"W": [ev]}, st, PVT, pd.DataFrame(), REF)
        np.testing.assert_allclose(adj["W"]["j_multiplier"](np.array([100.0, 200.0])), [1.0, 0.5])

    def test_invalid_values_never_become_the_reference(self) -> None:
        ev1, ev2 = _event(skin=float("nan")), _event(skin=1.0, start="2025-06-01T00:00:00+00:00")
        ch = PropertyChange("W", "2025-08-01", skin=4.0)
        st = CouplingSettings(enabled=True, decisions={ev1["event_id"]: EventDecision(apply_pressure=False),
                                                        ev2["event_id"]: EventDecision(apply_pressure=False)},
                              property_changes=[ch, PropertyChange("W", "2025-07-01", kh_md_ft=-5.0)])
        adj, notes = build_ipr_adjustments({"W": [ev1, ev2]}, st, PVT, pd.DataFrame(), REF)
        fn = adj["W"]["j_multiplier"]
        re = math.sqrt(160 * 43560 / math.pi)
        lt = math.log(re / 0.33) - 0.75
        np.testing.assert_allclose(fn(np.array([200.0, 250.0])), [1.0, (lt + 1.0) / (lt + 4.0)])  # ref = skin 1, not NaN
        self.assertEqual(sum(1 for n in notes if n["level"] == "BLOCK"), 2)  # NaN skin and kh <= 0

    def test_stimulated_well_negative_skin_is_allowed(self) -> None:
        ev1, ev2 = _event(skin=-6.0), _event(skin=-2.0, start="2025-06-01T00:00:00+00:00")
        st = CouplingSettings(enabled=True, decisions={e["event_id"]: EventDecision(apply_pressure=False) for e in (ev1, ev2)})
        adj, notes = build_ipr_adjustments({"W": [ev1, ev2]}, st, PVT, pd.DataFrame(), REF)
        self.assertIn("W", adj)
        self.assertLess(adj["W"]["j_multiplier"](np.array([200.0]))[0], 1.0)

    def test_pressure_anchor_upper_bound(self) -> None:
        pvt = PVT.assign(datum_tvd_ft=8000.0)
        ev = _event(dietz=25000.0)
        st = CouplingSettings(enabled=True, decisions={ev["event_id"]: EventDecision(apply_skin=False)})
        adj, notes = build_ipr_adjustments({"W": [ev]}, st, pvt, pd.DataFrame(), REF)
        self.assertNotIn("W", adj)
        self.assertTrue(any(n["level"] == "BLOCK" and "exceeds" in n["message"] for n in notes))

    def test_forced_window_is_not_applied_by_default(self) -> None:
        d = default_decision(_event(status="RADIAL_FORCED"), {"apply_pressure_default": True})
        self.assertFalse(d.use)


class TestPtaRateHistory(unittest.TestCase):
    def _daily(self, uptime_last):
        days = pd.date_range("2025-03-01", periods=4, freq="D", tz="UTC")
        return pd.DataFrame({"date": days, "oil_rate_alloc_stbd": [1000.0, 1000.0, 600.0, 1000.0 * uptime_last],
                             "water_rate_alloc_stbd": [0.0, 0.0, 0.0, 0.0], "uptime_frac": [1.0, 1.0, 0.6, uptime_last]})

    def test_final_rate_not_zeroed_by_early_shut_in(self) -> None:
        from vfm.pta.runner import _rate_points_from_daily
        shut = pd.Timestamp("2025-03-04 00:20", tz="UTC")       # shut-in day has ~1 % uptime
        pts = _rate_points_from_daily(self._daily(0.014), shut)
        self.assertAlmostEqual(pts[-1].oil_surface_rate, 1000.0)  # on-stream rate of 03-03 (600 / 0.6)
        self.assertAlmostEqual(pts[-2].oil_surface_rate, 600.0)   # whole day before: volume-correct average

    def test_shut_in_at_midnight_carries_q_n_on_previous_day(self) -> None:
        from vfm.pta.runner import _rate_points_from_daily
        pts = _rate_points_from_daily(self._daily(0.0), pd.Timestamp("2025-03-04 00:00", tz="UTC"))
        self.assertEqual(len(pts), 3)
        self.assertAlmostEqual(pts[-1].oil_surface_rate, 1000.0)

    def test_engineer_change_with_naive_date_and_clipping(self) -> None:
        ev = _event(skin=0.0)
        st = CouplingSettings(enabled=True, max_j_multiplier=1.2,
                              decisions={ev["event_id"]: EventDecision(apply_pressure=False)},
                              property_changes=[PropertyChange("W", "2025-05-01", skin=-4.0)])
        adj, notes = build_ipr_adjustments({"W": [ev]}, st, PVT, pd.DataFrame(), REF)
        m_after = adj["W"]["j_multiplier"](np.array([200.0]))[0]
        self.assertAlmostEqual(m_after, 1.2)  # clipped
        self.assertTrue(any(n["level"] == "WARN" and "clipped" in n["message"] for n in notes))

    def test_non_radial_event_cannot_push_skin_without_override(self) -> None:
        ev1, ev2 = _event(skin=0.0), _event(status="NO_RADIAL", skin=None, start="2025-06-01T00:00:00+00:00")
        st = CouplingSettings(enabled=True, decisions={ev1["event_id"]: EventDecision(apply_pressure=False),
                                                        ev2["event_id"]: EventDecision(apply_pressure=False)})
        adj, notes = build_ipr_adjustments({"W": [ev1, ev2]}, st, PVT, pd.DataFrame(), REF)
        self.assertNotIn("W", adj)
        self.assertTrue(any("No credible radial flow" in n["message"] for n in notes))

    def test_pressure_anchor_below_flowing_pressure_is_rejected(self) -> None:
        ev = _event(dietz=2500.0)
        st = CouplingSettings(enabled=True, decisions={ev["event_id"]: EventDecision(apply_skin=False)})
        pwf = {"W": pd.Series([2600.0], index=pd.DatetimeIndex([pd.Timestamp("2025-03-10", tz="UTC")]))}
        adj, notes = build_ipr_adjustments({"W": [ev]}, st, PVT, pd.DataFrame(), REF, pwf)
        self.assertNotIn("W", adj)
        self.assertTrue(any(n["level"] == "BLOCK" for n in notes))

    def test_settings_round_trip(self) -> None:
        st = CouplingSettings(enabled=True, decisions={"W@x": EventDecision(skin_override=3.0)},
                              property_changes=[PropertyChange("W", "2025-05-01", skin=1.0)],
                              estimators={"W": {"ipr": False}})
        back = CouplingSettings.from_dict(st.to_dict())
        self.assertEqual(back, st)


class TestGeneralization(unittest.TestCase):
    def test_column_map_and_missing_column_message(self) -> None:
        df = pd.DataFrame({"t": [1], "well": ["A"], "PDG": [3000.0], "THP": [300.0], "THP_ds": [200.0], "GL": [500.0]})
        out = apply_column_map(df, {"timestamp": "t", "well_name": "well", "dh_pressure_psia": "PDG",
                                    "whp_psia": "THP", "whp_downstream_psia": "THP_ds", "gas_lift_rate_mscfd": "GL"},
                               "high_freq")
        self.assertIn("dh_pressure_psia", out.columns)
        with self.assertRaises(KeyError) as ctx:
            apply_column_map(df, {}, "high_freq")
        self.assertIn("schema.high_freq", str(ctx.exception))

    def test_deep_merge_overrides_only_given_keys(self) -> None:
        base = {"paths": {"a": "1", "b": "2"}, "x": 1}
        self.assertEqual(deep_merge(base, {"paths": {"b": "3"}}), {"paths": {"a": "1", "b": "3"}, "x": 1})


if __name__ == "__main__":
    unittest.main()
