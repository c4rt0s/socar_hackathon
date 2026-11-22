"""Tests for the state/reconciler backports, PTA methods, deconvolution, model zoo and PI link."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vfm.preprocessing.states import classify_well_states  # noqa: E402


class TestStatesBackport(unittest.TestCase):
    def test_low_dp_at_open_choke_is_stall_not_shut_in(self) -> None:
        ts = pd.date_range("2025-09-06 08:00", periods=30, freq="1min", tz="UTC")
        whp = np.full(30, 207.0)
        whp[20:25] = 194.0                       # dp_choke ~1 psi at 50 % choke (GL-03 case)
        df = pd.DataFrame({"timestamp": ts, "well_name": "W", "whp_psia": whp, "whp_downstream_psia": 193.0})
        ev = pd.DataFrame({"timestamp": [pd.Timestamp("2025-01-01", tz="UTC")], "well_name": ["W"], "choke_opening_pct": [50.0]})
        out = classify_well_states(df, ev)
        self.assertFalse(out["is_shut_in"].any())
        self.assertTrue(out["is_flow_stall"].iloc[20:25].all())
        self.assertTrue(out["is_transient"].iloc[20:25].all())

    def test_no_backfill_before_first_choke_event(self) -> None:
        ts = pd.date_range("2025-01-01 00:00", periods=10, freq="1min", tz="UTC")
        df = pd.DataFrame({"timestamp": ts, "well_name": "W", "whp_psia": 300.0, "whp_downstream_psia": 200.0})
        ev = pd.DataFrame({"timestamp": [ts[5]], "well_name": ["W"], "choke_opening_pct": [0.0]})
        out = classify_well_states(df, ev)
        self.assertTrue(out["choke_opening_pct"].iloc[:5].isna().all())   # future choke not borrowed
        self.assertFalse(out["is_shut_in"].iloc[:5].any())
        self.assertTrue(out["is_state_unknown"].iloc[:5].all())
        self.assertTrue(out["is_shut_in"].iloc[5:].all())


class TestReconcilerBackport(unittest.TestCase):
    def test_calendar_complete_and_exact_window_anchor(self) -> None:
        from vfm.allocation.reconciler import aggregate_daily_vfm, anchor_daily_to_well_tests
        ts = pd.date_range("2025-01-01", "2025-01-04 23:59", freq="1min", tz="UTC")
        ts = ts[(ts < "2025-01-02") | (ts >= "2025-01-03")]          # whole day 2 missing (SCADA outage)
        n = len(ts)
        preds = pd.DataFrame({"timestamp": ts, "well_name": "W", "is_shut_in": False, "oil_rate_stbd": 1000.0,
                              "water_rate_stbd": 500.0, "gas_rate_mscfd": 300.0, "total_gas_rate_mscfd": 1100.0,
                              "liquid_rate_stbd": 1500.0})
        daily = aggregate_daily_vfm(preds)
        self.assertEqual(len(daily), 4)
        self.assertTrue(bool(daily.loc[1, "is_imputed_day"]))
        self.assertAlmostEqual(float(daily.loc[1, "data_completeness"]), 0.0)
        # test spanning midnight: 23:00 day 3 -> 11:00 day 4
        wt = pd.DataFrame({"well_name": ["W"], "test_start": [pd.Timestamp("2025-01-03 23:00", tz="UTC")],
                           "test_end": [pd.Timestamp("2025-01-04 11:00", tz="UTC")], "oil_rate_stbd": [1100.0],
                           "water_rate_stbd": [500.0], "gas_rate_mscfd": [330.0]})
        out = anchor_daily_to_well_tests(daily, wt)
        self.assertAlmostEqual(float(out["oil_rate_alloc_stbd"].iloc[-1]), 1100.0, places=6)
        self.assertIn("total_gas_rate_alloc_mscfd", out.columns)
        self.assertAlmostEqual(float(out["total_gas_rate_alloc_mscfd"].iloc[-1]), 330.0 + 800.0, places=6)


class TestPtaMethodsAndDeconvolution(unittest.TestCase):
    def test_methods_pwd_matches_kernel(self) -> None:
        from vfm.pta import kernel as K
        from vfm.pta import methods as M
        for td, cd, s in ((10.0, 500.0, 3.0), (1e3, 500.0, 3.0), (1e5, 0.0, -1.0), (50.0, 200.0, -1.0)):
            self.assertAlmostEqual(float(M.pwd(np.array([td]), cd, s)[0]), K.pwd(td, cd, s), places=5)

    def test_model_match_and_horner_recover_synthetic(self) -> None:
        from vfm.pta import kernel as K
        from vfm.pta import methods as M
        case = K.DEMO_CASES["storage_skin"]
        pts, rates, shut, phys, truth = K.generate_synthetic_buildup(case)
        res = K.analyze_buildup(pts, rates, shut, phys)
        ser = res["diagnostics"]["series"]
        dt, dp = np.array(ser["dt_hr"]), np.array(ser["dp_psi"])
        mm = M.model_match(dt, dp, 200.0, 1000.0, case.viscosity_cp, case.h_ft, case.porosity, case.ct_per_psi,
                           case.rw_ft, k0_md=20, skin0=0, c0_bbl_psi=0.02)
        self.assertAlmostEqual(mm["k_md"] / 50.0, 1.0, delta=0.02)
        self.assertAlmostEqual(mm["skin"], 3.0, delta=0.2)
        pi = M.productivity_indices(1000.0, 1250.0, 3000.0, 3100.0)
        self.assertIsNone(pi["pi_actual_stbd_psi"])

    def test_unit_response_matches_kernel_and_deconvolution_recovers_rates(self) -> None:
        from vfm.allocation.deconvolution import deconvolve_rates, radial_unit_response
        from vfm.pta.kernel import pwd
        k, h, mu, phi, ct, rw, S = 50.0, 40.0, 1.2, 0.2, 1.2e-5, 0.354, 3.0
        g = radial_unit_response(k * h, h, mu, phi, ct, rw, S)
        tdc = 0.0002637 * k / (phi * mu * ct * rw ** 2)
        for t in (1.0, 100.0):
            self.assertAlmostEqual(float(g(np.array([t]))[0]), 141.2 * mu / (k * h) * pwd(tdc * t, 0.0, S), places=4)
        # infinite-acting synthetic: rates 1000 -> 0 (shut-in) -> 1500, recovered with p_init unknown
        edges = np.arange(0, 241, 12.0)
        q = np.r_[np.full(8, 1000.0), np.zeros(2), np.full(10, 1500.0)]
        t_obs = np.arange(0.5, 240, 0.5)
        G = np.array([[g(np.array([t - a]))[0] - g(np.array([t - b]))[0] if t > a else 0.0
                       for a, b in zip(edges[:-1], edges[1:])] for t in t_obs])
        p = 4000.0 - G @ q
        res = deconvolve_rates(t_obs, p, edges, g, fixed_zero=q == 0, smoothing=1e-6)
        self.assertAlmostEqual(float(np.mean(res["q"][q > 0])), float(np.mean(q[q > 0])), delta=20.0)
        self.assertAlmostEqual(res["p_init"], 4000.0, delta=2.0)


class TestModelZoo(unittest.TestCase):
    def _hf(self):
        ts = pd.date_range("2025-03-01", periods=600, freq="1min", tz="UTC")
        choke = np.where(np.arange(600) < 300, 40.0, 0.0)
        return pd.DataFrame({"timestamp": ts, "well_name": "W", "choke_opening_pct": choke, "whp_psia": 300.0,
                             "whp_downstream_psia": 200.0, "dh_pressure_psia": 2500.0, "gas_lift_rate_mscfd": 800.0})

    def test_every_model_zeroes_shut_in_minutes(self) -> None:
        from vfm.allocation.models import TestBasedModel
        wt = pd.DataFrame({"well_name": ["W"], "test_start": [pd.Timestamp("2025-02-01", tz="UTC")],
                           "test_end": [pd.Timestamp("2025-02-01 12:00", tz="UTC")], "oil_rate_stbd": [1000.0],
                           "water_rate_stbd": [500.0], "gas_rate_mscfd": [300.0], "liquid_rate_stbd": [1500.0]})
        for mode in ("step", "linear"):
            m = TestBasedModel(mode).fit(wt, {}, pd.DataFrame(), pd.DataFrame())
            p = m.predict(self._hf(), "W")
            self.assertTrue((p["liquid_rate_stbd"].iloc[300:] == 0).all())
            self.assertTrue((p["total_gas_rate_mscfd"].iloc[300:] == 0).all())
            self.assertAlmostEqual(float(p["oil_rate_stbd"].iloc[0]), 1000.0)
            self.assertAlmostEqual(float(p["total_gas_rate_mscfd"].iloc[0]), 1100.0)

    def test_registry_lists_all_families(self) -> None:
        from vfm.allocation.models import list_allocation_models
        fam = {m["family"] for m in list_allocation_models()}
        self.assertTrue({"physics", "test-based", "regression/ML", "hybrid", "convolution"} <= fam)


class TestPiLink(unittest.TestCase):
    def test_apply_pi_fixes_reference_j(self) -> None:
        from vfm.coupling import CouplingSettings, EventDecision, build_ipr_adjustments
        ev = {"event_id": "W@x", "well_name": "W", "shut_in_start": "2025-03-01T00:00:00+00:00", "status": "RADIAL",
              "derived": {}, "result": {"results": {"apparent_skin": 1.0, "kh_md_ft": 2000.0}},
              "methods": {"productivity": {"pi_actual_stbd_psi": 3.2, "pi_theoretical_stbd_psi": 3.0}}}
        pvt = pd.DataFrame({"well_name": ["W"], "wellbore_radius_ft": [0.33], "estimated_drainage_area_acres": [160.0]})
        st = CouplingSettings(enabled=True, decisions={"W@x": EventDecision(apply_pressure=False, apply_skin=False, apply_pi=True)})
        adj, _ = build_ipr_adjustments({"W": [ev]}, st, pvt, pd.DataFrame(), pd.Timestamp("2025-01-01", tz="UTC"))
        self.assertAlmostEqual(adj["W"]["fixed_j"], 3.2)
        st.decisions["W@x"].pi_choice = "manual"
        st.decisions["W@x"].pi_override = 500.0   # out of range -> blocked
        adj, notes = build_ipr_adjustments({"W": [ev]}, st, pvt, pd.DataFrame(), pd.Timestamp("2025-01-01", tz="UTC"))
        self.assertNotIn("W", adj)
        self.assertTrue(any(n["level"] == "BLOCK" for n in notes))

    def test_fixed_j_is_used_by_the_ipr_fit(self) -> None:
        from vfm.physics.reservoir import fit_pr_and_pi
        ts = pd.date_range("2025-01-10", periods=6, freq="30D", tz="UTC")
        wt = pd.DataFrame({"well_name": "W", "test_start": ts, "test_end": ts + pd.Timedelta("12h"),
                           "dh_pressure_psia": [2600.0, 2580, 2610, 2590, 2570, 2600], "liquid_rate_stbd": 1500.0,
                           "water_cut_pct": 30.0, "oil_gravity_api": 30.0, "gas_sg": 0.75, "water_salinity_ppm": 1e5})
        pvt = pd.DataFrame({"well_name": ["W"], "datum_tvd_ft": [8200.0], "bubble_point_pressure_psia": [1300.0], "oil_fvf_rb_stb": [1.15]})
        comp = pd.DataFrame({"well_name": ["W"], "gauge_tvd_ft": [8000.0], "water_fvf_rb_stb": [1.02], "solution_gor_at_pb_scf_stb": [250.0]})
        p = fit_pr_and_pi(wt, {}, pvt, comp, fixed_j={"W": 2.5})["W"]
        self.assertAlmostEqual(p["J"], 2.5)
        self.assertEqual(p["j_source"], "fixed (external PI)")


if __name__ == "__main__":
    unittest.main()
