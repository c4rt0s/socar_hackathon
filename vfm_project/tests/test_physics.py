"""Unit tests for physical laws, unit conversions, and anti-leakage guards in VFM."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vfm.io.loaders import load_and_harmonize_well_tests
from vfm.modeling.targets import reconstruct_phase_rates
from vfm.physics.choke import (
    choke_bean_area_ft2,
    choke_bean_diameter_in,
    choke_rate,
)
from vfm.modeling.physics_vfm import PhysicsVFM
from vfm.physics.reservoir import (
    compute_wellbore_fluid_gradient,
    estimate_pr_from_buildup,
    fit_pr_and_pi,
    gauge_to_datum_pressure,
    ipr_rate,
)
from vfm.physics.vlp import invert_vlp_q_liq, vlp_pressure_drop


class TestPhysics(unittest.TestCase):
    def test_oilfield_unit_conversions_and_sm3_mscf_scaling(self) -> None:
        # 1. Oilfield API gravity & brine gradient bounds
        grad_oil = compute_wellbore_fluid_gradient(api=35.0, wc_frac=0.0, gor_scf_stb=250.0, gas_sg=0.7, bo=1.15)
        grad_water = compute_wellbore_fluid_gradient(api=35.0, wc_frac=1.0, salinity_ppm=100_000.0, bw=1.02)
        self.assertGreater(grad_oil, 0.25)
        self.assertLess(grad_oil, 0.40)
        self.assertGreater(grad_water, 0.42)
        self.assertLess(grad_water, 0.50)
        self.assertGreater(grad_water, grad_oil)

        # 2. Choke bean diameter & area scaling (128/64ths = 2.0 in at 100%)
        d_full = choke_bean_diameter_in(100.0, choke_max_size_64ths=128.0)
        d_half = choke_bean_diameter_in(50.0, choke_max_size_64ths=128.0)
        self.assertAlmostEqual(d_full, 2.0, places=6)
        self.assertAlmostEqual(d_half, 1.0, places=6)
        # Area scales with d^2 -> 4x area when diameter doubles
        a_full = choke_bean_area_ft2(d_full)
        a_half = choke_bean_area_ft2(d_half)
        self.assertAlmostEqual(a_full / a_half, 4.0, places=6)

        # 3. Sm3/d -> Mscf/d harmonization (1 Mscf = 28.3168466 Sm3)
        csv_text = (
            "well_name,test_start,test_end,oil_rate_stbd,water_rate_stbd,liquid_rate_stbd,"
            "gas_rate_mscfd,total_gas_rate_mscfd,gas_lift_rate_mscfd,gor_scf_stb\n"
            "GL-06,2025-01-18 00:00:00,2025-01-18 12:00:00,1000.0,200.0,1200.0,7079.21165,28316.8466,750.0,250.0\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "wt_sm3.csv"
            p.write_text(csv_text, encoding="utf-8")
            wt, audit = load_and_harmonize_well_tests(p)
            self.assertEqual(len(audit["unit_conversions"]), 1)
            self.assertAlmostEqual(float(wt.loc[0, "gas_rate_mscfd"]), 250.0, places=3)
            self.assertAlmostEqual(float(wt.loc[0, "total_gas_rate_mscfd"]), 1000.0, places=3)

    def test_gauge_to_datum_pressure_sign_and_magnitude(self) -> None:
        grad = 0.35  # psi/ft
        p_gauge = 3000.0

        # Datum deeper than gauge -> higher pressure at datum
        p_deeper = gauge_to_datum_pressure(p_gauge, gauge_tvd_ft=8000.0, datum_tvd_ft=8300.0, fluid_grad_psi_ft=grad)
        self.assertAlmostEqual(p_deeper, 3000.0 + 0.35 * 300.0, places=6)
        self.assertGreater(p_deeper, p_gauge)

        # Datum shallower than gauge -> lower pressure at datum
        p_shallower = gauge_to_datum_pressure(p_gauge, gauge_tvd_ft=8000.0, datum_tvd_ft=7800.0, fluid_grad_psi_ft=grad)
        self.assertAlmostEqual(p_shallower, 3000.0 - 0.35 * 200.0, places=6)
        self.assertLess(p_shallower, p_gauge)

        # Equal TVD -> identical pressure
        p_same = gauge_to_datum_pressure(p_gauge, gauge_tvd_ft=8000.0, datum_tvd_ft=8000.0, fluid_grad_psi_ft=grad)
        self.assertAlmostEqual(p_same, p_gauge, places=6)

    def test_choke_monotonicity_and_closure(self) -> None:
        # Zero when closed or zero/negative delta-P
        q_closed = choke_rate(
            choke_opening_pct=0.0,
            whp_psia=300.0,
            whp_downstream_psia=200.0,
            wc_frac=0.3,
            gor_scf_stb=250.0,
            gas_lift_rate_mscfd=500.0,
            oil_api=35.0,
            gas_sg=0.75,
        )
        self.assertEqual(q_closed, 0.0)

        q_no_dp = choke_rate(
            choke_opening_pct=50.0,
            whp_psia=200.0,
            whp_downstream_psia=210.0,
            wc_frac=0.3,
            gor_scf_stb=250.0,
            gas_lift_rate_mscfd=500.0,
            oil_api=35.0,
            gas_sg=0.75,
        )
        self.assertEqual(q_no_dp, 0.0)

        # NaN sensor propagation
        q_nan = choke_rate(
            choke_opening_pct=50.0,
            whp_psia=np.nan,
            whp_downstream_psia=200.0,
            wc_frac=0.3,
            gor_scf_stb=250.0,
            gas_lift_rate_mscfd=500.0,
            oil_api=35.0,
            gas_sg=0.75,
        )
        self.assertTrue(np.isnan(q_nan))

        # Monotonicity wrt choke opening (d)
        pcts = np.array([20.0, 35.0, 50.0, 70.0, 90.0])
        q_d = choke_rate(
            choke_opening_pct=pcts,
            whp_psia=350.0,
            whp_downstream_psia=220.0,
            wc_frac=0.25,
            gor_scf_stb=250.0,
            gas_lift_rate_mscfd=600.0,
            oil_api=35.0,
            gas_sg=0.75,
        )
        self.assertTrue(np.all(np.diff(q_d) > 0.0))

        # Monotonicity wrt pressure drop (decreasing downstream pressure while subcritical)
        whp_ds = np.array([320.0, 290.0, 260.0, 230.0])
        q_dp = choke_rate(
            choke_opening_pct=50.0,
            whp_psia=350.0,
            whp_downstream_psia=whp_ds,
            wc_frac=0.25,
            gor_scf_stb=250.0,
            gas_lift_rate_mscfd=600.0,
            oil_api=35.0,
            gas_sg=0.75,
        )
        self.assertTrue(np.all(np.diff(q_dp) > 0.0))

    def test_ipr_c1_continuity_at_bubble_point(self) -> None:
        j_val = 5.0
        pr = 3500.0
        pb = 1500.0
        eps = 1e-4

        # Value continuity at Pwf = Pb
        q_at_pb = ipr_rate(j_val, pr, pb, Pb=pb)
        q_above = ipr_rate(j_val, pr, pb + eps, Pb=pb)
        q_below = ipr_rate(j_val, pr, pb - eps, Pb=pb)
        self.assertAlmostEqual(q_at_pb, j_val * (pr - pb), places=6)
        self.assertAlmostEqual(q_above, q_at_pb, places=2)
        self.assertAlmostEqual(q_below, q_at_pb, places=2)

        # Slope continuity dq/dPwf = -J on both sides of Pb
        slope_right = (ipr_rate(j_val, pr, pb + eps, Pb=pb) - q_at_pb) / eps
        slope_left = (q_at_pb - ipr_rate(j_val, pr, pb - eps, Pb=pb)) / eps
        self.assertAlmostEqual(slope_right, -j_val, places=3)
        self.assertAlmostEqual(slope_left, -j_val, places=3)

        # Zero inflow when Pwf >= Pr
        self.assertEqual(ipr_rate(j_val, pr, pr, Pb=pb), 0.0)
        self.assertEqual(ipr_rate(j_val, pr, pr + 100.0, Pb=pb), 0.0)

    def test_vlp_monotonicity_sensitivity_and_inversion(self) -> None:
        common = dict(
            whp_psia=300.0,
            gas_lift_rate_mscfd=700.0,
            wc_frac=0.30,
            gor_scf_stb=250.0,
            gauge_tvd_ft=8000.0,
            gl_valve_tvd_ft=5000.0,
            tubing_id_in=2.992,
            oil_api=35.0,
            gas_sg=0.75,
            lift_gas_sg=0.65,
            bubble_point_psia=1400.0,
            bo=1.14,
            bw=1.02,
            mu_oil_cp=1.5,
            water_salinity_ppm=100_000.0,
            res_temp_degf=200.0,
            wh_temp_degf=120.0,
        )

        # 1. Higher rate increases friction -> higher dp above unloading minimum
        q_grid = np.array([1500.0, 2500.0, 3500.0, 4500.0])
        dp_q = vlp_pressure_drop(q_liq_stbd=q_grid, **common)
        self.assertTrue(np.all(np.diff(dp_q) > 0.0))

        # 2. Higher water cut increases mixture density -> higher dp
        dp_low_wc = vlp_pressure_drop(q_liq_stbd=2500.0, **{**common, "wc_frac": 0.1})
        dp_high_wc = vlp_pressure_drop(q_liq_stbd=2500.0, **{**common, "wc_frac": 0.7})
        self.assertGreater(dp_high_wc, dp_low_wc)

        # 3. Higher gas-lift rate lightens column -> lower dp (below friction-dominated regime)
        dp_low_gl = vlp_pressure_drop(q_liq_stbd=2000.0, **{**common, "gas_lift_rate_mscfd": 300.0})
        dp_high_gl = vlp_pressure_drop(q_liq_stbd=2000.0, **{**common, "gas_lift_rate_mscfd": 1200.0})
        self.assertLess(dp_high_gl, dp_low_gl)

        # 4. Exact round-trip inversion for q_liq
        q_true = 2800.0
        dp_true = float(vlp_pressure_drop(q_liq_stbd=q_true, **common))
        q_inv = invert_vlp_q_liq(dp_tubing_psia=dp_true, **common)
        self.assertAlmostEqual(float(q_inv), q_true, delta=2.0)

        # 5. Unreachable / ill-conditioned dp returns NaN
        q_bad = invert_vlp_q_liq(dp_tubing_psia=50.0, **common)
        self.assertTrue(np.isnan(q_bad))

    def test_mass_balance_and_nonnegativity_identities(self) -> None:
        q_liq = np.array([2000.0, 1500.0, -50.0])
        wc = np.array([0.35, 0.40, 0.25])
        gor = np.array([250.0, 300.0, 200.0])
        q_gl = np.array([800.0, 600.0, 500.0])
        shut_in = np.array([False, True, False])

        rec = reconstruct_phase_rates(q_liq, wc, gor, q_gl, is_shut_in=shut_in)

        # Non-negativity
        for key in ("liquid_rate_stbd", "oil_rate_stbd", "water_rate_stbd", "gas_rate_mscfd", "total_gas_rate_mscfd"):
            self.assertTrue(np.all(rec[key] >= 0.0))

        # Mass balance on flowing row 0: qo + qw = q_liq, q_total_gas = qo * GOR / 1000 + q_gl
        self.assertAlmostEqual(
            rec["oil_rate_stbd"][0] + rec["water_rate_stbd"][0],
            rec["liquid_rate_stbd"][0],
            places=6,
        )
        self.assertAlmostEqual(
            rec["total_gas_rate_mscfd"][0],
            rec["oil_rate_stbd"][0] * gor[0] / 1000.0 + q_gl[0],
            places=6,
        )

        # Exact shut-in zeroing on row 1
        for key in ("liquid_rate_stbd", "oil_rate_stbd", "water_rate_stbd", "gas_rate_mscfd", "total_gas_rate_mscfd"):
            self.assertEqual(float(rec[key][1]), 0.0)


class TestLogitWaterCutAndCleaning(unittest.TestCase):
    def test_logit_wc_bounds_and_field_slope_shrinkage(self) -> None:
        from vfm.modeling.physics_vfm import _fit_logit_wc_trend, _sigmoid
        # one test only: the well inherits the field slope instead of a flat trend
        tm, b0, b1 = _fit_logit_wc_trend(np.array([10.0]), np.array([0.30]), 0.004, 20000.0, True)
        self.assertAlmostEqual(b1, 0.004)
        wc_far = _sigmoid(b0 + b1 * (np.array([-3000.0, 3000.0]) - tm))  # ~8 years either side
        self.assertTrue(np.all((wc_far > 0) & (wc_far < 1)))  # bounded even far outside the data
        # many well-spread tests: the well's own slope dominates
        t = np.linspace(0, 240, 9)
        own = 0.3 + 0.0005 * t
        _, _, b1_many = _fit_logit_wc_trend(t, own, -0.001, 20000.0, False)
        self.assertGreater(b1_many, 0.0)
        _, _, b1_none = _fit_logit_wc_trend(t, own, None, 20000.0, False)      # no field slope: plain OLS
        self.assertGreater(b1_none, b1_many)

    def test_field_wc_slope_is_causal(self) -> None:
        from vfm.modeling.physics_vfm import field_logit_wc_slope
        ref = pd.Timestamp("2025-01-01", tz="UTC")
        starts = pd.date_range("2025-01-10", periods=6, freq="30D", tz="UTC")
        wt = pd.DataFrame({"well_name": "A", "test_start": starts, "test_end": starts + pd.Timedelta("12h"),
                           "water_cut_pct": [20, 21, 22, 23, 60, 90]})  # last two tests jump
        early = field_logit_wc_slope(wt, ref, max_time=starts[3] + pd.Timedelta("1D"))
        full = field_logit_wc_slope(wt, ref, max_time=None)
        wt_future_changed = wt.assign(water_cut_pct=[20, 21, 22, 23, 5, 5])
        self.assertAlmostEqual(early, field_logit_wc_slope(wt_future_changed, ref, starts[3] + pd.Timedelta("1D")))
        self.assertGreater(full, early)

    def test_gas_lift_hampel_keeps_genuine_zero_steps(self) -> None:
        from vfm.preprocessing.cleaning import clean_well_telemetry
        ts = pd.date_range("2025-01-01", periods=60, freq="1min", tz="UTC")
        gl = np.r_[np.full(30, 800.0), np.zeros(30)]   # lift gas stops at a shut-in
        gl[10] = 2500.0                                # a real spike while flowing
        df = pd.DataFrame({"timestamp": ts, "well_name": "W", "gas_lift_rate_mscfd": gl,
                           "whp_psia": 300.0, "whp_downstream_psia": 200.0, "dh_pressure_psia": 3000.0})
        out, audit = clean_well_telemetry(df)
        self.assertTrue(np.all(out["gas_lift_rate_mscfd"].iloc[30:] == 0.0))
        self.assertAlmostEqual(float(out["gas_lift_rate_mscfd"].iloc[10]), 800.0)
        self.assertEqual(audit["hampel_spikes_masked"]["gas_lift_rate_mscfd"], 1)

    def test_interpolation_never_bridges_long_gaps(self) -> None:
        from vfm.preprocessing.cleaning import clean_well_telemetry
        ts = pd.DatetimeIndex(list(pd.date_range("2025-01-01 00:00", periods=10, freq="1min", tz="UTC"))
                              + list(pd.date_range("2025-01-02 12:00", periods=10, freq="1min", tz="UTC")))
        p = np.r_[np.full(10, 3000.0), np.full(10, 3100.0)]
        p[9] = -999.25   # sentinel right before a 36 h SCADA gap
        p[3] = -999.25   # sentinel inside 1-minute data
        df = pd.DataFrame({"timestamp": ts, "well_name": "W", "dh_pressure_psia": p, "whp_psia": 300.0,
                           "whp_downstream_psia": 200.0, "gas_lift_rate_mscfd": 800.0})
        out, _ = clean_well_telemetry(df)
        self.assertTrue(np.isnan(out["dh_pressure_psia"].iloc[9]))          # not bridged across 36 h
        self.assertAlmostEqual(float(out["dh_pressure_psia"].iloc[3]), 3000.0)  # short gap filled


class TestBuildupRadialAcceptance(unittest.TestCase):
    PVT = {
        "oil_fvf_rb_stb": 1.2, "oil_viscosity_cp": 1.0, "net_pay_thickness_ft": 50.0,
        "porosity_frac": 0.2, "total_compressibility_1_psi": 1.2e-5, "wellbore_radius_ft": 0.33,
        "estimated_drainage_area_acres": 600.0, "datum_tvd_ft": 8200.0,
    }
    COMP = {"gauge_tvd_ft": 8000.0, "water_fvf_rb_stb": 1.02, "solution_gor_at_pb_scf_stb": 250.0}

    def _buildup(self, p_of_t):
        t0 = pd.Timestamp("2025-06-01 00:00:00+00:00")
        ts = pd.date_range(t0 - pd.Timedelta("30min"), t0 + pd.Timedelta("48h"), freq="1min")
        dt = (ts - t0).total_seconds().to_numpy() / 3600.0
        p = np.where(dt <= 0.0, 3000.0, 3000.0 + p_of_t(np.maximum(dt, 1e-9)))
        df = pd.DataFrame({"timestamp": ts, "dh_pressure_psia": p})
        return estimate_pr_from_buildup(
            df, t0, t0 + pd.Timedelta("48h"), prior_rate_stbd=1000.0, pvt_row=self.PVT,
            comp_row=self.COMP, fluid_grad_psi_ft=0.35, wc_frac=0.0, producing_time_hr=50_000.0,
        )

    def test_flat_derivative_recovers_kh(self) -> None:
        # dp = 40 ln(1 + t/0.05): derivative d(dp)/d(ln t) -> 40 psi, i.e. radial flow
        res = self._buildup(lambda t: 40.0 * np.log1p(t / 0.05))
        kh_true = 70.6 * 1000.0 * 1.2 * 1.0 / 40.0
        self.assertNotEqual(res["quality_flag"], "NO_RADIAL_P_END")
        self.assertAlmostEqual(res["kh_md_ft"] / kh_true, 1.0, delta=0.03)
        self.assertTrue(np.isfinite(res["skin"]))

    def test_falling_derivative_reports_no_kh(self) -> None:
        # exponential approach to a plateau: derivative humps and falls, never flat
        res = self._buildup(lambda t: 300.0 * (1.0 - np.exp(-t / 6.0)))
        self.assertEqual(res["quality_flag"], "NO_RADIAL_P_END")
        for key in ("kh_md_ft", "k_md", "skin", "p_star_gauge_psia", "t_pss_hr"):
            self.assertTrue(np.isnan(res[key]), key)
        self.assertAlmostEqual(res["pr_gauge_psia"], res["p_end_gauge_psia"], places=6)


class TestReservoirPressurePrior(unittest.TestCase):
    def _frames(self):
        ts = pd.date_range("2025-01-10", periods=6, freq="30D", tz="UTC")
        wt = pd.DataFrame({
            "well_name": "W", "test_start": ts, "test_end": ts + pd.Timedelta("12h"),
            "dh_pressure_psia": 2600.0 + np.array([0, 3, -2, 1, -1, 2]), "liquid_rate_stbd": 1500.0,
            "water_cut_pct": 30.0, "oil_gravity_api": 30.0, "gas_sg": 0.75, "water_salinity_ppm": 1e5,
        })
        pvt = pd.DataFrame({"well_name": ["W"], "datum_tvd_ft": [8200.0], "bubble_point_pressure_psia": [1300.0],
                            "oil_fvf_rb_stb": [1.15]})
        comp = pd.DataFrame({"well_name": ["W"], "gauge_tvd_ft": [8000.0], "water_fvf_rb_stb": [1.02],
                             "solution_gor_at_pb_scf_stb": [250.0]})
        return wt, pvt, comp

    def test_prior_gradient_sets_pr_when_tests_cannot(self) -> None:
        # nearly constant Pwf: Pr and J are collinear, so the prior decides Pr0
        wt, pvt, comp = self._frames()
        for g in (0.39, 0.41):
            p = fit_pr_and_pi(wt, {}, pvt, comp, pr_gradient_prior_psi_ft=g)["W"]
            self.assertAlmostEqual(p["Pr0_datum_psia"] / 8200.0, g, delta=0.005)
            self.assertEqual(p["pr_prior_source"], "caller")
        p = fit_pr_and_pi(wt, {}, pvt, comp)["W"]
        self.assertEqual(p["pr_prior_source"], "nominal_default")


class TestPhysicsVFMGorConsistency(unittest.TestCase):
    def test_estimators_use_calibration_gor_and_allocation_uses_gor_hat(self) -> None:
        captured = {}

        def fake_choke(params, **kw):
            captured["choke"] = np.asarray(kw["gor_scf_stb"])
            return np.full(len(kw["whp_psia"]), 1000.0)

        def fake_vlp(params, **kw):
            captured["vlp"] = np.asarray(kw["gor_scf_stb"])
            return np.full(len(kw["whp_psia"]), 1000.0)

        model = PhysicsVFM()
        model.well_models_ = {"W": {
            "ipr_params": None, "choke_params": {"cd": 0.84}, "vlp_params": {"slip_k": 1.0},
            "w_ipr": 0.0, "w_choke": 1.0, "w_vlp": 1.0, "wc_mean": 0.2, "wc_slope": 0.0,
            "wc_t_mean": 0.0, "gor_hat": 300.0, "rs_pb": 250.0,
            "default_api": 35.0, "default_gas_sg": 0.75, "default_salinity_ppm": 1e5,
        }}
        hf = pd.DataFrame({
            "timestamp": pd.date_range("2025-03-01", periods=3, freq="1min", tz="UTC"),
            "well_name": "W", "choke_opening_pct": 40.0, "whp_psia": 300.0,
            "whp_downstream_psia": 150.0, "dh_pressure_psia": 2500.0, "gas_lift_rate_mscfd": 500.0,
        })
        with mock.patch("vfm.modeling.physics_vfm.predict_choke_q_liq", fake_choke), \
                mock.patch("vfm.modeling.physics_vfm.predict_vlp_q_liq", fake_vlp):
            out = model.predict(hf)
        self.assertTrue(np.all(captured["choke"] == 250.0))
        self.assertTrue(np.all(captured["vlp"] == 250.0))
        np.testing.assert_allclose(out["gas_rate_mscfd"], 1000.0 * 0.8 * 300.0 / 1000.0)


if __name__ == "__main__":
    unittest.main()
