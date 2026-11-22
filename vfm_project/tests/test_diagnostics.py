"""Unit tests for PTA Qo Sensitivity Tornado & VFM Feature Diagnostics (Correlation + SHAP)."""

from __future__ import annotations

import math
import unittest

import numpy as np
import pandas as pd

from vfm.diagnostics import (
    _corr_payload,
    _pta_k_and_skin,
    compute_pta_qo_sensitivity,
    retrain_custom_vfm,
)


class TestPTAQoSensitivityLaws(unittest.TestCase):
    def test_exact_qo_linearity_and_skin_logarithmic_shift(self):
        bo, mu, h, phi, ct, rw = 1.25, 1.4, 80.0, 0.18, 1.5e-5, 0.354
        m, p1hr, pwf0, qo = 40.0, 3100.0, 2600.0, 1000.0
        k0, kh0, s0, _ = _pta_k_and_skin(qo, m, bo, mu, h, phi, ct, rw, p1hr, pwf0)
        k_hi, kh_hi, s_hi, _ = _pta_k_and_skin(qo * 1.15, m, bo, mu, h, phi, ct, rw, p1hr, pwf0)
        k_lo, kh_lo, s_lo, _ = _pta_k_and_skin(qo * 0.85, m, bo, mu, h, phi, ct, rw, p1hr, pwf0)

        # Exact 1:1 linear proportionality of k and kh with Qo
        self.assertAlmostEqual(k_hi / k0, 1.15, places=10)
        self.assertAlmostEqual(k_lo / k0, 0.85, places=10)
        self.assertAlmostEqual(kh_hi / kh0, 1.15, places=10)

        # Exact logarithmic skin shift: dS = -1.1513 * log10(Qo / Qo0)
        self.assertAlmostEqual(s_hi - s0, -1.1513 * math.log10(1.15), places=10)
        self.assertAlmostEqual(s_lo - s0, -1.1513 * math.log10(0.85), places=10)

    def test_viscosity_cancels_in_skin_factor(self):
        bo, mu, h, phi, ct, rw = 1.20, 1.5, 65.0, 0.20, 1.2e-5, 0.354
        m, p1hr, pwf0, qo = 30.0, 3200.0, 2700.0, 900.0
        k1, _, s1, _ = _pta_k_and_skin(qo, m, bo, mu * 0.8, h, phi, ct, rw, p1hr, pwf0)
        k2, _, s2, _ = _pta_k_and_skin(qo, m, bo, mu * 1.2, h, phi, ct, rw, p1hr, pwf0)
        self.assertAlmostEqual(k2 / k1, 1.5, places=10)
        # Viscosity mu_o cancels algebraically in k / (phi * mu_o * ct * rw^2)
        self.assertAlmostEqual(s1, s2, places=10)

    def test_compute_pta_qo_sensitivity_end_to_end_synthetic(self):
        ts = pd.date_range("2026-03-01", periods=100, freq="h", tz="UTC")
        hf = pd.DataFrame({
            "timestamp": ts,
            "dh_pressure_psia": np.linspace(2600.0, 2605.0, len(ts)),
            "choke_opening_pct": np.full(len(ts), 35.0),
        })
        pvt = {
            "oil_fvf_rb_stb": 1.22,
            "oil_viscosity_cp": 1.35,
            "net_pay_thickness_ft": 75.0,
            "porosity_frac": 0.19,
            "total_compressibility_1_psi": 1.4e-5,
            "wellbore_radius_ft": 0.354,
        }
        comp = {"water_fvf_rb_stb": 1.02}
        tests = pd.DataFrame({
            "well_name": ["TEST-1"],
            "test_start": [pd.Timestamp("2026-02-20T00:00:00Z")],
            "test_end": [pd.Timestamp("2026-02-20T06:00:00Z")],
            "oil_rate_stbd": [1080.0],
            "water_rate_stbd": [500.0],
        })
        daily = pd.DataFrame({
            "well_name": ["TEST-1"] * 5,
            "date": pd.date_range("2026-02-28", periods=5, freq="D", tz="UTC"),
            "oil_rate_vfm_stbd": [1000.0, 1005.0, 995.0, 1000.0, 1000.0],
            "water_rate_vfm_stbd": [500.0] * 5,
        })
        ev = {
            "shut_in_start": pd.Timestamp("2026-03-04T00:00:00Z"),
            "shut_in_end": pd.Timestamp("2026-03-05T00:00:00Z"),
            "duration_hr": 24.0,
        }
        res = compute_pta_qo_sensitivity(
            well="TEST-1", ev=ev, hf_w=hf, pvt_row=pvt, comp_row=comp,
            tests_df=tests, model_daily={"synth_stacked_v2": daily},
            hyb={"qo_stbd": 1000.0, "pwf_3h_med": 2602.0, "logic3_intersection": {"m": 50.0, "k": 71.3, "skin": 1.2}},
            horner_plot={"tp_hr": 2000.0, "m_psi_cycle": 50.0, "p_star_psi": 3250.0},
            qo_pct=10.0,
        )
        self.assertTrue(res["available"])
        # Confirm param_mu has zero skin span and param_qo has exact +-10% k shift
        p_map = {b["key"]: b for b in res["parameter_tornado_bars"]}
        self.assertAlmostEqual(p_map["param_mu"]["skin_span"], 0.0, places=3)
        self.assertAlmostEqual(p_map["param_qo"]["k_pct_high"], 10.0, places=1)
        self.assertAlmostEqual(p_map["param_qo"]["k_pct_low"], -10.0, places=1)
        # Confirm Perrine multiphase bar captures WC = 33.3%
        q_map = {b["key"]: b for b in res["qo_root_cause_bars"]}
        self.assertGreater(q_map["qo_multiphase_perrine"]["k_max"], res["base"]["k_md"])


        # Confirm coupled four_outputs_tornado at +-20% Qo accounts for BOTH tp(Qo)=24*Np/Qo AND parameters!
        res20 = compute_pta_qo_sensitivity(
            well="TEST-1", ev=ev, hf_w=hf, pvt_row=pvt, comp_row=comp,
            tests_df=tests, model_daily={"synth_stacked_v2": daily},
            hyb={"qo_stbd": 1000.0, "pwf_3h_med": 2602.0, "logic3_intersection": {"m": 50.0, "k": 71.3, "skin": 1.2}},
            horner_plot={"tp_hr": 200.0, "m_psi_cycle": 50.0, "p_star_psi": 3250.0, "win_hr": [1.0, 20.0]},
            qo_pct=20.0,
        )
        f4 = {r["key"]: r for r in res20["four_outputs_tornado"]}
        self.assertEqual(set(f4.keys()), {"permeability_k_md", "skin_S", "delta_p_skin_psi", "horner_slope_m"})
        # Coupled Horner slope m changes genuinely via tp = 24*Np/Qo when tp is finite!
        self.assertNotAlmostEqual(f4["horner_slope_m"]["at_qo_high"], f4["horner_slope_m"]["at_qo_low"], places=3)
        self.assertGreater(f4["permeability_k_md"]["pct_at_qo_high"], 18.0)
        self.assertLess(f4["permeability_k_md"]["pct_at_qo_low"], -18.0)
        self.assertLess(f4["skin_S"]["delta_at_qo_high"], 0.0)
        self.assertGreater(f4["skin_S"]["delta_at_qo_low"], 0.0)


class TestCustomFeatureRetrain(unittest.TestCase):
    def test_retrain_custom_vfm_strictly_uses_selected_features(self):
        rng = np.random.default_rng(12)
        wells = ["W-1", "W-2"]
        d_grid = pd.date_range("2026-01-01", periods=10, freq="D", tz="UTC")
        hf_map = {}
        test_rows = []
        for wi, w in enumerate(wells):
            ts = pd.date_range("2026-01-01", periods=240, freq="h", tz="UTC")
            chk = 20.0 + 25.0 * (wi + 0.5) * rng.uniform(0.7, 1.3, len(ts))
            whp = 300.0 + 40.0 * rng.normal(size=len(ts))
            pds = 140.0 + 10.0 * rng.normal(size=len(ts))
            pdg = 2600.0 - 4.0 * chk + 15.0 * rng.normal(size=len(ts))
            hf_map[w] = pd.DataFrame({
                "timestamp": ts,
                "choke_opening_pct": chk,
                "whp_psia": whp,
                "whp_downstream_psia": pds,
                "dh_pressure_psia": pdg,
                "dh_temperature_degf": np.full(len(ts), 185.0),
                "annulus_pressure_psia": np.full(len(ts), 900.0),
                "gas_lift_rate_mscfd": np.full(len(ts), 600.0),
            })
            for k in range(6):
                t0 = ts[k * 35]
                t1 = t0 + pd.Timedelta(hours=4)
                c_med = float(chk[k * 35 : k * 35 + 5].mean())
                qo = 25.0 * c_med + 100.0 * wi + rng.normal(0, 5)
                test_rows.append({
                    "well_name": w, "test_start": t0, "test_end": t1,
                    "oil_rate_stbd": qo, "water_rate_stbd": 0.4 * qo,
                    "gas_rate_mscfd": 0.25 * qo, "gor_scf_stb": 250.0,
                })
        tests_df = pd.DataFrame(test_rows)
        comp_df = pd.DataFrame({"well_name": wells, "tubing_id_in": [2.992, 3.958], "gauge_tvd_ft": [8000.0, 8500.0]})
        pvt_df = pd.DataFrame({"well_name": wells, "net_pay_thickness_ft": [60.0, 90.0], "bubble_point_pressure_psia": [1400.0, 1500.0]})

        for mt in ("lgbm", "ridge", "rf"):
            res, daily = retrain_custom_vfm(
                selected_features=["choke_opening_pct", "dp_choke_raw_psi", "tubing_id_in"],
                model_type=mt,
                tests_df=tests_df,
                hf_by_well=hf_map,
                comp_df=comp_df,
                pvt_df=pvt_df,
                date_grid=d_grid,
            )
            self.assertEqual(res["selected_features"], ["choke_opening_pct", "dp_choke_raw_psi", "tubing_id_in"])
            self.assertEqual(res["n_features"], 3)
            self.assertEqual(len(daily), 20)
            self.assertTrue(np.isfinite(daily["oil_rate_alloc_stbd"]).all())
            self.assertLess(res["shap"]["oil"]["max_efficiency_error"], 1e-2)
            self.assertEqual(len(res["shap"]["oil"]["features"]), 3)

        # Verify automated preset grid search ranks combinations and matches winner retrain
        from vfm.diagnostics import grid_search_vfm_presets
        gs, best_daily = grid_search_vfm_presets(
            model_types=["ridge", "lgbm"],
            optimize_metric="composite",
            tests_df=tests_df,
            hf_by_well=hf_map,
            comp_df=comp_df,
            pvt_df=pvt_df,
            date_grid=d_grid,
            custom_features=["choke_opening_pct", "tubing_id_in"],
        )
        self.assertGreaterEqual(gs["n_combinations"], 4)
        lb = gs["leaderboard"]
        self.assertEqual(lb[0]["rank"], 1)
        scores = [r["score"] for r in lb]
        self.assertEqual(scores, sorted(scores))
        self.assertEqual(gs["best"]["model_type"], gs["best_custom_result"]["model_type"])
        self.assertEqual(len(best_daily), 20)


class TestCorrelationPayload(unittest.TestCase):
    def test_symmetry_unit_diagonal_and_constant_column_drop(self):
        rng = np.random.default_rng(7)
        x = rng.normal(size=40)
        df = pd.DataFrame({
            "a": x,
            "b": 2.0 * x + 0.1 * rng.normal(size=40),
            "c": -x + 0.2 * rng.normal(size=40),
            "const_col": np.ones(40),
        })
        specs = [
            {"key": "a", "label": "A", "group": "G", "used_in_v2": True},
            {"key": "b", "label": "B", "group": "G", "used_in_v2": True},
            {"key": "c", "label": "C", "group": "G", "used_in_v2": False},
            {"key": "const_col", "label": "Const", "group": "G", "used_in_v2": False},
        ]
        out = _corr_payload(df, specs, method="pearson")
        self.assertEqual(len(out["columns"]), 3)  # zero-variance column excluded cleanly
        mat = np.asarray(out["matrix"], dtype=float)
        np.testing.assert_allclose(mat, mat.T, atol=1e-6)
        np.testing.assert_allclose(np.diag(mat), np.ones(3), atol=1e-6)
        self.assertGreater(mat[0, 1], 0.9)
        self.assertLess(mat[0, 2], -0.8)


if __name__ == "__main__":
    unittest.main()
