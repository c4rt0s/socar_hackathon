"""Tests for PTA Q_oil vs Q_total rate_basis toggle, adaptive Horner fallback, and deliverable exports."""

from __future__ import annotations

import io
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vfm import audit as A  # noqa: E402
from vfm.pta import kernel as K  # noqa: E402
from vfm.pta.runner import PtaRunSettings, _adaptive_horner_window, run_pta_event  # noqa: E402
from vfm.workbench import Workbench  # noqa: E402


class TestPtaRateBasisAndDeliverables(unittest.TestCase):
    def test_rate_basis_oil_vs_total_liquid(self) -> None:
        t0 = pd.Timestamp("2025-01-01 00:00:00", tz="UTC")
        shut_start = pd.Timestamp("2025-01-10 00:00:00", tz="UTC")
        shut_end = pd.Timestamp("2025-01-11 00:00:00", tz="UTC")
        dt_hr = np.concatenate([[0.0], np.geomspace(1 / 60.0, 24.0, 90)])
        m_true = 40.0
        tp_hr = 216.0
        pws = np.where(dt_hr == 0.0, 2500.0, 3200.0 - m_true * np.log10((tp_hr + dt_hr) / dt_hr))
        ts = [shut_start + pd.Timedelta(hours=float(h)) for h in dt_hr]
        hf = pd.DataFrame({"timestamp": ts, "dh_pressure_psia": pws})

        days = pd.date_range("2025-01-01", "2025-01-10", freq="1D", tz="UTC")
        daily = pd.DataFrame({
            "date": days,
            "oil_rate_alloc_stbd": np.full(len(days), 1000.0),
            "water_rate_alloc_stbd": np.full(len(days), 1000.0),
            "uptime_frac": np.ones(len(days)),
        })
        tests_df = pd.DataFrame({
            "test_start": [t0], "test_end": [t0 + pd.Timedelta(hours=6)],
            "oil_rate_stbd": [1000.0], "water_rate_stbd": [1000.0],
            "oil_gravity_api": [35.0], "water_cut_pct": [50.0],
            "gas_sg": [0.7], "water_salinity_ppm": [20000.0],
        })
        pvt = {
            "oil_fvf_rb_stb": 1.2, "oil_viscosity_cp": 1.0,
            "total_compressibility_1_psi": 1.2e-5, "porosity_frac": 0.2,
            "net_pay_thickness_ft": 50.0, "wellbore_radius_ft": 0.3,
            "datum_tvd_ft": 8000.0, "estimated_drainage_area_acres": 320.0,
        }
        comp = {"gauge_tvd_ft": 7900.0, "water_fvf_rb_stb": 1.0, "solution_gor_at_pb_scf_stb": 250.0}

        ev_oil = run_pta_event("W", shut_start, shut_end, hf, tests_df, pvt, comp,
                               PtaRunSettings(rate_basis="oil", prepend_cumulative_history=False), daily)
        ev_tot = run_pta_event("W", shut_start, shut_end, hf, tests_df, pvt, comp,
                               PtaRunSettings(rate_basis="total_liquid", prepend_cumulative_history=False), daily)

        q_oil_res = ev_oil["result"]["inputs"]["rate_source"]["q_final_reservoir_bbl_day"]
        q_tot_res = ev_tot["result"]["inputs"]["rate_source"]["q_final_reservoir_bbl_day"]
        self.assertAlmostEqual(q_oil_res, 1200.0, places=2)
        self.assertAlmostEqual(q_tot_res, 2200.0, places=2)
        k_oil = ev_oil["result"]["results"]["k_md"]
        k_tot = ev_tot["result"]["results"]["k_md"]
        self.assertIsNotNone(k_oil)
        self.assertIsNotNone(k_tot)
        self.assertAlmostEqual(k_tot / k_oil, 2200.0 / 1200.0, delta=0.05)

    def test_adaptive_horner_window_avoids_late_plateau(self) -> None:
        dt = np.geomspace(0.05, 72.0, 200)
        tp = 1000.0
        x = np.log10((tp + dt) / dt)
        # Middle-time linear Horner m=25, flattening out after dt > 12h
        p = np.where(dt <= 12.0, 3200.0 - 25.0 * x, 3200.0 - 25.0 * np.log10((tp + 12.0) / 12.0))
        win, tol = _adaptive_horner_window(dt, p, tp)
        self.assertIsNotNone(win)
        self.assertLessEqual(win[1], 16.0)

    def test_export_table_bytes_xlsx_and_csv(self) -> None:
        df = pd.DataFrame({
            "date": ["2025-01-01"],
            "well_name": ["GL-01"],
            "oil_rate_stbd": [1000.5],
            "water_rate_stbd": [200.25],
            "gas_rate_mscfd": [300.125],
        })
        raw_xlsx, ctype_x, ext_x = Workbench.export_table_bytes(df, "xlsx", "VFM")
        self.assertEqual(ext_x, "xlsx")
        self.assertIn("spreadsheetml", ctype_x)
        back = pd.read_excel(io.BytesIO(raw_xlsx), sheet_name="VFM")
        self.assertEqual(list(back.columns), list(df.columns))
        self.assertAlmostEqual(float(back["oil_rate_stbd"].iloc[0]), 1000.5, places=3)

        raw_csv, ctype_c, ext_c = Workbench.export_table_bytes(df, "csv")
        self.assertEqual(ext_c, "csv")
        self.assertIn("text/csv", ctype_c)
        self.assertIn("GL-01", raw_csv.decode("utf-8"))

    def test_score_audit_and_metric_backcalc(self) -> None:
        import base64
        ref = pd.DataFrame({
            "well_name": ["GL-01", "GL-01", "GL-02", "GL-02"],
            "date": ["2025-07-01", "2025-08-01", "2025-07-01", "2025-08-01"],
            "oil_rate_stbd": [1000.0, 900.0, 1200.0, 1100.0],
            "water_rate_stbd": [400.0, 450.0, 800.0, 850.0],
            "gas_rate_mscfd": [250.0, 225.0, 360.0, 330.0],
            "liquid_rate_stbd": [1400.0, 1350.0, 2000.0, 1950.0],
        })
        sub_good = ref.copy()
        sub_good["oil_rate_stbd"] *= 0.95
        sub_good["water_rate_stbd"] *= 0.95
        sub_good["gas_rate_mscfd"] *= 0.95
        sub_bad = ref.copy()
        sub_bad["oil_rate_stbd"] *= 0.85
        sub_bad["water_rate_stbd"] *= 0.85
        sub_bad["gas_rate_mscfd"] *= 0.85

        b1 = base64.b64encode(sub_good.drop(columns=["liquid_rate_stbd"]).to_csv(index=False).encode()).decode()
        b2 = base64.b64encode(sub_bad.drop(columns=["liquid_rate_stbd"]).to_csv(index=False).encode()).decode()
        df1, aud1 = A.decode_uploaded_file({"name": "sub_5pct.csv", "content_b64": b1})
        df2, aud2 = A.decode_uploaded_file({"name": "sub_15pct.csv", "content_b64": b2})
        ev1 = A.evaluate_submission(df1, ref, aud1, reported_score=25.0)
        ev2 = A.evaluate_submission(df2, ref, aud2, reported_score=15.0)
        bc = A.backcalculate_metrics([ev1, ev2], max_score=30.0)
        self.assertGreater(len(bc["formulas"]), 5)
        best = bc["formulas"][0]
        self.assertLess(best["mae_vs_reported"], 0.5)
        self.assertIn("cap=30%", best["formula"])


if __name__ == "__main__":
    unittest.main()
