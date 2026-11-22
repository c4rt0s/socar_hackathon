"""Unit tests for VFM cleaning, harmonization, physics features, targets, and allocation."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vfm.allocation import anchor_daily_to_well_tests, reconcile_to_manifold_totals
from vfm.io import load_and_harmonize_well_tests
from vfm.modeling import decompose_well_test_targets, reconstruct_phase_rates
from vfm.preprocessing import clean_well_telemetry, classify_well_states


class TestVFMPipeline(unittest.TestCase):
    def test_clean_telemetry_dedup_sentinel_and_spike(self) -> None:
        ts = pd.date_range("2025-01-01 00:00:00+00:00", periods=20, freq="1min")
        df = pd.DataFrame({
            "timestamp": ts,
            "well_name": "GL-01",
            "whp_psia": [290.0] * 20,
            "whp_downstream_psia": [200.0] * 20,
            "dh_pressure_psia": [3240.0] * 20,
            "dh_temperature_degf": [203.0] * 20,
            "annulus_pressure_psia": [1175.0] * 20,
            "gas_lift_rate_mscfd": [820.0] * 20,
        })
        df.loc[5, "whp_psia"] = -999.25
        df.loc[10, "dh_pressure_psia"] = 9000.0  # out-of-bounds spike
        df.loc[14, "whp_psia"] = 950.0           # isolated Hampel spike
        df = pd.concat([df, df.iloc[[3]]], ignore_index=True)  # duplicate row

        cleaned, audit = clean_well_telemetry(
            df,
            physical_bounds={"dh_pressure_psia": (100.0, 6000.0)},
        )
        self.assertEqual(audit["duplicates_removed"], 1)
        self.assertEqual(len(cleaned), 20)
        self.assertAlmostEqual(float(cleaned.loc[5, "whp_psia"]), 290.0, places=3)
        self.assertAlmostEqual(float(cleaned.loc[10, "dh_pressure_psia"]), 3240.0, places=3)
        self.assertAlmostEqual(float(cleaned.loc[14, "whp_psia"]), 290.0, places=3)

    def test_state_classification_shutin_and_transient(self) -> None:
        ts = pd.date_range("2025-01-01 00:00:00+00:00", periods=90, freq="1min")
        df = pd.DataFrame({
            "timestamp": ts,
            "well_name": "GL-01",
            "whp_psia": [290.0] * 90,
            "whp_downstream_psia": [200.0] * 90,
        })
        choke_ev = pd.DataFrame({
            "timestamp": pd.to_datetime(["2025-01-01 00:00:00+00:00", "2025-01-01 00:40:00+00:00"]),
            "well_name": ["GL-01", "GL-01"],
            "choke_opening_pct": [40.0, 0.0],
        })
        labeled = classify_well_states(df, choke_ev, stabilization_buffer_min=30)
        self.assertTrue(bool(labeled.loc[5, "is_transient"]))
        self.assertTrue(bool(labeled.loc[35, "is_steady_flowing"]))
        self.assertTrue(bool(labeled.loc[50, "is_shut_in"]))

    def test_well_test_sign_and_sm3_unit_harmonization(self) -> None:
        csv_text = (
            "well_name,test_start,test_end,oil_rate_stbd,water_rate_stbd,liquid_rate_stbd,"
            "gas_rate_mscfd,total_gas_rate_mscfd,gas_lift_rate_mscfd,gor_scf_stb\n"
            "GL-01,2025-01-03 00:00:00,2025-01-03 12:00:00,1000.0,500.0,1500.0,-250.0,1050.0,800.0,250.0\n"
            "GL-06,2025-01-18 00:00:00,2025-01-18 12:00:00,800.0,-400.0,1200.0,5663.37,26901.00,-750.0,250.0\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "wt.csv"
            p.write_text(csv_text, encoding="utf-8")
            wt, audit = load_and_harmonize_well_tests(p)
            self.assertEqual(len(audit["sign_fixes"]), 3)
            self.assertEqual(len(audit["unit_conversions"]), 1)
            g6 = wt[wt["well_name"] == "GL-06"].iloc[0]
            self.assertAlmostEqual(float(g6["gas_rate_mscfd"]), 200.0, places=1)
            self.assertAlmostEqual(float(g6["total_gas_rate_mscfd"]), 950.0, places=1)

    def test_target_round_trip_and_shutin_zeroing(self) -> None:
        wt = pd.DataFrame({
            "oil_rate_stbd": [1200.0],
            "water_rate_stbd": [800.0],
            "gas_rate_mscfd": [360.0],
        })
        dec = decompose_well_test_targets(wt)
        rec = reconstruct_phase_rates(
            q_liq_stbd=np.array([dec["q_liq_stbd"].iloc[0], 2000.0]),
            water_cut_frac=np.array([dec["water_cut_frac"].iloc[0], 0.4]),
            gor_form_scf_stb=np.array([dec["gor_form_scf_stb"].iloc[0], 300.0]),
            gas_lift_rate_mscfd=np.array([800.0, 800.0]),
            is_shut_in=np.array([False, True]),
        )
        self.assertAlmostEqual(float(rec["oil_rate_stbd"][0]), 1200.0, places=4)
        self.assertAlmostEqual(float(rec["water_rate_stbd"][0]), 800.0, places=4)
        self.assertAlmostEqual(float(rec["gas_rate_mscfd"][0]), 360.0, places=4)
        self.assertEqual(float(rec["oil_rate_stbd"][1]), 0.0)
        self.assertEqual(float(rec["total_gas_rate_mscfd"][1]), 0.0)

    def test_anchoring_and_manifold_reconciliation(self) -> None:
        dates = pd.to_datetime(["2025-01-01", "2025-01-02"], utc=True)
        daily = pd.DataFrame({
            "well_name": ["GL-01", "GL-01", "GL-02", "GL-02"],
            "date": [dates[0], dates[1], dates[0], dates[1]],
            "oil_rate_vfm_stbd": [1000.0, 1000.0, 1500.0, 1500.0],
            "water_rate_vfm_stbd": [500.0, 500.0, 500.0, 500.0],
            "gas_rate_vfm_mscfd": [300.0, 300.0, 400.0, 400.0],
        })
        wt = pd.DataFrame({
            "well_name": ["GL-01", "GL-02"],
            "test_start": [dates[0], dates[0]],
            "oil_rate_stbd": [1100.0, 1500.0],
            "water_rate_stbd": [500.0, 500.0],
            "gas_rate_mscfd": [300.0, 400.0],
        })
        anchored = anchor_daily_to_well_tests(daily, wt)
        totals = pd.DataFrame({
            "date": dates,
            "oil_total_meas_stbd": [2800.0, 2800.0],
        })
        rec = reconcile_to_manifold_totals(anchored, totals)
        day0_sum = rec[rec["date"] == dates[0]]["oil_rate_alloc_stbd"].sum()
        self.assertAlmostEqual(float(day0_sum), 2800.0, places=4)


if __name__ == "__main__":
    unittest.main()
