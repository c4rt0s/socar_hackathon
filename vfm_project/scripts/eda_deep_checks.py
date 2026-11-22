"""Deep diagnostic checks on shut-ins, GL-02 choke behavior, GL-03 shut-in flags, and GL-04 gas-lift."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vfm.config import load_config
from vfm.io import load_frame, load_static_tables


def main() -> None:
    cfg = load_config()
    cache_dir = cfg.resolve_path("cache_dir")
    eda_dir = cfg.resolve_path("outputs_dir") / "eda"
    eda_dir.mkdir(parents=True, exist_ok=True)

    completion, pvt, choke_events = load_static_tables(
        cfg.resolve_path("well_completion_csv"),
        cfg.resolve_path("reservoir_pvt_csv"),
        cfg.resolve_path("choke_events_csv"),
        schema=cfg.schema,
    )
    wt = pd.read_csv(cache_dir / "well_tests_harmonized.csv")
    wt["test_start"] = pd.to_datetime(wt["test_start"], utc=True)
    wt["test_end"] = pd.to_datetime(wt["test_end"], utc=True)

    with (cfg.resolve_path("outputs_dir") / "qc_audit_report.json").open("r", encoding="utf-8") as f:
        qc = json.load(f)

    print("=== SHUT-IN / TRANSIENT MINUTES FROM QC REPORT ===")
    for w, info in sorted(qc["high_freq_wells"].items()):
        print(
            f"{w}: raw={info['raw_rows']}, dedup={info['deduped_rows']}, "
            f"shut_in={info['shut_in_minutes']}, transient={info['transient_minutes']}, "
            f"steady={info['steady_flowing_minutes']}"
        )

    # Check non-zero shut-in rows for wells not expected to shut in (e.g. GL-03, GL-05, GL-07, GL-09)
    print("\n=== NON-STANDARD SHUT-IN EVENTS (choke > 0.5 but dp_choke <= 2.0) ===")
    for w in sorted(completion["well_name"].unique()):
        hf = load_frame(cache_dir / "wells" / w)
        si = hf[hf["is_shut_in"]]
        if len(si):
            non_zero_chk = si[si["choke_opening_pct"] > 0.5]
            print(
                f"{w}: total_shut_in={len(si)}, choke<=0.5={len(si)-len(non_zero_chk)}, "
                f"choke>0.5 & dp<=2={len(non_zero_chk)}"
            )
            if len(non_zero_chk) > 0:
                print(
                    non_zero_chk[["timestamp", "whp_psia", "whp_downstream_psia", "dh_pressure_psia", "choke_opening_pct", "dp_choke_psi"]]
                    .head(5)
                    .to_string(index=False)
                )

    # Build-up summary for the 6 actual shut-ins
    print("\n=== SHUT-IN BUILD-UP SUMMARY ===")
    pbu_rows = []
    for w in ("GL-01", "GL-02", "GL-04", "GL-06", "GL-08", "GL-10"):
        hf = load_frame(cache_dir / "wells" / w).sort_values("timestamp").reset_index(drop=True)
        si = hf[hf["is_shut_in"] & (hf["choke_opening_pct"] <= 0.5)]
        t_start = si["timestamp"].min()
        t_end = si["timestamp"].max()
        dur_h = (t_end - t_start).total_seconds() / 3600.0
        pre = hf[(hf["timestamp"] >= t_start - pd.Timedelta(hours=2)) & (hf["timestamp"] < t_start)]
        pwf_before = float(pre["dh_pressure_psia"].median())
        p_si_start = float(si["dh_pressure_psia"].iloc[0])
        p_si_end = float(si["dh_pressure_psia"].iloc[-1])
        p_si_max = float(si["dh_pressure_psia"].max())
        last_hour = si[si["timestamp"] >= t_end - pd.Timedelta(hours=1)]
        dp_last_hour = float(last_hour["dh_pressure_psia"].iloc[-1] - last_hour["dh_pressure_psia"].iloc[0])
        pbu_rows.append({
            "well_name": w,
            "shut_in_start": t_start,
            "shut_in_end": t_end,
            "duration_hours": round(dur_h, 2),
            "pwf_2h_before_psia": round(pwf_before, 2),
            "p_si_first_psia": round(p_si_start, 2),
            "p_si_last_psia": round(p_si_end, 2),
            "p_si_max_psia": round(p_si_max, 2),
            "buildup_dp_psi": round(p_si_end - pwf_before, 2),
            "dp_last_1h_psi": round(dp_last_hour, 3),
        })
    pbu_df = pd.DataFrame(pbu_rows)
    pbu_df.to_csv(eda_dir / "shutin_buildup_summary.csv", index=False)
    print(pbu_df.to_string(index=False))

    # Inspect GL-02 across all 8 tests
    print("\n=== GL-02 TEST-BY-TEST BREAKDOWN ===")
    hf2 = load_frame(cache_dir / "wells" / "GL-02").sort_values("timestamp")
    wt2 = wt[wt["well_name"] == "GL-02"].sort_values("test_start")
    rows2 = []
    for _, r in wt2.iterrows():
        win = hf2[(hf2["timestamp"] >= r["test_start"]) & (hf2["timestamp"] <= r["test_end"])]
        chk = float(win["choke_opening_pct"].median())
        whp = float(win["whp_psia"].median())
        whp_ds = float(win["whp_downstream_psia"].median())
        dp = whp - whp_ds
        d_in = (chk / 100.0) * 2.0
        # Gilbert proxy: q ~ whp * d^1.89 / glr^0.546 (or with subcritical dp correction)
        glr = float(r["glr_scf_stb"])
        gilbert = whp * ((chk / 100.0 * 128.0) ** 1.89) / (glr ** 0.546)
        rows2.append({
            "test_start": str(r["test_start"])[:10],
            "choke_pct": chk,
            "whp": round(whp, 2),
            "whp_ds": round(whp_ds, 2),
            "dp_choke": round(dp, 2),
            "dh_p": round(float(win["dh_pressure_psia"].median()), 2),
            "wc_pct": r["water_cut_pct"],
            "glr": glr,
            "qliq": r["liquid_rate_stbd"],
            "d2_sqrt_dp": round((d_in ** 2) * np.sqrt(dp), 4),
            "d_sqrt_dp": round(d_in * np.sqrt(dp), 4),
            "d2_sqrt_whp_dp": round((d_in ** 2) * np.sqrt(whp * dp), 2),
            "gilbert_crit": round(gilbert, 1),
        })
    print(pd.DataFrame(rows2).to_string(index=False))

    # Inspect GL-04 gas lift & annulus variability
    print("\n=== GL-04 TEST-BY-TEST GAS LIFT & STD ===")
    hf4 = load_frame(cache_dir / "wells" / "GL-04").sort_values("timestamp")
    wt4 = wt[wt["well_name"] == "GL-04"].sort_values("test_start")
    rows4 = []
    for _, r in wt4.iterrows():
        win = hf4[(hf4["timestamp"] >= r["test_start"]) & (hf4["timestamp"] <= r["test_end"])]
        rows4.append({
            "test_start": str(r["test_start"])[:10],
            "choke_pct": round(float(win["choke_opening_pct"].median()), 1),
            "whp": round(float(win["whp_psia"].median()), 1),
            "ann_p": round(float(win["annulus_pressure_psia"].median()), 1),
            "gl_rate": round(float(win["gas_lift_rate_mscfd"].median()), 1),
            "qliq": r["liquid_rate_stbd"],
            "whp_std_15m_med": round(float(win["whp_std_15m"].median()), 2),
            "ann_std_15m_med": round(float(win["annulus_std_15m"].median()), 2),
        })
    print(pd.DataFrame(rows4).to_string(index=False))


if __name__ == "__main__":
    main()
