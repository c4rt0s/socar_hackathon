"""Step 09: Benchmark every allocation model (vfm.allocation.models) on the well tests.

Two protocols, both scored on the exact [test_start, test_end] windows at 1-minute resolution:
  * walk_forward (strictly causal): for test k of a well, models see only tests with
    test_end <= test_start_k and telemetry up to test_start_k (models that need the pressure
    history get telemetry up to test_end_k: the test's own rate is never visible). The field
    Pr prior, field water-cut slope and PTA events are also restricted to the cutoff.
    Non-causal models (test_linear) are skipped here.
  * loto: leave-one-test-out (the held-out test is removed; everything else is visible).

Per-well models are fitted on the well's own tests; pooled models (ML) on all wells' tests.

    python scripts/09_model_benchmark.py                      # all models
    python scripts/09_model_benchmark.py physics_wls gbm      # a subset

Writes outputs/model_benchmark_detail.csv, model_benchmark_summary.csv, model_benchmark.md.
"""

from __future__ import annotations

from pathlib import Path
import sys
import time
from typing import Any, Dict, List

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vfm.allocation.models import REGISTRY, make_allocation_model  # noqa: E402
from vfm.config import load_config  # noqa: E402
from vfm.io import load_frame, load_static_tables  # noqa: E402
from vfm.modeling.physics_vfm import field_logit_wc_slope  # noqa: E402
from vfm.physics.reservoir import DEFAULT_REF_TIMESTAMP, fit_pr_and_pi  # noqa: E402
from vfm.pta.runner import PtaRunSettings, detect_pta_events, run_pta_event  # noqa: E402

PHASES = ("oil_rate_stbd", "water_rate_stbd", "gas_rate_mscfd", "liquid_rate_stbd")


def window_mean(pred: pd.DataFrame, hf: pd.DataFrame, t0, t1) -> Dict[str, float]:
    ts = pd.to_datetime(hf["timestamp"], utc=True)
    m = ((ts >= t0) & (ts <= t1)).to_numpy()
    if "is_steady_flowing" in hf.columns and (m & hf["is_steady_flowing"].to_numpy(dtype=bool)).any():
        m = m & hf["is_steady_flowing"].to_numpy(dtype=bool)
    return {c: float(np.nanmean(pred.loc[m, c].to_numpy(dtype="float64"))) if m.any() else np.nan for c in PHASES}


def summarize(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (cv, model), g in detail.groupby(["cv_mode", "model"]):
        r: Dict[str, Any] = {"cv_mode": cv, "model": model, "n_tests": int(len(g))}
        for c in PHASES:
            t, p = g[f"true_{c}"].to_numpy(float), g[f"pred_{c}"].to_numpy(float)
            ok = np.isfinite(t) & np.isfinite(p)
            r[f"{c}_mae"] = float(np.mean(np.abs(p[ok] - t[ok]))) if ok.any() else np.nan
            r[f"{c}_mape"] = float(np.mean(np.abs(p[ok] - t[ok]) / np.maximum(np.abs(t[ok]), 1.0)) * 100) if ok.any() else np.nan
            r[f"{c}_bias"] = float(np.mean(p[ok] - t[ok])) if ok.any() else np.nan
            r[f"{c}_n"] = int(ok.sum())
        rows.append(r)
    return pd.DataFrame(rows).sort_values(["cv_mode", "liquid_rate_stbd_mae"])


def main(selected: List[str]) -> None:
    t_all = time.perf_counter()
    cfg = load_config()
    cache = cfg.resolve_path("cache_dir")
    out_dir = cfg.resolve_path("outputs_dir")
    comp, pvt, _ = load_static_tables(cfg.resolve_path("well_completion_csv"), cfg.resolve_path("reservoir_pvt_csv"),
                                      cfg.resolve_path("choke_events_csv"), schema=cfg.schema)
    wt = load_frame(cache / "well_tests_harmonized")
    wt["test_start"] = pd.to_datetime(wt["test_start"], utc=True)
    wt["test_end"] = pd.to_datetime(wt["test_end"], utc=True)
    wt = wt.sort_values(["well_name", "test_start"]).reset_index(drop=True)
    wells = sorted(wt["well_name"].unique())
    hf = {w: load_frame(cache / "wells" / w) for w in wells}
    ref = cfg.reference_timestamp(DEFAULT_REF_TIMESTAMP)

    # PTA events once (rate history from tests before each shut-in: causal); a fold only uses
    # events that ended before its cutoff.
    pta_settings = PtaRunSettings.from_config(cfg.section("pta"))
    pta_settings.rate_source = "well_tests"
    pvt_by, comp_by = pvt.set_index("well_name"), comp.set_index("well_name")
    events: Dict[str, List[Dict[str, Any]]] = {}
    for w in wells:
        events[w] = [run_pta_event(w, s0, s1, hf[w], wt[wt.well_name == w], pvt_by.loc[w], comp_by.loc[w], pta_settings)
                     for s0, s1 in detect_pta_events(hf[w])]
    grads = pd.DataFrame([{"end": pd.Timestamp(b["shut_in_end"]), "g": b["pr_gradient_psi_ft"]}
                          for p in fit_pr_and_pi(wt, hf, pvt, comp).values() for b in p["buildup_anchors"]])

    def prior_at(cut):
        sub = grads if cut is None else grads[grads["end"] <= cut]
        return float(sub["g"].median()) if len(sub) else None

    from vfm.modeling.physics_vfm import physics_kwargs_from_config
    phys_kw = physics_kwargs_from_config(cfg)
    models = selected or list(REGISTRY)
    phys_cache: Dict[Any, Any] = {}
    feat_cache: Dict[Any, Any] = {}
    rows: List[Dict[str, Any]] = []
    for name in models:
        t0 = time.perf_counter()
        proto = make_allocation_model(name)
        for cv in ("walk_forward", "loto"):
            if cv == "walk_forward" and not proto.causal:
                continue
            for well in wells:
                wg = wt[wt.well_name == well].reset_index(drop=True)
                for k in range(len(wg)):
                    if cv == "walk_forward" and k == 0:
                        continue
                    test = wg.iloc[k]
                    if cv == "walk_forward":
                        cut = test["test_start"]
                        train = wt[wt.test_end <= cut]
                        ctx = {"pr_prior": prior_at(cut), "wc_global_slope": field_logit_wc_slope(wt, ref, cut),
                               "pta_events": events, "ref_timestamp": ref, "fold_key": (cv, well, k)}
                    else:
                        cut = None
                        train = wt[~((wt.well_name == well) & (wt.test_start == test["test_start"]))]
                        ctx = {"pr_prior": prior_at(None), "wc_global_slope": field_logit_wc_slope(train, ref, None),
                               "pta_events": events, "ref_timestamp": ref, "fold_key": (cv, well, k)}
                    ctx.update(physics_cache=phys_cache, feature_cache=feat_cache, physics_kwargs=phys_kw)
                    model = make_allocation_model(name)
                    if model.pooled:
                        fit_tests, fit_hf = train, {w: (hf[w] if cut is None else hf[w][pd.to_datetime(hf[w].timestamp, utc=True) <= cut]) for w in wells}
                        fit_comp, fit_pvt = comp, pvt
                    else:
                        fit_tests = train[train.well_name == well]
                        h = hf[well] if cut is None else hf[well][pd.to_datetime(hf[well].timestamp, utc=True) <= cut]
                        fit_hf = {well: h}
                        fit_comp, fit_pvt = comp[comp.well_name == well], pvt[pvt.well_name == well]
                    if fit_tests[fit_tests.well_name == well].empty and not model.pooled:
                        continue
                    try:
                        model.fit(fit_tests, fit_hf, fit_comp, fit_pvt, max_time=cut, context=ctx)
                        ts = pd.to_datetime(hf[well]["timestamp"], utc=True)
                        if model.needs_history:
                            h = hf[well][ts <= test["test_end"]]
                        else:
                            h = hf[well][(ts >= test["test_start"]) & (ts <= test["test_end"])]
                        pred = model.predict(h, well)
                        est = window_mean(pred, h, test["test_start"], test["test_end"])
                    except Exception as exc:  # a failing model must not stop the benchmark
                        print(f"  {name} {cv} {well} test {k + 1}: {type(exc).__name__}: {exc}")
                        est = {c: np.nan for c in PHASES}
                    rec = {"cv_mode": cv, "model": name, "family": model.family, "well_name": well,
                           "test_idx": k + 1, "test_start": test["test_start"], "test_end": test["test_end"]}
                    for c in PHASES:
                        rec[f"true_{c}"] = float(test[c])
                        rec[f"pred_{c}"] = est[c]
                    rows.append(rec)
        print(f"{name:<20} done in {time.perf_counter() - t0:6.1f}s", flush=True)

    detail = pd.DataFrame(rows)
    out_detail = out_dir / "model_benchmark_detail.csv"
    if selected and out_detail.exists():  # merge a subset run into the existing results
        old = pd.read_csv(out_detail)
        detail = pd.concat([old[~old["model"].isin(models)], detail], ignore_index=True)
    detail.to_csv(out_detail, index=False)
    summary = summarize(detail)
    summary.to_csv(out_dir / "model_benchmark_summary.csv", index=False)
    lines = ["# Allocation model benchmark", "",
             "Walk-forward = strictly causal; LOTO = leave-one-test-out. MAE in STB/d (gas Mscf/d); MAPE in %.", ""]
    for cv in ("walk_forward", "loto"):
        s = summary[summary.cv_mode == cv]
        lines += [f"## {cv}", "", "| Model | N | Oil MAE | Oil MAPE | Water MAE | Water MAPE | Gas MAE | Gas MAPE | Liquid MAE | Liquid MAPE |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for _, r in s.iterrows():
            lines.append(f"| {r.model} | {r.n_tests} | " + " | ".join(
                f"{r[f'{c}_mae']:.2f} | {r[f'{c}_mape']:.2f}%" for c in PHASES) + " |")
        lines.append("")
    (out_dir / "model_benchmark.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nTotal {time.perf_counter() - t_all:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1:])
