#!/usr/bin/env python3
"""Pure-Horner (no Bourdet) PTA case generator for single-phase oil (Q_oil).

Produces three clearly named deliverable cases in CSV and XLSX format:
  1. pta_horner_case1_all_data_r2_090.(csv|xlsx)
       All shut-in points up to t_open; earliest start index giving the most
       late-time points with Horner linear fit R^2 >= 0.90.
  2. pta_horner_case2_trim10_r2_090.(csv|xlsx)
       Same R^2 >= 0.90 late-time max-points logic after trimming the final
       10% of build-up points.
  3. pta_horner_case2_clean_late_r2_9995.(csv|xlsx)
       Pure Horner (no Bourdet) maximizing linear span on uniform Horner-x
       spacing with strict linearity (R^2 >= 0.9995) and allowing late-time
       boundary-flattened tail removal, then refit on all raw 1-min points
       inside the selected [dt_start, dt_end] window.
"""

from __future__ import annotations

from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
VFM_PROJECT = ROOT / "vfm_project"
OUT_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(VFM_PROJECT / "src"))
from vfm.config import load_config  # noqa: E402

EXPORT_COLS = ["well_name", "t_shut", "duration_hr", "m_psi_cycle", "k_md", "skin", "dP_skin"]


def _fit_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float] | None:
    xm, ym = float(x.mean()), float(y.mean())
    dx, dy = x - xm, y - ym
    sxx = float(np.dot(dx, dx))
    if sxx <= 1e-12:
        return None
    slope = float(np.dot(dx, dy) / sxx)
    intercept = ym - slope * xm
    resid = y - (intercept + slope * x)
    syy = float(np.dot(dy, dy))
    r2 = 1.0 - float(np.dot(resid, resid)) / syy if syy > 1e-12 else 0.0
    return -slope, intercept, r2  # m (>0), p_star, r2


def _horner_props(m: float, p_star: float, pwf: float, tp_hr: float, qo: float, pvt_row: pd.Series) -> dict:
    p1hr = p_star - m * np.log10(tp_hr + 1.0)
    k = 162.6 * qo * float(pvt_row["oil_fvf_rb_stb"]) * float(pvt_row["oil_viscosity_cp"]) / (
        m * float(pvt_row["net_pay_thickness_ft"])
    )
    denom = (
        float(pvt_row["porosity_frac"])
        * float(pvt_row["oil_viscosity_cp"])
        * float(pvt_row["total_compressibility_1_psi"])
        * float(pvt_row["wellbore_radius_ft"]) ** 2
    )
    skin = 1.1513 * ((p1hr - pwf) / m - np.log10(k / denom) + 3.23)
    dp_skin = 0.87 * m * skin
    return {
        "m_psi_cycle": float(m),
        "k_md": float(k),
        "skin": float(skin),
        "dP_skin": float(dp_skin),
        "p_star": float(p_star),
        "p_1hr": float(p1hr),
    }


def fit_late_max_points(
    dt: np.ndarray,
    pws: np.ndarray,
    pwf: float,
    tp_hr: float,
    qo: float,
    pvt_row: pd.Series,
    r2_thresh: float = 0.90,
    tail_trim_frac: float = 0.0,
) -> dict:
    """Late-time Horner fit pinned at end*(1 - tail_trim_frac), maximizing points with R^2 >= r2_thresh."""
    n_total = len(dt)
    end_idx = max(12, int(np.floor(n_total * (1.0 - tail_trim_frac))))
    x = np.log10((tp_hr + dt[:end_idx]) / dt[:end_idx])
    y = pws[:end_idx]

    chosen = None
    best_fallback = None
    for i in range(0, end_idx - 10):
        res = _fit_line(x[i:], y[i:])
        if res is None:
            continue
        m, p_star, r2 = res
        if m <= 0:
            continue
        if best_fallback is None or r2 > best_fallback[5]:
            best_fallback = (i, end_idx - 1, end_idx - i, m, p_star, r2)
        if r2 >= r2_thresh:
            chosen = (i, end_idx - 1, end_idx - i, m, p_star, r2)
            break

    i0, i1, n_pts, m, p_star, r2 = chosen if chosen is not None else best_fallback
    out = _horner_props(m, p_star, pwf, tp_hr, qo, pvt_row)
    out.update(dt_start_hr=float(dt[i0]), dt_end_hr=float(dt[i1]), n_pts=int(n_pts), n_total=int(n_total), r2=float(r2))
    return out


def fit_clean_late_horner(
    dt: np.ndarray,
    pws: np.ndarray,
    pwf: float,
    tp_hr: float,
    qo: float,
    pvt_row: pd.Series,
    r2_thresh: float = 0.9995,
    min_cycles: float = 0.45,
    n_per_decade: int = 20,
) -> dict:
    """Pure Horner (no Bourdet) maximizing linear span on uniform Horner-x bins (R^2 >= r2_thresh),
    allowing removal of late-time boundary-flattened data, then refitting on all raw 1-min points."""
    edges = 10.0 ** np.arange(np.floor(np.log10(dt.min())), np.log10(dt.max()) + 1e-9, 1.0 / n_per_decade)
    idx = np.digitize(dt, edges)
    tb, pb = [], []
    for b in np.unique(idx):
        s = idx == b
        tb.append(float(np.exp(np.mean(np.log(dt[s])))))
        pb.append(float(np.median(pws[s])))
    tb_arr, pb_arr = np.asarray(tb), np.asarray(pb)
    xb = np.log10((tp_hr + tb_arr) / tb_arr)

    best = None
    fallback = None
    for i in range(len(tb_arr)):
        if tb_arr[i] < 0.25:
            continue
        for j in range(len(tb_arr), i + 8, -1):
            if np.log10(tb_arr[j - 1] / tb_arr[i]) < min_cycles:
                break
            res = _fit_line(xb[i:j], pb_arr[i:j])
            if res is None:
                continue
            m, p_star, r2 = res
            if m <= 0:
                continue
            if fallback is None or r2 > fallback[5]:
                fallback = (tb_arr[i], tb_arr[j - 1], j - i, m, p_star, r2)
            if r2 >= r2_thresh:
                span = j - i
                if best is None or span > best[2] or (span == best[2] and r2 > best[5]):
                    best = (tb_arr[i], tb_arr[j - 1], span, m, p_star, r2)
                break

    dt_lo, dt_hi = (best if best is not None else fallback)[:2]
    sel = (dt >= dt_lo) & (dt <= dt_hi)
    x_raw = np.log10((tp_hr + dt[sel]) / dt[sel])
    m, p_star, r2_raw = _fit_line(x_raw, pws[sel])
    out = _horner_props(m, p_star, pwf, tp_hr, qo, pvt_row)
    out.update(
        dt_start_hr=float(dt_lo),
        dt_end_hr=float(dt_hi),
        n_pts=int(sel.sum()),
        n_total=int(len(dt)),
        r2=float(r2_raw),
    )
    return out


def load_shutins_and_data():
    cfg = load_config()
    ce = pd.read_csv(cfg.resolve_path("choke_events_csv"), parse_dates=["timestamp"]).sort_values(
        ["well_name", "timestamp"]
    )
    pvt = pd.read_csv(cfg.resolve_path("reservoir_pvt_csv")).set_index("well_name")
    comp = pd.read_csv(cfg.resolve_path("well_completion_csv")).set_index("well_name")
    daily = pd.read_csv(VFM_PROJECT / "outputs" / "daily_allocated_rates_physics.csv")
    daily["date"] = pd.to_datetime(daily["date"], utc=True).dt.tz_localize(None)

    shutins = []
    for w, g in ce.groupby("well_name"):
        g = g.reset_index(drop=True)
        for i, r in g.iterrows():
            if r["choke_opening_pct"] == 0 and i + 1 < len(g):
                t0, t1 = r["timestamp"], g.loc[i + 1, "timestamp"]
                shutins.append(
                    {"well_name": w, "t_shut": t0, "t_open": t1, "duration_hr": (t1 - t0).total_seconds() / 3600.0}
                )
    si_df = pd.DataFrame(shutins).sort_values(["well_name", "t_shut"]).reset_index(drop=True)
    windows = {r["well_name"]: (r["t_shut"] - pd.Timedelta("15min"), r["t_open"]) for _, r in si_df.iterrows()}

    parts = {w: [] for w in windows}
    for chunk in pd.read_csv(
        cfg.resolve_path("high_freq_csv"), usecols=["timestamp", "well_name", "dh_pressure_psia"], chunksize=500_000
    ):
        chunk = chunk[chunk["well_name"].isin(windows)]
        if chunk.empty:
            continue
        chunk["timestamp"] = pd.to_datetime(chunk["timestamp"])
        for w, g in chunk.groupby("well_name"):
            lo, hi = windows[w]
            sub = g[(g["timestamp"] >= lo) & (g["timestamp"] <= hi)]
            if not sub.empty:
                parts[w].append(sub)

    hf = {}
    for w in sorted(windows):
        df = pd.concat(parts[w], ignore_index=True).drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
        df["dh_pressure_psia"] = df["dh_pressure_psia"].replace([-999.25, -999.0], np.nan)
        hf[w] = df.dropna(subset=["dh_pressure_psia"])
    return si_df, hf, pvt, comp, daily


def pre_shutin_qo_tp(daily: pd.DataFrame, comp: pd.DataFrame, well: str, t_shut: pd.Timestamp) -> tuple[float, float]:
    dw = daily[daily["well_name"] == well].sort_values("date")
    day_shut = t_shut.floor("D")
    qo = None
    for _, r in dw[dw["date"] <= day_shut].iloc[::-1].iterrows():
        if r["uptime_frac"] >= 0.25 and np.isfinite(r["oil_rate_alloc_stbd"]):
            qo = float(r["oil_rate_alloc_stbd"] / r["uptime_frac"])
            break
    cum_2025 = float(dw[dw["date"] < day_shut]["oil_rate_alloc_stbd"].sum()) + qo * (
        (t_shut - day_shut).total_seconds() / 86400.0
    )
    tp_hr = 24.0 * (float(comp.loc[well, "cum_oil_at_2025_01_01_stb"]) + cum_2025) / qo
    return qo, tp_hr


def main() -> None:
    si_df, hf, pvt, comp, daily = load_shutins_and_data()

    cases = {
        "pta_horner_case1_all_data_r2_090": lambda dt, pws, pwf, tp, qo, P: fit_late_max_points(
            dt, pws, pwf, tp, qo, P, r2_thresh=0.90, tail_trim_frac=0.0
        ),
        "pta_horner_case2_trim10_r2_090": lambda dt, pws, pwf, tp, qo, P: fit_late_max_points(
            dt, pws, pwf, tp, qo, P, r2_thresh=0.90, tail_trim_frac=0.10
        ),
        "pta_horner_case2_clean_late_r2_9995": lambda dt, pws, pwf, tp, qo, P: fit_clean_late_horner(
            dt, pws, pwf, tp, qo, P, r2_thresh=0.9995
        ),
    }

    for name, fn in cases.items():
        rows, diag = [], []
        for _, r in si_df.iterrows():
            w, t0, t1, dur = r["well_name"], r["t_shut"], r["t_open"], float(r["duration_hr"])
            df = hf[w]
            pre = df[(df["timestamp"] >= t0 - pd.Timedelta("10min")) & (df["timestamp"] <= t0)]["dh_pressure_psia"]
            pwf = float(pre.iloc[-4:-1].mean()) if len(pre) >= 4 else float(pre.iloc[-1])
            post = df[(df["timestamp"] > t0) & (df["timestamp"] <= t1)]
            dt = (post["timestamp"] - t0).dt.total_seconds().to_numpy() / 3600.0
            pws = post["dh_pressure_psia"].to_numpy()
            qo, tp_hr = pre_shutin_qo_tp(daily, comp, w, t0)
            res = fn(dt, pws, pwf, tp_hr, qo, pvt.loc[w])
            rows.append(
                {
                    "well_name": w,
                    "t_shut": t0.strftime("%Y-%m-%d %H:%M:%S"),
                    "duration_hr": round(dur, 2),
                    "m_psi_cycle": res["m_psi_cycle"],
                    "k_md": res["k_md"],
                    "skin": res["skin"],
                    "dP_skin": res["dP_skin"],
                }
            )
            diag.append(
                {
                    "well_name": w,
                    "qo_stbd": round(qo, 2),
                    "tp_hr": round(tp_hr, 1),
                    "pwf_psia": round(pwf, 2),
                    "dt_start_hr": round(res["dt_start_hr"], 3),
                    "dt_end_hr": round(res["dt_end_hr"], 3),
                    "n_pts": res["n_pts"],
                    "n_total": res["n_total"],
                    "r2": round(res["r2"], 6),
                    "p_star_psia": round(res["p_star"], 2),
                }
            )
        out_df = pd.DataFrame(rows, columns=EXPORT_COLS)
        diag_df = pd.DataFrame(diag)
        csv_path = OUT_DIR / f"{name}.csv"
        xlsx_path = OUT_DIR / f"{name}.xlsx"
        out_df.to_csv(csv_path, index=False)
        with pd.ExcelWriter(xlsx_path) as writer:
            out_df.to_excel(writer, sheet_name="PTA_Results", index=False)
            diag_df.to_excel(writer, sheet_name="Fit_Diagnostics", index=False)
        print(f"\n=== Saved: {csv_path.name} & {xlsx_path.name} ===")
        print(pd.concat([out_df, diag_df[["dt_start_hr", "dt_end_hr", "n_pts", "r2"]]], axis=1).to_string(index=False))


if __name__ == "__main__":
    main()
