# Virtual Flow Metering (VFM), Allocation & PTA Workbench

Daily oil, water and gas rates for gas-lift wells from 1-minute telemetry, calibrated against
well tests, with pressure build-up (PTA) analysis linked to the allocation. The default dataset
is GL-01..GL-10 in `../data/raw/`; any dataset that follows the schema works (see
"Using a new dataset").

Two analysers share one engine and one dashboard:

- **Allocation** (`vfm.allocation.models`): twelve interchangeable rate models, among them
  PhysicsVFM (inverse-variance WLS of IPR, choke and tubing VLP), single-estimator physics,
  pressure-rate deconvolution, ridge/GBM/random-forest/kNN regression, a physics×ML hybrid and
  test-based baselines, all scored by the same causal benchmark.
- **PTA** (`vfm.pta`): every choke-closed shut-in is analysed with the Bourdet-derivative kernel
  (superposition, radial-window QC, kh, k, skin, p*, storage) plus Horner, MDH, a nonlinear model
  match (radial + storage + skin, Stehfest), Dietz average pressure and productivity indices
  (actual, theoretical pseudo-steady-state, ideal), flow efficiency and damage ratio.

## Requirements

Python 3.10+ with **numpy and pandas only**. Scripts, tests and the dashboard add `src/` to the
path themselves. `requirements.txt` lists optional packages used only by the legacy HybridVFM.

## Run the pipeline

```powershell
python -m unittest discover -s tests -v       # 46 unit tests
python scripts/01_audit_and_prepare.py        # REQUIRED first: builds data_cache/ (~1.4 GB pickles, ~3 min)
python scripts/05_reservoir_pressure.py       # build-ups -> Pr(t) and productivity index
python scripts/06_calibrate_choke_vlp.py      # per-well choke Cd and VLP slip calibration
python scripts/07_physics_vfm_daily.py        # daily allocated rates, all wells (uncoupled PhysicsVFM)
python scripts/08_benchmark.py                # PhysicsVFM vs baselines, walk-forward + LOTO (~3 min)
python scripts/09_model_benchmark.py          # every allocation model, walk-forward + LOTO (~40 min)
python dashboard/server.py                    # workbench on http://127.0.0.1:8765
```

`scripts/09_model_benchmark.py physics_wls gbm` reruns a subset and merges it into the results.

## Dashboard tabs

`python dashboard/server.py` (standard library HTTP server; the page loads Chart.js from a CDN).
The first start fits the baseline (~1.5 min) and caches it; later starts take seconds.

| Tab | What the engineer does |
|---|---|
| **Allocation** | Pick the active allocation model (computed in the background), compare it with the physics baseline, the PTA-coupled physics and the well tests per phase; inspect the three physics estimators, reservoir pressure Pr(t) and the J multiplier; switch estimators off per well; enter well-condition changes (date, skin, kh). |
| **PTA** | Per build-up: log-log plot with the radial window and the model-match curves; derivative vs Horner vs MDH vs model match vs the project's internal estimate; PI, flow efficiency, Dietz p̄; QC flags; re-run with another rate source (VFM daily allocation or well tests), smoothing or a manual radial window (force-accept is recorded as a WARN). |
| **PTA ↔ VFM link** | One table with PTA and VFM values side by side (kh from PTA vs kh implied by the VFM's J; p̄ vs the VFM's Pr; PTA PI vs the VFM's J) and the decision editor: use / ignore each build-up and apply its pressure, skin, kh and/or PI to the VFM. |
| **Model comparison** | Walk-forward and LOTO results for every model (MAE, MAPE, bias per phase), per-well heat table, predicted-vs-measured scatter, re-run button, and daily rates of several models side by side. |
| **Correlations** | Pearson/Spearman/Kendall heatmap of well-test, daily (active model + sensor means) or hourly sensor variables, per well or all wells, with a scatter of any pair and the strongest pairs. |

Judgement is saved to `outputs/engineer_judgement.json`; **Export** writes a scenario folder;
**Reset** returns to the baseline.

## Allocation models

| Model | Family | Notes |
|---|---|---|
| `physics_wls` | physics | PhysicsVFM: IPR + choke + VLP, inverse-variance WLS (default) |
| `physics_ipr`, `physics_choke`, `physics_vlp` | physics | one estimator only |
| `hybrid_physics_gbm` | hybrid | PhysicsVFM × GBM correction factor (shrunk, clipped; learned on in-sample physics) |
| `deconvolution` | convolution | downhole-pressure deconvolution with a radial + closed-boundary unit response (kh/skin from an accepted PTA, else from the IPR), free depletion trend, last-test rate prior |
| `ridge`, `gbm`, `rf`, `knn` | regression/ML | pooled across wells on log-relative rates and choke/pressure features (leave-one-out well references), numpy-only (`vfm.modeling.ml_numpy`) |
| `test_step` | test-based | last well test carried forward |
| `test_linear` | test-based | interpolation between tests (hindcast only; LOTO only) |

Every model zeroes shut-in minutes exactly and shares one phase split (logit water-cut trend,
Rs-shrunk GOR). Walk-forward folds give models only tests ending before the evaluated test and
causally computed field priors.

## PTA ↔ VFM coupling

Direction PTA ← VFM: the PTA rate history comes from the *uncoupled* VFM daily allocation (or the
tests), so there is no feedback loop. Direction PTA → VFM (`src/vfm/coupling.py`), per build-up:

1. **Pressure**: the Dietz average pressure (or last shut-in pressure / typed value) replaces the
   well's Pr(t) anchor.
2. **Skin / kh**: J scales with `[kh(t)/kh_ref]·(ln(re/rw) − 0.75 + S_ref)/(ln(re/rw) − 0.75 + S(t))`;
   the first observation is the reference, so a later build-up or engineer change moves J.
3. **PI**: the IPR's J is fixed to the PTA productivity index (actual, theoretical or typed).

Guardrails: coupling is off by default; each event and property has its own switch; defaults use
only build-ups with credible radial flow (engineer-forced windows never by default); pressures must
be positive, below 0.8 psi/ft × datum depth and above nearby flowing pressures; skin/kh values are
validated; J multipliers are clipped to [0.25, 4]; PI must be within 0.05–120 STB/d/psi. A judgement
that contradicts the tests raises the IPR's leave-one-test-out error, which automatically lowers its
weight in the reconciliation.

## Benchmark (strictly causal walk-forward, N = 75 tests)

PhysicsVFM after the legacy-cleaning backports, logit water cut, EWMA transients and
critical pressure ratio 0.40 (script 08; MAE in STB/d, gas Mscf/d):

| Phase | Before upgrade | After | LOTO before → after |
|---|---|---|---|
| Oil | 36.76 (3.23%) | **31.71 (2.75%)** | 28.93 → 26.46 |
| Water | 24.83 (3.84%) | **24.55 (3.73%)** | 20.75 → 20.60 |
| Gas | 15.12 (6.15%) | **14.17 (5.69%)** | 14.51 → 14.33 |
| Liquid | 36.98 (2.08%) | **33.48 (1.90%)** | 33.87 → 30.32 |

Last-test persistence: liquid 76.11 STB/d (4.32%). All other allocation models are compared in
`outputs/model_benchmark.md` (script 09) and in the Model comparison tab.

## Using a new dataset

1. Write a small config with only the keys that differ, e.g. `configs/my_field.json`:

   ```json
   {
     "paths": {"high_freq_csv": "D:/field_x/raw/scada.csv", "well_tests_csv": "D:/field_x/raw/tests.csv",
               "choke_events_csv": "D:/field_x/raw/chokes.csv", "well_completion_csv": "D:/field_x/raw/completion.csv",
               "reservoir_pvt_csv": "D:/field_x/raw/pvt.csv", "cache_dir": "data_cache_field_x",
               "outputs_dir": "outputs_field_x", "judgement_file": "outputs_field_x/engineer_judgement.json"},
     "schema": {"high_freq": {"dh_pressure_psia": "PDG_PRESSURE", "whp_psia": "THP"}},
     "assumptions": {"wellhead_temp_degf": 110.0, "choke_critical_pressure_ratio": 0.45},
     "modeling": {"reference_timestamp": "2026-01-01T00:00:00Z"},
     "pta": {"cumulative_oil_column": "cum_oil_stb", "cumulative_oil_date": "2026-01-01T00:00:00Z"}
   }
   ```

   `schema` maps canonical column names to yours; units must match the canonical names (psia, °F,
   STB/d, Mscf/d, %, ft). Missing columns raise an error naming the section to fix. The file is
   merged over `configs/default.json` (a UTF-8 BOM is fine).
2. `python scripts/01_audit_and_prepare.py configs/my_field.json`.
3. `$env:VFM_CONFIG = "configs/my_field.json"` for scripts 05–09, or
   `python dashboard/server.py --config configs/my_field.json`.

## Outputs (`outputs/`)

| File | Contents |
|---|---|
| `qc_audit_report.json` | Cleaning audit per well (sentinels, bounds, spikes, short gaps filled, duplicates) |
| `reservoir_params.csv` | Per-well build-up summary, Pr(t), J and prior source |
| `choke_vlp_params.csv`, `choke_vlp_loto_errors.csv` | Choke and VLP calibration with LOTO errors |
| `daily_allocated_rates_physics.csv` | 2,730 rows: VFM and test-anchored rates, uptime, completeness, transient counts, per-estimator means |
| `benchmark.md`, `benchmark_detail.csv` | PhysicsVFM vs baselines (script 08) |
| `model_benchmark.md`, `model_benchmark_summary.csv`, `model_benchmark_detail.csv` | All allocation models (script 09) |
| `scenarios/<name>/` | Dashboard exports |

## Code layout

```text
src/vfm/
├── io/loaders.py               # CSV loading with column maps; well-test sign and Sm3/d fixes
├── preprocessing/cleaning.py   # sentinels, bounds, per-sensor Hampel floors, gas-lift-safe despiking, time-aware gap fill
├── preprocessing/states.py     # shut-in = choke closed; flow stall and transient flags; no choke back-fill
├── physics/reservoir.py        # datum correction, build-up (Bourdet/Horner/MBH), Pr(t), IPR with coupling hooks
├── physics/choke.py, vlp.py    # multiphase choke; two-segment tubing VLP and inversion
├── modeling/physics_vfm.py     # PhysicsVFM: WLS, logit water cut, EWMA transients, coupling hooks
├── modeling/ml_numpy.py        # ridge, tree, GBM, random forest, kNN (numpy)
├── allocation/models.py        # allocation model zoo (one interface)
├── allocation/deconvolution.py # unit response + bounded deconvolution solver
├── allocation/daily.py         # daily aggregation and exact-window test anchoring
├── allocation/reconciler.py    # legacy daily API (fixed) and manifold reconciliation
├── pta/kernel.py               # build-up kernel (spec implementation)
├── pta/methods.py              # Horner, MDH, model match, productivity indices
├── pta/runner.py               # project data -> PTA events
├── coupling.py                 # engineer judgement and guardrails
└── workbench.py                # engine behind the dashboard
dashboard/  server.py · static/index.html · static/tab_models.js
```

## Known limitations

- **Small data.** 85 well tests in total; ML models are pooled across wells on relative rates and
  are evaluated honestly, but cannot learn much more than choke/pressure response shapes.
- **Deconvolution in a depleting closed system** fixes rate *changes* from pressure, not the level
  (a constant rate and the depletion trend look alike), so it relies on a last-test rate prior.
- **The hybrid correction** is learned on in-sample physics fits; it is shrunk and clipped.
- **Build-up kh and skin** are only reported with credible radial flow; GL-01 and GL-10 show
  falling derivatives (boundary/depletion) and are used only as last-shut-in pressures.
- **Reservoir pressure for wells without build-ups** (GL-03, 05, 07, 09) is weakly determined; it
  uses the median build-up gradient as a prior.
- **Constant wellhead temperature** (120 °F) and a fitted critical pressure ratio (0.40, chosen by
  benchmark; on walk-forward GL-04 and GL-10 are slightly worse than with 0.55, all other wells
  and all field-level metrics better).
- **Uncertainty is optimistic**: reconciliation outlier minutes are flagged but still used.
