# SOCAR Hackathon 2026 — Full Project Handover (< 30 MB per File)

This is a complete, unmodified copy of the working VFM × PTA workflow where **every single file is strictly under 30 MB**, making it ready both for **direct upload to Claude** and for **full local reproduction**.

## Folder Layout
- `vfm_project/` — Core Python package (`src/vfm/`), interactive dashboard (`dashboard/`), configs (`configs/default.json`), scripts (`scripts/`), 55 unit tests (`tests/`), outputs (`outputs/`), lightweight cache (`data_cache/`), and trained model (`models/vfm_stacked_v2.pkl`).
- `vfm_export_tool/` — Complete VFM training, PTA analysis, deliverable builder (`build_deliverables.py`), and executed notebook builder (`build_executed_notebook.py`).
- `deliverables/` — Official Phase 1 VFM & Phase 2 PTA deliverables (`CSV/XLSX`), 32 PTA plots + 4-output `Qo ±20%` Tornado PNG, VFM Correlation Matrix & SHAP PNGs, `vfm_feature_diagnostics.xlsx`, trained `vfm_stacked_v2.pkl`, and executed `SOCAR_2026_VFM_PTA_Solution.ipynb`.
- `pta_project/` — Reference PTA scripts.
- `data/raw/` — Official hackathon metadata CSVs (`well_tests.csv`, `well_completion.csv`, `reservoir_pvt.csv`, `choke_events.csv`) plus the 256.84 MB `high_freq_data.csv` zipped and divided into standalone `< 25 MB` tomes:
- `data/raw/high_freq_data_tomes/high_freq_data_tome_01.zip` — **21.69 MB** (`1,472,751` rows, valid standalone CSV inside `.zip`)
- `data/raw/high_freq_data_tomes/high_freq_data_tome_02.zip` — **21.62 MB** (`1,472,436` rows, valid standalone CSV inside `.zip`)
- `data/raw/high_freq_data_tomes/high_freq_data_tome_03.zip` — **14.22 MB** (`971,886` rows, valid standalone CSV inside `.zip`)
- `claude_upload_zips/` — Ready-to-upload bundles (every `.zip` is strictly `< 25 MB`):
- `claude_upload_zips/01_code_dashboard_scripts_tests.zip` — **0.48 MB**
- `claude_upload_zips/02_deliverables_plots_notebook.zip` — **4.97 MB**
- `claude_upload_zips/03_trained_model_cache_hourly.zip` — **13.55 MB**
- `claude_upload_zips/04_raw_metadata_and_well_tests.zip` — **0.01 MB**
- `claude_upload_zips/05_high_freq_data_tome_01.zip` — **21.69 MB**
- `claude_upload_zips/06_high_freq_data_tome_02.zip` — **21.62 MB**
- `claude_upload_zips/07_high_freq_data_tome_03.zip` — **14.22 MB**

## Uploading to Claude for Verification
Upload the numbered `.zip` files from `claude_upload_zips/` directly (all are standard `.zip` files well under Claude's 30 MB per-file limit). Each `high_freq_data_tome_XX.zip` contains a valid standalone CSV with headers so Claude can inspect individual tomes OR merge them with `reassemble_tomes.py`.

## Full Local Reproduction (Zero Workflow Changes)
```bash
# 1. Restore data/raw/high_freq_data.csv bit-for-bit (SHA-256 verified)
python reassemble_tomes.py

# 2. Rebuild per-well Parquet cache (excluded from handover because each file is ~39 MB)
cd vfm_project
python scripts/01_audit_and_prepare.py

# 3. Run all 55 unit tests
python -m unittest discover -s tests -q

# 4. Launch interactive VFM & PTA Workbench dashboard
python dashboard/server.py
```

- Original `high_freq_data.csv` SHA-256: `0d7c2c33f0c8b2577f7ae94248c20e66a80c9e69cbe2343d46c3a371b45fcf62` (`3,917,073` rows, `256.84 MB`)
