"""Step 03: Train and validate Hybrid VFM using chronological holdout and leave-one-well-out CV."""

from __future__ import annotations

import json
from pathlib import Path
import pickle
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vfm.config import load_config
from vfm.io import load_frame
from vfm.modeling import (
    HybridVFM,
    chronological_holdout_eval,
    detect_best_backend,
    leave_one_well_out_eval,
)


def main() -> None:
    cfg = load_config()
    cache_dir = cfg.resolve_path("cache_dir")
    models_dir = cfg.resolve_path("models_dir")
    outputs_dir = cfg.resolve_path("outputs_dir")
    models_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_frame(cache_dir / "well_test_features")
    qc_ds = dataset[dataset["passes_qc"]].reset_index(drop=True)
    backend = detect_best_backend()
    print(f"Training HybridVFM on {len(qc_ds)} QC-passing tests using backend='{backend}'")

    chrono_metrics, chrono_eval = chronological_holdout_eval(
        qc_ds,
        cutoff_utc=cfg.modeling["time_split_cutoff"],
        backend=backend,
    )
    lowo_metrics, lowo_eval = leave_one_well_out_eval(qc_ds, backend=backend)

    print("\nChronological Holdout Metrics (test_start >= " + cfg.modeling["time_split_cutoff"] + "):")
    for col, m in chrono_metrics.items():
        print(f"  {col:<18} MAE={m['mae']:>7.2f}  RMSE={m['rmse']:>7.2f}  MAPE={m['mape_pct']:>6.2f}%  R2={m['r2']:>6.4f}")

    print("\nLeave-One-Well-Out Metrics:")
    for col, m in lowo_metrics.items():
        print(f"  {col:<18} MAE={m['mae']:>7.2f}  RMSE={m['rmse']:>7.2f}  MAPE={m['mape_pct']:>6.2f}%  R2={m['r2']:>6.4f}")

    full_model = HybridVFM(backend=backend, random_state=cfg.modeling["random_state"]).fit(qc_ds)
    with (models_dir / "hybrid_vfm.pkl").open("wb") as f:
        pickle.dump(full_model, f)

    chrono_eval.to_csv(outputs_dir / "chronological_eval.csv", index=False)
    lowo_eval.to_csv(outputs_dir / "leave_one_well_out_eval.csv", index=False)
    with (outputs_dir / "validation_report.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "backend": backend,
                "n_qc_tests": int(len(qc_ds)),
                "chronological_holdout": chrono_metrics,
                "leave_one_well_out": lowo_metrics,
            },
            f,
            indent=2,
        )
    print(f"\nSaved model to {models_dir / 'hybrid_vfm.pkl'}")


if __name__ == "__main__":
    main()
