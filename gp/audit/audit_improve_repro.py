"""M6: reproduction checks of the improvement study. Imports `floodnet` by design (like audit_imbalance_repro.py):
retrains one run per new training path (aux10 U-Net, recipe U-Net, rf_context RF; 25% budget, seed 42) into gp/.tmp,
never touching runs/, and compares them with the frozen study runs. Writes output/audit/improve_repro.json."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    import joblib
    import torch

    from floodnet import training
    from floodnet.common import config, environment, inside, read_json, write_json
    from floodnet.improve import ImproveVariant, load_config, run_dir, variant_cfg
    environment()
    cfg, icfg = config(), load_config()
    scratch = ROOT / ".tmp" / f"audit_improve_{time.strftime('%Y%m%d_%H%M%S')}"
    original_run_path = training.run_path
    training.run_path = lambda method, budget, seed, smoke=False, variant=None: inside(scratch / (variant or "base") / f"{method}_{budget}_s{seed}")
    report = {"scratch": str(scratch.relative_to(ROOT)), "retrained": {}}
    try:
        for variant, method in (("aux10", "unet"), ("recipe", "unet"), ("rf_context", "rf")):
            vcfg = variant_cfg(cfg, variant, icfg)
            trainer = training.train_unet if method == "unet" else training.train_rf
            directory = trainer(vcfg, 25, 42, variant=ImproveVariant(variant))
            frozen = run_dir(method, 25, 42, variant)
            key = f"{variant}/{method}_25_s42"
            if method == "unet":
                a = torch.load(directory / "best.pt", map_location="cpu", weights_only=False)
                b = torch.load(frozen / "best.pt", map_location="cpu", weights_only=False)
                differing = [k for k in a["model"] if not torch.equal(a["model"][k], b["model"][k])]
                report["retrained"][key] = {"tensors": len(a["model"]), "differing_tensors": len(differing), "pass": not differing}
            else:
                same_rows = read_json(directory / "sampled_regions.json") == read_json(frozen / "sampled_regions.json")
                mine, theirs = joblib.load(directory / "model.joblib"), joblib.load(frozen / "model.joblib")
                mine.n_jobs = theirs.n_jobs = 1   # thread-order rounding (imbalance ledger): compare trees, predict single-threaded
                trees = all(np.array_equal(p.tree_.feature, q.tree_.feature) and np.array_equal(p.tree_.threshold, q.tree_.threshold)
                            and np.array_equal(p.tree_.value, q.tree_.value) for p, q in zip(mine.estimators_, theirs.estimators_))
                cache = ROOT / "data/features" / read_json(frozen / "run.json")["identity"]["feature_identity"][:16]
                x = np.concatenate([np.load(p)["x"] for p in sorted(cache.glob("test_*.npz"))])
                equal = bool(np.array_equal(mine.predict(x), theirs.predict(x)) and np.array_equal(mine.predict_proba(x), theirs.predict_proba(x)))
                report["retrained"][key] = {"same_training_rows": same_rows, "identical_trees": trees, "trees": len(mine.estimators_),
                                            "features": int(x.shape[1]), "test_regions": len(x), "identical_predictions": equal,
                                            "pass": same_rows and trees and equal}
            torch.cuda.empty_cache()
    finally:
        training.run_path = original_run_path
    report["pass"] = all(r["pass"] for r in report["retrained"].values())
    write_json(ROOT / "output/audit/improve_repro.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
