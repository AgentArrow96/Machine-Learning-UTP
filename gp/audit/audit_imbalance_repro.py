"""I8: reproduction checks of the class-imbalance study. Imports `floodnet` by design (like audit_retrain.py):
retrains one run per family into gp/.tmp (never touching runs/) and regenerates synthetic samples from the
selected generator checkpoints. Writes output/audit/imbalance_repro.json."""
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
    from floodnet.data import Store
    from floodnet.gan_common import generate, load_selected
    from floodnet.imbalance import Variant, family, layouts_for, load_config, set_dir
    environment()
    cfg, icfg = config(), load_config()
    store = Store(cfg)
    scratch = ROOT / ".tmp" / f"audit_imbalance_{time.strftime('%Y%m%d_%H%M%S')}"
    original_run_path = training.run_path
    training.run_path = lambda method, budget, seed, smoke=False, variant=None: inside(scratch / (variant or "base") / f"{method}_{budget}_s{seed}")
    report = {"scratch": str(scratch.relative_to(ROOT)), "retrained": {}, "regenerated": {}}
    try:
        for variant, method in (("oversample", "unet"), ("rgan", "rf")):
            v = Variant(variant, 25, store, cfg, icfg)
            trainer = training.train_unet if method == "unet" else training.train_rf
            directory = trainer(cfg, 25, 42, variant=v)
            frozen = original_run_path(method, 25, 42, variant=variant)
            if method == "unet":
                a = torch.load(directory / "best.pt", map_location="cpu", weights_only=False)["model"]
                b = torch.load(frozen / "best.pt", map_location="cpu", weights_only=False)["model"]
                differing = [k for k in a if not torch.equal(a[k], b[k])]
                report["retrained"][f"{variant}/{method}_25_s42"] = {"tensors": len(a), "differing_tensors": len(differing), "pass": not differing}
            else:
                same_rows = read_json(directory / "sampled_regions.json") == read_json(frozen / "sampled_regions.json")
                mine, theirs = joblib.load(directory / "model.joblib"), joblib.load(frozen / "model.joblib")
                # Grader fix (ledger): with n_jobs>1 scikit-learn sums the trees' probabilities in thread order, so even
                # one model predicted twice differs by ~1e-16. Compare the trees themselves, then predict single-threaded.
                mine.n_jobs = theirs.n_jobs = 1
                trees = all(np.array_equal(a.tree_.feature, b.tree_.feature) and np.array_equal(a.tree_.threshold, b.tree_.threshold)
                            and np.array_equal(a.tree_.value, b.tree_.value) for a, b in zip(mine.estimators_, theirs.estimators_))
                cache = ROOT / "data/features" / read_json(frozen / "run.json")["identity"]["feature_identity"][:16]
                features = sorted(cache.glob("test_*.npz"))
                x = np.concatenate([np.load(p)["x"] for p in features])
                equal = bool(np.array_equal(mine.predict(x), theirs.predict(x)) and np.array_equal(mine.predict_proba(x), theirs.predict_proba(x)))
                report["retrained"][f"{variant}/{method}_25_s42"] = {"same_training_rows": same_rows, "identical_trees": trees, "trees": len(mine.estimators_),
                                                                    "test_regions": len(x), "identical_predictions": equal, "pass": same_rows and trees and equal}
    finally:
        training.run_path = original_run_path
    layouts, _, _ = layouts_for(cfg, icfg, store, 100)
    for kind in ("rgan", "oasis", "spade"):
        generator, _ = load_selected(kind, 100, family(kind), icfg)
        stored = np.load(set_dir(kind, 100) / "rgb.npy", mmap_mode="r")
        again = generate(generator, layouts, icfg["seed"])[:16]
        report["regenerated"][f"{kind}_100"] = {"samples": 16, "byte_identical": bool(np.array_equal(again, np.asarray(stored[:16])))}
        del generator
        torch.cuda.empty_cache()
    report["pass"] = all(r["pass"] for r in report["retrained"].values()) and all(r["byte_identical"] for r in report["regenerated"].values())
    write_json(ROOT / "output/audit/imbalance_repro.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
