"""R1 + L4: retrain all 12 models from scratch into a scratch folder and record which images training touched.

This is the one audit script that imports `floodnet`, by design: it reruns the project's own training code
(same config, data and seeds) to test reproducibility. Frozen `runs/` are never written (G1 checks this).
The retrained models are then scored by the independent `audit_eval.py --root <scratch>`.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

AUDIT = Path(__file__).resolve().parent
ROOT = AUDIT.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(AUDIT))
from independent import read_json, write_json  # noqa: E402

import floodnet.data as fdata  # noqa: E402
import floodnet.neural as fneural  # noqa: E402
import floodnet.training as ftraining  # noqa: E402
from floodnet.common import config, environment, inside  # noqa: E402
from floodnet.evaluation import final_runs  # noqa: E402


def main():
    environment()
    cfg = config()
    scratch_name = f".tmp/audit_runs_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    scratch = inside(scratch_name)
    scratch.mkdir(parents=True)
    # Redirect every run directory into the scratch folder; frozen runs/ stay untouched.
    ftraining.run_path = lambda method, budget, seed, smoke=False: inside(f"{scratch_name}/{method}_{budget}_s{seed}")

    # L4 instrumentation (main process: datasets are built here and pickled to loader workers unchanged).
    log = {"select": [], "datasets": [], "rf_tables": []}
    original_select = fdata.Store.select
    def select(self, split, budget=None):
        log["select"].append([split, budget])
        return original_select(self, split, budget)
    fdata.Store.select = select
    original_dataset_init = fneural.ImageDataset.__init__
    def dataset_init(self, store, records, cfg, seed, training=False):
        log["datasets"].append({"training": training, "seed": seed, "ids": [r["id"] for r in records]})
        return original_dataset_init(self, store, records, cfg, seed, training)
    fneural.ImageDataset.__init__ = dataset_init
    original_table = ftraining.training_table
    def training_table(cfg, store, budget, seed):
        x, y, references = original_table(cfg, store, budget, seed)
        log["rf_tables"].append({"budget": budget, "seed": seed, "ids": sorted({r[0] for r in references}), "rows": len(y)})
        return x, y, references
    ftraining.training_table = training_table

    timings, runs = {}, []
    for method, budget, seed in final_runs(cfg):
        before = {k: len(v) for k, v in log.items()}
        started = time.perf_counter()
        (ftraining.train_rf if method == "rf" else ftraining.train_unet)(cfg, budget, seed)
        timings[f"{method}_{budget}_s{seed}"] = time.perf_counter() - started
        runs.append({"run": f"{method}_{budget}_s{seed}", "method": method, "budget": budget, "seed": seed,
                     "select_calls": log["select"][before["select"]:], "datasets": log["datasets"][before["datasets"]:],
                     "rf_tables": log["rf_tables"][before["rf_tables"]:]})

    # L4: compare what training touched with the budget lists and the validation split.
    processed = read_json(ROOT / "data/manifests/processed.json")
    split_of = {r["id"]: r["split"] for r in processed["records"]}
    val_ids = sorted(i for i, s in split_of.items() if s == "val")
    budgets = {b: sorted(read_json(ROOT / f"data/manifests/budget_{b}.json")["ids"]) for b in (25, 100)}
    isolation = []
    for run in runs:
        touched = [i for d in run["datasets"] for i in d["ids"]] + [i for t in run["rf_tables"] for i in t["ids"]]
        row = {"run": run["run"], "test_ids_touched": sum(split_of[i] == "test" for i in touched),
               "test_select_calls": sum(split == "test" for split, _ in run["select_calls"])}
        if run["method"] == "unet":
            train_sets = [d["ids"] for d in run["datasets"] if d["training"]]
            val_sets = [d["ids"] for d in run["datasets"] if not d["training"]]
            row.update(train_equals_budget=len(train_sets) == 1 and sorted(train_sets[0]) == budgets[run["budget"]],
                       val_equals_val_split=len(val_sets) == 1 and sorted(val_sets[0]) == val_ids,
                       train_images=len(train_sets[0]) if train_sets else 0)
        else:
            row.update(train_equals_budget=len(run["rf_tables"]) == 1 and run["rf_tables"][0]["ids"] == budgets[run["budget"]],
                       train_images=len(run["rf_tables"][0]["ids"]) if run["rf_tables"] else 0)
        isolation.append(row)
    l4_pass = all(r["test_ids_touched"] == 0 and r["test_select_calls"] == 0 and r["train_equals_budget"] and r.get("val_equals_val_split", True)
                  for r in isolation)

    # R1 (artefact level): compare retrained artefacts with the frozen ones.
    import joblib
    import torch
    feature_dir = next(p for p in (ROOT / "data/features").iterdir() if p.is_dir() and p.name != "overlap" and (p / "manifest.json").exists())
    eval_ids = [i for i, s in split_of.items() if s in ("val", "test")]
    features = [np.load(feature_dir / f"{i}.npz")["x"] for i in eval_ids]
    comparisons = []
    for method, budget, seed in final_runs(cfg):
        name = f"{method}_{budget}_s{seed}"
        frozen, fresh = ROOT / "runs" / name, scratch / name
        frozen_run, fresh_run = read_json(frozen / "run.json"), read_json(fresh / "run.json")
        if method == "unet":
            a = torch.load(frozen / "best.pt", map_location="cpu", weights_only=False)["model"]
            b = torch.load(fresh / "best.pt", map_location="cpu", weights_only=False)["model"]
            differing = [k for k in a if not torch.equal(a[k], b[k])]
            max_diff = max((float((a[k].float() - b[k].float()).abs().max()) for k in differing), default=0.0)
            curve = lambda path: [e["val_flooded_miou"] for e in read_json(path / "history.json")]
            history_equal = curve(frozen) == curve(fresh)
            comparisons.append({"run": name, "tensors": len(a), "differing_tensors": len(differing), "max_abs_difference": max_diff,
                                "best_epoch": [frozen_run["best_epoch"], fresh_run["best_epoch"]],
                                "epochs": [frozen_run["epochs"], fresh_run["epochs"]], "validation_curve_equal": history_equal,
                                "bitwise_identical": not differing})
        else:
            a, b = joblib.load(frozen / "model.joblib"), joblib.load(fresh / "model.joblib")
            a.n_jobs = b.n_jobs = 1
            pa, pb = np.concatenate([a.predict(f) for f in features]), np.concatenate([b.predict(f) for f in features])
            comparisons.append({"run": name, "regions_predicted": int(len(pa)), "differing_region_predictions": int((pa != pb).sum()),
                                "region_class_counts": [frozen_run["region_class_counts"], fresh_run["region_class_counts"]],
                                "sampled_regions_equal": read_json(frozen / "sampled_regions.json") == read_json(fresh / "sampled_regions.json"),
                                "bitwise_identical": bool((pa == pb).all()) and frozen_run["region_class_counts"] == fresh_run["region_class_counts"]})
    write_json("retrain.json", {"scratch": scratch_name, "seconds": timings, "L4": {"runs": isolation, "pass": l4_pass},
                                "R1_artifacts": comparisons, "instrumentation": [{k: v for k, v in r.items() if k != "datasets"} |
                                                                                 {"dataset_sizes": [len(d["ids"]) for d in r["datasets"]]} for r in runs]})
    print(json.dumps({"scratch": scratch_name, "L4": l4_pass, "bitwise_identical": sum(c["bitwise_identical"] for c in comparisons)}))


if __name__ == "__main__":
    main()
