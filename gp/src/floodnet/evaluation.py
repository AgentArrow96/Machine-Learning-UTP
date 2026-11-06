"""Shared held-out evaluation, experiment freeze, and end-to-end latency."""
from __future__ import annotations

import time
import numpy as np
import joblib
from .common import ProjectError, digest, inside, log, now, read_json, relative, sha256, write_json
from .data import Store
from .features import extract_regions
from .metrics import confusion, score


class RFEngine:
    device = "CPU"

    def __init__(self, directory, run):
        self.cfg = run["config"]
        self.model = joblib.load(directory / run["checkpoint"])
        # One image is ~250 regions: thread fan-out costs more than it saves and inflates latency.
        self.model.n_jobs = 1

    def predict(self, rgb, geometry):
        from .data import remap
        from .features import region_features
        x, _, regions = region_features(rgb, geometry, self.cfg)
        labels = self.model.predict(x)
        if self.cfg["rf"].get("aux10"):
            labels = remap(labels.astype(np.int64)[None]).ravel()  # 10-class forest -> 3 classes
        result = np.zeros(regions.shape, dtype=np.uint8)
        valid = regions >= 0
        result[valid] = labels[regions[valid]]
        return result

    def synchronize(self):
        pass


class UNetEngine:
    device = "CUDA"

    def __init__(self, directory, run):
        import torch
        from .neural import build_model, require_cuda
        require_cuda()
        self.torch = torch
        self.cfg = run["config"]
        self.dtype = torch.bfloat16 if "bfloat16" in run["precision"] else torch.float16
        self.model = build_model(run["config"], pretrained=False).cuda()
        saved = torch.load(directory / run["checkpoint"], map_location="cpu", weights_only=False)
        if saved["identity"] != run["identity"]:
            raise ProjectError("U-Net checkpoint provenance differs from run metadata")
        self.model.load_state_dict(saved["model"]); self.model.eval()

    def logits(self, rgb):
        from .neural import tensor_image
        torch = self.torch
        x = tensor_image(rgb)[None].pin_memory().cuda(non_blocking=True)
        from .neural import three_class
        with torch.inference_mode(), torch.autocast("cuda", dtype=self.dtype):
            return three_class(self.model(x), self.cfg)

    def predict(self, rgb, geometry):
        return self.logits(rgb).argmax(1)[0].cpu().numpy().astype(np.uint8)

    def probabilities(self, rgb):
        """Float32 softmax of the same logits that predict() takes the argmax of."""
        with self.torch.inference_mode():
            return self.torch.softmax(self.logits(rgb).float(), 1)[0].cpu().numpy()

    def synchronize(self):
        self.torch.cuda.synchronize()


def checkpoint_file(directory, run):
    """Hybrid runs train nothing: their checkpoint is the base U-Net run's checkpoint."""
    base = run.get("base_run")
    if not base:
        return directory / run["checkpoint"]
    return inside(base.get("directory", f"runs/{base['run_id']}")) / base["checkpoint"]


def load_engine(directory, run):
    if sha256(checkpoint_file(directory, run)) != run["checkpoint_sha256"]:
        raise ProjectError("Checkpoint hash changed since training completed")
    method = run["identity"]["method"]
    if method == "hybrid":
        from .hybrid import HybridEngine
        return HybridEngine(run)
    return (RFEngine if method == "rf" else UNetEngine)(directory, run)


def final_runs(cfg):
    """Every (method, budget, seed) combination of the final experiment, in display order."""
    return [(method, budget, seed) for method in ("rf", "unet") for budget in (25, 100) for seed in cfg["train_seeds"]]


def freeze_experiment(cfg, store):
    from .training import run_path
    runs = []
    configs = {}
    overlap_path = inside("data/manifests/overlap.json")
    if not overlap_path.exists():
        raise ProjectError("Run the overlap stage before held-out test evaluation")
    overlap = read_json(overlap_path)
    if overlap["processed_identity"] != store.manifest["identity"]:
        raise ProjectError("Overlap analysis belongs to a different preprocessing manifest")
    for method, budget, seed in final_runs(cfg):
        directory = run_path(method, budget, seed)
        path = directory / "run.json"
        if not path.exists():
            raise ProjectError(f"All {len(final_runs(cfg))} final models must complete validation before test evaluation; missing {directory.name}")
        run = read_json(path)
        val_path = directory / "evaluation_val.json"
        if run["status"] != "complete" or run["run_type"] != "final" or not val_path.exists():
            raise ProjectError(f"Final trained/validated run missing: {directory.name}")
        identity = run["identity"]
        if identity["processed_identity"] != store.manifest["identity"] or identity["budget_identity"] != store.manifest["budgets"][str(budget)]["identity"]:
            raise ProjectError("Final runs do not share the active dataset and budgets")
        if method in configs and configs[method] != identity["config_hash"]:
            raise ProjectError("The same method must share one configuration across budgets and seeds")
        configs[method] = identity["config_hash"]
        if sha256(directory / run["checkpoint"]) != run["checkpoint_sha256"]:
            raise ProjectError("A final checkpoint changed")
        validation = read_json(val_path)
        if (validation["run_identity"] != identity or validation["checkpoint_sha256"] != run["checkpoint_sha256"]
                or validation["image_count"] != len(store.select("val")) or validation["split"] != "val"):
            raise ProjectError("Stale validation result")
        runs.append({"run_id": run["run_id"], "directory": relative(directory), "identity": identity,
                     "checkpoint_sha256": run["checkpoint_sha256"], "validation_sha256": sha256(val_path)})
    frozen = {"schema_version": 2, "processed_identity": store.manifest["identity"], "train_seeds": cfg["train_seeds"],
              "overlap_identity": overlap["identity"], "runs": runs}
    if len({r["identity"]["source_identity"] for r in runs}) != 1:
        raise ProjectError("All final runs must use one frozen implementation")
    frozen["identity"] = digest(frozen)
    path = inside("runs/final_experiment.json")
    if path.exists():
        old = read_json(path)
        if old["identity"] != frozen["identity"]:
            raise ProjectError("Final experiment is already frozen with different artifacts; preserve it before starting a new experiment.")
        return old
    frozen["frozen_at"] = now()
    write_json(path, frozen)
    return frozen


def subset_metrics(per_image, ids):
    """Pool the saved per-image confusion counts over a chosen image subset (no re-inference)."""
    ids = set(ids)
    matrices = [item["metrics"]["confusion_matrix"] for item in per_image if item["id"] in ids]
    if len(matrices) != len(ids):
        raise ProjectError("Subset contains images missing from the evaluation")
    return score(np.sum(np.array(matrices, dtype=np.int64), axis=0) if matrices else np.zeros((3, 3), dtype=np.int64))


def benchmark(engine, store, records, settings):
    samples = [(store.get(r)[0], r["geometry"]) for r in records[:settings["images"]]]
    if not samples:
        raise ProjectError("No inference timing images")
    for i in range(settings["warmup"]):
        engine.predict(*samples[i % len(samples)])
    engine.synchronize()
    durations = []
    for rgb, geometry in samples:
        engine.synchronize(); started = time.perf_counter()
        engine.predict(rgb, geometry)
        engine.synchronize(); durations.append(time.perf_counter() - started)
    return {"device": engine.device, "batch_size": 1, "warmup_images": settings["warmup"],
            "image_ids": [r["id"] for r in records[:len(samples)]], "seconds": durations,
            "median_seconds": float(np.median(durations)), "p95_seconds": float(np.percentile(durations, 95)),
            "boundary": "Processed RGB already in host RAM -> CPU pixel prediction. Original decoding excluded; RF includes fresh SLIC/features; U-Net includes normalisation/transfers/argmax."}


def evaluate(cfg, run_directory, split="val", final=False):
    directory = inside(run_directory)
    run = read_json(directory / "run.json")
    if run["status"] != "complete":
        raise ProjectError("Evaluate a completed run")
    store = Store(cfg)
    if run["identity"]["processed_identity"] != store.manifest["identity"]:
        raise ProjectError("Run uses a different preprocessing manifest")
    frozen = None
    if split == "test":
        if not final or run["run_type"] != "final":
            raise ProjectError("Held-out test requires --final and every completed, validated final run")
        if "undersample_config_sha256" in run["identity"]:
            # The uniform / undersampling study has its own freeze, before its single test evaluation.
            from .undersample import freeze_undersample
            frozen = freeze_undersample(cfg, store)
            if relative(directory) not in frozen["tested"]:
                raise ProjectError("Only the uniform RF, the chosen ratio and the blackening runs are scored on the test split")
        elif "variant" in run["identity"]:
            # Class-imbalance runs were added after every original test result; they have their own freeze.
            from .imbalance import freeze_imbalance
            frozen = freeze_imbalance(cfg, store)
        elif run["identity"]["method"] == "hybrid":
            # The hybrid was added after the 12-run freeze; it has its own freeze before test.
            from .hybrid import freeze_hybrid
            frozen = freeze_hybrid(cfg, store)
        else:
            frozen = freeze_experiment(cfg, store)
        # Class-imbalance runs share names across variants, so their freeze lists them by directory.
        run_key = relative(directory) if "variant" in run["identity"] else run["run_id"]
        if run_key not in [r["run_id"] for r in frozen["runs"]]:
            raise ProjectError("Run is outside the frozen experiment")
    output = directory / f"evaluation_{split}.json"
    records = store.select(split)
    if run["run_type"] == "smoke":
        records = records[:4]
    if output.exists():
        cached = read_json(output)
        if (cached.get("complete") and cached["run_identity"] == run["identity"]
                and cached["checkpoint_sha256"] == run["checkpoint_sha256"]
                and cached["split"] == split and cached["image_count"] == len(records)
                and [r["id"] for r in cached["per_image"]] == [r["id"] for r in records]
                and cached["experiment_identity"] == (frozen["identity"] if frozen else None)
                and sha256(checkpoint_file(directory, run)) == run["checkpoint_sha256"]
                and sha256(cached["predictions"]) == cached["predictions_sha256"]):
            log(f"Verified existing {split} evaluation: {run['run_id']}")
            return cached
        raise ProjectError("Existing evaluation has changed/incompatible artifacts; preserve and investigate before rerunning.")
    engine = load_engine(directory, run)
    predictions_path = directory / f"predictions_{split}.npy"
    predictions = np.lib.format.open_memmap(predictions_path, mode="w+", dtype=np.uint8,
                                           shape=(len(records), store.manifest["size"], store.manifest["size"]))
    total = np.zeros((3, 3), dtype=np.int64)
    per_image = []
    started = time.perf_counter()
    for i, record in enumerate(records):
        rgb, truth = store.get(record)
        predicted = engine.predict(rgb, record["geometry"])
        predicted[truth == 255] = 0
        predictions[i] = predicted
        matrix = confusion(truth, predicted)
        total += matrix
        per_image.append({"id": record["id"], "prediction_index": i, "metrics": score(matrix)})
        if i % 100 == 0:
            log(f"Evaluate {run['run_id']} {split}: {i + 1}/{len(records)}")
    predictions.flush(); del predictions
    evaluation_seconds = time.perf_counter() - started
    result = {"schema_version": 1, "complete": True, "split": split, "created_at": now(), "run_id": run["run_id"],
              "run_type": run["run_type"], "run_identity": run["identity"], "checkpoint_sha256": run["checkpoint_sha256"],
              "experiment_identity": frozen["identity"] if frozen else None, "image_count": len(records),
              "metrics": score(total), "per_image": per_image, "predictions": relative(predictions_path),
              "predictions_sha256": sha256(predictions_path), "evaluation_seconds": evaluation_seconds,
              "timing": benchmark(engine, store, records, cfg["timing"])}
    write_json(output, result)
    log(f"{run['run_id']} {split}: flooded IoU={result['metrics']['flooded_miou']}; 3-class mIoU={result['metrics']['miou']}")
    return result
