"""Hybrid model: saved U-Net probabilities refined by a fully connected CRF. No new training.

Methodology section C: reuse each U-Net checkpoint, start with 5 mean-field iterations, choose
spatial/RGB smoothing on validation images only, keep one setting for both budgets, and let
validation adopt or reject refinement. Added after the 12-run freeze, so it has its own freeze.
"""
from __future__ import annotations

import importlib.metadata
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from .common import ProjectError, digest, inside, log, now, read_json, relative, sha256, write_json
from .data import Store, crop
from .metrics import confusion, score

SELECTION_VERSION = 1


def hybrid_config(path=None):
    hcfg = read_json(path or "configs/hybrid.json")
    if hcfg["backend"] != "densecrf":
        raise ProjectError(f"Unsupported hybrid backend: {hcfg['backend']}")
    return hcfg


def backend_version():
    return f"pydensecrf2 {importlib.metadata.version('pydensecrf2')}"


def crf_params(hcfg, sxy, srgb, compat):
    return {"iterations": hcfg["iterations"], "gaussian": dict(hcfg["gaussian"]),
            "bilateral": {"sxy": sxy, "srgb": srgb, "compat": compat}}


def grid(hcfg):
    return [crf_params(hcfg, sxy, srgb, hcfg["bilateral_compat"])
            for sxy in hcfg["bilateral_grid"]["sxy"] for srgb in hcfg["bilateral_grid"]["srgb"]]


def refine_valid(probs, rgb, params):
    """Mean-field DenseCRF on unpadded content: probs (C,H,W) float, rgb (H,W,3) uint8."""
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_softmax
    classes, height, width = probs.shape
    crf = dcrf.DenseCRF2D(width, height, classes)
    crf.setUnaryEnergy(np.ascontiguousarray(unary_from_softmax(np.ascontiguousarray(probs, dtype=np.float32))))
    crf.addPairwiseGaussian(sxy=params["gaussian"]["sxy"], compat=params["gaussian"]["compat"])
    crf.addPairwiseBilateral(sxy=params["bilateral"]["sxy"], srgb=params["bilateral"]["srgb"],
                             rgbim=np.ascontiguousarray(rgb, dtype=np.uint8), compat=params["bilateral"]["compat"])
    q = np.array(crf.inference(params["iterations"]), dtype=np.float32).reshape(classes, height, width)
    return q.argmax(0).astype(np.uint8)


def valid_window(array, geometry):
    g = geometry
    return array[..., g["top"]:g["top"] + g["height"], g["left"]:g["left"] + g["width"]]


def refine(probs, rgb, geometry, params):
    """Refine the valid rectangle only; padding keeps the U-Net argmax (it is never scored)."""
    labels = probs.argmax(0).astype(np.uint8)
    valid_window(labels, geometry)[...] = refine_valid(valid_window(probs, geometry), crop(rgb, geometry), params)
    return labels


def tuning_job(job):
    """Confusion matrices of one validation image for every candidate setting."""
    probs, rgb, truth, settings = job
    return np.stack([confusion(truth, refine_valid(probs, rgb, params)) for params in settings])


class HybridEngine:
    device = "CUDA + CPU CRF"

    def __init__(self, run):
        from .evaluation import UNetEngine
        base_directory = inside(run["base_run"].get("directory", f"runs/{run['base_run']['run_id']}"))
        base = read_json(base_directory / "run.json")
        if base["identity"] != run["identity"]["base_identity"]:
            raise ProjectError("Hybrid base U-Net run changed since refinement was configured")
        self.unet = UNetEngine(base_directory, base)
        self.params = run["crf"]

    def predict(self, rgb, geometry):
        return refine(self.unet.probabilities(rgb), rgb, geometry, self.params)

    def synchronize(self):
        self.unet.synchronize()


def unet_runs(cfg):
    from .evaluation import final_runs
    return [(budget, seed) for method, budget, seed in final_runs(cfg) if method == "unet"]


def hybrid_runs(cfg):
    return [("hybrid", budget, seed) for budget, seed in unet_runs(cfg)]


def selection_key(cfg, hcfg, store):
    base = read_json("runs/final_experiment.json")
    return digest({"version": SELECTION_VERSION, "hybrid": hcfg, "backend": backend_version(),
                   "processed": store.manifest["identity"], "base_experiment": base["identity"]})


def evaluate_settings(cfg, store, settings, hcfg):
    """Pool validation confusion counts per U-Net run for each setting, plus its own U-Net baseline."""
    from .evaluation import load_engine
    from .training import run_path
    records = store.select("val")
    per_run = {}
    for budget, seed in unet_runs(cfg):
        directory = run_path("unet", budget, seed)
        run = read_json(directory / "run.json")
        engine = load_engine(directory, run)
        baseline = np.zeros((3, 3), dtype=np.int64)
        jobs = []
        for record in records:
            rgb, truth = store.get(record)
            probs = engine.probabilities(rgb)
            baseline += confusion(truth, probs.argmax(0).astype(np.uint8))
            g = record["geometry"]
            jobs.append((valid_window(probs, g).copy(), crop(rgb, g).copy(), crop(truth, g).copy(), settings))
        stored = read_json(directory / "evaluation_val.json")["metrics"]
        if score(baseline) != stored:
            raise ProjectError(f"{run['run_id']}: U-Net probabilities do not reproduce the stored validation result")
        started = time.perf_counter()
        totals = np.zeros((len(settings), 3, 3), dtype=np.int64)
        with ProcessPoolExecutor(max_workers=hcfg["workers"]) as pool:
            for matrices in pool.map(tuning_job, jobs, chunksize=4):
                totals += matrices
        base_score = score(baseline)["flooded_miou"]
        per_run[run["run_id"]] = {"budget": budget, "seed": seed, "unet_flooded_miou": base_score,
                                  "flooded_miou": [score(m)["flooded_miou"] for m in totals],
                                  "gain": [score(m)["flooded_miou"] - base_score for m in totals],
                                  "seconds": time.perf_counter() - started}
        log(f"CRF tuning {run['run_id']}: U-Net {base_score:.4f}; best setting gain {max(per_run[run['run_id']]['gain']):+.4f}")
    return per_run


def select(rounds):
    """Highest mean validation gain over every U-Net run; earlier settings win ties."""
    best = None
    for round_index, round_ in enumerate(rounds):
        for index, params in enumerate(round_["settings"]):
            gain = float(np.mean([item["gain"][index] for item in round_["per_run"].values()]))
            if best is None or gain > best["mean_gain"]:
                best = {"params": params, "mean_gain": gain, "round": round_index, "index": index}
    return best


def tune(cfg, hcfg):
    store = Store(cfg)
    key = selection_key(cfg, hcfg, store)
    path = inside("runs/hybrid_selection.json")
    if path.exists():
        existing = read_json(path)
        if existing["key"] == key:
            log("Verified existing CRF selection; reuse.")
            return existing
        if inside("runs/hybrid_experiment.json").exists():
            raise ProjectError("The hybrid is frozen with a different CRF selection; preserve it before re-tuning.")
    started = time.perf_counter()
    first = grid(hcfg)
    rounds = [{"name": "spatial/RGB grid", "settings": first, "per_run": evaluate_settings(cfg, store, first, hcfg)}]
    best = select(rounds)
    if best["mean_gain"] <= 0:
        # Methodology: the one allowed adjustment is a weaker smoothing strength when refinement hurts.
        b = best["params"]["bilateral"]
        weaker = [crf_params(hcfg, b["sxy"], b["srgb"], compat) for compat in hcfg["weaker_compat"]]
        rounds.append({"name": "weaker smoothing strength", "settings": weaker, "per_run": evaluate_settings(cfg, store, weaker, hcfg)})
        best = select(rounds)
    selection = {"schema_version": 1, "key": key, "backend": backend_version(), "hybrid_config": hcfg,
                 "processed_identity": store.manifest["identity"],
                 "base_experiment_identity": read_json("runs/final_experiment.json")["identity"],
                 "rule": "Highest mean validation flooded-mIoU gain over each run's own U-Net, across all U-Net runs; one setting for every budget and seed.",
                 "rounds": rounds, "selected": best["params"], "selected_mean_gain": best["mean_gain"],
                 "adopted": best["mean_gain"] > 0}
    selection["identity"] = digest(selection)
    selection["created_at"] = now()
    selection["elapsed_seconds"] = time.perf_counter() - started
    write_json(path, selection)
    verdict = "adopted" if selection["adopted"] else "rejected (no validation gain)"
    log(f"CRF selection {best['params']['bilateral']}: mean validation gain {best['mean_gain']:+.4f}; refinement {verdict}")
    return selection


def create_run(cfg, budget, seed):
    """Record a hybrid run: the base U-Net checkpoint plus the validation-selected CRF settings."""
    from .training import run_path, source_identity, versions
    selection = read_json("runs/hybrid_selection.json")
    base_directory = run_path("unet", budget, seed)
    base = read_json(base_directory / "run.json")
    if base["status"] != "complete" or base["run_type"] != "final":
        raise ProjectError(f"{base_directory.name} is not a completed final U-Net run")
    if sha256(base_directory / base["checkpoint"]) != base["checkpoint_sha256"]:
        raise ProjectError(f"{base_directory.name}: checkpoint changed since training")
    directory = run_path("hybrid", budget, seed)
    ids = base["identity"]
    identity = {"method": "hybrid", "budget": budget, "seed": seed, "processed_identity": ids["processed_identity"],
                "audit_identity": ids["audit_identity"], "budget_identity": ids["budget_identity"],
                "config_hash": digest(selection["selected"]), "base_run_id": base["run_id"], "base_identity": ids,
                "base_checkpoint_sha256": base["checkpoint_sha256"], "selection_identity": selection["identity"],
                "crf": selection["selected"], "backend": selection["backend"], "source_identity": source_identity()}
    meta = directory / "run.json"
    if meta.exists():
        old = read_json(meta)
        without_source = lambda value: {k: v for k, v in value.items() if k != "source_identity"}
        if old["status"] == "complete" and without_source(old["identity"]) == without_source(identity):
            log(f"{directory.name} already recorded; reuse.")
            return directory
        raise ProjectError(f"{directory.name} exists with a different base run or CRF selection; preserve it first.")
    directory.mkdir(parents=True, exist_ok=True)
    write_json(meta, {"schema_version": 1, "run_id": directory.name, "run_type": "final", "status": "complete",
                      "identity": identity, "base_run": {"run_id": base["run_id"], "checkpoint": base["checkpoint"]},
                      "checkpoint_sha256": base["checkpoint_sha256"], "config": base["config"], "crf": selection["selected"],
                      "precision": base["precision"], "gpu": base.get("gpu"), "versions": {**versions(), "pydensecrf2": selection["backend"]},
                      "training_seconds": None, "epochs": None, "best_epoch": None, "created_at": now(), "completed_at": now(),
                      "note": "No training: refines the base U-Net checkpoint's probabilities with the validation-selected DenseCRF."})
    log(f"Recorded {directory.name} (refines {base['run_id']})")
    return directory


def freeze_hybrid(cfg, store):
    """Freeze the six hybrid runs and their CRF selection before any hybrid test evaluation."""
    from .training import run_path
    base = read_json("runs/final_experiment.json")
    selection = read_json("runs/hybrid_selection.json")
    frozen_bases = {r["run_id"]: r for r in base["runs"]}
    selected_index = next((round_["per_run"], round_["settings"].index(selection["selected"]))
                          for round_ in selection["rounds"] if selection["selected"] in round_["settings"])
    runs = []
    for _, budget, seed in hybrid_runs(cfg):
        directory = run_path("hybrid", budget, seed)
        if not (directory / "run.json").exists():
            raise ProjectError(f"All hybrid runs must be recorded and validated before test evaluation; missing {directory.name}")
        run = read_json(directory / "run.json")
        identity = run["identity"]
        frozen_base = frozen_bases.get(identity["base_run_id"])
        if (frozen_base is None or frozen_base["identity"] != identity["base_identity"]
                or frozen_base["checkpoint_sha256"] != run["checkpoint_sha256"]):
            raise ProjectError(f"{directory.name} does not refine a frozen U-Net checkpoint")
        if identity["selection_identity"] != selection["identity"] or run["crf"] != selection["selected"]:
            raise ProjectError(f"{directory.name} does not use the current CRF selection")
        val_path = directory / "evaluation_val.json"
        if not val_path.exists():
            raise ProjectError(f"{directory.name} needs its validation evaluation before test")
        validation = read_json(val_path)
        if validation["run_identity"] != identity or validation["image_count"] != len(store.select("val")):
            raise ProjectError(f"{directory.name}: stale validation result")
        per_run, index = selected_index
        tuned = per_run[identity["base_run_id"]]["flooded_miou"][index]
        if abs(validation["metrics"]["flooded_miou"] - tuned) > 1e-12:
            raise ProjectError(f"{directory.name}: validation evaluation does not reproduce the tuning score")
        runs.append({"run_id": run["run_id"], "directory": relative(directory), "identity": identity,
                     "checkpoint_sha256": run["checkpoint_sha256"], "validation_sha256": sha256(val_path)})
    frozen = {"schema_version": 1, "base_experiment_identity": base["identity"], "selection_identity": selection["identity"],
              "selected": selection["selected"], "adopted": selection["adopted"], "runs": runs}
    frozen["identity"] = digest(frozen)
    path = inside("runs/hybrid_experiment.json")
    if path.exists():
        old = read_json(path)
        if old["identity"] != frozen["identity"]:
            raise ProjectError("The hybrid is already frozen with different artifacts; preserve it before changing it.")
        return old
    frozen["frozen_at"] = now()
    write_json(path, frozen)
    return frozen
