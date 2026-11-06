"""Improvement study: 10-class supervision, training recipe, TTA, 512-px input, RF context, ResNet-34, combined.

Pre-registered in configs/improve.json and docs/improve.md. Baselines = the 18 frozen runs. Every prediction is scored
on the frozen 256-px test grid against the same masks; validation alone decides the combined model; the study is
frozen before its single test evaluation. Runs live in runs/improve/<variant>/; caches in data/improve/.
"""
from __future__ import annotations

import copy
import time

import numpy as np

from .common import ProjectError, digest, inside, log, now, read_json, relative, sha256, write_json
from .data import Store, prepare_record, preprocessing_key
from .metrics import confusion, score

CONFIG = "configs/improve.json"
UNET_COMPONENTS = ("aux10", "recipe", "tta", "res512", "encoder")
RF_COMPONENTS = ("aux10", "res512", "rf_context")


def load_config():
    return read_json(CONFIG)


def plan(icfg=None, prune=True):
    """Every planned run. A combined model whose method gained no component on validation would equal the baseline,
    so it is not trained (recorded in runs/improve/combine.json)."""
    icfg = icfg or load_config()
    combine = inside("runs/improve/combine.json")
    empty = set()
    if prune and combine.exists():
        components = read_json(combine)["components"]
        empty = {m for m, family in (("unet", "unet"), ("hybrid", "unet"), ("rf", "rf")) if not components[family]}
    return [(variant, method, budget, seed) for variant, spec in icfg["variants"].items() for method in spec["methods"]
            if not (variant == "combined" and method in empty) for budget in icfg["budgets"] for seed in icfg["train_seeds"]]


def study_path(variant):
    return f"improve/{variant}"


def run_dir(method, budget, seed, variant):
    from .training import run_path
    return run_path(method, budget, seed, variant=study_path(variant))


# ---------- pre-registration ----------

def preregister():
    path = inside("runs/improve/preregistration.json")
    record = {"config_sha256": sha256(CONFIG), "protocol_sha256": sha256("docs/improve.md"), "evals_sha256": sha256("audit/improve_evals.md"),
              "runs_planned": len(plan(prune=False))}
    if path.exists():
        old = read_json(path)
        if {k: old[k] for k in record} == record:
            return old
        if any(inside("runs/improve").glob("*/*/run.json")):
            raise ProjectError("Pre-registered settings changed after runs started; preserve the study first")
    record["registered_at"] = now()
    write_json(path, record)
    return record


def check_preregistration():
    path = inside("runs/improve/preregistration.json")
    if not path.exists():
        raise ProjectError("Run `floodnet improve preregister` first")
    record = read_json(path)
    if record["config_sha256"] != sha256(CONFIG):
        raise ProjectError("configs/improve.json changed after pre-registration")
    return record


# ---------- pure helpers (tested) ----------

def area_weights(n_in, n_out):
    """(n_out, n_in) box-filter weights: each output pixel averages the input pixels it covers, by overlap length."""
    scale = n_in / n_out
    edges = np.arange(n_out + 1) * scale
    lower, upper = edges[:-1, None], edges[1:, None]
    pixels = np.arange(n_in)[None]
    return np.clip(np.minimum(upper, pixels + 1) - np.maximum(lower, pixels), 0, None) / scale


def to_256(probs, g_model, g256):
    """Class probabilities at model size -> the frozen 256-px grid: area-average the unpadded content onto the 256-px
    content (2x2 block means when the content is exactly twice the size; box weights otherwise, e.g. 343 -> 171 rows),
    then place them at the 256-px geometry. Padding gets class 0."""
    classes = probs.shape[0]
    out = np.zeros((classes, 256, 256), dtype=np.float32)
    out[0] = 1
    content = probs[:, g_model["top"]:g_model["top"] + g_model["height"], g_model["left"]:g_model["left"] + g_model["width"]]
    rows, cols = area_weights(content.shape[1], g256["height"]), area_weights(content.shape[2], g256["width"])
    pooled = rows @ content.astype(np.float64) @ cols.T
    out[:, g256["top"]:g256["top"] + g256["height"], g256["left"]:g256["left"] + g256["width"]] = pooled
    return out


def one_hot(labels, classes=3):
    return (np.arange(classes)[:, None, None] == labels[None]).astype(np.float32)


def scaled_crf(params, size):
    """The frozen CRF setting, with its pixel-unit kernel widths scaled to the model size (pre-registered)."""
    if size == 256:
        return params
    factor = size / 256
    out = copy.deepcopy(params)
    out["gaussian"]["sxy"] = params["gaussian"]["sxy"] * factor
    out["bilateral"]["sxy"] = params["bilateral"]["sxy"] * factor
    return out


def passes_combine_rule(differences):
    """differences = {budget: [seed-matched validation changes]}: positive mean and >= 2 of 3 seeds positive, at both budgets."""
    return all(np.mean(values) > 0 and sum(v > 0 for v in values) >= 2 for values in differences.values())


# ---------- variant configurations ----------

def variant_cfg(cfg, variant, icfg, components=None):
    """Base configuration + the variant's pre-registered changes (combined: the selected components)."""
    out = copy.deepcopy(cfg)
    names = components if variant == "combined" else [variant]
    for name in names:
        spec = icfg["variants"][name]
        out["unet"].update(spec.get("unet", {}))
        out["rf"].update(spec.get("rf", {}))
        if "size" in spec:
            out["size"] = spec["size"]
    return out


def inference_of(variant, icfg, components=None):
    names = components if variant == "combined" else [variant]
    return {"tta": "dihedral8"} if any(icfg["variants"][n].get("inference", {}).get("tta") for n in names) else {}


class ImproveVariant:
    """What training.py needs from a variant (identity keys; no extra data; plain store; plain loss)."""
    loss = "ce"

    def __init__(self, variant, components=None):
        self.name = study_path(variant)
        self.variant, self.components = variant, components

    def identity(self):
        return {"variant": self.name, "improve_config_sha256": sha256(CONFIG), "components": self.components}

    def unet_extra(self, budget):
        return []

    def wrap_store(self, store):
        return store

    def rf_extra(self, budget):
        return None


# ---------- 512-px cache (own manifest; never touches data/manifests) ----------

def prepare_size(cfg, size):
    """The released images resized like the frozen cache, at another size. Same image IDs and budgets."""
    from concurrent.futures import ThreadPoolExecutor
    from .common import SPLITS
    scfg = {**cfg, "size": size}
    manifest_path = inside(f"data/improve/processed_{size}.json")
    audit_manifest = read_json("data/manifests/audit.json")
    key = preprocessing_key(scfg, audit_manifest)
    if manifest_path.exists() and read_json(manifest_path).get("identity") == key and read_json(manifest_path).get("complete"):
        log(f"{size}-px cache already complete; reuse")
        return read_json(manifest_path)
    base = read_json("data/manifests/processed.json")
    directory = inside(f"data/improve/processed/{key[:16]}")
    directory.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    records = []
    for split in SPLITS:
        split_records = [r for r in audit_manifest["records"] if r["split"] == split]
        rgb = np.lib.format.open_memmap(directory / f"{split}_rgb.npy", mode="w+", dtype=np.uint8, shape=(len(split_records), size, size, 3))
        mask = np.lib.format.open_memmap(directory / f"{split}_mask.npy", mode="w+", dtype=np.uint8, shape=(len(split_records), size, size))
        with ThreadPoolExecutor(max_workers=cfg["preprocess_workers"]) as pool:
            for start in range(0, len(split_records), 32):
                batch = list(enumerate(split_records))[start:start + 32]
                for (index, record), (image, labels, geometry) in zip(batch, pool.map(lambda pair: prepare_record(pair[1], size), batch)):
                    rgb[index], mask[index] = image, labels
                    records.append({**record, "index": index, "geometry": geometry,
                                    "processed_counts": np.bincount(labels[labels != 255], minlength=3).astype(int).tolist()})
        rgb.flush(); mask.flush()
        del rgb, mask
        log(f"{size}-px cache: {split} done")
    budgets = {}
    for budget, item in base["budgets"].items():
        entry = {"budget": item["budget"], "seed": item["seed"], "ids": item["ids"], "processed_identity": key}
        entry["identity"] = digest(entry)
        budgets[budget] = entry
    manifest = {"schema_version": 1, "identity": key, "audit_identity": audit_manifest["identity"], "complete": True,
                "directory": relative(directory), "size": size, "records": records, "budgets": budgets, "created_at": now(),
                "elapsed_seconds": time.perf_counter() - started, "same_ids_as": base["identity"]}
    write_json(manifest_path, manifest)
    return manifest


# ---------- runs that train nothing (TTA, hybrids) ----------

def record_inference_run(cfg, icfg, variant, method, budget, seed, components=None):
    """U-Net TTA run (frozen baseline checkpoint) or hybrid run (variant U-Net + frozen CRF, scaled to the size)."""
    from .training import run_path, source_identity, versions
    directory = run_dir(method, budget, seed, variant)
    inference = inference_of(variant, icfg, components)
    if method == "unet":   # tta: the frozen baseline U-Net, predicted with test-time augmentation
        base_directory = run_path("unet", budget, seed)
    else:
        base_directory = run_path("unet", budget, seed) if variant == "tta" else run_dir("unet", budget, seed, variant)
    base = read_json(base_directory / "run.json")
    while "base_run" in base:   # a TTA-only record: refine the checkpoint it points to
        base_directory = inside(base["base_run"]["directory"])
        base = read_json(base_directory / "run.json")
    if base["status"] != "complete" or sha256(base_directory / base["checkpoint"]) != base["checkpoint_sha256"]:
        raise ProjectError(f"{relative(base_directory)} is not a completed, intact U-Net run")
    crf = scaled_crf(read_json("runs/hybrid_selection.json")["selected"], base["config"]["size"]) if method == "hybrid" else None
    identity = {"method": method, "budget": budget, "seed": seed, "variant": study_path(variant), "components": components,
                "improve_config_sha256": sha256(CONFIG), "base_identity": base["identity"], "base_checkpoint_sha256": base["checkpoint_sha256"],
                "inference": inference, "crf": crf, "processed_identity": base["identity"]["processed_identity"],
                "budget_identity": base["identity"]["budget_identity"], "source_identity": source_identity()}
    meta = directory / "run.json"
    if meta.exists():
        old = read_json(meta)
        strip = lambda value: {k: v for k, v in value.items() if k != "source_identity"}
        if old["status"] == "complete" and strip(old["identity"]) == strip(identity):
            return directory
        raise ProjectError(f"{relative(directory)} exists with a different base run")
    directory.mkdir(parents=True, exist_ok=True)
    write_json(meta, {"schema_version": 1, "run_id": directory.name, "run_type": "final", "status": "complete", "identity": identity,
                      "base_run": {"run_id": base["run_id"], "checkpoint": base["checkpoint"], "directory": relative(base_directory)},
                      "checkpoint_sha256": base["checkpoint_sha256"], "config": base["config"], "crf": crf, "inference": inference,
                      "precision": base["precision"], "gpu": base.get("gpu"), "versions": versions(), "training_seconds": None,
                      "created_at": now(), "completed_at": now()})
    return directory


# ---------- engines: every variant predicts on the frozen 256-px grid ----------

class Engine:
    """Predicts 3-class labels on the 256-px grid for one run, whatever its method, size or inference."""

    def __init__(self, directory, run):
        from .evaluation import RFEngine, checkpoint_file
        self.run, self.method = run, run["identity"]["method"]
        self.cfg = run["config"]
        self.size = self.cfg["size"]
        self.store = Store(self.cfg) if self.size != 256 else None
        self.tta = bool(run.get("inference", {}).get("tta"))
        if self.method == "rf":
            self.rf = RFEngine(directory, run)
            self.device = "CPU"
        else:
            import torch
            from .neural import build_model, require_cuda
            require_cuda()
            self.torch = torch
            self.dtype = torch.bfloat16 if "bfloat16" in run["precision"] else torch.float16
            self.model = build_model(self.cfg, pretrained=False).cuda().eval()
            saved = torch.load(checkpoint_file(directory, run), map_location="cpu", weights_only=False)
            self.model.load_state_dict(saved["model"])
            self.device = "CUDA" + (" + CPU CRF" if self.method == "hybrid" else "")

    def model_input(self, record):
        if self.store is None:
            return None, record
        model_record = self.store.by_id[record["id"]]
        return self.store.get(model_record)[0], model_record

    def probabilities(self, rgb):
        from .neural import tensor_image, three_class
        torch = self.torch
        x = tensor_image(rgb)[None].cuda()

        def softmax(batch):
            with torch.autocast("cuda", dtype=self.dtype):
                return torch.softmax(three_class(self.model(batch), self.cfg).float(), 1)
        with torch.inference_mode():
            if self.tta:
                from .neural import dihedral_mean
                return dihedral_mean(softmax, x)[0].cpu().numpy()
            return softmax(x)[0].cpu().numpy()

    def predict(self, rgb256, record):
        rgb_model, model_record = self.model_input(record)
        rgb = rgb256 if rgb_model is None else rgb_model
        g_model, g256 = model_record["geometry"], record["geometry"]
        if self.method == "rf":
            probs = one_hot(self.rf.predict(rgb, g_model))
        else:
            probs = self.probabilities(rgb)
            if self.method == "hybrid":
                from .hybrid import refine
                probs = one_hot(refine(probs, rgb, g_model, self.run["crf"]))
        if self.size != 256:
            probs = to_256(probs, g_model, g256)
        return probs.argmax(0).astype(np.uint8)

    def synchronize(self):
        if self.method != "rf":
            self.torch.cuda.synchronize()


def evaluate_improve(cfg, directory, split="val", final=False):
    directory = inside(directory)
    run = read_json(directory / "run.json")
    if run["status"] != "complete":
        raise ProjectError("Evaluate a completed run")
    store = Store(cfg)
    frozen = None
    if split == "test":
        if not final:
            raise ProjectError("Held-out test requires --final")
        frozen = freeze_improve(cfg, store)
        if relative(directory) not in [r["run_id"] for r in frozen["runs"]]:
            raise ProjectError("Run is outside the frozen improvement study")
    output = directory / f"evaluation_{split}.json"
    records = store.select(split)
    if output.exists():
        cached = read_json(output)
        if (cached.get("complete") and cached["run_identity"] == run["identity"] and cached["image_count"] == len(records)
                and cached["experiment_identity"] == (frozen["identity"] if frozen else None)
                and sha256(cached["predictions"]) == cached["predictions_sha256"]):
            return cached
        raise ProjectError(f"{relative(output)} exists with different artifacts; preserve and investigate")
    engine = Engine(directory, run)
    predictions_path = directory / f"predictions_{split}.npy"
    predictions = np.lib.format.open_memmap(predictions_path, mode="w+", dtype=np.uint8, shape=(len(records), 256, 256))
    total, per_image = np.zeros((3, 3), dtype=np.int64), []
    started = time.perf_counter()
    for i, record in enumerate(records):
        rgb, truth = store.get(record)
        predicted = engine.predict(rgb, record)
        predicted[truth == 255] = 0
        predictions[i] = predicted
        matrix = confusion(truth, predicted)
        total += matrix
        per_image.append({"id": record["id"], "prediction_index": i, "metrics": score(matrix)})
    predictions.flush()
    del predictions
    seconds = time.perf_counter() - started
    timings = []
    for record in records[:cfg["timing"]["images"]]:
        rgb, _ = store.get(record)
        engine.synchronize(); t = time.perf_counter()
        engine.predict(rgb, record)
        engine.synchronize(); timings.append(time.perf_counter() - t)
    result = {"schema_version": 1, "complete": True, "split": split, "created_at": now(), "run_id": run["run_id"], "run_type": "final",
              "run_identity": run["identity"], "checkpoint_sha256": run["checkpoint_sha256"],
              "experiment_identity": frozen["identity"] if frozen else None, "image_count": len(records), "metrics": score(total),
              "per_image": per_image, "predictions": relative(predictions_path), "predictions_sha256": sha256(predictions_path),
              "evaluation_seconds": seconds,
              "timing": {"device": engine.device, "median_seconds": float(np.median(timings)), "p95_seconds": float(np.percentile(timings, 95)),
                         "boundary": "Model-size RGB in RAM -> 3-class labels on the 256-px grid (incl. TTA, CRF and 512->256 averaging)"}}
    write_json(output, result)
    log(f"{relative(directory)} {split}: flooded IoU={result['metrics']['flooded_miou']:.4f}")
    return result


# ---------- combine rule (validation only) ----------

def validation_difference(variant, method, budget, seed, components=None):
    from .training import run_path
    mine = read_json(run_dir(method, budget, seed, variant) / "evaluation_val.json")["metrics"]["flooded_miou"]
    base = read_json(run_path(method, budget, seed) / "evaluation_val.json")["metrics"]["flooded_miou"]
    return mine - base


def decide_combination(icfg=None):
    """Apply the pre-registered combine rule to the component runs' validation results; write runs/improve/combine.json."""
    icfg = icfg or load_config()
    path = inside("runs/improve/combine.json")
    if path.exists():
        return read_json(path)
    evidence, chosen = {}, {"unet": [], "rf": []}
    for method, components in (("unet", UNET_COMPONENTS), ("rf", RF_COMPONENTS)):
        for component in components:
            differences = {b: [validation_difference(component, method, b, s) for s in icfg["train_seeds"]] for b in icfg["budgets"]}
            passed = passes_combine_rule(differences)
            evidence[f"{component}/{method}"] = {"differences": {str(b): v for b, v in differences.items()}, "passes": passed}
            if passed:
                chosen[method].append(component)
    result = {"rule": icfg["combine_rule"], "evidence": evidence, "components": chosen, "decided_at": now(),
              "based_on": "validation only (test split sealed until the study freeze)"}
    write_json(path, result)
    log(f"Combined model components: U-Net {chosen['unet'] or 'none'}; RF {chosen['rf'] or 'none'}")
    return result


def combined_components(method):
    components = read_json("runs/improve/combine.json")["components"]
    return components["unet" if method in ("unet", "hybrid") else "rf"]


# ---------- train / freeze / test ----------

def train_improve(cfg, variant, method, budget, seed):
    from .training import train_rf, train_unet
    check_preregistration()
    icfg = load_config()
    components = combined_components(method) if variant == "combined" else None
    if variant == "combined" and not components:
        raise ProjectError(f"No component passed the combine rule for {method}; the combined model is not trained")
    if (method == "unet" and variant == "tta") or method == "hybrid" or (variant == "combined" and method == "unet" and components == ["tta"]):
        directory = record_inference_run(cfg, icfg, variant, method, budget, seed, components)
    else:
        vcfg = variant_cfg(cfg, variant, icfg, [c for c in components if c != "tta"] if components else None)
        trainer = train_rf if method == "rf" else train_unet
        directory = trainer(vcfg, budget, seed, resume=True, variant=ImproveVariant(variant, components))
        run = read_json(directory / "run.json")
        inference = inference_of(variant, icfg, components)
        if inference and run.get("inference") != inference:
            run["inference"] = inference
            write_json(directory / "run.json", run)
    evaluate_improve(cfg, relative(directory), "val")
    return directory


def freeze_improve(cfg, store, planned=None):
    icfg = load_config() if inside(CONFIG).exists() else None
    planned = planned if planned is not None else plan(icfg)
    runs = []
    for variant, method, budget, seed in planned:
        directory = run_dir(method, budget, seed, variant)
        val = directory / "evaluation_val.json"
        if not (directory / "run.json").exists() or not val.exists():
            raise ProjectError(f"All improvement runs must be trained and validated before test; missing {relative(directory)}")
        run = read_json(directory / "run.json")
        validation = read_json(val)
        if validation["run_identity"] != run["identity"]:
            raise ProjectError(f"{relative(directory)}: stale validation result")
        runs.append({"run_id": relative(directory), "identity": run["identity"], "checkpoint_sha256": run["checkpoint_sha256"],
                     "validation_sha256": sha256(val)})
    frozen = {"schema_version": 1, "preregistration": check_preregistration(), "combine": read_json("runs/improve/combine.json"),
              "base_experiment_identity": read_json("runs/final_experiment.json")["identity"], "runs": runs}
    frozen["identity"] = digest(frozen)
    path = inside("runs/improve/experiment.json")
    if path.exists():
        old = read_json(path)
        if old["identity"] != frozen["identity"]:
            raise ProjectError("The improvement study is already frozen with different artifacts")
        return old
    frozen["frozen_at"] = now()
    write_json(path, frozen)
    log(f"Improvement study frozen: {len(runs)} runs")
    return frozen


def stage_test(cfg):
    for variant, method, budget, seed in plan():
        evaluate_improve(cfg, relative(run_dir(method, budget, seed, variant)), "test", final=True)


# ---------- report ----------

ORIGINAL = ["background", "building flooded", "building non-flooded", "road flooded", "road non-flooded", "water", "tree", "vehicle", "pool", "grass"]


def error_breakdown(directory, store, layouts):
    """Where a run's false alarms land (original classes) and what its misses become, on the test split."""
    result = read_json(directory / "evaluation_test.json")
    predictions = np.load(inside(result["predictions"]), mmap_mode="r")
    position = {i["id"]: i["prediction_index"] for i in result["per_image"]}
    fp = {1: np.zeros(10, np.int64), 2: np.zeros(10, np.int64)}
    for record in store.select("test"):
        _, truth = store.get(record)
        p = np.asarray(predictions[position[record["id"]]])
        valid = truth != 255
        for c in (1, 2):
            fp[c] += np.bincount(np.minimum(layouts[record["id"]][(p == c) & (truth != c) & valid], 9), minlength=10)
    return {name: {ORIGINAL[k]: float(v / max(fp[c].sum(), 1)) for k, v in enumerate(fp[c]) if v} for c, name in ((1, "building"), (2, "road"))}


def report(cfg):
    from .gan_common import masks10_for
    from .imbalance import compare
    from .training import run_path
    icfg, store = load_config(), Store(cfg)
    combos = sorted({(v, m, b) for v, m, b, _ in plan(icfg)}, key=lambda k: (list(icfg["variants"]).index(k[0]), k[1], k[2]))
    comparisons = [compare(study_path(v), m, b, icfg) | {"variant": v} for v, m, b in combos]
    layouts = {r["id"]: masks10_for(r, 256) for r in store.select("test")}
    errors = {"baseline": {f"{m}_{b}": error_breakdown(run_path(m, b, icfg["seed"]), store, layouts) for m in ("rf", "unet", "hybrid") for b in icfg["budgets"]}}
    for v, m, b in combos:
        errors.setdefault(v, {})[f"{m}_{b}"] = error_breakdown(run_dir(m, b, icfg["seed"], v), store, layouts)
    out = {"schema_version": 1, "created_at": now(), "experiment": read_json("runs/improve/experiment.json")["identity"],
           "combine": read_json("runs/improve/combine.json"), "comparisons": comparisons, "errors_seed42": errors}
    write_json("output/improve/results.json", out)
    return out


# ---------- stages ----------

def fetch_encoder(name="resnet34"):
    """One-off download of ImageNet encoder weights into the project cache (training itself stays offline)."""
    import os
    os.environ.pop("HF_HUB_OFFLINE", None)
    import segmentation_models_pytorch as smp
    smp.encoders.get_encoder(name, weights="imagenet")
    log(f"{name} ImageNet weights cached")


def stage_features(cfg, variant):
    from .features import build_features
    check_preregistration()
    return build_features(variant_cfg(cfg, variant, load_config(), combined_components("rf") if variant == "combined" else None))


# ---------- web export ----------

def validation_preview(icfg):
    """Seed-matched validation changes for every finished variant run (a preview; the decision uses the test split)."""
    from .training import run_path

    def summary(path):
        metrics = read_json(path)["metrics"]
        return metrics["flooded_miou"], [c["iou"] for c in metrics["per_class"][1:]]

    out = []
    for variant, method, budget in dict.fromkeys((v, m, b) for v, m, b, _ in plan(icfg)):
        row = {"variant": variant, "method": method, "budget": budget, "seeds": [], "scores": [], "differences": [], "class_iou_differences": []}
        for seed in icfg["train_seeds"]:
            mine = run_dir(method, budget, seed, variant) / "evaluation_val.json"
            if mine.exists():
                (score_v, classes_v), (score_b, classes_b) = summary(mine), summary(run_path(method, budget, seed) / "evaluation_val.json")
                row["seeds"].append(seed)
                row["scores"].append(score_v)
                row["differences"].append(score_v - score_b)
                row["class_iou_differences"].append([a - b for a, b in zip(classes_v, classes_b, strict=True)])
        out.append(row | {"complete": len(row["seeds"]) == len(icfg["train_seeds"])})
    return out


def export_improve(cfg):
    """Web export of the improvement study. Works mid-study: anything not produced yet is exported as missing (the page
    shows it as pending) and no value is filled in. After the freeze and test evaluation the same call writes the final page."""
    from .gan_common import masks10_for
    from .training import run_path
    icfg, store = load_config(), Store(cfg)
    results_path, frozen_path = inside("output/improve/results.json"), inside("runs/improve/experiment.json")
    final = results_path.exists() and frozen_path.exists()
    results = read_json(results_path) if final else {"comparisons": [], "errors_seed42": {}}
    frozen = read_json(frozen_path) if frozen_path.exists() else None
    combine_path = inside("runs/improve/combine.json")
    errors = results["errors_seed42"]
    if not final:   # the baselines' error analysis comes from their own (already scored) frozen test predictions
        layouts = {r["id"]: masks10_for(r, 256) for r in store.select("test")}
        errors = {"baseline": {f"{m}_{b}": error_breakdown(run_path(m, b, icfg["seed"]), store, layouts)
                               for m in ("rf", "unet", "hybrid") for b in icfg["budgets"]}}
    baseline = {}
    for method in ("rf", "unet", "hybrid"):
        for budget in icfg["budgets"]:
            tests = [read_json(run_path(method, budget, s) / "evaluation_test.json")["metrics"] for s in icfg["train_seeds"]]
            baseline[f"{method}_{budget}"] = {"flooded_miou": [t["flooded_miou"] for t in tests],
                                              "class_iou": [[c["iou"] for c in t["per_class"][1:]] for t in tests],
                                              "class_recall": [[c["recall"] for c in t["per_class"][1:]] for t in tests]}
    costs, validated, planned = {}, 0, plan(icfg)
    for variant, method, budget, seed in planned:
        directory = run_dir(method, budget, seed, variant)
        if (directory / "evaluation_val.json").exists():
            validated += 1
        if (directory / "run.json").exists():
            run = read_json(directory / "run.json")
            if run.get("status") == "complete" and run.get("training_seconds") is not None:
                costs.setdefault(f"{variant}_{method}_{budget}", []).append(run["training_seconds"])
    for method in ("rf", "unet"):
        for budget in icfg["budgets"]:
            costs[f"baseline_{method}_{budget}"] = [read_json(run_path(method, budget, s) / "run.json")["training_seconds"] for s in icfg["train_seeds"]]
    preregistration = inside("runs/improve/preregistration.json")
    first_test = min(read_json(inside(r["run_id"]) / "evaluation_test.json")["created_at"] for r in frozen["runs"]) if final else None
    out = {"schema_version": 1, "status": "complete" if final else "in progress", "created_at": now(),
           "experiment_identity": frozen["identity"] if frozen else None,
           "preregistered_at": read_json(preregistration)["registered_at"] if preregistration.exists() else None,
           "frozen_at": frozen["frozen_at"] if frozen else None, "first_test_evaluation": first_test,
           "progress": {"runs_planned": len(planned), "runs_validated": validated},
           "config": {k: icfg[k] for k in ("study", "combine_rule", "hybrid", "scoring", "slic_at_512", "unchanged", "decision_rule")},
           "variants": dict(icfg["variants"]), "budgets": icfg["budgets"], "seeds": icfg["train_seeds"],
           "combine": read_json(combine_path) if combine_path.exists() else None, "baseline_test": baseline, "errors_seed42": errors,
           "comparisons": results["comparisons"], "validation_preview": validation_preview(icfg), "costs": costs,
           "runs": len(frozen["runs"]) if frozen else None}
    if "C:" + "\\" in str(out):
        raise ProjectError("Absolute paths leaked into the improvement export")
    write_json(inside("web/data/improve.json"), out)
    log(f"Improvement export ({out['status']}): {validated}/{len(planned)} runs validated, {len(results['comparisons'])} test comparisons")
    return out


# ---------- app "after" bundle: the pre-registered combined models, shaped like results.json ----------

AFTER_METHODS = ("rf", "unet", "hybrid")


def after_run_id(method, budget, seed):
    """Improved runs reuse the baseline run ids inside their folders; the app needs them namespaced."""
    return f"improve/combined/{method}_{budget}_s{seed}"



def after_methodology(before, after, crf_before, crf_after):
    """Before/after rows of each pipeline, read from the frozen run configurations (rows with equal values are kept and
    marked unchanged, so the page shows what stayed the same as well)."""
    def row(aspect, old, new):
        return {"aspect": aspect, "before": old, "after": new, "changed": old != new}

    def px(cfg):
        return f"{cfg['size']}×{cfg['size']} px"

    rf_b, rf_a, un_b, un_a = before["rf"], after["rf"], before["unet"], after["unet"]
    slic = lambda cfg: f"SLIC, {cfg['slic']['n_segments']} superpixels per image (about {cfg['size'] ** 2 // cfg['slic']['n_segments']:,} px each)"
    features = lambda cfg: ("50 per region: the original 22, the mean of those 22 over its touching neighbours, its centroid (y, x), its area share "
                            f"and the mean colour of the coarse SLIC region ({cfg['rf']['context']['coarse_segments']} segments) it lies in"
                            if cfg["rf"].get("context") else "22 per region: RGB and HSV mean and standard deviation, 10-bin LBP texture histogram")
    labels = lambda cfg: ("10 original FloodNet classes (region majority); predictions mapped back to the 3 classes"
                          if cfg["rf"].get("aux10") else "3 classes (region majority): Other, flooded building, flooded road")
    forest = lambda cfg: f"{cfg['rf']['n_estimators']} trees, depth {cfg['rf']['max_depth']}, {cfg['rf']['class_weight']} class weights"
    cap = lambda cfg: f"{cfg['rf']['max_regions']:,} regions, every flooded region kept"
    scoring = lambda cfg: "on the 256 px test grid" if cfg["size"] == 256 else f"{cfg['size']} px prediction area-averaged onto the same 256 px test grid"
    rate = lambda v: f"{v:.0e}".replace("e-0", "e-")
    schedule = lambda u: (f"cosine decay {rate(u['lr'])} → {rate(u['lr_min'])}" if u.get("schedule") == "cosine" else f"constant {rate(u['lr'])}")
    stopping = lambda u: f"up to {u['max_epochs']} epochs, stop after {u['patience']} without improvement"
    kept = lambda u: (f"running average of the weights (EMA, decay {u['ema']}), best validation epoch" if u.get("ema")
                      else "the trained weights at the best validation epoch")
    inference = lambda run: ("mean of 8 predictions: the image flipped and rotated, each mapped back (test-time augmentation)"
                             if run.get("tta") else "one forward pass")
    encoder = lambda u: f"{u['encoder'].replace('resnet', 'ResNet-')}, {'ImageNet' if u['encoder_weights'] == 'imagenet' else u['encoder_weights']} weights"
    head = lambda u: ("3 class scores + 10 auxiliary class scores" if u.get("aux10_weight") else "3 class scores")
    kernel = lambda c, k: (f"sxy {c[k]['sxy']:g}, weight {c[k]['compat']}" if k == "gaussian"
                           else f"sxy {c[k]['sxy']:g}, srgb {c[k]['srgb']}, weight {c[k]['compat']}")
    return {
        "rf": [row("Input resolution", px(rf_b), px(rf_a)), row("Superpixels", slic(rf_b), slic(rf_a)),
               row("Region features", features(rf_b), features(rf_a)), row("Training labels", labels(rf_b), labels(rf_a)),
               row("Forest", forest(rf_b), forest(rf_a)), row("Training cap", cap(rf_b), cap(rf_a)), row("Scoring", scoring(rf_b), scoring(rf_a))],
        "unet": [row("Input resolution", px(un_b), px(un_a)),
                 row("Encoder", encoder(un_b['unet']), encoder(un_a['unet'])),
                 row("Output head", head(un_b["unet"]), head(un_a["unet"])),
                 row("Loss", "class-weighted cross-entropy", "class-weighted cross-entropy"),
                 row("Learning rate", schedule(un_b["unet"]), schedule(un_a["unet"])), row("Training length", stopping(un_b["unet"]), stopping(un_a["unet"])),
                 row("Weights kept", kept(un_b["unet"]), kept(un_a["unet"])),
                 row("Batch, optimiser, augmentation", f"batch {un_b['unet']['batch_size']}, AdamW, flips + rotations", f"batch {un_a['unet']['batch_size']}, AdamW, flips + rotations"),
                 row("Prediction", inference(before["unet_inference"]), inference(after["unet_inference"])), row("Scoring", scoring(un_b), scoring(un_a))],
        "hybrid": [row("Base U-Net", "the original U-Net of the same budget and seed", "the improved U-Net of the same budget and seed"),
                   row("Spatial kernel", kernel(crf_before, "gaussian"), kernel(crf_after, "gaussian")),
                   row("Appearance kernel", kernel(crf_before, "bilateral"), kernel(crf_after, "bilateral")),
                   row("Iterations", str(crf_before["iterations"]), str(crf_after["iterations"])),
                   row("Setting chosen", "validation grid (6 settings)", "the same setting, pixel widths doubled for the 512 px input (no new tuning)")]}


def export_after(cfg):
    """web/data/after.json + per-image prediction PNGs of the 18 combined runs (seed-42 runs for the explorer).

    Same field shapes as results.json (runs, summary, budget_effects, crf_effects, samples) so the app can switch its
    data source between the frozen originals ("before") and the combined models ("after"). Reads only frozen,
    hash-checked study artifacts; never writes outside web/."""
    import shutil

    from PIL import Image

    from .data import crop
    from .evaluation import subset_metrics
    from .export import STRATA_FIELDS, check_url, crf_effects, flat_metrics, slim_history, summarise, target_strata
    icfg, store = load_config(), Store(cfg)
    frozen = read_json("runs/improve/experiment.json")
    results_path = inside("output/improve/results.json")
    combine = read_json("runs/improve/combine.json")
    overlap = read_json("data/manifests/overlap.json")
    if overlap["processed_identity"] != store.manifest["identity"]:
        raise ProjectError("Overlap analysis belongs to a different preprocessing manifest")
    clean_ids = overlap["summary"]["test"]["clean_ids"]
    strata = target_strata(store, overlap)
    seeds, primary = icfg["train_seeds"], icfg["seed"]
    web = inside("web")
    identity = digest({"experiment": frozen["identity"], "results": sha256(results_path)})[:16]
    images = web / f"data/after/{identity}/images"
    if images.parent.exists():
        shutil.rmtree(images.parent)
    images.mkdir(parents=True)
    runs, explorer = [], []
    for method in AFTER_METHODS:
        if not combine["components"]["rf" if method == "rf" else "unet"]:
            raise ProjectError(f"No combined {method} model was trained")
        for budget in icfg["budgets"]:
            for seed in seeds:
                directory = run_dir(method, budget, seed, "combined")
                result, run = read_json(directory / "evaluation_test.json"), read_json(directory / "run.json")
                if result["experiment_identity"] != frozen["identity"] or result["run_identity"] != run["identity"]:
                    raise ProjectError(f"{relative(directory)}: test result is not from the frozen study")
                if sha256(result["predictions"]) != result["predictions_sha256"]:
                    raise ProjectError(f"{relative(directory)}: predictions changed after evaluation")
                history = directory / "history.json"
                public = {"id": after_run_id(method, budget, seed), "method": method, "budget": budget, "seed": seed,
                          "status": run["status"], "metrics": result["metrics"],
                          "clean_metrics": subset_metrics(result["per_image"], clean_ids), "clean_images": len(clean_ids),
                          "overlap_strata": {stratum: {k: flat_metrics(subset_metrics(result["per_image"], strata[stratum]))[k] for k in STRATA_FIELDS}
                                             for stratum in ("low", "high")},
                          "training_seconds": run.get("training_seconds"),
                          "timing": {k: result["timing"][k] for k in ("device", "median_seconds", "p95_seconds", "boundary")},
                          "peak_gpu_allocated_bytes": run.get("peak_gpu_allocated_bytes"), "device": run.get("gpu", "CPU"),
                          "precision": run.get("precision"), "checkpoint_sha256": run["checkpoint_sha256"],
                          "epochs": run.get("epochs"), "best_epoch": run.get("best_epoch"),
                          "size": run["config"]["size"], "inference": run.get("inference") or {},
                          "history": slim_history(read_json(history)) if history.exists() else []}
                runs.append(public)
                if seed == primary:
                    explorer.append((public, result))
    samples = {}
    for public, result in explorer:
        predictions = np.load(inside(result["predictions"]), mmap_mode="r")
        rows = {item["id"]: item for item in result["per_image"]}
        for record in store.select("test"):
            item = rows[record["id"]]
            path = images / f"{record['id']}_{public['method']}_{public['budget']}.png"
            Image.fromarray(crop(np.asarray(predictions[item["prediction_index"]]), record["geometry"])).save(path, optimize=True)
            sample = samples.setdefault(record["id"], {"predictions": {}, "metrics": {}})
            sample["predictions"][public["id"]] = path.relative_to(web).as_posix()
            sample["metrics"][public["id"]] = {k: flat_metrics(item["metrics"])[k] for k in ("flooded_miou", "building_iou", "road_iou")}
    summary, budget_effects = summarise(runs, seeds)
    report_json = read_json(results_path)
    comparisons = [c for c in report_json["comparisons"] if c["variant"] == "combined"]
    from .training import run_path
    config_of = lambda directory: read_json(directory / "run.json")["config"]
    before_cfg = {"rf": config_of(run_path("rf", 25, primary)), "unet": config_of(run_path("unet", 25, primary)), "unet_inference": {}}
    after_cfg = {"rf": config_of(run_dir("rf", 25, primary, "combined")), "unet": config_of(run_dir("unet", 25, primary, "combined")),
                 "unet_inference": read_json(run_dir("unet", 25, primary, "combined") / "run.json").get("inference") or {}}
    methodology = after_methodology(before_cfg, after_cfg, read_json("runs/hybrid_selection.json")["selected"],
                                    read_json(run_dir("hybrid", 25, primary, "combined") / "run.json")["crf"])
    single = [{k: c[k] for k in ("variant", "method", "budget", "mean_difference", "ci", "decision")}
              for c in report_json["comparisons"] if c["variant"] != "combined"]
    out = {"schema_version": 1, "status": "complete", "created_at": now(), "identity": identity,
           "experiment_identity": frozen["identity"], "frozen_at": frozen["frozen_at"], "components": combine["components"],
           "label": "Pre-registered combined model", "explorer_seed": primary, "explorer_runs": [p["id"] for p, _ in explorer],
           "runs": runs, "summary": summary, "budget_effects": budget_effects, "crf_effects": crf_effects(runs, seeds),
           "comparisons": comparisons, "samples": samples, "methodology": methodology, "single_changes": single,
           "errors_seed42": {k: report_json["errors_seed42"][k] for k in ("baseline", "combined")},
           "combine": {"rule": combine["rule"], "evidence": combine["evidence"], "decided_at": combine["decided_at"]},
           "study": {"runs": len(plan(icfg)), "component_runs": len([r for r in plan(icfg) if r[0] != "combined"]),
                     "changes_tested": len(icfg["variants"]) - 1, "preregistered_at": read_json("runs/improve/preregistration.json")["registered_at"]}}
    for sample in samples.values():
        for url in sample["predictions"].values():
            check_url(url, web)
    if "C:" + "\\" in str(out):
        raise ProjectError("Absolute paths leaked into the after export")
    write_json(web / "data/after.json", out)
    for old in (web / "data/after").iterdir():
        if old.is_dir() and old.name != identity:
            shutil.rmtree(old)
    log(f"After export: {len(runs)} combined runs, {len(samples)} test samples x {len(explorer)} explorer runs")
    return out
