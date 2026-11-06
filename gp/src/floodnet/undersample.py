"""Uniform Before pipeline, undersampling pipeline and blackening test.

Pre-registered in configs/undersample.json and docs/undersample.md. Uniform: every model uses the U-Net's capped class
weights and all training data (the RF is retrained; the U-Net and U-Net + CRF already match). Undersampling: every
model keeps all target pixels/regions and r 'Other' ones per target one, without class weights; r is chosen on
validation. Blackening: as undersampling, trained on photos whose 'Other' pixels are black, scored on real photos.
Runs live in runs/undersample/<uniform|r1|r3|r10|blacken>/; the study is frozen before its single test evaluation.
"""
from __future__ import annotations

import copy

import numpy as np

from .common import ProjectError, digest, inside, log, now, read_json, relative, sha256, write_json
from .data import Store

CONFIG = "configs/undersample.json"
ROOT = "runs/undersample"
FAMILY = {"rf": "rf", "unet": "unet", "hybrid": "unet"}


def load_config():
    return read_json(CONFIG)


def study_path(kind):
    return f"undersample/{kind}"


def ratio_kind(ratio):
    return f"r{ratio}"


def run_dir(method, budget, seed, kind):
    from .training import run_path
    return run_path(method, budget, seed, variant=study_path(kind))


def baseline_dir(method, budget, seed):
    """The uniform Before of a configuration: the uniform RF, or the frozen original U-Net / U-Net + CRF."""
    from .training import run_path
    return run_dir("rf", budget, seed, "uniform") if method == "rf" else run_path(method, budget, seed)


def plan(ucfg=None):
    ucfg = ucfg or load_config()
    budgets, seeds = ucfg["budgets"], ucfg["train_seeds"]
    runs = [("uniform", "rf", b, s) for b in budgets for s in seeds]
    for ratio in ucfg["undersample"]["ratios"]:
        runs += [(ratio_kind(ratio), m, b, s) for m in ucfg["undersample"]["methods"] for b in budgets for s in seeds]
    runs += [("blacken", m, b, s) for m in ucfg["blacken"]["methods"] for b in budgets for s in seeds]
    return runs


def choose_ratio(scores):
    """{ratio: [validation flooded mIoU per seed]} -> the ratio with the best mean; ties go to the smaller ratio."""
    return min(scores, key=lambda r: (-float(np.mean(scores[r])), r))


def ratio_choice():
    path = inside(f"{ROOT}/ratio.json")
    if not path.exists():
        raise ProjectError("Choose the undersampling ratio on validation first (`floodnet undersample choose`)")
    return read_json(path)


def chosen_kind(method, budget):
    return ratio_kind(ratio_choice()["choice"][FAMILY[method]][str(budget)])


def tested(ucfg=None):
    """Runs scored on the test split: the uniform RF, the chosen ratio per budget and family, and the blackening test."""
    ucfg = ucfg or load_config()
    out = []
    for kind, method, budget, seed in plan(ucfg):
        if kind in ("uniform", "blacken") or kind == chosen_kind(method, budget):
            out.append((kind, method, budget, seed))
    return out


# ---------- pre-registration ----------

def preregister():
    path = inside(f"{ROOT}/preregistration.json")
    record = {"config_sha256": sha256(CONFIG), "protocol_sha256": sha256("docs/undersample.md"),
              "evals_sha256": sha256("audit/undersample_evals.md"), "runs_planned": len(plan())}
    if path.exists():
        old = read_json(path)
        if {k: old[k] for k in record} == record:
            return old
        if any(inside(ROOT).glob("*/*/run.json")):
            raise ProjectError("Pre-registered settings changed after runs started; preserve the study first")
    record["registered_at"] = now()
    write_json(path, record)
    return record


def check_preregistration():
    path = inside(f"{ROOT}/preregistration.json")
    if not path.exists():
        raise ProjectError("Run `floodnet undersample preregister` first")
    record = read_json(path)
    if record["config_sha256"] != sha256(CONFIG):
        raise ProjectError("configs/undersample.json changed after pre-registration")
    return record


# ---------- configurations ----------

def variant_cfg(cfg, kind, method, budget, ucfg=None):
    ucfg = ucfg or load_config()
    out = copy.deepcopy(cfg)
    if kind == "uniform":
        out["rf"].update(ucfg["uniform"]["rf"])
        return out
    ratio = int(kind[1:]) if kind.startswith("r") else ratio_choice()["choice"][FAMILY[method]][str(budget)]
    out["unet"].update(ucfg["undersample"]["unet"], undersample_ratio=ratio)
    out["rf"].update(ucfg["undersample"]["rf"], ratio=ratio)
    if kind == "blacken":
        out["unet"]["blacken"] = True
        out["rf"]["blacken"] = True
    return out


class StudyVariant:
    """What training.py needs from this study (identity keys; no extra data; plain store; plain loss)."""
    loss = "ce"

    def __init__(self, kind):
        self.kind, self.name = kind, study_path(kind)

    def identity(self):
        return {"variant": self.name, "undersample_config_sha256": sha256(CONFIG)}

    def unet_extra(self, budget):
        return []

    def wrap_store(self, store):
        return store

    def rf_extra(self, budget):
        return None


def record_hybrid(kind, budget, seed):
    """U-Net + CRF of a study U-Net: its checkpoint + the frozen, validation-selected CRF setting (not re-tuned)."""
    from .training import source_identity, versions
    selection = read_json("runs/hybrid_selection.json")
    base_directory = run_dir("unet", budget, seed, kind)
    base = read_json(base_directory / "run.json")
    if base["status"] != "complete" or sha256(base_directory / base["checkpoint"]) != base["checkpoint_sha256"]:
        raise ProjectError(f"{relative(base_directory)} is not a completed, intact U-Net run")
    directory = run_dir("hybrid", budget, seed, kind)
    ids = base["identity"]
    identity = {"method": "hybrid", "budget": budget, "seed": seed, "processed_identity": ids["processed_identity"],
                "audit_identity": ids["audit_identity"], "budget_identity": ids["budget_identity"],
                "config_hash": digest(selection["selected"]), "base_run_id": relative(base_directory), "base_identity": ids,
                "base_checkpoint_sha256": base["checkpoint_sha256"], "selection_identity": selection["identity"],
                "crf": selection["selected"], "backend": selection["backend"], "source_identity": source_identity(),
                "variant": study_path(kind), "undersample_config_sha256": sha256(CONFIG)}
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
                      "checkpoint_sha256": base["checkpoint_sha256"], "config": base["config"], "crf": selection["selected"],
                      "precision": base["precision"], "gpu": base.get("gpu"), "versions": versions(), "training_seconds": None,
                      "epochs": None, "best_epoch": None, "created_at": now(), "completed_at": now(),
                      "note": "No training: refines the study U-Net's probabilities with the frozen DenseCRF setting."})
    return directory


# ---------- stages ----------

def stage_features_blacken(cfg):
    from .features import build_features
    check_preregistration()
    blackened = variant_cfg(cfg, "r1", "rf", 25)
    blackened["rf"]["blacken"] = True
    return build_features(blackened)


def train(cfg, kind, method, budget, seed):
    from .evaluation import evaluate
    from .training import train_rf, train_unet
    check_preregistration()
    if method == "hybrid":
        directory = record_hybrid(kind, budget, seed)
    else:
        trainer = train_rf if method == "rf" else train_unet
        directory = trainer(variant_cfg(cfg, kind, method, budget), budget, seed, resume=True, variant=StudyVariant(kind))
    evaluate(cfg, relative(directory), "val")
    return directory


def decide_ratio(ucfg=None):
    """Apply the pre-registered ratio rule to the validation results; write runs/undersample/ratio.json."""
    ucfg = ucfg or load_config()
    path = inside(f"{ROOT}/ratio.json")
    if path.exists():
        return read_json(path)
    evidence, choice = {}, {"unet": {}, "rf": {}}
    for family in ("unet", "rf"):
        for budget in ucfg["budgets"]:
            scores = {r: [read_json(run_dir(family, budget, s, ratio_kind(r)) / "evaluation_val.json")["metrics"]["flooded_miou"]
                          for s in ucfg["train_seeds"]] for r in ucfg["undersample"]["ratios"]}
            choice[family][str(budget)] = choose_ratio(scores)
            evidence[f"{family}_{budget}"] = {str(r): v for r, v in scores.items()}
    result = {"rule": ucfg["ratio_rule"], "choice": choice, "validation_flooded_miou": evidence, "decided_at": now(),
              "based_on": "validation only (test split sealed until the study freeze)"}
    write_json(path, result)
    log(f"Undersampling ratio chosen on validation: {choice}")
    return result


def freeze_undersample(cfg, store, planned=None):
    from .evaluation import checkpoint_file
    ucfg = load_config()
    planned = planned if planned is not None else plan(ucfg)
    runs = []
    for kind, method, budget, seed in planned:
        directory = run_dir(method, budget, seed, kind)
        val = directory / "evaluation_val.json"
        if not (directory / "run.json").exists() or not val.exists():
            raise ProjectError(f"All undersampling-study runs must be trained and validated before test; missing {relative(directory)}")
        run = read_json(directory / "run.json")
        if run["status"] != "complete" or read_json(val)["run_identity"] != run["identity"]:
            raise ProjectError(f"{relative(directory)}: incomplete run or stale validation result")
        if sha256(checkpoint_file(directory, run)) != run["checkpoint_sha256"]:
            raise ProjectError(f"{relative(directory)}: checkpoint changed")
        runs.append({"run_id": relative(directory), "identity": run["identity"], "checkpoint_sha256": run["checkpoint_sha256"],
                     "validation_sha256": sha256(val)})
    frozen = {"schema_version": 1, "preregistration": check_preregistration(), "ratio": ratio_choice(),
              "tested": [relative(run_dir(m, b, s, k)) for k, m, b, s in tested(ucfg)],
              "base_experiment_identity": read_json("runs/final_experiment.json")["identity"],
              "hybrid_experiment_identity": read_json("runs/hybrid_experiment.json")["identity"], "runs": runs}
    frozen["identity"] = digest(frozen)
    path = inside(f"{ROOT}/experiment.json")
    if path.exists():
        old = read_json(path)
        if old["identity"] != frozen["identity"]:
            raise ProjectError("The undersampling study is already frozen with different artifacts")
        return old
    frozen["frozen_at"] = now()
    write_json(path, frozen)
    log(f"Undersampling study frozen: {len(runs)} runs, {len(frozen['tested'])} to be tested")
    return frozen


def stage_test(cfg):
    from .evaluation import evaluate
    for kind, method, budget, seed in tested():
        evaluate(cfg, relative(run_dir(method, budget, seed, kind)), "test", final=True)


# ---------- report ----------

def headline(matrix):
    """Minority (flooded) IoU, majority IoU, weighted mIoU, 3-class mIoU and accuracy of one pooled confusion matrix."""
    m = np.asarray(matrix, dtype=np.float64)
    tp, support, predicted = np.diag(m), m.sum(1), m.sum(0)
    iou = tp / np.maximum(support + predicted - tp, 1)
    return {"minority": float(iou[1:].mean()), "majority": float(iou[0]), "weighted": float((support / support.sum() * iou).sum()),
            "miou3": float(iou.mean()), "accuracy": float(tp.sum() / m.sum())}


def headlines(kind, method, budget, ucfg, baseline):
    """Per-seed headline metrics of both arms (descriptive; the decision uses only the minority IoU)."""
    arms = {"variant": lambda s: run_dir(method, budget, s, kind), "baseline": lambda s: baseline(method, budget, s)}
    out = {}
    for arm, directory in arms.items():
        rows = [headline(read_json(directory(s) / "evaluation_test.json")["metrics"]["confusion_matrix"]) for s in ucfg["train_seeds"]]
        out[arm] = {k: [r[k] for r in rows] for k in rows[0]}
    out["differences"] = {k: [v - b for v, b in zip(out["variant"][k], out["baseline"][k])] for k in out["variant"]}
    return out


def report(cfg):
    from .imbalance import compare
    from .training import run_path
    ucfg = load_config()
    comparisons, blacken, uniform = [], [], []
    for budget in ucfg["budgets"]:
        uniform.append(compare(study_path("uniform"), "rf", budget, ucfg) | {"kind": "uniform", "baseline": "first version",
                                                                           "headline": headlines("uniform", "rf", budget, ucfg, run_path)})
        for method in ucfg["undersample"]["methods"]:
            kind = chosen_kind(method, budget)
            comparisons.append(compare(study_path(kind), method, budget, ucfg, baseline=baseline_dir)
                               | {"kind": kind, "baseline": "uniform Before", "headline": headlines(kind, method, budget, ucfg, baseline_dir)})
            blacken.append(compare(study_path("blacken"), method, budget, ucfg, baseline=baseline_dir)
                           | {"kind": "blacken", "baseline": "uniform Before", "headline": headlines("blacken", method, budget, ucfg, baseline_dir)})
    out = {"schema_version": 1, "created_at": now(), "experiment": read_json(f"{ROOT}/experiment.json")["identity"],
           "ratio": ratio_choice(), "uniform_vs_first": uniform, "comparisons": comparisons, "blacken": blacken,
           "first_version_runs": [relative(run_path("rf", b, s)) for b in ucfg["budgets"] for s in ucfg["train_seeds"]]}
    write_json("output/undersample/results.json", out)
    return out


# ---------- app bundles: web/data/uniform.json and web/data/undersample.json (same shape as after.json) ----------

def public_run(directory, method, budget, seed, clean_ids, strata):
    from .evaluation import subset_metrics
    from .export import STRATA_FIELDS, flat_metrics, slim_history
    result, run = read_json(directory / "evaluation_test.json"), read_json(directory / "run.json")
    if result["run_identity"] != run["identity"] or sha256(result["predictions"]) != result["predictions_sha256"]:
        raise ProjectError(f"{relative(directory)}: stale or changed test result")
    history = directory / "history.json"
    # The original hybrids name their U-Net only by run id (runs/<id>); study hybrids store its directory.
    base_run = run.get("base_run")
    base = read_json(inside(base_run.get("directory") or f"runs/{base_run['run_id']}") / "run.json") if base_run else run
    return {"id": relative(directory).removeprefix("runs/"), "method": method, "budget": budget, "seed": seed, "status": run["status"],
            "metrics": result["metrics"], "clean_metrics": subset_metrics(result["per_image"], clean_ids), "clean_images": len(clean_ids),
            "overlap_strata": {stratum: {k: flat_metrics(subset_metrics(result["per_image"], strata[stratum]))[k] for k in STRATA_FIELDS}
                               for stratum in ("low", "high")},
            "training_seconds": run.get("training_seconds"),
            "timing": {k: result["timing"][k] for k in ("device", "median_seconds", "p95_seconds", "boundary")},
            "peak_gpu_allocated_bytes": run.get("peak_gpu_allocated_bytes"), "device": run.get("gpu", "CPU"),
            "precision": run.get("precision"), "checkpoint_sha256": run["checkpoint_sha256"],
            "epochs": run.get("epochs"), "best_epoch": run.get("best_epoch"), "size": run["config"]["size"],
            "region_count": run.get("region_count"), "region_class_counts": run.get("region_class_counts"),
            "class_weights": run.get("class_weights") or base.get("class_weights"),
            "other_keep_probability": base.get("other_keep_probability"),
            "val_flooded_miou": read_json(val)["metrics"]["flooded_miou"] if (val := directory / "evaluation_val.json").exists() else None,
            "history": slim_history(read_json(history)) if history.exists() else []}, result


def pipeline_bundle(cfg, name, directories, identity_parts, extra):
    """Write web/data/<name>.json + the seed-42 explorer masks of every configuration in `directories`."""
    import shutil

    from PIL import Image

    from .data import crop
    from .export import check_url, crf_effects, flat_metrics, summarise, target_strata
    ucfg, store = load_config(), Store(cfg)
    overlap = read_json("data/manifests/overlap.json")
    clean_ids, strata = overlap["summary"]["test"]["clean_ids"], target_strata(store, overlap)
    seeds, primary = ucfg["train_seeds"], ucfg["seed"]
    web = inside("web")
    identity = digest(identity_parts)[:16]
    images = web / f"data/{name}/{identity}/images"
    if images.parent.exists():
        shutil.rmtree(images.parent)
    images.mkdir(parents=True)
    runs, explorer = [], []
    for (method, budget, seed), directory in directories.items():
        public, result = public_run(directory, method, budget, seed, clean_ids, strata)
        runs.append(public)
        if seed == primary:
            explorer.append((public, result))
    # Frozen original runs already have their masks in the main export; link to those instead of copying them.
    exported = {s["id"]: s["predictions"] for s in read_json("web/data/results.json")["samples"]}
    samples = {}
    for public, result in explorer:
        predictions = np.load(inside(result["predictions"]), mmap_mode="r")
        rows = {item["id"]: item for item in result["per_image"]}
        for record in store.select("test"):
            item = rows[record["id"]]
            url = exported.get(record["id"], {}).get(public["id"])
            if url is None:
                path = images / f"{record['id']}_{public['method']}_{public['budget']}.png"
                Image.fromarray(crop(np.asarray(predictions[item["prediction_index"]]), record["geometry"])).save(path, optimize=True)
                url = path.relative_to(web).as_posix()
            sample = samples.setdefault(record["id"], {"predictions": {}, "metrics": {}})
            sample["predictions"][public["id"]] = url
            sample["metrics"][public["id"]] = {k: flat_metrics(item["metrics"])[k] for k in ("flooded_miou", "building_iou", "road_iou")}
    summary, budget_effects = summarise(runs, seeds)
    out = {"schema_version": 1, "status": "complete", "created_at": now(), "identity": identity, "explorer_seed": primary,
           "explorer_runs": [p["id"] for p, _ in explorer], "runs": runs, "summary": summary, "budget_effects": budget_effects,
           "crf_effects": crf_effects(runs, seeds), "samples": samples, **extra}
    for sample in samples.values():
        for url in sample["predictions"].values():
            check_url(url, web)
    if "C:" + "\\" in str(out):
        raise ProjectError(f"Absolute paths leaked into the {name} export")
    write_json(web / f"data/{name}.json", out)
    for old in (web / f"data/{name}").iterdir():
        if old.is_dir() and old.name != identity:
            shutil.rmtree(old)
    log(f"{name} export: {len(runs)} runs, {len(samples)} test samples x {len(explorer)} explorer runs")
    return out, images.parent


def blacken_example(cfg, folder, web):
    """One training photo before/after blackening, and one test photo predicted by the blackened and the uniform U-Net."""
    from PIL import Image

    from .data import crop
    from .neural import blacken
    from .training import run_path
    from .visuals import class_rgb, overlay
    store = Store(cfg)
    colours = class_rgb()

    def save(array, geometry, stem):
        Image.fromarray(crop(array, geometry)).save(folder / f"{stem}.webp", quality=88)
        return (folder / f"{stem}.webp").relative_to(web).as_posix()

    def predicted(directory, image_id):
        result = read_json(directory / "evaluation_test.json")
        index = next(i["prediction_index"] for i in result["per_image"] if i["id"] == image_id)
        return np.asarray(np.load(inside(result["predictions"]), mmap_mode="r")[index])

    train = next(r for r in store.select("train", 100) if r["processed_counts"][1] > 2000 and r["processed_counts"][2] > 2000)
    rgb, truth = store.get(train)
    test = next(r for r in store.select("test") if r["processed_counts"][1] > 2000 and r["processed_counts"][2] > 2000)
    trgb, ttruth = store.get(test)
    g = test["geometry"]
    return {"train_id": train["id"], "train_photo": save(rgb, train["geometry"], "train_photo"),
            "train_blackened": save(blacken(rgb, truth), train["geometry"], "train_blackened"),
            "test_id": test["id"], "test_truth": save(overlay(trgb, ttruth, colours, .55, .08), g, "test_truth"),
            "test_uniform": save(overlay(trgb, predicted(run_path("unet", 100, 42), test["id"]), colours, .55, .08), g, "test_uniform"),
            "test_blacken": save(overlay(trgb, predicted(run_dir("unet", 100, 42, "blacken"), test["id"]), colours, .55, .08), g, "test_blacken")}


def export_bundles(cfg):
    from .export import summarise, target_strata
    from .training import run_path
    ucfg, store = load_config(), Store(cfg)
    frozen = read_json(f"{ROOT}/experiment.json")
    results = read_json("output/undersample/results.json")
    budgets, seeds = ucfg["budgets"], ucfg["train_seeds"]
    parts = {"experiment": frozen["identity"], "results": sha256("output/undersample/results.json")}
    uniform = {(m, b, s): (run_dir("rf", b, s, "uniform") if m == "rf" else run_path(m, b, s))
               for m in ("rf", "unet", "hybrid") for b in budgets for s in seeds}
    pipeline_bundle(cfg, "uniform", uniform, parts | {"bundle": "uniform"},
                    {"label": "Uniform Before", "experiment_identity": frozen["identity"], "frozen_at": frozen["frozen_at"],
                     "comparisons": results["uniform_vs_first"], "rule": ucfg["uniform"]})
    chosen = {(m, b, s): run_dir(m, b, s, chosen_kind(m, b)) for m in ("rf", "unet", "hybrid") for b in budgets for s in seeds}
    out, folder = pipeline_bundle(cfg, "undersample", chosen, parts | {"bundle": "undersample"},
                                  {"label": "Undersampling pipeline", "experiment_identity": frozen["identity"], "frozen_at": frozen["frozen_at"],
                                   "comparisons": results["comparisons"], "ratio": results["ratio"], "rule": ucfg["undersample"]["rule"],
                                   "ratios": ucfg["undersample"]["ratios"], "blacken_rule": ucfg["blacken"]["rule"],
                                   "preregistered_at": read_json(f"{ROOT}/preregistration.json")["registered_at"],
                                   "study": {"runs": len(plan(ucfg)), "tested": len(frozen["tested"])}})
    overlap = read_json("data/manifests/overlap.json")
    clean_ids, strata = overlap["summary"]["test"]["clean_ids"], target_strata(store, overlap)
    blacken_runs = [public_run(run_dir(m, b, s, "blacken"), m, b, s, clean_ids, strata)[0]
                    for m in ("rf", "unet", "hybrid") for b in budgets for s in seeds]
    out["blacken"] = {"runs": [{k: r[k] for k in ("id", "method", "budget", "seed", "metrics", "training_seconds")} for r in blacken_runs],
                      "summary": summarise(blacken_runs, seeds)[0], "comparisons": results["blacken"],
                      "example": blacken_example(cfg, folder, inside("web"))}
    write_json(inside("web/data/undersample.json"), out)
    return out
