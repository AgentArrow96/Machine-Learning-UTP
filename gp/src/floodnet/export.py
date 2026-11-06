"""Portable, versioned results bundle. No checkpoints or absolute paths leave runs/."""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import PurePosixPath
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .common import CLASSES, COLOURS, ProjectError, digest, inside, log, now, read_json, sha256, write_json
from .data import Store, crop
from .evaluation import final_runs, subset_metrics
from .features import feature_key
from .hybrid import hybrid_runs

SCHEMA_VERSION = 3
METHODS = ("rf", "unet", "hybrid")
METHOD_NAMES = {"rf": "SLIC + RF", "unet": "U-Net", "hybrid": "U-Net + CRF"}

LIMITATIONS = [
    "This is a research prototype using one event/region, not an operational disaster-response service.",
    "The target is flooded buildings and flooded roads. Other includes floodwater, vegetation and non-flooded infrastructure.",
    "Evaluation is at 256 x 256 with aspect-ratio padding excluded. Thin boundaries may disappear during resizing.",
    "Pixel coverage does not measure physical area, flood depth, building counts or road passability.",
    "Three training seeds share one fixed 25% image subset: the spread reflects training randomness only, "
    "and three replicates are too few for a significance test.",
    "Most official test frames, and nearly all test frames containing flooded targets, share ground with a training "
    "frame (verified by keypoint matching). The test split therefore mostly measures performance on areas already "
    "seen in training, not on a new flood area.",
    "The hybrid's DenseCRF settings were chosen on validation from a small pre-set grid (six spatial/colour pairs). "
    "The best setting lay at the edge of that grid; finer settings were not tried, to keep tuning minimal.",
]

STRATA_FIELDS = ("flooded_miou", "building_iou", "road_iou")

# Flat metric names used by summaries, CSVs and the app.
METRIC_FIELDS = ("flooded_miou", "miou", "building_iou", "road_iou", "building_dice", "road_dice",
                 "building_recall", "road_recall")


def flat_metrics(metrics):
    building, road = metrics["per_class"][1], metrics["per_class"][2]
    return {"flooded_miou": metrics["flooded_miou"], "miou": metrics["miou"],
            "building_iou": building["iou"], "road_iou": road["iou"], "building_dice": building["dice"],
            "road_dice": road["dice"], "building_recall": building["recall"], "road_recall": road["recall"]}


def stats(values):
    """Mean, sample SD, range and the per-seed values; None entries make the statistic undefined."""
    if not values or any(v is None for v in values):
        return {"mean": None, "sd": None, "min": None, "max": None, "values": values}
    array = np.asarray(values, dtype=np.float64)
    return {"mean": float(array.mean()), "sd": float(array.std(ddof=1)) if len(array) > 1 else None,
            "min": float(array.min()), "max": float(array.max()), "values": [float(v) for v in values]}


def summarise(runs, seeds):
    summary, effects = [], []
    for method in METHODS:
        groups = {}
        for budget in (25, 100):
            group = sorted((r for r in runs if r["method"] == method and r["budget"] == budget), key=lambda r: r["seed"])
            if [r["seed"] for r in group] != sorted(seeds):
                continue
            groups[budget] = group
            summary.append({"method": method, "budget": budget, "n": len(group), "seeds": [r["seed"] for r in group],
                            "run_ids": [r["id"] for r in group],
                            "full": {k: stats([flat_metrics(r["metrics"])[k] for r in group]) for k in METRIC_FIELDS},
                            "clean": {k: stats([flat_metrics(r["clean_metrics"])[k] for r in group]) for k in METRIC_FIELDS},
                            **({f"overlap_{stratum}": {k: stats([r["overlap_strata"][stratum][k] for r in group]) for k in STRATA_FIELDS}
                                for stratum in ("low", "high")} if all(r.get("overlap_strata") for r in group) else {}),
                            "training_seconds": stats([r["training_seconds"] for r in group]),
                            "inference_median_seconds": stats([r["timing"]["median_seconds"] for r in group]),
                            "epochs": stats([r["epochs"] for r in group]) if method == "unet" else None,
                            "best_epoch": stats([r["best_epoch"] for r in group]) if method == "unet" else None})
        if len(groups) == 2:
            # Same seed at both budgets: a paired, seed-matched budget effect.
            effects.append({"method": method, "seeds": sorted(seeds),
                            "full": stats([b["metrics"]["flooded_miou"] - a["metrics"]["flooded_miou"]
                                           for a, b in zip(groups[25], groups[100])]),
                            "clean": stats([b["clean_metrics"]["flooded_miou"] - a["clean_metrics"]["flooded_miou"]
                                            for a, b in zip(groups[25], groups[100])])})
    return summary, effects


def crf_effects(runs, seeds):
    """Seed-matched change from each U-Net run to the hybrid that refines it (methodology section 5)."""
    effects = []
    for budget in (25, 100):
        pairs = []
        for seed in sorted(seeds):
            unet = next((r for r in runs if r["method"] == "unet" and r["budget"] == budget and r["seed"] == seed), None)
            hybrid = next((r for r in runs if r["method"] == "hybrid" and r["budget"] == budget and r["seed"] == seed), None)
            if unet and hybrid:
                pairs.append((unet, hybrid))
        if len(pairs) != len(seeds):
            continue
        change = lambda key: stats([flat_metrics(h["metrics"])[key] - flat_metrics(u["metrics"])[key] for u, h in pairs])
        effects.append({"budget": budget, "seeds": sorted(seeds),
                        "full": {key: change(key) for key in ("flooded_miou", "miou", "building_iou", "road_iou",
                                                              "building_recall", "road_recall")},
                        **({f"overlap_{stratum}": stats([h["overlap_strata"][stratum]["flooded_miou"] - u["overlap_strata"][stratum]["flooded_miou"]
                                                         for u, h in pairs]) for stratum in ("low", "high")}
                           if all(u.get("overlap_strata") and h.get("overlap_strata") for u, h in pairs) else {})})
    return effects


def tuning_section(selection):
    rows = []
    for round_ in selection["rounds"]:
        for index, params in enumerate(round_["settings"]):
            gains = [item["gain"][index] for item in round_["per_run"].values()]
            rows.append({"round": round_["name"], "params": params, "mean_gain": float(np.mean(gains)),
                         "min_gain": float(np.min(gains)), "max_gain": float(np.max(gains)),
                         "per_run": {run_id: item["gain"][index] for run_id, item in round_["per_run"].items()},
                         "selected": params == selection["selected"]})
    return {"backend": selection["backend"], "rule": selection["rule"], "selected": selection["selected"],
            "selected_mean_gain": selection["selected_mean_gain"], "adopted": selection["adopted"],
            "grid": rows, "tuning_seconds": selection["elapsed_seconds"],
            "unet_validation": {run_id: item["unet_flooded_miou"] for run_id, item in selection["rounds"][0]["per_run"].items()}}


def representatives(samples, count=6):
    selected = []
    for presence in ((True, True), (True, False), (False, True)):
        group = sorted([s for s in samples if tuple(s["presence"]) == presence], key=lambda s: (s["target_fraction"], s["id"]))
        if group:
            for i in (len(group) // 4, 3 * len(group) // 4):
                if group[i]["id"] not in selected:
                    selected.append(group[i]["id"])
    for sample in sorted(samples, key=lambda s: s["id"]):
        if len(selected) >= count:
            break
        if sample["id"] not in selected:
            selected.append(sample["id"])
    return selected[:count]


def check_url(url, web):
    path = PurePosixPath(url)
    if path.is_absolute() or ".." in path.parts or ":" in url or not (web / path).is_file():
        raise ProjectError(f"Invalid/missing web asset: {url}")


def check_scores(values):
    if any(v is not None and not 0 <= v <= 1 for v in values):
        raise ProjectError("Invalid score range")


def validate_bundle(bundle, web):
    if bundle["schema_version"] != SCHEMA_VERSION:
        raise ProjectError("Unknown web export schema")
    ids = [r["id"] for r in bundle["runs"]]
    if len(ids) != len(set(ids)):
        raise ProjectError("Duplicate exported run IDs")
    explorer = set(bundle["explorer_runs"])
    if not explorer <= set(ids):
        raise ProjectError("Explorer references an unexported run")
    for run in bundle["runs"]:
        for metrics in (run["metrics"], run["clean_metrics"]):
            if metrics is not None:
                check_scores(flat_metrics(metrics).values())
        for stratum in (run.get("overlap_strata") or {}).values():
            check_scores(stratum.values())
    for sample in bundle["samples"]:
        for url in (sample["rgb"], sample["truth"], *sample["predictions"].values()):
            check_url(url, web)
        if set(sample["predictions"]) != explorer or set(sample["metrics"]) != explorer:
            raise ProjectError(f"Sample {sample['id']} lacks explorer predictions")
        for metrics in sample["metrics"].values():
            check_scores(metrics.values())
        for url in sample.get("figures", {}).values():
            check_url(url, web)
    if bundle["dataset"] and bundle["dataset"].get("overlap"):
        for example in bundle["dataset"]["overlap"]["examples"]:
            check_url(example["eval_rgb"], web); check_url(example["train_rgb"], web)
    for url in bundle["downloads"].values():
        check_url(url, web)
    encoded = json.dumps(bundle, allow_nan=False)
    if "C:\\" in encoded or ".pt\"" in encoded or "model.joblib" in encoded:
        raise ProjectError("Training paths/checkpoints leaked into the web bundle")


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def qualitative_figure(sample, runs, web, assets, budget):
    """RGB, ground truth and every method's prediction for one test image (methodology section 5)."""
    rgb = Image.open(web / sample["rgb"]).convert("RGB")
    width, height = rgb.size
    font = ImageFont.load_default(size=13)
    columns = 3
    canvas = Image.new("RGB", (width * columns, (height + 30) * 2 + 52), "white")
    draw = ImageDraw.Draw(canvas)
    seed = runs[0]["seed"]
    draw.text((8, 6), f"FloodNet image {sample['image_id']} | {budget}% labels | seed {seed} | 256x256 (padding excluded)", fill="#0f172a", font=font)
    panel_specs = [("RGB", None), ("Ground truth", sample["truth"])]
    for method in METHODS:
        run = next((r for r in runs if r["method"] == method and r["budget"] == budget), None)
        if run:
            panel_specs.append((METHOD_NAMES[method], sample["predictions"][run["id"]]))
    palette = np.array([[148, 163, 184], [245, 158, 11], [8, 145, 178]], dtype=np.float32)
    for index, (label, mask_url) in enumerate(panel_specs):
        left, top = (index % columns) * width, 30 + (index // columns) * (height + 30)
        display = rgb
        if mask_url:
            with Image.open(web / mask_url) as image:
                labels = np.asarray(image)
            alpha = np.where(labels == 0, .09, .55)[..., None]
            display = Image.fromarray(np.round(np.asarray(rgb) * (1-alpha) + palette[labels] * alpha).astype(np.uint8))
        draw.text((left + 6, top + 2), label, fill="#0f172a", font=font)
        canvas.paste(display, (left, top + 24))
    draw.text((8, canvas.height - 19), "Orange: flooded building | Cyan: flooded road | Research prototype", fill="#64748b", font=font)
    path = assets / "figures" / f"{sample['id']}_budget{budget}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    return path.relative_to(web).as_posix()


def slim_history(history):
    keys = ("epoch", "train_loss", "val_loss", "val_flooded_miou", "val_miou", "train_seconds")
    return [{k: entry[k] for k in keys} for entry in history]


def overlap_section(store, overlap, images, web):
    test_ids = {r["id"] for r in store.select("test")}
    verified = [p for p in overlap["pairs"] if p["verified"] and p["eval_id"] in test_ids]
    # Illustrate with the strongest pair per distinct test image (deterministic ordering).
    examples, seen = [], set()
    for pair in sorted(verified, key=lambda p: (-p["overlap"], p["eval_id"], p["train_id"])):
        if pair["eval_id"] in seen:
            continue
        seen.add(pair["eval_id"])
        train = store.by_id[pair["train_id"]]
        train_path = images / f"{train['id']}_rgb.webp"
        Image.fromarray(crop(store.get(train)[0], train["geometry"])).save(train_path, quality=90)
        examples.append({"eval_id": pair["eval_id"], "train_id": pair["train_id"], "overlap": pair["overlap"],
                         "inliers": pair["inliers"], "eval_rgb": (images / f"{pair['eval_id']}_rgb.webp").relative_to(web).as_posix(),
                         "train_rgb": train_path.relative_to(web).as_posix()})
        if len(examples) == 3:
            break
    maxima = [overlap["per_image"][i]["max_train_overlap"] for i in sorted(test_ids)]
    edges = [0, 1e-9, .25, .5, .75, 1.0001]
    histogram = [{"label": label, "count": int(sum(lo <= m < hi for m in maxima))}
                 for label, lo, hi in zip(("None", "Under 25%", "25–50%", "50–75%", "75–100%"), edges[:-1], edges[1:])]
    summary = {split: {k: v for k, v in item.items() if k != "clean_ids"} | {"clean": len(item["clean_ids"])}
               for split, item in overlap["summary"].items()}
    strata = target_strata(store, overlap)
    clean = set(overlap["summary"]["test"]["clean_ids"])
    test = store.select("test")
    targets = {"images": len(strata["low"]) + len(strata["high"]),
               "overlapping": sum(bool(overlap["per_image"][i]["verified_train_ids"]) for i in strata["low"] + strata["high"]),
               "split_overlap": strata["threshold"], "low_images": len(strata["low"]), "high_images": len(strata["high"]),
               "clean_with_building": sum(r["id"] in clean and r["processed_counts"][1] > 0 for r in test),
               "clean_with_road": sum(r["id"] in clean and r["processed_counts"][2] > 0 for r in test)}
    return {"identity": overlap["identity"], "method": overlap["method"], "settings": overlap["settings"],
            "summary": summary, "target_images": targets, "test_max_overlap_histogram": histogram, "examples": examples}


def target_strata(store, overlap):
    """Split target-containing test images at their median training overlap.

    Almost every flooded test scene overlaps a training frame, so an overlap-free subset holds
    too few targets to score. Comparing lower- and higher-overlap halves of the target images
    keeps the class mix comparable and asks whether more shared ground means higher scores."""
    targets = sorted(r["id"] for r in store.select("test") if r["processed_counts"][1] or r["processed_counts"][2])
    overlaps = {i: overlap["per_image"][i]["max_train_overlap"] for i in targets}
    threshold = float(np.median(list(overlaps.values())))
    return {"threshold": threshold, "low": [i for i in targets if overlaps[i] < threshold],
            "high": [i for i in targets if overlaps[i] >= threshold]}


def prune_exports(web, keep):
    exports = web / "data/exports"
    for directory in exports.iterdir() if exports.exists() else []:
        if directory.is_dir() and directory.name != keep:
            shutil.rmtree(inside(directory))
            log(f"Removed stale web export {directory.name}")


def export_web(cfg):
    web = inside("web")
    web.mkdir(parents=True, exist_ok=True)
    seeds = cfg["train_seeds"]
    primary = seeds[0]
    expected = final_runs(cfg) + hybrid_runs(cfg)
    bundle = {"schema_version": SCHEMA_VERSION, "exported_at": now(), "status": "incomplete", "title": "FloodNet / Infrastructure in focus",
              "classes": [{"id": i, "name": name, "colour": COLOURS[i]} for i, name in enumerate(CLASSES)],
              "protocol": {"subset_seed": cfg["subset_seed"], "train_seeds": seeds, "explorer_seed": primary, "size": cfg["size"],
                           "budgets": [25, 100], "ignore_index": 255, "expected_runs": len(expected), "methods": list(METHODS),
                           "trained_runs": len(final_runs(cfg)), "hybrid_runs": len(hybrid_runs(cfg)),
                           "rf_region_sampling": cfg["rf"]["region_sampling"], "rf_max_regions": cfg["rf"]["max_regions"],
                           "primary_metric": "Mean of flooded-building and flooded-road IoU from pooled split confusion counts"},
              "dataset": None, "runs": [], "summary": [], "budget_effects": [], "crf_effects": [], "hybrid": None,
              "explorer_runs": [], "samples": [],
              "representatives": [], "limitations": LIMITATIONS, "reproducibility": None,
              "sources": [{"title": "FloodNet official dataset", "url": "https://github.com/BinaLab/FloodNet-Supervised_v1.0"},
                          {"title": "FloodNet paper", "url": "https://doi.org/10.1109/ACCESS.2021.3090981"},
                          {"title": "CDLA-Permissive 1.0", "url": "https://cdla.io/permissive-1-0/"}],
              "missing_runs": [], "downloads": {}}
    processed_path = inside("data/manifests/processed.json")
    if not processed_path.exists():
        bundle["message"] = "Data preparation and final evaluations have not completed. No measured results are available."
        write_json(web / "data/results.json", bundle)
        return bundle
    store = Store(cfg)
    audit = read_json("data/manifests/audit.json")
    overlap_path = inside("data/manifests/overlap.json")
    overlap = read_json(overlap_path) if overlap_path.exists() else None
    if overlap and overlap["processed_identity"] != store.manifest["identity"]:
        raise ProjectError("Overlap analysis belongs to a different preprocessing manifest")
    clean_ids = overlap["summary"]["test"]["clean_ids"] if overlap else None
    strata = target_strata(store, overlap) if overlap else None
    bundle["dataset"] = {"name": "FloodNet-Supervised v1.0", "counts": audit["counts"], "original_classes": audit["original_classes"],
                         "distributions": audit["distributions"], "processed_distributions": {}, "audit_identity": audit["identity"],
                         "processed_identity": store.manifest["identity"], "budget_counts": {k: len(v["ids"]) for k, v in store.manifest["budgets"].items()},
                         "exact_duplicate_groups": audit["exact_duplicates"], "targets_lost_images": store.manifest["targets_lost_images"],
                         "source": audit.get("source"), "preprocessing_seconds": store.manifest["elapsed_seconds"],
                         "orientation_policy": audit.get("orientation_policy"),
                         "exif_rotated_images": sum(r.get("exif_orientation", 1) != 1 for r in audit["records"]), "overlap": None}
    feature_manifest_path = inside(f"data/features/{feature_key(cfg, store)[:16]}/manifest.json")
    bundle["dataset"]["feature_preparation_seconds"] = read_json(feature_manifest_path)["elapsed_seconds"] if feature_manifest_path.exists() else None
    for split in ("train", "val", "test"):
        counts = np.array([r["processed_counts"] for r in store.select(split)])
        bundle["dataset"]["processed_distributions"][split] = {"pixel_counts": counts.sum(0).tolist(), "image_presence": (counts > 0).sum(0).tolist()}
    frozen_path = inside("runs/final_experiment.json")
    frozen = read_json(frozen_path) if frozen_path.exists() else None
    if frozen and overlap and frozen.get("overlap_identity") != overlap["identity"]:
        raise ProjectError("Frozen experiment used a different overlap analysis")
    hybrid_frozen_path = inside("runs/hybrid_experiment.json")
    hybrid_frozen = read_json(hybrid_frozen_path) if hybrid_frozen_path.exists() else None
    if hybrid_frozen and (not frozen or hybrid_frozen["base_experiment_identity"] != frozen["identity"]):
        raise ProjectError("The hybrid freeze refers to a different 12-run experiment")
    export_id = (digest({"base": frozen["identity"], "hybrid": hybrid_frozen["identity"]}) if hybrid_frozen
                 else frozen["identity"] if frozen else store.manifest["identity"])[:16]
    selection_path = inside("runs/hybrid_selection.json")
    if selection_path.exists():
        selection = read_json(selection_path)
        if hybrid_frozen and selection["identity"] != hybrid_frozen["selection_identity"]:
            raise ProjectError("CRF selection changed after the hybrid freeze")
        bundle["hybrid"] = tuning_section(selection)
    assets = web / f"data/exports/{export_id}"
    images = assets / "images"
    images.mkdir(parents=True, exist_ok=True)
    if overlap:
        bundle["dataset"]["overlap"] = overlap_section(store, overlap, images, web)
    from .training import run_path
    loaded = []
    for method, budget, seed in expected:
        directory = run_path(method, budget, seed)
        result_path = directory / "evaluation_test.json"
        if not result_path.exists():
            bundle["missing_runs"].append(directory.name)
            continue
        result, run = read_json(result_path), read_json(directory / "run.json")
        freeze = hybrid_frozen if method == "hybrid" else frozen
        if not freeze or result["experiment_identity"] != freeze["identity"] or result["run_identity"] != run["identity"] or result["run_type"] != "final":
            raise ProjectError("Unfrozen or incompatible test result cannot be exported")
        if result["image_count"] != audit["counts"]["test"] or result["checkpoint_sha256"] != run["checkpoint_sha256"]:
            raise ProjectError("Incomplete or stale test evaluation")
        if sha256(result["predictions"]) != result["predictions_sha256"]:
            raise ProjectError("Prediction file changed after evaluation")
        clean = subset_metrics(result["per_image"], clean_ids) if clean_ids is not None else None
        overlap_strata = {stratum: {k: flat_metrics(subset_metrics(result["per_image"], strata[stratum]))[k] for k in STRATA_FIELDS}
                          for stratum in ("low", "high")} if strata else None
        public_run = {"id": run["run_id"], "method": method, "budget": budget, "seed": seed, "status": run["status"],
                      "metrics": result["metrics"], "clean_metrics": clean, "clean_images": len(clean_ids) if clean_ids is not None else None,
                      "overlap_strata": overlap_strata,
                      "training_seconds": run["training_seconds"], "timing": {k: result["timing"][k] for k in ("device", "median_seconds", "p95_seconds", "boundary")},
                      "peak_gpu_allocated_bytes": run.get("peak_gpu_allocated_bytes"), "peak_gpu_reserved_bytes": run.get("peak_gpu_reserved_bytes"),
                      "device": run.get("gpu", "CPU"), "precision": run.get("precision"), "identity": run["identity"],
                      "checkpoint_sha256": run["checkpoint_sha256"], "epochs": run.get("epochs"), "best_epoch": run.get("best_epoch"),
                      "region_count": run.get("region_count"), "region_class_counts": run.get("region_class_counts"),
                      "crf": run.get("crf"), "base_run_id": run.get("base_run", {}).get("run_id"),
                      "history": slim_history(read_json(directory / "history.json")) if (directory / "history.json").exists() else []}
        bundle["runs"].append(public_run)
        loaded.append((public_run, result, run))
    if loaded:
        first = loaded[0][2]
        hybrid_run = next((run for _, _, run in loaded if run["identity"]["method"] == "hybrid"), None)
        bundle["reproducibility"] = {"versions": hybrid_run["versions"] if hybrid_run else first["versions"], "config": first["config"],
                                     "crf": hybrid_run["crf"] if hybrid_run else None}
    explorer = [(public_run, result) for public_run, result, _ in loaded if public_run["seed"] == primary]
    bundle["explorer_runs"] = [public_run["id"] for public_run, _ in explorer]
    per_image_rows = {public_run["id"]: {item["id"]: item for item in result["per_image"]} for public_run, result, _ in loaded}
    for public_run, result in explorer:
        public_run["_predictions"] = np.load(inside(result["predictions"]), mmap_mode="r")
    for record in store.select("test"):
        rgb, truth = store.get(record)
        rgb_path, truth_path = images / f"{record['id']}_rgb.webp", images / f"{record['id']}_truth.png"
        Image.fromarray(crop(rgb, record["geometry"])).save(rgb_path, quality=92)
        Image.fromarray(crop(truth, record["geometry"])).save(truth_path)
        image_overlap = overlap["per_image"][record["id"]] if overlap else None
        sample = {"id": record["id"], "image_id": record["image_id"], "rgb": rgb_path.relative_to(web).as_posix(),
                  "truth": truth_path.relative_to(web).as_posix(), "presence": [record["processed_counts"][c] > 0 for c in (1, 2)],
                  "target_fraction": sum(record["processed_counts"][1:]) / sum(record["processed_counts"]),
                  "geometry": record["geometry"], "predictions": {}, "metrics": {},
                  "train_overlap": image_overlap["max_train_overlap"] if image_overlap else None,
                  "overlapping_train_ids": image_overlap["verified_train_ids"] if image_overlap else None}
        for public_run, _ in explorer:
            item = per_image_rows[public_run["id"]].get(record["id"])
            if item is None:
                raise ProjectError("Evaluation missing a test image")
            prediction_path = images / f"{record['id']}_{public_run['id']}.png"
            Image.fromarray(crop(public_run["_predictions"][item["prediction_index"]], record["geometry"])).save(prediction_path)
            sample["predictions"][public_run["id"]] = prediction_path.relative_to(web).as_posix()
            sample["metrics"][public_run["id"]] = {k: flat_metrics(item["metrics"])[k] for k in ("flooded_miou", "building_iou", "road_iou")}
        bundle["samples"].append(sample)
    for public_run, _ in explorer:
        del public_run["_predictions"]
    bundle["representatives"] = representatives(bundle["samples"])
    bundle["summary"], bundle["budget_effects"] = summarise(bundle["runs"], seeds)
    bundle["crf_effects"] = crf_effects(bundle["runs"], seeds)
    if len(bundle["runs"]) == len(expected) and overlap:
        bundle["status"] = "complete"
        bundle["experiment_identity"] = frozen["identity"]
        bundle["hybrid_experiment_identity"] = hybrid_frozen["identity"]
        write_json(assets / "experiment.json", frozen)
        write_json(assets / "hybrid_experiment.json", hybrid_frozen)
        bundle["downloads"]["experiment"] = (assets / "experiment.json").relative_to(web).as_posix()
        bundle["downloads"]["hybrid_experiment"] = (assets / "hybrid_experiment.json").relative_to(web).as_posix()
        explorer_runs = [public_run for public_run, _ in explorer]
        figures = []
        for sample in bundle["samples"]:
            if sample["id"] in bundle["representatives"]:
                sample["figures"] = {str(b): qualitative_figure(sample, explorer_runs, web, assets, b) for b in (25, 100)}
                figures.append({"id": sample["id"], "figures": sample["figures"]})
        write_json(assets / "figures.json", figures)
        bundle["downloads"]["figures"] = (assets / "figures.json").relative_to(web).as_posix()
    else:
        bundle["message"] = f"Final experiment is incomplete: {len(bundle['runs'])} of {len(expected)} frozen test evaluations exported."
    aggregate_rows, per_rows = [], []
    for run in bundle["runs"]:
        full, clean = flat_metrics(run["metrics"]), flat_metrics(run["clean_metrics"]) if run["clean_metrics"] else {}
        aggregate_rows.append({"run_id": run["id"], "method": run["method"], "budget": run["budget"], "seed": run["seed"],
                               **full, "clean_flooded_miou": clean.get("flooded_miou"), "clean_miou": clean.get("miou"),
                               "low_overlap_target_flooded_miou": (run["overlap_strata"] or {}).get("low", {}).get("flooded_miou"),
                               "high_overlap_target_flooded_miou": (run["overlap_strata"] or {}).get("high", {}).get("flooded_miou"),
                               "clean_images": run["clean_images"], "training_seconds": run["training_seconds"],
                               "inference_median_seconds": run["timing"]["median_seconds"], "epochs": run["epochs"],
                               "best_epoch": run["best_epoch"]})
        for image_id, item in sorted(per_image_rows[run["id"]].items()):
            metrics = flat_metrics(item["metrics"])
            per_rows.append({"image_id": image_id, "run_id": run["id"], "seed": run["seed"], "flooded_miou": metrics["flooded_miou"],
                             "miou": metrics["miou"], "building_iou": metrics["building_iou"], "road_iou": metrics["road_iou"],
                             "train_overlap": overlap["per_image"][image_id]["max_train_overlap"] if overlap else None})
    summary_rows = []
    for item in bundle["summary"]:
        row = {"method": item["method"], "budget": item["budget"], "n": item["n"], "seeds": " ".join(map(str, item["seeds"]))}
        for scope in ("full", "clean"):
            for key in ("flooded_miou", "miou", "building_iou", "road_iou"):
                row[f"{scope}_{key}_mean"], row[f"{scope}_{key}_sd"] = item[scope][key]["mean"], item[scope][key]["sd"]
        summary_rows.append(row)
    write_csv(assets / "aggregate.csv", aggregate_rows,
              ["run_id", "method", "budget", "seed", *METRIC_FIELDS, "clean_flooded_miou", "clean_miou", "clean_images",
               "low_overlap_target_flooded_miou", "high_overlap_target_flooded_miou",
               "training_seconds", "inference_median_seconds", "epochs", "best_epoch"])
    write_csv(assets / "summary.csv", summary_rows,
              ["method", "budget", "n", "seeds"] + [f"{scope}_{key}_{stat}" for scope in ("full", "clean")
                                                    for key in ("flooded_miou", "miou", "building_iou", "road_iou") for stat in ("mean", "sd")])
    write_csv(assets / "per_image.csv", per_rows, ["image_id", "run_id", "seed", "flooded_miou", "miou", "building_iou", "road_iou", "train_overlap"])
    write_json(assets / "budgets.json", store.manifest["budgets"])
    downloads = [("aggregate", "aggregate.csv"), ("summary", "summary.csv"), ("per_image", "per_image.csv"), ("budgets", "budgets.json")]
    if bundle["hybrid"]:
        tuning_runs = sorted(bundle["hybrid"]["grid"][0]["per_run"])
        write_csv(assets / "crf_tuning.csv",
                  [{"round": row["round"], "bilateral_sxy": row["params"]["bilateral"]["sxy"], "bilateral_srgb": row["params"]["bilateral"]["srgb"],
                    "bilateral_compat": row["params"]["bilateral"]["compat"], "mean_validation_gain": row["mean_gain"],
                    "selected": row["selected"], **{f"gain_{run_id}": row["per_run"][run_id] for run_id in tuning_runs}}
                   for row in bundle["hybrid"]["grid"]],
                  ["round", "bilateral_sxy", "bilateral_srgb", "bilateral_compat", "mean_validation_gain", "selected",
                   *[f"gain_{run_id}" for run_id in tuning_runs]])
        downloads.append(("crf_tuning", "crf_tuning.csv"))
    if overlap:
        write_csv(assets / "overlap_pairs.csv",
                  [{k: p[k] for k in ("eval_id", "train_id", "matches", "inliers", "inlier_ratio", "overlap")} | {"sources": " ".join(p["sources"])}
                   for p in overlap["pairs"] if p["verified"]],
                  ["eval_id", "train_id", "sources", "matches", "inliers", "inlier_ratio", "overlap"])
        downloads.append(("overlap", "overlap_pairs.csv"))
    for key, filename in downloads:
        bundle["downloads"][key] = (assets / filename).relative_to(web).as_posix()
    validate_bundle(bundle, web)
    # Publish the manifest last; existing readers never see references to unwritten assets.
    write_json(web / "data/results.json", bundle)
    prune_exports(web, export_id)
    log(f"Web export {bundle['status']}: {len(bundle['runs'])}/{len(expected)} runs; {len(bundle['samples'])} test samples")
    return bundle
