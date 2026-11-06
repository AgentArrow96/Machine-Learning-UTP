"""E1-E5, L1-L3, L5-L7: independent end-to-end evaluation of every run. Never imports `floodnet`.

Inputs: 256x256 arrays derived from the RAW release by audit_data.py (not the project cache).
Models: rebuilt from their checkpoints with segmentation_models_pytorch / joblib / pydensecrf.
Scoring: sklearn.metrics.confusion_matrix. Test masks are loaded only after all predictions exist.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from independent import DERIVED, MEAN, OUT, ROOT, STD, confusion, guard, read_json, scores, write_json

os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
os.environ.setdefault("TORCH_HOME", str(ROOT / ".cache/torch"))
os.environ["HF_HUB_OFFLINE"] = "1"

METHODS = ("rf", "unet", "hybrid")
BUDGETS = (25, 100)
SEEDS = (42, 43, 44)
WORKERS = 12


# ---------- inputs ----------

def load_images(split):
    data = np.load(DERIVED / f"{split}.npz")
    geometry = json.loads((DERIVED / f"{split}_geometry.json").read_text(encoding="utf-8"))
    return [str(i) for i in data["ids"]], data["rgb"], geometry


def load_masks(split):
    return np.load(DERIVED / f"{split}.npz")["mask"]


def window(array, g):
    return array[..., g["top"]:g["top"] + g["height"], g["left"]:g["left"] + g["width"]]


# ---------- U-Net ----------

def build_unet(config, weights=None):
    import segmentation_models_pytorch as smp
    u = config["unet"]
    return smp.Unet(encoder_name=u["encoder"], encoder_weights=weights, decoder_channels=tuple(u["decoder_channels"]),
                    in_channels=3, classes=3, activation=None)


def load_unet(run_dir):
    import torch
    run = read_json(run_dir / "run.json")
    model = build_unet(run["config"])
    state = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)["model"]
    model.load_state_dict(state)
    return model.cuda().eval(), run


def unet_logits(model, rgb):
    """Batch-one bf16 forward pass on ImageNet-normalised RGB, like the documented inference."""
    import torch
    x = torch.from_numpy(np.ascontiguousarray(((rgb.astype(np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1)))[None].cuda()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        return model(x)


def unet_predict(model, rgb_all):
    import torch
    labels, probs = [], []
    for rgb in rgb_all:
        logits = unet_logits(model, rgb)
        labels.append(logits.argmax(1)[0].cpu().numpy().astype(np.uint8))
        with torch.inference_mode():
            probs.append(torch.softmax(logits.float(), 1)[0].cpu().numpy())
    return np.stack(labels), probs


# ---------- DenseCRF ----------

def crf_labels(probs, rgb, params):
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_softmax
    classes, height, width = probs.shape
    crf = dcrf.DenseCRF2D(width, height, classes)
    crf.setUnaryEnergy(np.ascontiguousarray(unary_from_softmax(np.ascontiguousarray(probs, dtype=np.float32))))
    crf.addPairwiseGaussian(sxy=params["gaussian"]["sxy"], compat=params["gaussian"]["compat"])
    crf.addPairwiseBilateral(sxy=params["bilateral"]["sxy"], srgb=params["bilateral"]["srgb"],
                             rgbim=np.ascontiguousarray(rgb, dtype=np.uint8), compat=params["bilateral"]["compat"])
    return np.array(crf.inference(params["iterations"]), dtype=np.float32).reshape(classes, height, width).argmax(0).astype(np.uint8)


def crf_job(job):
    probs, rgb, params = job
    return crf_labels(probs, rgb, params)


def crf_grid_job(job):
    probs, rgb, truth, settings = job
    return np.stack([confusion(truth, crf_labels(probs, rgb, p)) for p in settings])


def hybrid_predict(probs, rgb_all, geometry, ids, params, pool):
    jobs = [(window(p, geometry[i]).copy(), window(rgb.transpose(2, 0, 1), geometry[i]).transpose(1, 2, 0).copy(), params)
            for p, rgb, i in zip(probs, rgb_all, ids)]
    refined = list(pool.map(crf_job, jobs, chunksize=4))
    out = np.stack([p.argmax(0).astype(np.uint8) for p in probs])
    for k, (labels, i) in enumerate(zip(refined, ids)):
        window(out[k], geometry[i])[...] = labels
    return out


# ---------- Random forest ----------

def feature_dir():
    candidates = [p for p in (ROOT / "data/features").iterdir() if p.is_dir() and (p / "manifest.json").exists() and p.name != "overlap"]
    assert len(candidates) == 1, candidates
    return candidates[0]


def independent_rf_features(job):
    """Region descriptors written from the protocol text: SLIC 250 / compactness 10 on unpadded content;
    RGB and HSV mean+std; 10-bin normalised uniform LBP (P=8, R=1)."""
    from scipy import ndimage
    from skimage.color import rgb2gray, rgb2hsv
    from skimage.feature import local_binary_pattern
    from skimage.segmentation import slic
    rgb, g = job
    content = rgb[g["top"]:g["top"] + g["height"], g["left"]:g["left"] + g["width"]]
    segments = slic(content, n_segments=250, compactness=10, max_num_iter=10, enforce_connectivity=True, start_label=1, channel_axis=-1)
    labels, regions = np.unique(segments, return_inverse=True)
    regions = regions.reshape(segments.shape)
    index = np.arange(len(labels))
    image = content.astype(np.float64) / 255
    columns = []
    for space in (image, rgb2hsv(image)):
        columns.append(np.stack([ndimage.mean(space[..., c], regions, index) for c in range(3)], 1))
        columns.append(np.stack([ndimage.standard_deviation(space[..., c], regions, index) for c in range(3)], 1))
    lbp = local_binary_pattern(np.round(rgb2gray(image) * 255).astype(np.uint8), P=8, R=1, method="uniform").astype(int)
    histogram = np.stack([np.bincount(lbp[regions == r], minlength=10)[:10] for r in index]).astype(np.float64)
    histogram /= histogram.sum(1, keepdims=True)
    full = np.full(rgb.shape[:2], -1, dtype=np.int64)
    full[g["top"]:g["top"] + g["height"], g["left"]:g["left"] + g["width"]] = regions
    return np.concatenate(columns + [histogram], 1).astype(np.float32), full


def rf_expand(model, features, regions):
    labels = model.predict(features).astype(np.uint8)
    out = np.zeros(regions.shape, dtype=np.uint8)
    valid = regions >= 0
    out[valid] = labels[regions[valid]]
    return out


# ---------- scoring ----------

def per_image_confusions(predictions, masks):
    return np.stack([confusion(t, p) for t, p in zip(masks, predictions)])


def summary(cms):
    s = scores(cms.sum(0))
    return {"flooded_miou": s["flooded_miou"], "miou": s["miou"], "building_iou": s["iou"][1], "road_iou": s["iou"][2],
            "building_recall": s["recall"][1], "road_recall": s["recall"][2], "confusion": cms.sum(0).tolist()}


def stored_predictions(run_dir, ids):
    result = read_json(run_dir / "evaluation_test.json")
    array = np.load(run_dir / "predictions_test.npy", mmap_mode="r")
    position = {item["id"]: item["prediction_index"] for item in result["per_image"]}
    return np.stack([array[position[i]] for i in ids]), result


def compare(mine, theirs, masks):
    padded = mine.copy()
    padded[masks == 255] = 0  # the project stores 0 in padding; padding is never scored
    return {"differing_pixels": int(np.count_nonzero(padded != theirs)),
            "differing_images": int(np.count_nonzero((padded != theirs).reshape(len(mine), -1).any(1)))}


# ---------- main evaluation of 18 runs ----------

def evaluate_root(root, tag, pool, crf_params):
    ids, rgb_all, geometry = load_images("test")
    predictions, timings, comparisons = {}, {}, {}
    # Predict everything first: no test mask is in memory yet.
    import joblib
    cached = feature_dir()
    project_features = {i: np.load(cached / f"{i}.npz") for i in ids}
    independent = list(pool.map(independent_rf_features, [(rgb, geometry[i]) for rgb, i in zip(rgb_all, ids)], chunksize=4))
    feature_drift = float(np.max([np.abs(f - project_features[i]["x"]).max() if f.shape == project_features[i]["x"].shape else np.inf
                                  for (f, _), i in zip(independent, ids)]))
    region_mismatch = int(sum(not np.array_equal(r, project_features[i]["regions"]) for (_, r), i in zip(independent, ids)))
    for budget in BUDGETS:
        for seed in SEEDS:
            started = time.perf_counter()
            model = joblib.load(root / f"rf_{budget}_s{seed}" / "model.joblib")
            model.n_jobs = 1
            predictions[f"rf_{budget}_s{seed}"] = np.stack([rf_expand(model, project_features[i]["x"], project_features[i]["regions"]) for i in ids])
            predictions[f"rf_{budget}_s{seed}#independent_features"] = np.stack([rf_expand(model, f, r) for f, r in independent])
            timings[f"rf_{budget}_s{seed}"] = time.perf_counter() - started
            started = time.perf_counter()
            unet, _ = load_unet(root / f"unet_{budget}_s{seed}")
            labels, probs = unet_predict(unet, rgb_all)
            predictions[f"unet_{budget}_s{seed}"] = labels
            timings[f"unet_{budget}_s{seed}"] = time.perf_counter() - started
            started = time.perf_counter()
            predictions[f"hybrid_{budget}_s{seed}"] = hybrid_predict(probs, rgb_all, geometry, ids, crf_params, pool)
            timings[f"hybrid_{budget}_s{seed}"] = time.perf_counter() - started
            del unet, probs
    masks = load_masks("test")
    cms = {name: per_image_confusions(p, masks) for name, p in predictions.items()}
    metrics = {name: summary(c) for name, c in cms.items()}
    for name in predictions:
        if "#" in name:
            continue
        run_dir = ROOT / "runs" / name  # always compare against the FROZEN stored predictions
        if (run_dir / "predictions_test.npy").exists():
            theirs, result = stored_predictions(run_dir, ids)
            comparisons[name] = {**compare(predictions[name], theirs, masks),
                                 "confusion_equal": metrics[name]["confusion"] == result["metrics"]["confusion_matrix"]}
    np.savez_compressed(OUT / f"per_image_cms_{tag}.npz", ids=np.array(ids), **{k.replace("#", "__"): v for k, v in cms.items()})
    keep = {k.replace("#", "__"): v for k, v in predictions.items() if k.endswith("_s42")}
    np.savez_compressed(OUT / f"predictions_s42_{tag}.npz", ids=np.array(ids), **keep)
    return {"metrics": metrics, "comparisons": comparisons, "seconds": timings,
            "independent_features": {"max_abs_feature_difference": feature_drift, "images_with_different_superpixels": region_mismatch}}, predictions, masks, ids


# ---------- controls (frozen runs only) ----------

def derangement(n, seed=7):
    rng = np.random.default_rng(seed)
    while True:
        perm = rng.permutation(n)
        if not np.any(perm == np.arange(n)):
            return perm


def controls(predictions, masks, ids, pool):
    out = {}
    perm = derangement(len(masks))
    out["L1"] = {name: scores(per_image_confusions(p, masks[perm]).sum(0))["flooded_miou"] for name, p in predictions.items() if "#" not in name}
    train_masks = load_masks("train")
    counts = np.array([(train_masks == c).sum() for c in range(3)], dtype=np.float64)
    priors = counts / counts.sum()
    rng = np.random.default_rng(11)
    shape = masks.shape
    baselines = {"all_other": np.zeros(shape, np.uint8), "all_building": np.ones(shape, np.uint8),
                 "all_road": np.full(shape, 2, np.uint8), "class_prior_random": rng.choice(3, size=shape, p=priors).astype(np.uint8)}
    out["L2"] = {"train_pixel_priors": priors.tolist(),
                 "baselines": {name: scores(per_image_confusions(p, masks).sum(0)) for name, p in baselines.items()}}
    # L3: networks that never saw a FloodNet label.
    import torch
    _, rgb_all, _ = load_images("test")
    config = read_json(ROOT / "runs/unet_100_s42/run.json")["config"]
    untrained = {}
    for name, weights in (("imagenet_encoder_random_decoder", "imagenet"), ("fully_random", None)):
        torch.manual_seed(0)
        model = build_unet(config, weights).cuda().eval()
        labels, _ = unet_predict(model, rgb_all)
        untrained[name] = scores(per_image_confusions(labels, masks).sum(0))
        del model
    out["L3"] = untrained
    # L7: perfect copies on target-bearing images, and per-image IoU distribution.
    valid = masks != 255
    target_share = np.array([((m == 1) | (m == 2))[v].mean() for m, v in zip(masks, valid)])
    perfect = {}
    histograms = {}
    for name, p in predictions.items():
        if "#" in name:
            continue
        exact = [ids[k] for k in range(len(masks)) if target_share[k] > .01 and np.array_equal(p[k][valid[k]], masks[k][valid[k]])]
        perfect[name] = exact
        per_image = [scores(confusion(masks[k], p[k]))["flooded_miou"] for k in range(len(masks)) if target_share[k] > .01]
        histograms[name] = np.histogram([v for v in per_image if v is not None], bins=10, range=(0, 1))[0].tolist()
    out["L7"] = {"target_images_over_1pct": int((target_share > .01).sum()), "perfect_copies": perfect, "per_image_iou_histograms": histograms}
    return out


def selection_checks(pool):
    """L5: checkpoint selection and CRF selection used validation only, recomputed independently."""
    histories = {}
    for budget in BUDGETS:
        for seed in SEEDS:
            run_dir = ROOT / f"runs/unet_{budget}_s{seed}"
            history = read_json(run_dir / "history.json")
            scores_ = [e["val_flooded_miou"] for e in history]
            first_best = int(np.argmax(scores_)) + 1
            histories[run_dir.name] = {"best_epoch_recorded": read_json(run_dir / "run.json")["best_epoch"], "first_argmax_epoch": first_best,
                                       "epochs": len(history)}
    selection = read_json(ROOT / "runs/hybrid_selection.json")
    settings = selection["rounds"][0]["settings"]
    ids, rgb_all, geometry = load_images("val")
    val_masks = load_masks("val")
    recomputed = {}
    for budget in BUDGETS:
        for seed in SEEDS:
            name = f"unet_{budget}_s{seed}"
            model, _ = load_unet(ROOT / "runs" / name)
            labels, probs = unet_predict(model, rgb_all)
            base = scores(per_image_confusions(labels, val_masks).sum(0))["flooded_miou"]
            jobs = [(window(p, geometry[i]).copy(), window(rgb.transpose(2, 0, 1), geometry[i]).transpose(1, 2, 0).copy(),
                     window(m, geometry[i]).copy(), settings) for p, rgb, m, i in zip(probs, rgb_all, val_masks, ids)]
            totals = np.sum(list(pool.map(crf_grid_job, jobs, chunksize=4)), axis=0)
            recomputed[name] = {"unet_val": base, "gains": [scores(t)["flooded_miou"] - base for t in totals]}
            del model, probs
    recorded = selection["rounds"][0]["per_run"]
    max_gain_difference = max(abs(a - b) for name in recomputed for a, b in zip(recomputed[name]["gains"], recorded[name]["gain"]))
    max_base_difference = max(abs(recomputed[n]["unet_val"] - recorded[n]["unet_flooded_miou"]) for n in recomputed)
    mean_gain = np.mean([recomputed[n]["gains"] for n in recomputed], axis=0)
    chosen = settings[int(np.argmax(mean_gain))]
    return {"histories": histories, "crf_recomputed_mean_gains": mean_gain.tolist(), "crf_max_gain_difference": max_gain_difference,
            "crf_max_unet_val_difference": max_base_difference, "crf_choice_recomputed": chosen, "crf_choice_recorded": selection["selected"],
            "pass": all(h["best_epoch_recorded"] == h["first_argmax_epoch"] for h in histories.values())
            and max_gain_difference < 1e-12 and max_base_difference < 1e-12 and chosen == selection["selected"]}


def chronology():
    """L6: every freeze precedes every test evaluation (recorded timestamps and file mtimes)."""
    parse = lambda s: datetime.fromisoformat(s)
    mtime = lambda p: datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc)
    frozen = read_json(ROOT / "runs/final_experiment.json")
    hybrid = read_json(ROOT / "runs/hybrid_experiment.json")
    selection = read_json(ROOT / "runs/hybrid_selection.json")
    rows, ok = [], True
    for method in METHODS:
        for budget in BUDGETS:
            for seed in SEEDS:
                path = ROOT / f"runs/{method}_{budget}_s{seed}/evaluation_test.json"
                created = parse(read_json(path)["created_at"])
                freeze = parse(hybrid["frozen_at"] if method == "hybrid" else frozen["frozen_at"])
                freeze_file = ROOT / ("runs/hybrid_experiment.json" if method == "hybrid" else "runs/final_experiment.json")
                good = freeze < created and mtime(freeze_file) < mtime(path)
                ok &= good
                rows.append({"run": path.parent.name, "freeze": freeze.isoformat(), "test_evaluated": created.isoformat(), "freeze_before_test": good})
    selection_before = parse(selection["created_at"]) < parse(hybrid["frozen_at"])
    return {"rows": rows, "selection_created": selection["created_at"], "selection_before_hybrid_freeze": selection_before, "pass": ok and selection_before}


def published(metrics):
    """E5: results.json, evaluation files, CSVs and docs/results.md all carry the same numbers."""
    data = read_json(ROOT / "web/data/results.json")
    problems = []
    for run in data["runs"]:
        evaluation = read_json(ROOT / f"runs/{run['id']}/evaluation_test.json")
        if run["metrics"] != evaluation["metrics"]:
            problems.append(f"results.json != evaluation_test.json for {run['id']}")
        if run["metrics"]["confusion_matrix"] != metrics[run["id"]]["confusion"]:
            problems.append(f"results.json != independent recomputation for {run['id']}")
    with (ROOT / "web" / data["downloads"]["aggregate"]).open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if abs(float(row["flooded_miou"]) - metrics[row["run_id"]]["flooded_miou"]) > 1e-12:
                problems.append(f"aggregate.csv differs for {row['run_id']}")
    names = {"rf": "SLIC + RF", "unet": "U-Net", "hybrid": "U-Net + DenseCRF"}
    doc = (ROOT / "docs/results.md").read_text(encoding="utf-8")
    checked = 0
    for item in data["summary"]:
        values = [metrics[f"{item['method']}_{item['budget']}_s{s}"]["flooded_miou"] * 100 for s in SEEDS]
        mean, sd = float(np.mean(values)), float(np.std(values, ddof=1))
        row = re.search(rf"^\| {re.escape(names[item['method']])} \| {item['budget']}% \| \**([\d.]+) ± ([\d.]+)\** \| ([\d.]+) / ([\d.]+) / ([\d.]+) \|", doc, re.M)
        if not row:
            problems.append(f"docs/results.md row missing for {item['method']} {item['budget']}%")
            continue
        shown = [float(v) for v in row.groups()]
        expected = [mean, sd, *values]
        checked += 1
        if any(abs(a - b) > .05 + 1e-9 for a, b in zip(shown, expected)):
            problems.append(f"docs/results.md {item['method']} {item['budget']}%: shows {shown}, recomputed {[round(v, 2) for v in expected]}")
    return {"problems": problems, "doc_rows_checked": checked, "pass": not problems and checked == 6}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(ROOT / "runs"))
    parser.add_argument("--tag", default="frozen")
    parser.add_argument("--controls", action="store_true", help="also run E5 and L1-L7 (frozen runs only)")
    args = parser.parse_args()
    root = Path(args.root)
    started = time.perf_counter()
    crf_params = read_json(ROOT / "runs/hybrid_selection.json")["selected"]
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        report, predictions, masks, ids = evaluate_root(root, args.tag, pool, crf_params)
        comparisons = report["comparisons"]
        identical = lambda prefix: [n for n in comparisons if n.startswith(prefix) and comparisons[n]["differing_pixels"] == 0 and comparisons[n]["confusion_equal"]]
        drift = {f"rf_{b}_s{s}": abs(report["metrics"][f"rf_{b}_s{s}#independent_features"]["flooded_miou"] - report["metrics"][f"rf_{b}_s{s}"]["flooded_miou"]) * 100
                 for b in BUDGETS for s in SEEDS}
        report["E1"] = {"identical_runs": identical("unet_"), "pass": len(identical("unet_")) == 6}
        report["E2"] = {"identical_runs": identical("hybrid_"), "pass": len(identical("hybrid_")) == 6}
        report["E3"] = {"identical_runs": identical("rf_"), "pass": len(identical("rf_")) == 6}
        report["E4"] = {"abs_delta_points": drift, "pass": all(v <= 1.0 for v in drift.values())}
        if args.controls:
            report["E5"] = published(report["metrics"])
            control = controls(predictions, masks, ids, pool)
            real = {n: report["metrics"][n]["flooded_miou"] for n in report["metrics"] if "#" not in n}
            best_trivial = max(v["flooded_miou"] for v in control["L2"]["baselines"].values())
            report["L1"] = {"shuffled_flooded_miou": control["L1"], "pass": all(v < .05 for v in control["L1"].values())}
            report["L2"] = {**control["L2"], "best_trivial_flooded_miou": best_trivial, "weakest_real_model": min(real.values()),
                            "pass": control["L2"]["baselines"]["all_other"]["flooded_miou"] == 0 and min(real.values()) >= 10 * best_trivial}
            report["L3"] = {**control["L3"], "pass": all(v["flooded_miou"] < .10 for v in control["L3"].values())}
            report["L5"] = selection_checks(pool)
            report["L6"] = chronology()
            report["L7"] = {**control["L7"], "pass": not any(control["L7"]["perfect_copies"].values())}
    report["seconds_total"] = time.perf_counter() - started
    guard()
    write_json(f"eval_{args.tag}.json", report)
    print(json.dumps({k: v["pass"] for k, v in report.items() if isinstance(v, dict) and "pass" in v}))


if __name__ == "__main__":
    main()
