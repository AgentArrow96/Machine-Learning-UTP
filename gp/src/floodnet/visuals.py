"""Display-only exports for the results app: sharper test photos and a step-by-step model walkthrough.

Both read the frozen runs and the raw release without changing them. The walkthrough recomputes each
model's intermediate steps for a few representative test images and fails unless every final mask
equals the frozen test prediction pixel for pixel.
"""
from __future__ import annotations

import re
import shutil
import zipfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image, ImageOps

from .common import (
    CLASSES,
    COLOURS,
    ProjectError,
    digest,
    inside,
    log,
    now,
    read_json,
    sha256,
    write_json,
)
from .data import (
    MEAN,
    ORIGINAL_CLASSES,
    STD,
    Store,
    crop,
    mask_array,
    remap,
    resize_pair,
)

HIRES_SIZE = 1024
WALKTHROUGH_VERSION = 1
PALETTE_FILE = "data/raw/ColorMasks-FloodNetv1.0/ColorPalette-Values.xlsx"


# ---------- Sharper photos (git-ignored) ----------

def oriented_photo(record):
    """The raw photo exactly as the pipeline reads it: hash-checked, then EXIF-oriented."""
    if sha256(record["image"]) != record["image_sha256"]:
        raise ProjectError(f"Original changed since audit: {record['id']}")
    with Image.open(inside(record["image"])) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def fit_long_side(image, size):
    scale = size / max(image.size)
    return image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))), Image.Resampling.LANCZOS)


def check_aspect(size, geometry):
    """The processed crop and the sharp photo must show the same frame (rounding of both sides allowed)."""
    width, height = size
    if abs(geometry["height"] - height * geometry["width"] / width) > 0.75:
        raise ProjectError(f"Sharp photo {width}x{height} does not match processed geometry {geometry}")


def hires_job(job):
    record, path = job
    image = fit_long_side(oriented_photo(record), HIRES_SIZE)
    check_aspect(image.size, record["geometry"])
    image.save(path, "WEBP", quality=82)
    return record["id"], list(image.size)


def export_hires(cfg):
    store = Store(cfg)
    bundle = read_json("web/data/results.json")
    ids = [r["id"] for r in store.select("test")]
    ids += [e["train_id"] for e in ((bundle.get("dataset") or {}).get("overlap") or {}).get("examples", [])]
    directory = inside("web/data/hires")
    directory.mkdir(parents=True, exist_ok=True)
    jobs = [(store.by_id[i], directory / f"{i}.webp") for i in dict.fromkeys(ids)]
    with ThreadPoolExecutor(8) as pool:
        sizes = dict(pool.map(hires_job, jobs))
    index = {"schema_version": 1, "created_at": now(), "processed_identity": store.manifest["identity"], "long_side": HIRES_SIZE,
             "images": {i: {"url": f"data/hires/{i}.webp", "size": sizes[i]} for i in sizes}}
    write_json(directory / "index.json", index)
    log(f"Sharp photos: {len(sizes)} images at {HIRES_SIZE} px in web/data/hires (git-ignored)")
    return index


# ---------- Walkthrough helpers ----------

def official_palette():
    """Name -> RGB from the release's ColorPalette-Values.xlsx (read as XML)."""
    with zipfile.ZipFile(inside(PALETTE_FILE)) as archive:
        strings = re.findall(r"<t[^>]*>([^<]*)</t>", archive.read("xl/sharedStrings.xml").decode("utf-8"))
        sheet = archive.read("xl/worksheets/sheet1.xml").decode("utf-8")
    rows = {}
    for ref, index in re.findall(r'<c r="([A-Z]+\d+)"[^>]*t="s"[^>]*><v>(\d+)</v>', sheet):
        column, row = re.match(r"([A-Z]+)(\d+)", ref).groups()
        rows.setdefault(row, {})[column] = strings[int(index)]
    return {row["G"]: [int(v) for v in re.findall(r"\d+", row["H"])] for row in rows.values() if row.get("G") and row.get("H")}


def original_colours():
    palette = official_palette()
    names = [name.lower().replace(" ", "-") for name in ORIGINAL_CLASSES]
    missing = [n for n in names if n not in palette]
    if missing:
        raise ProjectError(f"Official palette lacks {missing}")
    return [palette[n] for n in names]


def pca_rgb(features):
    """(C, H, W) activations -> (H, W, 3) uint8: the first three principal components as colour.

    Component signs are fixed (largest loading positive) and each channel is stretched between its
    1st and 99th percentile, so the same activations always give the same picture."""
    channels, height, width = features.shape
    x = features.reshape(channels, -1).T.astype(np.float64)
    x = x - x.mean(0)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    components = vt[:3]
    components *= np.sign(components[np.arange(len(components)), np.abs(components).argmax(1)])[:, None]
    projected = x @ components.T
    if projected.shape[1] < 3:
        projected = np.pad(projected, ((0, 0), (0, 3 - projected.shape[1])))
    low, high = np.percentile(projected, 1, axis=0), np.percentile(projected, 99, axis=0)
    scaled = (projected - low) / np.where(high > low, high - low, 1)
    return np.round(np.clip(scaled, 0, 1) * 255).astype(np.uint8).reshape(height, width, 3)


def scaled_window(geometry, size, resolution):
    """The valid (unpadded) rectangle of a 256-px geometry at a coarser feature resolution."""
    s = resolution / size
    g = geometry
    return {"top": int(np.floor(g["top"] * s)), "left": int(np.floor(g["left"] * s)),
            "height": int(np.ceil((g["top"] + g["height"]) * s)) - int(np.floor(g["top"] * s)),
            "width": int(np.ceil((g["left"] + g["width"]) * s)) - int(np.floor(g["left"] * s))}


def encode_regions(regions):
    """Region index (-1 = padding) -> RGB PNG with index+1 in R (low byte) and G (high byte)."""
    value = (np.asarray(regions, dtype=np.int64) + 1)
    if value.max() > 65535:
        raise ProjectError("Too many regions to encode")
    out = np.zeros((*value.shape, 3), dtype=np.uint8)
    out[..., 0], out[..., 1] = value & 255, value >> 8
    return out


def decode_regions(encoded):
    encoded = np.asarray(encoded, dtype=np.int64)
    return encoded[..., 0] + (encoded[..., 1] << 8) - 1


def crf_iterations(probs, rgb, params):
    """Label map after each mean-field step (index 0 = before any step); same model as hybrid.refine_valid."""
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_softmax
    classes, height, width = probs.shape
    crf = dcrf.DenseCRF2D(width, height, classes)
    crf.setUnaryEnergy(np.ascontiguousarray(unary_from_softmax(np.ascontiguousarray(probs, dtype=np.float32))))
    crf.addPairwiseGaussian(sxy=params["gaussian"]["sxy"], compat=params["gaussian"]["compat"])
    crf.addPairwiseBilateral(sxy=params["bilateral"]["sxy"], srgb=params["bilateral"]["srgb"],
                             rgbim=np.ascontiguousarray(rgb, dtype=np.uint8), compat=params["bilateral"]["compat"])
    q, scratch, scratch2 = crf.startInference()
    steps = [np.array(q, dtype=np.float32).reshape(classes, height, width).argmax(0).astype(np.uint8)]
    for _ in range(params["iterations"]):
        crf.stepInference(q, scratch, scratch2)
        steps.append(np.array(q, dtype=np.float32).reshape(classes, height, width).argmax(0).astype(np.uint8))
    return steps


def image_scores(truth, prediction):
    from .export import flat_metrics
    from .metrics import confusion, score
    flat = flat_metrics(score(confusion(truth, prediction)))
    return {k: flat[k] for k in ("flooded_miou", "building_iou", "road_iou")}


def frozen_prediction(run_id, image_id):
    result = read_json(f"runs/{run_id}/evaluation_test.json")
    if sha256(result["predictions"]) != result["predictions_sha256"]:
        raise ProjectError(f"Frozen predictions of {run_id} changed")
    item = next(i for i in result["per_image"] if i["id"] == image_id)
    return np.load(inside(result["predictions"]), mmap_mode="r")[item["prediction_index"]].copy()


def match(prediction, frozen, geometry):
    """Compare on the valid rectangle (evaluation zeroes padding before saving)."""
    a, b = crop(prediction, geometry), crop(frozen, geometry)
    differing = int(np.count_nonzero(a != b))
    if differing:
        raise ProjectError(f"Walkthrough differs from the frozen test prediction in {differing} pixels")
    return {"pixels": int(a.size), "differing": differing, "matches_frozen_prediction": True}


# ---------- Walkthrough export ----------

class Writer:
    def __init__(self, web, directory):
        self.web, self.directory = web, directory

    def png(self, name, array):
        path = self.directory / f"{name}.png"
        Image.fromarray(np.ascontiguousarray(array)).save(path, optimize=True)
        return path.relative_to(self.web).as_posix()

    def webp(self, name, image):
        path = self.directory / f"{name}.webp"
        image.save(path, "WEBP", quality=85)
        return path.relative_to(self.web).as_posix()


def shared_steps(cfg, record, rgb, truth, write):
    from skimage.color import rgb2gray
    from skimage.feature import local_binary_pattern
    from skimage.segmentation import find_boundaries

    from .features import extract_regions
    g, key = record["geometry"], record["id"]
    original = mask_array(inside(record["mask"]))
    if sha256(record["mask"]) != record["mask_sha256"]:
        raise ProjectError(f"Original mask changed since audit: {key}")
    full = oriented_photo(record)
    # The preparation shown must be the real one: redo it from the raw pair and compare with the cache.
    prepared_rgb, prepared_truth, prepared_geometry = resize_pair(np.asarray(full), remap(original), cfg["size"])
    if prepared_geometry != g or not np.array_equal(prepared_rgb, rgb) or not np.array_equal(prepared_truth, truth):
        raise ProjectError(f"Re-running preparation for {key} does not reproduce the cached arrays")
    photo = fit_long_side(full, 768)
    check_aspect(photo.size, g)
    colours = np.array(original_colours(), dtype=np.uint8)
    preview = Image.fromarray(colours[original]).resize(photo.size, Image.Resampling.NEAREST)
    content, truth_crop = crop(rgb, g), crop(truth, g)
    x, y, regions = extract_regions(rgb, g, cfg["slic"], truth)
    region_crop = crop(regions, g)
    area = np.bincount(region_crop.ravel(), minlength=len(x))
    boundaries = find_boundaries(region_crop, mode="inner")
    outlined = content.copy()
    outlined[boundaries] = [255, 255, 255]
    mean_colour = np.round(x[:, :3] * 255).astype(np.uint8)[region_crop]
    lbp = local_binary_pattern(np.round(rgb2gray(content) * 255).astype(np.uint8), P=8, R=1, method="uniform")
    lbp_image = np.round(lbp / 9 * 255).astype(np.uint8)
    original_counts = np.bincount(original.ravel(), minlength=10)
    shared = {"photo": write.webp(f"{key}_photo", photo), "photo_size": list(photo.size),
              "original_mask": write.png(f"{key}_original_mask", np.asarray(preview)),
              "original_size": [int(original.shape[1]), int(original.shape[0])],
              "original_shares": (original_counts / original_counts.sum()).round(5).tolist(),
              "input": write.png(f"{key}_input", rgb), "rgb": write.png(f"{key}_rgb", content),
              "truth": write.png(f"{key}_truth", truth_crop),
              "truth_counts": np.bincount(truth_crop.ravel(), minlength=3).tolist(), "preparation_reproduced": True,
              "slic": {"boundaries": write.png(f"{key}_slic", outlined), "mean_colour": write.png(f"{key}_mean_colour", mean_colour),
                       "lbp": write.png(f"{key}_lbp", lbp_image), "regions": write.png(f"{key}_regions", encode_regions(region_crop)),
                       "region_count": len(x), "settings": cfg["slic"], "area": area.tolist(),
                       "features": np.round(x, 4).tolist(), "truth_label": y.tolist()}}
    if not np.array_equal(decode_regions(encode_regions(region_crop)), region_crop):
        raise ProjectError("Region map encoding failed")
    return shared, x


def rf_steps(rf, run_id, record, rgb, truth, x, write, budget):
    from .features import extract_regions
    g, key = record["geometry"], record["id"]
    model = rf.model
    if list(model.classes_) != [0, 1, 2]:
        raise ProjectError("Unexpected forest classes")
    probabilities = model.predict_proba(x)
    votes = np.zeros((len(x), 3), dtype=np.int64)
    for tree in model.estimators_:
        votes[np.arange(len(x)), tree.predict(x).astype(np.int64)] += 1
    _, _, regions = extract_regions(rgb, g, rf.cfg["slic"])
    prediction = rf.predict(rgb, g)
    if not np.array_equal(prediction[regions >= 0], probabilities.argmax(1)[regions[regions >= 0]]):
        raise ProjectError("Forest probabilities do not explain the prediction")
    return {"run_id": run_id, "trees": len(model.estimators_), "probabilities": np.round(probabilities, 4).tolist(),
            "votes": votes.tolist(), "mask": write.png(f"{key}_rf_{budget}", crop(prediction, g)),
            "scores": image_scores(crop(truth, g), crop(prediction, g)),
            "check": match(prediction, frozen_prediction(run_id, key), g)}, prediction


def unet_steps(unet, run_id, record, rgb, truth, write, budget):
    torch = unet.torch
    g, key = record["geometry"], record["id"]
    captured = {}
    hooks = [unet.model.encoder.register_forward_hook(lambda m, i, o: captured.__setitem__("encoder", [t.float().cpu().numpy()[0] for t in o]))]
    for index, block in enumerate(unet.model.decoder.blocks):
        hooks.append(block.register_forward_hook(lambda m, i, o, index=index: captured.__setitem__(f"decoder{index}", o.float().cpu().numpy()[0])))
    try:
        logits = unet.logits(rgb)
    finally:
        for hook in hooks:
            hook.remove()
    prediction = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
    with torch.inference_mode():
        probs = torch.softmax(logits.float(), 1)[0].cpu().numpy()

    def stage(name, features):
        window = scaled_window(g, rgb.shape[0], features.shape[1])
        # PCA over the unpadded window only, so padding cannot dominate the components.
        image = pca_rgb(features[:, window["top"]:window["top"] + window["height"], window["left"]:window["left"] + window["width"]])
        return {"url": write.png(f"{key}_unet_{budget}_{name}", image), "channels": int(features.shape[0]),
                "resolution": int(features.shape[1])}
    encoder = [stage(f"enc{i}", f) for i, f in enumerate(captured["encoder"][1:], 1)]
    decoder = [stage(f"dec{i}", captured[f"decoder{i}"]) for i in range(len(unet.model.decoder.blocks))]
    probs_crop = crop(np.moveaxis(probs, 0, -1), g)
    return {"run_id": run_id, "encoder": encoder, "decoder": decoder,
            "probabilities": write.png(f"{key}_unet_{budget}_probs", np.round(probs_crop * 255).astype(np.uint8)),
            "mask": write.png(f"{key}_unet_{budget}", crop(prediction, g)),
            "scores": image_scores(crop(truth, g), crop(prediction, g)),
            "check": match(prediction, frozen_prediction(run_id, key), g)}, prediction


def hybrid_steps(hybrid, run_id, record, rgb, truth, unet_prediction, write, budget):
    from .hybrid import refine_valid, valid_window
    g, key = record["geometry"], record["id"]
    probs = hybrid.unet.probabilities(rgb)
    window, content = valid_window(probs, g), crop(rgb, g)
    steps = crf_iterations(window, content, hybrid.params)
    if not np.array_equal(steps[-1], refine_valid(window, content, hybrid.params)):
        raise ProjectError("Stepped mean-field inference differs from DenseCRF inference")
    prediction = probs.argmax(0).astype(np.uint8)
    valid_window(prediction, g)[...] = steps[-1]
    base = crop(unet_prediction, g)
    truth_crop = crop(truth, g)
    final = steps[-1]
    return {"run_id": run_id, "params": hybrid.params,
            "iterations": [write.png(f"{key}_crf_{budget}_it{i}", s) for i, s in enumerate(steps)],
            "changed_from_unet": [int(np.count_nonzero(s != base)) for s in steps],
            "changed_from_previous": [0] + [int(np.count_nonzero(a != b)) for a, b in zip(steps[1:], steps[:-1])],
            "fixed": int(np.count_nonzero((base != truth_crop) & (final == truth_crop))),
            "broken": int(np.count_nonzero((base == truth_crop) & (final != truth_crop))),
            "mask": write.png(f"{key}_hybrid_{budget}", final),
            "scores": image_scores(truth_crop, final),
            "check": match(prediction, frozen_prediction(run_id, key), g)}


def export_walkthrough(cfg, rf_variant=None):
    """`rf_variant` (e.g. "undersample/uniform") replays that study's forests instead of the frozen originals."""
    from .evaluation import load_engine
    from .export import check_url
    from .features import FEATURE_NAMES
    bundle = read_json("web/data/results.json")
    if bundle["status"] != "complete":
        raise ProjectError("Export the complete results first (export-web)")
    store = Store(cfg)
    seed = bundle["protocol"]["explorer_seed"]
    examples = bundle["representatives"]
    web = inside("web")
    parts = {"version": WALKTHROUGH_VERSION, "experiment": bundle["experiment_identity"],
             "hybrid": bundle["hybrid_experiment_identity"], "examples": examples, "seed": seed}
    identity = digest(parts | {"rf_variant": rf_variant} if rf_variant else parts)[:16]
    directory = web / f"data/walkthrough/{identity}"
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    write = Writer(web, directory)
    engines = {}
    importances = {}
    for budget in (25, 100):
        for method in ("rf", "unet", "hybrid"):
            run_id = f"{rf_variant}/{method}_{budget}_s{seed}" if method == "rf" and rf_variant else f"{method}_{budget}_s{seed}"
            run_directory = inside(f"runs/{run_id}")
            engines[method, budget] = (run_id, load_engine(run_directory, read_json(run_directory / "run.json")))
        importances[str(budget)] = np.round(engines["rf", budget][1].model.feature_importances_, 5).tolist()
    output = []
    for key in examples:
        record = store.by_id[key]
        rgb, truth = store.get(record)
        log(f"Walkthrough {key}")
        shared, x = shared_steps(cfg, record, rgb, truth, write)
        item = {"id": key, "image_id": record["image_id"], "geometry": record["geometry"],
                "presence": [bool(record["processed_counts"][c]) for c in (1, 2)], "shared": shared, "budgets": {}}
        for budget in (25, 100):
            rf_id, rf = engines["rf", budget]
            unet_id, unet = engines["unet", budget]
            hybrid_id, hybrid = engines["hybrid", budget]
            rf_part, _ = rf_steps(rf, rf_id, record, rgb, truth, x, write, budget)
            unet_part, unet_prediction = unet_steps(unet, unet_id, record, rgb, truth, write, budget)
            hybrid_part = hybrid_steps(hybrid, hybrid_id, record, rgb, truth, unet_prediction, write, budget)
            item["budgets"][str(budget)] = {"rf": rf_part, "unet": unet_part, "hybrid": hybrid_part}
        output.append(item)
    hybrid_params = engines["hybrid", 100][1].params
    result = {"schema_version": WALKTHROUGH_VERSION, "created_at": now(), "identity": identity, "seed": seed, "budgets": [25, 100],
              "experiment_identity": bundle["experiment_identity"], "hybrid_experiment_identity": bundle["hybrid_experiment_identity"],
              "rf_variant": rf_variant,
              "classes": [{"id": i, "name": n, "colour": COLOURS[i]} for i, n in enumerate(CLASSES)],
              "original_classes": [{"id": i, "name": n, "colour": c} for i, (n, c) in enumerate(zip(ORIGINAL_CLASSES, original_colours()))],
              "normalisation": {"mean": MEAN.round(3).tolist(), "std": STD.round(3).tolist()},
              "feature_names": FEATURE_NAMES, "feature_importances": importances, "crf": hybrid_params,
              "selection": "The six representative test scenes, chosen by target presence and pixel coverage, never by prediction quality.",
              "examples": output}
    urls = [u for e in output for u in walk_urls(e)]
    for url in urls:
        check_url(url, web)
    encoded = str(result)
    if "C:\\" in encoded or ":/" in encoded.replace("https://", ""):
        raise ProjectError("Absolute paths leaked into the walkthrough")
    write_json(web / "data/walkthrough.json", result)
    for old in (web / "data/walkthrough").iterdir():
        if old.is_dir() and old.name != identity:
            shutil.rmtree(old)
    checks = sum(1 for e in output for b in e["budgets"].values() for m in b.values() if m["check"]["matches_frozen_prediction"])
    log(f"Walkthrough: {len(output)} images, {len(urls)} assets; {checks} final masks equal the frozen test predictions")
    return result


def walk_urls(value, prefix="data/walkthrough/"):
    if isinstance(value, dict):
        for v in value.values():
            yield from walk_urls(v, prefix)
    elif isinstance(value, list):
        for v in value:
            yield from walk_urls(v, prefix)
    elif isinstance(value, str) and value.startswith(prefix):
        yield value


# ---------- Preprocessing page ----------

def strata_counts(records, ids):
    """Images per target-presence stratum (the strata the 25% subset is drawn from)."""
    by_id = {r["id"]: r for r in records}
    out = {"both": 0, "building": 0, "road": 0, "neither": 0}
    for i in ids:
        building, road = (by_id[i]["processed_counts"][c] > 0 for c in (1, 2))
        out["both" if building and road else "building" if building else "road" if road else "neither"] += 1
    return out


def class_presence(records):
    """How many annotations contain each original class ID 0..9."""
    counts = [0] * 10
    for record in records:
        for class_id in record["mask_ids"]:
            counts[class_id] += 1
    return counts


def edge_agreement(rgb, mask):
    """Mean image gradient on mask boundaries relative to the whole image (higher = better aligned)."""
    grey = rgb.astype(np.float64).mean(2)
    gradient = np.abs(np.diff(grey, axis=1))[:-1, :] + np.abs(np.diff(grey, axis=0))[:, :-1]
    boundary = (mask[:, 1:] != mask[:, :-1])[:-1, :] | (mask[1:, :] != mask[:-1, :])[:, :-1]
    return float(gradient[boundary].mean() / gradient.mean()) if boundary.any() else None


def augmentation_draws(rgb, mask, seed, count, strength):
    """Replay the U-Net training augmentation for epochs 0..count-1 of one image, with the drawn parameters."""
    from .neural import augment
    draws = []
    for epoch in range(count):
        rng = np.random.default_rng(np.random.SeedSequence([seed, epoch, 0]))
        params = {"flip_lr": bool(rng.random() < .5), "flip_ud": bool(rng.random() < .5), "quarter_turns": int(rng.integers(4)),
                  "contrast": float(rng.uniform(1 - strength, 1 + strength)), "brightness": float(rng.uniform(-strength, strength))}
        image, labels = augment(rgb, mask, seed, epoch, 0, strength)
        draws.append((image, labels, params))
    return draws


def overlay(rgb, labels, colours, alpha=.5, background=.08):
    """Blend class colours over a photo; class 0 is only lightly tinted, 255 (padding) is left as is."""
    rgb = np.asarray(rgb, dtype=np.float32)
    palette = np.array(colours, dtype=np.float32)
    safe = np.where(labels == 255, 0, labels)
    weight = np.where(labels == 255, 0, np.where(safe == 0, background, alpha))[..., None]
    return np.round(rgb * (1 - weight) + palette[safe] * weight).astype(np.uint8)


def hatch_padding(rgb, geometry):
    out = rgb.copy()
    g = geometry
    yy, xx = np.mgrid[:out.shape[0], :out.shape[1]]
    pad = (yy < g["top"]) | (yy >= g["top"] + g["height"]) | (xx < g["left"]) | (xx >= g["left"] + g["width"])
    out[pad & ((xx + yy) % 8 < 3)] = 148
    return out


def class_rgb():
    return [[int(c[i:i + 2], 16) for i in (1, 3, 5)] for c in COLOURS]


def export_preprocessing(cfg):
    """Every preparation step with its real numbers, plus a few deterministic example images."""
    store = Store(cfg)
    audit = read_json("data/manifests/audit.json")
    processed = store.manifest
    source = read_json("data/downloads/source.json")
    web = inside("web")
    identity = digest({"audit": audit["identity"], "processed": processed["identity"], "version": 1})[:16]
    directory = web / f"data/preprocessing/{identity}"
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    write = Writer(web, directory)
    records = audit["records"]
    log("Preprocessing: re-hashing the downloaded archive")
    archive_ok = sha256(source["archive"]) == source["archive_sha256"]
    if not archive_ok:
        raise ProjectError("Archive hash differs from source.json")
    steps = {}
    steps["acquire"] = {"archive_bytes": source["archive_bytes"], "archive_sha256": source["archive_sha256"], "archive_verified": archive_ok,
                        "files_hashed": 2 * len(records), "counts": audit["counts"], "expected_counts": cfg["expected_counts"],
                        "licence": source["licence"], "repository": source["repository"]}
    modes = sorted({r["mask_mode"] for r in records})
    steps["pairing"] = {"rule": "<id>.jpg ↔ <id>_lab.png within the same split folder", "pairs": len(records), "unpaired": 0,
                        "mask_modes": modes, "ids_in_range": all(0 <= i <= 9 for r in records for i in r["mask_ids"]),
                        "size_mismatches": 0, "class_presence": class_presence(records), "original_classes": ORIGINAL_CLASSES}
    near = sorted(audit["near_duplicate_candidates"], key=lambda c: (c["ahash_distance"], c["ids"]))
    examples, used = [], set()
    # The three closest pairs that share no image, so three different scenes are shown.
    chosen = []
    for candidate in near:
        if len(chosen) == 3:
            break
        if not used & set(candidate["ids"]):
            chosen.append(candidate)
            used |= set(candidate["ids"])
    for candidate in chosen:
        urls = []
        for image_id in candidate["ids"]:
            record = store.by_id[image_id]
            urls.append(write.webp(f"near_{image_id}", Image.fromarray(crop(store.get(record)[0], record["geometry"]))))
        examples.append({"ids": candidate["ids"], "distance": candidate["ahash_distance"], "cross_split": candidate["cross_split"], "urls": urls})
    steps["duplicates"] = {"exact_groups": len(audit["exact_duplicates"]), "near_candidates": len(near),
                           "near_cross_split": sum(c["cross_split"] for c in near), "near_note": audit["near_duplicate_note"],
                           "examples": examples, "removed": 0}
    rotated = sorted((r for r in records if r["exif_orientation"] != 1), key=lambda r: r["id"])
    pick = next((r for r in rotated if r["original_counts"][1] or r["original_counts"][3]), rotated[0])
    colours10 = original_colours()
    mask = mask_array(inside(pick["mask"]))
    if sha256(pick["image"]) != pick["image_sha256"] or sha256(pick["mask"]) != pick["mask_sha256"]:
        raise ProjectError(f"Original changed since audit: {pick['id']}")
    with Image.open(inside(pick["image"])) as image:
        raw = image.convert("RGB").resize((800, 600), Image.Resampling.BILINEAR)
        fixed = ImageOps.exif_transpose(image).convert("RGB").resize((800, 600), Image.Resampling.BILINEAR)
    small_mask = np.asarray(Image.fromarray(mask).resize((800, 600), Image.Resampling.NEAREST))
    steps["orientation"] = {"rotated": len(rotated), "total": len(records), "policy": audit["orientation_policy"],
                            "example": {"id": pick["id"], "exif_orientation": pick["exif_orientation"],
                                        "raw": write.webp("exif_raw", Image.fromarray(overlay(raw, small_mask, colours10, .45, .45))),
                                        "fixed": write.webp("exif_fixed", Image.fromarray(overlay(fixed, small_mask, colours10, .45, .45))),
                                        "agreement_raw": edge_agreement(np.asarray(raw), small_mask),
                                        "agreement_fixed": edge_agreement(np.asarray(fixed), small_mask)}}
    sizes = []
    for size in sorted({(r["width"], r["height"]) for r in records}):
        group = sorted((r for r in processed["records"] if (r["geometry"]["original_width"], r["geometry"]["original_height"]) == size), key=lambda r: r["id"])
        example = group[0]
        rgb, _ = store.get(example)
        sizes.append({"size": list(size), "count": len(group), "geometry": example["geometry"], "example_id": example["id"],
                      "padded": write.png(f"padded_{size[0]}x{size[1]}", hatch_padding(rgb, example["geometry"]))})
    steps["resize"] = {"size": cfg["size"], "sizes": sizes, "rgb_resampling": "Lanczos", "mask_resampling": "nearest", "padding_label": 255}
    train_original = audit["distributions"]["train"]["pixel_counts"]
    train_processed = np.array([r["processed_counts"] for r in store.select("train")]).sum(0).tolist()
    steps["remap"] = {"mapping": audit["verified_original_mapping"], "original_train_pixels": train_original,
                      "processed_train_pixels": train_processed}
    original_total = sum(r["width"] * r["height"] for r in records)
    processed_total = int(sum(sum(r["processed_counts"]) for r in processed["records"]))
    steps["resolution"] = {"original_target_pixels": processed["original_target_pixels"], "processed_target_pixels": processed["processed_target_pixels"],
                           "original_total": original_total, "processed_total": processed_total,
                           "targets_lost_images": {k: len(v) for k, v in processed["targets_lost_images"].items()}}
    cache = inside(processed["directory"])
    steps["cache"] = {"identity": processed["identity"], "seconds": processed["elapsed_seconds"],
                      "arrays": {f"{s}_{k}": list(np.load(cache / f"{s}_{k}.npy", mmap_mode="r").shape) for s in ("train", "val", "test") for k in ("rgb", "mask")},
                      "rederived_by_audit": "Audit D7 re-derived all 2,343 arrays from the raw files with 0 differing pixels."}
    train_records = store.select("train")
    steps["budgets"] = {"seed": cfg["subset_seed"], "counts": {k: len(v["ids"]) for k, v in processed["budgets"].items()},
                        "strata": {k: strata_counts(train_records, v["ids"]) for k, v in processed["budgets"].items()},
                        "validation": strata_counts(store.select("val"), [r["id"] for r in store.select("val")]),
                        "test": strata_counts(store.select("test"), [r["id"] for r in store.select("test")])}
    budget25 = set(processed["budgets"]["25"]["ids"])
    sample = next(r for r in sorted(train_records, key=lambda r: r["id"]) if r["id"] in budget25 and r["processed_counts"][1] and r["processed_counts"][2])
    rgb, labels = store.get(sample)
    draws = []
    for i, (image, mask_, params) in enumerate(augmentation_draws(rgb, labels, cfg["train_seeds"][0], 8, cfg["unet"]["brightness_contrast"])):
        draws.append({**params, "epoch": i + 1, "image": write.png(f"augment_{i}", image), "overlay": write.png(f"augment_{i}_mask", overlay(image, mask_, class_rgb()))})
    steps["augmentation"] = {"example_id": sample["id"], "seed": cfg["train_seeds"][0], "strength": cfg["unet"]["brightness_contrast"],
                             "original": write.png("augment_original", rgb), "draws": draws}
    valid = crop(rgb, sample["geometry"]).astype(np.float32) / 255
    normalised = (valid - MEAN) / STD
    steps["normalisation"] = {"mean": MEAN.round(3).tolist(), "std": STD.round(3).tolist(),
                              "example_range": [float(normalised.min()), float(normalised.max())],
                              "theoretical_range": [float(((0 - MEAN) / STD).min()), float(((1 - MEAN) / STD).max())]}
    result = {"schema_version": 1, "created_at": now(), "identity": identity, "audit_identity": audit["identity"],
              "processed_identity": processed["identity"], "original_classes": [{"id": i, "name": n, "colour": c} for i, (n, c) in enumerate(zip(ORIGINAL_CLASSES, colours10))],
              "classes": [{"id": i, "name": n, "colour": COLOURS[i]} for i, n in enumerate(CLASSES)], "steps": steps}
    from .export import check_url
    for url in walk_urls(result, "data/preprocessing/"):
        check_url(url, web)
    if "C:\\" in str(result):
        raise ProjectError("Absolute paths leaked into the preprocessing export")
    write_json(web / "data/preprocessing.json", result)
    for old in (web / "data/preprocessing").iterdir():
        if old.is_dir() and old.name != identity:
            shutil.rmtree(old)
    log(f"Preprocessing export: {len(list(walk_urls(result, 'data/preprocessing/')))} assets")
    return result
