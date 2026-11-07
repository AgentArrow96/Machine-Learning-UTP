"""Audited originals, disk-backed preprocessing, and deterministic image budgets."""
from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import math
import re
import time

import numpy as np
from PIL import Image, ImageOps
import psutil

from .common import ProjectError, SPLITS, digest, inside, log, now, read_json, relative, sha256, write_json

ORIGINAL_CLASSES = ["Background", "Building flooded", "Building non-flooded", "Road flooded",
                    "Road non-flooded", "Water", "Tree", "Vehicle", "Pool", "Grass"]
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def mask_array(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        values = np.asarray(image).copy()
        mode = image.mode
    if values.ndim != 2 or not np.issubdtype(values.dtype, np.integer):
        raise ProjectError(f"Expected indexed/integer annotation; found mode {mode} at {path}")
    ids = np.unique(values)
    if np.any((ids < 0) | (ids > 9)):
        raise ProjectError(f"Unknown original label IDs {ids.tolist()} at {path}")
    return values.astype(np.uint8)


def remap(values: np.ndarray) -> np.ndarray:
    if values.ndim != 2 or np.any(values > 9) or np.any(values < 0):
        raise ProjectError("Remapping requires original two-dimensional integer labels 0..9.")
    result = np.zeros(values.shape, dtype=np.uint8)
    result[values == 1] = 1
    result[values == 3] = 2
    return result


def pair_key(path: Path) -> str:
    return re.sub(r"(?:_lab|_label|_mask)$", "", path.stem, flags=re.IGNORECASE).lower()


def discover(raw: Path) -> list[dict]:
    groups = {s: {"images": {}, "masks": {}} for s in SPLITS}
    for path in sorted(raw.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}:
            continue
        if "__MACOSX" in path.parts or path.name.startswith("."):
            continue
        parts = [part.lower() for part in path.relative_to(raw).parts[:-1]]
        split_candidates = {s for s in SPLITS if any(part == s or part.startswith(s + "-") or part.startswith(s + "_") for part in parts)}
        if len(split_candidates) != 1:
            raise ProjectError(f"Cannot identify one published split for {path}")
        split = split_candidates.pop()
        role = "masks" if any("label" in part or "mask" in part for part in parts) else "images"
        key = pair_key(path)
        if key in groups[split][role]:
            raise ProjectError(f"Duplicate {role} ID {key} within {split}")
        groups[split][role][key] = path
    records = []
    for split, group in groups.items():
        if group["images"].keys() != group["masks"].keys():
            missing = sorted(group["images"].keys() ^ group["masks"].keys())
            raise ProjectError(f"Unpaired {split} images/masks: {missing[:20]}")
        for key in sorted(group["images"]):
            records.append({"id": f"{split}_{key}", "image_id": key, "split": split,
                            "image": relative(group["images"][key]), "mask": relative(group["masks"][key])})
    return records


def inspect_record(record: dict) -> dict:
    values = mask_array(inside(record["mask"]))
    with Image.open(inside(record["image"])) as image:
        image.load()
        orientation = image.getexif().get(274, 1)
        if orientation not in (1, 3):
            raise ProjectError(f"Unreviewed EXIF orientation {orientation}: {record['id']}")
        # Official annotations use the displayed orientation. Release has 57 EXIF=3 JPEGs.
        # Visual alignment checked against masks: apply EXIF to RGB, leave annotation indices alone.
        oriented = ImageOps.exif_transpose(image)
        if oriented.size != (values.shape[1], values.shape[0]):
            raise ProjectError(f"Image/mask dimension mismatch: {record['id']}")
        grey = np.asarray(oriented.convert("L").resize((8, 8), Image.Resampling.LANCZOS))
        ahash = int("".join("1" if b else "0" for b in (grey >= grey.mean()).ravel()), 2)
        width, height = oriented.size
    counts = np.bincount(values.ravel(), minlength=10).astype(int).tolist()
    with Image.open(inside(record["mask"])) as mask_image:
        mask_mode = mask_image.mode
    return {**record, "width": width, "height": height, "pair_valid": True,
            "mask_mode": mask_mode, "exif_orientation": orientation,
            "mask_ids": np.flatnonzero(counts).tolist(), "original_counts": counts,
            "image_sha256": sha256(record["image"]), "mask_sha256": sha256(record["mask"]), "ahash": f"{ahash:016x}"}


def audit(cfg: dict, raw: str = "data/raw") -> dict:
    started = time.perf_counter()
    raw_path = inside(raw)
    # Official Dropbox also includes colourised visualisations; only indexed release masks are labels.
    if (raw_path / "FloodNet-Supervised_v1.0").is_dir():
        raw_path = raw_path / "FloodNet-Supervised_v1.0"
    records = discover(raw_path)
    counts = dict(Counter(r["split"] for r in records))
    if counts != cfg["expected_counts"]:
        raise ProjectError(f"Release counts differ: observed {counts}; expected {cfg['expected_counts']}. Resolve the release before preparation.")
    completed = []
    with ThreadPoolExecutor(max_workers=cfg["preprocess_workers"]) as pool:
        for index, record in enumerate(pool.map(inspect_record, records)):
            completed.append(record)
            if index % 100 == 0:
                log(f"Audited {index + 1}/{len(records)} image-mask pairs")
    duplicates = defaultdict(list)
    for record in completed:
        duplicates[record["image_sha256"]].append(record["id"])
    exact = [ids for ids in duplicates.values() if len(ids) > 1]
    distributions = {}
    for split in SPLITS:
        selected = [r for r in completed if r["split"] == split]
        matrix = np.array([r["original_counts"] for r in selected], dtype=np.int64)
        distributions[split] = {"pixel_counts": matrix.sum(0).tolist(), "image_presence": (matrix > 0).sum(0).tolist()}
    manifest = {"schema_version": 1, "created_at": now(), "verified_original_mapping": {"1": 1, "3": 2, "others_0_to_9": 0},
                "counts": counts, "original_classes": ORIGINAL_CLASSES, "distributions": distributions,
                "records": completed, "exact_duplicates": exact,
                "overlap_note": "Scene overlap is verified separately by the overlap stage (keypoint matching).",
                "elapsed_seconds": time.perf_counter() - started}
    manifest["orientation_policy"] = "Apply RGB EXIF orientation (release values 1/3); masks already use annotation/display coordinates. Visually verified."
    manifest["identity"] = digest({"records": completed, "counts": counts, "mapping": manifest["verified_original_mapping"]})
    source_path = inside("data/downloads/source.json")
    if source_path.exists():
        manifest["source"] = read_json(source_path)
    write_json("data/manifests/audit.json", manifest)
    log(f"Audit complete: {counts}; {len(exact)} exact-duplicate groups.")
    return manifest


def resize_pair(rgb: np.ndarray, mask: np.ndarray, size: int = 256):
    h, w = mask.shape
    if rgb.shape != (h, w, 3):
        raise ProjectError("RGB/mask shape mismatch")
    scale = min(size / w, size / h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    left, top = (size - nw) // 2, (size - nh) // 2
    image = np.asarray(Image.fromarray(rgb).resize((nw, nh), Image.Resampling.LANCZOS))
    labels = np.asarray(Image.fromarray(mask).resize((nw, nh), Image.Resampling.NEAREST))
    output_rgb = np.broadcast_to(np.round(MEAN * 255).astype(np.uint8), (size, size, 3)).copy()
    output_mask = np.full((size, size), 255, dtype=np.uint8)
    output_rgb[top:top + nh, left:left + nw] = image
    output_mask[top:top + nh, left:left + nw] = labels
    geometry = {"original_width": w, "original_height": h, "width": nw, "height": nh, "left": left, "top": top}
    return output_rgb, output_mask, geometry


def prepare_record(record: dict, size: int):
    # Hash verification detects changes since audit instead of silently reusing stale originals.
    if sha256(record["image"]) != record["image_sha256"] or sha256(record["mask"]) != record["mask_sha256"]:
        raise ProjectError(f"Original changed since audit: {record['id']}")
    with Image.open(inside(record["image"])) as image:
        rgb = np.asarray(ImageOps.exif_transpose(image).convert("RGB"))
    return resize_pair(rgb, remap(mask_array(inside(record["mask"]))), size)


def subset_ids(records: list[dict], seed: int, budget: int) -> list[str]:
    records = sorted(records, key=lambda r: r["id"])
    if budget == 100:
        return [r["id"] for r in records]
    if budget != 25:
        raise ProjectError("Budgets are 25 or 100")
    n = math.floor(len(records) * 0.25)
    strata = defaultdict(list)
    for record in records:
        strata[(record["processed_counts"][1] > 0, record["processed_counts"][2] > 0)].append(record["id"])
    keys = sorted(strata)
    quotas = {key: math.floor(n * len(strata[key]) / len(records)) for key in keys}
    remainder_order = sorted(keys, key=lambda key: (-(n * len(strata[key]) / len(records) - quotas[key]), key))
    for key in remainder_order[:n - sum(quotas.values())]:
        quotas[key] += 1
    # Ensure any available target class is represented when the budget can support it.
    for target in (0, 1):
        if any(key[target] for key in keys) and not any(key[target] and quotas[key] for key in keys):
            recipient = next(key for key in keys if key[target])
            donors = [key for key in keys if quotas[key] > 1]
            if not donors:
                raise ProjectError("Budget too small to preserve target-class representation")
            donor = max(donors, key=lambda key: quotas[key])
            quotas[donor] -= 1
            quotas[recipient] += 1
    rng = np.random.default_rng(seed)
    selected = []
    for key in keys:
        selected.extend(rng.choice(strata[key], size=quotas[key], replace=False).tolist())
    return sorted(selected)


def preprocessing_key(cfg: dict, audit_manifest: dict) -> str:
    return digest({"audit": audit_manifest["identity"], "size": cfg["size"], "version": cfg["preprocess_version"],
                   "rgb_resize": "pillow_lanczos", "mask_resize": "nearest", "padding": "encoder_mean/255", "mapping": [1, 3],
                   "orientation": "rgb_exif_transpose_masks_in_display_coordinates"})


def prepare(cfg: dict) -> dict:
    audit_manifest = read_json("data/manifests/audit.json")
    key = preprocessing_key(cfg, audit_manifest)
    directory = inside(f"data/processed/{key[:16]}")
    directory.mkdir(parents=True, exist_ok=True)
    progress_path = directory / "progress.json"
    progress = read_json(progress_path) if progress_path.exists() else {"identity": key, "records": {}}
    if progress["identity"] != key:
        raise ProjectError("Preprocessing cache identity mismatch")
    started = time.perf_counter()
    previous_elapsed = progress.get("elapsed_seconds", 0.)
    peak_rss = psutil.Process().memory_info().rss
    size = cfg["size"]
    all_records = []
    for split in SPLITS:
        records = [r for r in audit_manifest["records"] if r["split"] == split]
        paths = [directory / f"{split}_{kind}.npy" for kind in ("rgb", "mask")]
        existed = all(path.exists() for path in paths)
        shapes = [(len(records), size, size, 3), (len(records), size, size)]
        arrays = [np.lib.format.open_memmap(path, mode="r+" if existed else "w+", dtype=np.uint8, shape=shape)
                  for path, shape in zip(paths, shapes)]
        if not existed:
            for record in records:
                progress["records"].pop(record["id"], None)
        pending = [(i, r) for i, r in enumerate(records) if r["id"] not in progress["records"]]
        # map over bounded batches to avoid retaining decoded arrays for the entire dataset.
        with ThreadPoolExecutor(max_workers=cfg["preprocess_workers"]) as pool:
            for start in range(0, len(pending), cfg["preprocess_workers"]):
                batch = pending[start:start + cfg["preprocess_workers"]]
                prepared = pool.map(lambda pair: prepare_record(pair[1], size), batch)
                for (index, record), (rgb, mask, geometry) in zip(batch, prepared):
                    arrays[0][index] = rgb
                    arrays[1][index] = mask
                    processed_counts = np.bincount(mask[mask != 255], minlength=3).astype(int).tolist()
                    progress["records"][record["id"]] = {**record, "index": index, "geometry": geometry,
                                                         "processed_counts": processed_counts}
                for array in arrays:
                    array.flush()
                progress["elapsed_seconds"] = previous_elapsed + time.perf_counter() - started
                write_json(progress_path, progress)
                peak_rss = max(peak_rss, psutil.Process().memory_info().rss)
                if start % 100 == 0:
                    log(f"Prepared {split}: {min(start + len(batch), len(pending))}/{len(pending)} pending images")
        for record in records:
            all_records.append(progress["records"][record["id"]])
        del arrays
    train = [r for r in all_records if r["split"] == "train"]
    budgets = {}
    for budget in (25, 100):
        # Subset seed is independent of training seeds: every replicate uses the same image IDs.
        ids = subset_ids(train, cfg["subset_seed"], budget)
        item = {"budget": budget, "seed": cfg["subset_seed"], "ids": ids, "processed_identity": key}
        item["identity"] = digest(item)
        write_json(f"data/manifests/budget_{budget}.json", item)
        budgets[str(budget)] = item
    manifest = {"schema_version": 1, "identity": key, "audit_identity": audit_manifest["identity"], "complete": True,
                "directory": relative(directory), "size": size, "records": all_records, "budgets": budgets,
                "created_at": now(), "elapsed_seconds": previous_elapsed + time.perf_counter() - started,
                "invocation_seconds": time.perf_counter() - started, "peak_host_rss_bytes": peak_rss,
                "original_target_pixels": [sum(r["original_counts"][c] for r in all_records) for c in (1, 3)],
                "processed_target_pixels": [sum(r["processed_counts"][c] for r in all_records) for c in (1, 2)],
                "targets_lost_images": {str(c): [r["id"] for r in all_records if r["original_counts"][orig] > 0 and r["processed_counts"][c] == 0]
                                       for c, orig in ((1, 1), (2, 3))}}
    write_json("data/manifests/processed.json", manifest)
    log("Processed cache and nested budgets complete.")
    return manifest


class Store:
    def __init__(self, cfg: dict | None = None):
        self.manifest = read_json("data/manifests/processed.json")
        if not self.manifest.get("complete"):
            raise ProjectError("Preparation is incomplete")
        if cfg is not None and self.manifest["identity"] != preprocessing_key(cfg, read_json("data/manifests/audit.json")):
            raise ProjectError("Stale preprocessing cache: run prepare for this audit/configuration")
        self.records = self.manifest["records"]
        self.by_id = {r["id"]: r for r in self.records}
        self.arrays = {}

    def __getstate__(self):
        state = self.__dict__.copy()
        state["arrays"] = {}  # Windows workers reopen memmaps; never pickle image arrays.
        return state

    def get(self, record: dict):
        split = record["split"]
        if split not in self.arrays:
            directory = inside(self.manifest["directory"])
            self.arrays[split] = tuple(np.load(directory / f"{split}_{kind}.npy", mmap_mode="r") for kind in ("rgb", "mask"))
        rgb, mask = self.arrays[split]
        return rgb[record["index"]].copy(), mask[record["index"]].copy()

    def select(self, split: str, budget: int | None = None):
        records = [r for r in self.records if r["split"] == split]
        if budget is not None:
            if split != "train":
                raise ProjectError("Training budgets cannot be applied to validation/test")
            allowed = set(self.manifest["budgets"][str(budget)]["ids"])
            records = [r for r in records if r["id"] in allowed]
        return records


def crop(array, geometry):
    g = geometry
    return array[g["top"]:g["top"] + g["height"], g["left"]:g["left"] + g["width"]]
