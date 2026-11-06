"""Independent re-implementation of the documented data protocol and scoring.

Written from docs/experiment_protocol.md, not from src/floodnet. Never imports `floodnet`.
Scoring uses scikit-learn's confusion_matrix as the reference implementation.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/FloodNet-Supervised_v1.0"
OUT = ROOT / "output/audit"
DERIVED = OUT / "derived"
SPLITS = ("train", "val", "test")
SIZE = 256
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
PAD_RGB = np.round(MEAN * 255).astype(np.uint8)  # protocol: RGB padding uses the ImageNet mean
IGNORE = 255
FLOODED_BUILDING, FLOODED_ROAD = 1, 3  # release note: 'Building-flooded':1, 'Road-flooded':3


def guard():
    """Fail loudly if the project package leaked into an independent script."""
    leaked = [name for name in sys.modules if name == "floodnet" or name.startswith("floodnet.")]
    assert not leaked, f"independence violated: {leaked}"


def discover():
    """Pair images and masks straight from the release folders: {split: [(id, image, mask)]}."""
    pairs = {}
    for split in SPLITS:
        images = {p.stem: p for p in (RAW / split / f"{split}-org-img").glob("*.jpg")}
        masks = {p.stem.removesuffix("_lab"): p for p in (RAW / split / f"{split}-label-img").glob("*_lab.png")}
        pairs[split] = {"images": images, "masks": masks}
    return pairs


def sha256(path, block=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while chunk := stream.read(block):
            digest.update(chunk)
    return digest.hexdigest()


def remap(mask):
    """Protocol: original 1 -> 1 (flooded building), 3 -> 2 (flooded road), every other ID -> 0."""
    out = np.zeros(mask.shape, dtype=np.uint8)
    out[mask == FLOODED_BUILDING] = 1
    out[mask == FLOODED_ROAD] = 2
    return out


def load_raw(image_path, mask_path):
    with Image.open(image_path) as image:
        rgb = ImageOps.exif_transpose(image).convert("RGB")
    with Image.open(mask_path) as mask_image:
        mask = np.asarray(mask_image).copy()
    return rgb, mask


def fit_and_pad(rgb_image, mask):
    """Protocol: keep aspect ratio, fit inside 256 (Lanczos RGB / nearest mask), centre-pad."""
    width, height = rgb_image.size
    if mask.shape != (height, width):
        raise ValueError(f"image {rgb_image.size} vs mask {mask.shape}")
    scale = min(SIZE / width, SIZE / height)
    new_w, new_h = max(1, round(width * scale)), max(1, round(height * scale))
    left, top = (SIZE - new_w) // 2, (SIZE - new_h) // 2
    small_rgb = np.asarray(rgb_image.resize((new_w, new_h), Image.Resampling.LANCZOS))
    small_mask = np.asarray(Image.fromarray(remap(mask)).resize((new_w, new_h), Image.Resampling.NEAREST))
    rgb = np.empty((SIZE, SIZE, 3), dtype=np.uint8)
    rgb[:] = PAD_RGB
    labels = np.full((SIZE, SIZE), IGNORE, dtype=np.uint8)
    rgb[top:top + new_h, left:left + new_w] = small_rgb
    labels[top:top + new_h, left:left + new_w] = small_mask
    return rgb, labels, {"left": left, "top": top, "width": new_w, "height": new_h}


def derive(job):
    image_id, image_path, mask_path = job
    rgb_image, mask = load_raw(image_path, mask_path)
    rgb, labels, geometry = fit_and_pad(rgb_image, mask)
    return image_id, rgb, labels, geometry


def load_derived(split):
    """Arrays produced by audit_data.py from the raw release (never from the project cache)."""
    data = np.load(DERIVED / f"{split}.npz")
    geometry = json.loads((DERIVED / f"{split}_geometry.json").read_text(encoding="utf-8"))
    return list(data["ids"]), data["rgb"], data["mask"], geometry


def confusion(truth, prediction):
    """3x3 counts over valid (non-padding) pixels; rows = truth, columns = prediction."""
    from sklearn.metrics import confusion_matrix
    valid = truth != IGNORE
    return confusion_matrix(truth[valid].ravel(), prediction[valid].ravel(), labels=[0, 1, 2]).astype(np.int64)


def scores(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    tp = np.diag(matrix)
    support, predicted = matrix.sum(1), matrix.sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        iou = tp / (support + predicted - tp)
        recall = tp / support
        dice = 2 * tp / (support + predicted)
    defined = ~np.isnan(iou)
    return {"iou": iou.tolist(), "recall": recall.tolist(), "dice": dice.tolist(),
            "flooded_miou": float(np.nanmean(iou[1:])) if defined[1:].any() else None,
            "miou": float(np.nanmean(iou)) if defined.any() else None}


def artifact_hashes():
    """G1: SHA-256 of every frozen artifact the audit must not change."""
    files = [p for base in ("runs", "data/processed", "data/manifests") for p in sorted((ROOT / base).rglob("*"))
             if p.is_file() and "archive" not in p.relative_to(ROOT).parts]
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=8) as pool:
        digests = list(pool.map(sha256, files))
    return {p.relative_to(ROOT).as_posix(): d for p, d in zip(files, digests)}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    path.write_text(json.dumps(value, indent=2, default=_json_default), encoding="utf-8")
    return path


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value))
