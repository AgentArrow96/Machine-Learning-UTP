"""Deterministic SLIC, 22 region descriptors, and source-keyed feature caches."""
import time
import numpy as np
from skimage.color import rgb2hsv, rgb2gray
from skimage.feature import local_binary_pattern
from skimage.segmentation import slic

from .common import ProjectError, digest, inside, log, read_json, relative, write_json
from .data import Store, crop

FEATURE_NAMES = [f"{space}_{stat}_{channel}" for space, channels in (("rgb", "rgb"), ("hsv", "hsv"))
                 for stat in ("mean", "std") for channel in channels] + [f"lbp_{i}" for i in range(10)]


def extract_regions(rgb, geometry, settings, truth=None):
    content = crop(rgb, geometry)
    # Segment unpadded content: padding cannot alter region placement or colour statistics.
    labels = slic(content, n_segments=settings["n_segments"], compactness=settings["compactness"],
                  max_num_iter=settings["max_num_iter"], enforce_connectivity=True, start_label=1, channel_axis=-1)
    unique = np.unique(labels)
    indices = np.searchsorted(unique, labels)
    n = len(unique)
    flat = indices.ravel()
    counts = np.bincount(flat, minlength=n).astype(np.float64)
    colour = content.astype(np.float64) / 255
    hsv = rgb2hsv(colour)
    features = []
    for values in (colour, hsv):
        means = np.stack([np.bincount(flat, weights=values[..., c].ravel(), minlength=n) / counts for c in range(3)], 1)
        variances = np.stack([np.bincount(flat, weights=values[..., c].ravel() ** 2, minlength=n) / counts for c in range(3)], 1) - means ** 2
        features.extend([means, np.sqrt(np.maximum(variances, 0))])
    grey = np.round(rgb2gray(content) * 255).astype(np.uint8)
    lbp = local_binary_pattern(grey, P=8, R=1, method="uniform").astype(np.int64)
    histogram = np.bincount(flat * 10 + lbp.ravel(), minlength=n * 10).reshape(n, 10) / counts[:, None]
    x = np.concatenate([*features, histogram], axis=1).astype(np.float32)
    region_map = np.full(rgb.shape[:2], -1, dtype=np.int32)
    g = geometry
    region_map[g["top"]:g["top"] + g["height"], g["left"]:g["left"] + g["width"]] = indices
    y = None
    if truth is not None:
        valid_truth = crop(truth, geometry)
        if np.any(valid_truth > 2):
            raise ProjectError("Padding/invalid class inside valid image rectangle")
        votes = np.bincount(flat * 3 + valid_truth.ravel(), minlength=n * 3).reshape(n, 3)
        y = votes.argmax(1).astype(np.uint8)
    return x, y, region_map


def region_features(rgb, geometry, cfg, truth=None):
    """The RF's region rows: 22 colour and texture features per SLIC superpixel."""
    return extract_regions(rgb, geometry, cfg["slic"], truth)


def feature_key(cfg, store):
    return digest({"processed": store.manifest["identity"], "slic": cfg["slic"], "feature_version": 1, "features": FEATURE_NAMES})


def feature_path(cfg, store, record):
    return inside(f"data/features/{feature_key(cfg, store)[:16]}/{record['id']}.npz")


def build_features(cfg):
    store = Store(cfg)
    key = feature_key(cfg, store)
    directory = inside(f"data/features/{key[:16]}")
    directory.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    old_manifest = directory / "manifest.json"
    previous_elapsed = read_json(old_manifest).get("elapsed_seconds", 0.) if old_manifest.exists() else 0.
    counts = {}
    for index, record in enumerate(store.records):
        path = feature_path(cfg, store, record)
        if path.exists():
            try:
                with np.load(path, allow_pickle=False) as cached:
                    if str(cached["identity"]) != key or cached["x"].shape[1] != len(FEATURE_NAMES):
                        raise ProjectError("Feature identity mismatch")
                    counts[record["id"]] = len(cached["x"])
                continue
            except (OSError, ValueError, KeyError) as error:
                raise ProjectError(f"Incomplete feature cache {path}: {error}") from error
        rgb, truth = store.get(record)
        x, y, regions = region_features(rgb, record["geometry"], cfg, truth)
        temporary = path.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, x=x, y=y, regions=regions, identity=key)
        temporary.replace(path)
        counts[record["id"]] = len(x)
        if index % 100 == 0:
            log(f"SLIC features {index + 1}/{len(store.records)}")
    result = {"identity": key, "processed_identity": store.manifest["identity"], "directory": relative(directory),
              "feature_names": FEATURE_NAMES, "region_counts": counts, "elapsed_seconds": previous_elapsed + time.perf_counter() - started, "complete": True}
    write_json(directory / "manifest.json", result)
    return result


def training_table(cfg, store, budget, seed):
    """Region rows for the RF, in a seeded order. With `keep_targets` every flooded region is kept and the
    `max_regions` cap only thins "Other"; the report's configuration sets a cap above the region count, so every
    training region is used."""
    records = store.select("train", budget)
    sampling = cfg["rf"]["region_sampling"]
    if sampling not in ("uniform", "keep_targets"):
        raise ProjectError(f"Unknown RF region sampling: {sampling}")
    cap = cfg["rf"]["max_regions"]
    xs, ys, references = [], [], []
    # Shuffle each image's regions, then round-robin across images to cap large scenes fairly.
    rng = np.random.default_rng(seed)
    rows = []
    for record in records:
        path = feature_path(cfg, store, record)
        if not path.exists():
            raise ProjectError("Missing SLIC cache; run features first")
        with np.load(path, allow_pickle=False) as cached:
            if str(cached["identity"]) != feature_key(cfg, store):
                raise ProjectError("Stale SLIC cache")
            rows.append((record["id"], cached["x"].copy(), cached["y"].copy(), rng.permutation(len(cached["y"]))))
    if sampling == "keep_targets":
        # Uniform capping discards most minority regions once the budget exceeds the cap,
        # hiding the budget effect. Keep every flooded region; the cap only thins "Other".
        for image_id, x, y, order in rows:
            for region in order[y[order] > 0]:
                xs.append(x[region]); ys.append(y[region]); references.append([image_id, int(region)])
        if len(xs) > cap:
            raise ProjectError(f"{len(xs):,} target regions exceed the {cap:,}-region cap")
        rows = [(image_id, x, y, order[y[order] == 0]) for image_id, x, y, order in rows]
    for position in range(max((len(row[3]) for row in rows), default=0)):
        for image_id, x, y, order in rows:
            if position < len(order) and len(xs) < cap:
                region = int(order[position])
                xs.append(x[region]); ys.append(y[region]); references.append([image_id, region])
        if len(xs) == cap:
            break
    x, y = np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.uint8)
    if set(np.unique(y).tolist()) != {0, 1, 2}:
        raise ProjectError("Training superpixels do not contain all three classes; inspect majority labels and budget.")
    return x, y, references
