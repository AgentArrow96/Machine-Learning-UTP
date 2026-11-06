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


CONTEXT_NAMES = [f"neighbour_{name}" for name in FEATURE_NAMES] + ["centroid_y", "centroid_x", "area_share",
                                                                  "coarse_mean_r", "coarse_mean_g", "coarse_mean_b"]


def context_features(x, regions, rgb, coarse_segments, compactness=10):
    """Improvement study (rf_context): each region's 22 features + the mean of its 4-connected neighbours' features,
    its centroid (0-1) and area share, and the mean colour of the coarse SLIC region it mostly lies in.
    `regions` (n region ids from 0, -1 = padding) and `rgb` cover the same pixels."""
    n = len(x)
    valid = regions >= 0
    pairs = set()
    for a, b in ((regions[:, :-1], regions[:, 1:]), (regions[:-1, :], regions[1:, :])):
        edge = (a != b) & (a >= 0) & (b >= 0)
        pairs.update(zip(np.minimum(a, b)[edge].tolist(), np.maximum(a, b)[edge].tolist()))   # unordered: each neighbour once
    neighbour = np.zeros_like(x)
    counts = np.zeros(n)
    for a, b in pairs:
        neighbour[a] += x[b]; neighbour[b] += x[a]; counts[a] += 1; counts[b] += 1
    lonely = counts == 0
    neighbour[~lonely] /= counts[~lonely, None]
    neighbour[lonely] = x[lonely]
    ys, xs = np.nonzero(valid)
    ids = regions[valid]
    area = np.bincount(ids, minlength=n).astype(np.float64)
    height, width = regions.shape
    cy = np.bincount(ids, weights=ys, minlength=n) / np.maximum(area, 1) / max(height - 1, 1)
    cx = np.bincount(ids, weights=xs, minlength=n) / np.maximum(area, 1) / max(width - 1, 1)
    coarse = slic(rgb, n_segments=coarse_segments, compactness=compactness, max_num_iter=10, enforce_connectivity=True,
                  start_label=0, channel_axis=-1)
    colour = rgb.reshape(-1, 3).astype(np.float64) / 255
    coarse_ids = coarse.reshape(-1)
    coarse_n = coarse_ids.max() + 1
    coarse_mean = np.stack([np.bincount(coarse_ids, weights=colour[:, c], minlength=coarse_n) for c in range(3)], 1) \
        / np.bincount(coarse_ids, minlength=coarse_n)[:, None]
    votes = np.zeros((n, coarse_n), dtype=np.int64)
    np.add.at(votes, (ids, coarse[valid]), 1)
    enclosing = coarse_mean[votes.argmax(1)]
    return np.concatenate([x, neighbour, cy[:, None], cx[:, None], (area / max(area.sum(), 1))[:, None], enclosing], 1).astype(np.float32)


def region_features(rgb, geometry, cfg, truth=None):
    """The RF's region rows: the 22 project features, or 50 with the improvement study's context features."""
    x, y, regions = extract_regions(rgb, geometry, cfg["slic"], truth)
    context = cfg["rf"].get("context")
    if context:
        x = context_features(x, crop(regions, geometry), crop(rgb, geometry), context["coarse_segments"], cfg["slic"]["compactness"])
    return x, y, regions


def region_labels10(record, regions, size, masks10=None):
    """10-class majority label of each region (aux10 RF), from the original annotation at the model size."""
    from .gan_common import masks10_for
    layout = masks10 if masks10 is not None else masks10_for(record, size)
    valid = regions >= 0
    votes = np.zeros((regions.max() + 1, 10), dtype=np.int64)
    np.add.at(votes, (regions[valid], np.minimum(layout[valid], 9)), 1)
    return votes.argmax(1).astype(np.uint8)


def feature_key(cfg, store):
    identity = {"processed": store.manifest["identity"], "slic": cfg["slic"], "feature_version": 1, "features": FEATURE_NAMES}
    if cfg["rf"].get("context"):
        # Only the improvement study's context features change the key; the frozen baseline key is untouched.
        identity.update(context=cfg["rf"]["context"], features=FEATURE_NAMES + CONTEXT_NAMES)
    if cfg["rf"].get("blacken"):
        # Blackening test: features of training photos whose 'Other' pixels are painted black (own cache).
        identity.update(blacken=True)
    return digest(identity)


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
    black = bool(cfg["rf"].get("blacken"))
    # The blackening test only changes the training photos; validation and test are scored on real photos.
    records = store.select("train") if black else store.records
    for index, record in enumerate(records):
        path = feature_path(cfg, store, record)
        if path.exists():
            try:
                with np.load(path, allow_pickle=False) as cached:
                    if str(cached["identity"]) != key or cached["x"].shape[1] != (50 if cfg["rf"].get("context") else 22):
                        raise ProjectError("Feature identity mismatch")
                    counts[record["id"]] = len(cached["x"])
                continue
            except (OSError, ValueError, KeyError) as error:
                raise ProjectError(f"Incomplete feature cache {path}: {error}") from error
        rgb, truth = store.get(record)
        if black:
            from .neural import blacken
            rgb = blacken(rgb, truth)
        x, y, regions = region_features(rgb, record["geometry"], cfg, truth)
        temporary = path.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, x=x, y=y, regions=regions, identity=key)
        temporary.replace(path)
        counts[record["id"]] = len(x)
        if index % 100 == 0:
            log(f"SLIC features {index + 1}/{len(records)}")
    result = {"identity": key, "processed_identity": store.manifest["identity"], "directory": relative(directory),
              "feature_names": FEATURE_NAMES + (CONTEXT_NAMES if cfg["rf"].get("context") else []), "region_counts": counts, "elapsed_seconds": previous_elapsed + time.perf_counter() - started, "complete": True}
    write_json(directory / "manifest.json", result)
    return result


def training_table(cfg, store, budget, seed, extra=None):
    """Region rows for the RF. `extra` (class-imbalance variants only) = {"images": [(id, x, y)], "target_rows": (x, y, ids)}:
    extra images join the round-robin like real ones; extra target rows are kept like real target regions."""
    records = store.select("train", budget)
    sampling = cfg["rf"]["region_sampling"]
    if sampling not in ("uniform", "keep_targets", "ratio"):
        raise ProjectError(f"Unknown RF region sampling: {sampling}")
    cap = cfg["rf"]["max_regions"]
    xs, ys, references = [], [], []
    # Shuffle each image's regions, then round-robin across images to cap large scenes fairly.
    rng = np.random.default_rng(seed)
    rows = []
    aux10 = cfg["rf"].get("aux10")
    if aux10:
        from .gan_common import masks10_cache
        layouts, layout_ids = masks10_cache(cfg, store, "train")
        layout_index = {image_id: i for i, image_id in enumerate(layout_ids)}
    for record in records:
        path = feature_path(cfg, store, record)
        if not path.exists():
            raise ProjectError("Missing SLIC cache; run features first")
        with np.load(path, allow_pickle=False) as cached:
            if str(cached["identity"]) != feature_key(cfg, store):
                raise ProjectError("Stale SLIC cache")
            # aux10: the forest learns the 10-class region majority (targets are IDs 1 and 3 there).
            y = region_labels10(record, cached["regions"], cfg["size"], np.asarray(layouts[layout_index[record["id"]]])) if aux10 else cached["y"].copy()
            rows.append((record["id"], cached["x"].copy(), y, rng.permutation(len(cached["y"]))))
    if extra:
        # A separate generator keeps the real rows' permutations identical to the variant-free table.
        extra_rng = np.random.default_rng([seed, 1])
        rows.extend((image_id, x, y, extra_rng.permutation(len(y))) for image_id, x, y in extra.get("images", []))
    if sampling == "ratio":
        # Undersampling pipeline: every target region, plus a seeded random sample of ratio x #target 'Other' regions.
        other = []
        for image_id, x, y, order in rows:
            for region in order:
                if y[region] > 0:
                    xs.append(x[region]); ys.append(y[region]); references.append([image_id, int(region)])
                else:
                    other.append((image_id, x, region))
        count = min(len(other), round(cfg["rf"]["ratio"] * len(xs)))
        for k in np.sort(np.random.default_rng([seed, 2]).choice(len(other), count, replace=False)):
            image_id, x, region = other[k]
            xs.append(x[region]); ys.append(0); references.append([image_id, int(region)])
        if len(xs) > cap:
            raise ProjectError(f"{len(xs):,} ratio-sampled regions exceed the {cap:,}-region cap")
        rows = []
    if sampling == "keep_targets":
        # Uniform capping discards most minority regions once the budget exceeds the cap,
        # hiding the budget effect. Keep every flooded region; the cap only thins "Other".
        is_target = (lambda labels: np.isin(labels, (1, 3))) if aux10 else (lambda labels: labels > 0)
        for image_id, x, y, order in rows:
            for region in order[is_target(y[order])]:
                xs.append(x[region]); ys.append(y[region]); references.append([image_id, int(region)])
        if extra and extra.get("target_rows") is not None:
            ex, ey, eids = extra["target_rows"]
            for k in range(len(ey)):
                xs.append(ex[k]); ys.append(ey[k]); references.append([eids[k], -1])
        if len(xs) > cap:
            raise ProjectError(f"{len(xs):,} target regions exceed the {cap:,}-region cap")
        rows = [(image_id, x, y, order[~is_target(y[order])]) for image_id, x, y, order in rows]
    for position in range(max((len(row[3]) for row in rows), default=0)):
        for image_id, x, y, order in rows:
            if position < len(order) and len(xs) < cap:
                region = int(order[position])
                xs.append(x[region]); ys.append(y[region]); references.append([image_id, region])
        if len(xs) == cap:
            break
    x, y = np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.uint8)
    needed = {1, 3} if aux10 else {0, 1, 2}
    if not needed <= set(np.unique(y).tolist()) or (not aux10 and set(np.unique(y).tolist()) != {0, 1, 2}):
        raise ProjectError("Training superpixels do not contain all three classes; inspect majority labels and budget.")
    return x, y, references
