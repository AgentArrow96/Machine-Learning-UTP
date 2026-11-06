"""Verified scene overlap between training and evaluation images.

Consecutive UAV frames often share ground. Candidate train frames are proposed by frame-ID
proximity and ImageNet-embedding similarity, then verified geometrically: ORB keypoints,
RANSAC homography, and the fraction of the evaluation frame covered by the warped train frame.
The analysis reads images only (never predictions), so it cannot bias model selection.
"""
from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image, ImageOps

from .common import ProjectError, digest, inside, log, now, read_json, relative, write_json

VERSION = 2
GRID = (120, 160)  # rows, cols used to rasterise the warped footprint


def grey_original(path, long_side):
    with Image.open(inside(path)) as image:
        # JPEG draft decoding downsamples during decode, avoiding full 12-megapixel decodes.
        image.draft("L", (long_side, long_side))
        oriented = ImageOps.exif_transpose(image).convert("L")
        scale = long_side / max(oriented.size)
        size = (max(1, round(oriented.width * scale)), max(1, round(oriented.height * scale)))
        return np.asarray(oriented.resize(size, Image.Resampling.BILINEAR), dtype=np.float32) / 255


def orb_features(grey, keypoints):
    from skimage.feature import ORB
    detector = ORB(n_keypoints=keypoints, fast_threshold=0.05)
    detector.detect_and_extract(grey)
    # skimage reports (row, col); homographies use (x, y).
    return detector.keypoints[:, ::-1].astype(np.float32), detector.descriptors, np.array(grey.shape)


def cache_features(job):
    record_id, image, target, long_side, keypoints = job
    if os.path.exists(target):
        return record_id
    try:
        points, descriptors, shape = orb_features(grey_original(image, long_side), keypoints)
    except RuntimeError:  # textureless frame (e.g. open water): no keypoints
        points, descriptors, shape = np.zeros((0, 2), np.float32), np.zeros((0, 256), bool), np.array([0, 0])
    temporary = target[:-4] + ".tmp.npz"
    np.savez_compressed(temporary, points=points, descriptors=descriptors, shape=shape)
    os.replace(temporary, target)
    return record_id


def footprint_overlap(model, source_shape, target_shape):
    """Fraction of the target frame covered by the source frame mapped through `model`."""
    from skimage.draw import polygon
    h, w = source_shape
    corners = model(np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64))
    if not np.all(np.isfinite(corners)):
        return 0.
    th, tw = target_shape
    rows = corners[:, 1] / th * GRID[0]
    cols = corners[:, 0] / tw * GRID[1]
    # Two views of the same ground plane keep the quadrilateral convex with the source's
    # winding order; a fold or mirror image means RANSAC fitted a degenerate model.
    edges = np.roll(corners, -1, 0) - corners
    cross = edges[:, 0] * np.roll(edges, -1, 0)[:, 1] - edges[:, 1] * np.roll(edges, -1, 0)[:, 0]
    if not np.all(cross > 0):
        return 0.
    rr, cc = polygon(rows, cols, shape=GRID)
    return len(rr) / (GRID[0] * GRID[1])


def verify_pair(first, second, settings, seed):
    """Match `first` (evaluation) against `second` (train); return evidence for the pair."""
    from skimage.feature import match_descriptors
    from skimage.measure import ransac
    from skimage.transform import ProjectiveTransform
    result = {"matches": 0, "inliers": 0, "inlier_ratio": 0., "overlap": 0., "verified": False}
    if len(first["points"]) < 4 or len(second["points"]) < 4:
        return result
    matches = match_descriptors(second["descriptors"], first["descriptors"], cross_check=True, max_ratio=0.8)
    result["matches"] = int(len(matches))
    if len(matches) < max(4, settings["min_inliers"]):
        return result
    source = second["points"][matches[:, 0]].astype(np.float64)
    target = first["points"][matches[:, 1]].astype(np.float64)
    model, inliers = ransac((source, target), ProjectiveTransform, min_samples=4,
                            residual_threshold=settings["residual_threshold"], max_trials=settings["max_trials"],
                            rng=np.random.default_rng(seed))
    if model is None or inliers is None:
        return result
    count = int(inliers.sum())
    result.update(inliers=count, inlier_ratio=count / len(matches))
    if count >= settings["min_inliers"] and result["inlier_ratio"] >= settings["min_inlier_ratio"]:
        result["overlap"] = footprint_overlap(model, second["shape"], first["shape"])
        result["verified"] = result["overlap"] >= settings["min_overlap"]
    return result


def load_cached(path):
    with np.load(path) as cached:
        return {"points": cached["points"], "descriptors": cached["descriptors"], "shape": tuple(int(v) for v in cached["shape"])}


def verify_job(job):
    first_path, second_path, settings, seed = job
    return verify_pair(load_cached(first_path), load_cached(second_path), settings, seed)


def embeddings(cfg, store, records):
    """Global-pooled ImageNet ResNet-18 descriptors of the processed RGB (retrieval only)."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    import torch
    from .neural import build_model, tensor_image
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = build_model(cfg).encoder.to(device).eval()
    vectors = []
    with torch.inference_mode():
        for start in range(0, len(records), 64):
            batch = torch.stack([tensor_image(store.get(r)[0]) for r in records[start:start + 64]]).to(device)
            pooled = encoder(batch)[-1].float().mean((2, 3))
            vectors.append(torch.nn.functional.normalize(pooled, dim=1).cpu().numpy())
    return np.concatenate(vectors)


def candidates(cfg, store, records):
    settings = cfg["overlap"]
    train = [r for r in records if r["split"] == "train"]
    evaluated = [r for r in records if r["split"] != "train"]
    by_number = {int(r["image_id"]): r["id"] for r in train if r["image_id"].isdigit()}
    vectors = dict(zip([r["id"] for r in records], embeddings(cfg, store, records)))
    train_matrix = np.stack([vectors[r["id"]] for r in train])
    train_index = {r["id"]: i for i, r in enumerate(train)}
    pairs = {}
    for record in evaluated:
        sources = {}
        if record["image_id"].isdigit():
            number = int(record["image_id"])
            for delta in range(-settings["frame_window"], settings["frame_window"] + 1):
                if delta and number + delta in by_number:
                    sources.setdefault(by_number[number + delta], set()).add("frame")
        similarity = train_matrix @ vectors[record["id"]]
        for index in np.argsort(-similarity, kind="stable")[:settings["embedding_neighbours"]]:
            sources.setdefault(train[index]["id"], set()).add("embedding")
        for train_id, kinds in sources.items():
            pairs[(record["id"], train_id)] = {"sources": sorted(kinds),
                                               "embedding_similarity": float(similarity[train_index[train_id]])}
    return pairs


def overlap_key(cfg, store):
    return digest({"version": VERSION, "processed": store.manifest["identity"], "settings": cfg["overlap"]})


def analyse(cfg):
    from .data import Store
    started = time.perf_counter()
    store = Store(cfg)
    settings = cfg["overlap"]
    key = overlap_key(cfg, store)
    path = inside("data/manifests/overlap.json")
    if path.exists() and read_json(path).get("identity") == key:
        log("Verified existing overlap analysis; reuse.")
        return read_json(path)
    records = store.records
    by_id = {r["id"]: r for r in records}
    pairs = candidates(cfg, store, records)
    log(f"Overlap: {len(pairs):,} candidate evaluation/train pairs")
    directory = inside(f"data/features/overlap/{digest({'long_side': settings['long_side'], 'keypoints': settings['keypoints'], 'processed': store.manifest['identity']})[:16]}")
    directory.mkdir(parents=True, exist_ok=True)
    needed = sorted({i for pair in pairs for i in pair})
    feature_paths = {i: str(directory / f"{i}.npz") for i in needed}
    jobs = [(i, by_id[i]["image"], feature_paths[i], settings["long_side"], settings["keypoints"]) for i in needed]
    with ProcessPoolExecutor(max_workers=settings["workers"]) as pool:
        for index, _ in enumerate(pool.map(cache_features, jobs, chunksize=4)):
            if index % 250 == 0:
                log(f"Overlap keypoints {index + 1}/{len(jobs)}")
    ordered = sorted(pairs)
    seed = cfg["subset_seed"]
    verify_jobs = [(feature_paths[a], feature_paths[b], settings, seed) for a, b in ordered]
    results = []
    with ProcessPoolExecutor(max_workers=settings["workers"]) as pool:
        for index, result in enumerate(pool.map(verify_job, verify_jobs, chunksize=8)):
            results.append(result)
            if index % 1000 == 0:
                log(f"Overlap verification {index + 1}/{len(verify_jobs)}")
    pair_rows = [{"eval_id": a, "train_id": b, **pairs[(a, b)], **result} for (a, b), result in zip(ordered, results)]
    per_image = {}
    for record in records:
        if record["split"] == "train":
            continue
        verified = [p for p in pair_rows if p["eval_id"] == record["id"] and p["verified"]]
        per_image[record["id"]] = {"max_train_overlap": max((p["overlap"] for p in verified), default=0.),
                                   "verified_train_ids": sorted(p["train_id"] for p in verified)}
    summary = {}
    for split in ("val", "test"):
        ids = [r["id"] for r in records if r["split"] == split]
        overlapping = [i for i in ids if per_image[i]["verified_train_ids"]]
        summary[split] = {"images": len(ids), "overlapping_train": len(overlapping),
                          "clean_ids": sorted(set(ids) - set(overlapping)),
                          "verified_pairs": sum(p["verified"] for p in pair_rows if p["eval_id"].startswith(split + "_"))}
    manifest = {"schema_version": 1, "identity": key, "version": VERSION, "created_at": now(),
                "processed_identity": store.manifest["identity"], "settings": settings, "feature_directory": relative(directory),
                "method": ("Candidates: train frames within ±frame_window IDs plus top embedding neighbours "
                           "(ImageNet ResNet-18, processed RGB). Verification: ORB keypoints on EXIF-oriented "
                           "greyscale originals, cross-checked Hamming matches, RANSAC homography; verified when "
                           "inliers, inlier ratio and covered evaluation-frame fraction pass the thresholds."),
                "summary": summary, "per_image": per_image, "pairs": pair_rows,
                "elapsed_seconds": time.perf_counter() - started}
    write_json(path, manifest)
    for split in ("val", "test"):
        s = summary[split]
        log(f"Overlap {split}: {s['overlapping_train']}/{s['images']} images overlap a train frame; {len(s['clean_ids'])} clean")
    return manifest


def check_pair(cfg, eval_id, train_id):
    """Diagnostic for threshold calibration: verify one explicit pair without caching."""
    from .data import Store
    store = Store(cfg)
    settings = cfg["overlap"]
    loaded = []
    for record_id in (eval_id, train_id):
        record = store.by_id.get(record_id)
        if record is None:
            raise ProjectError(f"Unknown record {record_id}")
        points, descriptors, shape = orb_features(grey_original(record["image"], settings["long_side"]), settings["keypoints"])
        loaded.append({"points": points, "descriptors": descriptors, "shape": tuple(shape)})
    return verify_pair(*loaded, settings, cfg["subset_seed"])
