"""D1-D8: raw data, label meaning, preprocessing and budget checks. Independent of `floodnet`."""
from __future__ import annotations

import json
import re
import sys
import time
import zipfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent))
from independent import (DERIVED, RAW, ROOT, SPLITS, derive, discover, guard, read_json, sha256, write_json)

WORKERS = 12
COLOUR = ROOT / "data/raw/ColorMasks-FloodNetv1.0"


def orientation_and_counts(job):
    """EXIF orientation of the photo and per-ID pixel counts of the full-resolution mask."""
    image_id, image_path, mask_path = job
    with Image.open(image_path) as image:
        orientation = image.getexif().get(274, 1)
    with Image.open(mask_path) as mask_image:
        mask = np.asarray(mask_image)
    ids = np.unique(mask)
    return image_id, int(orientation), mask_image.mode, ids.tolist(), np.bincount(mask.ravel(), minlength=10)[:10].tolist(), int(mask.max())


def edge_agreement(rgb, mask):
    """Mean image gradient on mask boundaries relative to the whole image (higher = aligned)."""
    grey = rgb.astype(np.float64).mean(2)
    gradient = np.abs(np.diff(grey, axis=1))[:-1, :] + np.abs(np.diff(grey, axis=0))[:, :-1]
    boundary = (mask[:, 1:] != mask[:, :-1])[:-1, :] | (mask[1:, :] != mask[:-1, :])[:, :-1]
    return float(gradient[boundary].mean() / gradient.mean()) if boundary.any() else None


def exif_job(job):
    image_id, image_path, mask_path = job
    with Image.open(mask_path) as mask_image:
        mask = np.asarray(mask_image.resize((800, 600), Image.Resampling.NEAREST))
    with Image.open(image_path) as image:
        raw = np.asarray(image.convert("RGB").resize((800, 600), Image.Resampling.BILINEAR))
        rotated = np.asarray(ImageOps.exif_transpose(image).convert("RGB").resize((800, 600), Image.Resampling.BILINEAR))
    return image_id, edge_agreement(rotated, mask), edge_agreement(raw, mask)


def palette():
    """Name -> RGB colour from the release's ColorPalette-Values.xlsx (read as XML, no extra deps)."""
    archive = zipfile.ZipFile(COLOUR / "ColorPalette-Values.xlsx")
    strings = re.findall(r"<t[^>]*>([^<]*)</t>", archive.read("xl/sharedStrings.xml").decode("utf-8"))
    sheet = archive.read("xl/worksheets/sheet1.xml").decode("utf-8")
    cells = dict(re.findall(r'<c r="([A-Z]+\d+)"[^>]*t="s"[^>]*><v>(\d+)</v>', sheet))
    names = {}
    for ref, index in cells.items():
        column, row = re.match(r"([A-Z]+)(\d+)", ref).groups()
        names.setdefault(row, {})[column] = strings[int(index)]
    result = {}
    for row in names.values():
        name, colour = row.get("G"), row.get("H")
        if name and colour:
            result[name] = tuple(int(v) for v in re.findall(r"\d+", colour))
    return result


def colour_job(job):
    """Root cause of the first D5 run (kept in data_run1_D5_stretch_method.json): the official colour masks
    render the TOP-LEFT 4000x3000 window of the indexed mask, so 4592x3072 masks must be cropped, not stretched."""
    image_id, mask_path, colour_path = job
    with Image.open(colour_path) as colour_image:
        colour = np.asarray(colour_image.convert("RGB"))
    with Image.open(mask_path) as mask_image:
        size = mask_image.size
        window = mask_image.crop((0, 0, min(size[0], 4000), min(size[1], 3000)))
        mask = np.asarray(window.resize((colour.shape[1], colour.shape[0]), Image.Resampling.NEAREST))
    packed = (colour[..., 0].astype(np.int64) << 16) | (colour[..., 1].astype(np.int64) << 8) | colour[..., 2]
    table = {}
    for value in np.unique(mask):
        colours, counts = np.unique(packed[mask == value], return_counts=True)
        table[int(value)] = {int(c): int(n) for c, n in zip(colours, counts)}
    return image_id, table, list(size)


def majority_agreement(mask_path, colour_path, crop_window):
    """Share of pixels whose colour is the majority colour of their mask ID (root-cause evidence for D5)."""
    with Image.open(colour_path) as colour_image:
        colour = np.asarray(colour_image.convert("RGB"))
    with Image.open(mask_path) as mask_image:
        source = mask_image.crop((0, 0, min(mask_image.size[0], 4000), min(mask_image.size[1], 3000))) if crop_window else mask_image
        mask = np.asarray(source.resize((colour.shape[1], colour.shape[0]), Image.Resampling.NEAREST))
    packed = (colour[..., 0].astype(np.int64) << 16) | (colour[..., 1].astype(np.int64) << 8) | colour[..., 2]
    return float(sum(np.unique(packed[mask == v], return_counts=True)[1].max() for v in np.unique(mask)) / mask.size)


def alignment_job(job):
    """Edge agreement of every photo with its own mask and with another image's mask (control)."""
    (image_id, image_path, mask_path), (_, _, other_mask_path) = job
    with Image.open(image_path) as image:
        rgb = np.asarray(ImageOps.exif_transpose(image).convert("RGB").resize((800, 600), Image.Resampling.BILINEAR))
    with Image.open(mask_path) as mask_image:
        own = np.asarray(mask_image.resize((800, 600), Image.Resampling.NEAREST))
    with Image.open(other_mask_path) as other_image:
        other = np.asarray(other_image.resize((800, 600), Image.Resampling.NEAREST))
    return image_id, edge_agreement(rgb, own), edge_agreement(rgb, other)


def main():
    started = time.perf_counter()
    report = {}
    audit = read_json(ROOT / "data/manifests/audit.json")
    stored = {r["id"]: r for r in audit["records"]}
    note = (RAW / "DATASET-VERSION-NOTE-v1.0.txt").read_text(encoding="utf-8", errors="replace")
    note_counts = dict(zip(("train", "val", "test"), map(int, re.search(r"Train: (\d+), Val: (\d+), Test: (\d+)", note).groups())))

    # D1 counts and pairing
    pairs = discover()
    d1 = {}
    jobs = {}
    for split in SPLITS:
        images, masks = pairs[split]["images"], pairs[split]["masks"]
        complete = sorted(set(images) & set(masks), key=lambda s: int(s) if s.isdigit() else s)
        d1[split] = {"images": len(images), "masks": len(masks), "complete_pairs": len(complete),
                     "unpaired": sorted(set(images) ^ set(masks)), "release_note": note_counts[split]}
        jobs[split] = [(f"{split}_{key}", images[key], masks[key]) for key in complete]
    report["D1"] = {"detail": d1, "pass": all(v["complete_pairs"] == v["release_note"] == v["images"] == v["masks"] and not v["unpaired"]
                                            for v in d1.values()) and [d1[s]["complete_pairs"] for s in SPLITS] == [1445, 450, 448]}

    # D2 raw file hashes and the archive hash
    files = [(image_id, kind, path) for split in SPLITS for image_id, image, mask in jobs[split] for kind, path in (("image", image), ("mask", mask))]
    with ThreadPoolExecutor(max_workers=8) as pool:
        digests = list(pool.map(lambda item: (item[0], item[1], sha256(item[2])), files))
    mismatches = [(i, k) for i, k, h in digests if stored.get(i, {}).get(f"{k}_sha256") != h]
    source = read_json(ROOT / "data/downloads/source.json")
    archive = ROOT / source["archive"]
    archive_hash = sha256(archive) if archive.exists() else None
    report["D2"] = {"files_hashed": len(digests), "mismatches": mismatches[:20], "mismatch_count": len(mismatches),
                    "archive_sha256": archive_hash, "archive_expected": source["archive_sha256"],
                    "pass": not mismatches and len(digests) == 4686 and archive_hash == source["archive_sha256"]}
    image_hashes = {i: h for i, k, h in digests if k == "image"}

    # D3 split disjointness
    keys = {split: {image_id.split("_", 1)[1] for image_id, _, _ in jobs[split]} for split in SPLITS}
    hashes = {split: {image_hashes[image_id] for image_id, _, _ in jobs[split]} for split in SPLITS}
    shared_ids = {f"{a}-{b}": len(keys[a] & keys[b]) for a, b in (("train", "val"), ("train", "test"), ("val", "test"))}
    shared_hashes = {f"{a}-{b}": len(hashes[a] & hashes[b]) for a, b in (("train", "val"), ("train", "test"), ("val", "test"))}
    report["D3"] = {"shared_ids": shared_ids, "shared_image_hashes": shared_hashes,
                    "pass": not any(shared_ids.values()) and not any(shared_hashes.values())}

    # D4 mask IDs and per-class pixel counts (full resolution)
    all_jobs = [job for split in SPLITS for job in jobs[split]]
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        meta = {r[0]: r for r in pool.map(orientation_and_counts, all_jobs, chunksize=8)}
    bad_ids = [i for i, m in meta.items() if not set(m[3]) <= set(range(10))]
    count_mismatch = [i for i, m in meta.items() if m[4] != stored[i]["original_counts"]]
    totals = {split: np.sum([meta[j[0]][4] for j in jobs[split]], axis=0).tolist() for split in SPLITS}
    report["D4"] = {"masks_checked": len(meta), "mask_modes": dict(Counter(m[2] for m in meta.values())),
                    "masks_with_ids_outside_0_9": bad_ids, "count_mismatches": count_mismatch[:20],
                    "count_mismatch_total": len(count_mismatch), "pixel_totals_by_split": totals,
                    "pass": not bad_ids and not count_mismatch and len(meta) == 2343}

    # D5 label meaning via the official colour masks and palette
    names = palette()
    colour_dirs = {"train": "ColorMasks-TrainSet", "val": "ColorMasks-ValSet", "test": "ColorMasks-TestSet"}
    rng = np.random.default_rng(20261024)
    with_targets = [j for j in all_jobs if meta[j[0]][4][1] or meta[j[0]][4][3]]
    others = [j for j in all_jobs if not (meta[j[0]][4][1] or meta[j[0]][4][3])]
    sample = [with_targets[i] for i in rng.choice(len(with_targets), 75, replace=False)] + \
             [others[i] for i in rng.choice(len(others), 75, replace=False)]
    colour_jobs = []
    for image_id, _, mask_path in sample:
        split, key = image_id.split("_", 1)
        colour_jobs.append((image_id, mask_path, COLOUR / colour_dirs[split] / f"{key}_lab.png"))
    missing = [str(j[2]) for j in colour_jobs if not j[2].exists()]
    colour_jobs = [j for j in colour_jobs if j[2].exists()]
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        tables = list(pool.map(colour_job, colour_jobs))
    votes = {}
    for _, table, _ in tables:
        for value, counts in table.items():
            for colour, count in counts.items():
                votes.setdefault(value, Counter())[colour] += count
    by_colour = {rgb: name for name, rgb in names.items()}
    pack = lambda c: (c[0] << 16) | (c[1] << 8) | c[2]
    mapping = {}
    for value, counter in votes.items():
        colour, _ = counter.most_common(1)[0]
        rgb = (colour >> 16, (colour >> 8) & 255, colour & 255)
        mapping[value] = {"majority_colour": rgb, "palette_name": by_colour.get(rgb)}
    agree = sum(votes[v][pack(mapping[v]["majority_colour"])] for v in votes)
    total = sum(sum(c.values()) for c in votes.values())
    # Agreement per image and per photo size, against the PALETTE colour of each ID (not the majority vote).
    palette_of = {v: names.get(m["palette_name"]) for v, m in mapping.items()}
    per_size = {}
    for _, table, size in tables:
        hit = sum(counts.get(pack(palette_of[v]), 0) for v, counts in table.items() if palette_of.get(v))
        n = sum(sum(counts.values()) for counts in table.values())
        per_size.setdefault(f"{size[0]}x{size[1]}", []).append(hit / n)
    report["D5"] = {"palette": {k: list(v) for k, v in names.items()}, "id_to_colour": {str(k): {"colour": list(v["majority_colour"]), "name": v["palette_name"]} for k, v in sorted(mapping.items())},
                    "sample_masks": len(tables), "missing_colour_masks": missing, "pixel_agreement": agree / total,
                    "agreement_by_photo_size": {k: {"images": len(v), "mean": float(np.mean(v)), "min": float(np.min(v))} for k, v in per_size.items()},
                    "method_note": "First run stretched the whole mask to 1024x1024 and failed (79.4%); root cause: colour masks render the top-left 4000x3000 window. Criterion unchanged; comparison now crops that window.",
                    "root_cause_evidence": {f"test_{key}": {"photo_size": list(Image.open(RAW / f"test/test-label-img/{key}_lab.png").size),
                                                            "stretched": majority_agreement(RAW / f"test/test-label-img/{key}_lab.png", COLOUR / f"ColorMasks-TestSet/{key}_lab.png", False),
                                                            "top_left_4000x3000": majority_agreement(RAW / f"test/test-label-img/{key}_lab.png", COLOUR / f"ColorMasks-TestSet/{key}_lab.png", True)}
                                            for key in ("10163", "6336", "7486")},
                    "pass": mapping.get(1, {}).get("palette_name") == "building-flooded" and mapping.get(3, {}).get("palette_name") == "road-flooded"
                    and agree / total >= .99 and not missing}

    # D6b (added after the D5 investigation): every photo aligns with its own mask far better than with another's.
    rng_pairs = np.random.default_rng(5)
    shuffled = rng_pairs.permutation(len(all_jobs))
    while np.any(shuffled == np.arange(len(all_jobs))):
        shuffled = rng_pairs.permutation(len(all_jobs))
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        aligned = list(pool.map(alignment_job, [(all_jobs[k], all_jobs[shuffled[k]]) for k in range(len(all_jobs))], chunksize=8))
    valid_rows = [(i, a, b) for i, a, b in aligned if a is not None and b is not None]
    wins = sum(a > b for _, a, b in valid_rows)
    report["D6b"] = {"images_compared": len(valid_rows), "own_mask_better": int(wins),
                     "median_own": float(np.median([a for _, a, _ in valid_rows])), "median_other": float(np.median([b for _, _, b in valid_rows])),
                     "own_mask_not_better": [i for i, a, b in valid_rows if a <= b][:20],
                     "pass": wins >= .99 * len(valid_rows)}

    # D6 EXIF orientation choice
    rotated_jobs = [j for j in all_jobs if meta[j[0]][1] != 1]
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        exif = list(pool.map(exif_job, rotated_jobs))
    better = sum(1 for _, rot, raw in exif if rot is not None and raw is not None and rot > raw)
    report["D6"] = {"orientations": dict(Counter(m[1] for m in meta.values())), "rotated_images": len(exif), "rotation_better": better,
                    "median_edge_agreement_rotated": float(np.median([r for _, r, _ in exif if r])),
                    "median_edge_agreement_unrotated": float(np.median([u for _, _, u in exif if u])),
                    "pass": len(exif) == 57 and better >= 55}

    # D7 independent derivation of all processed arrays vs the project cache
    processed = read_json(ROOT / "data/manifests/processed.json")
    cache_dir = ROOT / processed["directory"]
    index = {r["id"]: r["index"] for r in processed["records"]}
    DERIVED.mkdir(parents=True, exist_ok=True)
    d7 = {}
    for split in SPLITS:
        with ProcessPoolExecutor(max_workers=WORKERS) as pool:
            derived = list(pool.map(derive, jobs[split], chunksize=4))
        ids = [d[0] for d in derived]
        rgb = np.stack([d[1] for d in derived])
        mask = np.stack([d[2] for d in derived])
        geometry = {d[0]: d[3] for d in derived}
        np.savez(DERIVED / f"{split}.npz", ids=np.array(ids), rgb=rgb, mask=mask)
        (DERIVED / f"{split}_geometry.json").write_text(json.dumps(geometry), encoding="utf-8")
        cache_rgb = np.load(cache_dir / f"{split}_rgb.npy", mmap_mode="r")
        cache_mask = np.load(cache_dir / f"{split}_mask.npy", mmap_mode="r")
        rgb_bad = mask_bad = 0
        bad_images = []
        for position, image_id in enumerate(ids):
            k = index[image_id]
            r = int(np.count_nonzero(cache_rgb[k] != rgb[position]))
            m = int(np.count_nonzero(cache_mask[k] != mask[position]))
            rgb_bad += r; mask_bad += m
            if r or m:
                bad_images.append(image_id)
        d7[split] = {"images": len(ids), "rgb_mismatch_values": rgb_bad, "mask_mismatch_pixels": mask_bad, "mismatching_images": bad_images[:20],
                     "mismatching_image_count": len(bad_images), "processed_target_pixels": [int((mask == c).sum()) for c in (1, 2)]}
    report["D7"] = {"detail": d7, "pass": all(v["rgb_mismatch_values"] == 0 and v["mask_mismatch_pixels"] == 0 for v in d7.values())}

    # D8 budgets
    budget = {b: read_json(ROOT / f"data/manifests/budget_{b}.json")["ids"] for b in (25, 100)}
    train_ids = {j[0] for j in jobs["train"]}
    other_ids = {j[0] for s in ("val", "test") for j in jobs[s]}
    derived_train = np.load(DERIVED / "train.npz")
    presence = {i: ((m == 1).any(), (m == 2).any()) for i, m in zip(derived_train["ids"], derived_train["mask"])}
    strata = lambda ids: {f"building={a},road={b}": n for (a, b), n in sorted(Counter(presence[i] for i in ids).items())}
    report["D8"] = {"counts": {str(b): len(v) for b, v in budget.items()}, "25_in_train": set(budget[25]) <= train_ids,
                    "100_equals_train": set(budget[100]) == train_ids, "nested": set(budget[25]) <= set(budget[100]),
                    "val_or_test_in_budgets": len((set(budget[25]) | set(budget[100])) & other_ids),
                    "strata_25": strata(budget[25]), "strata_train": strata(train_ids),
                    "pass": len(budget[25]) == 361 and set(budget[25]) <= train_ids and set(budget[100]) == train_ids
                    and not ((set(budget[25]) | set(budget[100])) & other_ids)}
    report["seconds"] = time.perf_counter() - started
    guard()
    write_json("data.json", report)
    print(json.dumps({k: v["pass"] for k, v in report.items() if isinstance(v, dict) and "pass" in v}))


if __name__ == "__main__":
    main()
