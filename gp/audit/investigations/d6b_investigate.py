"""Root-cause D6b: is the mask misaligned, or is the edge metric weak? Check where alignment peaks."""
import json
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit_data import edge_agreement  # noqa: E402
from independent import OUT, RAW  # noqa: E402

OFFSETS = [(dy, dx) for dy in range(-24, 25, 8) for dx in range(-24, 25, 8)]


def shifted(mask, dy, dx):
    out = np.roll(mask, (dy, dx), axis=(0, 1))
    return out


def job(image_id):
    split, key = image_id.split("_", 1)
    with Image.open(RAW / split / f"{split}-org-img" / f"{key}.jpg") as image:
        rgb = np.asarray(ImageOps.exif_transpose(image).convert("RGB").resize((800, 600), Image.Resampling.BILINEAR))
    with Image.open(RAW / split / f"{split}-label-img" / f"{key}_lab.png") as mask_image:
        mask = np.asarray(mask_image.resize((800, 600), Image.Resampling.NEAREST))
    scores = {f"{dy},{dx}": edge_agreement(rgb, shifted(mask, dy, dx)) for dy, dx in OFFSETS}
    transforms = {"rot180": mask[::-1, ::-1], "flipud": mask[::-1], "fliplr": mask[:, ::-1]}
    scores.update({k: edge_agreement(rgb, v) for k, v in transforms.items()})
    best = max(scores, key=lambda k: scores[k] if scores[k] is not None else -1)
    classes = len(np.unique(mask))
    return image_id, scores["0,0"], best, scores[best], classes


if __name__ == "__main__":
    # data.json keeps only 20 failing IDs; recompute the full list with the same seeded derangement as D6b.
    from audit_data import alignment_job  # noqa: E402
    from independent import SPLITS, discover  # noqa: E402
    pairs = discover()
    # Same numeric order as audit_data.py, so the seeded derangement (and the failing set) is identical to D6b's.
    order = lambda key: int(key) if key.isdigit() else key
    jobs = [(f"{s}_{k}", pairs[s]["images"][k], pairs[s]["masks"][k]) for s in SPLITS
            for k in sorted(set(pairs[s]["images"]) & set(pairs[s]["masks"]), key=order)]
    rng = np.random.default_rng(5)
    perm = rng.permutation(len(jobs))
    while np.any(perm == np.arange(len(jobs))):
        perm = rng.permutation(len(jobs))
    with ProcessPoolExecutor(12) as pool:
        aligned = list(pool.map(alignment_job, [(jobs[k], jobs[perm[k]]) for k in range(len(jobs))], chunksize=8))
    not_better = [i for i, a, b in aligned if a is not None and b is not None and a <= b]
    with ProcessPoolExecutor(12) as pool:
        rows = list(pool.map(job, not_better, chunksize=4))
    at_zero = [r for r in rows if r[2] == "0,0"]
    near_zero = [r for r in rows if r[2] in {f"{dy},{dx}" for dy in (-8, 0, 8) for dx in (-8, 0, 8)}]
    transformed = [r for r in rows if r[2] in ("rot180", "flipud", "fliplr")]
    far = [r for r in rows if r not in near_zero and r not in transformed]
    summary = {"not_better_count": len(not_better), "peak_at_zero_offset": len(at_zero), "peak_within_8px": len(near_zero),
               "peak_at_flip_or_rotation": [r[0] for r in transformed], "peak_far_shift": [(r[0], r[2], round(r[3], 3), round(r[1], 3)) for r in far],
               "median_own_score": float(np.median([r[1] for r in rows])), "class_counts": {int(c): sum(r[4] == c for r in rows) for c in sorted({r[4] for r in rows})}}
    (OUT / "d6b_root_cause.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=1)[:3000])
