"""S1 paired bootstrap over test images and S2 external plausibility. Never imports `floodnet`.

Input: per-image confusion matrices from the independent evaluation (output/audit/per_image_cms_frozen.npz).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from independent import OUT, guard, write_json

RESAMPLES = 5000
SEEDS = (42, 43, 44)
# FloodNet paper (Rahnemoonfar et al., IEEE Access 2021; arXiv 2012.02951), Table 4, test set, 10 classes, 713x713 input.
PUBLISHED = {"ENet": {"building_flooded": 6.94, "road_flooded": 12.49},
             "DeepLabV3+": {"building_flooded": 32.7, "road_flooded": 52.00},
             "PSPNet (ResNet-101)": {"building_flooded": 68.93, "road_flooded": 82.16}}


def flooded_miou(pooled):
    """pooled: (..., 9) flattened 3x3 confusion counts -> mean IoU of classes 1 and 2."""
    m = pooled.reshape(*pooled.shape[:-1], 3, 3)
    tp = np.stack([m[..., c, c] for c in (1, 2)], -1)
    union = np.stack([m[..., c, :].sum(-1) + m[..., :, c].sum(-1) - m[..., c, c] for c in (1, 2)], -1)
    return (tp / union).mean(-1)


def class_iou(pooled, c):
    m = pooled.reshape(*pooled.shape[:-1], 3, 3)
    return m[..., c, c] / (m[..., c, :].sum(-1) + m[..., :, c].sum(-1) - m[..., c, c])


def main():
    data = np.load(OUT / "per_image_cms_frozen.npz")
    runs = {k: data[k].reshape(len(data["ids"]), 9).astype(np.float64) for k in data.files if k != "ids" and "__" not in k}
    n = len(data["ids"])
    rng = np.random.default_rng(2026)
    weights = rng.multinomial(n, np.full(n, 1 / n), size=RESAMPLES).astype(np.float64)  # paired: same resample for every run

    def seed_mean(method, budget, resampled=True):
        values = [flooded_miou(weights @ runs[f"{method}_{budget}_s{s}"]) if resampled else flooded_miou(runs[f"{method}_{budget}_s{s}"].sum(0))
                  for s in SEEDS]
        return np.mean(values, axis=0)

    comparisons = {
        "unet_minus_rf_25": (("unet", 25), ("rf", 25)), "unet_minus_rf_100": (("unet", 100), ("rf", 100)),
        "rf_100_minus_25": (("rf", 100), ("rf", 25)), "unet_100_minus_25": (("unet", 100), ("unet", 25)),
        "hybrid_100_minus_25": (("hybrid", 100), ("hybrid", 25)),
        "hybrid_minus_unet_25": (("hybrid", 25), ("unet", 25)), "hybrid_minus_unet_100": (("hybrid", 100), ("unet", 100)),
    }
    s1 = {}
    for name, (a, b) in comparisons.items():
        point = float(seed_mean(*a, resampled=False) - seed_mean(*b, resampled=False)) * 100
        samples = (seed_mean(*a) - seed_mean(*b)) * 100
        s1[name] = {"difference_points": point, "ci95": [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))],
                    "p_greater_than_zero": float((samples > 0).mean())}

    # Per-image wins on target-bearing test images (seed-averaged per-image flooded mIoU; undefined images skipped).
    totals = runs["unet_100_s42"].reshape(n, 3, 3).sum(2)  # truth support per class (identical for every run)
    target = (totals[:, 1] + totals[:, 2]) > 0

    def per_image(method, budget):
        stacked = []
        for s in SEEDS:
            m = runs[f"{method}_{budget}_s{s}"].reshape(n, 3, 3)
            with np.errstate(invalid="ignore", divide="ignore"):
                ious = [m[:, c, c] / (m[:, c, :].sum(1) + m[:, :, c].sum(1) - m[:, c, c]) for c in (1, 2)]
            stacked.append(np.nanmean(np.stack(ious, 1), 1))
        return np.nanmean(np.stack(stacked), 0)

    wins = {}
    for name, (a, b) in comparisons.items():
        pa, pb = per_image(*a)[target], per_image(*b)[target]
        wins[name] = {"images": int(target.sum()), "a_better": int((pa > pb).sum()), "ties": int((pa == pb).sum()), "b_better": int((pa < pb).sum())}

    unet100 = {k: float(np.mean([class_iou(runs[f"unet_100_s{s}"].sum(0), c) for s in SEEDS])) * 100 for k, c in (("building_flooded", 1), ("road_flooded", 2))}
    hybrid100 = {k: float(np.mean([class_iou(runs[f"hybrid_100_s{s}"].sum(0), c) for s in SEEDS])) * 100 for k, c in (("building_flooded", 1), ("road_flooded", 2))}
    best = PUBLISHED["PSPNet (ResNet-101)"]
    s2 = {"published": PUBLISHED, "ours_unet_100": unet100, "ours_hybrid_100_observation": hybrid100,
          "caveat": "Not like-for-like: the paper scores 10 classes at 713x713; this project scores 3 classes at 256x256 on the same official split.",
          "pass": unet100["building_flooded"] <= best["building_flooded"] and unet100["road_flooded"] <= best["road_flooded"]}
    guard()
    write_json("stats.json", {"S1": {"resamples": RESAMPLES, "comparisons": s1, "per_image_wins": wins}, "S2": s2})
    print(json.dumps({k: [round(v["difference_points"], 2), [round(x, 2) for x in v["ci95"]], v["p_greater_than_zero"]] for k, v in s1.items()}))


if __name__ == "__main__":
    main()
