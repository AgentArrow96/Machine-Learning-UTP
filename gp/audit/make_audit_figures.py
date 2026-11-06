"""Proof figures from the audit's own predictions: a seeded RANDOM sample (not curated), worst cases,
and randomly chosen verified train-test overlap pairs. Never imports `floodnet`."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from independent import DERIVED, OUT, ROOT, confusion, guard, read_json, scores

os.environ["MPLCONFIGDIR"] = str(ROOT / ".cache/matplotlib")
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PALETTE = np.array([[148, 163, 184], [245, 158, 11], [8, 145, 178]], dtype=np.float32)
PANELS = (("rf_100_s42", "SLIC + RF"), ("unet_100_s42", "U-Net"), ("hybrid_100_s42", "U-Net + CRF"))
FIGURES = OUT / "figures"


def overlay(rgb, labels, valid):
    alpha = np.where(labels == 0, .08, .6)[..., None] * valid[..., None]
    colours = PALETTE[np.where(valid, labels, 0)]
    return np.round(rgb * (1 - alpha) + colours * alpha).astype(np.uint8)


def crop(array, g):
    return array[g["top"]:g["top"] + g["height"], g["left"]:g["left"] + g["width"]]


def grid(rows, title, path, predictions, rgb_all, masks, geometry, ids):
    fig, axes = plt.subplots(len(rows), 5, figsize=(15, 2.45 * len(rows)), layout="constrained")
    axes = np.atleast_2d(axes)
    for r, k in enumerate(rows):
        g = geometry[ids[k]]
        valid = masks[k] != 255
        panels = [("RGB", crop(rgb_all[k], g)), ("Ground truth", crop(overlay(rgb_all[k], masks[k], valid), g))]
        for name, label in PANELS:
            s = scores(confusion(masks[k], predictions[name][k]))["flooded_miou"]
            panels.append((f"{label}  IoU {'n/a' if s is None else f'{s * 100:.0f}%'}", crop(overlay(rgb_all[k], predictions[name][k], valid), g)))
        for c, (label, image) in enumerate(panels):
            axes[r, c].imshow(image)
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
            axes[r, c].set_title(label if c else f"{ids[k]} | {label}", fontsize=8)
    fig.suptitle(title, fontsize=11)
    fig.savefig(path, dpi=110, facecolor="white")
    plt.close(fig)


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    data = np.load(DERIVED / "test.npz")
    ids = [str(i) for i in data["ids"]]
    rgb_all, masks = data["rgb"], data["mask"]
    geometry = json.loads((DERIVED / "test_geometry.json").read_text(encoding="utf-8"))
    stored = np.load(OUT / "predictions_s42_frozen.npz")
    pred_ids = [str(i) for i in stored["ids"]]
    assert pred_ids == ids
    predictions = {name: stored[name] for name, _ in PANELS}
    valid = masks != 255
    share = np.array([((m == 1) | (m == 2))[v].mean() for m, v in zip(masks, valid)])
    rng = np.random.default_rng(31)
    targets = np.flatnonzero(share > .01)
    empty = np.flatnonzero(share == 0)
    random_rows = sorted(rng.choice(targets, 12, replace=False).tolist()) + sorted(rng.choice(empty, 4, replace=False).tolist())
    grid(random_rows, "Seeded random sample (seed 31): 12 test images with >1% flooded pixels + 4 without (100% labels, training seed 42)",
         FIGURES / "random_sample.png", predictions, rgb_all, masks, geometry, ids)
    worst = {}
    for name, label in PANELS:
        per_image = [(scores(confusion(masks[k], predictions[name][k]))["flooded_miou"] or 0, k) for k in targets]
        rows = [k for _, k in sorted(per_image)[:5]]
        worst[name] = [ids[k] for k in rows]
        grid(rows, f"Five worst target images for {label} (lowest per-image flooded mIoU)", FIGURES / f"worst_{name}.png",
             predictions, rgb_all, masks, geometry, ids)
    # Randomly chosen verified train-test overlap pairs, drawn from the independently derived arrays.
    overlap = read_json(ROOT / "data/manifests/overlap.json")
    verified = [p for p in overlap["pairs"] if p["verified"] and p["eval_id"].startswith("test_")]
    picks = [verified[i] for i in rng.choice(len(verified), 12, replace=False)]
    train = np.load(DERIVED / "train.npz")
    train_index = {str(i): k for k, i in enumerate(train["ids"])}
    train_geometry = json.loads((DERIVED / "train_geometry.json").read_text(encoding="utf-8"))
    test_index = {i: k for k, i in enumerate(ids)}
    fig, axes = plt.subplots(4, 6, figsize=(16, 9), layout="constrained")
    for n, pair in enumerate(picks):
        r, c = divmod(n, 3)
        a = crop(rgb_all[test_index[pair["eval_id"]]], geometry[pair["eval_id"]])
        b = crop(train["rgb"][train_index[pair["train_id"]]], train_geometry[pair["train_id"]])
        for offset, (image, label) in enumerate(((a, pair["eval_id"]), (b, pair["train_id"]))):
            ax = axes[r, c * 2 + offset]
            ax.imshow(image); ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{label}" + (f"  ({pair['overlap'] * 100:.0f}% shared, {pair['inliers']} inliers)" if offset == 0 else ""), fontsize=7)
    fig.suptitle("12 randomly chosen verified test-train overlap pairs (seeded): left = test frame, right = training frame", fontsize=11)
    fig.savefig(FIGURES / "overlap_pairs_random.png", dpi=100, facecolor="white")
    plt.close(fig)
    guard()
    (OUT / "figures.json").write_text(json.dumps({"random_sample": [ids[k] for k in random_rows], "worst": worst,
                                                   "overlap_pairs": [[p["eval_id"], p["train_id"]] for p in picks]}, indent=2), encoding="utf-8")
    print("figures written to", FIGURES)


if __name__ == "__main__":
    main()
