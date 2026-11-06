"""Quantify and visualise the flip/rotation candidates from the D6b investigation."""
import json
import os
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit_data import edge_agreement  # noqa: E402
from independent import OUT, RAW, ROOT  # noqa: E402


def load(image_id, size=(800, 600)):
    split, key = image_id.split("_", 1)
    with Image.open(RAW / split / f"{split}-org-img" / f"{key}.jpg") as image:
        exif = image.getexif().get(274, 1)
        raw_size = image.size
        rgb = np.asarray(ImageOps.exif_transpose(image).convert("RGB").resize(size, Image.Resampling.BILINEAR))
    with Image.open(RAW / split / f"{split}-label-img" / f"{key}_lab.png") as mask_image:
        mask = np.asarray(mask_image.resize(size, Image.Resampling.NEAREST))
    return rgb, mask, exif, raw_size


def job(image_id):
    rgb, mask, exif, raw_size = load(image_id)
    s = {"own": edge_agreement(rgb, mask), "rot180": edge_agreement(rgb, mask[::-1, ::-1]),
         "flipud": edge_agreement(rgb, mask[::-1]), "fliplr": edge_agreement(rgb, mask[:, ::-1])}
    flooded = int(((mask == 1) | (mask == 3)).sum())
    return image_id, s, exif, list(raw_size), flooded, len(np.unique(mask))


if __name__ == "__main__":
    summary = json.loads((OUT / "d6b_root_cause.json").read_text(encoding="utf-8"))
    ids = summary["peak_at_flip_or_rotation"]
    with ProcessPoolExecutor(12) as pool:
        rows = list(pool.map(job, ids))
    ratio = [(r[0], max(r[1]["rot180"], r[1]["flipud"], r[1]["fliplr"]) / r[1]["own"], max(("rot180", "flipud", "fliplr"), key=lambda k: r[1][k]), r) for r in rows]
    ratio.sort(key=lambda t: -t[1])
    strong = [t for t in ratio if t[1] > 1.5]
    print("candidates", len(rows), "ratio>1.5:", len(strong), "ratio>1.25:", sum(t[1] > 1.25 for t in ratio))
    print("best transform counts", {k: sum(t[2] == k for t in ratio) for k in ("rot180", "flipud", "fliplr")})
    print("exif of candidates", {e: sum(t[3][2] == e for t in ratio) for e in {t[3][2] for t in ratio}})
    print("sizes", {tuple(s): sum(tuple(t[3][3]) == tuple(s) for t in ratio) for s in {tuple(t[3][3]) for t in ratio}})
    print("top 12:", [(t[0], round(t[1], 2), t[2], round(t[3][1]["own"], 2), t[3][4]) for t in ratio[:12]])
    print("bottom 6:", [(t[0], round(t[1], 3), t[2]) for t in ratio[-6:]])
    os.environ["MPLCONFIGDIR"] = str(ROOT / ".cache/matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    palette = np.array([[0, 0, 0], [255, 0, 0], [180, 120, 120], [160, 150, 20], [140, 140, 140], [61, 230, 250], [0, 82, 255], [255, 0, 245], [255, 235, 0], [4, 250, 7]], np.float32)
    show = [t[0] for t in ratio[:4]] + [t[0] for t in ratio[len(ratio) // 2:len(ratio) // 2 + 2]]
    fig, axes = plt.subplots(len(show), 3, figsize=(12, 3 * len(show)), layout="constrained")
    for r, image_id in enumerate(show):
        rgb, mask, exif, _ = load(image_id, (400, 300))
        best = next(t for t in ratio if t[0] == image_id)
        transformed = {"rot180": mask[::-1, ::-1], "flipud": mask[::-1], "fliplr": mask[:, ::-1]}[best[2]]
        for c, (title, m) in enumerate((("photo", None), ("mask as published", mask), (f"mask {best[2]}", transformed))):
            image = rgb if m is None else np.round(rgb * .45 + palette[m] * .55).astype(np.uint8)
            axes[r, c].imshow(image); axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
            axes[r, c].set_title(f"{image_id} EXIF={exif} | {title}" + (f" | ratio {best[1]:.2f}" if c == 2 else ""), fontsize=8)
    fig.savefig(OUT / "figures/d6b_orientation_candidates.png", dpi=100, facecolor="white")
    (OUT / "d6b_orientation.json").write_text(json.dumps([{"id": t[0], "best_transform_over_own": t[1], "transform": t[2], "scores": t[3][1], "exif": t[3][2],
                                                           "flooded_pixels": t[3][4]} for t in ratio], indent=1), encoding="utf-8")
