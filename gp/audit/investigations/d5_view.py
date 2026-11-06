import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit_data import edge_agreement  # noqa: E402
from independent import OUT, RAW, ROOT  # noqa: E402

os.environ["MPLCONFIGDIR"] = str(ROOT / ".cache/matplotlib")
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COLOUR = ROOT / "data/raw/ColorMasks-FloodNetv1.0"
DIRS = {"train": "ColorMasks-TrainSet", "val": "ColorMasks-ValSet", "test": "ColorMasks-TestSet"}
PALETTE = np.array([[0, 0, 0], [255, 0, 0], [180, 120, 120], [160, 150, 20], [140, 140, 140], [61, 230, 250], [0, 82, 255], [255, 0, 245], [255, 235, 0], [4, 250, 7]], np.float32)

ids = ["test_7450", "val_6630", "train_6600"]
fig, axes = plt.subplots(len(ids), 3, figsize=(13, 3.4 * len(ids)), layout="constrained")
for r, image_id in enumerate(ids):
    split, key = image_id.split("_", 1)
    with Image.open(RAW / split / f"{split}-org-img" / f"{key}.jpg") as photo:
        exif = photo.getexif().get(274, 1)
        rgb_full = ImageOps.exif_transpose(photo).convert("RGB")
    mask_img = Image.open(RAW / split / f"{split}-label-img" / f"{key}_lab.png")
    rgb = np.asarray(rgb_full.crop((0, 0, 4000, 3000)).resize((400, 300), Image.Resampling.BILINEAR)).astype(np.float32)
    mask = np.asarray(mask_img.crop((0, 0, 4000, 3000)).resize((400, 300), Image.Resampling.NEAREST))
    colour = np.asarray(Image.open(COLOUR / DIRS[split] / f"{key}_lab.png").convert("RGB").resize((400, 300), Image.Resampling.NEAREST)).astype(np.float32)
    full_rgb = np.asarray(rgb_full.resize((800, 600), Image.Resampling.BILINEAR))
    own = edge_agreement(full_rgb, np.asarray(mask_img.resize((800, 600), Image.Resampling.NEAREST)))
    ccode = np.asarray(Image.open(COLOUR / DIRS[split] / f"{key}_lab.png").convert("L").resize((800, 600), Image.Resampling.NEAREST))
    colour_edges = edge_agreement(np.asarray(rgb_full.crop((0, 0, 4000, 3000)).resize((800, 600), Image.Resampling.BILINEAR)), ccode)
    panels = [("photo (top-left 4000x3000)", rgb), (f"indexed mask (used) | edge agreement {own:.2f}", rgb * .45 + PALETTE[mask] * .55),
              (f"official colour mask | edge agreement {colour_edges:.2f}", rgb * .45 + colour * .55)]
    for c, (title, image) in enumerate(panels):
        axes[r, c].imshow(np.clip(image, 0, 255).astype(np.uint8)); axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
        axes[r, c].set_title(f"{image_id} EXIF={exif} | {title}", fontsize=8)
fig.savefig(OUT / "figures/d5_low_agreement_examples.png", dpi=100, facecolor="white")
print("saved")
