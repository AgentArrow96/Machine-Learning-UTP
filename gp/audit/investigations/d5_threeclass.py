import json
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit_data import palette  # noqa: E402
from independent import OUT, RAW, ROOT, remap  # noqa: E402

COLOUR = ROOT / "data/raw/ColorMasks-FloodNetv1.0"
DIRS = {"train": "ColorMasks-TrainSet", "val": "ColorMasks-ValSet", "test": "ColorMasks-TestSet"}
EXPECTED = ["background", "building-flooded", "building-non-flooded", "road-flooded", "road-non-flooded", "water", "tree", "vehicle", "pool", "grass"]


def job(image_id):
    names = palette()
    split, key = image_id.split("_", 1)
    colour = np.asarray(Image.open(COLOUR / DIRS[split] / f"{key}_lab.png").convert("RGB"))
    mask = Image.open(RAW / split / f"{split}-label-img" / f"{key}_lab.png")
    small = np.asarray(mask.crop((0, 0, min(mask.size[0], 4000), min(mask.size[1], 3000))).resize((colour.shape[1], colour.shape[0]), Image.Resampling.NEAREST))
    packed = (colour[..., 0].astype(np.int64) << 16) | (colour[..., 1].astype(np.int64) << 8) | colour[..., 2]
    colour_ids = np.full(packed.shape, 255, np.uint8)
    for value, name in enumerate(EXPECTED):
        r, g, b = names[name]
        colour_ids[packed == ((r << 16) | (g << 8) | b)] = value
    ten = float((colour_ids == small).mean())
    three_a, three_b = remap(small), remap(np.where(colour_ids == 255, 0, colour_ids))
    flooded_in_either = (three_a > 0) | (three_b > 0)
    disagreements = {f"{EXPECTED[a]}->{EXPECTED[b]}": int(((small == a) & (colour_ids == b)).sum())
                     for a in range(10) for b in range(10) if a != b and ((small == a) & (colour_ids == b)).any()}
    return image_id, {"ten_class_agreement": ten, "three_class_agreement": float((three_a == three_b).mean()),
                      "flooded_pixels_indexed": int((three_a > 0).sum()), "flooded_pixels_colour": int((three_b > 0).sum()),
                      "flooded_disagreement_pixels": int((flooded_in_either & (three_a != three_b)).sum()),
                      "top_relabels": dict(sorted(disagreements.items(), key=lambda kv: -kv[1])[:3])}


if __name__ == "__main__":
    low = [r["id"] for r in json.loads((OUT / "d5_large_photo_agreement.json").read_text(encoding="utf-8")) if r["agreement"] < .99]
    with ProcessPoolExecutor(12) as pool:
        results = dict(pool.map(job, low))
    for k, v in results.items():
        print(k, round(v["ten_class_agreement"], 3), round(v["three_class_agreement"], 4), v["flooded_disagreement_pixels"], v["top_relabels"])
    total_flooded_disagreement = sum(v["flooded_disagreement_pixels"] for v in results.values())
    print("images:", len(results), "| min 3-class agreement:", round(min(v["three_class_agreement"] for v in results.values()), 4),
          "| images with any flooded-class disagreement:", sum(v["flooded_disagreement_pixels"] > 0 for v in results.values()),
          "| total flooded disagreement pixels (1024px scale):", total_flooded_disagreement)
    (OUT / "d5_three_class_check.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
