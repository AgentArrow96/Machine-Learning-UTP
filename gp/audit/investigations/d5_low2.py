import json
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit_data import palette  # noqa: E402
from independent import OUT, RAW, ROOT  # noqa: E402

COLOUR = ROOT / "data/raw/ColorMasks-FloodNetv1.0"
DIRS = {"train": "ColorMasks-TrainSet", "val": "ColorMasks-ValSet", "test": "ColorMasks-TestSet"}
EXPECTED = ["background", "building-flooded", "building-non-flooded", "road-flooded", "road-non-flooded", "water", "tree", "vehicle", "pool", "grass"]


def agreement(image_id):
    names = palette()
    split, key = image_id.split("_", 1)
    colour = np.asarray(Image.open(COLOUR / DIRS[split] / f"{key}_lab.png").convert("RGB"))
    mask = np.asarray(Image.open(RAW / split / f"{split}-label-img" / f"{key}_lab.png"))
    with Image.open(RAW / split / f"{split}-org-img" / f"{key}.jpg") as photo:
        exif = photo.getexif().get(274, 1)
    lut = np.array([names[n] for n in EXPECTED], dtype=np.uint8)
    out = {"exif": exif}
    for name, m in (("top_left", mask[:3000, :4000]), ("rot180_top_left", mask[::-1, ::-1][:3000, :4000]), ("bottom_right", mask[-3000:, -4000:])):
        small = np.asarray(Image.fromarray(m).resize((colour.shape[1], colour.shape[0]), Image.Resampling.NEAREST))
        out[name] = float((lut[small] == colour).all(-1).mean())
    return image_id, out


if __name__ == "__main__":
    rows = json.loads((OUT / "d5_large_photo_agreement.json").read_text(encoding="utf-8"))
    low = [r["id"] for r in rows if r["agreement"] < .99]
    with ProcessPoolExecutor(12) as pool:
        results = list(pool.map(agreement, low))
    for image_id, r in results:
        print(image_id, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
    (OUT / "d5_low_agreement_root_cause.json").write_text(json.dumps(dict(results), indent=1), encoding="utf-8")
