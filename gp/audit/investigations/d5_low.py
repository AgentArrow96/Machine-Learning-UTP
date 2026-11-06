import json
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit_data import colour_job, palette  # noqa: E402
from independent import OUT, ROOT, SPLITS, discover  # noqa: E402

COLOUR = ROOT / "data/raw/ColorMasks-FloodNetv1.0"
DIRS = {"train": "ColorMasks-TrainSet", "val": "ColorMasks-ValSet", "test": "ColorMasks-TestSet"}

if __name__ == "__main__":
    candidates = json.loads((OUT / "d6b_orientation.json").read_text(encoding="utf-8"))
    print("D6b candidates with flooded pixels:", sum(c["flooded_pixels"] > 0 for c in candidates), "of", len(candidates),
          "| strongest (ratio>1.25) with flooded pixels:", sum(c["flooded_pixels"] > 0 and c["best_transform_over_own"] > 1.25 for c in candidates))
    pairs = discover()
    names = palette()
    pack = lambda c: (c[0] << 16) | (c[1] << 8) | c[2]
    expected = {0: "background", 1: "building-flooded", 2: "building-non-flooded", 3: "road-flooded", 4: "road-non-flooded",
                5: "water", 6: "tree", 7: "vehicle", 8: "pool", 9: "grass"}
    jobs = []
    for split in SPLITS:
        for key, mask in pairs[split]["masks"].items():
            with Image.open(mask) as m:
                if m.size == (4592, 3072):
                    jobs.append((f"{split}_{key}", mask, COLOUR / DIRS[split] / f"{key}_lab.png"))
    with ProcessPoolExecutor(12) as pool:
        tables = list(pool.map(colour_job, jobs, chunksize=4))
    rows = []
    for image_id, table, size in tables:
        hit = sum(counts.get(pack(names[expected[v]]), 0) for v, counts in table.items())
        total = sum(sum(c.values()) for c in table.values())
        rows.append((hit / total, image_id))
    rows.sort()
    print("4592x3072 colour masks checked:", len(rows), "| below 99%:", sum(a < .99 for a, _ in rows), "| lowest 5:", [(i, round(a, 3)) for a, i in rows[:5]])
    (OUT / "d5_large_photo_agreement.json").write_text(json.dumps([{"id": i, "agreement": a} for a, i in rows], indent=1), encoding="utf-8")
