import json

import numpy as np
import pytest

from floodnet.common import ROOT
from floodnet.visuals import (
    augmentation_draws,
    class_presence,
    edge_agreement,
    strata_counts,
)


def test_strata_counts_split_by_target_presence():
    records = [{"id": "a", "processed_counts": [9, 1, 1]}, {"id": "b", "processed_counts": [9, 1, 0]},
               {"id": "c", "processed_counts": [9, 0, 2]}, {"id": "d", "processed_counts": [9, 0, 0]},
               {"id": "e", "processed_counts": [9, 0, 0]}]
    assert strata_counts(records, ["a", "b", "c", "d", "e"]) == {"both": 1, "building": 1, "road": 1, "neither": 2}
    assert strata_counts(records, ["a", "d"]) == {"both": 1, "building": 0, "road": 0, "neither": 1}


def test_class_presence_counts_masks_containing_each_class():
    records = [{"mask_ids": [0, 1, 3]}, {"mask_ids": [0, 9]}, {"mask_ids": [3]}]
    assert class_presence(records) == [2, 1, 0, 2, 0, 0, 0, 0, 0, 1]


def test_edge_agreement_prefers_the_aligned_mask():
    rng = np.random.default_rng(0)
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[10:30, 5:40] = 1
    mask[40:60, 20:50] = 2
    rgb = np.stack([mask * 90 + rng.integers(0, 10, mask.shape)] * 3, -1).astype(np.uint8)
    assert edge_agreement(rgb, mask) > edge_agreement(rgb, mask[::-1, ::-1])


def test_augmentation_draws_replay_the_training_augmentation():
    from floodnet.neural import augment
    rng = np.random.default_rng(1)
    rgb = rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)
    mask = rng.integers(0, 3, (16, 16), dtype=np.uint8)
    draws = augmentation_draws(rgb, mask, seed=42, count=3, strength=0.1)
    assert len(draws) == 3
    for epoch, (image, labels, params) in enumerate(draws):
        expected = augment(rgb, mask, 42, epoch, 0, 0.1)
        assert np.array_equal(image, expected[0]) and np.array_equal(labels, expected[1])
        assert set(params) == {"flip_lr", "flip_ud", "quarter_turns", "contrast", "brightness"}


def test_exported_preprocessing_matches_the_manifests():
    path = ROOT / "web/data/preprocessing.json"
    if not path.exists():
        pytest.skip("Preprocessing page not exported")
    text = path.read_text(encoding="utf-8")
    assert "C:\\" not in text
    data = json.loads(text)
    audit = json.loads((ROOT / "data/manifests/audit.json").read_text(encoding="utf-8"))
    processed = json.loads((ROOT / "data/manifests/processed.json").read_text(encoding="utf-8"))
    steps = data["steps"]
    assert steps["acquire"]["counts"] == audit["counts"]
    assert steps["acquire"]["files_hashed"] == 2 * len(audit["records"])
    assert steps["duplicates"]["exact_groups"] == len(audit["exact_duplicates"])
    assert steps["duplicates"]["near_candidates"] == len(audit["near_duplicate_candidates"])
    assert steps["orientation"]["rotated"] == sum(r["exif_orientation"] != 1 for r in audit["records"])
    example = steps["orientation"]["example"]
    assert example["agreement_fixed"] > example["agreement_raw"]
    assert steps["budgets"]["counts"] == {k: len(v["ids"]) for k, v in processed["budgets"].items()}
    urls = [u for u in json.dumps(data).split('"') if u.startswith("data/preprocessing/")]
    assert urls and all((ROOT / "web" / u).is_file() for u in urls)
