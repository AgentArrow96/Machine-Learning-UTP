import json

import numpy as np
import pytest

from floodnet.common import ROOT, ProjectError
from floodnet.visuals import (
    check_aspect,
    decode_regions,
    encode_regions,
    pca_rgb,
    scaled_window,
    walk_urls,
)

def test_pca_colours_are_deterministic():
    rng = np.random.default_rng(7)
    features = rng.normal(size=(16, 10, 12)).astype(np.float32)
    first = pca_rgb(features)
    assert first.shape == (10, 12, 3) and first.dtype == np.uint8
    assert np.array_equal(first, pca_rgb(features.copy()))
    assert pca_rgb(features[:2]).shape == (10, 12, 3)


def test_region_map_round_trips_through_rgb():
    regions = np.array([[-1, 0, 1], [255, 256, 1000]])
    assert np.array_equal(decode_regions(encode_regions(regions)), regions)


def test_sharp_photo_must_show_the_processed_frame():
    geometry = {"width": 256, "height": 171, "top": 42, "left": 0}
    check_aspect((1024, 685), geometry)
    with pytest.raises(ProjectError):
        check_aspect((1024, 768), geometry)


def test_feature_windows_follow_the_unpadded_rectangle():
    geometry = {"width": 256, "height": 192, "top": 32, "left": 0}
    assert scaled_window(geometry, 256, 8) == {"top": 1, "left": 0, "height": 6, "width": 8}
    assert scaled_window(geometry, 256, 256) == {"top": 32, "left": 0, "height": 192, "width": 256}


def test_exported_walkthrough_is_portable_and_verified():
    path = ROOT / "web/data/walkthrough.json"
    if not path.exists():
        pytest.skip("Walkthrough not exported")
    text = path.read_text(encoding="utf-8")
    assert "C:\\" not in text and "/runs/" not in text and ".pt\"" not in text
    walk = json.loads(text)
    urls = list(walk_urls(walk))
    assert urls and all((ROOT / "web" / u).is_file() for u in urls)
    checks = [m["check"] for e in walk["examples"] for b in e["budgets"].values() for m in b.values()]
    assert len(checks) == len(walk["examples"]) * 2 * 2
    assert all(c["matches_frozen_prediction"] and c["differing"] == 0 for c in checks)
