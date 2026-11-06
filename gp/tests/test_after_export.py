"""The app's "after" bundle (web/data/after.json): the pre-registered combined models, shaped like results.json."""
import json

import numpy as np
import pytest
from PIL import Image

from floodnet.common import ROOT, config

AFTER = ROOT / "web/data/after.json"
needs_export = pytest.mark.skipif(not AFTER.exists(), reason="needs `floodnet improve export-after`")


def bundle():
    return json.loads(AFTER.read_text(encoding="utf-8"))


def test_after_run_ids_are_namespaced_and_never_collide_with_baseline_ids():
    from floodnet.improve import after_run_id
    assert after_run_id("unet", 25, 42) == "improve/combined/unet_25_s42"
    assert after_run_id("unet", 25, 42) != "unet_25_s42"


@needs_export
def test_after_summary_means_equal_the_frozen_test_evaluations():
    out = bundle()
    assert len(out["runs"]) == 18 and len({r["id"] for r in out["runs"]}) == 18
    for item in out["summary"]:
        scores = [json.loads((ROOT / f"runs/improve/combined/{item['method']}_{item['budget']}_s{s}/evaluation_test.json")
                             .read_text(encoding="utf-8"))["metrics"]["flooded_miou"] for s in item["seeds"]]
        assert item["full"]["flooded_miou"]["mean"] == pytest.approx(np.mean(scores), abs=1e-12)


@needs_export
def test_after_comparisons_are_after_minus_before_of_the_same_seeds():
    out = bundle()
    assert {(c["method"], c["budget"]) for c in out["comparisons"]} == {(m, b) for m in ("rf", "unet", "hybrid") for b in (25, 100)}
    for c in out["comparisons"]:
        after = next(s for s in out["summary"] if s["method"] == c["method"] and s["budget"] == c["budget"])
        assert after["full"]["flooded_miou"]["values"] == pytest.approx(c["variant_scores"], abs=1e-12)
        assert np.subtract(c["variant_scores"], c["baseline_scores"]) == pytest.approx(c["differences"], abs=1e-12)


@needs_export
def test_after_explorer_pngs_equal_the_stored_predictions():
    from floodnet.data import Store, crop
    out, store = bundle(), Store(config())
    records = store.select("test")[::40]
    for run_id in out["explorer_runs"]:
        result = json.loads((ROOT / "runs" / run_id / "evaluation_test.json").read_text(encoding="utf-8"))
        predictions = np.load(ROOT / result["predictions"], mmap_mode="r")
        index = {item["id"]: item["prediction_index"] for item in result["per_image"]}
        for record in records:
            url = out["samples"][record["id"]]["predictions"][run_id]
            assert not url.startswith("/") and ":" not in url
            with Image.open(ROOT / "web" / url) as image:
                np.testing.assert_array_equal(np.asarray(image), crop(np.asarray(predictions[index[record["id"]]]), record["geometry"]))
