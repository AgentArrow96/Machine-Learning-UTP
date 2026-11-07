import io
import zipfile
import numpy as np
import pytest
from PIL import Image

from floodnet.common import ProjectError, inside
from floodnet.acquire import safe_members
from floodnet.data import remap, resize_pair, subset_ids, mask_array, preprocessing_key
from floodnet.metrics import confusion, score
from floodnet.features import extract_regions, FEATURE_NAMES


def test_remap_and_invalid_encoding(tmp_path):
    original = np.arange(10, dtype=np.uint8)[None]
    assert remap(original).tolist() == [[0, 1, 0, 2, 0, 0, 0, 0, 0, 0]]
    with pytest.raises(ProjectError):
        remap(np.array([[255]], dtype=np.uint8))
    path = tmp_path / "coloured.png"
    Image.new("RGB", (4, 4), "red").save(path)
    with pytest.raises(ProjectError, match="indexed"):
        mask_array(path)


def test_pair_resize_padding_and_nearest_labels():
    mask = np.array([[1, 1, 2, 2], [1, 1, 2, 2]], dtype=np.uint8)
    rgb = np.stack([mask * 80] * 3, axis=-1)
    image, labels, g = resize_pair(rgb, mask, 8)
    assert g == {"original_width": 4, "original_height": 2, "width": 8, "height": 4, "left": 0, "top": 2}
    assert np.all(labels[:2] == 255) and np.all(labels[6:] == 255)
    assert set(np.unique(labels)) == {1, 2, 255}
    assert labels[3, 1] == 1 and labels[3, 6] == 2
    predicted = labels.copy(); predicted[labels == 255] = 99
    metrics = score(confusion(labels, predicted))
    assert metrics["valid_pixels"] == 32 and metrics["flooded_miou"] == 1


def test_metrics_hand_calculated_false_positives_and_absence():
    truth = np.array([[0, 1, 1], [2, 2, 255]], dtype=np.uint8)
    predicted = np.array([[0, 1, 0], [2, 0, 9]], dtype=np.uint8)
    actual = score(confusion(truth, predicted))
    assert actual["confusion_matrix"] == [[1, 0, 0], [1, 1, 0], [1, 0, 1]]
    assert actual["flooded_miou"] == .5
    assert actual["per_class"][1]["dice"] == pytest.approx(2/3)
    assert actual["per_class"][2]["recall"] == .5
    absent = score(confusion(np.zeros((2, 2), dtype=np.uint8), np.zeros((2, 2), dtype=np.uint8)))
    assert absent["flooded_miou"] is None and absent["per_class"][1]["recall"] is None
    false_positive = score(confusion(np.array([[0]]), np.array([[1]])))
    assert false_positive["per_class"][1]["iou"] == 0 and false_positive["per_class"][1]["recall"] is None
    with pytest.raises(ProjectError):
        confusion(np.array([[1]]), np.array([[255]]))


def test_budgets_are_nested_reproducible_and_target_represented():
    records = [{"id": f"train_{i:04}", "processed_counts": [90, int(i % 4 in (0, 1))*5, int(i % 4 in (0, 2))*5]} for i in range(101)]
    limited = subset_ids(records, 42, 25)
    assert len(limited) == 25 and limited == subset_ids(records[::-1], 42, 25)
    assert set(limited) < set(subset_ids(records, 42, 100))
    selected = [r for r in records if r["id"] in limited]
    assert all(any(r["processed_counts"][c] for r in selected) for c in (1, 2))


def test_cache_identity_changes_for_sources_and_geometry(cfg):
    first = preprocessing_key(cfg, {"identity": "first"})
    assert first != preprocessing_key(cfg, {"identity": "second"})
    cfg["preprocess_version"] += 1
    assert first != preprocessing_key(cfg, {"identity": "first"})


def test_exif_rgb_alignment_with_display_coordinate_masks(project):
    from floodnet.common import relative
    from floodnet.data import prepare_record, inspect_record
    root, _ = project
    rgb = np.zeros((12, 20, 3), dtype=np.uint8)
    rgb[:, :10] = [230, 10, 10]; rgb[:, 10:] = [10, 10, 230]
    exif = Image.Exif(); exif[274] = 3
    image_path, mask_path = root / 'rotated.jpg', root / 'mask.png'
    Image.fromarray(rgb).save(image_path, exif=exif)
    mask = np.zeros((12, 20), dtype=np.uint8); mask[:, 10:] = 1
    Image.fromarray(mask).save(mask_path)
    record = inspect_record({'id':'train_rotated','split':'train','image':relative(image_path),'mask':relative(mask_path)})
    processed, labels, _ = prepare_record(record, 20)
    assert record['exif_orientation'] == 3
    assert processed[10, 15, 0] > processed[10, 5, 0]
    assert labels[10, 15] == 1 and labels[10, 5] == 0


def test_superpixels_exclude_padding_and_labels():
    rgb = np.full((32, 32, 3), 127, dtype=np.uint8)
    truth = np.full((32, 32), 255, dtype=np.uint8)
    truth[4:28, :] = 1
    geometry = {"top": 4, "left": 0, "height": 24, "width": 32}
    settings = {"n_segments": 12, "compactness": 10, "max_num_iter": 10}
    x, y, regions = extract_regions(rgb, geometry, settings, truth)
    rgb[:4] = 255; rgb[28:] = 0
    x2, y2, regions2 = extract_regions(rgb, geometry, settings, truth)
    assert x.shape[1] == len(FEATURE_NAMES) == 22
    assert np.isfinite(x).all() and np.all(y == 1)
    assert np.all(regions[:4] == -1) and np.all(regions[28:] == -1)
    np.testing.assert_array_equal(x, x2)
    np.testing.assert_array_equal(regions, regions2)


def test_archive_root_sentinel_and_traversal_rejection():
    for bad in ("../outside", "/absolute", "C:/drive", "ok/../../outside"):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as z:
            z.writestr(bad, "x")
        buffer.seek(0)
        with zipfile.ZipFile(buffer) as z, pytest.raises(ProjectError):
            safe_members(z)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as z:
        z.writestr("/", ""); z.writestr("safe/file", "x")
    buffer.seek(0)
    with zipfile.ZipFile(buffer) as z:
        assert [x.filename for x in safe_members(z)] == ["safe/file"]


def test_outputs_cannot_escape_project():
    with pytest.raises(ProjectError):
        inside("../forbidden.txt")


def test_atomic_records_retry_transient_windows_locks(project, monkeypatch):
    from pathlib import Path
    import floodnet.common as common
    root, _ = project
    original = Path.replace
    attempts = []
    def occasionally_locked(self, target):
        attempts.append(1)
        if len(attempts) < 3:
            raise PermissionError('transient test lock')
        return original(self, target)
    monkeypatch.setattr(Path, 'replace', occasionally_locked)
    monkeypatch.setattr(common.time, 'sleep', lambda _: None)
    common.write_json('atomic.json', {'valid':True})
    assert common.read_json('atomic.json') == {'valid':True} and len(attempts) == 3


def test_augmentation_alignment_and_resume_determinism():
    from floodnet.neural import augment
    mask = np.zeros((8, 8), dtype=np.uint8); mask[:3, :4] = 1
    rgb = np.stack([mask * 100] * 3, -1)
    first_rgb, first_mask = augment(rgb, mask, 42, 3, 7, 0)
    second_rgb, second_mask = augment(rgb, mask, 42, 3, 7, 0)
    np.testing.assert_array_equal(first_mask, first_rgb[..., 0] // 100)
    np.testing.assert_array_equal(first_mask, second_mask)
    np.testing.assert_array_equal(first_rgb, second_rgb)


def test_cuda_required(monkeypatch):
    import torch
    from floodnet.neural import require_cuda
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ProjectError, match="CPU fallback is disabled"):
        require_cuda()


def test_epoch_sampler_resumes_independently_of_worker_lifetime():
    from floodnet.neural import EpochSampler
    class Epoch:
        value = 3
    class Samples:
        epoch = Epoch()
        def __len__(self):
            return 17
    first = list(EpochSampler(Samples(), 42))
    resumed = list(EpochSampler(Samples(), 42))
    assert first == resumed and sorted(first) == list(range(17))
    changed = Samples(); changed.epoch = type('Epoch', (), {'value':4})()
    assert list(EpochSampler(changed, 42)) != first


def test_report_forest_uses_the_unet_class_weights_and_every_region():
    """configs/base.json: the forest handles the imbalance like the U-Net (same capped weights, no region cap)."""
    from floodnet.common import config
    from floodnet.neural import class_weights
    from floodnet.training import rf_class_weight
    cfg = config()
    assert cfg["rf"]["class_weight"] == "pixel_clipped" and cfg["rf"]["max_regions"] >= 10 ** 9
    records = [{"processed_counts": [900, 30, 70]}, {"processed_counts": [500, 0, 20]}]
    expected = class_weights(records, cfg["unet"]["weight_clip"])
    assert rf_class_weight(cfg, records) == {c: pytest.approx(float(expected[c])) for c in range(3)}
