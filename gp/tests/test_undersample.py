import copy

import numpy as np
import pytest

from floodnet.common import ROOT, config


def real_data():
    return (ROOT / "data/manifests/processed.json").exists()


needs_data = pytest.mark.skipif(not real_data(), reason="needs the prepared FloodNet cache")


def labels():
    """6x6 label map: Other (0) with a 2x2 building (1), a 1x2 road (2) and a padding row (255)."""
    y = np.zeros((6, 6), dtype=np.uint8)
    y[1:3, 1:3] = 1
    y[4, 2:4] = 2
    y[5, :] = 255
    return y


# ---------- paths and identities never touch earlier studies ----------

def test_undersample_runs_live_in_their_own_folder():
    from floodnet.training import run_path
    assert run_path("rf", 25, 42, variant="undersample/uniform") == ROOT / "runs/undersample/uniform/rf_25_s42"
    assert run_path("unet", 100, 44, variant="undersample/r3") == ROOT / "runs/undersample/r3/unet_100_s44"
    assert run_path("rf", 25, 42) == ROOT / "runs/rf_25_s42"


@needs_data
def test_blackening_changes_only_the_rf_feature_key():
    from floodnet.data import Store
    from floodnet.features import feature_key
    cfg = config()
    store = Store(cfg)
    black = copy.deepcopy(cfg)
    black["rf"]["blacken"] = True
    weights = copy.deepcopy(cfg)
    weights["rf"]["class_weight"] = "pixel_clipped"
    assert feature_key(black, store) != feature_key(cfg, store)
    assert feature_key(weights, store) == feature_key(cfg, store)


# ---------- uniform Before: the RF uses the U-Net's class weights ----------

def test_pixel_clipped_rf_weights_equal_the_unet_weights():
    from floodnet.neural import class_weights
    from floodnet.training import rf_class_weight
    records = [{"processed_counts": [900, 30, 70]}, {"processed_counts": [500, 0, 20]}]
    expected = class_weights(records, [0.25, 4.0])
    weights = rf_class_weight({"rf": {"class_weight": "pixel_clipped"}, "unet": {"weight_clip": [0.25, 4.0]}}, records)
    assert weights == {c: pytest.approx(float(expected[c])) for c in range(3)}
    assert rf_class_weight({"rf": {"class_weight": None}, "unet": {"weight_clip": [0.25, 4.0]}}, records) is None
    assert rf_class_weight({"rf": {"class_weight": "balanced"}, "unet": {"weight_clip": [0.25, 4.0]}}, records) == "balanced"


# ---------- undersampling: U-Net pixels ----------

def test_keep_probability_gives_ratio_r_other_pixels_per_target_pixel():
    from floodnet.neural import other_keep_probability
    records = [{"processed_counts": [9000, 100, 200]}, {"processed_counts": [1000, 0, 0]}]
    assert other_keep_probability(records, 3) == pytest.approx(3 * 300 / 10000)
    assert other_keep_probability(records, 1000) == 1.0


def test_undersample_mask_drops_only_other_pixels_and_is_fixed_per_image():
    from floodnet.neural import undersample_mask
    y = np.zeros((200, 200), dtype=np.uint8)
    y[:20] = 1
    y[20:40] = 2
    y[190:] = 255
    a = undersample_mask(y, 0.25, seed=42, index=7)
    assert np.array_equal(a, undersample_mask(y, 0.25, seed=42, index=7))
    assert not np.array_equal(a, undersample_mask(y, 0.25, seed=42, index=8))
    assert np.array_equal(a[:40], y[:40]) and np.all(a[190:] == 255)
    other = y == 0
    assert set(np.unique(a[other]).tolist()) <= {0, 255}
    assert (a[other] == 0).mean() == pytest.approx(0.25, abs=0.02)
    assert np.array_equal(undersample_mask(y, 1.0, seed=42, index=7), y)


def test_focus_example_keeps_the_real_photo_and_moves_the_label_with_the_augmentation():
    from floodnet.neural import augment, focus_example
    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
    y = np.zeros((64, 64), dtype=np.uint8)
    y[10:20, 10:30] = 1
    image, label = focus_example(rgb, y, keep=0.3, blacken_photo=False, seed=42, epoch=3, index=5, strength=0.1)
    plain_image, plain_label = augment(rgb, y, 42, 3, 5, 0.1)
    assert np.array_equal(image, plain_image)
    kept = label != 255
    assert np.array_equal(label[kept], plain_label[kept])
    assert np.array_equal(label == 1, plain_label == 1)


def test_blackening_paints_only_other_pixels_of_the_photo():
    from floodnet.neural import blacken
    y = labels()
    rgb = np.full((6, 6, 3), 200, dtype=np.uint8)
    out = blacken(rgb, y)
    assert np.all(out[y == 0] == 0)
    assert np.all(out[y == 1] == 200) and np.all(out[y == 2] == 200) and np.all(out[y == 255] == 200)
    assert np.all(rgb == 200)


# ---------- undersampling: RF regions ----------

@needs_data
def test_ratio_table_keeps_every_target_region_and_r_times_other():
    from floodnet.data import Store
    from floodnet.features import training_table
    cfg = copy.deepcopy(config())
    store = Store(cfg)
    _, base_y, _ = training_table(cfg, store, 25, 42)
    cfg["rf"].update(region_sampling="ratio", ratio=3, max_regions=10 ** 9)
    _, y, refs = training_table(cfg, store, 25, 42)
    targets = int((base_y > 0).sum())
    assert int((y > 0).sum()) == targets
    assert int((y == 0).sum()) == min(int((base_y == 0).sum()), round(3 * targets))
    _, _, again_refs = training_table(cfg, store, 25, 42)
    assert refs == again_refs
    other_refs = {tuple(r) for r, label in zip(refs, y) if label == 0}
    assert other_refs != {tuple(r) for r, label in zip(training_table(cfg, store, 25, 43)[2], training_table(cfg, store, 25, 43)[1]) if label == 0}


# ---------- ratio choice ----------

def test_ratio_choice_takes_the_best_validation_mean_and_breaks_ties_low():
    from floodnet.undersample import choose_ratio
    assert choose_ratio({1: [0.5, 0.52, 0.51], 3: [0.55, 0.53, 0.54], 10: [0.50, 0.49, 0.52]}) == 3
    assert choose_ratio({1: [0.5, 0.5, 0.5], 3: [0.5, 0.5, 0.5], 10: [0.4, 0.4, 0.4]}) == 1


# ---------- report: the three headline metrics ----------

def test_headline_metrics_follow_their_definitions():
    from floodnet.undersample import headline
    m = np.array([[90, 5, 5], [2, 6, 2], [1, 1, 8]])
    iou = [90 / 103, 6 / 16, 8 / 17]
    out = headline(m)
    assert out["minority"] == pytest.approx((iou[1] + iou[2]) / 2)
    assert out["majority"] == pytest.approx(iou[0])
    assert out["weighted"] == pytest.approx((100 * iou[0] + 10 * iou[1] + 10 * iou[2]) / 120)
    assert out["accuracy"] == pytest.approx(104 / 120)
