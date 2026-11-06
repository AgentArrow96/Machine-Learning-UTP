import numpy as np
import pytest
import torch

from floodnet.common import ROOT, config
from floodnet.data import remap


def real_data():
    return (ROOT / "data/manifests/processed.json").exists()


needs_data = pytest.mark.skipif(not real_data(), reason="needs the prepared FloodNet cache")


# ---------- paths never touch frozen artifacts ----------

def test_improve_runs_live_in_their_own_folder_and_old_paths_are_unchanged():
    from floodnet.training import run_path
    assert run_path("unet", 25, 42, variant="improve/aux10") == ROOT / "runs/improve/aux10/unet_25_s42"
    assert run_path("unet", 25, 42, variant="oversample") == ROOT / "runs/imbalance/oversample/unet_25_s42"
    assert run_path("unet", 25, 42) == ROOT / "runs/unet_25_s42"


def test_non_baseline_caches_and_preflight_never_write_into_data_manifests():
    from floodnet.data import manifest_path_for
    from floodnet.training import preflight_path_for
    cfg = config()
    assert manifest_path_for(cfg) == "data/manifests/processed.json"
    assert manifest_path_for({**cfg, "size": 512}) == "data/improve/processed_512.json"
    assert preflight_path_for(cfg, None) == "data/manifests/unet_preflight.json"
    assert preflight_path_for(cfg, object()).startswith("data/improve/unet_preflight_")


# ---------- 10-class auxiliary supervision ----------

def test_aux10_model_has_13_outputs_and_predicts_with_the_3_class_head():
    from floodnet.neural import build_model, three_class
    cfg = config()
    cfg["unet"] = {**cfg["unet"], "aux10_weight": 0.5}
    model = build_model(cfg, pretrained=False)
    out = model(torch.zeros(1, 3, 64, 64))
    assert out.shape[1] == 13 and three_class(out, cfg).shape[1] == 3
    plain = config()
    assert three_class(torch.zeros(1, 3, 8, 8), plain).shape[1] == 3


def test_paired_augmentation_moves_both_label_maps_identically():
    from floodnet.neural import augment_pair
    rng = np.random.default_rng(0)
    y10 = rng.integers(0, 10, (32, 32)).astype(np.uint8)
    y10[:4] = 255
    y3 = np.where(y10 == 255, 255, remap(np.where(y10 == 255, 0, y10))).astype(np.uint8)
    rgb = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
    for epoch in range(6):
        image, a3, a10 = augment_pair(rgb, y3, y10, seed=42, epoch=epoch, index=3, strength=.1)
        assert np.array_equal(a3, np.where(a10 == 255, 255, remap(np.where(a10 == 255, 0, a10))))


def test_aux_loss_adds_the_weighted_10_class_term():
    from floodnet.neural import aux_loss
    logits = torch.zeros(2, 13, 4, 4)
    y3 = torch.zeros(2, 4, 4, dtype=torch.long)
    y10 = torch.full((2, 4, 4), 9, dtype=torch.long)
    base = torch.nn.functional.cross_entropy(logits[:, :3], y3)
    extra = torch.nn.functional.cross_entropy(logits[:, 3:], y10)
    weight = torch.ones(3)
    assert torch.isclose(aux_loss(logits, y3, y10, weight, 0.5), base + 0.5 * extra)


@needs_data
def test_rf_aux10_region_labels_remap_to_the_3_class_majority_for_target_regions():
    from floodnet.data import Store
    from floodnet.features import feature_path, region_labels10
    cfg = config()
    store = Store(cfg)
    record = store.select("train", 25)[0]
    with np.load(feature_path(cfg, store, record)) as cached:
        y3, regions = cached["y"], cached["regions"]
    y10 = region_labels10(record, regions, cfg["size"])
    assert len(y10) == len(y3) and set(np.unique(y10)) <= set(range(10))
    # where the 10-class majority is a target, the 3-class majority must be that target too (ties aside, almost always)
    agree = np.mean([(remap(np.array([[v]]))[0, 0] == t) for v, t in zip(y10, y3) if v in (1, 3)] or [1])
    assert agree > .95


# ---------- training recipe ----------

def test_cosine_schedule_and_ema():
    from floodnet.neural import cosine_lr, ema_update
    assert cosine_lr(0, 60, 1e-4, 1e-6) == pytest.approx(1e-4)
    assert cosine_lr(59, 60, 1e-4, 1e-6) == pytest.approx(1e-6)
    assert 1e-6 < cosine_lr(30, 60, 1e-4, 1e-6) < 1e-4
    ema, live = torch.nn.Linear(2, 1), torch.nn.Linear(2, 1)
    with torch.no_grad():
        ema.weight.fill_(1.); live.weight.fill_(3.)
    ema_update(ema, live, 0.9)
    assert torch.allclose(ema.weight, torch.full_like(ema.weight, 1.2))


# ---------- test-time augmentation ----------

def test_dihedral_tta_is_exact_for_an_equivariant_model_and_averages_otherwise():
    from floodnet.neural import dihedral_mean
    x = torch.rand(1, 3, 8, 8)
    identity = lambda t: torch.softmax(t, 1)
    assert torch.allclose(dihedral_mean(identity, x), identity(x), atol=1e-6)
    fixed = torch.rand(1, 3, 8, 8)
    constant = lambda t: fixed
    expected = torch.stack([fixed, fixed.flip(3), fixed.flip(2), fixed.flip(2).flip(3),
                            fixed.transpose(2, 3), fixed.transpose(2, 3).flip(3), fixed.transpose(2, 3).flip(2), fixed.transpose(2, 3).flip(2).flip(3)]).mean(0)
    assert dihedral_mean(constant, x).shape == fixed.shape
    assert not torch.allclose(dihedral_mean(constant, x), fixed) or torch.allclose(expected, fixed)


# ---------- 512-px scoring on the 256 grid ----------

@pytest.mark.parametrize("g256,g512", [({"top": 32, "left": 0, "height": 192, "width": 256}, {"top": 64, "left": 0, "height": 384, "width": 512}),
                                       ({"top": 42, "left": 0, "height": 171, "width": 256}, {"top": 85, "left": 0, "height": 342, "width": 512})])
def test_512_probabilities_are_averaged_onto_the_256_geometry(g256, g512):
    from floodnet.improve import to_256
    probs = np.zeros((3, 512, 512), dtype=np.float32)
    probs[0] = 1
    t, l, h, w = g512["top"], g512["left"], g512["height"], g512["width"]
    probs[:, t:t + h, l:l + w] = 0
    probs[2, t:t + h, l:l + w] = 1
    out = to_256(probs, g512, g256)
    assert out.shape == (3, 256, 256)
    window = out[:, g256["top"]:g256["top"] + g256["height"], g256["left"]:g256["left"] + g256["width"]]
    assert np.allclose(window[2], 1)


def test_512_content_that_is_not_exactly_twice_is_area_averaged():
    # Regression: 4592x3072 photos give 171 rows at 256 px but 343 (not 342) at 512 px (15% of images).
    from floodnet.improve import to_256
    g256, g512 = {"top": 42, "left": 0, "height": 171, "width": 256}, {"top": 84, "left": 0, "height": 343, "width": 512}
    probs = np.zeros((3, 512, 512), dtype=np.float32)
    probs[0] = 1
    content = probs[:, 84:84 + 343, :]
    content[...] = 0
    content[1, :172] = 1                       # top half (by area) building, bottom half road
    content[2, 172:] = 1
    out = to_256(probs, g512, g256)[:, 42:42 + 171]
    assert np.allclose(out.sum(0), 1, atol=1e-6)                # still a distribution
    assert np.allclose(out[1].mean(), 172 / 343, atol=1e-6)     # area preserved
    assert out[:, :85].argmax(0).max() == 1 and out[:, 86:].argmax(0).min() == 2


def test_area_averaging_equals_2x2_block_mean_when_exact():
    from floodnet.improve import to_256
    rng = np.random.default_rng(0)
    g256, g512 = {"top": 32, "left": 0, "height": 192, "width": 256}, {"top": 64, "left": 0, "height": 384, "width": 512}
    probs = rng.random((3, 512, 512)).astype(np.float32)
    block = probs[:, 64:448].reshape(3, 192, 2, 256, 2).mean((2, 4))
    assert np.allclose(to_256(probs, g512, g256)[:, 32:224], block, atol=1e-6)


def test_crf_parameters_scale_with_resolution():
    from floodnet.improve import scaled_crf
    params = {"iterations": 5, "gaussian": {"sxy": 3, "compat": 3}, "bilateral": {"sxy": 20, "srgb": 5, "compat": 10}}
    scaled = scaled_crf(params, 512)
    assert scaled["gaussian"]["sxy"] == 6 and scaled["bilateral"]["sxy"] == 40 and scaled["bilateral"]["srgb"] == 5
    assert scaled_crf(params, 256) == params


# ---------- RF context features ----------

def test_context_features_average_the_neighbours():
    from floodnet.features import context_features
    regions = np.array([[0, 0, 1, 1], [0, 0, 1, 1], [2, 2, 2, 2], [2, 2, 2, 2]])
    x = np.array([[1.] * 22, [3.] * 22, [5.] * 22], dtype=np.float32)
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    out = context_features(x, regions, rgb, coarse_segments=2)
    assert out.shape == (3, 22 + 22 + 3 + 3)
    assert np.allclose(out[0, 22:44], 4.)          # region 0 touches 1 and 2 -> mean(3, 5)
    assert np.allclose(out[2, 22:44], 2.)          # region 2 touches 0 and 1 -> mean(1, 3)
    assert (out[:, 44:46] >= 0).all() and (out[:, 44:46] <= 1).all() and out[1, 46] == pytest.approx(4 / 16)


def test_context_features_count_each_neighbour_once():
    # Regression (audit M4/M5): regions 0 and 1 touch as 0|1 in one row and 1|0 in another; 1 must count once.
    from floodnet.features import context_features
    regions = np.array([[0, 1, 2], [1, 0, 2], [2, 2, 2]])
    x = np.array([[1.] * 22, [3.] * 22, [9.] * 22], dtype=np.float32)
    out = context_features(x, regions, np.zeros((3, 3, 3), dtype=np.uint8), coarse_segments=2)
    assert np.allclose(out[0, 22:44], 6.)          # region 0 touches 1 and 2 once each -> mean(3, 9)


# ---------- combine rule ----------

def test_combine_rule_needs_positive_means_at_both_budgets_and_two_of_three_seeds():
    from floodnet.improve import passes_combine_rule
    good = {25: [.01, .02, -.005], 100: [.01, .002, .003]}
    weak = {25: [.01, -.02, -.005], 100: [.01, .002, .003]}
    negative = {25: [.01, .02, .03], 100: [-.03, .002, .003]}
    assert passes_combine_rule(good) and not passes_combine_rule(weak) and not passes_combine_rule(negative)
