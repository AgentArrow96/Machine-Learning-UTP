import pickle

import numpy as np
import pytest
import torch

from floodnet.common import ROOT, config, read_json
from floodnet.data import remap


def real_data():
    return (ROOT / "data/manifests/processed.json").exists()


needs_data = pytest.mark.skipif(not real_data(), reason="needs the prepared FloodNet cache")


# ---------- the original 18 runs are untouched by the new hooks ----------

@needs_data
def test_identity_without_variant_equals_the_stored_identity():
    from floodnet.data import Store
    from floodnet.training import identity
    cfg = config()
    store = Store(cfg)
    for method in ("rf", "unet"):
        stored = read_json(f"runs/{method}_25_s42/run.json")["identity"]
        current = identity(stored_config(method), store, method, 25, 42)
        assert {k: v for k, v in current.items() if k != "source_identity"} == {k: v for k, v in stored.items() if k != "source_identity"}


def stored_config(method):
    return read_json(f"runs/{method}_25_s42/run.json")["config"]


@needs_data
def test_training_table_without_extras_reproduces_the_frozen_rf_sample():
    from floodnet.data import Store
    from floodnet.features import training_table
    cfg = config()
    _, _, references = training_table(cfg, Store(cfg), 25, 42)
    assert references == read_json("runs/rf_25_s42/sampled_regions.json")


# ---------- data recipes ----------

@needs_data
def test_ten_class_masks_remap_to_the_cached_labels():
    from floodnet.data import Store
    from floodnet.gan_common import masks10_for
    cfg = config()
    store = Store(cfg)
    for record in store.select("train")[:5] + store.select("val")[:3]:
        masks10 = masks10_for(record, cfg["size"])
        _, truth = store.get(record)
        valid = truth != 255
        assert np.array_equal(masks10 == 255, ~valid)
        assert np.array_equal(remap(masks10[valid].reshape(1, -1)).ravel(), truth[valid])


def toy_budget(n=12, size=64, seed=0):
    rng = np.random.default_rng(seed)
    masks, ids = [], []
    for i in range(n):
        m = np.full((size, size), 9, dtype=np.uint8)
        m[:8] = 255  # padding rows
        if i % 2 == 0:
            m[20:30, 10:25] = 1
        if i % 3 == 0:
            m[40:50, 30:60] = 3
        m[12:16, :] = rng.integers(0, 10)
        masks.append(m)
        ids.append(f"train_{i}")
    return np.stack(masks), ids


def test_layouts_are_deterministic_budget_only_and_add_targets():
    from floodnet.gan_common import make_layouts
    masks, ids = toy_budget()
    settings = {"paste_min": 1, "paste_max": 2, "min_component_pixels": 16, "paste_classes": [1, 3]}
    first, sources = make_layouts(masks, ids, count=10, seed=42, settings=settings)
    second, _ = make_layouts(masks, ids, count=10, seed=42, settings=settings)
    assert np.array_equal(first, second)
    assert set(np.unique(first)) <= set(range(10)) | {255}
    assert all({src["base"]} <= set(ids) and set(src["pasted_from"]) <= set(ids) for src in sources)
    assert all(np.array_equal(layout == 255, masks[ids.index(src["base"])][::src["flip_ud"] and -1 or 1, ::src["flip_lr"] and -1 or 1] == 255)
               for layout, src in zip(first, sources))
    base_targets = [np.isin(masks[ids.index(s["base"])], [1, 3]).sum() for s in sources]
    assert all(np.isin(layout, [1, 3]).sum() > b for layout, b in zip(first, base_targets))


def test_layouts_survive_bases_that_are_already_full_of_targets():
    """Regression: whole-frame flooded-road layouts cannot take any paste; editing must pick another base, not fail."""
    from floodnet.gan_common import make_layouts
    masks = np.full((6, 64, 64), 3, dtype=np.uint8)
    masks[:, :8] = 255
    masks[0, 8:] = 9
    masks[0, 20:30, 10:25] = 1
    ids = [f"train_{i}" for i in range(6)]
    settings = {"paste_min": 1, "paste_max": 2, "min_component_pixels": 16, "paste_classes": [1, 3]}
    layouts, sources = make_layouts(masks, ids, count=12, seed=0, settings=settings)
    assert len(layouts) == 12 and all(s["pasted_from"] for s in sources)
    assert all(np.isin(layout, [1, 3]).sum() > np.isin(masks[ids.index(s["base"])], [1, 3]).sum() for layout, s in zip(layouts, sources))


def test_finished_sets_are_not_rebuilt(project, monkeypatch):
    """Retries must not overwrite a known-good synthetic set (a rebuilt npz changes its hash and the set identity)."""
    from floodnet import imbalance
    root, _ = project
    folder = root / "data/synthetic/sets/smote_25"
    folder.mkdir(parents=True)
    calls = []

    def build(variant, budget, cfg, icfg, store):
        calls.append(variant)
        (folder / "rows.npz").write_bytes(b"first")
        (folder / "manifest.json").write_text('{"identity": "abc"}', encoding="utf-8")
        return {"identity": "abc"}
    monkeypatch.setattr(imbalance, "set_dir", lambda variant, budget: folder)
    monkeypatch.setattr(imbalance, "build_row_set", build)
    monkeypatch.setattr(imbalance, "check_preregistration", dict)
    monkeypatch.setattr(imbalance, "load_config", lambda: {"variants": {"smote": {"kind": "rows"}}})
    monkeypatch.setattr(imbalance, "Store", lambda cfg: None)
    imbalance.stage_set({}, "smote", 25)
    before = (folder / "rows.npz").read_bytes()
    imbalance.stage_set({}, "smote", 25)
    assert calls == ["smote"] and (folder / "rows.npz").read_bytes() == before


def test_chessmix_is_deterministic_and_budget_only():
    from floodnet.imbalance import chessmix
    masks, ids = toy_budget(size=128)
    rgb = np.random.default_rng(1).integers(0, 256, (*masks.shape, 3), dtype=np.uint8)
    labels = np.stack([np.where(m == 255, 255, remap(np.where(m == 255, 0, m))) for m in masks]).astype(np.uint8)
    settings = {"grid": 4, "patch": 32, "candidates_per_patch": 4, "target_weight_epsilon": 1}
    a = chessmix(rgb, labels, ids, count=3, seed=7, settings=settings)
    b = chessmix(rgb, labels, ids, count=3, seed=7, settings=settings)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    assert set(np.unique(a[1])) <= {0, 1, 2, 255}
    assert all(set(s) <= set(ids) for s in a[2])
    assert (a[1] == 255).sum() >= a[1].size // 2 - 1


def test_oversampling_draws_only_target_images_from_the_budget():
    from floodnet.imbalance import oversample_ids
    records = [{"id": f"train_{i}", "processed_counts": [5, int(i % 3 == 0), int(i % 4 == 0)]} for i in range(20)]
    budget = [f"train_{i}" for i in range(12)]
    chosen = oversample_ids(records, budget, count=30, seed=42)
    assert chosen == oversample_ids(records, budget, count=30, seed=42) and len(chosen) == 30
    allowed = {r["id"] for r in records if r["id"] in budget and (r["processed_counts"][1] or r["processed_counts"][2])}
    assert set(chosen) <= allowed


def test_smote_rows_lie_between_real_neighbours():
    from floodnet.imbalance import smote
    rng = np.random.default_rng(3)
    x = rng.random((40, 22)).astype(np.float32)
    x[:, 12:] /= x[:, 12:].sum(1, keepdims=True)
    synthetic, pairs = smote(x, count=50, k=5, seed=11)
    assert synthetic.shape == (50, 22)
    for row, (i, j, u) in zip(synthetic, pairs):
        assert 0 <= u <= 1 and np.allclose(row, x[i] + u * (x[j] - x[i]), atol=1e-6)
    assert np.allclose(synthetic[:, 12:].sum(1), 1, atol=1e-5)


class RealStub:
    def get(self, record):
        return np.zeros((4, 4, 3), np.uint8), np.zeros((4, 4), np.uint8)


def test_smote_neighbours_are_chunked_but_identical():
    """Regression: SMOTE built an (n, n, 22) array at once (about 14 GB for 9,000 road rows), starving the GPU process.
    Chunked distances must give exactly the rows of the original all-at-once computation."""
    from floodnet import imbalance
    rng = np.random.default_rng(5)
    x = rng.random((300, 22)).astype(np.float32)

    def reference(x, count, k, seed):
        r = np.random.default_rng(seed)
        x = np.asarray(x, dtype=np.float64)
        distances = ((x[:, None, :] - x[None, :, :]) ** 2).sum(-1)
        np.fill_diagonal(distances, np.inf)
        neighbours = np.argsort(distances, 1)[:, :k]
        rows = []
        for _ in range(count):
            i = int(r.integers(len(x)))
            j = int(neighbours[i, r.integers(neighbours.shape[1])])
            rows.append(x[i] + float(r.random()) * (x[j] - x[i]))
        return np.asarray(rows, dtype=np.float32)
    assert np.array_equal(imbalance.smote(x, 200, 5, 9)[0], reference(x, 200, 5, 9))
    assert imbalance.SMOTE_CHUNK * len(x) * x.shape[1] * 8 < 2 ** 30


def test_mixed_store_serves_synthetic_records_and_pickles_without_arrays(tmp_path):
    from floodnet.neural import MixedStore
    rgb = np.full((2, 4, 4, 3), 7, np.uint8)
    mask = np.full((2, 4, 4), 2, np.uint8)
    np.save(tmp_path / "rgb.npy", rgb)
    np.save(tmp_path / "mask.npy", mask)
    store = MixedStore(RealStub(), {"syn": tmp_path})
    image, labels = store.get({"id": "syn_1", "synthetic": "syn", "index": 1})
    assert image[0, 0, 0] == 7 and labels[0, 0] == 2
    assert store.get({"id": "train_1"})[0].sum() == 0
    clone = pickle.loads(pickle.dumps(store))
    assert clone.arrays == {} and clone.get({"id": "syn_0", "synthetic": "syn", "index": 0})[1][0, 0] == 2


# ---------- losses and networks ----------

def test_dice_cross_entropy_ignores_padding_and_rewards_overlap():
    from floodnet.neural import DiceCrossEntropy
    weights = torch.tensor([0.25, 2.0, 1.0])
    loss = DiceCrossEntropy(weights)
    target = torch.zeros((1, 8, 8), dtype=torch.long)
    target[0, 2:5, 2:5] = 1
    target[0, 5:7, :] = 2
    target[0, 0] = 255
    perfect = torch.full((1, 3, 8, 8), -20.)
    perfect.scatter_(1, target.clamp(max=2).unsqueeze(1), 20.)
    wrong = torch.full((1, 3, 8, 8), -20.)
    wrong[:, 0] = 20.
    assert loss(perfect, target) < 1e-3
    assert loss(wrong, target) > 1.0
    assert torch.allclose(loss.weight, weights)


def test_gan_training_fails_closed_on_exploding_losses():
    """Regression: OASIS reached d = 3.4e8 (finite) and kept training; any loss beyond 1e4 must stop the run."""
    from floodnet.common import ProjectError
    from floodnet.gan_common import check_losses
    check_losses({"d": 3.3, "g_adv": 1.6}, "toy", 10)
    for bad in ({"d": 3.4e8}, {"labelmix": float("nan")}, {"g_adv": float("inf")}):
        with pytest.raises(ProjectError):
            check_losses(bad, "toy", 2000)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_generation_does_not_depend_on_leftover_cudnn_state():
    """Regression: a U-Net retrain earlier in the same process left cudnn.deterministic=True, which switched the
    R-GAN's transposed convolutions to another algorithm (+-1 grey level on 0.46% of values)."""
    from floodnet.gan_common import generate
    from floodnet.rgan import ImageGenerator
    torch.manual_seed(0)
    generator = ImageGenerator(ngf=8).cuda()
    layouts = np.full((2, 256, 256), 9, dtype=np.uint8)
    layouts[:, :40] = 255
    layouts[:, 100:140, 50:200] = 3
    before = torch.backends.cudnn.deterministic
    try:
        torch.backends.cudnn.deterministic = False
        a = generate(generator, layouts, 42)
        torch.backends.cudnn.deterministic = True
        b = generate(generator, layouts, 42)
        assert torch.backends.cudnn.deterministic is True
    finally:
        torch.backends.cudnn.deterministic = before
    assert np.array_equal(a, b)


def test_active_gan_time_excludes_pauses():
    """Standby/stalls (a 3-hour gap for one 500-iteration step) must not count as training time."""
    from floodnet.imbalance import active_training_seconds
    log = """=== gan --variant oasis --budget 100
[2026-10-24T11:00:00+00:00] oasis_100 500/20000: d=1
[2026-10-24T11:04:00+00:00] oasis_100 1000/20000: d=1
[2026-10-24T11:08:00+00:00] oasis_100 1500/20000: d=1
[2026-10-24T14:08:00+00:00] oasis_100 2000/20000: d=1
[2026-10-24T14:10:00+00:00] oasis_100 checkpoint 2000: validation KID 0.5
[2026-10-24T14:14:00+00:00] spade_25 500/10000: d=1
=== gan --variant oasis --budget 100
[2026-10-24T17:00:00+00:00] oasis_100: resuming at iteration 2000
[2026-10-24T17:04:00+00:00] oasis_100 2500/20000: d=1"""
    result = active_training_seconds(log, "oasis", 100)
    # Steps: first (typical 240) + 240 + 240 + pause (replaced by 240) + checkpoint 120; resumed: 240 (from the resume line).
    assert result["active_seconds"] == 240 + 240 + 240 + 240 + 120 + 240
    assert result["pauses"] == [{"from": "2026-10-24T11:08:00+00:00", "to": "2026-10-24T14:08:00+00:00", "seconds": 10800}]
    # An earlier, abandoned fresh run (e.g. the diverged OASIS) is not part of the final generator's time.
    abandoned = "=== gan --variant oasis --budget 100\n[2026-10-24T09:00:00+00:00] oasis_100 500/20000: d=1\n[2026-10-24T09:04:00+00:00] oasis_100 1000/20000: d=1\n"
    assert active_training_seconds(abandoned + log, "oasis", 100)["active_seconds"] == result["active_seconds"]


def test_relativistic_average_losses_have_the_right_direction():
    from floodnet.gan_common import relativistic_d_loss, relativistic_g_loss
    real, fake = torch.full((4, 1), 3.), torch.full((4, 1), -3.)
    assert relativistic_d_loss(real, fake) < relativistic_d_loss(fake, real)
    assert relativistic_g_loss(real, fake) > relativistic_g_loss(fake, real)


def has_spectral_norm(module):
    return any(hasattr(m, "weight_orig") or "parametrizations" in dict(m.named_children()) for m in module.modules())


def test_image_rgan_has_the_three_named_components():
    from floodnet.rgan import ImageDiscriminator, ImageGenerator
    g, d = ImageGenerator(ngf=8), ImageDiscriminator(ndf=8)
    assert has_spectral_norm(g) and has_spectral_norm(d)
    assert any(isinstance(m, torch.nn.CELU) for m in d.modules())
    cond = torch.zeros((2, 11, 256, 256))
    cond[:, 9] = 1
    image = g(cond)
    assert image.shape == (2, 3, 256, 256) and image.abs().max() <= 1
    logits, features = d(image, cond)
    assert logits.ndim == 4 and len(features) >= 3


def test_tabular_rgan_generates_valid_region_rows():
    from floodnet.rgan import TabularDiscriminator, TabularGenerator
    g, d = TabularGenerator(noise=8, hidden=16), TabularDiscriminator(hidden=16)
    assert has_spectral_norm(g) and any(isinstance(m, torch.nn.CELU) for m in d.modules())
    rows = g(torch.randn(5, 8), torch.tensor([0, 1, 0, 1, 1]))
    assert rows.shape == (5, 22) and rows.min() >= 0 and rows.max() <= 1
    assert torch.allclose(rows[:, 12:].sum(1), torch.ones(5), atol=1e-5)
    assert d(rows, torch.tensor([0, 1, 0, 1, 1])).shape == (5, 1)


def test_oasis_and_spade_shapes():
    from floodnet.gan_oasis import OasisDiscriminator, OasisGenerator
    from floodnet.gan_spade import SpadeDiscriminator, SpadeGenerator
    cond = torch.zeros((2, 11, 256, 256))
    cond[:, 9] = 1
    for generator in (OasisGenerator(nf=8, noise_channels=4), SpadeGenerator(nf=8)):
        out = generator(cond)
        assert out.shape == (2, 3, 256, 256) and out.abs().max() <= 1
    assert OasisDiscriminator(nf=8)(torch.zeros(2, 3, 256, 256)).shape == (2, 12, 256, 256)
    outputs = SpadeDiscriminator(ndf=8)(torch.zeros(2, 3, 256, 256), cond)
    assert len(outputs) == 2 and all(len(features) >= 3 for features in outputs)


def test_oasis_discriminator_blocks_do_not_amplify_their_input_scale():
    """Regression: unnormalised residual blocks grew activations 3-11x per block (2.3 -> 2.3e8 over 12 blocks).
    With GroupNorm in the main path, a block's output scale must not follow a 1000x larger input."""
    from floodnet.gan_oasis import ResDown, ResUp
    torch.manual_seed(0)
    for block in (ResDown(16, 32), ResUp(16, 32)):
        x = torch.randn(2, 16, 16, 16)
        small, large = block(x).abs().max(), block(1000 * x).abs().max()
        assert large < 50 * small


def test_oasis_class_balanced_loss_weights_rare_classes_more():
    from floodnet.gan_oasis import class_balanced_weights
    labels = torch.zeros((1, 16, 16), dtype=torch.long)
    labels[0, :2, :2] = 3
    weights = class_balanced_weights(labels, classes=11)
    assert weights[3] > weights[0] > 0 and weights[5] == 0


# ---------- quality metrics ----------

def test_kid_is_near_zero_for_the_same_distribution_and_grows_with_shift():
    from floodnet.gan_common import kid
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(300, 16)), rng.normal(size=(300, 16))
    c = rng.normal(loc=1.0, size=(300, 16))
    same = kid(a, b, subsets=10, subset_size=100, seed=1)
    shifted = kid(a, c, subsets=10, subset_size=100, seed=1)
    assert abs(same["mean"]) < 0.05 and shifted["mean"] > 10 * max(abs(same["mean"]), 0.01)


def test_mmd_detects_a_shift():
    from floodnet.imbalance import rbf_mmd
    rng = np.random.default_rng(0)
    a, b, c = rng.random((200, 22)), rng.random((200, 22)), rng.random((200, 22)) + .3
    assert rbf_mmd(a, b) < rbf_mmd(a, c)


# ---------- freeze ----------

def test_freeze_refuses_until_every_run_is_validated(project):
    from floodnet.common import ProjectError
    from floodnet.imbalance import freeze_imbalance
    _, cfg = project
    with pytest.raises(ProjectError):
        freeze_imbalance(cfg, None, plan=[("oversample", "unet", 25, 42)])
