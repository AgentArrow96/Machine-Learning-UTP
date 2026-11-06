"""Shared machinery for the three mask-to-image GANs of the class-imbalance study.

Everything a GAN family does not decide for itself lives here so the comparison isolates the generator:
10-class layouts at 256 px, layout editing (extra flooded targets), conditioning, DiffAugment, the relativistic
losses, ImageNet ResNet-18 features, KID, the resumable training loop with validation-KID checkpointing, synthesis
with a data-contract check, and the quality report (KID references, mask consistency, memorisation).
"""
from __future__ import annotations

import time

import numpy as np
from PIL import Image

from .common import (
    ProjectError,
    digest,
    inside,
    log,
    now,
    read_json,
    sha256,
    write_json,
)
from .data import MEAN, STD, mask_array, remap

CONDITION_CHANNELS = 11  # 10 original classes + padding


# ---------- layouts ----------

def masks10_for(record, size):
    """Original 10-class annotation resized and padded exactly like `resize_pair` does for the 3-class mask."""
    if sha256(record["mask"]) != record["mask_sha256"]:
        raise ProjectError(f"Original annotation changed since audit: {record['id']}")
    values = mask_array(inside(record["mask"]))
    g = record["geometry"]
    labels = np.asarray(Image.fromarray(values).resize((g["width"], g["height"]), Image.Resampling.NEAREST))
    out = np.full((size, size), 255, dtype=np.uint8)
    out[g["top"]:g["top"] + g["height"], g["left"]:g["left"] + g["width"]] = labels
    return out


def masks10_cache(cfg, store, split):
    """Cached (N, 256, 256) 10-class layouts for a split, keyed by the processed identity."""
    directory = inside("data/synthetic")
    directory.mkdir(parents=True, exist_ok=True)
    # 256 px keeps its original file name; other sizes (improvement study) get their own cache.
    suffix = "" if cfg["size"] == 256 else f"_{cfg['size']}"
    path = directory / f"masks10_{split}{suffix}.npy"
    meta = directory / f"masks10_{split}{suffix}.json"
    records = store.select(split)
    key = digest({"processed": store.manifest["identity"], "split": split, "ids": [r["id"] for r in records]})
    if path.exists() and meta.exists() and read_json(meta)["key"] == key and sha256(path) == read_json(meta)["sha256"]:
        return np.load(path, mmap_mode="r"), [r["id"] for r in records]
    masks = np.stack([masks10_for(r, cfg["size"]) for r in records])
    for record, layout in zip(records, masks):
        _, truth = store.get(record)
        valid = truth != 255
        if not np.array_equal(layout == 255, ~valid) or not np.array_equal(remap(np.where(valid, layout, 0))[valid], truth[valid]):
            raise ProjectError(f"10-class layout does not remap to the cached labels: {record['id']}")
    np.save(path, masks)
    write_json(meta, {"key": key, "sha256": sha256(path), "created_at": now(), "records": len(records)})
    log(f"Cached {len(records)} 10-class layouts for {split}")
    return np.load(path, mmap_mode="r"), [r["id"] for r in records]


def components(mask, classes, min_pixels):
    from scipy import ndimage
    found = []
    for c in classes:
        labelled, _ = ndimage.label(mask == c)
        for index, box in enumerate(ndimage.find_objects(labelled), 1):
            piece = labelled[box] == index
            if piece.sum() >= min_pixels:
                found.append((c, piece))
    return found


def make_layouts(masks, ids, count, seed, settings):
    """Edited budget layouts with extra flooded targets (labels first; a GAN then renders the photo).

    Base: a random target-containing budget layout, flipped (geometry-preserving). Then 1-2 flooded-building or
    flooded-road components from *other* budget layouts are pasted into the valid area, only over non-target pixels.
    Returns (layouts, sources) where sources record every image used (budget-isolation evidence)."""
    rng = np.random.default_rng(seed)
    classes = settings["paste_classes"]
    has_target = [i for i, m in enumerate(masks) if np.isin(m, classes).any()]
    if not has_target:
        raise ProjectError("No target-containing layouts in this budget")
    pieces = [(i, c, p) for i in has_target for c, p in components(np.asarray(masks[i]), classes, settings["min_component_pixels"])]
    if not pieces:
        raise ProjectError("No target components large enough to paste")
    layouts, sources, rejected_bases = [], [], 0
    for _ in range(count):
        # A base already covered by targets (e.g. a whole-frame flooded road) cannot take a paste: draw another base.
        for _attempt in range(50):
            base = int(rng.choice(has_target))
            flip_lr, flip_ud = bool(rng.random() < .5), bool(rng.random() < .5)
            layout = np.asarray(masks[base])[::-1 if flip_ud else 1, ::-1 if flip_lr else 1].copy()
            valid = layout != 255
            pasted, pasted_classes = [], []
            for _ in range(int(rng.integers(settings["paste_min"], settings["paste_max"] + 1))):
                # Up to 5 components per paste slot; a component that paints nothing here is replaced by another.
                for _try in range(5):
                    others = [k for k, piece in enumerate(pieces) if piece[0] != base] or list(range(len(pieces)))
                    source, cls, piece = pieces[int(rng.choice(others))]
                    piece = piece[::-1 if rng.random() < .5 else 1, ::-1 if rng.random() < .5 else 1]
                    h, w = piece.shape
                    if h > layout.shape[0] or w > layout.shape[1]:
                        continue
                    best = None
                    for _ in range(10):
                        top, left = int(rng.integers(0, layout.shape[0] - h + 1)), int(rng.integers(0, layout.shape[1] - w + 1))
                        window = layout[top:top + h, left:left + w]
                        paint = piece & (window != 255) & ~np.isin(window, classes)
                        if best is None or paint.sum() > best[2].sum():
                            best = (top, left, paint)
                    top, left, paint = best
                    if paint.sum() == 0:
                        continue
                    layout[top:top + h, left:left + w][paint] = cls
                    pasted.append(ids[source])
                    pasted_classes.append(int(cls))
                    break
            if pasted:
                break
            rejected_bases += 1
        if not pasted or not np.array_equal(layout == 255, ~valid):
            raise ProjectError("Layout editing failed to add a target inside the valid area after 50 bases")
        layouts.append(layout)
        sources.append({"base": ids[base], "flip_lr": flip_lr, "flip_ud": flip_ud, "pasted_from": pasted, "pasted_classes": pasted_classes,
                        "rejected_bases_so_far": rejected_bases})
    return np.stack(layouts), sources


def geometry_of(layout):
    rows, cols = np.flatnonzero((layout != 255).any(1)), np.flatnonzero((layout != 255).any(0))
    return {"top": int(rows[0]), "left": int(cols[0]), "height": int(rows[-1] - rows[0] + 1), "width": int(cols[-1] - cols[0] + 1)}


def labels_of(layout):
    """3-class training labels of a 10-class layout (255 padding kept)."""
    return np.where(layout == 255, 255, remap(np.where(layout == 255, 0, layout))).astype(np.uint8)


def check_contract(rgb, labels, layouts):
    """Synthetic data contract (eval I3): shapes, IDs, labels = remap(layout), padding."""
    n = len(layouts)
    if rgb.shape != (n, 256, 256, 3) or rgb.dtype != np.uint8 or labels.shape != (n, 256, 256) or labels.dtype != np.uint8:
        raise ProjectError(f"Synthetic shapes/dtypes wrong: {rgb.shape} {labels.shape}")
    if not set(np.unique(labels).tolist()) <= {0, 1, 2, 255}:
        raise ProjectError("Synthetic labels outside {0,1,2,255}")
    for layout, label in zip(layouts, labels):
        if not np.array_equal(label, labels_of(layout)):
            raise ProjectError("Synthetic labels differ from remap(layout)")
    pad = np.round(MEAN * 255).astype(np.uint8)
    if not all(np.all(image[label == 255] == pad) for image, label in zip(rgb, labels)):
        raise ProjectError("Synthetic padding is not the cache's mean colour")
    return {"samples": n, "checked": ["shapes", "dtypes", "label ids", "labels == remap(layout)", "padding colour"]}


# ---------- tensors, losses, augmentation ----------

def condition(layouts):
    """(B, H, W) uint8 layouts -> (B, 11, H, W) one-hot (channel 10 = padding)."""
    import torch
    index = layouts.long().clone()
    index[index == 255] = 10
    return torch.nn.functional.one_hot(index, CONDITION_CHANNELS).permute(0, 3, 1, 2).float()


def to_unit(rgb_uint8):
    """(B, H, W, 3) uint8 -> (B, 3, H, W) in [-1, 1]."""
    return rgb_uint8.permute(0, 3, 1, 2).float() / 127.5 - 1


def relativistic_d_loss(real_logits, fake_logits):
    import torch.nn.functional as F
    real_rel = real_logits - fake_logits.mean()
    fake_rel = fake_logits - real_logits.mean()
    return (F.binary_cross_entropy_with_logits(real_rel, torch_ones(real_rel)) + F.binary_cross_entropy_with_logits(fake_rel, torch_zeros(fake_rel))) / 2


def relativistic_g_loss(real_logits, fake_logits):
    import torch.nn.functional as F
    real_rel = real_logits - fake_logits.mean()
    fake_rel = fake_logits - real_logits.mean()
    return (F.binary_cross_entropy_with_logits(fake_rel, torch_ones(fake_rel)) + F.binary_cross_entropy_with_logits(real_rel, torch_zeros(real_rel))) / 2


def torch_ones(x):
    import torch
    return torch.ones_like(x)


def torch_zeros(x):
    import torch
    return torch.zeros_like(x)


def diff_augment(images, cond, generator):
    """DiffAugment for conditional GANs: the same flip and translation on image and layout, colour on the image only.

    Applied identically to real and fake batches inside each discriminator step (Zhao et al., 2020)."""
    import torch
    b = images.shape[0]
    device = images.device
    flip = torch.rand(b, 1, 1, 1, device=device, generator=generator) < .5
    images, cond = torch.where(flip, images.flip(3), images), torch.where(flip, cond.flip(3), cond)
    brightness = (torch.rand(b, 1, 1, 1, device=device, generator=generator) - .5) * .4
    mean = images.mean(1, keepdim=True)
    saturation = torch.rand(b, 1, 1, 1, device=device, generator=generator) * .6 + .7
    contrast = torch.rand(b, 1, 1, 1, device=device, generator=generator) * .4 + .8
    images = (images + brightness - mean) * saturation + mean
    images = (images - images.mean((1, 2, 3), keepdim=True)) * contrast + images.mean((1, 2, 3), keepdim=True)
    shift = images.shape[-1] // 8
    dx, dy = (int(v) for v in torch.randint(-shift, shift + 1, (2,), device=device, generator=generator))
    images = torch.roll(images, (dy, dx), (2, 3))
    cond = torch.roll(cond, (dy, dx), (2, 3))
    return images, cond


# ---------- features and KID ----------

class FeatureNet:
    """Frozen ImageNet ResNet-18 (the project's offline encoder weights) for similarity, perceptual and KID features."""
    def __init__(self, device="cuda"):
        import os

        import segmentation_models_pytorch as smp
        import torch
        os.environ["HF_HUB_OFFLINE"] = "1"
        self.torch = torch
        self.encoder = smp.encoders.get_encoder("resnet18", weights="imagenet").to(device).eval()
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self.mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(STD, device=device).view(1, 3, 1, 1)

    def stages(self, images_unit):
        x = ((images_unit + 1) / 2 - self.mean) / self.std
        return self.encoder(x)[2:]

    def pooled(self, images_unit):
        return self.stages(images_unit)[-1].mean((2, 3))


def similarity_loss(features, fake, real):
    """Similarity-measure loss (Advanced R-GAN component, implemented as feature cosine distance)."""
    import torch.nn.functional as F
    total = 0
    for a, b in zip(features.stages(fake), features.stages(real)):
        total = total + (1 - F.cosine_similarity(a.flatten(1), b.flatten(1), dim=1)).mean()
    return total


def perceptual_loss(features, fake, real):
    import torch.nn.functional as F
    return sum(F.l1_loss(a, b) for a, b in zip(features.stages(fake), features.stages(real)))


def kid(a, b, subsets, subset_size, seed):
    """Kernel Inception Distance (Binkowski et al. 2018) on any feature matrix: unbiased MMD^2, cubic polynomial kernel."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    d = a.shape[1]
    size = min(subset_size, len(a), len(b))
    values = []
    for _ in range(subsets):
        x, y = a[rng.choice(len(a), size, replace=False)], b[rng.choice(len(b), size, replace=False)]
        kxx, kyy, kxy = (x @ x.T / d + 1) ** 3, (y @ y.T / d + 1) ** 3, (x @ y.T / d + 1) ** 3
        m = size
        values.append((kxx.sum() - np.trace(kxx)) / (m * (m - 1)) + (kyy.sum() - np.trace(kyy)) / (m * (m - 1)) - 2 * kxy.mean())
    return {"mean": float(np.mean(values)), "sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0}


def pooled_features(features, images_uint8, batch=32):
    import torch
    out = []
    with torch.inference_mode():
        for i in range(0, len(images_uint8), batch):
            x = to_unit(torch.as_tensor(np.ascontiguousarray(images_uint8[i:i + batch])).cuda())
            out.append(features.pooled(x).float().cpu().numpy())
    return np.concatenate(out)


# ---------- training, selection, synthesis ----------

LOSS_LIMIT = 1e4


def check_losses(losses, name, iteration):
    """Fail closed when a GAN diverges: non-finite *or* exploding losses (a finite 3e8 is still a broken run)."""
    for key, value in losses.items():
        if not np.isfinite(value) or abs(value) > LOSS_LIMIT:
            raise ProjectError(f"{name}: {key} = {value:.3g} at iteration {iteration} (diverged)")


def gan_directory(kind, budget):
    return inside(f"data/synthetic/gans/{kind}_{budget}")


def generate(generator, layouts, seed, batch=16):
    """Render layouts deterministically (fixed noise seed); padding painted with the cache's mean colour.

    The cuDNN mode is pinned to the one the stored synthetic sets were rendered with (PyTorch defaults), so leftover
    global state - e.g. a U-Net retrain earlier in the process sets cudnn.deterministic=True - cannot switch the
    transposed-convolution algorithm and shift pixel values."""
    import torch
    out = np.empty((len(layouts), 256, 256, 3), dtype=np.uint8)
    noise = torch.Generator(device="cuda").manual_seed(seed)
    generator.eval()
    with torch.inference_mode(), torch.backends.cudnn.flags(enabled=True, benchmark=False, deterministic=False):
        for i in range(0, len(layouts), batch):
            part = torch.as_tensor(np.ascontiguousarray(layouts[i:i + batch])).cuda()
            images = generator(condition(part), noise=noise) if getattr(generator, "uses_noise", False) else generator(condition(part))
            images = ((images.clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy()
            out[i:i + batch] = images
    pad = np.round(MEAN * 255).astype(np.uint8)
    out[np.asarray(layouts) == 255] = pad
    generator.train()
    return out


def train_image_gan(kind, budget, cfg, icfg, family, store):
    """Resumable GAN training on one budget's labelled images; checkpoints scored by validation KID."""
    import torch
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    settings = icfg["image_gan"]
    total = settings["iterations"][str(budget)]
    directory = gan_directory(kind, budget)
    directory.mkdir(parents=True, exist_ok=True)
    budget_ids = set(store.manifest["budgets"][str(budget)]["ids"])
    train_masks, train_ids = masks10_cache(cfg, store, "train")
    keep = [i for i, image_id in enumerate(train_ids) if image_id in budget_ids]
    used_ids = [train_ids[i] for i in keep]
    if set(used_ids) != budget_ids:
        raise ProjectError("GAN training set differs from the budget")
    records = {r["id"]: r for r in store.select("train")}
    real = torch.as_tensor(np.stack([store.get(records[i])[0] for i in used_ids])).cuda()
    layouts = torch.as_tensor(np.ascontiguousarray(train_masks[keep])).cuda()
    val_masks, val_ids = masks10_cache(cfg, store, "val")
    val_records = {r["id"]: r for r in store.select("val")}
    val_real = np.stack([store.get(val_records[i])[0] for i in val_ids])
    features = FeatureNet()
    val_features = pooled_features(features, val_real)
    run_key = digest({"kind": kind, "budget": budget, "settings": settings, "family": settings[kind], "ids": used_ids,
                      "preregistration": read_json("runs/imbalance/preregistration.json")["config_sha256"]})
    torch.manual_seed(icfg["seed"])
    np.random.seed(icfg["seed"])
    generator, discriminator = family.build(settings[kind])
    generator, discriminator = generator.cuda(), discriminator.cuda()
    opt_g = torch.optim.Adam(generator.parameters(), lr=settings["lr_generator"], betas=tuple(settings["betas"]))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=settings["lr_discriminator"], betas=tuple(settings["betas"]))
    sampler = torch.Generator(device="cuda").manual_seed(icfg["seed"])
    history, start, elapsed = [], 0, 0.
    latest = directory / "latest.pt"
    if latest.exists():
        state = torch.load(latest, map_location="cpu", weights_only=False)
        if state["key"] != run_key:
            raise ProjectError(f"{directory.name}: existing GAN state has a different configuration; preserve it first")
        generator.load_state_dict(state["generator"]); discriminator.load_state_dict(state["discriminator"])
        opt_g.load_state_dict(state["opt_g"]); opt_d.load_state_dict(state["opt_d"])
        sampler.set_state(state["sampler"]); torch.set_rng_state(state["torch_rng"]); torch.cuda.set_rng_state(state["cuda_rng"])
        history, start, elapsed = state["history"], state["iteration"], state["elapsed"]
        log(f"{directory.name}: resuming at iteration {start}")
    write_json(directory / "training_set.json", {"budget": budget, "ids": used_ids, "key": run_key})
    session = time.perf_counter()
    running = {}
    for iteration in range(start, total):
        index = torch.randint(0, len(real), (settings["batch_size"],), device="cuda", generator=sampler)
        batch_real, batch_cond = to_unit(real[index]), condition(layouts[index])
        losses = family.step(generator, discriminator, opt_g, opt_d, batch_real, batch_cond, features, settings[kind], sampler)
        check_losses(losses, directory.name, iteration + 1)
        for name, value in losses.items():
            running[name] = running.get(name, 0.) + value
        done = iteration + 1
        if done % 500 == 0:
            log(f"{directory.name} {done}/{total}: " + ", ".join(f"{k}={v / 500:.3f}" for k, v in running.items()))
            running = {}
        if done % settings["checkpoint_every"] == 0 or done == total:
            torch.cuda.synchronize()
            generated = generate(generator, np.asarray(val_masks), icfg["seed"])
            score = kid(pooled_features(features, generated), val_features, icfg["quality"]["kid_subsets"], icfg["quality"]["kid_subset_size"], icfg["seed"])
            path = directory / f"generator_{done:06d}.pt"
            torch.save({"generator": generator.state_dict(), "iteration": done, "key": run_key}, path)
            history.append({"iteration": done, "val_kid": score, "checkpoint": path.name, "sha256": sha256(path)})
            state = {"key": run_key, "generator": generator.state_dict(), "discriminator": discriminator.state_dict(),
                     "opt_g": opt_g.state_dict(), "opt_d": opt_d.state_dict(), "sampler": sampler.get_state(),
                     "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(), "history": history,
                     "iteration": done, "elapsed": elapsed + time.perf_counter() - session}
            torch.save(state, directory / "latest.tmp")
            (directory / "latest.tmp").replace(latest)
            write_json(directory / "history.json", history)
            log(f"{directory.name} checkpoint {done}: validation KID {score['mean']:.4f} ± {score['sd']:.4f}")
    best = min(history, key=lambda h: (h["val_kid"]["mean"], h["iteration"]))
    selection = {"kind": kind, "budget": budget, "key": run_key, "rule": settings["selection"], "selected": best, "history": history,
                 "iterations": total, "training_seconds": (read_json(directory / "selected.json")["training_seconds"]
                                                         if (directory / "selected.json").exists() and start == total else elapsed + time.perf_counter() - session),
                 "selected_at": now()}
    write_json(directory / "selected.json", selection)
    log(f"{directory.name}: selected iteration {best['iteration']} (validation KID {best['val_kid']['mean']:.4f})")
    return selection


def load_selected(kind, budget, family, icfg):
    import torch
    directory = gan_directory(kind, budget)
    selection = read_json(directory / "selected.json")
    path = directory / selection["selected"]["checkpoint"]
    if sha256(path) != selection["selected"]["sha256"]:
        raise ProjectError(f"{path.name} changed after selection")
    generator, _ = family.build(icfg["image_gan"][kind])
    generator.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["generator"])
    return generator.cuda(), selection


def mean_colour_painting(layouts, real_rgb, real_layouts):
    """Trivial 'generator' for the KID reference: each class painted with its mean training colour."""
    colours = np.zeros((11, 3), dtype=np.float64)
    for c in range(10):
        pixels = real_rgb[real_layouts == c]
        colours[c] = pixels.reshape(-1, 3).mean(0) if len(pixels) else 0
    colours[10] = np.round(MEAN * 255)
    index = np.where(layouts == 255, 10, layouts)
    return colours[index].round().astype(np.uint8)


def quality_report(kind, budget, cfg, icfg, family, store, synthetic_rgb):
    """KID (with references), mask consistency with the frozen U-Net, memorisation (eval I4)."""
    import torch

    from .evaluation import UNetEngine
    from .metrics import confusion, score
    q = icfg["quality"]
    features = FeatureNet()
    generator, selection = load_selected(kind, budget, family, icfg)
    val_masks, val_ids = masks10_cache(cfg, store, "val")
    val_records = {r["id"]: r for r in store.select("val")}
    val_real = np.stack([store.get(val_records[i])[0] for i in val_ids])
    generated = generate(generator, np.asarray(val_masks), icfg["seed"])
    budget_ids = set(store.manifest["budgets"][str(budget)]["ids"])
    train_masks, train_ids = masks10_cache(cfg, store, "train")
    keep = [i for i, image_id in enumerate(train_ids) if image_id in budget_ids]
    train_records = {r["id"]: r for r in store.select("train")}
    train_real = np.stack([store.get(train_records[train_ids[i]])[0] for i in keep])
    f_val, f_gen, f_train = pooled_features(features, val_real), pooled_features(features, generated), pooled_features(features, train_real)
    painted = mean_colour_painting(np.asarray(val_masks), train_real, np.asarray(train_masks[keep]))
    f_paint = pooled_features(features, painted)
    kid_args = (q["kid_subsets"], q["kid_subset_size"], icfg["seed"])
    unet_dir = inside(f"runs/unet_{budget}_s{icfg['seed']}")
    engine = UNetEngine(unet_dir, read_json(unet_dir / "run.json"))
    total = np.zeros((3, 3), dtype=np.int64)
    for image, layout in zip(generated, np.asarray(val_masks)):
        prediction = engine.predict(image, None)
        total += confusion(labels_of(layout), prediction)
    generated_score = score(total)["flooded_miou"]
    real_score = read_json(unet_dir / "evaluation_val.json")["metrics"]["flooded_miou"]
    f_syn = pooled_features(features, synthetic_rgb)

    def nearest(a, b):
        a = a / np.linalg.norm(a, axis=1, keepdims=True)
        b = b / np.linalg.norm(b, axis=1, keepdims=True)
        return (a @ b.T).max(1)
    real_to_train, syn_to_train = nearest(f_val, f_train), nearest(f_syn, f_train)
    threshold = float(np.percentile(real_to_train, q["memorisation_percentile"]))
    report = {"kind": kind, "budget": budget, "selected_iteration": selection["selected"]["iteration"],
              "kid": {"generated_vs_real_val": kid(f_gen, f_val, *kid_args), "real_train_vs_real_val": kid(f_train, f_val, *kid_args),
                      "mean_colour_painting_vs_real_val": kid(f_paint, f_val, *kid_args)},
              "mask_consistency": {"frozen_unet": unet_dir.name, "flooded_miou_on_generated_val": generated_score,
                                   "flooded_miou_on_real_val": real_score},
              "memorisation": {"real_val_to_train_median": float(np.median(real_to_train)), "synthetic_to_train_median": float(np.median(syn_to_train)),
                               "real_val_p99": threshold, "synthetic_above_real_p99": float(np.mean(syn_to_train > threshold)),
                               "pass": bool(np.mean(syn_to_train > threshold) <= 0.05)},
              "created_at": now()}
    del engine, generator
    torch.cuda.empty_cache()
    return report
