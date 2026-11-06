"""CUDA-only U-Net, reproducible augmentation, and Windows-safe lazy loading."""
from __future__ import annotations

import multiprocessing
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler

from .common import ProjectError
from .data import MEAN, STD


def require_cuda():
    if not torch.cuda.is_available() or torch.version.cuda is None:
        raise ProjectError("CUDA is required. Run scripts/setup.ps1 and doctor --gpu-smoke using gp/.venv. CPU fallback is disabled.")
    return torch.device("cuda:0")


def seed_all(seed):
    torch.set_num_threads(8)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_model(cfg, pretrained=True):
    import segmentation_models_pytorch as smp
    # Improvement study (aux10): 3 prediction logits + 10 auxiliary original-class logits from the same decoder.
    classes = 13 if cfg["unet"].get("aux10_weight") else 3
    return smp.Unet(encoder_name=cfg["unet"]["encoder"],
                    encoder_weights=cfg["unet"]["encoder_weights"] if pretrained else None,
                    decoder_channels=tuple(cfg["unet"]["decoder_channels"]), in_channels=3, classes=classes, activation=None)


def three_class(logits, cfg):
    """The 3-class prediction logits (the auxiliary 10-class head, if any, is never used for prediction)."""
    return logits[:, :3] if cfg["unet"].get("aux10_weight") else logits


def aux_loss(logits, y3, y10, weight, aux_weight, ignore_index=255):
    """Class-weighted 3-class cross-entropy + aux_weight x unweighted 10-class cross-entropy."""
    main = torch.nn.functional.cross_entropy(logits[:, :3], y3, weight=weight.to(logits.dtype), ignore_index=ignore_index)
    extra = torch.nn.functional.cross_entropy(logits[:, 3:], y10, ignore_index=ignore_index)
    return main + aux_weight * extra


def augment_pair(rgb, y3, y10, seed, epoch, index, strength):
    """The training augmentation applied to the image with both label maps: the same seeded draw moves both."""
    image, a3 = augment(rgb, y3, seed, epoch, index, strength)
    _, a10 = augment(rgb, y10, seed, epoch, index, strength)
    return image, a3, a10


def other_keep_probability(records, ratio):
    """Undersampling pipeline: share of 'Other' pixels kept so the budget has `ratio` Other pixels per target pixel."""
    counts = np.array([r["processed_counts"] for r in records], dtype=np.float64).sum(0)
    return float(min(1.0, ratio * (counts[1] + counts[2]) / counts[0]))


def undersample_mask(mask, keep, seed, index):
    """A fixed, seeded subset of 'Other' pixels keeps its label; the rest become 255 (ignored by the loss).
    The draw depends on the image only, so every epoch sees the same subset. Targets and padding are untouched."""
    if keep >= 1:
        return mask
    rng = np.random.default_rng(np.random.SeedSequence([seed, index, 7919]))
    out = mask.copy()
    out[(mask == 0) & (rng.random(mask.shape) >= keep)] = 255
    return out


def blacken(rgb, mask):
    """Blackening test: the photo's 'Other' pixels are painted black; targets and padding are unchanged."""
    out = rgb.copy()
    out[mask == 0] = 0
    return out


def focus_example(rgb, mask, keep, blacken_photo, seed, epoch, index, strength):
    """One training example of the undersampling pipeline: the photo is augmented with its real mask (so dropped
    labels never alter the photo), the undersampled label map follows the same flips and rotations."""
    if blacken_photo:
        rgb = blacken(rgb, mask)
    image, _, label = augment_pair(rgb, mask, undersample_mask(mask, keep, seed, index), seed, epoch, index, strength)
    return image, label


def cosine_lr(epoch, max_epochs, lr, lr_min):
    import math
    return lr_min + (lr - lr_min) * (1 + math.cos(math.pi * epoch / max(1, max_epochs - 1))) / 2


def ema_update(ema_model, model, decay):
    """Exponential moving average of parameters and buffers (BatchNorm statistics are averaged too)."""
    with torch.no_grad():
        for e, m in zip(ema_model.state_dict().values(), model.state_dict().values()):
            if e.dtype.is_floating_point:
                e.mul_(decay).add_(m.detach(), alpha=1 - decay)
            else:
                e.copy_(m)


def dihedral_mean(probability_fn, x):
    """Test-time augmentation: average class probabilities over the 8 flips/rotations, each mapped back."""
    total = None
    for transpose in (False, True):
        base = x.transpose(2, 3) if transpose else x
        for flip_v in (False, True):
            for flip_h in (False, True):
                t = base.flip(2) if flip_v else base
                t = t.flip(3) if flip_h else t
                p = probability_fn(t)
                p = p.flip(3) if flip_h else p
                p = p.flip(2) if flip_v else p
                p = p.transpose(2, 3) if transpose else p
                total = p if total is None else total + p
    return total / 8


def precision():
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def tensor_image(rgb):
    values = (rgb.astype(np.float32) / 255 - MEAN) / STD
    return torch.from_numpy(np.ascontiguousarray(values.transpose(2, 0, 1)))


def augment(rgb, mask, seed, epoch, index, strength):
    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch, index]))
    if rng.random() < 0.5:
        rgb, mask = np.fliplr(rgb), np.fliplr(mask)
    if rng.random() < 0.5:
        rgb, mask = np.flipud(rgb), np.flipud(mask)
    k = int(rng.integers(4))
    rgb, mask = np.rot90(rgb, k), np.rot90(mask, k)
    contrast, brightness = rng.uniform(1-strength, 1+strength), rng.uniform(-strength, strength)
    values = np.clip(((rgb.astype(np.float32) / 255 - .5) * contrast + .5 + brightness) * 255, 0, 255).astype(np.uint8)
    values[mask == 255] = np.round(MEAN * 255).astype(np.uint8)
    return np.ascontiguousarray(values), np.ascontiguousarray(mask)


class MixedStore:
    """Real cache plus synthetic arrays behind the same `.get(record)` interface (class-imbalance study).

    Synthetic records carry {"synthetic": key, "index": i}; their arrays live in `sources[key]/{rgb,mask}.npy`.
    Like Store, pickling drops the memory maps so Windows workers reopen them."""
    def __init__(self, store, sources):
        self.store, self.sources, self.arrays = store, {k: str(v) for k, v in sources.items()}, {}

    def __getstate__(self):
        state = self.__dict__.copy()
        state["arrays"] = {}
        return state

    def get(self, record):
        key = record.get("synthetic")
        if key is None:
            return self.store.get(record)
        if key not in self.arrays:
            from pathlib import Path
            folder = Path(self.sources[key])
            self.arrays[key] = (np.load(folder / "rgb.npy", mmap_mode="r"), np.load(folder / "mask.npy", mmap_mode="r"))
        rgb, mask = self.arrays[key]
        return rgb[record["index"]].copy(), mask[record["index"]].copy()


class DiceCrossEntropy(torch.nn.Module):
    """Weighted cross-entropy + soft Dice over the two target classes (equal weight; padding 255 ignored).

    Exposes `.weight` like CrossEntropyLoss so the training loop's loss bookkeeping is unchanged."""
    def __init__(self, weight, targets=(1, 2), ignore_index=255):
        super().__init__()
        self.register_buffer("weight", torch.as_tensor(weight, dtype=torch.float32))
        self.targets, self.ignore_index = targets, ignore_index

    def forward(self, logits, target):
        ce = torch.nn.functional.cross_entropy(logits, target, weight=self.weight.to(logits.dtype), ignore_index=self.ignore_index)
        valid = (target != self.ignore_index).unsqueeze(1)
        probs = torch.softmax(logits.float(), 1) * valid
        onehot = torch.nn.functional.one_hot(target.clamp(max=logits.shape[1] - 1), logits.shape[1]).permute(0, 3, 1, 2).float() * valid
        dice = []
        for c in self.targets:
            p, t = probs[:, c], onehot[:, c]
            dice.append(1 - (2 * (p * t).sum() + 1) / (p.sum() + t.sum() + 1))
        return ce + torch.stack(dice).mean()


class ImageDataset(Dataset):
    def __init__(self, store, records, cfg, seed, training=False):
        self.store, self.records, self.cfg, self.seed, self.training = store, records, cfg, seed, training
        self.epoch = multiprocessing.Value("i", 0)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        rgb, mask = self.store.get(self.records[index])
        if self.training:
            rgb, mask = augment(rgb, mask, self.seed, self.epoch.value, index, self.cfg["unet"]["brightness_contrast"])
        return tensor_image(rgb), torch.from_numpy(mask.astype(np.int64))


class AuxImageDataset(ImageDataset):
    """Training images with the 3-class and the 10-class label maps (aux10 variant)."""
    def __init__(self, store, records, cfg, seed, masks10):
        super().__init__(store, records, cfg, seed, training=True)
        self.masks10 = masks10

    def __getitem__(self, index):
        rgb, mask = self.store.get(self.records[index])
        y10 = np.asarray(self.masks10[self.records[index]["id"]])
        rgb, mask, y10 = augment_pair(rgb, mask, y10, self.seed, self.epoch.value, index, self.cfg["unet"]["brightness_contrast"])
        return tensor_image(rgb), torch.from_numpy(mask.astype(np.int64)), torch.from_numpy(y10.astype(np.int64))


class FocusImageDataset(ImageDataset):
    """Undersampling pipeline: fixed subset of 'Other' pixels supervised; optionally blackened photos (blackening test)."""
    def __init__(self, store, records, cfg, seed, keep, blacken_photo):
        super().__init__(store, records, cfg, seed, training=True)
        self.keep, self.blacken_photo = keep, blacken_photo

    def __getitem__(self, index):
        rgb, mask = self.store.get(self.records[index])
        image, label = focus_example(rgb, mask, self.keep, self.blacken_photo, self.seed, self.epoch.value, index,
                                     self.cfg["unet"]["brightness_contrast"])
        return tensor_image(image), torch.from_numpy(label.astype(np.int64))


def worker_seed(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed); random.seed(seed)
    torch.set_num_threads(1)


class EpochSampler(Sampler):
    """Order depends on epoch, not persistent-worker creation or resumed iterator state."""
    def __init__(self, dataset, seed):
        self.dataset, self.seed = dataset, seed

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.dataset.epoch.value, 867]))
        return iter(rng.permutation(len(self.dataset)).tolist())


def loader(dataset, cfg, generator, shuffle=False):
    workers = cfg["unet"]["num_workers"]
    kwargs = {"batch_size": cfg["unet"]["batch_size"], "shuffle": False, "num_workers": workers,
              "pin_memory": True, "generator": generator, "worker_init_fn": worker_seed}
    if shuffle:
        kwargs["sampler"] = EpochSampler(dataset, dataset.seed)
    if workers:
        kwargs.update(persistent_workers=True, prefetch_factor=2, multiprocessing_context="spawn")
    return DataLoader(dataset, **kwargs)


def class_weights(records, limits):
    counts = np.array([r["processed_counts"] for r in records], dtype=np.float64).sum(0)
    if np.any(counts == 0):
        raise ProjectError(f"Training subset lacks a target class: {counts.tolist()}")
    weights = 1 / (counts / counts.sum())
    return np.clip(weights / weights.mean(), *limits).astype(np.float32)


def rng_state(generator):
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all(), "loader": generator.get_state()}


def restore_rng(state, generator):
    random.setstate(state["python"]); np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"]); torch.cuda.set_rng_state_all(state["cuda"])
    generator.set_state(state["loader"])
