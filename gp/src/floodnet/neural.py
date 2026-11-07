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
    return smp.Unet(encoder_name=cfg["unet"]["encoder"],
                    encoder_weights=cfg["unet"]["encoder_weights"] if pretrained else None,
                    decoder_channels=tuple(cfg["unet"]["decoder_channels"]), in_channels=3, classes=3, activation=None)


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
