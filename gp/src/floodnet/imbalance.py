"""Class-imbalance study: pre-registration, data recipes, variant runs, own freeze, statistics and the web export.

Pre-registered in configs/imbalance.json and docs/imbalance.md. Baseline = the 18 frozen runs (untouched). Every
recipe draws only from one budget's training images; validation selects GAN checkpoints; test is scored once, after
`runs/imbalance/experiment.json` freezes all variant runs.
"""
from __future__ import annotations

import time

import numpy as np

from .common import (
    ProjectError,
    digest,
    inside,
    log,
    now,
    read_json,
    relative,
    sha256,
    write_json,
)
from .data import MEAN, Store

IMAGE_VARIANTS = ("oversample", "chessmix", "rgan", "oasis", "spade")
GAN_FAMILIES = ("rgan", "oasis", "spade")
ROW_VARIANTS = ("smote", "rgan_tabular")
SMOTE_CHUNK = 256
CONFIG = "configs/imbalance.json"


def load_config():
    return read_json(CONFIG)


def family(kind):
    from . import gan_oasis, gan_spade, rgan
    return {"rgan": rgan, "oasis": gan_oasis, "spade": gan_spade}[kind]


def plan(icfg=None):
    icfg = icfg or load_config()
    return [(variant, method, budget, seed) for variant, spec in icfg["variants"].items() for method in spec["methods"]
            for budget in icfg["budgets"] for seed in icfg["train_seeds"]]


# ---------- pre-registration ----------

def preregister():
    path = inside("runs/imbalance/preregistration.json")
    record = {"config_sha256": sha256(CONFIG), "protocol_sha256": sha256("docs/imbalance.md"),
              "evals_sha256": sha256("audit/imbalance_evals.md"), "runs_planned": len(plan())}
    if path.exists():
        old = read_json(path)
        if {k: old[k] for k in record} != record:
            if inside("data/synthetic/gans").exists():
                raise ProjectError("Pre-registered settings changed after GAN training started; record a ruling and preserve the old study first")
            log("Pre-registration updated before any GAN training")
        else:
            return old
    record["registered_at"] = now()
    write_json(path, record)
    return record


def check_preregistration():
    path = inside("runs/imbalance/preregistration.json")
    if not path.exists():
        raise ProjectError("Run `floodnet imbalance preregister` first")
    record = read_json(path)
    if record["config_sha256"] != sha256(CONFIG):
        raise ProjectError("configs/imbalance.json changed after pre-registration")
    return record


# ---------- data recipes (budget-only) ----------

def oversample_ids(records, budget_ids, count, seed):
    """Control: target-containing budget images drawn with replacement (augmentation then varies each copy)."""
    budget = set(budget_ids)
    pool = sorted(r["id"] for r in records if r["id"] in budget and (r["processed_counts"][1] or r["processed_counts"][2]))
    rng = np.random.default_rng(seed)
    return [pool[i] for i in rng.integers(0, len(pool), count)]


def chessmix(rgb, labels, ids, count, seed, settings):
    """ChessMix (Pereira & dos Santos 2021): chessboard mosaics of patches, preferring patches rich in rare classes.

    Filled squares hold a randomly flipped/turned patch from a budget image chosen with probability proportional to its
    target pixels (+epsilon), taking the window with most target pixels among a few random candidates. Empty squares
    are the cache's mean colour with label 255 (ignored) - our reading of the paper (documented assumption)."""
    rng = np.random.default_rng(seed)
    grid, patch = settings["grid"], settings["patch"]
    size = grid * patch
    targets = np.array([np.isin(lab, (1, 2)).sum() for lab in labels], dtype=np.float64)
    weights = (targets + settings["target_weight_epsilon"]) / (targets + settings["target_weight_epsilon"]).sum()
    pad = np.round(MEAN * 255).astype(np.uint8)
    images = np.empty((count, size, size, 3), dtype=np.uint8)
    masks = np.full((count, size, size), 255, dtype=np.uint8)
    sources = []
    for n in range(count):
        images[n] = pad
        used = []
        for r in range(grid):
            for c in range(grid):
                if (r + c) % 2:
                    continue
                k = int(rng.choice(len(labels), p=weights))
                valid = np.argwhere(labels[k] != 255)
                (top0, left0), (bottom, right) = valid.min(0), valid.max(0)
                best = None
                for _ in range(settings["candidates_per_patch"]):
                    top = int(rng.integers(top0, max(top0, bottom - patch + 1) + 1))
                    left = int(rng.integers(left0, max(left0, right - patch + 1) + 1))
                    window = labels[k][top:top + patch, left:left + patch]
                    if window.shape != (patch, patch):
                        continue
                    score = np.isin(window, (1, 2)).sum() - 10 * (window == 255).sum()
                    if best is None or score > best[0]:
                        best = (score, top, left)
                if best is None:
                    continue
                _, top, left = best
                image_patch, label_patch = rgb[k][top:top + patch, left:left + patch], labels[k][top:top + patch, left:left + patch]
                turns = int(rng.integers(4))
                image_patch, label_patch = np.rot90(image_patch, turns), np.rot90(label_patch, turns)
                if rng.random() < .5:
                    image_patch, label_patch = image_patch[:, ::-1], label_patch[:, ::-1]
                images[n, r * patch:(r + 1) * patch, c * patch:(c + 1) * patch] = image_patch
                masks[n, r * patch:(r + 1) * patch, c * patch:(c + 1) * patch] = label_patch
                used.append(ids[k])
        images[n][masks[n] == 255] = pad
        sources.append(used)
    return images, masks, sources


def smote(x, count, k, seed):
    """SMOTE (Chawla et al. 2002): new rows on segments between a minority row and one of its k nearest neighbours."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=np.float64)
    # Row chunks keep memory at SMOTE_CHUNK x n x d (the all-at-once n x n x d array reached ~14 GB at 100%);
    # the arithmetic per row is unchanged, so the neighbours and rows are identical.
    k = min(k, len(x) - 1)
    neighbours = np.empty((len(x), k), dtype=np.int64)
    for start in range(0, len(x), SMOTE_CHUNK):
        part = ((x[start:start + SMOTE_CHUNK, None, :] - x[None, :, :]) ** 2).sum(-1)
        part[np.arange(len(part)), np.arange(start, start + len(part))] = np.inf
        neighbours[start:start + len(part)] = np.argsort(part, 1)[:, :k]
    rows, pairs = [], []
    for _ in range(count):
        i = int(rng.integers(len(x)))
        j = int(neighbours[i, rng.integers(neighbours.shape[1])])
        u = float(rng.random())
        rows.append(x[i] + u * (x[j] - x[i]))
        pairs.append((i, j, u))
    return np.asarray(rows, dtype=np.float32), pairs


def rbf_mmd(a, b):
    """RBF-kernel MMD^2 (median-distance bandwidth) between two row sets."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    joint = np.concatenate([a, b])
    sq = (joint ** 2).sum(1)
    distances = np.maximum(sq[:, None] + sq[None, :] - 2 * joint @ joint.T, 0)
    bandwidth = max(np.median(distances), 1e-12)
    kernel = np.exp(-distances / bandwidth)
    n = len(a)
    return float(kernel[:n, :n].mean() + kernel[n:, n:].mean() - 2 * kernel[:n, n:].mean())


def region_rows(rgb, labels, geometry, slic):
    """RF rows of a synthetic image: the project's own region descriptors; regions whose majority is 255 are dropped."""
    from .data import crop
    from .features import extract_regions
    x, _, regions = extract_regions(rgb, geometry, slic)
    content, truth = crop(regions, geometry), crop(labels, geometry)
    votes = np.zeros((len(x), 4), dtype=np.int64)
    np.add.at(votes, (content.ravel(), np.where(truth.ravel() == 255, 3, truth.ravel())), 1)
    majority = votes.argmax(1)
    keep = majority < 3
    return x[keep], majority[keep].astype(np.uint8)


# ---------- synthetic sets ----------

def set_dir(variant, budget):
    return inside(f"data/synthetic/sets/{variant}_{budget}")


def budget_ids(store, budget):
    return list(store.manifest["budgets"][str(budget)]["ids"])


def synthetic_count(store, budget, icfg):
    return round(len(budget_ids(store, budget)) * icfg["synthetic_fraction"])


def layouts_for(cfg, icfg, store, budget):
    from .gan_common import make_layouts, masks10_cache
    path = inside(f"data/synthetic/layouts_{budget}.npz")
    masks, ids = masks10_cache(cfg, store, "train")
    allowed = set(budget_ids(store, budget))
    keep = [i for i, image_id in enumerate(ids) if image_id in allowed]
    key = digest({"budget": budget, "ids": [ids[i] for i in keep], "layouts": icfg["layouts"], "seed": icfg["seed"],
                  "count": synthetic_count(store, budget, icfg)})
    if path.exists():
        with np.load(path, allow_pickle=False) as cached:
            if str(cached["key"]) == key:
                return cached["layouts"], read_json(path.with_suffix(".json"))["sources"], key
    layouts, sources = make_layouts(np.asarray(masks[keep]), [ids[i] for i in keep], synthetic_count(store, budget, icfg), icfg["seed"], icfg["layouts"])
    np.savez_compressed(path, layouts=layouts, key=key)
    write_json(path.with_suffix(".json"), {"key": key, "budget": budget, "sources": sources, "created_at": now()})
    return layouts, sources, key


def build_image_set(variant, budget, cfg, icfg, store):
    """Writes rgb.npy / mask.npy / manifest.json for one image variant and budget (or the id list for oversample)."""
    from .gan_common import (
        check_contract,
        generate,
        geometry_of,
        labels_of,
        load_selected,
    )
    directory = set_dir(variant, budget)
    directory.mkdir(parents=True, exist_ok=True)
    count = synthetic_count(store, budget, icfg)
    allowed = budget_ids(store, budget)
    records = {r["id"]: r for r in store.select("train")}
    manifest = {"variant": variant, "budget": budget, "count": count, "created_at": now()}
    started = time.perf_counter()
    if variant == "oversample":
        chosen = oversample_ids(store.select("train"), allowed, count, icfg["seed"])
        manifest.update(ids=chosen, sources=sorted(set(chosen)))
    elif variant == "chessmix":
        ordered = sorted(allowed)
        rgb = np.stack([store.get(records[i])[0] for i in ordered])
        labels = np.stack([store.get(records[i])[1] for i in ordered])
        images, masks, sources = chessmix(rgb, labels, ordered, count, icfg["seed"], icfg["chessmix"])
        np.save(directory / "rgb.npy", images)
        np.save(directory / "mask.npy", masks)
        manifest.update(sources=sorted({s for used in sources for s in used}), geometry=[{"top": 0, "left": 0, "height": 256, "width": 256}] * count,
                        rgb_sha256=sha256(directory / "rgb.npy"), mask_sha256=sha256(directory / "mask.npy"))
    else:
        layouts, sources, layout_key = layouts_for(cfg, icfg, store, budget)
        generator, selection = load_selected(variant, budget, family(variant), icfg)
        images = generate(generator, layouts, icfg["seed"])
        labels = np.stack([labels_of(layout) for layout in layouts])
        contract = check_contract(images, labels, layouts)
        np.save(directory / "rgb.npy", images)
        np.save(directory / "mask.npy", labels)
        manifest.update(sources=sorted({s["base"] for s in sources} | {p for s in sources for p in s["pasted_from"]}),
                        layout_key=layout_key, generator=selection["selected"], contract=contract,
                        geometry=[geometry_of(layout) for layout in layouts],
                        rgb_sha256=sha256(directory / "rgb.npy"), mask_sha256=sha256(directory / "mask.npy"))
    if not set(manifest["sources"]) <= set(allowed):
        raise ProjectError(f"{variant} {budget}%: a source image lies outside the budget")
    manifest["seconds"] = time.perf_counter() - started
    manifest["identity"] = digest({k: v for k, v in manifest.items() if k not in ("created_at", "seconds")})
    write_json(directory / "manifest.json", manifest)
    log(f"Synthetic set {variant} {budget}%: {count} samples from {len(manifest['sources'])} budget images")
    return manifest


def image_region_rows(variant, budget, cfg, store):
    """Cached RF rows for every synthetic image of a set (project descriptors; 255-majority regions dropped)."""
    directory = set_dir(variant, budget)
    manifest = read_json(directory / "manifest.json")
    path = directory / "regions.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as cached:
            if str(cached["identity"]) == manifest["identity"]:
                return [(f"{variant}_{budget}_{i:04d}", cached["x"][a:b], cached["y"][a:b]) for i, (a, b) in enumerate(zip(cached["offsets"][:-1], cached["offsets"][1:]))]
    rgb, labels = np.load(directory / "rgb.npy", mmap_mode="r"), np.load(directory / "mask.npy", mmap_mode="r")
    xs, ys, offsets = [], [], [0]
    for i in range(len(rgb)):
        x, y = region_rows(np.asarray(rgb[i]), np.asarray(labels[i]), manifest["geometry"][i], cfg["slic"])
        xs.append(x); ys.append(y); offsets.append(offsets[-1] + len(y))
    np.savez_compressed(path, x=np.concatenate(xs), y=np.concatenate(ys), offsets=np.array(offsets), identity=manifest["identity"])
    return [(f"{variant}_{budget}_{i:04d}", x, y) for i, (x, y) in enumerate(zip(xs, ys))]


def real_target_rows(cfg, store, budget):
    from .features import feature_path
    rows, labels = [], []
    for record in store.select("train", budget):
        with np.load(feature_path(cfg, store, record), allow_pickle=False) as cached:
            keep = cached["y"] > 0
            rows.append(cached["x"][keep]); labels.append(cached["y"][keep])
    return np.concatenate(rows), np.concatenate(labels)


def val_target_rows(cfg, store):
    from .features import feature_path
    rows, labels = [], []
    for record in store.select("val"):
        with np.load(feature_path(cfg, store, record), allow_pickle=False) as cached:
            keep = cached["y"] > 0
            rows.append(cached["x"][keep]); labels.append(cached["y"][keep])
    return np.concatenate(rows), np.concatenate(labels)


def build_row_set(variant, budget, cfg, icfg, store):
    """SMOTE or tabular R-GAN rows: as many target rows per class as the image R-GAN set gives the RF."""
    directory = set_dir(variant, budget)
    directory.mkdir(parents=True, exist_ok=True)
    match = image_region_rows(icfg["rows_match_variant"], budget, cfg, store)
    target_counts = {c: int(sum((y == c).sum() for _, _, y in match)) for c in (1, 2)}
    x, y = real_target_rows(cfg, store, budget)
    started = time.perf_counter()
    manifest = {"variant": variant, "budget": budget, "target_counts": target_counts, "real_target_rows": {c: int((y == c).sum()) for c in (1, 2)},
                "matched_to": icfg["rows_match_variant"], "created_at": now()}
    if variant == "smote":
        parts = [smote(x[y == c], target_counts[c], icfg["variants"]["smote"]["k"], icfg["seed"] + c)[0] for c in (1, 2)]
        rows = np.concatenate(parts)
        labels = np.concatenate([np.full(target_counts[c], c, dtype=np.uint8) for c in (1, 2)])
    else:
        import torch

        from .rgan import sample, train_tabular
        val_x, val_y = val_target_rows(cfg, store)
        generator, selection = train_tabular(x, (y - 1).astype(np.int64), val_x, (val_y - 1).astype(np.int64), icfg["tabular_gan"], icfg["seed"])
        rows, index = sample(generator, {0: target_counts[1], 1: target_counts[2]}, icfg["seed"])
        labels = (index + 1).astype(np.uint8)
        torch.save(generator.state_dict(), directory / "generator.pt")
        manifest.update(selection=selection, generator_sha256=sha256(directory / "generator.pt"),
                        val_mmd_real_train={c: rbf_mmd(x[y == c][:2000], val_x[val_y == c]) for c in (1, 2)})
    np.savez_compressed(directory / "rows.npz", x=rows, y=labels)
    manifest.update(rows_sha256=sha256(directory / "rows.npz"), seconds=time.perf_counter() - started)
    manifest["identity"] = digest({k: v for k, v in manifest.items() if k not in ("created_at", "seconds")})
    write_json(directory / "manifest.json", manifest)
    log(f"Row set {variant} {budget}%: {len(labels)} target rows {target_counts}")
    return manifest


# ---------- variants for the training hooks ----------

class Variant:
    """What training.py needs from a class-imbalance variant (identity keys, extra data, store wrapper, loss)."""
    def __init__(self, name, budget, store, cfg, icfg):
        self.name, self.budget, self.store, self.cfg, self.icfg = name, budget, store, cfg, icfg
        self.kind = icfg["variants"][name]["kind"]
        self.loss = "dice_ce" if name == "dice" else "ce"
        self.manifest = read_json(set_dir(name, budget) / "manifest.json") if self.kind in ("images", "rows") else None

    def identity(self):
        return {"variant": self.name, "imbalance_config_sha256": sha256(CONFIG),
                "synthetic_identity": self.manifest["identity"] if self.manifest else None}

    def unet_extra(self, budget):
        if self.kind != "images":
            return []
        if self.name == "oversample":
            by_id = {r["id"]: r for r in self.store.select("train")}
            return [by_id[i] for i in self.manifest["ids"]]
        labels = np.load(set_dir(self.name, budget) / "mask.npy", mmap_mode="r")
        return [{"id": f"{self.name}_{budget}_{i:04d}", "split": "synthetic", "synthetic": self.name, "index": i,
                 "geometry": g, "processed_counts": np.bincount(np.asarray(labels[i]).ravel(), minlength=256)[:3].tolist()}
                for i, g in enumerate(self.manifest["geometry"])]

    def wrap_store(self, store):
        from .neural import MixedStore
        if self.kind == "images" and self.name != "oversample":
            return MixedStore(store, {self.name: set_dir(self.name, self.budget)})
        return store

    def rf_extra(self, budget):
        if self.name == "oversample":
            from .features import feature_path
            rows = []
            for k, image_id in enumerate(self.manifest["ids"]):
                with np.load(feature_path(self.cfg, self.store, self.store.by_id[image_id]), allow_pickle=False) as cached:
                    rows.append((f"{image_id}#copy{k}", cached["x"].copy(), cached["y"].copy()))
            return {"images": rows}
        if self.kind == "images":
            return {"images": image_region_rows(self.name, budget, self.cfg, self.store)}
        with np.load(set_dir(self.name, budget) / "rows.npz", allow_pickle=False) as cached:
            x, y = cached["x"].copy(), cached["y"].copy()
        return {"target_rows": (x, y, [f"{self.name}_{budget}_row{i}" for i in range(len(y))])}


def create_variant_hybrid(cfg, variant, budget, seed):
    """Hybrid record for a variant U-Net: its checkpoint + the frozen, validation-selected CRF (not re-tuned)."""
    from .training import run_path, source_identity, versions
    selection = read_json("runs/hybrid_selection.json")
    base_directory = run_path("unet", budget, seed, variant=variant)
    base = read_json(base_directory / "run.json")
    if base["status"] != "complete" or sha256(base_directory / base["checkpoint"]) != base["checkpoint_sha256"]:
        raise ProjectError(f"{relative(base_directory)} is not a completed, intact U-Net run")
    directory = run_path("hybrid", budget, seed, variant=variant)
    ids = base["identity"]
    identity = {"method": "hybrid", "budget": budget, "seed": seed, "processed_identity": ids["processed_identity"],
                "audit_identity": ids["audit_identity"], "budget_identity": ids["budget_identity"],
                "config_hash": digest(selection["selected"]), "base_run_id": relative(base_directory), "base_identity": ids,
                "base_checkpoint_sha256": base["checkpoint_sha256"], "selection_identity": selection["identity"],
                "crf": selection["selected"], "backend": selection["backend"], "source_identity": source_identity(),
                "variant": variant, "imbalance_config_sha256": ids["imbalance_config_sha256"], "synthetic_identity": ids["synthetic_identity"]}
    meta = directory / "run.json"
    if meta.exists():
        old = read_json(meta)
        strip = lambda value: {k: v for k, v in value.items() if k != "source_identity"}
        if old["status"] == "complete" and strip(old["identity"]) == strip(identity):
            return directory
        raise ProjectError(f"{relative(directory)} exists with a different base run")
    directory.mkdir(parents=True, exist_ok=True)
    write_json(meta, {"schema_version": 1, "run_id": directory.name, "run_type": "final", "status": "complete", "identity": identity,
                      "base_run": {"run_id": base["run_id"], "checkpoint": base["checkpoint"], "directory": relative(base_directory)},
                      "checkpoint_sha256": base["checkpoint_sha256"], "config": base["config"], "crf": selection["selected"],
                      "precision": base["precision"], "gpu": base.get("gpu"), "versions": versions(), "training_seconds": None,
                      "epochs": None, "best_epoch": None, "created_at": now(), "completed_at": now(),
                      "note": "No training: refines the variant U-Net's probabilities with the frozen DenseCRF setting."})
    return directory


# ---------- freeze ----------

def gan_quality_complete(icfg):
    missing = [f"{kind}_{budget}" for kind in GAN_FAMILIES for budget in icfg["budgets"]
               if not (set_dir(kind, budget) / "quality.json").exists()]
    return missing


def freeze_imbalance(cfg, store, plan=None):
    """Freeze every variant run after validation and before any variant test evaluation (eval I6)."""
    from .training import run_path
    icfg = load_config() if inside(CONFIG).exists() else None
    planned = plan if plan is not None else globals()["plan"](icfg)
    runs = []
    for variant, method, budget, seed in planned:
        directory = run_path(method, budget, seed, variant=variant)
        if not (directory / "run.json").exists():
            raise ProjectError(f"All class-imbalance runs must be trained and validated before test; missing {relative(directory)}")
        run = read_json(directory / "run.json")
        val_path = directory / "evaluation_val.json"
        if run["status"] != "complete" or not val_path.exists():
            raise ProjectError(f"{relative(directory)} lacks a completed run or its validation result")
        validation = read_json(val_path)
        if validation["run_identity"] != run["identity"] or validation["image_count"] != len(store.select("val")):
            raise ProjectError(f"{relative(directory)}: stale validation result")
        from .evaluation import checkpoint_file
        if sha256(checkpoint_file(directory, run)) != run["checkpoint_sha256"]:
            raise ProjectError(f"{relative(directory)}: checkpoint changed")
        runs.append({"run_id": relative(directory), "variant": variant, "identity": run["identity"],
                     "checkpoint_sha256": run["checkpoint_sha256"], "validation_sha256": sha256(val_path)})
    missing = gan_quality_complete(icfg)
    if missing:
        raise ProjectError(f"GAN quality reports missing before freeze: {missing}")
    preregistration = check_preregistration()
    frozen = {"schema_version": 1, "preregistration": preregistration, "base_experiment_identity": read_json("runs/final_experiment.json")["identity"],
              "hybrid_experiment_identity": read_json("runs/hybrid_experiment.json")["identity"], "runs": runs,
              "gan_quality_sha256": {f"{k}_{b}": sha256(set_dir(k, b) / "quality.json") for k in GAN_FAMILIES for b in icfg["budgets"]}}
    frozen["identity"] = digest(frozen)
    path = inside("runs/imbalance/experiment.json")
    if path.exists():
        old = read_json(path)
        if old["identity"] != frozen["identity"]:
            raise ProjectError("The class-imbalance study is already frozen with different artifacts")
        return old
    frozen["frozen_at"] = now()
    write_json(path, frozen)
    log(f"Class-imbalance study frozen: {len(runs)} runs")
    return frozen


# ---------- statistics ----------

def per_image_matrices(directory):
    result = read_json(directory / "evaluation_test.json")
    return [item["id"] for item in result["per_image"]], np.array([item["metrics"]["confusion_matrix"] for item in result["per_image"]], dtype=np.int64)


def flooded_miou(matrices):
    """Pooled flooded-class mIoU for (..., 3, 3) confusion matrices (vectorised)."""
    tp = np.diagonal(matrices, axis1=-2, axis2=-1)
    union = matrices.sum(-1) + matrices.sum(-2) - tp
    iou = np.divide(tp, union, out=np.full(tp.shape, np.nan), where=union > 0)
    return np.nanmean(iou[..., 1:], -1)


def class_metrics(matrix):
    tp = np.diag(matrix).astype(np.float64)
    out = {}
    for c, name in ((1, "building"), (2, "road")):
        out[f"{name}_iou"] = tp[c] / (matrix[c].sum() + matrix[:, c].sum() - tp[c]) if matrix[c].sum() + matrix[:, c].sum() - tp[c] else None
        out[f"{name}_recall"] = tp[c] / matrix[c].sum() if matrix[c].sum() else None
        out[f"{name}_precision"] = tp[c] / matrix[:, c].sum() if matrix[:, c].sum() else None
    return out


def compare(variant, method, budget, icfg, baseline=None):
    """Seed-matched comparison of a variant with its baseline (default: the frozen original run of the same budget and
    seed; `baseline(method, budget, seed)` may point elsewhere)."""
    from .training import run_path
    seeds = icfg["train_seeds"]
    baseline = baseline or (lambda m, b, s: run_path(m, b, s))
    ids = None
    arms = {"variant": [], "baseline": []}
    for seed in seeds:
        for arm, directory in (("variant", run_path(method, budget, seed, variant=variant)), ("baseline", baseline(method, budget, seed))):
            image_ids, matrices = per_image_matrices(directory)
            if ids is None:
                ids = image_ids
            if image_ids != ids:
                raise ProjectError("Per-image order differs between runs")
            arms[arm].append(matrices)
    variant_m, baseline_m = np.stack(arms["variant"]), np.stack(arms["baseline"])  # (seeds, images, 3, 3)
    v_scores, b_scores = flooded_miou(variant_m.sum(1)), flooded_miou(baseline_m.sum(1))
    differences = v_scores - b_scores
    rng = np.random.default_rng(icfg["statistics"]["bootstrap_seed"])
    n = len(ids)
    counts = np.stack([np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(icfg["statistics"]["bootstrap"])])  # (B, n)
    flat_v, flat_b = variant_m.reshape(len(seeds), n, 9), baseline_m.reshape(len(seeds), n, 9)
    boot_v = np.einsum("bn,snk->bsk", counts, flat_v).reshape(-1, len(seeds), 3, 3)
    boot_b = np.einsum("bn,snk->bsk", counts, flat_b).reshape(-1, len(seeds), 3, 3)
    boot = (flooded_miou(boot_v) - flooded_miou(boot_b)).mean(1)
    alpha = (1 - icfg["statistics"]["ci"]) / 2
    low, high = (float(v) for v in np.quantile(boot, [alpha, 1 - alpha]))
    if low > 0 and np.all(differences > 0):
        decision = "helps"
    elif high < 0 and np.all(differences < 0):
        decision = "hurts"
    else:
        decision = "no clear effect"
    pooled_v, pooled_b = variant_m.sum((0, 1)), baseline_m.sum((0, 1))
    return {"variant": variant, "method": method, "budget": budget, "seeds": seeds,
            "variant_scores": v_scores.tolist(), "baseline_scores": b_scores.tolist(), "differences": differences.tolist(),
            "mean_difference": float(differences.mean()), "ci": [low, high], "p_positive": float((boot > 0).mean()), "decision": decision,
            "variant_classes": {k: [class_metrics(m.sum(0))[k] for m in variant_m] for k in class_metrics(pooled_v)},
            "baseline_classes": {k: [class_metrics(m.sum(0))[k] for m in baseline_m] for k in class_metrics(pooled_b)}}


def report(cfg):
    icfg = load_config()
    started = time.perf_counter()
    combos = sorted({(v, m, b) for v, m, b, _ in plan(icfg)}, key=lambda k: (list(icfg["variants"]).index(k[0]), k[1], k[2]))
    comparisons = [compare(v, m, b, icfg) for v, m, b in combos]
    out = {"schema_version": 1, "created_at": now(), "experiment": read_json("runs/imbalance/experiment.json")["identity"],
           "comparisons": comparisons, "seconds": time.perf_counter() - started}
    write_json("output/imbalance/results.json", out)
    return out


# ---------- stages (driven by `floodnet imbalance <stage>` and scripts/run_imbalance.py) ----------

def stage_gan(cfg, kind, budget):
    from .gan_common import train_image_gan
    check_preregistration()
    store = Store(cfg)
    directory = inside(f"data/synthetic/gans/{kind}_{budget}")
    if (directory / "selected.json").exists():
        selection = read_json(directory / "selected.json")
        if selection["iterations"] == load_config()["image_gan"]["iterations"][str(budget)]:
            log(f"{kind} {budget}%: already trained and selected; reuse")
            return selection
    return train_image_gan(kind, budget, cfg, load_config(), family(kind), store)


def set_complete(variant, budget):
    folder = set_dir(variant, budget)
    needed = ["manifest.json"] + (["quality.json"] if variant in GAN_FAMILIES else []) \
        + (["regions.npz"] if variant not in ROW_VARIANTS and variant != "oversample" else [])
    return all((folder / name).exists() for name in needed)


def stage_set(cfg, variant, budget):
    """Build a synthetic set once. A finished set is never rebuilt: a rewritten npz changes its hash, the set identity
    and therefore the provenance of every run already trained on it (retries must be idempotent)."""
    check_preregistration()
    if set_complete(variant, budget):
        log(f"Synthetic set {variant} {budget}%: already built; reuse")
        return read_json(set_dir(variant, budget) / "manifest.json")
    icfg, store = load_config(), Store(cfg)
    if variant in ROW_VARIANTS:
        return build_row_set(variant, budget, cfg, icfg, store)
    manifest = build_image_set(variant, budget, cfg, icfg, store)
    if variant != "oversample":
        image_region_rows(variant, budget, cfg, store)
    if variant in GAN_FAMILIES:
        from .gan_common import quality_report
        rgb = np.load(set_dir(variant, budget) / "rgb.npy", mmap_mode="r")
        write_json(set_dir(variant, budget) / "quality.json", quality_report(variant, budget, cfg, icfg, family(variant), store, np.asarray(rgb)))
    return manifest


def train_variant(cfg, variant, method, budget, seed):
    from .evaluation import evaluate
    from .training import run_path, train_rf, train_unet
    check_preregistration()
    icfg, store = load_config(), Store(cfg)
    if method == "hybrid":
        directory = create_variant_hybrid(cfg, variant, budget, seed)
    else:
        trainer = train_rf if method == "rf" else train_unet
        directory = trainer(cfg, budget, seed, resume=True, variant=Variant(variant, budget, store, cfg, icfg))
        # Only the data or the loss may differ from the frozen baseline run with the same budget and seed.
        if read_json(directory / "run.json")["identity"]["config_hash"] != read_json(run_path(method, budget, seed) / "run.json")["identity"]["config_hash"]:
            raise ProjectError(f"{relative(directory)} used different model settings than its baseline")
    evaluate(cfg, relative(directory), "val")
    return directory


def stage_test(cfg):
    from .evaluation import evaluate
    from .training import run_path
    for variant, method, budget, seed in plan():
        evaluate(cfg, relative(run_path(method, budget, seed, variant=variant)), "test", final=True)


# ---------- web export ----------

REVIEW = [
    {"name": "Advanced R-GAN", "family": "Regularised GAN (tabular)", "run": "yes", "url": "https://doi.org/10.1016/j.aej.2024.10.084",
     "cite": "Lee, Jung, Moon & Rho, Alexandria Eng. J. 111 (2025)", "what": "Generates minority rows with spectral normalisation, a similarity-measure loss and a CELU discriminator",
     "fit": "Applied to the RF's region rows; its components regularise the image R-GAN"},
    {"name": "Relativistic GAN", "family": "Adversarial loss", "run": "yes (inside R-GAN)", "url": "https://arxiv.org/abs/1807.00734",
     "cite": "Jolicoeur-Martineau (2018)", "what": "The discriminator judges whether real data look more realistic than fake", "fit": "Stabilises the image R-GAN"},
    {"name": "pix2pix", "family": "Mask-to-image cGAN", "run": "yes (R-GAN backbone)", "url": "https://arxiv.org/abs/1611.07004",
     "cite": "Isola et al. (2016)", "what": "U-Net generator + PatchGAN discriminator", "fit": "Renders edited layouts as photos"},
    {"name": "OASIS", "family": "Semantic image synthesis", "run": "yes", "url": "https://arxiv.org/abs/2012.04781",
     "cite": "Sushko et al. (2020)", "what": "Segmentation discriminator with a class-balanced loss, 3D noise, LabelMix", "fit": "Built so rare classes still get realistic texture"},
    {"name": "SPADE / GauGAN", "family": "Semantic image synthesis", "run": "yes", "url": "https://arxiv.org/abs/1903.07291",
     "cite": "Park et al. (2019)", "what": "Spatially-adaptive normalisation keeps the layout through the generator", "fit": "Standard reference"},
    {"name": "ChessMix", "family": "Non-generative mixing (remote sensing)", "run": "yes", "url": "https://arxiv.org/abs/2108.11535",
     "cite": "Pereira & dos Santos (2021)", "what": "Chessboard mosaics of patches favouring rare classes", "fit": "Cheap baseline for the GANs"},
    {"name": "SMOTE", "family": "Tabular oversampling", "run": "yes (RF)", "url": "https://doi.org/10.1613/jair.953",
     "cite": "Chawla et al. (2002)", "what": "Interpolates between minority neighbours", "fit": "Classic remedy for the RF's region rows"},
    {"name": "Generalised Dice", "family": "Loss", "run": "yes (Dice + CE, U-Net)", "url": "https://arxiv.org/abs/1707.03237",
     "cite": "Sudre et al. (2017)", "what": "Overlap-based loss robust to imbalance", "fit": "Loss-side remedy, no new data"},
    {"name": "Spectral normalisation", "family": "GAN regulariser", "run": "used", "url": "https://arxiv.org/abs/1802.05957", "cite": "Miyato et al. (2018)",
     "what": "Bounds each layer's Lipschitz constant", "fit": "Component of the R-GAN"},
    {"name": "DiffAugment", "family": "Limited-data GAN training", "run": "used", "url": "https://arxiv.org/abs/2006.10738", "cite": "Zhao et al. (2020)",
     "what": "Same differentiable augmentation on real and fake", "fit": "Needed with 361-1,445 training images"},
    {"name": "StyleGAN2-ADA", "family": "Unconditional GAN", "run": "no", "url": "https://arxiv.org/abs/2006.06676", "cite": "Karras et al. (2020)",
     "what": "Adaptive discriminator augmentation", "fit": "Makes images without masks; would need a second mask generator"},
    {"name": "BAGAN", "family": "Class-balancing GAN", "run": "no", "url": "https://arxiv.org/abs/1803.09655", "cite": "Mariani et al. (2018)",
     "what": "Generates minority image classes", "fit": "Image-level classes, not pixel labels"},
    {"name": "CB-GAN", "family": "Class-balancing GAN", "run": "no", "url": "https://arxiv.org/abs/2106.09402", "cite": "Rangwani et al. (2021)",
     "what": "Classifier-in-the-loop regulariser for long-tailed generation", "fit": "Image classes only"},
    {"name": "RCGAN", "family": "Regularised cycle GAN", "run": "no", "url": "https://arxiv.org/abs/2001.06591", "cite": "Yang et al. (2020)",
     "what": "Anomaly detection", "fit": "Not a data generator"},
    {"name": "SatSynth", "family": "Diffusion (aerial)", "run": "no", "url": "https://arxiv.org/abs/2403.16605", "cite": "Toker et al. (2024)",
     "what": "Joint aerial image-mask generation", "fit": "Not a GAN; far heavier"},
    {"name": "Long-tail diffusion caution", "family": "Diffusion augmentation", "run": "informs design", "url": "https://arxiv.org/abs/2602.04749",
     "cite": "Wijenayake et al. (2026)", "what": "Generators that mirror the training data keep rare classes rare", "fit": "Why layouts get extra targets"},
    {"name": "Temporal Graph Network (fraud)", "family": "Temporal graph network", "run": "no (lesson recorded)", "url": "https://doi.org/10.3390/a17120552",
     "cite": "Saldana-Ulloa, De Ita Luna & Marcial-Romero, Algorithms 17(12):552 (2024)",
     "what": "Event-based temporal graphs of payment entities; more event types and denser graphs improved detection",
     "fit": "Different data; transferable idea: give the RF relational context between neighbouring superpixels"},
    {"name": "Focal loss", "family": "Loss", "run": "no (Dice chosen)", "url": "https://arxiv.org/abs/1708.02002", "cite": "Lin et al. (2017)",
     "what": "Down-weights easy pixels", "fit": "Alternative loss remedy"},
    {"name": "Class-balanced loss", "family": "Loss weights", "run": "no", "url": "https://arxiv.org/abs/1901.05555", "cite": "Cui et al. (2019)",
     "what": "Effective-number weighting", "fit": "Close to the existing clipped weights"},
    {"name": "Copy-paste", "family": "Augmentation", "run": "informs design", "url": "https://arxiv.org/abs/2012.07177", "cite": "Ghiasi et al. (2020)",
     "what": "Pastes object instances into other images", "fit": "Inspires the layout editing (labels pasted, GAN renders pixels)"},
]


def validation_preview(icfg):
    """Seed-matched validation changes for every finished variant run (a preview; the decision uses the test split)."""
    from .training import run_path
    out = []
    for variant, spec in icfg["variants"].items():
        for method in spec["methods"]:
            for budget in icfg["budgets"]:
                seeds, differences, scores = [], [], []
                for seed in icfg["train_seeds"]:
                    mine = run_path(method, budget, seed, variant=variant) / "evaluation_val.json"
                    if mine.exists():
                        value = read_json(mine)["metrics"]["flooded_miou"]
                        base = read_json(run_path(method, budget, seed) / "evaluation_val.json")["metrics"]["flooded_miou"]
                        seeds.append(seed)
                        scores.append(value)
                        differences.append(value - base)
                out.append({"variant": variant, "method": method, "budget": budget, "seeds": seeds, "scores": scores, "differences": differences,
                            "complete": len(seeds) == len(icfg["train_seeds"])})
    return out


def export_imbalance(cfg):
    """Web export of the study. Works mid-study: anything not produced yet is exported as missing (the page shows it as
    pending) and no value is ever filled in. After the freeze and test evaluation the same call writes the final page."""
    import shutil

    from PIL import Image

    from .export import check_url
    from .gan_common import geometry_of, labels_of
    from .training import run_path
    from .visuals import Writer, class_rgb, overlay, walk_urls
    icfg, store = load_config(), Store(cfg)
    results_path, frozen_path = inside("output/imbalance/results.json"), inside("runs/imbalance/experiment.json")
    final = results_path.exists() and frozen_path.exists()
    results = read_json(results_path) if final else {"comparisons": []}
    frozen = read_json(frozen_path) if frozen_path.exists() else None
    web = inside("web")
    identity = digest({"experiment": frozen["identity"], "results": sha256(results_path)} if final else {"interim": now()})[:16]
    directory = web / f"data/imbalance/{identity}"
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    write = Writer(web, directory)
    original = {"weight_clip": cfg["unet"]["weight_clip"], "unet_class_weights": {}, "rf_region_counts": {}, "unclipped_building_over_other": {}}
    for budget in icfg["budgets"]:
        original["unet_class_weights"][str(budget)] = read_json(run_path("unet", budget, icfg["seed"]) / "run.json")["class_weights"]
        original["rf_region_counts"][str(budget)] = read_json(run_path("rf", budget, icfg["seed"]) / "run.json")["region_class_counts"]
        counts = np.array([r["processed_counts"] for r in store.select("train", budget)], dtype=np.float64).sum(0)
        original["unclipped_building_over_other"][str(budget)] = float(counts[0] / counts[1])
    train = np.array([r["processed_counts"] for r in store.select("train")], dtype=np.float64)
    original["train_pixel_shares"] = (train.sum(0) / train.sum()).tolist()
    original["train_images"] = len(train)
    original["images_without_targets"] = int(((train[:, 1] == 0) & (train[:, 2] == 0)).sum())
    sets = {}
    for variant in [v for v in icfg["variants"] if icfg["variants"][v]["kind"] != "loss"]:
        for budget in icfg["budgets"]:
            path = set_dir(variant, budget) / "manifest.json"
            if path.exists():
                manifest = read_json(path)
                sets[f"{variant}_{budget}"] = {k: manifest[k] for k in ("count", "target_counts", "real_target_rows", "seconds", "selection", "val_mmd_real_train")
                                               if k in manifest} | {"source_images": len(manifest.get("sources", []))}
            else:
                sets[f"{variant}_{budget}"] = None
    gans, gallery, pick = {}, {"layouts": [], "renderings": {}}, list(range(4))
    log_path = inside("output/imbalance/pipeline.log")
    log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    layouts, sources, _ = layouts_for(cfg, icfg, store, 100)
    for i in pick:
        g = geometry_of(layouts[i])
        box = (slice(g["top"], g["top"] + g["height"]), slice(g["left"], g["left"] + g["width"]))
        blank = np.full((256, 256, 3), 241, dtype=np.uint8)
        gallery["layouts"].append({"index": i, "base": sources[i]["base"], "pasted_from": sources[i]["pasted_from"],
                                   "labels": write.png(f"layout_{i}", overlay(blank, labels_of(layouts[i]), class_rgb(), 1, 1)[box])})
    for kind in GAN_FAMILIES:
        for budget in icfg["budgets"]:
            selected, quality = inside(f"data/synthetic/gans/{kind}_{budget}") / "selected.json", set_dir(kind, budget) / "quality.json"
            if selected.exists() and quality.exists():
                selection = read_json(selected)
                active = active_training_seconds(log_text, kind, budget)
                gans[f"{kind}_{budget}"] = {"selected_iteration": selection["selected"]["iteration"], "iterations": selection["iterations"],
                                            "history": [{"iteration": h["iteration"], "kid": h["val_kid"]["mean"]} for h in selection["history"]],
                                            "training_seconds": selection["training_seconds"], "active_seconds": active["active_seconds"],
                                            "pauses": active["pauses"], "quality": read_json(quality)}
            else:
                gans[f"{kind}_{budget}"] = None
        if set_complete(kind, 100):
            rgb = np.load(set_dir(kind, 100) / "rgb.npy", mmap_mode="r")
            gallery["renderings"][kind] = []
            for i in pick:
                g = geometry_of(layouts[i])
                image = np.asarray(rgb[i])[g["top"]:g["top"] + g["height"], g["left"]:g["left"] + g["width"]]
                gallery["renderings"][kind].append(write.webp(f"{kind}_{i}", Image.fromarray(image)))
        else:
            gallery["renderings"][kind] = None
    chess_rgb, chess_mask = np.load(set_dir("chessmix", 100) / "rgb.npy", mmap_mode="r"), np.load(set_dir("chessmix", 100) / "mask.npy", mmap_mode="r")
    gallery["chessmix"] = [{"image": write.webp(f"chessmix_{i}", Image.fromarray(np.asarray(chess_rgb[i]))),
                            "labels": write.png(f"chessmix_{i}_labels", overlay(np.asarray(chess_rgb[i]), np.asarray(chess_mask[i]), class_rgb(), .6, .1))}
                           for i in range(2)]
    costs, validated = {}, 0
    for variant, method, budget, seed in plan(icfg):
        run_directory = run_path(method, budget, seed, variant=variant)
        if (run_directory / "evaluation_val.json").exists():
            validated += 1
        if (run_directory / "run.json").exists():
            run = read_json(run_directory / "run.json")
            if run.get("status") == "complete" and run.get("training_seconds") is not None:
                costs.setdefault(f"{variant}_{method}_{budget}", []).append(run["training_seconds"])
    for method in ("rf", "unet"):
        for budget in icfg["budgets"]:
            costs[f"baseline_{method}_{budget}"] = [read_json(run_path(method, budget, s) / "run.json")["training_seconds"] for s in icfg["train_seeds"]]
    first_test = min(read_json(inside(r["run_id"]) / "evaluation_test.json")["created_at"] for r in frozen["runs"]) if final else None
    out = {"schema_version": 1, "status": "complete" if final else "in progress", "created_at": now(), "identity": identity,
           "experiment_identity": frozen["identity"] if frozen else None,
           "preregistered_at": read_json("runs/imbalance/preregistration.json")["registered_at"],
           "frozen_at": frozen["frozen_at"] if frozen else None, "first_test_evaluation": first_test,
           "progress": {"runs_planned": len(plan(icfg)), "runs_validated": validated,
                        "gans_done": sum(v is not None for v in gans.values()), "gans_planned": len(gans)},
           "config": {k: icfg[k] for k in ("synthetic_fraction", "decision_rule", "unchanged", "image_gan", "tabular_gan", "layouts", "chessmix")},
           "variants": dict(icfg["variants"]), "original": original, "review": REVIEW, "sets": sets, "gans": gans,
           "gallery": gallery, "comparisons": results["comparisons"], "validation_preview": validation_preview(icfg), "costs": costs,
           "runs": len(frozen["runs"]) if frozen else None}
    for url in walk_urls(out, "data/imbalance/"):
        check_url(url, web)
    if "C:" + "\\" in str(out):
        raise ProjectError("Absolute paths leaked into the class-imbalance export")
    write_json(web / "data/imbalance.json", out)
    for old in (web / "data/imbalance").iterdir():
        if old.is_dir() and old.name != identity:
            shutil.rmtree(old)
    log(f"Class-imbalance export ({out['status']}): {validated}/{len(plan(icfg))} runs validated, {len(results['comparisons'])} test comparisons")
    return out


def active_training_seconds(log_text, kind, budget):
    """Training time of one GAN from its own progress lines, with pauses (standby, a stalled machine) taken out.

    Within each session (a `=== gan` stage), every gap between consecutive lines of this GAN counts as logged, except
    gaps longer than 3x the GAN's typical (median) gap: those are pauses and count as one typical gap. The first step of
    a session without a resume line also counts as one typical gap. Re-done iterations after a resume are real work."""
    import re
    from datetime import datetime
    from itertools import pairwise
    header = f"=== gan --variant {kind} --budget {budget}"
    pattern = re.compile(rf"^\[([^\]]+)\] {re.escape(kind)}_{budget}[ :]")
    sessions, current = [], None
    for line in log_text.splitlines():
        if line.startswith("=== "):
            current = [] if line.strip() == header else None
            if current is not None:
                sessions.append(current)
            continue
        match = pattern.match(line)
        if current is not None and match:
            current.append((datetime.fromisoformat(match.group(1)), "resuming" in line, match.group(1)))
    # Only the final generator's lineage counts: drop sessions before the last fresh (non-resumed) start.
    fresh = [i for i, s in enumerate(sessions) if s and not s[0][1]]
    sessions = sessions[fresh[-1]:] if fresh else sessions
    gaps = [(b[0] - a[0]).total_seconds() for s in sessions for a, b in pairwise(s)]
    if not gaps:
        return {"active_seconds": None, "pauses": [], "typical_step_seconds": None}
    typical = float(np.median(gaps))
    active, pauses = 0., []
    for s in sessions:
        if s and not s[0][1]:
            active += typical
        for a, b in pairwise(s):
            gap = (b[0] - a[0]).total_seconds()
            if gap > 3 * typical:
                active += typical
                pauses.append({"from": a[2], "to": b[2], "seconds": gap})
            else:
                active += gap
    return {"active_seconds": active, "pauses": pauses, "typical_step_seconds": typical}
