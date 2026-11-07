"""Four final runs; atomic checkpoints and provenance-aware resume."""
from __future__ import annotations

import importlib.metadata
import platform
import time
import numpy as np
import joblib
import psutil
from sklearn.ensemble import RandomForestClassifier

from .common import ProjectError, ROOT, digest, inside, log, now, read_json, replace_file, sha256, write_json
from .data import Store
from .features import feature_key, training_table


def versions():
    names = ("torch", "torchvision", "segmentation-models-pytorch", "scikit-image", "scikit-learn", "numpy", "Pillow")
    return {"python": platform.python_version(), **{name: importlib.metadata.version(name) for name in names}}


def source_identity():
    return digest({path.name: sha256(path) for path in sorted((ROOT / "src/floodnet").glob("*.py"))})


def identity(cfg, store, method, budget, seed):
    return {"config_hash": digest(cfg), "processed_identity": store.manifest["identity"],
            "audit_identity": store.manifest["audit_identity"], "budget_identity": store.manifest["budgets"][str(budget)]["identity"],
            "feature_identity": feature_key(cfg, store) if method == "rf" else None,
            "method": method, "budget": budget, "seed": seed, "source_identity": source_identity()}


def run_path(method, budget, seed, smoke=False):
    return inside(f"runs/{'smoke/' if smoke else ''}{method}_{budget}_s{seed}")


def reusable_complete_run(directory, old, provenance):
    """A finished run stays valid when only the source hash changed and its checkpoint is intact.

    Run identity records the code that trained it; later edits (for example the export) must
    not force retraining. Configuration, data, budget, method and seed must still match exactly."""
    if old["status"] != "complete":
        return False
    without_source = lambda value: {k: v for k, v in value.items() if k != "source_identity"}
    if without_source(old["identity"]) != without_source(provenance):
        return False
    if sha256(directory / old["checkpoint"]) != old["checkpoint_sha256"]:
        raise ProjectError(f"{directory.name}: completed checkpoint changed since training")
    if old["identity"]["source_identity"] != provenance["source_identity"]:
        log(f"{directory.name}: source changed since training; reusing the completed, checksum-verified run")
    return True


def rf_class_weight(cfg, records):
    """The forest's class weights. 'pixel_clipped' = the U-Net's capped inverse-frequency pixel weights of the same
    budget, so both models handle the class imbalance the same way; any other value is passed to scikit-learn unchanged."""
    choice = cfg["rf"]["class_weight"]
    if choice != "pixel_clipped":
        return choice
    from .neural import class_weights
    return {c: float(w) for c, w in enumerate(class_weights(records, cfg["unet"]["weight_clip"]))}


def train_rf(cfg, budget, seed, resume=False, smoke=False):
    store = Store(cfg)
    directory = run_path("rf", budget, seed, smoke)
    directory.mkdir(parents=True, exist_ok=True)
    run = {"schema_version": 1, "run_id": directory.name, "run_type": "smoke" if smoke else "final",
           "identity": identity(cfg, store, "rf", budget, seed), "config": cfg, "versions": versions(), "started_at": now(), "status": "running"}
    meta = directory / "run.json"
    if meta.exists():
        old = read_json(meta)
        if resume and reusable_complete_run(directory, old, run["identity"]):
            log("RF run already complete; nothing to resume."); return directory
        if old["identity"] != run["identity"]:
            raise ProjectError("Existing RF run has incompatible provenance. Preserve it and use matching configuration.")
        if not resume:
            raise ProjectError("Run exists. Use --resume to reuse/restart this compatible run.")
    write_json(meta, run)
    try:
        prepared = time.perf_counter()
        x, y, references = training_table(cfg, store, budget, seed)
        write_json(directory / "sampled_regions.json", references)
        run["region_count"] = len(y)
        run["region_class_counts"] = np.bincount(y, minlength=3).tolist()
        run["table_seconds"] = time.perf_counter() - prepared
        weights = rf_class_weight(cfg, store.select("train", budget))
        if isinstance(weights, dict):
            run["class_weights"] = [weights[c] for c in range(3)]
        model = RandomForestClassifier(n_estimators=cfg["rf"]["n_estimators"], max_depth=cfg["rf"]["max_depth"],
                                       class_weight=weights, random_state=seed, n_jobs=cfg["rf"]["n_jobs"])
        started = time.perf_counter()
        model.fit(x, y)
        run["training_seconds"] = time.perf_counter() - started
        temporary = directory / "model.tmp.joblib"
        joblib.dump(model, temporary)
        replace_file(temporary, directory / "model.joblib")
        if not np.array_equal(model.predict(x[:100]), joblib.load(directory / "model.joblib").predict(x[:100])):
            raise ProjectError("RF checkpoint reload changed predictions")
        run.update(status="complete", completed_at=now(), checkpoint="model.joblib", checkpoint_sha256=sha256(directory / "model.joblib"),
                   peak_gpu_allocated_bytes=None, host_rss_after_fit_bytes=psutil.Process().memory_info().rss)
        write_json(meta, run)
        log(f"RF {budget}% trained on {len(y):,} regions in {run['training_seconds']:.1f}s")
        return directory
    except BaseException as error:
        run.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=str(error))
        write_json(meta, run)
        raise


def save_checkpoint(path, state):
    import torch
    path = inside(path)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(state, temporary)
    replace_file(temporary, path)


PREFLIGHT = "data/manifests/unet_preflight.json"


def _backward_once(cfg, rgb, mask, batch):
    """One forward/backward pass at this batch size; returns the loss. Its GPU tensors are freed when it returns."""
    import torch
    from .neural import build_model, precision, tensor_image
    model = build_model(cfg).cuda()
    x = tensor_image(rgb)[None].repeat(batch, 1, 1, 1).cuda()
    y = torch.from_numpy(mask.astype(np.int64))[None].repeat(batch, 1, 1).cuda()
    with torch.autocast("cuda", dtype=precision()):
        loss = torch.nn.functional.cross_entropy(model(x), y, ignore_index=255)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"), error_if_nonfinite=True)
    torch.cuda.synchronize()
    return float(loss.detach())


def unet_preflight(cfg, store, path=PREFLIGHT):
    """Select one batch size for both budgets using an actual image and backward pass."""
    import torch
    from .neural import precision, require_cuda
    require_cuda()
    record = store.select("train")[0]
    rgb, mask = store.get(record)
    for batch in (4, 2):
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        try:
            loss = _backward_once(cfg, rgb, mask, batch)
        except torch.OutOfMemoryError:
            loss = None
        torch.cuda.empty_cache()
        if loss is None:
            continue
        result = {"batch_size": batch, "gpu": torch.cuda.get_device_name(), "dtype": str(precision()),
                  "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "config_hash": digest(cfg),
                  "processed_identity": store.manifest["identity"], "tested_at": now(), "loss": loss}
        write_json(path, result)
        return result
    raise ProjectError("U-Net backward preflight failed at batch 4 and 2. Free GPU memory and retry; resolution remains frozen.")


def train_unet(cfg, budget, seed, resume=False, smoke=False):
    import copy
    import os
    # Training uses the pretrained cache verified by doctor; never depend on live Hub metadata.
    os.environ["HF_HUB_OFFLINE"] = "1"
    import torch
    from .neural import (ImageDataset, build_model, class_weights, loader, precision, require_cuda,
                         restore_rng, rng_state, seed_all)
    from .metrics import score
    require_cuda()
    store = Store(cfg)
    preflight_path = inside(PREFLIGHT)
    preflight = read_json(preflight_path) if preflight_path.exists() else None
    if preflight is None or preflight["config_hash"] != digest(cfg) or preflight["processed_identity"] != store.manifest["identity"]:
        preflight = unet_preflight(cfg, store)
    cfg = copy.deepcopy(cfg)
    cfg["unet"]["batch_size"] = preflight["batch_size"]
    seed_all(seed)
    directory = run_path("unet", budget, seed, smoke)
    directory.mkdir(parents=True, exist_ok=True)
    provenance = identity(cfg, store, "unet", budget, seed)
    metadata_path = directory / "run.json"
    checkpoint = None
    if metadata_path.exists():
        old = read_json(metadata_path)
        if resume and reusable_complete_run(directory, old, provenance):
            log("U-Net run already complete."); return directory
        if old["identity"] != provenance:
            raise ProjectError("Checkpoint/configuration/dataset/source identity differs. Incompatible resume rejected.")
        if not resume:
            raise ProjectError("Run exists; use --resume with its exact configuration.")
        if (directory / "latest.pt").exists():
            # These checkpoints are generated locally by this project, never accepted as uploads.
            checkpoint = torch.load(directory / "latest.pt", map_location="cpu", weights_only=False)
            if checkpoint["identity"] != provenance:
                raise ProjectError("Checkpoint identity differs")
    records = store.select("train", budget)
    weights = class_weights(records, cfg["unet"]["weight_clip"])
    training_records = records[:8] if smoke else records
    validation_records = store.select("val")[:4] if smoke else store.select("val")
    training_data = ImageDataset(store, training_records, cfg, seed, training=True)
    validation_data = ImageDataset(store, validation_records, cfg, seed)
    generator = torch.Generator().manual_seed(seed)
    training_loader = loader(training_data, cfg, generator, shuffle=True)
    validation_generator = torch.Generator().manual_seed(seed + 1)
    validation_loader = loader(validation_data, cfg, validation_generator)
    model = build_model(cfg, pretrained=checkpoint is None).cuda()
    optimiser = torch.optim.AdamW(model.parameters(), lr=cfg["unet"]["lr"], weight_decay=cfg["unet"]["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=precision() == torch.float16)
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights, device="cuda"), ignore_index=255)
    history, best, stale, first_epoch, elapsed = [], -1., 0, 0, 0.
    if checkpoint:
        model.load_state_dict(checkpoint["model"]); optimiser.load_state_dict(checkpoint["optimiser"])
        scaler.load_state_dict(checkpoint["scaler"])
        history, best, stale, first_epoch, elapsed = checkpoint["history"], checkpoint["best"], checkpoint["stale"], checkpoint["epoch"] + 1, checkpoint["elapsed_seconds"]
        restore_rng(checkpoint["rng"], generator)
        validation_generator.set_state(checkpoint["validation_generator"])
    run = {"schema_version": 1, "run_id": directory.name, "run_type": "smoke" if smoke else "final", "identity": provenance,
           "config": cfg, "versions": versions(), "status": "running", "started_at": old["started_at"] if metadata_path.exists() else now(),
           "class_weights": weights.tolist(), "gpu": torch.cuda.get_device_name(), "precision": str(precision()),
           "gpu_preflight": preflight, "training_images": len(training_records), "validation_images": len(validation_records),
           "reproducibility": "Seeded Python/NumPy/PyTorch, deterministic augmentation and cuDNN; no cross-version bitwise guarantee."}
    write_json(metadata_path, run)
    torch.cuda.reset_peak_memory_stats()
    session_start = time.perf_counter()
    max_epochs = 1 if smoke else cfg["unet"]["max_epochs"]
    try:
        for epoch in range(first_epoch, max_epochs):
            if stale >= cfg["unet"]["patience"]:
                break
            training_data.epoch.value = epoch
            epoch_start = time.perf_counter()
            model.train()
            total_loss, valid_count = 0., 0
            for x, y in training_loader:
                x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
                optimiser.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=precision()):
                    loss = criterion(model(x), y)
                if not torch.isfinite(loss):
                    raise ProjectError(f"Non-finite training loss at epoch {epoch + 1}")
                scaler.scale(loss).backward()
                scaler.unscale_(optimiser)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"), error_if_nonfinite=True)
                scaler.step(optimiser); scaler.update()
                denominator = float(criterion.weight[y[y != 255]].sum())
                total_loss += float(loss.detach()) * denominator
                valid_count += denominator
            torch.cuda.synchronize()
            train_seconds = time.perf_counter() - epoch_start
            model.eval()
            matrix = torch.zeros((3, 3), dtype=torch.int64, device="cuda")
            val_loss, val_count = 0., 0
            val_start = time.perf_counter()
            with torch.inference_mode():
                for x, y in validation_loader:
                    x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
                    with torch.autocast("cuda", dtype=precision()):
                        output = model(x); loss = criterion(output, y)
                    if not torch.isfinite(loss):
                        raise ProjectError("Non-finite validation loss")
                    valid = y != 255
                    matrix += torch.bincount(y[valid] * 3 + output.argmax(1)[valid], minlength=9).reshape(3, 3)
                    denominator = float(criterion.weight[y[valid]].sum())
                    val_count += denominator; val_loss += float(loss) * denominator
            torch.cuda.synchronize()
            metrics = score(matrix.cpu().numpy())
            current = metrics["flooded_miou"]
            if current is None:
                raise ProjectError("Validation flooded-class IoU is undefined")
            improved = current > best
            best, stale = (current, 0) if improved else (best, stale + 1)
            entry = {"epoch": epoch + 1, "train_loss": total_loss / valid_count, "val_loss": val_loss / val_count,
                     "val_flooded_miou": current, "val_miou": metrics["miou"], "train_seconds": train_seconds,
                     "validation_seconds": time.perf_counter() - val_start, "images_per_second": len(training_records) / train_seconds,
                     "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(), "validation_metrics": metrics}
            history.append(entry)
            state = {"identity": provenance, "model": model.state_dict(), "optimiser": optimiser.state_dict(), "scaler": scaler.state_dict(),
                     "epoch": epoch, "history": history, "best": best, "stale": stale, "rng": rng_state(generator),
                     "validation_generator": validation_generator.get_state(), "elapsed_seconds": elapsed + time.perf_counter() - session_start}
            save_checkpoint(directory / "latest.pt", state)
            if improved:
                save_checkpoint(directory / "best.pt", state)
            write_json(directory / "history.json", history)
            run.update(epochs=len(history), best_validation_flooded_miou=best, training_seconds=state["elapsed_seconds"],
                       peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(), peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved())
            write_json(metadata_path, run)
            log(f"U-Net {budget}% epoch {epoch + 1}/{max_epochs}: loss={entry['train_loss']:.4f}, val flooded IoU={current:.4f}, {train_seconds:.1f}s train")
        selected = torch.load(directory / "best.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(selected["model"])
        model.eval()
        x, _ = validation_data[0]
        with torch.inference_mode(), torch.autocast("cuda", dtype=precision()):
            before = model(x[None].cuda()).argmax(1)
            reloaded = build_model(cfg, pretrained=False).cuda()
            reloaded.load_state_dict(selected["model"]); reloaded.eval()
            after = reloaded(x[None].cuda()).argmax(1)
        if not torch.equal(before, after):
            raise ProjectError("U-Net checkpoint reload changed predictions")
        run.update(status="complete", completed_at=now(), checkpoint="best.pt", checkpoint_sha256=sha256(directory / "best.pt"),
                   best_epoch=selected["epoch"] + 1, stopped_early=len(history) < max_epochs)
        write_json(metadata_path, run)
        return directory
    except BaseException as error:
        run.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=str(error))
        write_json(metadata_path, run)
        raise
