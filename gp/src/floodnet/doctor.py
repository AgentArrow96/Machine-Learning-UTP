"""Actionable dependency and GPU diagnostics; smoke results are not model scores."""
import importlib.metadata
import shutil
import sys
import time
import psutil
from .common import ROOT, ProjectError, digest, now, write_json


def doctor(cfg, gpu_smoke=False):
    report = {"created_at": now(), "python": sys.version, "executable": sys.executable, "root": str(ROOT),
              "ram_bytes": psutil.virtual_memory().total, "free_disk_bytes": shutil.disk_usage(ROOT).free,
              "logical_cpus": psutil.cpu_count(), "dependencies": {}, "gpu_smoke": None}
    for name in ("torch", "torchvision", "segmentation-models-pytorch", "scikit-image", "scikit-learn", "numpy", "Pillow", "pydensecrf2", "pytest"):
        try:
            report["dependencies"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            report["dependencies"][name] = None
    try:
        import torch
        report["cuda_available"] = torch.cuda.is_available()
        report["torch_cuda_build"] = torch.version.cuda
        if report["cuda_available"]:
            report["gpu"] = torch.cuda.get_device_name()
            report["gpu_capability"] = list(torch.cuda.get_device_capability())
            report["gpu_total_bytes"] = torch.cuda.get_device_properties(0).total_memory
        if gpu_smoke:
            from .neural import build_model, precision, require_cuda, seed_all
            require_cuda(); seed_all(cfg["train_seeds"][0])
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            model = build_model(cfg).cuda()
            optimiser = torch.optim.AdamW(model.parameters(), lr=cfg["unet"]["lr"])
            x = torch.randn(2, 3, cfg["size"], cfg["size"], device="cuda")
            y = torch.randint(0, 3, (2, cfg["size"], cfg["size"]), device="cuda")
            with torch.autocast("cuda", dtype=precision()):
                loss = torch.nn.functional.cross_entropy(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"), error_if_nonfinite=True)
            optimiser.step(); torch.cuda.synchronize()
            report["gpu_smoke"] = {"passed": True, "kind": "synthetic_tensor_hardware_check_only", "loss": float(loss.detach()),
                                   "seconds": time.perf_counter() - started, "precision": str(precision()),
                                   "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "config_hash": digest(cfg)}
    except Exception as error:
        report["error"] = str(error)
        write_json("runs/doctor.json", report)
        raise ProjectError(f"Environment verification failed: {error}") from error
    write_json("runs/doctor.json", report)
    return report
