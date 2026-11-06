"""Run the pre-registered uniform / undersampling study (resumable; each stage in its own process).

  --group gpu : U-Nets of the ratio grid (r = 1, 3, 10), then their U-Net + CRF records (CRF runs on the CPU)
  --group cpu : the uniform RF, the RFs of the ratio grid, the blackened-photo feature cache
  --group all : both groups, then ratio choice (validation) -> blackening runs -> freeze -> test -> report -> export
                -> walkthrough (replays the uniform Before forests; needs the GPU)
The gpu and cpu groups can run side by side (own status/log files); `all` reuses everything they finished.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATHS = {"status": ROOT / "output/undersample/pipeline_status.json", "log": ROOT / "output/undersample/pipeline.log"}


def run(args, status):
    name = " ".join(str(a) for a in args)
    if status.get(name) == "done":
        return
    started = time.perf_counter()
    with PATHS["log"].open("a", encoding="utf-8") as log:
        log.write(f"\n=== {name}\n")
        log.flush()
        result = subprocess.run([sys.executable, "-u", "-m", "floodnet", "undersample", *map(str, args)], cwd=ROOT, stdout=log,
                                stderr=subprocess.STDOUT, check=False)
    status[name] = "done" if result.returncode == 0 else f"failed ({result.returncode})"
    status[f"{name} seconds"] = round(time.perf_counter() - started, 1)
    PATHS["status"].write_text(json.dumps(status, indent=2), encoding="utf-8")
    if result.returncode:
        raise SystemExit(f"Stage failed: {name} (see {PATHS['log']})")


def config():
    return json.loads((ROOT / "configs/undersample.json").read_text(encoding="utf-8"))


def train(status, ucfg, kind, methods):
    for method in methods:
        for budget in ucfg["budgets"]:
            for seed in ucfg["train_seeds"]:
                run(["train", "--kind", kind, "--method", method, "--budget", budget, "--seed", seed], status)


def gpu(status, ucfg):
    for ratio in ucfg["undersample"]["ratios"]:
        train(status, ucfg, f"r{ratio}", ("unet", "hybrid"))


def cpu(status, ucfg):
    train(status, ucfg, "uniform", ("rf",))
    for ratio in ucfg["undersample"]["ratios"]:
        train(status, ucfg, f"r{ratio}", ("rf",))
    run(["features-blacken"], status)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", choices=("all", "gpu", "cpu"), default="all")
    args = parser.parse_args()
    if args.group != "all":
        PATHS.update(status=ROOT / f"output/undersample/pipeline_status_{args.group}.json", log=ROOT / f"output/undersample/pipeline_{args.group}.log")
    PATHS["status"].parent.mkdir(parents=True, exist_ok=True)
    status = json.loads(PATHS["status"].read_text(encoding="utf-8")) if PATHS["status"].exists() else {}
    ucfg = config()
    run(["preregister"], status)
    if args.group == "gpu":
        return gpu(status, ucfg)
    if args.group == "cpu":
        return cpu(status, ucfg)
    cpu(status, ucfg)
    gpu(status, ucfg)
    run(["choose"], status)
    train(status, ucfg, "blacken", ("rf", "unet", "hybrid"))
    run(["freeze"], status)
    run(["test"], status)
    run(["report"], status)
    run(["export"], status)
    run(["walkthrough"], status)


if __name__ == "__main__":
    main()
