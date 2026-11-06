"""Run the pre-registered improvement study (resumable; each stage in its own process).

  --group gpu : encoder download, then U-Net (+ hybrid) runs of aux10, recipe, tta, encoder, res512
  --group cpu : 512-px cache, context / 512-px features, then the RF runs of aux10, rf_context, res512
  --group all : both groups in sequence, then combine -> combined runs -> freeze -> test -> report -> export
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
PATHS = {"status": ROOT / "output/improve/pipeline_status.json", "log": ROOT / "output/improve/pipeline.log"}


def run(args, status):
    name = " ".join(str(a) for a in args)
    if status.get(name) == "done":
        return
    started = time.perf_counter()
    with PATHS["log"].open("a", encoding="utf-8") as log:
        log.write(f"\n=== {name}\n")
        log.flush()
        result = subprocess.run([sys.executable, "-u", "-m", "floodnet", "improve", *map(str, args)], cwd=ROOT, stdout=log,
                                stderr=subprocess.STDOUT, check=False)
    status[name] = "done" if result.returncode == 0 else f"failed ({result.returncode})"
    status[f"{name} seconds"] = round(time.perf_counter() - started, 1)
    PATHS["status"].write_text(json.dumps(status, indent=2), encoding="utf-8")
    if result.returncode:
        raise SystemExit(f"Stage failed: {name} (see {PATHS['log']})")


def config():
    return json.loads((ROOT / "configs/improve.json").read_text(encoding="utf-8"))


def train(status, icfg, variant, methods):
    for method in methods:
        for budget in icfg["budgets"]:
            for seed in icfg["train_seeds"]:
                run(["train", "--variant", variant, "--method", method, "--budget", budget, "--seed", seed], status)


def gpu(status, icfg):
    run(["fetch-encoder"], status)
    for variant in ("aux10", "recipe", "tta", "encoder"):
        train(status, icfg, variant, ("unet", "hybrid"))
    while not (ROOT / "data/improve/processed_512.json").exists():   # built by the cpu group
        time.sleep(30)
    train(status, icfg, "res512", ("unet", "hybrid"))


def cpu(status, icfg):
    run(["prepare512"], status)
    run(["features", "--variant", "rf_context"], status)
    run(["features", "--variant", "res512"], status)
    for variant in ("aux10", "rf_context", "res512"):
        train(status, icfg, variant, ("rf",))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", choices=("all", "gpu", "cpu"), default="all")
    args = parser.parse_args()
    if args.group != "all":
        PATHS.update(status=ROOT / f"output/improve/pipeline_status_{args.group}.json", log=ROOT / f"output/improve/pipeline_{args.group}.log")
    PATHS["status"].parent.mkdir(parents=True, exist_ok=True)
    status = json.loads(PATHS["status"].read_text(encoding="utf-8")) if PATHS["status"].exists() else {}
    icfg = config()
    run(["preregister"], status)
    if args.group == "gpu":
        return gpu(status, icfg)
    if args.group == "cpu":
        return cpu(status, icfg)
    cpu(status, icfg)
    gpu(status, icfg)
    run(["combine"], status)
    combine = json.loads((ROOT / "runs/improve/combine.json").read_text(encoding="utf-8"))["components"]
    if any(c in ("rf_context", "res512") for c in combine["rf"]):
        run(["features", "--variant", "combined"], status)
    methods = [m for m, family in (("rf", "rf"), ("unet", "unet"), ("hybrid", "unet")) if combine[family]]
    train(status, icfg, "combined", methods)
    run(["freeze"], status)
    run(["test"], status)
    run(["report"], status)
    run(["export"], status)


if __name__ == "__main__":
    main()
