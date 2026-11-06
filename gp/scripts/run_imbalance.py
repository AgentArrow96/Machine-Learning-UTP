"""Run the pre-registered class-imbalance study end to end (resumable; each stage in its own process).

Order: pre-register -> 10-class layouts -> GAN-free sets and runs (oversample, ChessMix, Dice) -> the three image
GANs -> their sets + quality reports -> tabular sets (SMOTE, tabular R-GAN) -> their runs -> freeze -> test -> report.
Usage: .\\.venv\\Scripts\\python.exe scripts\\run_imbalance.py [--until STAGE]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATUS = ROOT / "output/imbalance/pipeline_status.json"
LOG = ROOT / "output/imbalance/pipeline.log"
PATHS = {"status": STATUS, "log": LOG}


def config():
    return json.loads((ROOT / "configs/imbalance.json").read_text(encoding="utf-8"))


def run(args, status):
    name = " ".join(str(a) for a in args)
    if status.get(name) == "done":
        return
    started = time.perf_counter()
    with PATHS["log"].open("a", encoding="utf-8") as log:
        log.write(f"\n=== {name}\n")
        log.flush()
        result = subprocess.run([sys.executable, "-u", "-m", "floodnet", "imbalance", *map(str, args)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
    status[name] = "done" if result.returncode == 0 else f"failed ({result.returncode})"
    status[f"{name} seconds"] = round(time.perf_counter() - started, 1)
    PATHS["status"].write_text(json.dumps(status, indent=2), encoding="utf-8")
    if result.returncode:
        raise SystemExit(f"Stage failed: {name} (see {PATHS['log']})")


def runs_for(icfg, variants):
    return [(v, m, b, s) for v in variants for m in icfg["variants"][v]["methods"] for b in icfg["budgets"] for s in icfg["train_seeds"]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--until", choices=("free", "gans", "runs", "test", "report"), default="report")
    parser.add_argument("--group", choices=("all", "cpu"), default="all",
                        help="cpu: only the GPU-free stages (SMOTE rows; RF runs of finished GAN sets), to run beside a GAN")
    args = parser.parse_args()
    if args.group == "cpu":
        PATHS.update(status=ROOT / "output/imbalance/pipeline_status_cpu.json", log=ROOT / "output/imbalance/pipeline_cpu.log")
    PATHS["status"].parent.mkdir(parents=True, exist_ok=True)
    status = json.loads(PATHS["status"].read_text(encoding="utf-8")) if PATHS["status"].exists() else {}
    icfg = config()
    budgets = icfg["budgets"]
    if args.group == "cpu":
        # Random Forest training and validation run on the CPU; finished sets are never rebuilt (stage_set).
        for budget in budgets:
            run(["set", "--variant", "smote", "--budget", budget], status)
        for v, m, b, s in runs_for(icfg, ("rgan", "spade", "smote")):
            if m == "rf":
                run(["train", "--variant", v, "--method", m, "--budget", b, "--seed", s], status)
        return
    run(["preregister"], status)
    run(["masks10"], status)
    for variant in ("oversample", "chessmix"):
        for budget in budgets:
            run(["set", "--variant", variant, "--budget", budget], status)
    order = {"rf": 0, "unet": 1, "hybrid": 2}
    for v, m, b, s in sorted(runs_for(icfg, ("oversample", "chessmix", "dice")), key=lambda r: (r[0], order[r[1]], r[2], r[3])):
        run(["train", "--variant", v, "--method", m, "--budget", b, "--seed", s], status)
    if args.until == "free":
        return
    for kind in ("rgan", "spade", "oasis"):
        for budget in budgets:
            run(["gan", "--variant", kind, "--budget", budget], status)
            run(["set", "--variant", kind, "--budget", budget], status)
    for variant in ("smote", "rgan_tabular"):
        for budget in budgets:
            run(["set", "--variant", variant, "--budget", budget], status)
    if args.until == "gans":
        return
    for v, m, b, s in sorted(runs_for(icfg, ("rgan", "spade", "oasis", "smote", "rgan_tabular")), key=lambda r: (r[0], order[r[1]], r[2], r[3])):
        run(["train", "--variant", v, "--method", m, "--budget", b, "--seed", s], status)
    if args.until == "runs":
        return
    run(["freeze"], status)
    run(["test"], status)
    if args.until == "test":
        return
    run(["report"], status)
    run(["export"], status)


if __name__ == "__main__":
    main()
