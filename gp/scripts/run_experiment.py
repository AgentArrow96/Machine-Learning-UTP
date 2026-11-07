"""Run the agreed stages with durable logs and safe stage reuse.

Use the gp virtual environment. Run download separately first if network access
needs approval. This script never modifies the original app or project PDFs.
"""
import os
import argparse
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from floodnet.common import config, environment, now, write_json


def main():
    environment()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-prepared", action="store_true", help="Reuse audited/prepared data and feature caches; train still validates their identities")
    args = parser.parse_args()
    seeds = config()["train_seeds"]
    stages = [] if args.from_prepared else [["audit"], ["prepare"], ["features"]]
    # Overlap uses images only; it is cached and must exist before the frozen test evaluation.
    stages.append(["overlap"])
    runs = [(model, budget, seed) for model in ("rf", "unet") for budget in (25, 100) for seed in seeds]
    for model, budget, seed in runs:
        stages += [["train", "--model", model, "--budget", str(budget), "--seed", str(seed), "--resume"],
                   ["evaluate", "--run", f"runs/{model}_{budget}_s{seed}", "--split", "val"]]
    for model, budget, seed in runs:
        stages.append(["evaluate", "--run", f"runs/{model}_{budget}_s{seed}", "--split", "test", "--final"])
    stages.append(["export-web"])
    stages.append(["report-assets"])
    status = {"started_at": now(), "status": "running", "completed": []}
    logs = PROJECT / "runs/pipeline.log"
    logs.parent.mkdir(parents=True, exist_ok=True)
    with logs.open("a", encoding="utf-8", buffering=1) as log:
        for stage in stages:
            status["active"] = stage
            write_json("runs/pipeline_status.json", status)
            command = ([sys.executable, "-u", str(PROJECT / "scripts/make_report_assets.py")] if stage == ["report-assets"]
                       else [sys.executable, "-u", "-m", "floodnet", *stage])
            message = f"\n[{now()}] STAGE {' '.join(stage)}\n"
            print(message, flush=True); log.write(message)
            child = subprocess.Popen(command, cwd=PROJECT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, encoding="utf-8", errors="replace",
                                     creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0)
            try:
                for line in child.stdout:
                    print(line, end="", flush=True); log.write(line)
                code = child.wait()
            except KeyboardInterrupt:
                child.send_signal(__import__('signal').CTRL_BREAK_EVENT) if os.name == "nt" else child.terminate()
                status.update(status="interrupted", finished_at=now())
                write_json("runs/pipeline_status.json", status)
                raise
            if code:
                status.update(status="failed", exit_code=code, finished_at=now())
                write_json("runs/pipeline_status.json", status)
                raise SystemExit(code)
            status["completed"].append(stage)
    status.update(status="complete", active=None, finished_at=now())
    write_json("runs/pipeline_status.json", status)


if __name__ == "__main__":
    main()
