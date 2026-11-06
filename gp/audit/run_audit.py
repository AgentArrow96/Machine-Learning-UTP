"""Run the whole audit: python audit/run_audit.py [--skip-retrain]

Order: G1 baseline -> data x2 -> independent evaluation x2 (+controls) -> retrain 12 models -> independent
evaluation of the retrained models -> statistics x2 -> verification loop -> figures -> G1 check -> report.
Each step runs in its own process so the independent steps can prove `floodnet` was never imported.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

AUDIT = Path(__file__).resolve().parent
ROOT = AUDIT.parent
OUT = ROOT / "output/audit"
PY = sys.executable


def step(name, *args):
    started = time.perf_counter()
    print(f"\n=== {name} ===", flush=True)
    result = subprocess.run([PY, "-u", *map(str, args)], cwd=ROOT)
    print(f"--- {name}: exit {result.returncode} in {time.perf_counter() - started:.0f}s", flush=True)
    if result.returncode:
        raise SystemExit(f"{name} failed with exit code {result.returncode}")


def keep(source, target):
    shutil.copyfile(OUT / source, OUT / target)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-retrain", action="store_true", help="reuse output/audit/retrain.json from an earlier full run")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    step("G1 baseline", "-c", "import sys; sys.path.insert(0, 'audit'); from independent import artifact_hashes, write_json; write_json('g1_before.json', artifact_hashes())")
    for run in ("A", "B"):
        step(f"data checks ({run})", AUDIT / "audit_data.py")
        keep("data.json", f"data_{run}.json")
    for run in ("A", "B"):
        step(f"independent evaluation + controls ({run})", AUDIT / "audit_eval.py", "--controls")
        keep("eval_frozen.json", f"eval_frozen_{run}.json")
    if not (args.skip_retrain and (OUT / "retrain.json").exists()):
        step("retrain 12 models from scratch", AUDIT / "audit_retrain.py")
    scratch = json.loads((OUT / "retrain.json").read_text(encoding="utf-8"))["scratch"]
    step("independent evaluation of retrained models", AUDIT / "audit_eval.py", "--root", ROOT / scratch, "--tag", "retrained")
    for run in ("A", "B"):
        step(f"statistics ({run})", AUDIT / "audit_stats.py")
        keep("stats.json", f"stats_{run}.json")
    step("verification loop", AUDIT / "audit_code.py")
    step("proof figures", AUDIT / "make_audit_figures.py")
    # Root-cause investigations of the D5/D6b findings (systematic-debugging); independent of `floodnet`.
    for name in ("d6b_investigate", "d6b_orientation", "d5_low", "d5_low2", "d5_threeclass", "d5_view"):
        step(f"investigation {name}", AUDIT / "investigations" / f"{name}.py")
    step("G1 after", "-c", "import sys; sys.path.insert(0, 'audit'); from independent import artifact_hashes, write_json; write_json('g1_after.json', artifact_hashes())")
    step("report", AUDIT / "audit_report.py")
    print(f"\nAudit finished in {(time.perf_counter() - started) / 60:.1f} min. Report: docs/audit.md; evidence: output/audit/")


if __name__ == "__main__":
    main()
