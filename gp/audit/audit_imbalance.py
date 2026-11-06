"""Independent checks of the class-imbalance study (evals I1-I4, I6, I7, I9, I10). Never imports `floodnet`.

Run twice for pass^2:  .venv\\Scripts\\python.exe audit\\audit_imbalance.py  (writes output/audit/imbalance_<n>.json)
Reproduction checks that need project code (I8) live in audit_imbalance_repro.py.
"""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_eval import (
    feature_dir,
    hybrid_predict,
    load_images,
    load_masks,
    load_unet,
    per_image_confusions,
    rf_expand,
    stored_predictions,
    unet_predict,
)
from independent import (
    MEAN,
    ROOT,
    artifact_hashes,
    guard,
    read_json,
    scores,
    sha256,
    write_json,
)

ICFG = read_json(ROOT / "configs/imbalance.json")
SEEDS, BUDGETS = ICFG["train_seeds"], ICFG["budgets"]


def plan():
    return [(v, m, b, s) for v, spec in ICFG["variants"].items() for m in spec["methods"] for b in BUDGETS for s in SEEDS]


def when(text):
    return datetime.fromisoformat(text)


def remap10(layout):
    """10-class layout -> 3-class labels written from the protocol text (1 -> 1, 3 -> 2, else 0; 255 kept)."""
    out = np.zeros(layout.shape, dtype=np.uint8)
    out[layout == 1] = 1
    out[layout == 3] = 2
    out[layout == 255] = 255
    return out


def budget_sets():
    processed = read_json(ROOT / "data/manifests/processed.json")
    return {int(k): set(v["ids"]) for k, v in processed["budgets"].items()}


def check_isolation_and_contract():
    budgets = budget_sets()
    report = {"I2": [], "I3": []}
    for variant, spec in ICFG["variants"].items():
        if spec["kind"] == "loss":
            continue
        for budget in BUDGETS:
            folder = ROOT / f"data/synthetic/sets/{variant}_{budget}"
            manifest = read_json(folder / "manifest.json")
            sources = set(manifest.get("sources", []))
            entry = {"set": f"{variant}_{budget}", "sources": len(sources), "outside_budget": len(sources - budgets[budget]),
                     "val_or_test": sum(s.startswith(("val_", "test_")) for s in sources)}
            if variant in ("rgan", "oasis", "spade"):
                training = read_json(ROOT / f"data/synthetic/gans/{variant}_{budget}/training_set.json")["ids"]
                entry["gan_training_ids_equal_budget"] = set(training) == budgets[budget]
                layouts = np.load(ROOT / f"data/synthetic/layouts_{budget}.npz")["layouts"]
                rgb, mask = np.load(folder / "rgb.npy", mmap_mode="r"), np.load(folder / "mask.npy", mmap_mode="r")
                pad = np.round(MEAN * 255).astype(np.uint8)
                bad = sum(not np.array_equal(np.asarray(mask[i]), remap10(layouts[i])) or not np.all(np.asarray(rgb[i])[layouts[i] == 255] == pad)
                          for i in range(len(layouts)))
                report["I3"].append({"set": f"{variant}_{budget}", "samples": len(layouts), "contract_failures": int(bad),
                                     "shape_ok": rgb.shape == (len(layouts), 256, 256, 3) and mask.shape == (len(layouts), 256, 256),
                                     "label_ids": sorted(np.unique(np.asarray(mask)).tolist())})
            report["I2"].append(entry)
    layouts_ok = []
    for budget in BUDGETS:
        sources = read_json(ROOT / f"data/synthetic/layouts_{budget}.json")["sources"]
        ids = {s["base"] for s in sources} | {p for s in sources for p in s["pasted_from"]}
        layouts_ok.append({"budget": budget, "images": len(ids), "outside_budget": len(ids - budgets[budget])})
    report["I2_layouts"] = layouts_ok
    rf_refs = []
    for variant, method, budget, seed in plan():
        if method != "rf":
            continue
        refs = read_json(ROOT / f"runs/imbalance/{variant}/rf_{budget}_s{seed}/sampled_regions.json")
        real = {r[0].split("#")[0] for r in refs if r[0].startswith("train_")}
        rf_refs.append({"run": f"{variant}/rf_{budget}_s{seed}", "real_images": len(real), "outside_budget": len(real - budgets[budget]),
                        "val_or_test": sum(r[0].startswith(("val_", "test_")) for r in refs)})
    report["I2_rf_rows"] = rf_refs
    report["I2_pass"] = all(e["outside_budget"] == 0 and e["val_or_test"] == 0 and e.get("gan_training_ids_equal_budget", True) for e in report["I2"]) \
        and all(e["outside_budget"] == 0 for e in layouts_ok) and all(e["outside_budget"] == 0 and e["val_or_test"] == 0 for e in rf_refs)
    report["I3_pass"] = all(e["contract_failures"] == 0 and e["shape_ok"] and set(e["label_ids"]) <= {0, 1, 2, 255} for e in report["I3"])
    return report


def check_chronology():
    prereg = when(read_json(ROOT / "runs/imbalance/preregistration.json")["registered_at"])
    first_gan = min(min(p.stat().st_mtime for p in (ROOT / f"data/synthetic/gans/{k}_{b}").glob("generator_*.pt"))
                    for k in ("rgan", "oasis", "spade") for b in BUDGETS)
    quality = {f"{k}_{b}": when(read_json(ROOT / f"data/synthetic/sets/{k}_{b}/quality.json")["created_at"]) for k in ("rgan", "oasis", "spade") for b in BUDGETS}
    gan_runs = {f"{v}/{m}_{b}_s{s}": when(read_json(ROOT / f"runs/imbalance/{v}/{m}_{b}_s{s}/run.json")["started_at"] if m != "hybrid"
                                          else read_json(ROOT / f"runs/imbalance/{v}/{m}_{b}_s{s}/run.json")["created_at"])
                for v, m, b, s in plan() if v in ("rgan", "oasis", "spade")}
    frozen = read_json(ROOT / "runs/imbalance/experiment.json")
    frozen_at = when(frozen["frozen_at"])
    tests = {r["run_id"]: when(read_json(ROOT / r["run_id"] / "evaluation_test.json")["created_at"]) for r in frozen["runs"]}
    validations = {r["run_id"]: when(read_json(ROOT / r["run_id"] / "evaluation_val.json")["created_at"]) for r in frozen["runs"]}
    return {"I1": {"registered_at": prereg.isoformat(), "first_gan_checkpoint": datetime.fromtimestamp(first_gan).astimezone().isoformat(),
                   "pass": prereg.timestamp() < first_gan,
                   "config_hash_unchanged": read_json(ROOT / "runs/imbalance/preregistration.json")["config_sha256"] == sha256(ROOT / "configs/imbalance.json")},
            "I4": {"latest_quality_report": max(quality.values()).isoformat(), "first_gan_variant_run": min(gan_runs.values()).isoformat(),
                   "pass": all(quality[f"{run.split('/')[0]}_{run.split('_')[1]}"] < t for run, t in gan_runs.items())},
            "I6": {"runs": len(frozen["runs"]), "planned": len(plan()), "frozen_at": frozen_at.isoformat(),
                   "last_validation": max(validations.values()).isoformat(), "first_test": min(tests.values()).isoformat(),
                   "pass": len(frozen["runs"]) == len(plan()) and max(validations.values()) < frozen_at < min(tests.values())}}


def rescore():
    """I7: every variant run re-predicted from its checkpoint and scored with sklearn; compared with its stored predictions."""
    import joblib
    ids, rgb_all, geometry = load_images("test")
    cached = feature_dir()
    features = {i: np.load(cached / f"{i}.npz") for i in ids}
    predictions = {}
    with ProcessPoolExecutor(12) as pool:
        for variant, method, budget, seed in plan():
            run_dir = ROOT / f"runs/imbalance/{variant}/{method}_{budget}_s{seed}"
            key = f"{variant}/{method}_{budget}_s{seed}"
            if method == "rf":
                model = joblib.load(run_dir / "model.joblib")
                model.n_jobs = 1
                predictions[key] = np.stack([rf_expand(model, features[i]["x"], features[i]["regions"]) for i in ids])
            elif method == "unet":
                model, _ = load_unet(run_dir)
                predictions[key], _ = unet_predict(model, rgb_all)
                del model
            else:
                run = read_json(run_dir / "run.json")
                model, _ = load_unet(ROOT / run["base_run"]["directory"])
                _, probs = unet_predict(model, rgb_all)
                predictions[key] = hybrid_predict(probs, rgb_all, geometry, ids, run["crf"], pool)
                del model, probs
    masks = load_masks("test")
    out = {}
    for key, prediction in predictions.items():
        run_dir = ROOT / "runs/imbalance" / key
        theirs, result = stored_predictions(run_dir, ids)
        mine = prediction.copy()
        mine[masks == 255] = 0
        cms = per_image_confusions(prediction, masks)
        total = cms.sum(0)
        out[key] = {"differing_pixels": int(np.count_nonzero(mine != theirs)), "confusion_equal": total.tolist() == result["metrics"]["confusion_matrix"],
                    "flooded_miou": scores(total)["flooded_miou"], "per_image_cms_sha": sha256_bytes(cms.tobytes())}
    return {"runs": out, "pass": all(v["differing_pixels"] == 0 and v["confusion_equal"] for v in out.values())}


def sha256_bytes(data):
    import hashlib
    return hashlib.sha256(data).hexdigest()


def bootstrap_check():
    """I9: an independent re-implementation of the pre-registered bootstrap and decision rule."""
    results = read_json(ROOT / "output/imbalance/results.json")["comparisons"]
    stats = ICFG["statistics"]
    mismatches = []

    def flooded(m):
        tp = np.diagonal(m, axis1=-2, axis2=-1).astype(np.float64)
        union = m.sum(-1) + m.sum(-2) - tp
        with np.errstate(invalid="ignore", divide="ignore"):
            iou = tp / union
        return np.nanmean(iou[..., 1:], -1)
    for c in results:
        arms = []
        for root in (f"runs/imbalance/{c['variant']}", "runs"):
            per_seed = []
            for seed in SEEDS:
                result = read_json(ROOT / root / f"{c['method']}_{c['budget']}_s{seed}" / "evaluation_test.json")
                per_seed.append(np.array([i["metrics"]["confusion_matrix"] for i in result["per_image"]], dtype=np.int64))
            arms.append(np.stack(per_seed))
        v, b = arms
        diffs = flooded(v.sum(1)) - flooded(b.sum(1))
        rng = np.random.default_rng(stats["bootstrap_seed"])
        n = v.shape[1]
        counts = np.stack([np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(stats["bootstrap"])])
        bv = np.einsum("bn,snij->bsij", counts, v)
        bb = np.einsum("bn,snij->bsij", counts, b)
        boot = (flooded(bv) - flooded(bb)).mean(1)
        a = (1 - stats["ci"]) / 2
        low, high = np.quantile(boot, [a, 1 - a])
        decision = "helps" if low > 0 and (diffs > 0).all() else "hurts" if high < 0 and (diffs < 0).all() else "no clear effect"
        same = np.allclose(diffs, c["differences"], atol=1e-12) and abs(low - c["ci"][0]) < 1e-12 and abs(high - c["ci"][1]) < 1e-12 and decision == c["decision"]
        if not same:
            mismatches.append({"comparison": f"{c['variant']}/{c['method']}/{c['budget']}", "mine": [float(low), float(high), decision], "theirs": [*c["ci"], c["decision"]]})
    return {"comparisons": len(results), "mismatches": mismatches, "pass": not mismatches}


def check_g1():
    before = read_json(ROOT / "output/audit/g1_after.json")
    now = artifact_hashes()
    changed = [k for k, v in before.items() if now.get(k) != v]
    return {"original_files": len(before), "changed_or_missing": changed, "pass": not changed}


def main():
    run = sum(1 for _ in (ROOT / "output/audit").glob("imbalance_[0-9].json")) + 1
    started = time.perf_counter()
    report = {"run": run, **check_isolation_and_contract(), **check_chronology()}
    report["I7"] = rescore()
    report["I9"] = bootstrap_check()
    report["I10"] = check_g1()
    report["seconds"] = time.perf_counter() - started
    guard()
    path = write_json(f"imbalance_{run}.json", report)
    summary = {k: v.get("pass") if isinstance(v, dict) else v for k, v in report.items() if k.startswith("I") and isinstance(v, dict)}
    summary.update(I2=report["I2_pass"], I3=report["I3_pass"])
    print(json.dumps({"written": str(path.relative_to(ROOT)), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
