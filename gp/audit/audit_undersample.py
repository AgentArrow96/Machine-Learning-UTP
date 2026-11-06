"""Independent checks of the uniform Before and undersampling study (evals U1-U7). Never imports `floodnet`.

Run twice for pass^2:  .venv\\Scripts\\python.exe audit\\audit_undersample.py  (writes output/audit/undersample_<n>.json)
Written from configs/undersample.json and docs/undersample.md. Inputs and masks are the arrays audit_data.py derived from
the raw release; scoring uses scikit-learn; the bootstrap, the ratio rule and the class weights are re-implemented here.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_eval import WORKERS, independent_rf_features, per_image_confusions, stored_predictions
from independent import IGNORE, ROOT, artifact_hashes, guard, load_derived, read_json, scores, sha256, write_json

UCFG = read_json(ROOT / "configs/undersample.json")
SEEDS, BUDGETS, RATIOS = UCFG["train_seeds"], UCFG["budgets"], UCFG["undersample"]["ratios"]
FAMILY = {"rf": "rf", "unet": "unet", "hybrid": "unet"}
STUDY = ROOT / "runs/undersample"
CLIP = (0.25, 4.0)


def when(text):
    return datetime.fromisoformat(text)


def run_dir(kind, method, budget, seed):
    return STUDY / kind / f"{method}_{budget}_s{seed}"


def baseline_dir(method, budget, seed):
    """docs/undersample.md: the uniform Before = the uniform RF, or the frozen original U-Net / U-Net + CRF."""
    return run_dir("uniform", "rf", budget, seed) if method == "rf" else ROOT / f"runs/{method}_{budget}_s{seed}"


def plan():
    runs = [("uniform", "rf", b, s) for b in BUDGETS for s in SEEDS]
    runs += [(f"r{r}", m, b, s) for r in RATIOS for m in UCFG["undersample"]["methods"] for b in BUDGETS for s in SEEDS]
    runs += [("blacken", m, b, s) for m in UCFG["blacken"]["methods"] for b in BUDGETS for s in SEEDS]
    return runs


def started(run):
    return when(run.get("started_at") or run["created_at"])


def val_score(directory):
    return read_json(directory / "evaluation_val.json")["metrics"]["flooded_miou"]


# ---------- U6: ratio rule ----------

def my_choice():
    choice = {"unet": {}, "rf": {}}
    for family in choice:
        for budget in BUDGETS:
            means = {r: float(np.mean([val_score(run_dir(f"r{r}", family, budget, s)) for s in SEEDS])) for r in RATIOS}
            best = max(means.values())
            choice[family][str(budget)] = min(r for r, m in means.items() if m == best)
    return choice


def tested(choice):
    return [(k, m, b, s) for k, m, b, s in plan() if k in ("uniform", "blacken") or k == f"r{choice[FAMILY[m]][str(b)]}"]


def check_ratio():
    stored = read_json(STUDY / "ratio.json")
    mine = my_choice()
    return {"mine": mine, "stored": stored["choice"], "pass": mine == stored["choice"]}


# ---------- U1 / U3: chronology ----------

def check_chronology(choice):
    prereg = read_json(STUDY / "preregistration.json")
    runs = {(k, m, b, s): read_json(run_dir(k, m, b, s) / "run.json") for k, m, b, s in plan()}
    first = min(started(r) for r in runs.values())
    u1 = {"registered_at": prereg["registered_at"], "first_run_started": first.isoformat(),
          "config_hash_unchanged": prereg["config_sha256"] == sha256(ROOT / "configs/undersample.json"),
          "evals_hash_unchanged": prereg["evals_sha256"] == sha256(ROOT / "audit/undersample_evals.md"),
          "runs_planned": prereg["runs_planned"], "plan_size": len(runs)}
    u1["pass"] = (when(prereg["registered_at"]) < first and u1["config_hash_unchanged"] and u1["evals_hash_unchanged"]
                  and prereg["runs_planned"] == len(runs))
    frozen = read_json(STUDY / "experiment.json")
    frozen_at = when(frozen["frozen_at"])
    expected = tested(choice)
    validations = [when(read_json(run_dir(*key) / "evaluation_val.json")["created_at"]) for key in runs]
    tests = {key: read_json(run_dir(*key) / "evaluation_test.json") for key in expected}
    untested_with_test = [f"{k}/{m}_{b}_s{s}" for k, m, b, s in plan() if (k, m, b, s) not in expected and (run_dir(k, m, b, s) / "evaluation_test.json").exists()]
    ratio_at = when(read_json(STUDY / "ratio.json")["decided_at"])
    blacken_started = [started(r) for key, r in runs.items() if key[0] == "blacken"]
    u3 = {"frozen_runs": len(frozen["runs"]), "planned": len(runs), "tested": len(expected), "frozen_at": frozen_at.isoformat(),
          "last_validation": max(validations).isoformat(), "first_test": min(when(t["created_at"]) for t in tests.values()).isoformat(),
          "frozen_set_equals_plan": sorted(r["run_id"] for r in frozen["runs"]) == sorted(f"runs/undersample/{k}/{m}_{b}_s{s}" for k, m, b, s in runs),
          "tested_set_equals_rule": sorted(frozen["tested"]) == sorted(f"runs/undersample/{k}/{m}_{b}_s{s}" for k, m, b, s in expected),
          "tests_bound_to_freeze": all(t["experiment_identity"] == frozen["identity"] for t in tests.values()),
          "untested_runs_with_a_test_file": untested_with_test,
          "ratio_decided_before_blackening_runs": all(ratio_at < t for t in blacken_started)}
    u3["pass"] = (u3["frozen_set_equals_plan"] and u3["tested_set_equals_rule"] and u3["tests_bound_to_freeze"] and not untested_with_test
                  and u3["ratio_decided_before_blackening_runs"] and max(validations) < frozen_at < min(when(t["created_at"]) for t in tests.values()))
    return {"U1": u1, "U3": u3}


# ---------- U2: frozen artifacts ----------

def committed_sha(path):
    blob = subprocess.run(["git", "-C", str(ROOT.parent), "show", f"HEAD:gp/{path}"], capture_output=True, check=True).stdout
    import hashlib
    return hashlib.sha256(blob).hexdigest()


def check_frozen():
    before = read_json(ROOT / "output/audit/g1_after.json")
    now = artifact_hashes()
    changed = [k for k, v in before.items() if now.get(k) != v]
    bundles = {p: sha256(ROOT / p) == committed_sha(p) for p in ("web/data/results.json", "web/data/after.json")}
    return {"original_files": len(before), "changed_or_missing": changed, "bundles_equal_committed": bundles,
            "pass": not changed and all(bundles.values())}


# ---------- U4: rules in the saved artifacts ----------

def budget_counts(ids_by_budget):
    ids, _, masks, _ = load_derived("train")
    index = {str(i): n for n, i in enumerate(ids)}
    return {b: np.array([np.bincount(masks[index[i]][masks[index[i]] != IGNORE].ravel(), minlength=3)[:3] for i in v]).sum(0)
            for b, v in ids_by_budget.items()}


def my_weights(counts):
    inverse = 1 / (counts / counts.sum())
    return np.clip(inverse / inverse.mean(), *CLIP)


def regions(cache, ids):
    totals = np.zeros(3, dtype=np.int64)
    for i in ids:
        with np.load(cache / f"{i}.npz") as f:
            totals += np.bincount(f["y"], minlength=3)[:3]
    return totals


def blacken_features(pool, run, ids):
    """Blacken the independently derived photo on its 'Other' pixels and rebuild the 22 descriptors here."""
    cache = ROOT / "data/features" / run["identity"]["feature_identity"][:16]
    all_ids, rgb, masks, geometry = load_derived("train")
    index = {str(i): n for n, i in enumerate(all_ids)}
    jobs = []
    for i in ids:
        image = rgb[index[i]].copy()
        image[masks[index[i]] == 0] = 0
        jobs.append((image, geometry[i]))
    worst, shapes = 0.0, True
    for i, (x, _) in zip(ids, pool.map(independent_rf_features, jobs, chunksize=2)):
        with np.load(cache / f"{i}.npz") as f:
            shapes &= x.shape == f["x"].shape
            if x.shape == f["x"].shape:
                worst = max(worst, float(np.abs(x - f["x"]).max()))
    real_cache = ROOT / "data/features" / read_json(ROOT / "runs/rf_25_s42/run.json")["identity"]["feature_identity"][:16]
    with np.load(cache / f"{ids[0]}.npz") as black, np.load(real_cache / f"{ids[0]}.npz") as real:
        differs = black["x"].shape != real["x"].shape or not np.array_equal(black["x"], real["x"])
    only_train = not any(p.name.startswith(("val_", "test_")) for p in cache.glob("*.npz"))
    return {"images": len(ids), "same_region_count": shapes, "max_abs_difference": worst, "differs_from_real_photos": differs,
            "cache_has_only_training_photos": only_train, "pass": shapes and worst < 1e-4 and differs and only_train}


def check_rules(pool, choice):
    import joblib
    manifest = read_json(ROOT / "data/manifests/processed.json")
    ids_by_budget = {int(b): v["ids"] for b, v in manifest["budgets"].items()}
    counts = budget_counts(ids_by_budget)
    out = {"pixel_counts": {b: c.tolist() for b, c in counts.items()}, "runs": {}}
    ok = True
    totals = {}
    for kind, method, budget, seed in plan():
        directory = run_dir(kind, method, budget, seed)
        run = read_json(directory / "run.json")
        key = f"{kind}/{method}_{budget}_s{seed}"
        if method == "hybrid":
            base = ROOT / run["base_run"]["directory"]
            crf_ok = run["crf"] == read_json(ROOT / "runs/hybrid_selection.json")["selected"]
            base_ok = base == run_dir(kind, "unet", budget, seed)
            out["runs"][key] = {"same_crf": crf_ok, "refines_matching_unet": base_ok}
            ok &= crf_ok and base_ok
            continue
        if method == "rf":
            model = joblib.load(directory / run["checkpoint"])
            have = np.array(run["region_class_counts"])
            cache = ROOT / "data/features" / run["identity"]["feature_identity"][:16]
            if (cache, budget) not in totals:
                totals[cache, budget] = regions(cache, ids_by_budget[budget])
            every = totals[cache, budget]
            if kind == "uniform":
                weights = model.class_weight
                rule = (have.tolist() == every.tolist() and isinstance(weights, dict)
                        and np.allclose([weights[c] for c in range(3)], my_weights(counts[budget]), atol=1e-6))
            else:
                ratio = int(kind[1:]) if kind.startswith("r") else choice["rf"][str(budget)]
                targets = every[1:]
                want_other = min(every[0], round(ratio * targets.sum()))
                rule = model.class_weight is None and have[1:].tolist() == targets.tolist() and have[0] == want_other
            out["runs"][key] = {"region_class_counts": have.tolist(), "rule_holds": bool(rule)}
        else:
            ratio = int(kind[1:]) if kind.startswith("r") else choice["unet"][str(budget)]
            c = counts[budget]
            want = min(1.0, ratio * (c[1] + c[2]) / c[0])
            rule = abs(run["other_keep_probability"] - want) < 1e-12 and np.allclose(run["class_weights"], 1)
            out["runs"][key] = {"other_keep_probability": run["other_keep_probability"], "expected": want, "rule_holds": bool(rule)}
        ok &= bool(rule)
    sample = [i for i in ids_by_budget[100]][::120][:12]
    out["blackened_features"] = blacken_features(pool, read_json(run_dir("blacken", "rf", 25, SEEDS[0]) / "run.json"), sample)
    out["pass"] = bool(ok and out["blackened_features"]["pass"])
    return out


# ---------- U5: re-scoring ----------

def rescore(choice):
    ids, _, masks, _ = load_derived("test")
    ids = [str(i) for i in ids]
    out = {}
    for kind, method, budget, seed in tested(choice):
        directory = run_dir(kind, method, budget, seed)
        theirs, result = stored_predictions(directory, ids)
        cms = per_image_confusions(theirs, masks)
        stored = {item["id"]: item["metrics"]["confusion_matrix"] for item in result["per_image"]}
        out[f"{kind}/{method}_{budget}_s{seed}"] = {
            "confusion_equal": cms.sum(0).tolist() == result["metrics"]["confusion_matrix"],
            "per_image_equal": all(c.tolist() == stored[i] for c, i in zip(cms, ids)),
            "flooded_miou_equal": abs(scores(cms.sum(0))["flooded_miou"] - result["metrics"]["flooded_miou"]) < 1e-12}
    return {"runs": len(out), "detail": out, "pass": all(all(v.values()) for v in out.values())}


# ---------- U7: bootstrap ----------

def flooded(m):
    tp = np.diagonal(m, axis1=-2, axis2=-1).astype(np.float64)
    union = m.sum(-1) + m.sum(-2) - tp
    with np.errstate(invalid="ignore", divide="ignore"):
        iou = tp / union
    return np.nanmean(iou[..., 1:], -1)


def arm(directory_of, order):
    """Per-image matrices of the 3 seeds in the stored image order; every run must list the images in that same order,
    so resample k draws the same images in both arms."""
    rows = []
    for seed in SEEDS:
        result = read_json(directory_of(seed) / "evaluation_test.json")
        if order and [i["id"] for i in result["per_image"]] != order:
            raise AssertionError(f"image order differs: {directory_of(seed)}")
        order[:] = [i["id"] for i in result["per_image"]]
        rows.append(np.array([i["metrics"]["confusion_matrix"] for i in result["per_image"]], dtype=np.int64))
    return np.stack(rows)


def bootstrap_check(choice):
    results = read_json(ROOT / "output/undersample/results.json")
    stats = UCFG["statistics"]
    first = lambda m, b, s: ROOT / f"runs/{m}_{b}_s{s}"  # noqa: E731
    expected = [("uniform", "rf", b, first) for b in BUDGETS]
    expected += [(f"r{choice[FAMILY[m]][str(b)]}", m, b, baseline_dir) for b in BUDGETS for m in UCFG["undersample"]["methods"]]
    expected += [("blacken", m, b, baseline_dir) for b in BUDGETS for m in UCFG["blacken"]["methods"]]
    stored = {(c["kind"], c["method"], c["budget"]): c for c in results["uniform_vs_first"] + results["comparisons"] + results["blacken"]}
    mismatches = []
    for kind, method, budget, baseline in expected:
        c = stored.get((kind, method, budget))
        if c is None:
            mismatches.append({"comparison": f"{kind}/{method}/{budget}", "missing": True})
            continue
        order = []
        v = arm(lambda s: run_dir(kind, method, budget, s), order)
        b = arm(lambda s: baseline(method, budget, s), order)
        diffs = flooded(v.sum(1)) - flooded(b.sum(1))
        rng = np.random.default_rng(stats["bootstrap_seed"])
        n = v.shape[1]
        boot = np.empty(stats["bootstrap"])
        for k in range(stats["bootstrap"]):
            weight = np.bincount(rng.integers(0, n, n), minlength=n)
            boot[k] = (flooded(np.tensordot(weight, v, axes=([0], [1]))) - flooded(np.tensordot(weight, b, axes=([0], [1])))).mean()
        a = (1 - stats["ci"]) / 2
        low, high = np.quantile(boot, [a, 1 - a])
        decision = "helps" if low > 0 and (diffs > 0).all() else "hurts" if high < 0 and (diffs < 0).all() else "no clear effect"
        same = (np.allclose(diffs, c["differences"], atol=1e-12) and abs(low - c["ci"][0]) < 1e-9 and abs(high - c["ci"][1]) < 1e-9
                and decision == c["decision"])
        if not same:
            mismatches.append({"comparison": f"{kind}/{method}/{budget}", "mine": [float(low), float(high), decision], "theirs": [*c["ci"], c["decision"]]})
    return {"comparisons": len(expected), "stored": len(stored), "mismatches": mismatches, "pass": not mismatches and len(stored) == len(expected)}


def main():
    run = sum(1 for _ in (ROOT / "output/audit").glob("undersample_[0-9].json")) + 1
    t = time.perf_counter()
    report = {"run": run}
    report["U6"] = check_ratio()
    choice = report["U6"]["mine"]
    report.update(check_chronology(choice))
    report["U2"] = check_frozen()
    with ProcessPoolExecutor(WORKERS) as pool:
        report["U4"] = check_rules(pool, choice)
    report["U5"] = rescore(choice)
    report["U7"] = bootstrap_check(choice)
    report["seconds"] = time.perf_counter() - t
    guard()
    path = write_json(f"undersample_{run}.json", report)
    summary = {k: v["pass"] for k, v in sorted(report.items()) if k.startswith("U")}
    print(json.dumps({"written": str(path.relative_to(ROOT)), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
