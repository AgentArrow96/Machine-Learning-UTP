"""Verify saved predictions, seed summaries, hybrid provenance, web assets/CSVs, figures and acceptance reports."""
import csv
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from PIL import Image
from floodnet.common import config, inside, now, read_json, sha256, write_json
from floodnet.data import Store, crop
from floodnet.evaluation import checkpoint_file, final_runs
from floodnet.export import STRATA_FIELDS, crf_effects, flat_metrics, stats, target_strata
from floodnet.hybrid import hybrid_runs
from floodnet.metrics import confusion, score


def main():
    cfg = config()
    data = read_json("web/data/results.json")
    frozen = read_json("runs/final_experiment.json")
    hybrid_frozen = read_json("runs/hybrid_experiment.json")
    selection = read_json("runs/hybrid_selection.json")
    overlap = read_json("data/manifests/overlap.json")
    expected = final_runs(cfg) + hybrid_runs(cfg)
    assert data["status"] == "complete" and len(data["runs"]) == len(expected) == data["protocol"]["expected_runs"]
    assert data["experiment_identity"] == frozen["identity"] and frozen["overlap_identity"] == overlap["identity"]
    assert data["hybrid_experiment_identity"] == hybrid_frozen["identity"]
    assert hybrid_frozen["base_experiment_identity"] == frozen["identity"] and hybrid_frozen["selection_identity"] == selection["identity"]
    store = Store(cfg)
    samples = {s["id"]: s for s in data["samples"]}
    test = store.select("test")
    assert set(samples) == {r["id"] for r in test} and len(test) == 448
    clean_ids = set(overlap["summary"]["test"]["clean_ids"])
    strata = target_strata(store, overlap)
    assert len(strata["low"]) + len(strata["high"]) == data["dataset"]["overlap"]["target_images"]["images"]
    test_index = {record["id"]: i for i, record in enumerate(test)}
    checks, prediction_assets = [], 0
    for exported in data["runs"]:
        run_dir = inside(f"runs/{exported['id']}")
        run = read_json(run_dir / "run.json")
        result = read_json(run_dir / "evaluation_test.json")
        assert run["identity"]["seed"] == exported["seed"] and run["status"] == "complete"
        assert sha256(checkpoint_file(run_dir, run)) == exported["checkpoint_sha256"]
        if exported["method"] == "hybrid":
            # The hybrid trains nothing: it must refine the frozen U-Net checkpoint with the same budget and seed.
            base = read_json(inside(f"runs/unet_{exported['budget']}_s{exported['seed']}") / "run.json")
            assert run["base_run"]["run_id"] == base["run_id"] and run["checkpoint_sha256"] == base["checkpoint_sha256"]
            assert run["crf"] == selection["selected"] and result["experiment_identity"] == hybrid_frozen["identity"]
            round_ = next(r for r in selection["rounds"] if selection["selected"] in r["settings"])
            tuned = round_["per_run"][base["run_id"]]["flooded_miou"][round_["settings"].index(selection["selected"])]
            validation = read_json(run_dir / "evaluation_val.json")["metrics"]["flooded_miou"]
            assert abs(validation - tuned) < 1e-12, (exported["id"], validation, tuned)
        assert exported["metrics"] == result["metrics"]
        predictions = np.load(inside(result["predictions"]), mmap_mode="r")
        pooled = np.zeros((3, 3), dtype=np.int64)
        clean = np.zeros((3, 3), dtype=np.int64)
        explorer = exported["id"] in data["explorer_runs"]
        for i, record in enumerate(test):
            _, truth = store.get(record)
            matrix = confusion(truth, predictions[i])
            pooled += matrix
            if record["id"] in clean_ids:
                clean += matrix
            if explorer:
                sample = samples[record["id"]]
                with Image.open(inside("web") / sample["predictions"][run["run_id"]]) as image:
                    np.testing.assert_array_equal(np.asarray(image), crop(predictions[i], record["geometry"]))
                prediction_assets += 1
        assert score(pooled) == result["metrics"]
        assert score(clean) == exported["clean_metrics"] and exported["clean_images"] == len(clean_ids)
        for stratum, ids in (("low", strata["low"]), ("high", strata["high"])):
            matrix = np.sum([confusion(store.get(store.by_id[i])[1], predictions[test_index[i]]) for i in ids], axis=0)
            assert {k: flat_metrics(score(matrix))[k] for k in STRATA_FIELDS} == exported["overlap_strata"][stratum]
        if exported["method"] in ("unet", "hybrid"):
            assert "RTX" in exported["device"] and "bfloat16" in exported["precision"]
        checks.append(f"{run['run_id']}: checkpoint hash, pooled, clean-subset and overlap-strata scores recomputed from all pixel predictions"
                      + ("; refines the frozen U-Net and reproduces its validation tuning score" if exported["method"] == "hybrid" else ""))
    for item in data["summary"]:
        group = sorted((r for r in data["runs"] if r["method"] == item["method"] and r["budget"] == item["budget"]), key=lambda r: r["seed"])
        assert [r["seed"] for r in group] == sorted(cfg["train_seeds"])
        for scope, key in (("full", "metrics"), ("clean", "clean_metrics")):
            for field, stat in item[scope].items():
                assert stat == stats([flat_metrics(r[key])[field] for r in group]), (item["method"], item["budget"], scope, field)
        for stratum in ("low", "high"):
            for field in STRATA_FIELDS:
                assert item[f"overlap_{stratum}"][field] == stats([r["overlap_strata"][stratum][field] for r in group])
    checks.append(f"{len(data['summary'])} seed summaries (mean/SD/range, full and clean) match the individual runs")
    assert data["crf_effects"] == crf_effects(data["runs"], cfg["train_seeds"]) and len(data["crf_effects"]) == 2
    assert data["hybrid"]["selected"] == selection["selected"] and data["hybrid"]["adopted"] == selection["adopted"]
    checks.append("Seed-matched CRF effects and the validation selection match the runs and the frozen selection record")
    rf_counts = {r["budget"]: r["region_class_counts"] for r in data["runs"] if r["method"] == "rf"}
    regions = {}
    features = Path(inside(f"data/features"))
    # The baseline RF's own cache (the improvement study adds context / 512-px caches next to it).
    feature_dir = features / read_json("runs/rf_25_s42/run.json")["identity"]["feature_identity"][:16]
    for budget in (25, 100):
        ids = set(store.manifest["budgets"][str(budget)]["ids"])
        totals = np.zeros(3, dtype=np.int64)
        for record_id in ids:
            with np.load(feature_dir / f"{record_id}.npz") as cached:
                totals += np.bincount(cached["y"], minlength=3)
        regions[budget] = totals.tolist()
        assert rf_counts[budget][1:] == regions[budget][1:], (budget, rf_counts[budget], regions[budget])
        assert sum(rf_counts[budget]) == min(sum(regions[budget]), cfg["rf"]["max_regions"])
    checks.append(f"RF fitted tables keep every flooded region: 25% {rf_counts[25]}, 100% {rf_counts[100]}")
    with (inside("web") / data["downloads"]["aggregate"]).open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == len(expected)
    for row in rows:
        run = next(r for r in data["runs"] if r["id"] == row["run_id"])
        assert float(row["flooded_miou"]) == run["metrics"]["flooded_miou"]
        assert float(row["clean_flooded_miou"]) == run["clean_metrics"]["flooded_miou"]
    with (inside("web") / data["downloads"]["per_image"]).open(encoding="utf-8", newline="") as stream:
        assert sum(1 for _ in csv.DictReader(stream)) == 448 * len(expected)
    checks.append(f"Aggregate CSV ({len(expected)} runs) and per-image CSV ({448 * len(expected)} rows) agree with the export")
    exports = [p.name for p in (inside("web") / "data/exports").iterdir()]
    assert exports == [data["downloads"]["summary"].split("/")[2]], exports
    checks.append("Exactly one export directory remains (stale exports pruned)")
    walkthrough_path = inside("web/data/walkthrough.json")
    if walkthrough_path.exists():
        walk = read_json(walkthrough_path)
        assert walk["experiment_identity"] == frozen["identity"] and walk["hybrid_experiment_identity"] == hybrid_frozen["identity"]
        # A walkthrough that replays the uniform Before forests is checked against the uniform bundle's masks.
        samples = {s["id"]: s["predictions"] for s in data["samples"]}
        if walk.get("rf_variant"):
            samples = {k: v["predictions"] for k, v in read_json("web/data/uniform.json")["samples"].items()}
        compared = 0
        for example in walk["examples"]:
            for budget, methods in example["budgets"].items():
                for method, part in methods.items():
                    exported = samples[example["id"]][part["run_id"]]
                    with Image.open(inside("web") / part["mask"]) as a, Image.open(inside("web") / exported) as b:
                        assert np.array_equal(np.asarray(a), np.asarray(b)), (example["id"], budget, method)
                    compared += 1
        checks.append(f"{compared} walkthrough final masks equal the exported explorer predictions of the same runs"
                      + (f" (forests: {walk['rf_variant']})" if walk.get("rf_variant") else ""))
    preprocessing_path = inside("web/data/preprocessing.json")
    if preprocessing_path.exists():
        steps = read_json(preprocessing_path)["steps"]
        audit = read_json("data/manifests/audit.json")
        assert steps["acquire"]["counts"] == audit["counts"] and steps["acquire"]["files_hashed"] == 2 * len(audit["records"])
        assert steps["duplicates"]["exact_groups"] == len(audit["exact_duplicates"]) and steps["duplicates"]["removed"] == 0
        assert steps["orientation"]["rotated"] == sum(r["exif_orientation"] != 1 for r in audit["records"])
        assert steps["budgets"]["counts"] == {k: len(v["ids"]) for k, v in store.manifest["budgets"].items()}
        urls = [u for u in json.dumps(steps).split('"') if u.startswith("data/preprocessing/")]
        assert urls and all((inside("web") / u).is_file() for u in urls)
        checks.append(f"Preprocessing page: counts, duplicates, orientation and budgets equal the manifests; {len(urls)} assets present")
    imbalance_path = inside("web/data/imbalance.json")
    if imbalance_path.exists():
        study = read_json(imbalance_path)
        frozen_study = read_json("runs/imbalance/experiment.json")
        assert study["experiment_identity"] == frozen_study["identity"]
        assert study["comparisons"] == read_json("output/imbalance/results.json")["comparisons"]
        for comparison in study["comparisons"]:
            for seed, value in zip(comparison["seeds"], comparison["variant_scores"]):
                stored = read_json(f"runs/imbalance/{comparison['variant']}/{comparison['method']}_{comparison['budget']}_s{seed}/evaluation_test.json")
                assert abs(stored["metrics"]["flooded_miou"] - value) < 1e-12, (comparison["variant"], comparison["method"], seed)
        urls = [u for u in json.dumps(study).split('"') if u.startswith("data/imbalance/")]
        assert urls and all((inside("web") / u).is_file() for u in urls)
        checks.append(f"Class-imbalance page: {len(study['comparisons'])} comparisons equal the study results and every variant score equals its frozen test evaluation; {len(urls)} assets present")
    improve_path = inside("web/data/improve.json")
    if improve_path.exists():
        study = read_json(improve_path)
        if study["status"] == "complete":
            assert study["experiment_identity"] == read_json("runs/improve/experiment.json")["identity"]
            assert study["comparisons"] == read_json("output/improve/results.json")["comparisons"]
            for comparison in study["comparisons"]:
                for seed, value in zip(comparison["seeds"], comparison["variant_scores"]):
                    stored = read_json(f"runs/improve/{comparison['variant']}/{comparison['method']}_{comparison['budget']}_s{seed}/evaluation_test.json")
                    assert abs(stored["metrics"]["flooded_miou"] - value) < 1e-12, (comparison["variant"], comparison["method"], seed)
        for key, entry in study["baseline_test"].items():
            method, budget = key.split("_")
            for seed, value in zip(study["seeds"], entry["flooded_miou"]):
                assert abs(read_json(f"runs/{method}_{budget}_s{seed}/evaluation_test.json")["metrics"]["flooded_miou"] - value) < 1e-12, key
        checks.append(f"Improvements page ({study['status']}): {len(study['comparisons'])} comparisons equal the study results; baseline scores equal the frozen test evaluations")
    after_path = inside("web/data/after.json")
    if after_path.exists():
        after = read_json(after_path)
        assert after["experiment_identity"] == read_json("runs/improve/experiment.json")["identity"]
        assert after["comparisons"] == [c for c in read_json("output/improve/results.json")["comparisons"] if c["variant"] == "combined"]
        for run in after["runs"]:
            result = read_json(f"runs/{run['id']}/evaluation_test.json")
            assert run["metrics"] == result["metrics"], run["id"]
        after_pngs = 0
        for run_id in after["explorer_runs"]:
            result = read_json(f"runs/{run_id}/evaluation_test.json")
            predictions = np.load(inside(result["predictions"]), mmap_mode="r")
            for item in result["per_image"]:
                record = store.by_id[item["id"]]
                with Image.open(inside("web") / after["samples"][item["id"]]["predictions"][run_id]) as image:
                    np.testing.assert_array_equal(np.asarray(image), crop(np.asarray(predictions[item["prediction_index"]]), record["geometry"]))
                after_pngs += 1
        assert [p.name for p in inside("web/data/after").iterdir()] == [after["identity"]]
        checks.append(f"After view: {len(after['runs'])} combined runs equal their frozen test evaluations; {after_pngs} explorer masks equal the stored predictions; Compare deltas equal the study report")
    study_results = inside("output/undersample/results.json")
    for name in ("uniform", "undersample"):
        path = inside(f"web/data/{name}.json")
        if not path.exists():
            continue
        bundle = read_json(path)
        results = read_json(study_results)
        assert bundle["experiment_identity"] == read_json("runs/undersample/experiment.json")["identity"] == results["experiment"]
        assert bundle["comparisons"] == (results["uniform_vs_first"] if name == "uniform" else results["comparisons"]), name
        groups = [bundle["runs"]] + ([bundle["blacken"]["runs"]] if name == "undersample" else [])
        for runs in groups:
            for run in runs:
                assert run["metrics"] == read_json(f"runs/{run['id']}/evaluation_test.json")["metrics"], run["id"]
        pngs = 0
        for run_id in bundle["explorer_runs"]:
            result = read_json(f"runs/{run_id}/evaluation_test.json")
            predictions = np.load(inside(result["predictions"]), mmap_mode="r")
            for item in result["per_image"]:
                record = store.by_id[item["id"]]
                with Image.open(inside("web") / bundle["samples"][item["id"]]["predictions"][run_id]) as image:
                    np.testing.assert_array_equal(np.asarray(image), crop(np.asarray(predictions[item["prediction_index"]]), record["geometry"]))
                pngs += 1
        assert [p.name for p in inside(f"web/data/{name}").iterdir()] == [bundle["identity"]]
        extra = ""
        if name == "undersample":
            assert bundle["blacken"]["comparisons"] == results["blacken"] and bundle["ratio"] == results["ratio"]
            example = [v for k, v in bundle["blacken"]["example"].items() if not k.endswith("_id")]
            assert all((inside("web") / u).is_file() for u in example)
            extra = f"; {len(bundle['blacken']['runs'])} blackening runs and {len(example)} example images present"
        checks.append(f"{name}.json: {sum(map(len, groups))} runs equal their frozen test evaluations; {pngs} explorer masks equal the stored predictions; comparisons equal the study report{extra}")
    figures = read_json("output/figures/manifest.json")
    assert figures["experiment_identity"] == frozen["identity"] and figures["source_export_sha256"] == sha256("web/data/results.json")
    for item in figures["files"]:
        assert sha256(inside("output/figures") / item["file"]) == item["sha256"]
    checks.append(f"{len(figures['files'])} standalone scientific figure files match their source-export/provenance hashes")
    log = inside("runs/pipeline.log").read_text(encoding="utf-8")
    warnings = log.count("sklearn.utils.parallel.delayed")
    assert warnings == 0, f"{warnings} sklearn parallel warnings in the pipeline log"
    checks.append("Pipeline log has no sklearn parallel-configuration warnings")
    pytest_root = ET.parse(inside("output/pytest.xml")).getroot()
    suites = list(pytest_root.iter("testsuite"))
    test_count = sum(int(s.attrib.get("tests", 0)) for s in suites)
    assert sum(int(s.attrib.get("failures", 0)) + int(s.attrib.get("errors", 0)) for s in suites) == 0
    assert sum(int(s.attrib.get("skipped", 0)) for s in suites) == 0
    browser_log = inside("output/playwright/checks.log").read_text(encoding="utf-8")
    browser = json.loads(next(line for line in browser_log.splitlines() if line.startswith('{"status":')))
    assert browser["status"] == "passed" and browser["run_count"] == len(expected) and browser["sample_count"] == 448
    report = {"status": "passed", "verified_at": now(), "experiment_identity": frozen["identity"],
              "hybrid_experiment_identity": hybrid_frozen["identity"], "test_images": 448,
              "runs": len(expected), "clean_test_images": len(clean_ids), "prediction_pngs_verified": prediction_assets,
              "pytest_tests": test_count, "browser_checks": len(browser["checks"]), "browser_report": browser, "checks": checks}
    write_json("output/verification.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
