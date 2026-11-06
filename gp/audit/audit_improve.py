"""Independent checks of the improvement study (evals M1-M5, M7, M8). Never imports `floodnet`.

Run twice for pass^2:  .venv\\Scripts\\python.exe audit\\audit_improve.py  (writes output/audit/improve_<n>.json)
Reproduction checks that need project code (M6) live in audit_improve_repro.py.
Written from configs/improve.json and docs/improve.md: 512-px inputs are derived here from the raw release, every
prediction is rebuilt from the stored checkpoints / cached RF features and scored with scikit-learn on the 256-px grid.
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
    WORKERS,
    hybrid_predict,
    load_images,
    load_masks,
    per_image_confusions,
    rf_expand,
    stored_predictions,
)
from independent import (
    IGNORE,
    MEAN,
    PAD_RGB,
    ROOT,
    STD,
    artifact_hashes,
    discover,
    guard,
    load_raw,
    read_json,
    remap,
    scores,
    sha256,
    write_json,
)

ICFG = read_json(ROOT / "configs/improve.json")
SEEDS, BUDGETS = ICFG["train_seeds"], ICFG["budgets"]
FAMILY = {"rf": "rf", "unet": "unet", "hybrid": "unet"}


def when(text):
    return datetime.fromisoformat(text)


def combine_components():
    return read_json(ROOT / "runs/improve/combine.json")["components"]


def plan():
    """The protocol's runs; a combined model whose family gained no component is not trained (docs/improve.md)."""
    chosen = combine_components()
    return [(v, m, b, s) for v, spec in ICFG["variants"].items() for m in spec["methods"]
            if not (v == "combined" and not chosen[FAMILY[m]]) for b in BUDGETS for s in SEEDS]


def run_dir(variant, method, budget, seed):
    return ROOT / f"runs/improve/{variant}/{method}_{budget}_s{seed}"


def started(run):
    return when(run.get("started_at") or run["created_at"])


# ---------- M1 / M3: chronology ----------

def check_chronology():
    prereg = read_json(ROOT / "runs/improve/preregistration.json")
    runs = {f"{v}/{m}_{b}_s{s}": read_json(run_dir(v, m, b, s) / "run.json") for v, m, b, s in plan()}
    first = min(started(r) for r in runs.values())
    m1 = {"registered_at": prereg["registered_at"], "first_run_started": first.isoformat(),
          "config_hash_unchanged": prereg["config_sha256"] == sha256(ROOT / "configs/improve.json"),
          "evals_hash_unchanged": prereg["evals_sha256"] == sha256(ROOT / "audit/improve_evals.md"),
          "runs_planned_at_registration": prereg["runs_planned"]}
    m1["pass"] = when(prereg["registered_at"]) < first and m1["config_hash_unchanged"] and m1["evals_hash_unchanged"]
    frozen = read_json(ROOT / "runs/improve/experiment.json")
    frozen_at = when(frozen["frozen_at"])
    validations = {k: when(read_json(ROOT / "runs/improve" / k / "evaluation_val.json")["created_at"]) for k in runs}
    tests = {k: read_json(ROOT / "runs/improve" / k / "evaluation_test.json") for k in runs}
    combine = when(read_json(ROOT / "runs/improve/combine.json")["decided_at"])
    combined_started = [started(r) for k, r in runs.items() if k.startswith("combined/")]
    m3 = {"runs": len(frozen["runs"]), "planned": len(runs), "frozen_at": frozen_at.isoformat(),
          "last_validation": max(validations.values()).isoformat(), "first_test": min(when(t["created_at"]) for t in tests.values()).isoformat(),
          "tests_bound_to_freeze": all(t["experiment_identity"] == frozen["identity"] for t in tests.values()),
          "frozen_set_equals_plan": sorted(r["run_id"].removeprefix("runs/improve/") for r in frozen["runs"]) == sorted(runs),
          "combine_decided_before_combined_runs": all(combine < t for t in combined_started)}
    m3["pass"] = (m3["frozen_set_equals_plan"] and m3["tests_bound_to_freeze"] and m3["combine_decided_before_combined_runs"]
                  and max(validations.values()) < frozen_at < min(when(t["created_at"]) for t in tests.values()))
    return {"M1": m1, "M3": m3}


# ---------- M2: frozen artifacts ----------

def check_frozen():
    before = read_json(ROOT / "output/audit/g1_after.json")
    now = artifact_hashes()
    changed = [k for k, v in before.items() if now.get(k) != v]
    imbalance = read_json(ROOT / "runs/imbalance/experiment.json")
    exported = read_json(ROOT / "web/data/imbalance.json")["experiment_identity"]
    return {"original_files": len(before), "changed_or_missing": changed, "imbalance_freeze_matches_export": imbalance["identity"] == exported,
            "pass": not changed and imbalance["identity"] == exported}


# ---------- independent inputs at 512 px ----------

def fit_and_pad(rgb_image, size):
    """Protocol at another size: keep aspect ratio, fit inside `size` (Lanczos), centre-pad with the ImageNet mean."""
    from PIL import Image
    width, height = rgb_image.size
    scale = min(size / width, size / height)
    new_w, new_h = max(1, round(width * scale)), max(1, round(height * scale))
    left, top = (size - new_w) // 2, (size - new_h) // 2
    rgb = np.empty((size, size, 3), dtype=np.uint8)
    rgb[:] = PAD_RGB
    rgb[top:top + new_h, left:left + new_w] = np.asarray(rgb_image.resize((new_w, new_h), Image.Resampling.LANCZOS))
    return rgb, {"left": left, "top": top, "width": new_w, "height": new_h}


def derive512(job):
    image_path, mask_path = job
    rgb_image, _ = load_raw(image_path, mask_path)
    return fit_and_pad(rgb_image, 512)


def load512(ids, pool):
    pairs = discover()["test"]
    keys = [i.removeprefix("test_") for i in ids]   # derived ids are "<split>_<file stem>"
    out = list(pool.map(derive512, [(pairs["images"][k], pairs["masks"][k]) for k in keys], chunksize=4))
    return np.stack([o[0] for o in out]), {i: o[1] for i, o in zip(ids, out)}


def box_resample(content, out_h, out_w):
    """Area (box-filter) resampling written from the definition: output pixel o covers input [o*s, (o+1)*s) and takes
    the overlap-weighted mean of the input pixels there (a 2x2 block mean when s = 2)."""
    def weights(n_in, n_out):
        s = n_in / n_out
        w = np.zeros((n_out, n_in))
        for o in range(n_out):
            a, b = o * s, (o + 1) * s
            for i in range(int(np.floor(a)), min(n_in, int(np.ceil(b)))):
                w[o, i] = (min(b, i + 1) - max(a, i)) / s
        return w
    wr, wc = weights(content.shape[1], out_h), weights(content.shape[2], out_w)
    return np.stack([wr @ c.astype(np.float64) @ wc.T for c in content])


def to256(probs, g_model, g256):
    """Area-averaged unpadded content placed at the 256-px geometry (class 0 in padding)."""
    classes = probs.shape[0]
    out = np.zeros((classes, 256, 256), dtype=np.float32)
    out[0] = 1
    content = probs[:, g_model["top"]:g_model["top"] + g_model["height"], g_model["left"]:g_model["left"] + g_model["width"]]
    out[:, g256["top"]:g256["top"] + g256["height"], g256["left"]:g256["left"] + g256["width"]] = box_resample(content, g256["height"], g256["width"])
    return out


def one_hot(labels):
    return (np.arange(3)[:, None, None] == labels[None]).astype(np.float32)


# ---------- independent U-Net inference ----------

def load_weights(directory):
    """Follow inference-only records (TTA, hybrid) to the trained checkpoint."""
    import segmentation_models_pytorch as smp
    import torch
    run = read_json(directory / "run.json")
    while "base_run" in run:
        directory = ROOT / run["base_run"]["directory"]
        run = read_json(directory / "run.json")
    u = run["config"]["unet"]
    model = smp.Unet(encoder_name=u["encoder"], encoder_weights=None, decoder_channels=tuple(u["decoder_channels"]), in_channels=3,
                     classes=13 if u.get("aux10_weight") else 3, activation=None)
    model.load_state_dict(torch.load(directory / run["checkpoint"], map_location="cpu", weights_only=False)["model"])
    return model.cuda().eval(), directory


def unet_probs(model, rgb_all, tta):
    """3-class softmax (first 3 logits) of each image, batch one, bf16; TTA = mean over the 8 dihedral views mapped back."""
    import torch
    out = []
    for rgb in rgb_all:
        x = torch.from_numpy(np.ascontiguousarray(((rgb.astype(np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1)))[None].cuda()

        def soft(t):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return torch.softmax(model(t)[:, :3].float(), 1)
        with torch.inference_mode():
            if not tta:
                out.append(soft(x)[0].cpu().numpy())
                continue
            total = None
            for transpose in (False, True):
                base = x.transpose(2, 3) if transpose else x
                for vertical in (False, True):
                    for horizontal in (False, True):
                        t = base.flip(2) if vertical else base
                        t = t.flip(3) if horizontal else t
                        p = soft(t)
                        p = p.flip(3) if horizontal else p
                        p = p.flip(2) if vertical else p
                        p = p.transpose(2, 3) if transpose else p
                        total = p if total is None else total + p
            out.append((total / 8)[0].cpu().numpy())
    return out


def expected_crf(size):
    selected = read_json(ROOT / "runs/hybrid_selection.json")["selected"]
    params = json.loads(json.dumps(selected))
    params["gaussian"]["sxy"] = selected["gaussian"]["sxy"] * size / 256
    params["bilateral"]["sxy"] = selected["bilateral"]["sxy"] * size / 256
    return params


# ---------- M4 / M5: re-prediction and re-scoring ----------

def context_check(cache, base_cache, ids):
    """rf_context rows: first 22 columns = the frozen baseline features; neighbour means (4-connected), centroid and
    area share recomputed here from the cached region maps."""
    worst = 0.0
    for i in ids:
        mine, base = np.load(cache / f"{i}.npz"), np.load(base_cache / f"{i}.npz")
        x, regions = mine["x"], mine["regions"]
        if not np.array_equal(x[:, :22], base["x"]) or not np.array_equal(regions, base["regions"]):
            return {"pass": False, "image": i, "reason": "base columns or regions differ"}
        rows, cols = np.nonzero(regions >= 0)
        top, left = rows.min(), cols.min()
        r = regions[top:rows.max() + 1, left:cols.max() + 1]
        n = len(x)
        adjacency = np.zeros((n, n), dtype=bool)
        for a, b in ((r[:, :-1], r[:, 1:]), (r[:-1, :], r[1:, :])):
            edge = (a != b) & (a >= 0) & (b >= 0)
            adjacency[a[edge], b[edge]] = adjacency[b[edge], a[edge]] = True
        counts = adjacency.sum(1)
        neighbour = np.where(counts[:, None] > 0, adjacency.astype(np.float64) @ x[:, :22] / np.maximum(counts, 1)[:, None], x[:, :22])
        ys, xs = np.nonzero(r >= 0)
        ids_ = r[r >= 0]
        area = np.bincount(ids_, minlength=n)
        cy = np.bincount(ids_, weights=ys, minlength=n) / area / (r.shape[0] - 1)
        cx = np.bincount(ids_, weights=xs, minlength=n) / area / (r.shape[1] - 1)
        mine_cols = np.concatenate([neighbour, cy[:, None], cx[:, None], (area / area.sum())[:, None]], 1)
        worst = max(worst, float(np.abs(mine_cols - x[:, 22:47]).max()))
    return {"images": len(ids), "max_abs_difference": worst, "pass": worst < 1e-5}


def rescore():
    import joblib
    ids, rgb256, geometry256 = load_images("test")
    masks = load_masks("test")
    runs = plan()
    out, extra = {}, {}
    with ProcessPoolExecutor(WORKERS) as pool:
        rgb512, geometry512 = load512(ids, pool) if any(read_json(run_dir(*r) / "run.json")["config"]["size"] == 512 for r in runs) else (None, None)
        cached_probs = {}
        for variant, method, budget, seed in runs:
            directory = run_dir(variant, method, budget, seed)
            run = read_json(directory / "run.json")
            size = run["config"]["size"]
            rgb_all, geometry = (rgb256, geometry256) if size == 256 else (rgb512, geometry512)
            if method == "rf":
                model = joblib.load(directory / run["checkpoint"])
                model.n_jobs = 1
                cache = ROOT / "data/features" / run["identity"]["feature_identity"][:16]
                labels = []
                for i in ids:
                    f = np.load(cache / f"{i}.npz")
                    label = rf_expand(model, f["x"], f["regions"])
                    labels.append(remap(label) if run["config"]["rf"].get("aux10") else label)
                prediction = np.stack(labels) if size == 256 else np.stack([to256(one_hot(p), geometry[i], geometry256[i]).argmax(0) for p, i in zip(labels, ids)])
                if run["config"]["rf"].get("context"):
                    base_cache = ROOT / "data/features" / read_json(ROOT / f"runs/rf_{budget}_s{seed}/run.json")["identity"]["feature_identity"][:16]
                    extra.setdefault("rf_context_features", context_check(cache, base_cache, ids))
            else:
                tta = bool(run.get("inference", {}).get("tta"))
                model, weights = load_weights(directory)
                key = (str(weights), tta)
                if key not in cached_probs:
                    cached_probs.clear()
                    cached_probs[key] = unet_probs(model, rgb_all, tta)
                del model
                probs = cached_probs[key]
                if method == "unet":
                    labels = [p.argmax(0).astype(np.uint8) for p in probs]
                else:
                    crf_ok = run["crf"] == expected_crf(size)
                    extra.setdefault("crf_settings", {})[f"{variant}/{method}_{budget}_s{seed}"] = crf_ok
                    labels = list(hybrid_predict(probs, rgb_all, geometry, ids, run["crf"], pool))
                prediction = np.stack(labels) if size == 256 else np.stack(
                    [(to256(one_hot(p) if method == "hybrid" else q, geometry[i], geometry256[i])).argmax(0) for p, q, i in zip(labels, probs, ids)])
            prediction = prediction.astype(np.uint8)
            theirs, result = stored_predictions(directory, ids)
            mine = prediction.copy()
            mine[masks == IGNORE] = 0
            cms = per_image_confusions(mine, masks)
            out[f"{variant}/{method}_{budget}_s{seed}"] = {
                "differing_pixels": int(np.count_nonzero(mine != theirs)),
                "confusion_equal": cms.sum(0).tolist() == result["metrics"]["confusion_matrix"],
                "per_image_equal": all(c.tolist() == p["metrics"]["confusion_matrix"] for c, p in zip(cms, sorted(result["per_image"], key=lambda e: ids.index(e["id"])))),
                "flooded_miou": scores(cms.sum(0))["flooded_miou"]}
    crf_pass = all(extra.get("crf_settings", {}).values())
    context_pass = extra.get("rf_context_features", {"pass": True})["pass"]
    return {"runs": out, **extra,
            "pass": all(v["differing_pixels"] == 0 and v["confusion_equal"] and v["per_image_equal"] for v in out.values()) and crf_pass and context_pass}


# ---------- M7: combine rule ----------

def check_combine():
    stored = read_json(ROOT / "runs/improve/combine.json")
    mine = {"unet": [], "rf": []}
    for family, components in (("unet", ("aux10", "recipe", "tta", "res512", "encoder")), ("rf", ("aux10", "res512", "rf_context"))):
        for component in components:
            ok = True
            for budget in BUDGETS:
                diffs = [read_json(run_dir(component, family, budget, s) / "evaluation_val.json")["metrics"]["flooded_miou"]
                         - read_json(ROOT / f"runs/{family}_{budget}_s{s}/evaluation_val.json")["metrics"]["flooded_miou"] for s in SEEDS]
                ok &= float(np.mean(diffs)) > 0 and sum(d > 0 for d in diffs) >= 2
            if ok:
                mine[family].append(component)
    return {"mine": mine, "stored": stored["components"], "pass": mine == stored["components"]}


# ---------- M8: bootstrap ----------

def bootstrap_check():
    results = read_json(ROOT / "output/improve/results.json")["comparisons"]
    stats = ICFG["statistics"]
    mismatches = []

    def flooded(m):
        tp = np.diagonal(m, axis1=-2, axis2=-1).astype(np.float64)
        union = m.sum(-1) + m.sum(-2) - tp
        with np.errstate(invalid="ignore", divide="ignore"):
            iou = tp / union
        return np.nanmean(iou[..., 1:], -1)
    expected = {(v, m, b) for v, m, b, _ in plan()}
    found = {(c["variant"], c["method"], c["budget"]) for c in results}
    for c in results:
        arms = []
        for root in (f"runs/improve/{c['variant']}", "runs"):
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
        boot = (flooded(np.einsum("bn,snij->bsij", counts, v)) - flooded(np.einsum("bn,snij->bsij", counts, b))).mean(1)
        a = (1 - stats["ci"]) / 2
        low, high = np.quantile(boot, [a, 1 - a])
        decision = "helps" if low > 0 and (diffs > 0).all() else "hurts" if high < 0 and (diffs < 0).all() else "no clear effect"
        same = np.allclose(diffs, c["differences"], atol=1e-12) and abs(low - c["ci"][0]) < 1e-12 and abs(high - c["ci"][1]) < 1e-12 and decision == c["decision"]
        if not same:
            mismatches.append({"comparison": f"{c['variant']}/{c['method']}/{c['budget']}", "mine": [float(low), float(high), decision], "theirs": [*c["ci"], c["decision"]]})
    return {"comparisons": len(results), "complete": expected == found, "mismatches": mismatches, "pass": not mismatches and expected == found}


def main():
    run = sum(1 for _ in (ROOT / "output/audit").glob("improve_[0-9].json")) + 1
    t = time.perf_counter()
    report = {"run": run, **check_chronology(), "M2": check_frozen()}
    report["M4_M5"] = rescore()
    report["M7"] = check_combine()
    report["M8"] = bootstrap_check()
    report["seconds"] = time.perf_counter() - t
    guard()
    path = write_json(f"improve_{run}.json", report)
    summary = {k: v["pass"] for k, v in report.items() if k.startswith("M")}
    print(json.dumps({"written": str(path.relative_to(ROOT)), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
