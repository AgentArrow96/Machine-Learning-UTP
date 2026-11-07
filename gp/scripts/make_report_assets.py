"""Figures and numbers for the final report (SLIC + RF and U-Net). Reads the frozen exports and runs; changes no run.

Reads web/data/results.json, plus web/data/uniform.json when present (the exported bundle of the report's retrained
forests next to the frozen U-Nets; without it the runs in results.json are used). Writes report/numbers.json,
report/figures/ and the app's Presentation-mode files (web/data/presentation.json, web/data/presentation/)."""
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from floodnet.common import config, environment, inside, read_json, write_json

environment()
os.environ["MPLCONFIGDIR"] = str(inside(".cache/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

METHODS = ("rf", "unet")
BUDGETS = (25, 100)
COLOURS = {"rf": "#059669", "unet": "#4f46e5"}
SHADES = {("rf", 25): "#6ee7b7", ("rf", 100): "#059669", ("unet", 25): "#a5b4fc", ("unet", 100): "#4f46e5"}
NAMES = {"rf": "SLIC + RF", "unet": "U-Net"}
CLASSES = ("Other", "Flooded building", "Flooded road")
PROPOSAL = "pdf/TEB3123 Group 3_Ammar_Ariff_Haikal_Aisar_CJ.pdf"
OUT = inside("report")
FIG = OUT / "figures"


def metrics(matrix):
    m = np.asarray(matrix, dtype=np.float64)
    tp, support, predicted = np.diag(m), m.sum(1), m.sum(0)
    iou = tp / (support + predicted - tp)
    out = {"minority": iou[1:].mean(), "majority": iou[0], "miou3": iou.mean(), "weighted": (support / support.sum() * iou).sum(), "accuracy": tp.sum() / m.sum()}
    for c, name in ((1, "building"), (2, "road")):
        out |= {f"{name}_iou": iou[c], f"{name}_precision": tp[c] / predicted[c], f"{name}_recall": tp[c] / support[c], f"{name}_dice": 2 * tp[c] / (support[c] + predicted[c])}
    return {k: float(v) for k, v in out.items()}


def stat(values):
    a = np.asarray(values, dtype=np.float64)
    return {"mean": float(a.mean()), "sd": float(a.std(ddof=1)), "min": float(a.min()), "max": float(a.max()), "values": a.tolist()}


def collect(uniform, first):
    def runs(bundle, method, budget):
        return sorted((r for r in bundle["runs"] if r["method"] == method and r["budget"] == budget), key=lambda r: r["seed"])

    out = {"configs": {}, "budget_effect": {}, "first_version_rf": {}}
    summary = {(s["method"], s["budget"]): s for s in uniform["summary"]}
    for method in METHODS:
        for budget in BUDGETS:
            rows = runs(uniform, method, budget)
            per = [metrics(r["metrics"]["confusion_matrix"]) for r in rows]
            s = summary[method, budget]
            out["configs"][f"{method}_{budget}"] = {
                "metrics": {k: stat([p[k] for p in per]) for k in per[0]},
                "confusion_pooled_over_seeds": np.sum([r["metrics"]["confusion_matrix"] for r in rows], axis=0).tolist(),
                "training_seconds": stat([r["training_seconds"] for r in rows]),
                "inference_ms": stat([r["timing"]["median_seconds"] * 1000 for r in rows]),
                "peak_gpu_mib": max((r["peak_gpu_allocated_bytes"] or 0) for r in rows) / 2 ** 20,
                "epochs": [r["epochs"] for r in rows], "best_epoch": [r["best_epoch"] for r in rows],
                "val_minority": stat([read_json(inside(f"runs/{r['id']}/evaluation_val.json"))["metrics"]["flooded_miou"] for r in rows]),
                "class_weights": rows[0]["class_weights"], "region_class_counts": rows[0]["region_class_counts"],
                "overlap_low": s["overlap_low"]["flooded_miou"]["mean"], "overlap_high": s["overlap_high"]["flooded_miou"]["mean"]}
        keys = out["configs"][f"{method}_25"]["metrics"]
        out["budget_effect"][method] = {k: stat(np.subtract(out["configs"][f"{method}_100"]["metrics"][k]["values"], keys[k]["values"])) for k in keys}
    for budget in BUDGETS if uniform.get("comparisons") else ():
        per = [metrics(r["metrics"]["confusion_matrix"]) for r in runs(first, "rf", budget)]
        c = next(x for x in uniform["comparisons"] if x["budget"] == budget)
        out["first_version_rf"][str(budget)] = {"metrics": {k: stat([p[k] for p in per]) for k in per[0]}, "region_class_counts": runs(first, "rf", budget)[0]["region_class_counts"],
                                                "change": c["mean_difference"], "ci": c["ci"], "decision": c["decision"]}
    counts = np.array(first["dataset"]["processed_distributions"]["test"]["pixel_counts"], dtype=np.float64)
    share = counts[0] / counts.sum()
    out["all_other_reference"] = {"minority": 0.0, "majority": share, "miou3": share / 3, "weighted": share * share, "accuracy": share}
    train = np.array(first["dataset"]["processed_distributions"]["train"]["pixel_counts"], dtype=np.float64)
    out["train_pixel_share"] = (train / train.sum()).tolist()
    out["test_pixel_share"] = (counts / counts.sum()).tolist()
    out["overlap"] = {"test": {k: v for k, v in first["dataset"]["overlap"]["summary"]["test"].items() if k != "clean_ids"},
                      "val": {k: v for k, v in first["dataset"]["overlap"]["summary"]["val"].items() if k != "clean_ids"},
                      "target_images": first["dataset"]["overlap"]["target_images"]}
    return out


ORIGINAL = ["background", "building flooded", "building non-flooded", "road flooded", "road non-flooded", "water", "tree", "vehicle", "pool", "grass"]


def masks10(record, size):
    """The original 10-class annotation, resized and padded exactly like the 3-class mask."""
    from floodnet.common import ProjectError, sha256
    from floodnet.data import mask_array
    if sha256(record["mask"]) != record["mask_sha256"]:
        raise ProjectError(f"Original annotation changed since audit: {record['id']}")
    g = record["geometry"]
    labels = np.asarray(Image.fromarray(mask_array(inside(record["mask"]))).resize((g["width"], g["height"]), Image.Resampling.NEAREST))
    out = np.full((size, size), 255, dtype=np.uint8)
    out[g["top"]:g["top"] + g["height"], g["left"]:g["left"] + g["width"]] = labels
    return out


def error_breakdown(directory, store, layouts):
    """Where a run's false alarms land among the original classes, on the test split."""
    result = read_json(directory / "evaluation_test.json")
    predictions = np.load(inside(result["predictions"]), mmap_mode="r")
    position = {i["id"]: i["prediction_index"] for i in result["per_image"]}
    fp = {1: np.zeros(10, np.int64), 2: np.zeros(10, np.int64)}
    for record in store.select("test"):
        _, truth = store.get(record)
        p = np.asarray(predictions[position[record["id"]]])
        valid = truth != 255
        for c in (1, 2):
            fp[c] += np.bincount(np.minimum(layouts[record["id"]][(p == c) & (truth != c) & valid], 9), minlength=10)
    return {name: {ORIGINAL[k]: float(v / max(fp[c].sum(), 1)) for k, v in enumerate(fp[c]) if v} for c, name in ((1, "building"), (2, "road"))}


def errors(numbers, uniform):
    """Share of each model's flooded false alarms per original class, mean over the 3 seeds."""
    from floodnet.data import Store
    store = Store(config())
    layouts = {r["id"]: masks10(r, 256) for r in store.select("test")}
    out = {}
    for method in METHODS:
        for budget in BUDGETS:
            rows = [r for r in uniform["runs"] if r["method"] == method and r["budget"] == budget]
            parts = [error_breakdown(inside(f"runs/{r['id']}"), store, layouts) for r in rows]
            out[f"{method}_{budget}"] = {t: {name: float(np.mean([p[t].get(name, 0.0) for p in parts])) for name in sorted({n for p in parts for n in p[t]})} for t in ("building", "road")}
    numbers["false_alarms"] = out


def proposal_figures():
    """Figures reused from the group's proposal PDF (kept outside the repository); skipped when it is absent."""
    if not inside(PROPOSAL).exists():
        return False
    target = FIG / "proposal"
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    subprocess.run(["pdfimages", "-png", str(inside(PROPOSAL)), str(target / "p")], check=True)
    names = {0: "utp_logo", 1: "fig_pixel_share", 2: "fig_class_presence", 3: "fig_6334", 4: "fig_6335", 5: "fig_samples", 6: "fig_rf_pipeline", 8: "fig_unet"}
    for path in sorted(target.glob("p-*.png")):
        index = int(path.stem.split("-")[1])
        if index in names:
            path.rename(target / f"{names[index]}.png")
        else:
            path.unlink()
    return True


def save(fig, name):
    fig.savefig(FIG / f"{name}.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def seed_dots(ax, x, values):
    ax.scatter(x + np.linspace(-.05, .05, len(values)), np.asarray(values) * 100, s=18, color="white", edgecolor="#0f172a", linewidth=.9, zorder=3)


def figures(numbers, uniform):
    c = numbers["configs"]
    fig, ax = plt.subplots(figsize=(7.5, 4), layout="constrained")
    for i, method in enumerate(METHODS):
        stats = [c[f"{method}_{b}"]["metrics"]["minority"] for b in BUDGETS]
        x = np.arange(2) + (i - .5) * .36
        means = np.array([s["mean"] for s in stats]) * 100
        bars = ax.bar(x, means, .32, color=COLOURS[method], label=NAMES[method])
        for p, s in zip(x, stats):
            seed_dots(ax, p, s["values"])
        ax.bar_label(bars, labels=[f"{m:.1f}%" for m in means], padding=10, fontsize=9)
    ax.set(xticks=[0, 1], xticklabels=["25% labels", "100% labels"], ylim=(0, 80), ylabel="Minority IoU (%)")
    ax.legend(frameon=False, loc="upper left")
    save(fig, "budget_comparison")

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), layout="constrained", sharey=True)
    for ax, key, title in zip(axes, ("building_iou", "road_iou"), ("Flooded building IoU", "Flooded road IoU")):
        for i, (method, budget) in enumerate(SHADES):
            s = c[f"{method}_{budget}"]["metrics"][key]
            bar = ax.bar(i, s["mean"] * 100, .72, color=SHADES[method, budget], label=f"{NAMES[method]}, {budget}%")
            seed_dots(ax, i, s["values"])
            ax.bar_label(bar, labels=[f"{s['mean'] * 100:.1f}%"], padding=8, fontsize=8.5)
        ax.set(title=title, xticks=[], ylim=(0, 90))
    axes[0].set_ylabel("IoU (%)")
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, ncols=4, loc="outside lower center")
    save(fig, "per_class_iou")

    fig, axes = plt.subplots(2, 2, figsize=(7.6, 6.2), layout="constrained")
    for ax, (method, budget) in zip(axes.flat, ((m, b) for m in METHODS for b in BUDGETS)):
        m = np.asarray(c[f"{method}_{budget}"]["confusion_pooled_over_seeds"], dtype=np.float64)
        share = m / m.sum(1, keepdims=True)
        ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
        ax.set(title=f"{NAMES[method]}, {budget}% labels", xticks=range(3), yticks=range(3), xticklabels=("Other", "Building", "Road"), yticklabels=("Other", "Building", "Road"), xlabel="Predicted", ylabel="True")
        ax.spines[:].set_visible(False)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{share[i, j] * 100:.1f}%", ha="center", va="center", color="white" if share[i, j] > .55 else "#0f172a", fontsize=10)
    save(fig, "confusion_matrices")

    labels = [f"{NAMES[m]}\n{b}%" for m in METHODS for b in BUDGETS]
    shades = [SHADES[m, b] for m in METHODS for b in BUDGETS]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), layout="constrained")
    for ax, key, unit, title in ((axes[0], "training_seconds", "s", "Training time (mean of 3 seeds)"), (axes[1], "inference_ms", "ms", "Median prediction time per image")):
        values = [c[f"{m}_{b}"][key]["mean"] for m in METHODS for b in BUDGETS]
        bars = ax.bar(labels, values, .7, color=shades)
        ax.bar_label(bars, labels=[f"{v:.1f} {unit}" for v in values], padding=3, fontsize=9)
        ax.set(title=title, ylabel=unit, ylim=(0, max(values) * 1.18))
    save(fig, "compute_comparison")

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), layout="constrained", sharey=True)
    for ax, budget in zip(axes, BUDGETS):
        for run, colour in zip(sorted((r for r in uniform["runs"] if r["method"] == "unet" and r["budget"] == budget), key=lambda r: r["seed"]), ("#0891b2", "#d97706", "#7c3aed")):
            v = [e["val_flooded_miou"] * 100 for e in run["history"]]
            ax.plot(range(1, len(v) + 1), v, color=colour, lw=2, label=f"seed {run['seed']}")
            ax.scatter(run["best_epoch"], v[run["best_epoch"] - 1], s=70, color="white", edgecolor=colour, linewidth=2, zorder=3)
        ax.set(title=f"U-Net, {budget}% labels", xlabel="Epoch", ylim=(0, 80))
    axes[0].set_ylabel("Validation minority IoU (%)")
    axes[0].legend(frameon=False, loc="lower right")
    save(fig, "unet_learning_curves")

    false = numbers["false_alarms"]
    order = ["water", "road non-flooded", "building non-flooded", "tree", "grass", "background", "vehicle", "pool"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), layout="constrained", sharey=True)
    for ax, target, name in ((axes[0], "road", "flooded road"), (axes[1], "building", "flooded building")):
        y = np.arange(len(order))
        for i, method in enumerate(METHODS):
            values = [false[f"{method}_100"][target].get(n, 0) * 100 for n in order]
            bars = ax.barh(y + (i - .5) * .38, values, .36, color=COLOURS[method], label=NAMES[method])
            ax.bar_label(bars, labels=[f"{v:.0f}%" for v in values], padding=2, fontsize=8)
        ax.set(title=f"False {name} pixels, true class", yticks=y, yticklabels=[n.capitalize() for n in order], xlabel="Share of false alarms (%)", xlim=(0, 100))
    axes[0].invert_yaxis()
    axes[0].legend(frameon=False, loc="lower right")
    save(fig, "false_alarms")


def scenes(results, uniform):
    colours = {c["id"]: np.array([int(c["colour"][i:i + 2], 16) for i in (1, 3, 5)], dtype=np.float32) for c in results["classes"]}
    web = inside("web")
    samples = {s["id"]: s for s in results["samples"]}
    seed = results["protocol"]["explorer_seed"]
    run = {m: next(r["id"] for r in uniform["runs"] if r["method"] == m and r["budget"] == 100 and r["seed"] == seed) for m in METHODS}

    def overlay(photo, mask):
        out = photo.copy()
        for c in (1, 2):
            out[mask == c] = .35 * out[mask == c] + .65 * colours[c]
        return out.astype(np.uint8)

    fig, axes = plt.subplots(len(results["representatives"]), 4, figsize=(8.2, 8.6), layout="constrained")
    for row, key in zip(axes, results["representatives"]):
        s = samples[key]
        photo = np.asarray(Image.open(web / s["rgb"]).convert("RGB"), dtype=np.float32)
        panels = [("Photo", photo.astype(np.uint8)), ("Ground truth", overlay(photo, np.asarray(Image.open(web / s["truth"]))))]
        for m in METHODS:
            item = uniform["samples"][key]
            panels.append((f"{NAMES[m]} (IoU {item['metrics'][run[m]]['flooded_miou'] * 100:.0f}%)" if item["metrics"][run[m]]["flooded_miou"] is not None else NAMES[m],
                           overlay(photo, np.asarray(Image.open(web / item["predictions"][run[m]])))))
        for ax, (title, image) in zip(row, panels):
            ax.imshow(image)
            ax.set_title(title, fontsize=8)
            ax.axis("off")
        row[0].text(-.04, .5, s["image_id"], transform=row[0].transAxes, rotation=90, va="center", ha="right", fontsize=8)
    save(fig, "test_scenes")


def web_assets(numbers):
    """The app's Presentation mode: its own small data file and compressed figures."""
    target = inside("web/data/presentation")
    target.mkdir(parents=True, exist_ok=True)
    write_json(inside("web/data/presentation.json"), {"false_alarms": numbers["false_alarms"], "feature_importance": numbers["feature_importance"]})
    sources = {"samples": FIG / "proposal/fig_samples.png", "rf_pipeline": FIG / "proposal/fig_rf_pipeline.png", "unet": FIG / "proposal/fig_unet.png", "scenes": FIG / "test_scenes.png"}
    for name, source in sources.items():
        if not source.exists():
            continue
        image = Image.open(source).convert("RGB")
        image.thumbnail((1500, 1500))
        image.save(target / f"{name}.webp", quality=88)


def from_results(results):
    """The bundle shape this script reads, built from results.json alone (no first-version comparison)."""
    samples = {s["id"]: {"predictions": s["predictions"], "metrics": s["metrics"]} for s in results["samples"]}
    return {"runs": results["runs"], "summary": results["summary"], "samples": samples, "comparisons": []}


def importances(bundle, seed):
    """Forest feature importances of the explorer-seed runs, from the saved models."""
    import joblib
    from floodnet.features import FEATURE_NAMES
    out = {}
    for budget in BUDGETS:
        run = next(r for r in bundle["runs"] if r["method"] == "rf" and r["budget"] == budget and r["seed"] == seed)
        model = joblib.load(inside(f"runs/{run['id']}/model.joblib"))
        out[str(budget)] = sorted(zip(FEATURE_NAMES, np.round(model.feature_importances_, 5).tolist()), key=lambda x: -x[1])
    return out


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold"})
    results = read_json("web/data/results.json")
    uniform = read_json("web/data/uniform.json") if inside("web/data/uniform.json").exists() else from_results(results)
    numbers = collect(uniform, results)
    errors(numbers, uniform)
    numbers["feature_importance"] = importances(uniform, results["protocol"]["explorer_seed"])
    numbers["dataset"] = {k: results["dataset"][k] for k in ("counts", "budget_counts", "exif_rotated_images", "preprocessing_seconds", "feature_preparation_seconds")}
    numbers["preprocessing"] = {k: read_json("web/data/preprocessing.json")["steps"][k] for k in ("pairing", "budgets", "resolution", "normalisation")}
    numbers["duplicates"] = {k: v for k, v in read_json("web/data/preprocessing.json")["steps"]["duplicates"].items() if k != "examples"}
    write_json(OUT / "numbers.json", numbers)
    proposal_figures()
    figures(numbers, uniform)
    scenes(results, uniform)
    shutil.copy(inside("output/figures/test_train_overlap.png"), FIG / "test_train_overlap.png")
    shutil.copy(inside("output/figures/dataset_distribution.png"), FIG / "dataset_distribution.png")
    web_assets(numbers)
    print("report assets written to", OUT)


if __name__ == "__main__":
    main()
