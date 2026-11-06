"""Standalone report figures from the verified export; no model fitting or tuning."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from floodnet.common import ProjectError, environment, inside, log, now, read_json, sha256, write_json

environment()
os.environ["MPLCONFIGDIR"] = str(inside(".cache/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

METHODS = ("rf", "unet", "hybrid")
COLOURS = {"rf": "#059669", "unet": "#4f46e5", "hybrid": "#7c3aed"}
NAMES = {"rf": "SLIC + RF", "unet": "U-Net", "hybrid": "U-Net + CRF"}
SHADES = {("rf", 25): "#6ee7b7", ("rf", 100): "#059669", ("unet", 25): "#a5b4fc", ("unet", 100): "#4f46e5",
          ("hybrid", 25): "#c4b5fd", ("hybrid", 100): "#7c3aed"}
SEED_COLOURS = ["#0891b2", "#d97706", "#7c3aed", "#e11d48"]


def main():
    data = read_json("web/data/results.json")
    if data["status"] != "complete" or len(data["runs"]) != data["protocol"]["expected_runs"]:
        raise ProjectError("Report figures require every completed held-out evaluation")
    directory = inside("output/figures")
    directory.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.titleweight": "bold", "svg.fonttype": "none"})
    seeds = data["protocol"]["train_seeds"]
    explorer_seed = data["protocol"]["explorer_seed"]
    summary = {(s["method"], s["budget"]): s for s in data["summary"]}
    files = []

    def save(fig, name):
        for extension in ("png", "svg"):
            path = directory / f"{name}.{extension}"
            fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
            files.append({"file": path.relative_to(directory).as_posix(), "sha256": sha256(path)})
        plt.close(fig)

    def seed_bars(ax, field, scope):
        """Mean bars, min-max whiskers and one dot per seed for every method at both budgets."""
        for index, method in enumerate(METHODS):
            stats = [summary[(method, b)][scope][field] for b in (25, 100)]
            x = np.arange(2) + (index - 1) * .28
            means = np.array([s["mean"] for s in stats]) * 100
            low = means - np.array([s["min"] for s in stats]) * 100
            high = np.array([s["max"] for s in stats]) * 100 - means
            bars = ax.bar(x, means, .26, color=COLOURS[method], alpha=.88, label=NAMES[method],
                          yerr=[low, high], capsize=5, error_kw={"elinewidth": 1.4, "ecolor": "#0f172a"})
            for position, stat in zip(x, stats):
                ax.scatter(position + np.linspace(-.06, .06, len(stat["values"])), np.array(stat["values"]) * 100,
                           s=16, color="white", edgecolor="#0f172a", linewidth=.9, zorder=3)
            ax.bar_label(bars, labels=[f"{m:.1f}%" for m in means], padding=14, fontsize=8)
        ax.set(xticks=[0, 1], xticklabels=["25% labels", "100% labels"], ylim=(0, 100))
        ax.yaxis.grid(True, alpha=.15); ax.set_axisbelow(True)

    fig, ax = plt.subplots(figsize=(8.5, 4.8), layout="constrained")
    seed_bars(ax, "flooded_miou", "full")
    ax.set(ylabel="Flooded-class mean IoU (%)",
           title=f"Held-out test split ({data['dataset']['counts']['test']} images) | seeds {', '.join(map(str, seeds))}")
    ax.legend(frameon=False, loc="upper left")
    ax.text(1, -.13, "Bars = seed mean; whiskers = min-max; dots = individual seeds", transform=ax.transAxes, ha="right", fontsize=8, color="#64748b")
    save(fig, "budget_comparison")

    targets = data["dataset"]["overlap"]["target_images"]
    fig, ax = plt.subplots(figsize=(11, 4.8), layout="constrained")
    configurations = [(m, b) for m in METHODS for b in (25, 100)]
    for offset, (stratum, colour, label) in enumerate((("low", "#0891b2", f"Lower-overlap half ({targets['low_images']} images)"),
                                                        ("high", "#d97706", f"Higher-overlap half ({targets['high_images']} images)"))):
        stats = [summary[c][f"overlap_{stratum}"]["flooded_miou"] for c in configurations]
        means = np.array([s["mean"] for s in stats]) * 100
        errors = [means - np.array([s["min"] for s in stats]) * 100, np.array([s["max"] for s in stats]) * 100 - means]
        bars = ax.bar(np.arange(len(configurations)) + (offset - .5) * .36, means, .34, color=colour, yerr=errors, capsize=4, label=label)
        ax.bar_label(bars, labels=[f"{m:.1f}" for m in means], padding=10, fontsize=8)
    ax.set(xticks=np.arange(len(configurations)), xticklabels=[f"{NAMES[m]} {b}%" for m, b in configurations], ylim=(0, 100), ylabel="Flooded-class mean IoU (%)",
           title=f"Target test images split at median training overlap ({targets['split_overlap'] * 100:.0f}%)")
    ax.legend(frameon=False, loc="upper left"); ax.yaxis.grid(True, alpha=.15); ax.set_axisbelow(True)
    save(fig, "overlap_sensitivity")

    fig, ax = plt.subplots(figsize=(10, 4.8), layout="constrained")
    for i, (method, budget) in enumerate(configurations):
        item = summary[(method, budget)]["full"]
        means = [item[k]["mean"] * 100 for k in ("building_iou", "road_iou")]
        errors = [(item[k]["sd"] or 0) * 100 for k in ("building_iou", "road_iou")]
        ax.bar(np.arange(2) + (i - 2.5) * .14, means, .14, yerr=errors, capsize=3, color=SHADES[(method, budget)], label=f"{NAMES[method]} {budget}%")
    ax.set(xticks=np.arange(2), xticklabels=["Flooded building", "Flooded road"], ylim=(0, 100), ylabel="IoU (%)",
           title="Per-class held-out IoU (mean ± SD across seeds)")
    ax.legend(frameon=False, ncols=3); ax.yaxis.grid(True, alpha=.15); ax.set_axisbelow(True)
    save(fig, "per_class_iou")

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), layout="constrained")
    for column, budget in enumerate((25, 100)):
        runs = sorted((r for r in data["runs"] if r["method"] == "unet" and r["budget"] == budget), key=lambda r: r["seed"])
        ax = axes[0, column]
        for i, run in enumerate(runs):
            epochs = [e["epoch"] for e in run["history"]]
            ax.plot(epochs, [e["val_flooded_miou"] * 100 for e in run["history"]], color=SEED_COLOURS[i], label=f"Seed {run['seed']}")
            ax.scatter([run["best_epoch"]], [run["history"][run["best_epoch"] - 1]["val_flooded_miou"] * 100], color="white",
                       edgecolor=SEED_COLOURS[i], linewidth=1.8, zorder=3)
        ax.set(title=f"U-Net | {budget}% labels: validation flooded mIoU", xlabel="Epoch", ylabel="Validation flooded mIoU (%)", ylim=(0, 100))
        ax.legend(frameon=False, fontsize=8)
        ax = axes[1, column]
        run = next(r for r in runs if r["seed"] == explorer_seed)
        epochs = [e["epoch"] for e in run["history"]]
        ax.plot(epochs, [e["train_loss"] for e in run["history"]], label="Train", color=COLOURS["unet"])
        ax.plot(epochs, [e["val_loss"] for e in run["history"]], label="Validation", color="#d97706", linestyle="--")
        ax.axvline(run["best_epoch"], linestyle=":", color="#64748b", label=f"Selected epoch {run['best_epoch']}")
        ax.set(title=f"U-Net | {budget}% labels: loss (seed {explorer_seed})", ylabel="Weighted cross-entropy", xlabel="Epoch")
        ax.legend(frameon=False, fontsize=8)
    save(fig, "unet_learning_curves")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), layout="constrained")
    labels = [f"{NAMES[m]}\n{b}%" for m, b in configurations]
    colours = [COLOURS[m] for m, _ in configurations]
    for ax, key, scale, title, ylabel in ((axes[0], "training_seconds", 1, "Model training time (log scale)", "Seconds"),
                                          (axes[1], "inference_median_seconds", 1000, "Batch-one inference median", "Milliseconds")):
        # The hybrid trains nothing, so it has no training-time bar.
        shown = [i for i, c in enumerate(configurations) if summary[c][key]["mean"] is not None]
        stats = [summary[configurations[i]][key] for i in shown]
        means = np.array([s["mean"] for s in stats]) * scale
        ax.bar([labels[i] for i in shown], means, color=[colours[i] for i in shown],
               yerr=[means - np.array([s["min"] for s in stats]) * scale, np.array([s["max"] for s in stats]) * scale - means], capsize=4)
        ax.set(title=title, ylabel=ylabel)
        ax.tick_params(axis="x", labelsize=8)
        if key == "training_seconds":
            ax.set_yscale("log")
    nets = [(b, max(r["peak_gpu_allocated_bytes"] for r in data["runs"] if r["method"] == "unet" and r["budget"] == b)) for b in (25, 100)]
    axes[2].bar([f"U-Net\n{b}%" for b, _ in nets], [peak / 2**20 for _, peak in nets], color=COLOURS["unet"])
    axes[2].set(title="Peak training GPU allocation", ylabel="MiB")
    fig.suptitle("Compute comparison | RF: CPU; U-Net: CUDA; U-Net + CRF: CUDA + CPU CRF, no training | mean of seeds, whiskers = min-max\n"
                 "Shared data preparation / RF feature extraction excluded from training bars", fontsize=11)
    save(fig, "compute_comparison")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    original = np.array(data["dataset"]["distributions"]["train"]["pixel_counts"], dtype=float)
    shares = original / original.sum() * 100
    axes[0].bar(np.arange(10), shares, color=["#f59e0b" if i == 1 else "#0891b2" if i == 3 else "#94a3b8" for i in range(10)])
    axes[0].set(xticks=np.arange(10), xticklabels=data["dataset"]["original_classes"], ylabel="Training pixel share (%)", title="Original ten classes (log scale)")
    axes[0].set_yscale("log"); axes[0].tick_params(axis="x", labelrotation=50)
    for label in axes[0].get_xticklabels():
        label.set_ha("right")
    processed = np.array(data["dataset"]["processed_distributions"]["train"]["pixel_counts"], dtype=float)
    values = processed / processed.sum() * 100
    bars = axes[1].bar(["Other", "Flooded\nbuilding", "Flooded\nroad"], values, color=[c["colour"] for c in data["classes"]])
    axes[1].bar_label(bars, labels=[f"{v:.2f}%" for v in values], padding=3)
    axes[1].set(ylim=(0, 110), ylabel="Training pixel share (%)", title="Processed three classes (padding ignored)")
    save(fig, "dataset_distribution")

    fig, axes = plt.subplots(2, 3, figsize=(13, 8), layout="constrained")
    by_budget = [(m, b) for b in (25, 100) for m in METHODS]
    for ax, (method, budget) in zip(axes.ravel(), by_budget):
        run = next(r for r in data["runs"] if r["method"] == method and r["budget"] == budget and r["seed"] == explorer_seed)
        matrix = np.array(run["metrics"]["confusion_matrix"], dtype=float)
        norm = matrix / matrix.sum(1, keepdims=True)
        image = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for row in range(3):
            for col in range(3):
                ax.text(col, row, f"{norm[row, col] * 100:.1f}%", ha="center", va="center", color="white" if norm[row, col] > .55 else "#0f172a")
        ax.set(xticks=range(3), yticks=range(3), xticklabels=["Other", "Building", "Road"], yticklabels=["Other", "Building", "Road"],
               xlabel="Prediction", ylabel="Ground truth", title=f"{NAMES[method]} | {budget}% | seed {explorer_seed}")
    fig.colorbar(image, ax=axes.ravel().tolist(), label="Fraction of ground-truth class", shrink=.75)
    save(fig, "confusion_matrices")

    overlap = data["dataset"]["overlap"]
    fig, ax = plt.subplots(figsize=(7.5, 4.2), layout="constrained")
    histogram = overlap["test_max_overlap_histogram"]
    bars = ax.bar([h["label"] for h in histogram], [h["count"] for h in histogram],
                  color=["#059669" if h["label"] == "None" else "#d97706" for h in histogram])
    ax.bar_label(bars, padding=3)
    test = overlap["summary"]["test"]
    ax.set(ylabel="Test images", xlabel="Largest verified overlap with any training frame",
           title=f"Scene overlap: {test['overlapping_train']} of {test['images']} test images share ground with training")
    save(fig, "test_train_overlap")

    hybrid = data["hybrid"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained")
    for budget, offset in ((25, -.17), (100, .17)):
        effect = next(e for e in data["crf_effects"] if e["budget"] == budget)["full"]
        keys = ("flooded_miou", "building_iou", "road_iou")
        means = np.array([effect[k]["mean"] for k in keys]) * 100
        errors = [means - np.array([effect[k]["min"] for k in keys]) * 100, np.array([effect[k]["max"] for k in keys]) * 100 - means]
        bars = axes[0].bar(np.arange(3) + offset, means, .32, color=SHADES[("hybrid", budget)], yerr=errors, capsize=4, label=f"{budget}% labels")
        axes[0].bar_label(bars, labels=[f"{m:+.1f}" for m in means], padding=6, fontsize=8)
    axes[0].axhline(0, color="#0f172a", linewidth=.8)
    axes[0].set(xticks=np.arange(3), xticklabels=["Flooded mIoU", "Building IoU", "Road IoU"], ylabel="Change vs. own U-Net (points)",
                title="Test-set effect of DenseCRF refinement (seed-matched)")
    axes[0].legend(frameon=False)
    grid_rows = [row for row in hybrid["grid"] if row["round"] == hybrid["grid"][0]["round"]]
    sxys = sorted({row["params"]["bilateral"]["sxy"] for row in grid_rows})
    srgbs = sorted({row["params"]["bilateral"]["srgb"] for row in grid_rows})
    heat = np.full((len(srgbs), len(sxys)), np.nan)
    for row in grid_rows:
        heat[srgbs.index(row["params"]["bilateral"]["srgb"]), sxys.index(row["params"]["bilateral"]["sxy"])] = row["mean_gain"] * 100
    limit = np.nanmax(np.abs(heat))
    image = axes[1].imshow(heat, cmap="RdBu", vmin=-limit, vmax=limit)
    for r in range(len(srgbs)):
        for c in range(len(sxys)):
            selected = hybrid["selected"]["bilateral"]["sxy"] == sxys[c] and hybrid["selected"]["bilateral"]["srgb"] == srgbs[r]
            axes[1].text(c, r, f"{heat[r, c]:+.1f}" + ("\nselected" if selected else ""), ha="center", va="center", fontsize=9,
                         fontweight="bold" if selected else "normal", color="white" if abs(heat[r, c]) > .6 * limit else "#0f172a")
    axes[1].set(xticks=range(len(sxys)), xticklabels=sxys, yticks=range(len(srgbs)), yticklabels=srgbs,
                xlabel="Appearance kernel spatial range (sxy)", ylabel="Colour range (srgb)",
                title="Validation tuning: mean change over 6 U-Net runs (points)")
    fig.colorbar(image, ax=axes[1], shrink=.8)
    save(fig, "crf_refinement")

    write_json(directory / "manifest.json", {
        "created_at": now(), "experiment_identity": data["experiment_identity"],
        "hybrid_experiment_identity": data.get("hybrid_experiment_identity"),
        "source_export_sha256": sha256("web/data/results.json"), "script_sha256": sha256(Path(__file__)),
        "matplotlib_version": matplotlib.__version__,
        "note": (f"Training seeds {seeds}; bars are seed means with min-max whiskers. Confusion matrices and loss curves show seed "
                 f"{explorer_seed}. Official test split; overlap sensitivity splits target-containing test images at their median training overlap; 256x256, padding excluded."),
        "files": files})
    log(f"Saved {len(files)} report figure files and provenance manifest to {directory}")


if __name__ == "__main__":
    main()
