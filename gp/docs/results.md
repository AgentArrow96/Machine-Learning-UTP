# Measured results (v2.1: three approaches)

Completed 24 October 2026 on the official FloodNet-Supervised v1.0 release, 256×256 aspect-preserving inputs, one fixed 25% image subset (seed 42), and three seeds (42, 43, 44) for every configuration:

- **12 trained models:** SLIC + Random Forest and ResNet-18 U-Net at 25% and 100% labels. They were frozen with the scene-overlap analysis before the 448 test images were scored.
- **6 hybrid runs:** U-Net + DenseCRF (methodology section C), added afterwards without retraining. The CRF setting was chosen on validation only, and the hybrid runs were frozen separately before their own test evaluation.

Frozen identities: 12 trained runs `65a77d261e7b0266be386f0560135adfa8b02bf6dbe7cfee4a3ff57e6c7743b7`; hybrid runs and CRF selection `8708b9323865d5763b5d3998b5128da7ebf29d287f7cfb08d889604ca6404db7`.

## Held-out segmentation

Primary score: mean IoU of flooded building and flooded road, pooled over all test pixels (padding ignored). Values are mean ± sample SD across the three seeds.

| Method | Labels | Flooded mIoU | Per seed (42 / 43 / 44) | 3-class mIoU | Building IoU | Road IoU |
|---|---:|---:|---|---:|---:|---:|
| SLIC + RF | 25% | 23.4 ± 0.2 | 23.4 / 23.2 / 23.6 | 46.6 ± 0.2 | 26.5 ± 0.4 | 20.4 ± 0.1 |
| SLIC + RF | 100% | 22.5 ± 0.2 | 22.3 / 22.7 / 22.6 | 45.4 ± 0.2 | 24.1 ± 0.3 | 20.9 ± 0.2 |
| U-Net | 25% | 52.3 ± 2.2 | 54.4 / 50.1 / 52.3 | 67.2 ± 1.6 | 64.3 ± 2.4 | 40.3 ± 4.9 |
| U-Net | 100% | 63.3 ± 2.7 | 60.2 / 64.5 / 65.2 | 74.9 ± 1.9 | 68.2 ± 0.9 | 58.4 ± 5.6 |
| U-Net + DenseCRF | 25% | 55.1 ± 2.4 | 57.1 / 52.5 / 55.8 | 69.2 ± 1.7 | 68.7 ± 1.7 | 41.6 ± 5.4 |
| U-Net + DenseCRF | 100% | **65.5 ± 3.4** | 61.6 / 66.8 / 68.0 | 76.5 ± 2.4 | 71.0 ± 0.5 | 59.9 ± 7.1 |

**Method gap.** The U-Net beats SLIC + RF by about 29 points at 25% and 41 points at 100%. Every U-Net seed is above every RF seed. The best approach at both budgets is the hybrid U-Net + DenseCRF.

**Budget effect (seed-matched, 100% minus 25%).**
- **U-Net:** +11.0 ± 4.6 points (+5.7, +14.4, +12.9). Every seed improved. The v1 single-seed run (seed 42) happened to show the *smallest* gain of the three, so v1 understated the benefit of more labels.
- **U-Net + DenseCRF:** +10.3 ± 5.2 points (+4.4, +14.3, +12.3). The budget effect carries over almost unchanged.
- **SLIC + RF:** −0.9 ± 0.3 points (−1.1, −0.6, −1.1). With every flooded region now used for training, RF does find more flooded pixels at 100%: building recall rises from 70.7% to 80.2% and road recall from 53.6% to 69.2%. It also raises more false alarms, so IoU does not improve. Two-thirds or more of RF's false-alarm pixels fall in the 396 test scenes that contain no flooded infrastructure, and that count rises by 54% at 100% labels (seed 42). In the explorer, many of those false alarms land on open floodwater (image 7486 is an example), which suggests the superpixel colour/texture features are the bottleneck rather than the amount of labels.

The epoch-to-epoch validation score of the U-Net swings by up to about 16 points. The three seeds show the result is stable in direction; three replicates are not a significance test.

## Hybrid: does CRF refinement help?

Each hybrid run refines the exact U-Net checkpoint with the same budget and seed, so the paired difference isolates the CRF (methodology section 5).

**Validation tuning (test never used).** Six appearance-kernel settings were scored on all 450 validation images for each of the six U-Net runs (2 min 12 s on 12 CPU workers).

| Appearance kernel (sxy / srgb, weight 10) | Mean validation change | Range over 6 runs |
|---|---:|---|
| **20 / 5 (selected)** | **+3.2** | +2.6 to +3.9 |
| 20 / 13 | +1.3 | +0.7 to +2.7 |
| 50 / 5 | +1.6 | +1.3 to +2.4 |
| 50 / 13 | −2.9 | −5.1 to −1.1 |
| 80 / 5 | −0.0 | −0.8 to +0.7 |
| 80 / 13 | −5.3 | −7.6 to −3.6 |

- Smaller spatial and colour ranges work better at this resolution. The common PASCAL VOC defaults (80 / 13) would cost 5.3 points.
- Because the best setting helped every run, validation **adopted** refinement and the weaker-strength fallback was not needed.
- The selected setting sits at the edge of the grid. Smaller ranges were not tried, in keeping with the methodology's minimal-tuning policy, so the CRF benefit may be slightly understated.

**Test-set effect (hybrid minus its own U-Net, seed-matched, points).**

| Labels | Flooded mIoU | Per seed (42 / 43 / 44) | Building IoU | Road IoU | Building recall | Road recall |
|---|---:|---|---:|---:|---:|---:|
| 25% | **+2.9** | +2.7 / +2.4 / +3.4 | +4.4 | +1.3 | +3.0 | −1.2 |
| 100% | **+2.2** | +1.4 / +2.4 / +2.8 | +2.8 | +1.5 | +1.1 | −1.2 |

- The CRF improved flooded-class mIoU for every seed at both budgets, close to the validation estimate (+3.2).
- Most of the gain comes from flooded buildings, whose sharp edges and uniform roofs suit a colour-aware CRF.
- Road recall drops slightly (about −1 point) as the CRF smooths away some thin road pixels, which is the risk the methodology flagged. Road IoU still rises because fewer false-alarm pixels remain (one seed at 100% shows −0.2 road IoU).
- The gain is similar in the lower- and higher-overlap halves of the target images (+2.3 vs +1.9 at 100%), so it is not an artefact of memorised scenes.
- **Cost:** no training; about 73 ms more per image (single-threaded CPU CRF), so batch-one inference rises from 5.6 ms to about 79 ms (p95 96 ms).

## Scene overlap between training and test

ORB keypoint matching with RANSAC homographies (method in [experiment_protocol.md](experiment_protocol.md)) verified 2,186 overlapping evaluation/training frame pairs:

- 357 of 448 test images (80%) and 363 of 450 validation images (81%) share ground with at least one training frame, often 50–100% of the frame.
- 49 of the 52 test images that contain flooded buildings or roads overlap training. The flooded areas were photographed densely, so almost every flooded test scene was seen, from a neighbouring frame or a later flight, during training.
- The 91 overlap-free test images contain only 2 with flooded buildings and 3 with flooded roads. Their target IoU (U-Net 100%: 49.2%) is driven by false alarms in target-free scenes, so it is **not** used as a leakage estimate. It is reported in the all-runs CSV for transparency.

Sensitivity check: the 52 target-containing test images were split at their median overlap (68%) into two halves of 26 images, and every run was scored on each half.

| Method | Labels | Lower-overlap half | Higher-overlap half |
|---|---:|---:|---:|
| SLIC + RF | 25% | 40.0 ± 0.3 | 39.8 ± 0.0 |
| SLIC + RF | 100% | 44.6 ± 0.2 | 44.3 ± 0.2 |
| U-Net | 25% | 55.4 ± 2.4 | 53.4 ± 1.3 |
| U-Net | 100% | 65.8 ± 2.8 | 63.1 ± 2.4 |
| U-Net + DenseCRF | 25% | 58.4 ± 2.9 | 56.1 ± 1.5 |
| U-Net + DenseCRF | 100% | 68.2 ± 3.9 | 65.0 ± 2.8 |

Within the observed range, more shared ground did not mean higher scores, so there is no sign that the method ranking or the budget effect rests on memorised scenes. The check is weak: the lower-overlap half still mostly overlaps training. **The official split measures performance on areas already seen in training, not on a new flood area.** This is a property of the published FloodNet split, not of these models. (On target images, RF-100 also beats RF-25, which confirms that RF's lower full-split score comes from false alarms in target-free scenes.)

## Training and compute

| Configuration | Training time (mean, range) | Median inference | Epochs (selected) | Peak GPU allocated |
|---|---:|---:|---|---:|
| SLIC + RF 25% | 1.6 s | 42.0 ms | — | N/A (CPU) |
| SLIC + RF 100% | 3.3 s (3.2–3.4) | 44.9 ms | — | N/A (CPU) |
| U-Net 25% | 80 s (68–87) | 5.5 ms | 28 (22), 23 (17), 30 (24) | 512 MiB |
| U-Net 100% | 193 s (158–219) | 5.6 ms | 21 (15), 28 (22), 26 (20) | 512 MiB |
| U-Net + DenseCRF 25% / 100% | none (reuses U-Net) | 78.2 / 79.0 ms | — | as U-Net, plus CPU CRF |

All U-Net runs trained on the RTX 5080 Laptop GPU with CUDA 12.8 / PyTorch 2.11.0+cu128, BF16 and batch 4. Training is deterministic: seed 42 reproduced the v1 runs exactly (same epochs, selected checkpoints, validation and test scores). RF inference now predicts single-threaded, which removed the 22,366 scikit-learn parallel warnings of v1 and cut RF latency from about 64 ms to 42–45 ms per image. The remaining RF time is mostly SLIC and feature extraction. Shared preparation took 4.1 minutes, RF feature extraction 1.7 minutes, and the overlap analysis about 14 minutes on first run (keypoints are then cached; re-verifying all pairs takes about 6 minutes).

Inference timing: batch one, ten warm-ups, the first 50 sorted test IDs. It starts from processed RGB in host memory and ends with a CPU mask, with CUDA synchronisation. This compares CPU and GPU pipelines on this laptop, not algorithms on equal hardware.

## Hybrid implementation notes

- **Backend:** `pydensecrf2` 1.1. No Windows/cp312 wheels exist for any DenseCRF package, so it is compiled from source with the installed MSVC 2022 Build Tools (`pip install -e .[crf]`). The methodology's superpixel-smoothing fallback was not needed.
- **Inputs:** the U-Net softmax probabilities come from the same logits as the frozen U-Net predictions. The tuning step checks this by reproducing every U-Net validation score exactly before any CRF runs.
- **Freeze:** the hybrid has its own freeze (`runs/hybrid_experiment.json`), which references the 12-run freeze and the CRF selection. Each hybrid validation evaluation must reproduce its tuning score exactly before test.
- **Reuse fix:** a completed run is now reused when only the source-code hash changed and its checkpoint hash verifies. Before this, any code change after training made the pipeline refuse the finished runs. The 12 original runs were reused, not retrained.

## Changes from v1

v1 (preserved in `runs/archive/v1_f341bd24/`) reported RF 23.8% / 23.7% and U-Net 54.4% / 60.2% from a single seed. v2 corrects:

1. **RF region sampling.** v1 sampled regions uniformly under the 100k cap, so RF-100 saw only 1,510 of 4,155 flooded-building and 2,969 of 8,970 flooded-road regions, barely more than RF-25. That made the RF budget comparison meaningless. v2 keeps every flooded region (RF-100 now fits 86,875 / 4,155 / 8,970 regions). The RF result is now a genuine finding: more labels raise recall but not IoU.
2. **Seeds.** One seed became three, with seed-matched budget effects. The U-Net budget gain is +11.0 points on average, not the +5.7 from the single v1 seed.
3. **Overlap.** The 8×8 average-hash screen (2,019 unverified "candidates") was replaced by verified geometric matching, plus the stratified sensitivity check above.
4. **RF inference.** Single-threaded prediction per image, with no warnings.
5. **Results app.** Restyled to the lecture app's design system, all classes prefixed `fn-`, readable (unminified) code, and the export now committed under `web/data/`.

The export and presentation code (`src/floodnet/export.py`, `web/`) was refined after the 12 runs were trained and frozen, to add the overlap-stratified analysis and the hybrid. Training and scoring code for those runs is unchanged, `UNetEngine.predict` was refactored to share its logits with the hybrid, and every exported number is recomputed from the saved predictions by `scripts/verify_delivery.py`.

## Verification

- 34 tests pass with none skipped (`output/pytest.xml`), including a real CUDA checkpoint round trip, the correction tests and the hybrid tests. The hybrid tests cover noise removal, padding, selection, the weaker-strength fallback, the freeze gate, run reuse and the paired effects.
- 41 browser checks pass at 1440 px and 390 px with no console errors (`output/playwright/checks.log`).
- `scripts/verify_delivery.py` recomputes, from the raw prediction arrays for all 18 runs:
  - pooled, clean-subset and overlap-strata scores
  - the seed summaries and CRF paired effects
  - the RF region tables
  - the CSVs, the 2,688 explorer PNGs, the checkpoint hashes and the figure hashes

  It also confirms that each hybrid refines its frozen U-Net checkpoint and reproduces its validation tuning score, and that the pipeline log has no scikit-learn warnings (`output/verification.json`).

| Openable result | Producing code |
|---|---|
| [Standalone app](../web/index.html) (serve over HTTP; see README) | [App](../web/assets/js/app.js), [charts](../web/assets/js/charts.js), [styles](../web/assets/css/main.css) |
| [Seed summary CSV](../web/data/exports/d41ec3fe08e7197b/summary.csv), [all runs CSV](../web/data/exports/d41ec3fe08e7197b/aggregate.csv), [per-image CSV](../web/data/exports/d41ec3fe08e7197b/per_image.csv) | [Evaluation](../src/floodnet/evaluation.py), [export](../src/floodnet/export.py) |
| [CRF tuning CSV](../web/data/exports/d41ec3fe08e7197b/crf_tuning.csv), [CRF figure](../output/figures/crf_refinement.png) | [Hybrid](../src/floodnet/hybrid.py), [settings](../configs/hybrid.json) |
| [Verified overlap pairs CSV](../web/data/exports/d41ec3fe08e7197b/overlap_pairs.csv) | [Overlap analysis](../src/floodnet/overlap.py) |
| [Budget comparison](../output/figures/budget_comparison.png), [overlap sensitivity](../output/figures/overlap_sensitivity.png), [test–train overlap](../output/figures/test_train_overlap.png) | [Report plotting script](../scripts/make_figures.py) |
| [Learning curves](../output/figures/unet_learning_curves.png), [per-class IoU](../output/figures/per_class_iou.png), [confusion matrices](../output/figures/confusion_matrices.png), [compute](../output/figures/compute_comparison.png), [dataset distribution](../output/figures/dataset_distribution.png) | Report plotting script |
| [12-run freeze](../runs/final_experiment.json), [hybrid freeze](../runs/hybrid_experiment.json), [CRF selection](../runs/hybrid_selection.json) | Evaluation and hybrid freeze gates |
