# Experiment protocol (v2)

v2 replaces the single-seed v1 experiment. The v1 artifacts are preserved under `runs/archive/v1_f341bd24/`; see "Changes from v1" in [results.md](results.md).

## Question and targets

Compare practical SLIC + Random Forest and pretrained ResNet-18 U-Net pipelines under 25% and 100% labelled-image budgets. Original FloodNet mask IDs 1 and 3 map to flooded building and flooded road; remaining valid IDs map to Other. Padding is 255 and ignored. Other includes water and non-flooded infrastructure.

Use the official release split (1445/450/448 images). Audit before training. The pixel distributions and example figures in the proposal are historical inputs, not substitutes for this run's audit.

## Preparation and budgets

The release contains 57 JPEGs with EXIF orientation 3. Annotations use display coordinates: apply EXIF transpose to RGB before resizing and leave indexed masks unchanged. (Checked quantitatively: mask boundaries align with image edges about 2× better after the transpose.) Audit records orientation and rejects unreviewed orientation values.

Resize RGB with Pillow Lanczos and masks with nearest neighbour, preserving aspect ratio inside a centred 256-square frame. RGB padding uses the ImageNet mean; mask padding uses 255. Store geometry and before/after class distributions.

One 25% image subset is stratified by processed target presence (both/building/road/neither) using `subset_seed` 42, with floor-rounded total size and deterministic remainder allocation. The 100% budget is the entire official training split. No validation/test images enter either budget. Every training seed uses the same subset.

## Seeds

Each of the four configurations (2 methods × 2 budgets) is trained with `train_seeds` 42, 43 and 44: 12 runs in total. The training seed controls RF bootstrap/feature sampling and region sampling, U-Net initialisation of the decoder, data order and augmentation. Report the mean, sample standard deviation and min–max across seeds. Budget effects are computed seed-matched (the 100% run minus the 25% run with the same seed). Three replicates show stability; they are not a significance test.

## Scene overlap between splits

Consecutive UAV frames overlap, and the official split assigns frames independently. The `overlap` stage verifies shared ground between every validation/test image and candidate training frames:

- **Candidates:** training frames within ±5 frame IDs, plus the 5 nearest training images by cosine similarity of ImageNet ResNet-18 pooled features of the processed RGB.
- **Verification:** ORB keypoints (2,000) on EXIF-oriented greyscale originals scaled to 1,024 px, cross-checked Hamming matches, RANSAC homography (residual 4 px, 2,000 trials, fixed RNG). A pair is verified when it has at least 20 inliers (RANSAC only runs on pairs with at least 20 cross-checked matches; unrelated frames typically give fewer than 5 matches, and the weakest verified pairs, with 20–30 inliers, were inspected visually and show the same structures, often from a different flight and rotated), an inlier ratio of at least 0.25, an orientation-preserving footprint, and the warped training frame covers at least 10% of the evaluation frame.
- **Use:** headline results stay on the official split, as the methodology requires. The overlap analysis reads only images, and its identity is frozen with the experiment before test evaluation.
- **Sensitivity check:** an overlap-free ("clean") test subset turned out to contain almost no flooded targets (91 images; 2 with flooded buildings, 3 with flooded roads), because the flooded areas were photographed densely and 49 of the 52 target-containing test images overlap training. Its target IoU is dominated by false alarms on target-free scenes, so it is reported in the CSVs but not used as a leakage estimate. Instead, the 52 target-containing test images are split at their median training overlap and every run is scored on each half by pooling the saved per-image confusion counts (no re-inference). This shows whether more shared ground means higher scores; it cannot measure performance on an unseen flood area.

## Traditional pipeline

SLIC uses approximately 250 connected superpixels and compactness 10 on unpadded content. Region features are RGB/HSV mean/std plus ten normalised uniform LBP bins (P=8, R=1). Majority ground-truth labels use the lowest class ID to break ties.

The forest uses 150 trees, maximum depth 20, balanced class weights, the training seed, and up to eight CPU workers for fitting. **Region sampling (`keep_targets`):** every flooded-building and flooded-road region from the budget's images is kept; the remainder of the 100k-region cap is filled with Other regions, shuffled within each image and drawn round-robin across images. (v1 sampled uniformly, which discarded about two-thirds of the flooded regions at the 100% budget.) Sampled image/region IDs are saved; all three classes must be present. Inference predicts one image at a time with a single-threaded forest.

## Neural pipeline

ImageNet ResNet-18 encoder; U-Net decoder channels 128/64/32/16/8; three logits. ImageNet mean/std normalisation. AdamW LR=1e-4, weight decay=1e-4, batch=4. A real-data backward preflight may choose batch=2 for all runs. BF16 preferred, FP16/scaler otherwise. No CPU training fallback.

For training-subset pixel proportions p, form inverse-frequency weights 1/p, normalise to arithmetic mean one, then clip to [0.25, 4]. Weighted cross-entropy ignores padding. Training/validation loss curves accumulate the weighted-loss denominator rather than averaging batch means.

Training uses matched horizontal/vertical flips, random quarter-turns, and image-only contrast/brightness magnitude 0.1. Validation/test transforms are deterministic. Maximum 30 epochs; stop after six consecutive non-improving validation flooded-mIoU epochs. A strictly greater score replaces the best checkpoint; ties keep the earlier one. No hyperparameter search.

## Hybrid pipeline (U-Net + DenseCRF)

Methodology section C, added after the 12-run freeze. No network is trained: each of the six final U-Net checkpoints is reused unchanged, and its float32 softmax probabilities (the same logits whose argmax gives the U-Net result) are refined by a fully connected CRF (`pydensecrf2` 1.1, built from source with MSVC because no Windows wheels exist). The CRF runs on the CPU over the unpadded image only: unary `-log p`, a spatial Gaussian kernel (sxy 3, weight 3) and an appearance kernel on the 256-px RGB, 5 mean-field iterations, then argmax.

Settings are chosen on validation only (`configs/hybrid.json`): appearance kernel sxy ∈ {20, 50, 80} × srgb ∈ {5, 13}, weight 10. The selected setting maximises the mean validation flooded-mIoU gain over each run's own U-Net across all six U-Net runs, and one setting is shared by both budgets and all seeds. If no setting helps, the one adjustment the methodology allows is tried: a weaker appearance weight (5, 3). Validation adopts refinement only if the mean gain is positive; the hybrid is reported on test either way. The selection (`runs/hybrid_selection.json`) and the six hybrid runs are frozen in `runs/hybrid_experiment.json`, which references the 12-run freeze, before any hybrid test evaluation. Each hybrid's validation evaluation must reproduce its tuning score exactly. The paired effect of the CRF is the seed-matched change from each U-Net run to its refinement.

## Metrics and timing

Confusion rows are ground truth, columns predictions. Pool counts over the evaluated images before computing IoU/Dice/recall. The primary score is the mean IoU of the defined target classes; on the complete release both targets must be supported. Three-class mIoU averages defined class IoUs. Undefined denominators are null. Per-image scores include a defined-target count and are never averaged to form headline scores.

All 12 models, their validation artifacts and the overlap analysis are frozen before held-out testing. Report every result, including weak performance. Training times exclude shared preprocessing/feature extraction; U-Net includes validation/checkpointing. Inference is batch one, after ten warm-up images, on the first 50 sorted split IDs. Report median/P95 latency, use CUDA synchronisation, and include all per-image computation from processed host RGB to CPU prediction (for the hybrid: U-Net forward pass, softmax and a single-threaded CPU CRF). Original decoding is excluded for every method.

## Limits

One event and region, three seeds on one subset, downsampled masks, verified neighbouring-frame overlap, different training procedures, and pretrained external ImageNet knowledge all limit interpretation. These results do not establish physical flooded area, depth, road passability, deployment readiness, adoption, or operational benefit. The hybrid's CRF setting lies at the edge of its small validation grid. Full ten-class segmentation is deferred.

Sources: supplied methodology/proposal PDFs in `pdf/`; [official dataset](https://github.com/BinaLab/FloodNet-Supervised_v1.0); [paper DOI](https://doi.org/10.1109/ACCESS.2021.3090981).
