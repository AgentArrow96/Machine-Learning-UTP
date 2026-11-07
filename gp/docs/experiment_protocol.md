# Experiment protocol

The protocol of the final report: SLIC + Random Forest and a ResNet-18 U-Net, two label budgets, three seeds. An earlier single-seed version of the experiment is preserved under `runs/archive/v1_f341bd24/` on the original machine.

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

The forest uses 150 trees, maximum depth 20, the training seed, and up to eight CPU workers for fitting. **Class imbalance:** the forest is weighted exactly like the U-Net, with the capped inverse-frequency pixel weights of the same training images (`class_weight: "pixel_clipped"`, see the neural pipeline), and it is fitted on **every** training region (`max_regions` 10⁹, so the `keep_targets` sampler keeps all regions). Region IDs are saved; all three classes must be present.

A first version used sklearn's uncapped "balanced" weights and a 100k-region cap that kept every flooded region but only about a third of the Other regions at 100%. It was evaluated, then replaced by the rule above so that both models handle the imbalance the same way; only the six forests were retrained, and the first version is kept and reported as a reference. Inference predicts one image at a time with a single-threaded forest.

## Neural pipeline

ImageNet ResNet-18 encoder; U-Net decoder channels 128/64/32/16/8; three logits. ImageNet mean/std normalisation. AdamW LR=1e-4, weight decay=1e-4, batch=4. A real-data backward preflight may choose batch=2 for all runs. BF16 preferred, FP16/scaler otherwise. No CPU training fallback.

For training-subset pixel proportions p, form inverse-frequency weights 1/p, normalise to arithmetic mean one, then clip to [0.25, 4]. Weighted cross-entropy ignores padding. Training/validation loss curves accumulate the weighted-loss denominator rather than averaging batch means.

Training uses matched horizontal/vertical flips, random quarter-turns, and image-only contrast/brightness magnitude 0.1. Validation/test transforms are deterministic. Maximum 30 epochs; stop after six consecutive non-improving validation flooded-mIoU epochs. A strictly greater score replaces the best checkpoint; ties keep the earlier one. No hyperparameter search.

## Metrics and timing

Confusion rows are ground truth, columns predictions. Pool counts over the evaluated images before computing IoU/Dice/recall. The primary score is the mean IoU of the defined target classes; on the complete release both targets must be supported. Three-class mIoU averages defined class IoUs. Undefined denominators are null. Per-image scores include a defined-target count and are never averaged to form headline scores.

All 12 models, their validation artifacts and the overlap analysis are frozen before held-out testing. Report every result, including weak performance. Training times exclude shared preprocessing/feature extraction; U-Net includes validation/checkpointing. Inference is batch one, after ten warm-up images, on the first 50 sorted split IDs. Report median/P95 latency, use CUDA synchronisation, and include all per-image computation from processed host RGB to CPU prediction. Original decoding is excluded for every method.

## Limits

One event and region, three seeds on one subset, downsampled masks, verified neighbouring-frame overlap, different training procedures, and pretrained external ImageNet knowledge all limit interpretation. These results do not establish physical flooded area, depth, road passability, deployment readiness, adoption, or operational benefit. Full ten-class segmentation is deferred.

Sources: supplied methodology/proposal PDFs in `pdf/`; [official dataset](https://github.com/BinaLab/FloodNet-Supervised_v1.0); [paper DOI](https://doi.org/10.1109/ACCESS.2021.3090981).
