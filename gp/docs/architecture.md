# Architecture and phase dependencies

The training workspace owns acquisition, preprocessing, models and evaluations. The web app owns presentation. Their only runtime boundary is a versioned JSON/image export; the app never imports a checkpoint or starts a model server.

1. **Foundation:** fixed configuration, class order, paths and provenance utilities.
2. **Environment:** isolated CUDA-enabled Python; cached pretrained encoder; synthetic-tensor hardware check.
3. **Acquisition:** authors' README → official Dropbox ZIP → hashed archive → CRC-checked extraction.
4. **Audit:** identify canonical indexed annotations; pairing, geometry, labels, hashes, exact duplicates.
5. **Preparation:** original files → aspect-preserving 256 input → per-split RGB/mask memmaps; fixed budget IDs (subset seed).
6. **Features:** deterministic SLIC + colour/texture descriptors. Features contain no fitted validation/test information.
7. **Overlap:** candidate train frames (frame IDs + ImageNet embeddings) → cached ORB keypoints → RANSAC homography → verified overlap per validation/test image; feeds the overlap-stratified sensitivity check.
8. **Training:** RF and U-Net per budget and per training seed (12 runs). Data selection is shared; learning procedures differ.
9. **Evaluation:** common pixel-confusion implementation; complete official validation/test splits; explicit experiment freeze covering all 12 runs and the overlap analysis.
10. **Export:** seed summaries, overlap-strata and clean-subset scores, and display assets; validate, publish the manifest last, prune stale exports.
11. **Presentation:** static, responsive app with a Full details and a Presentation mode; missing results remain missing.

## Memory and concurrency

Original images are decoded in bounded preparation batches (four workers). Epochs use already-processed data. RGB/mask memmaps store uint8 values; the shared cache is about 586 MiB. Normalised float tensors exist only for loaded batches.

Windows data-loader workers reopen memmaps after spawn instead of receiving pickled arrays. Two persistent workers with prefetch factor two and pinned host buffers feed the GPU. Worker processes never create CUDA tensors. Entry points use main guards. U-Net runs execute sequentially.

SLIC feature rows are small (22 float32 values); the forest is fitted on every training region with at most eight CPU workers. Inference recomputes SLIC/features and predicts with one thread per image. The overlap stage decodes originals with JPEG draft scaling and runs keypoint extraction and pair verification in a pool of processes; keypoints are cached under `data/features/overlap/`.

## Identity and recovery

The audit identity derives from original file hashes and annotations. Preprocessing identity includes that audit plus geometry/remapping version. Feature identity adds SLIC settings and feature names. Overlap identity adds its settings and algorithm version. Run identity includes configuration, budget, training seed, data/features and source hashes.

Preparation flushes disk arrays before marking image records complete. Changed sources/settings select a different cache. Interrupted RF fitting restarts; U-Net resumes at a completed epoch. A completed run is reused when everything except the source-code hash matches and its checkpoint hash verifies, so later code changes (for example to the export) never force retraining. Its shuffle and augmentation depend on seed/epoch/image, so resuming does not depend on worker lifetime. The final experiment rejects changed checkpoint/validation artifacts; starting a new experiment requires archiving the frozen one first (as v1 was).

## Boundaries

All project-owned paths are resolved and constrained beneath `gp/`. Original PDFs and raw dataset files are preserved. Colourised dataset visualisations are retained but excluded from indexed-mask discovery. No backend, account system or live inference is introduced.
