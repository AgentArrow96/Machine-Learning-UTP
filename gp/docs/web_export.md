# Web export contract — schema version 3

`web/data/results.json` is UTF-8 JSON with finite numeric values or explicit nulls. Top-level fields: `schema_version`, `exported_at`, `status`, `title`, `protocol`, `classes`, `dataset`, `runs`, `summary`, `budget_effects`, `explorer_runs`, `samples`, `representatives`, `limitations`, `reproducibility`, `sources`, `missing_runs`, `downloads`. A complete export also includes `experiment_identity` (12 trained runs and the overlap analysis).

- `status`: `complete` only when every configured run (2 methods × 2 budgets × each training seed, 12 in total) has a frozen test evaluation and the overlap analysis exists. Otherwise `incomplete`, with `message`.
- `protocol`: subset seed, training seeds, the seed shown in the explorer, input size, budgets, expected run count and RF region sampling/cap.
- `classes`: ordered IDs 0/1/2, names and shared colour palette.
- `dataset`: original/processed distributions, counts, budget counts, identities, source/licence, EXIF policy and `overlap`. `overlap` holds the method, thresholds, per-split summary, `target_images` (how many target-containing test images overlap training, the median split and clean-subset target counts), a histogram of each test image's largest verified training overlap, and up to three example pairs with display images. Null before preparation.
- `runs`: one entry per run: method (`rf`/`unet`), budget, seed, official-split `metrics`, `clean_metrics` (test images without verified training overlap; few targets, so not a leakage estimate) and `clean_images`, `overlap_strata` (flooded mIoU and target IoUs on the lower- and higher-overlap halves of the target-containing test images), checkpoint hash, identity, training history, timing and device. Configuration and versions appear once in `reproducibility`.
- `metrics`: pooled confusion matrix, valid-pixel count, primary `flooded_miou`, three-class `miou`, defined-target count, and ordered per-class IoU/Dice/recall/support. Scores are fractions 0–1; null means undefined.
- `summary`: per method and budget, `{mean, sd, min, max, values}` across seeds for flooded mIoU, mIoU and target-class IoU/Dice/recall, for `full` and `clean`, `overlap_low`/`overlap_high` stats for the strata, plus training time, inference median and U-Net epochs.
- `budget_effects`: per method, seed-matched 100%−25% differences in flooded mIoU for `full` and `clean`.
- `explorer_runs`: run IDs (the explorer seed) whose prediction PNGs are exported.
- `samples`: test image ID, RGB/ground-truth URLs, valid geometry, target presence, target pixel fraction, largest training overlap and overlapping training IDs, plus prediction URLs and slim per-image scores for the explorer runs.
- `representatives`: up to six sample IDs chosen by target-presence strata and coverage quartiles, never by model quality.
- `downloads`: relative links to the seed summary, all-runs and per-image CSVs, verified overlap pairs, budget IDs and the freeze manifest.

Display RGB is WebP. Mask PNGs store class IDs as greyscale values 0/1/2; a canvas maps them to the legend. All assets are cropped to the saved valid rectangle. Display transformations do not change the evaluated metrics.

Every asset URL is relative to the web root; absolute paths, parent traversal, checkpoints and Python serialisations are excluded. Assets are written into an identity-specific directory and validated, the top-level manifest is replaced atomically, and other export directories are then removed. Incomplete exports are allowed and labelled.

`web/data/uniform.json` has the same `runs`, `summary`, `budget_effects`, `explorer_runs` and `samples` shape for the report's forests (retrained with the U-Net's class weights and every region) next to the unchanged U-Nets, plus `comparisons` against the first-version forests in `results.json` and its own `experiment_identity`. The app shows `uniform.json` and keeps the first-version forests as a reference. `scripts/make_report_assets.py` reads both and writes `web/data/presentation.json` and `web/data/presentation/` for the Presentation mode.

The site has two pages: `web/index.html`, a static story page whose numbers are checked against `uniform.json` and `results.json` by `scripts/browser_checks.js`, and `web/app.html`, the dashboard that reads these exports. Old dashboard links such as `index.html#results` are forwarded to `app.html#results`. The story page's photos (`web/data/landing/`) are made from the original test images by `scripts/make_landing_images.py`.

`web/data/` is committed so the app can be served from any checkout. To share it, copy the whole `web/` folder and serve it over HTTP; no training environment is needed. Linking from the lecture app later requires only a relative link (and staging `gp/web` in the Pages workflow).

## Walkthrough and sharp photos

`web/data/walkthrough.json` (schema version 1, committed) and `web/data/walkthrough/<identity>/` are written by `floodnet export-walkthrough` for the six `representatives`, at both budgets, using the explorer seed. Per example: the original photo and 10-class annotation (official palette), the padded 256-px input, the remapped truth, SLIC outlines, region mean colours, LBP codes, an RGB-encoded region map (index + 1 in R + 256·G, 0 = padding) with all 22 features and truth-majority labels per region; per budget: forest probabilities and per-tree votes per region, U-Net encoder/decoder PCA images, packed softmax probabilities (R/G/B = other/building/road × 255), and the U-Net mask, each with image scores and a `check` showing the final mask equals the frozen test prediction (`differing` = 0). The export also reruns preparation from the raw pair and fails unless it reproduces the cached arrays. It lives outside `data/exports/` and is not pruned by `export-web`; rerun it after a new export.

`web/data/hires/` (git-ignored) holds `floodnet export-hires` output: hash-checked, EXIF-oriented 1024-px WebP photos of every test image and of the training frames in the overlap examples, plus `index.json`. The app uses them only in the expanded viewer and falls back to the 256-px images when the folder is absent.

## Preprocessing page

`web/data/preprocessing.json` (schema 1, committed) + `web/data/preprocessing/<identity>/` are written by `floodnet export-preprocessing`: per step (`acquire`, `pairing`, `duplicates`, `orientation`, `resize`, `remap`, `resolution`, `cache`, `budgets`, `augmentation`, `normalisation`) the numbers from `audit.json`, `processed.json` and `source.json` (the archive is re-hashed during the export), plus deterministic example images.
