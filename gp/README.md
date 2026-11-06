# FloodNet: infrastructure in focus

Reproducible SLIC + Random Forest and ResNet-18 U-Net experiments at **25% and 100% labelled training data**, each trained with **three seeds** (42, 43, 44), the **hybrid U-Net + DenseCRF** refinement of every U-Net run, plus a verified check for scenes shared between training and test frames, followed by a standalone static results explorer. All project-owned files stay in this folder. The explorer follows the design of `../app/` but does not load anything from it.

## Quick start — Windows PowerShell

Python 3.12, an NVIDIA driver and the Microsoft C++ Build Tools are prerequisites (the hybrid's `pydensecrf2` has no Windows wheels and is compiled during setup). This project's environment is independent of the repository-root environment. The setup command installs CUDA PyTorch, the declared dependencies, and pretrained encoder weights **inside this project**; it checks a GPU optimiser step. It does not install a system CUDA toolkit.

**Terminal 1 — environment and official data acquisition**

```powershell
Set-Location -LiteralPath 'C:\Important Files\ML\gp'
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup.ps1
.\.venv\Scripts\python.exe -u -m floodnet download
```

The official archive is approximately 12.2 GiB. It also includes colourised visualisations, which are preserved but never used as integer training annotations. Downloads retain a `.part` file. Rerun the download command after interruption; a server that does not support resuming may require a restart. Archive extraction is restartable and checks existing originals against archive CRCs.

**Terminal 1 — complete experiment**

```powershell
Set-Location -LiteralPath 'C:\Important Files\ML\gp'
.\.venv\Scripts\python.exe -u .\scripts\run_experiment.py
```

This runs audit → preparation → features → overlap → six RF runs → six CUDA U-Net runs (each with validation) → frozen test evaluations → CRF validation tuning → six hybrid runs (validation, then their own frozen test evaluation) → web export → report figures. Completed runs are reused (checkpoint hashes verified) even when the source code changed later. Stage status and console output are saved in `runs/pipeline_status.json` and `runs/pipeline.log`. Existing compatible trained runs/evaluations are verified and reused. An interrupted U-Net resumes at its latest completed epoch. On the RTX 5080 laptop the 12 training runs take roughly 15–20 minutes; the first overlap analysis takes about 14 minutes and its keypoints are cached afterwards.

**Terminal 2 — standalone results app (HTTP server process)**

```powershell
Set-Location -LiteralPath 'C:\Important Files\ML\gp'
.\.venv\Scripts\python.exe -m http.server 8765 --bind 127.0.0.1 --directory web
```

Open **http://127.0.0.1:8765/**. Keep the HTTP server terminal open; stop it with Ctrl+C. Model training does not need to remain active to view completed exports. Reload the page after re-exporting results. After editing anything in `web/assets/`, bump the `?v=` number on the asset links in `web/index.html` so browsers fetch the new files. Opening the HTML with `file://` is unsupported because the app fetches JSON and images.

Optional display exports add to the app (all read the frozen runs and raw data without changing them):

```powershell
# Sharper 1024-px test photos for the expanded image viewer (~77 MB, git-ignored in web/data/hires/; without them the viewer uses the 256-px images)
.\.venv\Scripts\python.exe -m floodnet export-hires
# "How the models work" walkthrough (CUDA; ~20 s). Fails unless every final mask equals the frozen test prediction
.\.venv\Scripts\python.exe -m floodnet export-walkthrough
# "Data preprocessing" page (~15 s; re-hashes the downloaded archive)
.\.venv\Scripts\python.exe -m floodnet export-preprocessing
```

**Class-imbalance study** (pre-registered in `configs/imbalance.json` and `docs/imbalance.md`; evals in `audit/imbalance_evals.md`, task ledger in `audit/imbalance_ledger.md`). It adds 114 runs in `runs/imbalance/` (oversampling, ChessMix, R-GAN, OASIS, SPADE, Dice + CE, SMOTE, tabular R-GAN) with their own freeze, and never changes the 18 original runs. Generators and synthetic data live in the git-ignored `data/synthetic/`. About 9–10 h on the RTX 5080; every stage resumes:

```powershell
.\.venv\Scripts\python.exe .\scripts\run_imbalance.py            # all stages, ends with the web export
.\.venv\Scripts\python.exe .\audit\audit_imbalance.py            # independent checks (run twice for pass^2)
.\.venv\Scripts\python.exe .\audit\audit_imbalance_repro.py      # bitwise retraining and regeneration checks
```

In the app, select any photo or prediction panel to expand it (scroll or +/− to zoom, drag to pan, ←/→ to flip between panels at the same view). Charts show values on hover or keyboard focus, have legend toggles, an **Expand** view and default annotations that **Notes on/off** hides.

## Run individual phases

Run these from `gp/`. The installed module resolves its data/output paths relative to the package's project root, including when called from a different working directory. `--config` is a global option before the command and must refer to a JSON file inside `gp/`.

```powershell
.\.venv\Scripts\python.exe -m floodnet doctor --gpu-smoke
.\.venv\Scripts\python.exe -u -m floodnet audit
.\.venv\Scripts\python.exe -u -m floodnet prepare
.\.venv\Scripts\python.exe -u -m floodnet features
.\.venv\Scripts\python.exe -u -m floodnet overlap

# One run per model, budget and seed (seeds come from configs/base.json: 42, 43, 44)
.\.venv\Scripts\python.exe -u -m floodnet train --model rf --budget 25 --seed 42
.\.venv\Scripts\python.exe -u -m floodnet train --model unet --budget 100 --seed 43
.\.venv\Scripts\python.exe -u -m floodnet evaluate --run runs/unet_100_s43 --split val

# Hybrid U-Net + DenseCRF (no training): validation-only tuning, then one record per U-Net run
.\.venv\Scripts\python.exe -u -m floodnet tune-crf
.\.venv\Scripts\python.exe -u -m floodnet refine --budget 100 --seed 42
.\.venv\Scripts\python.exe -u -m floodnet evaluate --run runs/hybrid_100_s42 --split val

# After all runs have validation results
.\.venv\Scripts\python.exe -u -m floodnet evaluate --run runs/rf_25_s42 --split test --final
.\.venv\Scripts\python.exe -m floodnet export-web
.\.venv\Scripts\python.exe -m floodnet export-walkthrough
.\.venv\Scripts\python.exe .\scripts\make_figures.py
```

`scripts/run_experiment.py` issues every train/evaluate command in order. Test evaluation requires all 12 completed final runs, their validation results and the overlap analysis; hybrid test evaluation additionally requires the CRF selection and all six hybrid runs with validation results, which are frozen in `runs/hybrid_experiment.json`. The first `--final` call freezes their configuration, checkpoint, validation and overlap hashes. Changed final artifacts cannot silently replace the frozen experiment: to start a new experiment, move `runs/final_experiment.json` and the run folders into `runs/archive/` first (the v1 single-seed experiment is kept in `runs/archive/v1_f341bd24/`).

To check the overlap thresholds on one pair: `.\.venv\Scripts\python.exe -m floodnet overlap-check --eval test_6336 --train train_6335`.

## Resume and diagnose

```powershell
.\.venv\Scripts\python.exe -u -m floodnet train --model unet --budget 25 --resume
.\.venv\Scripts\python.exe -u .\scripts\run_experiment.py --from-prepared
```

`--from-prepared` skips acquisition/audit/preparation/features (overlap still runs and is reused when cached); each training stage still validates its active manifests. RF fitting restarts if interrupted. U-Net resumes from the latest completed epoch with model, optimiser, precision scaler, random state and epoch-defined augmentation/shuffle. A smoke run uses `train --model unet --budget 25 --smoke` (any `--seed`); it writes under `runs/smoke/`, uses eight training/four validation images for one epoch, and can never enter the final export.

- **No CUDA:** use this folder's `.venv` and rerun setup/doctor. CPU fallback is deliberately disabled for U-Net.
- **Out of memory:** real-data preflight tries batch 4 then 2 and records one selection shared across budgets. Free GPU memory if both fail; resolution is not silently changed.
- **Release mismatch / invalid masks:** inspect the audit error and originals. The pipeline will not invent partitions or convert unexplained colour masks into labels.
- **Incompatible resume:** preserve existing runs; restore their exact source/configuration/manifests or start a separately preserved experiment. Do not overwrite provenance to bypass the check.
- **Changed originals:** re-audit and prepare. Source/configuration changes invalidate matching caches and prevent incompatible checkpoint resume.
- **Network restriction:** acquisition/setup need access to the official Dropbox/PyTorch/Hugging Face sources. Once data, weights and dependencies are cached, training and the app work locally.

## Validation

```powershell
Set-Location -LiteralPath 'C:\Important Files\ML\gp'
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp .tmp\pytest-manual
node --check .\web\assets\app.js
```

Use a fresh `--basetemp` name for independent runs if Windows holds old files open. Tests cover labels, geometry, padding, hand-calculated metrics, subsets, cache identity, safe archive paths, augmentation/order reproducibility, CPU-fallback rejection, target-preserving RF sampling, overlap verification on a shifted crop, subset pooling, the overlap-strata split, the 12-run freeze gate, single-threaded RF inference, seed summaries, a synthetic-fixture RF pipeline and export validation. Fixture scores are test assertions, never project results.

## Output map

| Artifact | Purpose |
|---|---|
| `data/downloads/source.json` | Official source, licence and archive hash |
| `data/manifests/audit.json` | Original pairing, labels, distributions and overlap candidates |
| `data/manifests/processed.json` | Cache geometry, processed statistics and nested budgets |
| `data/manifests/budget_25.json`, `budget_100.json` | Exact training image IDs |
| `data/manifests/overlap.json` | Candidate/verified train–evaluation scene pairs and per-image training overlap |
| `runs/doctor.json` | Hardware/software and CUDA check |
| `runs/<method>_<budget>_s<seed>/run.json` | Training settings, provenance, status and compute logs |
| `runs/<run>/history.json` | U-Net learning curves |
| `runs/<run>/evaluation_test.json` | Aggregate/per-image held-out metrics and latency |
| `runs/<run>/predictions_test.npy` | Raw processed-resolution class predictions |
| `runs/final_experiment.json` | Frozen 12-run + overlap identity |
| `runs/hybrid_selection.json` | CRF validation grid, per-run gains and the selected setting |
| `runs/hybrid_experiment.json` | Frozen hybrid runs + CRF selection (references the 12-run freeze) |
| `configs/hybrid.json` | CRF settings and validation grid (kept separate so the trained runs' configuration hash never changes) |
| `runs/archive/v1_f341bd24/` | Preserved v1 (single-seed, uniform RF sampling) experiment |
| `web/data/results.json` | Portable, versioned app interface |
| `web/data/exports/<identity>/` | CSVs, manifests and display images |

Generated data/environments/runs are ignored by Git. The app's exported data in `web/data/` is **not** ignored, so a checkout can serve the explorer directly. To share the finished explorer, copy the **whole `web/` folder including `web/data/`** and serve it over HTTP. It requires neither Python model libraries nor access to checkpoints. Future integration can link to this folder using relative URLs.

Standalone scientific charts are in `output/figures/` (PNG and SVG with a source/hash manifest). Rebuild them after a complete export with `.\.venv\Scripts\python.exe .\scripts\make_figures.py`; future complete pipeline runs also generate them. Matched qualitative report PNGs are listed in the exported `figures.json`, with six scenes at each budget.

Browser QA uses the official Playwright CLI and an isolated headless Edge profile. Its runner is `node scripts/browser_qa.cjs` after opening the documented `floodnet` CLI session and starting the HTTP server. Create an empty `.playwright` folder in `gp/` first (it is git-ignored): it marks the Playwright workspace so the `npx` client and the runner find the same session. Open the session and run the checks from the same PowerShell window. Current screenshots and checks are in `output/playwright/`. The final acceptance test report is `output/pytest.xml`; `scripts/verify_delivery.py` verifies those reports, all 1,792 exported (seed-42) prediction PNGs against raw arrays, pooled, clean-subset and overlap-strata metrics for all 12 runs, seed summaries, RF region tables, CSVs, checkpoint hashes, the absence of sklearn parallel warnings and the scientific figures.

With the HTTP server running, use a separate terminal for browser QA:

```powershell
Set-Location -LiteralPath 'C:\Important Files\ML\gp'
$env:npm_config_cache = Join-Path (Get-Location) '.cache\npm'
$env:PLAYWRIGHT_BROWSERS_PATH = Join-Path (Get-Location) '.cache\ms-playwright'
$env:PLAYWRIGHT_MCP_OUTPUT_DIR = Join-Path (Get-Location) 'output\playwright'
$env:TEMP = Join-Path (Get-Location) '.tmp'
$env:TMP = $env:TEMP
npx.cmd --yes --package @playwright/cli@0.1.22 playwright-cli -s=floodnet open http://127.0.0.1:8765/ --browser msedge --profile .cache/browser-profile
node .\scripts\browser_qa.cjs
npx.cmd --offline --yes --package @playwright/cli@0.1.22 playwright-cli -s=floodnet close
```

To regenerate acceptance evidence, run pytest with `--junitxml output/pytest.xml`, run the browser checks against the complete export, then run `.\.venv\Scripts\python.exe .\scripts\verify_delivery.py`.

The full protocol is in [docs/experiment_protocol.md](docs/experiment_protocol.md), the architecture in [docs/architecture.md](docs/architecture.md), the export contract in [docs/web_export.md](docs/web_export.md), and measured results in [docs/results.md](docs/results.md).
