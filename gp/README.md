# FloodNet: infrastructure in focus

The code behind the final report: SLIC + Random Forest and a ResNet-18 U-Net segmenting flooded buildings and flooded roads at **25% and 100% labelled training data**, each trained with **three seeds** (42, 43, 44), plus a verified check for scenes shared between training and test frames, followed by a standalone static results app. All project-owned files stay in this folder.

Both models handle the class imbalance the same way: the forest uses the U-Net's capped inverse-frequency class weights and every training region (`configs/base.json`: `"class_weight": "pixel_clipped"`, no region cap).

## Where to find the models and the results

**Model code** (all in [src/floodnet/](src/floodnet/)):

| Part | File | What it contains |
|---|---|---|
| SLIC + Random Forest features | [features.py](src/floodnet/features.py) | SLIC superpixels, the 22 colour/texture features per region, region labels, the training table |
| Random Forest training | [training.py](src/floodnet/training.py) (`train_rf`, `rf_class_weight`) | Fits the 150-tree forest with the U-Net's class weights |
| U-Net model | [neural.py](src/floodnet/neural.py) (`build_model`, `class_weights`) | ResNet-18 U-Net, capped inverse-frequency class weights, data loading and augmentation |
| U-Net training | [training.py](src/floodnet/training.py) (`train_unet`) | CUDA training with checkpoint selection on validation flooded mIoU |
| Prediction and evaluation | [evaluation.py](src/floodnet/evaluation.py), [metrics.py](src/floodnet/metrics.py) | Both models' inference, the experiment freeze, pooled IoU/Dice/recall |
| Data and preparation | [data.py](src/floodnet/data.py), [overlap.py](src/floodnet/overlap.py) | Audit, 3-class remapping, resizing/padding, label budgets, train–test scene overlap |
| Settings | [configs/base.json](configs/base.json) | Every model and training setting, seeds and budgets |
| Command line | [cli.py](src/floodnet/cli.py), [scripts/run_experiment.py](scripts/run_experiment.py) | `python -m floodnet …` commands and the full pipeline |

**Results:**

- **Interactive app:** https://agentarrow96.github.io/Machine-Learning-UTP/floodnet/ (source in [web/](web/); run it locally as shown below). *Full details* has every result; *Presentation* follows the report's deliverables.
- **The report's numbers:** [web/data/uniform.json](web/data/uniform.json), `summary` (mean, SD and range over seeds 42–44 per model and budget) and `runs` (every run's metrics, confusion matrix and checkpoint hash).
- **First-version forest and dataset facts:** [web/data/results.json](web/data/results.json). The first version used sklearn "balanced" weights and a 100k-region cap and is kept for reference only.
- **CSV downloads:** [web/data/exports/d41ec3fe08e7197b/](web/data/exports/d41ec3fe08e7197b/) has seed summaries, all runs and per-image scores. The U-Net rows match the report; the forest rows in these CSVs are the first version.

Test-set flooded-class mIoU (mean of flooded building and flooded road IoU, mean ± SD over three seeds):

| Model | 25% labels | 100% labels |
|---|---|---|
| SLIC + Random Forest | 27.6% ± 0.1 | 31.8% ± 0.0 |
| ResNet-18 U-Net | 52.3% ± 2.2 | 63.3% ± 2.7 |

Checkpoints, predictions and the dataset are not in Git (tens of gigabytes). Running the pipeline below recreates them.

## Quick start (Windows PowerShell)

Python 3.12 and an NVIDIA driver are prerequisites. This project's environment is independent of the repository-root environment. The setup command installs CUDA PyTorch, the declared dependencies, and pretrained encoder weights **inside this project**; it checks a GPU optimiser step. It does not install a system CUDA toolkit.

### Terminal 1: environment and official data

```powershell
Set-Location -LiteralPath 'C:\Important Files\ML\gp'
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup.ps1
.\.venv\Scripts\python.exe -u -m floodnet download
```

The official archive is approximately 12.2 GiB. It also includes colourised visualisations, which are preserved but never used as integer training annotations. Downloads keep a `.part` file; rerun the command after an interruption. Archive extraction is restartable and checks existing originals against archive CRCs.

### Terminal 1: complete experiment

```powershell
.\.venv\Scripts\python.exe -u .\scripts\run_experiment.py
```

This runs audit → preparation → features → overlap → six RF runs → six CUDA U-Net runs (each with validation) → frozen test evaluations → web export → report figures and numbers. Stage status and console output are saved in `runs/pipeline_status.json` and `runs/pipeline.log`. Existing compatible runs and evaluations are verified (checkpoint hashes) and reused. An interrupted U-Net resumes at its latest completed epoch. On the RTX 5080 laptop the 12 training runs take roughly 15–20 minutes; the first overlap analysis takes about 14 minutes and its keypoints are cached afterwards.

### Terminal 2: results app

```powershell
Set-Location -LiteralPath 'C:\Important Files\ML\gp'
.\.venv\Scripts\python.exe -m http.server 8765 --bind 127.0.0.1 --directory web
```

Open **http://127.0.0.1:8765/**. The sidebar switches between **Full details** and **Presentation** (the report's deliverables, simplified). Keep the server terminal open; stop it with Ctrl+C. After editing anything in `web/assets/`, bump the `?v=` number on the asset links in `web/index.html` so browsers fetch the new files. Opening the HTML with `file://` is unsupported because the app fetches JSON and images.

Optional display exports for the app (they read the frozen runs and raw data without changing them):

```powershell
# Sharper 1024-px test photos for the expanded image viewer (~77 MB, git-ignored in web/data/hires/)
.\.venv\Scripts\python.exe -m floodnet export-hires
# "How the models work" walkthrough (CUDA; ~20 s). Fails unless every final mask equals the frozen test prediction
.\.venv\Scripts\python.exe -m floodnet export-walkthrough
# "Data preprocessing" page (~15 s; re-hashes the downloaded archive)
.\.venv\Scripts\python.exe -m floodnet export-preprocessing
```

In the app, select any photo or prediction panel to expand it (scroll or +/− to zoom, drag to pan, ←/→ to flip between panels at the same view). Charts show values on hover or keyboard focus, have legend toggles, an **Expand** view and default annotations that **Notes on/off** hides.

## Run individual phases

Run these from `gp/`. `--config` is a global option before the command and must refer to a JSON file inside `gp/`.

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

# After all 12 runs have validation results
.\.venv\Scripts\python.exe -u -m floodnet evaluate --run runs/rf_25_s42 --split test --final
.\.venv\Scripts\python.exe -m floodnet export-web
.\.venv\Scripts\python.exe .\scripts\make_report_assets.py
```

Test evaluation requires all 12 completed runs, their validation results and the overlap analysis. The first `--final` call freezes their configuration, checkpoint, validation and overlap hashes in `runs/final_experiment.json`; changed artifacts cannot silently replace the frozen experiment. To start a new experiment, move `runs/final_experiment.json` and the run folders into `runs/archive/` first.

To check the overlap thresholds on one pair: `.\.venv\Scripts\python.exe -m floodnet overlap-check --eval test_6336 --train train_6335`.

**About the committed app data.** The forests in the report were retrained once with the configuration above after a first version (sklearn "balanced" weights and a 100k-region cap) had been evaluated; the U-Nets were not changed. On the original machine the report's forests are kept in `runs/undersample/uniform/` and the first-version forests in `runs/rf_*`. The committed `web/data/uniform.json` holds the report's forests with the U-Nets, and `web/data/results.json` holds the first-version forests, which the app keeps as a reference. A fresh run of this code trains the report's forests directly under `runs/rf_*`. Retraining `rf_25_s42` with this code reproduces the report's forest exactly (0 differing test pixels).

## Resume and diagnose

```powershell
.\.venv\Scripts\python.exe -u -m floodnet train --model unet --budget 25 --resume
.\.venv\Scripts\python.exe -u .\scripts\run_experiment.py --from-prepared
```

`--from-prepared` skips acquisition, audit, preparation and features (overlap still runs and is reused when cached). RF fitting restarts if interrupted. U-Net resumes from the latest completed epoch with model, optimiser, precision scaler, random state and epoch-defined augmentation/shuffle. A smoke run uses `train --model unet --budget 25 --smoke`; it writes under `runs/smoke/`, uses eight training and four validation images for one epoch, and can never enter the final export.

- **No CUDA:** use this folder's `.venv` and rerun setup/doctor. CPU fallback is deliberately disabled for U-Net.
- **Out of memory:** a real-data preflight tries batch 4 then 2 and records one selection shared across budgets.
- **Release mismatch / invalid masks:** inspect the audit error and originals. The pipeline will not invent partitions or convert unexplained colour masks into labels.
- **Incompatible resume:** preserve existing runs; restore their exact source, configuration and manifests, or start a separately preserved experiment.
- **Network restriction:** acquisition and setup need the official Dropbox, PyTorch and Hugging Face sources. Once cached, training and the app work offline.

## Validation

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp .tmp\pytest-manual
node --check .\web\assets\js\app.js
```

Use a fresh `--basetemp` name if Windows holds old files open. Tests cover labels, geometry, padding, hand-calculated metrics, subsets, cache identity, safe archive paths, augmentation/order reproducibility, CPU-fallback rejection, the forest's class weights and region sampling, overlap verification on a shifted crop, the overlap-strata split, the 12-run freeze gate, single-threaded RF inference, seed summaries, a synthetic-fixture pipeline and export validation. Fixture scores are test assertions, never project results.

Browser QA uses the official Playwright CLI and an isolated headless Edge profile. Create an empty `.playwright` folder in `gp/` first (it is git-ignored), start the HTTP server, then in a separate terminal:

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
| `web/data/results.json`, `uniform.json` | The app's data (see above) |
| `web/data/exports/<identity>/` | CSVs, manifests and display images |
| `report/numbers.json`, `report/figures/` | Numbers and figures for the report (`scripts/make_report_assets.py`) |

Generated data, environments and runs are ignored by Git. The app's exported data in `web/data/` is committed, so a checkout can serve the app directly: copy the whole `web/` folder and serve it over HTTP. It needs neither the model libraries nor the checkpoints.

The protocol is in [docs/experiment_protocol.md](docs/experiment_protocol.md), the architecture in [docs/architecture.md](docs/architecture.md) and the export contract in [docs/web_export.md](docs/web_export.md).
