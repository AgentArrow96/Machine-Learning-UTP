# Audit eval definitions (written before any audit code ran)

Method: ECC `eval-harness` (define evals first, code-based graders, PASS/FAIL, pass^k for release-critical checks) and Superpowers `verification-before-completion` (no claim without fresh evidence). Skill texts used: `C:\Important Files\audit-skills\` (local clones; push disabled; outside every working folder).

Criteria below are copied from the approved plan and are not edited after results exist. Results are recorded in `output/audit/results.json` and summarised in `docs/audit.md`.

**Independence rule.** `audit/independent.py`, `audit/audit_data.py`, `audit/audit_eval.py` and `audit/audit_stats.py` must never import the project package `floodnet` (asserted at the end of each script). Only `audit/audit_retrain.py` imports it, by design, to retrain with the project's own code.

**Repetition.** Deterministic evals (D*, E*, L*, S1) are run twice by `audit/run_audit.py`; a check passes only if both runs pass and agree (pass^2 = 1.00). R* is itself a reproduction and runs once.

| ID | What | PASS criterion |
|---|---|---|
| D1 | Counts and pairing | 1445 / 450 / 448 complete image–mask pairs, matching `DATASET-VERSION-NOTE-v1.0.txt` |
| D2 | Raw hashes | all 4,686 raw-file SHA-256 equal `data/manifests/audit.json`; archive SHA-256 equals `data/downloads/source.json` |
| D3 | Split disjointness | 0 shared IDs and 0 shared image-file hashes across train / val / test |
| D4 | Mask IDs and pixel counts | every mask's IDs ⊆ {0..9}; per-image per-class pixel counts equal the stored counts exactly |
| D5 | Label meaning | indexed ID 1 ↔ palette `building-flooded`, ID 3 ↔ `road-flooded` (via the official colour masks); pixel agreement ≥ 99% on a seeded 150-mask sample |
| D6 | EXIF choice | rotated RGB has higher mask-edge agreement than unrotated for ≥ 55 of the 57 orientation-3 images |
| D7 | Cache fidelity | 0 mismatching pixels between independently derived 256×256 RGB / masks and `data/processed/*.npy` (all 2,343 images) |
| D8 | Budgets | 25% list = 361 IDs, all in train, nested in 100%, 0 val/test IDs; strata counts reported |
| E1 | U-Net, independent evaluation | 6/6 runs: pixel-identical predictions to `predictions_test.npy` and identical pooled confusion matrices |
| E2 | Hybrid, independent evaluation | 6/6 runs identical (as E1) |
| E3 | RF with project features | 6/6 runs identical (as E1) |
| E4 | RF with independently re-implemented features | per run, abs(Δ flooded mIoU) ≤ 1.0 point (drift reported) |
| E5 | Published numbers | results.json = evaluation_test.json = CSVs; `docs/results.md` table values equal results.json within rounding (±0.05 points) |
| L1 | Shuffled-label control | every model's flooded mIoU < 5% against deranged test masks |
| L2 | Trivial baselines | all-Other = 0; all-building / all-road / class-prior random reported; every real model ≥ 10× the best trivial baseline |
| L3 | Untrained controls | random-decoder (ImageNet encoder) and fully random U-Net both < 10% flooded mIoU |
| L4 | Training-set isolation (instrumented) | training IDs == budget list, validation IDs == the 450 val IDs, test IDs seen by training or selection == 0 |
| L5 | Selection used validation only | best epoch == first argmax of val flooded mIoU in all 6 U-Net histories; CRF choice == argmax of an independently recomputed validation grid whose gains equal `runs/hybrid_selection.json` |
| L6 | Chronology | every freeze time < every test-evaluation time (manifest timestamps and file modification times) |
| L7 | No perfect copies | 0 test images with > 1% target pixels where any model's prediction equals the mask on every valid pixel; per-image IoU histogram reported |
| R1 | Retrain all 12 models | per run: `torch.equal` on every tensor of `best.pt` (count of differing tensors and max abs difference reported); RF identical predictions on cached features; PASS if test flooded mIoU within 0.1 point (bitwise reported separately) |
| R2 | Retrained hybrids | 6/6 within 0.1 point of the frozen hybrid scores |
| S1 | Bootstrap | 5,000 paired image resamples; 95% CI and P(Δ > 0) for U-Net − RF, 100% − 25% per method, hybrid − U-Net per budget (reported; no pass threshold) |
| S2 | External plausibility | our 100% U-Net flooded-building and flooded-road IoU are not above the best published FloodNet figure (PSPNet, 713 px; paper Table 4: 68.93 / 82.16) — a flag, not proof |
| V1 | Tests | fresh `pytest`: 0 failed, 0 skipped |
| V2 | Build | `python -m compileall src audit scripts` exits 0; `node --check` on the app scripts exits 0 |
| V3 | Lint (bug rules) | `ruff check --select F,E9` on `src audit scripts`: 0 findings |
| V4 | Types | `pyright src` error count reported |
| V5 | Security | 0 secrets / API keys / absolute user paths in `src/`, `web/data/results.json` |
| V6 | Inventory | `gp/` is untracked: file inventory reported instead of a diff |
| G1 | Artifacts untouched | SHA-256 of every file under `runs/` (excluding `runs/archive/`), `data/processed/` and `data/manifests/` identical before and after the audit |

## Addendum (added after the first run; original criteria above unchanged)

- **D5 measurement fix.** First run: FAIL (pixel agreement 79.4%; raw output kept in `output/audit/data_run1_D5_stretch_method.json`). Root cause (`systematic-debugging`, single-image reproduction): the official colour masks render the **top-left 4000×3000 window** of the indexed mask, so the 352 masks of size 4592×3072 must be cropped, not stretched (image 7486: 0.641 stretched vs 1.000 cropped). The PASS criterion is unchanged; only the comparison method was corrected.
- **D6b (new).** Because of that finding, every one of the 2,343 photos is checked against its own mask and against a different image's mask (seeded derangement). PASS if the own-mask edge agreement is higher for ≥ 99% of images.
