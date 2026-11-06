# Task ledger: preprocessing page + class-imbalance study

Method: superpowers `executing-plans` (task loop, completion contract: named tests ran and their output was read
in-session; every deviation gets a `Ruling:`), superpowers `test-driven-development` (red → green), ECC
`verification-loop` per part, ECC `mle-reviewer` review at the end of each part. Skill files:
`C:\Important Files\audit-skills\superpowers\skills\*` and `...\ECC\skills\*`, `...\ECC\agents\mle-reviewer.md`.

| # | Task | Tests / evals | Status | Evidence |
|---|---|---|---|---|
| 0 | Checkpoint the app: branch `floodnet-results-app`, stage only `gp/web` (3,911 files, no hires), push | staged-path checks | done | commit 86ddae2 pushed; `git ls-remote` shows it; main still 9352800 |
| A1 | Preprocessing export (`floodnet export-preprocessing`) + page; red→green tests `tests/test_preprocessing.py` | P1 (5 tests) | done | red: ImportError; green: 5 passed; export re-hashed the 13 GB archive (match); 27 assets |
| B0 | Pre-registration files written first: `configs/imbalance.json`, `docs/imbalance.md` (protocol + review, arXiv IDs verified via the arXiv API), `audit/imbalance_evals.md` | I1 | done | files exist before any GAN code ran |
| B1 | Red→green: `tests/test_imbalance.py` (17 tests: identities, training_table regression, layouts, ChessMix, oversampling, SMOTE, MixedStore, Dice+CE, relativistic losses, R-GAN/OASIS/SPADE components and shapes, KID, MMD, freeze guard) | I3, I5, V | done | red: 15 failed (ModuleNotFoundError); green: full suite 62 passed |
| B2 | GAN speed smoke test (random tensors, nothing saved) before pre-registration | – | done | R-GAN 12.6 it/s (1.8 GB); SPADE 4.4 it/s (4.5 GB); OASIS 1.76 it/s (13.5 GB) |

Ruling (B2): equal iteration budgets for every GAN set to 20,000 (100%) / 10,000 (25%) instead of 30,000 / 15,000, and the OASIS discriminator width set to 32 (10.1 GB, 2.35 it/s), because the planned budget would have taken ≈ 11 h for the GANs alone with OASIS near the 16 GB limit. Decided from the speed smoke test only, before pre-registration and before any real GAN iteration; applies equally to all three families. 20k iterations ≈ 110 epochs at 100% and ≈ 220 epochs at 25% (batch 8).

Observation (systematic-debugging, no change made): R-GAN 25% validation KID rose monotonically 0.271 → 0.319 → 0.402 → 0.432 → 0.491 (iterations 2k → 10k) while training L1/similarity kept falling (0.119 → 0.085 by 4k): the generator overfits its 361 images. The pre-registered rule selected iteration 2,000 — the *first* checkpoint, i.e. the edge of the evaluated range, so an earlier optimum cannot be excluded. Disclosed as a limitation; settings are not changed mid-study.

Failure + fix (systematic-debugging): stage `set --variant rgan --budget 25` stopped with "Layout editing failed to add a target inside the valid area". Reproduced on the real budget: 17 of 643 target components in the 25% budget span the whole frame (171×256); when every component drawn for a layout painted nothing (only non-target, non-padding pixels may be painted), the code raised instead of trying again. Hypothesis: a paste slot gives up after one unpaintable component. One change: a paste slot now tries up to 5 components, and a base that takes no paste is redrawn (≤ 50 times). Regression test `test_layouts_survive_bases_that_are_already_full_of_targets` failed before the fix and passes after (suite: 63 passed). Real dry-run: 25% → 180 layouts, 100% → 722 layouts, 0 bases rejected, every layout gains target pixels (median ≈ 700). Ruling: implementation bug against the pre-registered protocol ("1–2 components pasted"), fixed before any layout was used; GAN training (which uses only real layouts) is unaffected; no setting changed.

Failure + fix (systematic-debugging): OASIS 25% diverged between iterations 1,500 and 2,000 (d = 3.4e8, LabelMix = 3.4e16; validation KID 3.75, worse than a flat colour painting at 1.50). The losses stayed finite, so the non-finite guard missed it. Reproduced on real data with instrumentation (diverged at 1,200). First hypothesis (unnormalised 1x1 head) rejected: head weight norm stayed 2.0-2.3. Trace: every spectrally normalised weight had sigma = 1.00 (spectral norm worked), but block outputs grew 3-11x per block through the 12 residual blocks (2.3 -> 3,524 at the bottleneck -> 2.3e8 at the output); flattened-matrix spectral norm does not bound a 3x3 convolution's gain, and residual sums compound it. Changes: (1) GroupNorm pre-normalisation inside the OASIS discriminator's residual blocks (first block keeps raw RGB, which is bounded) - a documented deviation from the paper's spectral-norm-only discriminator; (2) every GAN now stops on any loss above 1e4, not only on non-finite values. Regression tests (red then green): `test_oasis_discriminator_blocks_do_not_amplify_their_input_scale`, `test_gan_training_fails_closed_on_exploding_losses`; suite 65 passed. Verification: the fixed OASIS trained 3,000 iterations on the real 25% budget with logits 10-13 and d 1.4-2.2. The diverged run is preserved in data/synthetic/gans_archive/oasis_25_diverged_unnormalised; R-GAN and SPADE are unaffected (their losses never exceeded 1e4).

Speed-up (approved): GPU-free stages (SMOTE rows; RF runs of the R-GAN, SPADE and SMOTE variants) run in a second worker (`run_imbalance.py --group cpu`, own status/log files) beside OASIS. Finished synthetic sets are never rebuilt (`stage_set`; test `test_finished_sets_are_not_rebuilt`, red then green). OASIS's own training is unchanged (fp32/TF32, equal 20k/10k iterations).

Failure + fix: OASIS 25% stopped at 07:40 with "CUDA error: out of memory" while the CPU worker built the SMOTE 100% rows. Root cause: `smote` computed an n x n x 22 float64 distance array at once (about 9,000 real road rows -> ~14 GB host RAM); on Windows the GPU driver also relies on system memory, so OASIS's allocation failed. One change: distances in chunks of 256 rows (same per-row arithmetic). Regression test `test_smote_neighbours_are_chunked_but_identical` (red then green; suite 67 passed). Both stored SMOTE sets (25%: 14,626 rows, 100%: 53,638 rows) are reproduced byte-for-byte by the chunked code, so nothing built on them changes. OASIS 25% resumes from its saved iteration-2,000 state (resumed GAN training is not bitwise identical to an uninterrupted run; selection is still by validation KID).

App (interim): `floodnet imbalance export` now works mid-study. Missing pieces are exported as null and shown as "Pending" (no value is ever filled in); the test section is replaced by a clearly labelled validation preview until the freeze. Verified without a browser (GPU busy): every page section rendered in Node against the real interim export for both budgets (no undefined/NaN; 16/16 assets present); suite 67 passed; `node --check` on all scripts.

Pause (user request, GPU needed elsewhere): pipeline stopped during OASIS 100% after checkpoint 18,000; GPU released (183 MiB, 0%). Resume with `scripts/run_imbalance.py`; OASIS 100% continues from its last saved state and finished stages are skipped.

Reproduction check (I8) - first run FAILED twice, both root-caused (systematic-debugging):
1. `rgan/rf_25_s42`: "identical_predictions: false" although training rows matched. All 150 trees were identical (features, thresholds, values); with n_jobs=8 scikit-learn sums tree probabilities in thread order, so even one model predicted twice differs by ~4e-16 (n_jobs=1: exactly equal). Grader bug: the check now compares the trees and predicts single-threaded.
2. `rgan_100` regeneration not byte-identical: in a fresh process regeneration matched the stored set exactly; the repro script had retrained a U-Net first, which sets cudnn.deterministic=True globally, switching the R-GAN's 8 transposed convolutions to another algorithm (+-1 grey level on 0.46% of values). Fix: `generate()` pins the cuDNN mode the sets were rendered with. Regression test `test_generation_does_not_depend_on_leftover_cudnn_state` (red then green).
Re-run: PASS (U-Net 182/182 tensors identical; RF identical trees + predictions on 84,643 test regions; R-GAN, OASIS, SPADE regenerate byte-identical).

| # | Task | Evals | Status | Evidence |
|---|---|---|---|---|
| B3 | 114 runs trained + validated, frozen, tested once | I6 | done | frozen 18:37:04 UTC, last validation 18:36:48, first test 18:37:35 |
| B4 | Independent audit, twice | I1-I4, I6, I7, I9, I10 | PASS (pass^2) | `output/audit/imbalance_1.json`, `imbalance_2.json`: 0 differing pixels over 114 runs; identical per-image confusion hashes across passes; 0/38 bootstrap mismatches; 140 frozen files unchanged |
| B5 | Reproduction | I8 | PASS (after grader fix) | `output/audit/imbalance_repro.json` |
| B6 | Delivery + browser | I11 | PASS | `verify_delivery.py` passed; browser QA 76 checks, 0 console errors; screenshots reviewed (1440 / 390 px) |

VERIFICATION REPORT (ECC verification-loop)
- Tests: 68 passed, 0 failed, 0 skipped.
- Coverage (pytest + coverage from a throwaway folder): new modules FAIL the pre-registered 80% target - gan_spade 79%, gan_oasis 69%, rgan 52%, gan_common 40%, visuals 28%, imbalance 26%. The uncovered code is the GPU training/orchestration/export paths; they ran end-to-end on the real data and are checked by the independent audit (pass^2) and the reproduction check, but not by unit tests.
- Lint: ruff 0 findings on all new/changed files. compileall OK. node --check OK on all scripts.
- Security: no absolute paths or secrets in the web exports (checked by each export and verify_delivery).
- Result: NOT READY on the coverage criterion only; every other gate passes.

REVIEW (ECC mle-reviewer checklist) - Decision: APPROVE WITH WARNINGS for the research comparison.
- [MEDIUM] Coverage below target / src/floodnet/{imbalance,gan_common,visuals}.py / GPU paths verified only end-to-end / Fix: small-tensor integration tests per stage.
- [MEDIUM] Post-hoc study / docs/imbalance.md / added after original test results were seen / Mitigated: pre-registration hash before training, own freeze, single test evaluation, disclosed.
- [LOW] One generator per budget and family / generator variance unmeasured / disclosed.
- [LOW] R-GAN 25% selected its first checkpoint / earlier optimum not excluded / disclosed.
- [LOW] Cost times include Windows standby pauses for two GANs / disclosed on the page.

Cost display (user request: exclude the standby hours): GAN training time on the page is now *active* time from each GAN's own progress log - gaps longer than 3x the GAN's typical 500-iteration step count as one typical step; only the final generator's lineage counts (the archived diverged OASIS run is excluded; iterations redone after the OOM crash and the requested pause are included as real work). Test `test_active_gan_time_excludes_pauses` (red then green; suite 69 passed). Result: SPADE 25% 66.5 -> 39.7 min (one 29-min pause), SPADE 100% 207.4 -> 81.0 min (128 min), OASIS 100% 383.8 -> 215.3 min (184 min), OASIS 25% 97.4 -> 105.4 min (adds the 8 min redone after the crash), R-GAN unchanged (14.0 / 27.9 min). Windows logged Modern Standby only for part of the OASIS gap (11:56-12:53 UTC); the machine made no progress for the rest either, which is why progress, not the event log, defines a pause. Recorded clock times stay in `selected.json`; the export carries both.
