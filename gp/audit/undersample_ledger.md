# Task ledger: uniform Before and undersampling pipeline

Method as for the improvement study (tests first, systematic debugging, verification before completion).

| # | Task | Evals | Status | Evidence |
|---|---|---|---|---|
| 0 | Pre-registration files written before code: `configs/undersample.json`, `docs/undersample.md`, `audit/undersample_evals.md` | U1 | done | hashed in `runs/undersample/preregistration.json` (2026-10-31T07:22:36Z, 78 runs) before any run |
| 1 | Tests first (red → green): 9 tests in `tests/test_undersample.py` (run paths, blacken feature key, pixel-clipped RF weights, keep probability, fixed seeded keep mask, augmentation keeps the photo and moves the mask, blackening only on "Other", ratio region table, ratio rule) | V | done | 98 passed (full suite) |
| 2 | Training paths: RF `class_weight="pixel_clipped"` and `region_sampling="ratio"`; U-Net `FocusImageDataset` (ignore mask before augmentation, optional blackened photo), `class_weighting: false`; blackened RF feature cache (train split only) | U4 | done | baseline path unchanged: `audit_imbalance_repro.py` rerun after the patch → oversample U-Net 0/182 differing tensors, R-GAN RF identical trees and predictions |
| 3 | Study module `undersample.py` (plan, variant configs, hybrid records, ratio choice, freeze, test, report with seed-matched majority / weighted / 3-class / accuracy next to the bootstrap on minority IoU), CLI stages, `scripts/run_undersample.py` (GPU and CPU groups side by side) | U3, U6 | done | code review against `configs/undersample.json` |
| 4 | App bundles `web/data/uniform.json` and `web/data/undersample.json` (same schema as `after.json` plus ratio, blackening and example images); walkthrough replays the uniform forests (`--rf-variant`) | V | done | `verify_delivery.py` extended: bundles equal the evaluations, PNG masks equal the stored predictions, walkthrough masks equal the uniform bundle |
| 5 | App: Before = uniform pipeline (first version kept as "Before v1" and in a reference card), Undersampling view and page, Compare with Undersampling or with After (each against its own baseline), corrected class-weight equation | V | done | browser QA rewritten to navigate by route in every view (overflow 1440 / 390 px, LaTeX, leaks, decisions, masks) |
| 6 | Independent grader `audit/audit_undersample.py` (U1–U7; no `floodnet` import; masks from the raw-derived arrays; sklearn scoring; class weights, ratio rule and bootstrap re-implemented) | U1–U7 | done | `output/audit/undersample_1.json` and `_2.json`: U1–U7 all pass both times (pass^2) |
| 7 | Study run: GPU and CPU groups side by side, then `--group all`; ratio chosen on validation (U-Net r = 10 at both budgets, RF r = 3); frozen 2026-10-31T08:44:54Z (78 runs, 42 tested); report, export, walkthrough | U3, U6 | done | 0 failed stages; `output/undersample/results.json`; results appended to `docs/undersample.md` |
| 8 | Delivery checks | V | done | pytest 99 passed; `verify_delivery.py` passed (uniform.json 18 runs / 2,688 masks, undersample.json 36 runs / 2,688 masks + 18 blackening runs, walkthrough 36 masks equal the uniform bundle); browser QA 278 checks passed, 0 console errors; G1 140/140 unchanged; `results.json` / `after.json` equal the committed versions |

Incident 1 (found by the browser QA, before any commit): in the Before view the walkthrough check list and the
"All runs" table showed run ids such as `undersample/uniform/rf_25_s42`, which leaked the study name into the Before
view. Fix: display the run name without the study folder (walkthrough adds "(Before, retrained forest)"); the stored
ids are unchanged.

Incident 2 (found by a dry run of the export on frozen runs): the original hybrid runs record their U-Net only by run id,
so the bundle export's base-run lookup (`base_run.directory`) would have failed on them. Fix: fall back to
`runs/<run_id>`. No artifact had been written.

Observation (not a defect): the app's class-weight equation read clip(N / 3n_c), but `neural.class_weights` divides the
inverse frequency by its mean over the classes; the equation in `web/assets/js/math.js` was corrected to match the code.

Ruling (before any run): "uniform" keeps the U-Net and U-Net + CRF runs unchanged, because they already use the
capped class weights on every training pixel; only the forest is retrained. The first-version forest stays in
`runs/rf_*` and in the app.

Ruling (before any run): the user's blackened-photo idea is run as a separate test, not as the pipeline, because the
mask needed to blacken a photo does not exist for a new photo; it is trained on blackened training photos and scored
on real validation and test photos only.
