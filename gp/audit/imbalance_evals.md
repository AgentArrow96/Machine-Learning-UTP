# Evals for the preprocessing page and the class-imbalance study

Written before any implementation (ECC `eval-harness`: evals first, code graders, PASS/FAIL; deterministic checks
run twice, pass^2). Criteria are not edited after results exist; results are appended in `output/imbalance/evals.json`
and summarised in `docs/imbalance.md`.

| ID | PASS criterion | Grader |
|---|---|---|
| P1 | Preprocessing export numbers equal `audit.json` / `processed.json` / `source.json`; the EXIF example's aligned overlay has higher mask-edge agreement than the misaligned one; every asset URL resolves | `tests/test_preprocessing.py`, `scripts/verify_delivery.py` |
| I1 | `configs/imbalance.json` and its SHA-256 recorded (`runs/imbalance/preregistration.json`) before the first GAN iteration (timestamp order) | `audit/audit_imbalance.py` |
| I2 | Budget isolation: every source image of every GAN, layout, ChessMix mosaic, oversample copy, SMOTE row and tabular R-GAN row belongs to that budget's training IDs; 0 val/test IDs | instrumented logs + `audit/audit_imbalance.py` |
| I3 | Synthetic data contract holds for 100% of samples: shapes, label IDs {0,1,2,255}, label == remap(layout), padding = 255, identity hash stored | `imbalance.check_contract` + audit |
| I4 | GAN quality (KID with references, mask consistency, memorisation) recorded for all 3 GANs × 2 budgets before any downstream GAN-variant run starts; failures reported, not swapped | audit (timestamps) |
| I5 | `identity()` with `variant=None` equals the stored identity of all 18 original runs; resumed pipeline re-verifies them unchanged | `tests/test_imbalance.py`, audit |
| I6 | All 114 runs validated, then frozen; freeze timestamp precedes every new test evaluation; each test evaluated once | audit |
| I7 | Independent re-scoring (no `floodnet` import) of all 114 runs from raw files matches exactly, twice (pass^2) | `audit/audit_eval.py --root` |
| I8 | Retraining one run per family (oversample U-Net 25% s42, R-GAN RF 25% s42) is bitwise identical; regenerating 16 synthetic samples from the selected checkpoint is byte-identical | audit |
| I9 | Bootstrap (seeded) identical on rerun (pass^2); decision labels follow the pre-registered rule mechanically | audit |
| I10 | G1: the 140 original frozen files unchanged; only `gp/web` is ever committed, on branch `floodnet-results-app` | audit + git |
| I11 | `verify_delivery.py` passes incl. preprocessing/imbalance exports; browser QA passes at 1440 and 390 px with no console errors | scripts |
| V | pytest all pass; coverage ≥ 80% on new modules; ruff 0 on new files; compileall + `node --check` OK; 0 secrets/absolute paths in exports | verification loop |
