# Task ledger: improvement study

Method as for the class-imbalance study (superpowers executing-plans + test-driven-development + systematic-debugging
+ verification-before-completion; ECC eval-harness, verification-loop, mle-workflow, pytorch-patterns, mle-reviewer).

| # | Task | Evals | Status | Evidence |
|---|---|---|---|---|
| 0 | Pre-registration files written before code: `configs/improve.json`, `docs/improve.md`, `audit/improve_evals.md` | M1 | done | written before any improve code ran |

Ruling (before any run): the plan said "hybrid CRF re-selected on validation at 512". Pre-registered instead as a
geometric rescaling of the frozen setting (pixel-unit sxy doubled), because re-tuning would need its own validation
grid on 512-px U-Nets; rescaling keeps the one-setting-for-all rule and adds no tuning degrees of freedom.

| # | Task | Evals | Status | Evidence |
|---|---|---|---|---|
| 1 | Tests first (red → green): 13 tests in `tests/test_improve.py` (aux head + remap, augment-pair consistency, aux loss, region 10-class labels, cosine/EMA, dihedral TTA, 512→256 geometry, CRF scaling, context features, combine rule, run/manifest/preflight paths) | V | done | 82 passed (full suite) |
| 2 | Training paths: aux10 dataset/loss, cosine LR, EMA (validated + saved), 13-output head; RF 10-class labels, context features; 512-px cache under `data/improve/` | M2, M6 | done | baseline path unchanged: `audit_imbalance_repro.py` rerun after the patch → oversample U-Net 0/182 differing tensors, R-GAN RF identical trees/predictions; G1 140/140 unchanged |
| 3 | Engine (RF/U-Net/hybrid × TTA × 512→256), `evaluate_improve`, combine decision, freeze, test, report | M3, M4, M7 | done | code review against `configs/improve.json` scoring text |
| 4 | Web export (`floodnet improve export`, works mid-study with pending markers) + page 12 "Improvements" | V | done | browser QA: overflow 1440/390 px, model/budget switches, tooltips, 0 console errors |
| 5 | Independent graders: `audit/audit_improve.py` (M1–M5, M7, M8; no `floodnet` import; 512-px inputs re-derived from raw) and `audit/audit_improve_repro.py` (M6) | M1–M8 | written | run after the study |
| 6 | Pre-registered 2026-10-24T23:21:10Z (96 runs); GPU and CPU groups launched side by side | M1 | running | `runs/improve/preregistration.json` |

Ruling (before the combine decision and before any combined run): if no component of a family passes the combine
rule, its combined model would be identical to the frozen baseline (no change to train), so those combined runs are
not trained and not compared; `plan()` drops them once `runs/improve/combine.json` exists. The pre-registration keeps
the full count (96); the export and the audit use the pruned plan.

Incident 1 (systematic debugging; found by the independent grader before any run used the features):
`context_features` collected neighbour pairs as ordered tuples, so two regions touching as A|B in one row and B|A in
another were counted twice and the "mean of neighbouring regions' features" became a weighted mean (independent
recomputation over the 448 test caches: max |difference| 0.12). Root cause: ordered pair set. Fix: unordered pairs
(min, max). Regression test `test_context_features_count_each_neighbour_once` (red → green). The feature cache
`data/features/683bc1ae6f54347e` was moved to `.tmp/superseded/` and rebuilt with the same key (config unchanged);
recheck max |difference| 1.3e-7 (float32), pass. Manifest metadata also listed only the 22 base names; fixed to 50.
No RF had been trained on the superseded cache.

Observation 1 (not a code defect; no change made — the variant is pre-registered): aux10 U-Net + CRF at 25% s42 fell
30 points on validation (U-Net alone −3.4). Systematic debugging: (a) 10-class layouts align with the 3-class masks
and padding on 300/300 training images (100% agreement); (b) no gradient clipping or loss-weight defect; (c) the
aux10 U-Net's 3-class softmax is near-uniform where it predicts a flooded building (mean p(building) 0.35 vs 0.74 for
the baseline U-Net on the same 12 validation images; building logits ≤ 0.12); training curve: validation flooded mIoU
0.05–0.19 for 19 epochs, then a late rise, 3-class validation loss 0.21 vs 0.11. Root cause (hypothesis consistent
with all evidence): the pre-registered 13-output head shares smp's final 8-channel decoder stage, so 10 auxiliary
classes compete with the 3-class readout for 8 features; the 3-class head keeps its argmax but loses its margins, and
DenseCRF (frozen setting, unary = −log p) then smooths the weak target probabilities away. Reported as a result; a
separate auxiliary head on a wider decoder stage is noted as future work, not run.

Incident 2 / ruling (before any 512-px result existed): the first res512 RF evaluation stopped with "Content (343, 512)
is not an integer multiple of (171, 256)". The pre-registered scoring text assumed every 512-px content is exactly 2x
the 256-px content; 4592x3072 photos round to 171 rows at 256 px but 343 at 512 px (train 217, val 70, test 65
images). Ruling: the "2x2 averaging" is implemented as area (box-filter) averaging of the unpadded content onto the
256-px content, which is exactly the 2x2 block mean when the content is exactly 2x (test
`test_area_averaging_equals_2x2_block_mean_when_exact`) and the overlap-weighted mean otherwise (regression test
`test_512_content_that_is_not_exactly_twice_is_area_averaged`, red → green). `configs/improve.json` is unchanged (its
hash is pre-registered); this ruling is the record. The independent grader re-implements the same definition with
explicit loops (max |difference| 0.0 on both geometries). The res512 RF 25% s42 forest itself was trained before the
stop and is reused unchanged; no validation or test number of any 512-px run had been produced.

Grader fix (audit code only): the first two `audit_improve.py` runs stopped before scoring because the derived test
ids are "test_<stem>" while the raw release is keyed by "<stem>". Fixed the lookup; the independently derived 512-px
inputs then equal the project's 512-px cache pixel-for-pixel (24/24 images checked, geometry fields equal).

| # | Task | Evals | Status | Evidence |
|---|---|---|---|---|
| 7 | 78 component runs trained + validated; combine decided on validation (RF: aux10 + res512 + rf_context; U-Net/hybrid: recipe + tta + res512); 18 combined runs | M7 | done | `runs/improve/combine.json` decided 01:3xZ, before the first combined run |
| 8 | Freeze (96 runs) 03:13:00Z → test scored once per run 03:13:24Z–04:31:40Z → report → export | M3 | done | `runs/improve/experiment.json`, `output/improve/results.json`, `web/data/improve.json` |
| 9 | Reproduction (M6): aux10 U-Net, recipe U-Net (0/182 differing tensors each), rf_context RF (identical rows, trees, predictions on 84,643 test regions) | M6 | pass | `output/audit/improve_repro.json` |
| 10 | Results appended to `docs/improve.md`; pre-results text still hashes to the pre-registered protocol hash (5eda9f…) | M1 | done | prefix hash checked before appending |
