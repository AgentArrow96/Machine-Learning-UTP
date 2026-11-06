# Improvement study: pre-registered protocol

Written **before** any training. Settings: `configs/improve.json` (SHA-256 recorded in
`runs/improve/preregistration.json` before the first run). Evals: `audit/improve_evals.md`. Ledger: `audit/improve_ledger.md`.

## Why (error analysis of the frozen test predictions)

| Finding (U-Net 100%, seed 42 unless noted) | Value |
|---|---|
| Road false alarms that land on open water (RF) | 49% (61%) |
| Building false alarms that land on non-flooded buildings | 48% |
| Misses of either target predicted as "Other" | 97–98% |
| Wrong pixels within 3 px of a target boundary (U-Net + CRF) | 43% (40%) |
| Road pixels in components narrower than 6 px | 1% (median width 68 px at 256 px) |
| Validation flooded-mIoU swing between epochs | 5–6 points; no learning-rate schedule |
| RF precision, building / road | 0.25 / 0.23 |

Diagnosis: the 3-class remap merges water, non-flooded roads and non-flooded buildings into "Other" — the very
classes the models confuse with the targets. The released 10-class labels can teach those distinctions at no data
cost. Boundaries, a noisy training curve and the RF's context-free regions are the other levers.

## Variants (baselines = the frozen 18 runs)

| Variant | Methods | What changes |
|---|---|---|
| `aux10` | RF, U-Net, U-Net + CRF | U-Net: 13 outputs (3-class head used for prediction + 10-class auxiliary head, loss = class-weighted 3-class CE + 0.5 × 10-class CE). RF: forest learns the 10-class region majority; output remapped to 3 classes (keep-every-target rule applies to IDs 1 and 3). |
| `recipe` | U-Net, U-Net + CRF | cosine learning rate 1e-4 → 1e-6, up to 60 epochs, patience 12, EMA weights (0.999) validated and saved |
| `tta` | U-Net, U-Net + CRF | no training: frozen U-Nets, softmax averaged over the 8 flips/rotations |
| `res512` | RF, U-Net, U-Net + CRF | 512-px cache (same images, budgets, geometry rule); scored on the 256-px grid (2×2 averaging) |
| `rf_context` | RF | 22 region features + mean of neighbouring regions' features + centroid and area + mean colour of the enclosing coarse (60-segment) SLIC region |
| `encoder` | U-Net, U-Net + CRF | ResNet-34 ImageNet encoder |
| `combined` | RF, U-Net, U-Net + CRF | the components that pass the combine rule on validation |

Combine rule, CRF handling, scoring and the decision rule are fixed verbatim in `configs/improve.json`.
3 seeds × 2 budgets for every method of every variant: 18 + 12 + 12 + 18 + 6 + 12 + 18 = 96 runs.

## Results

Appended after the freeze and the single test evaluation.

Frozen 2026-10-25T03:13:00Z (96 runs, `runs/improve/experiment.json`); test split scored once per run between
03:13:24Z and 04:31:40Z. Combine decision (validation only, `runs/improve/combine.json`): Random Forest = aux10 +
res512 + rf_context; U-Net and U-Net + CRF = recipe + tta + res512 (aux10 and encoder left out). Test flooded-class
mIoU, mean of 3 seeds; change vs the frozen model of the same budget and seed, 95% paired-bootstrap interval:

| Model | Labels | Original | Combined (decision) | Single changes that clearly helped |
|---|---|---|---|---|
| SLIC + RF | 25% | 23.4% | **37.8%** (+14.4 [+12.1, +16.9], helps) | context +5.9, aux10 +4.2, res512 +0.9 |
| SLIC + RF | 100% | 22.5% | **35.9%** (+13.4 [+11.2, +15.8], helps) | context +7.2, aux10 +2.9, res512 +1.5 |
| U-Net | 25% | 52.3% | **61.3%** (+9.0 [+6.7, +11.4], helps) | res512 +4.3, recipe +3.9, tta +2.8 |
| U-Net | 100% | 63.3% | 65.1% (+1.8 [−0.2, +3.9], no clear effect) | recipe +2.3, tta +1.6 |
| U-Net + CRF | 25% | 55.1% | **62.3%** (+7.1 [+4.7, +9.7], helps) | res512 +4.7, recipe +3.6, tta +1.7 |
| U-Net + CRF | 100% | 65.5% | 65.5% (+0.1 [−1.9, +2.1], no clear effect) | recipe +2.0 |

Clearly hurt: ResNet-34 encoder (U-Net 100% −4.3, U-Net + CRF 100% −3.7) and aux10 on the U-Net + CRF at 100%
(−2.3). 19 of 32 comparisons helped, 3 hurt, 10 no clear effect.

What the numbers say:

- **Representation was the Random Forest's limit.** Context (neighbours, position, coarse colour) and the 10-class
  labels together lift it by 13–14 points at both budgets, the largest effect in the project. With 10-class labels
  the RF's road false alarms on open water fall from 61% to 40% of its road false alarms (seed 42, 100%).
- **For the U-Net, training and inference mattered, not capacity.** The cosine/EMA recipe (road false alarms on
  water 49% → 31%), test-time augmentation and 512-px input each help at 25%; a bigger encoder hurts.
- **10-class supervision through a shared head did not help the U-Net** and broke the CRF: the 13-output head on
  smp's 8-channel last decoder stage gives near-uniform 3-class probabilities (ledger, observation 1). A separate
  auxiliary head is future work.
- **At 100% labels the combined U-Net is not clearly better**: the 512-px component helped seeds 42 and 43 and cost
  seed 44 6–9 points (validation had shown the same pattern, −1.8), so the seeds disagree. The recipe alone is the
  only clearly helpful U-Net change at 100% for both the U-Net and the hybrid; this is stated as evidence, not as a
  model chosen on test.
- Rulings and incidents during the study (neighbour double counting fixed before use; area averaging for the 15% of
  images whose 512-px content is 343 rows) are in `audit/improve_ledger.md`.
