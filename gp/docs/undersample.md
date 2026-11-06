# Uniform Before pipeline and undersampling pipeline

Pre-registered in `configs/undersample.json` (hashed in `runs/undersample/preregistration.json` before any run).

## Why

The original pipeline handled class imbalance differently per model:

| Model | Original handling |
|---|---|
| U-Net | pixel class weights, inverse frequency, capped at 0.25–4; no resampling |
| SLIC + RF | sklearn "balanced" weights (uncapped); at 100% labels the 100,000-region cap kept every target region but only 86,875 of 262,227 "Other" regions |
| U-Net + CRF | the U-Net's handling (no training of its own) |

Results are only comparable if every model is treated the same way, so:

1. **Uniform Before**: every model uses the U-Net's capped class weights and all training data. The RF is retrained
   (`runs/undersample/uniform/rf_*`); the U-Net and U-Net + CRF already match and are reused.
2. **Undersampling pipeline**: every model keeps all target pixels/regions and r "Other" pixels/regions per target
   one, without class weights. r ∈ {1, 3, 10} is chosen on validation per budget and model family.
3. **Blackening test**: the undersampling pipeline trained on photos whose "Other" pixels are painted black.
   Blackening needs the ground-truth mask, which no new photo has, so it is scored on real photos only.

The majority IoU (≈ 91–98%), weighted mIoU (≈ 89–97%) and minority IoU (≈ 22–66%) of the original models show the
imbalance problem: the scores that count "Other" look excellent for every model, while the flooded classes are where
the models fail. Undersampling targets exactly that.

## Protocol

- Same images, budgets (25% / 100%), seeds (42, 43, 44), U-Net and RF settings, and test grid as every earlier run.
- Choices (the ratio) on validation only; all runs frozen in `runs/undersample/experiment.json`; the test split is
  scored once, only for the uniform RF, the chosen-ratio runs and the blackening runs.
- Comparisons: seed-matched differences in test flooded mIoU, 5,000-resample paired image bootstrap, pre-registered
  decision rule (helps / hurts / no clear effect).
- Reported metrics: minority IoU (primary), majority IoU, weighted mIoU, 3-class mIoU, pixel accuracy, per-class IoU /
  precision / recall / Dice.

## Results

Frozen 2026-10-31T08:44:54Z (`runs/undersample/experiment.json`, identity `9964d170dab0ae72…`): 78 runs trained and
validated, 42 scored once on the test split (6 uniform forests, 18 chosen-ratio runs, 18 blackening runs). Test split,
448 images, seeds 42 / 43 / 44. Minority IoU = flooded mIoU (primary, decided by the bootstrap); majority IoU, weighted
mIoU and accuracy are descriptive. Full numbers: `output/undersample/results.json`.

### Ratio chosen on validation

| Family | Labels | r = 1 | r = 10 | r = 3 | Chosen |
|---|---|---|---|---|---|
| SLIC + RF | 100% | 22.0% | 30.8% | 31.9% | r = 3 |
| SLIC + RF | 25% | 20.7% | 23.9% | 28.3% | r = 3 |
| U-Net (and U-Net + CRF) | 100% | 61.8% | 67.9% | 64.2% | r = 10 |
| U-Net (and U-Net + CRF) | 25% | 56.7% | 59.8% | 57.0% | r = 10 |

### Uniform Before: first-version forest → uniform forest

| Model | Labels | Minority IoU (first version → uniform) | Δ (seeds) | 95% CI | Decision | Majority IoU | Weighted mIoU | Accuracy |
|---|---|---|---|---|---|---|---|---|
| SLIC + RF | 25% | 23.4% → 27.6% | +4.2 pp (+4.3 pp, +4.3 pp, +4.0 pp) | [+1.7 pp, +6.7 pp] | helps | 93.0% → 96.0% (+3.1 pp) | 90.5% → 93.6% (+3.1 pp) | 93.0% → 96.1% (+3.0 pp) |
| SLIC + RF | 100% | 22.5% → 31.8% | +9.3 pp (+9.5 pp, +9.2 pp, +9.3 pp) | [+6.3 pp, +12.5 pp] | helps | 91.1% → 96.5% (+5.4 pp) | 88.7% → 94.2% (+5.6 pp) | 91.2% → 96.5% (+5.4 pp) |

### Undersampling (chosen r) vs uniform Before

| Model | Labels | Minority IoU (Before → undersampling) | Δ (seeds) | 95% CI | Decision | Majority IoU | Weighted mIoU | Accuracy |
|---|---|---|---|---|---|---|---|---|
| SLIC + RF | 25% | 27.6% → 26.5% | −1.1 pp (−1.3 pp, −1.2 pp, −0.9 pp) | [−3.6 pp, +1.2 pp] | no clear effect | 96.0% → 93.6% (−2.5 pp) | 93.6% → 91.2% (−2.4 pp) | 96.1% → 93.6% (−2.5 pp) |
| SLIC + RF | 100% | 31.8% → 29.8% | −2.0 pp (−2.0 pp, −2.2 pp, −1.7 pp) | [−4.8 pp, +0.5 pp] | no clear effect | 96.5% → 94.3% (−2.2 pp) | 94.2% → 92.1% (−2.2 pp) | 96.5% → 94.3% (−2.2 pp) |
| U-Net | 25% | 52.3% → 55.5% | +3.2 pp (+1.8 pp, +3.8 pp, +3.9 pp) | [+0.8 pp, +5.5 pp] | helps | 96.9% → 97.8% (+0.9 pp) | 95.3% → 96.3% (+1.0 pp) | 97.0% → 97.8% (+0.8 pp) |
| U-Net | 100% | 63.3% → 61.5% | −1.8 pp (−0.2 pp, −1.4 pp, −3.9 pp) | [−3.6 pp, −0.1 pp] | hurts | 98.2% → 98.2% (−0.1 pp) | 97.0% → 96.8% (−0.2 pp) | 98.3% → 98.2% (−0.1 pp) |
| U-Net + CRF | 25% | 55.1% → 56.8% | +1.6 pp (+2.3 pp, −1.3 pp, +3.9 pp) | [−1.1 pp, +4.1 pp] | no clear effect | 97.2% → 97.9% (+0.8 pp) | 95.7% → 96.5% (+0.8 pp) | 97.2% → 98.0% (+0.7 pp) |
| U-Net + CRF | 100% | 65.5% → 63.2% | −2.2 pp (+0.2 pp, −2.3 pp, −4.6 pp) | [−4.1 pp, −0.5 pp] | no clear effect | 98.4% → 98.3% (−0.1 pp) | 97.2% → 97.0% (−0.2 pp) | 98.4% → 98.3% (−0.1 pp) |

### Blackening test (trained on blackened photos, scored on real photos) vs uniform Before

| Model | Labels | Minority IoU (Before → blackened) | Δ (seeds) | 95% CI | Decision | Majority IoU | Weighted mIoU | Accuracy |
|---|---|---|---|---|---|---|---|---|
| SLIC + RF | 25% | 27.6% → 6.1% | −21.5 pp (−21.6 pp, −21.5 pp, −21.5 pp) | [−25.7 pp, −17.2 pp] | hurts | 96.0% → 0.0% (−96.0 pp) | 93.6% → 0.2% (−93.4 pp) | 96.1% → 3.0% (−93.0 pp) |
| SLIC + RF | 100% | 31.8% → 7.0% | −24.9 pp (−24.8 pp, −25.0 pp, −24.8 pp) | [−28.4 pp, −21.2 pp] | hurts | 96.5% → 0.0% (−96.5 pp) | 94.2% → 0.2% (−94.0 pp) | 96.5% → 3.1% (−93.4 pp) |
| U-Net | 25% | 52.3% → 21.5% | −30.8 pp (−32.0 pp, −28.8 pp, −31.6 pp) | [−36.7 pp, −25.7 pp] | hurts | 96.9% → 0.4% (−96.5 pp) | 95.3% → 1.0% (−94.3 pp) | 97.0% → 3.3% (−93.7 pp) |
| U-Net | 100% | 63.3% → 24.6% | −38.7 pp (−36.3 pp, −38.4 pp, −41.4 pp) | [−43.9 pp, −33.5 pp] | hurts | 98.2% → 0.2% (−98.0 pp) | 97.0% → 0.9% (−96.1 pp) | 98.3% → 3.2% (−95.0 pp) |
| U-Net + CRF | 25% | 55.1% → 25.0% | −30.1 pp (−30.6 pp, −27.6 pp, −32.2 pp) | [−36.5 pp, −24.6 pp] | hurts | 97.2% → 0.1% (−97.1 pp) | 95.7% → 0.8% (−94.8 pp) | 97.2% → 3.0% (−94.2 pp) |
| U-Net + CRF | 100% | 65.5% → 27.6% | −37.9 pp (−34.7 pp, −37.9 pp, −41.1 pp) | [−43.3 pp, −32.4 pp] | hurts | 98.4% → 0.0% (−98.4 pp) | 97.2% → 0.9% (−96.4 pp) | 98.4% → 3.1% (−95.3 pp) |

### Reading

- **Uniform Before.** Giving the forest the U-Net's capped class weights and every training region helped at both
  budgets (+4.2 and +9.3 pp minority IoU, every seed), and its majority IoU rose too (+3.1 / +5.4 pp). The first-version
  forest's uncapped "balanced" weights and region cap made it over-predict floods.
- **Undersampling.** Not a better fix than the class weights: it helped only the U-Net at 25% labels (+3.2 pp), hurt the
  U-Net at 100% (−1.8 pp) and made no clear difference for the forest or the U-Net + CRF. The forest lost background
  quality (majority IoU −2.5 / −2.2 pp); the U-Net family kept it (within ±1 pp). Validation chose the mildest ratio for
  the U-Net (r = 10: 50% of "Other" pixels kept at 25%, 41% at 100%), a sign that it needs the background examples.
- **Blackening.** Trained on blackened photos, every model collapsed on real photos (−22 to −39 pp minority IoU,
  majority IoU ≈ 0): with "not black = flood" learned, a real photo, which has no black, is called flooded almost
  everywhere. Blackening needs the true mask, so it cannot be applied to a new photo in any case.
- **Conclusion.** Before (capped class weights, all training data, the same for every model) stays the reference
  pipeline; the three-class task is kept.

## Reproduce

```powershell
.venv\Scripts\python.exe scripts\run_undersample.py --group gpu   # U-Nets of the ratio grid + their CRF records
.venv\Scripts\python.exe scripts\run_undersample.py --group cpu   # uniform RF, ratio-grid RFs, blackened feature cache
.venv\Scripts\python.exe scripts\run_undersample.py --group all   # the rest (merge the two status files first)
.venv\Scripts\python.exe audit\audit_undersample.py               # independent checks U1-U7; run twice (pass^2)
```
