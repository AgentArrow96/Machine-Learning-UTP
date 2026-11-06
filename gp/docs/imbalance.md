# Class imbalance: original handling, method review and the pre-registered study

This document is written **before** any new training (pre-registration). Results are appended at the end only after
the study's own freeze and single test evaluation. Settings: `configs/imbalance.json` (its SHA-256 is recorded in
`runs/imbalance/preregistration.json` before the first GAN iteration). Evals: `audit/imbalance_evals.md`.

## 1. How the original pipeline handled the imbalance

| Where | What | Numbers (from the frozen run records) |
|---|---|---|
| U-Net loss | Inverse-frequency class weights computed on each budget's training images, normalised to mean 1, clipped to [0.25, 4] | 25%: Other 0.25 / building 2.01 / road 0.96; 100%: 0.25 / 1.90 / 1.08 (unclipped, building would be ≈ 68× Other; the clip keeps it ≈ 8×) |
| Random Forest | `class_weight="balanced"` and, under the 100,000-region cap, every flooded region kept (only "Other" thinned) | 100%: 86,875 Other / 4,155 building / 8,970 road regions |
| Data | The 25% subset is stratified by target presence | strata shares equal in 25% and 100% |
| Metric | Headline is flooded-class mIoU, never pixel accuracy | "Other" is 96% of pixels |

The imbalance has two sides: **pixels** (flooded building 1.4%, flooded road 2.5% of training pixels) and **images**
(1,280 of 1,445 training images contain neither target).

## 2. Literature review

Following ECC `scientific-thinking-literature-review` (question → search plan → screening → extraction → synthesis →
citation check).

**Question.** Which methods can reduce the effect of rare *pixel classes* on a segmentation model trained on a few
hundred to 1,445 aerial images, and can GAN-generated training data (in particular the "R-GAN" family) help?

**Search plan.** Web and arXiv searches on: regularised / relativistic GANs for imbalance; GAN and diffusion
augmentation for semantic segmentation and aerial imagery; loss-based and sampling-based remedies; plus the two papers
named by the user (Advanced R-GAN; the journalclub.io source paper on temporal graph networks).

**Screening (inclusion).** Must either (a) generate image–mask pairs or tabular minority samples, or (b) change the
loss/sampling for rare classes; must be trainable offline on one 16 GB GPU within hours. Excluded from *running* (kept
in the review) when the method needs per-pixel labels it cannot produce, or is not a data generator.

| Method | Family | What it does | Fits pixel-level imbalance? | Run here? |
|---|---|---|---|---|
| **Advanced R-GAN** (Lee, Jung, Moon & Rho, *Alexandria Eng. J.* 111, 2025, doi:10.1016/j.aej.2024.10.084) | Regularised GAN, tabular | Generates minority (fraud) rows; generator with spectral normalisation built on STGAN, similarity-measure loss, CELU discriminator; LightGBM downstream | Directly for the RF's 22-number region rows; its three components transfer to an image GAN | **Yes**: tabular (RF) and as the image R-GAN's regularisation |
| Relativistic GAN (Jolicoeur-Martineau, 2018, arXiv:1807.00734) | Adversarial loss | Discriminator judges whether real is *more* realistic than fake; stabilises training | As a loss inside any generator | **Yes**, in the image R-GAN |
| pix2pix (Isola et al., 2016, arXiv:1611.07004) | Mask-to-image cGAN | U-Net generator + PatchGAN | Renders label layouts as photos | **Yes**, backbone of the image R-GAN |
| SPADE / GauGAN (Park et al., 2019, arXiv:1903.07291) | Semantic image synthesis | Spatially-adaptive normalisation keeps the layout through the network | Yes; standard reference | **Yes** |
| OASIS (Sushko et al., 2020, arXiv:2012.04781) | Semantic image synthesis | (N+1)-class segmentation discriminator with class-balanced loss; 3D noise; LabelMix | Yes; designed so rare classes still get realistic texture | **Yes** |
| Spectral normalisation (Miyato et al., 2018, arXiv:1802.05957) | GAN regulariser | Bounds layer Lipschitz constant | Component | Used |
| DiffAugment (Zhao et al., 2020, arXiv:2006.10738) | Limited-data GAN training | Same differentiable augmentation on real and fake | Needed: 361–1,445 training images | Used |
| StyleGAN2-ADA (Karras et al., 2020, arXiv:2006.06676) | Unconditional GAN, limited data | Adaptive discriminator augmentation | No masks → would need a separate mask generator | No (reviewed) |
| BAGAN (Mariani et al., 2018, arXiv:1803.09655) | Class-balancing GAN | Generates images of minority *image* classes | Image-level classes, not pixel labels | No (reviewed) |
| CB-GAN (Rangwani et al., 2021, arXiv:2106.09402) | Class-balancing GAN | Classifier-in-the-loop regulariser for long-tailed generation | Image classes only | No (reviewed) |
| RCGAN (Yang et al., 2020, arXiv:2001.06591) | Regularised cycle GAN | Anomaly *detection* | Not a data generator | No (reviewed) |
| Semantic-synthesis augmentation (Katiyar et al., 2020, arXiv:2011.12636) | Evaluation of synthesis | Synthesis metrics are biased towards well-recognised classes | Warns to evaluate rare classes separately | Informs our quality checks |
| SatSynth (Toker et al., CVPR 2024, arXiv:2403.16605) | Diffusion, aerial | Jointly generates aerial image–mask pairs | Yes, but not a GAN and far heavier | No (reviewed) |
| Long-tail diffusion caution (Wijenayake et al., 2026, arXiv:2602.04749) | Diffusion augmentation | A generator that mirrors the training distribution keeps rare classes rare | Motivates editing layouts to contain extra targets | Informs layout design |
| ChessMix (Pereira & dos Santos, 2021, arXiv:2108.11535) | Non-generative mixing, remote sensing | Chessboard mosaics of patches, favouring rare-class patches | Yes, cheap | **Yes** |
| SMOTE (Chawla et al., 2002, doi:10.1613/jair.953) | Tabular oversampling | Interpolates between minority neighbours | For the RF's region rows | **Yes** |
| Focal loss (Lin et al., 2017, arXiv:1708.02002) | Loss | Down-weights easy pixels | Yes | Reviewed (Dice chosen as the loss remedy) |
| Generalised Dice (Sudre et al., 2017, arXiv:1707.03237) | Loss | Overlap-based loss robust to imbalance | Yes | **Yes** (Dice + CE) |
| Class-balanced loss (Cui et al., 2019, arXiv:1901.05555) | Loss weights | Effective-number weighting | Close to our clipped inverse-frequency weights | Reviewed |
| Copy-paste (Ghiasi et al., 2020, arXiv:2012.07177) | Augmentation | Pastes object instances into other images | Inspires our layout editing (labels pasted, GAN renders pixels) | Reviewed |
| Temporal Graph Network for payment fraud (Saldaña-Ulloa, De Ita Luna & Marcial-Romero, *Algorithms* 17(12):552, 2024, doi:10.3390/a17120552) — the source paper of the journalclub.io R-GAN episode | Temporal graph network | Models transactions as event-based temporal graphs (users, cards, devices, bank accounts); more interaction-event types and denser graphs improved detection | Different data (event streams), not a generator | No; lesson recorded below |
| KID (Bińkowski et al., 2018, arXiv:1801.01401) | Metric | Unbiased kernel distance between feature distributions | Quality check | Used |

**Synthesis.** GANs that only reproduce the training distribution do not, by themselves, fix rarity (Wijenayake et al.).
Useful generators for segmentation must be *mask-conditioned* (pix2pix/SPADE/OASIS) and be fed layouts that contain
more of the rare classes. With 361–1,445 training images, GAN training needs limited-data care (DiffAugment) and checks
for memorisation. The Advanced R-GAN is a tabular method; its three regularisation components are transferred to the
image generator and the method itself is applied, faithfully in kind, to the RF's tabular region rows. The TGN paper's
transferable idea is relational context: the RF currently classifies each superpixel in isolation; a region-adjacency
graph with a graph network is recorded as future work, not run.

**Citation check.** arXiv IDs were verified against the arXiv API on 2026-10-24 (title, first author, year). The
Advanced R-GAN and TGN entries were verified from their DOAJ/MDPI listings and abstracts only: the full texts returned
HTTP 403, so architectural details beyond the abstracts are **assumptions** and are marked as such in §3.

## 3. Pre-registered study

Baseline: the frozen 18 runs (unchanged). New runs live in `runs/imbalance/<variant>/<method>_<budget>_s<seed>/`.

| Variant | RF | U-Net | U-Net + CRF |
|---|---|---|---|
| oversample (control) | ✓ | ✓ | ✓ |
| chessmix | ✓ | ✓ | ✓ |
| rgan (image) | ✓ | ✓ | ✓ |
| oasis | ✓ | ✓ | ✓ |
| spade | ✓ | ✓ | ✓ |
| dice (Dice + CE loss) | – | ✓ | ✓ |
| smote | ✓ | – | – |
| rgan_tabular | ✓ | – | – |

114 new runs (3 seeds × 2 budgets each).

- **Amount.** M = 50% of the budget's image count extra samples (722 at 100%, 180 at 25%) for every image variant;
  tabular variants add as many target rows per class as the rgan image variant gives the RF.
- **Budget isolation.** Every generator, layout, mosaic, duplicate and synthetic row is built only from that budget's
  training images; validation is used only for checkpoint selection and the decision; test is scored once after the freeze.
- **Unchanged.** Class weights (from real images), architecture, optimiser, epochs, patience, validation checkpointing,
  seeds 42/43/44, and the frozen DenseCRF setting.
- **Generators.** One per budget and family, seed 42, equal iteration budgets (20,000 at 100%, 10,000 at 25%, batch 8;
  set from a speed smoke test before pre-registration, see `audit/imbalance_ledger.md`), a checkpoint every 2,000
  iterations, the checkpoint with the lowest validation KID is used. Generator-seed
  variance is not measured (disclosed).
- **Assumptions (abstract-only papers).** Advanced R-GAN's "similarity-measure loss" is implemented as feature-cosine
  similarity (images: ImageNet ResNet-18 features; rows: RBF-MMD + cosine of means); its STGAN base is replaced by a
  pix2pix generator for images and an MLP for rows. ChessMix's empty squares are image-mean pixels with label 255.
  SPADE's VGG-19 perceptual loss uses the cached ResNet-18 to stay offline.
- **Decision rule.** "Helps" only if the 95% bootstrap CI (5,000 paired image resamples) of the seed-mean pooled
  flooded-mIoU difference against the baseline excludes 0 **and** all 3 seed-matched differences are positive; "hurts"
  mirrored; otherwise "no clear effect".

## 4. Results (test split, scored once after the freeze)

Frozen 2026-10-24 18:37:04 UTC (114 runs, all validated); first test evaluation 18:37:35 UTC. Sources:
`output/imbalance/results.json`, `runs/imbalance/experiment.json`; independent checks `output/audit/imbalance_1.json`,
`imbalance_2.json`, `imbalance_repro.json`. Flooded-class mIoU, mean of 3 seeds; differences are seed-matched against
the frozen original runs; CI = 5,000 paired image-bootstrap resamples.

**8 of 38 comparisons clearly helped, 9 clearly hurt, 21 showed no clear effect.**

| Variant | Model | Budget | Original → with remedy | Change (seeds 42/43/44) | 95% CI | Decision |
|---|---|---|---|---|---|---|
| Oversampling (control) | U-Net + CRF | 25% | 55.1% → 62.4% | **+7.3** (+5.0 / +10.3 / +6.6) | +4.0 to +10.9 | helps |
| Oversampling (control) | U-Net + CRF | 100% | 65.5% → 66.3% | **+0.8** (+2.7 / +1.3 / −1.6) | −0.9 to +2.6 | no clear effect |
| Oversampling (control) | RF | 25% | 23.4% → 22.4% | **−1.0** (−1.1 / −0.8 / −1.1) | −1.7 to −0.4 | hurts |
| Oversampling (control) | RF | 100% | 22.5% → 20.1% | **−2.4** (−2.1 / −2.5 / −2.5) | −3.0 to −1.8 | hurts |
| Oversampling (control) | U-Net | 25% | 52.3% → 59.6% | **+7.3** (+4.4 / +10.1 / +7.4) | +4.3 to +10.6 | helps |
| Oversampling (control) | U-Net | 100% | 63.3% → 64.1% | **+0.9** (+1.8 / +1.5 / −0.8) | −0.7 to +2.5 | no clear effect |
| ChessMix | U-Net + CRF | 25% | 55.1% → 59.2% | **+4.0** (+3.8 / +6.1 / +2.2) | +1.1 to +7.1 | helps |
| ChessMix | U-Net + CRF | 100% | 65.5% → 64.2% | **−1.2** (−0.2 / −1.8 / −1.8) | −3.4 to +0.9 | no clear effect |
| ChessMix | RF | 25% | 23.4% → 21.9% | **−1.5** (−1.5 / −1.2 / −1.8) | −2.2 to −0.8 | hurts |
| ChessMix | RF | 100% | 22.5% → 17.0% | **−5.5** (−5.2 / −5.6 / −5.6) | −6.4 to −4.4 | hurts |
| ChessMix | U-Net | 25% | 52.3% → 55.9% | **+3.6** (+4.0 / +4.9 / +1.8) | +1.0 to +6.2 | helps |
| ChessMix | U-Net | 100% | 63.3% → 62.5% | **−0.7** (−0.3 / −0.8 / −1.1) | −2.6 to +1.1 | no clear effect |
| R-GAN (image) | U-Net + CRF | 25% | 55.1% → 57.4% | **+2.2** (+0.1 / +5.5 / +1.1) | +0.1 to +4.3 | helps |
| R-GAN (image) | U-Net + CRF | 100% | 65.5% → 63.0% | **−2.4** (+3.7 / −7.3 / −3.6) | −4.9 to +0.0 | no clear effect |
| R-GAN (image) | RF | 25% | 23.4% → 23.2% | **−0.2** (−0.2 / −0.1 / −0.5) | −1.4 to +0.8 | no clear effect |
| R-GAN (image) | RF | 100% | 22.5% → 17.8% | **−4.7** (−4.5 / −4.8 / −4.7) | −5.8 to −3.6 | hurts |
| R-GAN (image) | U-Net | 25% | 52.3% → 54.2% | **+1.9** (−0.4 / +4.8 / +1.4) | −0.0 to +3.7 | no clear effect |
| R-GAN (image) | U-Net | 100% | 63.3% → 60.0% | **−3.2** (+2.0 / −8.3 / −3.4) | −5.4 to −1.1 | no clear effect |
| OASIS | U-Net + CRF | 25% | 55.1% → 56.2% | **+1.0** (−2.0 / +4.3 / +0.8) | −2.8 to +5.0 | no clear effect |
| OASIS | U-Net + CRF | 100% | 65.5% → 63.1% | **−2.4** (+2.0 / −4.2 / −5.0) | −4.1 to −0.8 | no clear effect |
| OASIS | RF | 25% | 23.4% → 19.2% | **−4.2** (−4.2 / −4.0 / −4.5) | −5.4 to −3.1 | hurts |
| OASIS | RF | 100% | 22.5% → 16.2% | **−6.3** (−6.0 / −6.4 / −6.5) | −7.4 to −5.1 | hurts |
| OASIS | U-Net | 25% | 52.3% → 53.6% | **+1.4** (−2.1 / +4.6 / +1.5) | −2.1 to +4.9 | no clear effect |
| OASIS | U-Net | 100% | 63.3% → 60.9% | **−2.4** (+1.4 / −3.8 / −4.9) | −3.9 to −1.0 | no clear effect |
| SPADE | U-Net + CRF | 25% | 55.1% → 58.8% | **+3.7** (+2.5 / +5.3 / +3.1) | +0.9 to +6.6 | helps |
| SPADE | U-Net + CRF | 100% | 65.5% → 64.6% | **−0.8** (+3.2 / −1.3 / −4.5) | −2.8 to +1.0 | no clear effect |
| SPADE | RF | 25% | 23.4% → 22.6% | **−0.8** (−0.8 / −0.5 / −1.1) | −1.7 to +0.1 | no clear effect |
| SPADE | RF | 100% | 22.5% → 17.0% | **−5.5** (−5.2 / −5.7 / −5.6) | −6.6 to −4.3 | hurts |
| SPADE | U-Net | 25% | 52.3% → 54.6% | **+2.3** (+1.7 / +4.1 / +1.1) | −0.1 to +4.8 | no clear effect |
| SPADE | U-Net | 100% | 63.3% → 61.7% | **−1.5** (+1.9 / −0.9 / −5.6) | −3.3 to +0.1 | no clear effect |
| Dice + CE | U-Net + CRF | 25% | 55.1% → 59.1% | **+4.0** (+3.8 / +5.8 / +2.3) | +2.3 to +5.6 | helps |
| Dice + CE | U-Net + CRF | 100% | 65.5% → 63.1% | **−2.3** (+1.0 / −4.9 / −3.1) | −5.0 to +0.0 | no clear effect |
| Dice + CE | U-Net | 25% | 52.3% → 56.2% | **+4.0** (+4.2 / +5.8 / +1.8) | +2.5 to +5.4 | helps |
| Dice + CE | U-Net | 100% | 63.3% → 60.9% | **−2.4** (+0.4 / −5.1 / −2.4) | −4.6 to −0.4 | no clear effect |
| SMOTE | RF | 25% | 23.4% → 23.0% | **−0.4** (−0.6 / −0.1 / −0.6) | −1.0 to +0.1 | no clear effect |
| SMOTE | RF | 100% | 22.5% → 19.7% | **−2.8** (−2.7 / −2.9 / −2.9) | −3.5 to −2.1 | hurts |
| R-GAN (tabular) | RF | 25% | 23.4% → 24.0% | **+0.6** (+0.7 / +0.7 / +0.4) | −0.5 to +1.6 | no clear effect |
| R-GAN (tabular) | RF | 100% | 22.5% → 22.1% | **−0.4** (−0.3 / −0.4 / −0.5) | −1.1 to +0.2 | no clear effect |

What the numbers say:

- **Few labels (25%) + U-Net / U-Net + CRF:** rebalancing helps. The simplest remedy is the strongest: oversampling the
  real target images (+7.3 points for both). ChessMix (+3.6 / +4.0) and Dice + CE (+4.0 / +4.0) also clearly help.
  Among the GANs only the CRF-refined variants of SPADE (+3.7) and R-GAN (+2.2) pass the rule; the U-Net-only GAN
  gains (+1.4 to +2.3) are not consistent across seeds.
- **All labels (100%):** no remedy clearly helps any model. Several point down (R-GAN U-Net −3.2, OASIS −2.4,
  Dice −2.4), but the seeds disagree, so none is labelled "hurts".
- **Random Forest:** every image-based remedy hurts at 100% (ChessMix −5.5, SPADE −5.5, R-GAN −4.7, OASIS −6.3,
  oversampling −2.4) and SMOTE −2.8; the tabular R-GAN (the paper's own setting) is neutral (+0.6 / −0.4). More target
  regions mostly add false alarms for a region classifier that only sees colour and texture.
- **GANs vs no GAN:** none of the three GANs beats plain oversampling of real target images at either budget. Their
  quality checks passed (no memorisation; validation KID R-GAN 0.122 < SPADE 0.185 < OASIS 0.470 at 100%), so this is
  not a broken-generator artefact.

Independent verification: all 114 runs re-predicted from their checkpoints by code that never imports the project
(0 differing pixels, identical confusion counts; pass^2); the bootstrap and decisions re-implemented independently
(0 mismatches, pass^2); budget isolation and the synthetic data contract hold for every set; pre-registration
precedes the first GAN checkpoint; every run was frozen before any test evaluation; the 140 original frozen files are
unchanged. Reproduction: a retrained U-Net is bitwise identical (182/182 tensors), a retrained R-GAN Random Forest has
identical trees and predictions, and all three generators regenerate their images byte for byte.

Limits: added after the original test results were seen (choices fixed in advance or on validation; test scored once);
one generator per budget and family; the R-GAN at 25% selected its first checkpoint (an earlier optimum cannot be
excluded); OASIS needed normalised discriminator blocks; most test scenes overlap training frames.
