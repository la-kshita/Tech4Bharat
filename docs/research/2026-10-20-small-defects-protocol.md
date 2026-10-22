# Small defects and overfitting: training-only nested study

The corrected checkpoint failed the requested recall target on the already
observed official test benchmark. This study keeps that test closed and fixes
the candidate list, fitting schedule, and selection rule before results.
It aims for >=95% recall with normal false-positive rate <10%; success is not
promised and no existing split can become a pristine factory confirmation set.

## Established methods and exact experimental scope

| ID | Candidate | Purpose and limits |
|---|---|---|
| S01 | Corrected coarse supervised control | Frozen ResNet18 layer2/3 features, 32x80 grid, hidden32 local head; exact positive-oriented image-score training. New folds and 30 fixed epochs, so not an exact rerun of the earlier 60+12 schedule. |
| S02 | Regularized coarse supervised bundle | Hidden16, spatial channel dropout 0.2, AdamW decay0.05, positive-image soft-Dice weight0.5. Tests a combined lower-capacity/foreground-learning hypothesis; no claim isolating each ingredient. |
| S03 | Fine regularized supervised bundle | Unprojected layer1/2 features, 192 channels at 64x160, no 3x3 feature averaging, top-three-cell image score. Aims to retain small responses without averaging nine fine-grid cells. |
| S04 | Multiscale fine regularized bundle | Same fine grid and regularization with all 448 layer1/2/3 channels, preserving coarse semantic context as well as finer detail. |
| S05 | Compact fine PatchCore-style normal memory | 192-channel fine features; normal-only normalization; 64 sampled patches per normal image; 512 full-dimensional bank vectors selected by a 32-dimensional projected greedy coreset; exact chunked nearest-neighbor distances and maximum score. A CPU-bounded variant, not a reproduction of the paper's larger WideResNet configuration or published accuracy. |
| S06 | Linear normal-feature reconstruction | Normal-only PCA with rank64 on the same 192-channel fine patch samples; maximum residual score. A low-capacity reconstruction control, not a claim to implement the full nonlinear DFR architecture. |
| S07 | Classical morphology | Native-resolution Gaussian sigma0.8, bright/dark top-hat responses at 7/15/31-pixel scales, contrast floor0.03, mean of strongest 0.01% pixels. No learned defect labels; separate normal calibration controls alarms. |

All supervised candidates use batch4, AdamW learning rate0.001, 30 epochs,
gradient clipping5, gamma2 weighted focal pixel loss with positive weight8,
plus image BCE of their exact inference pooling rule. No early stopping, epoch
selection, hard-example mining, or image-augmentation search. The control's
image score averages the top ceil(0.001*N) valid feature cells; fine supervised
models average the top three valid cells. Padding is excluded everywhere.
Feature-channel normalization uses only fitted normal images in that specific
fit. This and the fold design are common changes from older experiments.

The methods have industrial precedent at the family level, not proof for this
project. [PatchCore](https://www.amazon.science/code-and-datasets/towards-total-recall-in-industrial-anomaly-detection)
and its [official implementation](https://github.com/amazon-science/patchcore-inspection)
provide normal-patch memory-bank anomaly detection. [DFR](https://arxiv.org/abs/2012.07122)
studies multiscale feature reconstruction; S06 is explicitly a simpler linear
control. [HALCON gray top-hat](https://www.mvtec.com/doc/halcon/2211/en/gray_tophat.html)
documents a commercial machine-vision morphology operator; S07 uses SciPy's
corresponding operations, not HALCON software or its proprietary detector.
[Intel's Anomalib/OpenVINO tutorial](https://www.intel.com/content/www/us/en/developer/articles/training/defect-detection-with-anomalib.html)
supports the relevance of normal-feature models to industrial inspection.

[EfficientAD](https://openaccess.thecvf.com/content/WACV2024/html/Batzner_EfficientAD_Accurate_Visual_Anomaly_Detection_at_Millisecond-Level_Latencies_WACV_2024_paper.html)
and [DRAEM](https://openaccess.thecvf.com/content/ICCV2021/html/Zavrtanik_DRAEM_-_A_Discriminatively_Trained_Reconstruction_Embedding_for_Surface_Anomaly_ICCV_2021_paper.html)
are relevant future alternatives. Their full training/teacher/reconstruction
pipelines are not silently approximated and relabeled as successful runs here.
The current locked environment has CPU-only Torch; the bounded initial suite
uses the available frozen backbone and streaming caches. A GPU run or commercial
vendor comparison would be a separate recorded experiment, not evidence already
collected by this study.

## Data and nested evaluation

Only the versioned official-training manifest is read. The 417 existing normal
calibration originals remain reserved during candidate selection, leaving
**1,914 originals: 246 defective and 1,668 normal**. All of these were previously
used or observed in the project. Repartitioning does not erase that exposure.

Assign whole audited appearance groups to three folds, stratifying defect labels
and source-mask area <=0.5%. Preserve physical-product uncertainty. Frozen
ImageNet features can be cached for all pool images because no data-dependent
feature fitting occurs. Every fitted normalization, head, PCA basis, or normal
bank is recomputed within its allowed training subset.

For each outer fold O:

1. Let the other two folds be A and B. Each candidate is fit on A and validated
   on B, and separately fit on B and validated on A. Within each training fold,
   select 200 normal originals for calibration, then 256 different normals for
   fitting, plus that fold's positives for supervised fitting. Non-supervised
   methods ignore the positive examples; actual learning IDs are recorded.
2. Select the method using only those two inner results. O is absent from
   fitting, calibration, and selection inputs for this decision. Reusable inner
   fits generate separate result files for the two possible validation folds;
   each outer selector reads only its permitted pair.
3. Refit that selected method on A+B, again separating 200 normal calibration
   images and 256 fitted normals, plus all A+B positives for supervised fitting.
   Evaluate O once with the frozen threshold. Retain every outer prediction.

All choices and the final full-training recipe are frozen **before any outer
evaluation**. Select the final recipe using all six directed inner comparisons,
then fit it using all 246 training defects if supervised and 256 pool normals.
Calibrate it once on the reserved 417 normals. This final recipe selection sees
all pool images through inner comparisons; it is **not itself independently
tested by the outer results**. The outer results estimate the predeclared
fold-specific selection procedure, whose selected method may vary by fold.
The final checkpoint remains a candidate for new independent confirmation.

**Supplement registered before any outer result:** train the fixed S01 control
on each outer training partition with the same fitting IDs, calibration IDs,
seed, and 30-epoch schedule as that outer selected model. Run this comparator
after the main study completes; it never participates in selection. Record the
registration and source hash in `paired-control-plan.json`. This supplies an
apples-to-apples paired baseline for internal improvement claims. Retain
discordant counts and per-fold differences; do not compare different datasets
or treat repeated inner predictions as independent observations.

**Additional stability supplement registered before outer results:** repeat
each fold-specific selected outer fit with seed offsets +1000 and +2000, keeping
its method, fitting IDs, calibration IDs, and schedule identical. Report both
complete three-fold repeat outcomes; neither can replace the primary or final
checkpoint. These isolate outer-refit initialization/order variation while
holding inner selection fixed. They do not repeat the entire inner search or
increase the number of independent images. Registration/source hash are saved
in `refit-repeat-plan.json`; this is not post hoc best-seed selection.

**Full historical-recipe reference, also registered before outer results:** the
30-epoch S01 control is a common-budget control, not the prior best checkpoint's
full recipe. A separate fixed reference reproduces that recipe on each outer
training partition: 60 epochs with the legacy learned image objective, training-only
mining of 128 hard normals, 12 refinement epochs, then 12 corrected-objective
continuation epochs; seed4217, original whole-canvas normalization from 128 base
normals and the fitting positives. Use identical outer images and normal
calibration IDs. Report both the study's .08 calibration policy and the original
.10 policy, fixed in advance. This reference never changes inner selection or
the final candidate. Its registration is independently published in
`2026-10-20-small-defects-legacy-registration.json`. It prevents an improvement
claim based only on beating a shorter-trained baseline.

Each calibration uses only normal scores with alpha0.08 and strict
`score > threshold`. The 8% calibration level is a predeclared margin below the
10% workload ceiling, not a post hoc adjustment to improve a held-out result.
Never choose a threshold using held-out defect labels or held-out normals.

## Fixed selection, proof, and failure rules

- A candidate is eligible only if both inner normal FPRs are strictly <10%.
  Among eligible methods, maximize worst inner recall, then mean recall, then
  small-defect recall; minimize mean FPR and use method name for deterministic
  ties. If none is eligible, the predeclared fallback minimizes worst FPR before
  recall and is explicitly labeled as failing the inner constraint.
- Report every candidate and every resource/fit failure. A missing prediction
  is never treated as a successful detection. Incomplete methods are marked
  incomplete and excluded from selection, with their failed attempts preserved.
- Pool outer **decisions** exactly once per pool original. Report per-fold and
  pooled counts, recall, normal FPR, precision, coverage, and small-defect recall.
  Different outer-model raw score scales are not pooled into AUROC/AP.
- Report descriptive exact 95% intervals and the joint screen using one-sided
  97.5% recall/FPR bounds. The joint screen requires recall lower>=95% and FPR
  upper<10%, under independent representative item assumptions. Also distinguish
  the simpler point-target result. Unknown physical groups and prior exposure
  constrain interpretation even if internal metrics pass.
  In particular, overlapping training sets make pooled cross-validation errors
  dependent. A Clopper-Pearson calculation on pooled outer counts is a descriptive
  binomial reference, **not an exact confidence guarantee for cross-validation**.
  Report fold variability and each fold's result, and do not turn a passing
  descriptive bound into independent proof for the final checkpoint.
- Score supervised fitting images separately to reveal fitting/validation gaps.
  Report outer head replay after weight reload in reverse cached-feature order,
  and native-image/cache parity for 16 fixed final-candidate examples. These
  checks establish implementation consistency, not statistical generalization.
  A separate full native replay will additionally regenerate all
  primary/control/repeat/full-reference evaluations on the 1,914 original RGB images, without
  using cached feature arrays. Require all decisions to match and score
  differences <=1e-5; report exact equality separately. This is an inference
  audit, not an additional model-selection or accuracy experiment.
- Keep original test images, prior benchmark snapshots, old predictions, and old
  checkpoints unchanged. Do not inspect individual test errors, rescore test
  images, or change candidates/thresholds after seeing outer results in this study.
- No new factory confirmation is possible without new, nonoverlapping, blindly
  labeled images. Preserve the [fresh-data requirements](2026-10-20-fresh-confirmation-plan.md).

The original study already established a small-defect/generalization concern.
This protocol acts on that hypothesis without further test access. It does not
claim that prior project-level test exposure has been undone or that internal CV
provides absolute certainty for factory operation.

## Reproduction and accounting

Configuration: `components/configs/recall_small_defects.json`. Worker source:
`components/src/component_inspector/research/ksdd2_small_defects.py`. From the
repository root, using the existing locked recall environment and downloaded
KSDD2 official-training originals:

```powershell
components/.venv-p0/Scripts/python.exe components/tools/run_small_defects.py --root components/outputs/recall-small-defects
```

One owned worker at a time, four CPU threads, max1,800 seconds per phase and
RSS cap min(4GiB, 60% currently available RAM). Source/config/dependency hashes,
model/feature hashes, original/group IDs, exact predictions, thresholds, losses,
runtime versions, and controller resource logs are preserved. Files are
exclusive; a failed phase is not blindly overwritten. The controller may resume
completed phases after interruption without refitting them. No new candidate
search is justified by a failed outer result.
