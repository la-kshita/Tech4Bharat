# Corrected KSDD2 checkpoint: frozen benchmark validation protocol

Prepared before any corrected-checkpoint official-test prediction. This study
tests generalization beyond the development split, while preserving the fact
that the official test set was already observed by the project in Stage5.

## Frozen primary and secondary models

Primary: `recall-alignment/warm_aligned/head.pt`, SHA-256
`ffca032083bd4c7a003c198b70b7dc5eb10c4d7dee40392fd7a81c9160f560d9`.
Its existing normal-only calibration threshold is **-1.684261679649353**,
with strict `score > threshold`. No new calibration is allowed in this study.

The score is the mean of the highest `ceil(.001*N)` segmentation logits in the
valid feature ROI. The learned image classifier is unused. Geometry, frozen
ResNet18 weights, feature normalization, and float16 feature roundtrip remain
identical to the saved pipeline. The machine-readable protocol hashes source,
weights, metadata, fitting IDs, calibration records, and historical evidence.

Secondary: the three previously trained corrected cold-start models, seeds
4217, 4218, and 4219, each with its own existing threshold. Report all three to
check retraining consistency. Do not select a new primary based on test results.
There is no training, augmentation search, threshold sweep, or ensemble search.

## Population and exposure

Use every official KSDD2 test image: **110 defective and 894 normal images**.
The 2,331 official training images contain 246 defects and 2,085 normal images.
Every training defect has already been used for fitting or observed development;
there is no remaining untouched positive confirmation subset in that split.
The user confirms that no separate new labeled factory batch is available.

The official test images were excluded from fitting, mining, normalization, and
calibration of these corrected heads. However, earlier project models were
tested on them, so these are **not pristine project-level confirmation data**.
Any confidence bounds are descriptive benchmark statistics, not an independent
factory certification. Never relabel training images, repeated observations, or
augmented images as a fresh holdout. Keep the old Stage5 archive unchanged.

## Checks and acceptance

1. Recompute all 3,335 usable original image hashes, decoded hashes, mask hashes,
   mask/label agreement, and the existing conservative perceptual duplicate audit.
   Verify train/test ID, pixel hash, and appearance group disjointness. Physical
   product and lot IDs are unknown; appearance checks cannot establish those.
2. Verify fitting IDs and calibration IDs, and reproduce each original threshold
   from its saved normal-only calibration scores. Require complete predictions.
3. Run all four frozen checkpoints in one native image-to-feature pass. No mask,
   label, filename embedding, or historical prediction enters the neural net.
   Save label-free predictions before computing metrics. Abort on integrity errors.
4. In a fresh worker process, regenerate features and scores for all 1,004 images
   in reverse order. Compare every score and decision exactly. This tests software
   determinism; it does not add statistical observations.
5. Report counts, recall, normal false-positive rate, precision, coverage, AUROC,
   and exact 95% Clopper-Pearson intervals. For the primary confidence screen,
   require the one-sided 97.5% recall lower bound to be >=95% and the one-sided
   97.5% false-positive upper bound to be <10%. Bonferroni gives at least 95%
   simultaneous coverage for the pair under independent representative sampling.
6. Separately report point-target attainment: recall >=95%, false alarms <10%,
   and full coverage. Point success and confidence success are different results.
7. Score each model's existing 420 fitting images as an overfitting diagnostic,
   and compare fitting, previously observed development, and official-test rates.
   Never use training performance as evidence of unseen-image accuracy.

Report every outcome, including failures. No adaptive retry or tuning on these
test labels. A new truly independent confirmation requires separately acquired,
blindly labeled production images with product/lot/shift tracking and a frozen
model before access. Finite testing cannot establish absolute certainty.

## Reproduction

From the repository root in the locked recall environment:

```powershell
components/.venv-p0/Scripts/python.exe components/tools/run_recall_validation.py --root components/outputs/recall-validation
```

The controller freezes the protocol, then runs integrity, inference, full replay,
and metric reporting in separate owned workers, one at a time. Each phase has a
1,800-second cap, four CPU threads, and an RSS cap of the smaller of 4 GiB and
60% of currently available RAM. Results and failures are exclusive files.

Dataset: [ViCoS KSDD2](https://www.vicos.si/resources/kolektorsdd2/), industrial
surface images with scratches, spots, and other surface imperfections. Statistical
implementation: [SciPy exact proportion intervals](https://docs.scipy.org/doc/scipy-1.15.2/reference/generated/scipy.stats._result_classes.BinomTestResult.proportion_ci.html).
