# Warm-recipe small-defect study: verified internal results

**The requested across-fold consistency target is not achieved in this study.**

The first execution was interrupted by lid-triggered Windows Modern Standby. The failed phase saved no trained checkpoint or predictions. Its plan, empty log, resource failure, system power events and deterministic timeout diagnosis are preserved. A [published recovery registration](2026-10-21-warm-small-recovery-registration.json) reused three completed caches and eleven completed checkpoints byte for byte in a separate output root, then retried the interrupted fit with the original seed and continued the unchanged numerical protocol. This was an infrastructure recovery, not a search for a better seed or outcome. The failure does not count as a successful model result. See the [interruption report](2026-10-21-warm-small-recovery.md).

The locked development selection is **longer108, normal-calibration alpha 0.08**. All nine development checks passed: **False**. All three second-partition checks passed: **False**. Each required recall >=95% and normal false-positive rate <10%. A pooled average cannot rescue a failing fold or seed.

In plain English: a saved checkpoint repeats exactly, but training the recipe again changes which defects it misses. Extra training reduced pooled false alarms in every seed/partition comparison, while small-defect catches improved little. Five of nine development cases and two of three second-partition cases passed; five of twelve failed. This is not a consistently 95% detector.

The registered 95% criterion concerns overall defective-image recall. Small-defect recall is reported separately; passing the overall criterion would not establish 95% recall within that harder subset.

These are KSDD2 industrial surface-defect images from the official training set. All 1,914 originals had already been observed in the project. The second partition changes fitting/calibration/evaluation membership and initialization; it does not supply unseen factory images or erase historical selection exposure. The official test remained closed, the 417 reserved normals remained unused, and no independent production guarantee is established. Recall here means correctly flagging a defective image, including the small-defect subset; pixel localization accuracy is a separate claim.

## Every development candidate and fixed calibration policy

| Method | Alpha | Passed cases / 9 | Worst recall | Worst normal FPR | Worst small / 15 |
|---|---:|---:|---:|---:|---:|
| baseline84 | 0.05 | 3/9 | 87.80% | 8.81% | 7/15 |
| baseline84 | 0.08 | 2/9 | 89.02% | 13.31% | 8/15 |
| longer108 | 0.05 | 2/9 | 89.02% | 6.47% | 7/15 |
| longer108 | 0.08 | 5/9 | 90.24% | 9.89% | 8/15 |
| small108 | 0.05 | 2/9 | 89.02% | 7.19% | 7/15 |
| small108 | 0.08 | 3/9 | 91.46% | 10.43% | 9/15 |
| tiles108 | 0.05 | 3/9 | 85.37% | 7.73% | 9/15 |
| tiles108 | 0.08 | 4/9 | 90.24% | 12.23% | 10/15 |
| tiles_small108 | 0.05 | 3/9 | 84.15% | 7.37% | 8/15 |
| tiles_small108 | 0.08 | 2/9 | 90.24% | 11.51% | 10/15 |

## Variation across all three seeds for each experiment

| Method | Alpha | Pooled recall range | Pooled normal FPR range | Small catches range / 45 |
|---|---:|---:|---:|---:|
| baseline84 | 0.05 | 91.46% to 95.12% | 5.52% to 6.29% | 32 to 38 |
| baseline84 | 0.08 | 92.28% to 95.93% | 8.69% to 10.19% | 33 to 39 |
| longer108 | 0.05 | 91.06% to 94.72% | 4.44% to 5.46% | 31 to 37 |
| longer108 | 0.08 | 92.68% to 96.34% | 7.01% to 8.69% | 33 to 39 |
| small108 | 0.05 | 91.06% to 94.31% | 4.68% to 5.76% | 31 to 36 |
| small108 | 0.08 | 93.09% to 95.93% | 6.89% to 9.35% | 34 to 39 |
| tiles108 | 0.05 | 90.24% to 94.72% | 4.02% to 5.52% | 33 to 39 |
| tiles108 | 0.08 | 93.50% to 95.93% | 8.33% to 9.17% | 35 to 40 |
| tiles_small108 | 0.05 | 89.84% to 94.72% | 4.32% to 5.28% | 33 to 40 |
| tiles_small108 | 0.08 | 93.50% to 95.93% | 7.85% to 9.23% | 36 to 40 |

Each endpoint is a complete three-fold aggregate for one fixed seed, with 246 defective and 1,668 normal originals. These ranges describe variability, not independent samples or a reason to choose the best seed.

All methods and both policies were fixed before the first new feature-cache or fit phase. Alpha .05 and .08 are two declared development policies, not a threshold scan on the official test. Every threshold uses only 200 disjoint normal training originals with strict score > threshold. Selection sees only the nine development cases per policy. No best seed is chosen.

## Paired aggregates, kept separate for each seed

| Stage | Seed | Recipe / policy | TP / 246 | FP / 1,668 | Recall | Normal FPR | Small / 45 |
|---|---:|---|---:|---:|---:|---:|---:|
| dev | 4217 | baseline84 / 0.08 | 236/246 | 147/1,668 | 95.93% | 8.81% | 39/45 |
| dev | 4217 | longer108 / 0.08 | 237/246 | 134/1,668 | 96.34% | 8.03% | 39/45 |
| dev | 4218 | baseline84 / 0.08 | 233/246 | 170/1,668 | 94.72% | 10.19% | 35/45 |
| dev | 4218 | longer108 / 0.08 | 232/246 | 145/1,668 | 94.31% | 8.69% | 36/45 |
| dev | 4219 | baseline84 / 0.08 | 227/246 | 145/1,668 | 92.28% | 8.69% | 33/45 |
| dev | 4219 | longer108 / 0.08 | 228/246 | 117/1,668 | 92.68% | 7.01% | 33/45 |
| confirm | 4220 | baseline84 / 0.08 | 232/246 | 124/1,668 | 94.31% | 7.43% | 36/45 |
| confirm | 4220 | longer108 / 0.08 | 234/246 | 110/1,668 | 95.12% | 6.59% | 36/45 |

Each aggregate uses exactly 246 defects and 1,668 normals for that seed and partition. The same images recur across seeds and partitions; denominators are not added to fabricate a larger independent sample. Raw scores from different models are not pooled into AUROC/AP. The same-alpha baseline comparison separates training changes from calibration-policy changes; baseline alpha .08 also keeps the prior common operating policy visible.

## Locked second-partition results

| Fold | Method | Alpha | TP / 82 | FP / 556 | Recall | Normal FPR | Small / 15 | Point goal |
|---|---|---:|---:|---:|---:|---:|---:|---|
| 0 | baseline84 | 0.08 | 75/82 | 50/556 | 91.46% | 8.99% | 11/15 | Fail |
| 0 | longer108 | 0.08 | 76/82 | 49/556 | 92.68% | 8.81% | 11/15 | Fail |
| 1 | baseline84 | 0.08 | 79/82 | 38/556 | 96.34% | 6.83% | 13/15 | Pass |
| 1 | longer108 | 0.08 | 79/82 | 28/556 | 96.34% | 5.04% | 12/15 | Pass |
| 2 | baseline84 | 0.08 | 78/82 | 36/556 | 95.12% | 6.47% | 12/15 | Pass |
| 2 | longer108 | 0.08 | 79/82 | 33/556 | 96.34% | 5.94% | 13/15 | Pass |

## Detections gained and lost against the same-policy baseline

| Stage | Seed | Small gained / lost | Large gained / lost | Normal alarms added / avoided |
|---|---:|---:|---:|---:|
| dev | 4217 | 0 / 0 | 1 / 0 | 20 / 33 |
| dev | 4218 | 1 / 0 | 0 / 2 | 15 / 40 |
| dev | 4219 | 1 / 1 | 1 / 0 | 18 / 46 |
| confirm | 4220 | 1 / 1 | 2 / 0 | 23 / 37 |

Gains and losses refer to the exact same originals and alpha within each fold/seed pair. For defects, newly flagged means a detection gained; for normal items, it means a false alarm added. Full per-original IDs and all candidate comparisons are retained in [paired changes](warm-small/paired-changes.json). Large-defect counts are stored separately alongside small-defect counts in [summary](warm-small/summary.json).

The selected method and alpha were frozen before any second-partition inference. No change in candidate, threshold, seed, number of epochs, crop geometry or small-defect definition followed these results. The baseline and selected recipe use identical fitting and calibration parent identities within each pair.

## What was actually changed

The full baseline is the prior 60+12+12 recipe. Its three audited seed4217 development checkpoints are reused byte-for-byte; all other baselines are fresh fits. Each continuation adds 24 fixed aligned-objective epochs from the same fold/seed baseline, with the same optimizer reset, presentation order, and fitting parents. longer108 is the matched continuation control. small108 weights small-positive loss by 3. tiles108 substitutes a native 160x160 crop for a full image with probability .5, enlarged to 256x256. tiles_small108 combines those two factors. None changes backbone weights, head capacity, or baseline moments.

A separate [target-visibility diagnostic](2026-10-21-warm-small-target-visibility.json) checked all 246 defective originals, including all 45 small defects. Every coarse target retains positive cells inside the valid ROI, and every parent retains a positive crop. Complete erasure of an image’s positive target does not occur. However, [partial boundary exclusion](2026-10-21-warm-small-target-boundaries.json) affects 59 originals, including four small defects: some coarse positive cells fall outside the registered valid ROI. Geometry effects are not fully ruled out. These checks do not establish that frozen features retain enough discriminative detail or uniquely identify the generalization-gap cause; the registered geometry and all numerical training remain unchanged.

Positive training crops are chosen from the fitting parent’s positive GT crops; normal parents use any crop. At inference, masks, labels and size strata do not choose views: tiled recipes score the full image and all ten overlapping crops, taking their maximum. The complete score is calibrated on normal training parents, accounting for the additional opportunities to flag an image. Every view remains attached to its original parent; crops are not independent samples.

Sources and all implementation limits are in the [preregistered protocol](2026-10-20-warm-small-protocol.md). This is an adaptation of slicing and mask-supervised inspection ideas using our existing frozen ResNet18 features, not a claim to reproduce full SAHI or SegDec-Net results.

## Verifiable evidence and limits

The separate arithmetic/role verifier checked **6234 feature artifacts**, derived labels and size fractions from **1914 hashed original masks**, independently reconstructed the second partition, 6 normalization sets, 7056 hard-normal candidate scores from saved epoch60 weights, and 181440 training-crop choices from registered generators. It verified unchanged normalization and unused classifier weights in continuations, every calibration quantile, all metric rows, the development-only selection, and the paired role exclusions.

It also recomputed **10200 calibration scores** directly from saved convolution weights, independently of the experiment’s head/scoring functions, and checked the dependency lock, installed library versions and pretrained backbone hash.

Full native replay regenerated **32538 head scores** and **65076 decisions** on the same 1,914 original RGB images without cached features or GT masks. Scores matched within 1e-5 and every decision matched; the three replay records report exact equality separately. These are repeated implementation checks, not additional independent images. The verifier was authored in this same work session; third-party certification is not claimed.

The first staged-publication checker omitted the CRLF resource supervisor from its six-file newline allowlist. Its [failed attempt](2026-10-21-publication-check-attempt-1.json) and exact [checker source](2026-10-21-publication-check-attempt-1.py) are preserved. The repaired checker permits canonical Git LF only for the seven explicitly identified CRLF source files, requiring their working bytes to retain their frozen hashes. Explicit checkout attributes preserve those bytes. No numeric source, checkpoint, score or threshold changed during this audit repair.

Preflight: 131 software tests passed with 45 upstream warnings; old baseline scores matched on the two checked originals; all ten native crop feature shapes/values passed the smoke check. Historical evidence remains unchanged. Nine additional audit tests passed, including rejection of forged counts, flags, labels, duplicate originals, calibration membership, small-defect counts and thresholds, plus paired gain/loss semantics. Worker resources include all measured elapsed times and owned-process-tree RSS peaks; the fixed limits are four threads, 1,800 seconds and min(4GiB,60% free RAM) per phase. Timings are measurements of these runs, not controlled hardware claims.

The independent audit itself completed in 645.015 seconds, with sampled owned-process-tree RSS peaking at 360,566,784 bytes. Its separate [resource record](2026-10-21-warm-small-audit-resources.json) and [raw log](2026-10-21-warm-small-audit-worker.log) are preserved outside the verifier's own export tree, avoiding a live-log or self-hash cycle.

Post-study addendum, 22 October: the [three-seed ensemble extension](2026-10-21-seed-ensemble-results.md)
is complete. Its primary policy failed, while its predeclared diagnostic policy
passed all six internal point targets. Neither outcome changes this study's
failed twelve-case criterion. The [fitting-gap diagnostic](2026-10-21-fitting-gap-results.md)
found 100% fitting recall and lower held-out recall. Final full software checks
passed [148 tests](software-checks/summary.json). The
[local checkpoint package receipt](2026-10-21-warm-small-checkpoint-package.json)
records exact archive and member hashes; the weights and pretrained backbone
are preserved locally, with large feature arrays and original RGB excluded.

Overlapping training sets, repeated original images, prior project exposure and unknown physical-product identity limit statistical interpretation. Binomial intervals stored per record are descriptive, not formal exact CV confidence or a production guarantee. A new sealed, blindly labeled factory batch is still required for independent confirmation; see the [collection plan](2026-10-20-fresh-confirmation-plan.md).

Frozen source commit: `eebfeb73f61fdcb218de6a9ffad9a1bad183130b`. Protocol identity: `b38e03009c68fc963d81a22d5dda3e7263cdfe6f85577566a952001913b79749`. The snapshot was published before new cache/fitting phases. Model tensors and arrays remain local, with hashes; exported records preserve exact IDs, thresholds, predictions, losses, parent/view usage and logs.

Evidence: [summary](warm-small/summary.json), [selection](warm-small/selection.json), [verification](warm-small/verification.json), [export hash index](warm-small/export-index.json), [preregistration snapshot](2026-10-20-warm-small-preregistration.json).

## Every development fold and seed

| Method | Alpha | Seed | Fold | TP / 82 | FP / 556 | Small / 15 | Point goal |
|---|---:|---:|---:|---:|---:|---:|---|
| baseline84 | 0.05 | 4217 | 0 | 79/82 | 36/556 | 12/15 | Pass |
| baseline84 | 0.05 | 4217 | 1 | 79/82 | 49/556 | 13/15 | Pass |
| baseline84 | 0.05 | 4217 | 2 | 76/82 | 20/556 | 13/15 | Fail |
| baseline84 | 0.05 | 4218 | 0 | 74/82 | 35/556 | 7/15 | Fail |
| baseline84 | 0.05 | 4218 | 1 | 76/82 | 34/556 | 12/15 | Fail |
| baseline84 | 0.05 | 4218 | 2 | 78/82 | 23/556 | 13/15 | Pass |
| baseline84 | 0.05 | 4219 | 0 | 72/82 | 24/556 | 7/15 | Fail |
| baseline84 | 0.05 | 4219 | 1 | 77/82 | 24/556 | 12/15 | Fail |
| baseline84 | 0.05 | 4219 | 2 | 76/82 | 44/556 | 13/15 | Fail |
| baseline84 | 0.08 | 4217 | 0 | 80/82 | 52/556 | 13/15 | Pass |
| baseline84 | 0.08 | 4217 | 1 | 79/82 | 61/556 | 13/15 | Fail |
| baseline84 | 0.08 | 4217 | 2 | 77/82 | 34/556 | 13/15 | Fail |
| baseline84 | 0.08 | 4218 | 0 | 75/82 | 48/556 | 8/15 | Fail |
| baseline84 | 0.08 | 4218 | 1 | 80/82 | 74/556 | 14/15 | Fail |
| baseline84 | 0.08 | 4218 | 2 | 78/82 | 48/556 | 13/15 | Pass |
| baseline84 | 0.08 | 4219 | 0 | 73/82 | 42/556 | 8/15 | Fail |
| baseline84 | 0.08 | 4219 | 1 | 77/82 | 49/556 | 12/15 | Fail |
| baseline84 | 0.08 | 4219 | 2 | 77/82 | 54/556 | 13/15 | Fail |
| longer108 | 0.05 | 4217 | 0 | 79/82 | 26/556 | 12/15 | Pass |
| longer108 | 0.05 | 4217 | 1 | 79/82 | 36/556 | 13/15 | Pass |
| longer108 | 0.05 | 4217 | 2 | 75/82 | 17/556 | 12/15 | Fail |
| longer108 | 0.05 | 4218 | 0 | 74/82 | 30/556 | 7/15 | Fail |
| longer108 | 0.05 | 4218 | 1 | 77/82 | 27/556 | 13/15 | Fail |
| longer108 | 0.05 | 4218 | 2 | 73/82 | 17/556 | 11/15 | Fail |
| longer108 | 0.05 | 4219 | 0 | 74/82 | 26/556 | 8/15 | Fail |
| longer108 | 0.05 | 4219 | 1 | 77/82 | 29/556 | 12/15 | Fail |
| longer108 | 0.05 | 4219 | 2 | 75/82 | 36/556 | 13/15 | Fail |
| longer108 | 0.08 | 4217 | 0 | 80/82 | 45/556 | 13/15 | Pass |
| longer108 | 0.08 | 4217 | 1 | 79/82 | 50/556 | 13/15 | Pass |
| longer108 | 0.08 | 4217 | 2 | 78/82 | 39/556 | 13/15 | Pass |
| longer108 | 0.08 | 4218 | 0 | 76/82 | 49/556 | 9/15 | Fail |
| longer108 | 0.08 | 4218 | 1 | 78/82 | 55/556 | 14/15 | Pass |
| longer108 | 0.08 | 4218 | 2 | 78/82 | 41/556 | 13/15 | Pass |
| longer108 | 0.08 | 4219 | 0 | 74/82 | 33/556 | 8/15 | Fail |
| longer108 | 0.08 | 4219 | 1 | 77/82 | 36/556 | 12/15 | Fail |
| longer108 | 0.08 | 4219 | 2 | 77/82 | 48/556 | 13/15 | Fail |
| small108 | 0.05 | 4217 | 0 | 79/82 | 27/556 | 12/15 | Pass |
| small108 | 0.05 | 4217 | 1 | 79/82 | 36/556 | 13/15 | Pass |
| small108 | 0.05 | 4217 | 2 | 74/82 | 15/556 | 11/15 | Fail |
| small108 | 0.05 | 4218 | 0 | 74/82 | 31/556 | 7/15 | Fail |
| small108 | 0.05 | 4218 | 1 | 77/82 | 30/556 | 13/15 | Fail |
| small108 | 0.05 | 4218 | 2 | 73/82 | 17/556 | 11/15 | Fail |
| small108 | 0.05 | 4219 | 0 | 74/82 | 27/556 | 8/15 | Fail |
| small108 | 0.05 | 4219 | 1 | 77/82 | 29/556 | 12/15 | Fail |
| small108 | 0.05 | 4219 | 2 | 77/82 | 40/556 | 13/15 | Fail |
| small108 | 0.08 | 4217 | 0 | 80/82 | 39/556 | 13/15 | Pass |
| small108 | 0.08 | 4217 | 1 | 79/82 | 49/556 | 13/15 | Pass |
| small108 | 0.08 | 4217 | 2 | 77/82 | 41/556 | 13/15 | Fail |
| small108 | 0.08 | 4218 | 0 | 77/82 | 49/556 | 10/15 | Fail |
| small108 | 0.08 | 4218 | 1 | 78/82 | 58/556 | 14/15 | Fail |
| small108 | 0.08 | 4218 | 2 | 79/82 | 49/556 | 13/15 | Pass |
| small108 | 0.08 | 4219 | 0 | 75/82 | 32/556 | 9/15 | Fail |
| small108 | 0.08 | 4219 | 1 | 77/82 | 37/556 | 12/15 | Fail |
| small108 | 0.08 | 4219 | 2 | 77/82 | 46/556 | 13/15 | Fail |
| tiles108 | 0.05 | 4217 | 0 | 80/82 | 40/556 | 14/15 | Pass |
| tiles108 | 0.05 | 4217 | 1 | 78/82 | 15/556 | 14/15 | Pass |
| tiles108 | 0.05 | 4217 | 2 | 74/82 | 12/556 | 11/15 | Fail |
| tiles108 | 0.05 | 4218 | 0 | 79/82 | 43/556 | 13/15 | Pass |
| tiles108 | 0.05 | 4218 | 1 | 77/82 | 21/556 | 13/15 | Fail |
| tiles108 | 0.05 | 4218 | 2 | 77/82 | 27/556 | 12/15 | Fail |
| tiles108 | 0.05 | 4219 | 0 | 70/82 | 36/556 | 9/15 | Fail |
| tiles108 | 0.05 | 4219 | 1 | 75/82 | 25/556 | 12/15 | Fail |
| tiles108 | 0.05 | 4219 | 2 | 77/82 | 31/556 | 12/15 | Fail |
| tiles108 | 0.08 | 4217 | 0 | 80/82 | 60/556 | 14/15 | Fail |
| tiles108 | 0.08 | 4217 | 1 | 79/82 | 39/556 | 14/15 | Pass |
| tiles108 | 0.08 | 4217 | 2 | 77/82 | 40/556 | 12/15 | Fail |
| tiles108 | 0.08 | 4218 | 0 | 79/82 | 61/556 | 13/15 | Fail |
| tiles108 | 0.08 | 4218 | 1 | 77/82 | 37/556 | 13/15 | Fail |
| tiles108 | 0.08 | 4218 | 2 | 80/82 | 52/556 | 14/15 | Pass |
| tiles108 | 0.08 | 4219 | 0 | 74/82 | 68/556 | 10/15 | Fail |
| tiles108 | 0.08 | 4219 | 1 | 78/82 | 30/556 | 13/15 | Pass |
| tiles108 | 0.08 | 4219 | 2 | 78/82 | 55/556 | 12/15 | Pass |
| tiles_small108 | 0.05 | 4217 | 0 | 80/82 | 41/556 | 14/15 | Pass |
| tiles_small108 | 0.05 | 4217 | 1 | 78/82 | 21/556 | 14/15 | Pass |
| tiles_small108 | 0.05 | 4217 | 2 | 75/82 | 10/556 | 12/15 | Fail |
| tiles_small108 | 0.05 | 4218 | 0 | 79/82 | 36/556 | 13/15 | Pass |
| tiles_small108 | 0.05 | 4218 | 1 | 77/82 | 22/556 | 13/15 | Fail |
| tiles_small108 | 0.05 | 4218 | 2 | 77/82 | 26/556 | 12/15 | Fail |
| tiles_small108 | 0.05 | 4219 | 0 | 69/82 | 34/556 | 8/15 | Fail |
| tiles_small108 | 0.05 | 4219 | 1 | 76/82 | 24/556 | 13/15 | Fail |
| tiles_small108 | 0.05 | 4219 | 2 | 76/82 | 30/556 | 12/15 | Fail |
| tiles_small108 | 0.08 | 4217 | 0 | 80/82 | 64/556 | 14/15 | Fail |
| tiles_small108 | 0.08 | 4217 | 1 | 79/82 | 37/556 | 14/15 | Pass |
| tiles_small108 | 0.08 | 4217 | 2 | 77/82 | 40/556 | 12/15 | Fail |
| tiles_small108 | 0.08 | 4218 | 0 | 79/82 | 57/556 | 13/15 | Fail |
| tiles_small108 | 0.08 | 4218 | 1 | 77/82 | 31/556 | 13/15 | Fail |
| tiles_small108 | 0.08 | 4218 | 2 | 79/82 | 43/556 | 13/15 | Pass |
| tiles_small108 | 0.08 | 4219 | 0 | 74/82 | 63/556 | 10/15 | Fail |
| tiles_small108 | 0.08 | 4219 | 1 | 77/82 | 32/556 | 13/15 | Fail |
| tiles_small108 | 0.08 | 4219 | 2 | 79/82 | 59/556 | 13/15 | Fail |

## Reproduce

Use the locked environment and official-training originals/parent cache. Audit this preserved run into a new empty export path. The existing evidence and registered fit directories are exclusive and must not be overwritten.

```powershell
components/.venv-p0/Scripts/python.exe components/tools/verify_warm_small.py --root C:/Users/rehtr/Downloads/tech4bharat/components/outputs/recall-warm-small-resumed-20261021 --export docs/research/warm-small-audit-repeat
```

A new fit requires new empty paths and a newly published preregistration snapshot; pass its path using --registration. New timestamps, paths and source commits produce a new protocol identity, not a copy of this study’s identity.

```powershell
components/.venv-p0/Scripts/python.exe -m component_inspector.research.ksdd2_warm_small --phase freeze --root components/outputs/new-warm-small
# Publish matching protocol/preflight snapshots before starting new cache or fit phases.
components/.venv-p0/Scripts/python.exe components/tools/run_warm_small.py --root components/outputs/new-warm-small
# Run replay_warm_small.py serially for --fold 0, then 1, then 2.
# Audit the new run with its own --registration path and a new empty --export path.
```
