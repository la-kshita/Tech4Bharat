# Recall research decision, 22 October 2026

The latest batch is complete. The most promising observed candidate is the equal
average of three fixed `longer108` models at the predeclared diagnostic alpha .08.
It caught at least 95% of defective images with fewer than 10% false alarms in
all six internal folds. **The registered primary alpha .05 failed, so the study
is not a successful primary qualification.** We do not switch that label after
seeing the results.

| Setting | Development partition | Second partition | Folds meeting both targets |
|---|---|---|---|
| Primary alpha .05 | 232/246 defects (94.31%); 89/1,668 normal alarms (5.34%) | 234/246 (95.12%); 76/1,668 (4.56%) | 3/6 |
| Predeclared diagnostic alpha .08 | 238/246 defects (96.75%); 119/1,668 normal alarms (7.13%) | 236/246 (95.93%); 117/1,668 (7.01%) | 6/6 |

For the diagnostic setting, individual folds span 95.12%–97.56% recall and
6.12%–8.63% normal false alarms. These ranges are observed point estimates, not
confidence guarantees. It caught 40/45 small-defect images (88.89%) in development
and 37/45 (82.22%) in the second partition. Small-defect fold catches range from
12/15 to 14/15. **Small-defect recall has not reached 95%.**

In everyday terms, averaging three models helps cover some of their different
mistakes. The more sensitive diagnostic setting catches more defects while
remaining inside the observed false-alarm budget. It still misses roughly one
in six small-defect images on the second partition. The previous size weighting,
zoomed crops and extra training did not consistently solve that problem.

## What the overfitting check says

All 18 selected-family heads caught every one of their 164 fitting defects,
including all 30 small fitting defects, and falsely flagged none of their 256
fitting normals at either fixed policy. Held-out individual-head recall ranged
from 87.80% to 98.78% at alpha .05 and 90.24% to 98.78% at .08. The ensemble also
caught every common fitting defect but missed held-out defects.

That is a measured generalization gap. We cannot claim the system is free of
overfitting, nor identify memorization, feature limitations or distribution
variation as its unique cause. These fitting scores never selected the model,
threshold or reported qualification. [Full diagnostic and raw IDs](2026-10-21-fitting-gap-results.md).

## What is verified

The data are KSDD2 industrial surface images. The task is flagging whether an
image contains a visible defect, not detecting every flaw pixel. Each partition
uses the same 1,914 previously observed official-training originals: 246 defective
and 1,668 normal, including 45 small-defect images. Small means a ground-truth
defect mask occupying at most 0.5% of the source image. We never add repeated
members or partitions to claim more independent images.

Original-image replay regenerated all new-member scores without feature caches
or masks. Independent member-score combination reproduced all 7,656 ensemble
decisions with zero differences in flags. Separate arithmetic and role checks
verified mask labels, source hashes, fitting/calibration/evaluation exclusions,
normalization, hard-example mining, normal-only cutoffs and metric counts.
All 148 final software tests passed; 45 upstream deprecation warnings remain.
These checks support correct implementation and reporting, not generalization
to a new factory. The audit was authored in this work session, not certified by
an external reviewer.

The official test stayed closed throughout this batch. Both partitions reuse
previously observed training images; re-splitting them does not remove prior
exposure. There is no separate fresh factory batch. No independently qualified
production model or newly trained all-data deployment checkpoint is claimed.

## Next evidence needed

Keep this diagnostic ensemble as the leading development candidate. To qualify
its .08 setting, prospectively freeze the full procedure and threshold on an
independent batch before seeing its labels or predictions. Follow the
[fresh-image confirmation plan](2026-10-20-fresh-confirmation-plan.md); another
split or seed from these same originals cannot supply that evidence.

For further small-defect development, the next distinct experiment should test
a compact segmentation backbone fine-tuned on native-resolution crops, with
regularization and a fixed augmentation schedule, against the preserved frozen
feature control. Full end-to-end feature learning has not been tested in this
batch. Register its limited candidates, parent-level folds, normal-only
calibration and stopping rule before running it. Use development roles for any
capacity/augmentation selection; retain a separate sealed confirmation batch.
Do not repeatedly tune against the six reported outcomes until they pass.

The authors' [Mixed SegDec-Net paper](https://arxiv.org/abs/2104.06064) provides
an industrial surface-inspection precedent for jointly learned segmentation
and classification. It motivates that next architecture test; it does not
establish that our proposed crop adaptation will improve recall or meet the
false-alarm budget.

Evidence: [all five warm-recipe methods](2026-10-20-warm-small-results.md),
[all ensemble folds and policies](2026-10-21-seed-ensemble-results.md),
[fitting-gap results](2026-10-21-fitting-gap-results.md),
[final software counts](software-checks/summary.json),
[checkpoint receipt](2026-10-21-seed-ensemble-checkpoint-package.json).
