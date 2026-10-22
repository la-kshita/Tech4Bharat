# Corrected model validation: the 95% recall goal did not hold

The frozen best corrected checkpoint detected **100/110 test defects (90.91%
recall)** and falsely flagged **69/894 normal images (7.72%)**. It missed ten
defective items. **False alarms passed the requested point target; recall failed.**
The earlier 97.56% development recall must not be presented as confirmed unseen
performance. No model, threshold, or candidate was changed after this result.

## What was tested

KolektorSDD2 (KSDD2) contains industrial surface images with scratches, minor
spots, and other surface imperfections. The task here is image-level
**defective versus normal** detection, using a localized segmentation-logit map
internally. This is not a defect-type classifier or evidence for all component
categories. The official dataset has 2,331 training images (246 defective,
2,085 normal) and 1,004 test images (110 defective, 894 normal).
[Official dataset description](https://www.vicos.si/resources/kolektorsdd2/).

All 1,004 official test images were excluded from these heads' fitting, mining,
feature normalization, and threshold calibration. However, earlier project
models had already been evaluated on them in Stage5. They are therefore
**unseen by the corrected models' fitting, but an already observed project
benchmark**, not pristine independent confirmation. The user confirms there is
no separate new labeled factory batch available. A new random split, crop, or
augmentation of existing images cannot supply that missing evidence.

Before scoring, the [frozen protocol](2026-10-20-validation-protocol.md) selected
`warm_aligned` as primary, fixed its checkpoint and threshold, and specified all
three cold-start corrected seeds as secondary consistency checks. The primary
is one corrected continuation of the earlier selected checkpoint; it is not a
fourth independent repeat of the cold-start training recipe.

## Fixed test outcomes

| Frozen checkpoint | TP | FN | FP | TN | Recall | Normal false-positive rate | Requested point goal |
|---|---:|---:|---:|---:|---:|---:|---|
| **Primary: corrected continuation** | **100** | **10** | **69** | **825** | **90.91%** | **7.72%** | **Fail: recall** |
| Corrected cold-start seed4217 | 99 | 11 | 62 | 832 | 90.00% | 6.94% | Fail: recall |
| Corrected cold-start seed4218 | 101 | 9 | 53 | 841 | 91.82% | 5.93% | Fail: recall |
| Corrected cold-start seed4219 | 100 | 10 | 58 | 836 | 90.91% | 6.49% | Fail: recall |

Every image was scored: 100% coverage and zero technical rejects. The normal
false-positive denominator is all 894 good images, not all flagged images.
Primary precision was 59.17% on this dataset's class mixture; AUROC was 0.97410
and average precision 0.92392. These ranking metrics do not override failure at
the frozen operating threshold. Seed4218 is reported alongside every other seed;
it does not replace the primary using test-set selection. None met 95% recall.

For historical context, the archived original PaDiM benchmark caught 84/110
defects with 77/894 false alarms: 76.36% recall and 8.61% FPR. The current primary
caught 16 more defects and made eight fewer false alarms on these same images.
It also uses labeled defect training examples, unlike the original normal-only
baseline. This improvement does not establish the new 95% goal.

## Overfitting and consistency checks

| Primary checkpoint population | Detected defects | Recall | False alarms on normals | Normal FPR |
|---|---:|---:|---:|---:|
| Existing fitting images; diagnostic only | 164/164 | 100.00% | 3/256 | 1.17% |
| Previously observed development images | 80/82 | 97.56% | 24/417 | 5.76% |
| Official test images excluded from fitting | 100/110 | 90.91% | 69/894 | 7.72% |

Recall dropped **6.65 percentage points** from development to test, while false
alarms increased **1.96 points**. The fitting-to-test gap and repeated selection
on the same development items are consistent with overfitting and/or development
selection optimism. Sampling differences and defect difficulty may also
contribute. This experiment cannot uniquely identify the cause or prove that
overfitting is absent. Perfect training recall is not evidence of deployment
accuracy.

The three corrected cold-start models scored 90.00%–91.82% recall with
5.93%–6.94% false alarms on the same test items. There was no earlier-style
catastrophic seed collapse, but **all three consistently failed the recall
target**. They all detected 96 defects and all missed six; four defects were
caught by only one seed and four by two seeds. Shared misses indicate systematic
difficult cases as well as some seed sensitivity. Do not pool these runs as
330 independent defective items.

The complete native pipeline was rerun in a **fresh process, reverse image
order, all 1,004 images, all four checkpoints**. Every score and every decision
matched exactly: maximum score difference zero and zero decision mismatches.
The heads' parameters and buffers remained unchanged. This establishes
reproducibility for these fixed artifacts on this CPU environment, not future
training or domain generalization.

An additional exploratory random-label control asks how often a random subset
of the same 169 flagged images would include at least 100 of the 110 defects.
The conditional hypergeometric probability is approximately **5.85e-79**.
This supports real discrimination beyond random flags; it does not rule out
overfitting, prior benchmark exposure, correlated products, or domain shift.

## Integrity and leakage audit

- Recomputed source-image, decoded-pixel, and mask hashes for all **3,335 usable
  originals**; every hash matched the historical audited release.
- Verified binary masks, image/mask dimensions, and image-label agreement.
- No cross-split image ID, source-byte hash, decoded-pixel hash, or appearance
  group overlap. A fresh conservative pHash/thumbnail duplicate audit found no
  duplicate pairs. Physical product and lot identities remain unknown; appearance
  auditing cannot prove physical independence.
- Verified each model's 420 fitting IDs: 164 defects and 256 normals, all in
  permitted training roles, with no calibration or development overlap.
- Reproduced all four original thresholds from their separate **417 normal
  calibration scores**. No test image recalibrated any threshold.
- Labels, masks, IDs, and historical scores were excluded from neural-network
  inputs. Saved first-pass predictions contain no truth labels; labels were
  joined for metrics after inference and exact replay.
- Independently recounted every saved prediction and recomputed the confidence
  bounds using beta quantiles, separately from the main binomial-test function.
- All **126 frozen historical archive files** retained their original hashes.
  Two preexisting unmatched local copy files (`10301 (copy).png` and
  `10301_GT (copy).png`) remain quarantined outside the official 3,335 images.
  No official test item was excluded.

Hash/mask agreement checks consistency with supplied annotations. It does not
substitute for independent human review of every label or prove that no
unrecorded historical exposure ever occurred.

## Uncertainty: what can be claimed

Primary exact 95% Clopper-Pearson intervals:

| Rate | Estimate | Exact 95% interval |
|---|---:|---:|
| Defect recall | 90.91% | **83.92%–95.55%** |
| Normal false-positive rate | 7.72% | **6.05%–9.67%** |

The preregistered stronger screen used one-sided 97.5% bounds for each of the
two primary rates, giving at least 95% simultaneous coverage under independent
representative binomial sampling. Its recall lower bound is 83.92% and FPR upper
bound 9.67%: **the joint confidence target also fails**. Every secondary model
failed its descriptive confidence screen too.

These intervals quantify sampling uncertainty under their assumptions. Prior
project benchmark exposure, unknown physical grouping, and a possible future
factory distribution change limit their interpretation. An upper recall bound
above 95% does not establish recall >=95%. No finite dataset provides absolute
certainty about all future images. There is no validated 95%+ factory claim here.
[Exact interval implementation](https://docs.scipy.org/doc/scipy-1.15.2/reference/generated/scipy.stats._result_classes.BinomTestResult.proportion_ci.html).

## Exploratory missed-defect diagnosis

Ground-truth area slices were computed after prediction as diagnostics; they
were not preregistered acceptance tests and did not change any model or threshold.

| Defect area / original image area | Detected / total | Recall |
|---|---:|---:|
| <=0.1% | 0/1 | 0.00% |
| >0.1% to 0.5% | 13/16 | 81.25% |
| >0.5% to 2% | 38/42 | 90.48% |
| >2% | 49/51 | 96.08% |

Combining the first two descriptive slices gives **13/17 (76.47%)** recall for
defects occupying at most 0.5% of an image, versus 87/93 (93.55%) for larger
defects. The smallest slice has only one image, so it cannot establish a stable
size-specific rate. Source-edge defects were detected in 53/58 images (91.38%),
and interior defects in 47/52 (90.38%); the misses are not exclusively an edge
problem. Size alone does not explain all failures: two large defects were missed.

These results motivate future training-only studies of small-defect resolution,
positive-example diversity, regularization, and a label-separated inner
validation scheme. They do not establish a causal fix. The test images and
their masks are now observed evidence and must not become a tuning target
followed by another claim of fresh confirmation.

## Run identity, resources, and reproduction

Protocol identity:
`0d4b112f51488773ef28b670de7ae6774e4c9b0821e4d327e70203f4b9a32d3e`.
Inference/audit source was frozen at commit `495ddbb`. The independent export
verifier was committed at `c6efa0b`. Source hashes are authoritative and recorded
in the protocol; subsequent reporting commits do not alter frozen inference code.

Primary head SHA-256:
`ffca032083bd4c7a003c198b70b7dc5eb10c4d7dee40392fd7a81c9160f560d9`.
Primary strict threshold: **-1.684261679649353**. ResNet18 weights SHA-256:
`80c49dee3da4822c009c5a7fe591e9223c5a2cfcf95a4067ca4dfb5a7b89c612`.
Parent training-source manifest:
`39fabf3c59c6b579bbefa7aba642d2a59f67003a336b1460eb75a2c230c5fd7e`.
Exact package versions, lock hash, fitting IDs, source hashes, test identities,
per-image scores, all thresholds, counts, and resource records are exported below.

| Phase | Wall time | Peak owned-process-tree RSS |
|---|---:|---:|
| Integrity audit | 38.11 s | 0.411 GiB |
| Full first inference pass, four heads | 255.38 s | 0.540 GiB |
| Full fresh-process replay, four heads | 251.55 s | 0.538 GiB |
| Reporting and fitting-image diagnostics | 31.58 s | 0.358 GiB |

All four phases completed within their 1,800-second and measured RAM caps;
four CPU threads, one worker at a time. First-pass image-to-four-head p95 was
0.2651 seconds. This is a CPU benchmark, not a deployed application measurement.
Software suite: **120 passed**, 45 upstream deprecation warnings, 25.46 seconds.
Software success and deterministic replay do not mean the detector met recall.

From the repository root, after recreating the frozen alignment models and locked
environment using the [alignment report](2026-10-20-alignment-results.md):

```powershell
components/.venv-p0/Scripts/python.exe components/tools/run_recall_validation.py --root components/outputs/recall-validation
components/.venv-p0/Scripts/python.exe components/tools/verify_recall_validation.py --root components/outputs/recall-validation --export docs/research/validation
```

Existing paths are exclusive and preserved. Use new empty reproduction and export
directories if the original evidence exists; a reproduction is not new validation
data and must not introduce a new candidate or changed threshold.

Evidence: [protocol](validation/protocol.json), [primary full predictions and
metrics](validation/warm_aligned-results.json), [all four outcomes](validation/summary.json),
[integrity audit](validation/integrity.json), [exact full replay](validation/replay.json),
[independent recount](validation/verification.json), [missed-defect and seed
diagnostics](validation/diagnostics.json), [environment](validation/environment.json),
[complete byte-hash export index](validation/export-index.json).

Decision: **do not claim 95%+ validated recall or production readiness**. Preserve
the measured improvement and this failed target check. Improve using training and
properly separated development data, then obtain a genuinely new sealed batch
under the [fresh confirmation plan](2026-10-20-fresh-confirmation-plan.md).
The original archived Stage5 no-go remains unchanged.
