# Remaining recall strategies: round 2

Completed 20 October 2026 after the user authorized the remaining strategies and improvement of the strongest candidate. **Best single run: 97.56% recall / 5.28% FPR, but the recipe fails one of its two repeat runs catastrophically. Do not call this a reliable 95%+ training pipeline.** The follow-up corrects training/scoring alignment; see the [alignment study](2026-10-20-alignment-results.md).

The retained parent is round 1 supervised KSDD2: 77/82 detected defects, 27/417 normal false alarms (93.90% recall / 6.47% FPR). The same training/evaluation membership is retained for direct comparison. Those development outcomes are already observed and now guide model selection. New results are **observed-development comparisons**, not independent confirmation.

## Fixed initial experiments

- **R04:** finer 64x160 feature grid from layers 1–3 with a fixed random 448-to-96 channel projection to keep CPU cost comparable. The RGB input remains approximately native resolution; this tests finer latent detail, not invented information from image upscaling. No claim about RGB cable gland is made.
- **R05:** same supervised architecture/IDs/60 epochs, replacing weighted pixel BCE with weighted focal loss (gamma 2). Image loss and optimizer stay fixed.
- **R06:** same saved segmentation maps, compare maximum, top 0.1% mean and maximum local 3x3 average. Each score rule gets its own normal-only threshold.
- **R07:** diagnose normal peak locations. KSDD2 is a texture surface; object-background masking and pose alignment are deferred unless task evidence justifies them. Padding is already excluded and edge defects must not be removed.
- **R08:** one actual image brightness/contrast variant (0.8–1.2) per training original, randomly used with probability 0.5. Masks and geometry stay paired. Compare original images plus fixed paired brightness/blur stress without threshold changes.
- **R09:** nested 50 of the original 100 PaDiM channels, same 128 fitting IDs/covariance formula/regularization. No defective image enters Gaussian fitting.
- **R10:** pair the strongest eligible supervised head with compact PatchCore using two fixed fusion rules. Split the 417 calibration normals into disjoint normalization and threshold subsets; never assume two separate 10%-FPR detectors produce a 10%-FPR ensemble.

At most one compound focal-plus-lighting training variant and one training-only hard-negative refinement may follow, if relevant. Check the best learned configuration with two additional initialization seeds; these are stability checks on observed development data, not new independent items. The round 1 results, model states, phase markers and original archived benchmark remain unchanged.

Each phase runs alone with at most 4 CPU threads, RSS <=min(4 GiB, 60% available RAM) and an 1,800-second pilot limit. Resource-stopped attempts and unsupported/deferred strategies are retained. Cutoffs remain alpha-0.10 normal calibration; evaluation labels will not be used to lower them to hit a desired headline.

Protocol: `components/configs/recall_round2.json`. Local artifacts: `components/outputs/recall-round2/`.

## Completed initial comparisons

All rows below use the same 82 defective and 417 normal development originals. Each cutoff is rank-calibrated at alpha 0.10 on reserved normals; the ensemble uses 208 normals for score normalization and the other 209 for threshold calibration. The 499 evaluation items never enter fitting, normalization or threshold calibration. Technical coverage is 100% in every completed comparison.

| Strategy / variant | Found / missed defects | Normal false alarms | Recall | Normal FPR |
|---|---:|---:|---:|---:|
| Previous R01 supervised image head | 77 / 5 | 27 / 417 | 93.90% | 6.47% |
| R04 finer features, image head | 72 / 10 | 28 / 417 | 87.80% | 6.71% |
| R05 focal loss, image head | 77 / 5 | 26 / 417 | 93.90% | 6.24% |
| R06 parent map maximum | 79 / 3 | 31 / 417 | 96.34% | 7.43% |
| R06 parent map top 0.1% mean | 79 / 3 | 27 / 417 | 96.34% | 6.47% |
| R06 parent map maximum local 3x3 average | 79 / 3 | 30 / 417 | 96.34% | 7.19% |
| R08 lighting augmentation, image head | 77 / 5 | 30 / 417 | 93.90% | 7.19% |
| R09 PaDiM nested 50 channels | 58 / 24 | 38 / 417 | 70.73% | 9.11% |
| R10 focal image head + PatchCore, max normal CDF | 75 / 7 | 26 / 417 | 91.46% | 6.24% |
| R10 focal image head + PatchCore, weighted robust z | 77 / 5 | 21 / 417 | 93.90% | 5.04% |

**R04:** the finer-grid, compressed-channel configuration worsened both operating measures. This is a test of one CPU-oriented spatial-detail variant, not an exhaustive high-resolution or overlapping-tile study. RGB upscaling, overlapping tiles and cable-gland RGB were not tested.

**R05:** focal loss alone preserved recall and removed one normal false alarm. It was not a recall improvement with the learned image classifier. Applying the fixed R06 rule afterward gives a different outcome, recorded below.

**R06:** the best parent-map rule finds two defects missed by the learned image classifier and loses none of its previous catches. It swaps two normal false alarms for two others, leaving the total unchanged at 27. This changes how segmentation logits become the image decision; it is not a monotonic rescaling of the old image score or an evaluation-derived threshold reduction. Padding stays excluded.

**R07:** deferred with a diagnostic and reason. KSDD2 contains surface textures, with no established removable object background or pose ambiguity. About 32.85% of normal evaluation maxima occur within two feature cells of the valid-ROI boundary; that does not establish that those pixels are background. Deleting them could remove edge defects. Foreground masking/alignment was not fitted or claimed as an improvement.

**R08:** mild brightness/contrast training did not improve nominal recall and added three false alarms with the image head. Fixed-cutoff stress results show a mixed robustness tradeoff, not a general robustness win.

**R09:** retaining 50 rather than 100 nested PaDiM channels worsened both measures relative to the reproduced 100-channel comparator (73.17% recall / 6.71% FPR). Reject this variant. The covariance equation and regularization stayed pinned; other shrinkage/covariance estimators were not searched.

**R10:** the initial selected raw image head was focal because it tied the original head's recall with one fewer false alarm. Weighted fusion with PatchCore removes five more false alarms from that head, but does not recover additional defects. Max-CDF fusion reduces recall. Neither overtakes the stronger localized scoring results. Fusion is calibrated as a whole detector, rather than combining two independently thresholded 10%-FPR alarms.

## Apply the fixed localized scoring rule to each new head

The top-0.1%-mean rule was selected from the observed parent comparison and then applied identically to each new segmentation head, without another pooling-rule search per head. These are adaptive development comparisons and remain part of model selection.

| Training configuration + localized scoring | Found / missed defects | Normal false alarms | Recall | Normal FPR |
|---|---:|---:|---:|---:|
| Finer features | 74 / 8 | 43 / 417 | 90.24% | 10.31% |
| Focal loss | 80 / 2 | 29 / 417 | 97.56% | 6.95% |
| Lighting augmentation | 78 / 4 | 26 / 417 | 95.12% | 6.24% |

Focal loss plus localized scoring is stronger than either change assessed using the old image head alone. The finer-feature variant still fails the false-alarm requirement.

## Bounded refinement of the strongest candidate

| Follow-up | Found / missed defects | Normal false alarms | Recall | Normal FPR |
|---|---:|---:|---:|---:|
| Combined focal + lighting, learned image head | 79 / 3 | 27 / 417 | 96.34% | 6.47% |
| Combined focal + lighting, localized score | 78 / 4 | 27 / 417 | 95.12% | 6.47% |
| **Focal + localized score + hard normal refinement** | **80 / 2** | **22 / 417** | **97.56%** | **5.28%** |
| Refined head + PatchCore, max normal CDF | 78 / 4 | 27 / 417 | 95.12% | 6.47% |
| Refined head + PatchCore, weighted robust z | 79 / 3 | 22 / 417 | 96.34% | 5.28% |

The single combined-training variant was permitted because both isolated changes preserved the previous raw image head's recall under the FPR limit. It did not overtake focal training plus localized scoring.

For hard-normal refinement, rank the 1,123 unused normal fitting originals with the selected focal/localized head; add the 128 highest-scoring normals to the original 128 normals and 164 defects. Verify their provider masks are normal. Fine-tune the same head for 12 epochs with a fresh AdamW optimizer, learning rate 0.0002, weight decay 0.0001, batch size 4 and gradient clipping 5. Retain training-only normalization, focal pixel loss and image BCE. No lighting augmentation is used in this refinement. The fitting set is 420 images; calibration/evaluation images and groups are excluded. The inherited 60-epoch parent and the new refined head remain separate artifacts.

This preserves exactly the same 80 detected defects and the same two misses as focal/localized scoring. Fourteen previously false-flagged normals become correct, while seven different normals become false flags: a net reduction of seven. The false alarms are not a strict subset of the parent's false alarms.

This is a useful complete refinement recipe, not an isolated causal estimate for mining: it adds hard normals **and** 12 continuation epochs. A matched continuation-only control was not run, so the improvement cannot be attributed exclusively to the extra hard normals.

Calibrated fusion with PatchCore does not improve this refined candidate. Standalone PatchCore also misses its two remaining defects; the chosen weighted fusion loses one detected defect without removing a normal false alarm. Retain the single refined head.

The selected checkpoint, inference rule and threshold were frozen in `selected-pipeline.json` before either repeat seed was evaluated. Selected artifact SHA-256: `180084ca4586143aa707d83a11232f130f917a612e7acd94a07776ba16c9c3fe`. The normal-only cutoff is **0.4049701691**, applied as strict `score > cutoff`. This is a raw logit-derived score, not a defect probability. Seed 4217 remains the canonical run; repeat seeds cannot replace it in the headline table.

### Remaining missed defects: post-selection diagnosis

After selection, inspect the two source masks without changing any fit or cutoff. `10403.png` has 545 positive source pixels (0.359% of the image) in a narrow annotated region. `11404.png` has 3,847 positive pixels (2.665%) and its annotation touches the source-image boundary. Their scores are -0.03086 and 0.36277, respectively. The remaining failures are not exclusively one-pixel defects, and cropping away image boundaries would remove real annotated defect area. These masks are diagnostic development-label access after selection, never training examples. Do not lower the threshold using these two missed evaluation scores.

## Repeat-training result: instability detected, not hidden

The entire selected recipe was rerun from initialization, including training-only mining and refinement, with seeds 4218 and 4219. The canonical seed had already been frozen; repeats cannot replace it in the headline.

| Initialization seed | Found / missed defects | Normal false alarms | Recall | FPR |
|---|---:|---:|---:|---:|
| 4217, canonical | 80 / 2 | 22 / 417 | 97.56% | 5.28% |
| 4218 | 80 / 2 | 30 / 417 | 97.56% | 7.19% |
| **4219, failed** | **3 / 79** | **42 / 417** | **3.66%** | **10.07%** |

Two good runs do not cancel the failed run. These are initialization checks on the same development membership, not new independent samples; pooling their denominators would be misleading. The original best checkpoint remains a real observed result, but the recipe cannot be recommended as repeatable.

A full 916-score replay of the failed checkpoint matches every saved calibration/evaluation score exactly. Normalization tensors are identical across seeds, cache paths match and weights are finite. The bad seed learns image-classifier weights `[-1.1482, 0.5430, -0.7115]` for maximum, top-0.5% mean and global mean. Successful seeds learn positive maximum coefficients. That negative maximum coefficient can reward decreasing a positive image's strongest local defect score, even though inference flags larger localized scores. The mean training-defect pixel logit is 1.05 in the failed seed versus 3.64 in the canonical seed; training-background mean is -9.15 versus -4.29. Low combined training loss did not guarantee the intended decision behavior.

The corrective study replaces the learned image-classifier loss with BCE of the exact positive-oriented localized inference score. Focal pixel supervision, data roles, backbone, pooling, normal calibration and fixed seeds remain explicit. The old results and failed artifacts stay immutable. [Replay](round2/instability-replay.json), [targeted probes](round2/instability-probes.json), [full stability record](round2/stability-summary.json).

## Paired image-condition stress, original fixed cutoffs

These are transformed versions of the same observed development items, not new independent data. The cutoffs were not recalibrated after each transform. This initial stress study compares the original and lighting-augmented **learned image heads**, not every subsequently selected scoring/fusion variant.

| Transform | Original image head recall / FPR | Lighting-trained image head recall / FPR |
|---|---:|---:|
| Brightness x0.6 | 79.27% / 0.24% | 87.80% / 1.20% |
| Brightness x1.4 | 98.78% / 33.33% | 98.78% / 21.82% |
| Gaussian blur radius 2 | 82.93% / 3.60% | 95.12% / 41.49% |

The lighting-trained head misses fewer darkened defects and lowers bright-image false alarms, but bright-image FPR remains above 10%. Its high blur recall comes with 173/417 normal false alarms. Do not call that a successful operating point. Image acquisition control and further condition-specific validation remain necessary.

## Resources and software checks

The full suite passed: **113 tests**, 45 pinned-backend deprecation warnings, 24.85 seconds. Checks cover role exclusion, tensor save/load, streamed training, the actual fine/combined/repeat training branches, focal gradients, padding-safe pooling, strict under-10% candidate eligibility, repeat-seed exclusion from headline selection, and disjoint ensemble normalization/calibration.

| Selected completed phase | Supervised wall time | Peak owned-worker-tree RSS |
|---|---:|---:|
| Original-map aggregation comparison | 10.77 s | 0.35 GiB |
| Focal training and scoring | 232.38 s | 0.48 GiB |
| Lighting cache extraction | 89.77 s | 0.55 GiB |
| Lighting training and scoring | 339.64 s | 0.48 GiB |
| Fine-feature extraction | 352.45 s | 0.57 GiB |
| Fine-feature training and scoring | 377.33 s | 0.46 GiB |
| PaDiM 50-channel fit and scoring | 429.80 s | 0.86 GiB |
| Paired brightness/blur stress | 343.34 s | 0.55 GiB |
| Combined training and both score rules | 365.23 s | 0.48 GiB |
| Hard-normal mining, refinement and scoring | 152.70 s | 0.51 GiB |

These include process startup and phase-specific preparation/scoring, not just optimizer time. They depend on competing machine load and do not establish a controlled speed comparison. Resource measurements sample the owned worker process tree, not total machine memory. Feature caches are research scratch data; new-image inference recomputes the frozen features from the input image.

## Reproduction and evidence policy

Recreate round 1 first using its [recorded environment, data/weight identities and commands](2026-10-19-round1-results.md#reproduction-commands). This continuation uses the same Python 3.12.14 CPU environment and `components/requirements-recall.lock`; no dependency upgrade was made.

From the repository root in PowerShell, using new empty run directories:

```powershell
components/.venv-p0/Scripts/python.exe -m component_inspector.research.ksdd2_round2 --phase prepare --root components/outputs/recall-round2 --parent components/outputs/recall-round1
components/.venv-p0/Scripts/python.exe components/tools/run_recall_round2.py --root components/outputs/recall-round2 --weights .cache/recall-downloads/model.safetensors
components/.venv-p0/Scripts/python.exe components/tools/run_recall_round2_followup.py --root components/outputs/recall-round2
components/.venv-p0/Scripts/python.exe components/tools/verify_recall_round2.py --root components/outputs/recall-round2
components/.venv-p0/Scripts/python.exe -m pytest components/tests -q

# Substitute the selected_result from selected-for-stability.json for METHOD.
components/.venv-p0/Scripts/python.exe components/tools/benchmark_recall_round2.py --root components/outputs/recall-round2 --weights .cache/recall-downloads/model.safetensors --method METHOD
components/.venv-p0/Scripts/python.exe components/tools/summarize_recall_round2.py --root components/outputs/recall-round2 --output-directory docs/research/round2-replication
```

Outputs and phase completions are exclusive. Reuse saved completions or choose a new run directory; do not overwrite a failed attempt or select a favorable rerun silently. The initial protocol was prepared before its new source file was committed. `source-provenance.json` resolves that source hash to commit `defa74c7eaed6f0d0d3fd37a2f31e75e948d26ee`; every initial phase records that exact source hash. The later training-function change adds a reporting-label override for repeat runs and leaves the numerical training calculation unchanged. Follow-up source and dependency hashes are recorded before those experiments begin.

Source code, exact predictions, calibration scores, compact model/training metadata, feature-index transform records and resource measurements are portable evidence. Raw images, feature arrays and trained tensor files remain local and ignored by Git. Repeating these experiments reproduces development work; it does not create new independent validation data.
