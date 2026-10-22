# Small-defect and overfitting study: verified training-only results

**The primary internal nested result fails the requested point target.** The selected nested procedure caught **231/246 defects (93.90% recall)** with **144/1,668 normal false alarms (8.63%)**. The official test set was not rescored. These are internal cross-validation results, not new independent factory proof.

The selected procedure did not improve on the simpler paired control: it caught no more defects and produced no fewer false alarms.

The full prior-recipe reference averages **95.93% recall / 8.81% normal FPR** at the fixed .08 calibration policy. Its fold results below determine consistency; a passing pooled average on reused training images does not establish reliable 95% performance or improve the earlier official-test benchmark.

## What this result means

KSDD2 contains industrial surface images with visible imperfections such as scratches and spots. The reported goal is image-level detection: flag an image containing a defect, while keeping false flags on normal images below the budget. Small-defect recall is also image-level, stratified by ground-truth mask area. These counts do not establish pixel-level localization accuracy or cable-gland performance.

The study used only official KSDD2 training originals. Whole audited appearance groups were assigned to three folds, each containing 82 defects, 556 normals, and 15 small-defect images. Seven fixed candidates were compared using inner folds. Each outer-fold model was selected, fitted, and calibrated without that outer fold. All choices and the final candidate were frozen before outer predictions. The final candidate was selected from inner comparisons across all training data; the nested result estimates the fold-specific selection procedure, not an independent test of that final checkpoint.

These images had already been used or observed during earlier project work. Nested separation reduces new leakage and selection optimism; it cannot erase historical exposure or establish unknown physical-product independence. There is still no new sealed, blindly labeled production batch. A better internal result must not be advertised as a confirmed improvement from the earlier 90.91% official-test result: the cohorts differ.

## Paired outer comparison on the same images

| Procedure | TP / FN | FP / normal count | Recall | Normal FPR | Small defects caught |
|---|---:|---:|---:|---:|---:|
| Fixed coarse supervised control | 232 / 14 | 132 / 1,668 | 94.31% | 7.91% | 36/45 |
| Inner-selected nested procedure | 231 / 15 | 144 / 1,668 | 93.90% | 8.63% | 34/45 |

The paired difference is **-0.41 percentage points recall** and **+0.72 points FPR**. The selected procedure gained 3 defects and lost 4 relative to the control; it cleared 47 old false alarms and introduced 59 new ones. These comparisons use identical outer originals, fitting IDs, and calibration IDs. They are internal paired observations, not a formal claim of significance under cross-validation dependence.

## Full prior-recipe reference

The common-budget control above uses 30 epochs. A separately preregistered reference reproduces the prior best warm-corrected recipe with 60 base epochs, 12 hard-normal refinement epochs, and 12 aligned-objective continuation epochs. It uses seed4217, 128 base normals plus 128 mined normals, the original whole-canvas normalization, and the same outer positives and calibration IDs. This reference is not an additional selectable candidate and cannot change the frozen final model.

| Fixed full prior recipe operating policy | TP / FN | FP / 1,668 | Recall | Normal FPR | Small / 45 |
|---|---:|---:|---:|---:|---:|
| Calibration alpha 0.08 | 236 / 10 | 147 / 1,668 | 95.93% | 8.81% | 39/45 |
| Calibration alpha 0.1 | 239 / 7 | 175 / 1,668 | 97.15% | 10.49% | 41/45 |

| Reference fold | Calibration alpha | TP / 82 | FP / 556 | Recall | Normal FPR | Small / 15 |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 0.08 | 80/82 | 52/556 | 97.56% | 9.35% | 13/15 |
| 0 | 0.1 | 81/82 | 66/556 | 98.78% | 11.87% | 14/15 |
| 1 | 0.08 | 79/82 | 61/556 | 96.34% | 10.97% | 13/15 |
| 1 | 0.1 | 79/82 | 65/556 | 96.34% | 11.69% | 13/15 |
| 2 | 0.08 | 77/82 | 34/556 | 93.90% | 6.12% | 13/15 |
| 2 | 0.1 | 79/82 | 44/556 | 96.34% | 7.91% | 14/15 |

At the common .08 calibration policy, the selected nested procedure caught **-5 defects** and made **-3 false alarms** relative to the full prior recipe. Both .08 and the historical .10 policy were fixed before outer outcomes. Do not claim an overall improvement over the full prior recipe unless these comparisons support it; a gain over the shorter control alone is insufficient. See [full-recipe comparison](small-defects/legacy-comparison.json).

The full prior recipe fails at least one common-policy fold target. Even a passing pooled average would not establish consistent per-fold performance.

## Every outer fold

| Outer fold | Inner-selected method | TP / FN | FP / 556 | Recall | Normal FPR | Small / 15 |
|---|---|---:|---:|---:|---:|---:|
| 0 | multiscale_regularized | 74 / 8 | 37 / 556 | 90.24% | 6.65% | 7/15 |
| 1 | coarse_regularized | 80 / 2 | 67 / 556 | 97.56% | 12.05% | 14/15 |
| 2 | coarse_regularized | 77 / 5 | 40 / 556 | 93.90% | 7.19% | 13/15 |

All outer images were accounted for exactly once. Raw logits and distance scores from different outer models are not pooled into one AUROC/AP. Training-image diagnostics are kept separately and cannot qualify unseen performance.

## Every candidate and experiment

Each row summarizes six directed inner comparisons: three independently fitted one-fold models, each evaluated on the other two folds. An image appears twice in those inner comparisons; the six results are not six independent datasets.

| Candidate | Fitting recall range | Held-out inner recall range | Inner FPR range | Small-defect recall range | Inner point-goal cases / 6 | Total fit seconds |
|---|---:|---:|---:|---:|---:|---:|
| coarse_control | 100.00%–100.00% | 91.46%–97.56% | 6.65%–10.79% | 66.67%–93.33% | 1/6 | 260.30 |
| coarse_regularized | 100.00%–100.00% | 90.24%–97.56% | 5.94%–10.61% | 60.00%–93.33% | 2/6 | 237.60 |
| fine_patchcore | Not recorded for non-supervised methods | 51.22%–58.54% | 6.47%–10.79% | 26.67%–53.33% | 0/6 | 29.28 |
| fine_pca | Not recorded for non-supervised methods | 17.07%–37.80% | 4.86%–15.47% | 0.00%–20.00% | 0/6 | 28.15 |
| fine_regularized | 100.00%–100.00% | 90.24%–98.78% | 2.34%–8.45% | 73.33%–100.00% | 2/6 | 679.61 |
| morphology | Not recorded for non-supervised methods | 39.02%–52.44% | 6.12%–8.63% | 13.33%–53.33% | 0/6 | 0.02 |
| multiscale_regularized | 100.00%–100.00% | 89.02%–96.34% | 7.19%–11.87% | 53.33%–93.33% | 2/6 | 2496.05 |

The exact configurations and industrial sources are in the [preregistered protocol](2026-10-20-small-defects-protocol.md). The regularized heads are experimental bundles, compact PatchCore is not the paper-scale WideResNet implementation, PCA is a linear reconstruction control, and SciPy morphology is not proprietary HALCON software. EfficientAD/DRAEM were researched but are not recorded as trained experiments.

## Refit stability without seed selection

| Fixed outer-refit seed set | TP / FN | FP / 1,668 | Recall | Normal FPR | Point goal |
|---|---:|---:|---:|---:|---|
| Primary | 231 / 15 | 144 / 1,668 | 93.90% | 8.63% | Fail |
| Offset +1000 | 231 / 15 | 129 / 1,668 | 93.90% | 7.73% | Fail |
| Offset +2000 | 231 / 15 | 158 / 1,668 | 93.90% | 9.47% | Fail |

Each repeat holds the fold-specific inner-selected method and data membership fixed, then changes outer-refit initialization/order only. The entire inner search was not repeated. These are the same 246 defects and 1,668 normals each time; the denominators are not tripled. No repeat replaces the primary or final model.

## Small defects, uncertainty, and overfitting

The nested procedure detected **34/45 small defects (75.56%)**, versus 36/45 for the paired control. Small means a source mask occupying at most 0.5% of the original image, not a resized feature-mask fraction. There are only 15 small defects per fold; retain the counts instead of claiming a precise factory-wide small-defect guarantee.

Of the 15 missed defects, 11 were in the small-defect stratum. Larger defects were caught in 197/201 cases. This is a diagnostic of the frozen result, not permission to retune on these outer examples. Training fit can look excellent while this held-out small-defect slice remains weak.

Descriptive Clopper-Pearson references on pooled counts are 90.14%–96.55% recall and 7.33%–10.08% normal FPR. **Overlapping cross-validation training sets make errors dependent, so these binomial intervals are not formal exact CV confidence or independent proof for the final candidate.** Point targets, descriptive bound targets, refit consistency, and production confirmation are separate claims.

No positive training image lost its entire valid target. Some projected border target cells are outside the nearest-sampled ROI: 59 coarse-grid and 40 fine-grid images have this partial boundary effect. It is preserved in the target-coverage audit, not silently fixed after results. Fine features retain more such targets, but this does not establish perfect boundary representation.

## Final candidate and verifiable evidence

Final inner-selected method: **fine_regularized**. Head/model SHA-256: `dd77ecf687ee7fed5e1d9e0dd06afedc2aad966f341645e90ab9eb0494ccb9cf`. Frozen strict threshold: **score > -0.10871299356222153**. Its normal-only calibration uses the reserved 417 originals after method selection. These calibration scores are not a separate performance test.

Native/cache parity on 16 fixed final-candidate images: maximum score difference 0.0; all decisions matched. Native calibration-pipeline p95 was 0.2740 seconds on CPU. Outer heads were reloaded and replayed in reverse cached-feature order. Parity does not prove generalization.

A separate full native replay regenerated all **11,484 primary/control/repeat/legacy model-threshold evaluations on the 1,914 original RGB images**, including the two predeclared legacy thresholds on the same weights, without reading cached feature arrays, and checked the archived decisions. See [native replay](small-defects/full-native-replay.json) and [native prediction rows](small-defects/full-native-predictions.json).

Independent verification recounted every inner/outer/control/repeat row, verified 8148 feature artifacts, reconstructed 14 distinct normal-only moment sets, checked selections against permitted inner inputs, and checked the immutable source/model/threshold lineage. Old benchmark evidence is unchanged. Software suite before training: **125 passed**, 45 upstream warnings. The arithmetic verifier is a separate implementation authored in this same work session; no third-party certification is claimed.

The first audit attempt incorrectly expected supervised fitting diagnostics for unsupervised methods and stopped with a missing-file error. The audit and report readers were corrected to match the frozen output schema. No models, selection rules, thresholds, or predictions changed. The [failed audit attempt](small-defects/verification-attempt-1.json) is preserved.

Protocol identity: `b7ce2f1f5dff4c5e66cf47e8f1bf44331e3a2bd00a33c191e6bcac9e2a28698f`. Frozen main source commit: `de6ad3d2a32a50de7be94bbcaf9b19823382469a`. Source/config hashes in the protocol are authoritative. New feature arrays contain 16.39 GiB of logical data; model tensors and arrays remain local. Exact predictions, IDs, hashes, losses, selection inputs, timestamps, and resource logs are exported.

A lossless NTFS compression intervention reduced generated cache disk allocation during the study. It did not change array values, fitting rules, models, or thresholds; the final verifier checks every feature hash. Compression overlapped one fitting phase, so elapsed times are not clean hardware speed comparisons. See [storage intervention](small-defects/cache-storage-optimization.json).

Evidence: [nested outcomes](small-defects/nested-results.json), [paired comparison](small-defects/paired-comparison.json), [all refit repeats](small-defects/refit-stability.json), [independent verification](small-defects/verification.json), [final candidate](small-defects/final--frozen.json), [selection record](small-defects/selection.json), [timing/storage](small-defects/timing-and-storage.json), [complete byte-hash index](small-defects/export-index.json).

A local ZIP preserves all audited model checkpoints, the exact backbone, source code, and exported evidence. Its [package receipt](2026-10-20-small-defects-checkpoint-package.json) records the archive hash and location. All member hashes are verified after writing. It excludes dataset images and cached arrays; it is a checkpoint backup, not a new evaluation or a standalone installer.

The first ZIP membership check failed because Windows checkpoint paths were compared with ZIP-normalized paths. The first ZIP is preserved; the corrected archive uses normalized member names and passes all hash checks. See [packaging attempt](2026-10-20-small-defects-package-attempt-1.json).

## Every directed inner result

### coarse_control

| Train fold → validation fold | TP / 82 | FP / 556 | Small / 15 |
|---|---:|---:|---:|
| 0 → 1 | 80/82 | 56/556 | 14/15 |
| 0 → 2 | 75/82 | 60/556 | 11/15 |
| 1 → 0 | 77/82 | 37/556 | 11/15 |
| 1 → 2 | 76/82 | 43/556 | 12/15 |
| 2 → 0 | 77/82 | 40/556 | 10/15 |
| 2 → 1 | 78/82 | 51/556 | 13/15 |

### coarse_regularized

| Train fold → validation fold | TP / 82 | FP / 556 | Small / 15 |
|---|---:|---:|---:|
| 0 → 1 | 80/82 | 48/556 | 14/15 |
| 0 → 2 | 78/82 | 42/556 | 13/15 |
| 1 → 0 | 77/82 | 33/556 | 11/15 |
| 1 → 2 | 74/82 | 39/556 | 12/15 |
| 2 → 0 | 75/82 | 46/556 | 9/15 |
| 2 → 1 | 80/82 | 59/556 | 14/15 |

### fine_patchcore

| Train fold → validation fold | TP / 82 | FP / 556 | Small / 15 |
|---|---:|---:|---:|
| 0 → 1 | 47/82 | 44/556 | 7/15 |
| 0 → 2 | 42/82 | 36/556 | 4/15 |
| 1 → 0 | 43/82 | 36/556 | 4/15 |
| 1 → 2 | 44/82 | 37/556 | 5/15 |
| 2 → 0 | 43/82 | 56/556 | 4/15 |
| 2 → 1 | 48/82 | 60/556 | 8/15 |

### fine_pca

| Train fold → validation fold | TP / 82 | FP / 556 | Small / 15 |
|---|---:|---:|---:|
| 0 → 1 | 14/82 | 36/556 | 2/15 |
| 0 → 2 | 15/82 | 27/556 | 0/15 |
| 1 → 0 | 21/82 | 55/556 | 3/15 |
| 1 → 2 | 17/82 | 30/556 | 0/15 |
| 2 → 0 | 27/82 | 86/556 | 3/15 |
| 2 → 1 | 31/82 | 60/556 | 3/15 |

### fine_regularized

| Train fold → validation fold | TP / 82 | FP / 556 | Small / 15 |
|---|---:|---:|---:|
| 0 → 1 | 78/82 | 43/556 | 14/15 |
| 0 → 2 | 75/82 | 29/556 | 12/15 |
| 1 → 0 | 76/82 | 27/556 | 12/15 |
| 1 → 2 | 74/82 | 13/556 | 11/15 |
| 2 → 0 | 75/82 | 47/556 | 11/15 |
| 2 → 1 | 81/82 | 46/556 | 15/15 |

### morphology

| Train fold → validation fold | TP / 82 | FP / 556 | Small / 15 |
|---|---:|---:|---:|
| 0 → 1 | 41/82 | 34/556 | 8/15 |
| 0 → 2 | 32/82 | 34/556 | 2/15 |
| 1 → 0 | 39/82 | 48/556 | 6/15 |
| 1 → 2 | 34/82 | 42/556 | 3/15 |
| 2 → 0 | 39/82 | 47/556 | 6/15 |
| 2 → 1 | 43/82 | 36/556 | 8/15 |

### multiscale_regularized

| Train fold → validation fold | TP / 82 | FP / 556 | Small / 15 |
|---|---:|---:|---:|
| 0 → 1 | 79/82 | 55/556 | 14/15 |
| 0 → 2 | 75/82 | 66/556 | 11/15 |
| 1 → 0 | 73/82 | 43/556 | 9/15 |
| 1 → 2 | 78/82 | 51/556 | 12/15 |
| 2 → 0 | 74/82 | 40/556 | 8/15 |
| 2 → 1 | 77/82 | 43/556 | 13/15 |

## Reproduce

The verifier defaults are bound to this published study and its two registration snapshots. Recount the preserved local study with a new empty export directory. A completely new fit also requires newly published registration snapshots before outer inference; the original snapshots cannot match a new timestamp/path-based protocol identity. Prepare those copies in an isolated repository using the published JSON formats and preserve this original study. Source/data path setup and the parent feature cache are prerequisites; this is not a standalone fresh-machine installer.

Use new empty output/export paths and the locked recall environment. Freeze and register the supplements, publish the matching registration snapshots in that isolated copy before any outer evaluation, then run serially:

```powershell
components/.venv-p0/Scripts/python.exe -m component_inspector.research.ksdd2_small_defects --phase freeze --root components/outputs/recall-small-defects
components/.venv-p0/Scripts/python.exe components/tools/run_control_small_defects.py --phase register --root components/outputs/recall-small-defects
components/.venv-p0/Scripts/python.exe components/tools/run_repeat_small_defects.py --phase register --root components/outputs/recall-small-defects
components/.venv-p0/Scripts/python.exe components/tools/run_legacy_small_defects.py --phase register --root components/outputs/recall-small-defects
components/.venv-p0/Scripts/python.exe components/tools/run_small_defects.py --root components/outputs/recall-small-defects
components/.venv-p0/Scripts/python.exe components/tools/run_control_small_defects.py --phase run --root components/outputs/recall-small-defects
components/.venv-p0/Scripts/python.exe components/tools/run_repeat_small_defects.py --phase run --root components/outputs/recall-small-defects
components/.venv-p0/Scripts/python.exe components/tools/run_legacy_small_defects.py --phase run --root components/outputs/recall-small-defects
components/.venv-p0/Scripts/python.exe components/tools/replay_native_small_defects.py --root components/outputs/recall-small-defects
components/.venv-p0/Scripts/python.exe components/tools/verify_small_defects.py --root components/outputs/recall-small-defects --export docs/research/small-defects
```

Independent factory confirmation remains incomplete. Keep the [fresh-image confirmation plan](2026-10-20-fresh-confirmation-plan.md); do not reopen or retune the official test to manufacture a passing result.

## Next development hypothesis

First isolate training schedule with matched objectives, data, capacity, and regularization: the 30-epoch candidates and full prior recipe differ in several factors, so their comparison cannot identify a duration effect. Another useful hypothesis is native-resolution overlapping tiles with mask-aware training crops to preserve tiny marks before the backbone downsamples them. Test these as newly registered development studies with fixed normal-only calibration, not by changing thresholds on these outer predictions. The current bundled changes do not isolate resolution, dropout, capacity, Dice loss, or score pooling as individual causes. Keep single-factor controls when assessing a future gain. These additional hypotheses have not been trained in this study and are not claimed results.
