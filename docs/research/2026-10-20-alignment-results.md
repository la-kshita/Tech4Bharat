# Training/scoring alignment after the recall study

Completed 20 October 2026. **All three corrected initialization runs meet the requested development target.** The corrected continuation catches **80/82 defects with 24/417 normal false alarms (97.56% recall / 5.76% FPR)**. The earlier saved checkpoint remains the numerical best at 97.56% / 5.28%, but its original training recipe was unstable when retrained. This follow-up fixes that training/scoring conflict and preserves all results.

**Later validation:** the frozen corrected continuation reached only **100/110
defects (90.91% recall), with 69/894 normal false alarms (7.72%)**, on official
test images excluded from fitting. All three cold-start seeds also missed the
95% target. These development numbers did not establish test performance. See
the [full validation and overfitting audit](2026-10-20-validation-results.md).

## Why another experiment was necessary

The [remaining-strategy study](2026-10-20-round2-results.md) reached 80/82 detected defects and 22/417 normal false alarms (97.56% recall / 5.28% FPR) with focal training, localized scoring and hard-normal refinement. A full repeated training run with seed 4218 also caught 80/82, but seed 4219 collapsed to 3/82 with 42/417 false alarms. Two good runs did not establish a reliable training recipe.

The failed checkpoint reproduces all 916 stored calibration/evaluation scores exactly. Across the three seeds, channel normalization tensors are identical, feature-cache paths match and model weights are finite. The failed seed's learned image-classifier coefficient on the maximum local defect logit is -1.148, while successful seeds learn positive coefficients. This permits the image-classification objective to encourage decreasing the strongest local score on a defective image, even though the inference rule interprets larger localized scores as more defective. Background-logit shortcuts can fit training labels while conflicting with localized inference.

Ranked diagnosis: (1) learned-classifier/scoring conflict; (2) normalization or artifact mismatch; (3) insufficient localization learning. Targeted probes exclude the normalization mismatch and show the sign conflict; the corrective experiments test its practical effect. The original bad result remains in the evidence and is not discarded as an inconvenient seed.

## Correction and fixed experiment

The image loss now directly uses **BCE of the same top-0.1%-mean segmentation logit used at inference**. Weighted focal pixel supervision remains gamma 2, positive weight 8. The unused learned classifier is frozen and contributes neither to training nor inference. This removes its ability to reverse the meaning of the localization score.

Everything else follows the recorded recipe: frozen ResNet18 layers 2/3; 384-channel 32x80 cached features with a float16 roundtrip; 128 original normals plus 164 labeled defects; 60 base epochs with AdamW learning rate 0.001; training-only mining of 128 additional normals from 1,123 eligible originals; 12 refinement epochs at learning rate 0.0002. Batch size is 4, weight decay 0.0001 and gradient clipping 5. Normalization uses only the original fitting images. No lighting augmentation is added.

Seeds **4217, 4218 and 4219** were recorded before fitting. Seed 4217 remains canonical; both repeats are always reported and cannot silently replace it. Each trained model gets its own normal-only rank cutoff at alpha 0.10 using the same 417 reserved normals. Evaluation remains the same 82 defects and 417 normals, all from the provider's training split. No official test image enters this study. These outcomes are continued selection on already observed development data, not independent confirmation.

## Regression checks before full training

A test at the actual objective seam first failed under the old learned-classifier loss: flipping an unused classifier's sign changed the loss and segmentation gradients. It passed after using the aligned objective. Additional tests verify that positive-image loss pushes the selected valid peak upward, padding receives no score gradient, and differentiable training pooling exactly matches the existing inference pooling. The full suite passed **116 tests** with 45 pinned-backend deprecation warnings in 28.72 seconds.

The corrected full three-seed training experiment also passes the practical target, including the previously failing seed. A software property test alone would not establish that result. Original-image parity and record audits supplement the measured experiment.

## Observed canonical result

The predeclared corrected seed4217 detects **79/82 defects** and falsely flags **32/417 normals**: **96.34% recall / 7.67% FPR**, with 100% technical coverage. Its cutoff, derived only from the 417 calibration normals, is -1.7987803221 with strict `score > cutoff`. The raw score is logit-derived and is not a calibrated probability. Its numerical scale differs from the old model, so the lower-looking cutoff is not an evaluation-driven relaxation of the old threshold.

| Comparable development result | Detected / missed defects | Normal false alarms | Recall | FPR |
|---|---:|---:|---:|---:|
| Reproduced PaDiM baseline | 60 / 22 | 28 / 417 | 73.17% | 6.71% |
| Previous round 1 supervised candidate | 77 / 5 | 27 / 417 | 93.90% | 6.47% |
| Round 2 best single checkpoint, unstable training recipe | 80 / 2 | 22 / 417 | 97.56% | 5.28% |
| Corrected canonical objective | 79 / 3 | 32 / 417 | 96.34% | 7.67% |

The corrected canonical run meets the requested operating target, but does not outperform the old best single checkpoint. The correction removes the reproduced failure mechanism and the target survives the same fixed repeat seeds.

| Corrected fixed seed | Found / missed defects | Normal false alarms | Recall | FPR |
|---|---:|---:|---:|---:|
| 4217, predeclared canonical | 79 / 3 | 32 / 417 | 96.34% | 7.67% |
| 4218 | 78 / 4 | 29 / 417 | 95.12% | 6.95% |
| 4219, previously collapsed | 78 / 4 | 30 / 417 | 95.12% | 7.19% |

The corrected recipe reaches **95.12%–96.34% recall at 6.95%–7.67% FPR** across all three fixed initializations. The previous failed seed improves from 3/82 to 78/82 detections. This supports the training/scoring alignment correction for this observed split; it does not prove all future seeds or new factory data will meet 95%. Repeated runs use the same images, so their denominators must not be pooled as independent validation.

The canonical corrected checkpoint was frozen before either corrected repeat was evaluated: SHA-256 `0392b0ffd689c1915ef44b719c21370cd8c2946ce9e52fa009d368d6db92d2ac`. This is a new artifact; no old checkpoint or failed result was overwritten.

## One additional fixed-checkpoint refinement

Before seeing the final corrected repeat, predeclare one additional test of the earlier best saved checkpoint (`round2/hard_refined`, SHA-256 `180084ca4586143aa707d83a11232f130f917a612e7acd94a07776ba16c9c3fe`). Keep its existing 420 fitting IDs and normalization; use the corrected aligned objective for 12 more epochs, fresh AdamW learning rate 0.0002, batch size 4, seed4217. No new mining, images, feature geometry or threshold rule is introduced. The model and original result stay separate.

This asks whether the objective correction can improve the highest measured checkpoint, rather than relying only on fresh initialization. It is one fixed-checkpoint continuation, not an additional independent from-scratch seed or a clean isolation of objective effects from extra training. The protocol and dependency hashes were recorded before fitting.

**Measured result:** 80/82 defects detected, 24/417 normal false alarms, **97.56% recall / 5.76% FPR**. It preserves exactly the same 80 catches as the old best checkpoint. Four old false alarms disappear and six different normals become false flags, a net increase of two. This does **not** improve on the old 22-false-alarm numerical best. Retain both checkpoints, without pretending the correction won every operating measure.

## Verification, latency and decision

All three fixed runs and the additional continuation passed the saved-record audit: exact metric recount, complete common evaluation membership, 417 normal-only calibration IDs, strict threshold flags, training-only fitting/mining IDs, fixed 60/12-epoch histories, finite states and matching artifact/source hashes. The learned classifier stays at initialization in the corrected cold runs and stays unchanged from its source in the continuation. Its weights cannot drive training.

| Actual image-processing check | Warm p95 per image | Maximum score difference from saved evaluation |
|---|---:|---:|
| Earlier numerical best checkpoint | 0.275 s | 0 |
| Corrected canonical checkpoint | 0.232 s | 0 |
| Corrected saved-checkpoint continuation | 0.239 s | 0 |

Each check uses eight fixed development normals and eight fixed defects, not selected by scores. It includes source hashing, image decode, geometry, pretrained backbone, cache-equivalent precision and final scoring on CPU with batch size 1. Every checked flag also matches. These small local timing samples exclude UI, network and cameras; cached-head-only time is not being presented as full inference latency.

Corrected full training/refinement/scoring phases took 481.75 s, 483.97 s and 451.16 s, with sampled peak owned-worker-tree RSS of approximately 0.50–0.52 GiB. The extra continuation took 113.78 s with a 0.48-GiB peak. Every worker completed within its recorded memory/time cap. Native inference peaks were approximately 0.54–0.55 GiB. All timings depend on machine load.

**Decision:** preserve the 97.56% / 5.28% checkpoint as the best numerical result on this development set. Use the aligned objective for future retraining: it survived all three tested initializations at 95.12%–96.34% recall and under-10% FPR. The corrected continuation is a 97.56% / 5.76% alternative with aligned final training. Training instability is different from inference randomness: the older fixed checkpoint still reproduces its own scores exactly.

The next useful step is to freeze candidate artifacts and evaluate once on new untouched images, including edge defects, low contrast and changing illumination. Further tuning on these same 82 observed defects cannot establish factory recall. The corrected canonical run's descriptive Wilson 95% intervals are 89.79%–98.75% recall and 5.49%–10.63% FPR; selection and shared physical origins prevent treating these as a formal post-selection production guarantee. The earlier lighting/blur study tested other image heads, so it does not establish corrected-model robustness. The original archived Stage 5 no-go and cable-gland scope remain unchanged.

**Plain English:** the best saved model now misses two of these 82 defective items instead of five in the previous supervised result, and wrongly flags about five of every 100 good items. We also caught a training setup that could fail badly on retraining and corrected it; the repaired setup met the goal in all three fixed runs. These are development results, not proof from new factory items.

Evidence: [decision record](alignment/candidate-decision.json), [all fixed-seed results](alignment/stability-summary.json), [metric/split verification](alignment/verification.json), [software checks](alignment/software-checks.json), [corrected canonical predictions](alignment/aligned.json), [continuation predictions](alignment/warm_aligned.json), [native canonical inference](alignment/aligned-live-benchmark.json), [native continuation inference](alignment/warm_aligned-live-benchmark.json), [earlier best native inference](alignment/legacy-best-live-benchmark.json), [complete export index](alignment/summary.json). Raw images, feature arrays and model tensors stay local and ignored; exact score records, metadata and rebuilding code are tracked.

## Reproduce

Recreate the [round 1 data, feature cache and pinned environment](2026-10-19-round1-results.md#reproduction-commands), then run from the repository root in PowerShell:

```powershell
components/.venv-p0/Scripts/python.exe components/tools/run_recall_alignment.py --root components/outputs/recall-alignment --parent components/outputs/recall-round1

# Optional continuation; first recreate the round 2 saved checkpoint using its report.
components/.venv-p0/Scripts/python.exe -m component_inspector.research.ksdd2_warm_alignment --phase prepare --root components/outputs/recall-alignment --prior components/outputs/recall-round2
components/.venv-p0/Scripts/python.exe -m component_inspector.research.ksdd2_warm_alignment --phase train --root components/outputs/recall-alignment

components/.venv-p0/Scripts/python.exe components/tools/verify_recall_alignment.py --root components/outputs/recall-alignment
components/.venv-p0/Scripts/python.exe components/tools/benchmark_recall_round2.py --root components/outputs/recall-alignment --weights .cache/recall-downloads/model.safetensors --method aligned
components/.venv-p0/Scripts/python.exe components/tools/benchmark_recall_round2.py --root components/outputs/recall-alignment --weights .cache/recall-downloads/model.safetensors --method warm_aligned
components/.venv-p0/Scripts/python.exe components/tools/summarize_recall_round2.py --root components/outputs/recall-alignment --output-directory docs/research/alignment-replication
```

Use a new empty run directory for a changed experiment. Each worker runs alone with four CPU threads, at most min(4 GiB, 60% available RAM), and a 1,800-second phase cap. All completed and failed attempts remain recorded. Raw datasets, feature arrays and trained tensors are local ignored artifacts; source code, configurations, exact score rows and compact evidence are tracked.

The recorded continuation was supervised with `run_bounded`; its direct optional worker commands above reproduce the calculation without that controller's RSS/time enforcement. The three-seed runner includes supervision automatically.
