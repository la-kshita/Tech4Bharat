# Recall experiment ledger

Created 19 October 2026; updated 22 October 2026. **The verified warm-recipe sweep fails the across-fold consistency target.** The fixed rule selected `longer108` at alpha .08. Across seeds4217/4218/4219 it caught **237/232/228 of 246 defects (96.34%/94.31%/92.68%)**, with **134/145/117 false alarms among 1,668 normals** and **39/36/33 small catches among 45**. Its worst fold reached 90.24% recall. Five of nine development cases passed. On the locked second partition it caught 234/246 (95.12%) with 110/1,668 false alarms (6.59%), but only two of three folds passed. Pooled success cannot rescue the failing fold. See [full results](2026-10-20-warm-small-results.md).

All five methods and both predeclared policies are reported. Size weighting and zoomed crops did not meet every fold/seed target. Independent verification checked 6,234 feature artifacts, 1,914 raw GT masks, six normalization sets, 7,056 mining scores, 10,200 direct-weight calibration scores, 181,440 view choices and all 102 metric records. Every one of 65,076 raw-RGB replay decisions matched exactly. The original [standby interruption and exact recovery](2026-10-21-warm-small-recovery.md), [publication-check repair](2026-10-21-publication-check-attempt-1.json), [complete target retention](2026-10-21-warm-small-target-visibility.json) and [partial boundary exclusions](2026-10-21-warm-small-target-boundaries.json) remain documented. No learned artifact or threshold changed during audit/report repairs.

The [prospective three-seed average](2026-10-21-seed-ensemble-results.md) completed with fixed seeds4221/4222 added to seed4220 on the second partition. Its family-selection rule, equal-weight sigmoid score, primary alpha .05 and diagnostic alpha .08 were published before main selection, second-partition predictions or any ensemble scoring. Primary alpha .05 passed three of six folds: pooled 232/246 defects with 89/1,668 false alarms in development and 234/246 with 76/1,668 in the second partition. **Primary qualification failed.** Predeclared diagnostic alpha .08 passed all six point targets (95.12%–97.56% recall, 6.12%–8.63% FPR), with 40/45 and 37/45 small catches. It remains diagnostic, not a substituted primary success. Independent RGB replay reproduced all 7,656 ensemble decisions; role/mining/score audits passed. The [registered fitting-gap diagnostic](2026-10-21-fitting-gap-results.md) found 100% fitting recall and zero fitting false alarms in all 18 selected-family heads, while held-out recall was 87.80%–98.78% at alpha .05 and 90.24%–98.78% at .08. This gap does not uniquely identify its cause. All 148 final software tests passed with 45 upstream warnings. The official test remains closed; no independent new factory data exists. See the [plain-English decision](2026-10-22-recall-status.md).

The preceding completed study found no verified improvement from seven fixed small-defect methods. Its nested procedure caught **231/246 defects (93.90%) with 144/1,668 false alarms (8.63%)**. Both predeclared seed repeats also caught 231/246. The full prior recipe averaged 236/246 (95.93%) and 147/1,668 (8.81%), but failed a target in two of three folds. These are previously observed official-training images; no new factory confirmation is available. See [full small-defect results](2026-10-20-small-defects-results.md).

The earlier frozen benchmark primary scored **100/110 defects (90.91%) and 69/894 normal false alarms (7.72%)**. All three corrected cold-start seeds also failed recall (90.00%–91.82%). This official test set stayed closed in the new study. See [validation results](2026-10-20-validation-results.md).

Baseline, bounded R01–R10 review and alignment correction are complete. Earlier development results were: best numerical checkpoint **97.56% / 5.28%** (unstable original retraining); three corrected fixed seeds **95.12%–96.34% / 6.95%–7.67%**; one corrected continuation **97.56% / 5.76%**. Those development targets did not generalize to the official test. R07 is deferred with a task-specific reason. See [round 1](2026-10-19-round1-results.md), [round 2](2026-10-20-round2-results.md) and [alignment correction](2026-10-20-alignment-results.md).

Research and source rationale: [ten strategies](2026-10-19-recall-strategies.md). Saved-score-only diagnostic: [audit JSON](2026-10-19-observed-score-audit.json).

## S01–S07 completed training-only study

Status **completed; new procedure fails the point target and does not beat the paired control**. Frozen main source `de6ad3d2a32a50de7be94bbcaf9b19823382469a`; protocol `b7ce2f1f5dff4c5e66cf47e8f1bf44331e3a2bd00a33c191e6bcac9e2a28698f`. All seven candidates completed three inner fits and six directed inner comparisons. These overlapping inner comparisons are development observations, not independent samples.

| Attempt | Fixed method | Recorded disposition |
|---|---|---|
| S01 | Corrected coarse supervised control, 30 epochs | Paired outer control: 232/246 defects, 132/1,668 false alarms, 36/45 small defects; fails recall |
| S02 | Coarse head with smaller capacity, dropout, weight decay and Dice loss | Selected in two outer folds; neither met both targets |
| S03 | Fine layer1/2 supervised features and top-three-cell score | Final inner-selected candidate; no independent performance test of that final checkpoint |
| S04 | Fine multiscale layer1/2/3 supervised features | Selected in outer fold 0; 74/82 defects and 37/556 false alarms, 7/15 small defects; fails recall |
| S05 | Compact fine PatchCore-style memory | All six inner comparisons fail recall; bounded configuration, not paper-scale PatchCore |
| S06 | Normal-only PCA reconstruction | All six inner comparisons fail recall; linear control, not full DFR |
| S07 | Native grayscale bright/dark top-hat morphology | All six inner comparisons fail recall; no learned defect labels |

The selected procedure caught 34/45 small defects; 11 of its 15 misses were small. Fitting recall was 100% for all supervised inner heads, while held-out results were weaker. This demonstrates a generalization gap without identifying its unique cause. No epochs, thresholds, methods, or seeds were selected on outer or official-test outcomes.

Matched refit offsets +1000 and +2000 retained the selected methods and exact fit/calibration memberships: both caught 231/246, with 129 and 158 false alarms. The full 60+12+12 prior-recipe reference caught 236/246 with 147 false alarms at fixed alpha .08; at the separately fixed historical .10 policy it caught 239/246 but made 175/1,668 false alarms (10.49%, fails). No favorable repeat or policy replaced the primary or final candidate.

Every one of 11,484 model/threshold evaluations regenerated from the 1,914 original training RGB images matched exactly. Verification checked 8,148 feature artifacts and independently reconstructed 14 distinct normal-only moment sets; protected historical evidence is unchanged. The first verifier attempt had an output-schema error, and the first ZIP check had a Windows path-normalization error; both failed attempts are preserved and documented. No model or prediction changed during those repairs. All experiment rows, hashes, losses, timings, method sources, proof records and limitations are in the [study report](2026-10-20-small-defects-results.md), [export index](small-defects/export-index.json), and [checkpoint receipt](2026-10-20-small-defects-checkpoint-package.json).

Publication checks matched all exported evidence bytes against staged Git blobs. Six older source dependencies were captured with CRLF working bytes while Git stores their canonical LF content. Explicit `.gitattributes` settings now reproduce those original checkout bytes; their code and frozen working hashes were unchanged. The local archive contains the exact source bytes used by the study.

## Fixed historical comparator

| Recipe / method | TP | FN | FP | TN | Recall | Normal FPR | Run |
|---|---:|---:|---:|---:|---:|---:|---|
| KSDD2 / PaDiM | 84 | 26 | 77 | 817 | 76.36% | 8.61% | frozen-20261019T041652Z-9c825801 |
| Cable gland RGB / PaDiM | 11 | 76 | 1 | 20 | 12.64% | 4.76% | frozen-20261019T041652Z-9c825801 |

Original decision: no-go. Observed-score threshold envelopes do not change it.

## Planned attempts

| ID | Hypothesis / parent | Current status | Measured outcome | Next evidence needed |
|---|---|---|---|---|
| P0 | Reproduce baseline with exact local identities | Completed | 60/82 defects; 28/417 normal alarms | Calibration score parity passed; 73.17% recall / 6.71% FPR on the common development set |
| R01 | Real-mask supervised segmentation plus image classifier / P0 | Completed; retained parent | 77/82 defects; 27/417 normal alarms | 93.90% recall / 6.47% FPR; stronger round 2 descendant selected |
| R02 | Compact PatchCore / P0 with matched normal IDs | Completed | 66/82 defects; 32/417 normal alarms | 80.49% recall / 7.67% FPR; improves on P0 but below R01 |
| R03 | Representative normals, then larger normal pool / selected parent | Completed | 128: 67/82 defects, 32/417 alarms; 256: 65/82 defects, 42/417 alarms | 128 marginally improves R02; 256 fails the FPR target and worsens recall |
| R04 | Finer latent detail / fixed RGB geometry | Completed; reject tested variant | Raw head: 72/82 defects, 28/417 alarms; localized: 74/82, 43/417 | Neither beats the parent; localized FPR 10.31% fails budget; overlap tiling not tested |
| R05 | Focal loss and hard examples / R01 | Completed; corrected | Best old checkpoint: 80/82 defects, 22/417 alarms | Original seed4219 collapses; corrected objective removes learned-classifier conflict and survives all three tested seeds |
| R06 | Robust image aggregation / supervised maps | Completed; selected with R05 | Parent top 0.1% mean: 79/82 defects, 27/417 alarms | 96.34% / 6.47%; applying the same rule to focal/refined maps improves further |
| R07 | Conservative foreground/pose / selected parent | Deferred with reason | Normal peak diagnostic saved | KSDD2 surface texture has no justified removable background/pose transform; padding already excluded |
| R08 | Permitted illumination variation / R01 | Completed; not selected | Raw: 77/82 defects, 30/417 alarms; localized: 78/82, 26/417 | Lighting and combined training do not beat selected recipe; fixed-cutoff stress reveals bright/blur false-alarm failures |
| R09 | Channel dimension / P0 | Completed; reject | 58/82 defects, 38/417 alarms | 70.73% / 9.11%; lower recall and more false alarms than original 100 channels |
| R10 | Complementary calibrated pair / selected head + PatchCore | Completed; not selected | Initial weighted: 77/82, 21/417; refined weighted: 79/82, 22/417 | Useful initial FPR reduction, but fusion loses one catch versus refined single head with same FP count |
| R05/R06 correction | Train the exact positive-oriented localized inference score | Completed; use for future retraining | Canonical 79/82, 32/417; repeats each 78/82, with 29 and 30/417 alarms | All three meet 95%+ / under-10%; extra continuation of old best: 80/82, 24/417 |
| V01 | Frozen corrected primary plus three fixed seeds / already observed official test benchmark | Completed; target failed | Primary 100/110 defects, 69/894 alarms; seeds 99–101/110, 53–62/894 | 90.91% / 7.72% primary; all seeds below 95%; fresh independent data still required |

## Required record for every actual attempt

Append an entry rather than overwriting an earlier run. Use status `completed`, `failed`, `resource_stopped`, `inconclusive` or `deferred_with_reason`. Never populate unmeasured fields with zero. Distinguish a 30-minute pilot from a completed full training configuration.

```text
Attempt ID / UTC timestamps / method ID:
Status and reason:
Recipe and intended claim (development / observed benchmark / new confirmation):
Code commit, dirty-file hashes, environment lock and numerical versions:
Source identity, manifest hash, original/group/parent IDs and prior exposure:
Parent baseline and exactly one changed factor:
Fitting IDs, inner/outer fold IDs, seed and label access rules:
Backbone provider/revision/hash, model artifact/hash and feature/cache geometry:
Hyperparameters, actual fit duration and stopping rule:
Image score and threshold source/rule (selection versus final calibration):
TP/FN/FP/TN; rejected defects/normals; recall/FPR/precision/coverage:
AP/AUROC; small-defect/per-folder slices; fixed stress results:
Recall difference at the same FPR budget and paired uncertainty:
Cold start / warm p95 / peak process-tree RSS / artifact and cache bytes:
Score, label-join, raw-map, diagnostic gallery and resource-log paths/hashes:
Software checks actually run and outcome:
What improved, what worsened, likely cause versus untested hypothesis:
Retain / reject / next bounded test and decision rationale:
```

## Final candidate decision record

Preserve all earlier development checkpoints and the failed repeat. The aligned objective removed the demonstrated training/scoring conflict, but **the frozen corrected primary and all three cold-start seeds failed the 95% recall target on the official test benchmark**. Their 90.00%–91.82% test recall supersedes any implication that the earlier development success validated the goal. Original-image replay is exact and metric/split audits pass; those checks do not establish generalization. More training and hard examples changed together in refinement, so their net gain is not attributable exclusively to mining. Improve using training and separated development data, then obtain a genuinely new sealed batch; do not tune the now-observed official test until it passes. No factory acceptance is established and the original Stage 5 no-go remains unchanged historical evidence.

## V01 completed attempt

Frozen source commit `495ddbb`; protocol `0d4b112f51488773ef28b670de7ae6774e4c9b0821e4d327e70203f4b9a32d3e`. Primary head `ffca032083bd4c7a003c198b70b7dc5eb10c4d7dee40392fd7a81c9160f560d9`, existing strict threshold -1.684261679649353, no fitting or calibration changes. Status **completed; failed recall point and confidence targets**. All 1,004 test images scored twice in separate processes, with reverse order on replay: every score and decision equal. All 3,335 usable originals reaudited; no train/test duplicate or fitting/calibration leakage found; physical independence unknown. Primary exact 95% intervals: recall 83.92%–95.55%, FPR 6.05%–9.67%. Training/development/test recall was 100% / 97.56% / 90.91%, consistent with an optimism/generalization gap; its unique cause is unproven. The at-most-0.5%-area defect slice caught only 13/17; this is exploratory, not a tuning decision. First-pass p95 for all four heads was 0.2651 s; peak owned-process-tree RSS 0.540 GiB. All four phases completed within caps, all 126 historical archive files unchanged, 120 software tests passed. Exact package versions, lock hash, timing, source and model hashes, prediction rows, defect slices, and independent verification: [full results](2026-10-20-validation-results.md) and [export index](validation/export-index.json). No separately new labeled data is available; independent confirmation is incomplete.
