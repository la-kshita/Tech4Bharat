# Prospective three-seed ensemble: verified internal results

**The fixed ensemble does not meet the requested overall targets in every fold.**

The inherited family is **longer108**. The primary alpha remains **0.05**; alpha .08 is diagnostic and cannot replace a failed primary. Each fold requires overall defective-image recall >=95% and normal false-positive rate <10%. Small-defect counts are reported separately; overall success does not establish 95% recall within that subset.

The preceding single-model sweep passed all twelve of its own checks: **False**. Its [results](2026-10-20-warm-small-results.md) remain unchanged. This extension combines three members into one method and has six ensemble fold cases; it does not relabel failed single-model checks as successful or claim three independent ensemble repeats.

Both partitions reuse the same 1,914 previously observed KSDD2 training originals. Each partition has 246 defective and 1,668 normal images, including 45 small-defect images. Member/partition denominators are not added to manufacture an independent sample. The official test remained closed, the 417 reserved normals remained unused, and no separate factory batch exists. These are internal development and partition/initialization robustness checks, not independent factory validation or a production guarantee.

## All six primary-policy fold results

| Stage | Fold | TP / 82 | FP / 556 | Recall | Normal FPR | Small / 15 | Large / 67 | Goal |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| dev | 0 | 76/82 | 25/556 | 92.68% | 4.50% | 9/15 | 67/67 | Fail |
| dev | 1 | 80/82 | 36/556 | 97.56% | 6.47% | 14/15 | 66/67 | Pass |
| dev | 2 | 76/82 | 28/556 | 92.68% | 5.04% | 12/15 | 64/67 | Fail |
| confirm | 0 | 76/82 | 21/556 | 92.68% | 3.78% | 10/15 | 66/67 | Fail |
| confirm | 1 | 78/82 | 27/556 | 95.12% | 4.86% | 12/15 | 66/67 | Pass |
| confirm | 2 | 80/82 | 28/556 | 97.56% | 5.04% | 13/15 | 67/67 | Pass |

## Aggregates, kept separate by partition

| Stage | Primary policy | TP / 246 | FP / 1,668 | Recall | Normal FPR | Small / 45 |
|---|---|---:|---:|---:|---:|---:|
| dev | alpha .05 | 232/246 | 89/1,668 | 94.31% | 5.34% | 35/45 |
| confirm | alpha .05 | 234/246 | 76/1,668 | 95.12% | 4.56% | 35/45 |

A passing aggregate does not rescue a failed fold. Pooled raw scores from differently fitted/calibrated fold models are not used to claim a single AUROC/AP. Stored binomial intervals are descriptive under the overlapping-data setting; they are not formal exact CV guarantees.

## Same-policy member comparisons

| Stage | Member seed | Recipe | TP / 246 | FP / 1,668 | Recall | Normal FPR | Small / 45 |
|---|---:|---|---:|---:|---:|---:|---:|
| dev | 4217 | baseline84 | 234/246 | 105/1,668 | 95.12% | 6.29% | 38/45 |
| dev | 4217 | longer108 | 233/246 | 79/1,668 | 94.72% | 4.74% | 37/45 |
| dev | 4218 | baseline84 | 228/246 | 92/1,668 | 92.68% | 5.52% | 32/45 |
| dev | 4218 | longer108 | 224/246 | 74/1,668 | 91.06% | 4.44% | 31/45 |
| dev | 4219 | baseline84 | 225/246 | 92/1,668 | 91.46% | 5.52% | 32/45 |
| dev | 4219 | longer108 | 226/246 | 91/1,668 | 91.87% | 5.46% | 33/45 |
| confirm | 4220 | baseline84 | 228/246 | 63/1,668 | 92.68% | 3.78% | 34/45 |
| confirm | 4220 | longer108 | 228/246 | 66/1,668 | 92.68% | 3.96% | 33/45 |
| confirm | 4221 | baseline84 | 232/246 | 84/1,668 | 94.31% | 5.04% | 36/45 |
| confirm | 4221 | longer108 | 230/246 | 84/1,668 | 93.50% | 5.04% | 35/45 |
| confirm | 4222 | baseline84 | 236/246 | 72/1,668 | 95.93% | 4.32% | 38/45 |
| confirm | 4222 | longer108 | 235/246 | 68/1,668 | 95.53% | 4.08% | 37/45 |

These are fixed members and matched baseline controls, not alternatives from which a winning seed was selected. Per-original small/large detections gained or lost, and normal false alarms added or avoided, are retained for each member pair in [paired changes](seed-ensemble/paired-changes.json).

## Diagnostic policy: every fold, no substitution

This predeclared setting passed all six observed point targets, with recall
95.12%–97.56% and normal FPR 6.12%–8.63%. Development totals are 238/246 defects
(96.75%), 119/1,668 false alarms (7.13%), and 40/45 small catches (88.89%).
Second-partition totals are 236/246 defects (95.93%), 117/1,668 false alarms
(7.01%), and 37/45 small catches (82.22%). It is the leading development
candidate for a future sealed study, but does not replace the failed primary
qualification. In plain English: three-model averaging helps, while small defects
and generalization remain unresolved. [Decision and next evidence](2026-10-22-recall-status.md).

| Stage | Fold | TP / 82 | FP / 556 | Recall | Normal FPR | Small / 15 |
|---|---:|---:|---:|---:|---:|---:|
| dev | 0 | 79/82 | 36/556 | 96.34% | 6.47% | 12/15 |
| dev | 1 | 80/82 | 48/556 | 97.56% | 8.63% | 14/15 |
| dev | 2 | 79/82 | 35/556 | 96.34% | 6.29% | 14/15 |
| confirm | 0 | 78/82 | 34/556 | 95.12% | 6.12% | 12/15 |
| confirm | 1 | 78/82 | 39/556 | 95.12% | 7.01% | 12/15 |
| confirm | 2 | 80/82 | 44/556 | 97.56% | 7.91% | 13/15 |

## Fixed method and verification

The family was inherited from the original development-only selection. The extension was registered before that selection and before either second-partition or ensemble outcomes. Development members are 4217/4218/4219; second-partition members are 4220/4221/4222. The added members reuse the original numerical worker, training roles and schedules, with their separately published initialization seeds. All three positive-oriented image logits are sigmoid-transformed and equally averaged. No best-seed gate, fitted temperature, fusion weighting or post-result threshold is used. The complete ensemble score is calibrated once on its 200 normal parents.

The independent arithmetic/role audit checked 1914 raw mask hashes/labels, 4074 feature artifacts, 3 base-moment sets, 4704 epoch60 hard-normal scores, 2400 direct-weight calibration scores, and 0 crop presentation choices. It checked all roles, losses/usage schedules, parent checkpoint links, preserved classifier states, thresholds, component counts and equal-weight ensemble arithmetic using a separate SciPy expit calculation. All parent-study evidence bytes matched its export index.

Native replay regenerated 7656 new-member head scores and 15312 decisions from original RGB without masks or cached arrays. Combining those scores with the original study’s independently replayed native scores reproduced all 7656 ensemble decisions with zero flag mismatches; maximum score difference 2.22044604925e-16. These repeated checks do not add independent images. The audit is from the same work session, not third-party certification.

Eight new ensemble preflight tests passed; the combined focused run with the nine arithmetic-audit tests passed all 17. The original full preflight passed 131 tests with 45 upstream warnings. Timings and RSS are measured resource records, not controlled latency benchmarks. Every fit/replay/audit phase uses four threads, 1,800 seconds and min(4GiB,60% free RAM).

Final full software checks passed **148 tests**, with zero failures, errors or
skips and 45 upstream deprecation warnings. See the [exact counts](software-checks/summary.json)
and [raw log](software-checks/worker.log). These are software tests, not 148 new
model evaluations. The independent ensemble audit completed in 56.078 seconds,
peaking at 387,837,952 sampled owned-process-tree RSS bytes; its separate
[resources](2026-10-21-seed-ensemble-audit-resources.json) and
[log](2026-10-21-seed-ensemble-audit-worker.log) are preserved outside its export tree.
The [registered post-fit diagnostic](2026-10-21-fitting-gap-results.md) found
100% fitting recall in all 18 selected-family heads and lower held-out recall.
It changed no model, calibration cutoff or qualification.

The [verified local checkpoint package receipt](2026-10-21-seed-ensemble-checkpoint-package.json)
records the archive hash, exact member count and complete member/CRC checks.
The package contains the added checkpoints and the nested verified main-study
archive; original RGB and large feature arrays are excluded with provenance
preserved. Checkpoint weights remain local rather than uploaded in Git.

The original standby interruption and exact recovery remain documented in the [main study](2026-10-20-warm-small-results.md). No model-outcome-based retry or seed removal occurs in this extension. Generalization beyond observed KSDD2 images still requires the [fresh factory confirmation plan](2026-10-20-fresh-confirmation-plan.md).

See the [prospective protocol](2026-10-21-seed-ensemble-protocol.md) for the original Deep Ensembles paper and industrial ensemble precedent. This is an adaptation of those ideas; their published accuracy is not ours.

Protocol identity: `1b85991799852e768a2010d1910e9f00a663da7916be8b423a8642527ba8712d`. Numerical source commit: `9455b1ff9daa2657c2e9f6aa1240cd6dd8117b6e`. Source/configuration and new member protocols were published before scoring or added fitting. Evidence: [preregistration](2026-10-21-seed-ensemble-preregistration.json), [summary](seed-ensemble/audit-summary.json), [verification](seed-ensemble/verification.json), [export hash index](seed-ensemble/export-index.json).

## Reproduce the preserved audit

Use the locked environment, preserved originals/caches, member checkpoints and main-study proof. Use a new empty export path; do not overwrite archived fit or proof directories. New fitting needs new paths and a newly published registration. Checkpoints and arrays remain local with recorded hashes.

```powershell
components/.venv-p0/Scripts/python.exe components/tools/verify_seed_ensemble.py --root C:/Users/rehtr/Downloads/tech4bharat/components/outputs/recall-seed-ensemble --export docs/research/seed-ensemble-audit-repeat
```
