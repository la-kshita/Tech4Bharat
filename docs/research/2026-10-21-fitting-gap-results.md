# Fitting-versus-held-out diagnostic

**Fitting performance is not validation. None of these fitting scores changes the selected model, threshold or qualification.**

Each member uses its original normal-only calibration cutoff. Fitting sets contain 164 defects (30 small) and 256 normals; held-out folds contain 82 defects (15 small) and 556 normals. Originals repeat across members and folds; these denominators are never added as independent samples.

A fitting/held-out gap describes generalization behavior without uniquely distinguishing overfitting, inadequate features, label noise or distribution differences. A small gap would not prove freedom from overfitting or independent factory generalization. No further model fitting or threshold tuning follows from this diagnostic.

| Alpha | Fitting recall range across 18 heads | Held-out recall range |
|---:|---:|---:|
| 0.05 | 100.00% to 100.00% | 87.80% to 98.78% |
| 0.08 | 100.00% to 100.00% | 90.24% to 98.78% |

| Stage | Seed | Fold | Alpha | Fitting TP / 164 | Held-out TP / 82 | Fitting FP / 256 | Held-out FP / 556 | Fitting small / 30 | Held-out small / 15 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dev | 4217 | 0 | 0.05 | 164/164 | 79/82 | 0/256 | 26/556 | 30/30 | 12/15 |
| dev | 4217 | 0 | 0.08 | 164/164 | 80/82 | 0/256 | 45/556 | 30/30 | 13/15 |
| dev | 4218 | 0 | 0.05 | 164/164 | 74/82 | 0/256 | 30/556 | 30/30 | 7/15 |
| dev | 4218 | 0 | 0.08 | 164/164 | 76/82 | 0/256 | 49/556 | 30/30 | 9/15 |
| dev | 4219 | 0 | 0.05 | 164/164 | 74/82 | 0/256 | 26/556 | 30/30 | 8/15 |
| dev | 4219 | 0 | 0.08 | 164/164 | 74/82 | 0/256 | 33/556 | 30/30 | 8/15 |
| dev | 4217 | 1 | 0.05 | 164/164 | 79/82 | 0/256 | 36/556 | 30/30 | 13/15 |
| dev | 4217 | 1 | 0.08 | 164/164 | 79/82 | 0/256 | 50/556 | 30/30 | 13/15 |
| dev | 4218 | 1 | 0.05 | 164/164 | 77/82 | 0/256 | 27/556 | 30/30 | 13/15 |
| dev | 4218 | 1 | 0.08 | 164/164 | 78/82 | 0/256 | 55/556 | 30/30 | 14/15 |
| dev | 4219 | 1 | 0.05 | 164/164 | 77/82 | 0/256 | 29/556 | 30/30 | 12/15 |
| dev | 4219 | 1 | 0.08 | 164/164 | 77/82 | 0/256 | 36/556 | 30/30 | 12/15 |
| dev | 4217 | 2 | 0.05 | 164/164 | 75/82 | 0/256 | 17/556 | 30/30 | 12/15 |
| dev | 4217 | 2 | 0.08 | 164/164 | 78/82 | 0/256 | 39/556 | 30/30 | 13/15 |
| dev | 4218 | 2 | 0.05 | 164/164 | 73/82 | 0/256 | 17/556 | 30/30 | 11/15 |
| dev | 4218 | 2 | 0.08 | 164/164 | 78/82 | 0/256 | 41/556 | 30/30 | 13/15 |
| dev | 4219 | 2 | 0.05 | 164/164 | 75/82 | 0/256 | 36/556 | 30/30 | 13/15 |
| dev | 4219 | 2 | 0.08 | 164/164 | 77/82 | 0/256 | 48/556 | 30/30 | 13/15 |
| confirm | 4220 | 0 | 0.05 | 164/164 | 72/82 | 0/256 | 24/556 | 30/30 | 9/15 |
| confirm | 4220 | 0 | 0.08 | 164/164 | 76/82 | 0/256 | 49/556 | 30/30 | 11/15 |
| confirm | 4221 | 0 | 0.05 | 164/164 | 74/82 | 0/256 | 23/556 | 30/30 | 10/15 |
| confirm | 4221 | 0 | 0.08 | 164/164 | 77/82 | 0/256 | 48/556 | 30/30 | 11/15 |
| confirm | 4222 | 0 | 0.05 | 164/164 | 76/82 | 0/256 | 22/556 | 30/30 | 11/15 |
| confirm | 4222 | 0 | 0.08 | 164/164 | 78/82 | 0/256 | 52/556 | 30/30 | 12/15 |
| confirm | 4220 | 1 | 0.05 | 164/164 | 78/82 | 0/256 | 21/556 | 30/30 | 12/15 |
| confirm | 4220 | 1 | 0.08 | 164/164 | 79/82 | 0/256 | 28/556 | 30/30 | 12/15 |
| confirm | 4221 | 1 | 0.05 | 164/164 | 77/82 | 0/256 | 29/556 | 30/30 | 12/15 |
| confirm | 4221 | 1 | 0.08 | 164/164 | 78/82 | 0/256 | 44/556 | 30/30 | 12/15 |
| confirm | 4222 | 1 | 0.05 | 164/164 | 78/82 | 0/256 | 28/556 | 30/30 | 12/15 |
| confirm | 4222 | 1 | 0.08 | 164/164 | 78/82 | 0/256 | 43/556 | 30/30 | 12/15 |
| confirm | 4220 | 2 | 0.05 | 164/164 | 78/82 | 0/256 | 21/556 | 30/30 | 12/15 |
| confirm | 4220 | 2 | 0.08 | 164/164 | 79/82 | 0/256 | 33/556 | 30/30 | 13/15 |
| confirm | 4221 | 2 | 0.05 | 164/164 | 79/82 | 0/256 | 32/556 | 30/30 | 13/15 |
| confirm | 4221 | 2 | 0.08 | 164/164 | 81/82 | 0/256 | 58/556 | 30/30 | 14/15 |
| confirm | 4222 | 2 | 0.05 | 164/164 | 81/82 | 0/256 | 18/556 | 30/30 | 14/15 |
| confirm | 4222 | 2 | 0.08 | 164/164 | 81/82 | 0/256 | 44/556 | 30/30 | 14/15 |

## Ensemble common fitting positives

Only common fitting positives are summarized. Members have different mined normal sets, so no ensemble fitting-normal FPR is claimed. These are the fixed equal-weight scores and existing joint thresholds, not best-member choices or a replacement for held-out outcomes.

| Stage | Fold | Alpha | Fitting TP / 164 | Held-out TP / 82 | Fitting small / 30 | Held-out small / 15 |
|---|---:|---:|---:|---:|---:|---:|
| dev | 0 | 0.05 | 164/164 | 76/82 | 30/30 | 9/15 |
| dev | 0 | 0.08 | 164/164 | 79/82 | 30/30 | 12/15 |
| dev | 1 | 0.05 | 164/164 | 80/82 | 30/30 | 14/15 |
| dev | 1 | 0.08 | 164/164 | 80/82 | 30/30 | 14/15 |
| dev | 2 | 0.05 | 164/164 | 76/82 | 30/30 | 12/15 |
| dev | 2 | 0.08 | 164/164 | 79/82 | 30/30 | 14/15 |
| confirm | 0 | 0.05 | 164/164 | 76/82 | 30/30 | 10/15 |
| confirm | 0 | 0.08 | 164/164 | 78/82 | 30/30 | 12/15 |
| confirm | 1 | 0.05 | 164/164 | 78/82 | 30/30 | 12/15 |
| confirm | 1 | 0.08 | 164/164 | 78/82 | 30/30 | 12/15 |
| confirm | 2 | 0.05 | 164/164 | 80/82 | 30/30 | 13/15 |
| confirm | 2 | 0.08 | 164/164 | 80/82 | 30/30 | 13/15 |

The registered forward scorer and independent convolution/pooling agreed within 1e-5 on all 7560 fitting scores, with identical flags at all 15120 individual cutoff decisions. Checkpoint hashes were unchanged. No official test image was accessed; all existing previously-observed-data limitations still apply.

Evidence: [raw predictions](fitting-gap/diagnostic.json), [export hashes](fitting-gap/export-index.json), [prospective diagnostic](2026-10-21-fitting-gap-protocol.md), [source hash](2026-10-21-fitting-gap-preregistration.json).
