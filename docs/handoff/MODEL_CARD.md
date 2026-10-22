# Frozen surface-defect ensemble

## Intended use

An experimental ML building block for human-reviewed **KSDD2-style industrial
surface images**. It flags whether an image contains a visible surface defect.
The teammate can now integrate a fixed, reproducible model into the MVP.
It does not recognize arbitrary products, diagnose defect subtypes or accept
products automatically. Cable-gland, PCB and other recipes are not supported
by this selected model.

## Exact freeze

| Field | Value |
|---|---|
| Model ID | `ksdd2-ensemble-mvp-v1` |
| Manifest | `components/model_registry/ksdd2-ensemble-mvp-v1/manifest.json` |
| Identity | `8ddcac9907af4c033cbc9ff208f452ce572ea6b251da373cf267c2a3df95f355` |
| Family | Supervised `longer108`: frozen ResNet18 features, small learned segmentation head |
| Instances | Existing second-partition fold 0, seeds 4220, 4221, 4222 |
| Image rule | Each head: top ceil(0.001 × valid feature cells) mean positive logit; ensemble: equal mean of three sigmoid scores |
| Calibration | 200 disjoint normal parents, alpha .08, rank 185, strict `score > threshold` |
| Numeric threshold | **0.33108379983608666** |
| Preprocessing | RGB, 90° clockwise, aspect-preserving letterbox H256×W640; ImageNet normalization once |
| Features | layer2 + nearest-upsampled layer3; 3×3 average; C384×32×80; original float16 rounding preserved |
| Deployment target | Local CPU, four math threads, one request at a time |

The first-numbered fold was frozen for handoff without refitting or choosing
the highest-performing fold. The exact checkpoint bytes and saved calibration
cutoff are preserved. It is not a new model trained on the entire pool.
SHA-256 and size checks protect the manifest, three heads and standard backbone.
The trained classifier branch is retained in the checkpoints but unused by
the selected positive-oriented image score.

## Measured evidence

| Population | Recall | Normal false alarms | Small defects |
|---|---|---|---|
| This exact frozen fold instance | 78/82 = **95.12%** | 34/556 = **6.12%** | 12/15 = **80.00%** |
| Procedure, development aggregate | 238/246 = **96.75%** | 119/1,668 = **7.13%** | 40/45 = **88.89%** |
| Procedure, second-partition aggregate | 236/246 = **95.93%** | 117/1,668 = **7.01%** | 37/45 = **82.22%** |

The diagnostic .08 procedure met overall recall >=95% and normal FPR <10% in
all six internal folds, with per-fold recall 95.12%–97.56% and FPR 6.12%–8.63%.
The original predeclared **primary .05** setting passed only three of six
folds. The .08 diagnostic was not substituted as a successful primary result.
Its selection for the MVP is a disclosed development decision made after
those results, not a new independent qualification.

The two partitions reuse the same 1,914 previously observed official-training
originals: 246 defective and 1,668 normal, including 45 small defects. Do not
add denominators across members or partitions. Small means GT defect area
<=0.5% of source image; this is an evaluation label, never an inference input.
These numbers concern defective **images**, not every defect region/pixel.

## Limitations and interpretation

- **Small defects remain difficult.** The second-partition procedure caught
  37/45 small-defect images, below 95%; the frozen instance caught 12/15.
- **Generalization remains unproven.** All selected heads caught 100% of their
  fitting defects but missed held-out defects. A measured fitting/held-out gap
  remains; no proof of freedom from overfitting is claimed.
- **No fresh factory validation.** All current data were previously observed.
  The official test stayed closed during this batch and handoff. No separate
  factory batch exists. Earlier official-test failures remain in the research
  record and do not become successful results for this new ensemble.
- **Scores are not probabilities.** Neither score nor overlay is calibrated
  defect probability or severity. Attention localization has not been qualified.
- **Domain and operating conditions matter.** Lighting, focus, camera, material
  and product changes may affect results. Dimension/exposure warnings are
  provisional diagnostics, not validated automatic rejection criteria.
- **Human review is required.** A below-cutoff result does not certify a product,
  lot or shipment. No human time-saving or real buyer acceptance was measured.

KSDD2 training-data use and attribution are recorded in
`components/docs/DATA_SOURCES.md` (CC BY-NC-SA 4.0). The pretrained backbone is
`timm/resnet18.a1_in1k`, pinned revision
`491b427b45c94c7fb0e78b5474cc919aff584bbf`; its provider metadata/terms remain
separate. This is a research MVP handoff, not a claim of commercial clearance.
Original images are not redistributed in the model registry.

## Evidence and change policy

[Every ensemble fold](../research/2026-10-21-seed-ensemble-results.md),
[fitting-gap diagnostic](../research/2026-10-21-fitting-gap-results.md),
[portable adapter replay](portable-model-parity.json), and
[fresh-image confirmation plan](../research/2026-10-20-fresh-confirmation-plan.md).

App integration should preserve this manifest and score rule. A changed
threshold, feature geometry, aggregation or new training run is a new model
version, with its own documented selection and evaluation. The current
official test must not become a threshold-tuning set.
