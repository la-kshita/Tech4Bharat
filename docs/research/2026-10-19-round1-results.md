# KSDD2 recall experiments: round 1

**Completed and reviewed 20 October 2026 (IST).** User authorized baseline reproduction and R01–R03 on 19 October. All five fixed candidate runs completed. **The supervised candidate is best: 93.90% recall at 6.47% normal false alarms. The requested 95%+ recall goal was not reached.**

## Predeclared comparison

Source: original KSDD2 ZIP, 853,126,555 bytes, SHA-256 `edcdb486809b24f1d17b785e30c52fafc5999554dd5fe18ddf77b61ceb6f36a8`. Frozen ResNet18 safetensors SHA-256 `80c49dee3da4822c009c5a7fe591e9223c5a2cfcf95a4067ca4dfb5a7b89c612`. Both were downloaded from the documented providers and verified before use.

The original audited manifest is unchanged. A v2 manifest uses local verified paths, retains original/appearance-group IDs and excludes every official test image before decoding. Its identity is `39fabf3c59c6b579bbefa7aba642d2a59f67003a336b1460eb75a2c230c5fd7e` (local absolute paths contribute to this identity).

| Role | Images | Use |
|---|---:|---|
| Original normal fitting subset | 128 | PaDiM, basic PatchCore and supervised negatives |
| Eligible full normal fitting pool | 1,251 | Normal-only diversity selection for R03 |
| Supervised fitting defects | 164 | R01 training masks/image labels only |
| Reserved normal calibration | 417 | Each fixed pipeline's rank threshold, alpha 0.10 |
| Evaluation normals | 417 | Same normal images for every candidate |
| Evaluation defects | 82 | Same defects for every candidate; never used for fitting or early stopping |

The split, architecture, 60 training epochs, learning rate and scoring rules were fixed before new candidate evaluation. No epoch is chosen using the evaluation group. This is **one grouped development split**, not the planned later multi-seed comparison or independent field validation. Previously observed benchmark images are not relabeled as fresh confirmation data.

## Setup and software verification

- Recreated Python 3.12.14 with CPU torch 2.10.0, torchvision 0.25.0, Anomalib 2.7.0, timm 1.0.30 and the pinned numerical dependencies. Dependency consistency check passed for 83 packages.
- Historical lock hashes for NumPy and SciPy did not match the official CPython 3.12 Windows wheels. `requirements-recall.lock` preserves package versions and corrects those two hashes after checking PyPI; the historical lock remains unchanged.
- Corrected the reproduction guide's weight URL/format: the worker needs the hash-pinned safetensors file, not the formerly suggested `.pth` file.
- The first pytest invocation hit a missing parent folder for its explicit temporary directory. Created that directory and reran the suite: 104 passed. After adding a real two-epoch streaming-training fixture, the final suite passed **105 tests, 45 upstream deprecation warnings, 29.89 s**. Tests cover test-split rejection before decode, group leakage, diversity selection, masked nearest-neighbor scores, supervised gradients/artifact parity, chunked Gaussian equivalence and exclusion of calibration/evaluation IDs from supervised fitting.
- Free RAM is lower than in the teammate's run. Baseline fitting uses disk-backed embeddings and position chunks with the same pretrained network, retained channels, sample IDs, covariance calculation, regularization and scoring. The implementation is checked against the pinned library on a fixture and against historical calibration scores on real data. It does not silently lower resolution or feature dimension.

## Results

Baseline reproduction completed. All 417 recreated calibration scores match the saved historical calibration at `atol=rtol=1e-5`; the maximum absolute score difference is 0.0002632. Save/load score difference is zero. The recreated threshold is 47.6604576 versus the historical 47.6603584. This is numerical reproduction within tolerance, not a claim of identical serialized model bytes.

On the shared **development** comparison set, the baseline detects 60/82 defects (73.17%), misses 22, falsely flags 28/417 normals (6.71%) and correctly leaves 389 normals unflagged. Coverage is 100%. Its image AUROC is 0.9186 and AP is 0.7794. This differs from historical test recall because the evaluation images differ; it is not a decline inferred from comparing percentages across different sets.

Baseline resource measurements: 7.64 s fitting, 256.95 s complete fit/calibration/evaluation phase, 1,829,490,688-byte peak process-tree RSS (1.70 GiB), warm p95 0.559 s. The disk-backed fitting method stayed within the launch-time memory cap.

### Measured comparison on the same 82 defects and 417 normals

| Experiment | Detected / missed defects | False alarms / correct normals | Recall | Normal FPR | Decision |
|---|---:|---:|---:|---:|---|
| P0 reproduced PaDiM baseline | 60 / 22 | 28 / 389 | 73.17% | 6.71% | Fixed comparator |
| R01 supervised segmentation + image head | 77 / 5 | 27 / 390 | **93.90%** | **6.47%** | Best candidate; below 95% goal |
| R02 compact PatchCore, original 128 normals | 66 / 16 | 32 / 385 | 80.49% | 7.67% | Improves on P0; below supervised |
| R03 diversity-selected 128 normals | 67 / 15 | 32 / 385 | 81.71% | 7.67% | One extra detected defect versus R02; marginal |
| R03 diversity-selected 256 normals | 65 / 17 | 42 / 375 | 79.27% | 10.07% | Reject: poorer recall and exceeds FPR limit |

All five runs scored every expected evaluation image: 100% technical coverage, zero rejects. Each model's threshold came from the same 417 reserved normal calibration images under the predeclared rank rule, not from defective evaluation labels. No threshold was lowered to manufacture a 95% result.

### How each experiment went

**R01 — teach the model what defects look like.** Trained on 164 labeled defects plus the original 128 normal fitting images. The frozen ResNet18 backbone feeds a small segmentation head and learned image classifier. This is a CPU-oriented adaptation, not a reproduction of official SuperSimpleNet. All 164 projected training masks retained positive cells. The fixed 60-epoch run decreased training loss from 0.8194 to 0.0281; epoch selection never used evaluation scores. It finds 17 more defects overall than P0 and reduces misses from 22 to 5, a 77.3% reduction in missed defects on this split. The detailed paired audit shows 19 previously missed defects gained and two previously detected defects lost; it is not a strict superset of P0 detections.

Its image AUROC is 0.9884 and AP is 0.9600. AP is not recall: the actual operating recall is 0.9390. Its 95% Wilson recall interval is 86.51%–97.37%; normal-FPR interval is 4.49%–9.26%. These are sample intervals with unknown physical-item independence, not a guarantee of field performance.

**Resource interruption and recovery.** The first R01 attempt was deliberately terminated before evaluation when system available RAM fell to approximately 0.2 GiB. Its worker log, controller reason and resource record are retained. It ran for 433.31 s with a 1.39-GiB peak; it produced no quality result. A second attempt reads batches from disk, with the same seed, data, normalization calculations, model, optimizer and 60 epochs. Its first-epoch loss exactly matches the first attempt (0.8193888286613438), supporting the unchanged training calculation. The successful attempt used a 0.47-GiB peak and completed its phase in 1,648.11 s, including 1,535.88 s fitting. It was not restarted because of accuracy or selected from multiple evaluated fits.

**R02 — compare each patch against known good patches.** A compact PatchCore-style implementation uses the same pretrained backbone, image geometry and original 128 normal fitting IDs. It detects six more defects than P0 but adds four normal false alarms. The bank is capped at 10,000 vectors; a bounded per-image candidate sample precedes projected greedy coverage. That CPU approximation is explicitly recorded and is not an exact published PatchCore reproduction. It completed under the memory/time caps but did not approach the supervised result.

**R03 — improve the variety of good examples.** Kept the PatchCore recipe fixed and compared a diversity-selected 128-image normal subset with its nested 256-image expansion. The eligible pool contains only audited normal fitting groups. Selecting 128 differently finds one additional defect with the same false-alarm count, which is too small to call a reliable improvement from one split. Expanding to 256 worsens both operational measures and fails the under-10% FPR requirement. More normal examples did not automatically make this detector better.

### Resources and native inference

| Completed phase | Wall time | Peak owned-process-tree RSS |
|---|---:|---:|
| Baseline fit, calibration and evaluation | 256.95 s | 1.70 GiB |
| Shared feature extraction | 336.89 s | 0.59 GiB |
| Supervised attempt 2, fit and evaluation | 1,648.11 s | 0.47 GiB |
| Basic PatchCore fit and evaluation | 1,027.08 s | 0.48 GiB |
| Both normal-coverage variants combined | 884.66 s | 0.51 GiB |

All completed phases stayed within their launch-time RSS and 1,800-second time caps. Timing was affected by competing resource use, so differences in run duration do not establish algorithmic speed rankings.

A separate native check loads the saved supervised model and performs actual source-hash verification, decode, geometry, backbone extraction, cache-equivalent quantization and scoring on 16 fixed development normals. It matches the saved scores **exactly**, with warm p95 **0.245 s per image**, cold pipeline 0.249 s and model initialization 4.84 s. This verifies real CPU inference rather than treating cached-head time as total latency. It is a small development sample without a web service or camera.

The same native check also matches all saved scores exactly for basic PatchCore and both coverage variants. Their warm p95 values are 0.599 s, 0.590 s and 0.605 s, respectively. All are below the five-second provisional latency screen on this small sample. [Native benchmark records](round1/summary.json) retain the fixed sample IDs and timing scope.

### Evidence and limits

- [Compact machine-readable summary](round1/summary.json) and [paired changes](round1/paired-changes.json).
- Full saved predictions and calibration scores: [P0](round1/baseline.json), [R01](round1/supervised.json), [R02](round1/patchcore.json), [R03 128](round1/coverage128.json), [R03 256](round1/coverage256.json).
- [Successful supervised resource measurement](round1/supervised-attempt2-resources.json), [interrupted attempt](round1/supervised-resources.json), [controller explanation](round1/supervised-controller-stop.json) and [wheel-hash reconciliation](round1/wheel-hash-reconciliation.json).
- [Final verification record](round1/verification.json): role separation, metric recounts, common evaluation membership and exported-prediction byte equality all passed. Historical manifests and archived benchmark files have no Git changes.
- Local full run: `components/outputs/recall-round1/`; source import: `components/outputs/recall-data/`. Trained models, raw images and feature caches remain local and ignored by Git. The portable source code/configuration, compact evidence and complete score rows are tracked.

The supervised candidate passes the original provisional 90% recall / 10% FPR point-estimate screen on this development split, but fails the user's stricter 95%+ recall target. It is the candidate to retain, not a claim of a completed reliable inspector. These are single-split development results, not multi-seed evidence, untouched confirmation or factory validation. No official test image was decoded or scored, no cable-gland experiment was run, and no new brightness/blur stress benchmark was performed in this round. The original Stage 5 no-go remains an unchanged historical result.

**Plain English:** showing the model real defect examples worked much better than only teaching it what good items look like. It now misses five of these 82 defects instead of 22, while wrongly flagging about six or seven out of 100 good items. The bigger good-example collection did not help. We made a large improvement, but have not yet reached 95% or proved it on new factory images.

## Reproduction commands

Run from the repository root in PowerShell. Use a dedicated Python 3.12 environment and the corrected lock. `uv` may be installed independently; this checkout bootstrapped it into `.cache/bootstrap/bin/uv.exe`.

```powershell
uv venv --python 3.12.14 components/.venv-p0
uv pip install --python components/.venv-p0/Scripts/python.exe -r components/requirements-recall.lock --torch-backend cpu
uv pip install --python components/.venv-p0/Scripts/python.exe --no-deps -e components

# Obtain and hash-verify the two source files using the provider links in the guide.
components/.venv-p0/Scripts/python.exe -m component_inspector.cli acquire --dataset ksdd2 --archive .cache/recall-downloads/KolektorSDD2.zip --root components/outputs/recall-data
components/.venv-p0/Scripts/python.exe -m component_inspector.research.ksdd2_round1 --phase prepare --root components/outputs/recall-round1 --raw-root components/outputs/recall-data/data/raw/ksdd2
components/.venv-p0/Scripts/python.exe components/tools/run_recall_round1.py --root components/outputs/recall-round1 --weights .cache/recall-downloads/model.safetensors
```

Use a new run directory for a changed experiment; phase completions and outputs are exclusive. Reproduction depends on provider availability and the exact recorded source bytes. Running these commands again is development replication, not new independent validation.
