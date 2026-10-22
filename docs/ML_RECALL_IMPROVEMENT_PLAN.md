# Tech4Bharat: repository understanding and recall improvement plan

19 October 2026. Inspected repository commit `f7f6750` (`main`). Planning and saved-output analysis only; no new model was fit or run. Proposed experiments below amend the original fixed-model protocol and do not alter its completed benchmark.

**Follow-up:** the teammate subsequently committed source inventories and Stage 2 manifests at `fb4a370`. The newer [ten-strategy research review](research/2026-10-19-recall-strategies.md) supersedes this document's file-availability statements and initial experiment shortlist. It includes KSDD2 score analysis using the newly available labels. Raw datasets and trained weights remain excluded. Implementation is awaiting the user's go.

**20 October update:** the user approved baseline reproduction and the first three strategies. [Round 1 results](research/2026-10-19-round1-results.md) are complete: supervised KSDD2 reaches 93.90% recall / 6.47% normal FPR on the common development split. The initial awaiting-approval status above is historical.

## Recommendation

Optimize **defect recall subject to a normal false-alarm budget**, separately for each recipe. Retain the existing provisional target of recall >=90% and normal false-positive rate (FPR) <=10%; aim for >=95% recall as a stretch. Select the highest-recall candidate that satisfies the workload and resource constraints. These are targets, not promised results.

Start the improvement cycle on **KSDD2**, which has a much stronger baseline and labeled training defects. Keep **RGB cable gland** as a separate research track. This prioritization changes execution order, not the identity of the intended mechanical-component product. Success on KSDD2 cannot qualify cable gland.

## 1. What the team is building

A narrow visual inspection assistant: choose a product recipe, upload a prescribed view, obtain an experimental anomaly score/attention map, and let a person record the final decision. Later stages add batch history, duplicate/rework accounting and reports.

The active plan supersedes the earlier simulation/PCB proposal:

1. Acquire and audit sources; preserve duplicate groups and data roles.
2. Fit a reference-image comparator and PaDiM independently per recipe.
3. Check development behavior and CPU feasibility.
4. Freeze code, data, artifacts and thresholds; evaluate and stress-test.
5. Only qualifying recipes proceed to local inference, human review, batch records, review exercise and packaging (stages 6–10).

The current ML implementation uses a frozen pretrained ResNet18, layers 1–3, 100 retained channels, and Gaussian normal statistics from 128 normal images per recipe. It does **not** fine-tune the neural network. Increasing epochs is therefore not an applicable improvement to this model.

Implemented CLI commands include `prepare`, `develop`, `spike`, `freeze`, `calibrate`, `evaluate`, `stress` and `report`. Application and web directories are scaffolds. The historical `experiment/` PCB/browser work is a different experiment.

Useful code map:

| Concern | Current files under `components/src/component_inspector/` |
|---|---|
| Sources, grouping, role allocation | `data/acquire.py`, `audit.py`, `split.py`, `prepare.py` |
| RGB geometry and original-coordinate maps | `vision/geometry.py` |
| Comparator and normal-model fitting | `vision/reference.py`, `develop.py`, `padim_adapter.py`, `spike_worker.py` |
| Normal-only threshold calibration | `evaluation/calibration.py` |
| Frozen runs, predictions, metrics, reports | `evaluation/freeze.py`, `runner.py`, `phase_worker.py`, `metrics.py`, `reports.py` |

The README, execution-plan status header, recipe status fields and some reproduction instructions lag the implementation. Use the Stage 5 evidence and frozen effective configuration to understand the completed run. Historical requests contain the teammate's absolute Windows paths; they are provenance, not directly portable commands.

## 2. What the evidence actually says

The immutable snapshot is [frozen-20261019T041652Z-9c825801](../components/historical_benchmark/frozen-20261019T041652Z-9c825801/REPORT.md). Its decision was **no-go**.

| Dataset / method | Detected defects | Recall | Normal false alarms | Normal FPR | Warm p95 |
|---|---:|---:|---:|---:|---:|
| Cable gland / reference | 3/87 | 3.45% | 1/21 | 4.76% | 0.058 s |
| Cable gland / PaDiM | 11/87 | 12.64% | 1/21 | 4.76% | 0.168 s |
| KSDD2 / reference | 74/110 | 67.27% | 75/894 | 8.39% | 0.148 s |
| KSDD2 / PaDiM | 84/110 | 76.36% | 77/894 | 8.61% | 0.362 s |

All expected test images were scored, with no technical rejects. PaDiM image AUROC was 0.731 for cable gland and 0.932 for KSDD2. Macro pixel AUROC around 0.97 does not establish successful image-level detection. Cable-gland PaDiM detected **0/22 thread defects**, 3/21 bent, 3/22 cut and 5/22 hole defects.

Stress evidence matters: KSDD2 PaDiM blur raises normal FPR to 14.09%. The RGB comparator flags essentially every KSDD2 normal under the tested brightness changes. Improving only the comparator's nominal recall would miss this robustness problem.

### Additional retrospective score analysis

Using only archived cable-gland prediction scores, with binary labels recoverable from the verified `test/good` versus defect-folder paths:

| Method | Best observed recall with <=2/21 false alarms | Fewest observed normal false alarms needed for >=79/87 detected defects |
|---|---:|---:|
| Reference | 9/87 = 10.34% | 20/21 = 95.24% |
| PaDiM | 42/87 = 48.28% | 15/21 = 71.43% |

This sweeps the archived scores with the original strict `score > threshold` comparison, including tied-score groups. It is **post hoc diagnosis**, not a replacement operating threshold, unbiased performance estimate or a new passing result. It shows a threshold alone cannot meet the original cable-gland target on these observed scores. The old no-go remains unchanged.

KSDD2 row-level labels are not included in the archived prediction rows, and the audited manifest is excluded from Git. Do not infer its labels from numeric filenames or claim its full threshold trade-off was reconstructed here.

## 3. Likely improvement opportunities

These are testable hypotheses, not established causes of the misses.

| Observation | Hypothesis | Development check |
|---|---|---|
| PaDiM uses one maximum over the entire valid image rectangle | Normal background, texture or particular regions dominate image scores | Save peak coordinates, foreground/background statistics and score maps on development images |
| Valid ROI excludes letterbox padding, not object background | Restricting irrelevant background could reduce false alarms | Compare a conservative image-derived object ROI, retaining the complete silhouette and edge margin |
| Cable images are resized from 400x400 to 256x256 | Small defect signals may be weakened by resizing and map smoothing | Compare 384x384 full frame against 256x256 on development data |
| KSDD2 is rotated 90 degrees into 256x640 | It already preserves approximately native spatial detail | Measure audited dimensions first; do not assume that upscaling supplies missing information |
| PaDiM fits position-specific statistics | Pose/appearance variation may be poorly represented | Measure normal development score versus pose/appearance; compare PatchCore |
| 100-dimensional covariance from 128 normals | Covariance estimates or retained channels may be sensitive | One bounded 50-channel comparison; verify finite distances and numerical behavior |
| KSDD2 has 246 labeled training anomalies, all currently development-only | A supervised defect head could learn known subtle defects that normal-only fitting misses | New grouped supervised train/tuning roles and a lightweight segmentation/detection candidate |

No code audit or gallery establishes that cable-gland defects are invisible in RGB. If geometry/appearance experiments remain weak, assess RGB visibility explicitly before proposing depth. Depth or a different product/view would be a separate scope decision.

## 4. Evaluation design before model changes

**Preserve the entire original benchmark.** Existing test results, examples and labels have been reviewed; those sets are now historical benchmarks. Rescoring them can support a labeled development comparison, but cannot create an untouched confirmation set.

Create a versioned v2 manifest and protocol with parent image/group IDs, acquisition identity and a record of prior exposure. Never overwrite the Stage 2 manifest or old freeze. Split originals before crops, patches or augmentations; keep all derivatives of a group together.

For KSDD2's supervised branch, partition the 246 official-training anomalies approximately 70/30 into supervised training and tuning, by audited group. Report actual counts; do not split neighboring crops independently. Assign normal training and tuning groups explicitly as well. Keep operational normal calibration disjoint from both training and model selection. Previously viewed development data remain development data even if repartitioned.

There are 1,123 KSDD2 normal `unused_fit` images in the audit summary. A new protocol can allocate some to normal-model fitting or a reserved calibration pool after checking groups and prior exposure. Current access guards deliberately prohibit using them without a manifest change. Cable gland has only five unused normal fit images and no genuine training anomalies: it cannot support the same selection design or a convincing newly reserved defect confirmation set from the existing training pool.

Use grouped inner folds of KSDD2 development/training data for candidate comparisons. Obtain out-of-fold normal and anomalous scores; compare recall at an empirically constrained FPR. Select the architecture, preprocessing and score rule there. Then fit the selected recipe and calibrate its final operational threshold once on the reserved normal pool. Do not select a model by repeatedly examining final-calibration or test performance.

New independent confirmation needs new unreviewed, nonoverlapping items from the **same intended task/capture conditions**, with verified labels and better physical-item identity. A different product category is a generalization study, not confirmation of cable-gland inspection. Rephotographs or transforms of an observed part are not new independent items. If new data are unavailable, report development gains and historical benchmark comparisons without an independent-validation claim.

## 5. Bounded experiment sequence

Complete this sequence serially. Retain the current model as the fixed comparator. Change one factor at a time, record negative results, and avoid a large architecture/hyperparameter sweep.

| Priority / experiment | Concrete change | Selection evidence and stopping rule |
|---|---|---|
| P0: reproducibility and diagnostics | Restore Python 3.12, locked CPU dependencies, audited source data and model artifacts; add a development-only evaluation command | Verify model/map parity, role access and valid map coordinates; produce score/error tables before experiments |
| P1: image score | Compare current max with one fixed local-region average (e.g. maximum 3x3 average on the feature-distance grid); use the same maps | Choose using grouped KSDD2 tuning scores; reject if small-defect recall falls. This requires raw pre-smoothing maps, not only the current returned smoothed map |
| P1: cable geometry | Compare 256 and 384 square full-frame inputs, no crop | Preserve the complete object/edge defects; retain only with stronger development diagnostics and acceptable resources; defect-recall evidence still requires appropriate data |
| P2: normal representation | Compare a compact PatchCore candidate with frozen ResNet18 mid-level features, same input geometry and normal training pool | Start with a 5% coreset and explicit bank cap of 10,000 vectors, chunk distance calculations; profile build/inference before a full run. This compact configuration is an experiment, not published PatchCore performance |
| P2: supervised KSDD2 | Frozen backbone plus a small trainable segmentation head and image score; use real training masks | Cache features for CPU training; begin with weighted BCE, validate one Dice-loss addition only if needed; balanced sampling and hard negatives from training folds only; stop if bounded CPU feasibility fails |
| P3: covariance / normal coverage | Only if diagnostics support it: 50 versus 100 PaDiM channels, or 128 versus 256 KSDD2 normal groups | Separate runs with fixed remaining settings; the 256-image branch needs explicit cap/manifest changes and a memory check |
| P3: capture robustness | One conservative brightness augmentation policy, based on training data; consider overlap tiles only if native-size detail is still the bottleneck | Original and fixed stress results both reported. Tile every region, merge overlaps, recalibrate image scores; PaDiM must retain consistent spatial semantics |

A local-region average can reject isolated peaks but can also erase a real tiny anomaly; it is not automatically better than maximum scoring. Similarly, foreground extraction and registration can hide boundary defects. They require dedicated edge/geometry checks and should be attempted only when diagnostics support them.

Budget: at most four initial KSDD2 configurations (baseline, score variant, PatchCore, supervised head) and one cable-resolution comparison. The later P3 branches are conditional, not an automatic search grid. Recheck the two leading candidates across three group-split seeds only when initial gains exceed observed variation. Seeds are stability checks, not independent confirmation.

An OR ensemble is optional only after complementary development misses are demonstrated. Normalize/fuse scores using training/development data, then calibrate the combined system once. Combining two models at separate 10% FPR thresholds does not preserve a 10% system FPR.

## 6. Metrics and acceptance

Primary reporting: detected defects / all expected defects, missed defects, normal false alarms / all normal items, and technical coverage. Technical rejects count as misses in defect recall. Keep each recipe separate.

Also report image AP, AUROC, precision, review/flag rate, per-folder recall where labels exist, small-defect strata, original versus fixed brightness/blur stress results, warm p95, cold start, peak process-tree RSS and artifact size. Save scores, truth labels in evaluation-only outputs, and raw maps for the **new** run so later analyses do not require re-inference.

Compute Wilson intervals and group bootstrap intervals with a physical-independence caveat. The existing provisional 90%/10% screens use point estimates; requiring the recall lower confidence bound to exceed 90% would be a stricter, separate criterion requiring more data. Cable gland has just 21 test normals, so each false alarm changes FPR by 4.76 percentage points.

For workload planning, at 5% defect prevalence, 90% recall and 10% normal FPR flag approximately 14% of items and give about 32% precision: `0.05*0.90 + 0.95*0.10`. This is an illustrative scenario, not measured factory prevalence or evidence that unflagged items can bypass human inspection. FPR is not the fraction of alerts that are wrong.

Maintain native CPU inference, batch size 1, no paid dependencies, warm p95 <=5 seconds and <=4 GiB owned-process-tree RSS; retain the stricter fit cap of min(4 GiB, 60% available RAM). KSDD2 P0 fitting previously approached 3 GiB. Larger full-frame PaDiM maps can grow covariance storage substantially, so profile before scaling resolution or fit count.

Reopen application stages only for a recipe supported by its declared evidence gate. If improvement is shown only on development or observed benchmarks, label the outcome an experimental research assistant; do not imply factory validation or human time savings.

## 7. Implementation handoff

Suggested work packages, in order:

1. **Portable baseline + development evaluator.** Restore the environment, rehydrate sources/artifacts, create `components/configs/protocol_v2.yaml` and an additive v2 manifest. Add explicit development scoring with labels joined only after prediction. Add score distributions and error slices. Acceptance: no calibration/test access during selection, exact used-ID logs, unchanged historical hashes.
2. **Score and model candidates.** Parameterize model/score identities rather than editing frozen P0 in place; add separate score/PatchCore/supervised-head implementations. Extend fit-role guards for supervised training without weakening normal-only PaDiM fitting. Acceptance: actual configured values in artifact metadata, valid maps, save/load parity, reproducible prediction IDs and measured resource caps.
3. **Candidate comparison.** Export one per-recipe table with grouped tuning metrics at FPR budgets of 5%, 10% and 20%; 20% is a workload comparison, not a quietly relaxed acceptance target. Select one candidate per recipe before final calibration. Acceptance: highest feasible recall, stability evidence, recorded negative results and complete run identities.
4. **Calibration and confirmation.** Freeze the chosen pipeline, use its reserved normal calibration pool, then evaluate once on a genuinely new confirmation cohort when available. Acceptance: full confusion counts/intervals/stress/resources, with the appropriate development-only or independent-confirmation label.

Planning estimate: baseline/data restoration and diagnostics 1–2 working days; bounded candidates 2–4 working days; reporting/packaging 1 day. These are engineering estimates, not training benchmarks. New representative confirmation data and any required human labels have no established availability or deadline.

This clone contains code, tests, historical scores and selected gallery images, but not the raw archives, full manifests, trained weights or original virtual environments. Only Python 3.10 was discovered on PATH during this review; supported model execution needs a separate Python 3.12 environment. No model experiment or full test-suite replication was performed here. Saved row counts and strict-threshold flags were checked for all four benchmark outputs.

## 8. Sources and scope of recommendations

- [Current execution plan](../Tech4Bharat_Codex_Execution_Plan.md), [Stage 5 report](../components/docs/STAGE5_REPORT.md), [failure analysis](../components/docs/STAGE5_FAILURE_ANALYSIS.md), [audit summary](../components/docs/STAGE2_AUDIT_SUMMARY.json), and [immutable archive](../components/historical_benchmark/README.md): project scope and recorded evidence.
- [PaDiM paper](https://arxiv.org/abs/2011.08785): pretrained patch embeddings and Gaussian normal modeling. Proposed changes are our hypotheses, not measured improvements from this paper on these data.
- [PatchCore paper](https://arxiv.org/abs/2106.08265) and [Anomalib PatchCore documentation](https://anomalib.readthedocs.io/en/latest/markdown/guides/reference/models/image/patchcore.html): representative normal patch memory and coreset modeling support trying this comparator. Published MVTec AD AUROC is not a recall promise for KSDD2 or RGB MVTec 3D-AD; verify APIs against the installed pinned version.
- [Official KSDD2 source](https://www.vicos.si/resources/kolektorsdd2/) and [authors' mixed-supervision implementation](https://github.com/vicoslab/mixed-segdec-net-comind2021): labeled training defects motivate a supervised branch. A lightweight frozen-backbone head is our CPU-oriented proposal, not a reproduction of the authors' complete model.
- [Official MVTec 3D-AD source](https://www.mvtec.com/research-teaching/datasets/mvtec-3d-ad): dataset modality context. The existing project uses RGB only and retains uncertainty about the supplied early-review archive's current release identity.

Use the actual measured recall, false alarms and task scope in the project narrative. A defensible goal is: **improve visible-defect detection while keeping human review workload measurable**. A numerical success claim comes after the result.
