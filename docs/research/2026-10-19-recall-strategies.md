# Ten recall improvement strategies for Tech4Bharat

Research completed 19 October 2026 against upstream commit `fb4a370ff5c77901e27c4024734527d99cca8824`. Fork: [RAK2315/Tech4Bharat](https://github.com/RAK2315/Tech4Bharat). Working branch: `research/recall-improvement`.

**Status: research complete; model implementation and training await the user's go.** This review adds documentation and analyzes already saved outputs. No new inference, fitting, dataset download, numerical environment installation or operational threshold change was performed.

**20 October update:** the user subsequently approved baseline reproduction and R01–R03. Those runs are now complete; see the [measured results](2026-10-19-round1-results.md). The status and approval checkpoint above/below describe this original research snapshot, not a new approval requirement for the already completed work.

## Decision and target

Prioritize **supervised KSDD2 learning**, with **PatchCore as the strongest first normal-only comparator**. Improve the cable-gland RGB recipe separately; its failures are much larger and it lacks labeled training defects. KSDD2 success will not qualify cable-gland inspection.

Rank methods by evidence on the same task, fit to repository failure modes, available labels, CPU feasibility and how easily an experiment can isolate the effect. The confidence labels below express engineering judgment, not measured probabilities. No paper establishes the probability that our project will pass the gate, and a ten-method list cannot promise ten improvements.

The decision objective is highest defect recall **subject to normal FPR <=10%**, with recall >=90% as the initial gate and >=95% as the stretch. Higher precision and lower review burden break close ties; then prefer lower latency and memory. Compare FPR <=5% as an additional operating scenario, without trading away the primary gate. FPR is the fraction of normal items falsely flagged; specificity is `1 - FPR`. AP/AUROC and recall are different metrics.

## Updated repository findings

- The fork contains the teammate's new source inventories, audited Stage 2 membership and reproduction guide. The earlier analysis lacked row-level KSDD2 labels; those are now available in the manifest.
- Raw images/archives, pretrained files, fitted model states and the virtual environment remain excluded. Source links/hashes and rebuilding instructions are supplied instead.
- Baseline: frozen ResNet18 PaDiM, layers 1–3, 100 retained channels, 128 normal fitting groups, maximum raw output-map value over valid nonpadding pixels. Normal-only rank calibration uses alpha 0.10. Neural-network weights are not optimized in this fit.
- KSDD2: 246 training defects currently assigned to development, 128 selected training normals, 1,123 unused normal fit images, 417 development normals and 417 calibration normals. Cable gland: 128 selected normal fit groups, 68 development images, 44 calibration normals, no genuine training defects.
- Cable gland is reduced from 400x400 to 256x256. KSDD2 is rotated 90 degrees into 256x640, already approximately native scale. A higher nominal input size is not automatically more information.
- The existing test sets and selected galleries have already been examined. They remain historical benchmarks; splitting or reshuffling them now cannot produce independent confirmation.

### Reproducibility prerequisites for after approval

1. **Weight-file mismatch:** `components/docs/REPRODUCTION.md` suggests a `.pth` file. `vision/spike_worker.py` actually calls `safetensors.torch.load_file` and requires SHA-256 `80c49dee3da4822c009c5a7fe591e9223c5a2cfcf95a4067ca4dfb5a7b89c612`. Use the recorded `model.safetensors` at provider revision `491b427b45c94c7fb0e78b5474cc919aff584bbf`; do not rename or substitute the `.pth` file.
2. **Line endings affect byte hashes:** this Windows checkout uses CRLF. The committed manifest's LF-normalized hash is the historical `f1fdd911...`; its checkout-byte hash is `82b04168...`. Its canonical JSON hash still matches the content-addressed directory `b769d859...`. Restore/verify the intended exact bytes and define v2 serialization explicitly. Do not weaken old integrity checks or rewrite frozen claims to force a match.
3. **Historical absolute paths:** manifest raw-image and mask paths reference the teammate's machine. Rebuild a local, verified manifest with retained source IDs/hashes and record a new identity. Do not edit the archived historical manifest in place.
4. Restore a dedicated Python 3.12 CPU environment, pinned dependencies, exact source bytes and model candidates. Only Python 3.10 was found on PATH in the prior inspection. Run meaningful role/geometry/save-load/metric checks before model experiments.

These are setup requirements, not entries counted toward the ten improvement hypotheses. They have been diagnosed but not fixed in this research phase.

## What saved scores rule out

The [retrospective score audit](2026-10-19-observed-score-audit.json) joins archived predictions to exact manifest IDs, checks the original flags and TP/FP counts, and enumerates attainable strict-threshold decisions including score ties. No model was rerun.

| PaDiM recipe | Original frozen recall / FPR | Best observed recall at <=10% FPR | Lowest observed FPR needed for >=90% recall |
|---|---:|---:|---:|
| KSDD2 | 84/110 = 76.36% / 77/894 = 8.61% | 88/110 = 80.00%, with 88/894 = 9.84% FPR | 196/894 = 21.92%, detecting 99/110 |
| Cable gland RGB | 11/87 = 12.64% / 1/21 = 4.76% | 42/87 = 48.28%, with 2/21 = 9.52% FPR | 15/21 = 71.43%, detecting 79/87 |

At >=95% observed recall, the minimum observed FPR rises to 31.88% for KSDD2 and 85.71% for cable gland. These are **post hoc score envelopes on already observed tests**, not deployable thresholds or new performance claims. They are optimistic for these scores because labels choose the cuts. Neither recipe can reach the stated target through a global threshold adjustment alone on these outputs.

## Ranked shortlist

| Rank / ID | Strategy | Best fit | Confidence in useful improvement | Main constraint |
|---|---|---|---|---|
| 1 / R01 | Train on real defects with segmentation and an image classifier | KSDD2 | High relative to the shortlist | New supervised roles and CPU training budget |
| 2 / R02 | Replace positional Gaussian scoring with compact PatchCore | Both, especially pose-variable cable gland | High relative to normal-only alternatives | Memory-bank construction and nearest-neighbor cost |
| 3 / R03 | Expand representative normal coverage | KSDD2 | Medium–high | More normals can also make some anomalies less separable |
| 4 / R04 | Preserve small-defect detail with resolution/overlap tiles | Cable gland first | Medium–high when detail loss is verified | Memory, seams, multiplicity and spatial semantics |
| 5 / R05 | Balance defect learning and mine hard training examples | Supervised KSDD2 | Medium–high, conditional on R01 | Overfitting and label noise |
| 6 / R06 | Improve map-to-image aggregation | Both | Medium | Tiny real defects can look like isolated noise |
| 7 / R07 | Conservative foreground ROI and pose handling | Cable gland | Medium, conditional on diagnostics | Edge defects and deformation can be removed |
| 8 / R08 | Learn permitted illumination variation | Both; robustness first | Medium for stress robustness | Augmentation can hide faint real defects |
| 9 / R09 | Stabilize PaDiM covariance and feature dimension | Both | Medium–low, inexpensive comparator | Statistical stability is not guaranteed recall gain |
| 10 / R10 | Calibrate a small ensemble of complementary winners | KSDD2 after R01/R02 | Conditional | Added false alarms and runtime |

These are distinct hypotheses, including refinements to shared models. They are not ten unrelated architecture implementations. Dependency experiments compare a refined model against its immediate parent, as well as the fixed baseline.

### R01 — Use the labeled defects: supervised segmentation plus image detection

**Evidence.** The original KSDD2 work supports mixed pixel/image supervision [S1]. SuperSimpleNet explicitly learns from labeled abnormal images and reports 97.4% KSDD2 detection AP in its ICPR paper [S2]; AP does not imply 97.4% recall at our FPR. The authors' implementation is available [S3]. This is more relevant evidence than transplanting a normal-only MVTec leaderboard result.

**Experiment.** Add a frozen ResNet18 feature extractor, a small segmentation head and a separate image classifier; preserve thin-mask signal when projecting labels to feature grids. Use real KSDD2 training masks and normal images, with explicitly grouped supervised fitting and validation roles. Cache frozen features for CPU feasibility. This small-head configuration is our adaptation, not an exact SuperSimpleNet reproduction. If feasible, compare a pinned official SuperSimpleNet configuration later as a separate artifact; its larger backbone and paper training budget are not assumed to fit our CPU limits.

**Measure / stop.** Select on grouped development recall at FPR <=10%, AP and small-defect recall. Keep the official test untouched by selection. Stop a CPU pilot if it breaches resource/time caps; never use an official KSDD2 checkpoint as an independently trained validation candidate without auditing its data exposure. No corresponding supervised cable-gland branch exists without new labeled training defects.

### R02 — Compact PatchCore

**Evidence.** PatchCore retains representative normal patch embeddings through coreset selection and compares patches against a memory bank [S4]. This offers a different assumption from PaDiM's position-specific Gaussian. Its paper's MVTec AD performance is supporting evidence, not a forecast for these recipes.

**Experiment.** Begin with the same frozen ResNet18, geometry and 128 normal groups as P0 to isolate the scoring/model change. Use layers 2–3 and a 3x3 neighborhood, a 5% coreset capped at 10,000 vectors, chunked CPU distances and exact source/feature identities. These are compact pilot settings, not claimed published defaults. Profile the coreset build as well as inference; a capped final bank does not cap temporary construction memory.

**Measure / stop.** Compare recall/FPR, latency and process-tree peak RSS against P0. Keep separate banks per recipe. If good normals remain extreme or many anomalies retrieve near-identical normal patches, save those development examples and diagnose the representation before growing the bank.

### R03 — Better coverage of legitimate normal appearances

**Evidence.** Our audit contains 1,123 KSDD2 normal fit images unused by P0; the selected 128 are deterministically ordered rather than selected for feature diversity. Representative normal context is a key PatchCore design principle [S4]. This motivates a project-specific coverage experiment; it does not prove that more samples always improve recall.

**Experiment.** First compare two 128-group normal subsets: current order versus diversity selection using frozen normal-only embeddings. Then compare the better selection with 256 normal groups. Treat subset choice and sample count as separate ablations. Select only from the approved normal fitting pool; preserve independent development/calibration groups. Never reject difficult but correctly labeled normals just because their scores are high.

**Measure / stop.** Inspect the normal upper-score tail and recall at the same FPR budget. Retain only if anomaly separation holds or improves. Larger PaDiM fits need changed hard caps, a new manifest/protocol and memory profiling; streaming statistics may become necessary. Cable gland has little unused normal capacity, so prioritize coverage quality there rather than promising a large dataset expansion.

### R04 — Preserve small defects through resolution and overlap-aware tiles

**Evidence.** The tiled-ensemble paper studies dedicated models per tile location, including PaDiM and PatchCore, to preserve fine detail while controlling peak memory [S5]. Our cable-gland resize reduces spatial detail; KSDD2's current transform approximately preserves it.

**Experiment.** Start with cable-gland full-frame 256 versus 384 square inputs. Attempt four overlapping, full-coverage native-scale tiles only if the development diagnostics show remaining small-defect loss or full-frame memory is excessive. Freeze tile coordinates/overlap and restore maps to original coordinates. For positional PaDiM, retain a separate model/state per tile position; do not pool arbitrary tiles into a shared spatial Gaussian. Load tile models sequentially if needed.

**Measure / stop.** Count edge and small-defect misses, seam responses, summed latency, disk and resident memory. Calibrate the complete image score after all tile fusion because extra patches create more chances for false alarms. Interpolation itself does not recover detail; no center crop may silently remove defects.

### R05 — Balance training and use hard examples

**Evidence.** Defect pixels occupy a small share of these surface images. The mixed-supervision work addresses imbalanced losses [S1]; focal loss specifically reduces the dominance of easy examples in dense classification [S6]. Its transfer to our segmentation head is a hypothesis, not a demonstrated KSDD2 gain from the focal-loss paper.

**Experiment.** After R01, hold architecture and data fixed. Compare natural sampling with balanced image sampling plus a training-derived positive-pixel weight. Only if needed, compare one focal-loss setting against weighted BCE. Mine difficult normals and subtle defects from training or a nested inner-training split; do not recycle outer validation errors into training while continuing to call their scores held out. Do not count repeated crops as new parts.

**Measure / stop.** Check recall by mask-area stratum and false alarms on genuine scratches/textures labeled normal. Recalibrate after changing loss because score scales shift. Stop if improvement is concentrated in augmented copies or label-noisy examples; preserve the untouched validation groups.

### R06 — Better image scores from useful local maps

**Evidence.** Our detector uses a single extreme map pixel. Good macro pixel AUROC does not guarantee useful image ranking. AnomalyDINO studies robust aggregation of patch distances [S7]; this motivates a bounded comparison, not a claim that its aggregation transfers unchanged to PaDiM.

**Experiment.** Using saved **development** maps from a new run, compare max with the mean of the highest 0.1% valid map values and one maximum local 3x3 average on the native distance grid. Retain strict valid-pixel handling. Compare a learned image head under R01 separately; do not credit an architecture change as a pooling-only win. A strictly monotonic transform of the scalar max cannot improve its ranking or score envelope.

**Measure / stop.** Use identical maps so this is cheap and isolates aggregation. Include tiny defects and known normal extreme pixels. Keep raw pre-smoothing distance maps if needed; the current adapter only returns its smoothed map. Discard pooling if it suppresses tiny anomalies. Freeze the score rule before operational normal calibration.

### R07 — Foreground and pose handling without deleting defects

**Evidence.** Current `valid_roi` excludes padding, not object background. AnomalyDINO evaluates foreground masking and rotation but also documents failed masks and cases where rotation invariance is undesirable [S7]. There is no direct proof that either causes our cable-gland failure.

**Experiment.** First log development peak coordinates and normal pose-related score variation. If background dominates, compare full frame with a training-designed foreground region dilated to retain silhouette/edge margins, using image content only. Treat rigid alignment as a separate comparison and retain the original view/map. Do not elastically warp away bends; pose handling must not make abnormal geometry normal. Textured KSDD2 surfaces generally do not need an object mask.

**Measure / stop.** Require foreground-coverage/edge fixtures and map-coordinate checks. Reject masks that clip thread tips, holes or silhouette changes. A failed mask becomes an explicit review/retake condition, never a normal decision. Do not use ground-truth defect masks as inference ROIs.

### R08 — Illumination robustness through conservative training variation

**Evidence.** The archived stress runs already expose brightness sensitivity and KSDD2 PaDiM blur FPR of 14.09%. Torchvision supplies controlled brightness/contrast transforms [S8]; the documentation establishes the operation, not an accuracy gain for our task.

**Experiment.** After splitting originals, try a fixed mild brightness/contrast range, initially 0.8–1.2, on eligible training images. For the supervised branch retain paired geometric image/mask transforms. For normal-only models treat transformed samples as correlated derivatives, not additional independent normal parts. Compare one training policy at a time. Do not use per-image brightness centering without checking whether discoloration is itself a defect signal.

**Measure / stop.** Report original development results plus fixed brightness 0.6/1.4 and blur stress conditions. These observed stress choices are robustness comparisons, not untouched confirmation. A focus warning can request human retake but cannot create defect information lost to blur. Reject a policy that improves stress FPR by erasing faint-defect recall.

### R09 — Better-conditioned PaDiM statistics

**Evidence.** Current 100-feature covariance is estimated from 128 normal samples per spatial position. The adapter records fixed Gaussian stabilization. Covariance shrinkage addresses estimation/inversion problems when observations are limited relative to dimension [S9]; this is not proof of a present numerical bug.

**Experiment.** Inspect development covariance condition numbers and normal score outliers. Compare 50 retained channels against 100 using a nested, fixed channel subset. If conditioning remains suspect, compare a single data-driven shrinkage estimate with the existing ridge-stabilized covariance, keeping geometry and normal IDs fixed. Save an explicitly new backend identity rather than silently changing Anomalib internals or old artifacts.

**Measure / stop.** Measure score/map parity, finite distances, recall/FPR and seed sensitivity. Do not optimize condition numbers instead of detection. Stop if the normal tail becomes smoother but true anomalies collapse into the normal distribution. Increasing to hundreds of dimensions is not the default remedy under this memory/sample budget.

### R10 — A calibrated ensemble only when misses are complementary

**Evidence.** PatchCore's authors provide ensemble configurations [S4], and tiled ensembles study aggregation across specialized models [S5]. Neither establishes that our particular pair has complementary errors. This method therefore ranks last until measured development predictions exist.

**Experiment.** Pair at most two strong, independently scored candidates, preferably the supervised detector and PatchCore. First calculate overlap of missed development defects. Normalize or fit a small fusion rule using training/inner-development data, cross-fit the fusion during comparison, then calibrate the final combined score once on separate normals. Do not build a ten-model voting system.

**Measure / stop.** Compare the ensemble against the stronger single model at the same 10% FPR, including combined CPU latency and memory. An OR of two separate 10%-FPR detectors does not retain 10% system FPR. Stop if the extra model adds review work without meaningful recall gain. Counts and paired changes matter more than a small rounded percentage increase.

## Execution sequence proposed for after the go

**Phase A — recover a trustworthy baseline.** Resolve weight format, serialized-byte identity and local paths; restore sources/environment and verify software/fit/save-load behavior on authorized development inputs. Preserve all historical artifacts and phase reservations. Create additive v2 protocol and manifest identities. This phase can proceed on KSDD2 without making cable-gland access a universal blocker.

**Phase B — first bounded comparisons.** Cache baseline development maps, then test R06 (cheap score comparison), R02 (compact PatchCore) and R01 (small supervised head), followed by R03 if normal-tail diagnostics support it. Run R04 on cable gland as a separate track with no claim that normal-only diagnostics establish defect recall. This order differs from evidence ranking because cost and dependencies matter.

**Phase C — targeted refinements.** Apply R05 to the promising supervised parent, R08 where robustness is weak, and R07/R09 only if their diagnostic triggers occur. R10 requires complementary errors first. All ten hypotheses will receive a recorded outcome: run result, resource failure or justified deferral. Unsupported experiments will not be silently reported as successful or run merely to fill the list.

**Phase D — select, freeze and confirm.** Confirm the leading two development candidates across three fixed group-split seeds where practical, then choose one per recipe. Freeze preprocessing, training IDs, hyperparameters, model state and scoring before operational normal calibration. Evaluate an unreviewed confirmation cohort once when it exists. Until then, state development or historical-benchmark gains with their exposure limits.

No paid services, GPU dependency or hosted training is assumed. Profile one CPU fit worker at a time. Maintain fit RSS <=min(4 GiB, 60% available RAM), inference owned-process-tree RSS <=4 GiB and warm p95 <=5 s. Proposal: stop an initial training pilot at 30 minutes and record partial/failed status; a partial fit is never compared as an equivalent full-paper reproduction. Use chunked feature caches on disk and never trade away label/split integrity to reduce runtime.

### Data roles and selection rules

- Every original and derivative retains recipe, group, source identity and parent. Reuse the audited duplicate groups; disclose unknown physical-item identities. Do not pool raw scores, normal statistics or thresholds across recipes.
- For supervised KSDD2 comparisons, create fixed grouped outer folds from the current training-anomaly/development-normal pool. Each score must come from a model not trained on that outer group. Use an inner split or fixed preregistered training length for early stopping; outer folds must not also tune epochs. The old normal fit pool supplies fitting negatives; R03 explicitly changes its selection/count.
- This remains development cross-validation: choosing among ten approaches on these scores induces selection optimism. Repeated seeds quantify sensitivity; they are not new validation data.
- Retain a separate normal pool for the selected candidate's final threshold. Existing normal calibration data are historical/previously exposed; fresh allocation from eligible unused KSDD2 normal groups is possible only with documented exposure and whole-group checks. Do not label it a fresh defect confirmation set.
- New independent confirmation requires new, nonoverlapping **physical items** with labels and capture conditions representing the same task. Transforms, alternate photos of old parts, a different object category or a new upstream commit do not provide that automatically. New representative data collection and expert review are not assumed available.

### Acceptance and experiment documentation

Use the [experiment ledger](recall-experiment-ledger.md). Every run records commit, v2 manifest/source/parent IDs, method/config, fitting and threshold rules, model hashes, resource limits, failure slices, and exact output paths. Record failed and deferred attempts as well as winners.

Primary table: TP/FN/FP/TN, all-defect recall, normal FPR, precision, and technical coverage separately per recipe. Rejected defects remain misses. Aim to score all valid benchmark inputs; disclose every failure and count it in coverage. Include image AP/AUROC, per-folder recall when known, small-mask strata, stress results, paired changes from the parent, warm p95, cold start, peak RSS and model/cache size.

For method selection, an initial **practical improvement** is at least +3 percentage points of grouped development recall at the same FPR budget, without resource failure or a major stress/per-folder regression. This is an engineering filter, not a significance test; repeat marginal gains across fixed seeds and report paired uncertainty. The ultimate gate stays >=90% recall and <=10% FPR. Do not lower the target when an experiment fails. With only 21 cable-gland test normals, FPR advances in 4.76-point steps, so strong population claims remain unsupported.

Save raw new development maps, scores and exact joins so threshold/aggregation diagnostics do not require repeated inference. Evaluation-only labels are joined after prediction and never passed as model/ROI inputs. For future confirmation report Wilson intervals, group-bootstrap intervals with an identity caveat and counts; a 90% recall point estimate does not establish a 90% lower confidence bound.

## Alternatives researched but not in the initial ten

**AnomalyDINO / stronger feature backbone:** a credible fallback if compact ResNet features limit separation [S7]. Its published few-shot setting and higher-resolution transformer pipeline differ from our data/compute assumptions. Try only after a bounded CPU/memory pilot; do not assume a bigger backbone always helps.

**EfficientAD:** a credible student–teacher alternative [S10], but our immediate failure is detection quality rather than the existing subsecond inference time. Training and reproduction costs make it secondary to using actual KSDD2 defect labels and a compact PatchCore comparison. Published GPU latency does not establish CPU latency here.

**Synthetic defect generation:** potentially useful for normal-only cable gland [S11], but generated cut/paste anomalies may not resemble thread, bend or fine-geometry failures. Real labeled defects already exist for KSDD2. Keep synthetic samples as labeled derivatives and never use them as the only evidence of real-defect recall.

**Depth fusion or a different product/view:** potentially relevant if RGB evidence is inadequate, but changes the active RGB-only product scope and is not included in this implementation proposal.

## Primary sources consulted

Sources checked on 19 October 2026. Evidence strength above distinguishes same-dataset measurements, mechanisms demonstrated elsewhere, library functionality and our own project-specific hypotheses.

- **S1 — Mixed supervision / KSDD2 authors:** [paper](https://arxiv.org/abs/2104.06064), [full text](https://arxiv.org/html/2104.06064v3), [official code](https://github.com/vicoslab/mixed-segdec-net-comind2021), [dataset](https://www.vicos.si/resources/kolektorsdd2/). Supports real-mask/image supervision; our CPU head and new splits are adaptations.
- **S2 — SuperSimpleNet ICPR 2024:** [paper and full text](https://arxiv.org/html/2408.03143v2). Supports supervised surface-defect detection, separate image head and the cited AP; paper hardware/training differ from our pilot.
- **S3 — SuperSimpleNet authors' code:** [repository](https://github.com/blaz-r/SuperSimpleNet). Distinguishes ICPR/JIMS variants and documents hardware/checkpoint caveats. Pin a revision before adopting code; do not combine different variants' results.
- **S4 — PatchCore:** [paper](https://arxiv.org/abs/2106.08265), [CVPR full-text page](https://openaccess.thecvf.com/content/CVPR2022/papers/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper), [authors' code](https://github.com/amazon-science/patchcore-inspection). Supports representative normal memory, coreset selection and ensemble implementation, not our target recall.
- **S5 — Tiled ensembles:** [paper](https://arxiv.org/abs/2403.04932). Supports overlap and per-location models; our geometry/resource choices need measurement.
- **S6 — Focal loss:** [original paper](https://arxiv.org/abs/1708.02002). Supports emphasizing hard dense-classification examples; KSDD2 loss settings and gains are unmeasured here.
- **S7 — AnomalyDINO:** [paper/full text](https://arxiv.org/html/2405.14529v3). Supports patch aggregation, masking/rotation analysis and stronger-feature fallback; its warnings also motivate conservative ROI handling.
- **S8 — Torchvision:** [ColorJitter documentation](https://docs.pytorch.org/vision/main/generated/torchvision.transforms.ColorJitter.html). Defines transforms only; it provides no project performance guarantee. Verify against pinned torchvision, not only current main docs.
- **S9 — Covariance estimation:** [scikit-learn documentation and original references](https://scikit-learn.org/stable/modules/covariance.html). Supports shrinkage/conditioning rationale; v2 PaDiM behavior remains an experiment.
- **S10 — EfficientAD:** [original paper](https://arxiv.org/abs/2303.14535). A secondary architecture lead; GPU speed must not be quoted as native CPU performance.
- **S11 — CutPaste:** [original paper](https://arxiv.org/abs/2104.04015). Supports synthetic anomaly learning as a fallback mechanism; synthetic success does not validate actual defects.
- **Repository evidence:** [Stage 5 report](../../components/docs/STAGE5_REPORT.md), [failure analysis](../../components/docs/STAGE5_FAILURE_ANALYSIS.md), [source/audit summary](../../components/docs/STAGE2_AUDIT_SUMMARY.json), [current reproduction guide](../../components/docs/REPRODUCTION.md), [initial analysis](../ML_RECALL_IMPROVEMENT_PLAN.md).

## Approval checkpoint

The proposed go covers baseline reproduction, additive v2 split/evaluation machinery and bounded experiments R01–R10, with honest run logs and conditional deferrals. It does not promise all ten will run regardless of diagnostic relevance or available compute. No model implementation has started. **The next action is to ask the user for the go before executing Phase A.**
