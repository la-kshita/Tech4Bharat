# Tech4Bharat — ML model handoff

The ML improvement work is complete for this handoff. The team selected a
reproducible three-model surface-defect ensemble for the experimental MVP.
**Application implementation is left for the teammate to continue.**

## Start here

- **[Teammate / Codex handoff](docs/handoff/START_HERE.md)** — setup and prediction interface.
- **[Next steps](docs/handoff/NEXT_STEPS.md)** — resume the application from T06.
- **[Frozen model card](docs/handoff/MODEL_CARD.md)** — exact selection and limitations.
- [Repository map](docs/handoff/REPOSITORY_MAP.md) and [documentation map](docs/README.md).
- [All research reports and proof](docs/research/README.md).
- [Agent instructions](AGENTS.md).

## Selected model

**`ksdd2-ensemble-mvp-v1`**: three supervised `longer108` ResNet18-feature heads,
seeds 4220/4221/4222, existing confirmation fold 0. The three unchanged heads and
manifest are committed in `components/model_registry/`; the standard pretrained
backbone is acquired with a pinned checksum. No refitting or best-fold search.

The more-sensitive, predeclared diagnostic setting met **95%+ overall recall
and under 10% normal false alarms in all six internal folds**. Procedure-level
second-partition results were **95.93% recall / 7.01% false alarms**, with
**37/45 small defects caught (82.22%)**. The one exact frozen instance caught
**78/82 defects (95.12%)**, falsely flagged **34/556 normals (6.12%)**, and caught
**12/15 small defects (80%)**.

These are internal observations on previously observed KSDD2 training images.
The original primary setting failed; the diagnostic result does not replace it.
A fitting/held-out generalization gap remains. The official test stayed closed
through this batch; fresh factory validation is pending. Scores are not defect
probabilities, and human review is required. The model card preserves full context.

## Use the ML interface

After following the [locked-environment setup](docs/handoff/START_HERE.md):

```powershell
components/.venv-p0/Scripts/python.exe components/tools/prepare_frozen_model.py --download-backbone
components/.venv-p0/Scripts/python.exe components/tools/predict_frozen_model.py PATH_TO_IMAGE.png
```

For Python integration, load `component_inspector.vision.ksdd2_ensemble.Ensemble`
once and call `predict_bytes`. This is a standalone ML interface, not a running
API or UI. No training data or labels are needed for prediction.

## Project boundaries

Source, tests, selected small model heads, configuration, locks and evidence are
tracked. Datasets, large feature caches, installed bundles and environments remain
ignored. Historical experiment paths are preserved for reproducibility.

The existing [execution plan](Tech4Bharat_Codex_Execution_Plan.md) describes the
later app/review/report work. [Component history](components/README.md),
[dataset provenance](components/docs/DATA_SOURCES.md), and the earlier
[PCB experiment](experiment/README.md) remain reference material. This handoff
adds no application, cloud deployment or claim of production acceptance.
