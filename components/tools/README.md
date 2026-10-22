# ML tools

Run from the repository root in the locked Python environment.

## Teammate integration

- `prepare_frozen_model.py`: copy the three committed frozen heads and verify
  the exact pretrained backbone, using a local file/cache or an explicit download.
- `predict_frozen_model.py`: one-image prediction with the pinned model;
  emits JSON, no labels, training data, database or web server required.
- `verify_frozen_model.py`: developer parity regression against 638 archived
  original TRAIN predictions. It needs the original local research dataset and
  manifest and does not add new accuracy evidence.

## Preserved research workflows

`run_*` executes registered studies, `replay_*` regenerates native scores,
`verify_*` checks roles/arithmetic/provenance, `report_*` writes reports, and
`package_*` preserves exact evidence. Round 1/2, alignment, validation,
small-defect, warm-small and seed-ensemble scripts belong to separate immutable
protocols. They are not application startup commands.

`resume_warm_small.py` and `diagnose_warm_small_interruption.py` preserve the
documented standby recovery. `finish_*`/`finalize_recall_checks.py` were serial
study controllers, not recurring jobs. Do not restart completed pipelines as
part of normal integration.
