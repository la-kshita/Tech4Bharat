# Repository map and folder ownership

```text
AGENTS.md                         Instructions for the next coding agent
README.md                         Active ML handoff entry point
docs/
  README.md                       Documentation map
  handoff/                        Current model card, next steps and parity proof
  research/                       Dated reports and immutable JSON/log evidence
  superpowers/plans/              Earlier implementation playbook
components/
  model_registry/                 Committed selected heads + manifest (small)
  src/component_inspector/
    vision/ksdd2_ensemble.py       Portable prediction interface for the teammate
    vision/                       Existing media/geometry/native model helpers
    research/                     Preserved numerical experiments
    data/, evaluation/            Existing audit, split and evaluation tools
    application/                  Reserved application scaffold; teammate resumes
  tools/                          ML commands; see its README for tool groups
  tests/                          Software tests, including artifact integrity
  configs/                        Historical registered study configurations
  requirements-recall.lock        Exact verified ML environment
  web/                            Reserved frontend scaffold; not implemented
  docs/                           Historical component setup/stage ledgers
  historical_benchmark/           Protected historical benchmark snapshot
  models/                         Ignored installed frozen bundle
  outputs/                        Ignored large research runs/caches/packages
  .venv-p0/                       Ignored local Python environment
.cache/                           Ignored downloaded model/dependency cache
experiment/                       Earlier PCB work, preserved separately
Tech4Bharat_*Plan*.md              Earlier planning documents, linked by handoff
```

The active model is separated from experiments in `model_registry/`; no original
dataset or cached feature arrays are committed there. Keep runtime-created
uploads, SQLite databases and reports in a separate ignored application-data
directory when the teammate implements the app.

Frozen source/evidence paths are retained because protocols and hash indices
reference them. Navigation indices organize that history without breaking
reproduction. The dated plans and original stage ledgers remain available;
current entry points clearly identify them as historical.

An interrupted initial app-scaffolding attempt was stopped at the user's scope
correction. Its unused dependency-lock/environment draft is parked under ignored
`components/outputs/paused-app-scaffold-20261022/`; no API/UI/dependency changes
from it are part of the PR. It is not a project milestone or required setup.
