# Start here

This change hands improved ML research back to the application team.
Read **[docs/handoff/START_HERE.md](docs/handoff/START_HERE.md)** first, then
**[docs/handoff/NEXT_STEPS.md](docs/handoff/NEXT_STEPS.md)**.

The user explicitly limited this task to ML finalization, repository
organization and a PR. Application implementation is left for the teammate;
no API, UI, database or deployment was built in this handoff.

## Working rules

- Active selection: `ksdd2-ensemble-mvp-v1`, three `longer108` supervised
  ResNet18 feature heads, seeds 4220/4221/4222, existing confirmation fold 0.
  Use the committed manifest and saved cutoff; alpha .08 is not the cutoff.
- The model card describes research results and limits. Do not label the model
  production-validated or claim the original primary study passed.
- Keep the official KSDD2 test closed during integration. Never tune thresholds,
  choose seeds or select architecture using test outcomes.
- Preserve `components/src/component_inspector/research/`, frozen numerical
  dependencies, `components/historical_benchmark/`, `experiment/` and recorded
  research evidence. Their hashes/paths support reproduction. Add app adapters
  separately; do not refactor frozen research code as part of app integration.
- `vision/ksdd2_ensemble.py` is the portable ML interface. Its replay proof is
  in `docs/handoff/`; changes affecting scores need a new parity check before
  being called equivalent. The manifest and three heads are committed;
  the standard pretrained backbone is acquired by the checksum-verified helper.
- Environments, datasets, feature caches and installed bundles stay ignored.
  Do not put original benchmark images or model files into web public assets.
- New model experiments need a new protocol and separate output directory.
  Preserve failures. One ML worker at a time; four CPU threads and the existing
  bounded supervisor for batch research/replay.
- Human review records must come from actual people. Test decisions and batch
  fixtures must be labelled fictional. Scores are not defect probabilities;
  model flags cannot accept products.
- Use `rg` first for repository searches. Do not spawn subagents unless the
  user explicitly requests delegation. Avoid moving frozen evidence paths.

## Commands (repository root)

```powershell
components/.venv-p0/Scripts/python.exe components/tools/prepare_frozen_model.py --download-backbone
components/.venv-p0/Scripts/python.exe components/tools/predict_frozen_model.py PATH_TO_IMAGE.png
components/.venv-p0/Scripts/python.exe -m pytest components/tests -q
```

Fresh-clone setup and adapter usage are in `docs/handoff/START_HERE.md`.
Old dated plans/status notes are historical; the handoff identifies what is
implemented, what is deferred, and where the application team resumes.
