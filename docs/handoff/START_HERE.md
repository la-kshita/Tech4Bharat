# ML handoff: start here

**Scope:** model improvement and ML handoff are complete. The teammate resumes
application development. The user explicitly stopped API/UI implementation in
this task. No application backend, frontend or database was added.

## Read in this order

1. [Model card](MODEL_CARD.md): the chosen instance, exact threshold and limits.
2. [Next steps](NEXT_STEPS.md): where the application team should resume.
3. [Repository map](REPOSITORY_MAP.md): source, evidence and local-only folders.
4. [Research index](../research/README.md): all completed methods and failures.

## Selected model

`ksdd2-ensemble-mvp-v1` is an average of three supervised ResNet18-feature
`longer108` heads. We froze the existing first-numbered confirmation fold,
seeds 4220/4221/4222, and its saved threshold. There is no new fit or search
for the best seed/fold. The three 55,793-byte heads are committed in
`components/model_registry/ksdd2-ensemble-mvp-v1/`; the standard pretrained
backbone is fetched separately and checked against its fixed SHA-256.

The committed manifest is the authority for model files, versions, geometry,
score rule, threshold and source evidence. Its identity is
`8ddcac9907af4c033cbc9ff208f452ce572ea6b251da373cf267c2a3df95f355`.

## Fresh clone: Windows / Python 3.12

Run from the repository root. Install [uv](https://docs.astral.sh/uv/) and use
Python 3.12. The existing lock was verified on Windows; other platforms need
their own compatibility/parity check before claiming equivalence.

```powershell
uv venv --python 3.12 components/.venv-p0
uv pip install --python components/.venv-p0/Scripts/python.exe --torch-backend cpu --require-hashes -r components/requirements-recall.lock
uv pip install --python components/.venv-p0/Scripts/python.exe --no-deps -e components
components/.venv-p0/Scripts/python.exe components/tools/prepare_frozen_model.py --download-backbone
components/.venv-p0/Scripts/python.exe components/tools/predict_frozen_model.py PATH_TO_IMAGE.png
components/.venv-p0/Scripts/python.exe -m pytest components/tests -q
```

The backbone is about 46.8 MB from the pinned timm Hugging Face revision. On
the original workstation, the helper reuses the verified local cache. Offline:
pass `--backbone PATH_TO_model.safetensors` instead of `--download-backbone`.
The installer rejects changed files; it never silently replaces a different
model. No original KSDD2 images, masks or feature caches are required to predict.

## Integration interface

```python
from component_inspector.vision.ksdd2_ensemble import Ensemble

model = Ensemble()  # Load once; use one native inference request at a time.
prediction = model.predict_bytes(uploaded_png_or_jpeg_bytes)
```

The result includes `model_id`, `model_identity`, `recipe_id`, `score`,
`threshold`, `flagged`, `status`, `warnings`, image dimensions and a base64 PNG
attention overlay. `not_flagged` means below cutoff, not accepted or defect-free.
The adapter preserves original-image overlay coordinates. Attention is a
visual aid, not validated defect segmentation or a severity/probability map.
Failures raise errors; the application must surface them as failed inspections,
never fill them with a normal score. Encoded limits are 10 MiB, decoded images
20 MP, minimum side 128 pixels, single-frame actual PNG/JPEG only. Out-of-KSDD2
capture dimensions generate a warning; matching dimensions do not establish
that an image belongs to the supported domain.

The implementation is in `components/src/component_inspector/vision/ksdd2_ensemble.py`.
It imports no research runners and reads no dataset manifests, masks or labels.
The underlying numerical environment remains the existing research lock, with
no app dependencies added. Pin the model manifest and check errors in the
teammate's eventual service; do not accept arbitrary uploaded model files.

## Verification and current limits

The portable adapter is checked against all 638 archived held-out TRAIN scores
for this exact fold. See [parity evidence](portable-model-parity.json) and
[resources](portable-model-resources.json). This is implementation regression,
not another independent accuracy sample. [Software counts](software-checks.json)
record the handoff checks actually run.

All 638 scores matched exactly, with zero flag mismatches; the bounded replay
completed in 160.906 seconds with 403,922,944 sampled owned-process-tree RSS
bytes at peak. All 156 software tests passed with 45 upstream deprecation
warnings. An [isolated source-only smoke check](isolated-source-smoke.json)
also predicted successfully with the existing locked environment and no
research output/data directories. This does not claim a fresh-machine install.

The promising .08 diagnostic setting met both point targets in six internal
folds; the original primary setting failed. Fresh factory validation and a real
human review exercise remain pending. The model card keeps these distinctions
visible while the team moves on with the MVP.
