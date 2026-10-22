# Selected model registry

`ksdd2-ensemble-mvp-v1/` contains the three unchanged selected trained heads and
their portable manifest. This is the only active model selection for the MVP.
The heads total 167,379 bytes. The larger standard pretrained backbone is
downloaded from its pinned provider revision, with exact hash/size checks.

```powershell
# From the repository root, using the locked environment:
components/.venv-p0/Scripts/python.exe components/tools/prepare_frozen_model.py --download-backbone
components/.venv-p0/Scripts/python.exe components/tools/predict_frozen_model.py PATH_TO_IMAGE.png
```

The installed bundle belongs in ignored `components/models/`, not the web
public directory. See [setup](../../docs/handoff/START_HERE.md) and the
[model card](../../docs/handoff/MODEL_CARD.md) for exact selection and limits.
No original dataset images, new training or app implementation are included.
