# Reproduction guide

This guide recreates the Stage 5 development inputs and fitted model candidates
locally without committing or redistributing the raw datasets or large model
states. Run commands from the repository root in PowerShell.

The archived Stage 5 test set has already been observed. Recreating its files is
useful for audit and software verification, but it must not be repeatedly scored
to choose a new model, image score or threshold. Use development data for design
decisions and a new untouched confirmation set for the final comparison.

## 1. Clone and create the Python environment

Python 3.12 and `uv` are required. The recorded Stage 5 numerical environment is
described in `historical_benchmark/frozen-20261019T041652Z-9c825801/freeze.json`.

```powershell
git clone https://github.com/la-kshita/Tech4Bharat.git
Set-Location Tech4Bharat
uv venv --python 3.12 components/.venv-p0
uv pip install --python components/.venv-p0/Scripts/python.exe -r components/requirements-p0.lock
uv pip install --python components/.venv-p0/Scripts/python.exe --no-deps -e components
```

Verify the software before acquiring data:

```powershell
.\components\.venv-p0\Scripts\python.exe -m component_inspector.cli doctor --json
.\components\.venv-p0\Scripts\python.exe -m pytest components/tests -q --basetemp=components/.cache/colleague-pytest
```

The reviewed repository state passed 99 tests. Package availability can change;
the lock and frozen version record are the authoritative dependency evidence.

## 2. KSDD2

- Provider page: https://www.vicos.si/resources/kolektorsdd2/
- Direct provider download: https://data.vicos.si/datasets/KSDD/KolektorSDD2.zip
- Recorded bytes: `853126555`
- Recorded SHA-256:
  `edcdb486809b24f1d17b785e30c52fafc5999554dd5fe18ddf77b61ceb6f36a8`
- Licence recorded by the project: CC BY-NC-SA 4.0.
- Citation: Jakob Bozic, Domen Tabernik and Danijel Skocaj, *Mixed
  supervision for surface-defect detection: from weakly to fully supervised
  learning*, Computers in Industry, 2021.

The acquisition command downloads from the provider, applies bounded extraction,
records provenance and verifies the resulting inventory:

```powershell
.\components\.venv-p0\Scripts\python.exe -m component_inspector.cli acquire --dataset ksdd2
```

Expected local paths:

```text
components/data/archives/edcdb486809b24f1d17b785e30c52fafc5999554dd5fe18ddf77b61ceb6f36a8.zip
components/data/raw/ksdd2/
components/data/manifests/ksdd2_source.json
components/data/manifests/ksdd2_edcdb486809b24f1d17b785e30c52fafc5999554dd5fe18ddf77b61ceb6f36a8_inventory.json
```

If the provider bytes change, do not relabel them with the historical hash. Treat
them as a new source release and generate a new benchmark identity.

## 3. MVTec 3D-AD cable gland

- Official dataset page:
  https://www.mvtec.com/research-teaching/datasets/mvtec-3d-ad
- Recorded direct download origin:
  https://www.mydrive.ch/shares/45892/8836f2fd7407ec1c54dc165a5b8a2e91/download/428792309-1643274318/cable_gland.tar.xz
- Recorded bytes: `354399396`
- Recorded SHA-256:
  `63728f5ab3b37feaa03a796eefd44e572afe74e49b28e8fc3b19ec98838aeaed`
- Licence/notice retained in the supplied archive: CC BY-NC-SA 4.0, MVTec
  copyright 2021.
- Citation: Paul Bergmann, David Sattlegger and Carsten Steger,
  *The MVTec 3D-AD Dataset for Unsupervised 3D Anomaly Detection and
  Localization*, VISAPP 2022.

The recorded archive contains an early-review notice and has no authenticated
current provider release ID or provider-published checksum. The URL and SHA-256
reproduce the exact locally evaluated bytes; they do not prove that those bytes
are the provider's current release. Review the provider's current terms and the
included notice before use.

Download the archive to any local path, verify its hash, then import it through
the bounded acquisition command:

```powershell
$cableArchive = 'C:\PATH\TO\cable_gland.tar.xz'
Get-FileHash -Algorithm SHA256 -LiteralPath $cableArchive
.\components\.venv-p0\Scripts\python.exe -m component_inspector.cli acquire --dataset mvtec_3d_ad --category cable_gland --archive $cableArchive
```

Expected local paths:

```text
components/data/archives/63728f5ab3b37feaa03a796eefd44e572afe74e49b28e8fc3b19ec98838aeaed.tar
components/data/raw/mvtec_3d_cable_gland/cable_gland/
components/data/manifests/mvtec_3d_cable_gland_source.json
components/data/manifests/mvtec_3d_cable_gland_63728f5ab3b37feaa03a796eefd44e572afe74e49b28e8fc3b19ec98838aeaed_inventory.json
```

The Stage 5 recipe uses RGB only. XYZ and camera-calibration files remain source
evidence but are excluded from model inputs.

## 4. Verify sources and recreate the Stage 2 manifest

The committed manifests provide the historical membership and hashes for review.
To verify local raw files and recreate the prepared manifest:

```powershell
.\components\.venv-p0\Scripts\python.exe -m component_inspector.cli acquire --dataset ksdd2
.\components\.venv-p0\Scripts\python.exe -m component_inspector.cli acquire --dataset mvtec_3d_ad --category cable_gland --archive 'C:\PATH\TO\cable_gland.tar.xz'
.\components\.venv-p0\Scripts\python.exe -m component_inspector.cli prepare --config components/configs/protocol.yaml
```

Historical selected manifest:

```text
components/data/manifests/stage2/b769d8598112008728d354145526815eceeaaf32e477861af22c25a541a1ecc9/manifest.json
```

- Manifest SHA-256:
  `f1fdd9113745565f00827529f475989192465562e881478a3f6d3d31f59b4863`
- Membership hash:
  `6d24d51d1858c67b08d14340998310c8a4e419ab7f232270d4f322a7bcd53ca2`

If a fresh preparation produces different identities, investigate source bytes,
code versions and configuration before fitting models. Do not overwrite the
historical manifests.

## 5. Recreate development model candidates

The selected Stage 5 model states are intentionally not stored in Git. The exact
selected artifact identities and SHA-256 hashes are recorded in `freeze.json`.

Recreate the reference baseline development artifacts:

```powershell
.\components\.venv-p0\Scripts\python.exe -m component_inspector.cli develop `
  --config components/configs/protocol.yaml `
  --manifest components/data/manifests/stage2/b769d8598112008728d354145526815eceeaaf32e477861af22c25a541a1ecc9/manifest.json
```

PaDiM used the pinned `resnet18.a1_in1k` pretrained weights. The recorded provider
URL is:

https://huggingface.co/timm/resnet18.a1_in1k/resolve/491b427b45c94c7fb0e78b5474cc919aff584bbf/model.safetensors

The worker requires the safetensors bytes with SHA-256
`80c49dee3da4822c009c5a7fe591e9223c5a2cfcf95a4067ca4dfb5a7b89c612`.
The `.pth` URL in earlier versions of this guide points to a different format
and does not satisfy the worker's pinned hash/loader. Do not rename that file.

The Stage 4 worker records the resolved weight identity in
`docs/STAGE4_FEASIBILITY_SUMMARY.json`. Use a verified locally downloaded weight
file when running the feasibility command:

```powershell
.\components\.venv-p0\Scripts\python.exe -m component_inspector.cli spike `
  --config components/configs/protocol.yaml `
  --manifest components/data/manifests/stage2/b769d8598112008728d354145526815eceeaaf32e477861af22c25a541a1ecc9/manifest.json `
  --weights 'C:\PATH\TO\model.safetensors'
```

The historical selected development summaries are:

```text
components/docs/STAGE3_DEVELOPMENT_SUMMARY.json
components/docs/STAGE4_FEASIBILITY_SUMMARY.json
```

Compare recreated model metadata and state hashes with
`historical_benchmark/frozen-20261019T041652Z-9c825801/freeze.json`. Exact model
state equality can depend on the frozen dependency/runtime combination; a
different state must receive a new run identity.

## 6. Do not repeat the historical final evaluation for model selection

The historical calibration, prediction rows, aggregate metrics, galleries and
worker logs are committed under:

```text
components/historical_benchmark/frozen-20261019T041652Z-9c825801/
```

Use those saved outputs to inspect Stage 5. Do not remove phase reservations or
rerun the same holdout until it appears to pass. For a next-version experiment:

1. Diagnose and select changes using development data.
2. Pre-register the candidate and acceptance rules.
3. Freeze a new run identity.
4. Evaluate once on a new untouched confirmation set.
5. Compare that result with the historical Stage 5 report.

## Licence and storage boundary

Raw datasets, archives and trained model states remain ignored locally. This
repository provides source links, attribution, hashes, manifests and rebuilding
commands instead of redistributing multi-gigabyte third-party data. Both recorded
dataset sources carry CC BY-NC-SA 4.0 obligations; public availability does not
grant commercial permission or provider endorsement.
