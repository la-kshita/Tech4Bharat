# Current handoff — 22 October 2026

ML finalization and teammate handoff are complete. The active experimental
candidate is `ksdd2-ensemble-mvp-v1`; exact heads/calibration and a standalone
prediction adapter are supplied. Application implementation is intentionally
left to the teammate. See [start here](../../docs/handoff/START_HERE.md),
[model card](../../docs/handoff/MODEL_CARD.md) and
[next steps](../../docs/handoff/NEXT_STEPS.md).

The dated stage entries below are preserved history. Their original no-go,
setup-only and no-Git statements do not describe the current repository;
none of those earlier benchmark decisions has been rewritten as a success.

---

# Current checkpoint - stage 5 run finished, NO GO (19 October 2026)

Calibration, original evaluation, paired stress and reports are complete. Both methods failed the 90% recall screen on both datasets. Application stages 6-10 are stopped by the plan gate. Software suite: 99 passed. Reporting limitations remain explicit. See STAGE5_REPORT.md and STAGE5_VERIFICATION.json. Earlier checkpoints below are history.

---

# Current checkpoint - stage 5 in progress (19 October 2026)

Stages 0-4 complete. Normal-only calibration complete for both datasets and both methods. Original final evaluation is running once against frozen settings. No application eligibility claim yet. See IMPLEMENTATION_PROGRESS.md for current evidence; older checkpoints below are historical.

---

# Current checkpoint - cable-gland scope update

The active datasets are MVTec 3D-AD cable_gland (RGB only) and KSDD2.
Both are acquired; audit/splits and models remain pending. The user requested
updating the prior files for this replacement. See [scope amendment](CABLE_GLAND_SCOPE.md).
Active configuration is mvtec_3d_cable_gland_rgb + ksdd2_surface; metal nut is
deferred. Old statements below requiring metal-nut access are historical.
Exact cable-gland download source/early-review notice reconciliation is pending
before freeze; do not fabricate provider identity. No training started.

Scope-update verification: all configuration YAML parsed; active recipe IDs,
archive hash and RGB file globs agree with the imported source record.
Matched 223 training, 23 validation and 108 test RGB paths (no test decoding).
Full component suite: 58 passed in 3.54 s using the reproduction command above
with workspace-local basetemp. git diff --check passed. Old source/test/doctor
snapshots are retained as dated history, not rewritten as new results.
Sequential stage authorization and restrictions remain as recorded below;
this edit updates supporting files, not a claim that the stages have executed.

---

# Setup progress — 18 October 2026

Plan: [Tech4Bharat_Codex_Execution_Plan.md](../../Tech4Bharat_Codex_Execution_Plan.md).

## Initial authorised work (historical)

User requested the virtual environment, initial files and folders only.
This is the setup portion of T01, not authorisation to run the entire plan.

## Rulings

- No Git repository exists in this workspace. Use the separate `components/`
  directory; do not initialise a repository, create a worktree or make commits.
- Reuse the available bundled Python 3.12.14 interpreter through a new environment;
  the system default is Python 3.13 and does not match this plan.
- Use project-local `.cache/uv` because the sandbox's default uv cache is unwritable.
- Install only lightweight setup/dev dependencies now. ML packages, frontend
  dependencies, datasets, models and training remain deferred.
- Preserve the old `experiment/` environment/results and all plan documents.

## Verification

- Created `components/.venv-p0` with Python **3.12.14**, system packages disabled.
- Resolved eight lightweight setup/dev packages into `requirements-setup.lock`
  with distribution hashes; installed the project editable into this environment.
- Installed PyYAML 6.0.3, psutil 7.2.2 and pytest 9.1.1 plus their locked dependencies.
- Wrote diagnostic command tests first: two failed before `cli.py` existed;
  after implementation, `python -m pytest tests -v` passed **2/2**.
- `python -m component_inspector.cli doctor --json` confirms isolated supported
  Python, readable draft configurations, and no installed torch/torchvision/anomalib.
- Dataset acquisition and model readiness remain false; only `doctor` is implemented.
- Dependency setup required escalation because the sandbox blocked uv cache/path
  access. The reviewed escalation succeeded; no action remains blocked.
- All of T01 is **not** complete: acquisition and its safety tests remain future work.

Runtime snapshot: [SETUP_ENVIRONMENT.json](SETUP_ENVIRONMENT.json).

## Subsequent authorisation: local Git repository

The user subsequently requested Git setup for the project. This supersedes the
earlier no-initialisation ruling above: initialise the workspace repository and
commit the existing research and scaffold, excluding local dependencies, datasets,
weights and generated binary artifacts. No remote publication was requested.

## Remaining implementation after initial setup (historical)

Official archive access/licence/source ledger, safe acquisition and extraction tests,
and actual dataset acquisition remain unimplemented. Follow with T02 split auditing
before any model fitting. No performance or readiness result is implied by setup.

## Stage 0 verification — 18 October 2026

- Gate passed for the existing scaffold only. Stage 1 was not executed.
- Branch `main` tracks `origin/main`; HEAD before this verification was `d30447b`.
  Origin is `https://github.com/la-kshita/Tech4Bharat.git`. Remote visibility and
  remote freshness were not checked online; confirm private visibility manually.
- Existing user changes (`README.md` modified and root `docs/` untracked) were
  preserved. No applicable AGENTS.md was found in the workspace or its ancestors.
- Doctor exited 0: Python 3.12.14, isolated/supported, system packages disabled;
  16 logical CPUs, 7,734,337,536 bytes available RAM (~7.20 GiB) and
  659,205,365,760 bytes free disk (~613.93 GiB) at the saved snapshot.
- Node v24.21.0 and npm 11.19.0 are available. Frontend dependencies/build and
  future ML compatibility were not tested. No environment/package changes made.
- Doctor is the only implemented CLI command. All four YAML configurations load;
  backend pins are proposed, protocol is draft, recipe validation is pending.
  Raw archive/data/model directories contain placeholders only; readiness is false.
- Existing tests passed 2/2, including rejection of unsupported `fit` (exit 2).
  The first run emitted a pytest cache permission warning. A second run with
  `-p no:cacheprovider` passed 2/2 without warnings; cache permissions were not changed.
- Optional dependency consistency checks were unavailable: `python -m pip check`
  failed because this uv-created environment has no pip; `uv pip check` failed
  while opening the default cache (os error 183). No dependency consistency pass
  is claimed. These do not prevent the verified doctor/tests from running.
- `git check-ignore -v` confirmed exclusions for environments, raw data, archives,
  models, project/root caches and frontend node_modules. Tracked data/models are
  `.gitkeep` files only.
- Filesystem agrees with the playbook's scaffold scope. The earlier no-Git ruling
  above is historical and superseded; REPRODUCE.md was absent and is now supplied.
- Evidence: [STAGE0_DOCTOR.json](STAGE0_DOCTOR.json),
  [STAGE0_TESTS.txt](STAGE0_TESTS.txt), and commands in [REPRODUCE.md](REPRODUCE.md).
- Next authorised prompt: stage 1 / remaining T01 in the build playbook. Dataset
  source/terms inspection, safe acquisition and tests remain future work.


## Stage 1 / remaining T01 — 18 October 2026

Current authorisation is safe acquisition only. Software is verified; the complete
stage/data gate is **blocked**, because MVTec requires the user's personal form
step. Stage 2 and all model/application stages were not started.

- Working branch: `stage-1-safe-acquisition`, based on stage 0 commit `b36d144`.
  User's modified root README and untracked root docs remain separate.
- Added real `acquire` CLI and local `--archive` import; source registry permits
  only MVTec metal_nut and KSDD2. Blocked access exits 3, failures exit 1, unsupported
  commands exit 2. `doctor` lists acquire but its readiness flags remain scaffold
  defaults, not a dataset-ledger audit.
- Download/extraction use fixed byte/member/expansion bounds, safe portable names,
  regular files only, temporary extraction, one training-only PNG decode, SHA256
  archives/inventories, and no-overwrite/idempotent verification. No downloaded
  script or pickle is executed. No final-test image was viewed/decoded for design.
- MVTec CLI ran and recorded `blocked_access`; zero bytes downloaded. Required
  action: personally use the official form and obtain the metal_nut archive if
  terms are acceptable, then provide its local path. No terms were accepted.
- KSDD2: 853,126,555 downloaded bytes; 852,612,567 extracted bytes in 6,677 files;
  archive + raw logical total 1,705,739,122 bytes (~1.589 GiB). One training image
  decoded at 229×645. Source/inventory/byte hashes and notices are in DATA_SOURCES.md
  and STAGE1_SOURCES.json. Reuse verified all saved hashes and file membership.
- Release layout includes train/test PNGs and five inert .pyb files. Training-file
  count needs reconciliation against provider image counts in stage 2; no label,
  mask-pairing, physical identity or split audit is claimed here.
- Pillow 12.1.0 installed into the reused venv; verified wheel/import/decode on
  Python 3.12.14, updated hashed lightweight lock and editable metadata. No ML
  packages, GPU, paid service or historical environment changes.
- Sandbox blocked initial network attempts; approved escalation enabled Pillow
  install and KSDD2 acquisition. Download-created data also required approved
  access for reuse verification; no ACL changes made. uv discovery initially
  failed in the sandbox, then succeeded with explicit Python and approved access.
- TDD: initial 40 acquisition cases failed for the absent module/CLI. New TAR/ZIP
  metadata, raw-extra-file, XZ and TAR-small-file regressions were each observed
  failing before fixes. Existing-behaviour checks cover CRC cleanup, retries,
  blocked CLI status and original doctor/unsupported-fit contract.
- Final suite: **53 passed** (51 source cases + 2 existing doctor cases).
  Commands/results and self-review are recorded in STAGE1_EVIDENCE.md;
  full test output is STAGE1_TESTS.txt. RED logs remain ignored under
  outputs/stage1/. No new model evidence is claimed by fictional archives.
- `uv --cache-dir components/.cache/uv pip check --python
  components/.venv-p0/Scripts/python.exe` passed (10 installed packages checked).
- Next: resume stage 1 with a legitimately obtained MVTec archive. Do not start
  dependent stage 2 without both sources or an explicit human scope amendment.

Rulings: work in the user-requested checkout with a dedicated local branch and
self-review (no subagents, per user instruction); preserve unrelated working-tree
changes. Cost/limit: no independent reviewer and no separate worktree isolation.
MVTec's personal access flow is a real blocker, not a reason to substitute a mirror.
The earlier source/scaffold statements above are dated historical verification.


## Sequential continuation authorised - 18 October 2026

The user now authorises all remaining numbered stages sequentially, with no routine
stage approval. This supersedes the earlier one-stage-at-a-time authorisation;
experimental gates, required personal actions and real human participation still
apply. No publication, push, purchase, personal terms acceptance or invented
human decisions is authorised. No subagents are authorised.

Progress inspected at commit cbd1c45 on stage-1-safe-acquisition. Stage 0 passed;
stage 1 software and KSDD2 acquisition passed. The archives directory contains
only the recorded KSDD2 ZIP; no MVTec archive has been provided there. The current
official MVTec AD page was rechecked and still presents a download form:
https://www.mvtec.com/research-teaching/datasets/mvtec-ad .

Fresh verification: components/.venv-p0/Scripts/python.exe -m pytest
components/tests -q -p no:cacheprovider -> 53 passed. The implemented acquire CLI
for mvtec_ad/category metal_nut again returned blocked_access, exit 3, with no
archive. Existing KSDD2/hash/exclusion/manual checks passed in the immediately
preceding verification; no new download, split, model or evaluation was run.

Genuine blocker: the user must personally obtain the MVTec metal_nut archive
through the official access flow if its terms are acceptable, then provide the
local archive path. Stage 2's explicit gate requires both sources or an explicit
human-approved scope amendment; sequential execution does not waive that gate.
There is no eligible later numbered stage while this prerequisite remains unmet.

Resume remaining stage 1 with --archive PATH, verify it, then continue stage 2
onward under the standing sequential authorisation. Do not repeat completed
acquisition or ask for routine stage approval. Preserve the user's unrelated
root README change and untracked root docs. No data or historical files changed.


## Cable-gland acquisition - 18 October 2026

User-supplied archive: `C:\Users\laksh\Downloads\cable_gland.tar.xz`.
Official source: https://www.mvtec.com/research-teaching/datasets/mvtec-3d-ad .
Status: acquired, not audited or split; separate identity `mvtec_3d_ad/cable_gland`.
Local import completed 2026-10-18T16:36:48.457546+00:00 (22:06:48 IST).
Original download time, exact download URL and provider release ID are unknown.

Archive bytes: 354399396. SHA256:
`63728f5ab3b37feaa03a796eefd44e572afe74e49b28e8fc3b19ec98838aeaed`.
Content-addressed copy under components:
`data/archives/63728f5ab3b37feaa03a796eefd44e572afe74e49b28e8fc3b19ec98838aeaed.tar`.
The existing importer stores TAR-family archives with a .tar suffix; the bytes
remain XZ-compressed and unchanged. Actual format is detected from content.
Raw root: `data/raw/mvtec_3d_cable_gland/cable_gland/`.
Inventory: 820 regular files, 759081950 extracted bytes; inventory SHA256:
`7e3f5e22b18edc9c3e9894a402973dbbf05345a8deea30b32e0c9879d707032c`.
Layout counts: 223 training RGB images, 23 validation RGB images, 108 test RGB
images (21 good; 22 cut; 22 hole; 21 bent; 22 thread). This is not a full audit.
Only train/good/rgb/000.png was decoded: RGB PNG, 400 x 400. No test image viewed.
XYZ files remain inert source data. Proposed modelling uses RGB only, not 3D.

Included notices: MVTec copyright 2021, CC BY-NC-SA 4.0. Retain attribution,
licence and changes; no commercial permission claimed or personal terms accepted.
Attribution: Bergmann, Sattlegger, Steger, The MVTec 3D-AD Dataset (VISAPP 2022).
The included readme describes an early review subset. The exact provider download
URL and release/notice reconciliation remain pending before experiment freeze.
No provider checksum verified: self-computed SHA256 identifies acquired bytes,
not provider identity. Import mode: local_archive_unverified_provider_identity.

Importer tests were written first: two new behavior tests failed before support
existed. After adding the separate source/category, CLI and training-RGB selector,
all 58 component tests passed (3.81 s). Command from components:
`.\.venv-p0\Scripts\python.exe -m pytest tests -q --basetemp=.cache/cable-green-1 --tb=short`.
KSDD2 and the historical blocked metal-nut record are preserved. The original
metal-nut protocol is not silently reinterpreted as cable gland. Next: document
replacement RGB-only recipe, data roles and validation handling before Stage 2.
No training, final evaluation, model readiness or complete Stage 2 claim.


Build-plan alignment - 18 October 2026: revised the root build plan for
RGB-only cable gland + KSDD2, referencing current execution/playbook details.
The prior technical review now has a current-scope header and explicit historical
evidence status; its old commands/counts are not active requirements. Verified
local Markdown links, stages 0-10, 134/45/44 role arithmetic and 79/87, 2/21
operating-point counts; git diff --check passed. Documentation-only changes in
this turn; no new training, experiment, or model/API/UI implementation.


Project-local archive copy - 18 October 2026: copied the supplied download to
components/data/archives/cable_gland.tar.xz (354399396 bytes); SHA256 matches
63728f5ab3b37feaa03a796eefd44e572afe74e49b28e8fc3b19ec98838aeaed.
Existing importer-owned archive/raw files and Downloads original were preserved.
Acquire reuse exited 0 after verifying archive/inventory and all 820 raw file
hashes and membership. The extra named copy adds 354399396 storage bytes.
Build/reproduction commands now use the project-local archive; recipe raw paths
remain unchanged. No training or test-image decoding occurred.
