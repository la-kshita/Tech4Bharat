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

# Implementation ledger - plan: docs/superpowers/plans/2026-10-18-codex-build-playbook.md

User instruction: implement each numbered stage, reverifying earlier stages at every gate.
Work in the existing dedicated stage-1-safe-acquisition checkout; preserve existing user edits.
No subagents, push, public deployment, purchases, personal terms acceptance or invented human decisions.

## Stage checkpoints

- Stage 0 reverified: 58 component tests passed (3.96 s); Python 3.12.14 isolated;
  16 logical CPUs; available RAM 7,380,770,816 bytes at snapshot; Node v24.21.0,
  npm 11.19.0. Data/model tracked paths are placeholders only.
- Stage 1: source/CLI safety tests pass. Fresh full-source reuse verification pending this run.
- Stage 2: implementing adapters, mask/duplicate audit, deterministic roles and access guards.
- Stages 3-10 pending prerequisites; no model or application completion claim.

## Rulings and pre-flight

Ruling: reuse the current checkout/dedicated branch and project-local ignored scratch rather
than create a checkout without the user data/current edits. Cost: author self-review and no
separate worktree isolation. The prior recorded user preference was work in this checkout.
Ruling: use this portable Markdown ledger and Python/PowerShell commands instead of Bash-only
skill helper scripts on Windows; keep real command evidence and RED/GREEN results here.
Pre-flight: acquisition -> adapter consumes separate dataset identity/raw root; config's
RGB-only paths agree. Adapter -> split preserves official partition and mask-derived labels.
Split -> fit must enforce normal-only role guards and source/config/manifest hashes.
Fit -> calibration/evaluation requires separate roles, both-recipe freeze and trusted bundles.
Evaluation -> application is conditional on measured per-recipe accuracy/resource gates.
Ruling: retain unknown exact download URL and early-review notice before freeze; current
local software/data audit may continue without inventing provenance. User confirms only
that the dataset is downloaded. The model/evaluation freeze must not falsely clear this.

## 19 October resume / Stage 2 implementation

- Stage 1 fresh acquire reuse succeeded for KSDD2 and cable gland (CLI exit 0),
  verifying source archives, inventories and all raw hashes/membership.
- Stage 2 initial seven tests failed for absent adapters/audit/split. Added
  records, separate RGB adapters, mask validation, decoded/pHash duplicate groups,
  deterministic roles and normal-only purpose guards. Prepare/tamper test failed
  for absent prepare module, then immutable hash-addressed manifest publishing added.
- Full suite now 66 passed in 13.57 s. Real prepare audit is running; no stage
  completion asserted until actual manifests/counts and repeat determinism verify.
- Dependency network downloads failed including approved retry (DNS to wheel host).
  Local .whl filenames in experiment were zero-byte placeholders. Verified every
  hashed installed RECORD file of NumPy 2.2.4 and SciPy 1.15.2 (0 missing/mismatch),
  repacked those exact verified bytes into project-local offline wheels and installed
  into components/.venv-p0. Did not mutate/import experiment model code or environment.
  NumPy 2.2.4 / SciPy 1.15.2 imports confirmed. Windows launcher install required
  approved sandbox escalation. Repack provenance is in .cache/offline-wheels/PROVENANCE.json.
Ruling: use the locally verified 2.2.4/1.15.2 dependencies instead of unavailable
network target 2.2.6/1.15.3. Cost: provider wheel hashes differ because these are
repacked installed bytes; disclose source and retain lock/provenance for recreation.
Ruling: near-duplicate candidates meeting pHash<=2 and thumbnail MAE<=2/255 are
conservatively grouped automatically with explicit candidate evidence; they do not
establish physical identity. Cost: may reduce available independent groups; report
counts and prevent cross-role contamination rather than claiming unique products.

- Regressed a near-identical opposite-label image pair: initial solid-image fixture
  did not exercise the branch; replaced with a fixed textured fixture and observed
  the expected duplicate-label failure. Preserved distinct-label near candidates
  rather than grouping/discarding tiny defects. Full suite: 67 passed in 9.86 s.
- Restarted the first real audit after this correction so code hashes and actual
  behavior agree. Interrupted only that owned prepare session; inspected and removed
  its known stale .prepare.lock. Raw data/manifests were not deleted or overwritten.
- Added per-source/decode progress logs. Current fresh prepare is running.
- requirements-audit.lock generated offline with repacked-wheel hashes. Installation
  is reproducible only with the documented local wheel artifacts or their exact
  verified reconstruction; no provider-wheel hash equivalence is claimed.
Ruling: near candidates with different labels are preserved/reported, not auto-merged;
otherwise tiny genuine defects could be removed as label conflicts. Physical identity
is still unknown; candidate evidence and the residual uncertainty remain disclosed.

## Stage 2 verified checkpoint

- Real audit reconciled 354 cable-gland RGB files and 3,335 usable KSDD2 images.
  Two extra KSDD2 copy-named PNGs were quarantined missing_mask, not counted as normal.
- Source class_ids.json uses 255=bent,254=cut,253=hole,252=thread. Added a failing
  252-valued mask regression, then mapped declared positive values correctly.
  Before this correction the audit would quarantine 66 valid cable-gland defects;
  the rejected/incomplete run was stopped, not treated as evidence.
- Two full corrected prepare runs produced identical membership hash
  6d24d51d1858c67b08d14340998310c8a4e419ab7f232270d4f322a7bcd53ca2
  and manifest SHA256 f1fdd9113745565f00827529f475989192465562e881478a3f6d3d31f59b4863.
- Per-recipe manifests, roles and source/code identities published content-addressed;
  docs/STAGE2_AUDIT.md and STAGE2_AUDIT_SUMMARY.json contain readable/compact evidence.
- fit/calibration/test access and cross-role leakage fixture checks passed.
  Viewed development-only cable gland and a KSDD2 anomaly/mask pair; no test gallery.
- Step 2 data gate passed; unknown physical IDs and source-notice uncertainty remain
  disclosed. Reverify archive/manifest/raw hashes before each dependent real run.

## Stage 3 in progress

Six geometry/reference tests failed before implementation. Added reversible full-frame
RGB letterbox/fixed rotation, valid-ROI scoring and chunked B0 references; then 74
suite tests passed. Artifact save/load/hash/recipe regression failed before storage
implementation. Added trusted saved banks and guarded development-only runner.
First full rerun had an OS WinError 5 directory rename in existing test_tar_roundtrip;
an unchanged rerun in a new scratch directory passed all 75 tests in 8.46 s.
Real B0 development is running after freshly revalidating both Stage 1 sources.
Ruling: normalize Gaussian residual smoothing by valid-ROI Gaussian weight at padded
edges so zero padding cannot suppress valid edge anomalies. Cost: explicit B0 boundary
behavior differs from unnormalized zero-padding smoothing; record code/config identity.

- Stage 3 ordering check: sorted JSON keys made the initial development-only diagnostic
  run process KSDD2 before cable gland. Both were development-only; no calibration/test
  access. Added an actual two-recipe runner fixture; it failed on the wrong order,
  then runner switched to protocol recipe order. Full suite 76 passed in 8.67 s.
  The initial diagnostic is superseded; a fresh source-reverified D1-first run is running.
- P0 preflight: all-wheel dry-run rejected freia==0.2, a mandatory dependency without
  a wheel. That restriction was broader than the plan (which forbids compiling PyTorch,
  not pure-Python source packages), so resolution is being retried with only torch/
  torchvision restricted to wheels. No dependency pins silently changed.

## Stage 3 verified checkpoint

D1-first corrected run completed 68 cable-gland and 663 KSDD2 development queries.
16 selected normal references per recipe; exact bank save/load raw map/score parity.
Warm p95 full pipeline: cable gland 0.05672 s; KSDD2 0.16176 s.
Peak sampled process RSS 208347136 bytes; model bytes 12585530 and 31459767.
All scores remain uncalibrated, flags/thresholds null; no test or calibration queries.
Evidence: docs/STAGE3_DEVELOPMENT_SUMMARY.json and referenced ignored immutable outputs.
Full suite 76 passed in 8.67 s before actual run. Earlier stages reverified by
source hash/membership checks and audited content-addressed manifest verification.

## Stage 4 preflight

Pinned primary backend resolved when allowing the pure-Python freia source package;
CPU-only install completed, preserving NumPy/SciPy/Pillow constraints. No fallback
or pins silently changed. Exact installed low-level module source is being inspected.
Candidate Gaussian adds 0.01 I plus stabilization 1e-5 I; maps use bilinear resize
and sigma-4 smoothing. Preserve these library details in artifact metadata.

- Stage 4 adapter tests: 3 passed, 45 upstream deprecation warnings (not suppressed).
  Normalization/frozen BN, finite shape/ROI max, stored-channel state and raw parity
  passed with random-backbone fixtures explicitly not used as accuracy evidence.
- Actual pretrained timm/resnet18.a1_in1k model.safetensors downloaded through hf CLI;
  revision 491b427b45c94c7fb0e78b5474cc919aff584bbf, SHA256/Hub ETag both
  80c49dee3da4822c009c5a7fe591e9223c5a2cfcf95a4067ca4dfb5a7b89c612.
  Config normalization is ImageNet mean/std, no classifier crop. Frozen feature
  weights loaded strictly for all used feature keys; unused classifier/layer4 keys disclosed.
- The Windows directory publication failure recurred in existing source tests under
  different fixture names. Added a regression that failed on WinError 5, then bounded
  retries for that precise error, with destination no-overwrite rechecked on each
  attempt. Source suite 57 passed. Full suite after fix: 81 passed, 45 upstream
  warnings in 49.99 s. No failing safety check was disabled or weakened.
- Real P0 child runs use an owned-process RSS/time supervisor with a tested kill-on-cap
  fixture. D1's Gaussian fit completed in 13.45 s; D2 in 27.77 s; development scoring
  continues. No calibration or test inference yet.
- Recovered exact cable-gland download HostUrl from original file Zone.Identifier.
  Recorded canonicalized metadata hash, retained notice and matched category counts
  in docs/CABLE_GLAND_SOURCE_RECONCILIATION.json. No need for user to send another URL.
Ruling: reconcile source notice for local research on the identified supplied archive
without claiming a verified current provider release. Exact download origin is now
recorded; early-review wording, unknown provider release/checksum and publication
review remain disclosed. Cost: results cannot be advertised as authenticated current
provider-release reproduction or commercial/factory evidence.

## Stage 4 memory-evidence correction

Initial P0 runs completed real fits/development parity, but supervisor reported ~4 MiB:
it measured only the Windows venv launcher, not its interpreter descendant. Those
runs are not a memory-feasibility pass; raw results retained, summary marked superseded.
Added nested-child 80 MB allocation regression: failed (completed instead of resource
limit), then changed supervisor to sum RSS over its owned process tree and terminate
owned descendants on a cap. Both supervisor tests passed in 0.60 s. Fresh P0 fits
for both datasets are running with the corrected monitor; no frozen/test run exists.

## Stage 4 verified checkpoint

Corrected process-tree monitor completed both real P0 development runs.
Cable gland: 128 fit normals, 68 development images, fit 12.47 s, warm p95
0.15851 s, peak owned-tree RSS 1627570176 bytes. KSDD2: 128 fit normals,
663 development images, fit 27.92 s, warm p95 0.36986 s, peak RSS 3173732352
bytes. Both within launch-specific fit caps and saved/reloaded raw-output parity.
Evidence: STAGE4_FEASIBILITY_SUMMARY.json (corrected run 38fc0d1f768474c62ecf4aa3-0ac3bf5e).
No calibration/final predictions were used for these development decisions.

## Stage 5 implementation checkpoint

Implemented frozen file/version graph, normal-only finite-sample quantile/display
scale, original per-method metrics with technical rejects, one-time phase workers,
sequential bounded CLI orchestration, fixed-threshold paired stress, and escaped
failure gallery/report. Fictional TP9/FN1/FP2/TN18 gives recall .9, FPR .1,
precision 9/11, coverage 1; missing denominator cannot pass a screen.
New gate/hash tests failed before implementation then passed. Worker repeat test
with alternate output folder exposed a loophole; global run unit reservations now
reject repeat access across all output folders (RED then GREEN).
Full suite before that final hardening: 97 passed, 45 upstream warnings, 55.20 s;
post-hardening focused tests: 4 passed. Final full suite is running before freeze.
Ruling: baseline-only pass is conditional and requires a scope decision before a
baseline demo, as required by the binding stage-5 instructions. Primary PaDiM
eligibility is determined separately for each recipe; no holdout tuning.

## Stage 5 frozen experiment

Final pre-freeze full suite: 97 passed, 45 upstream warnings in 54.70 s.
Freeze validation then rejected a PaDiM fit-ID ordering mismatch: membership was
identical (128/128 per recipe), but select_for preserves manifest record order,
whereas the selection list is separately ordered. Added expected-order regression
RED then GREEN (freeze tests 3 passed); verification now checks the actual worker
record order for PaDiM, retaining exact membership/state hashes. No model refit.
Source/config/manifest/artifact checks then passed and experiment frozen at
runs/frozen-20261019T041652Z-9c825801. Freeze SHA256:
49cecd4d593536eecfba7974f61968570581b44885606ab97d424ba5d86cff40.
Normal-only calibration started. No final predictions before calibration completion.
Manual fictional check: normals 1..19 -> n19, k18, threshold18; 18 unflagged,
19 flagged. n8 failure covered by passing calibration regression.

Stage 5 calibration completed successfully for all four method/recipe units.
Cable gland n44/k41; KSDD2 n417/k377. Thresholds and fixed display scales are
hash-linked in calibration_complete.json (12 result/resource file hashes).
No calibration rejects. Strict score>threshold; no label-selected thresholds.
Original final evaluation now reserved once, D1 then D2, reference then PaDiM.

Stage 5 reporting review: added post-prediction supplement outside the frozen
numerical runtime. It reads saved scores/flags and audited labels only, never
loads models or scores images. Verified-folder recall and appearance-group
bootstrap confidence intervals have a fictional regression (RED then GREEN).
Ruling: disclose omitted secondary outputs rather than rescore the holdout.
The frozen worker computed macro per-anomalous-image pixel AUROC but omitted
pixel AP and full raw-map archival. Those outputs are unavailable in this run;
fixed-scale gallery visualizations remain available. This is a reporting
limitation, not evidence that missing metrics are zero and not permission to
repeat/tune final inference. Image metrics and predeclared gate checks remain
available. Physical-product identity remains unknown; bootstrap appearance groups
cannot establish physical independence. Supplemental analysis cannot alter gates.

## Stage 5 original evaluation checkpoint - NO GO

All four original evaluations completed once. No technical rejects; coverage100%.
Cable gland: reference TP3/FN84/FP1/TN20 (recall3.45%, normalFPR4.76%);
PaDiM TP11/FN76/FP1/TN20 (recall12.64%, normalFPR4.76%).
KSDD2: reference TP74/FN36/FP75/TN819 (recall67.27%, normalFPR8.39%);
PaDiM TP84/FN26/FP77/TN817 (recall76.36%, normalFPR8.61%).
Warm p95 seconds: cable B0 .05797 / P0 .16827; KSDD2 B0 .14822 / P0 .36171.
All speed, memory, coverage and normal-FPR screens passed; all recall screens failed.
Primary eligibility empty and baseline-only eligibility empty: final decision no_go.
Per plan, stop application stages6-10. No threshold/model/geometry retuning or repeat
original inference. Paired stress has now started after evaluation_complete.json;
complete stress and failure analysis before stage5 reporting handoff.

## Stage 5 final checkpoint - executed, no-go with reporting limitations

All four normal-calibration units, all four original held-out units and all four
paired-stress units completed successfully. Original evaluation was not repeated.
All 12 fixed stress combinations retained original thresholds. Final KSDD2 PaDiM
stress recall/FPR: blur81.82%/14.09%, darker70%/4.81%, brighter80%/10.51%.
Generated immutable-run REPORT.md, gallery.html and SUPPLEMENT.json. Supplement
uses saved predictions only (no model loading/rescoring); includes defect-folder
recall, 1000-replicate seed4217 appearance-group bootstrap intervals and gallery hashes.
Final independent verification command: python components/.cache/verify_stage5.py
-> verified: freeze/source/model/runtime/version and all completed-phase hashes,
exact calibration/test ID memberships, unique expected rows, null calibration
flags, 12 exclusive unit reservations, 48 gallery image hashes. Access-role count
and ID hashes recorded in docs/STAGE5_VERIFICATION.json.
Final full suite: 99 passed, 45 upstream warnings in60.16s. git ls-files data/models
lists only intended .gitkeep placeholders. No push, publication, app implementation
or new test-driven threshold/model choice.
Final review: self-review by implementing agent (no independent reviewer, per
existing no-agent scope). Limitations retained: pixel AP unavailable, full raw test
maps not archived, unknown physical identities, early-review source notice and
unverified current provider release; no human time-saving measurement. These are
not concealed by software-test success. The original detection gate is no_go;
application stages6-10 are stopped, not claimed implemented. Future model work
requires a separately designed experiment and untouched confirmation evidence.
Evidence entry point: docs/STAGE5_REPORT.md and docs/STAGE5_FAILURE_ANALYSIS.md.
