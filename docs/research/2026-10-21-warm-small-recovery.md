# Recorded recovery of the warm-recipe sweep

The first execution is **incomplete**, with eleven completed development models
and three completed native-crop caches. No complete-study accuracy or across-fold
qualification can be claimed from that execution. The official test remains closed.

## Interruption evidence

The `dev-f2-s4217-longer108` worker started on 20 October 2026 at approximately
21:36:04 India time and wrote its plan at 21:36:16. Windows Kernel-Power event 506
records entry into Modern Standby at **21:36:23, reason: Lid**. The subsequent
standby interval ends at 22:40:46; the supervisor wrote its time-limit result at
22:40:47. It recorded 3,883.422 elapsed seconds against the unchanged 1,800-second
limit. Sampled RSS peaked at 518,631,424 bytes, below its 1,275,049,574-byte cap.

The worker log is empty, and no final `model.pt`, `training.json`, evaluation or
completion marker exists for that failed fit. Its plan, empty log and resource
record are preserved with hashes. The power events provide a matching external
trace of a system interruption; the long elapsed time is not evidence that this
model needs an hour of active computation. Earlier completed phases had also
slowed under memory pressure, but that alone does not explain this precise
standby/wake/timer sequence.

The frozen supervisor was tested with a real disposable sleeping child and a
simulated jump in its monotonic clock. It correctly classified a time limit and
reproduced the recorded elapsed overrun without ML fitting. The ordinary short
timeout control passed too. The numerical code and supervisor are unchanged.
This diagnostic checks the mechanism; the Windows trace supplies the evidence
for the actual standby cause. See [diagnosis](warm-small-recovery/diagnosis.json)
and [power events](warm-small-recovery/power-events.json).

## Recovery registered before further fitting

The [recovery registration](2026-10-21-warm-small-recovery-registration.json)
preserves the exact original protocol identity and source hashes:
`b38e03009c68fc963d81a22d5dda3e7263cdfe6f85577566a952001913b79749`.
It records every copied file, every failed-attempt file and diagnostic hash.
The original output root is retained; the resumed root is
`components/outputs/recall-warm-small-resumed-20261021`.

- Reuse the three completed cache indexes and eleven completed model phases byte
  for byte. Existing arrays remain at their original hashed paths. Do not refit
  completed methods, discard unfavorable results or relabel copies as new fits.
- Retry the interrupted continuation once from its original baseline and seed,
  then execute the remaining fixed folds, methods and seeds. Keep all epochs,
  calibration policies, selection rules and resource caps unchanged.
- Publish the recovery registration before resumed fitting. The wrapper verifies
  published bytes, the remote commit, copied artifacts, diagnostics and unchanged
  sources before launching the original serial runner.
- Request the Windows process-scoped `SYSTEM_REQUIRED` awake state while the
  controller and proof jobs run, then clear it on exit. Do not change a global
  power plan or lid policy. Closing the lid or explicitly putting the computer
  to sleep can still interrupt the run.
- A further failed phase stops the suite and remains visible. No automatic retry,
  extra seed, threshold search or model-outcome-based recovery is allowed.
- Include this interruption and its raw evidence in the final audit/export. The
  failure is not a successful model result and cannot support the recall claim.

The separate arithmetic-audit tests passed again: **9 passed**. The updated
recovery and proof helpers compiled. The actual retried continuation then
**completed in 176.688 seconds**, with peak sampled RSS 521,228,288 bytes, under
the same 1,800-second limit and its 583,414,579-byte startup cap. See its
[raw resource record](warm-small-recovery/retry-resources.json). This establishes
that this operational recovery succeeded; it does not establish complete-study
accuracy or guarantee that a later explicit sleep/lid closure cannot interrupt
another phase.

The original controller correctly stopped at the failed phase. Its waiting proof
helper subsequently stopped because no selection file existed; it did not run
replay, create a final report or produce a favorable selection from partial data.

## Continue and verify

```powershell
components/.venv-p0/Scripts/python.exe components/tools/resume_warm_small.py --phase run --root components/outputs/recall-warm-small-resumed-20261021
# After the fixed suite, run the replay/audit/report helper serially.
components/.venv-p0/Scripts/python.exe components/tools/finish_warm_small.py --root components/outputs/recall-warm-small-resumed-20261021
```

The dataset, prior exposure and fresh-factory-data limitations in the
[original preregistered protocol](2026-10-20-warm-small-protocol.md) still apply.
