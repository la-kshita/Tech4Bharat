# Fitting-versus-held-out diagnostic

This is a diagnostic of the already fixed model procedure, **not another method,
score rule, selection criterion, model refit or held-out qualification**. Register
its source hash before evaluating fitting scores. Run it serially after both
the main sweep and the supplementary ensemble have passed their implementation
audits. The inherited family, seeds, model states, preprocessing and normal-only
calibration thresholds are already fixed; none may change in response.

For each of the 18 selected-family member heads (nine development and nine
second-partition heads), score its exact 420 fitting parents at inference:
164 defects, including 30 small defects, and 256 normals. Apply both existing
alpha .05/.08 thresholds from that head's 200 disjoint calibration normals.
Never calibrate on the fitting predictions. Compare fitting recall/FPR and
small-defect catches with its existing 82-defect/556-normal held-out record.
Keep each model's denominators separate; repeated parents are not new samples.

Compute all scores both through the registered forward scorer and independent
convolutions/pooling from saved weights. Require differences <=1e-5 and identical
flags at both thresholds. Preserve all original IDs, predictions, thresholds,
source/model hashes and counts. Require checkpoint bytes unchanged afterward.

For each ensemble fold, score only its 164 common fitting positives using its
already fixed three-member mean and existing combined-score thresholds. Report
the 30 small fitting defects separately. Do not claim an ensemble fitting-normal
FPR: members have different mined-normal sets. This diagnostic cannot rescue a
failed held-out fold or establish 95% validation performance.

Use one bounded worker, four threads, 1,800 seconds and min(4GiB,60% available
RAM), with failures preserved. No official test image is accessed. A measured
fitting/held-out gap describes generalization behavior without uniquely proving
its cause; its absence would not certify freedom from overfitting or establish
independent factory generalization. All existing limitations remain.

```powershell
components/.venv-p0/Scripts/python.exe components/tools/audit_fitting_gap.py --root components/outputs/recall-seed-ensemble --output components/outputs/recall-fitting-gap
```
