# Fresh-image confirmation still required

The project has no separate new labeled factory batch. The official KSDD2 test
set was already observed by earlier project models; scoring a new frozen model
on it is useful benchmark evidence, but cannot undo that prior exposure.

## Collection and sealing

Collect a new batch from the intended inspection line and camera, across normal
shifts, lots, illumination changes, and expected defect sizes. Keep physical
product IDs, lot IDs, capture timestamps, and camera settings. Keep all views of
one product together. Do not reuse KSDD2 images, alternate crops, augmentations,
or a random re-split as independent data. A different unrelated defect dataset
answers a different domain-transfer question.

Freeze one model, preprocessing implementation, score rule, and threshold before
opening the new batch. Seal image identities and file hashes. Have a person
label normal/defective images without seeing model predictions, with a second
review for ambiguous items. Keep labels separate from inference inputs. Record
ambiguous items and technical rejects explicitly; do not silently remove errors.

For the historical V01 benchmark primary, the frozen record is `warm_aligned`, head SHA-256
`ffca032083bd4c7a003c198b70b7dc5eb10c4d7dee40392fd7a81c9160f560d9`,
and strict `score > -1.684261679649353`. The later [small-defect study](2026-10-20-small-defects-results.md)
has an inner-selected research candidate, but establishes no independently
validated replacement. Its full prior-recipe reference passes pooled internal
point targets while failing targets in two folds. If a future model is selected
for factory confirmation, create a new selection and sealing record before
touching the confirmation images.

The subsequent [verified warm-recipe sweep](2026-10-20-warm-small-results.md)
failed consistency across folds/seeds. Its selected control ranged from 92.68%
to 96.34% recall across seeds; a second partition pooled 95.12% but failed one
fold. The [completed three-seed ensemble](2026-10-21-seed-ensemble-results.md)
passed three of six primary-policy folds. Its predeclared diagnostic alpha .08
passed all six point targets, making it a candidate for a future prospectively
sealed study; it does not replace the failed primary. The fitting diagnostic
found 100% fitting recall and lower held-out recall. Both studies reuse observed training originals and
cannot replace the fresh factory confirmation described here. No independently
validated replacement for the historical reference is established.

## Sample size and decision

Choose sample size in advance, not by repeatedly checking whether the result
has started passing. If feasible, target **500 independently sampled defective
items and 1,000 normal items**, including difficult edge and small defects in
their intended operating mix. Sampling separate classes estimates recall and
false-positive rate; its artificial class ratio does not estimate factory
precision or defect prevalence. If defects must be collected across more time,
keep the collection and labeling rules fixed and track lot/product dependence.

Under an independent-binomial planning assumption, with true recall 97.5%,
500 defective items have about **80.9% probability** of producing a one-sided
97.5% recall lower bound >=95%. At that size, at most **15 misses** satisfy the
recall bound. With only 110 defects, at most one miss satisfies that bound,
and the same true 97.5% detector has only about 23.6% probability of passing it.
These are planning calculations, not measured factory performance or guarantees.

Score the sealed batch once. Require all of:

- Complete image accounting and technical-rejection reporting.
- Recall >=95% and normal false-positive rate <10% as point estimates.
- The one-sided exact 97.5% recall lower bound >=95%, and the one-sided exact
  97.5% false-positive upper bound <10%. These two bounds give at least 95%
  simultaneous coverage under the stated sampling assumptions.
- No product, source-image, duplicate-group, calibration, or fitting overlap.
- Clear reporting by lot/shift and predeclared defect-size groups; sufficient
  independent lots to avoid treating correlated views as independent items.

If it fails, keep the result, use it as observed development evidence, and reserve
a different untouched batch for the next model. Do not retune the threshold or
choose a better seed on this confirmation batch and call that validation.
Even a successful study supports the sampled operating conditions; it cannot
establish absolute certainty for every future image or a different factory.

Statistical implementation: [SciPy exact binomial intervals](https://docs.scipy.org/doc/scipy-1.15.2/reference/generated/scipy.stats._result_classes.BinomTestResult.proportion_ci.html).
