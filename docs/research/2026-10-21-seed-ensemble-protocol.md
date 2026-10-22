# Fixed three-seed ensemble extension

This is a **separate prospective extension**, not a repair of the warm sweep's
single-model outcomes or a claim that its twelve checks passed. On observed
development data, the longer-training control passed the first seed's three
folds, then caught only 76/82 defects on the second seed's first fold. The
small-defect-specific changes also failed a fold in the first seed. Initialization
sensitivity is a measured problem. All original runs and proof still finish.

Register this extension before the main sweep selects its family, before any of
its second-partition predictions, before computing any ensemble scores, and
before fitting either added seed. The source, configuration, parent protocol,
two member protocols and copied cache-index hashes are frozen and published.

## Method and primary policy

Use the family selected by the main sweep's existing **development-only** rule,
regardless of its second-partition outcomes. Do not pick a different family for
the ensemble after looking at those outcomes. For each image, transform each
of exactly three positive-oriented image logits with a numerically stable
sigmoid, then take their equally weighted arithmetic mean. These transformed
scores are not claimed to be calibrated defect probabilities. There is no
learned temperature, CDF normalization, fitted fusion, weighting search,
best-member gate or OR of three separately thresholded alarms.

Calibrate the **combined score** once on the same 200 disjoint normal parents
shared by all three members in that fold. The primary policy is fixed **alpha
0.05**, rank 191 of 200, strict `score > threshold`. Alpha .08 is diagnostic
only; it cannot rescue a failed primary qualification. No threshold or policy
will be chosen using ensemble development or second-partition results.

- Development: use all three existing seed4217/4218/4219 family heads for each
  of the three original development folds. Each evaluated original was excluded
  from every member's fitting, mining, normalization and calibration in its fold.
- Second partition: use existing seed4220 and fresh fixed seeds4221/4222 for each
  of the three predeclared group-stratified folds. New members use the original
  60+12+12 baseline and, if selected, its exact registered 24-epoch continuation.
  Each added member has its own published protocol overriding only the member
  initialization seed. Reuse the original numerical worker unchanged.
- Fit all registered new members, even if the first sweep's confirmation fails.
  No resampling, extra seed, new mask rule or model-outcome-based stopping.
- All fitting/calibration/held-out parent roles, masks, feature geometry, source
  backbone, capacities, optimizer schedules and exclusion of 417 reserved normals
  remain the original procedure. Native inference never uses GT-driven views.

The hypothesis is that averaging independent initializations can reduce variance
and recover complementary misses. It cannot create information all members
miss or eliminate bias. The general rationale is the original
[Deep Ensembles paper](https://arxiv.org/abs/1612.01474). Industrial anomaly
detection also uses ensembles in the authors' [PatchCore implementation](https://github.com/amazon-science/patchcore-inspection).
Our same-family supervised seed average is an adaptation, not a reproduction
of their architectures, uncertainty calibration or reported performance.

## Qualification and honest accounting

The new procedure has **six ensemble fold cases**: three development folds and
three second-partition folds. Every primary-policy case must catch >=95% of
defective images while falsely flagging <10% of normals. Overall image recall
and the small-defect subset are reported separately. A pooled average cannot
rescue a failed fold; the diagnostic policy cannot replace the primary.

Three members are combined into one method; they are not three independent
ensemble repeats. The two partitions use different fixed seed groups but repeat
the same 1,914 previously observed training originals. Counts are 246 defective
and 1,668 normal originals per partition, including 45 small defects; do not add
the denominators across members or partitions to invent a larger independent
sample. This does not relabel failed single-model checks as successful ones.

The official test stays closed and no fresh factory data exists. This remains
internal development/robustness evidence, even if every ensemble fold passes.
Physical product identities are unknown, and prior project exposure remains.
No production guarantee, pristine test result or absolute certainty is claimed.

## Verification and resource rules

Wait until the original serial fits, raw-RGB replay, independent audit and report
have finished before starting any added fitting. Keep one ML worker active,
four threads, 1,800 seconds and min(4GiB,60% free RAM) per phase. Keep the
process-scoped awake request while active, without altering global power/lid
settings. Any failed phase stops; preserve it and do not retry automatically.

Retain every new checkpoint, epoch60 mining checkpoint, prediction, normal
calibration score, fitting/calibration/mining identity, loss, view-use count and
resource log. Independently check original IDs, GT mask labels/small areas,
fitting-role exclusions, normalization, hard-normal rankings, classifier-state
preservation, continuation presentation choices, strict quantiles, counts,
equal-weight arithmetic and paired gains/losses against each same-policy member.

Replay all newly fitted evaluated heads from original RGB without feature caches
or masks, including matched baseline controls. Combine independently replayed
member scores with the main sweep's archived native scores; require ensemble
scores within 1e-5 and zero flag differences. Repeated head/policy evaluations
remain implementation checks, not additional independent images. Publish raw
evidence hashes, failures, exact denominators and all six fold outcomes.
