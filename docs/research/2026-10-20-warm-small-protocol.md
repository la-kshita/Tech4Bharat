# Warm-recipe small-defect study: fixed protocol before new results

The previous full recipe averaged 95.93% recall / 8.81% normal FPR internally,
but missed a target in two of three folds. The newer seven-method selection
procedure reached 93.90% recall in all three refit seed sets. Neither supports a
reliable 95% factory claim. This study improves the previous recipe and keeps
the official test closed throughout.

## Fixed experiments

| Method | Fixed change from the full 84-epoch warm recipe |
|---|---|
| baseline84 | Unchanged 60 base + 12 hard-normal refinement + 12 aligned-objective epochs |
| longer108 | Another 24 aligned-objective epochs, AdamW learning rate .0002 |
| small108 | Same 24 epochs, with weight 3 for small-positive examples in focal pixel and image BCE losses |
| tiles108 | Same 24 epochs, with 50% probability of a native zoomed crop per example |
| tiles_small108 | The factorial combination of zoomed crops and small-positive weighting |

All continuations start from exactly the same fold/seed baseline checkpoint and
use the same fitting parents, ordering seed, capacity, normalization, loss
family, optimizer reset, batch size 4, and update count. The two single changes
and their combination are compared with the matched longer108 control.
The original baseline remains visible; a gain over a deliberately shorter
model is not sufficient evidence. Small means original defect-mask area <=0.5%.
Weights scale both loss components and normalize their respective denominators.

The baseline uses the previous 128 normal + fitting-positive whole-canvas
float32 moments, weighted focal pixels (positive weight 8, gamma 2), learned
image objective for 60+12 epochs, and corrected positive-oriented top-0.1%
image-score BCE for the final 12. Mine 128 additional normals only from the
permitted training pool, excluding calibration and evaluation parents. Reuse
the already audited seed4217 baseline checkpoints without changing their bytes;
refit the full recipe for seeds4218/4219 and the second partition's seed4220.
No new mining or normalization updates occur during the four continuations.

### Crops and inference

Rotate native RGB and GT consistently by 90 degrees clockwise. A fixed
160x160-pixel grid with stride128 and a final edge-aligned start covers the
entire native image. Current training shapes produce exactly ten crops.
Resize each crop to 256x256, extract the same frozen ResNet18 layer2/3 features,
then the same 3x3 feature average; cache 384x32x32 float16 features. This changes
apparent defect scale before downsampling. It is not synthetic defect creation.

Training uses a full image with probability 0.5 and a crop otherwise. Positive
parents choose uniformly among their own positive GT crops; normal parents
choose uniformly among all ten crops. Positive image labels therefore remain
true. All views inherit the original parent's fitting role. Record per-parent
view usage. Cropped feature views are centered in a 32x80 canvas for mixed
training batches; padding contributes to neither image pooling nor pixel loss.
Keep the baseline mean/std unchanged in every continuation.

At inference, read no masks or labels: score the full image and **all ten**
crops, taking the maximum of their positive-oriented top-ceil(0.001*N) valid-cell
mean logits. Batched square-crop inference is valid because the learned head
contains only pointwise convolutions. Calibrate the complete eleven-view score
on training-only normal parents; this accounts for the extra chances to flag an
image. Full-image methods retain their single-view score. No GT-guided crop
selection, defect-size gating, or best-view rule is allowed at inference.

The industrial precedent is the dataset authors' [mask-supervised SegDec-Net
work](https://www2.vicos.si/publications/bozic2021mixed/) and
[official implementation](https://github.com/vicoslab/mixed-segdec-net-comind2021).
[SAHI's original paper](https://arxiv.org/abs/2202.06934) motivates slicing and
slice fine-tuning for small-object detail. This study implements an adaptation
with our frozen backbone and segmentation head, not the full SegDec-Net or SAHI
detector or either paper's published performance.

## Data separation and selection

Use only the existing 1,914 official-training pool originals: 246 defects and
1,668 normals, including 45 small-defect images. Keep the 417 reserved normals
outside every fit, calibration, and selection in this study. The whole audited
appearance-group development folds remain the prior three folds (82 defects,
556 normals, 15 small defects each), giving exact paired comparisons and audited
baseline reuse. **These images and these folds have already been observed.**
All results are development evidence, not pristine outer validation.

Fit every method in every development fold for seeds4217/4218/4219. The held-out
fold is excluded from normalization, baseline fitting, mining, continuation,
and calibration. Use the same 200 calibration normals and same baseline fitting
parents for every paired method/seed in that fold. Ground-truth masks may guide
training crops only within the fitting parents. Frozen pretrained-feature
caching on all pool originals fits no data-dependent parameters.

Predeclare exactly two normal-only quantile policies, alpha .05 and .08, strict
score > threshold. These are development candidates, not a held-out threshold
scan. Never use official-test data. A policy qualifies only if **each of all
nine fold/seed cases** achieves recall >=95% and normal FPR <10%.

Select using development records only: first require all nine FPRs <10%; among
eligible policies maximize worst recall, then worst small-defect recall, then
mean recall; minimize mean FPR and then complexity (extra epochs, tiles,
weighting), method name, alpha. If no FPR-eligible policy exists, minimize
worst FPR before the other objectives and explicitly record failure. No seed is
selected or discarded. Missing/failed phases stop the suite with their records
preserved; they never become zero errors or successful cases.

Before new fitting, predeclare a second group-stratified partition using seed
20261021 with the same per-fold counts. Lock the selected method and policy
before any inference for this partition. Fresh seed4220 refits the selected
recipe and the full baseline on each of its three training complements. This
checks partition and initialization sensitivity without choosing a better
confirmation seed, method, or threshold afterward. Both baseline policies stay
visible; only the selected policy is the selected-procedure check.

**Internal consistency requires all nine development checks and all three
second-partition checks to pass.** Pooled averages cannot rescue a failing fold.
An aggregate improvement or best run is insufficient. The second partition
still reuses observed images; changing membership does not make independent
data. There is no available new factory batch. Physical product IDs remain
unknown; keep appearance groups whole. No absolute production guarantee or
independent factory confirmation will be claimed even if all checks pass.

## Proof and accounting

- Freeze source/config/dependency hashes and publish the protocol snapshot in
  Git before the first new cache or fit phase. Do not change numeric code,
  parameters, candidate list, fold assignments, policies, or seed list afterward.
- Retain every checkpoint, calibration score, per-original prediction, loss
  curve, fitting/mining/normalization parent ID, view-use count and resource log.
  Reused historical baseline predictions/calibration must reproduce exactly.
- Independently recount metrics, confirm strict quantiles, verify all role
  exclusions, selections, model hashes, feature hashes and per-parent usage.
  Reconstruct normalization and, for new baselines, hard-normal rankings from
  the saved epoch60 checkpoint and permitted training candidates.
- Replay every evaluated fold/seed model from original RGB pixels without
  cached arrays; score every inference view without GT and require decisions to
  match, maximum absolute score difference <=1e-5. Record exact equality too.
  Split replay into bounded phases if needed. Repeated seeds, policies, views
  and partitions do not increase the number of independent original images.
- Report per-fold and per-seed small/large-defect counts, paired gains/losses,
  and all failures. Do not pool raw score scales into AUROC across models.
  Any binomial bounds on overlapping CV counts are descriptive only.
- One owned ML worker at a time, four threads, 1,800 seconds per phase, RSS
  cap min(4GiB,60% currently available RAM). Preserve resource failures and
  diagnose them; never resample or silently overwrite a failed attempt.
- Preserve historical source, datasets, predictions and no-go decisions. Any
  implementation defect discovered after freezing is a recorded invalidation
  and separately preregistered corrected run, not a concealed repair of results.

Source: `components/src/component_inspector/research/ksdd2_warm_small.py`.
Configuration: `components/configs/recall_warm_small.json`. Runner:
`components/tools/run_warm_small.py`. Freeze explicitly, publish the snapshot,
then run the fixed serial suite. Audit/report tooling may be corrected with
failed audit attempts preserved; it cannot alter learned artifacts or outcomes.
