# Where the teammate resumes

**Completed by this PR:** ML comparison experiments, honest reporting,
selection of an experimental MVP candidate, exact frozen heads/calibration,
a portable prediction adapter, integrity checks and continuation documentation.

**Intentionally left for the teammate:** application/backend/frontend,
persistence, real human review, reports/deployment and any new factory study.
The reserved `application/` and `web/` directories remain scaffolds. No web
dependencies, fake app completion or fictional human findings are included.

## Suggested next implementation slice

1. Follow [start here](START_HERE.md), prepare the model and run one local image
   through the prediction CLI. Load `Ensemble` once in the eventual service.
2. Resume **T06**, application foundation, using the existing
   [execution plan](../../Tech4Bharat_Codex_Execution_Plan.md). Use only
   `ksdd2_surface` with this experimental selected ensemble. Historical PaDiM
   choices, cable-gland eligibility and old no-go status are context; do not
   silently activate a different recipe or claim the gate was passed.
3. Build one bounded local inspection endpoint, with concurrency 1, real media
   validation, missing/corrupt artifact failures and explicit busy/timeout
   responses. Do not turn errors into a not-flagged result. Keep any native
   request lock until inference has actually stopped.
4. Continue **T07** with input image, optional attention overlay, model identity,
   below-cutoff/flagged result, warnings and separate human decisions. Never use
   “AI approved.” Persist actual human events; mark the real participation step
   pending until someone uses it.
5. Continue **T08** batch/history/report work: latest unique-product state,
   duplicate/rework behavior, uncertainty, immutable observations, draft versus
   genuinely acknowledged reports. The original plan specifies these contracts.
6. T09 human timing study and T10 hosting/deployment are later work. No time
   benefit, public live service or factory acceptance is established here.

Keep the UI positive and useful: this is a reproducible model ready for an
assisted prototype. Explain its current boundaries in a visible model-details
panel and reports. Use the model card's exact numbers and context.

## Prompt to paste into the teammate's Codex

> Read AGENTS.md, docs/handoff/START_HERE.md, MODEL_CARD.md and NEXT_STEPS.md.
> Continue application work from T06 using the existing frozen
> ksdd2-ensemble-mvp-v1 model through vision/ksdd2_ensemble.py. First run the
> documented setup and smoke inference, then implement one bounded local
> image-inspection endpoint and the T07 human-review flow. Preserve all research
> evidence and the pinned score/threshold. Do not retrain, reopen/tune on the
> official test, claim production accuracy, or fabricate human review outcomes.
> Keep actual human participation pending while completing independent software
> work. Ask only for missing product decisions that cannot be inferred from
> the plans and current handoff.

## If someone resumes ML work later

Use a separate registered experiment. Native-resolution feature learning with
regularization is a proposed next hypothesis, not a measured improvement.
Independent confirmation needs a genuinely new batch, following the existing
fresh-data plan. Another seed or re-split of these images is internal evidence.
