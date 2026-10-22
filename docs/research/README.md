# ML research index

Start with the [plain-English decision](2026-10-22-recall-status.md). The
chosen frozen instance is described in the [model card](../handoff/MODEL_CARD.md).

| Work | Report | Machine-readable evidence |
|---|---|---|
| Fixed three-seed ensemble | [Complete results](2026-10-21-seed-ensemble-results.md) | `seed-ensemble/` |
| Fitting versus held-out gap | [Diagnostic](2026-10-21-fitting-gap-results.md) | `fitting-gap/` |
| Five warm-recipe/small-defect variants | [Complete results](2026-10-20-warm-small-results.md) | `warm-small/`, `warm-small-recovery/` |
| Earlier seven small-defect methods | [Results](2026-10-20-small-defects-results.md) | `small-defects/` |
| Earlier corrected official-test benchmark | [Results](2026-10-20-validation-results.md) | `validation/` |
| Training/inference objective correction | [Results](2026-10-20-alignment-results.md) | `alignment/` |
| Original method comparisons | [Round 1](2026-10-19-round1-results.md), [round 2](2026-10-20-round2-results.md) | `round1/`, `round2/` |
| Ten researched methods | [Strategy review](2026-10-19-recall-strategies.md) | Sources linked in report |
| Complete chronological record | [Experiment ledger](recall-experiment-ledger.md) | Referenced per-run records |
| Requirements for independent confirmation | [Fresh-data plan](2026-10-20-fresh-confirmation-plan.md) | Future work, not completed validation |

Reports distinguish development candidates, primary failures and diagnostic
results. JSON/log folders retain original names and bytes because their export
indices and protocols refer to them. Large local checkpoint archives and caches
remain ignored; the three selected deployable heads are in
`components/model_registry/ksdd2-ensemble-mvp-v1/`.
