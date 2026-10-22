"""Report the verified prospective ensemble without changing its primary rule."""
import argparse
from pathlib import Path

from component_inspector.research.ksdd2_seed_ensemble import read


def pct(value): return f'{100*value:.2f}%'


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True); args=parser.parse_args()
    root=args.root.resolve(); p=read(root/'protocol.json'); raw=read(root/'summary.json')
    summary=read(root/'audit-summary.json'); v=read(root/'verification.json')
    main_root=Path(p['main_root']); main_summary=read(main_root/'summary.json')
    assert raw['protocol_identity']==summary['protocol_identity']==v['protocol_identity']==p['identity']
    passed=summary['all_six_primary_ensemble_goals_met']; method=raw['method']; checks=summary['checks']
    outcome=('All six predeclared ensemble folds pass the overall point targets.' if passed else
             'The fixed ensemble does not meet the requested overall targets in every fold.')
    lines=['# Prospective three-seed ensemble: verified internal results','',f'**{outcome}**','',
           f'The inherited family is **{method}**. The primary alpha remains **0.05**; '
           'alpha .08 is diagnostic and cannot replace a failed primary. Each fold requires '
           'overall defective-image recall >=95% and normal false-positive rate <10%. '
           'Small-defect counts are reported separately; overall success does not establish '
           '95% recall within that subset.', '',
           f'The preceding single-model sweep passed all twelve of its own checks: '
           f'**{main_summary["all_twelve_internal_goals_met"]}**. Its '
           '[results](2026-10-20-warm-small-results.md) remain unchanged. This extension '
           'combines three members into one method and has six ensemble fold cases; it '
           'does not relabel failed single-model checks as successful or claim three '
           'independent ensemble repeats.', '',
           'Both partitions reuse the same 1,914 previously observed KSDD2 training '
           'originals. Each partition has 246 defective and 1,668 normal images, including '
           '45 small-defect images. Member/partition denominators are not added to '
           'manufacture an independent sample. The official test remained closed, the '
           '417 reserved normals remained unused, and no separate factory batch exists. '
           'These are internal development and partition/initialization robustness checks, '
           'not independent factory validation or a production guarantee.', '',
           '## All six primary-policy fold results','',
           '| Stage | Fold | TP / 82 | FP / 556 | Recall | Normal FPR | Small / 15 | Large / 67 | Goal |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for stage in ('dev','confirm'):
        for fold in range(3):
            row=checks[f'{stage}-f{fold}-alpha-0.05.json']; c=row['counts']
            lines.append(f'| {stage} | {fold} | {c["TP"]}/82 | {c["FP"]}/556 | {pct(row["recall"])} | {pct(row["fpr"])} | {row["small_defects"]["detected"]}/15 | {row["large_defects"]["detected"]}/67 | {"Pass" if row["point_goal_met"] else "Fail"} |')
    lines += ['', '## Aggregates, kept separate by partition','',
              '| Stage | Primary policy | TP / 246 | FP / 1,668 | Recall | Normal FPR | Small / 45 |',
              '|---|---|---:|---:|---:|---:|---:|']
    for stage in ('dev','confirm'):
        rows=[checks[f'{stage}-f{f}-alpha-0.05.json'] for f in range(3)]
        tp=sum(r['counts']['TP'] for r in rows); fp=sum(r['counts']['FP'] for r in rows)
        small=sum(r['small_defects']['detected'] for r in rows)
        lines.append(f'| {stage} | alpha .05 | {tp}/246 | {fp}/1,668 | {pct(tp/246)} | {pct(fp/1668)} | {small}/45 |')
    lines += ['', 'A passing aggregate does not rescue a failed fold. Pooled raw scores '
              'from differently fitted/calibrated fold models are not used to claim a '
              'single AUROC/AP. Stored binomial intervals are descriptive under the '
              'overlapping-data setting; they are not formal exact CV guarantees.', '',
              '## Same-policy member comparisons','',
              '| Stage | Member seed | Recipe | TP / 246 | FP / 1,668 | Recall | Normal FPR | Small / 45 |',
              '|---|---:|---|---:|---:|---:|---:|---:|']
    methods=list(dict.fromkeys(['baseline84',method]))
    for stage in ('dev','confirm'):
        seeds=p['config']['development_members' if stage=='dev' else 'confirmation_members']
        for seed in seeds:
            location=main_root if stage=='dev' or seed==4220 else Path(p['members'][str(seed)]['root'])
            for candidate in methods:
                rows=[read(location/f'{stage}-f{fold}-s{seed}-{candidate}/alpha-0.05.json') for fold in range(3)]
                tp=sum(r['metrics']['counts']['TP'] for r in rows); fp=sum(r['metrics']['counts']['FP'] for r in rows)
                small=sum(r['small_defects']['detected'] for r in rows)
                lines.append(f'| {stage} | {seed} | {candidate} | {tp}/246 | {fp}/1,668 | {pct(tp/246)} | {pct(fp/1668)} | {small}/45 |')
    lines += ['', 'These are fixed members and matched baseline controls, not alternatives '
              'from which a winning seed was selected. Per-original small/large detections '
              'gained or lost, and normal false alarms added or avoided, are retained for '
              'each member pair in [paired changes](seed-ensemble/paired-changes.json).', '',
              '## Diagnostic policy: every fold, no substitution','',
              '| Stage | Fold | TP / 82 | FP / 556 | Recall | Normal FPR | Small / 15 |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for stage in ('dev','confirm'):
        for fold in range(3):
            r=checks[f'{stage}-f{fold}-alpha-0.08.json']; c=r['counts']
            lines.append(f'| {stage} | {fold} | {c["TP"]}/82 | {c["FP"]}/556 | {pct(r["recall"])} | {pct(r["fpr"])} | {r["small_defects"]["detected"]}/15 |')
    lines += ['', '## Fixed method and verification','',
              'The family was inherited from the original development-only selection. '
              'The extension was registered before that selection and before either '
              'second-partition or ensemble outcomes. Development members are 4217/4218/4219; '
              'second-partition members are 4220/4221/4222. The added members reuse the '
              'original numerical worker, training roles and schedules, with their '
              'separately published initialization seeds. All three positive-oriented '
              'image logits are sigmoid-transformed and equally averaged. No best-seed '
              'gate, fitted temperature, fusion weighting or post-result threshold is used. '
              'The complete ensemble score is calibrated once on its 200 normal parents.', '',
              f'The independent arithmetic/role audit checked {v["raw_training_mask_hashes_and_labels_checked"]} '
              f'raw mask hashes/labels, {v["feature_artifact_hashes_checked"]} feature artifacts, '
              f'{v["normalization_sets_reconstructed"]} base-moment sets, '
              f'{v["hard_normal_scores_reconstructed"]} epoch60 hard-normal scores, '
              f'{v["calibration_scores_reconstructed"]} direct-weight calibration scores, and '
              f'{v["training_crop_choices_reconstructed"]} crop presentation choices. It checked '
              'all roles, losses/usage schedules, parent checkpoint links, preserved classifier '
              'states, thresholds, component counts and equal-weight ensemble arithmetic '
              'using a separate SciPy expit calculation. All parent-study evidence bytes '
              'matched its export index.', '',
              f'Native replay regenerated {v["new_member_head_evaluations_from_rgb"]} new-member '
              f'head scores and {v["new_member_threshold_evaluations_from_rgb"]} decisions '
              'from original RGB without masks or cached arrays. Combining those scores '
              'with the original study’s independently replayed native scores reproduced '
              f'all {v["ensemble_decisions_from_independent_native_member_scores"]} ensemble '
              f'decisions with zero flag mismatches; maximum score difference '
              f'{v["max_native_ensemble_score_difference"]:.12g}. These repeated checks do '
              'not add independent images. The audit is from the same work session, not '
              'third-party certification.', '',
              'Eight new ensemble preflight tests passed; the combined focused run with '
              'the nine arithmetic-audit tests passed all 17. The original full preflight '
              'passed 131 tests with 45 upstream warnings. Timings and RSS are measured '
              'resource records, not controlled latency benchmarks. Every fit/replay/audit '
              'phase uses four threads, 1,800 seconds and min(4GiB,60% free RAM).', '',
              'The original standby interruption and exact recovery remain documented in '
              'the [main study](2026-10-20-warm-small-results.md). No model-outcome-based '
              'retry or seed removal occurs in this extension. Generalization beyond '
              'observed KSDD2 images still requires the '
              '[fresh factory confirmation plan](2026-10-20-fresh-confirmation-plan.md).', '',
              'See the [prospective protocol](2026-10-21-seed-ensemble-protocol.md) for '
              'the original Deep Ensembles paper and industrial ensemble precedent. '
              'This is an adaptation of those ideas; their published accuracy is not ours.', '',
              f'Protocol identity: `{p["identity"]}`. Numerical source commit: `{p["source_commit"]}`. '
              'Source/configuration and new member protocols were published before scoring '
              'or added fitting. Evidence: [preregistration](2026-10-21-seed-ensemble-preregistration.json), '
              '[summary](seed-ensemble/audit-summary.json), [verification](seed-ensemble/verification.json), '
              '[export hash index](seed-ensemble/export-index.json).', '',
              '## Reproduce the preserved audit','',
              'Use the locked environment, preserved originals/caches, member checkpoints '
              'and main-study proof. Use a new empty export path; do not overwrite archived '
              'fit or proof directories. New fitting needs new paths and a newly published '
              'registration. Checkpoints and arrays remain local with recorded hashes.', '',
              '```powershell',
              f'components/.venv-p0/Scripts/python.exe components/tools/verify_seed_ensemble.py --root {str(args.root).replace(chr(92),"/")} --export docs/research/seed-ensemble-audit-repeat',
              '```','']
    with args.output.open('x',encoding='utf-8',newline='\n') as stream: stream.write('\n'.join(lines))
    print(args.output)


if __name__=='__main__': main()
