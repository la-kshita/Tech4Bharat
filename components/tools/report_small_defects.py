"""Write the completed study report directly from independently verified records."""
import argparse
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def percent(x):
    return f'{100*x:.2f}%'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); root = args.root.resolve()
    p = read(root/'protocol.json'); selection = read(root/'selection.json')
    s = read(root/'nested-results.json'); paired = read(root/'paired-comparison.json')
    stability = read(root/'refit-stability.json'); verification = read(root/'verification.json')
    frozen = read(root/'final/frozen.json'); timings = read(root/'timing-and-storage.json')
    legacy = read(root/'legacy-comparison.json')
    c = s['nested_metrics']['counts']; control = paired['fixed_control_counts']
    reference = legacy['outcomes']['0.08']
    recall, fpr = c['TP']/246, c['FP']/1668
    small = s['small_defects']; small_control = paired['control_small_defects']
    goal = s['nested_confidence']['point_goal_met']
    stable = stability['all_three_point_goals_met']
    verdict = ('The primary and both predeclared refit repeats meet the internal point target.' if goal and stable else
               'The primary meets the internal point target, but at least one refit repeat does not.' if goal else
               'The primary internal nested result fails the requested point target.')
    control_verdict = (
        'The selected procedure did not improve on the simpler paired control: it caught '
        'no more defects and produced no fewer false alarms.'
        if c['TP'] <= control['TP'] and c['FP'] >= control['FP'] else
        'The selected procedure caught at least as many defects with no more false alarms '
        'than the paired control on this internal cohort.'
        if c['TP'] >= control['TP'] and c['FP'] <= control['FP'] else
        'The paired comparison shows a recall/false-alarm tradeoff; it is not an unqualified improvement.'
    )
    lines = [
        '# Small-defect and overfitting study: verified training-only results', '',
        f'**{verdict}** The selected nested procedure caught **{c["TP"]}/246 defects '
        f'({percent(recall)} recall)** with **{c["FP"]}/1,668 normal false alarms ({percent(fpr)})**. '
        'The official test set was not rescored. These are internal cross-validation results, '
        'not new independent factory proof.', '',
        control_verdict, '',
        f'The full prior-recipe reference averages **{percent(reference["recall"])} recall / '
        f'{percent(reference["normal_fpr"])} normal FPR** at the fixed .08 calibration policy. '
        'Its fold results below determine consistency; a passing pooled average on reused '
        'training images does not establish reliable 95% performance or improve the earlier '
        'official-test benchmark.', '',
        '## What this result means', '',
        'KSDD2 contains industrial surface images with visible imperfections such as scratches '
        'and spots. The reported goal is image-level detection: flag an image containing a '
        'defect, while keeping false flags on normal images below the budget. Small-defect '
        'recall is also image-level, stratified by ground-truth mask area. These counts do '
        'not establish pixel-level localization accuracy or cable-gland performance.', '',
        'The study used only official KSDD2 training originals. Whole audited appearance groups '
        'were assigned to three folds, each containing 82 defects, 556 normals, and 15 small-defect '
        'images. Seven fixed candidates were compared using inner folds. Each outer-fold model '
        'was selected, fitted, and calibrated without that outer fold. All choices and the final '
        'candidate were frozen before outer predictions. The final candidate was selected from '
        'inner comparisons across all training data; the nested result estimates the fold-specific '
        'selection procedure, not an independent test of that final checkpoint.', '',
        'These images had already been used or observed during earlier project work. Nested '
        'separation reduces new leakage and selection optimism; it cannot erase historical '
        'exposure or establish unknown physical-product independence. There is still no new '
        'sealed, blindly labeled production batch. A better internal result must not be advertised '
        'as a confirmed improvement from the earlier 90.91% official-test result: the cohorts differ.', '',
        '## Paired outer comparison on the same images', '',
        '| Procedure | TP / FN | FP / normal count | Recall | Normal FPR | Small defects caught |',
        '|---|---:|---:|---:|---:|---:|',
        f'| Fixed coarse supervised control | {control["TP"]} / {control["FN"]} | {control["FP"]} / 1,668 | {percent(control["TP"]/246)} | {percent(control["FP"]/1668)} | {small_control["detected"]}/45 |',
        f'| Inner-selected nested procedure | {c["TP"]} / {c["FN"]} | {c["FP"]} / 1,668 | {percent(recall)} | {percent(fpr)} | {small["detected"]}/45 |', '',
        f'The paired difference is **{100*paired["selected_minus_control_recall"]:+.2f} percentage '
        f'points recall** and **{100*paired["selected_minus_control_fpr"]:+.2f} points FPR**. '
        f'The selected procedure gained {len(paired["defects_gained_ids"])} defects and lost '
        f'{len(paired["defects_lost_ids"])} relative to the control; it cleared '
        f'{len(paired["false_alarms_cleared_ids"])} old false alarms and introduced '
        f'{len(paired["new_false_alarms_ids"])} new ones. These comparisons use identical outer '
        'originals, fitting IDs, and calibration IDs. They are internal paired observations, '
        'not a formal claim of significance under cross-validation dependence.', '',
        '## Full prior-recipe reference', '',
        'The common-budget control above uses 30 epochs. A separately preregistered '
        'reference reproduces the prior best warm-corrected recipe with 60 base epochs, '
        '12 hard-normal refinement epochs, and 12 aligned-objective continuation epochs. '
        'It uses seed4217, 128 base normals plus 128 mined normals, the original whole-canvas '
        'normalization, and the same outer positives and calibration IDs. This reference '
        'is not an additional selectable candidate and cannot change the frozen final model.', '',
        '| Fixed full prior recipe operating policy | TP / FN | FP / 1,668 | Recall | Normal FPR | Small / 45 |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for alpha, row in legacy['outcomes'].items():
        ct, sm = row['counts'], row['small_defects']
        lines.append(f'| Calibration alpha {alpha} | {ct["TP"]} / {ct["FN"]} | {ct["FP"]} / 1,668 | {percent(row["recall"])} | {percent(row["normal_fpr"])} | {sm["detected"]}/45 |')
    lines += ['', '| Reference fold | Calibration alpha | TP / 82 | FP / 556 | Recall | Normal FPR | Small / 15 |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for fold, policies in legacy['folds'].items():
        for alpha, row in policies.items():
            ct, sm = row['counts'], row['small_defects']
            lines.append(f'| {fold} | {alpha} | {ct["TP"]}/82 | {ct["FP"]}/556 | {percent(ct["TP"]/82)} | {percent(ct["FP"]/556)} | {sm["detected"]}/15 |')
    old = legacy['outcomes']['0.08']['counts']
    common_all_folds = all(row['0.08']['counts']['TP']/82 >= .95 and row['0.08']['counts']['FP']/556 < .1
                           for row in legacy['folds'].values())
    lines += ['', f'At the common .08 calibration policy, the selected nested procedure caught '
              f'**{c["TP"]-old["TP"]:+d} defects** and made **{c["FP"]-old["FP"]:+d} false alarms** '
              'relative to the full prior recipe. Both .08 and the historical .10 policy '
              'were fixed before outer outcomes. Do not claim an overall improvement over '
              'the full prior recipe unless these comparisons support it; a gain over the '
              'shorter control alone is insufficient. See '
              '[full-recipe comparison](small-defects/legacy-comparison.json).', '',
        ('The full prior recipe meets both point targets in every common-policy fold.' if common_all_folds else
         'The full prior recipe fails at least one common-policy fold target. Even a passing pooled average would not establish consistent per-fold performance.'), '',
        '## Every outer fold', '',
        '| Outer fold | Inner-selected method | TP / FN | FP / 556 | Recall | Normal FPR | Small / 15 |',
        '|---|---|---:|---:|---:|---:|---:|',
    ]
    for fold, row in s['outer_folds'].items():
        m = row['metrics']; ct = m['counts']; sm = row['small_defects']
        lines.append(f'| {fold} | {row["method"]} | {ct["TP"]} / {ct["FN"]} | {ct["FP"]} / 556 | {percent(m["recall"]["rate"])} | {percent(m["accepted_normal_fpr"]["rate"])} | {sm["detected"]}/15 |')
    lines += ['', 'All outer images were accounted for exactly once. Raw logits and distance scores '
              'from different outer models are not pooled into one AUROC/AP. Training-image '
              'diagnostics are kept separately and cannot qualify unseen performance.', '',
              '## Every candidate and experiment', '',
              'Each row summarizes six directed inner comparisons: three independently fitted '
              'one-fold models, each evaluated on the other two folds. An image appears twice '
              'in those inner comparisons; the six results are not six independent datasets.', '',
              '| Candidate | Fitting recall range | Held-out inner recall range | Inner FPR range | Small-defect recall range | Inner point-goal cases / 6 | Total fit seconds |',
              '|---|---:|---:|---:|---:|---:|---:|']
    inner_records = {}
    for method in p['config']['methods']:
        records, fit_recalls, fit_seconds = [], [], 0.
        for train in range(3):
            directory = root/f'inner-{method}-{train}'
            if not (directory/'complete.json').exists(): continue
            fit_seconds += read(directory/'training.json')['fit_seconds']
            if p['config']['methods'][method]['family'] == 'supervised':
                fit_recalls.append(read(directory/'fitting-diagnostic.json')['metrics']['recall']['rate'])
            for val in range(3):
                if train != val: records.append((train, val, read(directory/f'eval-{val}.json')))
        inner_records[method] = records
        if len(records) != 6:
            lines.append(f'| {method} | Incomplete; preserved failure | — | — | — | — | — |'); continue
        recalls = [r['metrics']['recall']['rate'] for _, _, r in records]
        fprs = [r['metrics']['accepted_normal_fpr']['rate'] for _, _, r in records]
        small_rates = [r['small_defects']['detected']/r['small_defects']['total'] for _, _, r in records]
        passed = sum(a >= .95 and b < .1 for a, b in zip(recalls, fprs))
        fit_range = f'{percent(min(fit_recalls))}–{percent(max(fit_recalls))}' if fit_recalls else 'Not recorded for non-supervised methods'
        lines.append(f'| {method} | {fit_range} | {percent(min(recalls))}–{percent(max(recalls))} | {percent(min(fprs))}–{percent(max(fprs))} | {percent(min(small_rates))}–{percent(max(small_rates))} | {passed}/6 | {fit_seconds:.2f} |')
    lines += ['', 'The exact configurations and industrial sources are in the '
              '[preregistered protocol](2026-10-20-small-defects-protocol.md). The regularized '
              'heads are experimental bundles, compact PatchCore is not the paper-scale '
              'WideResNet implementation, PCA is a linear reconstruction control, and SciPy '
              'morphology is not proprietary HALCON software. EfficientAD/DRAEM were researched '
              'but are not recorded as trained experiments.', '',
              '## Refit stability without seed selection', '',
              '| Fixed outer-refit seed set | TP / FN | FP / 1,668 | Recall | Normal FPR | Point goal |',
              '|---|---:|---:|---:|---:|---|',
              f'| Primary | {c["TP"]} / {c["FN"]} | {c["FP"]} / 1,668 | {percent(recall)} | {percent(fpr)} | {"Pass" if goal else "Fail"} |']
    for offset, row in stability['repeats'].items():
        ct = row['counts']
        lines.append(f'| Offset +{offset} | {ct["TP"]} / {ct["FN"]} | {ct["FP"]} / 1,668 | {percent(row["recall"])} | {percent(row["normal_fpr"])} | {"Pass" if row["point_goal_met"] else "Fail"} |')
    lines += ['', 'Each repeat holds the fold-specific inner-selected method and data membership '
              'fixed, then changes outer-refit initialization/order only. The entire inner search '
              'was not repeated. These are the same 246 defects and 1,668 normals each time; '
              'the denominators are not tripled. No repeat replaces the primary or final model.', '',
              '## Small defects, uncertainty, and overfitting', '',
              f'The nested procedure detected **{small["detected"]}/45 small defects '
              f'({percent(small["detected"]/45)})**, versus '
              f'{small_control["detected"]}/45 for the paired control. Small means a source mask '
              'occupying at most 0.5% of the original image, not a resized feature-mask fraction. '
              'There are only 15 small defects per fold; retain the counts instead of claiming '
              'a precise factory-wide small-defect guarantee.', '',
              f'Of the {c["FN"]} missed defects, {45-small["detected"]} were in the small-defect '
              f'stratum. Larger defects were caught in {c["TP"]-small["detected"]}/201 cases. '
              'This is a diagnostic of the frozen result, not permission to retune on these '
              'outer examples. Training fit can look excellent while this held-out small-defect '
              'slice remains weak.', '',
              f'Descriptive Clopper-Pearson references on pooled counts are '
              f'{percent(s["nested_confidence"]["recall_exact_95"][0])}–{percent(s["nested_confidence"]["recall_exact_95"][1])} recall '
              f'and {percent(s["nested_confidence"]["fpr_exact_95"][0])}–{percent(s["nested_confidence"]["fpr_exact_95"][1])} normal FPR. '
              '**Overlapping cross-validation training sets make errors dependent, so these '
              'binomial intervals are not formal exact CV confidence or independent proof '
              'for the final candidate.** Point targets, descriptive bound targets, refit '
              'consistency, and production confirmation are separate claims.', '',
              'No positive training image lost its entire valid target. Some projected border '
              'target cells are outside the nearest-sampled ROI: 59 coarse-grid and 40 fine-grid '
              'images have this partial boundary effect. It is preserved in the target-coverage '
              'audit, not silently fixed after results. Fine features retain more such targets, '
              'but this does not establish perfect boundary representation.', '',
              '## Final candidate and verifiable evidence', '',
              f'Final inner-selected method: **{frozen["method"]}**. Head/model SHA-256: '
              f'`{frozen["model_sha256"]}`. Frozen strict threshold: '
              f'**score > {frozen["threshold_record"]["threshold"]!r}**. Its normal-only '
              'calibration uses the reserved 417 originals after method selection. These '
              'calibration scores are not a separate performance test.', '',
              f'Native/cache parity on 16 fixed final-candidate images: maximum score '
              f'difference {frozen["max_native_cache_score_difference"]!r}; all decisions '
              f'matched. Native calibration-pipeline p95 was '
              f'{frozen["native_calibration_pipeline_p95_seconds"]:.4f} seconds on CPU. '
              'Outer heads were reloaded and replayed in reverse cached-feature order. '
              'Parity does not prove generalization.', '',
              'A separate full native replay regenerated all **11,484 primary/control/repeat/legacy '
              'model-threshold evaluations on the 1,914 original RGB images**, including the two '
              'predeclared legacy thresholds on the same weights, without reading cached feature '
              'arrays, and checked the archived decisions. See '
              '[native replay](small-defects/full-native-replay.json) and '
              '[native prediction rows](small-defects/full-native-predictions.json).', '',
              f'Independent verification recounted every inner/outer/control/repeat row, '
              f'verified {verification["feature_artifact_hashes_checked"]} feature artifacts, '
              f'reconstructed {verification["normal_only_moments_independently_reconstructed"]} '
              'distinct normal-only moment sets, checked selections against permitted inner '
              'inputs, and checked the immutable source/model/threshold lineage. Old benchmark '
              'evidence is unchanged. Software suite before training: **125 passed**, '
              '45 upstream warnings. The arithmetic verifier is a separate implementation '
              'authored in this same work session; no third-party certification is claimed.', '',
              ('The first audit attempt incorrectly expected supervised fitting diagnostics '
              'for unsupervised methods and stopped with a missing-file error. The audit '
              'and report readers were corrected to match the frozen output schema. '
              'No models, selection rules, thresholds, or predictions changed. The '
              '[failed audit attempt](small-defects/verification-attempt-1.json) is preserved.'
               if (root/'verification-attempt-1.json').exists() else ''), '',
              f'Protocol identity: `{p["identity"]}`. Frozen main source commit: '
              f'`{p["source_commit"]}`. Source/config hashes in the protocol are authoritative. '
              f'New feature arrays contain {timings["new_feature_array_bytes"]/1024**3:.2f} GiB of logical data; '
              'model tensors and arrays remain local. Exact predictions, IDs, hashes, losses, '
              'selection inputs, timestamps, and resource logs are exported.', '',
              'A lossless NTFS compression intervention reduced generated cache disk allocation '
              'during the study. It did not change array values, fitting rules, models, or '
              'thresholds; the final verifier checks every feature hash. Compression overlapped '
              'one fitting phase, so elapsed times are not clean hardware speed comparisons. '
              'See [storage intervention](small-defects/cache-storage-optimization.json).', '',
              'Evidence: [nested outcomes](small-defects/nested-results.json), '
              '[paired comparison](small-defects/paired-comparison.json), '
              '[all refit repeats](small-defects/refit-stability.json), '
              '[independent verification](small-defects/verification.json), '
              '[final candidate](small-defects/final--frozen.json), '
              '[selection record](small-defects/selection.json), '
              '[timing/storage](small-defects/timing-and-storage.json), '
              '[complete byte-hash index](small-defects/export-index.json).', '',
              'A local ZIP preserves all audited model checkpoints, the exact backbone, '
              'source code, and exported evidence. Its [package receipt]'
              '(2026-10-20-small-defects-checkpoint-package.json) records the archive hash '
              'and location. All member hashes are verified after writing. It excludes '
              'dataset images and cached arrays; it is a checkpoint backup, not a new '
              'evaluation or a standalone installer.', '',
              ('The first ZIP membership check failed because Windows checkpoint paths '
               'were compared with ZIP-normalized paths. The first ZIP is preserved; '
               'the corrected archive uses normalized member names and passes all hash '
               'checks. See [packaging attempt](2026-10-20-small-defects-package-attempt-1.json).'
               if (args.output.parent/'2026-10-20-small-defects-package-attempt-1.json').exists() else ''), '',
              '## Every directed inner result', '']
    for method, records in inner_records.items():
        lines += [f'### {method}', '', '| Train fold → validation fold | TP / 82 | FP / 556 | Small / 15 |', '|---|---:|---:|---:|']
        for train, val, row in records:
            ct, sm = row['metrics']['counts'], row['small_defects']
            lines.append(f'| {train} → {val} | {ct["TP"]}/82 | {ct["FP"]}/556 | {sm["detected"]}/15 |')
        lines.append('')
    lines += ['## Reproduce', '',
              'The verifier defaults are bound to this published study and its two registration '
              'snapshots. Recount the preserved local study with a new empty export directory. '
              'A completely new fit also requires newly published registration snapshots before '
              'outer inference; the original snapshots cannot match a new timestamp/path-based '
              'protocol identity. Prepare those copies in an isolated repository using the '
              'published JSON formats and preserve this original study. Source/data path '
              'setup and the parent feature cache are prerequisites; this is not a standalone '
              'fresh-machine installer.', '',
              'Use new empty output/export paths and the locked recall environment. Freeze and '
              'register the supplements, publish the matching registration snapshots in that '
              'isolated copy before any outer evaluation, then run serially:', '', '```powershell',
              'components/.venv-p0/Scripts/python.exe -m component_inspector.research.ksdd2_small_defects --phase freeze --root components/outputs/recall-small-defects',
              'components/.venv-p0/Scripts/python.exe components/tools/run_control_small_defects.py --phase register --root components/outputs/recall-small-defects',
              'components/.venv-p0/Scripts/python.exe components/tools/run_repeat_small_defects.py --phase register --root components/outputs/recall-small-defects',
              'components/.venv-p0/Scripts/python.exe components/tools/run_legacy_small_defects.py --phase register --root components/outputs/recall-small-defects',
              'components/.venv-p0/Scripts/python.exe components/tools/run_small_defects.py --root components/outputs/recall-small-defects',
              'components/.venv-p0/Scripts/python.exe components/tools/run_control_small_defects.py --phase run --root components/outputs/recall-small-defects',
              'components/.venv-p0/Scripts/python.exe components/tools/run_repeat_small_defects.py --phase run --root components/outputs/recall-small-defects',
              'components/.venv-p0/Scripts/python.exe components/tools/run_legacy_small_defects.py --phase run --root components/outputs/recall-small-defects',
              'components/.venv-p0/Scripts/python.exe components/tools/replay_native_small_defects.py --root components/outputs/recall-small-defects',
              'components/.venv-p0/Scripts/python.exe components/tools/verify_small_defects.py --root components/outputs/recall-small-defects --export docs/research/small-defects',
              '```', '',
              'Independent factory confirmation remains incomplete. Keep the '
              '[fresh-image confirmation plan](2026-10-20-fresh-confirmation-plan.md); '
              'do not reopen or retune the official test to manufacture a passing result.', '',
              '## Next development hypothesis', '',
              'First isolate training schedule with matched objectives, data, capacity, and '
              'regularization: the 30-epoch candidates and full prior recipe differ in several '
              'factors, so their comparison cannot identify a duration effect. Another useful '
              'hypothesis is native-resolution overlapping tiles with mask-aware training crops '
              'to preserve tiny marks before the backbone downsamples them. Test these as '
              'newly registered development studies with fixed normal-only calibration, '
              'not by changing thresholds on these outer predictions. The current bundled '
              'changes do not isolate resolution, dropout, capacity, Dice loss, or score pooling '
              'as individual causes. Keep single-factor controls when assessing a future gain. '
              'These additional hypotheses have not been trained in this study and are not '
              'claimed results.', '']
    with args.output.open('x', encoding='utf-8', newline='\n') as f: f.write('\n'.join(lines))
    print(args.output)


if __name__ == '__main__': main()
