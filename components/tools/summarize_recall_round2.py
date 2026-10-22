"""Export exact experiment records and a concise ranked observed-development table."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output-directory', type=Path, required=True)
    args = parser.parse_args()
    root, dest = args.root.resolve(), args.output_directory.resolve()
    dest.mkdir(parents=True, exist_ok=False)
    summary = {'claim': 'bounded selection on observed KSDD2 development data; not independent validation',
               'commit_at_export': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
               'protocol': json.loads((root/'protocol.json').read_text()), 'candidates': [],
               'repeat_training': [], 'exported_files': {}}
    for p in sorted(root.glob('*/result.json')):
        name = p.parent.name
        target = dest/f'{name}.json'
        shutil.copyfile(p, target)
        value = json.loads(p.read_text())
        summary['exported_files'][target.name] = hashlib.sha256(p.read_bytes()).hexdigest()
        if 'metrics' in value:
            compact = {k: v for k, v in value.items() if k not in ['predictions', 'calibration_scores']}
            if 'training_history' in compact.get('details', {}):
                detail = dict(compact['details'])
                history = detail.pop('training_history')
                detail.update(first_training_loss=history[0], final_training_loss=history[-1])
                compact['details'] = detail
            compact['source_result_file'] = target.name
            compact['meets_development_goal'] = bool(value['metrics']['recall']['rate'] >= .95 and value['metrics']['accepted_normal_fpr']['rate'] < .10)
            repeated = name.startswith('stability') or '_seed' in name
            summary['repeat_training' if repeated else 'candidates'].append(compact)
    for p in sorted(root.glob('*.json')):
        target = dest/p.name
        shutil.copyfile(p, target)
        summary['exported_files'][target.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    for p in sorted(root.glob('*/*.json')):
        if p.name not in {'model.json', 'training.json', 'index.json'}:
            continue
        target = dest/f'{p.parent.name}-{p.name}'
        shutil.copyfile(p, target)
        summary['exported_files'][target.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    def key(row):
        metric = row['metrics']
        return metric['accepted_normal_fpr']['rate'] < .10, metric['recall']['rate'], -metric['accepted_normal_fpr']['rate']
    summary['candidates'].sort(key=key, reverse=True)
    selected_path = root/'selected-for-stability.json'
    if selected_path.exists():
        summary['selected_canonical_configuration'] = json.loads(selected_path.read_text())
    with (dest/'summary.json').open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(summary, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')
    print('| Candidate | Detected defects | Normal false alarms | Recall | FPR |')
    print('|---|---:|---:|---:|---:|')
    for row in summary['candidates']:
        m = row['metrics']
        c = m['counts']
        print(f"| {row['method']} | {c['TP']}/82 | {c['FP']}/417 | {m['recall']['rate']:.2%} | {m['accepted_normal_fpr']['rate']:.2%} |")


if __name__ == '__main__':
    main()
