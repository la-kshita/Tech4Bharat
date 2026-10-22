"""Export measured development outputs and compact provenance into review docs."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

from component_inspector.data.acquire import sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = json.loads((root/'manifest.json').read_text())
    phases = {'baseline': 'baseline', 'supervised': 'supervised', 'patchcore': 'patchcore',
              'coverage128': 'coverage', 'coverage256': 'coverage'}
    output = {'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'claim': 'single-split KSDD2 development comparison; not independent confirmation',
              'code_commit_at_export': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              'manifest_identity': manifest['identity'], 'configuration': manifest['config'],
              'counts': manifest['counts'], 'methods': [], 'phase_resources': {}}
    for name, phase in phases.items():
        path = root/name/'result.json'
        resources = sorted(root.glob(f'{phase}*-resources.json'), key=lambda p: p.stat().st_mtime)
        if resources:
            resource = resources[-1]
            output['phase_resources'][phase] = json.loads(resource.read_text())
        if not path.exists():
            output['methods'].append({'method': name, 'status': 'not_completed'})
            continue
        value = json.loads(path.read_text())
        compact = {k: v for k, v in value.items() if k not in {'predictions', 'calibration_scores', 'training_history'}}
        compact.update(status='completed', raw_result_path=path.relative_to(root).as_posix(),
                       result_sha256=sha256(path), threshold_selected_from='417 reserved normal images only',
                       actual_goal_pass=bool(value['metrics']['recall']['rate'] >= .95
                                             and value['metrics']['accepted_normal_fpr']['rate'] < .10))
        if 'training_history' in value:
            compact['training_loss_first'] = value['training_history'][0]
            compact['training_loss_last'] = value['training_history'][-1]
        output['methods'].append(compact)
    if not any(v['status'] == 'completed' for v in output['methods']):
        raise ValueError('No measured results to export')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(output, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')
    print('| Method | TP | FN | FP | TN | Recall | Normal FPR | Goal |')
    print('|---|---:|---:|---:|---:|---:|---:|---|')
    for value in output['methods']:
        if value['status'] != 'completed':
            print(value['method'], value['status'])
            continue
        m = value['metrics']
        c = m['counts']
        print(f"| {value['method']} | {c['TP']} | {c['FN']} | {c['FP']} | {c['TN']} | {m['recall']['rate']:.2%} | {m['accepted_normal_fpr']['rate']:.2%} | {value['actual_goal_pass']} |")


if __name__ == '__main__':
    main()
