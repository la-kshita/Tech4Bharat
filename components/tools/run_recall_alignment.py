"""Run the corrected objective on all three fixed seeds, one worker at a time."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import psutil

from component_inspector.vision.resources import run_bounded


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--parent', type=Path, default=Path('components/outputs/recall-round1'))
    args = parser.parse_args()
    root = args.root.resolve()
    if not (root/'protocol.json').exists():
        subprocess.run([sys.executable, '-m', 'component_inspector.research.ksdd2_aligned',
                        '--phase', 'prepare', '--root', str(root), '--parent', str(args.parent.resolve())], check=True)
    for seed in [4217, 4218, 4219]:
        name = 'aligned' if seed == 4217 else f'aligned_seed{seed}'
        if (root/f'{name}-complete.json').exists():
            print('USE SAVED', name, flush=True)
            continue
        suffix, attempt = name, 1
        while (root/f'{suffix}-worker.log').exists():
            attempt += 1
            suffix = f'{name}-attempt{attempt}'
        command = [sys.executable, '-m', 'component_inspector.research.ksdd2_aligned',
                   '--phase', 'train', '--root', str(root), '--seed', str(seed)]
        cap = min(4*1024**3, int(.60*psutil.virtual_memory().available))
        print('START', name, 'RSS cap', cap, flush=True)
        resource = run_bounded(command, root/f'{suffix}-worker.log', max_rss_bytes=cap, max_seconds=1800)
        with (root/f'{suffix}-resources.json').open('x', encoding='utf-8', newline='\n') as handle:
            json.dump(resource, handle, indent=2, sort_keys=True)
            handle.write('\n')
        print('END', name, json.dumps(resource), flush=True)
        if resource['status'] != 'completed':
            print('Preserve unsuccessful attempt and continue other fixed seeds', flush=True)


if __name__ == '__main__':
    main()
