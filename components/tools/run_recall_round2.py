"""Serial round 2 worker supervision; preserves every attempted phase log."""
import argparse
import json
from pathlib import Path
import sys

import psutil

from component_inspector.vision.resources import run_bounded


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--phases', nargs='+', default=['aggregation', 'focal', 'lighting_cache', 'lighting', 'fine_cache', 'fine', 'padim50', 'ensemble', 'stress'])
    args = parser.parse_args()
    root = args.root.resolve()
    for phase in args.phases:
        if (root/f'{phase}-complete.json').exists():
            print('USE SAVED', phase, flush=True)
            continue
        suffix, attempt = phase, 1
        while (root/f'{suffix}-worker.log').exists():
            attempt += 1
            suffix = f'{phase}-attempt{attempt}'
        command = [sys.executable, '-m', 'component_inspector.research.ksdd2_round2',
                   '--phase', phase, '--root', str(root), '--weights', str(args.weights.resolve())]
        cap = min(4*1024**3, int(.60*psutil.virtual_memory().available))
        print('START', phase, 'RSS cap', cap, flush=True)
        resource = run_bounded(command, root/f'{suffix}-worker.log', max_rss_bytes=cap, max_seconds=1800)
        with (root/f'{suffix}-resources.json').open('x', encoding='utf-8', newline='\n') as handle:
            json.dump(resource, handle, indent=2, sort_keys=True)
            handle.write('\n')
        print('END', phase, json.dumps(resource), flush=True)
        if resource['status'] != 'completed':
            print('Phase incomplete; preserve evidence and continue unrelated phases', phase, flush=True)


if __name__ == '__main__':
    main()
