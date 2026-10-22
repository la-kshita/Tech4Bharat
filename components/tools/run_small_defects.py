"""Run the predeclared nested study serially with immutable resource records."""
import argparse
from pathlib import Path
import subprocess
import sys

import psutil

from component_inspector.research.ksdd2_round1 import read_json, write_json
from component_inspector.vision.resources import run_bounded


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    base = [sys.executable, '-m', 'component_inspector.research.ksdd2_small_defects', '--root', str(root)]
    if not (root/'protocol.json').exists():
        subprocess.run([*base, '--phase', 'freeze'], check=True)
    p = read_json(root/'protocol.json')

    def run(name, command, *, essential=True):
        result_path = root/f'{name}-resources.json'
        if result_path.exists():
            resource = read_json(result_path)
            if resource['status'] != 'completed' and essential:
                raise RuntimeError('Preserved essential phase failed: '+name)
            print('USE SAVED ATTEMPT', name, resource['status'], flush=True)
            return
        cap = min(4*1024**3, int(.60*psutil.virtual_memory().available))
        print('START', name, 'RSS cap', cap, flush=True)
        resources = run_bounded([*base, *command], root/f'{name}-worker.log',
                                max_rss_bytes=cap, max_seconds=1800)
        write_json(result_path, resources)
        print('END', name, resources, flush=True)
        if resources['status'] != 'completed' and essential:
            raise RuntimeError('Essential phase failed; preserve attempt: '+name)

    run('cache', ['--phase', 'cache'])
    for method in p['config']['methods']:
        for fold in range(3):
            run(f'inner-{method}-{fold}', ['--phase', 'inner', '--method', method, '--fold', str(fold)], essential=False)
    run('selection', ['--phase', 'select'])
    run('final', ['--phase', 'final'])
    for fold in range(3):
        run(f'outer-{fold}', ['--phase', 'outer', '--fold', str(fold)])
    run('summary', ['--phase', 'summarize'])


if __name__ == '__main__':
    main()
