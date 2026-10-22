"""Supervise the preregistered benchmark audit and two full inference passes."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import psutil

from component_inspector.research.ksdd2_round1 import write_json
from component_inspector.vision.resources import run_bounded


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    base = [sys.executable, '-m', 'component_inspector.research.ksdd2_validation', '--root', str(root)]
    if not (root/'protocol.json').exists():
        subprocess.run([*base, '--phase', 'freeze'], check=True)
    for phase in ('integrity', 'infer', 'replay', 'summarize'):
        # Existing results are never overwritten or used to tune another attempt.
        # A rerun requires inspection of the preserved attempt, not blind resume.
        cap = min(4*1024**3, int(.60*psutil.virtual_memory().available))
        print('START', phase, 'RSS cap', cap, flush=True)
        resources = run_bounded([*base, '--phase', phase], root/f'{phase}-worker.log',
                                max_rss_bytes=cap, max_seconds=1800)
        write_json(root/f'{phase}-resources.json', resources)
        print('END', phase, json.dumps(resources), flush=True)
        if resources['status'] != 'completed':
            raise RuntimeError(f'{phase} failed; preserved all artifacts; do not tune or blindly retry')


if __name__ == '__main__':
    main()
