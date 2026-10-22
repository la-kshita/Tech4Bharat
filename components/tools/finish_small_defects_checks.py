"""Run registered checks serially after the existing paired-control controller."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import psutil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--wait-control-pid', type=int)
    args = parser.parse_args()
    if args.wait_control_pid is not None:
        controller = psutil.Process(args.wait_control_pid)
        command = controller.cmdline()
        assert any(Path(part).name == 'run_control_small_defects.py' for part in command)
        assert '--phase' in command and command[command.index('--phase')+1] == 'run'
        print('Waiting for the existing control controller; no parallel fitting', flush=True)
        controller.wait(timeout=5400)
    for fold in range(3):
        record = json.loads((args.root/f'control-outer-{fold}-resources.json').read_text())
        assert record['status'] == 'completed', 'Preserve the failed control; do not refit automatically'
    for script, extra in [
        ('run_repeat_small_defects.py', ['--phase', 'run']),
        ('run_legacy_small_defects.py', ['--phase', 'run']),
        ('replay_native_small_defects.py', []),
    ]:
        print('SERIAL CHECK', script, flush=True)
        subprocess.run([sys.executable, str(Path(__file__).parent/script),
                        '--root', str(args.root), *extra], check=True)


if __name__ == '__main__':
    main()
