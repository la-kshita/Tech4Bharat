"""Run supplementary raw replay, independent audit and report after its fits."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

from component_inspector.vision.resources import run_bounded


def read(path): return json.loads(path.read_text(encoding='utf-8'))


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--wait-controller-pid',type=int)
    args=parser.parse_args(); root=args.root.resolve(); tools=Path(__file__).resolve().parent
    if args.wait_controller_pid is not None:
        process=psutil.Process(args.wait_controller_pid)
        assert any(Path(value).name=='run_seed_ensemble.py' for value in process.cmdline())
        deadline=time.monotonic()+43200
        print('WAIT for registered ensemble fits; no parallel proof worker',flush=True)
        while process.is_running():
            if time.monotonic()>deadline: raise TimeoutError('Ensemble controller wait expired')
            try: process.wait(timeout=30)
            except psutil.TimeoutExpired: continue
            break
    assert (root/'summary.json').exists()
    assert all(read(path)['status']=='completed' for path in root.glob('*-resources.json'))
    for shard in range(3):
        subprocess.run([sys.executable,str(tools/'replay_seed_ensemble.py'),'--root',str(root),'--shard',str(shard)],check=True)
    p=read(root/'protocol.json'); cfg=p['config']['resources']; audit=root.parent/(root.name+'-audit')
    audit.mkdir(exist_ok=False)
    cap=min(cfg['rss_cap_gib']*1024**3,int(cfg['available_ram_fraction']*psutil.virtual_memory().available))
    result=run_bounded([sys.executable,str(tools/'verify_seed_ensemble.py'),'--root',str(root),'--export','docs/research/seed-ensemble'],
                       audit/'worker.log',max_rss_bytes=cap,max_seconds=cfg['phase_seconds'])
    with (audit/'resources.json').open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(result,stream,indent=2,sort_keys=True); stream.write('\n')
    print(result,flush=True)
    if result['status']!='completed': raise RuntimeError('Preserve failed ensemble audit')
    subprocess.run([sys.executable,str(tools/'report_seed_ensemble.py'),'--root',str(root),
                    '--output','docs/research/2026-10-21-seed-ensemble-results.md'],check=True)
    print('ENSEMBLE PROOF AND REPORT COMPLETE',flush=True)


if __name__=='__main__':
    if os.name=='nt':
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.SetThreadExecutionState.argtypes=[ctypes.c_uint]; kernel.SetThreadExecutionState.restype=ctypes.c_uint
        if not kernel.SetThreadExecutionState(0x80000001): raise ctypes.WinError(ctypes.get_last_error())
        try: main()
        finally: kernel.SetThreadExecutionState(0x80000000)
    else: main()
