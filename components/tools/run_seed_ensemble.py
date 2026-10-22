"""Wait for the original proof, then train registered ensemble members serially."""
import argparse
import ctypes
import hashlib
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
    parser.add_argument('--wait-proof-pid',type=int)
    parser.add_argument('--registration',type=Path,default=Path('docs/research/2026-10-21-seed-ensemble-preregistration.json'))
    args=parser.parse_args(); root=args.root.resolve(); p=read(root/'protocol.json')
    snapshot=read(args.registration); assert snapshot['before_any_ensemble_score_or_added_fit']
    assert snapshot['protocol']==p
    for relative,value in snapshot['frozen_files'].items():
        assert hashlib.sha256((root/relative).read_bytes()).hexdigest()==value
    for filename,value in p['source_sha256'].items(): assert hashlib.sha256(Path(filename).read_bytes()).hexdigest()==value
    repo=Path(__file__).resolve().parents[2]
    relative=str(args.registration.resolve().relative_to(repo)).replace('\\','/')
    published=subprocess.check_output(['git','show','HEAD:'+relative],cwd=repo)
    assert hashlib.sha256(published).digest()==hashlib.sha256(args.registration.read_bytes()).digest()
    head=subprocess.check_output(['git','rev-parse','HEAD'],text=True,cwd=repo).strip()
    remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/research/recall-improvement'],text=True,cwd=repo).split()[0]
    assert head==remote,'Publish the extension registration first'
    if args.wait_proof_pid is not None:
        proof=psutil.Process(args.wait_proof_pid)
        assert any(Path(value).name=='finish_warm_small.py' for value in proof.cmdline())
        print('WAIT for original serial sweep and its proof; no additional ML worker',flush=True)
        deadline=time.monotonic()+43200
        while proof.is_running():
            if time.monotonic()>deadline: raise TimeoutError('Original proof did not finish; extension not started')
            try: proof.wait(timeout=30)
            except psutil.TimeoutExpired: continue
            break
    main_root=Path(p['main_root']); assert (main_root/'verification.json').exists()
    assert (repo/'docs/research/2026-10-20-warm-small-results.md').exists()
    cfg=p['config']['resources']
    def run(name,command):
        path=root/f'{name}-resources.json'
        if path.exists():
            assert read(path)['status']=='completed','Preserve failed extension phase: '+name
            print('USE COMPLETED',name,flush=True); return
        cap=min(cfg['rss_cap_gib']*1024**3,int(cfg['available_ram_fraction']*psutil.virtual_memory().available))
        print('START',name,flush=True)
        value=run_bounded(command,root/f'{name}-worker.log',max_rss_bytes=cap,max_seconds=cfg['phase_seconds'])
        with path.open('x',encoding='utf-8',newline='\n') as stream:
            json.dump(value,stream,indent=2,sort_keys=True); stream.write('\n')
        print('END',name,value,flush=True)
        if value['status']!='completed': raise RuntimeError('Preserve failed extension; no automatic retries')
    prefix=[sys.executable,'-m','component_inspector.research.ksdd2_seed_ensemble','--root',str(root)]
    run('prepare',[*prefix,'--phase','prepare'])
    method=read(root/'selection.json')['method']
    for seed,metadata in p['members'].items():
        for fold in range(3):
            for candidate in dict.fromkeys(['baseline84',method]):
                run(f'member-s{seed}-f{fold}-{candidate}',
                    [sys.executable,'-m','component_inspector.research.ksdd2_warm_small',
                     '--root',metadata['root'],'--phase','train','--stage','confirm',
                     '--seed',seed,'--fold',str(fold),'--method',candidate])
    run('compare',[*prefix,'--phase','compare'])
    print('ENSEMBLE NUMERICAL PHASES COMPLETE; independent proof still required',flush=True)


if __name__=='__main__':
    if os.name=='nt':
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.SetThreadExecutionState.argtypes=[ctypes.c_uint]; kernel.SetThreadExecutionState.restype=ctypes.c_uint
        if not kernel.SetThreadExecutionState(0x80000001): raise ctypes.WinError(ctypes.get_last_error())
        try: main()
        finally: kernel.SetThreadExecutionState(0x80000000)
    else: main()
