"""One bounded worker at a time; never overwrite failed warm-study phases."""
import argparse
from pathlib import Path
import sys

import psutil

from component_inspector.research.ksdd2_round1 import read_json, write_json
from component_inspector.vision.resources import run_bounded


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args(); root=args.root.resolve(); p=read_json(root/'protocol.json')
    command=[sys.executable,'-m','component_inspector.research.ksdd2_warm_small','--root',str(root)]

    def run(name,extra):
        path=root/f'{name}-resources.json'
        if path.exists():
            record=read_json(path)
            if record['status']!='completed': raise RuntimeError('Preserved failure: '+name)
            print('USE COMPLETED',name,flush=True); return
        cfg=p['config']['resources']; cap=min(cfg['rss_cap_gib']*1024**3,int(cfg['available_ram_fraction']*psutil.virtual_memory().available))
        print('START',name,'RSS cap',cap,flush=True)
        record=run_bounded([*command,*extra],root/f'{name}-worker.log',max_rss_bytes=cap,max_seconds=cfg['phase_seconds'])
        write_json(path,record); print('END',name,record,flush=True)
        if record['status']!='completed': raise RuntimeError('Preserve failed phase: '+name)

    for fold in range(3): run(f'cache-{fold}',['--phase','cache','--fold',str(fold)])
    for seed in p['config']['development_seeds']:
        for fold in range(3):
            for method in p['config']['methods']:
                run(f'dev-f{fold}-s{seed}-{method}', ['--phase','train','--stage','dev','--seed',str(seed),'--fold',str(fold),'--method',method])
    run('selection',['--phase','select'])
    selected=read_json(root/'selection.json')['method']
    for fold in range(3):
        for method in dict.fromkeys(['baseline84',selected]):
            run(f'confirm-f{fold}-{method}', ['--phase','train','--stage','confirm','--seed',str(p['config']['confirmation_seed']),
                                             '--fold',str(fold),'--method',method])


if __name__=='__main__': main()
