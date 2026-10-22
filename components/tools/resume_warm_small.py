"""Recorded infrastructure recovery; frozen numerical workers stay unchanged."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def read(path): return json.loads(path.read_text(encoding='utf-8'))
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value):
    with path.open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,indent=2,sort_keys=True); stream.write('\n')


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--phase',choices=['prepare','run'],required=True)
    parser.add_argument('--original',type=Path,default=Path('components/outputs/recall-warm-small'))
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--registration',type=Path,default=Path('docs/research/2026-10-21-warm-small-recovery-registration.json'))
    args=parser.parse_args(); original=args.original.resolve(); root=args.root.resolve(); repo=Path(__file__).resolve().parents[2]
    p=read(original/'protocol.json')
    for path,value in p['source_sha256'].items(): assert sha(Path(path))==value,'Frozen numerical source changed'
    if args.phase=='prepare':
        root.mkdir(exist_ok=False); copies={}; completed=[]; failures={}
        def copy(path):
            destination=root/path.relative_to(original); destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(path,destination); assert sha(path)==sha(destination)
            copies[str(destination.relative_to(root)).replace('\\','/')]={'source':str(path),'sha256':sha(path),'bytes':path.stat().st_size}
        for name in ('protocol.json','preflight.json'): copy(original/name)
        for resource_path in sorted(original.glob('*-resources.json')):
            record=read(resource_path); phase=resource_path.name.removesuffix('-resources.json')
            if record['status']!='completed':
                paths=[resource_path,original/(phase+'-worker.log')]
                paths.extend(path for path in (original/phase).rglob('*') if path.is_file())
                failures[phase]={'resource':record,'preserved_files':{str(path):{'sha256':sha(path),'bytes':path.stat().st_size} for path in paths}}
                continue
            folder=original/('tile-'+phase if phase.startswith('cache-') else phase)
            assert folder.exists()
            for path in sorted(folder.rglob('*')):
                if path.is_file() and path.suffix in ('.json','.pt','.log'): copy(path)
            copy(resource_path); copy(original/(phase+'-worker.log')); completed.append(phase)
        assert len(completed)==14 and set(failures)=={'dev-f2-s4217-longer108'}
        failed=failures['dev-f2-s4217-longer108']; assert failed['resource']['status']=='time_limit'
        assert not (original/'dev-f2-s4217-longer108/model.pt').exists()
        audit=original.parent/'recall-warm-small-recovery-audit'
        diagnostic={str(path):{'sha256':sha(path),'bytes':path.stat().st_size} for path in sorted(audit.rglob('*')) if path.is_file()}
        assert (audit/'power-events.json').exists() and (audit/'diagnosis.json').exists()
        value={'created_utc':datetime.now(timezone.utc).isoformat(),'protocol_identity':p['identity'],
               'registration_path':str(args.registration.resolve()),
               'original_root':str(original),'resumed_root':str(root),'copied_files':copies,
               'completed_phases_reused_without_refitting':completed,'preserved_failed_attempts':failures,
               'diagnostic_files':diagnostic,'source_sha256':{str(path.resolve()):sha(path) for path in
                    [Path(__file__),Path(__file__).parent/'diagnose_warm_small_interruption.py']},
               'numeric_changes':False,'seed_changes':False,'candidate_or_threshold_changes':False,
               'official_test_access':False,'before_any_resumed_fit':True,
               'recovery':'Resume the same preregistered protocol after a documented lid-triggered Modern Standby interruption. Reuse exact completed artifacts and cache indexes; keep original arrays at their hashed paths. Retry the incomplete fit once with its original seed and baseline. Any further failure stops the runner.',
               'awake_request':'Process-scoped Windows SYSTEM_REQUIRED while the controller runs, restored on exit. No global power plan or lid policy changes.'}
        write(root/'recovery.json',value); shutil.copyfile(root/'recovery.json',args.registration)
        print('PREPARED',root,'reused',len(completed),'phases; publish registration before run',flush=True); return
    recovery=read(root/'recovery.json'); assert recovery['before_any_resumed_fit'] and not recovery['numeric_changes']
    assert sha(root/'recovery.json')==sha(args.registration)
    relative=str(args.registration.resolve().relative_to(repo)).replace('\\','/')
    published=subprocess.check_output(['git','show','HEAD:'+relative],cwd=repo)
    assert hashlib.sha256(published).hexdigest()==sha(args.registration)
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/research/recall-improvement'],cwd=repo,text=True).split()[0]
    assert remote==head,'Publish registration before resumed fitting'
    for relative,value in recovery['copied_files'].items(): assert sha(root/relative)==sha(Path(value['source']))==value['sha256']
    for failure in recovery['preserved_failed_attempts'].values():
        for path,value in failure['preserved_files'].items(): assert sha(Path(path))==value['sha256']
    for path,value in recovery['diagnostic_files'].items(): assert sha(Path(path))==value['sha256']
    for path,value in recovery['source_sha256'].items(): assert sha(Path(path))==value
    if os.name!='nt': raise RuntimeError('This recorded recovery uses the Windows process awake request')
    import ctypes
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.SetThreadExecutionState.argtypes=[ctypes.c_uint]; kernel.SetThreadExecutionState.restype=ctypes.c_uint
    previous=kernel.SetThreadExecutionState(0x80000001)
    if not previous: raise ctypes.WinError(ctypes.get_last_error())
    write(root/'recovery-start.json',{'created_utc':datetime.now(timezone.utc).isoformat(),'publication_commit':head,
          'process_awake_request_succeeded':True,'global_power_settings_changed':False})
    result=None
    try:
        print('Resuming frozen serial suite; process-scoped awake request active.',flush=True)
        with (root/'recovery-controller.log').open('x',encoding='utf-8') as stream:
            result=subprocess.run([sys.executable,str(Path(__file__).parent/'run_warm_small.py'),'--root',str(root)],stdout=stream,stderr=subprocess.STDOUT)
    finally:
        restored=bool(kernel.SetThreadExecutionState(0x80000000))
        write(root/'recovery-end.json',{'created_utc':datetime.now(timezone.utc).isoformat(),'awake_request_cleared':restored,
              'returncode':None if result is None else result.returncode})
    if result.returncode: raise RuntimeError('Preserve failed recovery; no automatic retries')
    print('RESUMED SWEEP COMPLETE',flush=True)


if __name__=='__main__': main()
