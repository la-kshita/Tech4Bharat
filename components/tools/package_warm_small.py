"""Preserve exact checkpoint/source/evidence bytes and verify every ZIP member."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile


def read(path): return json.loads(path.read_text(encoding='utf-8'))
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path,value):
    with path.open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False); stream.write('\n')


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--receipt',type=Path,required=True); args=parser.parse_args()
    repo=Path(__file__).resolve().parents[2]; root=args.root.resolve()
    p=read(root/'protocol.json'); v=read(root/'verification.json'); s=read(root/'summary.json')
    assert p['identity']==v['protocol_identity']==s['protocol_identity']
    assert sha(Path(__file__).parent/'verify_warm_small.py')==v['verifier_sha256']
    audit=root.parent/(root.name+'-audit'); assert read(audit/'resources.json')['status']=='completed'
    paths={path.resolve() for path in root.rglob('*') if path.is_file() and path.suffix in ('.pt','.json','.log')}
    paths.update(Path(path).resolve() for path in p['source_sha256'])
    paths.update(path.resolve() for path in Path(__file__).parent.glob('*warm_small.py'))
    paths.add(repo/'components/requirements-recall.lock')
    paths.add(Path(__file__).parent/'verify_research_publication.py')
    paths.add(Path(__file__).parent/'finalize_recall_checks.py')
    paths.update(path.resolve() for path in (repo/'docs/research/software-checks').glob('*') if path.is_file())
    paths.update(path.resolve() for path in (repo/'components/tests').glob('test_warm_small*.py'))
    paths.update(path.resolve() for path in (repo/'docs/research/warm-small').glob('*') if path.is_file())
    paths.update(path.resolve() for path in (repo/'docs/research/warm-small-recovery').glob('*') if path.is_file())
    paths.update(path.resolve() for path in (repo/'docs/research').glob('2026-10-20-warm-small*') if path.is_file())
    paths.update(path.resolve() for path in (repo/'docs/research').glob('2026-10-21-warm-small*') if path.is_file())
    paths.update(path.resolve() for path in (repo/'docs/research').glob('2026-10-21-publication*') if path.is_file())
    paths.add(Path(__file__).parent/'diagnose_warm_small_interruption.py')
    paths.update(path.resolve() for path in audit.glob('*') if path.is_file())
    parent=Path(p['parent']); old=read(parent/'protocol.json'); paths.add(Path(old['weights']).resolve())
    paths.update(path.resolve() for path in parent.rglob('*') if path.is_file() and path.suffix in ('.json','.log'))
    paths.update(Path(value['path']).resolve() for value in p['baseline_reuse'].values())
    for key,value in v['model_artifacts'].items(): assert sha(root/key)==value['sha256']
    files={str(path.relative_to(repo)).replace('\\','/'):{'sha256':sha(path),'bytes':path.stat().st_size} for path in sorted(paths)}
    assert len(files)==len(paths)
    for name in files:
        pure=PurePosixPath(name); assert not pure.is_absolute() and '..' not in pure.parts and '\\' not in name
    index={'protocol_identity':p['identity'],'files':files,'payload_file_count':len(files),
           'scope':'Exact checkpoints, frozen sources, pretrained weights and completed evidence. Large feature caches and original dataset are excluded, with hashes preserved in evidence.',
           'portability':'Evidence retains original absolute paths. Reproduction on another location needs a newly registered run; this archive is forensic preservation, not a standalone installer.'}
    target=root/'checkpoint-evidence-verified.zip'
    with zipfile.ZipFile(target,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as package:
        for path in sorted(paths): package.write(path,str(path.relative_to(repo)).replace('\\','/'))
        package.writestr('package-index.json',json.dumps(index,indent=2,sort_keys=True)+'\n')
    with zipfile.ZipFile(target) as package:
        assert package.testzip() is None
        members={entry.filename.replace('\\','/'):entry for entry in package.infolist()}
        assert len(members)==len(package.infolist())==len(files)+1
        assert set(members)==set(files)|{'package-index.json'}
        assert json.loads(package.read(members['package-index.json']))==index
        for name,value in files.items():
            content=package.read(members[name]); assert len(content)==value['bytes']
            assert hashlib.sha256(content).hexdigest()==value['sha256']
            assert sha(repo/name)==value['sha256'],'Source changed during packaging: '+name
    receipt={'protocol_identity':p['identity'],'path':str(target),'sha256':sha(target),'bytes':target.stat().st_size,
             'payload_file_count':len(files),'total_zip_members':len(files)+1,
             'every_payload_hash_verified':True,'crc_verified':True,'safe_unique_member_names_verified':True,
             'all_twelve_internal_goals_met':s['all_twelve_internal_goals_met'],
             'independent_factory_confirmation':False,'source_sha256':sha(Path(__file__)),
             'excluded':'Original dataset and large feature arrays; their recorded hashes and raw data acquisition provenance are preserved.'}
    write(args.receipt,receipt); print(json.dumps(receipt,indent=2))


if __name__=='__main__': main()
