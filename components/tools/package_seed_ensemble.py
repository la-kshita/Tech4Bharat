"""Verified supplementary checkpoint package including the parent archive."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile

from component_inspector.research.ksdd2_seed_ensemble import read,sha


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--receipt',type=Path,required=True)
    parser.add_argument('--main-receipt',type=Path,default=Path('docs/research/2026-10-21-warm-small-checkpoint-package.json'))
    args=parser.parse_args(); repo=Path(__file__).resolve().parents[2]; root=args.root.resolve()
    p=read(root/'protocol.json'); v=read(root/'verification.json'); summary=read(root/'audit-summary.json')
    assert p['identity']==v['protocol_identity']==summary['protocol_identity']
    assert sha(Path(__file__).parent/'verify_seed_ensemble.py')==v['verifier_sha256']
    parent=read(args.main_receipt); archive=Path(parent['path']); assert sha(archive)==parent['sha256']
    assert parent['every_payload_hash_verified'] and parent['crc_verified']
    audit=root.parent/(root.name+'-audit'); assert read(audit/'resources.json')['status']=='completed'
    paths={path.resolve() for path in root.rglob('*') if path.is_file() and path.suffix in ('.pt','.json','.log')}
    paths.update(Path(path).resolve() for path in p['source_sha256'])
    paths.update(path.resolve() for path in Path(__file__).parent.glob('*seed_ensemble.py'))
    paths.add(Path(__file__).parent/'verify_research_publication.py')
    paths.add(Path(__file__).parent/'audit_fitting_gap.py')
    paths.add(Path(__file__).parent/'finalize_recall_checks.py')
    paths.update(path.resolve() for path in audit.glob('*') if path.is_file())
    paths.update([archive.resolve(),args.main_receipt.resolve(),repo/'components/tests/test_seed_ensemble.py',repo/'components/requirements-recall.lock',
                  repo/'docs/research/2026-10-20-warm-small-results.md',
                  repo/'docs/research/2026-10-22-recall-status.md'])
    for folder in ('seed-ensemble','warm-small','warm-small-recovery','fitting-gap','software-checks'):
        paths.update(path.resolve() for path in (repo/'docs/research'/folder).glob('*') if path.is_file())
    for pattern in ('2026-10-21-seed-ensemble*','2026-10-21-warm-small*','2026-10-21-publication*','2026-10-21-fitting-gap*'):
        paths.update(path.resolve() for path in (repo/'docs/research').glob(pattern) if path.is_file())
    for filename,value in v['model_artifacts'].items(): assert sha(filename)==value
    files={str(path.relative_to(repo)).replace('\\','/'):{'sha256':sha(path),'bytes':path.stat().st_size} for path in sorted(paths)}
    assert len(files)==len(paths)
    for name in files:
        pure=PurePosixPath(name); assert not pure.is_absolute() and '..' not in pure.parts and '\\' not in name
    index={'protocol_identity':p['identity'],'files':files,'payload_file_count':len(files),
           'scope':'Exact supplementary checkpoints, registered sources and evidence, plus the verified parent checkpoint archive. Original RGB and large feature arrays are excluded with provenance/hashes preserved.',
           'portability':'Preserves original absolute paths. Extract the nested parent archive for its backbone/checkpoints. Forensic preservation, not a standalone installer or independently validated production model.'}
    target=root/'checkpoint-evidence-verified.zip'
    with zipfile.ZipFile(target,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as package:
        for path in sorted(paths): package.write(path,str(path.relative_to(repo)).replace('\\','/'))
        package.writestr('package-index.json',json.dumps(index,indent=2,sort_keys=True)+'\n')
    with zipfile.ZipFile(target) as package:
        assert package.testzip() is None
        members={entry.filename.replace('\\','/'):entry for entry in package.infolist()}
        assert len(members)==len(package.infolist())==len(files)+1 and set(members)==set(files)|{'package-index.json'}
        assert json.loads(package.read(members['package-index.json']))==index
        for name,value in files.items():
            content=package.read(members[name]); assert len(content)==value['bytes'] and hashlib.sha256(content).hexdigest()==value['sha256']
            assert sha(repo/name)==value['sha256']
    receipt={'protocol_identity':p['identity'],'path':str(target),'sha256':sha(target),'bytes':target.stat().st_size,
             'payload_file_count':len(files),'total_zip_members':len(files)+1,'every_payload_hash_verified':True,
             'crc_verified':True,'safe_unique_member_names_verified':True,
             'all_six_primary_ensemble_goals_met':summary['all_six_primary_ensemble_goals_met'],
             'independent_factory_confirmation':False,'source_sha256':sha(Path(__file__)),
             'parent_archive_sha256':parent['sha256']}
    with args.receipt.open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(receipt,stream,indent=2,sort_keys=True); stream.write('\n')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__': main()
