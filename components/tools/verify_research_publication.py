"""Compare forensic evidence and frozen source hashes with staged Git blobs."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess


def sha(data): return hashlib.sha256(data).hexdigest()
def read(path): return json.loads(path.read_text(encoding='utf-8-sig'))


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--evidence',type=Path,action='append',default=[])
    parser.add_argument('--protocol',type=Path,action='append',default=[])
    parser.add_argument('--file',type=Path,action='append',default=[])
    parser.add_argument('--receipt',type=Path,action='append',default=[])
    parser.add_argument('--output',type=Path); args=parser.parse_args()
    repo=Path(__file__).resolve().parents[2]; targets={}; receipts=[]
    def add(path,expected=None):
        path=path.resolve(); relative=str(path.relative_to(repo)).replace('\\','/')
        actual=sha(path.read_bytes()); assert expected is None or expected==actual,'Local hash mismatch: '+relative
        assert relative not in targets or targets[relative]==actual
        targets[relative]=actual
    for folder in args.evidence:
        index=read(folder/'export-index.json'); add(folder/'export-index.json')
        for name,value in index['files'].items():
            path=folder/name; assert path.stat().st_size==value['bytes']; add(path,value['sha256'])
    for path in args.protocol:
        protocol=read(path); original=dict(protocol); expected=original.pop('identity')
        assert sha(json.dumps(original,sort_keys=True,separators=(',',':')).encode())==expected
        for filename,value in protocol['source_sha256'].items(): add(Path(filename),value)
    for path in args.file: add(path)
    for path in args.receipt:
        receipt=read(path); archive=Path(receipt['path'])
        assert archive.stat().st_size==receipt['bytes'] and sha(archive.read_bytes())==receipt['sha256']
        assert receipt['every_payload_hash_verified'] and receipt['crc_verified']; add(path)
        receipts.append({'path':str(archive),'sha256':receipt['sha256']})
    staged={}
    for row in subprocess.check_output(['git','ls-files','--stage','-z'],cwd=repo).split(b'\0'):
        if not row: continue
        metadata,filename=row.split(b'\t',1); mode,oid,stage=metadata.split(); name=filename.decode('utf-8')
        if name in targets:
            assert stage==b'0','Unmerged staged file: '+name
            staged[name]=oid.decode('ascii')
    missing=set(targets)-set(staged); assert not missing,'Untracked/unstaged evidence: '+str(sorted(missing)[:12])
    oids=list(dict.fromkeys(staged.values())); content=subprocess.check_output(['git','cat-file','--batch'],
                    input=('\n'.join(oids)+'\n').encode('ascii'),cwd=repo)
    stream=io.BytesIO(content); blobs={}
    for oid in oids:
        actual,kind,size=stream.readline().strip().split(); assert actual.decode()==oid and kind==b'blob'
        data=stream.read(int(size)); assert len(data)==int(size) and stream.read(1)==b'\n'
        blobs[oid]=sha(data)
    assert not stream.read()
    legacy={f'components/src/component_inspector/{name}.py' for name in
            ('data/acquire','evaluation/calibration','evaluation/metrics','vision/geometry','vision/padim_adapter','vision/spike_worker','vision/resources')}
    exact=0; normalized=[]
    for name,expected in targets.items():
        actual=blobs[staged[name]]
        if actual==expected: exact+=1
        else:
            assert name in legacy and sha((repo/name).read_bytes().replace(b'\r\n',b'\n'))==actual,'Staged bytes differ: '+name
            normalized.append(name)
    result={'files_checked':len(targets),'exact_byte_matches':exact,'frozen_crlf_sources_matching_canonical_git_lf':sorted(normalized),
            'checkpoint_receipts_verified':receipts,'verifier_sha256':sha(Path(__file__).read_bytes()),
            'scope':'Read-only comparison of local forensic hashes and staged Git blobs; not a new model evaluation or third-party certification.'}
    if args.output:
        with args.output.open('x',encoding='utf-8',newline='\n') as out: json.dump(result,out,indent=2,sort_keys=True); out.write('\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
