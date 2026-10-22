"""Preserve audited checkpoints, backbone, sources and evidence in a local ZIP."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--export', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    workspace = Path(__file__).resolve().parents[2]
    root, evidence = args.root.resolve(), args.export.resolve()
    protocol, verification = read(root/'protocol.json'), read(root/'verification.json')
    assert verification['protocol_identity'] == protocol['identity']
    assert verification['original_rgb_images_replayed'] == 1914
    assert not verification['official_test_scored']
    entries = {}

    def add(path, archive_name, expected=None):
        archive_name = archive_name.replace('\\', '/')
        data = path.read_bytes()
        if expected is not None: assert sha(data) == expected, path
        assert archive_name not in entries, archive_name
        entries[archive_name] = (path, sha(data), len(data))

    index = read(evidence/'export-index.json')
    for name, record in index['files'].items():
        add(evidence/name, 'evidence/'+name, record['sha256'])
    add(evidence/'export-index.json', 'evidence/export-index.json')
    for relative, record in read(root/'timing-and-storage.json')['model_artifacts'].items():
        add(root/relative, 'checkpoints/'+relative, record['sha256'])
    add(Path(protocol['weights']), 'backbone/model.safetensors', protocol['weights_sha256'])
    for path in sorted((workspace/'components/src').rglob('*.py')):
        add(path, 'source/'+path.relative_to(workspace).as_posix())
    sources = list((workspace/'components/tools').glob('*small_defects*.py'))
    sources += list((workspace/'components/configs').glob('*.json'))
    sources += [workspace/'components/requirements-recall.lock', workspace/'components/pyproject.toml']
    sources += [workspace/'docs/research'/name for name in (
        '2026-10-20-small-defects-protocol.md',
        '2026-10-20-small-defects-preregistration.json',
        '2026-10-20-small-defects-legacy-registration.json',
        '2026-10-20-small-defects-results.md',
    )]
    package_attempt = workspace/'docs/research/2026-10-20-small-defects-package-attempt-1.json'
    if package_attempt.exists(): sources.append(package_attempt)
    for path in sources: add(path, 'source/'+path.relative_to(workspace).as_posix())
    # The packaged frozen sources must match their registered logical bytes.
    for file, expected in protocol['source_sha256'].items():
        assert sha(Path(file).read_bytes()) == expected, file
    for registration, worker in (
        ('paired-control-plan.json', 'run_control_small_defects.py'),
        ('refit-repeat-plan.json', 'run_repeat_small_defects.py'),
        ('legacy-reference-plan.json', 'run_legacy_small_defects.py'),
    ):
        assert sha((workspace/'components/tools'/worker).read_bytes()) == read(root/registration)['source_sha256']
    assert sha((workspace/'components/tools/verify_small_defects.py').read_bytes()) == verification['verifier_sha256']
    assert sha((workspace/'components/tools/replay_native_small_defects.py').read_bytes()) == read(root/'full-native-replay.json')['source_sha256']
    manifest = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'protocol_identity': protocol['identity'],
        'scope': 'local checkpoint/evidence/source backup, not a new evaluation or standalone installer; original licensed KSDD2 files and path remapping remain necessary for reproduction',
        'excluded': 'dataset originals, image/target/feature arrays, virtual environment',
        'files': {name: {'sha256': digest, 'bytes': size}
                  for name, (_, digest, size) in sorted(entries.items())},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=3) as archive:
        for name, (path, digest, _) in sorted(entries.items()):
            data = path.read_bytes()
            assert sha(data) == digest, 'Artifact changed during packaging'
            archive.writestr(name, data)
        archive.writestr('package-index.json', json.dumps(manifest, indent=2, sort_keys=True)+'\n')
    with zipfile.ZipFile(args.output) as archive:
        assert archive.testzip() is None
        assert set(archive.namelist()) == set(entries) | {'package-index.json'}
        for name, (_, digest, _) in entries.items(): assert sha(archive.read(name)) == digest
    receipt = {'archive_path': str(args.output.resolve()), 'archive_sha256': sha(args.output.read_bytes()),
               'archive_bytes': args.output.stat().st_size, 'file_count': len(entries),
               'protocol_identity': protocol['identity'], 'all_member_hashes_verified_after_write': True,
               'scope': manifest['scope'], 'excluded': manifest['excluded']}
    with args.receipt.open('x', encoding='utf-8', newline='\n') as file:
        json.dump(receipt, file, indent=2, sort_keys=True); file.write('\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__': main()
