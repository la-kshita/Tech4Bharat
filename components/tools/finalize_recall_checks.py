"""Serial post-proof fitting diagnosis and final software checks; no new training."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import psutil

from component_inspector.vision.resources import run_bounded


def read(path): return json.loads(path.read_text(encoding='utf-8'))
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value):
    with path.open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,indent=2,sort_keys=True); stream.write('\n')


def export(source,destination):
    destination.mkdir(exist_ok=False); files={}
    for path in sorted(source.iterdir()):
        if path.is_file():
            shutil.copyfile(path,destination/path.name); assert sha(path)==sha(destination/path.name)
            files[path.name]={'sha256':sha(path),'bytes':path.stat().st_size}
    write(destination/'export-index.json',{'files':files,'scope':'Diagnostic/software evidence; not additional held-out model qualification.'})


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--wait-proof-pid',type=int); args=parser.parse_args()
    source_sha=sha(Path(__file__)); root=args.root.resolve(); tools=Path(__file__).resolve().parent
    if args.wait_proof_pid is not None:
        process=psutil.Process(args.wait_proof_pid)
        assert any(Path(value).name=='finish_seed_ensemble.py' for value in process.cmdline())
        deadline=time.monotonic()+43200; print('WAIT for both studies and their proof; no concurrent diagnostic or tests',flush=True)
        while process.is_running():
            if time.monotonic()>deadline: raise TimeoutError('Proof wait expired')
            try: process.wait(timeout=30)
            except psutil.TimeoutExpired: continue
            break
    assert (root/'verification.json').exists() and Path('docs/research/2026-10-21-seed-ensemble-results.md').exists()
    fitting=root.parent/'recall-fitting-gap'
    subprocess.run([sys.executable,str(tools/'audit_fitting_gap.py'),'--root',str(root),'--output',str(fitting)],check=True)
    diagnostic=read(fitting/'diagnostic.json'); assert read(fitting/'resources.json')['status']=='completed'
    assert diagnostic['protocol_identity']==read(root/'protocol.json')['identity']
    assert diagnostic['source_sha256']==read(Path('docs/research/2026-10-21-fitting-gap-preregistration.json'))['source_sha256']
    export(fitting,Path('docs/research/fitting-gap'))
    lines=['# Fitting-versus-held-out diagnostic','',
           '**Fitting performance is not validation. None of these fitting scores changes the selected model, threshold or qualification.**','',
           'Each member uses its original normal-only calibration cutoff. Fitting sets '
           'contain 164 defects (30 small) and 256 normals; held-out folds contain 82 '
           'defects (15 small) and 556 normals. Originals repeat across members and folds; '
           'these denominators are never added as independent samples.', '',
           'A fitting/held-out gap describes generalization behavior without uniquely '
           'distinguishing overfitting, inadequate features, label noise or distribution '
           'differences. A small gap would not prove freedom from overfitting or independent '
           'factory generalization. No further model fitting or threshold tuning follows '
           'from this diagnostic.', '',
           '| Alpha | Fitting recall range across 18 heads | Held-out recall range |',
           '|---:|---:|---:|']
    for alpha in (.05,.08):
        rows=[r for r in diagnostic['members'] if r['alpha']==alpha]
        lines.append(f'| {alpha:.2f} | {100*min(r["fitting_recall"] for r in rows):.2f}% to {100*max(r["fitting_recall"] for r in rows):.2f}% | {100*min(r["held_out_recall"] for r in rows):.2f}% to {100*max(r["held_out_recall"] for r in rows):.2f}% |')
    lines += ['', '| Stage | Seed | Fold | Alpha | Fitting TP / 164 | Held-out TP / 82 | Fitting FP / 256 | Held-out FP / 556 | Fitting small / 30 | Held-out small / 15 |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in diagnostic['members']:
        lines.append(f'| {row["stage"]} | {row["seed"]} | {row["fold"]} | {row["alpha"]:.2f} | {row["fitting_counts"]["TP"]}/164 | {row["held_out_counts"]["TP"]}/82 | {row["fitting_counts"]["FP"]}/256 | {row["held_out_counts"]["FP"]}/556 | {row["fitting_small"]["detected"]}/30 | {row["held_out_small"]["detected"]}/15 |')
    lines += ['', '## Ensemble common fitting positives','',
              'Only common fitting positives are summarized. Members have different mined '
              'normal sets, so no ensemble fitting-normal FPR is claimed. These are the '
              'fixed equal-weight scores and existing joint thresholds, not best-member '
              'choices or a replacement for held-out outcomes.', '',
              '| Stage | Fold | Alpha | Fitting TP / 164 | Held-out TP / 82 | Fitting small / 30 | Held-out small / 15 |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for row in diagnostic['ensembles']:
        held=read(root/f'{row["stage"]}-f{row["fold"]}-alpha-{row["alpha"]:.2f}.json')
        lines.append(f'| {row["stage"]} | {row["fold"]} | {row["alpha"]:.2f} | {row["fitting_positive_detected"]}/164 | {held["metrics"]["counts"]["TP"]}/82 | {row["fitting_small"]["detected"]}/30 | {held["small_defects"]["detected"]}/15 |')
    lines += ['', f'The registered forward scorer and independent convolution/pooling agreed '
              f'within 1e-5 on all {diagnostic["fitting_head_scores_checked"]} fitting scores, '
              f'with identical flags at all {diagnostic["individual_fitting_threshold_decisions_reproduced"]} '
              'individual cutoff decisions. Checkpoint hashes were unchanged. No official '
              'test image was accessed; all existing previously-observed-data limitations '
              'still apply.', '',
              'Evidence: [raw predictions](fitting-gap/diagnostic.json), '
              '[export hashes](fitting-gap/export-index.json), '
              '[prospective diagnostic](2026-10-21-fitting-gap-protocol.md), '
              '[source hash](2026-10-21-fitting-gap-preregistration.json).', '']
    with Path('docs/research/2026-10-21-fitting-gap-results.md').open('x',encoding='utf-8',newline='\n') as stream:
        stream.write('\n'.join(lines))
    software=root.parent/'recall-final-software'; software.mkdir(exist_ok=False)
    cfg=read(root/'protocol.json')['config']['resources']
    cap=min(cfg['rss_cap_gib']*1024**3,int(cfg['available_ram_fraction']*psutil.virtual_memory().available))
    command=[sys.executable,'-m','pytest','components/tests','-q','--junitxml',str(software/'junit.xml')]
    result=run_bounded(command,software/'worker.log',max_rss_bytes=cap,max_seconds=cfg['phase_seconds'])
    write(software/'resources.json',result); print(result,flush=True)
    if result['status']!='completed': raise RuntimeError('Preserve failed final software checks')
    suites=ET.parse(software/'junit.xml').getroot(); rows=list(suites) if suites.tag=='testsuites' else [suites]
    total={name:sum(int(row.attrib.get(name,0)) for row in rows) for name in ('tests','failures','errors','skipped')}
    assert total['failures']==total['errors']==0
    assert sha(Path(__file__))==source_sha
    write(software/'summary.json',{'command':command,'counts':total,'passed':total['tests']-total['skipped'],
          'source_sha256':source_sha,'scope':'Software checks only; no new KSDD2 test evaluation or model selection.'})
    export(software,Path('docs/research/software-checks'))
    print('POST-PROOF FITTING DIAGNOSTIC AND SOFTWARE CHECKS COMPLETE',total,flush=True)


if __name__=='__main__': main()
