"""Separately registered, equal-weight three-seed score ensemble."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from importlib.metadata import version
from pathlib import Path
import shutil
import subprocess

from component_inspector.evaluation.calibration import calibrate
from component_inspector.evaluation.metrics import evaluate

REPO=Path(__file__).resolve().parents[4]
CONFIG=REPO/'components/configs/recall_seed_ensemble.json'


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(value): return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def write(path,value):
    with Path(path).open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False); stream.write('\n')


def sigmoid(value):
    if not math.isfinite(value): raise ValueError('Nonfinite member logit')
    if value>=0: return 1./(1.+math.exp(-value))
    exp=math.exp(value); return exp/(1.+exp)


def aggregate(records):
    if len(records)!=3: raise ValueError('Exactly three fixed members required')
    predictions=[{row['id']:row for row in r['predictions']} for r in records]
    calibration=[{row['id']:row for row in r['calibration_scores']} for r in records]
    for record,pred,cal in zip(records,predictions,calibration):
        if len(pred)!=len(record['predictions']) or len(cal)!=len(record['calibration_scores']): raise ValueError('Duplicate member original')
    if any(set(p)!=set(predictions[0]) for p in predictions) or any(set(c)!=set(calibration[0]) for c in calibration):
        raise ValueError('Member original sets differ')
    if set(predictions[0]) & set(calibration[0]): raise ValueError('Calibration/evaluation overlap')
    rows=[]
    for identity in sorted(predictions[0]):
        source=[pred[identity] for pred in predictions]
        if len({r['label'] for r in source})!=1 or source[0]['label'] not in (0,1) or any(r['status']!='scored' for r in source): raise ValueError('Member label or coverage mismatch')
        rows.append({'id':identity,'label':source[0]['label'],
                     'score':math.fsum(sigmoid(r['score']) for r in source)/3.})
    cal=[{'id':identity,'score':math.fsum(sigmoid(member[identity]['score']) for member in calibration)/3.}
         for identity in sorted(calibration[0])]
    return rows,cal


def context(root):
    p=read(root/'protocol.json'); other=dict(p); expected=other.pop('identity'); assert digest(other)==expected
    for filename,value in p['source_sha256'].items(): assert sha(filename)==value,'Frozen ensemble source changed: '+filename
    assert {name:version(name) for name in p['versions']}==p['versions']
    assert sha(REPO/'components/requirements-recall.lock')==p['lock_sha256']
    main=Path(p['main_root']); assert sha(main/'protocol.json')==p['main_protocol_sha256']
    return p,main


def freeze(root,main):
    parent=read(main/'protocol.json'); cfg=read(CONFIG)
    assert not (main/'selection.json').exists() and not any(main.glob('confirm-*/alpha-*.json'))
    assert cfg['development_members']==parent['config']['development_seeds']
    assert cfg['confirmation_members']==[4220,4221,4222] and cfg['primary_alpha']==.05
    root.mkdir(exist_ok=False)
    sources={**parent['source_sha256'],**{str(path.resolve()):sha(path) for path in
             [Path(__file__),CONFIG,REPO/'components/tools/run_seed_ensemble.py']}}
    created=datetime.now(timezone.utc).isoformat(); members={}
    commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    for seed in cfg['confirmation_members'][1:]:
        directory=root/f'member-s{seed}'; directory.mkdir()
        value=json.loads(json.dumps(parent)); value.pop('identity'); value['created_utc']=created
        value['source_commit']=commit
        value['config']['confirmation_seed']=seed; value['source_sha256']=sources
        value['extension_parent_protocol']=parent['identity']
        value['claim']='Predeclared fresh member of a three-seed ensemble on observed official-training images; not factory confirmation'
        value['identity']=digest(value); write(directory/'protocol.json',value)
        copies={}
        for fold in range(3):
            cache=directory/f'tile-cache-{fold}'; cache.mkdir()
            for name in ('index.json','complete.json'):
                original=main/f'tile-cache-{fold}'/name; shutil.copyfile(original,cache/name)
                assert sha(original)==sha(cache/name); copies[str((cache/name).relative_to(root))]=sha(original)
        members[str(seed)]={'root':str(directory),'protocol_sha256':sha(directory/'protocol.json'),
                            'protocol_identity':value['identity'],'copied_cache_index_sha256':copies}
    value={'created_utc':created,'main_root':str(main),'main_protocol_identity':parent['identity'],
           'main_protocol_sha256':sha(main/'protocol.json'),'config':cfg,'source_sha256':sources,
           'source_commit':commit,'versions':parent['versions'],'lock_sha256':parent['lock_sha256'],
           'members':members,'registered_before_main_selection_and_confirmation':True,
           'no_ensemble_scores_computed':True,'no_additional_member_fit_started':True,
           'new_independent_factory_data':False,'official_test_access':False}
    value['identity']=digest(value); write(root/'protocol.json',value)
    print('ENSEMBLE FROZEN',value['identity'],flush=True)


def prepare(root):
    p,main=context(root)
    verified=read(main/'verification.json'); summary=read(main/'summary.json'); selected=read(main/'selection.json')
    assert verified['protocol_identity']==summary['protocol_identity']==p['main_protocol_identity']
    assert verified['selection_independently_verified']
    # Selection is inherited unchanged; second-partition metrics never choose
    # the family, members, aggregation, policy, or whether this extension runs.
    value={'protocol_identity':p['identity'],'method':selected['method'],
           'main_selection_identity':selected['identity'],'main_selection_sha256':sha(main/'selection.json'),
           'main_verification_sha256':sha(main/'verification.json'),
           'primary_alpha':p['config']['primary_alpha'],'no_confirmation_outcome_selection':True}
    value['identity']=digest(value); write(root/'selection.json',value)
    for seed,metadata in p['members'].items():
        directory=Path(metadata['root']); assert sha(directory/'protocol.json')==metadata['protocol_sha256']
        member=read(directory/'protocol.json')
        selection=dict(selected); selection.pop('identity'); selection['protocol_identity']=member['identity']
        selection['ensemble_extension_identity']=p['identity']; selection['parent_selection_identity']=selected['identity']
        selection['identity']=digest(selection); write(directory/'selection.json',selection)
    print('ENSEMBLE FAMILY LOCKED',selected['method'],'primary alpha',value['primary_alpha'],flush=True)


def sources_for(root,stage,fold):
    p,main=context(root); selected=read(root/'selection.json'); method=selected['method']
    assert sha(main/'selection.json')==selected['main_selection_sha256']
    if stage=='dev':
        return [main/f'dev-f{fold}-s{seed}-{method}' for seed in p['config']['development_members']]
    if stage!='confirm': raise ValueError('Unknown stage')
    return [main/f'confirm-f{fold}-s4220-{method}',
            *[Path(p['members'][str(seed)]['root'])/f'confirm-f{fold}-s{seed}-{method}'
              for seed in p['config']['confirmation_members'][1:]]]


def compare(root):
    p,main=context(root); parent=read(main/'protocol.json'); results={}
    for stage in ('dev','confirm'):
        for fold in range(3):
            directories=sources_for(root,stage,fold); records=[]; members=[]
            for directory in directories:
                assert sha(directory/'model.pt')==read(directory/'training.json')['model_sha256']==read(directory/'complete.json')['model_sha256']
                source=directory/'alpha-0.05.json'; records.append(read(source))
                members.append({'model_path':str(directory/'model.pt'),'model_sha256':sha(directory/'model.pt'),
                                'record_path':str(source),'record_sha256':sha(source)})
            rows,cal=aggregate(records)
            assert len(rows)==638 and len(cal)==200 and sum(r['label'] for r in rows)==82
            for alpha in (p['config']['primary_alpha'],p['config']['diagnostic_alpha']):
                threshold=calibrate([r['score'] for r in cal],alpha=alpha)
                predictions=[{**r,'flagged':r['score']>threshold['threshold'],'status':'scored'} for r in rows]
                small=[r for r in predictions if r['label'] and parent['areas'][r['id']]<=.005]
                metrics=evaluate(predictions,{r['id']:r['label'] for r in rows})
                record={'protocol_identity':p['identity'],'stage':stage,'fold':fold,'members':members,
                        'threshold_record':threshold,'calibration_scores':cal,'predictions':predictions,'metrics':metrics,
                        'small_defects':{'detected':sum(r['flagged'] for r in small),'total':len(small)},
                        'primary':alpha==p['config']['primary_alpha']}
                filename=f'{stage}-f{fold}-alpha-{alpha:.2f}.json'; write(root/filename,record)
                results[filename]={'counts':metrics['counts'],'small_defects':record['small_defects'],
                                  'recall':metrics['recall']['rate'],'fpr':metrics['accepted_normal_fpr']['rate'],
                                  'point_goal_met':metrics['recall']['rate']>=.95 and metrics['accepted_normal_fpr']['rate']<.1}
                print('ENSEMBLE RESULT',filename,results[filename],flush=True)
    primary=[row for name,row in results.items() if name.endswith(f'alpha-{p["config"]["primary_alpha"]:.2f}.json')]
    write(root/'summary.json',{'protocol_identity':p['identity'],'method':read(root/'selection.json')['method'],
          'primary_alpha':p['config']['primary_alpha'],'all_six_primary_ensemble_goals_met':all(r['point_goal_met'] for r in primary),
          'checks':results,'new_independent_factory_data':False,'official_test_access':False,
          'scope':p['config']['scope'],'does_not_change_main_twelve_single_model_outcomes':True})


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--phase',choices=['freeze','prepare','compare'],required=True)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--main-root',type=Path,default=Path('components/outputs/recall-warm-small-resumed-20261021'))
    args=parser.parse_args(); root=args.root.resolve()
    if args.phase=='freeze': freeze(root,args.main_root.resolve())
    elif args.phase=='prepare': prepare(root)
    else: compare(root)


if __name__=='__main__': main()
