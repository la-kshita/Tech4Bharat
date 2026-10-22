"""Independent arithmetic, role, view, normalization and mining audit."""
import argparse
from collections import Counter
import hashlib
import json
import math
from importlib.metadata import version
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(value): return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def write(path,value):
    if path.exists():
        assert read(path)==value,'Preserved audit sidecar differs: '+str(path)
        return
    with path.open('x',encoding='utf-8',newline='\n') as f:
        json.dump(value,f,indent=2,sort_keys=True,allow_nan=False); f.write('\n')


def identity(value):
    other=dict(value); expected=other.pop('identity'); assert digest(other)==expected


def finite(value):
    if isinstance(value,torch.Tensor): assert torch.isfinite(value).all()
    elif isinstance(value,dict):
        for child in value.values(): finite(child)


def recount(record,ids,truth,areas,cal_ids,alpha):
    rows=record['predictions']; assert len(rows)==len(ids) and {r['id'] for r in rows}==set(ids)
    cal=record['calibration_scores']; assert len(cal)==len({r['id'] for r in cal})==200
    assert {r['id'] for r in cal}==set(cal_ids)
    assert all(truth[r['id']]==0 and np.isfinite(r['score']) for r in cal)
    k=math.ceil(201*(1-alpha)); threshold=sorted(r['score'] for r in cal)[k-1]
    metadata=record['threshold_record']
    assert metadata['threshold']==threshold and metadata['alpha']==alpha
    assert metadata['n']==200 and metadata['k']==k and metadata['comparison']=='strict_greater'
    counts=Counter(TP=0,FN=0,FP=0,TN=0,rejected_anomalies=0,rejected_normals=0)
    for row in rows:
        i=row['id']; assert row['label']==truth[i] and row['status']=='scored' and np.isfinite(row['score'])
        assert row['flagged']==(row['score']>threshold)
        counts[('TP' if row['flagged'] else 'FN') if truth[i] else ('FP' if row['flagged'] else 'TN')]+=1
    assert dict(counts)==record['metrics']['counts']
    assert counts['TP']+counts['FN']==82 and counts['FP']+counts['TN']==556
    assert record['metrics']['recall']['rate']==counts['TP']/82
    assert record['metrics']['accepted_normal_fpr']['rate']==counts['FP']/556
    small=[r for r in rows if truth[r['id']] and areas[r['id']]<=.005]
    assert record['small_defects']=={'detected':sum(r['flagged'] for r in small),'total':15}
    return {'counts':dict(counts),'recall':counts['TP']/82,'fpr':counts['FP']/556,
            'small_defects':record['small_defects'],
            'large_defects':{'detected':counts['TP']-record['small_defects']['detected'],'total':67},
            'point_goal_met':counts['TP']/82>=.95 and counts['FP']/556<.1}


def score_base(state,x,roi):
    # Reconstruct the two pointwise layers directly from saved epoch60 weights;
    # do not call the experiment's head, pooling or scoring functions.
    x=(torch.from_numpy(x)[None]-state['mean'])/state['std']; weights=state['head']
    x=F.relu(F.conv2d(x,weights['segment.0.weight'],weights['segment.0.bias']))
    raw=F.conv2d(x,weights['segment.2.weight'],weights['segment.2.bias'])[0,0]
    values=raw[torch.from_numpy(roi)]; k=math.ceil(.001*len(values))
    return float(values.topk(k).values.mean())


def score_complete(state,identity,old_index,tile_index,tiles):
    entry=old_index[identity]
    x=np.load(entry['coarse_path'],allow_pickle=False).astype(np.float32)
    roi=np.load(entry['coarse_roi_path'],allow_pickle=False)
    value=score_base(state,x,roi)
    if tiles:
        x=np.load(tile_index[identity]['feature_path'],allow_pickle=False).astype(np.float32)
        x=(torch.from_numpy(x)-state['mean'])/state['std']; weights=state['head']
        x=F.relu(F.conv2d(x,weights['segment.0.weight'],weights['segment.0.bias']))
        raw=F.conv2d(x,weights['segment.2.weight'],weights['segment.2.bias'])[:,0].flatten(1)
        crop_scores=raw.topk(math.ceil(.001*raw.shape[1]),dim=1).values.mean(1)
        value=max(value,float(crop_scores.max()))
    return value


def paired_changes(baseline,candidate,truth,areas):
    """Keep each fold/seed paired; never treat repeated originals as new samples."""
    base={row['id']:row['flagged'] for row in baseline['predictions']}
    other={row['id']:row['flagged'] for row in candidate['predictions']}
    assert base.keys()==other.keys()
    result={}
    for stratum in ('small_defects','large_defects','all_defects','normals'):
        ids=[i for i in sorted(base) if (not truth[i] if stratum=='normals' else
             bool(truth[i]) and (stratum=='all_defects' or
             (areas[i]<=.005 if stratum=='small_defects' else areas[i]>.005)))]
        added=[i for i in ids if not base[i] and other[i]]
        removed=[i for i in ids if base[i] and not other[i]]
        result[stratum]={'total':len(ids),'baseline_flagged':sum(base[i] for i in ids),
                         'candidate_flagged':sum(other[i] for i in ids),
                         'newly_flagged_ids':added,'no_longer_flagged_ids':removed,
                         'net_change':len(added)-len(removed)}
        assert result[stratum]['candidate_flagged']-result[stratum]['baseline_flagged']==result[stratum]['net_change']
    return result


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--export',type=Path,required=True)
    parser.add_argument('--registration',type=Path,default=Path('docs/research/2026-10-20-warm-small-preregistration.json'))
    args=parser.parse_args(); root=args.root.resolve(); torch.set_num_threads(4); torch.set_num_interop_threads(1)
    p=read(root/'protocol.json'); identity(p); cfg=p['config']
    recovery=None
    if (root/'recovery.json').exists():
        recovery=read(root/'recovery.json'); original=Path(recovery['original_root'])
        assert recovery['before_any_resumed_fit'] and recovery['protocol_identity']==p['identity']
        assert not recovery['numeric_changes'] and not recovery['seed_changes'] and not recovery['candidate_or_threshold_changes']
        assert sha(root/'recovery.json')==sha(recovery['registration_path'])
        assert sha(root/'protocol.json')==sha(original/'protocol.json')
        for relative,value in recovery['copied_files'].items(): assert sha(root/relative)==sha(value['source'])==value['sha256']
        for failure in recovery['preserved_failed_attempts'].values():
            for path,value in failure['preserved_files'].items(): assert sha(path)==value['sha256']
        for path,value in recovery['diagnostic_files'].items(): assert sha(path)==value['sha256']
        for path,value in recovery['source_sha256'].items(): assert sha(path)==value
        assert read(root/'recovery-end.json')['returncode']==0
    snapshot=read(args.registration); assert snapshot['before_any_new_cache_or_fit']
    for name,value in snapshot['registrations'].items(): assert sha(root/name)==value['sha256'] and read(root/name)==value['record']
    for filename,expected in p['source_sha256'].items(): assert sha(filename)==expected,filename
    assert {name:version(name) for name in p['versions']}==p['versions']
    assert sha(Path(__file__).resolve().parents[1]/'requirements-recall.lock')==p['lock_sha256']
    parent=Path(p['parent']); old=read(parent/'protocol.json'); identity(old)
    assert p['parent_identity']==old['identity'] and sha(parent/'verification.json')==p['parent_verification_sha256']
    assert sha(old['weights'])==old['weights_sha256']
    manifest=read(Path(old['parent'])/'manifest.json'); identity(manifest); by={r['id']:r for r in manifest['records']}
    pool=set(p['pool_ids']); reserved=set(p['reserved_ids']); assert len(pool)==1914 and len(reserved)==417 and pool.isdisjoint(reserved)
    assert all(by[i]['official_split']=='train' for i in pool|reserved)
    truth={}; areas={}
    for i in sorted(pool):
        row=by[i]; assert sha(row['mask_path'])==row['hashes']['mask_sha256']
        with Image.open(row['mask_path']) as im: mask=np.asarray(im.convert('L'))>0
        truth[i]=int(mask.any()); assert truth[i]==row['label']
        if truth[i]: areas[i]=float(mask.mean())
    assert sum(truth.values())==246 and areas==p['areas']
    for stage in ('development','confirmation'):
        assignment=p[stage+'_folds']; assert set(assignment)==pool
        groups={}
        for i in pool:
            group=by[i]['group']; assert group not in groups or groups[group]==assignment[i]; groups[group]=assignment[i]
        for f in range(3):
            ids=[i for i in pool if assignment[i]==f]
            assert len(ids)==638 and sum(truth[i] for i in ids)==82 and sum(areas.get(i,1.)<=.005 for i in ids)==15
    # Independent second-partition reconstruction from the published rule.
    strata={}
    for i in p['pool_ids']:
        row=by[i]; key=(truth[i],bool(truth[i] and areas[i]<=.005))
        strata.setdefault(key,{}).setdefault(row['group'],[]).append(i)
    regenerated={}
    for key,groups in strata.items():
        for n,g in enumerate(sorted(groups,key=lambda g:digest([cfg['confirmation_partition_seed'],list(key),g]))):
            for i in groups[g]: regenerated[i]=n%3
    assert regenerated==p['confirmation_folds']
    indexes={}; positive_views={}; artifact_checks=0
    published_parent=read(Path(__file__).resolve().parents[2]/'docs/research/small-defects/export-index.json')['files']
    assert sha(parent/'features/index.json')==published_parent['features--index.json']['sha256']
    assert sha(parent/'cache-complete.json')==published_parent['cache-complete.json']['sha256']
    old_index=read(parent/'features/index.json')['rows']
    for i in sorted(pool):
        entry=old_index[i]
        kinds=['coarse','coarse_roi']+(['coarse_target'] if truth[i] else [])
        for kind in kinds:
            assert sha(entry[kind+'_path'])==entry[kind+'_sha256']; artifact_checks+=1
    for f in range(3):
        cache=root/f'tile-cache-{f}'; marker=read(cache/'complete.json')
        assert sha(cache/'index.json')==marker['index_sha256']
        index=read(cache/'index.json'); assert index['protocol_identity']==p['identity'] and marker['count']==638
        for i,entry in index['rows'].items():
            assert i in pool and p['development_folds'][i]==f and entry['parent_id']==i
            assert sha(entry['feature_path'])==entry['feature_sha256']; artifact_checks+=1
            features=np.load(entry['feature_path'],mmap_mode='r',allow_pickle=False)
            assert features.shape==(10,384,32,32) and features.dtype==np.float16 and np.isfinite(features).all()
            height,width=by[i]['original_shape'][:2]  # rotate: RGB width=original height
            xs=sorted(set([*range(0,height-160+1,128),height-160])); ys=sorted(set([*range(0,width-160+1,128),width-160]))
            boxes=[[x,y,x+160,y+160] for y in ys for x in xs]; assert entry['boxes']==boxes
            if truth[i]:
                assert sha(entry['target_path'])==entry['target_sha256']; artifact_checks+=1
                target=np.load(entry['target_path'],allow_pickle=False); assert target.shape==(10,32,32)
                with Image.open(by[i]['mask_path']) as im: mask=im.convert('L').transpose(Image.Transpose.ROTATE_270)
                independently=[]
                for box in boxes:
                    view=np.asarray(mask.crop(box).resize((256,256),Image.Resampling.NEAREST))>0
                    independently.append(view.reshape(32,8,32,8).max((1,3)))
                assert np.array_equal(target,np.stack(independently)) and target.any()
                positive_views[i]=[j for j in range(10) if target[j].any()]
            indexes[i]=entry
    assert set(indexes)==pool

    def expected_roles(stage,f):
        assignment=p['development_folds' if stage=='dev' else 'confirmation_folds']
        salt=f'outer-{f}' if stage=='dev' else f'confirm-{f}'
        seed=old['config']['seed'] if stage=='dev' else cfg['confirmation_partition_seed']
        available=[i for i in p['pool_ids'] if assignment[i]!=f]
        normals=sorted([i for i in available if not truth[i]],key=lambda i:digest([seed,salt,by[i]['group']]))
        cal=normals[:200]; base=sorted(normals[200:328])+sorted(i for i in available if truth[i])
        candidates=[i for i in available if not truth[i] and i not in set(base+cal)]
        evaluation=[i for i in p['pool_ids'] if assignment[i]==f]
        return base,cal,candidates,evaluation

    selected=read(root/'selection.json'); identity(selected); assert selected['protocol_identity']==p['identity']
    moments={}; models={}; records={}; checks={}; mining_checks=0; usage_checks=0; calibration_checks=0
    requests=[('dev',f,seed,method) for seed in cfg['development_seeds'] for f in range(3) for method in cfg['methods']]
    requests += [('confirm',f,cfg['confirmation_seed'],method) for f in range(3) for method in dict.fromkeys(['baseline84',selected['method']])]
    for stage,f,seed,method in requests:
        name=f'{stage}-f{f}-s{seed}-{method}'; output=root/name
        plan,training=read(output/'plan.json'),read(output/'training.json'); state=torch.load(output/'model.pt',weights_only=True)
        finite(state); complete=read(output/'complete.json'); assert complete['protocol_identity']==p['identity']
        assert sha(output/'model.pt')==training['model_sha256']==complete['model_sha256']
        assert plan['protocol_identity']==p['identity'] and training['seed']==plan['seed']==seed and training['method']==plan['method']==method
        base,cal,candidates,evaluation=expected_roles(stage,f)
        assert plan['base_ids']==base and plan['calibration_ids']==cal and plan['evaluation_ids']==evaluation
        fitting=training['fitting_ids']; assert len(fitting)==len(set(fitting))==420 and len(base)==292
        assert set(fitting).isdisjoint(set(cal+evaluation)|reserved)
        assert training['normalization_ids']==base and set(base).issubset(fitting)
        if method=='baseline84':
            hard=training['hard_ids']; assert len(hard)==128 and fitting==base+hard and set(hard).issubset(candidates)
            if training['reused']:
                assert stage=='dev' and seed==4217
                source=p['baseline_reuse'][str(f)]; assert sha(output/'model.pt')==source['sha256']
                assert sha(Path(source['path']).parent/'training.json')==source['training_sha256']
            else:
                for key,epochs,ids in [('base',60,base),('hard',12,fitting),('aligned',12,fitting)]:
                    fit=training[key]
                    assert [r['epoch'] for r in fit['history']]==list(range(1,epochs+1))
                    assert all(np.isfinite(r['loss']) for r in fit['history'])
                    assert fit['view_usage']=={i:{'full':epochs} for i in ids}
                checkpoint=torch.load(output/'base60.pt',weights_only=True); finite(checkpoint)
                assert torch.equal(checkpoint['mean'],state['mean']) and torch.equal(checkpoint['std'],state['std'])
                ranked=training['mining_scores']; assert {r['id'] for r in ranked}==set(candidates) and len(ranked)==len(candidates)
                assert ranked==sorted(ranked,key=lambda r:(-r['score'],r['id'])) and hard==[r['id'] for r in ranked[:128]]
                for row in ranked:
                    entry=old_index[row['id']]
                    x=np.load(entry['coarse_path'],mmap_mode='r',allow_pickle=False).astype(np.float32)
                    roi=np.load(entry['coarse_roi_path'],allow_pickle=False)
                    actual=score_base(checkpoint,x,roi)
                    assert abs(actual-row['score'])<=1e-5
                    mining_checks+=1
            key=tuple(base)
            if key not in moments:
                sums,squares,n=np.zeros(384),np.zeros(384),0
                for i in base:
                    x=np.load(old_index[i]['coarse_path'],mmap_mode='r',allow_pickle=False).astype(np.float64)
                    sums+=x.sum((1,2)); squares+=np.square(x).sum((1,2)); n+=x.shape[1]*x.shape[2]
                moments[key]=(sums/n,np.sqrt(np.maximum(squares/n-(sums/n)**2,.01)))
            mean,std=moments[key]
            assert np.allclose(state['mean'].numpy().ravel(),mean,atol=.002,rtol=.002)
            assert np.allclose(state['std'].numpy().ravel(),std,atol=.002,rtol=.002)
        else:
            parent_name=f'{stage}-f{f}-s{seed}-baseline84'; parent_state=models[parent_name]
            parent_training=read(root/parent_name/'training.json')
            assert training['parent_model_sha256']==parent_training['model_sha256'] and fitting==parent_training['fitting_ids']
            for key in ('mean','std'): assert torch.equal(state[key],parent_state[key])
            for key in ('classify.weight','classify.bias'): assert torch.equal(state['head'][key],parent_state['head'][key])
            continuation=training['continuation']; assert [r['epoch'] for r in continuation['history']]==list(range(1,25))
            assert all(np.isfinite(r['loss']) for r in continuation['history'])
            usage=continuation['view_usage']; assert set(usage)==set(fitting) and all(sum(v.values())==24 for v in usage.values())
            if not cfg['methods'][method]['tiles']:
                assert usage=={i:{'full':24} for i in fitting}
            else:
                order=torch.Generator().manual_seed(seed+9000); generator=torch.Generator().manual_seed(seed+18001)
                rebuilt={i:Counter() for i in fitting}
                for epoch in range(24):
                    for position in torch.randperm(len(fitting),generator=order):
                        i=fitting[int(position)]; tile='full'
                        if float(torch.rand((),generator=generator))<.5:
                            available=list(range(10))
                            if truth[i]:
                                available=positive_views[i]
                            tile=str(available[int(torch.randint(len(available),(1,),generator=generator))])
                        rebuilt[i][tile]+=1
                assert {i:dict(v) for i,v in rebuilt.items()}==usage
                usage_checks+=len(fitting)*24
        models[name]=state
        for alpha in cfg['alphas']:
            path=output/f'alpha-{alpha:.2f}.json'; record=read(path)
            result=recount(record,evaluation,truth,areas,cal,alpha)
            records[(stage,f,seed,method,alpha)]=record; checks[f'{name}/{alpha:.2f}']=result

        pair=[records[(stage,f,seed,method,alpha)] for alpha in cfg['alphas']]
        assert pair[0]['calibration_scores']==pair[1]['calibration_scores']
        assert [(r['id'],r['score']) for r in pair[0]['predictions']]==[(r['id'],r['score']) for r in pair[1]['predictions']]
        with torch.inference_mode():
            for row in pair[0]['calibration_scores']:
                value=score_complete(state,row['id'],old_index,indexes,cfg['methods'][method]['tiles'])
                assert abs(value-row['score'])<=1e-5
                calibration_checks+=1

    candidates={}
    for method in cfg['methods']:
        for alpha in cfg['alphas']:
            candidates[(method,alpha)]=[checks[f'dev-f{f}-s{seed}-{method}/{alpha:.2f}'] for seed in cfg['development_seeds'] for f in range(3)]
    def rank(item):
        (method,alpha),rows=item; worst=max(r['fpr'] for r in rows); c=cfg['methods'][method]
        return (worst>=.1,worst if worst>=.1 else 0.,-min(r['recall'] for r in rows),
                -min(r['small_defects']['detected']/15 for r in rows),-sum(r['recall'] for r in rows)/9,
                sum(r['fpr'] for r in rows)/9,c['extra_epochs'],c['tiles'],c['small_weighted'],method,alpha)
    choice=min(candidates.items(),key=rank)[0]; assert choice==(selected['method'],selected['alpha'])
    for files in selected['input_files'].values():
        for row in files:
            assert sha(row['path'])==row['sha256'] and Path(row['path']).parent.name.startswith('dev-')
    dev_pass=all(r['point_goal_met'] for r in candidates[choice]); assert dev_pass==selected['all_nine_development_goals_met']
    confirmation=[checks[f'confirm-f{f}-s{cfg["confirmation_seed"]}-{choice[0]}/{choice[1]:.2f}'] for f in range(3)]
    paired={}
    for (stage,f,seed,method,alpha),record in records.items():
        if method=='baseline84': continue
        baseline=records[(stage,f,seed,'baseline84',alpha)]
        paired[f'{stage}-f{f}-s{seed}-{method}/{alpha:.2f}']=paired_changes(baseline,record,truth,areas)
    write(root/'paired-changes.json',{'protocol_identity':p['identity'],'comparisons':paired,
          'meaning':'New defect flags are detections gained; removed defect flags are detections lost. New normal flags are false alarms added; removed normal flags are false alarms avoided.',
          'scope':'Same-policy, same-fold, same-seed baseline pairs; repeated originals across seeds are not independent evidence.'})
    reference_rows={key:{r['id']:r for r in record['predictions']} for key,record in records.items()}
    all_native=[]; native_heads=0; native_thresholds=0
    for fold in range(3):
        native=read(root/f'native-f{fold}.json')
        assert native['protocol_identity']==p['identity'] and native['selection_identity']==selected['identity']
        assert native['all_scores_within_1e_5'] and native['decision_mismatches']==0 and not native['official_test_accessed']
        assert not native['cache_arrays_read'] and not native['masks_read'] and native['originals']==638
        assert sha(Path(__file__).parent/'replay_warm_small.py')==native['source_sha256']
        for filename,value in native['model_hashes_unchanged'].items(): assert sha(filename)==value
        rows=read(root/f'native-f{fold}-predictions.json')['rows']; assert len(rows)==638
        for row in rows:
            i=row['id']; assert p['development_folds'][i]==fold
            expected_keys=[('dev',fold,seed,method) for seed in cfg['development_seeds'] for method in cfg['methods']]
            expected_keys += [('confirm',p['confirmation_folds'][i],cfg['confirmation_seed'],method) for method in dict.fromkeys(['baseline84',selected['method']])]
            assert set(row['predictions'])=={'/'.join(map(str,k)) for k in expected_keys}
            for key in expected_keys:
                actual=row['predictions']['/'.join(map(str,key))]
                for alpha in cfg['alphas']:
                    reference=reference_rows[(*key,alpha)][i]
                    assert abs(actual['score']-reference['score'])<=1e-5 and actual['flags'][f'{alpha:.2f}']==reference['flagged']
        all_native.extend(r['id'] for r in rows); native_heads+=native['head_evaluations']; native_thresholds+=native['threshold_evaluations']
    assert len(all_native)==len(set(all_native))==1914 and set(all_native)==pool
    assert native_heads==1914*(15+len(dict.fromkeys(['baseline84',selected['method']])))
    assert native_thresholds==2*native_heads
    summary={'protocol_identity':p['identity'],'selection_identity':selected['identity'],'selected_method':choice[0],'selected_alpha':choice[1],
             'all_nine_development_goals_met':dev_pass,'all_three_second_partition_goals_met':all(r['point_goal_met'] for r in confirmation),
             'all_twelve_internal_goals_met':dev_pass and all(r['point_goal_met'] for r in confirmation),
             'checks':checks,'qualification':{f'{method}/{alpha:.2f}':all(r['point_goal_met'] for r in rows) for (method,alpha),rows in candidates.items()},
             'originals':1914,'native_head_evaluations':native_heads,'native_threshold_evaluations':native_thresholds,
             'new_independent_factory_confirmation':False,'official_test_accessed':False,
             'documented_infrastructure_recovery':recovery is not None,
             'scope':'observed training originals; development selection and predeclared second-partition robustness, not fresh data'}
    write(root/'summary.json',summary)
    resources={path.name:read(path) for path in root.glob('*-resources.json')}
    assert all(r['status']=='completed' and r['peak_sampled_rss_bytes']<=r['rss_cap_bytes'] for r in resources.values())
    protected=['components/historical_benchmark','components/data/manifests','docs/research/small-defects','docs/research/validation','docs/research/alignment','docs/research/round1','docs/research/round2']
    assert not subprocess.check_output(['git','diff','--name-only',p['source_commit'],'--',*protected],text=True).strip()
    verification={'protocol_identity':p['identity'],'verifier_sha256':sha(Path(__file__)),
                  'feature_artifact_hashes_checked':artifact_checks,'raw_training_mask_hashes_and_labels_checked':len(truth),
                  'normalization_sets_reconstructed':len(moments),'hard_normal_scores_reconstructed':mining_checks,
                  'calibration_scores_reconstructed_from_weights':calibration_checks,
                  'training_crop_choices_reconstructed':usage_checks,'metric_records_recounted':len(checks),
                  'paired_comparisons_with_original_ids':len(paired),
                  'selection_independently_verified':True,'second_partition_assignment_reconstructed':True,
                  'no_official_test_access':True,'protected_old_evidence_unchanged':True,
                  'same_work_session_audit_not_third_party_certification':True,
                  'documented_infrastructure_recovery':None if recovery is None else {
                      'registration_sha256':sha(root/'recovery.json'),
                      'copied_phases':recovery['completed_phases_reused_without_refitting'],
                      'preserved_failed_phases':list(recovery['preserved_failed_attempts']),
                      'numeric_changes':False},
                  'model_artifacts':{str(path.relative_to(root)):{'sha256':sha(path),'bytes':path.stat().st_size} for path in root.rglob('*.pt')},
                  'resources':resources}
    write(root/'verification.json',verification)
    args.export.mkdir(parents=True,exist_ok=False); exported={}
    for path in sorted(root.rglob('*')):
        if path.is_file() and path.suffix in ('.json','.log'):
            name='--'.join(path.relative_to(root).parts); destination=args.export/name
            destination.write_bytes(path.read_bytes()); assert sha(destination)==sha(path)
            exported[name]={'sha256':sha(path),'bytes':path.stat().st_size}
    if recovery is not None:
        extra={}
        for failure in recovery['preserved_failed_attempts'].values():
            for path in failure['preserved_files']:
                path=Path(path); extra['interrupted--'+'--'.join(path.relative_to(original).parts)]=path
        for path in recovery['diagnostic_files']:
            path=Path(path); extra['diagnostic--'+path.name]=path
        for name,path in extra.items():
            destination=args.export/name; assert not destination.exists()
            destination.write_bytes(path.read_bytes()); assert sha(destination)==sha(path)
            exported[name]={'sha256':sha(path),'bytes':path.stat().st_size}
    write(args.export/'export-index.json',{'protocol_identity':p['identity'],'files':exported,'excluded':'model tensors and feature/target arrays remain local, with recorded hashes'})
    print(json.dumps({'selected':choice,'all_twelve_internal_goals_met':summary['all_twelve_internal_goals_met'],
                      'feature_hashes':artifact_checks,'native_threshold_evaluations':native_thresholds,'exported_files':len(exported)},indent=2))


if __name__=='__main__': main()
