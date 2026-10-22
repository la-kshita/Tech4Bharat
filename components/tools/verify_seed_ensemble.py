"""Independent member-role, mining, arithmetic and native-ensemble audit."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.special import expit
import torch

from component_inspector.research.ksdd2_seed_ensemble import context,read,sha,digest
from verify_warm_small import recount,paired_changes,score_base,score_complete,finite,write


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--export',type=Path,required=True)
    parser.add_argument('--registration',type=Path,default=Path('docs/research/2026-10-21-seed-ensemble-preregistration.json'))
    args=parser.parse_args(); root=args.root.resolve(); torch.set_num_threads(4); torch.set_num_interop_threads(1)
    p,main_root=context(root); parent=read(main_root/'protocol.json'); selected=read(root/'selection.json')
    snapshot=read(args.registration); assert snapshot['protocol']==p and snapshot['before_any_ensemble_score_or_added_fit']
    for relative,value in snapshot['frozen_files'].items(): assert sha(root/relative)==value
    main_verification=read(main_root/'verification.json')
    assert sha(main_root/'verification.json')==selected['main_verification_sha256']
    assert sha(Path(__file__).parent/'verify_warm_small.py')==main_verification['verifier_sha256']
    main_selection=read(main_root/'selection.json'); assert sha(main_root/'selection.json')==selected['main_selection_sha256']
    other=dict(selected); expected=other.pop('identity'); assert digest(other)==expected
    assert selected['method']==main_selection['method'] and selected['primary_alpha']==.05
    assert selected['main_selection_identity']==main_selection['identity'] and selected['no_confirmation_outcome_selection']
    method=selected['method']; methods=list(dict.fromkeys(['baseline84',method]))
    for relative,value in main_verification['model_artifacts'].items(): assert sha(main_root/relative)==value['sha256']
    export_parent=read(Path(__file__).resolve().parents[2]/'docs/research/warm-small/export-index.json')['files']
    for path in main_root.rglob('*'):
        if path.is_file() and path.suffix in ('.json','.log'):
            name='--'.join(path.relative_to(main_root).parts)
            assert sha(path)==export_parent[name]['sha256'],'Main evidence changed: '+name
    old_root=Path(parent['parent']); old=read(old_root/'protocol.json')
    old_published=read(Path(__file__).resolve().parents[2]/'docs/research/small-defects/export-index.json')['files']
    assert sha(old_root/'features/index.json')==old_published['features--index.json']['sha256']
    assert sha(old_root/'cache-complete.json')==old_published['cache-complete.json']['sha256']
    manifest=read(Path(old['parent'])/'manifest.json'); by={r['id']:r for r in manifest['records']}
    pool=set(parent['pool_ids']); reserved=set(parent['reserved_ids']); truth={}; areas={}
    for identity in sorted(pool):
        row=by[identity]; assert row['official_split']=='train' and sha(row['mask_path'])==row['hashes']['mask_sha256']
        with Image.open(row['mask_path']) as image: mask=np.asarray(image.convert('L'))>0
        truth[identity]=int(mask.any()); assert truth[identity]==row['label']
        if truth[identity]: areas[identity]=float(mask.mean())
    assert len(pool)==1914 and sum(truth.values())==246 and areas==parent['areas']
    old_index=read(old_root/'features/index.json')['rows']; tiles={}; feature_checks=0
    for fold in range(3):
        index=main_root/f'tile-cache-{fold}/index.json'
        assert sha(index)==read(index.parent/'complete.json')['index_sha256']
        tiles.update(read(index)['rows'])
    tiled=parent['config']['methods'][method]['tiles']; positive_views={}
    for identity in sorted(pool):
        entry=old_index[identity]
        for kind in ['coarse','coarse_roi']+(['coarse_target'] if truth[identity] else []):
            assert sha(entry[kind+'_path'])==entry[kind+'_sha256']; feature_checks+=1
        if tiled:
            assert sha(tiles[identity]['feature_path'])==tiles[identity]['feature_sha256']; feature_checks+=1
            if truth[identity]:
                assert sha(tiles[identity]['target_path'])==tiles[identity]['target_sha256']; feature_checks+=1
                target=np.load(tiles[identity]['target_path'],allow_pickle=False)
                positive_views[identity]=[i for i in range(10) if target[i].any()]
                assert positive_views[identity]
    records={}; model_hashes={}; mining=0; calibration=0; usage_count=0; moments={}
    for seed_text,metadata in p['members'].items():
        seed=int(seed_text); member=Path(metadata['root']); member_p=read(member/'protocol.json')
        assert sha(member/'protocol.json')==metadata['protocol_sha256']
        expected_config=dict(parent['config']); expected_config['confirmation_seed']=seed
        assert member_p['config']==expected_config and member_p['source_sha256']==p['source_sha256']
        assert member_p['extension_parent_protocol']==parent['identity']
        for fold in range(3):
            template=read(main_root/f'confirm-f{fold}-s4220-baseline84/plan.json')
            base=template['base_ids']; cal=template['calibration_ids']; evaluation=template['evaluation_ids']
            candidates=[i for i in parent['pool_ids'] if parent['confirmation_folds'][i]!=fold and not truth[i] and i not in set(base+cal)]
            assert len(candidates)==784 and len(base)==292 and len(cal)==200 and len(evaluation)==638
            baseline_state=None; baseline_training=None
            for candidate in methods:
                directory=member/f'confirm-f{fold}-s{seed}-{candidate}'
                plan=read(directory/'plan.json'); training=read(directory/'training.json')
                assert plan['protocol_identity']==member_p['identity'] and plan['stage']=='confirm' and plan['fold']==fold
                assert training['seed']==plan['seed']==seed and training['method']==plan['method']==candidate
                assert plan['base_ids']==base and plan['calibration_ids']==cal and plan['evaluation_ids']==evaluation
                fitting=training['fitting_ids']; assert len(fitting)==len(set(fitting))==420
                assert set(fitting).isdisjoint(set(cal+evaluation)|reserved) and training['normalization_ids']==base
                state=torch.load(directory/'model.pt',weights_only=True); finite(state)
                model_hashes[str(directory/'model.pt')]=sha(directory/'model.pt')
                assert model_hashes[str(directory/'model.pt')]==training['model_sha256']==read(directory/'complete.json')['model_sha256']
                assert read(directory/'complete.json')['protocol_identity']==member_p['identity']
                if candidate=='baseline84':
                    assert not training['reused']; hard=training['hard_ids']
                    assert fitting==base+hard and len(set(hard))==128 and set(hard).issubset(candidates)
                    for key,epochs,ids in [('base',60,base),('hard',12,fitting),('aligned',12,fitting)]:
                        fit=training[key]; assert [r['epoch'] for r in fit['history']]==list(range(1,epochs+1))
                        assert all(np.isfinite(r['loss']) for r in fit['history'])
                        assert fit['view_usage']=={i:{'full':epochs} for i in ids}
                    initial=torch.load(directory/'base60.pt',weights_only=True); finite(initial)
                    model_hashes[str(directory/'base60.pt')]=sha(directory/'base60.pt')
                    assert torch.equal(initial['mean'],state['mean']) and torch.equal(initial['std'],state['std'])
                    ranked=training['mining_scores']; assert {r['id'] for r in ranked}==set(candidates) and len(ranked)==784
                    assert ranked==sorted(ranked,key=lambda r:(-r['score'],r['id'])) and hard==[r['id'] for r in ranked[:128]]
                    with torch.inference_mode():
                        for row in ranked:
                            entry=old_index[row['id']]
                            value=score_base(initial,np.load(entry['coarse_path'],allow_pickle=False).astype(np.float32),np.load(entry['coarse_roi_path'],allow_pickle=False))
                            assert abs(value-row['score'])<=1e-5; mining+=1
                    if fold not in moments:
                        sums,squares,n=np.zeros(384),np.zeros(384),0
                        for i in base:
                            x=np.load(old_index[i]['coarse_path'],allow_pickle=False).astype(np.float64)
                            sums+=x.sum((1,2)); squares+=np.square(x).sum((1,2)); n+=x.shape[1]*x.shape[2]
                        moments[fold]=(sums/n,np.sqrt(np.maximum(squares/n-(sums/n)**2,.01)))
                    mean,std=moments[fold]
                    assert np.allclose(state['mean'].numpy().ravel(),mean,atol=.002,rtol=.002)
                    assert np.allclose(state['std'].numpy().ravel(),std,atol=.002,rtol=.002)
                    baseline_state,baseline_training=state,training
                else:
                    assert training['parent_model_sha256']==baseline_training['model_sha256'] and fitting==baseline_training['fitting_ids']
                    for key in ('mean','std'): assert torch.equal(state[key],baseline_state[key])
                    for key in ('classify.weight','classify.bias'): assert torch.equal(state['head'][key],baseline_state['head'][key])
                    fit=training['continuation']; assert [r['epoch'] for r in fit['history']]==list(range(1,25))
                    assert all(np.isfinite(r['loss']) for r in fit['history'])
                    usage=fit['view_usage']; assert set(usage)==set(fitting) and all(sum(v.values())==24 for v in usage.values())
                    if not tiled: assert usage=={i:{'full':24} for i in fitting}
                    else:
                        order=torch.Generator().manual_seed(seed+9000); generator=torch.Generator().manual_seed(seed+18001)
                        rebuilt={i:Counter() for i in fitting}
                        for epoch in range(24):
                            for index in torch.randperm(len(fitting),generator=order):
                                identity=fitting[int(index)]; tile='full'
                                if float(torch.rand((),generator=generator))<.5:
                                    available=positive_views[identity] if truth[identity] else list(range(10))
                                    tile=str(available[int(torch.randint(len(available),(1,),generator=generator))])
                                rebuilt[identity][tile]+=1
                        assert {i:dict(v) for i,v in rebuilt.items()}==usage; usage_count+=24*len(fitting)
                pair=[]
                for alpha in (.05,.08):
                    record=read(directory/f'alpha-{alpha:.2f}.json'); assert record['protocol_identity']==member_p['identity']
                    recount(record,evaluation,truth,areas,cal,alpha); records[(seed,fold,candidate,alpha)]=record; pair.append(record)
                assert pair[0]['calibration_scores']==pair[1]['calibration_scores']
                assert [(r['id'],r['score']) for r in pair[0]['predictions']]==[(r['id'],r['score']) for r in pair[1]['predictions']]
                with torch.inference_mode():
                    for row in pair[0]['calibration_scores']:
                        value=score_complete(state,row['id'],old_index,tiles,parent['config']['methods'][candidate]['tiles'])
                        assert abs(value-row['score'])<=1e-5; calibration+=1
                print('AUDITED MEMBER',seed,fold,candidate,flush=True)
    checks={}; pairs={}; ensemble_records={}; ensemble_maps={}
    for stage in ('dev','confirm'):
        seeds=p['config']['development_members' if stage=='dev' else 'confirmation_members']
        for fold in range(3):
            directories=[]
            for seed in seeds:
                directory=(main_root if stage=='dev' or seed==4220 else Path(p['members'][str(seed)]['root']))/f'{stage}-f{fold}-s{seed}-{method}'
                directories.append(directory)
            for alpha in (.05,.08):
                name=f'{stage}-f{fold}-alpha-{alpha:.2f}.json'; row=read(root/name)
                assert row['protocol_identity']==p['identity'] and row['primary']==(alpha==.05)
                sources=[read(directory/'alpha-0.05.json') for directory in directories]
                maps=[{r['id']:r for r in source['predictions']} for source in sources]
                cal_maps=[{r['id']:r for r in source['calibration_scores']} for source in sources]
                ids=set(maps[0]); cal_ids=set(cal_maps[0]); assert all(set(m)==ids for m in maps) and all(set(m)==cal_ids for m in cal_maps)
                for member,directory in zip(row['members'],directories):
                    assert member['model_path']==str(directory/'model.pt') and member['model_sha256']==sha(directory/'model.pt')
                    assert member['record_path']==str(directory/'alpha-0.05.json') and member['record_sha256']==sha(directory/'alpha-0.05.json')
                for value in row['predictions']:
                    expected=float(np.mean(expit([m[value['id']]['score'] for m in maps])))
                    assert abs(value['score']-expected)<=1e-12 and value['label']==truth[value['id']]
                for value in row['calibration_scores']:
                    expected=float(np.mean(expit([m[value['id']]['score'] for m in cal_maps])))
                    assert abs(value['score']-expected)<=1e-12 and truth[value['id']]==0
                checks[name]=recount(row,ids,truth,areas,cal_ids,alpha); ensemble_records[(stage,fold,alpha)]=row
                ensemble_maps[(stage,fold,alpha)]={r['id']:r for r in row['predictions']}
                for seed in seeds:
                    location=main_root if stage=='dev' or seed==4220 else Path(p['members'][str(seed)]['root'])
                    for candidate in methods:
                        source=read(location/f'{stage}-f{fold}-s{seed}-{candidate}'/f'alpha-{alpha:.2f}.json')
                        pairs[f'{stage}-f{fold}-member{seed}-{candidate}/{alpha:.2f}']=paired_changes(source,row,truth,areas)
    raw_summary=read(root/'summary.json'); assert raw_summary['protocol_identity']==p['identity'] and raw_summary['method']==method
    for name,row in checks.items():
        for key in ('counts','small_defects','recall','fpr','point_goal_met'): assert raw_summary['checks'][name][key]==row[key]
    primary=[row for name,row in checks.items() if name.endswith('alpha-0.05.json')]
    assert raw_summary['all_six_primary_ensemble_goals_met']==all(row['point_goal_met'] for row in primary)
    main_native={}; new_native={}; heads=0; thresholds=0
    for fold in range(3):
        for name in (f'native-f{fold}.json',f'native-f{fold}-predictions.json'):
            assert sha(main_root/name)==export_parent[name]['sha256']
        for row in read(main_root/f'native-f{fold}-predictions.json')['rows']:
            assert row['id'] not in main_native; main_native[row['id']]=row['predictions']
        proof=read(root/f'native-new-f{fold}.json')
        assert proof['protocol_identity']==p['identity'] and proof['source_sha256']==sha(Path(__file__).parent/'replay_seed_ensemble.py')
        assert proof['all_scores_within_1e_5'] and proof['decision_mismatches']==0 and proof['originals']==638
        assert not proof['cache_arrays_read'] and not proof['masks_read'] and not proof['official_test_accessed']
        for filename,value in proof['model_hashes_unchanged'].items(): assert sha(filename)==value
        rows=read(root/f'native-new-f{fold}-predictions.json')['rows']; assert len(rows)==638
        for row in rows:
            identity=row['id']; assert parent['development_folds'][identity]==fold and identity not in new_native
            assert set(row['predictions'])=={f'{seed}/{candidate}' for seed in p['members'] for candidate in methods}
            for key,actual in row['predictions'].items():
                seed,candidate=key.split('/'); evaluation_fold=parent['confirmation_folds'][identity]
                for alpha in (.05,.08):
                    source={r['id']:r for r in records[(int(seed),evaluation_fold,candidate,alpha)]['predictions']}[identity]
                    assert abs(source['score']-actual['score'])<=1e-5 and source['flagged']==actual['flags'][f'{alpha:.2f}']
            new_native[identity]=row['predictions']
        heads+=proof['head_evaluations']; thresholds+=proof['threshold_evaluations']
    assert set(main_native)==set(new_native)==pool and heads==1914*2*len(methods) and thresholds==2*heads
    ensemble_decisions=0; max_difference=0.
    for identity in sorted(pool):
        for stage in ('dev','confirm'):
            fold=parent['development_folds' if stage=='dev' else 'confirmation_folds'][identity]
            seeds=p['config']['development_members' if stage=='dev' else 'confirmation_members']; logits=[]
            for seed in seeds:
                if stage=='dev' or seed==4220: logits.append(main_native[identity][f'{stage}/{fold}/{seed}/{method}']['score'])
                else: logits.append(new_native[identity][f'{seed}/{method}']['score'])
            score=float(np.mean(expit(logits)))
            for alpha in (.05,.08):
                expected=ensemble_maps[(stage,fold,alpha)][identity]
                max_difference=max(max_difference,abs(score-expected['score']))
                assert abs(score-expected['score'])<=1e-5
                assert (score>ensemble_records[(stage,fold,alpha)]['threshold_record']['threshold'])==expected['flagged']
                ensemble_decisions+=1
    assert ensemble_decisions==7656
    resources={path.name:read(path) for path in root.glob('*-resources.json')}
    assert all(r['status']=='completed' and r['peak_sampled_rss_bytes']<=r['rss_cap_bytes'] for r in resources.values())
    write(root/'paired-changes.json',{'protocol_identity':p['identity'],'comparisons':pairs})
    write(root/'audit-summary.json',{'protocol_identity':p['identity'],'checks':checks,
          'all_six_primary_ensemble_goals_met':all(row['point_goal_met'] for row in primary),
          'new_independent_factory_data':False,'original_single_model_study_unchanged':True})
    verification={'protocol_identity':p['identity'],'verifier_sha256':sha(Path(__file__)),
                  'independent_arithmetic_helper_sha256':sha(Path(__file__).parent/'verify_warm_small.py'),
                  'raw_training_mask_hashes_and_labels_checked':len(truth),'feature_artifact_hashes_checked':feature_checks,
                  'normalization_sets_reconstructed':len(moments),'hard_normal_scores_reconstructed':mining,
                  'calibration_scores_reconstructed':calibration,'training_crop_choices_reconstructed':usage_count,
                  'new_member_head_evaluations_from_rgb':heads,'new_member_threshold_evaluations_from_rgb':thresholds,
                  'ensemble_decisions_from_independent_native_member_scores':ensemble_decisions,
                  'max_native_ensemble_score_difference':max_difference,'native_ensemble_flag_mismatches':0,
                  'paired_comparisons_with_original_ids':len(pairs),'model_artifacts':model_hashes,'resources':resources,
                  'main_verification_sha256_unchanged':sha(main_root/'verification.json'),
                  'no_official_test_access':True,'same_session_audit_not_third_party_certification':True}
    write(root/'verification.json',verification); args.export.mkdir(parents=True,exist_ok=False); files={}
    for path in sorted(root.rglob('*')):
        if path.is_file() and path.suffix in ('.json','.log'):
            name='--'.join(path.relative_to(root).parts); destination=args.export/name
            destination.write_bytes(path.read_bytes()); assert sha(destination)==sha(path)
            files[name]={'sha256':sha(path),'bytes':path.stat().st_size}
    write(args.export/'export-index.json',{'protocol_identity':p['identity'],'files':files,
          'excluded':'Model tensors and arrays remain local with recorded hashes; main study evidence is linked separately.'})
    print(json.dumps({'all_six_primary_ensemble_goals_met':raw_summary['all_six_primary_ensemble_goals_met'],
                      'hard_normal_scores_verified':mining,'native_ensemble_decisions_verified':ensemble_decisions,'exported_files':len(files)},indent=2))


if __name__=='__main__': main()
