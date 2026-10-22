"""Post-fit diagnostic only: never choose models or thresholds from fitting scores."""
import argparse
from pathlib import Path
import sys

import numpy as np
import psutil
from scipy.special import expit
import torch

from component_inspector.research.ksdd2_seed_ensemble import context,read,write,sha
from component_inspector.research.ksdd2_warm_small import context as warm_context,Store,load_head,score_views
from component_inspector.vision.resources import run_bounded
from verify_warm_small import score_complete


@torch.inference_mode()
def diagnose(root,output):
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    p,main=context(root); assert (root/'verification.json').exists()
    parent,old,manifest,by=warm_context(main); method=read(root/'selection.json')['method']
    store=Store(main,parent,old); old_index=read(Path(parent['parent'])/'features/index.json')['rows']
    tiles={}
    for fold in range(3): tiles.update(read(main/f'tile-cache-{fold}/index.json')['rows'])
    rows=[]; ensemble=[]; hashes={}; max_difference=0.; score_count=0
    for stage in ('dev','confirm'):
        seeds=p['config']['development_members' if stage=='dev' else 'confirmation_members']
        for fold in range(3):
            positive_scores={}; positive_ids=None
            for seed in seeds:
                source=main if stage=='dev' or seed==4220 else Path(p['members'][str(seed)]['root'])
                directory=source/f'{stage}-f{fold}-s{seed}-{method}'
                training=read(directory/'training.json'); fitting=training['fitting_ids']
                cal=read(directory/'plan.json')['calibration_ids']; evaluation=read(directory/'plan.json')['evaluation_ids']
                assert len(fitting)==len(set(fitting))==420 and set(fitting).isdisjoint(set(cal+evaluation)|set(parent['reserved_ids']))
                assert sha(directory/'model.pt')==training['model_sha256']
                hashes[str(directory/'model.pt')]=sha(directory/'model.pt')
                state=torch.load(directory/'model.pt',weights_only=True); head=load_head(state)
                values={}; direct_values={}
                for identity in fitting:
                    row=by[identity]; assert row['official_split']=='train'
                    value=score_views(head,state,store,row,parent['config']['methods'][method]['tiles'])
                    direct=score_complete(state,identity,old_index,tiles,parent['config']['methods'][method]['tiles'])
                    max_difference=max(max_difference,abs(value-direct)); assert abs(value-direct)<=1e-5
                    values[identity]=value; direct_values[identity]=direct; score_count+=1
                positives={i for i in fitting if by[i]['label']}; assert len(positives)==164
                assert positive_ids is None or positive_ids==positives; positive_ids=positives
                positive_scores[seed]={i:values[i] for i in positives}
                for alpha in (.05,.08):
                    held=read(directory/f'alpha-{alpha:.2f}.json'); threshold=held['threshold_record']['threshold']
                    assert all((values[i]>threshold)==(direct_values[i]>threshold) for i in fitting)
                    predictions=[{'id':i,'label':by[i]['label'],'score':values[i],'flagged':values[i]>threshold} for i in fitting]
                    counts={'TP':sum(r['label']==1 and r['flagged'] for r in predictions),
                            'FN':sum(r['label']==1 and not r['flagged'] for r in predictions),
                            'FP':sum(r['label']==0 and r['flagged'] for r in predictions),
                            'TN':sum(r['label']==0 and not r['flagged'] for r in predictions)}
                    small=[r for r in predictions if r['label'] and parent['areas'][r['id']]<=.005]
                    assert len(small)==30 and counts['TP']+counts['FN']==164 and counts['FP']+counts['TN']==256
                    rows.append({'stage':stage,'fold':fold,'seed':seed,'method':method,'alpha':alpha,
                                 'model_sha256':hashes[str(directory/'model.pt')],'threshold':threshold,
                                 'fitting_counts':counts,'fitting_recall':counts['TP']/164,'fitting_normal_fpr':counts['FP']/256,
                                 'fitting_small':{'detected':sum(r['flagged'] for r in small),'total':30},
                                 'held_out_counts':held['metrics']['counts'],'held_out_recall':held['metrics']['recall']['rate'],
                                 'held_out_normal_fpr':held['metrics']['accepted_normal_fpr']['rate'],'held_out_small':held['small_defects'],
                                 'predictions':predictions})
                print('FITTING GAP',stage,fold,seed,flush=True)
            for alpha in (.05,.08):
                threshold=read(root/f'{stage}-f{fold}-alpha-{alpha:.2f}.json')['threshold_record']['threshold']
                predictions=[]
                for identity in sorted(positive_ids):
                    value=float(np.mean(expit([positive_scores[seed][identity] for seed in seeds])))
                    predictions.append({'id':identity,'score':value,'flagged':value>threshold})
                small=[r for r in predictions if parent['areas'][r['id']]<=.005]
                ensemble.append({'stage':stage,'fold':fold,'alpha':alpha,'threshold':threshold,
                                 'fitting_positive_detected':sum(r['flagged'] for r in predictions),'fitting_positive_total':164,
                                 'fitting_small':{'detected':sum(r['flagged'] for r in small),'total':30},'predictions':predictions,
                                 'normal_fitting_fpr_not_claimed':'Members have different mined normal sets; only common fitting positives are summarized for the ensemble.'})
    for path,value in hashes.items(): assert sha(path)==value
    assert len(rows)==36 and len(ensemble)==12 and score_count==7560
    result={'protocol_identity':p['identity'],'source_sha256':sha(Path(__file__)),'role':'fitting diagnostic only; never a validation or qualification result',
            'method':method,'source_model_hashes_unchanged':hashes,'members':rows,'ensembles':ensemble,
            'fitting_head_scores_checked':score_count,'max_original_vs_independent_score_difference':max_difference,
            'all_scores_match_independent_convolution':True,'new_training_or_threshold_tuning':False,'official_test_access':False,
            'individual_fitting_threshold_decisions_reproduced':15120,
            'interpretation':'A fitting/held-out gap describes generalization behavior; it does not uniquely distinguish overfitting, insufficient features, label noise or distribution differences. Fitting originals and repeated members are not independent samples.'}
    write(output/'diagnostic.json',result); print('FITTING GAP DIAGNOSTIC COMPLETE',flush=True)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True); parser.add_argument('--worker',action='store_true')
    args=parser.parse_args(); root=args.root.resolve(); output=args.output.resolve()
    registration=read('docs/research/2026-10-21-fitting-gap-preregistration.json')
    assert registration['before_any_fitting_score_evaluation'] and registration['source_sha256']==sha(Path(__file__))
    if args.worker: diagnose(root,output); return
    output.mkdir(exist_ok=False); p,_=context(root); cfg=p['config']['resources']
    cap=min(cfg['rss_cap_gib']*1024**3,int(cfg['available_ram_fraction']*psutil.virtual_memory().available))
    value=run_bounded([sys.executable,str(Path(__file__).resolve()),'--root',str(root),'--output',str(output),'--worker'],
                     output/'worker.log',max_rss_bytes=cap,max_seconds=cfg['phase_seconds'])
    write(output/'resources.json',value); print(value,flush=True)
    if value['status']!='completed': raise RuntimeError('Preserve failed fitting diagnostic')


if __name__=='__main__': main()
