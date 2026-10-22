"""Rebuild all development/second-partition predictions from original RGB."""
import argparse
from pathlib import Path
import sys

import numpy as np
import psutil
import torch
from torch.nn import functional as F

from component_inspector.data.acquire import sha256
from component_inspector.research.ksdd2_round1 import image_array, read_json, write_json
from component_inspector.research.ksdd2_warm_small import (
    context, directory, extract_tiles, load_head, load_selection, native_rgb, score_views,
)
from component_inspector.vision.resources import run_bounded
from component_inspector.vision.spike_worker import pretrained_adapter


class Native:
    def __init__(self, full, roi, tiles): self.full,self.roi,self.tiles=full,roi,tiles
    def view(self,row,tile=None,target=False):
        if tile is not None or target: raise ValueError('No GT or training views in replay')
        return self.full,self.roi,None
    def array(self,identity,kind):
        if kind!='feature': raise ValueError('No target arrays in replay')
        return self.tiles


@torch.inference_mode()
def replay(root,fold):
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    p,old,manifest,by=context(root); selected=load_selection(root,p)
    models={}; expected={}; hashes={}
    requests=[('dev',fold,seed,method) for seed in p['config']['development_seeds'] for method in p['config']['methods']]
    requests += [('confirm',f,p['config']['confirmation_seed'],method) for f in range(3)
                 for method in dict.fromkeys(['baseline84',selected['method']])]
    for stage,f,seed,method in requests:
        output=directory(root,stage,f,seed,method)
        state=torch.load(output/'model.pt',weights_only=True,map_location='cpu')
        actual=sha256(output/'model.pt'); assert actual==read_json(output/'training.json')['model_sha256']
        key=(stage,f,seed,method); models[key]=(state,load_head(state),p['config']['methods'][method]['tiles'])
        hashes[str(output/'model.pt')]=actual
        for alpha in p['config']['alphas']:
            result=read_json(output/f'alpha-{alpha:.2f}.json')
            expected[(key,alpha)]=(result['threshold_record']['threshold'],{row['id']:row for row in result['predictions']})
    adapter=pretrained_adapter('ksdd2_surface',manifest['config'],Path(old['weights']))
    ids=[i for i in reversed(p['pool_ids']) if p['development_folds'][i]==fold]
    rows=[]; max_difference=0.; mismatches=0; head_evaluations=0; threshold_evaluations=0
    for n,identity in enumerate(ids):
        row=by[identity]; prepared=image_array(row,manifest['config'])
        tensors=adapter.model.feature_extractor(adapter._tensor(prepared.rgb_float32))
        f2,f3=tensors['layer2'],tensors['layer3']
        full=F.avg_pool2d(torch.cat([f2,F.interpolate(f3,size=f2.shape[-2:],mode='nearest')],1),3,1,1)[0].half().float().numpy()
        roi=F.interpolate(torch.from_numpy(prepared.valid_roi.astype(np.float32))[None,None],size=full.shape[-2:],mode='nearest')[0,0].bool().numpy()
        _,tiles=extract_tiles(adapter,native_rgb(row),p['config'])
        store=Native(full,roi,tiles); predictions={}
        active=[key for key in models if key[0]=='dev' or key[1]==p['confirmation_folds'][identity]]
        for key in active:
            state,head,uses_tiles=models[key]
            value=score_views(head,state,store,row,uses_tiles); head_evaluations+=1
            flags={}
            for alpha in p['config']['alphas']:
                threshold,reference=expected[(key,alpha)]
                prior=reference[identity]; flag=value>threshold
                max_difference=max(max_difference,abs(value-prior['score']))
                mismatches+=int(flag!=prior['flagged']); threshold_evaluations+=1
                flags[f'{alpha:.2f}']=flag
            predictions['/'.join(map(str,key))]={'score':value,'flags':flags}
        rows.append({'id':identity,'predictions':predictions})
        if n%50==0 or n+1==len(ids): print('NATIVE',fold,n+1,'/',len(ids),flush=True)
    for path,expected_hash in hashes.items(): assert sha256(path)==expected_hash
    value={'protocol_identity':p['identity'],'selection_identity':selected['identity'],'fold':fold,
           'source_sha256':sha256(Path(__file__)),'originals':len(rows),
           'head_evaluations':head_evaluations,'threshold_evaluations':threshold_evaluations,
           'max_absolute_score_difference':max_difference,'decision_mismatches':mismatches,
           'all_scores_exact':max_difference==0,'all_scores_within_1e_5':max_difference<=1e-5,
           'model_hashes_unchanged':hashes,'cache_arrays_read':False,'masks_read':False,'official_test_accessed':False,
           'scope':'previously observed training originals; replay checks implementation consistency, not new accuracy evidence'}
    write_json(root/f'native-f{fold}.json',value)
    write_json(root/f'native-f{fold}-predictions.json',{'protocol_identity':p['identity'],'rows':rows})
    if mismatches or max_difference>1e-5: raise ValueError('Native replay failed; preserve evidence')
    print('NATIVE PASSED',fold,max_difference,mismatches,flush=True)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--fold',type=int,choices=[0,1,2],required=True); parser.add_argument('--worker',action='store_true')
    args=parser.parse_args(); root=args.root.resolve()
    if args.worker: replay(root,args.fold); return
    cfg=read_json(root/'protocol.json')['config']['resources']
    cap=min(cfg['rss_cap_gib']*1024**3,int(cfg['available_ram_fraction']*psutil.virtual_memory().available))
    resource=run_bounded([sys.executable,str(Path(__file__).resolve()),'--root',str(root),'--fold',str(args.fold),'--worker'],
                         root/f'native-f{args.fold}-worker.log',max_rss_bytes=cap,max_seconds=cfg['phase_seconds'])
    write_json(root/f'native-f{args.fold}-resources.json',resource); print(resource,flush=True)
    if resource['status']!='completed': raise RuntimeError('Preserve native replay failure')


if __name__=='__main__': main()
