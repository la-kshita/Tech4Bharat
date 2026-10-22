"""Raw-RGB replay of every newly fitted member and matched baseline."""
import argparse
from pathlib import Path
import sys

import numpy as np
import psutil
import torch
from torch.nn import functional as F

from component_inspector.research.ksdd2_seed_ensemble import context,read,write,sha
from component_inspector.research.ksdd2_warm_small import context as warm_context,extract_tiles,load_head,native_rgb,score_views
from component_inspector.research.ksdd2_round1 import image_array
from component_inspector.vision.resources import run_bounded
from component_inspector.vision.spike_worker import pretrained_adapter


class Native:
    def __init__(self,full,roi,tiles): self.full,self.roi,self.tiles=full,roi,tiles
    def view(self,row,tile=None,target=False):
        if tile is not None or target: raise ValueError('No GT or training view in native replay')
        return self.full,self.roi,None
    def array(self,identity,kind):
        if kind!='feature' or self.tiles is None: raise ValueError('No target arrays in native replay')
        return self.tiles


@torch.inference_mode()
def replay(root,shard):
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    p,main=context(root); parent,old,manifest,by=warm_context(main)
    method=read(root/'selection.json')['method']; methods=list(dict.fromkeys(['baseline84',method]))
    uses_tiles=parent['config']['methods'][method]['tiles']; models={}; references={}; hashes={}
    for seed,metadata in p['members'].items():
        for fold in range(3):
            for candidate in methods:
                directory=Path(metadata['root'])/f'confirm-f{fold}-s{seed}-{candidate}'
                state=torch.load(directory/'model.pt',weights_only=True); key=(int(seed),fold,candidate)
                hashes[str(directory/'model.pt')]=sha(directory/'model.pt')
                assert hashes[str(directory/'model.pt')]==read(directory/'training.json')['model_sha256']
                models[key]=(state,load_head(state),parent['config']['methods'][candidate]['tiles'])
                for alpha in (.05,.08):
                    record=read(directory/f'alpha-{alpha:.2f}.json')
                    references[(key,alpha)]=(record['threshold_record']['threshold'],{r['id']:r for r in record['predictions']})
    adapter=pretrained_adapter('ksdd2_surface',manifest['config'],Path(old['weights']))
    ids=[i for i in reversed(parent['pool_ids']) if parent['development_folds'][i]==shard]
    rows=[]; max_difference=0.; mismatches=0; heads=0; decisions=0
    for n,identity in enumerate(ids):
        row=by[identity]; prepared=image_array(row,manifest['config'])
        features=adapter.model.feature_extractor(adapter._tensor(prepared.rgb_float32))
        full=F.avg_pool2d(torch.cat([features['layer2'],F.interpolate(features['layer3'],size=features['layer2'].shape[-2:],mode='nearest')],1),3,1,1)[0].half().float().numpy()
        roi=F.interpolate(torch.from_numpy(prepared.valid_roi.astype(np.float32))[None,None],size=full.shape[-2:],mode='nearest')[0,0].bool().numpy()
        tiles=extract_tiles(adapter,native_rgb(row),parent['config'])[1] if uses_tiles else None
        store=Native(full,roi,tiles); predictions={}; fold=parent['confirmation_folds'][identity]
        for key,(state,head,tiled) in models.items():
            if key[1]!=fold: continue
            score=score_views(head,state,store,row,tiled); heads+=1; flags={}
            for alpha in (.05,.08):
                threshold,expected=references[(key,alpha)]; before=expected[identity]
                max_difference=max(max_difference,abs(score-before['score']))
                flags[f'{alpha:.2f}']=score>threshold; mismatches+=int(flags[f'{alpha:.2f}']!=before['flagged']); decisions+=1
            predictions[f'{key[0]}/{key[2]}']={'score':score,'flags':flags}
        rows.append({'id':identity,'predictions':predictions})
        if n%100==0 or n+1==len(ids): print('NEW NATIVE',shard,n+1,'/',len(ids),flush=True)
    for path,value in hashes.items(): assert sha(path)==value
    result={'protocol_identity':p['identity'],'source_sha256':sha(Path(__file__)),'originals':len(rows),
            'head_evaluations':heads,'threshold_evaluations':decisions,'max_absolute_score_difference':max_difference,
            'all_scores_exact':max_difference==0,'all_scores_within_1e_5':max_difference<=1e-5,'decision_mismatches':mismatches,
            'model_hashes_unchanged':hashes,'cache_arrays_read':False,'masks_read':False,'official_test_accessed':False,
            'scope':'Repeated implementation check on observed training originals; not additional independent images'}
    write(root/f'native-new-f{shard}.json',result); write(root/f'native-new-f{shard}-predictions.json',{'rows':rows})
    if mismatches or max_difference>1e-5: raise ValueError('Preserve native replay failure')
    print('NEW NATIVE PASSED',shard,flush=True)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--shard',type=int,choices=[0,1,2],required=True); parser.add_argument('--worker',action='store_true')
    args=parser.parse_args(); root=args.root.resolve()
    if args.worker: replay(root,args.shard); return
    p,_=context(root); cfg=p['config']['resources']
    cap=min(cfg['rss_cap_gib']*1024**3,int(cfg['available_ram_fraction']*psutil.virtual_memory().available))
    value=run_bounded([sys.executable,str(Path(__file__).resolve()),'--root',str(root),'--shard',str(args.shard),'--worker'],
                      root/f'native-new-f{args.shard}-worker.log',max_rss_bytes=cap,max_seconds=cfg['phase_seconds'])
    write(root/f'native-new-f{args.shard}-resources.json',value); print(value,flush=True)
    if value['status']!='completed': raise RuntimeError('Preserve failed native phase')


if __name__=='__main__': main()
