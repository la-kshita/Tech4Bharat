"""Predeclared paired warm-recipe development study; official test is ineligible."""
import argparse
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version
import math
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F

from component_inspector.data.acquire import sha256
from component_inspector.evaluation.calibration import calibrate
from component_inspector.evaluation.metrics import evaluate
from component_inspector.research.ksdd2_round1 import DefectHead, digest, image_array, read_json, write_json
from component_inspector.research.ksdd2_small_defects import FeatureStore, context as old_context, guard, partition, pool

PROJECT = Path(__file__).resolve().parents[3]
REPO = PROJECT.parent
CONFIG = PROJECT/'configs/recall_warm_small.json'
COARSE = {'grid': 'coarse', 'channels': 384, 'pool': 'top_0.1_percent'}


def context(root):
    p = read_json(root/'protocol.json'); other = dict(p); expected = other.pop('identity')
    if digest(other) != expected: raise ValueError('Protocol changed')
    for path, value in p['source_sha256'].items():
        if sha256(path) != value: raise ValueError('Frozen source changed: '+path)
    old, manifest, by = old_context(Path(p['parent']))
    if old['identity'] != p['parent_identity']: raise ValueError('Parent changed')
    if sha256(Path(p['parent'])/'verification.json') != p['parent_verification_sha256']: raise ValueError('Parent audit changed')
    if sha256(PROJECT/'requirements-recall.lock') != p['lock_sha256']: raise ValueError('Dependency lock changed')
    if {name: version(name) for name in p['versions']} != p['versions']: raise ValueError('Environment changed')
    return p, old, manifest, by


def tile_boxes(width, height, size=160, stride=128):
    if min(width, height) < size: raise ValueError('Native image smaller than registered tile')
    def starts(length):
        return sorted(set([*range(0, length-size+1, stride), length-size]))
    return [(x, y, x+size, y+size) for y in starts(height) for x in starts(width)]


def assign_folds(old, by, seed):
    strata, groups = {}, {}
    for identity in old['pool_ids']:
        row = by[identity]; group = row['group']
        groups.setdefault(group, []).append(identity)
    for group, ids in groups.items():
        labels = {by[i]['label'] for i in ids}
        if len(labels) != 1: raise ValueError('Mixed group labels')
        label = labels.pop()
        small = label and min(old['positive_source_area_fraction'][i] for i in ids) <= .005
        strata.setdefault((label, bool(small)), []).append(group)
    assignment = {}
    for key, values in strata.items():
        for n, group in enumerate(sorted(values, key=lambda g: digest([seed, list(key), g]))):
            for identity in groups[group]: assignment[identity] = n % 3
    return assignment


def freeze(root, parent):
    old, manifest, by = old_context(parent)
    cfg = read_json(CONFIG)
    assert cfg['tile_input_size']==256 and cfg['tile_size']==160 and cfg['tile_stride']==128
    assert all(c['extra_epochs'] in (0,cfg['continuation_epochs']) for c in cfg['methods'].values())
    if not read_json(parent/'verification.json')['protected_old_benchmarks_unchanged']:
        raise ValueError('Parent is not audited')
    files = [Path(__file__), CONFIG, PROJECT/'tools/run_warm_small.py',
             PROJECT/'tools/run_legacy_small_defects.py', PROJECT/'src/component_inspector/vision/resources.py',
             *map(Path, old['source_sha256'])]
    confirm = assign_folds(old, by, cfg['confirmation_partition_seed'])
    value = {'created_utc': datetime.now(timezone.utc).isoformat(),
             'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
             'source_sha256': {str(path.resolve()): sha256(path) for path in files},
             'parent': str(parent), 'parent_identity': old['identity'], 'config': cfg,
             'parent_verification_sha256': sha256(parent/'verification.json'),
             'versions': {name: version(name) for name in old['versions']},
             'lock_sha256': sha256(PROJECT/'requirements-recall.lock'),
             'development_folds': old['folds'], 'confirmation_folds': confirm,
             'pool_ids': old['pool_ids'], 'reserved_ids': old['reserved_normal_calibration_ids'],
             'areas': old['positive_source_area_fraction'],
             'claim': 'development on previously observed official training images; second partition is a robustness check, not unseen data or independent confirmation',
             'baseline_reuse': {str(f): {'path': str(parent/f'legacy-outer-{f}/model.pt'),
                               'sha256': sha256(parent/f'legacy-outer-{f}/model.pt'),
                               'training_sha256': sha256(parent/f'legacy-outer-{f}/training.json'),
                               'plan_sha256': sha256(parent/f'legacy-outer-{f}/plan.json')}
                               for f in range(3)},
             'no_official_test_access': True}
    for assignment in (value['development_folds'], confirm):
        assert set(assignment) == set(value['pool_ids'])
        for fold in range(3):
            ids = [i for i in assignment if assignment[i] == fold]
            assert len(ids) == 638 and sum(by[i]['label'] for i in ids) == 82
            assert sum(value['areas'].get(i, 1.) <= .005 for i in ids) == 15
    root.mkdir(parents=True, exist_ok=False)
    value['identity'] = digest(value); write_json(root/'protocol.json', value)
    print('FROZEN', value['identity'], flush=True)


def roles(p, old, by, stage, fold):
    source = dict(old)
    source['folds'] = p['development_folds'] if stage == 'dev' else p['confirmation_folds']
    source['config'] = dict(old['config'])
    if stage == 'confirm': source['config']['seed'] = p['config']['confirmation_partition_seed']
    salt = f'outer-{fold}' if stage == 'dev' else f'confirm-{fold}'
    fitting, calibration = partition(source, by, [f for f in range(3) if f != fold], salt=salt)
    excluded = {i for i, value in source['folds'].items() if value == fold}
    guard(fitting+calibration, forbidden_ids=excluded | set(p['reserved_ids']))
    normals = [r for r in fitting if not r['label']][:128]
    base = sorted(normals, key=lambda r:r['id'])+sorted([r for r in fitting if r['label']], key=lambda r:r['id'])
    candidates = [by[i] for i in p['pool_ids'] if i not in excluded
                  and not by[i]['label'] and i not in {r['id'] for r in normals+calibration}]
    return base, calibration, candidates, [by[i] for i in p['pool_ids'] if i in excluded]


def native_rgb(row):
    guard([row])
    if sha256(row['path']) != row['hashes']['sha256']: raise ValueError('Image hash changed')
    with Image.open(row['path']) as image: return image.convert('RGB').transpose(Image.Transpose.ROTATE_270)


@torch.inference_mode()
def extract_tiles(adapter, rgb, cfg):
    boxes = tile_boxes(rgb.width, rgb.height, cfg['tile_size'], cfg['tile_stride'])
    values = []
    for box in boxes:
        array = np.asarray(rgb.crop(box).resize((256,256), Image.Resampling.BILINEAR), dtype=np.float32)/255.
        x = torch.from_numpy(array.transpose(2,0,1).copy())[None]
        features = adapter.model.feature_extractor((x-adapter.mean)/adapter.std)
        f2, f3 = features['layer2'], features['layer3']
        coarse = F.avg_pool2d(torch.cat([f2, F.interpolate(f3, size=f2.shape[-2:], mode='nearest')],1),3,1,1)
        values.append(coarse[0].half().numpy())
    return boxes, np.stack(values)


@torch.inference_mode()
def cache(root, fold):
    from component_inspector.vision.spike_worker import pretrained_adapter
    p, old, manifest, by = context(root)
    directory = root/f'tile-cache-{fold}'; directory.mkdir(exist_ok=False)
    adapter = pretrained_adapter('ksdd2_surface', manifest['config'], Path(old['weights']))
    entries = {}
    for n, identity in enumerate(i for i in p['pool_ids'] if p['development_folds'][i] == fold):
        row = by[identity]; rgb = native_rgb(row)
        boxes, features = extract_tiles(adapter, rgb, p['config'])
        assert len(boxes) == 10 and features.shape == (10,384,32,32) and np.isfinite(features).all()
        key = digest(identity); path = directory/f'{key}.npy'; np.save(path,features,allow_pickle=False)
        entry = {'parent_id':identity, 'boxes':boxes,'feature_path':str(path),'feature_sha256':sha256(path)}
        if row['label']:
            if sha256(row['mask_path']) != row['hashes']['mask_sha256']: raise ValueError('Mask changed')
            with Image.open(row['mask_path']) as im: mask=im.convert('L').transpose(Image.Transpose.ROTATE_270)
            targets=[]
            for box in boxes:
                target=np.asarray(mask.crop(box).resize((256,256),Image.Resampling.NEAREST))>0
                targets.append(F.adaptive_max_pool2d(torch.from_numpy(target.astype(np.float32))[None,None],(32,32))[0,0].byte().numpy())
            targets=np.stack(targets); assert targets.any()
            target_path=directory/f'{key}-target.npy'; np.save(target_path,targets,allow_pickle=False)
            entry.update(target_path=str(target_path), target_sha256=sha256(target_path))
        entries[identity]=entry
        if n%100==0: print('TILE CACHE',fold,n+1,flush=True)
    write_json(directory/'index.json',{'protocol_identity':p['identity'],'rows':entries})
    write_json(directory/'complete.json',{'index_sha256':sha256(directory/'index.json'),'count':len(entries)})


class Store:
    def __init__(self, root, p, old):
        self.full=FeatureStore(Path(p['parent']),old); self.entries={}; self.checked=set()
        for fold in range(3):
            directory=root/f'tile-cache-{fold}'
            if sha256(directory/'index.json') != read_json(directory/'complete.json')['index_sha256']:
                raise ValueError('Tile index changed')
            self.entries.update(read_json(directory/'index.json')['rows'])
        assert set(self.entries)==set(p['pool_ids'])

    def array(self, identity, kind):
        entry=self.entries[identity]; path=entry[kind+'_path']
        if path not in self.checked:
            if sha256(path)!=entry[kind+'_sha256']: raise ValueError('Tile cache changed')
            self.checked.add(path)
        return np.load(path,mmap_mode='r',allow_pickle=False)

    def view(self, row, tile=None, *, target=False):
        identity=row['id']
        if tile is None:
            feature=self.full.feature(identity,COARSE); roi=self.full.roi(identity,COARSE)
            label=self.full.target(identity,COARSE) if target and row['label'] else np.zeros(roi.shape,np.float32)
        else:
            feature=np.zeros((384,32,80),np.float32); feature[:,:,24:56]=self.array(identity,'feature')[tile]
            roi=np.zeros((32,80),bool); roi[:,24:56]=True
            label=np.zeros((32,80),np.float32)
            if target and row['label']: label[:,24:56]=self.array(identity,'target')[tile]
        return feature,roi,label


def load_head(state):
    head=DefectHead(384,32); head.load_state_dict(state['head']); head.eval(); return head


@torch.inference_mode()
def score_views(head, state, store, row, tiles):
    x,roi,_=store.view(row,None)
    raw=head.segment((torch.from_numpy(x)[None]-state['mean'])/state['std'])[:,0]
    values=[float(pool(raw,torch.from_numpy(roi)[None],COARSE['pool'])[0])]
    if tiles:
        # The head is pointwise. Batch all ten square views without artificial
        # feature padding; no labels or mask-driven view selection at inference.
        x=torch.from_numpy(np.array(store.array(row['id'],'feature'),dtype=np.float32))
        raw=head.segment((x-state['mean'])/state['std'])[:,0]
        values.extend(pool(raw,torch.ones_like(raw,dtype=torch.bool),COARSE['pool']).tolist())
    return max(values)


def fit(head, rows, store, state, *, seed, epochs, lr, aligned, cfg, method=None):
    order_generator=torch.Generator().manual_seed(seed)
    view_generator=torch.Generator().manual_seed(seed+9001)
    parameters=list(head.segment.parameters()) if aligned else list(head.parameters())
    optimizer=torch.optim.AdamW(parameters,lr=lr,weight_decay=cfg['weight_decay'])
    history=[]; start=time.perf_counter(); sampled=Counter(); usage={r['id']:Counter() for r in rows}
    for epoch in range(epochs):
        head.train(); total=0.
        for indices in torch.randperm(len(rows),generator=order_generator).split(4):
            batch=[rows[int(i)] for i in indices]; views=[]; weights=[]
            for row in batch:
                tile=None
                if method and method['tiles'] and float(torch.rand((),generator=view_generator))<cfg['tile_training_probability']:
                    available=list(range(10))
                    if row['label']:
                        targets=store.array(row['id'],'target'); available=[i for i in available if targets[i].any()]
                        if not available: raise ValueError('Positive source has no positive training crop')
                    tile=available[int(torch.randint(len(available),(1,),generator=view_generator))]
                views.append(store.view(row,tile,target=True)); sampled['full' if tile is None else 'tile']+=1
                usage[row['id']]['full' if tile is None else str(tile)]+=1
                weights.append(cfg['small_weight'] if method and method['small_weighted'] and row['label']
                               and cfg['_areas'][row['id']]<=cfg['small_max_fraction'] else 1.)
            x=torch.from_numpy(np.stack([v[0] for v in views])); roi=torch.from_numpy(np.stack([v[1] for v in views]))
            target=torch.from_numpy(np.stack([v[2] for v in views])); labels=torch.tensor([float(r['label']) for r in batch])
            weights=torch.tensor(weights)
            raw,image=head((x-state['mean'])/state['std'],roi)
            bce=F.binary_cross_entropy_with_logits(raw,target,pos_weight=torch.tensor(8.),reduction='none')
            probability=raw.sigmoid(); pt=probability*target+(1-probability)*(1-target)
            pixel=((bce*(1-pt).square()*roi).sum((1,2))*weights).sum()/((roi.sum((1,2))*weights).sum())
            if aligned: image=pool(raw,roi,COARSE['pool'])
            loss=pixel+(F.binary_cross_entropy_with_logits(image,labels,reduction='none')*weights).sum()/weights.sum()
            if not torch.isfinite(loss): raise ValueError('Nonfinite training loss')
            optimizer.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(parameters,5.); optimizer.step()
            total+=float(loss.detach())*len(batch)
        history.append({'epoch':epoch+1,'loss':total/len(rows)})
        if epoch%5==0 or epoch+1==epochs: print('TRAIN',epochs,epoch+1,history[-1]['loss'],flush=True)
    return {'history':history,'fit_seconds':time.perf_counter()-start,'view_counts':dict(sampled),
            'view_usage':{identity:dict(counts) for identity,counts in usage.items()}}


def directory(root, stage, fold, seed, method):
    return root/f'{stage}-f{fold}-s{seed}-{method}'


def evaluate_model(output, p, head, state, store, calibration, evaluation, tiles):
    cal=[{'id':r['id'],'score':score_views(head,state,store,r,tiles)} for r in calibration]
    scores={r['id']:score_views(head,state,store,r,tiles) for r in evaluation}
    for alpha in p['config']['alphas']:
        threshold=calibrate([r['score'] for r in cal],alpha=alpha)
        predictions=[{'id':r['id'],'label':r['label'],'score':scores[r['id']],
                      'flagged':scores[r['id']]>threshold['threshold'],'status':'scored'} for r in evaluation]
        small=[r for r in predictions if r['label'] and p['areas'][r['id']]<=.005]
        record={'protocol_identity':p['identity'],'threshold_record':threshold,'calibration_scores':cal,
                'predictions':predictions,'metrics':evaluate(predictions,{r['id']:r['label'] for r in evaluation}),
                'small_defects':{'detected':sum(r['flagged'] for r in small),'total':len(small)}}
        write_json(output/f'alpha-{alpha:.2f}.json',record)
        print('RESULT',output.name,alpha,record['metrics']['counts'],record['small_defects'],flush=True)


def train(root,stage,fold,seed,method):
    p,old,manifest,by=context(root); cfg=dict(p['config']); cfg['_areas']=p['areas']
    if stage not in ('dev','confirm') or fold not in (0,1,2) or method not in cfg['methods']:
        raise ValueError('Unregistered training request')
    if seed not in (cfg['development_seeds'] if stage=='dev' else [cfg['confirmation_seed']]):
        raise ValueError('Unregistered seed')
    if stage=='confirm':
        selected=load_selection(root,p); assert selected['method']==method or method=='baseline84'
    base,calibration,candidates,evaluation=roles(p,old,by,stage,fold)
    output=directory(root,stage,fold,seed,method); output.mkdir(exist_ok=False)
    store=Store(root,p,old)
    plan={'stage':stage,'fold':fold,'seed':seed,'method':method,'protocol_identity':p['identity'],
          'base_ids':[r['id'] for r in base],'calibration_ids':[r['id'] for r in calibration],
          'evaluation_ids':[r['id'] for r in evaluation],'no_test_access':True}
    write_json(output/'plan.json',plan)
    torch.manual_seed(seed)
    if method=='baseline84':
        reused=stage=='dev' and seed==4217
        if reused:
            source=p['baseline_reuse'][str(fold)]; assert sha256(source['path'])==source['sha256']
            assert sha256(Path(source['path']).parent/'training.json')==source['training_sha256']
            assert sha256(Path(source['path']).parent/'plan.json')==source['plan_sha256']
            parent_plan=read_json(Path(source['path']).parent/'plan.json')
            assert parent_plan['base_fitting_ids']==plan['base_ids'] and parent_plan['calibration_ids']==plan['calibration_ids']
            shutil.copyfile(source['path'],output/'model.pt')
            prior=read_json(Path(source['path']).parent/'training.json')
            training={'reused':True,'source':source,'fitting_ids':prior['fitting_ids'],'hard_ids':prior['hard_normal_ids'],
                      'normalization_ids':prior['normalization_ids'],'history_source':str(Path(source['path']).parent/'training.json')}
        else:
            sums,squares,count=torch.zeros(384),torch.zeros(384),0
            for row in base:
                x=torch.from_numpy(store.full.feature(row['id'],COARSE)); sums+=x.sum((1,2)); squares+=x.square().sum((1,2)); count+=x.shape[1]*x.shape[2]
            state={'mean':(sums/count).reshape(1,-1,1,1),
                   'std':(squares/count-(sums/count).square()).clamp_min(.01).sqrt().reshape(1,-1,1,1)}
            head=DefectHead(384,32)
            base_fit=fit(head,base,store,state,seed=seed,epochs=cfg['base_epochs'],lr=.001,aligned=False,cfg=cfg)
            torch.save({'head':head.state_dict(),**state},output/'base60.pt')
            ranked=sorted([(score_views(head.eval(),state,store,row,False),row) for row in candidates],key=lambda pair:(-pair[0],pair[1]['id']))
            hard=[row for _,row in ranked[:128]]
            guard(hard,forbidden_ids={r['id'] for r in base+calibration+evaluation}|set(p['reserved_ids']))
            for row in hard:
                if image_array(row,manifest['config'],mask=True).any(): raise ValueError('Mined normal has positive mask')
            rows=base+hard
            hard_fit=fit(head,rows,store,state,seed=seed,epochs=cfg['hard_epochs'],lr=.0002,aligned=False,cfg=cfg)
            head.classify.requires_grad_(False)
            aligned_fit=fit(head,rows,store,state,seed=seed,epochs=cfg['aligned_epochs'],lr=.0002,aligned=True,cfg=cfg)
            torch.save({'head':head.state_dict(),**state},output/'model.pt')
            training={'reused':False,'base':base_fit,'hard':hard_fit,'aligned':aligned_fit,
                      'fitting_ids':[r['id'] for r in rows],'hard_ids':[r['id'] for r in hard],
                      'normalization_ids':[r['id'] for r in base],
                      'mining_scores':[{'id':row['id'],'score':value} for value,row in ranked]}
    else:
        parent=directory(root,stage,fold,seed,'baseline84'); prior=read_json(parent/'training.json')
        assert sha256(parent/'model.pt')==prior['model_sha256']
        state=torch.load(parent/'model.pt',weights_only=True); head=load_head(state); head.classify.requires_grad_(False)
        rows=[by[i] for i in prior['fitting_ids']]
        guard(rows,forbidden_ids=set(plan['evaluation_ids']+plan['calibration_ids']+p['reserved_ids']))
        continuation=fit(head,rows,store,state,seed=seed+9000,epochs=cfg['methods'][method]['extra_epochs'],
                         lr=cfg['continuation_lr'],aligned=True,cfg=cfg,method=cfg['methods'][method])
        torch.save({'head':head.state_dict(),'mean':state['mean'],'std':state['std']},output/'model.pt')
        training={'parent_model_sha256':sha256(parent/'model.pt'),'fitting_ids':prior['fitting_ids'],
                  'normalization_ids':prior['normalization_ids'],'continuation':continuation}
    training.update(seed=seed,method=method,model_sha256=sha256(output/'model.pt'))
    write_json(output/'training.json',training)
    state=torch.load(output/'model.pt',weights_only=True); head=load_head(state)
    evaluate_model(output,p,head,state,store,calibration,evaluation,cfg['methods'][method]['tiles'])
    if stage=='dev' and seed==4217 and method=='baseline84':
        original=read_json(Path(p['baseline_reuse'][str(fold)]['path']).parent/f'eval-{fold}-alpha0.08.json')
        replay=read_json(output/'alpha-0.08.json')
        reference={row['id']:row for row in original['predictions']}
        assert all(row['score']==reference[row['id']]['score'] and row['flagged']==reference[row['id']]['flagged'] for row in replay['predictions'])
        assert replay['calibration_scores']==original['calibration_scores']
    write_json(output/'complete.json',{'protocol_identity':p['identity'],'model_sha256':sha256(output/'model.pt')})


def choose(candidates,methods):
    def key(name):
        method,alpha=name; rows=candidates[name]
        fpr=[r['metrics']['accepted_normal_fpr']['rate'] for r in rows]
        recall=[r['metrics']['recall']['rate'] for r in rows]
        small=[r['small_defects']['detected']/r['small_defects']['total'] for r in rows]
        cfg=methods[method]; worst=max(fpr)
        return (worst>=.1,worst if worst>=.1 else 0.,-min(recall),-min(small),-sum(recall)/len(rows),
                sum(fpr)/len(rows),cfg['extra_epochs'],cfg['tiles'],cfg['small_weighted'],method,alpha)
    return min(candidates,key=key)


def select(root):
    p,_,_,_=context(root)
    if any(root.glob('confirm-*/alpha-*.json')): raise ValueError('Select before second-partition results')
    candidates={}; inputs={}
    for method in p['config']['methods']:
        for alpha in p['config']['alphas']:
            rows=[]; files=[]
            for seed in p['config']['development_seeds']:
                for fold in range(3):
                    path=directory(root,'dev',fold,seed,method)/f'alpha-{alpha:.2f}.json'
                    rows.append(read_json(path)); files.append({'path':str(path),'sha256':sha256(path)})
            candidates[(method,alpha)]=rows; inputs[f'{method}/{alpha}']=files
    method,alpha=choose(candidates,p['config']['methods'])
    selected=candidates[(method,alpha)]
    value={'protocol_identity':p['identity'],'method':method,'alpha':alpha,'input_files':inputs,
           'all_nine_development_goals_met':all(r['metrics']['recall']['rate']>=.95 and r['metrics']['accepted_normal_fpr']['rate']<.1 for r in selected),
           'selected_before_second_partition_inference':True,'no_seed_selection':True,
           'claim':'development selection; previously observed images; not an unbiased or independent test result'}
    value['identity']=digest(value); write_json(root/'selection.json',value)
    print('SELECTED',method,alpha,'all nine:',value['all_nine_development_goals_met'],flush=True)


def load_selection(root,p):
    selected=read_json(root/'selection.json'); value=dict(selected); expected=value.pop('identity')
    if digest(value)!=expected or selected['protocol_identity']!=p['identity']: raise ValueError('Selection changed')
    for files in selected['input_files'].values():
        for record in files:
            if sha256(record['path'])!=record['sha256']: raise ValueError('Selection input changed')
    return selected


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--phase',choices=['freeze','cache','train','select'],required=True)
    parser.add_argument('--parent',type=Path,default=Path('components/outputs/recall-small-defects'))
    parser.add_argument('--fold',type=int,choices=[0,1,2]); parser.add_argument('--seed',type=int)
    parser.add_argument('--stage',choices=['dev','confirm']); parser.add_argument('--method')
    args=parser.parse_args(); torch.set_num_threads(4); torch.set_num_interop_threads(1)
    root=args.root.resolve()
    if args.phase=='freeze': freeze(root,args.parent.resolve())
    elif args.phase=='cache': cache(root,args.fold)
    elif args.phase=='select': select(root)
    else: train(root,args.stage,args.fold,args.seed,args.method)


if __name__=='__main__': main()
