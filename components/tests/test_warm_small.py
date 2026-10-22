import numpy as np
import pytest
import torch

from component_inspector.research.ksdd2_warm_small import choose, tile_boxes, score_views, DefectHead, fit
from component_inspector.research.ksdd2_small_defects import guard, pool


def test_native_tiles_cover_edges_without_padding_or_outside_reads():
    for width,height in [(602,184),(645,229),(660,242)]:
        covered=np.zeros((height,width),bool)
        boxes=tile_boxes(width,height)
        assert len(boxes)==10
        for x0,y0,x1,y1 in boxes:
            assert 0<=x0<x1<=width and 0<=y0<y1<=height
            assert (x1-x0,y1-y0)==(160,160)
            covered[y0:y1,x0:x1]=True
        assert covered.all()
    with pytest.raises(ValueError): tile_boxes(159,640)


def row(recall,fpr,small=.8):
    return {'metrics':{'recall':{'rate':recall},'accepted_normal_fpr':{'rate':fpr}},
            'small_defects':{'detected':int(small*15),'total':15}}


def test_selector_rejects_one_bad_seed_and_values_worst_fold():
    methods={name:{'extra_epochs':24,'tiles':False,'small_weighted':False} for name in ('lucky','steady','risky')}
    candidates={('lucky',.08):[row(.99,.05)]*8+[row(.90,.05)],
                ('steady',.08):[row(.96,.09)]*9,
                ('risky',.08):[row(.99,.01)]*8+[row(.99,.10)]}
    assert choose(candidates,methods)==('steady',.08)


def test_small_tile_score_ignores_padding_and_has_positive_gradient():
    raw=torch.zeros((1,32,80),requires_grad=True)
    roi=torch.zeros_like(raw,dtype=torch.bool); roi[:,:,24:56]=True
    with torch.no_grad(): raw[:,:,0]=100
    score=pool(raw,roi,'top_0.1_percent'); assert score.item()==0.
    score.sum().backward()
    assert raw.grad[~roi].sum()==0 and raw.grad[roi].sum()==1


def test_original_guard_rejects_test_and_outer_alias_parent():
    original={'id':'parent','recipe':'ksdd2_surface','official_split':'train','label':1}
    with pytest.raises(ValueError): guard([original],forbidden_ids={'parent'})
    with pytest.raises(ValueError): guard([{**original,'official_split':'test'}])


def test_inference_uses_all_tiles_without_reading_ground_truth():
    head=DefectHead(384,32).eval()
    with torch.no_grad():
        for parameter in head.parameters(): parameter.zero_()
        head.segment[0].weight[0,0,0,0]=1
        head.segment[2].weight[0,0,0,0]=1
    class LabelBlind:
        def view(self,row,tile=None,target=False):
            assert tile is None and not target
            return np.zeros((384,32,80),np.float32),np.ones((32,80),bool),None
        def array(self,identity,kind):
            assert kind=='feature'
            features=np.zeros((10,384,32,32),np.float32)
            features[:,0]=np.arange(1,11)[:,None,None]
            return features
    state={'mean':torch.zeros(1,384,1,1),'std':torch.ones(1,384,1,1)}
    for label in (0,1):
        assert score_views(head,state,LabelBlind(),{'id':'same','label':label},True)==10.


def test_training_crops_keep_the_positive_label_true_and_record_parent_usage():
    rows=[{'id':'defect','label':1},{'id':'normal','label':0}]
    class TrainingViews:
        def array(self,identity,kind):
            assert identity=='defect' and kind=='target'
            targets=np.zeros((10,32,32),np.uint8); targets[4,10,10]=1
            return targets
        def view(self,row,tile=None,target=False):
            assert target and tile is not None
            if row['label']: assert tile==4
            roi=np.zeros((32,80),bool); roi[:,24:56]=True
            mask=np.zeros((32,80),np.float32)
            if row['label']: mask[10,34]=1
            return np.zeros((384,32,80),np.float32),roi,mask
    state={'mean':torch.zeros(1,384,1,1),'std':torch.ones(1,384,1,1)}
    cfg={'weight_decay':.0001,'tile_training_probability':1.,'small_weight':3.,'small_max_fraction':.005,'_areas':{'defect':.001}}
    result=fit(DefectHead(384,32),rows,TrainingViews(),state,seed=19,epochs=1,lr=.0002,
               aligned=True,cfg=cfg,method={'tiles':True,'small_weighted':True})
    assert np.isfinite(result['history'][0]['loss'])
    assert result['view_usage']['defect']=={'4':1}
    assert all(sum(counts.values())==1 for counts in result['view_usage'].values())
