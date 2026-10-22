"""Adversarial checks of the separate auditor without importing an ML worker."""
import ast
from collections import Counter
from copy import deepcopy
import math
from pathlib import Path

import numpy as np
import pytest


source=Path(__file__).resolve().parents[1]/'tools/verify_warm_small.py'
tree=ast.parse(source.read_text(encoding='utf-8'))
functions=ast.Module(body=[node for node in tree.body if isinstance(node,ast.FunctionDef)
                          and node.name in ('recount','paired_changes')],type_ignores=[])
namespace={'Counter':Counter,'math':math,'np':np}
exec(compile(functions,str(source),'exec'),namespace)
recount=namespace['recount']; paired_changes=namespace['paired_changes']


def example():
    ids=[f'd{i}' for i in range(82)]+[f'n{i}' for i in range(556)]
    cal=[f'c{i}' for i in range(200)]
    truth={i:int(i.startswith('d')) for i in ids+cal}
    areas={f'd{i}':.001 if i<15 else .02 for i in range(82)}
    predictions=[]
    for i in ids:
        flag=int(i[1:])>=4 if truth[i] else int(i[1:])<55
        predictions.append({'id':i,'label':truth[i],'score':float(flag),'flagged':flag,'status':'scored'})
    record={'predictions':predictions,'calibration_scores':[{'id':i,'score':0.} for i in cal],
            'threshold_record':{'threshold':0.,'alpha':.08,'n':200,'k':185,'comparison':'strict_greater'},
            'metrics':{'counts':{'TP':78,'FN':4,'FP':55,'TN':501,'rejected_anomalies':0,'rejected_normals':0},
                       'recall':{'rate':78/82},'accepted_normal_fpr':{'rate':55/556}},
            'small_defects':{'detected':11,'total':15}}
    return record,ids,truth,areas,cal


def test_recount_handles_strict_threshold_ties_and_the_point_goal_boundary():
    row,ids,truth,areas,cal=example()
    result=recount(row,ids,truth,areas,cal,.08)
    assert result['point_goal_met'] and result['large_defects']=={'detected':67,'total':67}


@pytest.mark.parametrize('forgery',['counts','flag','label','duplicate','threshold','calibration_role','small'])
def test_recount_rejects_forged_or_inconsistent_evidence(forgery):
    row,ids,truth,areas,cal=example()
    if forgery=='counts': row['metrics']['counts']['TP']=82
    elif forgery=='flag': row['predictions'][0]['flagged']=True
    elif forgery=='label': row['predictions'][0]['label']=0
    elif forgery=='duplicate': row['predictions'][0]=deepcopy(row['predictions'][1])
    elif forgery=='threshold': row['threshold_record']['threshold']=-1.
    elif forgery=='calibration_role': row['calibration_scores'][0]['id']='d0'
    else: row['small_defects']['detected']=15
    with pytest.raises(AssertionError): recount(row,ids,truth,areas,cal,.08)


def test_paired_changes_distinguish_defect_gains_from_added_false_alarms():
    truth={'small-gain':1,'small-loss':1,'large':1,'normal-gain':0,'normal-loss':0}
    areas={'small-gain':.005,'small-loss':.001,'large':.006}
    base={'predictions':[{'id':i,'flagged':i.endswith('loss')} for i in truth]}
    other={'predictions':[{'id':i,'flagged':i.endswith('gain')} for i in truth]}
    result=paired_changes(base,other,truth,areas)
    assert result['small_defects']['newly_flagged_ids']==['small-gain']
    assert result['small_defects']['no_longer_flagged_ids']==['small-loss']
    assert result['small_defects']['net_change']==0
    assert result['large_defects']['total']==1
    assert result['normals']['newly_flagged_ids']==['normal-gain']
    assert result['normals']['no_longer_flagged_ids']==['normal-loss']
