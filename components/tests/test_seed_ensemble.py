from copy import deepcopy
import math

import pytest

from component_inspector.research.ksdd2_seed_ensemble import aggregate,sigmoid


def record(value):
    return {'predictions':[{'id':'defect','label':1,'score':value,'status':'scored'},
                           {'id':'normal','label':0,'score':-value,'status':'scored'}],
            'calibration_scores':[{'id':'cal-normal','score':-value}]}


def test_three_member_mean_is_not_best_seed_selection_or_logit_averaging():
    rows,cal=aggregate([record(10.),record(-2.),record(-2.)])
    value=next(row['score'] for row in rows if row['id']=='defect')
    expected=(sigmoid(10.)+2*sigmoid(-2.))/3
    assert value==pytest.approx(expected,abs=1e-15)
    assert value<.5 and sigmoid((10-2-2)/3)>.5
    assert cal[0]['score']==pytest.approx((sigmoid(-10.)+2*sigmoid(2.))/3,abs=1e-15)


def test_sigmoid_handles_extreme_finite_logits_without_overflow():
    assert sigmoid(10000.)==1. and sigmoid(-10000.)==0. and sigmoid(0.)==.5
    for value in (math.inf,-math.inf,math.nan):
        with pytest.raises(ValueError): sigmoid(value)


@pytest.mark.parametrize('corruption',['missing','duplicate','label','reject','overlap'])
def test_members_must_have_identical_unique_disjoint_complete_originals(corruption):
    rows=[record(1.),record(2.),record(3.)]; broken=rows[2]
    if corruption=='missing': broken['predictions'].pop()
    elif corruption=='duplicate': broken['predictions'].append(deepcopy(broken['predictions'][0]))
    elif corruption=='label': broken['predictions'][0]['label']=0
    elif corruption=='reject': broken['predictions'][0]['status']='rejected'
    else:
        for row in rows: row['calibration_scores'][0]['id']='normal'
    with pytest.raises(ValueError): aggregate(rows)


def test_exactly_three_predeclared_members_and_order_invariance():
    rows=[record(1.),record(2.),record(3.)]
    with pytest.raises(ValueError): aggregate(rows[:2])
    with pytest.raises(ValueError): aggregate(rows+rows[:1])
    assert aggregate(rows)==aggregate(list(reversed(rows)))
