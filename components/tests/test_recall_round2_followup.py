"""Check selection and calibration boundaries of the finite follow-up."""
import json

import numpy as np

from component_inspector.research import ksdd2_round2_followup as follow


def test_selection_rejects_ten_percent_and_never_selects_repeat_seed(tmp_path):
    for name, recall, fpr in [('aggregation_a', .96, .065), ('aggregation_b', .99, .10),
                              ('aggregation_c', .96, .07), ('stability_seed4218', 1., 0.)]:
        directory = tmp_path/name
        directory.mkdir()
        (directory/'result.json').write_text(json.dumps({
            'metrics': {'recall': {'rate': recall}, 'accepted_normal_fpr': {'rate': fpr}},
            'details': {'rule': 'maximum'}}))
    name, selected = follow.best(tmp_path)
    assert name == 'aggregation_a'
    assert selected['recipe'] == {'model': 'round1/supervised', 'rule': 'maximum', 'fusion': None}


def test_fusion_keeps_normalization_and_threshold_membership_disjoint(monkeypatch, tmp_path):
    calibration = [{'id': f'n{i}', 'v2_role': 'normal_calibration'} for i in range(24)]
    manifest = {'records': calibration}
    config = {'seed': 4217}
    first = {'calibration_scores': [{'id': r['id'], 'score': i/10} for i, r in enumerate(calibration)],
             'predictions': [{'id': 'eval', 'score': 3.}]}
    second = {'calibration_scores': [{'id': r['id'], 'score': i/20} for i, r in enumerate(calibration)],
              'predictions': [{'id': 'eval', 'score': 1.5}]}
    monkeypatch.setattr(follow, 'context', lambda root: (tmp_path, manifest, config))
    monkeypatch.setattr(follow, 'read_json', lambda path: second)
    monkeypatch.setattr(follow, 'result', lambda root, name, m, scores, cal, details:
                        {'scores': scores, 'cal': cal, 'details': details})
    out = follow.fuse(tmp_path, first, {'model': 'focal', 'rule': 'image'}, 'check', 'maximum_normal_cdf')
    details = out['details']
    assert set(details['normalization_ids']).isdisjoint(details['threshold_ids'])
    assert set(out['cal']) == set(details['threshold_ids'])
    assert len(details['normalization_ids']) == len(details['threshold_ids']) == 12
    assert len(details['normalizers'][0]['cdf_reference']) == 12
    assert np.isfinite(out['scores']['eval'])
