"""One predeclared objective-alignment refinement of the best saved checkpoint."""
import argparse
from pathlib import Path
import subprocess

import torch

from component_inspector.data.acquire import sha256
from component_inspector.research import ksdd2_aligned
from component_inspector.research.ksdd2_aligned import fit
from component_inspector.research.ksdd2_round1 import write_json
from component_inspector.research.ksdd2_round2 import cache_index, context, model_load, read_json
from component_inspector.research.ksdd2_round2_followup import score_head


def prepare(root, prior):
    parent, manifest, _ = context(root)
    source = prior/'hard_refined'
    record = read_json(source/'result.json')
    assert record['manifest_identity'] == manifest['identity']
    assert sha256(source/'head.pt') == record['details']['model_sha256']
    training = read_json(source/'training.json')
    write_json(root/'warm-alignment-plan.json', {
        'source_checkpoint': str(source), 'source_sha256': sha256(source/'head.pt'),
        'source_result': 'round2/hard_refined, best saved canonical checkpoint before the correction',
        'fitting_ids': training['fitting_ids'], 'hard_normal_ids': training['hard_normal_ids'],
        'epochs': 12, 'learning_rate': .0002, 'seed': 4217, 'variant_limit': 1,
        'objective': 'same corrected fixed-pool image BCE and weighted focal pixels',
        'new_mining_or_data': False, 'evaluation_for_early_stopping': False,
        'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'worker_source_sha256': sha256(Path(__file__)),
        'fit_dependency_sha256': sha256(Path(ksdd2_aligned.__file__)),
        'claim': 'one additional observed-development refinement of a fixed checkpoint; not another from-scratch repeat seed'})


def train(root):
    parent, manifest, _ = context(root)
    plan = read_json(root/'warm-alignment-plan.json')
    source = Path(plan['source_checkpoint'])
    assert sha256(source/'head.pt') == plan['source_sha256']
    head, state, metadata = model_load(source)
    head.classify.requires_grad_(False)
    by = {r['id']: r for r in manifest['records']}
    rows = [by[i] for i in plan['fitting_ids']]
    assert len(rows) == 420 and all(r['v2_role'] in {'normal_fit', 'positive_train'} for r in rows)
    directory = root/'warm_aligned'
    directory.mkdir(exist_ok=False)
    base = parent/'features'
    training = fit(head, rows, base, cache_index(base)['rows'], state['mean'], state['std'],
                   seed=plan['seed'], epochs=plan['epochs'], learning_rate=plan['learning_rate'], label='warm_aligned')
    torch.save({'head': head.state_dict(), 'mean': state['mean'], 'std': state['std']}, directory/'head.pt')
    write_json(directory/'model.json', {**metadata, 'cache': str(base),
               'image_objective': 'fixed positive-oriented top 0.1 percent logit BCE',
               'learned_classifier_used_for_training_or_inference': False,
               'warm_alignment': True, 'source_checkpoint_sha256': plan['source_sha256']})
    write_json(directory/'training.json', {**training, 'hard_normal_ids': plan['hard_normal_ids'],
               'additional_normals': 0, 'prior_training_unchanged': True,
               'source_checkpoint_sha256': plan['source_sha256'], 'fixed_schedule_no_evaluation_stopping': True})
    score_head(root, 'warm_aligned', 'warm_aligned', strategy='one bounded aligned-objective refinement of the best saved checkpoint')
    write_json(root/'warm_aligned-complete.json', {'source_sha256': sha256(Path(__file__)),
               'head_sha256': sha256(directory/'head.pt')})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['prepare', 'train'], required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--prior', type=Path, default=Path('components/outputs/recall-round2'))
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    if args.phase == 'prepare':
        prepare(args.root.resolve(), args.prior.resolve())
    else:
        train(args.root.resolve())


if __name__ == '__main__':
    main()
