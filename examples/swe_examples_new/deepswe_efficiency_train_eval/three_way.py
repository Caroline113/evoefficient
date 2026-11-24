"""Validation-gated recursive evolution; test results never enter selection."""
from __future__ import annotations
import copy
import math
from pathlib import Path
from .common import fresh_dir, freeze, validate_artifact, public_identity, sha, write_json, read_json
from .splits import load_split, PROTOCOL

def metrics(rows, cost_metric, dry_run=False):
    if not rows:
        raise ValueError('Cannot select on empty validation results')
    if any(r.get('synthetic') for r in rows):
        if not dry_run or not all(r.get('synthetic') for r in rows):
            raise ValueError('Synthetic validation cannot select a real strategy')
        return dict(pass_rate=1., mean_cost=0., synthetic=True)
    if any(r.get('status') == 'error' or not isinstance(r.get('success'), bool) for r in rows):
        raise ValueError('Invalid validation trial; inspect infrastructure failure before selection')
    values = [r.get(cost_metric) for r in rows]
    if (cost_metric == 'cost_units' and any(r.get('cost_priced') is not True for r in rows)
            or any(v is None or not math.isfinite(v) or v < 0 for v in values)):
        raise ValueError('Validation cost unavailable; supply pricing or explicitly select --selection-cost total_tokens')
    return dict(pass_rate=sum(r['success'] is True for r in rows)/len(rows),
                mean_cost=sum(values)/len(values), synthetic=False)

def select(candidates, floor, tier):
    eligible = [c for c in candidates if c['metrics']['pass_rate'] >= floor]
    def dominates(a, b):
        a, b = a['metrics'], b['metrics']
        return (a['pass_rate'] >= b['pass_rate'] and a['mean_cost'] <= b['mean_cost']
                and (a['pass_rate'] > b['pass_rate'] or a['mean_cost'] < b['mean_cost']))
    frontier = [c for c in eligible if not any(dominates(d, c) for d in eligible)]
    if not frontier:
        raise ValueError('No candidate meets initial validation performance floor')
    ps = [c['metrics']['pass_rate'] for c in frontier]
    cs = [c['metrics']['mean_cost'] for c in frontier]
    alpha, beta = {'lite': (1, 2), 'medium': (1, 1), 'heavy': (2, 1)}[tier]
    def score(c):
        m = c['metrics']
        dp = (max(ps)-m['pass_rate'])/(max(ps)-min(ps)) if max(ps)>min(ps) else 0
        dc = (m['mean_cost']-min(cs))/(max(cs)-min(cs)) if max(cs)>min(cs) else 0
        return alpha*dp*dp + beta*dc*dc
    return min(frontier, key=score), [c['id'] for c in frontier]

def run(args):
    from . import backend
    from .run_experiment import train_stage, evaluate_stage, summarize, save_progress
    if getattr(args, 'train_dir', None) or args.eval_dir or args.eval_limit or getattr(args, 'train_limit', None):
        raise ValueError('Fixed split cannot be combined with directory overrides or task limits')
    groups, split = load_split(args.split_manifest, allow_demo=args.dry_run)
    args.dataset = split['dataset_receipt']['dataset']
    args.protocol = PROTOCOL
    plan = dict(protocol=PROTOCOL, dataset=args.dataset, split_sha256=sha(args.split_manifest),
                split=split, training_tasks=[public_identity(t) for t in groups['train']],
                validation_tasks=[public_identity(t) for t in groups['val']],
                evaluation_tasks=[public_identity(t) for t in groups['test']],
                training_receipt={'split_sha256': sha(args.split_manifest)},
                grading=split['grading'], parameters=vars(args).copy(),
                test_evolution_enabled=False, validation_feedback_to_evolver=False)
    if args.command == 'evaluate':
        info = validate_artifact(args.artifact)
        if info.get('protocol') != PROTOCOL or info.get('split_sha256') != plan['split_sha256']:
            raise ValueError('Final artifact must match this exact split manifest')
        if info.get('dry_run') and not args.dry_run:
            raise ValueError('Synthetic artifact cannot be evaluated as a real experiment')
        args.mode = info['mode']
        artifact = Path(args.artifact)
    if args.command == 'plan':
        if args.output:
            write_json(args.output, plan)
        print({k: len(v) for k, v in groups.items()})
        return
    out = fresh_dir(args.output)
    write_json(out/'experiment_manifest.json', plan)
    if args.command != 'evaluate':
        all_train, all_ledger, all_val = [], [], []
        seed_args = copy.copy(args)
        seed_args.mode = 'static-control'
        seed_out = fresh_dir(out/'seed')
        parent = train_stage(seed_args, groups['train'], plan, seed_out)
        history = []
        def validate(path, name, epoch):
            validate_artifact(path)
            dest = fresh_dir(out/'val'/f'epoch_{epoch:03d}'/name)
            rows = backend.batch_solve(groups['val'], Path(path)/'strategies', dest, epoch, args, 'val')
            validate_artifact(path)
            save_progress(dest, rows, [])
            all_val.extend(rows)
            save_progress(out/'val', all_val, [])
            return dict(id=name, artifact=str(path), metrics=metrics(rows, args.selection_cost, args.dry_run))
        initial = validate(parent, 'seed', 0)
        floor = initial['metrics']['pass_rate']
        selected = initial
        for epoch in range(1, args.epochs+1) if args.mode != 'static-control' else []:
            candidates = [validate(parent, 'parent', epoch)]
            for number in range(1, args.candidates_per_epoch+1):
                name = f'epoch_{epoch:03d}_candidate_{number:03d}'
                candidate_out = fresh_dir(out/'candidates'/name)
                candidate_args = copy.copy(args)
                candidate_args.epochs = 1
                candidate_args.seed_workspace = str(Path(parent)/'strategies')
                candidate = train_stage(candidate_args, groups['train'], plan, candidate_out)
                all_train.extend(read_json(candidate_out/'train/results.json'))
                all_ledger.extend(read_json(candidate_out/'train/usage_ledger.json'))
                save_progress(out/'train', all_train, all_ledger)
                candidates.append(validate(candidate, name, epoch))
            if any(c['metrics']['pass_rate'] >= floor for c in candidates):
                selected, frontier = select(candidates, floor, args.selection_tier)
            else:
                selected, frontier = dict(selected), []
                selected['fallback'] = 'retain_previous_selected_no_eligible_candidate'
            parent = Path(selected['artifact'])
            history.append(dict(epoch=epoch, candidates=candidates, frontier=frontier, selected=selected))
            write_json(out/'selection_history.json', history)
        save_progress(out/'train', all_train, all_ledger)
        artifact = freeze(Path(parent)/'strategies', out/'artifact',
                          dict(protocol=PROTOCOL, split_sha256=plan['split_sha256'],
                               dataset=args.dataset, mode=args.mode, dry_run=args.dry_run,
                               training_tasks=plan['training_tasks'], validation_tasks=plan['validation_tasks'],
                               training_receipt=plan['training_receipt'], training_summary=summarize(all_train, all_ledger),
                               validation_summary=summarize(all_val, []), selection=selected,
                               selection_tier=args.selection_tier, selection_cost=args.selection_cost,
                               grading=plan['grading']))
    if args.command != 'train':
        evaluate_stage(args, groups['test'], artifact, out, phase='test')
        summary = read_json(out/'pipeline_summary.json')
        summary['validation'] = validate_artifact(artifact)['validation_summary']
        components = [summary['training'], summary['validation'], summary['evaluation']]
        summary['total_cost'] = (sum(c['solver_cost'] for c in components)
                                 if all(c['solver_cost'] is not None and not c['evolution_calls_recorded'] for c in components) else None)
        summary['protocol'] = PROTOCOL
        write_json(out/'pipeline_summary.json', summary)
    print(dict(output=str(out), artifact=str(artifact), dry_run=args.dry_run))
