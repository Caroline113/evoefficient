"""固定 DeepSWE 数据划分，默认按仓库隔离。"""
import argparse
import random
from pathlib import Path
from .common import *
from examples.tb_examples.emergent_efficiency_train_eval.splits import (
    PROTOCOL, audit, partition as task_partition,
)

def repo_of(task):
    repo = task['metadata'].get('repository_url')
    if not repo:
        raise ValueError(f"Missing repository_url: {task['id']}")
    return repo.lower().rstrip('/').removesuffix('.git')

def partition(tasks, train_count=10, test_count=75, seed=42, unit='repository'):
    if unit == 'task':
        return task_partition(tasks, train_count, test_count, seed)
    if train_count < 1 or test_count < 1 or train_count + test_count >= len(tasks):
        raise ValueError('Requested train/val/test sizes must be positive')
    groups = {}
    for t in sorted(tasks, key=lambda t: t['id']):
        groups.setdefault(repo_of(t), []).append(t)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    if len(keys) < 3:
        raise ValueError('Repository split requires at least three repositories')
    # Subset-sum finds the closest task count without cutting a repository.
    def choose(pool, target, reserve):
        states = {(0, 0): ()}
        for key in pool:
            for (n, k), subset in list(states.items()):
                if k + 1 <= len(pool) - reserve:
                    states.setdefault((n + len(groups[key]), k + 1), subset + (key,))
        valid = [(state, subset) for state, subset in states.items() if state[1] > 0]
        return min(valid, key=lambda item: (abs(item[0][0]-target), item[0][0]))[1]
    test = choose(keys, test_count, 2)
    rest = [k for k in keys if k not in test]
    train = choose(rest, train_count, 1)
    val = [k for k in rest if k not in train]
    result = {name: [t for k in ks for t in groups[k]]
              for name, ks in dict(train=train, val=val, test=test).items()}
    audit(result)
    return result

def load_split(path, allow_demo=False):
    info = read_json(path)
    if info.get('protocol') != PROTOCOL or info.get('split_unit') not in ('task','repository'):
        raise ValueError('Unsupported split')
    tasks = load_tasks(info['dataset_dir'])
    receipt = validate_receipt(info['dataset_dir'], tasks, allow_demo=allow_demo)
    if receipt != info['dataset_receipt']:
        raise ValueError('Dataset receipt changed since partitioning')
    by_id = {t['id']:t for t in tasks}
    ids = [i for k in ('train','val','test') for i in info['partitions'][k]]
    if len(ids) != len(set(ids)) or set(ids) != set(by_id):
        raise ValueError('Partitions must cover every source task exactly once')
    groups = {k:[by_id[i] for i in info['partitions'][k]] for k in ('train','val','test')}
    audit(groups)
    if info['split_unit'] == 'repository':
        repos = {k:{repo_of(t) for t in v} for k,v in groups.items()}
        if repos['train'] & repos['val'] or repos['train'] & repos['test'] or repos['val'] & repos['test']:
            raise ValueError('Repository leakage across partitions')
    return groups, info

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-dir', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--train-count', type=int, default=10)
    p.add_argument('--test-count', type=int, default=75)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--split-unit', choices=['repository','task'], default='repository')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args(argv)
    if Path(args.output).exists():
        raise ValueError('Refusing to overwrite fixed split')
    tasks = load_tasks(args.dataset_dir)
    receipt = validate_receipt(args.dataset_dir, tasks, allow_demo=args.dry_run)
    groups = partition(tasks,args.train_count,args.test_count,args.seed,args.split_unit)
    write_json(args.output, dict(protocol=PROTOCOL, dataset_dir=str(Path(args.dataset_dir).resolve()),
        dataset_receipt=receipt, split_unit=args.split_unit, seed=args.seed,
        requested_train_count=args.train_count, requested_test_count=args.test_count,
        partitions={k:[t['id'] for t in v] for k,v in groups.items()},
        grading='official_deepswe_v1.1_separate_harbor_verifier'))
    print({k:len(v) for k,v in groups.items()})
