"""Fixed, content-verified train/validation/test partitions of a pinned dataset."""
from __future__ import annotations
import argparse
import math
import random
from pathlib import Path
from .common import load_tasks, validate_receipt, ensure_disjoint, public_identity, read_json, write_json, sha

PROTOCOL = 'train_val_select_then_frozen_test'

def dpp_select(pool, features, count):
    """Greedy MAP, HarnessPro Appendix A.1 (epsilon=.1, theta=.7)."""
    import numpy as np
    if set(features) != {t['id'] for t in pool}:
        raise ValueError('DPP features must cover exactly the non-test pool; no test features')
    ratings = np.asarray([features[t['id']]['difficulty'] for t in pool], dtype=float)
    vectors = np.asarray([features[t['id']]['embedding'] for t in pool], dtype=float)
    if (not np.isfinite(ratings).all() or (ratings < 0).any() or (ratings > 10).any()
            or vectors.ndim != 2 or not np.isfinite(vectors).all()):
        raise ValueError('Invalid difficulty scores or fingerprint embeddings')
    norms = np.linalg.norm(vectors, axis=1)
    if (norms == 0).any():
        raise ValueError('Fingerprint embeddings must be nonzero')
    vectors = vectors / norms[:, None]
    q = np.maximum(ratings / 10, .1)
    w = (q / q.max()) ** (.7 / (2 * (1 - .7)))
    kernel = w[:, None] * (vectors @ vectors.T) * w[None, :]
    chosen = []
    for _ in range(count):
        scores = []
        for i in range(len(pool)):
            if i in chosen:
                continue
            subset = chosen + [i]
            sign, value = np.linalg.slogdet(kernel[np.ix_(subset, subset)])
            scores.append((value if sign > 0 else -math.inf, -i, i))
        best = max(scores)
        if not math.isfinite(best[0]):
            raise ValueError('DPP kernel rank is too low for requested training size')
        chosen.append(best[2])
    return [pool[i] for i in chosen]

def partition(tasks, train_count=10, test_count=None, seed=42, features=None):
    test_count = len(tasks) * 2 // 3 if test_count is None else test_count
    if train_count < 1 or test_count < 1 or train_count + test_count >= len(tasks):
        raise ValueError('train, val and test must all be nonempty')
    shuffled = sorted(tasks, key=lambda t: t['id'])
    random.Random(seed).shuffle(shuffled)
    test, pool = shuffled[:test_count], shuffled[test_count:]
    train = dpp_select(pool, features, train_count) if features is not None else pool[:train_count]
    ids = {t['id'] for t in train}
    val = [t for t in pool if t['id'] not in ids]
    groups = dict(train=train, val=val, test=test)
    audit(groups)
    return groups

def audit(groups):
    for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
        if not groups[a] or not groups[b]:
            raise ValueError('Empty partition')
        ensure_disjoint(groups[a], groups[b])

def load_split(path, allow_demo=False):
    info = read_json(path)
    if info.get('protocol') != PROTOCOL:
        raise ValueError('Unsupported split protocol')
    tasks = load_tasks(info['dataset_dir'])
    receipt = validate_receipt(info['dataset_dir'], tasks, allow_demo=allow_demo)
    if receipt != info['dataset_receipt']:
        raise ValueError('Dataset receipt changed since splitting')
    by_id = {t['id']: t for t in tasks}
    ids = [i for name in ('train', 'val', 'test') for i in info['partitions'][name]]
    if len(ids) != len(set(ids)) or set(ids) != set(by_id):
        raise ValueError('Partitions must cover each source task exactly once')
    groups = {name: [by_id[i] for i in info['partitions'][name]] for name in ('train', 'val', 'test')}
    audit(groups)
    return groups, info

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-dir', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--train-count', type=int, default=10)
    p.add_argument('--test-count', type=int)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--method', choices=['random', 'dpp'], default='random')
    p.add_argument('--features', help='JSON keyed by non-test task ID: difficulty, embedding (failure fingerprint)')
    p.add_argument('--dry-run', action='store_true', help='Allow synthetic source fixtures')
    args = p.parse_args(argv)
    if (args.method == 'dpp') != bool(args.features):
        p.error('--method dpp requires --features; random must not use features')
    if Path(args.output).exists():
        raise ValueError('Split output already exists; refusing to replace fixed partition')
    tasks = load_tasks(args.dataset_dir)
    receipt = validate_receipt(args.dataset_dir, tasks, allow_demo=args.dry_run)
    groups = partition(tasks, args.train_count, args.test_count, args.seed,
                       read_json(args.features) if args.features else None)
    info = dict(protocol=PROTOCOL, dataset_dir=str(Path(args.dataset_dir).resolve()),
                dataset_receipt=receipt, method=args.method, seed=args.seed,
                features_sha256=sha(args.features) if args.features else None,
                partitions={k: [t['id'] for t in v] for k, v in groups.items()},
                tasks=[public_identity(t) for t in tasks],
                grading='existing_official_harbor_verifier_not_label_free')
    write_json(args.output, info)
    print({k: len(v) for k, v in groups.items()})

if __name__ == '__main__':
    main()
