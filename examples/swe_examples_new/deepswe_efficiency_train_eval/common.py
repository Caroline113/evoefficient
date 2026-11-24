"""DeepSWE 数据身份校验；通用冻结及哈希操作复用 TB 实验。"""
from pathlib import Path
import re
import tomllib
from examples.tb_examples.emergent_efficiency_train_eval.common import (
    read_json, write_json, sha, tree_hashes, fresh_dir, load_tasks,
    ensure_disjoint, public_identity, freeze, validate_artifact,
)

DATASET = 'datacurve/deep-swe-1-1@1.1'
SOURCE = 'https://github.com/datacurve-ai/deep-swe.git'
REVISION = '0b9fabbb63b9104d678fe965e1632f2dd9eaa2ea'
TASK_COUNT = 113

def validate_receipt(root, tasks, *, allow_demo=False):
    receipt = read_json(Path(root)/'dataset_receipt.json')
    if receipt.get('dataset') != DATASET:
        raise ValueError('Expected DeepSWE v1.1 receipt, not TB or another dataset')
    if receipt.get('synthetic'):
        if not allow_demo:
            raise ValueError('Synthetic dataset requires --dry-run')
    else:
        if (receipt.get('source') != SOURCE or receipt.get('revision') != REVISION
                or len(tasks) != TASK_COUNT):
            raise ValueError('DeepSWE receipt must match the pinned official revision and all 113 tasks')
        for task in tasks:
            cfg = tomllib.loads((Path(task['path'])/'task.toml').read_text())
            verifier = cfg.get('verifier', {})
            if verifier.get('environment_mode') != 'separate' or not verifier.get('collect'):
                raise ValueError(f"Missing v1.1 separate verifier / committed-patch collection: {task['id']}")
    if receipt.get('tasks') != {t['id']: t['content_sha256'] for t in tasks}:
        raise ValueError('Task content changed since dataset preparation')
    return receipt
