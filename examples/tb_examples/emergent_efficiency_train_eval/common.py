from __future__ import annotations
import hashlib
import json
import re
import shutil
import tomllib
from pathlib import Path

DATASET = 'terminal-bench/terminal-bench@4.1.0'
CPU_DATASET = 'terminal-bench/terminal-bench-cpu@4.1.0'
VERSION = '4.1.0'
SOURCE = 'https://github.com/harbor-framework/terminal-bench.git'
STRATEGY_PATHS = ('manifest.yaml', 'prompts', 'skills', 'tools', 'memory')

def requires_accelerator(config):
    environments = [config.get('environment', {}), config.get('verifier', {}).get('environment', {}) or {}]
    return any(e.get('gpus', 0) or e.get('gpu_types') or e.get('tpu') for e in environments)

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding='utf-8')

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()

def tree_hashes(root):
    root = Path(root)
    return {p.relative_to(root).as_posix(): sha(p) for p in sorted(root.rglob('*')) if p.is_file() and '.git' not in p.parts and '__pycache__' not in p.parts}

def fresh_dir(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()): raise ValueError(f'Refusing to overwrite nonempty output: {path}')
    path.mkdir(parents=True, exist_ok=True)
    return path

def load_tasks(root):
    root = Path(root).resolve()
    if not root.is_dir(): raise ValueError(f'Dataset directory missing: {root}')
    dirs = [root] if (root/'task.toml').is_file() else sorted(p.parent for p in root.glob('*/task.toml'))
    if not dirs: raise ValueError(f'No Harbor task.toml files found under {root}; no TB2 download fallback is used')
    tasks = []
    for p in dirs:
        if p.is_symlink(): raise ValueError(f'Symlink task directories are not supported: {p}')
        cfg = tomllib.loads((p/'task.toml').read_text())
        instruction = (p/'instruction.md').read_text(encoding='utf-8')
        if not instruction.strip() or not (p/'tests/test.sh').is_file(): raise ValueError(f'Incomplete Harbor task: {p}')
        norm = re.sub(r'\s+', ' ', instruction).strip()
        files = tree_hashes(p)
        metadata=dict(cfg.get('metadata', {}));metadata['declared_task_name']=cfg.get('task',{}).get('name')
        tasks.append({'id': p.name, 'path': str(p), 'instruction': instruction, 'instruction_sha256': hashlib.sha256(norm.encode()).hexdigest(), 'files': files, 'content_sha256': hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(), 'metadata': metadata, 'requires_accelerator': requires_accelerator(cfg), 'agent_timeout_sec': cfg.get('agent', {}).get('timeout_sec', 28800)})
    if len({t['id'] for t in tasks}) != len(tasks): raise ValueError('Duplicate task IDs')
    return tasks

def identities(task):
    meta = task.get('metadata', {})
    full={str(x).lower() for x in [task['id'], meta.get('source_task_id'), meta.get('family'), meta.get('original_task_id'),meta.get('declared_task_name')] if x}
    return full|{x.rsplit('/',1)[-1] for x in full}

def ensure_disjoint(train, evaluation):
    hits = []
    for a in train:
        for b in evaluation:
            if identities(a) & identities(b) or a['instruction_sha256'] == b['instruction_sha256'] or a['content_sha256'] == b['content_sha256']:
                hits.append([a['id'], b['id']])
    if hits: raise ValueError(f'Training/evaluation leakage detected: {hits}')
    return {'passed': True, 'checks': ['task ID and declared source/family IDs', 'normalized instruction hash', 'task content hash'], 'semantic_near_duplicates_require_manual_review': True}

def public_identity(task):
    return {k: task[k] for k in ('id', 'path', 'instruction_sha256', 'content_sha256', 'metadata')}

def validate_receipt(root, tasks, *, training=False, allow_demo=False):
    name = 'training_manifest.json' if training else 'dataset_receipt.json'
    receipt = read_json(Path(root)/name)
    if training:
        if receipt.get('independent') is not True or not receipt.get('source'): raise ValueError('Training manifest must declare independent=true and a source/provenance')
        if receipt.get('demo') and not allow_demo: raise ValueError('Demo tasks are smoke-only; use --allow-demo-training explicitly')
    else:
        if receipt.get('dataset') not in (DATASET, CPU_DATASET): raise ValueError('Evaluation must be pinned to TB4.1 full or CPU dataset')
        cpu = receipt['dataset'] == CPU_DATASET
        count = 63 if cpu else 66
        if cpu and any(t.get('requires_accelerator') for t in tasks): raise ValueError('CPU dataset contains an accelerator task')
        if receipt.get('synthetic'):
            if not allow_demo: raise ValueError('Synthetic evaluation fixtures are allowed only in --dry-run')
        elif receipt.get('source') != SOURCE or receipt.get('tag') != f'v{VERSION}' or len(tasks) != count or not re.fullmatch(r'[0-9a-f]{40}',receipt.get('commit','')):
            raise ValueError(f'Evaluation receipt needs the pinned upstream source, tag, commit and all {count} tasks')
    expected = receipt.get('tasks', {})
    actual = {t['id']: t['content_sha256'] for t in tasks}
    if expected != actual: raise ValueError(f'{name} does not match current task content; re-prepare and review the dataset')
    return receipt

def freeze(workspace, output, metadata):
    output = fresh_dir(output); dest = output/'strategies'; dest.mkdir()
    for name in STRATEGY_PATHS:
        src = Path(workspace)/name
        if src.is_symlink() or (src.is_dir() and any(p.is_symlink() for p in src.rglob('*'))):raise ValueError('Strategy symlinks cannot be frozen; keep assets self-contained')
        if src.is_dir(): shutil.copytree(src, dest/name,ignore=shutil.ignore_patterns('.git','__pycache__','.pytest_cache'))
        elif src.is_file(): shutil.copy2(src, dest/name)
    # Observation histories, tests, raw tasks and git data are never copied.
    hashes = tree_hashes(dest)
    write_json(output/'artifact.json', {'format': 1, **metadata, 'strategy_hashes': hashes})
    return output

def validate_artifact(path):
    path = Path(path); info = read_json(path/'artifact.json')
    if tree_hashes(path/'strategies') != info['strategy_hashes']: raise ValueError('Frozen strategy artifact was modified')
    return info
