import pytest
from examples.tb_examples.emergent_efficiency_train_eval.common import (
    CPU_DATASET, DATASET, SOURCE, requires_accelerator, validate_receipt, write_json,
)

@pytest.mark.parametrize('config,expected', [
    ({'environment': {'gpus': 0}}, False),
    ({'environment': {'gpus': 1}}, True),
    ({'verifier': {'environment': {'gpus': 1}}}, True),
    ({'environment': {'tpu': {'count': 1}}}, True),
])
def test_accelerator_requirements(config, expected):
    assert requires_accelerator(config) is expected

def receipt(root, dataset, count):
    tasks = [{'id': f'task-{i}', 'content_sha256': f'hash-{i}', 'requires_accelerator': False} for i in range(count)]
    write_json(root/'dataset_receipt.json', {'dataset': dataset, 'source': SOURCE,
        'tag': 'v4.1.0', 'commit': 'a'*40,
        'tasks': {t['id']: t['content_sha256'] for t in tasks}})
    return tasks

@pytest.mark.parametrize('dataset,count', [(CPU_DATASET, 63), (DATASET, 66)])
def test_pinned_dataset_sizes(tmp_path, dataset, count):
    tasks = receipt(tmp_path, dataset, count)
    assert validate_receipt(tmp_path, tasks)['dataset'] == dataset
    with pytest.raises(ValueError, match='pinned upstream'):
        validate_receipt(tmp_path, tasks[:-1])

def test_cpu_refuses_accelerator_and_changed_content(tmp_path):
    tasks = receipt(tmp_path, CPU_DATASET, 63)
    tasks[0]['requires_accelerator'] = True
    with pytest.raises(ValueError, match='accelerator'):
        validate_receipt(tmp_path, tasks)
    tasks[0]['requires_accelerator'] = False
    tasks[0]['content_sha256'] = 'changed'
    with pytest.raises(ValueError, match='content'):
        validate_receipt(tmp_path, tasks)

@pytest.mark.parametrize('cpu', [False, True])
def test_prepare_from_pinned_source(tmp_path, monkeypatch, cpu):
    from examples.tb_examples.emergent_efficiency_train_eval import prepare_dataset as prep
    from examples.tb_examples.emergent_efficiency_train_eval.common import load_tasks, read_json
    src = tmp_path/'upstream'
    for i in range(66):
        task = src/'tasks'/f'task-{i:02d}'
        (task/'tests').mkdir(parents=True)
        (task/'instruction.md').write_text(f'Independent evaluation instruction {i}')
        (task/'task.toml').write_text(f'[environment]\ngpus = {int(i >= 63)}\n')
        (task/'tests/test.sh').write_text('exit 0\n')
    def git(args, **kwargs):
        return '' if args[1] == 'status' else SOURCE+'\n' if args[1] == 'remote' else 'a'*40+'\n'
    monkeypatch.setattr(prep.subprocess, 'check_output', git)
    output = tmp_path/'prepared'
    argv = ['benchmark', '--source-dir', str(src), '--output', str(output)]
    if cpu: argv.append('--cpu-only')
    prep.main(argv)
    tasks_root = output/('cpu-tasks' if cpu else 'tasks')
    tasks = load_tasks(tasks_root)
    assert len(tasks) == (63 if cpu else 66)
    report = validate_receipt(tasks_root, tasks)
    assert report['dataset'] == (CPU_DATASET if cpu else DATASET)
    assert len(report['excluded_accelerator_tasks']) == (3 if cpu else 0)
