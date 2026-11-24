import json
from pathlib import Path
import pytest
from examples.tb_examples.emergent_efficiency_train_eval.common import load_tasks, write_json, read_json, DATASET, validate_artifact
from examples.tb_examples.emergent_efficiency_train_eval.splits import main as split_main, load_split, partition, dpp_select
from examples.tb_examples.emergent_efficiency_train_eval.three_way import select, metrics
from examples.tb_examples.emergent_efficiency_train_eval.run_experiment import main
from test_protocol import task

def fixture(tmp_path):
    root = tmp_path/'data'
    for i in range(9):
        task(root, f'task{i}', f'Distinct instruction {i}')
    tasks = load_tasks(root)
    write_json(root/'dataset_receipt.json', dict(dataset=DATASET, synthetic=True,
                                               tasks={t['id']:t['content_sha256'] for t in tasks}))
    manifest = tmp_path/'split.json'
    split_main(['--dataset-dir',str(root),'--output',str(manifest),
                '--train-count','2','--test-count','4','--dry-run'])
    seed = tmp_path/'seed'
    (seed/'prompts').mkdir(parents=True)
    (seed/'prompts/system.md').write_text('seed')
    return root, manifest, seed

def test_partition_fixed_complete_and_immutable(tmp_path):
    root, manifest, _ = fixture(tmp_path)
    groups, info = load_split(manifest, True)
    assert {k:len(v) for k,v in groups.items()} == dict(train=2,val=3,test=4)
    assert {k:[t['id'] for t in v] for k,v in partition(load_tasks(root),2,4).items()} == info['partitions']
    info['partitions']['test'][0] = info['partitions']['train'][0]
    write_json(manifest, info)
    with pytest.raises(ValueError,match='exactly once'):
        load_split(manifest, True)

def test_task_tamper_rejected(tmp_path):
    root, manifest, _ = fixture(tmp_path)
    (root/'task0/instruction.md').write_text('modified')
    with pytest.raises(ValueError, match='content'):
        load_split(manifest, True)

def test_dpp_prefers_difficult_diverse():
    pool = [{'id': x} for x in 'abc']
    features = dict(a=dict(difficulty=10,embedding=[1,0]),
                    b=dict(difficulty=9,embedding=[1,0]),
                    c=dict(difficulty=8,embedding=[0,1]))
    assert [t['id'] for t in dpp_select(pool,features,2)] == ['a','c']
    with pytest.raises(ValueError,match='non-test'):
        dpp_select(pool,dict(features, test=dict(difficulty=1,embedding=[1,0])),2)

def candidate(name, rate, cost):
    return dict(id=name,metrics=dict(pass_rate=rate,mean_cost=cost))

def test_selection_floor_pareto_and_weights():
    cs = [candidate('parent',.8,10),candidate('cheap_failure',.5,1),
          candidate('balanced',.9,8),candidate('best',1.,12),candidate('dominated',.8,11)]
    chosen, frontier = select(cs,.8,'lite')
    assert chosen['id']=='balanced' and set(frontier)=={'balanced','best'}
    assert select(cs,.8,'heavy')[0]['id']=='best'
    assert select([candidate('a',1,0),candidate('b',1,0)],1,'medium')[0]['id']=='a'
    with pytest.raises(ValueError, match='cost unavailable'):
        metrics([dict(success=True,status='passed',cost_units=None,cost_priced=False)],'cost_units')
    with pytest.raises(ValueError, match='Invalid'):
        metrics([dict(success=False,status='error',total_tokens=1)],'total_tokens')

@pytest.mark.parametrize('accept',[True,False])
def test_train_val_test_order_and_parent_selection(tmp_path,monkeypatch,accept):
    _, manifest, seed = fixture(tmp_path)
    groups, _ = load_split(manifest, True)
    ids = {k:{t['id'] for t in v} for k,v in groups.items()}
    from examples.tb_examples.emergent_efficiency_train_eval import backend
    calls=[]
    def solve(tasks, ws, out, index, args, phase):
        assert {t['id'] for t in tasks} <= ids[phase]
        learned=(ws/'prompts/synthetic-training.md').exists()
        calls.append((phase,learned))
        # Passing candidates are cheaper; failing candidates must be rejected.
        return [dict(instance_id=t['id'],batch=index,success=(accept or not learned),
                     status='passed',synthetic=False,cost_priced=True,cost_units=1 if learned else 2,
                     total_tokens=10,conversation=[],steps=[]) for t in tasks]
    monkeypatch.setattr(backend,'batch_solve',solve)
    out=tmp_path/'run'
    main(['run','--split-manifest',str(manifest),'--seed-workspace',str(seed),
          '--output',str(out),'--epochs','2','--candidates-per-epoch','1','--dry-run'])
    assert calls[-1][0]=='test'
    assert all(phase!='test' for phase,_ in calls[:-4])
    info=validate_artifact(out/'artifact')
    assert (out/'artifact/strategies/prompts/synthetic-training.md').exists()==accept
    assert (out/'test/freeze_audit.json').is_file()
    history=read_json(out/'selection_history.json')
    assert len(history)==2
    # Selected parent is propagated into second epoch training.
    train_calls=[learned for phase,learned in calls if phase=='train']
    assert train_calls==[False,accept]
    assert read_json(out/'pipeline_summary.json')['validation']['tasks']==15
    with pytest.raises(ValueError,match='Synthetic'):
        main(['evaluate','--split-manifest',str(manifest),'--artifact',str(out/'artifact'),
              '--output',str(tmp_path/'real')])

def test_train_only_then_evaluate(tmp_path):
    _, manifest, seed = fixture(tmp_path)
    out=tmp_path/'train'
    main(['train','--split-manifest',str(manifest),'--seed-workspace',str(seed),
          '--output',str(out),'--dry-run'])
    assert not (out/'test').exists()
    final=tmp_path/'final'
    main(['evaluate','--split-manifest',str(manifest),'--artifact',str(out/'artifact'),
          '--output',str(final),'--dry-run'])
    assert read_json(final/'test/summary.json')['tasks']==4
    assert read_json(final/'test/summary.json')['successes'] is None
