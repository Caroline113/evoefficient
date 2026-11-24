import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from examples.swe_examples_new.deepswe_efficiency_train_eval import backend, common, splits
from examples.swe_examples_new.deepswe_efficiency_train_eval.run_experiment import main, parser
from examples.swe_examples_new.deepswe_efficiency_train_eval.three_way import select

def fixture(tmp_path):
    root=tmp_path/'tasks'
    for i in range(12):
        p=root/f'task{i:02d}'
        (p/'tests').mkdir(parents=True)
        (p/'tests/test.sh').write_text('exit 0')
        (p/'instruction.md').write_text(f'Unique software engineering request {i}')
        (p/'task.toml').write_text(
            'version="1.0"\n[metadata]\n'
            f'repository_url="https://github.com/example/repo{i//2}"\n'
            '[verifier]\nenvironment_mode="separate"\n'
            '[[verifier.collect]]\ncommand="git diff HEAD"\n')
    tasks=common.load_tasks(root)
    common.write_json(root/'dataset_receipt.json',
        dict(dataset=common.DATASET,synthetic=True,tasks={t['id']:t['content_sha256'] for t in tasks}))
    manifest=tmp_path/'split.json'
    splits.main(['--dataset-dir',str(root),'--output',str(manifest),
                 '--train-count','2','--test-count','4','--dry-run'])
    seed=tmp_path/'seed'
    (seed/'prompts').mkdir(parents=True)
    (seed/'prompts/system.md').write_text('Initial SWE agent')
    return root,manifest,seed

def test_repository_partition_and_integrity(tmp_path):
    root,manifest,_=fixture(tmp_path)
    groups,info=splits.load_split(manifest,True)
    repos={k:{splits.repo_of(t) for t in v} for k,v in groups.items()}
    assert not repos['train'] & repos['val']
    assert not repos['train'] & repos['test']
    assert not repos['val'] & repos['test']
    assert splits.partition(common.load_tasks(root),2,4)==groups
    info['partitions']['test'][0]=info['partitions']['train'][0]
    common.write_json(manifest,info)
    with pytest.raises(ValueError,match='exactly once'):
        splits.load_split(manifest,True)

def test_dataset_identity_and_tamper(tmp_path):
    root,manifest,_=fixture(tmp_path)
    with pytest.raises(ValueError,match='Synthetic'):
        splits.load_split(manifest,False)
    (root/'task00/instruction.md').write_text('changed')
    with pytest.raises(ValueError,match='content changed'):
        splits.load_split(manifest,True)

@pytest.mark.parametrize('mode',['static-control','normal-evolve','cost-aware'])
def test_full_protocol_and_separate_evaluation(tmp_path,monkeypatch,mode):
    _,manifest,seed=fixture(tmp_path)
    def forbidden(*a,**kw):
        raise AssertionError('dry run must never call real evolution')
    monkeypatch.setattr(backend,'evolve',forbidden)
    out=tmp_path/'trainval'
    main(['train','--split-manifest',str(manifest),'--seed-workspace',str(seed),
          '--output',str(out),'--mode',mode,'--epochs','2','--dry-run'])
    assert not (out/'test').exists()
    info=common.validate_artifact(out/'artifact')
    assert info['dataset']==common.DATASET
    assert info['dry_run']
    test=tmp_path/'test'
    main(['evaluate','--split-manifest',str(manifest),'--artifact',str(out/'artifact'),
          '--output',str(test),'--dry-run'])
    summary=common.read_json(test/'test/summary.json')
    assert summary['tasks']==4 and summary['successes'] is None
    assert common.read_json(test/'test/freeze_audit.json')['evaluation_evolution_calls']==0
    assert 'validation' in common.read_json(test/'pipeline_summary.json')

def test_parent_selection_propagates_without_test_feedback(tmp_path,monkeypatch):
    _,manifest,seed=fixture(tmp_path)
    groups,_=splits.load_split(manifest,True)
    calls=[]
    def solve(tasks,ws,out,index,args,phase):
        assert {t['id'] for t in tasks}<={t['id'] for t in groups[phase]}
        learned=(ws/'prompts/synthetic-training.md').exists()
        calls.append((phase,learned))
        return [dict(instance_id=t['id'],success=True,status='passed',synthetic=False,
                     cost_units=1 if learned else 2,cost_priced=True,total_tokens=1 if learned else 2,
                     batch=index,conversation=[],steps=[]) for t in tasks]
    monkeypatch.setattr(backend,'batch_solve',solve)
    out=tmp_path/'run'
    main(['run','--split-manifest',str(manifest),'--seed-workspace',str(seed),
          '--output',str(out),'--epochs','2','--candidates-per-epoch','1','--dry-run'])
    assert [learned for phase,learned in calls if phase=='train']==[False,True]
    first_test=next(i for i,(phase,_) in enumerate(calls) if phase=='test')
    assert all(p=='test' for p,_ in calls[first_test:])
    assert (out/'artifact/strategies/prompts/synthetic-training.md').exists()

@pytest.mark.parametrize('mode',['normal-evolve','cost-aware'])
def test_swe_evolution_bridge_mutates_and_accounts(tmp_path,monkeypatch,mode):
    from agent_evolve.algorithms.guided_synth.engine import GuidedSynthesisEngine
    from agent_evolve.algorithms.adaptive_skill import tools as adaptive_tools
    from agent_evolve.llm import factory
    from examples.swe_examples_new.emergent_efficiency import run_emergent_efficiency as swe
    from agent_evolve.llm.base import LLMResponse
    root,manifest,seed=fixture(tmp_path)
    args=parser().parse_args(['train','--split-manifest',str(manifest),
                              '--output',str(tmp_path/'unused'),'--mode',mode])
    tasks=common.load_tasks(root)
    row=dict(instance_id=tasks[0]['id'],task_instruction=tasks[0]['instruction'],
             task_metadata=tasks[0]['metadata'],success=True,score=1,conversation=[{'role':'assistant','content':'I inspected and fixed it'}],
             steps=[dict(tool='bash',input='pytest',output='passed')],cost_units=1,total_tokens=100,
             cumulative_input_tokens=80,cumulative_output_tokens=20,num_tool_calls=1)
    class Fake:
        def complete(self,messages,**kw):
            text='\n'.join(m.content for m in messages)
            if 'INTERVENTION CURATOR' in text:
                content='ACCEPT: targeted_verification'
            elif 'TRAINING EVIDENCE' in text:
                content='CONFIDENCE: HIGH\nACTION: NEW\nTYPE: skill\nNAME: targeted_verification\nDESCRIPTION: TRIGGER for code fixes; DO NOT TRIGGER before understanding.\nCONTENT:\nIdentify the affected behavior and locate its existing regression tests.\nRun a focused test before editing, implement the smallest change, and rerun the test. Check nearby edge cases and review the diff for unrelated edits before committing.'
            else:
                content=json.dumps(dict(proposals=[dict(name='focused_search',kind='prompt_fragment',
                    content='Search for the relevant implementation before reading whole files.',
                    confidence='HIGH',description='Focused search')]))
            return LLMResponse(content=content,usage={'input_tokens':10,'output_tokens':10})
    monkeypatch.setattr(factory,'create_llm_provider',lambda *a,**kw:Fake())
    monkeypatch.setattr(adaptive_tools,'create_default_llm',lambda *a,**kw:Fake())
    # Exercise the actual curator's parsing, decisions and filesystem mutation;
    # skip only git versioning, unrelated to this bridge.
    monkeypatch.setattr(GuidedSynthesisEngine,'evolve',
        lambda self,workspace,observation_logs,evo_number:self.step(workspace,observation_logs,None,None).metadata)
    old=swe._load_proposer_llm
    ledger=[]
    out=tmp_path/'evolution'
    backend.evolve([row],seed,out,1,args,ledger)
    assert (seed/'skills/targeted-verification/SKILL.md').exists()
    components={x['component'] for x in ledger}
    assert {'training_skill_reflection','ordinary_swe_curation'}<=components
    assert ('swe_efficiency_proposer' in components)==(mode=='cost-aware')
    assert swe._load_proposer_llm is old
    if mode=='cost-aware':
        assert common.read_json(out/'emergence/proposal_adoption.json')[0]['status']=='applied'
    assert all(x['cost'] is None for x in ledger)

def test_atif_and_committed_patch_prompt(tmp_path):
    trial=tmp_path/'trial'
    (trial/'agent').mkdir(parents=True)
    common.write_json(trial/'agent/trajectory.json',dict(steps=[
        dict(source='agent',message='inspect',tool_calls=[dict(function=dict(name='bash',arguments={'command':'git status'}))],
             observation={'results':[{'content':'clean'}]})]))
    legacy=SimpleNamespace(_load_trajectory=lambda _:([],[],''))
    steps,conversation,path=backend.load_training_trajectory(trial,legacy)
    assert steps[0]['tool']=='bash'
    assert 'git status' in conversation[0]['content']
    assert path.endswith('trajectory.json')
    prompt=backend.render_prompt(tmp_path)
    assert 'COMMIT' in prompt and 'separate clean verifier' in prompt

def test_harbor_config_does_not_override_official_verifier(tmp_path):
    prompt=tmp_path/'prompt'
    prompt.write_text('prompt')
    args=SimpleNamespace(harbor_model='model',delegate_agent='codex',delegate_kwargs='{}')
    cfg=backend.config_for({'path':str(tmp_path/'task')},tmp_path,prompt,tmp_path/'jobs','job',args)
    assert cfg['tasks']==[{'path':str((tmp_path/'task').resolve())}]
    assert 'verifier' not in cfg and 'override_timeout_sec' not in cfg['agents'][0]

def test_synthetic_artifact_cannot_bypass_with_real_source(tmp_path):
    _,manifest,seed=fixture(tmp_path)
    out=tmp_path/'train'
    main(['train','--split-manifest',str(manifest),'--seed-workspace',str(seed),
          '--output',str(out),'--mode','static-control','--dry-run'])
    changed=common.read_json(manifest)
    changed['seed']=999
    other=tmp_path/'other.json'
    common.write_json(other,changed)
    with pytest.raises(ValueError,match='exact split'):
        main(['evaluate','--split-manifest',str(other),'--artifact',str(out/'artifact'),
              '--output',str(tmp_path/'eval'),'--dry-run'])
