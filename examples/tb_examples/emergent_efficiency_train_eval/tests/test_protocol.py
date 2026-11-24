import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from examples.tb_examples.emergent_efficiency_train_eval.common import load_tasks,write_json,freeze,validate_artifact,ensure_disjoint,DATASET
from examples.tb_examples.emergent_efficiency_train_eval.traces import parse_session,prefix
from examples.tb_examples.emergent_efficiency_train_eval.analyze_prefix_costs import compute
from examples.tb_examples.emergent_efficiency_train_eval.run_experiment import main,summarize
from examples.tb_examples.emergent_efficiency_train_eval.backend import AuditProvider,config_for

def task(root,name,text):
    d=root/name;(d/'tests').mkdir(parents=True);(d/'environment').mkdir()
    (d/'task.toml').write_text('version="1.0"\n');(d/'instruction.md').write_text(text);(d/'tests/test.sh').write_text('exit 0\n');(d/'environment/Dockerfile').write_text('FROM python:3.11-slim\n')
    return d

def dataset(root,name,training=True):
    task(root,name,'Unique instruction '+name);tt=load_tasks(root)
    receipt={'tasks':{t['id']:t['content_sha256'] for t in tt}}
    receipt.update({'independent':True,'source':'unit test authored fixtures'} if training else {'dataset':DATASET,'synthetic':True})
    write_json(root/('training_manifest.json' if training else 'dataset_receipt.json'),receipt)
    return root

def session(path,totals):
    events=[]
    for n in totals:
        u={'input_tokens':n*8,'cached_input_tokens':n*6,'output_tokens':n*2,'total_tokens':n*10}
        events.extend([{'type':'token_usage_record','payload':{'thread_token_usage':u}},{'type':'event_msg','payload':{'type':'token_count','info':{'total_token_usage':u}}},{'type':'token_usage_record','payload':{'thread_token_usage':u}}])
    path.write_text('\n'.join(json.dumps(x) for x in events));return path

def test_prefix_cache_and_duplicate_sources(tmp_path):
    t=parse_session(session(tmp_path/'trace.jsonl',[1,2,3]));assert len(t['samples'])==3
    x=prefix(t,2,{'input':1,'cache':2,'output':3});assert x['input']==4 and x['cache']==12 and x['output']==4 and x['total']==20
    assert x['cost']==pytest.approx(40/1e6);assert x['truncated']

def test_reset_and_malformed_are_not_silently_repriced(tmp_path):
    p=session(tmp_path/'trace.jsonl',[2,1])
    with pytest.raises(ValueError,match='reset'):parse_session(p)
    p.write_text(p.read_text()+'\ninvalid json')
    with pytest.raises(ValueError):parse_session(p)

def test_same_rounds_pairing_and_unknown_prefix_success(tmp_path):
    runs={}
    for name,counts in [('normal',[3,1]),('cost-aware',[2,3])]:
        d=tmp_path/name;d.mkdir();rows=[]
        for i,n in enumerate(counts):
            sp=session(d/f'{i}.jsonl',range(1,n+1));rows.append({'instance_id':f't{i}','session_path':str(sp),'total_tokens':n*10,'success':i==0,'cost_priced':True,'input_price_per_1m':1,'cache_read_price_per_1m':2,'output_price_per_1m':3})
        write_json(d/'results.json',rows);runs[name]=d
    z=compute(runs,[2]);equal=[r for r in z['summaries'] if r['cohort']=='equal_K_exposure_tasks']
    assert all(r['task_ids']==['t0'] for r in equal)
    allrows=[r for r in z['summaries'] if r['cohort']=='all_common_valid_tasks'];assert all(r['tasks']==2 for r in allrows)
    capped=next(r for r in z['rows'] if r['run']=='normal' and r['task']=='t0');assert capped['success_at_cap'] is None

def test_missing_trace_is_excluded_not_zero(tmp_path):
    runs={}
    for name in ['a','b']:
        d=tmp_path/name;d.mkdir();write_json(d/'results.json',[{'instance_id':'missing','success':True}]);runs[name]=d
    z=compute(runs,[5]);assert len(z['excluded'])==2 and not z['rows']
    assert all(x['cost'] is None and x['tokens'] is None for x in z['summaries'])

def test_missing_price_is_null(tmp_path):
    runs={}
    for name in ['a','b']:
        d=tmp_path/name;d.mkdir();sp=session(d/'trace.jsonl',[1,2]);write_json(d/'results.json',[{'instance_id':'same','session_path':str(sp),'total_tokens':20}]);runs[name]=d
    z=compute(runs,[1]);assert all(x['cost'] is None for x in z['summaries'])

def test_renamed_duplicate_instruction_is_rejected(tmp_path):
    task(tmp_path/'train','different-name','same task instruction');task(tmp_path/'eval','target-name','same task  instruction')
    with pytest.raises(ValueError,match='leakage'):ensure_disjoint(load_tasks(tmp_path/'train'),load_tasks(tmp_path/'eval'))

def test_freeze_excludes_history_and_detects_tamper(tmp_path):
    ws=tmp_path/'workspace';(ws/'skills').mkdir(parents=True);(ws/'skills/one.md').write_text('learned rule');(ws/'evolution').mkdir();(ws/'evolution/raw.json').write_text('private training trace')
    dest=freeze(ws,tmp_path/'artifact',{'mode':'cost-aware'});assert not (dest/'strategies/evolution').exists();validate_artifact(dest)
    (dest/'strategies/skills/one.md').write_text('changed')
    with pytest.raises(ValueError,match='modified'):validate_artifact(dest)

@pytest.mark.parametrize('mode',['static-control','normal-evolve','cost-aware'])
def test_full_dry_run_freezes_and_never_evolves_eval(tmp_path,monkeypatch,mode):
    train=dataset(tmp_path/'training','train-fixture');ev=dataset(tmp_path/'benchmark','eval-fixture',False);seed=tmp_path/'seed';(seed/'prompts').mkdir(parents=True);(seed/'prompts/system.md').write_text('Seed agent')
    from examples.tb_examples.emergent_efficiency_train_eval import backend
    def forbidden(*a,**k):raise AssertionError('Dry run called a real evolution engine')
    monkeypatch.setattr(backend,'evolve',forbidden)
    out=tmp_path/'run';main(['run','--train-dir',str(train),'--eval-dir',str(ev),'--seed-workspace',str(seed),'--mode',mode,'--output',str(out),'--dry-run'])
    info=validate_artifact(out/'artifact');assert info['dry_run'];assert json.loads((out/'eval/freeze_audit.json').read_text())['evaluation_evolution_calls']==0
    if mode!='static-control':assert (out/'artifact/strategies/prompts/synthetic-training.md').is_file()
    assert json.loads((out/'eval/summary.json').read_text())['successes'] is None
    with pytest.raises(ValueError,match='Synthetic'):
        main(['evaluate','--artifact',str(out/'artifact'),'--eval-dir',str(ev),'--output',str(tmp_path/'not-dry')])

def test_zero_training_cost_is_known_for_static():
    s=summarize([],[]);assert s['solver_cost']==0 and s['solver_tokens']==0

def test_last_training_batch_evolves_before_freeze(tmp_path,monkeypatch):
    from examples.tb_examples.emergent_efficiency_train_eval import backend
    from examples.tb_examples.emergent_efficiency_adaptive import run_emergent_efficiency_tb_adaptive_harbor as legacy
    from examples.tb_examples.emergent_efficiency_train_eval.run_experiment import train_stage
    import shutil
    seed=tmp_path/'seed';(seed/'prompts').mkdir(parents=True);(seed/'prompts/system.md').write_text('seed')
    base=SimpleNamespace(_copy_seed_workspace=lambda src,dst,**kw:shutil.copytree(src,dst))
    monkeypatch.setattr(legacy,'_load_shared_runners',lambda:(base,None))
    monkeypatch.setattr(backend,'batch_solve',lambda tasks,ws,out,index,args,phase:[{'instance_id':t['id'],'conversation':[{'role':'assistant','content':'observed'}],'steps':[],'success':True,'cost_priced':True,'cost_units':1,'total_tokens':10} for t in tasks])
    calls=[]
    def evolve(rows,ws,out,index,args,ledger):
        calls.append(index);(ws/'prompts/learned.md').write_text(f'learned after batch {index}')
    monkeypatch.setattr(backend,'evolve',evolve)
    out=tmp_path/'run';out.mkdir();args=SimpleNamespace(dry_run=False,mode='normal-evolve',epochs=1,batch_size=2,seed_workspace=str(seed));plan={'dataset':DATASET,'training_tasks':[],'training_receipt':{}}
    artifact=train_stage(args,[{'id':str(i)} for i in range(5)],plan,out)
    assert calls==[1,2,3];assert (artifact/'strategies/prompts/learned.md').read_text()=='learned after batch 3'

def test_usage_ledger_tracks_empty_and_failed_requests():
    class Provider:
        def complete(self):return SimpleNamespace(content='',usage={'input_tokens':100,'output_tokens':5})
        def complete_with_tools(self):raise RuntimeError('request failed')
        def converse_loop(self):return SimpleNamespace(content='mutated skills',usage={'input_tokens':700,'output_tokens':90})
    records=[];p=AuditProvider(Provider(),records,'efficiency_proposer','model')
    p.complete()
    with pytest.raises(RuntimeError):p.complete_with_tools()
    assert records[0]['empty_content'];assert records[1]['usage'] is None and records[1]['cost'] is None
    p.converse_loop();assert records[2]['operation']=='converse_loop' and records[2]['usage']['input_tokens']==700

def test_real_harbor_normalization_with_mock_execution(tmp_path,monkeypatch):
    from examples.tb_examples.emergent_efficiency_train_eval import backend
    from agent_evolve.agents import harbor_usage
    d=task(tmp_path/'tasks','own-training','Training instruction');tt=load_tasks(d)[0]
    ws=tmp_path/'workspace';(ws/'prompts').mkdir(parents=True);(ws/'prompts/system.md').write_text('Seed instructions')
    out=tmp_path/'output';out.mkdir();sp=session(tmp_path/'trace.jsonl',[1])
    args=SimpleNamespace(harbor_model='deepseek-flash',delegate_agent='codex',delegate_kwargs='{}',harbor_cmd='harbor run',harbor_repo=None,process_timeout=60,model='deepseek:deepseek-v4-flash',mode='cost-aware',dataset=DATASET)
    import yaml
    def fake_run(cmd,**kw):
        cfg=yaml.safe_load(Path(cmd[-1]).read_text());trial=Path(cfg['jobs_dir'])/cfg['job_name']/'trial';(trial/'agent').mkdir(parents=True)
        write_json(trial/'result.json',{'verifier_result':{'rewards':{'reward':1}}})
        return SimpleNamespace(stdout='',stderr='',returncode=0)
    monkeypatch.setattr(backend.subprocess,'run',fake_run)
    monkeypatch.setattr(harbor_usage,'extract_harbor_usage',lambda *a,**kw:{'usage_artifact_path':str(sp),'input_tokens':2,'cache_read_input_tokens':6,'output_tokens':2,'total_tokens':10,'usage_record_quality':'ok'})
    row=backend.solve(tt,ws,out,1,args,'train')
    assert row['success'] and row['total_tokens']==10 and row['metadata']['benchmark']=='independent_training'

def test_backend_config_retains_native_task_timeout(tmp_path):
    pp=tmp_path/'prompt';pp.write_text('context');args=SimpleNamespace(harbor_model='model',delegate_agent='codex',delegate_kwargs='{}')
    cfg=config_for({'path':str(tmp_path/'tasks/a'),'id':'a'},tmp_path,pp,tmp_path/'jobs','job',args)
    assert cfg['tasks']==[{'path':str((tmp_path/'tasks/a').resolve())}];assert 'datasets' not in cfg;assert 'override_timeout_sec' not in cfg['agents'][0]
    assert cfg['agents'][0]['kwargs']['include_task_json_prompt'] is False


def test_coq_proxy_preserves_task_dockerfile(tmp_path):
    import yaml
    from examples.tb_examples.emergent_efficiency_train_eval.backend import configure_container_proxy

    environment = tmp_path/'task'/'environment'
    environment.mkdir(parents=True)
    dockerfile = environment/'Dockerfile'
    source = "FROM debian:bookworm-slim\nRUN su - coq -c 'opam init'\nRUN echo $PATH\n"
    dockerfile.write_text(source)
    inputs = tmp_path/'inputs'
    inputs.mkdir()
    config = {'environment': {'type': 'docker'}, 'agents': [{'kwargs': {}}]}
    proxy = 'http://10.0.2.100:17892'

    configure_container_proxy(config, {'id': 'coq-block-bound', 'path': str(environment.parent)}, inputs, proxy)

    assert dockerfile.read_text() == source
    overlay = Path(config['environment']['extra_docker_compose'][0])
    build = yaml.safe_load(overlay.read_text())['services']['main']['build']
    effective = Path(build['dockerfile']).read_text()
    assert effective.replace('--whitelist-environment=http_proxy,https_proxy,HTTP_PROXY,HTTPS_PROXY ', '') == source
    assert build['args']['HTTPS_PROXY'] == proxy
    assert config['agents'][0]['kwargs']['setup_proxy'] == proxy
