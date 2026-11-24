"""Harbor execution and read-only reuse of the original evolution helpers."""
from __future__ import annotations
import json
import os
import shlex
import subprocess
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from .common import write_json, read_json, tree_hashes
from .traces import parse_session

ADAPTER='agent_evolve.agents.terminal.harbor_adapter:AEvolveTerminalHarborAgent'

def render_prompt(workspace):
    root=Path(workspace); parts=[]
    # Include complete strategy text: a host-only skill name is not a loaded skill.
    for folder in ('prompts','skills','memory','tools'):
        for p in sorted((root/folder).rglob('*')):
            if p.is_file() and p.suffix in ('.md','.txt','.json','.jsonl','.py','.yaml','.yml'):
                parts.append(f'## Strategy asset: {p.relative_to(root).as_posix()}\n'+p.read_text(encoding='utf-8'))
    return '\n\n'.join(parts)+'\n\nExecution interface: use only tools actually exposed by the terminal agent. Skill text is supplied above in full. If read_skill or submit is not available, do not search for invented executables; use the supplied skill text and finish using the native agent interface. Preserve the task deliverable and let the official verifier judge correctness.\n'

def config_for(task,workspace,prompt_path,job_root,job_name,args):
    return {'job_name':job_name,'jobs_dir':str(job_root.resolve()),'n_attempts':1,'timeout_multiplier':1.0,'orchestrator':{'type':'local','n_concurrent_trials':1,'quiet':True,'kwargs':{}},'environment':{'type':'docker'},'agents':[{'name':'a-evolve-frozen-terminal','import_path':ADAPTER,'model_name':args.harbor_model,'kwargs':{'workspace_dir':str(workspace.resolve()),'system_prompt_path':str(prompt_path.resolve()),'include_task_json_prompt':False,'max_system_prompt_chars':max(24000,len(prompt_path.read_text())+1),'delegate_agent_name':args.delegate_agent,'delegate_model_name':args.harbor_model,'delegate_kwargs':json.loads(args.delegate_kwargs)}}],'tasks':[{'path':str(Path(task['path']).resolve())}]}


def configure_container_proxy(config, task, inputs, proxy):
    """Proxy Dockerfile fetches and delegate setup without touching task files."""
    import yaml

    proxy_keys = ('http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY')
    build = {'args': dict.fromkeys(proxy_keys, proxy)}
    if task['id'] == 'coq-block-bound':
        original = (Path(task['path'])/'environment/Dockerfile').read_text(encoding='utf-8')
        command = 'RUN su - coq -c '
        if original.count(command) != 1:
            raise ValueError('Unexpected Coq Dockerfile; review the proxy overlay')
        # The login shell clears inherited proxy variables without this option.
        keep = ','.join(proxy_keys)
        dockerfile = inputs/'Dockerfile.build_proxy'
        dockerfile.write_text(
            original.replace(command, f'RUN su --whitelist-environment={keep} - coq -c ', 1),
            encoding='utf-8',
        )
        build['dockerfile'] = str(dockerfile.resolve())
    overlay = inputs/'build_proxy.yaml'
    overlay.write_text(yaml.safe_dump({'services': {'main': {'build': build}}}), encoding='utf-8')
    config['environment']['extra_docker_compose'] = [str(overlay.resolve())]
    agent = config['agents'][0]
    agent['import_path'] = (
        'examples.tb_examples.emergent_efficiency_train_eval.harbor_proxy_agent:SetupProxyHarborAgent'
    )
    agent['kwargs']['setup_proxy'] = proxy

def solve(task,workspace,out,batch,args,phase):
    from agent_evolve.agents.terminal.dataset import TB2Task
    from agent_evolve.agents.harbor_usage import extract_harbor_usage
    from examples.tb_examples.emergent_efficiency_adaptive import run_emergent_efficiency_tb_adaptive_harbor as legacy
    base,harbor=legacy._load_shared_runners()
    prompt=render_prompt(workspace);inputs=out/'inputs'/task['id'];inputs.mkdir(parents=True,exist_ok=True);pp=inputs/'system_prompt.md';pp.write_text(prompt,encoding='utf-8')
    job_name=f'aev-train-eval-{phase}-{uuid.uuid4().hex[:12]}';jobs=out/'harbor_jobs';cfg=config_for(task,workspace,pp,jobs,job_name,args)
    proxy = os.environ.get('EVOEFFICIENT_CONTAINER_PROXY')
    if proxy:
        configure_container_proxy(cfg, task, inputs, proxy)
    cp=inputs/'harbor.yaml'
    import yaml
    cp.write_text(yaml.safe_dump(cfg,sort_keys=False),encoding='utf-8')
    cmd=[*shlex.split(args.harbor_cmd),'-c',str(cp.resolve())];env=dict(os.environ);repo=Path(__file__).resolve().parents[3];env['PYTHONPATH']=str(repo)+os.pathsep+env.get('PYTHONPATH','')
    started=time.time()
    try:
        done=subprocess.run(cmd,cwd=args.harbor_repo or repo,env=env,text=True,capture_output=True,timeout=args.process_timeout)
        stdout,stderr,returncode=done.stdout,done.stderr,done.returncode
    except subprocess.TimeoutExpired as e:
        stdout=e.stdout.decode(errors='replace') if isinstance(e.stdout,bytes) else e.stdout or '';stderr='Outer Harbor process timeout; inspect the job before resuming';returncode=-1
    (inputs/'stdout.txt').write_text(stdout,encoding='utf-8');(inputs/'stderr.txt').write_text(stderr,encoding='utf-8')
    candidates=[p for p in (jobs/job_name).rglob('result.json') if (p.parent/'agent').is_dir() or 'verifier_result' in read_json(p)]
    if len(candidates)!=1: raise RuntimeError(f'Expected one Harbor trial result for {task["id"]}, found {len(candidates)}. See {inputs}')
    rp=candidates[0];data=read_json(rp);trial=rp.parent;reward=((data.get('verifier_result') or {}).get('rewards') or {}).get('reward');passed=reward==1
    if getattr(args,'protocol',None) and (returncode != 0 or data.get('exception_info') or reward is None):
        raise RuntimeError(f'Invalid {phase} trial for {task["id"]}; inspect {rp} before selection/reporting')
    steps,conversation,trajectory_path=harbor._load_trajectory(trial)
    usage=extract_harbor_usage(data,trial_dir=trial,model_id=args.harbor_model,usage_source=f'train_eval_{phase}',include_codex_artifacts=True)
    trace_error=None;session=usage.get('usage_artifact_path');raw_tool_calls=None
    if session and Path(session).is_file():
        try:
            trace=parse_session(session)
            raw_tool_calls=trace['tool_calls']
            if not conversation: conversation=trace['conversation']
            if not steps: steps=trace['steps']
        except ValueError as e: trace_error=str(e)
    raw={'task_name':task['id'],'passed':passed,'score':float(reward or 0),'status':'passed' if passed else 'error' if data.get('exception_info') else 'failed','eval_output':json.dumps({'reward':reward,'exception_info':data.get('exception_info')},ensure_ascii=False),'usage':usage,'solve_time':time.time()-started,'total_time':time.time()-started,'conversation':conversation,'steps':steps,'raw_job_path':str(trial),'result_json_path':str(rp),'trajectory_path':trajectory_path,'returncode':returncode,'metadata':{},'harbor_command':cmd}
    compat=TB2Task(name=task['id'],prompt=task['instruction'],docker_image='',test_sh_path=str(Path(task['path'])/'tests/test.sh'),tests_dir=str(Path(task['path'])/'tests'),metadata=task['metadata'])
    row=harbor._normalize_harbor_tb_result(raw=raw,task=compat,workspace_dir=workspace,output_dir=out,system_prompt_text=prompt,model_id=args.model,batch_num=batch,mode=args.mode)
    row.update({'phase':phase,'benchmark':'independent_training' if phase=='train' and getattr(args,'protocol',None) is None else args.dataset,'status':raw['status'],'conversation':conversation,'steps':steps,'trace_error':trace_error,'synthetic':False})
    row['metadata']['benchmark']=row['benchmark']
    if raw_tool_calls is not None:row['num_tool_calls']=raw_tool_calls
    return row

def synthetic(task,workspace,out,batch,args,phase):
    # Explicit smoke fixtures, never presented as actual benchmark performance.
    return {'instance_id':task['id'],'task_name':task['id'],'batch':batch,'phase':phase,'mode':args.mode,'success':True,'score':1.0,'status':'synthetic','synthetic':True,'total_tokens':0,'cost_units':None,'cost_priced':False,'elapsed':0,'num_tool_calls':0,'conversation':[{'role':'assistant','content':'SYNTHETIC SMOKE; no model or Docker execution'}],'steps':[]}

def batch_solve(tasks,workspace,out,index,args,phase):
    fn=synthetic if args.dry_run else solve
    before=tree_hashes(workspace)
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        rows=list(pool.map(lambda t:fn(t,workspace,out,index,args,phase),tasks))
    if tree_hashes(workspace)!=before: raise ValueError('Solver modified the host strategy workspace; refusing to treat evaluation as frozen')
    return rows

class AuditProvider:
    """Record provider response usage and failed calls without inventing cache prices."""
    def __init__(self,provider,ledger,component,model): self.provider,self.ledger,self.component,self.model=provider,ledger,component,model
    def __getattr__(self,name):
        value=getattr(self.provider,name)
        if name not in ('complete','complete_with_tools','chat_with_tools','converse_loop'): return value
        def call(*a,**kw):
            started=time.time()
            try:
                r=value(*a,**kw);usage=getattr(r,'usage',{}) or {}
                self.ledger.append({'component':self.component,'model':self.model,'operation':name,'usage':usage,'cost':None,'cost_status':'not_priced_no_verified_cache_profile','elapsed':time.time()-started,'status':'ok','empty_content':not bool(getattr(r,'content',''))})
                return r
            except Exception as e:
                self.ledger.append({'component':self.component,'model':self.model,'operation':name,'usage':None,'cost':None,'cost_status':'unknown_failed_request','elapsed':time.time()-started,'status':'error','error':str(e)[:1000]});raise
        return call

def evolve(rows,workspace,out,index,args,ledger):
    from examples.tb_examples.emergent_efficiency_adaptive import run_emergent_efficiency_tb_adaptive_harbor as legacy
    from examples.tb_examples.emergent_efficiency_adaptive import experiment_controls as controls
    from agent_evolve.algorithms.adaptive_skill import AdaptiveSkillEngine
    from agent_evolve.algorithms.adaptive_skill.tools import create_default_llm
    from agent_evolve.contract.workspace import AgentWorkspace
    base,_=legacy._load_shared_runners();cfgargs=legacy.parse_args(['--mode',args.mode,'--model-id',args.model]);cfgargs.evolver_model=args.evolver_model or args.model
    config=legacy._build_adaptive_config(cfgargs);provider=AuditProvider(create_default_llm(config),ledger,'ordinary_evolution',cfgargs.evolver_model)
    if any(not r.get('conversation') or r.get('trace_error') for r in rows): raise ValueError('Training trace missing or invalid; refusing trajectory-blind evolution')
    logs=[{'task_id':r['instance_id'],'conversation':r['conversation'],'success':r['success'],'score':r['score'],'feedback_detail':r.get('feedback_detail',''),'steps':r['steps']} for r in rows]
    # Keep the original trajectory-only method; pin and account its judge too.
    from agent_evolve.algorithms.adaptive_skill import prompts
    from agent_evolve.llm import factory
    original_judge=prompts.judge_trajectories
    def judge(training_logs):
        original_factory=factory.create_llm_provider
        factory.create_llm_provider=lambda model_id,**kw:AuditProvider(original_factory(model_id,**kw),ledger,'trajectory_judge',model_id)
        try: return original_judge(training_logs,model_id=cfgargs.evolver_model,region=cfgargs.region)
        finally: factory.create_llm_provider=original_factory
    prompts.judge_trajectories=judge
    try: result=AdaptiveSkillEngine(config,llm=provider).evolve(AgentWorkspace(workspace),logs,evo_number=index)
    finally: prompts.judge_trajectories=original_judge
    write_json(out/'emergence'/f'batch_{index:03d}_ordinary.json',result)
    if args.mode=='cost-aware':
        analyzed=[controls.analyze_result(r) for r in rows];feedback=controls.failure_feedback(analyzed);shared=base._swe;original=shared._load_proposer_llm
        shared._load_proposer_llm=lambda _:AuditProvider(create_default_llm(config),ledger,'efficiency_proposer',cfgargs.evolver_model)
        try:
            with controls.failure_aware_proposer(shared,feedback): report=shared.propose_emergent_efficiency(batch_results=analyzed,batch_num=index,max_steps=cfgargs.max_steps,args=cfgargs)
        finally: shared._load_proposer_llm=original
        report['failure_token_distribution']=feedback
        write_json(out/'emergence'/f'batch_{index:03d}_proposal_report.json',report)
        base._apply_tb_proposals(workspace,report,output_dir=out,max_apply=cfgargs.max_proposals_per_batch)
    return result


