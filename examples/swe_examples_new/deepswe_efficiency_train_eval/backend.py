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
from examples.tb_examples.emergent_efficiency_train_eval.traces import parse_session

ADAPTER='examples.swe_examples_new.deepswe_efficiency_train_eval.harbor_agent:DeepSWECodexAgent'

def render_prompt(workspace):
    root=Path(workspace); parts=[]
    # Include complete strategy text: a host-only skill name is not a loaded skill.
    for folder in ('prompts','skills','memory','tools'):
        for p in sorted((root/folder).rglob('*')):
            if p.is_file() and p.suffix in ('.md','.txt','.json','.jsonl','.py','.yaml','.yml'):
                parts.append(f'## Strategy asset: {p.relative_to(root).as_posix()}\n'+p.read_text(encoding='utf-8'))
    return '\n\n'.join(parts)+'\n\nDeepSWE v1.1 delivery: implement the requested change in /app, verify it, and COMMIT your changes to Git before finishing. Only the committed diff against the base commit is graded in a separate clean verifier container. Never edit hidden tests or seek reference solutions. Strategy tool source below is guidance, not a registered tool. Execution interface: use only tools actually exposed by the terminal agent. Skill text is supplied above in full. If read_skill or submit is not available, do not search for invented executables; use the supplied skill text and finish using the native agent interface. Preserve the task deliverable and let the official verifier judge correctness.\n'

def config_for(task,workspace,prompt_path,job_root,job_name,args):
    if args.delegate_agent != 'codex':
        raise ValueError('This DeepSWE adapter currently supports native Harbor Codex only')
    kwargs = json.loads(args.delegate_kwargs)
    kwargs['system_prompt_path'] = str(prompt_path.resolve())
    return {
        'job_name':job_name, 'jobs_dir':str(job_root.resolve()), 'n_attempts':1,
        'timeout_multiplier':1.0,
        'orchestrator':{'type':'local','n_concurrent_trials':1,'quiet':True,'kwargs':{}},
        'environment':{'type':'docker'},
        'agents':[{'name':'deepswe-evolved-codex','import_path':ADAPTER,
                   'model_name':args.harbor_model,'kwargs':kwargs}],
        'tasks':[{'path':str(Path(task['path']).resolve())}],
    }

def load_training_trajectory(trial, legacy_harbor):
    """Accept legacy conversation JSON and Harbor's ATIF step trajectories."""
    steps, conversation, path = legacy_harbor._load_trajectory(trial)
    if conversation:
        return steps, conversation, path
    candidates = [trial/'agent'/'trajectory.json', *sorted((trial/'agent').rglob('*.trajectory.json'))]
    for candidate in candidates:
        if not candidate.is_file():
            continue
        data = read_json(candidate)
        records = data.get('steps') if isinstance(data, dict) else None
        if not isinstance(records, list) or not records:
            continue
        conversation, steps = [], []
        for record in records:
            conversation.append({'role':'assistant' if record.get('source')=='agent' else 'user',
                                 'content':json.dumps(record,ensure_ascii=False,default=str)})
            for call in record.get('tool_calls', []) or []:
                function = call.get('function', {})
                steps.append({'tool':function.get('name', 'unknown'),
                              'input':function.get('arguments', {}),
                              'output':record.get('observation', {})})
        return steps, conversation, str(candidate)
    return steps, conversation, path

def solve(task,workspace,out,batch,args,phase):
    from agent_evolve.agents.terminal.dataset import TB2Task
    from agent_evolve.agents.harbor_usage import extract_harbor_usage
    from examples.tb_examples.emergent_efficiency_adaptive import run_emergent_efficiency_tb_adaptive_harbor as legacy
    base,harbor=legacy._load_shared_runners()
    prompt=render_prompt(workspace);inputs=out/'inputs'/task['id'];inputs.mkdir(parents=True,exist_ok=True);pp=inputs/'system_prompt.md';pp.write_text(prompt,encoding='utf-8')
    job_name=f'aev-deepswe-{phase}-{uuid.uuid4().hex[:12]}';jobs=out/'harbor_jobs';cfg=config_for(task,workspace,pp,jobs,job_name,args)
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
    steps,conversation,trajectory_path=load_training_trajectory(trial,harbor)
    usage=extract_harbor_usage(data,trial_dir=trial,model_id=args.harbor_model,usage_source=f'train_eval_{phase}',include_codex_artifacts=True)
    trace_error=None;session=usage.get('usage_artifact_path');raw_tool_calls=None
    if session and Path(session).is_file() and Path(session).suffix == '.jsonl':
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
    row['task_instruction'] = task['instruction']
    row['task_metadata'] = task['metadata']
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


def evolution_config(args):
    from agent_evolve.config import EvolveConfig
    return EvolveConfig(evolver_model=args.evolver_model or args.model,
                        evolver_max_tokens=args.evolver_max_tokens,
                        evolver_temperature=args.evolver_temperature,
                        extra={'region':args.region})

REFLECTION_PROMPT = """Review the supplied training trajectory and propose ONE generalizable
intervention for future software-engineering tasks. Prefer enhancing an existing
skill over adding a narrow rule. Do not memorize task-specific solutions.
Use only this training evidence, not held-out tasks. A failed outcome alone does
not prove a root cause. Return this format:
CONFIDENCE: HIGH/MEDIUM/LOW
ACTION: NEW or ENHANCE (or NONE when no reusable lesson is supported)
TARGET: existing name if enhancing
TYPE: skill | fragment | tool | memory
NAME: generic_pattern_name
DESCRIPTION: TRIGGER when ...; DO NOT TRIGGER when ...
CONTENT:
A complete multiline procedure under 500 words (skills need at least two nonempty lines and 120 characters); tools require complete deterministic Strands @tool source.
"""

def evolve(rows, workspace, out, index, args, ledger):
    """SWE solver-proposal curation + the original SWE efficiency proposer."""
    from examples.swe_examples_new.emergent_efficiency import run_emergent_efficiency as swe
    from agent_evolve.algorithms.guided_synth.engine import GuidedSynthesisEngine
    from agent_evolve.algorithms.adaptive_skill.tools import create_default_llm
    from agent_evolve.llm.factory import create_llm_provider
    from agent_evolve.llm.base import LLMMessage
    from agent_evolve.types import Task
    from agent_evolve.contract.workspace import AgentWorkspace
    from types import SimpleNamespace

    if any(not r.get('conversation') or r.get('trace_error') for r in rows):
        raise ValueError('Missing or invalid training trajectory; refusing blind evolution')
    config = evolution_config(args)
    # Harbor delegates do not produce SWE _skill_proposal; explicitly bridge
    # train trajectories to the same structured proposal/curation interface.
    reflector = AuditProvider(create_llm_provider(args.model,region=args.region),
                              ledger,'training_skill_reflection',args.model)
    enriched = []
    for row in rows:
        evidence = dict(task=row['task_instruction'],conversation=row['conversation'],
                        steps=row['steps'])
        response = reflector.complete([
            LLMMessage(role='system',content=REFLECTION_PROMPT),
            LLMMessage(role='user',content=render_prompt(workspace)+'\nTRAINING EVIDENCE\n'
                       +json.dumps(evidence,ensure_ascii=False,default=str)),
        ],max_tokens=4096,temperature=.2)
        proposal = str(response.content or '').strip()
        if not proposal:
            raise ValueError('Empty skill reflection; usage recorded, no silent no-op evolution')
        if 'ACTION: NONE' not in proposal.upper() and GuidedSynthesisEngine(config,write_memory=False)._parse_intervention(proposal) is None:
            raise ValueError('Malformed skill reflection; inspect training evidence and usage ledger')
        enriched.append(dict(row,skill_proposal=proposal))
        write_json(out/'emergence'/f'batch_{index:03d}'/f"{row['instance_id']}_reflection.json",
                   dict(task_id=row['instance_id'],proposal=proposal))
    tasks = {r['instance_id']:Task(id=r['instance_id'],input=r['task_instruction'],
                                  metadata=r.get('task_metadata',{})) for r in enriched}
    observations = swe._results_to_observations(enriched,tasks)
    curator = AuditProvider(create_default_llm(config),ledger,'ordinary_swe_curation',
                            args.evolver_model or args.model)
    result = GuidedSynthesisEngine(config,llm=curator,write_memory=False).evolve(
        AgentWorkspace(workspace),observations,evo_number=index)
    write_json(out/'emergence'/f'batch_{index:03d}_ordinary.json',result)
    if args.mode == 'cost-aware':
        cfg = SimpleNamespace(**vars(args))
        cfg.model_id = args.model
        cfg.dry_run_proposer = False
        cfg.category_seed = 'none'
        cfg.evolver_call_retries = 4
        cfg.evolver_retry_initial_delay = 2.
        cfg.evolver_retry_max_delay = 30.
        # Used only as a diagnostic reference in original SWE cost feedback.
        # No external hard-stop budget is imposed on Harbor tasks.
        cfg.max_steps = 60
        original = swe._load_proposer_llm
        swe._load_proposer_llm = lambda _: AuditProvider(
            create_default_llm(config),ledger,'swe_efficiency_proposer',
            args.evolver_model or args.model)
        try:
            report = swe.propose_emergent_efficiency(
                batch_results=enriched,batch_num=index,max_steps=cfg.max_steps,args=cfg)
        finally:
            swe._load_proposer_llm = original
        swe._write_proposals(out,report)
        swe._apply_proposals(workspace,report,output_dir=out,
                             max_apply=args.max_proposals_per_batch)
    return result
