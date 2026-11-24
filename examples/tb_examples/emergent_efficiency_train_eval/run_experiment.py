#!/usr/bin/env python3
"""Learn on independent tasks, freeze, then evaluate TB 4.1 without evolution."""
from __future__ import annotations
import argparse
import importlib.util
import json
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
from pathlib import Path
if __package__ in (None,''): sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from examples.tb_examples.emergent_efficiency_train_eval.common import DATASET,load_tasks,validate_receipt,ensure_disjoint,public_identity,fresh_dir,tree_hashes,write_json,read_json,freeze,validate_artifact,sha

def parser():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    for name in ('plan','run','train','evaluate'):
        s=sub.add_parser(name)
        s.add_argument('--eval-dir',help='Prepared TB4.1.0 tasks directory (legacy protocol)')
        s.add_argument('--split-manifest',help='Fixed train/val/test manifest from prepare_dataset split')
        s.add_argument('--selection-tier',choices=['lite','medium','heavy'],default='medium')
        s.add_argument('--selection-cost',choices=['cost_units','total_tokens'],default='cost_units')
        s.add_argument('--candidates-per-epoch',type=int,default=2)
        s.add_argument('--output',required=name!='plan')
        s.add_argument('--model',default='deepseek:deepseek-v4-flash');s.add_argument('--evolver-model');s.add_argument('--harbor-model',default='deepseek-flash')
        s.add_argument('--harbor-repo',default='/home/azureuser/zj/harbor');s.add_argument('--harbor-cmd',default='.venv/bin/harbor run')
        s.add_argument('--delegate-agent',default='codex');s.add_argument('--delegate-kwargs',default='{}')
        s.add_argument('--parallel',type=int,default=2);s.add_argument('--batch-size',type=int,default=4);s.add_argument('--process-timeout',type=int,default=40000)
        s.add_argument('--eval-limit',type=int);s.add_argument('--dry-run',action='store_true')
        if name=='evaluate':s.add_argument('--artifact',required=True)
        else:
            s.add_argument('--train-dir');s.add_argument('--mode',choices=['static-control','normal-evolve','cost-aware'],default='cost-aware');s.add_argument('--epochs',type=int,default=1);s.add_argument('--train-limit',type=int);s.add_argument('--seed-workspace',default='seed_workspaces/terminal');s.add_argument('--allow-demo-training',action='store_true')
    s=sub.add_parser('preflight');s.add_argument('--harbor-repo',default='/home/azureuser/zj/harbor')
    return p

def build_plan(args):
    evaluation=load_tasks(args.eval_dir);receipt=validate_receipt(args.eval_dir,evaluation,allow_demo=args.dry_run)
    args.dataset=receipt['dataset']
    if args.command=='evaluate':
        artifact=validate_artifact(args.artifact)
        if artifact.get('protocol') == 'train_val_select_then_frozen_test': raise ValueError('Three-way artifact requires --split-manifest')
        if artifact.get('dataset') != receipt['dataset']: raise ValueError('Artifact and evaluation dataset identities differ')
        if artifact.get('dry_run') and not args.dry_run:raise ValueError('Synthetic training artifact cannot be evaluated as a real experiment')
        train=artifact['training_tasks'];leak=ensure_disjoint(train,evaluation);args.mode=artifact['mode']
        train_receipt=artifact['training_receipt']
    else:
        train=load_tasks(args.train_dir);train_receipt=validate_receipt(args.train_dir,train,training=True,allow_demo=args.allow_demo_training);leak=ensure_disjoint(train,evaluation)
    # Always check ALL train/eval identities before selecting limits.
    if args.eval_limit is not None:evaluation=evaluation[:args.eval_limit]
    if getattr(args,'train_limit',None) is not None:train=train[:args.train_limit]
    return train,evaluation,{'dataset':receipt['dataset'],'dataset_receipt':receipt,'training_receipt':train_receipt,'training_tasks':[public_identity(t) for t in train],'evaluation_tasks':[public_identity(t) for t in evaluation],'leakage_audit':leak,'mode':args.mode,'dry_run':args.dry_run,'protocol':'independent_train_then_frozen_eval','eval_evolution_enabled':False,'epochs':getattr(args,'epochs',None),'parameters':vars(args).copy(),'solver_max_rounds_enforced':False,'historical_round_accounting':'separate retrospective model_usage_update prefix analysis'}

def summarize(rows,ledger):
    actual=[r for r in rows if not r.get('synthetic')];priced=(not rows) or (bool(actual) and all(r.get('cost_priced') is True for r in actual))
    solve_cost=sum(r.get('cost_units',0) for r in actual) if priced else None
    return {'tasks':len(rows),'synthetic':any(r.get('synthetic') for r in rows),'successes':sum(r.get('success') is True for r in actual) if actual or not rows else None,'solver_tokens':sum(r.get('total_tokens',0) for r in actual) if actual or not rows else None,'solver_cost':solve_cost,'pass_rate':sum(r.get('success') is True for r in actual)/len(actual) if actual else None,'mean_solver_cost':solve_cost/len(actual) if actual and solve_cost is not None else None,'mean_solver_tokens':sum(r.get('total_tokens',0) for r in actual)/len(actual) if actual else None,'evolution_calls_recorded':len(ledger),'evolution_reported_usage':[{'component':x['component'],'usage':x['usage'],'model':x['model']} for x in ledger],'pipeline_cost':solve_cost if not ledger else None,'pipeline_cost_status':'complete_solver_only_no_evolution' if priced and not ledger else 'incomplete_not_all_components_priced','unknown_cost_calls':sum(x['cost'] is None for x in ledger),'note':'Provider usage may lack cache decomposition. Unknown cost is null, not zero. No judge/proposer response is intentionally excluded from the ledger.'}

def save_progress(out,rows,ledger):
    write_json(out/'results.json',rows);write_json(out/'usage_ledger.json',ledger);write_json(out/'summary.json',summarize(rows,ledger))

def train_stage(args,train,plan,out):
    from examples.tb_examples.emergent_efficiency_train_eval.backend import batch_solve,evolve
    workspace=out/'workspace'
    if args.dry_run:shutil.copytree(args.seed_workspace,workspace,ignore=shutil.ignore_patterns('.git','efficiency','__pycache__'))
    else:
        from examples.tb_examples.emergent_efficiency_adaptive import run_emergent_efficiency_tb_adaptive_harbor as legacy
        base,_=legacy._load_shared_runners();base._copy_seed_workspace(Path(args.seed_workspace),workspace,hide_efficiency=True)
    write_json(out/'seed_strategy_hashes.json',tree_hashes(workspace))
    rows=[];ledger=[];train_out=out/'train';train_out.mkdir()
    if args.mode!='static-control':
        index=0
        for epoch in range(args.epochs):
            for offset in range(0,len(train),args.batch_size):
                index+=1;batch_out=train_out/f'epoch_{epoch+1:03d}_batch_{index:03d}';batch_out.mkdir()
                batch=batch_solve(train[offset:offset+args.batch_size],workspace,batch_out,index,args,'train')
                for r in batch:r['epoch']=epoch+1
                rows.extend(batch);save_progress(train_out,rows,ledger)
                write_json(batch_out/'training_observations.json',[{'task_id':r['instance_id'],'conversation':r['conversation'],'steps':r['steps'],'success':r['success']} for r in batch])
                try:
                    if args.dry_run:
                        (workspace/'prompts').mkdir(exist_ok=True);(workspace/'prompts'/'synthetic-training.md').write_text('SYNTHETIC SMOKE artifact: no real efficiency strategy was learned.\n')
                    else:evolve(batch,workspace,train_out,index,args,ledger)
                finally:save_progress(train_out,rows,ledger)
                # The final training batch ALSO evolves before freezing.
    else:save_progress(train_out,[],ledger)
    artifact=freeze(workspace,out/'artifact',{'mode':args.mode,'dry_run':args.dry_run,'dataset':plan['dataset'],'training_tasks':plan['training_tasks'],'training_receipt':plan['training_receipt'],'training_usage_ledger_sha256':sha(train_out/'usage_ledger.json'),'training_summary':read_json(train_out/'summary.json')})
    return artifact

def evaluate_stage(args,tasks,artifact,out,phase='eval'):
    from examples.tb_examples.emergent_efficiency_train_eval.backend import batch_solve
    before=validate_artifact(artifact);eval_out=out/phase;eval_out.mkdir();rows=[]
    # No Observer, evolution engine or proposer is constructed here.
    def trial(index,task):
        task_root=eval_out/'trials'/task['id'];task_root.mkdir(parents=True)
        workspace=task_root/'strategies';shutil.copytree(Path(artifact)/'strategies',workspace)
        batch=batch_solve([task],workspace,task_root,index,args,phase)
        validate_artifact(artifact)
        return batch
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures=[pool.submit(trial,i,task) for i,task in enumerate(tasks,1)]
        for future in as_completed(futures):
            rows.extend(future.result());rows.sort(key=lambda r:r['batch']);save_progress(eval_out,rows,[])
    after=validate_artifact(artifact);write_json(eval_out/'freeze_audit.json',{'passed':before['strategy_hashes']==after['strategy_hashes'],'strategy_hashes':after['strategy_hashes'],'evaluation_evolution_calls':0})
    training=before['training_summary'];evaluation=read_json(eval_out/'summary.json')
    write_json(out/'pipeline_summary.json',{'training':training,'evaluation':evaluation,'total_cost':None if training['solver_cost'] is None or evaluation['solver_cost'] is None or training['evolution_calls_recorded'] else training['solver_cost']+evaluation['solver_cost'],'training_cost_amortization_note':'Report training solver, all evolution components and frozen evaluation separately; do not label evaluation-only saving as pipeline saving.'})
    return rows

def main(argv=None):
    p=parser();args=p.parse_args(argv)
    if args.command=='preflight':
        deps={x:bool(importlib.util.find_spec(x)) for x in ['pytest','yaml','matplotlib','harbor','strands']}
        print(json.dumps({'dependencies_in_project_venv':deps,'harbor_repo_exists':Path(args.harbor_repo).exists(),'dataset':DATASET},indent=2));return
    for k in ['parallel','batch_size','process_timeout','epochs','train_limit','eval_limit','candidates_per_epoch']:
        v=getattr(args,k,None)
        if v is not None and v<=0:p.error(f'--{k.replace("_","-")} must be positive')
    args.dataset=DATASET
    if not isinstance(json.loads(args.delegate_kwargs),dict):p.error('--delegate-kwargs must be a JSON object')
    if args.split_manifest:
        from examples.tb_examples.emergent_efficiency_train_eval.three_way import run
        return run(args)
    if not args.eval_dir or (args.command!='evaluate' and not args.train_dir):
        p.error('Supply --split-manifest or legacy --train-dir/--eval-dir')
    train,evaluation,plan=build_plan(args)
    args.dataset=plan['dataset']
    if args.command=='plan':
        if args.output:write_json(args.output,plan)
        print(json.dumps(plan,ensure_ascii=False,indent=2));return
    out=fresh_dir(args.output);plan['created_at']=datetime.now(timezone.utc).isoformat();write_json(out/'experiment_manifest.json',plan)
    artifact=Path(args.artifact) if args.command=='evaluate' else train_stage(args,train,plan,out)
    if args.command!='train':evaluate_stage(args,evaluation,artifact,out)
    print(json.dumps({'output':str(out),'artifact':str(artifact),'dry_run':args.dry_run},ensure_ascii=False))

if __name__=='__main__':main()


