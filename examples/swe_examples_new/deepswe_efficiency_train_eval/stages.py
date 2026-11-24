from __future__ import annotations
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from .common import *
def summarize(rows,ledger):
    actual=[r for r in rows if not r.get('synthetic')];priced=(not rows) or (bool(actual) and all(r.get('cost_priced') is True for r in actual))
    solve_cost=sum(r.get('cost_units',0) for r in actual) if priced else None
    return {'tasks':len(rows),'synthetic':any(r.get('synthetic') for r in rows),'successes':sum(r.get('success') is True for r in actual) if actual or not rows else None,'solver_tokens':sum(r.get('total_tokens',0) for r in actual) if actual or not rows else None,'solver_cost':solve_cost,'pass_rate':sum(r.get('success') is True for r in actual)/len(actual) if actual else None,'mean_solver_cost':solve_cost/len(actual) if actual and solve_cost is not None else None,'mean_solver_tokens':sum(r.get('total_tokens',0) for r in actual)/len(actual) if actual else None,'evolution_calls_recorded':len(ledger),'evolution_reported_usage':[{'component':x['component'],'usage':x['usage'],'model':x['model']} for x in ledger],'pipeline_cost':solve_cost if not ledger else None,'pipeline_cost_status':'complete_solver_only_no_evolution' if priced and not ledger else 'incomplete_not_all_components_priced','unknown_cost_calls':sum(x['cost'] is None for x in ledger),'note':'Provider usage may lack cache decomposition. Unknown cost is null, not zero. No judge/proposer response is intentionally excluded from the ledger.'}

def save_progress(out,rows,ledger):
    write_json(out/'results.json',rows);write_json(out/'usage_ledger.json',ledger);write_json(out/'summary.json',summarize(rows,ledger))

def train_stage(args,train,plan,out):
    from .backend import batch_solve,evolve
    workspace=out/'workspace'
    if args.dry_run:shutil.copytree(args.seed_workspace,workspace,ignore=shutil.ignore_patterns('.git','efficiency','__pycache__'))
    else:
        from examples.swe_examples_new.emergent_efficiency import run_emergent_efficiency as legacy
        legacy._copy_seed_workspace(Path(args.seed_workspace),workspace,hide_efficiency=True)
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
    from .backend import batch_solve
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

