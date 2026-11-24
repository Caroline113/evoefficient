"""DeepSWE v1.1：训练生成策略、验证筛选、冻结后 Harbor 测试。"""
import argparse
import json
import sys
from pathlib import Path
if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from examples.swe_examples_new.deepswe_efficiency_train_eval.stages import (
    train_stage, evaluate_stage, summarize, save_progress,
)

def parser():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    for name in ('plan','run','train','evaluate'):
        s=sub.add_parser(name)
        s.add_argument('--split-manifest',required=True)
        s.add_argument('--output',required=name!='plan')
        s.add_argument('--model',default='deepseek:deepseek-flash')
        s.add_argument('--evolver-model')
        s.add_argument('--harbor-model',default='deepseek-flash')
        s.add_argument('--harbor-repo',default=str(Path(__file__).resolve().parents[4]/'harbor'))
        s.add_argument('--harbor-cmd',default='.venv-cpu/bin/harbor run')
        s.add_argument('--delegate-agent',choices=['codex'],default='codex')
        s.add_argument('--delegate-kwargs',default='{}')
        s.add_argument('--parallel',type=int,default=2)
        s.add_argument('--batch-size',type=int,default=4)
        s.add_argument('--process-timeout',type=int,default=40000)
        s.add_argument('--selection-tier',choices=['lite','medium','heavy'],default='medium')
        s.add_argument('--selection-cost',choices=['cost_units','total_tokens'],default='total_tokens')
        s.add_argument('--candidates-per-epoch',type=int,default=2)
        s.add_argument('--dry-run',action='store_true')
        s.add_argument('--load-model-env',action='store_true',help='加载 .env 并对齐 Harbor 模型连接')
        s.add_argument('--evolver-max-tokens',type=int,default=8192)
        s.add_argument('--evolver-temperature',type=float,default=.2)
        s.add_argument('--max-proposals-per-batch',type=int,default=2)
        s.add_argument('--region',default='us-west-2')
        if name=='evaluate':
            s.add_argument('--artifact',required=True)
        else:
            s.add_argument('--mode',choices=['static-control','normal-evolve','cost-aware'],default='cost-aware')
            s.add_argument('--epochs',type=int,default=3)
            s.add_argument('--seed-workspace',default='seed_workspaces/swe')
        s.set_defaults(eval_dir=None,eval_limit=None,train_dir=None,train_limit=None)
    s=sub.add_parser('preflight')
    s.add_argument('--dataset-dir',required=True)
    return p

def main(argv=None):
    p=parser()
    args=p.parse_args(argv)
    if args.command=='preflight':
        from examples.swe_examples_new.deepswe_efficiency_train_eval.preflight import check
        return check(args.dataset_dir)
    for key in ('parallel','batch_size','process_timeout','candidates_per_epoch','epochs','evolver_max_tokens','max_proposals_per_batch'):
        if getattr(args,key,1)<=0:
            p.error(f'{key} must be positive')
    if not isinstance(json.loads(args.delegate_kwargs),dict):
        p.error('--delegate-kwargs must be a JSON object')
    if args.load_model_env and not args.dry_run:
        import os
        from dotenv import load_dotenv
        from agent_evolve.llm.factory import create_openai_client_for_model
        load_dotenv(Path(__file__).resolve().parents[3]/'.env',override=False)
        client,model,_=create_openai_client_for_model(args.model)
        os.environ['OPENAI_API_KEY']=client.api_key
        os.environ['OPENAI_BASE_URL']=str(client.base_url)
        client.close()
        args.harbor_model=model
    from examples.swe_examples_new.deepswe_efficiency_train_eval.three_way import run
    return run(args)

if __name__=='__main__':
    main()
