#!/usr/bin/env python3
"""Prepare pinned TB 4.1.0 or attest an independent Harbor training directory."""
import argparse
import subprocess
import shutil
import tomllib
import sys
from pathlib import Path
if __package__ in (None,''): sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from examples.tb_examples.emergent_efficiency_train_eval.common import DATASET,CPU_DATASET,VERSION,SOURCE,load_tasks,write_json,read_json,requires_accelerator

def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == 'split':
        from examples.tb_examples.emergent_efficiency_train_eval.splits import main as split_main
        return split_main(argv[1:])
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='cmd',required=True)
    e=sub.add_parser('benchmark');e.add_argument('--output',required=True);e.add_argument('--cpu-only',action='store_true');e.add_argument('--source-dir',help='Existing clean, fully checked out pinned upstream repository')
    t=sub.add_parser('training');t.add_argument('--path',required=True);t.add_argument('--source',required=True);t.add_argument('--demo',action='store_true');args=p.parse_args(argv)
    if args.cmd=='benchmark':
        root=Path(args.output).resolve()
        if root.exists(): raise SystemExit('Destination already exists; choose a fresh directory')
        source=Path(args.source_dir).resolve() if args.source_dir else root
        if args.source_dir:
            if subprocess.check_output(['git','status','--porcelain','--untracked-files=all'],cwd=source,text=True).strip(): raise ValueError('Source repository must be clean and fully checked out')
            origin=subprocess.check_output(['git','remote','get-url','origin'],cwd=source,text=True).strip()
            if origin != SOURCE: raise ValueError('Source repository must use the pinned upstream origin')
        else:
            subprocess.run(['git','clone','--depth','1','--branch',f'v{VERSION}',SOURCE,str(root)],check=True)
        tag_commit=subprocess.check_output(['git','rev-parse',f'v{VERSION}^{{commit}}'],cwd=source,text=True).strip()
        head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()
        if head != tag_commit: raise ValueError('Source HEAD must match the pinned tag')
        if args.source_dir: root.mkdir(parents=True)
        commit=head
        source_tasks=load_tasks(source/'tasks')
        if len(source_tasks)!=66: raise ValueError('Pinned upstream must contain all 66 tasks')
        tasks_root=root/('cpu-tasks' if args.cpu_only else 'tasks')
        excluded=[]
        if args.cpu_only or args.source_dir:
            tasks_root.mkdir()
            for task in source_tasks:
                if args.cpu_only and task['requires_accelerator']:
                    excluded.append(task['id']);continue
                shutil.copytree(task['path'],tasks_root/task['id'])
        tasks=load_tasks(tasks_root)
        expected=63 if args.cpu_only else 66
        if len(tasks)!=expected: raise ValueError(f'Expected {expected} pinned tasks, found {len(tasks)}; inspect upstream GPU declarations')
        receipt={'dataset':CPU_DATASET if args.cpu_only else DATASET,'excluded_accelerator_tasks':excluded,'source':SOURCE,'tag':f'v{VERSION}','commit':commit,'tasks':{x['id']:x['content_sha256'] for x in tasks}}
        write_json(tasks_root/'dataset_receipt.json',receipt);print(tasks_root)
    else:
        root=Path(args.path).resolve();tasks=load_tasks(root)
        if (root/'dataset_receipt.json').exists() or 'terminal-bench' in args.source.lower(): raise ValueError('Use genuinely independent training tasks, not an official benchmark copy')
        write_json(root/'training_manifest.json',{'independent':True,'source':args.source,'demo':args.demo,'tasks':{x['id']:x['content_sha256'] for x in tasks}});print(root/'training_manifest.json')

if __name__=='__main__': main()

