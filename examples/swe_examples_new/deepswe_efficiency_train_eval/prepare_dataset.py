"""准备固定版本的 DeepSWE v1.1 数据并生成三分清单。"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path
if __package__ in (None, ''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from examples.swe_examples_new.deepswe_efficiency_train_eval.common import *

def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == 'split':
        from examples.swe_examples_new.deepswe_efficiency_train_eval.splits import main as split
        return split(argv[1:])
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['benchmark'])
    p.add_argument('--output', required=True)
    p.add_argument('--source-dir', help='已检出固定官方提交的干净仓库')
    args=p.parse_args(argv)
    output=Path(args.output).resolve()
    if output.exists():
        raise ValueError('Output must be a new directory')
    if args.source_dir:
        source=Path(args.source_dir).resolve()
    else:
        source=output/'source'
        source.mkdir(parents=True)
        subprocess.run(['git','init',str(source)],check=True)
        subprocess.run(['git','remote','add','origin',SOURCE],cwd=source,check=True)
        subprocess.run(['git','fetch','--depth','1','origin',REVISION],cwd=source,check=True)
        subprocess.run(['git','checkout','--detach','FETCH_HEAD'],cwd=source,check=True)
    def git(*a):
        return subprocess.check_output(['git',*a],cwd=source,text=True).strip()
    if git('remote','get-url','origin') != SOURCE or git('rev-parse','HEAD') != REVISION:
        raise ValueError('Source must match pinned official DeepSWE revision')
    if git('status','--porcelain','--untracked-files=all'):
        raise ValueError('Source checkout must be clean')
    tasks=load_tasks(source/'tasks')
    receipt=dict(dataset=DATASET,source=SOURCE,revision=REVISION,
                 tasks={t['id']:t['content_sha256'] for t in tasks})
    target=output/'tasks'
    shutil.copytree(source/'tasks',target)
    write_json(target/'dataset_receipt.json',receipt)
    validate_receipt(target,load_tasks(target))
    print(target)

if __name__=='__main__':
    main()
