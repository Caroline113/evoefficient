#!/usr/bin/env python3
"""Create private migration archives, preserving Git worktree changes."""
import argparse,hashlib,json,os,subprocess,tarfile,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
def run(cmd,**kw):return subprocess.check_output([str(x) for x in cmd],**kw)
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def source(repo,dest):
 dest.mkdir(parents=True);base=['git','-C',repo]
 run([*base,'bundle','create',dest/'history.bundle','--all'])
 (dest/'tracked-worktree.patch').write_bytes(run([*base,'diff','--binary','HEAD','--']))
 (dest/'git-status.txt').write_bytes(run([*base,'status','--short']))
 (dest/'HEAD.txt').write_bytes(run([*base,'rev-parse','HEAD']))
 paths=run([*base,'ls-files','--others','--exclude-standard','-z']).decode().split('\0');paths=[p for p in paths if p]
 with tarfile.open(dest/'untracked.tar.gz','w:gz',compresslevel=3) as t:
  for p in paths:t.add(Path(repo)/p,arcname=p,recursive=False)
 # Supplement ignored source files (the project's existing ignore rules hide
 # some algorithm/example code), without copying environments or raw data.
 blocked={'.git','.venv','venv','env','node_modules','__pycache__','.pytest_cache','.cache','logs','datasets','private','archives'}
 extensions={'.py','.sh','.md','.txt','.yml','.yaml','.toml','.json','.html','.js','.ts','.tsx','.css','.jinja','.j2'}
 with tarfile.open(dest/'working-tree-sources.tar.gz','w:gz',compresslevel=3) as t:
  for current,dirs,files in os.walk(repo):
   dirs[:]=[d for d in dirs if d not in blocked]
   for name in files:
    p=Path(current)/name
    if p.suffix in extensions and p.stat().st_size<20*1024*1024 and not name.startswith('.env') and name not in {'credentials.json','secrets.json'} and not p.is_symlink():t.add(p,arcname=str(p.relative_to(repo)),recursive=False)
 return {'path':str(repo),'head':(dest/'HEAD.txt').read_text().strip(),'untracked_files':len(paths),'supplemental_source_archive':'working-tree-sources.tar.gz'}
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--name',required=True);p.add_argument('--harbor-root',default='/home/azureuser/zj/harbor');p.add_argument('--conda',default='/home/azureuser/anaconda3/bin/conda');p.add_argument('--conda-prefix',default='/home/azureuser/anaconda3/envs/aevolve');p.add_argument('--data',action='append',default=[]);p.add_argument('--pack-conda',action='store_true');p.add_argument('--reuse-conda-archive');args=p.parse_args()
 if not args.name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-._' for c in args.name):p.error('Invalid snapshot name')
 out=HERE/'archives'/args.name
 if out.exists():raise ValueError('Archive directory exists; choose a new name')
 out.mkdir(parents=True);manifest={'created_at_unix':time.time(),'repositories':{},'data_paths':args.data,'changed_during_backup':[],'archives':{}}
 for name,repo in [('project',ROOT),('harbor',Path(args.harbor_root))]:manifest['repositories'][name]=source(repo,out/'sources'/name)
 with tarfile.open(out/'source-repositories.tar.gz','w:gz',compresslevel=3) as t:t.add(out/'sources',arcname='sources')
 if args.data:
  with tarfile.open(out/'experiment-data.tar.gz','w:gz',compresslevel=3) as t:
   for name in args.data:
    path=(ROOT/name).resolve()
    if not path.is_relative_to(ROOT.resolve()) or path==ROOT.resolve():raise ValueError('Data paths must be explicit subdirectories of the project')
    if not path.exists():raise ValueError(f'Missing data path: {path}')
    paths=[path]+sorted(path.rglob('*')) if path.is_dir() else [path]
    for f in paths:
     a=f.stat(follow_symlinks=False);t.add(f,arcname=str(f.relative_to(ROOT)),recursive=False);b=f.stat(follow_symlinks=False)
     if (a.st_size,a.st_mtime_ns)!=(b.st_size,b.st_mtime_ns):manifest['changed_during_backup'].append(str(f.relative_to(ROOT)))
 if args.pack_conda:
  exe=Path(args.conda_prefix)/'bin/conda-pack'
  env=dict(os.environ);env['PATH']=str(Path(args.conda).parent)+os.pathsep+env.get('PATH','');env['CONDA_EXE']=args.conda
  log=out/'conda-pack.log'
  with log.open('w') as f:
   result=subprocess.run([str(exe),'-p',args.conda_prefix,'-o',str(out/'aevolve-conda.tar.gz')],stdout=f,stderr=subprocess.STDOUT,env=env)
  manifest['conda_pack_success']=result.returncode==0
  if result.returncode:manifest['conda_pack_error']='Inspect private conda-pack.log; no ignore-missing or ignore-editable option was applied'
 if args.reuse_conda_archive:
  import shutil
  shutil.copy2(args.reuse_conda_archive,out/'aevolve-conda.tar.gz');manifest['conda_pack_success']=True;manifest['reused_conda_archive']=args.reuse_conda_archive
 for f in out.glob('*.tar.gz'):manifest['archives'][f.name]={'bytes':f.stat().st_size,'sha256':digest(f)}
 dump(out/'manifest.json',manifest)
 (out/'SHA256SUMS').write_text(''.join(f'{x["sha256"]}  {name}\n' for name,x in manifest['archives'].items()))
 print(json.dumps({'output':str(out),'archives':manifest['archives'],'conda_pack_success':manifest.get('conda_pack_success'),'changed_files':len(manifest['changed_during_backup'])},indent=2))
if __name__=='__main__':main()
