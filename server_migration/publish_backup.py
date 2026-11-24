#!/usr/bin/env python3
"""Review/publish a private backup branch without changing HEAD or the real index."""
import argparse,hashlib,json,os,re,subprocess,tempfile
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;REPO='DeepSoftwareAnalytics/EvolveEfficient'
def call(cmd,*,env=None,input=None,check=True):
 p=subprocess.run(cmd,cwd=ROOT,env=env,input=input,text=True,capture_output=True)
 if check and p.returncode:raise RuntimeError(f'{cmd[0]} failed ({p.returncode}): {p.stderr[:1800]}')
 return p.stdout.strip()
def git(*args,**kw):return call(['git',*args],**kw)
def h(path):
 if not path.exists():return None
 digest=hashlib.sha256()
 with path.open('rb') as f:
  for data in iter(lambda:f.read(1024*1024),b''):digest.update(data)
 return digest.hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--snapshot',required=True);p.add_argument('--branch',default='backup/server-20261107');p.add_argument('--tag',default='migration-20261107');p.add_argument('--execute',action='store_true');a=p.parse_args()
 info=json.loads(call(['gh','repo','view',REPO,'--json','nameWithOwner,isPrivate,viewerPermission']))
 if not info['isPrivate'] or info['viewerPermission'] not in ('WRITE','ADMIN','MAINTAIN'):raise ValueError('Private repository with write permission required for these raw migration assets')
 assets=HERE/'archives'/a.snapshot;manifest=json.loads((assets/'manifest.json').read_text())
 if manifest.get('changed_during_backup') or not manifest.get('conda_pack_success'):raise ValueError('Archive consistency or Conda pack check failed')
 for name,m in manifest['archives'].items():
  path=assets/name
  if path.stat().st_size>=2*1024**3 or h(path)!=m['sha256']:raise ValueError(f'Invalid Release asset {name}')
 real_index=Path(git('rev-parse','--git-path','index'));real_index=real_index if real_index.is_absolute() else ROOT/real_index
 index_before=h(real_index);head=git('rev-parse','HEAD');branch_before=git('branch','--show-current')
 with tempfile.TemporaryDirectory(prefix='migration-index-') as td:
  env=dict(os.environ);env['GIT_INDEX_FILE']=str(Path(td)/'index')
  git('read-tree','HEAD',env=env);git('add','--all','--','.',env=env)
  changed=git('diff','--cached','--name-only','-z',env=env).split('\0');changed=[x for x in changed if x]
  scan=[];total=0
  for name in changed:
   path=ROOT/name
   if not path.is_file():continue
   if path.stat().st_size>=100*1024**2:raise ValueError(f'File requires LFS/Release, not Git: {name}')
   if name.startswith(('logs/','server_migration/private/','server_migration/archives/')):raise ValueError(f'Raw data was unexpectedly staged: {name}')
   if path.name.startswith('.env') and path.name not in ('.env.example','.env.template'):raise ValueError(f'Environment secret file staged: {name}')
   total+=path.stat().st_size
   if path.stat().st_size>4*1024**2:continue
   data=path.read_bytes()
   if b'\0' in data:continue
   text=data.decode('utf-8',errors='replace')
   if re.search(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|gh[pousr]_[A-Za-z0-9]{25,}|sk-[A-Za-z0-9_-]{30,}',text):scan.append(name)
  if scan:raise ValueError(f'Possible credentials require review before GitHub upload: {scan}')
  tree=git('write-tree',env=env)
  plan_path=HERE/'private'/a.snapshot/'publish-plan.json';plan_path.parent.mkdir(parents=True,exist_ok=True)
  if a.execute:
   plan=json.loads(plan_path.read_text())
   if plan['head']!=head or plan['tree']!=tree:raise ValueError('Working tree changed after review; regenerate the plan')
   commit=plan['commit']
  else:
   user=json.loads(call(['gh','api','user']));env['GIT_AUTHOR_NAME']=user['login'];env['GIT_COMMITTER_NAME']=user['login'];email=f'{user["id"]}+{user["login"]}@users.noreply.github.com';env['GIT_AUTHOR_EMAIL']=email;env['GIT_COMMITTER_EMAIL']=email
   commit=git('commit-tree',tree,'-p',head,env=env,input='Back up server code and migration manifests (2026-11-07)\n')
   plan={'repository':REPO,'head':head,'tree':tree,'commit':commit,'branch':a.branch,'tag':a.tag,'changed_files':changed,'git_payload_file_bytes':total,'assets':manifest['archives']};plan_path.write_text(json.dumps(plan,ensure_ascii=False,indent=2))
 if h(real_index)!=index_before or git('rev-parse','HEAD')!=head or git('branch','--show-current')!=branch_before:raise ValueError('Original Git state unexpectedly changed')
 if a.execute:
  git('config','--local','--replace-all','credential.https://github.com.helper','')
  git('config','--local','--add','credential.https://github.com.helper','!gh auth git-credential')
  git('remote','set-url','origin',f'https://github.com/{REPO}.git')
  git('push','origin',f'{commit}:refs/heads/{a.branch}')
  help_text=call(['gh','release','create','--help'])
  cmd=['gh','release','create',a.tag,'--repo',REPO,'--draft','--title','Server migration backup 2026-11-07','--notes-file',str(HERE/'README.md')]
  if '--target' in help_text:cmd.extend(['--target',a.branch])
  else:git('push','origin',f'{commit}:refs/tags/{a.tag}')
  call(cmd)
  call(['gh','release','upload',a.tag,'--repo',REPO,*[str(assets/name) for name in manifest['archives']],str(assets/'SHA256SUMS'),str(assets/'manifest.json')])
  releases=json.loads(call(['gh','api',f'repos/{REPO}/releases?per_page=100']))
  release=next(r for r in releases if r['tag_name']==a.tag)
  expected={name:m['bytes'] for name,m in manifest['archives'].items()}
  actual={x['name']:x['size'] for x in release['assets']}
  if any(actual.get(name)!=size for name,size in expected.items()):raise ValueError('Uploaded assets did not match local sizes')
  for asset in release['assets']:
   if asset['name'] in manifest['archives'] and asset.get('digest'):
    if asset['digest']!='sha256:'+manifest['archives'][asset['name']]['sha256']:raise ValueError('Remote asset digest mismatch')
  status={'branch_url':f'https://github.com/{REPO}/tree/{a.branch}','release_url':release['html_url'],'draft':release['draft'],'commit':commit,'assets':actual,'head_unchanged':git('rev-parse','HEAD')==head,'index_unchanged':h(real_index)==index_before}
  (HERE/'private'/a.snapshot/'publish-result.json').write_text(json.dumps(status,indent=2));print(json.dumps(status,indent=2))
 else:print(json.dumps({'plan':str(plan_path),'commit':commit,'changed_file_count':len(changed),'git_payload_file_bytes':total,'first_files':changed[:40],'main_and_index_unchanged':True},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
