#!/usr/bin/env python3
"""Read-only environment export; sanitized locks in exports/, raw data in private/."""
import argparse,hashlib,json,os,re,shutil,subprocess,sys
from pathlib import Path
from urllib.parse import urlsplit,urlunsplit,unquote
HERE=Path(__file__).resolve().parent
REPO=HERE.parent

def run(cmd,timeout=180):
    p=subprocess.run([str(x) for x in cmd],text=True,capture_output=True,timeout=timeout)
    if p.returncode:raise RuntimeError(f'Command {cmd[0]} failed with exit {p.returncode}; raw diagnostics kept private when available')
    return p.stdout

def scrub_text(s):
    # Never publish embedded URL passwords or conda environment variables.
    s=re.sub(r'(https?://)[^\s/@]+:[^\s/@]+@',r'\1<credentials-removed>@',s)
    s=re.sub(r'(https?://)[^\s/@]+@',r'\1<credentials-removed>@',s)
    s=re.sub(r'(/t/)[A-Za-z0-9_-]{8,}(/)',r'\1<token-removed>\2',s)
    s=re.sub(r'([?&](?:token|access_token|api_key|key|password)=)[^\s&"\']+',r'\1<token-removed>',s,flags=re.I)
    return s

def write(path,text):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text,encoding='utf-8')
def dump(path,value):write(path,json.dumps(value,ensure_ascii=False,indent=2))
def fresh(path):
    if path.exists():raise ValueError(f'Existing snapshot is preserved: {path}')
    path.mkdir(parents=True)

def export_python(name,python,out,private,source_roots):
    raw=run([python,HERE/'environment_probe.py']);dump(private/name/'environment.raw.json',json.loads(raw));probe=json.loads(raw)
    locks=[];local=[];direct=[];managed=[];conda_records={}
    for rec in Path(probe['prefix']).glob('conda-meta/*.json'):
        data=json.loads(rec.read_text());conda_records[data['name'].lower().replace('_','-')]=data
    for p in probe['packages']:
        n=p['name'];u=p['direct_url'];line=f'{n}=={p["version"]}'
        norm=n.lower().replace('_','-');url=(u or {}).get('url','');path=unquote(urlsplit(url).path) if url.startswith('file:') else None
        if path and not Path(path).exists() and norm in conda_records:
            managed.append({'name':n,'python_metadata_version':p['version'],'conda_version':conda_records[norm]['version']});continue
        if norm in ('a-evolve','harbor') or (u and str(u.get('url','')).startswith('file:')):
            url=(u or {}).get('url','');path=unquote(urlsplit(url).path) if url else None
            kind='project' if n.lower().replace('_','-')=='a-evolve' else 'harbor' if n.lower()=='harbor' else 'external'
            mapping=kind if kind!='external' else n;relative='.'
            if path:
                for root_name,root in source_roots.items():
                    if Path(path).is_relative_to(root):kind=root_name;mapping=root_name;relative=str(Path(path).relative_to(root));break
            local.append({'name':n,'version':p['version'],'source_kind':kind,'original_path':path,'source_subdir':relative,'editable':(u or {}).get('dir_info',{}).get('editable',False),'required_mapping':mapping})
        elif u:
            clean=scrub_text(u.get('url',''));vcs=u.get('vcs_info',{})
            if vcs:
                url=clean if clean.startswith(vcs['vcs']+'+') else vcs['vcs']+'+'+clean
                line=f'{n} @ {url}@{vcs.get("commit_id",vcs.get("requested_revision",""))}'
            else:line=f'{n} @ {clean}'
            direct.append({'name':n,'direct_url':json.loads(scrub_text(json.dumps(u)))})
            locks.append(line)
        else:locks.append(line)
    d=out/name;d.mkdir(parents=True)
    write(d/'requirements.lock.txt','\n'.join(locks)+'\n');dump(d/'local_sources.json',local)
    public={k:v for k,v in probe.items() if k!='packages'};public['package_count']=len(probe['packages']);public['source_package_count']=len(local);public['conda_managed_build_origins']=managed
    dump(d/'runtime.json',public);dump(d/'package_versions.json',[{'name':p['name'],'version':p['version']} for p in probe['packages']]);dump(d/'direct_sources.json',direct)
    return public

def main():
    import yaml
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--name',required=True);p.add_argument('--conda',default='/home/azureuser/anaconda3/bin/conda');p.add_argument('--conda-prefix',default='/home/azureuser/anaconda3/envs/aevolve');p.add_argument('--harbor-root',default='/home/azureuser/zj/harbor');args=p.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_.-]+',args.name):p.error('Use a simple snapshot name')
    out=HERE/'exports'/args.name;private=HERE/'private'/args.name;fresh(out);fresh(private)
    paths={'project_venv':REPO/'.venv/bin/python','harbor_venv':Path(args.harbor_root)/'.venv/bin/python','aevolve':Path(args.conda_prefix)/'bin/python'}
    envs={name:export_python(name,py,out,private,{'project':REPO,'harbor':Path(args.harbor_root)}) for name,py in paths.items()}
    conda_dir=out/'aevolve'
    for name,flags in [('environment.yml',['--no-builds']),('environment-history.yml',['--from-history'])]:
        raw=run([args.conda,'env','export','--prefix',args.conda_prefix,*flags]);write(private/'aevolve'/f'{name}.raw',raw)
        parsed=yaml.safe_load(raw);parsed.pop('prefix',None);parsed.pop('variables',None)
        if isinstance(parsed.get('dependencies'),list):
            parsed['dependencies']=[d for d in parsed['dependencies'] if not isinstance(d,dict)]
        # Pip packages and local sources are restored from the separate lock.
        write(conda_dir/name,scrub_text(yaml.safe_dump(parsed,sort_keys=False)))
    explicit=run([args.conda,'list','--prefix',args.conda_prefix,'--explicit']);write(private/'aevolve'/'conda-explicit.raw.txt',explicit);write(conda_dir/'conda-linux-64-explicit.txt',scrub_text(explicit))
    commands={'docker':['docker','--version'],'nvidia':['nvidia-smi','--query-gpu=name,driver_version','--format=csv,noheader'],'uv':['uv','--version'],'git':['git','--version']}
    system={}
    for name,cmd in commands.items():
        try:system[name]=scrub_text(run(cmd,timeout=20)).strip()
        except Exception:system[name]='not available in this shell'
    dump(out/'system.json',system);dump(out/'manifest.json',{'environments':envs,'conda_prefix':args.conda_prefix,'source_root':str(REPO),'harbor_root':args.harbor_root,'notes':['Conda YAML omits variables and pip subsection; restore pip lock and source packages separately.','venvs are recreated; copying .venv to a different path is not supported.','Conda explicit is platform-specific.','Local and editable dependencies must be restored from their source trees.']})
    sums={str(f.relative_to(out)):hashlib.sha256(f.read_bytes()).hexdigest() for f in out.rglob('*') if f.is_file()};dump(out/'SHA256.json',sums)
    print(json.dumps({'snapshot':str(out),'private_raw_exports':str(private),'package_counts':{n:v['package_count'] for n,v in envs.items()}},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
