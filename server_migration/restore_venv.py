#!/usr/bin/env python3
"""Plan or recreate a venv at a new path from a sanitized environment snapshot."""
import argparse,json,os,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--snapshot',required=True);p.add_argument('--environment',choices=['project_venv','harbor_venv'],required=True);p.add_argument('--python',required=True);p.add_argument('--destination',required=True);p.add_argument('--project-root',required=True);p.add_argument('--harbor-root',required=True);p.add_argument('--source',action='append',default=[],help='Extra local package NAME=/path');p.add_argument('--wheelhouse');p.add_argument('--execute',action='store_true');a=p.parse_args()
 snap=Path(a.snapshot)/a.environment;runtime=json.loads((snap/'runtime.json').read_text());locals=json.loads((snap/'local_sources.json').read_text())
 version=subprocess.check_output([a.python,'--version'],text=True).strip().split()[-1]
 if version!=runtime['python']:raise ValueError(f'Expected Python {runtime["python"]}, got {version}; supply the matching interpreter')
 dest=Path(a.destination)
 if dest.exists():raise ValueError('Destination already exists; this tool never overwrites an environment')
 roots={'project':a.project_root,'harbor':a.harbor_root}
 for x in a.source:
  n,sep,path=x.partition('=')
  if not sep:raise ValueError('Use NAME=/path for local source mappings')
  roots[n]=path
 missing=[x['required_mapping'] for x in locals if x['required_mapping'] not in roots or not Path(roots[x['required_mapping']]).exists()]
 if missing:raise ValueError(f'Missing local source mappings: {missing}')
 py=dest/'bin/python';cmds=[[a.python,'-m','venv',str(dest)],[str(py),'-m','pip','install','-r',str(snap/'requirements.lock.txt')]]
 if a.wheelhouse:cmds[1].extend(['--no-index','--find-links',a.wheelhouse])
 for x in locals:
  source=Path(roots[x['required_mapping']])/x.get('source_subdir','.')
  cmd=[str(py),'-m','pip','install','--no-deps','-e',str(source)]
  if a.wheelhouse:cmd.extend(['--no-index','--find-links',a.wheelhouse])
  cmds.append(cmd)
 cmds.append([str(py),'-m','pip','check'])
 print(json.dumps({'commands':cmds,'execute':a.execute},indent=2))
 if a.execute:
  for cmd in cmds:subprocess.run(cmd,check=True)
if __name__=='__main__':main()
