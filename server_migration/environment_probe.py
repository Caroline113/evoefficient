"""Run with the Python being inspected; no third-party packages required."""
import importlib.metadata as md
import json,platform,sys
from pathlib import Path
packages=[]
for d in md.distributions():
    direct=d.read_text('direct_url.json')
    packages.append({'name':d.metadata.get('Name',''),'version':d.version,'direct_url':json.loads(direct) if direct else None})
print(json.dumps({'python':platform.python_version(),'executable':sys.executable,'base_prefix':sys.base_prefix,'prefix':sys.prefix,'platform':platform.platform(),'machine':platform.machine(),'libc':platform.libc_ver(),'packages':sorted(packages,key=lambda x:x['name'].lower())},ensure_ascii=False))
