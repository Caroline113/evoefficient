"""Verify Conda relocation in a temporary directory, without touching live envs."""
import argparse,hashlib,json,os,subprocess,sys,tarfile,tempfile
from pathlib import Path
HERE=Path(__file__).resolve().parent
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--archive',default=str(HERE/'archives/20261107-final/aevolve-conda.tar.gz'));args=p.parse_args()
    archive=Path(args.archive);private=HERE/'private';private.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='conda-restore-check-',dir=private) as td:
        dest=Path(td)
        with tarfile.open(archive) as t:t.extractall(dest,filter='data')
        py=dest/'bin/python';env=dict(os.environ);env['PATH']=str(dest/'bin')+os.pathsep+env.get('PATH','');env['CONDA_PREFIX']=str(dest)
        subprocess.run([str(py),str(dest/'bin/conda-unpack')],check=True,env=env)
        code='import json,sys,platform,yaml,openai; print(json.dumps({"python":platform.python_version(),"prefix":sys.prefix,"yaml_import":True,"openai_import":True}))'
        result=json.loads(subprocess.check_output([str(py),'-c',code],text=True,env=env))
        assert Path(result['prefix'])==dest
        result['relocation_verified']=True;result.pop('prefix')
    out=HERE/'exports/20261107';(out/'restore-check.json').write_text(json.dumps(result,indent=2))
    sums={str(f.relative_to(out)):hashlib.sha256(f.read_bytes()).hexdigest() for f in out.rglob('*') if f.is_file() and f.name!='SHA256.json'}
    (out/'SHA256.json').write_text(json.dumps(sums,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
