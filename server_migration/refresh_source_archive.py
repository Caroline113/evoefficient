"""Refresh only the code archive after reviewing migration-tool changes."""
import argparse,json,tarfile,tempfile
from pathlib import Path
from server_migration.create_backup import source,digest
P=Path(__file__).resolve().parent
def main():
 p=argparse.ArgumentParser();p.add_argument('--snapshot',required=True);a=p.parse_args();out=P/'archives'/a.snapshot;manifest=json.loads((out/'manifest.json').read_text())
 with tempfile.TemporaryDirectory(dir=P/'private',prefix='sources-refresh-') as td:
  root=Path(td)/'sources'
  for name,rec in manifest['repositories'].items():rec.update(source(Path(rec['path']),root/name))
  archive=Path(td)/'source-repositories.tar.gz'
  with tarfile.open(archive,'w:gz',compresslevel=3) as t:t.add(root,arcname='sources')
  archive.replace(out/archive.name)
 f=out/'source-repositories.tar.gz';manifest['archives'][f.name]={'bytes':f.stat().st_size,'sha256':digest(f)}
 (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2));(out/'SHA256SUMS').write_text(''.join(f'{x["sha256"]}  {name}\n' for name,x in manifest['archives'].items()))
 print(json.dumps(manifest['archives'][f.name]))
if __name__=='__main__':main()
