import json,tarfile,subprocess
from pathlib import Path
from server_migration import export_environments as exporter
from server_migration.create_backup import source

def test_redacts_url_password_and_token_username():
    s=exporter.scrub_text('https://u:password@example.com/a https://ghp_secret@example.com/x?token=secret /t/0123456789/chan')
    assert 'password@' not in s and 'ghp_secret' not in s and 'token=secret' not in s and '/0123456789/' not in s

def test_conda_build_paths_are_not_local_source_requirements(tmp_path,monkeypatch):
    prefix=tmp_path/'aevolve';(prefix/'conda-meta').mkdir(parents=True)
    (prefix/'conda-meta/pip.json').write_text(json.dumps({'name':'pip','version':'26.1.1'}))
    probe={'prefix':str(prefix),'python':'3.11.15','packages':[{'name':'pip','version':'26.1.1','direct_url':{'url':'file:///missing/conda/build/pip'}}]}
    monkeypatch.setattr(exporter,'run',lambda *a,**kw:json.dumps(probe))
    out=tmp_path/'out';private=tmp_path/'private'
    info=exporter.export_python('aevolve','unused',out,private,{'project':tmp_path/'project','harbor':tmp_path/'harbor'})
    assert not (out/'aevolve/requirements.lock.txt').read_text().strip()
    assert json.loads((out/'aevolve/local_sources.json').read_text())==[]
    assert info['conda_managed_build_origins'][0]['name']=='pip'

def test_harbor_subprojects_map_to_restored_harbor_root(tmp_path,monkeypatch):
    harbor=tmp_path/'harbor';pkg=harbor/'packages/rewardkit';pkg.mkdir(parents=True)
    probe={'prefix':str(tmp_path/'venv'),'python':'3.13.17','packages':[{'name':'harbor-rewardkit','version':'0.2.4','direct_url':{'url':pkg.as_uri(),'dir_info':{'editable':True}}}]}
    monkeypatch.setattr(exporter,'run',lambda *a,**kw:json.dumps(probe))
    exporter.export_python('harbor_venv','unused',tmp_path/'out',tmp_path/'private',{'project':tmp_path/'project','harbor':harbor})
    row=json.loads((tmp_path/'out/harbor_venv/local_sources.json').read_text())[0]
    assert row['required_mapping']=='harbor' and row['source_subdir']=='packages/rewardkit'

def test_source_backup_keeps_deleted_untracked_and_ignored_code(tmp_path):
    repo=tmp_path/'repo';repo.mkdir()
    def git(*a):return subprocess.check_output(['git','-C',str(repo),*a])
    git('init');(repo/'tracked.py').write_text('original\n');(repo/'delete.py').write_text('delete\n');(repo/'.gitignore').write_text('ignored-code/\n')
    git('add','-A');git('-c','user.name=Test','-c','user.email=test@example.invalid','commit','-m','initial')
    (repo/'delete.py').unlink();(repo/'tracked.py').write_text('changed\n');(repo/'new.py').write_text('new\n');(repo/'ignored-code').mkdir();(repo/'ignored-code/needed.py').write_text('needed\n')
    dest=tmp_path/'backup';source(repo,dest)
    assert 'deleted file mode' in (dest/'tracked-worktree.patch').read_text()
    with tarfile.open(dest/'untracked.tar.gz') as t:assert 'new.py' in t.getnames()
    with tarfile.open(dest/'working-tree-sources.tar.gz') as t:assert 'ignored-code/needed.py' in t.getnames()
    assert (dest/'history.bundle').stat().st_size>0
