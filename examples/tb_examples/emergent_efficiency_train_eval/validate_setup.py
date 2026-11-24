#!/usr/bin/env python3
"""Offline schema/integration checks; execute with the Harbor environment's Python."""
import importlib.metadata
import json
import sys
import tempfile
import tomllib
from pathlib import Path
from types import SimpleNamespace
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from examples.tb_examples.emergent_efficiency_train_eval.backend import config_for

def main():
    from harbor.models.job.config import JobConfig
    from harbor.models.task.config import TaskConfig
    from examples.tb_examples.emergent_efficiency_train_eval.create_demo_training import create
    with tempfile.TemporaryDirectory() as td:
        root=Path(td);demo=create(root/'demo');pp=root/'prompt.md';pp.write_text('Seed prompt')
        args=SimpleNamespace(harbor_model='deepseek-flash',delegate_agent='codex',delegate_kwargs='{}')
        cfg=config_for({'path':str(demo/'train-ledger-aggregate'),'id':'train-ledger-aggregate'},root,pp,root/'jobs','schema-check',args)
        parsed=JobConfig.model_validate(cfg)
        assert len(parsed.tasks)==1 and Path(parsed.tasks[0].path)==(demo/'train-ledger-aggregate').resolve()
        for p in demo.glob('*/task.toml'):TaskConfig.model_validate(tomllib.loads(p.read_text()))
    fields=TaskConfig.model_fields
    verifier_type=fields['verifier'].annotation
    verifier_fields=getattr(verifier_type,'model_fields',{})
    assert 'environment_mode' in verifier_fields, 'Harbor must support TB4.1 separate verifier environments'
    print(json.dumps({'harbor_job_config_valid':True,'demo_task_configs_valid':True,'python':sys.executable,'task_config_fields':list(fields),'verifier_config_fields':list(verifier_fields),'tb41_separate_verifier_supported':'environment_mode' in verifier_fields},indent=2))

if __name__=='__main__':main()
