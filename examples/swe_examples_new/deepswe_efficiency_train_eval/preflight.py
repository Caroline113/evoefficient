"""在 Harbor 虚拟环境检查全部真实任务的配置兼容性。"""
import json
from pathlib import Path
import tomllib
from .common import load_tasks,validate_receipt
def check(root):
    from harbor.models.task.config import TaskConfig
    from harbor.models.job.config import JobConfig
    from .backend import config_for
    from .harbor_agent import DeepSWECodexAgent
    from harbor.agents.installed.base import BaseInstalledAgent
    from harbor.agents.installed.codex import Codex
    if not issubclass(DeepSWECodexAgent, BaseInstalledAgent) or DeepSWECodexAgent.MODEL_CONNECTION is not Codex.MODEL_CONNECTION:
        raise ValueError('Native Codex network/model capabilities were lost')
    from types import SimpleNamespace
    import tempfile
    tasks=load_tasks(root)
    validate_receipt(root,tasks)
    for task in tasks:
        cfg=tomllib.loads((Path(task['path'])/'task.toml').read_text())
        parsed=TaskConfig.model_validate(cfg)
        if parsed.verifier.environment_mode != 'separate' or not parsed.verifier.collect:
            raise ValueError('Harbor lost v1.1 verifier configuration')
    with tempfile.TemporaryDirectory() as td:
        root=Path(td)
        prompt=root/'prompt.md'
        prompt.write_text('schema check')
        args=SimpleNamespace(harbor_model='schema-check',delegate_agent='codex',delegate_kwargs='{}')
        JobConfig.model_validate(config_for(tasks[0],root,prompt,root/'jobs','schema-check',args))
    print(json.dumps(dict(tasks=len(tasks),separate_verifier=True,harbor_config_valid=True)))
