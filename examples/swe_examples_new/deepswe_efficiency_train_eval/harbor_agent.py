"""DeepSWE 策略注入：保留 Harbor 原生 Codex 的安装和网络能力。"""
from pathlib import Path
from harbor.agents.installed.codex import Codex

class DeepSWECodexAgent(Codex):
    """Subclassing preserves BaseInstalledAgent inference-host allowlisting."""
    def __init__(self, logs_dir, model_name=None, *, system_prompt_path, **kwargs):
        self.system_prompt_path = Path(system_prompt_path)
        if not self.system_prompt_path.is_file():
            raise ValueError('Frozen strategy prompt is missing')
        super().__init__(logs_dir=logs_dir, model_name=model_name, **kwargs)

    async def run(self, instruction, environment, context):
        strategy = self.system_prompt_path.read_text(encoding='utf-8')
        prompt = ('# DeepSWE v1.1 task\n' + instruction.strip()
                  + '\n\n# Frozen A-Evolve strategy\n' + strategy)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        (self.logs_dir/'deepswe_strategy_prompt.md').write_text(prompt,encoding='utf-8')
        await super().run(prompt,environment,context)
