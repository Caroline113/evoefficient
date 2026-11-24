#!/usr/bin/env python3
"""Plan or run a one-training-task/one-evaluation-task CPU workflow demo."""
import argparse
import os
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['plan', 'run'], nargs='?', default='plan')
    p.add_argument('--mode', choices=['static-control', 'normal-evolve', 'cost-aware'], default='cost-aware')
    p.add_argument('--eval-dir', default='datasets/terminal-bench-4.1.0-cpu/cpu-tasks')
    p.add_argument('--output')
    a = p.parse_args()
    os.chdir(ROOT)
    from dotenv import load_dotenv
    load_dotenv(ROOT/'.env', override=False)
    from agent_evolve.llm.factory import create_openai_client_for_model
    client, model, provider = create_openai_client_for_model('deepseek:deepseek-flash')
    # Scope solver authentication to this process; do not modify .env or Codex auth.
    os.environ['OPENAI_API_KEY'] = client.api_key
    os.environ['OPENAI_BASE_URL'] = str(client.base_url)
    client.close()
    print(f'DEMO ONLY: 1 authored training task + 1 CPU evaluation task; model={model}, provider={provider}', flush=True)
    harbor = ROOT.parent/'harbor'
    args = [a.action, '--train-dir', str(ROOT/'datasets/efficiency-train-demo'),
            '--eval-dir', str(Path(a.eval_dir).resolve()), '--allow-demo-training',
            '--mode', a.mode, '--train-limit', '1', '--eval-limit', '1',
            '--epochs', '1', '--batch-size', '1', '--parallel', '1',
            '--model', 'deepseek:deepseek-flash', '--evolver-model', 'deepseek:deepseek-flash',
            '--harbor-model', model, '--harbor-repo', str(harbor),
            '--harbor-cmd', str(harbor/'.venv-cpu/bin/harbor')+' run']
    if a.action == 'run':
        output = a.output or f'logs/tb41_cpu_demo/{a.mode}_{uuid.uuid4().hex[:12]}'
        args += ['--output', output]
    from examples.tb_examples.emergent_efficiency_train_eval.run_experiment import main as experiment
    experiment(args)

if __name__ == '__main__':
    main()
