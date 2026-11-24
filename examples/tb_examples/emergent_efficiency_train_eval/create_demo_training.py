#!/usr/bin/env python3
"""Create independently authored smoke tasks, not a research-scale training set."""
import argparse
import json
import sys
from pathlib import Path
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from examples.tb_examples.emergent_efficiency_train_eval.common import fresh_dir,load_tasks,write_json

def create(root):
    root=fresh_dir(root)
    ledger=[{'tenant':f't{i%7}','amount':i%19-7} for i in range(1500)]
    totals={f't{i}':sum(r['amount'] for r in ledger if r['tenant']==f't{i}') for i in range(7)}
    logs=[f'INFO heartbeat {i}' for i in range(12000)];logs[137]='ERROR code=E31';logs[9345]='ERROR code=E72'
    specs=[('train-ledger-aggregate','Read /app/ledger.jsonl. Write /app/answer.json mapping each tenant to its sum of amount. Inspect only what you need, then verify the full result.',{'ledger.jsonl':'\n'.join(json.dumps(x) for x in ledger)},totals),('train-sparse-log-diagnosis','Inspect /app/server.log and write /app/answer.json containing error_count (number of ERROR lines) and first_error_line (1-based). Preserve input logs.',{'server.log':'\n'.join(logs)}, {'error_count':2,'first_error_line':138}),('train-repair-config','Read /app/settings.json. Write /app/answer.json preserving service and nested options, but set retries=3 and options.verbose=false. Do not add other keys.',{'settings.json':json.dumps({'service':'worker','retries':99,'options':{'verbose':True,'workers':4}})},{'service':'worker','retries':3,'options':{'verbose':False,'workers':4}}),('train-record-deduplicate','Read /app/records.jsonl. Write /app/answer.json as a sorted list of distinct integer id values. Do not change the input.',{'records.jsonl':'\n'.join(json.dumps({'id':i%83,'payload':'x'*25}) for i in range(3000))},list(range(83)))]
    for name,instruction,files,expected in specs:
        p=root/name;(p/'environment/files').mkdir(parents=True);(p/'tests').mkdir()
        (p/'instruction.md').write_text(instruction+'\n')
        (p/'task.toml').write_text('version = "1.0"\n[metadata]\nsource = "independently authored workflow demo"\n[agent]\ntimeout_sec = 600\n[verifier]\ntimeout_sec = 60\n[environment]\ncpus = 1\nmemory_mb = 1024\nstorage_mb = 2048\n')
        (p/'environment/Dockerfile').write_text('FROM python:3.11-slim\nWORKDIR /app\nCOPY files/ /app/\n')
        for n,data in files.items():(p/'environment/files'/n).write_text(data,encoding='utf-8')
        code='import json\nfrom pathlib import Path\ntry:\n    ok=json.loads(Path("/app/answer.json").read_text())=='+repr(expected)+'\nexcept Exception:\n    ok=False\nPath("/logs/verifier").mkdir(parents=True,exist_ok=True)\nPath("/logs/verifier/reward.txt").write_text("1" if ok else "0")\nraise SystemExit(0 if ok else 1)\n'
        (p/'tests/check.py').write_text(code);(p/'tests/test.sh').write_text('#!/bin/bash\nset -eu\npython /tests/check.py\n')
    tasks=load_tasks(root);write_json(root/'training_manifest.json',{'independent':True,'source':'independently authored four-task workflow demo','demo':True,'tasks':{t['id']:t['content_sha256'] for t in tasks}})
    return root

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);args=p.parse_args();print(create(args.output))
