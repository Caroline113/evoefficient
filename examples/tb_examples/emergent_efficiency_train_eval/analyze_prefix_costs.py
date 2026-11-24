#!/usr/bin/env python3
"""Reprice recorded prefixes, never simulate unobserved continuations."""
from __future__ import annotations
import argparse
import csv
import html
import json
import sys
from collections import defaultdict
from pathlib import Path
if __package__ in (None, ''): sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from examples.tb_examples.emergent_efficiency_train_eval.common import read_json, write_json, fresh_dir, sha
from examples.tb_examples.emergent_efficiency_train_eval.traces import locate_session, parse_session, prices, prefix

def compute(runs, caps, override=None):
    rows=[]; errors=[]; hashes={}; mode_tasks={}; cache={}
    for label, path in runs.items():
        path=Path(path).resolve(); result_path=path if path.is_file() else path/'results.json'; run_dir=result_path.parent
        raw=read_json(result_path); results=raw if isinstance(raw,list) else raw['results']; hashes[str(result_path)]=sha(result_path)
        ids=set(); mode_tasks[label]=ids
        for row in results:
            task=str(row.get('instance_id') or row.get('task_name') or row.get('task_id') or '')
            if not task or task in ids: raise ValueError(f'{label}: missing or duplicate task ID {task}; select one trial/seed per run')
            ids.add(task)
            try:
                if row.get('synthetic'): raise ValueError('Synthetic rollout is not an observed execution')
                if int(row.get('cumulative_cache_write_input_tokens') or 0): raise ValueError('Cache-write tokens require a separate validated prefix/price schema')
                p=locate_session(row,run_dir); key=str(p)
                if key not in cache: cache[key]=parse_session(p); hashes[key]=sha(p)
                trace=cache[key]; profile=prices(row,override)
                if trace['samples'] and row.get('total_tokens') is not None and trace['samples'][-1]['total']!=int(row['total_tokens']): raise ValueError('Session final tokens disagree with results.json; wrong session or incomplete trace')
                for k in caps:
                    z=prefix(trace,k,profile); original_success=row.get('success')
                    rows.append({'run':label,'task':task,'cap':k,**z,'original_success':original_success,'success_at_cap':original_success if not z['truncated'] else None,'success_at_cap_status':'observed_final_result' if not z['truncated'] else 'unknown_no_prefix_verifier','session':key,'round_unit':trace['round_unit'],'price_source':'override' if override is not None else 'logged_profile' if profile else 'unpriced'})
            except (OSError,ValueError,KeyError) as e: errors.append({'run':label,'task':task,'reason':str(e)})
    labels=list(runs); common=set.intersection(*(mode_tasks[m] for m in labels)); summaries=[]
    for k in caps:
        lookup={(r['run'],r['task']):r for r in rows if r['cap']==k}
        comparable={t for t in common if all((m,t) in lookup for m in labels)}
        equal={t for t in comparable if all(lookup[m,t]['reached_cap'] for m in labels)}
        for cohort,taskset in [('all_common_valid_tasks',comparable),('equal_K_exposure_tasks',equal)]:
            for m in labels:
                rr=[lookup[m,t] for t in sorted(taskset)]; all_priced=bool(rr) and all(r['cost'] is not None for r in rr)
                summaries.append({'cap':k,'cohort':cohort,'run':m,'tasks':len(rr),'task_ids':sorted(taskset),'tokens':sum(r['total'] for r in rr) if rr else None,'cost':sum(r['cost'] for r in rr) if all_priced else None,'priced_tasks':sum(r['cost'] is not None for r in rr),'finished_before_K':sum(r['rounds_available']<k for r in rr),'truncated':sum(r['truncated'] for r in rr),'original_successes':sum(r['original_success'] is True for r in rr),'original_failures':sum(r['original_success'] is False for r in rr),'unknown_success_at_cap':sum(r['success_at_cap'] is None for r in rr)})
    return {'round_unit':'model_usage_update','runs':{k:str(v) for k,v in runs.items()},'caps':caps,'common_task_ids':sorted(common),'summaries':summaries,'rows':rows,'excluded':errors,'source_sha256':hashes,'notes':['All-common-valid caps each observed rollout at min(K,L); early finishes still have fewer rounds.','Equal-K uses identical task IDs whose recorded traces all reach K; it has survivorship/selection bias and is diagnostic.','Truncated prefixes have UNKNOWN success without a verifier checkpoint; final success is not carried back to K.','Usage updates are an auditable round proxy, not tool calls or a guaranteed count of HTTP requests.','No continuation after early finish is invented; missing usage/cost is null, never zero.','This is retrospective prefix accounting, not a rerun or online budget experiment.']}

def csv_file(path,rows):
    if not rows: return
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def render_report(out,data):
    def table(rows):
        keys=['cap','cohort','run','tasks','tokens','cost','finished_before_K','truncated','unknown_success_at_cap']
        return '<table><tr>'+''.join('<th>'+k+'</th>' for k in keys)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(r.get(k)))+'</td>' for k in keys)+'</tr>' for r in rows)+'</table>'
    body='<h1>历史轨迹统一轮次预算分析</h1><p>轮次单位为“独立的模型用量更新”，不与 tool calls 混用。这里重算已观察前缀的 token 与配置费用，不推测早停后的未观察行为。</p><p>all_common_valid_tasks：相同任务上取前 min(K,L) 轮；equal_K_exposure_tasks：各模式均达到 K 轮的相同任务子集。后一口径仍有筛选偏差，不能宣称消除了成功/失败差异。</p>'+table(data['summaries'])+'<h2>数据覆盖</h2><p>排除 '+str(len(data['excluded']))+' 条无法精确重算的记录。截断位置没有 verifier 快照时，成功状态标为 unknown。</p><pre>'+html.escape(json.dumps(data['excluded'],ensure_ascii=False,indent=2))+'</pre>'
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axs=plt.subplots(1,2,figsize=(13,5),layout='constrained')
        for ax,cohort in zip(axs,['all_common_valid_tasks','equal_K_exposure_tasks']):
            for label in data['runs']:
                rr=[r for r in data['summaries'] if r['run']==label and r['cohort']==cohort and r['cost'] is not None]
                if rr: ax.plot([r['cap'] for r in rr],[r['cost'] for r in rr],marker='o',label=label)
            ax.set_title(cohort.replace('_',' '));ax.set_xlabel('Model usage-update cap K');ax.set_ylabel('Configured solver cost');ax.grid(alpha=.2);ax.legend()
        fig.savefig(out/'prefix_costs.png',dpi=180);fig.savefig(out/'prefix_costs.svg');plt.close(fig)
        svg=(out/'prefix_costs.svg').read_text();body+=svg[svg.index('<svg'):]
    except ImportError: body+='<p>matplotlib 不可用，CSV/JSON 和表格已完整输出。</p>'
    doc='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>统一轮次预算分析</title><style>body{font:15px/1.8 sans-serif;color:#24394f;max-width:1500px;margin:35px auto;padding:20px}table{border-collapse:collapse;font-size:13px;width:100%}th,td{border:1px solid #dce5ee;padding:8px}th{background:#edf3f8}svg{width:100%;height:auto}pre{white-space:pre-wrap}</style>'+body+'</html>'
    (out/'结果分析报告.html').write_text(doc,encoding='utf-8')

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',action='append',required=True,help='LABEL=directory or results.json; repeat per mode');p.add_argument('--max-rounds',type=int,nargs='+',default=[20,40,80,120]);p.add_argument('--output',required=True);p.add_argument('--prices-json',help='Explicit {input,cache,output} prices per 1M; default retains logged prices');args=p.parse_args(argv)
    if min(args.max_rounds)<=0: p.error('All round caps must be positive')
    runs={}
    for item in args.run:
        label,sep,path=item.partition('=')
        if not sep or not label or label in runs: p.error('Use unique LABEL=PATH entries')
        runs[label]=path
    if len(runs)<2: p.error('At least two runs are needed for paired comparison')
    override=read_json(args.prices_json) if args.prices_json else None
    if override is not None and (set(override)!={'input','cache','output'} or any(float(x)<0 for x in override.values())): p.error('Invalid price profile')
    out=fresh_dir(args.output);data=compute(runs,sorted(set(args.max_rounds)),override)
    write_json(out/'prefix_analysis.json',data);csv_file(out/'task_prefix_metrics.csv',data['rows']);csv_file(out/'summary.csv',data['summaries']);render_report(out,data)
    print(json.dumps({'output':str(out),'valid_rows':len(data['rows']),'excluded':len(data['excluded'])},ensure_ascii=False))
    if not data['rows']: raise SystemExit('No usable trajectories; inspect prefix_analysis.json')

if __name__=='__main__': main()
