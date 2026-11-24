import collections
import json
import os
import re
import threading
import time
from datetime import datetime,timezone
from pathlib import Path
from .core import VERSION, Resolver, boolean, family, first, load_rows, normalize, number, redact, summarize, uid

PRUNE={'harbor_jobs','raw_harbor_results','conversations','sessions','workspace','strategies','system_prompts','harbor_inputs','inputs','node_modules','.git','__pycache__','figures','artifacts','emergence'}
RESULTS=['results.json','results.jsonl','interrupted_tasks.jsonl','per_instance_metrics.csv']
MARKERS=set(RESULTS+['summary.json','summary.csv','results.metrics.json','experiment_manifest.json','RUN_COMPLETE.json','run_status.json','prefix_analysis.json','pipeline_summary.json'])

class Store:
    def __init__(self,root,cache):
        self.root=Path(root).resolve();self.resolver=Resolver(root);self.cache=Path(cache).resolve()
        if self.cache.is_relative_to(self.root/'logs'):raise ValueError('Cache must be outside logs')
        self.cache.mkdir(parents=True,exist_ok=True);self.lock=threading.Lock();self.cancel=threading.Event();self.files={};self.trace_lock=threading.Lock()
        self.state={'scanning':False,'visited':0,'runs':0,'error':None,'last_scan':None}
        self.data={'schema_version':VERSION,'runs':[],'tasks':{},'issues':[],'analyses':[],'inventory':[]}
        try:
            obj=json.loads((self.cache/'index.json').read_text())
            if obj.get('schema_version')==VERSION and obj.get('root')==str(self.root):self.data=obj;self.state['last_scan']=obj.get('scanned_at')
        except (OSError,ValueError):pass
    def relative(self,p):return self.resolver.relative(p)
    def issue(self,kind,p,**extra):return {'kind':kind,'source':self.relative(p),**extra}
    def read(self,p,issues):
        try:
            self.resolver.safe(p)
            return load_rows(p,issues)
        except (ValueError,OSError) as e:
            issues.append({'kind':'unreadable_or_denied','source':str(p),'detail':type(e).__name__});return [],[]
    def object(self,p,issues):
        if not p.exists():return {}
        rows,_=self.read(p,issues)
        return rows[0] if rows else {}
    def start_scan(self):
        with self.lock:
            if self.state['scanning']:return False
            self.state.update(scanning=True,visited=0,runs=0,error=None);self.cancel.clear()
        threading.Thread(target=self.scan,daemon=True).start();return True
    def scan(self):
        result={'schema_version':VERSION,'root':str(self.root),'runs':[],'tasks':{},'issues':[],'analyses':[],'inventory':[]}
        issues=result['issues'];families=set()
        try:
            for directory,dirs,names in os.walk(self.root/'logs',followlinks=False):
                if self.cancel.is_set():raise RuntimeError('scan_cancelled_previous_snapshot_kept')
                p=Path(directory);self.state['visited']+=1
                dirs[:]=[x for x in dirs if x not in PRUNE and not x.startswith('.') and not (p/x).is_symlink()]
                if len(p.relative_to(self.root/'logs').parts)>14:dirs[:]=[];issues.append(self.issue('depth_limit',p));continue
                generic=[]
                for name in names:
                    q=p/name
                    if q.suffix not in ('.json','.jsonl','.csv') or name.startswith('.'):continue
                    if name in MARKERS:continue
                    try:
                        self.resolver.safe(q)
                        with q.open(encoding='utf-8-sig',errors='replace') as f:snippet=f.read(16384)
                        if re.search(r'instance_id|task_name|task_id',snippet):generic.append(q)
                    except (OSError,ValueError):continue
                if not set(names)&MARKERS and not generic:continue
                rel=self.relative(p)
                if 'prefix_analysis.json' in names:
                    obj=self.object(p/'prefix_analysis.json',issues)
                    result['analyses'].append({'id':uid(rel),'path':rel,'type':'prefix','caps':obj.get('caps',[]),'source_runs':obj.get('runs',{}),'rows':len(obj.get('rows',[])),'excluded':len(obj.get('excluded',[]))})
                    result['inventory'].append({'path':rel,'role':'derived_analysis','format':'prefix JSON/CSV'});dirs[:]=[];continue
                if not any(x in names for x in ['results.json','results.jsonl','experiment_manifest.json']) and (any(x in names for x in ['overall_comparison.csv','task_matrix.csv','S4_COMPARISON_REPORT.md']) or 'comparison' in p.name.lower()):
                    result['analyses'].append({'id':uid(rel),'path':rel,'type':'report','caps':[],'source_runs':{},'rows':None,'excluded':None})
                    result['inventory'].append({'path':rel,'role':'derived_report','format':'CSV/Markdown/HTML'});continue
                manifest=self.object(p/'experiment_manifest.json',issues)
                # Train/eval children belong to their protocol parent, not independent duplicate runs.
                parent=p.parent
                protocol_parent=parent if p.name in ('train','eval') and (parent/'experiment_manifest.json').exists() and ('train' in set(x.name for x in parent.iterdir() if x.is_dir())) else None
                if protocol_parent:continue
                paths=list(dict.fromkeys([p/n for n in RESULTS if (p/n).exists()]+generic))
                if (p/'summary.csv').exists():paths.append(p/'summary.csv')
                for phase in ['train','eval']:
                    paths.extend(p/phase/n for n in ['results.json','results.jsonl'] if (p/phase/n).exists())
                tasks=[];seen={};run_id=uid(rel);formats=[];offset=0
                for source in paths:
                    before=source.stat();rows,locations=self.read(source,issues);formats.append(source.suffix)
                    occurrence=collections.Counter()
                    for i,(r,loc) in enumerate(zip(rows,locations)):
                        t=normalize(r,run_id,self.relative(source),loc,i,manifest)
                        if not t:continue
                        t['source_size']=before.st_size;t['source_mtime_ns']=before.st_mtime_ns
                        key=json.dumps(t['identity'],sort_keys=True,default=str)
                        occurrence[key]+=1;key+=(f':occurrence:{occurrence[key]}')
                        if key in seen:
                            old=seen[key];old['sources'].append({'source':t['source'],'location':loc})
                            if any(old.get(k)!=t.get(k) for k in ['success','cost','tokens']):issues.append(self.issue('conflicting_result_sources',source,task=t['task'],location=loc))
                            continue
                        t['id']=uid([run_id,key]);seen[key]=t;tasks.append(t)
                    after=source.stat()
                    if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):issues.append(self.issue('changed_during_scan',source))
                summary={}
                for name in ['summary.json','results.metrics.json','RUN_COMPLETE.json','summary.csv']:
                    q=p/name
                    if q.exists():
                        rr,_=self.read(q,issues);summary[name]=redact(rr[:50])
                bench,provenance=family(rel,manifest)
                run={'id':run_id,'path':rel,'name':p.name,'family':bench,'family_source':provenance,
                    'version':first(manifest,'dataset','benchmark'),'mode':manifest.get('mode') or next((t['mode'] for t in tasks if t['mode']),None),
                    'models':sorted(set(str(t['model']) for t in tasks if t['model'])),
                    'date':first(manifest,'created_at','created'),'date_source':'manifest' if first(manifest,'created_at','created') else 'unknown',
                    'summary':summarize(tasks),'summary_only':not tasks,'formats':sorted(set(formats)),
                    'identity':{k:manifest.get(k) for k in ['benchmark','dataset','seed','repo_revision','task_count','protocol'] if k in manifest},
                    'role':'run' if tasks else 'summary_only_or_candidate','manifest_present':bool(manifest)}
                if not tasks and not summary and not manifest:
                    issues.append(self.issue('unrecognized_candidate',p));continue
                result['runs'].append(redact(run));result['tasks'][run_id]=tasks
                result['inventory'].append({'path':rel,'role':run['role'],'format':','.join(run['formats']),'tasks':len(tasks)})
                self.state['runs']=len(result['runs']);families.add(rel.split('/')[1] if '/' in rel else rel)
            for top in (self.root/'logs').iterdir():
                if top.is_dir() and not any(r['path']==self.relative(top) or r['path'].startswith(self.relative(top)+'/') for r in result['runs']) and not any(a['path'].startswith(self.relative(top)) for a in result['analyses']):
                    issues.append(self.issue('unrecognized_directory',top))
            for issue in issues:
                src=issue.get('source','')
                try:issue['source']=self.relative(Path(src)) if Path(src).is_absolute() else src
                except ValueError:issue['source']='[outside root]'
            result['issues']=redact(issues);result['scanned_at']=datetime.now(timezone.utc).isoformat();result['family_directories']=len(families)
            temp=self.cache/'index.json.tmp';temp.write_text(json.dumps(result,ensure_ascii=False,allow_nan=False));temp.replace(self.cache/'index.json')
            with self.lock:self.data=result;self.state['last_scan']=result['scanned_at']
        except Exception as e:self.state['error']=f'{type(e).__name__}: {e}'
        finally:self.state['scanning']=False
    def run(self,rid):
        r=next((r for r in self.data['runs'] if r['id']==rid),None)
        if not r:raise KeyError('run_not_found')
        return r
    def task(self,rid,tid):
        self.run(rid)
        t=next((t for t in self.data['tasks'].get(rid,[]) if t['id']==tid),None)
        if not t:raise KeyError('task_not_found')
        return t
    def raw_task(self,t):
        path=self.resolver.safe(self.root/t['source']);issues=[]
        stat=path.stat()
        if 'source_mtime_ns' in t and (t['source_size'],t['source_mtime_ns'])!=(stat.st_size,stat.st_mtime_ns):raise ValueError('source_changed_rescan_required')
        rows,locations=self.read(path,issues)
        if t['location'] not in locations:raise ValueError('source_changed_rescan_required')
        row=rows[locations.index(t['location'])]
        if str(first(row,'instance_id','task_name','task_id','instance','task') or '')!=t['task']:raise ValueError('source_changed_rescan_required')
        return row
    def register(self,p):
        p=self.resolver.safe(p)
        if not p.is_file():raise ValueError('not_a_file')
        key=uid(self.relative(p));self.files[key]=p
        return {'id':key,'path':self.relative(p),'bytes':p.stat().st_size,'kind':p.suffix.lower()}
    def detail(self,rid,tid):
        run=self.run(rid);t=self.task(rid,tid);raw=self.raw_task(t);base=self.root/run['path'];issues=[];files=[];mappings=[];trials=[]
        for key in ['conversation_path','system_prompt_path','raw_harbor_result_path','raw_job_path','result_json_path','trajectory_path','raw_result_path']:
            if not raw.get(key):continue
            p,method=self.resolver.resolve(raw[key],base);mappings.append({'field':key,'original':redact(raw[key]),'resolved':self.relative(p) if p else None,'method':method})
            if p and p.is_dir() and key=='raw_job_path':trials.append(p)
            elif p and p.is_file():files.append(self.register(p))
            elif not p:issues.append({'kind':'unresolved_reference','field':key})
        # Bounded fallback in this run only; never pick an arbitrary match.
        if not trials:
            matches=list((base/'harbor_jobs').glob('*/'+t['task']+'__*')) if re.fullmatch(r'[\w.\-]+',t['task']) else []
            if len(matches)==1:trials=matches
            elif len(matches)>1:issues.append({'kind':'ambiguous_trials','count':len(matches)})
        sessions=[]
        for trial in trials:
            for name in ['result.json','config.json','trial.log','agent/codex.txt','agent/a_evolve_terminal_adapter_prompt.md','verifier/test-stdout.txt','verifier/ctrf.json','verifier/reward.txt','artifacts/manifest.json']:
                p=trial/name
                if p.is_file():
                    entry=self.register(p);files.append(entry)
                    if entry['bytes']==0:issues.append({'kind':'empty_file','source':entry['path']})
            sessions.extend(p for p in (trial/'agent'/'sessions').glob('**/*.jsonl') if p.is_file())
            # Screenshot metadata only; bytes are loaded on demand.
            for folder in ['agent','artifacts']:
                count=0
                for p in (trial/folder).rglob('*'):
                    if p.is_file() and p.suffix.lower() in ('.png','.jpg','.jpeg','.webp'):
                        files.append(self.register(p));count+=1
                        if count>=100:break
        for name in [f'patch_{t["task"]}.diff']:
            if Path(name).name==name and (base/name).is_file():files.append(self.register(base/name))
        # Other agents may store screenshots, trajectories and evidence in raw result directories.
        for entry in list(files):
            if entry['kind']=='.jsonl' and 'rollout-' in entry['path']:sessions.append(self.files[entry['id']])
        sessions=list(dict.fromkeys(sessions))
        if len(sessions)>1:issues.append({'kind':'multiple_sessions_select_explicitly','count':len(sessions)})
        session_files=[self.register(p) for p in sessions]
        for entry in session_files:
            if entry not in files:files.append(entry)
        files=list({f['id']:f for f in files}.values())
        # Find definitions by exact task identity in bounded allowed roots. Multiple versions stay explicit.
        definitions=[]
        if re.fullmatch(r'[\w.\-]+',t['task']):
            for root in [self.root/'datasets',self.root/'agent_evolve'/'benchmarks']:
                for pattern in [f'*/tasks/{t["task"]}/instruction.md',f'*/{t["task"]}/instruction.md',f'{t["task"]}/instruction.md']:
                    for p in root.glob(pattern):
                        try:definitions.append(self.register(p))
                        except ValueError:continue
        definitions=list({x['id']:x for x in definitions}.values())
        if len(definitions)>1:issues.append({'kind':'ambiguous_task_definition','detail':'存在不同版本定义，请依据清单核对后选择'})
        evidence=[]
        for key in ['error','feedback_detail','detail','eval_output','timed_out','returncode']:
            if raw.get(key) not in (None,'',False):evidence.append({'source':t['source'],'location':t['location']+'/'+key,'text':str(redact(raw[key]))[:6000]})
        for entry in files:
            if entry['bytes'] and entry['bytes']<=4*1024*1024 and any(s in entry['path'] for s in ['/verifier/','trial.log','stderr','raw_harbor_results','agent/codex.txt']):
                p=self.files[entry['id']];text=redact(p.read_text(encoding='utf-8',errors='replace'))
                lines=text.splitlines()
                matches=[(n,s) for n,s in enumerate(lines,1) if re.search(r'error|fail|exception|timeout|traceback|401|403|429|connection|assert',s,re.I)]
                for n,line in matches[:10]:evidence.append({'file_id':entry['id'],'source':entry['path'],'location':f'line:{n}','text':line[:1200]})
                if entry['path'].endswith('reward.txt'):evidence.append({'source':entry['path'],'location':'line:1','text':'verifier reward: '+text[:100]})
        rules=[('authentication',r'\b401\b|unauthorized|invalid.api.key','认证被拒绝；具体凭据原因未确定'),('rate_limit',r'\b429\b|rate.limit','请求受到限流'),('network',r'connection.reset|connection.refused|could.not.resolve|network.unreachable','观察到网络连接错误'),('timeout',r'timed.out|timeout','观察到超时'),('container',r'docker.*error|container.*failed|image.*not.found','观察到容器或镜像错误'),('assertion',r'assertionerror|assert.*failed','观察到断言未通过')]
        diagnoses=[]
        for code,pattern,label in rules:
            hits=[i for i,e in enumerate(evidence) if re.search(pattern,e['text'],re.I)]
            if hits:diagnoses.append({'category':code,'observation':label,'evidence_indices':hits,'confidence':'rule_match','root_cause':'未确定；规则命中不代表因果确认'})
        if not diagnoses and t['state']=='failed':diagnoses=[{'category':'verification_failed','observation':'最终结果未通过','root_cause':'原因未确定，请查看 verifier 与执行轨迹','evidence_indices':list(range(len(evidence))),'confidence':'result_only'}]
        artifacts=[]
        for folder in ['emergence','system_prompts','workspace','artifact']:
            parent=base/folder
            if parent.is_dir():
                for p in parent.rglob('*'):
                    if p.is_file() and p.suffix in ('.json','.md','.yaml','.yml','.py','.txt'):
                        try:artifacts.append(self.register(p))
                        except ValueError:continue
                        if len(artifacts)>=100:break
        return redact({'task':t,'raw':raw,'files':files,'definitions':definitions,'sessions':session_files,'mappings':mappings,'issues':issues,'evidence':evidence,'diagnoses':diagnoses,'artifacts':artifacts,'artifact_warning':'这是运行级资产，除非日志提供版本关联，否则不能认定为该任务当时使用的策略。'})
    def analysis(self,aid):
        a=next((a for a in self.data['analyses'] if a['id']==aid),None)
        if not a:raise KeyError('analysis_not_found')
        obj=json.loads(self.resolver.safe(self.root/a['path']/'prefix_analysis.json').read_text()) if a['type']=='prefix' else {}
        return redact({'analysis':a,'summaries':obj.get('summaries',[]),'notes':obj.get('notes',[]),'round_unit':obj.get('round_unit'),'reports':[self.register(p) for p in (self.root/a['path']).iterdir() if p.is_file() and p.suffix in ('.html','.md','.csv','.png','.svg')],
            'linked_runs':[{k:r[k] for k in ['id','path','name']} for r in self.data['runs'] if any(r['path']==str(v).rstrip('/') for v in a['source_runs'].values())]})
    def compare(self,ids):
        if len(ids)<2 or len(ids)>6:raise ValueError('select_2_to_6_runs')
        runs=[self.run(r) for r in ids]
        families={r['family'] for r in runs}
        if len(families)!=1:raise ValueError('different_benchmarks_cannot_be_ranked')
        versions={str(r['version']) for r in runs if r['version']}
        if len(versions)>1:raise ValueError('different_dataset_versions')
        maps=[];ambiguous=[]
        for rid in ids:
            by=collections.defaultdict(list)
            for t in self.data['tasks'].get(rid,[]):
                if not t['synthetic']:by[t['task']].append(t)
            ambiguous.extend({'run_id':rid,'task':k,'attempts':len(v)} for k,v in by.items() if len(v)>1)
            maps.append({k:v[0] for k,v in by.items() if len(v)==1})
        common=sorted(set.intersection(*(set(x) for x in maps)))
        deltas=[]
        for i in range(1,len(maps)):
            gains=[];losses=[];currencies={};token_delta=0;token_pairs=0
            for task in common:
                a,b=maps[0][task],maps[i][task]
                if a['success'] is False and b['success'] is True:gains.append(task)
                if a['success'] is True and b['success'] is False:losses.append(task)
                if a['tokens'] is not None and b['tokens'] is not None:token_delta+=b['tokens']-a['tokens'];token_pairs+=1
                if a['cost'] is not None and b['cost'] is not None and a['currency']==b['currency'] and a['currency']!='未声明币种' and a.get('prices')==b.get('prices') and any(v is not None for v in (a.get('prices') or {}).values()):
                    group=currencies.setdefault(a['currency'],{'difference':0,'paired_tasks':0});group['difference']+=b['cost']-a['cost'];group['paired_tasks']+=1
            deltas.append({'baseline':ids[0],'candidate':ids[i],'success_gains':gains,'success_regressions':losses,'cost_differences':currencies,'token_difference':token_delta,'token_pairs':token_pairs})
        union=set.union(*(set(x) for x in maps))
        return {'runs':runs,'common_count':len(common),'union_count':len(union),'outside_intersection':len(union)-len(common),'deltas':deltas,'ambiguous_excluded':ambiguous,'comparability':'未确认（需核对任务内容哈希、种子与模型配置）','rows':[{'task':t,'attempts':[m[t] for m in maps]} for t in common]}
