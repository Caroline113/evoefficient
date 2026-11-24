"""Read-only adapters and safe file resolution. Python standard library only."""
import csv
import hashlib
import io
import json
import math
import re
from pathlib import Path

VERSION = 2
OLD_ROOT = '/home/azureuser/zj/a-evolve-new7/a-evolve-new'
SECRET = re.compile(r'api[_-]?key|authorization|password|secret|access[_-]?token|refresh[_-]?token|bearer', re.I)

def redact(value):
    if isinstance(value, dict):
        return {str(k): '[REDACTED]' if SECRET.search(str(k)) else redact(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)):
        return [redact(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value): return None
    if not isinstance(value,str): return value
    value = re.sub(r'(?i)(bearer\s+)[\w.\-+/=]+',r'\1[REDACTED]',value)
    value = re.sub(r'\bsk-[A-Za-z0-9_-]{8,}', '[REDACTED]', value)
    value = re.sub(r'(?i)((?:api[_-]?key|password|access_token|refresh_token|secret)\s*[=:]\s*[\"\x27]?)[^\s,;\"\x27}]+',r'\1[REDACTED]',value)
    value = re.sub(r'(https?://)[^\s/@]+:[^\s/@]+@',r'\1[REDACTED]@',value)
    value = re.sub(r'(?i)([?&](?:token|key|auth|signature|password)=)[^\s&#\"\x27]+',r'\1[REDACTED]',value)
    value = re.sub(r'\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+', '[REDACTED]',value)
    return value

def uid(value): return hashlib.sha256(str(value).encode()).hexdigest()[:20]
def number(value):
    if value is None or isinstance(value,bool): return None
    try:
        n=float(value)
        return n if math.isfinite(n) else None
    except (ValueError,TypeError): return None
def boolean(value):
    if isinstance(value,bool): return value
    if str(value).lower() in ('true','1'): return True
    if str(value).lower() in ('false','0'): return False
    return None
def first(row,*keys):
    return next((row[k] for k in keys if k in row and row[k] is not None),None)

class Resolver:
    def __init__(self,root):
        self.root=Path(root).resolve()
        self.allowed=[self.root/'logs', self.root/'datasets', self.root/'agent_evolve'/'benchmarks']
    def relative(self,p): return str(Path(p).relative_to(self.root))
    def safe(self,p):
        p=Path(p).resolve()
        if not any(p.is_relative_to(a.resolve()) for a in self.allowed): raise ValueError('path_outside_allowed_roots')
        if any(x.startswith('.') for x in p.relative_to(self.root).parts): raise ValueError('hidden_file_denied')
        if p.name.lower() in ('auth.json','id_rsa','id_ed25519') or p.suffix.lower() in ('.pem','.key'): raise ValueError('credential_file_denied')
        return p
    def resolve(self,value,run):
        value=str(value or '')
        if not value: return None,'missing_reference'
        mapped=value
        method='explicit'
        if value==OLD_ROOT or value.startswith(OLD_ROOT+'/'):
            mapped=str(self.root)+value[len(OLD_ROOT):];method='old_root_mapping'
        p=Path(mapped)
        candidates=[p] if p.is_absolute() else [self.root/p,run/p]
        for p in candidates:
            try:
                q=self.safe(p)
                if q.exists():return q,method if method!='explicit' else ('absolute' if Path(mapped).is_absolute() else 'project_or_run_relative')
            except ValueError: continue
        return None,'unresolved_or_denied'

def load_rows(path,issues):
    """Return (rows, locations); bad JSONL lines do not erase earlier records."""
    if path.stat().st_size==0:
        issues.append({'kind':'empty_file','source':str(path)});return [],[]
    if path.stat().st_size>128*1024*1024:
        issues.append({'kind':'file_too_large','source':str(path),'detail':'128MiB parsing limit'});return [],[]
    try:
        if path.suffix=='.jsonl':
            rows=[];locations=[]
            with path.open(encoding='utf-8-sig',errors='replace') as f:
                for n,line in enumerate(f,1):
                    if not line.strip():continue
                    try:
                        obj=json.loads(line)
                        if isinstance(obj,dict) and isinstance(obj.get('result'),dict):obj=obj['result']
                        if isinstance(obj,dict):rows.append(obj);locations.append(f'line:{n}')
                    except ValueError:
                        issues.append({'kind':'incomplete_jsonl_tail' if not line.endswith('\n') else 'invalid_jsonl_line','source':str(path),'location':f'line:{n}'})
            return rows,locations
        if path.suffix=='.csv':
            with path.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
            return rows,[f'line:{n+2}' for n in range(len(rows))]
        obj=json.loads(path.read_text(encoding='utf-8-sig'))
        if isinstance(obj,list):return [r for r in obj if isinstance(r,dict)],[f'/{n}' for n,r in enumerate(obj) if isinstance(r,dict)]
        if isinstance(obj,dict):
            for key in ['results','tasks','records']:
                if key=='tasks' and (not isinstance(obj.get(key),list) or not obj[key] or not all(isinstance(r,dict) and any(k in r for k in ['instance_id','task_name','task_id']) for r in obj[key])):continue
                if isinstance(obj.get(key),list):return [r for r in obj[key] if isinstance(r,dict)],[f'/{key}/{n}' for n,r in enumerate(obj[key]) if isinstance(r,dict)]
            return [obj],['/']
    except (ValueError,OSError) as e:
        issues.append({'kind':'parse_error','source':str(path),'detail':type(e).__name__})
    return [],[]

def family(path,manifest):
    explicit=first(manifest,'benchmark','dataset')
    if explicit:return str(explicit),'manifest'
    s=str(path).lower()
    for tokens,name in [(['visualwebarena'],'VisualWebArena'),(['osworld'],'OSWorld2'),(['gaia'],'GAIA'),(['mcp'],'MCP'),(['tb_','tb41','terminal'],'Terminal-Bench'),(['swe','evoefficient','emergent_efficiency','baseline'],'SWE / 未确认子集')]:
        if any(t in s for t in tokens):return name,'directory_inference'
    return '未识别','unknown'

def normalize(row,run_id,source,location,index,manifest):
    task=str(first(row,'instance_id','task_name','task_id','instance','task') or '')
    if not task:return None
    status=str(first(row,'status','result') or '').lower()
    success=boolean(first(row,'success','passed'))
    if success is None and status in ('pass','passed','success','true'):success=True
    if success is None and status in ('fail','failed','failure','false'):success=False
    if boolean(row.get('timed_out')) is True or 'timeout' in status:state='timeout'
    elif row.get('error') or status in ('error','exception'):state='error'
    elif status in ('interrupted','cancelled','canceled'):state='interrupted'
    elif status in ('running','in_progress'):state='running'
    elif success is True:state='success'
    elif success is False:state='failed'
    else:state='unknown'
    synthetic=boolean(row.get('synthetic')) is True or status=='synthetic' or any(boolean(manifest.get(k)) is True for k in ['dry_run','synthetic','dry_run_harbor','dry_run_codex'])
    phase=row.get('phase') or ('train' if '/train/' in source else 'eval' if '/eval/' in source else None)
    signature=[task,phase,row.get('batch'),row.get('epoch'),first(row,'attempt','attempt_id','trial_id','retry'),row.get('seed')]
    return redact({'id':uid([run_id,signature,index]),'task':task,'run_id':run_id,'state':state,'success':success,'score':number(row.get('score')),
        'phase':phase,'batch':row.get('batch'),'epoch':row.get('epoch'),'attempt':first(row,'attempt','attempt_id','trial_id','retry'),
        'mode':first(row,'mode','group') or manifest.get('mode'),'synthetic':synthetic,
        'turns_reported':number(first(row,'num_turns','turns')),'tools_reported':number(row.get('num_tool_calls')),
        'tokens':number(row.get('total_tokens')),'input':number(row.get('cumulative_input_tokens')),'output':number(row.get('cumulative_output_tokens')),
        'cache_read':number(row.get('cumulative_cache_read_input_tokens')),'cache_write':number(row.get('cumulative_cache_write_input_tokens')),
        'cost':number(row.get('cost_units')),'priced':boolean(row.get('cost_priced')),'currency':row.get('cost_currency') or '未声明币种',
        'price_source':row.get('cost_price_source'),'prices':{k:number(row.get(k)) for k in ['input_price_per_1m','output_price_per_1m','cache_read_price_per_1m','cache_write_price_per_1m']},
        'model':first(row,'main_model_id','model_id','model') or first(manifest,'model_id','model'),
        'elapsed':number(first(row,'elapsed','solve_time','elapsed_s')),'quality':row.get('usage_record_quality'),'estimated':boolean(row.get('usage_estimated')),
        'source':source,'location':location,'row_index':index,'sources':[{'source':source,'location':location}],
        'identity':signature,'has_reference':bool(any(row.get(k) for k in ['conversation_path','trajectory_path','raw_job_path']) or row.get('steps'))})

def summarize(tasks):
    real=[t for t in tasks if not t['synthetic']]
    states={s:sum(t['state']==s for t in real) for s in ['success','failed','error','timeout','interrupted','running','unknown']}
    known=[t for t in real if t['success'] is not None and t['state'] not in ('error','timeout','interrupted','running')]
    costs={};cost_groups={}
    for t in real:
        if t['cost'] is not None and t['priced'] is not False:
            g=costs.setdefault(t['currency'],{'known_sum':0,'priced_tasks':0,'price_sources':[]})
            g['known_sum']+=t['cost'];g['priced_tasks']+=1
            label=json.dumps([t['currency'],t.get('model'),t.get('price_source'),t.get('prices')],sort_keys=True)
            c=cost_groups.setdefault(label,{'currency':t['currency'],'model':t.get('model'),'price_source':t.get('price_source'),'prices':t.get('prices'),'known_sum':0,'tasks':0})
            c['known_sum']+=t['cost'];c['tasks']+=1
    return {'tasks':len(tasks),'real_tasks':len(real),'synthetic_tasks':len(tasks)-len(real),'states':states,
        'success_rate_known':sum(t['success'] is True for t in known)/len(known) if known else None,'known_result_denominator':len(known),
        'tokens_known_sum':sum(t['tokens'] for t in real if t['tokens'] is not None),'tokens_coverage':sum(t['tokens'] is not None for t in real),
        'costs':costs,'cost_groups':list(cost_groups.values()),'trace_reference_coverage':sum(t['has_reference'] for t in real),
        'elapsed_known_sum':sum(t['elapsed'] for t in real if t['elapsed'] is not None)}
