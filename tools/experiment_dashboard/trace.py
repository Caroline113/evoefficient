"""Lazy trajectory parsing. Never equate tool calls, turns and usage updates."""
import json
from pathlib import Path
from .core import redact,uid,number

def content_text(p):
    content=p.get('content',p.get('output',p.get('arguments',p.get('input',p))))
    if isinstance(content,str):return content
    return json.dumps(content,ensure_ascii=False,default=str)

def parse_session(path):
    events=[];issues=[];streams={'thread':[],'event':[]};calls={};outputs={};turns=set();previous_line=None
    with path.open('rb') as f:
        for line_no,raw in enumerate(f,1):
            if not raw.strip():continue
            try:obj=json.loads(raw)
            except ValueError:
                issues.append({'kind':'invalid_jsonl_line','line':line_no});continue
            if not isinstance(obj,dict):
                issues.append({'kind':'invalid_jsonl_event','line':line_no});continue
            p=obj.get('payload',{})
            if not isinstance(p,dict):p={'content':p}
            kind=obj.get('type','unknown');sub=p.get('type',kind);cid=p.get('call_id');turn=p.get('turn_id')
            if kind=='turn_context' and turn:turns.add(turn)
            if kind=='token_usage_record' and isinstance(p.get('thread_token_usage'),dict):streams['thread'].append((line_no,p['thread_token_usage'],turn))
            elif kind=='event_msg' and sub=='token_count':
                u=(p.get('info') or {}).get('total_token_usage')
                if isinstance(u,dict):streams['event'].append((line_no,u,turn))
            event={'index':len(events),'line':line_no,'type':kind,'subtype':sub,'role':p.get('role'),'timestamp':obj.get('timestamp') or p.get('timestamp'),'turn_id':turn,'call_id':cid,'name':p.get('name'),'text':redact(content_text(p))[:12000]}
            if kind=='response_item':
                if sub in ('function_call','custom_tool_call'):
                    if cid in calls:event['duplicate_call']=True
                    elif cid:calls[cid]=event['index']
                elif sub in ('function_call_output','custom_tool_call_output'):
                    event['call_event_index']=calls.get(cid)
                    if cid in outputs:event['duplicate_output']=True
                    if cid:outputs[cid]=event['index']
                    if cid not in calls:issues.append({'kind':'unmatched_tool_output','line':line_no})
            # Large metadata snapshots are summarized, but remain accessible as sanitized source.
            if kind in ('session_meta','world_state'):event['text']='元数据字段：'+', '.join(p.keys())
            events.append(event)
    for cid,index in calls.items():
        events[index]['output_event_index']=outputs.get(cid)
        if cid not in outputs:issues.append({'kind':'missing_tool_output','line':events[index]['line']})
    source='thread' if streams['thread'] else 'event';samples=[];last=None;segment=0
    for line,u,turn in streams[source]:
        incoming=number(u.get('input_tokens'));out=number(u.get('output_tokens'));cache=number(u.get('cached_input_tokens',0));total=number(u.get('total_tokens'))
        if incoming is None or out is None or cache is None or cache<0 or incoming<cache or out<0:
            issues.append({'kind':'invalid_usage','line':line});continue
        total=incoming+out if total is None else total
        if total!=incoming+out:issues.append({'kind':'invalid_total','line':line});continue
        current={'input':incoming-cache,'cache':cache,'output':out,'total':total}
        if last and total==last['total']:
            if current!=last:issues.append({'kind':'conflicting_usage_counter','line':line})
            continue
        reset=last and any(current[k]<last[k] for k in current)
        if reset:segment+=1;issues.append({'kind':'counter_reset','line':line});last=None
        if total<=0:continue
        samples.append({'round':len(samples)+1,'line':line,'turn_id':turn,'segment':segment,**current,'delta':{k:current[k]-last[k] for k in current} if last else (current if segment==0 else None)})
        last=current
    return {'events':events,'usage':samples,'issues':issues,'counts':{'model_usage_updates':len(samples),'tool_calls_recovered':len(calls),'dialogue_turns_recovered':len(turns) or None},'usage_source':source,'exact_prefix_accounting':not any(i['kind'] in ('invalid_jsonl_line','counter_reset','conflicting_usage_counter','invalid_usage','invalid_total') for i in issues)}

def trajectory(store,rid,tid,session=None):
    detail=store.detail(rid,tid);task=detail['task']
    options=detail['sessions']
    if session:
        selected=next((s for s in options if s['id']==session),None)
        if not selected:raise ValueError('session_not_in_task')
    elif len(options)==1:selected=options[0]
    elif len(options)>1:return {'events':[],'usage':[],'issues':[{'kind':'ambiguous_session_select_one'}],'sessions':options,'counts':{}}
    else:selected=None
    if selected:
        path=store.files[selected['id']];stat=path.stat();fingerprint=uid([str(path),stat.st_size,stat.st_mtime_ns,2]);cache=store.cache/f'trace-{fingerprint}.json'
        with store.trace_lock:
            if cache.exists():data=json.loads(cache.read_text())
            else:
                data=parse_session(path);tmp=cache.with_suffix('.tmp');tmp.write_text(json.dumps(redact(data),ensure_ascii=False));tmp.replace(cache)
        data['source']=selected;data['sessions']=options
    else:
        raw=store.raw_task(task);messages=[];provenance=None
        for entry in detail['files']:
            if '/conversations/' in entry['path'] and entry['kind']=='.json' and entry['bytes']<32*1024*1024:
                try:
                    obj=json.loads(store.files[entry['id']].read_text());messages=obj if isinstance(obj,list) else obj.get('conversation',obj.get('messages',[]));provenance=entry;break
                except (ValueError,AttributeError):pass
        if not messages:messages=raw.get('conversation') or raw.get('steps') or []
        events=[{'index':i,'line':None,'type':'conversation_or_step','role':p.get('role') if isinstance(p,dict) else None,'subtype':p.get('tool') if isinstance(p,dict) else None,'text':redact(json.dumps(p,ensure_ascii=False))[:12000]} for i,p in enumerate(messages)]
        data={'events':events,'usage':[],'counts':{'model_usage_updates':None,'tool_calls_recovered':None,'dialogue_turns_recovered':None},'issues':[{'kind':'raw_session_unavailable','detail':'仅有包装记录时，不声称恢复了完整过程'}],'source':provenance,'sessions':[],'exact_prefix_accounting':False}
    data['reported']={'turns':task['turns_reported'],'tool_calls':task['tools_reported']}
    data['search_scope']='完整已解析的当前所选会话（非所有日志）；展示文本每事件最多12000字符'
    return data
