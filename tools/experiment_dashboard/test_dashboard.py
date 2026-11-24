import csv
import http.client
import json
import threading
from pathlib import Path
import pytest
from .core import Resolver,load_rows,redact,normalize,summarize
from .store import Store
from .trace import parse_session
from .server import serve,csv_safe

@pytest.fixture
def project(tmp_path):
    (tmp_path/'logs').mkdir();return tmp_path

def write(p,obj):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj))

def test_formats_partial_jsonl_and_manifest(project):
    issues=[];p=project/'logs'/'results.jsonl';p.write_text('{"instance_id":"a","success":true}\n{"incomplete":')
    rows,loc=load_rows(p,issues)
    assert rows[0]['instance_id']=='a' and loc==['line:1']
    assert issues[0]['kind']=='incomplete_jsonl_tail'
    manifest=project/'logs'/'experiment_manifest.json';write(manifest,{'tasks':['a','b'],'benchmark':'GAIA'})
    assert load_rows(manifest,[])[0][0]['benchmark']=='GAIA'
    write(manifest,{'tasks':20,'benchmark':'GAIA'})
    assert load_rows(manifest,[])[0][0]['tasks']==20

def test_safe_paths_and_mapping(project,tmp_path):
    f=project/'logs'/'r'/'a.json';write(f,{})
    resolver=Resolver(project)
    path,method=resolver.resolve('/home/azureuser/zj/a-evolve-new7/a-evolve-new/logs/r/a.json',project/'logs'/'r')
    assert path==f and method=='old_root_mapping'
    with pytest.raises(ValueError):resolver.safe(project/'logs'/'..'/'secret.txt')
    secret=project/'secret.txt';secret.write_text('password')
    link=project/'logs'/'escape.txt';link.symlink_to(secret)
    with pytest.raises(ValueError):resolver.safe(link)

def test_redaction_and_csv():
    d=redact({'api_key':'secret-value','tokens':15,'note':'Bearer abc123 password=hello https://u:pw@example.com?a=2&token=foo sk-longsecretkey123'})
    assert d['api_key']=='[REDACTED]' and d['tokens']==15
    assert not any(x in d['note'] for x in ['abc123','hello','u:pw','token=foo','longsecret'])
    assert csv_safe('=CMD()').startswith("'")

def test_scan_nested_dedup_retries_summary_and_synthetic(project):
    a={'instance_id':'one','success':True,'total_tokens':100,'cost_units':2,'cost_priced':True,'cost_currency':'USD'}
    r=project/'logs'/'swe'/'run';write(r/'results.json',[a,{**a,'attempt':2},{**a,'instance_id':'mock','synthetic':True}])
    (r/'results.jsonl').write_text(json.dumps(a)+'\n'+json.dumps({**a,'attempt':2})+'\n')
    write(r/'experiment_manifest.json',{'benchmark':'SWE','tasks':['one']})
    q=project/'logs'/'mcp'/'baseline';q.mkdir(parents=True);(q/'summary.csv').write_text('metric,value\nsuccess_rate,0.5\n')
    derived=project/'logs'/'report';write(derived/'prefix_analysis.json',{'runs':{},'caps':[20],'rows':[1]})
    store=Store(project,project/'cache');store.scan()
    assert store.state['error'] is None
    assert len(store.data['runs'])==2 and len(store.data['analyses'])==1
    run=next(x for x in store.data['runs'] if not x['summary_only'])
    assert run['family']=='SWE' and run['summary']['tasks']==3 and run['summary']['real_tasks']==2
    assert run['summary']['costs']['USD']['known_sum']==4
    assert len(store.data['tasks'][run['id']][0]['sources'])==2

def test_missing_cost_and_failure_states():
    rows=[normalize({'instance_id':'a','success':False,'error':'connection reset'},'r','logs/a/results.json','/0',0,{}),normalize({'instance_id':'b','success':False},'r','logs/a/results.json','/1',1,{})]
    s=summarize(rows)
    assert s['states']['error']==1 and s['states']['failed']==1 and s['costs']=={}
    assert s['known_result_denominator']==1
    row=normalize({'task_id':'mcp-a','result':'PASS','elapsed_s':'2.5'},'r','logs/mcp/summary.csv','line:2',0,{})
    assert row['state']=='success' and row['elapsed']==2.5

def test_trace_dedup_reset_and_tool_links(tmp_path):
    events=[{'type':'turn_context','payload':{'turn_id':'turn-a'}},{'type':'response_item','payload':{'type':'function_call','call_id':'c','name':'bash','arguments':'{"command":"ls"}'}},{'type':'response_item','payload':{'type':'function_call_output','call_id':'c','output':'OK'}}]
    for n in [10,10,20,5]:events.append({'type':'token_usage_record','payload':{'thread_token_usage':{'input_tokens':n,'output_tokens':1,'cached_input_tokens':3,'total_tokens':n+1}}})
    p=tmp_path/'trace.jsonl';p.write_text('\n'.join(json.dumps(x) for x in events))
    d=parse_session(p)
    assert d['counts']=={'model_usage_updates':3,'tool_calls_recovered':1,'dialogue_turns_recovered':1}
    assert d['events'][1]['output_event_index']==2
    assert d['usage'][-1]['delta'] is None and not d['exact_prefix_accounting']

def test_cancel_preserves_old_snapshot(project):
    write(project/'logs'/'swe'/'results.json',[{'instance_id':'a'}])
    s=Store(project,project/'cache');s.scan();before=s.data
    s.cancel.set();s.scan();assert s.data is before and s.state['error']

def test_generic_name_and_source_change(project):
    p=project/'logs'/'gaia'/'custom-output.json'
    write(p,[{'instance_id':'a','success':True}])
    s=Store(project,project/'cache');s.scan()
    assert len(s.data['runs'])==1
    run=s.data['runs'][0];task=s.data['tasks'][run['id']][0]
    assert s.raw_task(task)['success'] is True
    write(p,[{'instance_id':'a','success':False}])
    with pytest.raises(ValueError,match='source_changed'):s.raw_task(task)

def test_compare_does_not_merge_attempts_or_currency(project):
    for name,currency in [('a','USD'),('b','CNY')]:
        write(project/'logs'/'gaia'/name/'results.json',[{'instance_id':'task','success':name=='b','cost_units':1,'cost_currency':currency,'input_price_per_1m':1}])
    s=Store(project,project/'cache');s.scan();ids=[r['id'] for r in s.data['runs']]
    d=s.compare(ids)
    assert d['common_count']==1 and d['deltas'][0]['cost_differences']=={}

def test_http_local_scan_guard_and_unknown_route(project):
    s=Store(project,project/'cache');s.scan();server=serve(s,port=0)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:
        c=http.client.HTTPConnection('127.0.0.1',server.server_port)
        c.request('GET','/api/overview');r=c.getresponse();assert r.status==200;r.read()
        c.request('POST','/api/scan');r=c.getresponse();assert r.status==403;r.read()
        c.request('GET','/api/files/12345');r=c.getresponse();assert r.status==404;r.read()
        c.request('GET','/',headers={'Host':'evil.example'});r=c.getresponse();assert r.status==403;r.read()
    finally:server.shutdown();server.server_close()
