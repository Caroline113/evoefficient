"""Local-only standard-library HTTP API and static UI."""
import argparse
import csv
import io
import json
import mimetypes
import re
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs,urlparse
from .core import redact
from .store import Store
from .trace import trajectory

STATIC=Path(__file__).parent/'static'

def csv_safe(x):
    s='' if x is None else str(x)
    return "'"+s if s.lstrip().startswith(('=','+','-','@','\t','\r')) else s

def serve(store,host='127.0.0.1',port=8765):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,fmt,*args):pass
        def send(self,data,status=200,kind='application/json; charset=utf-8',download=None):
            if kind.startswith('application/json'):data=json.dumps(redact(data),ensure_ascii=False,allow_nan=False).encode()
            elif isinstance(data,str):data=data.encode()
            self.send_response(status);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(data)))
            self.send_header('X-Content-Type-Options','nosniff');self.send_header('Cache-Control','no-store')
            self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; frame-src 'none'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'")
            if download:self.send_header('Content-Disposition',f'attachment; filename="{download}"')
            self.end_headers();self.wfile.write(data)
        def safe_host(self):
            hostname=(self.headers.get('Host') or '').split(':')[0]
            return hostname in ('127.0.0.1','localhost','[::1]')
        def do_POST(self):
            origin=self.headers.get('Origin')
            if not self.safe_host() or self.headers.get('X-Dashboard-Request')!='1' or origin and urlparse(origin).netloc!=self.headers.get('Host'):
                return self.send({'error':'request_origin_denied'},403)
            if self.path=='/api/scan':return self.send({'started':store.start_scan()})
            if self.path=='/api/scan/cancel':store.cancel.set();return self.send({'cancellation_requested':True})
            self.send({'error':'not_found'},404)
        def do_GET(self):
            if not self.safe_host():return self.send({'error':'host_denied'},403)
            parsed=urlparse(self.path);path=parsed.path;q=parse_qs(parsed.query)
            arg=lambda name,default='':q.get(name,[default])[0]
            try:
                if path=='/api/health':return self.send({'ok':True,'root':str(store.root/'logs'),'cache':str(store.cache),'scan':store.state,'schema_version':store.data['schema_version']})
                if path=='/api/overview':
                    runs=store.data['runs']
                    return self.send({'scan':store.state,'runs':len(runs),'tasks':sum(r['summary']['tasks'] for r in runs),'families':sorted(set(r['family'] for r in runs)),'issues':len(store.data['issues']),'analyses':store.data['analyses'],'inventory':store.data['inventory'],'summary_note':'总览仅汇总运行/尝试数量；跨基准不合并成功率或费用排名。'})
                if path=='/api/issues':return self.send({'issues':store.data['issues']})
                if path=='/api/runs':
                    runs=store.data['runs'];term=arg('q').lower()
                    runs=[r for r in runs if not term or term in json.dumps(r,ensure_ascii=False).lower()]
                    for key in ['family','mode']:
                        if arg(key):runs=[r for r in runs if r.get(key)==arg(key)]
                    if arg('synthetic')=='exclude':runs=[r for r in runs if r['summary']['real_tasks'] or r['summary_only']]
                    return self.send({'runs':runs,'count':len(runs)})
                if path=='/api/compare':return self.send(store.compare(arg('runs').split(',')))
                match=re.fullmatch(r'/api/analyses/([a-f0-9]+)',path)
                if match:return self.send(store.analysis(match[1]))
                match=re.fullmatch(r'/api/runs/([a-f0-9]+)(?:/tasks(?:/([a-f0-9]+)(?:/(trace|export))?)?)?',path)
                if match:
                    rid,tid,action=match.groups();run=store.run(rid)
                    if tid:
                        if action=='trace':
                            data=trajectory(store,rid,tid,arg('session') or None);events=data['events'];kind=arg('type');term=arg('q').lower()
                            filtered=[e for e in events if (not kind or kind in (e.get('type'),e.get('subtype'),e.get('role'))) and (not term or term in e['text'].lower())]
                            offset=max(0,int(arg('offset','0')));limit=min(100,max(1,int(arg('limit','50'))))
                            data={**data,'events':filtered[offset:offset+limit],'total_events':len(events),'filtered_events':len(filtered),'offset':offset,'limit':limit}
                            return self.send(data)
                        detail=store.detail(rid,tid)
                        if action=='export':
                            text='# '+detail['task']['task']+'\n\n来源：'+run['path']+'\n\n```json\n'+json.dumps(redact(detail),ensure_ascii=False,indent=2)+'\n```\n'
                            return self.send(text,kind='text/markdown; charset=utf-8',download='task.md')
                        # No full steps or conversation before opening the trajectory tab.
                        detail['raw']={k:v for k,v in detail['raw'].items() if k not in ('steps','conversation')}
                        return self.send(detail)
                    tasks=store.data['tasks'].get(rid,[])
                    if path.endswith('/tasks'):
                        state=arg('state');term=arg('q').lower();order=arg('sort','task')
                        tasks=[t for t in tasks if (not state or t['state']==state) and (not term or term in t['task'].lower())]
                        if arg('synthetic')=='exclude':tasks=[t for t in tasks if not t['synthetic']]
                        if order in ('cost','tokens','elapsed','turns_reported'):tasks.sort(key=lambda t:float(t.get(order)) if t.get(order) is not None else -1,reverse=True)
                        else:tasks.sort(key=lambda t:t['task'])
                        if arg('export'):
                            if arg('export')=='json':return self.send({'run':run,'tasks':tasks},download='tasks.json')
                            buf=io.StringIO();w=csv.writer(buf);keys=['task','state','synthetic','success','score','tokens','cost','currency','elapsed','phase','batch','turns_reported','tools_reported','source','location'];w.writerow(keys)
                            for t in tasks:w.writerow([csv_safe(redact(t.get(k))) for k in keys])
                            return self.send('\ufeff'+buf.getvalue(),kind='text/csv; charset=utf-8',download='tasks.csv')
                        offset=max(0,int(arg('offset','0')));limit=min(100,max(1,int(arg('limit','50'))))
                        return self.send({'tasks':tasks[offset:offset+limit],'count':len(tasks),'offset':offset,'limit':limit})
                    issues=[];base=store.root/run['path'];meta={}
                    for f in ['experiment_manifest.json','summary.json','results.metrics.json','summary.csv','accounting.json','pipeline_summary.json','run_status.json','RUN_COMPLETE.json','artifact/artifact.json','eval/freeze_audit.json','train/usage_ledger.json']:
                        p=base/f
                        if p.is_file():
                            rows,_=store.read(p,issues);meta[f]=redact(rows[:100])
                    if run['summary_only'] and (base/'results.jsonl').exists():
                        rows,_=store.read(base/'results.jsonl',issues);meta['results.jsonl (aggregate cycles, not task attempts)']=redact(rows[:100])
                    groups={}
                    for t in store.data['tasks'].get(rid,[]):
                        key=str([t.get('phase'),t.get('epoch'),t.get('batch')]);groups.setdefault(key,[]).append(t)
                    from .core import summarize
                    return self.send({'run':run,'metadata':meta,'issues':issues,'batches':[{'identity':key,'summary':summarize(rows)} for key,rows in groups.items()],'analyses':[a for a in store.data['analyses'] if run['path'] in a['source_runs'].values()]})
                match=re.fullmatch(r'/api/files/([a-f0-9]+)',path)
                if match:
                    p=store.files.get(match[1])
                    if not p:return self.send({'error':'file_not_registered_open_task_or_analysis_first'},404)
                    p=store.resolver.safe(p)
                    if p.stat().st_size>24*1024*1024:return self.send({'error':'file_size_limit_24MiB'},413)
                    content=p.read_bytes()
                    if p.suffix.lower() in ('.png','.jpg','.jpeg','.webp'):
                        valid=content.startswith(b'\x89PNG\r\n\x1a\n') or content.startswith(b'\xff\xd8\xff') or content.startswith(b'RIFF') and content[8:12]==b'WEBP'
                        if not valid:raise ValueError('invalid_image_signature')
                        return self.send(content,kind=mimetypes.guess_type(p.name)[0] or 'application/octet-stream')
                    text=content.decode('utf-8-sig',errors='replace')
                    try:text=json.dumps(redact(json.loads(text)),ensure_ascii=False,indent=2)
                    except ValueError:text=redact(text)
                    # HTML is deliberately served as inert text, never in application origin.
                    return self.send(text,kind='text/plain; charset=utf-8',download='evidence.txt' if arg('download') else None)
                if path in ('/','/index.html','/app.js','/style.css'):
                    p=STATIC/('index.html' if path=='/' else path[1:]);return self.send(p.read_bytes(),kind=mimetypes.guess_type(p.name)[0]+'; charset=utf-8')
                self.send({'error':'not_found'},404)
            except KeyError as e:self.send({'error':str(e)},404)
            except (ValueError,OSError) as e:self.send({'error':redact(str(e))},400)
            except Exception as e:self.send({'error':type(e).__name__,'detail':'请求失败，旧索引保持可用'},500)
    server=ThreadingHTTPServer((host,port),Handler)
    if not store.data['runs']:store.start_scan()
    return server

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2]);p.add_argument('--cache',type=Path);p.add_argument('--port',type=int,default=8765);p.add_argument('--scan-only',action='store_true');a=p.parse_args()
    store=Store(a.root,a.cache or a.root/'.experiment_dashboard_cache')
    if a.scan_only:
        store.state['scanning']=True;store.scan();print(json.dumps({'state':store.state,'runs':len(store.data['runs']),'issues':len(store.data['issues'])},ensure_ascii=False));return
    server=serve(store,port=a.port);print(f'Experiment Dashboard: http://127.0.0.1:{a.port}',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:store.cancel.set();server.server_close()

if __name__=='__main__':main()
