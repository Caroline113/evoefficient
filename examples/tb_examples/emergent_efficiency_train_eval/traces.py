"""Exact cumulative-usage prefixes; a round is a distinct model usage update."""
from __future__ import annotations
import json
from pathlib import Path

def normalize_counter(u):
    incoming, outgoing = int(u['input_tokens']), int(u['output_tokens'])
    cached = int(u.get('cached_input_tokens', 0))
    total = int(u.get('total_tokens', incoming + outgoing))
    if min(incoming, outgoing, cached) < 0 or cached > incoming or total != incoming + outgoing:
        raise ValueError('Invalid or ambiguous cumulative token counter')
    return {'input': incoming-cached, 'cache': cached, 'output': outgoing, 'total': total}

def parse_session(path):
    streams = {'thread': [], 'event': []}; calls = {}; outputs = {}; conversation = []; steps = []
    malformed = 0
    with Path(path).open(encoding='utf-8', errors='replace') as f:
        for line in f:
            if not line.strip():continue
            try: e = json.loads(line)
            except ValueError: malformed += 1; continue
            p = e.get('payload', {})
            if not isinstance(p, dict): continue
            if e.get('type') == 'token_usage_record' and isinstance(p.get('thread_token_usage'), dict):
                streams['thread'].append(normalize_counter(p['thread_token_usage']))
            elif e.get('type') == 'event_msg' and p.get('type') == 'token_count':
                u = (p.get('info') or {}).get('total_token_usage')
                if isinstance(u, dict): streams['event'].append(normalize_counter(u))
            if e.get('type') != 'response_item': continue
            kind = p.get('type'); cid = p.get('call_id') or p.get('id')
            if kind in ('function_call', 'custom_tool_call') and cid and cid not in calls:
                args = p.get('arguments', p.get('input', {}))
                if isinstance(args, str):
                    try: args = json.loads(args)
                    except ValueError: args = {'input': args}
                if not isinstance(args, dict): args = {'input': str(args)}
                call = {'name': p.get('name', 'unknown'), 'arguments': args, 'call_id': cid}
                calls[cid] = call
                conversation.append({'role': 'assistant', 'tool_calls': [{'function': call['name'], 'arguments': args}]})
            elif kind in ('function_call_output', 'custom_tool_call_output') and cid in calls and cid not in outputs:
                output = str(p.get('output', '')); outputs[cid] = output
                conversation.append({'role': 'tool', 'content': output, 'call_id': cid})
                steps.append({'tool': calls[cid]['name'], 'arguments': calls[cid]['arguments'], 'output': output, 'call_id': cid})
            elif kind == 'message' and p.get('role') == 'assistant':
                conversation.append({'role': 'assistant', 'content': p.get('content', []), 'channel': p.get('channel')})
    source = 'thread' if streams['thread'] else 'event'; samples = []
    for u in streams[source]:
        if samples and u['total'] < samples[-1]['total']: raise ValueError('Cumulative token counter reset; prefix cost cannot be reconstructed exactly')
        if samples and u['total'] == samples[-1]['total']:
            if u != samples[-1]: raise ValueError('Conflicting cumulative counters at the same total')
            continue
        if u['total'] > 0: samples.append(u)
    if malformed: raise ValueError(f'{malformed} malformed JSONL records; prefix rounds may be incomplete')
    return {'samples': samples, 'source': source, 'round_unit': 'model_usage_update', 'conversation': conversation, 'steps': steps, 'tool_calls': len(calls)}

def locate_session(row, run_dir):
    run_dir = Path(run_dir)
    def resolve(s):
        p = Path(s)
        return p if p.is_absolute() else run_dir/p
    for key in ('session_path', 'usage_artifact_path'):
        if row.get(key) and resolve(row[key]).is_file(): return resolve(row[key])
    raw_path = row.get('raw_harbor_result_path')
    if raw_path and resolve(raw_path).is_file():
        raw = json.loads(resolve(raw_path).read_text())
        s = (raw.get('usage') or {}).get('usage_artifact_path')
        if s:
            candidates = [resolve(s), Path.cwd()/s]
            for p in candidates:
                if p.is_file(): return p
    if row.get('raw_job_path'):
        paths = sorted(resolve(row['raw_job_path']).glob('agent/sessions/**/*.jsonl'))
        if len(paths) == 1: return paths[0]
        if len(paths) > 1: raise ValueError('Multiple sessions; explicit usage_artifact_path is required')
    raise ValueError('Raw model usage session unavailable')

def prices(row, override=None):
    if override is not None: return {k: float(v) for k,v in override.items()}
    keys = {'input': 'input_price_per_1m', 'cache': 'cache_read_price_per_1m', 'output': 'output_price_per_1m'}
    if row.get('cost_priced') is not True or any(row.get(k) is None for k in keys.values()): return None
    result = {k: float(row[field]) for k,field in keys.items()}
    if min(result.values()) < 0: raise ValueError('Negative price')
    return result

def cost(sample, profile):
    return sum(sample[k]*profile[k] for k in ('input','cache','output'))/1e6 if profile else None

def prefix(trace, cap, profile):
    if cap <= 0: raise ValueError('Round cap must be positive')
    ss = trace['samples']
    if not ss: raise ValueError('No cumulative model usage updates; no prefix estimate is invented')
    n = min(cap, len(ss)); u = ss[n-1]
    return {'rounds_available': len(ss), 'rounds_used': n, 'reached_cap': len(ss)>=cap, 'truncated': len(ss)>cap, **u, 'cost': cost(u, profile)}
