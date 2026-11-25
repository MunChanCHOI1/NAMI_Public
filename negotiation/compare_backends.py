"""Compare completed development runs. Offline analysis only; never calls a model."""
from __future__ import annotations

import argparse
from collections import defaultdict, deque
import hashlib
import json
import math
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def checked_calls(result):
    if not result.get('complete') or result.get('interrupted') or result.get('errors'):
        raise ValueError('Only complete runs without errors can be compared.')
    if 'calls' in result:
        calls = result['calls']
    else:
        if result['split'] != 'dev':
            raise ValueError('This comparison accepts development cases only.')
        calls = []
        for case in result['cases']:
            seen = set()
            for decision in case['agent']['internal_record']:
                model = decision.get('model', {})
                if not model.get('is_model'):
                    continue
                if model.get('cached'):
                    if model['trace_ref'] != model['request_sha256'] or model['trace_ref'] not in seen:
                        raise ValueError('A cached decision has no earlier source call in this case.')
                else:
                    calls.append(model)
                    seen.add(model['request_sha256'])
        if len(calls) != result['policies']['agent']['process']['model_requests']:
            raise ValueError('Model request count does not match the journal.')
    for call in calls:
        if call.get('status') != 'ok' or digest(call['request']) != call['request_sha256']:
            raise ValueError('A failed call or changed request was found.')
        if call['response']['metadata']['request_sha256'] != call['request_sha256']:
            raise ValueError('Response request hash does not match.')
        if set(call['predictions']) != set(call['request']['questions']):
            raise ValueError('Prediction questions do not match.')
        if any(isinstance(p, bool) or not isinstance(p, (int, float)) or not math.isfinite(p) or not 0 <= p <= 1
               for p in call['predictions'].values()):
            raise ValueError('Invalid probability.')
    return calls


def compare_calls(native, official, tolerance=0.01):
    """Match every occurrence, preserving duplicate requests and unmatched coverage."""
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError('Invalid tolerance.')
    pending = defaultdict(deque)
    for call in official:
        pending[call['request_sha256']].append(call)
    diffs, flips, matched, left_only = [], [], 0, 0
    for left in native:
        queue = pending[left['request_sha256']]
        if not queue:
            left_only += 1
            continue
        right = queue.popleft()
        if digest(left['request']) != digest(right['request']):
            raise ValueError('Matched request content differs.')
        matched += 1
        for qid, p in left['predictions'].items():
            q = right['predictions'][qid]
            diffs.append(abs(p-q))
            if (p >= 0.5) != (q >= 0.5):
                flips.append({'request_sha256': left['request_sha256'], 'question': qid,
                              'native': p, 'official': q})
    right_only = sum(len(q) for q in pending.values())
    return {'native_requests': len(native), 'official_requests': len(official),
            'matched_request_occurrences': matched, 'native_only_requests': left_only,
            'official_only_requests': right_only, 'matched_questions': len(diffs),
            'maximum_absolute_probability_difference': max(diffs) if diffs else None,
            'mean_absolute_probability_difference': sum(diffs)/len(diffs) if diffs else None,
            'diagnostic_absolute_tolerance': tolerance,
            'questions_exceeding_tolerance': sum(d > tolerance for d in diffs),
            'all_inputs_matched_and_within_tolerance': bool(diffs) and not left_only and not right_only and max(diffs) <= tolerance,
            'threshold_0_5_agreements': len(diffs)-len(flips), 'threshold_0_5_disagreements': flips,
            'native_backend_seconds': sum(c['latency_ms'] for c in native)/1000,
            'official_backend_seconds': sum(c['latency_ms'] for c in official)/1000}


def compare_cases(native, official):
    left = {c['case_id']: c['agent'] for c in native['cases']}
    right = {c['case_id']: c['agent'] for c in official['cases']}
    if set(left) != set(right):
        raise ValueError('Development case coverage differs.')
    rows = []
    for cid, a in left.items():
        b = right[cid]
        def actions(c):
            return [(d['round'], d['action'], (d.get('chosen') or {}).get('kind'),
                     (d.get('chosen') or {}).get('terms')) for d in c['internal_record']]
        rows.append({'case_id': cid, 'same_decisions_and_terms': actions(a) == actions(b),
                     'same_executed_actions': a['executed'] == b['executed'],
                     'same_outcome': (a['status'], a['agreement']) == (b['status'], b['agreement']),
                     'native_saving': a['realized_saving'], 'official_saving': b['realized_saving'],
                     'saving_difference': b['realized_saving']-a['realized_saving']})
    return {'cases': len(rows), 'same_decision_paths': sum(r['same_decisions_and_terms'] for r in rows),
            'same_executed_paths': sum(r['same_executed_actions'] for r in rows),
            'same_outcomes': sum(r['same_outcome'] for r in rows),
            'mean_saving_difference': sum(r['saving_difference'] for r in rows)/len(rows), 'rows': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('native-probe', 'native-dev', 'official-probe', 'official-dev', 'server-evidence', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    load = lambda path: json.loads(path.read_text(encoding='utf-8'))
    np, nd, sp, sd, server = [load(getattr(args, k)) for k in
                              ('native_probe','native_dev','official_probe','official_dev','server_evidence')]
    pins=np['provenance']['model_pins']
    if 'T4' not in server['gpu_before'] or server['reasoning'] or server['decision_temperature'] != 1.0:
        raise ValueError('The requested T4 decision-only configuration is missing.')
    if server['environment']['sglang'] != '0.5.22' or not server['launch']['health']['ok']:
        raise ValueError('The expected running official server is not evidenced.')
    if Path(server['launch']['checkpoint']).name != pins['model_revision'] or 'serve_decide.py' not in server['launch']['command']:
        raise ValueError('The server command/checkpoint does not match the pinned path.')
    if server['checkpoint_head_sha256'] != np['provenance']['checkpoint_metadata_sha256']['original_checkpoint']['jev_head.pt']:
        raise ValueError('The official checkpoint head differs from the native checkpoint.')
    for key, value in {**pins['official_source_files'],**pins['official_server_files']}.items():
        if server['official_source_sha256'][key] != value:
            raise ValueError('An official source hash differs: '+key)
    for baseline, other in ((np,sp),(nd,sd)):
        before=baseline['provenance']['input_sha256']; after=other['provenance']['input_sha256']
        if set(before)!=set(after):
            raise ValueError('Unexpected evaluation source set.')
        for name in before:
            patch=server['bridge_manifest'].get(name)
            expected=patch['after_sha256'] if patch else before[name]
            if after[name]!=expected or (patch and before[name]!=patch['before_sha256']):
                raise ValueError('Undeclared evaluation source change: '+name)
    native_probe,native_dev,official_probe,official_dev=map(checked_calls,(np,nd,sp,sd))
    if (len(native_probe),sum(len(c['predictions']) for c in native_probe),len(nd['cases']))!=(132,540,30):
        raise ValueError('The expected full development protocol is missing.')
    for call in native_probe+native_dev:
        meta=call['response']['metadata']
        if meta.get('mock') or not meta.get('head_loaded') or not meta.get('gpu_execution_performed'):
            raise ValueError('The native reference is not a recorded real-model GPU call.')
        if meta['model_revision']!=pins['model_revision'] or meta['dtype']!='fp16' or 'T4' not in meta['gpu_name']:
            raise ValueError('The native reference model or GPU configuration differs.')
    for call in official_probe+official_dev:
        meta=call['response']['metadata']
        if meta['backend']!='official_decide_http' or meta['decision_temperature']!=1.0 or meta['reasoning_requested']:
            raise ValueError('Unexpected HTTP execution configuration.')
        if pins['model_revision'] not in meta['served_model_ids']:
            raise ValueError('Reported server model does not match the checkpoint.')
    report={'scope':'one execution per backend on the same synthetic development inputs',
            'heldout_executed':False,'official_server_evidence':server,
            'probe':compare_calls(native_probe,official_probe),
            'dev_requests':compare_calls(native_dev,official_dev),
            'development':compare_cases(nd,sd),'official_probe_verdict':sp['verdict']['route'],
            'official_dev_agent':sd['policies']['agent'],'official_dev_minus_rule':sd['agent_minus']['rule_policy'],
            'source_hashes_verified':True,'request_hashes_verified':True,
            'input_files_sha256':{k:hashlib.sha256(getattr(args,k).read_bytes()).hexdigest() for k in
                                 ('native_probe','native_dev','official_probe','official_dev','server_evidence')},
            'limitations':['Official noul values are rounded to four decimals.',
                           'Latency includes warmup and different execution strategies; it is not a controlled benchmark.',
                           'Matching these development inputs does not establish general numerical or policy equivalence.']}
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('probe','dev_requests','development')},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
