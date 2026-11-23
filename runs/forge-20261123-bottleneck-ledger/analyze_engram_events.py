#!/usr/bin/env python3
"""Offline Engram readiness classifier. Fixture tests are not production evidence.

Input JSON: {evidence_kind:'measured'|'synthetic', records:[...]}
Required record keys: request_id, rank, chunk, layer, generation, data_generation,
clock_domain, t_need_ns, t_join_enter_ns, t_join_exit_ns, t_current_ready_ns,
t_all_done_ns, wait_binding ('whole_reader'|'current_layer'|'unknown').
Timing must be in a common monotonic domain. t_need_ns is the actual GPU dependency
ready time (pre-data work complete), not the host join-enter timestamp.
Optional t_reader_start_ns, other_ready_ns, need_uncertainty_ns.
"""
import argparse,json,pathlib
REQUIRED=('request_id','rank','chunk','layer','generation','data_generation','clock_domain',
 't_need_ns','t_join_enter_ns','t_join_exit_ns','t_current_ready_ns','t_all_done_ns','wait_binding')

def classify(r):
    base={k:r.get(k) for k in ('request_id','rank','chunk','layer')}
    missing=[k for k in REQUIRED if k not in r or r[k] is None]
    if missing:return {**base,'status':'unverified','reason':'missing fields','missing':missing}
    if r['generation']!=r['data_generation']:
        return {**base,'status':'invalid','reason':'stale generation; do not infer performance'}
    if not r['clock_domain'] or r.get('clock_alignment_verified') is not True:
        return {**base,'status':'unverified','reason':'clock alignment not verified'}
    fields=('t_need_ns','t_join_enter_ns','t_join_exit_ns','t_current_ready_ns','t_all_done_ns')
    if any(not isinstance(r[k],int) or r[k]<0 for k in fields):
        return {**base,'status':'invalid','reason':'invalid timestamps'}
    n,j0,j1,ready,done=(r[k] for k in fields)
    if j1<j0 or done<ready or j1<ready:
        return {**base,'status':'invalid','reason':'inconsistent ready/join chronology'}
    if r['wait_binding'] not in ('whole_reader','current_layer','unknown'):
        return {**base,'status':'invalid','reason':'unknown wait binding value'}
    u=r.get('need_uncertainty_ns',0)
    if not isinstance(u,int) or u<0:return {**base,'status':'invalid','reason':'invalid uncertainty'}
    # observed join-exit gap is an envelope, not causal savings or measured GPU idle.
    release_envelope=max(0,j1-n)
    current_supply=max(0,ready-n)
    delayed_ready_wait=max(0,j1-max(n,ready))
    hypothetical_whole=max(0,done-n)
    hypothetical_layer=max(0,ready-n)
    coupling=max(0,hypothetical_whole-hypothetical_layer)
    if r['wait_binding']=='whole_reader' and done>j1:
        return {**base,'status':'invalid','reason':'whole-reader join exited before all_done'}
    supported=(r['wait_binding']=='whole_reader' and ready<j0 and done>j0 and done<=j1)
    result={**base,'status':'classified','join_host_ms':(j1-j0)/1e6,
     'release_gap_envelope_ms':release_envelope/1e6,
     'current_supply_late_ms':current_supply/1e6,
     'ready_to_release_envelope_ms':delayed_ready_wait/1e6,
     'hypothetical_coupling_without_new_cost_ms':coupling/1e6 if r['wait_binding']=='whole_reader' else None,
     'coupling_pattern_supported':supported,'mechanism':'unresolved',
     'limits':['Counterfactual assumes other scheduling and costs fixed; not measured gain.',
              'Host release envelope is not actual GPU idle; compare independent GPU intervals.']}
    if supported:result['mechanism']='whole_reader_wait_after_current_ready'
    elif ready>n:result['mechanism']='current_data_late_or_overlap_insufficient'
    elif j1>n:result['mechanism']='data_ready_but_release_late_other_cause'
    else:result['mechanism']='no_exposed_release_wait'
    start=r.get('t_reader_start_ns')
    if isinstance(start,int):result['reader_started_after_need']=start>n
    # Uncertainty interval of dependency time; never subtract clocks across ranks.
    def delta(x):return max(0,done-x)-max(0,ready-x)
    result['coupling_need_uncertainty_range_ms']=([max(0,delta(n+u))/1e6,max(0,delta(max(0,n-u)))/1e6]
      if r['wait_binding']=='whole_reader' else None)
    return result

def analyze(doc):
    if doc.get('evidence_kind') not in ('measured','synthetic'):
        raise ValueError('explicit evidence_kind required')
    records=doc.get('records')
    if not isinstance(records,list):raise ValueError('records must be list')
    results=[classify(r) for r in records]
    return {'evidence_kind':doc['evidence_kind'],'results':results,
     'limits':['No summing waits across ranks: they may be the same critical interval.',
               'Synthetic fixtures only verify arithmetic and checks, not a production hypothesis.']}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('input');p.add_argument('--out');a=p.parse_args()
    d=analyze(json.loads(pathlib.Path(a.input).read_text()))
    text=json.dumps(d,indent=2,ensure_ascii=False)
    if a.out:
        out=pathlib.Path(a.out)
        if out.exists():raise FileExistsError('refuse to overwrite prior result')
        out.write_text(text+'\n')
    else:print(text)
