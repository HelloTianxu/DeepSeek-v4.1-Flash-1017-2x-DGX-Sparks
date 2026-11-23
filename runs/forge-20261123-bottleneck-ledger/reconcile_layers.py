#!/usr/bin/env python3
"""Per-layer chronological reconciliation; competing roofs remain hypotheses."""
import json,pathlib,statistics,math
R=pathlib.Path(__file__).resolve().parent;B=R.parent
trace=json.loads((B/'forge-20261123-steady-trace/timeline.json').read_text())['kernels']
reqs=json.loads((B/'forge-20261123-steady-trace/steady-summary.json').read_text())
storage=json.loads((R/'all-layer-expert-storage.json').read_text())
results=[]
for req in reqs:
 xs=[x for x in trace if req['start']<=x[0]<req['end']]
 gu=[(i,x) for i,x in enumerate(xs) if 'grouped_mma_kernel' in x[3]]
 labels=[]
 for idx,x in gu:
  comb=next(y for y in xs[idx+1:] if 'down_combine_kernel' in y[3] and y[2]==x[2])
  down=next(y for y in xs[idx+1:] if 'grouped_rows_kernel' in y[3] and y[2]==x[2])
  n=comb[4]
  # Ordered pattern verified against replay code: 5 encoder chunks*20 plus decoder128*20.
  li=len(labels)%20 if n!=128 else 20+sum(z['R']==128 for z in labels)
  st=storage[str(li)]
  dt=((x[1]-x[0])+(down[1]-down[0]))*1e-6
  labels.append({'layer_inferred':li,'R':n,'M':n*7,'core_ms':dt,
  'gateup_ms':(x[1]-x[0])*1e-6,'down_ms':(down[1]-down[0])*1e-6,
  'all_unique_half_GB':st['all_bytes']/2/1e9,
  'GB_at_spec_bandwidth':dt*273/1000,
  'once_read_all_effective_GBs':st['all_bytes']/2/1e6/dt,
  'min_unique_bytes_half':(6*st['routed_min_bytes']+st['shared_bytes'])/2,
  'active_experts_uniform_hypothesis':384*(1-(1-6/384)**n)+1})
 assert len(labels)==120 and len([x for x in labels if x['R']==128])==20
 bulk={li:statistics.mean(z['core_ms']/z['M'] for z in labels if z['layer_inferred']==li and z['R']==2048) for li in range(20)}
 tail=[z for z in labels if z['R']!=2048 and z['R']!=128]
 tail_surplus=sum(z['core_ms']-bulk[z['layer_inferred']]*z['M'] for z in tail)
 groups=[]
 for n in sorted(set(z['R'] for z in labels)):
  a=[z for z in labels if z['R']==n]
  groups.append({'R':n,'calls':len(a),'sum_ms':sum(z['core_ms'] for z in a),
  'median_total_core_ms_paired':statistics.median(z['core_ms'] for z in a),
  'bytes_at_273GBs_median_GB':statistics.median(z['GB_at_spec_bandwidth'] for z in a),
  'unique_all_half_GB_mean':statistics.mean(z['all_unique_half_GB'] for z in a),
  'once_read_all_GBs_median':statistics.median(z['once_read_all_effective_GBs'] for z in a)})
 # hypothetical whole-path proportionality to large shape; not a causal ceiling
 actual=sum(z['core_ms'] for z in labels)
 large=[z for z in labels if z['R']==2048]
 rate=sum(z['core_ms'] for z in large)/sum(z['M'] for z in large)
 project=rate*sum(z['M'] for z in labels)
 result={'request':req['request'],'groups':groups,'inferred_layer_rows':labels,
 'tail_only_same_layer_rate_surplus_ms':tail_surplus,
 'conditional_all_small_shapes_at_bulk_rate_saved_ms':actual-project,
 'reference_pattern':'R2048x20 four chunks; tailx20; R128x20 decoder replay'}
 results.append(result)
 print(json.dumps({k:v for k,v in result.items() if k!='inferred_layer_rows'},indent=2))
(R/'layer-shape-ledger.json').write_text(json.dumps({'limitations':['Layer labels inferred by trace order and bounded replay source, not NVTX IDs.',
 'Storage from serialized checkpoint, not DRAM traffic; all-active assumption cannot establish bandwidth utilization.',
 'Uniform active expert hypothesis not measured. Same-layer tail comparison still confounded by sequence position/routing.'], 'results':results},indent=2))
