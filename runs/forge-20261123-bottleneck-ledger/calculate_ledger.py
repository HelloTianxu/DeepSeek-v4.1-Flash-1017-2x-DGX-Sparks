#!/usr/bin/env python3
"""Reconcile existing trace, exact checkpoint metadata, and conditional roofs."""
import json,pathlib,re,statistics
R=pathlib.Path(__file__).resolve().parent
BASE=R.parent
cfg=json.loads((R/'checkpoint-config.json').read_text())['text_config']
sh=json.loads((BASE/'forge-20261123-shape-audit/shape-summary.json').read_text())['results']
trace=json.loads((BASE/'forge-20261123-steady-trace/timeline.json').read_text())['kernels']
requests=json.loads((BASE/'forge-20261123-steady-trace/steady-summary.json').read_text())
meta=json.loads((R/'layer0-tensor-metadata.json').read_text())
experts={}
for t in meta:
 m=re.search(r'ffn\.(?:experts\.(\d+)|shared_experts)\.',t['name'])
 if m:
  eid=m.group(1) or 'shared'
  experts.setdefault(eid,[]).append(t)
counts={eid:sum(t['bytes'] for t in ts) for eid,ts in experts.items()}
print('expert metadata groups',len(counts),'bytes range',min(counts.values()),max(counts.values()),'sum',sum(counts.values()))
D=cfg['hidden_size'];I=cfg['moe_intermediate_size'];slots=cfg['num_experts_per_tok']+cfg['n_shared_experts']
report={'inputs':{'D':D,'full_I':I,'slots':slots,'layers':cfg['num_hidden_layers'],
 'local_I_from_launch':18*64,'D_from_combine':40*128,
 'bandwidth_GBs_per_Spark_spec':273},'layer0_storage':{'experts':len(counts),'all_bytes':sum(counts.values()),'min_bytes':min(counts.values()),'max_bytes':max(counts.values())},'requests':[]}

def union(xs):
 total=0;end=-1
 for a,b in sorted(xs):
  if a>end:total+=b-a
  elif b>end:total+=b-end
  end=max(end,b)
 return total*1e-9
for req in requests:
 rid=req['request'];xs=[x for x in trace if req['start']<=x[0]<req['end']]
 shape=[x for x in sh if x['request']==rid and x['kind']=='gateup']
 tokenlayers=sum(x['R_from_combine_grid']*x['calls'] for x in shape)
 M=tokenlayers*slots
 # TP2 head partial: D=5120, I_local=1152 from measured grid; full output down N=5120.
 Fhead=M*6*D*(18*64)
 Ffull=Fhead*2
 gs=sum(x['sum_ms'] for x in sh if x['request']==rid)*1e-3
 T=req['span_ms']*1e-3
 N=next(x['R_from_combine_grid'] for x in shape if x['calls']==20 and x['R_from_combine_grid']!=128)+8192
 budget=N/3500
 expert=[x for x in xs if 'grouped_mma_kernel' in x[3] or 'grouped_rows_kernel' in x[3]]
 ranges=[(x[0],x[1]) for x in xs]
 categories={}
 for x in xs:
  name=x[3]
  if 'grouped_mma_kernel' in name or 'grouped_rows_kernel' in name:kind='expert_core'
  elif 'nccl' in name:kind='collective'
  elif 'rot_in_kernel' in name:kind='rotation'
  elif 'unpack_kernel' in name:kind='unpack'
  elif 'down_combine_kernel' in name or 'gateup_epilogue_kernel' in name:kind='expert_epilogue'
  elif name.startswith('_hc_'):kind='HC'
  elif '_gemm' in name or 'cutlass' in name:kind='other_GEMM'
  elif 'attn' in name:kind='attention_named'
  else:kind='other'
  categories.setdefault(kind,[]).append((x[0],x[1]))
 row={'request':rid,'T_s':T,'prompt_N_inferred':N,'measured_tps_window':N/T,
  'token_layers':tokenlayers,'token_layers_without_bounded_replay':N*40,
  'full_equivalent_expert_TFLOP':Ffull/1e12,'head_expert_TFLOP':Fhead/1e12,
  'head_core_sum_s':gs,'head_core_union_s':union([(x[0],x[1]) for x in expert]),
  'head_core_effective_TFLOPS':Fhead/1e12/gs,
  'gpu_union_s':union(ranges),'gpu_no_kernel_s':T-union(ranges),
  'target_3500_budget_s':budget,'required_saved_s':T-budget,
  'optimistic_core_only_zero_time_floor_s':T-gs,
  'optimistic_core_only_ceiling_tps':N/(T-gs),
  'category_sums_s':{k:sum(b-a for a,b in v)*1e-9 for k,v in categories.items()},
  'streams':sorted(set(x[2] for x in xs)),
  'shapes':[]}
 for x in shape:
  r=x['R_from_combine_grid'];m=x['M_inferred'];down=next(y for y in sh if y['request']==rid and y['kind']=='down' and y['R_from_combine_grid']==r)
  dt=x['median_ms']+down['median_ms'];f=m*6*D*1152
  row['shapes'].append({'R':r,'M':m,'head_GFLOP':f/1e9,'core_median_ms_sum':dt,
  'effective_TFLOPS':f/1e9/dt,'gateup_capacity':x['grid'][2]//2,'down_capacity':down['grid'][2],
  'unique_storage_all_experts_half_GB_layer0':sum(counts.values())/2/1e9,
  'conditional_ideal_single_read_all_experts_ms':sum(counts.values())/2/273e9*1e3,
  'capacity_rows_ratio_gu':x['grid'][2]/2*64/m,
  'capacity_rows_ratio_down':down['grid'][2]*32/m})
 report['requests'].append(row)
 print(json.dumps(row,indent=2))
report['limits']=['Layer0 checkpoint bytes are unique serialized storage, not measured rank-cache bytes or DRAM traffic.',
 'Arithmetic inferred TP2 layout from measured launch and matching reference; no measured CUDA args.',
 'No-kernel on head can be peer waiting; not automatically CPU/OS waste.',
 'Category sums may overlap; do not add to wall time or turn into removable savings.',
 'Counterfactual subtract-core floor assumes all observed core time lies on path and removal leaves everything else fixed; not realized improvement.',
 'Bandwidth roof needs actual traffic and sustainable bandwidth; no FP16 peak benchmark available.']
(R/'ledger.json').write_text(json.dumps(report,indent=2))
