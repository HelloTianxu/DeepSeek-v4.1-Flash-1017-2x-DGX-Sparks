#!/usr/bin/env python3
"""API overlap is correlation, not causal wait attribution."""
import json,pathlib
R=pathlib.Path(__file__).resolve().parent;B=R.parent
d=json.loads((B/'forge-20261123-steady-trace/timeline.json').read_text());q=json.loads((B/'forge-20261123-steady-trace/steady-summary.json').read_text())
def merge(xs):
 out=[]
 for a,b in sorted(xs):
  if not out or a>out[-1][1]:out.append([a,b])
  else:out[-1][1]=max(b,out[-1][1])
 return out

def length(xs):return sum(b-a for a,b in merge(xs))*1e-9

def intersection(a,b):
 a=merge(a);b=merge(b);i=j=0;out=[]
 while i<len(a) and j<len(b):
  lo=max(a[i][0],b[j][0]);hi=min(a[i][1],b[j][1])
  if hi>lo:out.append((lo,hi))
  if a[i][1]<b[j][1]:i+=1
  else:j+=1
 return out
out=[]
for r in q:
 lo,hi=r['start'],r['end'];ks=merge([(max(lo,x[0]),min(hi,x[1])) for x in d['kernels'] if x[0]<hi and x[1]>lo]);gaps=[];end=lo
 for a,b in ks:
  if a>end:gaps.append((end,a))
  end=max(end,b)
 if end<hi:gaps.append((end,hi))
 names={}
 for a,b,n in d['api']:
  if b<=lo or a>=hi:continue
  names.setdefault(n,[]).append((max(lo,a),min(hi,b)))
 stats=[]
 for n,xs in names.items():
  stats.append({'name':n,'calls':len(xs),'sum_clipped_s':sum(b-a for a,b in xs)*1e-9,
   'union_s':length(xs),'overlap_no_kernel_s':length(intersection(xs,gaps)),
   'overlap_gpu_kernel_s':length(intersection(xs,ks))})
 row={'request':r['request'],'no_kernel_s':length(gaps),'api_by_sum':sorted(stats,key=lambda x:-x['sum_clipped_s'])[:12],
 'largest_gaps_ms':[round((b-a)*1e-6,4) for a,b in sorted(gaps,key=lambda x:-(x[1]-x[0]))[:10]],
 'no_kernel_overlapping_any_API_s':length(intersection([x for xs in names.values() for x in xs],gaps))}
 out.append(row);print(json.dumps(row,indent=2))
(R/'wait-ledger.json').write_text(json.dumps({'limits':['API overlap with idle is temporal association, not causal removable waste.',
 'No worker GPU trace; peer wait and filesystem/CPU work unresolved.'],'results':out},indent=2))
