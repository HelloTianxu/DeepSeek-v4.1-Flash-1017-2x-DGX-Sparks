#!/usr/bin/env python3
"""Exact aligned-read geometry; uniform residues are a hypothesis, not routing data."""
import json,pathlib,math
R=pathlib.Path(__file__).resolve().parent
m=json.loads((R/'engram-absolute-offsets.json').read_text());out=[]
for t in m:
 b=t['absolute_data_start'];w=t['shape'][1];period=4096//math.gcd(4096,w)
 sizes=[]
 for row in range(period):
  off=b+row*w;lo=off//4096*4096;size=((off+w-lo+4095)//4096)*4096;sizes.append(size)
 assert all(s in (4096,8192) for s in sizes)
 out.append({'tensor':t['tensor'],'base_mod4096':b%4096,'row_bytes':w,'residue_period_rows':period,
 'crossing_residues':[i for i,s in enumerate(sizes) if s==8192],
 'crossing_fraction_uniform_residue_hypothesis':sizes.count(8192)/period,
 'mean_request_bytes_uniform_residue_hypothesis':sum(sizes)/period})
print(json.dumps(out,indent=2))
rows=24576
for layer in (1,14):
 a=[x for x in out if x['tensor'].startswith(f'layers.{layer}.')]
 weight=next(x for x in a if '.weight' in x['tensor']);scale=next(x for x in a if '.scale' in x['tensor'])
 assert weight['crossing_fraction_uniform_residue_hypothesis']==1/16
 assert scale['crossing_fraction_uniform_residue_hypothesis']==0
 avg=weight['mean_request_bytes_uniform_residue_hypothesis']+scale['mean_request_bytes_uniform_residue_hypothesis']
 print('layer',layer,'uniform request bytes',rows*avg,'amplification',avg/264)
 assert avg/264==32
(R/'engram-block-geometry.json').write_text(json.dumps({'evidence':'actual file offsets; exact geometry; routing residue distribution unmeasured','tensors':out,
 'uniform_residue_scenario':{'rows_per_rank_layer_chunk':rows,'request_bytes':rows*8448,'useful_bytes':rows*264,'amplification':32},
 'limits':['No measured routing IDs or duplicate block count; do not claim measured drive traffic.',
 'Block/request dedup can reduce repeated reads, but cannot infer benefit from row size alone.']},indent=2))
