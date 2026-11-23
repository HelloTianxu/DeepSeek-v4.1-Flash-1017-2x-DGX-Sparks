#!/usr/bin/env python3
"""Parse live function event spans; not GPU critical-path attribution."""
import json,pathlib
R=pathlib.Path(__file__).resolve().parent
starts={};readers={};out=[]
for line in (R/'engram-phase-trace.txt').read_text().splitlines():
 z=line.split()
 if not z or z[0] not in ['E','X','H','h','A','a','D','d']:continue
 tag=z[0];v=list(map(int,z[1:]));t,tid=v[:2]
 if tag=='E':readers[tid]={'tid':tid,'j':v[2],'chunk':v[3],'n':v[4],'start':t,'phases_ms':{}}
 elif tag=='X':
  r=readers.pop(tid);r['duration_ms']=(t-r['start'])/1e6;r['return_code']=v[5];r['other_ms']=r['duration_ms']-sum(r['phases_ms'].values());out.append(r)
 elif tag.isupper():starts[(tid,tag)]=t
 else:
  a=starts.pop((tid,tag.upper()));readers[tid]['phases_ms'][{'h':'hash','a':'aio_gather','d':'decode'}[tag]]=(t-a)/1e6
assert not readers and not starts and len(out)==10
bulk=[r for r in out if r['n']==2048]
total=sum(r['duration_ms'] for r in bulk);aio=sum(r['phases_ms']['aio_gather'] for r in bulk)
result={'evidence':'one synchronized live head request; 2048-row preparation spans','records':out,
 'bulk_total_ms':total,'bulk_aio_ms':aio,'bulk_aio_fraction':aio/total,
 'limits':['aioGather includes submit, wait, reap/copy, allocation and kernel/drive work; not pure SSD latency.', 'No same-request GPU clock mapping or paired observer controls.']}
p=R/'engram-phase-summary.json';assert not p.exists();p.write_text(json.dumps(result,indent=2))
for r in out:print(r['chunk'],r['j'],r['n'],round(r['duration_ms'],3),{k:round(v,3) for k,v in r['phases_ms'].items()})
print('bulk AIO fraction',aio/total)
