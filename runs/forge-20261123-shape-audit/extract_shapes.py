#!/usr/bin/env python3
"""Read existing trace only; no GPU execution or service changes."""
import json, pathlib, shlex, subprocess, hashlib
ROOT = pathlib.Path(__file__).resolve().parent
BASE = ROOT.parent / 'forge-20261123-steady-trace'
ranges = json.loads((BASE / 'steady-summary.json').read_text())
remote = r'''
import sqlite3,json,pathlib
p=pathlib.Path.home()/'.cache/forge-steady-20261123/trace.sqlite'
c=sqlite3.connect('file:'+str(p)+'?mode=ro',uri=True)
q="SELECT k.start,k.end,k.streamId,s.value,k.gridX,k.gridY,k.gridZ,k.blockX FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName WHERE s.value LIKE '%grouped_mma_kernel%' OR s.value LIKE '%grouped_rows_kernel%' OR s.value LIKE '%down_combine_kernel%' ORDER BY k.start"
print(json.dumps({'source':str(p),'rows':c.execute(q).fetchall()}))
c.close()
'''
p = subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=5','ztxjzh@192.168.1.118', 'timeout 12s python3 -c '+shlex.quote(remote)], text=True,capture_output=True,timeout=18)
if p.returncode:
    raise RuntimeError(f'remote query failed {p.returncode}: {p.stderr}')
doc=json.loads(p.stdout)
(ROOT / 'launches-raw.json').write_text(json.dumps(doc,indent=2))
results=[]
import statistics
for req in ranges:
    rows=[x for x in doc['rows'] if req['start'] <= x[0] < req['end']]
    groups={}
    for idx,x in enumerate(rows):
        name=x[3]
        if 'down_combine_kernel' in name: continue
        # Match subsequent combine on the same stream; R and slots are inferred
        # from reference launch ABI, not captured CUDA argument values.
        nxt=next((y for y in rows[idx+1:] if y[2]==x[2] and 'down_combine_kernel' in y[3]),None)
        if nxt is None: raise RuntimeError('no matching combine')
        r=nxt[4]; slots=nxt[7]//32
        kind='gateup' if 'grouped_mma_kernel' in name else 'down'
        key=(kind,r,slots,x[4],x[5],x[6])
        groups.setdefault(key,[]).append((x[1]-x[0])*1e-6)
    for key,ts in groups.items():
        kind,r,slots,gx,gy,gz=key
        results.append({'request':req['request'],'kind':kind,'R_from_combine_grid':r,
          'slots_from_combine_block':slots,'M_inferred':r*slots,
          'grid':[gx,gy,gz],'calls':len(ts),'median_ms':statistics.median(ts),
          'min_ms':min(ts),'max_ms':max(ts),'sum_ms':sum(ts),
          'ms_per_1K_pairs':statistics.median(ts)/(r*slots)*1000})
report={'evidence':'engineering trace; shape labels inferred using reference ABI',
 'script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
 'limitations':['Same request, not controlled shape sweep: layer, position, routing and cache differ.',
 'Grid work-list capacity is not valid tiles, padded rows, or active expert count.',
 'Kernel durations are not end-to-end critical-path savings.'], 'results':results}
(ROOT/'shape-summary.json').write_text(json.dumps(report,indent=2))
for x in results:
    print(x['request'],x['kind'],x['R_from_combine_grid'],x['M_inferred'],x['calls'],round(x['median_ms'],4),round(x['ms_per_1K_pairs'],4),x['grid'])
