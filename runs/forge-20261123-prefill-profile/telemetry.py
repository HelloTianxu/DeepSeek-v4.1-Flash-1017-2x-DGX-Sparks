#!/usr/bin/env python3
"""30-second read-only remote telemetry, no background processes on nodes."""
import concurrent.futures,json,pathlib,subprocess,shlex,time
root=pathlib.Path(__file__).resolve().parent
code=r'''import pathlib,json,time,subprocess
print(json.dumps({'at':time.time(),'stat':pathlib.Path('/proc/stat').read_text(),'diskstats':pathlib.Path('/proc/diskstats').read_text(),'memory':pathlib.Path('/proc/meminfo').read_text(),'gpu':subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,clocks.sm,power.draw,temperature.gpu','--format=csv,noheader'],capture_output=True,text=True).stdout,'processes':[{'pid':p.name,'stat':(p/'stat').read_text(),'io':''} for p in pathlib.Path('/proc').iterdir() if p.name.isdigit() and (p/'comm').exists() and 'tensorfold' in (p/'comm').read_text()],'rail':{str(p):p.read_text() for p in pathlib.Path('/sys/class/infiniband').glob('*/ports/1/counters/*') if p.name in ['port_xmit_data','port_rcv_data','port_xmit_wait']}}))'''
commands={'head':['ztxjzh@192.168.1.118'],'worker':['-J','ztxjzh@192.168.1.118','ztxjzh@192.168.100.11']}
def one(item):
 name,args=item
 try:
  s=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=4',*args,'python3 -c '+shlex.quote(code)],capture_output=True,text=True,timeout=9)
  return {'node':name,'exit':s.returncode,'data':json.loads(s.stdout) if s.returncode==0 else None,'error':s.stderr}
 except Exception as e:return {'node':name,'error':str(e)}
t0=time.monotonic()
with (root/'telemetry-overlap.jsonl').open('x') as f, concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
 while time.monotonic()-t0<30:
  for row in pool.map(one,commands.items()):f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush()
  print('telemetry elapsed',round(time.monotonic()-t0,1),flush=True);time.sleep(1)
