#!/usr/bin/env python3
"""Bounded Forge single-request probe; raw SSE evidence, no credentials in output."""
import argparse, hashlib, json, pathlib, random, statistics, time, urllib.request
p=argparse.ArgumentParser();p.add_argument('--arm',required=True);p.add_argument('--case',required=True,choices=['warm','prefill8','prefill16','code','prose','thinking','tool']);p.add_argument('--rep',type=int,default=0);p.add_argument('--match',default='forge-20261123');a=p.parse_args()
root=pathlib.Path(__file__).resolve().parent
cfg=json.loads((pathlib.Path('/Users/zhangtianxu/Documents/AI/DGS-Spark/dgx-tp2-monitor/nodes.json')).read_text());node=cfg['nodes'][0]
line=pathlib.Path(node['api_key_file']).read_text().strip().splitlines()[0];key=line.split(':',1)[-1].strip()
nonce=hashlib.sha256((f'{a.match}-{a.case}-{a.rep}-{a.arm}' if a.case.startswith('prefill') else f'{a.match}-{a.case}-{a.rep}').encode()).hexdigest()[:24]
if a.case.startswith('prefill'):
    # Distinct first position, natural local text, fixed within matched pair.
    corpus=(root/'corpus.txt').read_text(); chars=28000 if a.case=='prefill8' else 55000
    offset=(a.rep*1301)%4000; text=(corpus*8)[offset:offset+chars]
    content=f'Experiment ID {nonce}. Read the following technical notes carefully.\n{text}\nReply with the single word: ready.';budget=1
elif a.case=='code':
    content=f'Experiment ID {nonce}. Write a complete Python standard-library implementation of a bounded LRU cache with TTL, thread safety, monotonic time, generic type annotations, detailed docstrings and unittest tests for expiry, eviction and concurrency. Provide the full code without placeholders.';budget=512
elif a.case=='prose':
    content=f'Experiment ID {nonce}. Write a detailed historical essay on the printing press: metal movable type, Gutenberg, printing economics, literacy, censorship, and differences between Europe and East Asia. Use many paragraphs and concrete examples. Do not stop early.';budget=512
elif a.case=='thinking':
    content=f'Experiment ID {nonce}. Explain why the sum of the first n odd positive integers is n squared. Give two proofs.';budget=128
elif a.case=='tool':
    content=f'Experiment ID {nonce}. Use get_weather to obtain the current temperature in Beijing. Do not guess.';budget=128
else:
    content=f'Experiment ID {nonce}. Reply with the single word ready.';budget=16
body={'model':'DeepSeek-V4.1-Flash-1017','messages':[{'role':'user','content':content}],'max_tokens':budget,'temperature':0,'seed':1234,'stream':True,'stream_options':{'include_usage':True},'chat_template_kwargs':{'enable_thinking':a.case=='thinking'}}
if a.case=='tool':
    body['tools']=[{'type':'function','function':{'name':'get_weather','description':'Get current weather for a city','parameters':{'type':'object','properties':{'city':{'type':'string'}},'required':['city']}}}];body['tool_choice']='auto'
req=urllib.request.Request(node['api_url']+'/v1/chat/completions',json.dumps(body).encode(),{'Content-Type':'application/json','Authorization':'Bearer '+key})
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
def metrics():
    r=urllib.request.Request(node['api_url']+'/metrics',headers={'Authorization':'Bearer '+key})
    with opener.open(r,timeout=5) as s:return s.read().decode()
before=metrics();t0=time.monotonic();events=[];output=[];first=last=None;usage=None;server=None;finish=None;err=None
try:
    with opener.open(req,timeout=50) as resp:
        for raw in resp:
            line=raw.decode().strip()
            if not line.startswith('data:'):continue
            data=line[5:].strip()
            if data=='[DONE]':break
            j=json.loads(data);elapsed=time.monotonic()-t0;events.append({'at_s':elapsed,'data':j})
            if j.get('usage'):usage=j['usage']
            if j.get('tensorfold'):server=j['tensorfold']
            for choice in j.get('choices',[]):
                d=choice.get('delta',{});text=d.get('content') or d.get('reasoning_content') or d.get('reasoning') or d.get('tool_calls')
                if text:
                    if first is None:first=elapsed
                    last=elapsed;output.append(d)
                finish=choice.get('finish_reason') or finish
except Exception as e:err=f'{type(e).__name__}: {e}'
end=time.monotonic()-t0;after=metrics();gaps=[];ts=[e['at_s'] for e in events if any(any((ch.get('delta') or {}).get(k) for k in ['content','reasoning_content','reasoning','tool_calls']) for ch in e['data'].get('choices',[]))]
gaps=[b-a for a,b in zip(ts,ts[1:])];ct=(usage or {}).get('completion_tokens');pt=(usage or {}).get('prompt_tokens');cached=((usage or {}).get('prompt_tokens_details') or {}).get('cached_tokens')
r={'arm':a.arm,'case':a.case,'rep':a.rep,'request':body,'usage':usage,'server':server,'finish':finish,'error':err,'first_s':first,'last_s':last,'total_s':end,'sse_events':events,'output':output,'metrics_before':before,'metrics_after':after,'prompt_sha':hashlib.sha256(content.encode()).hexdigest(),'prefill_tps':(pt-cached)/first if pt is not None and cached is not None and first else None,'decode_tps':(ct-1)/(last-first) if ct and ct>1 and last>first else None,'max_sse_gap_s':max(gaps) if gaps else None,'p99_sse_gap_s':sorted(gaps)[min(len(gaps)-1,int(.99*len(gaps)))] if gaps else None,'peak_note':'SSE timing retained; chunks are not guaranteed individual tokens, so no subwindow token/s claimed.'}
f=root/f'{a.arm}-{a.case}-{a.rep}.json'
if f.exists():raise RuntimeError('Refuse overwrite existing experimental raw data')
f.write_text(json.dumps(r,ensure_ascii=False,indent=2));print(json.dumps({k:r[k] for k in ['arm','case','rep','usage','server','error','first_s','total_s','prefill_tps','decode_tps','max_sse_gap_s']},ensure_ascii=False),flush=True)
if err or usage is None:raise SystemExit(1)
