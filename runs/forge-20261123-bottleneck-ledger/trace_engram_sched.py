#!/usr/bin/env python3
import subprocess,shlex,pathlib
ROOT=pathlib.Path(__file__).resolve().parent
remote=r"""
import subprocess,pathlib,json
name='forge-engram-sched-20261123'
x=json.loads(subprocess.run(['docker','inspect','dsv41-1017-exl3-tf'],capture_output=True,text=True,check=True).stdout)[0];pid=x['State']['Pid']
p='/proc/'+str(pid)+'/root/opt/tensorfold/native/bin/tensorfold-native'
script=f'''uprobe:{p}:prompt.engramAhead1 /pid == {pid}/ {{ @eh[tid]=arg0; @j[tid]=arg5; @at[tid]=arg3; @n[tid]=arg4; printf("E %llu %d %llu %llu %llu\\n",nsecs,tid,@j[tid],@at[tid],@n[tid]); }}
uretprobe:{p}:prompt.engramAhead1 /pid == {pid}/ {{ printf("O %d %llu %llu %llu %llu %llu\\n",tid,@j[tid],@at[tid],@Offns[tid],@Runqueue[tid],@Switches[tid]); delete(@Offns[tid]); delete(@Runqueue[tid]); delete(@Switches[tid]); printf("I %d %llu %llu %llu %llu %llu %llu %llu\\n",tid,@j[tid],@at[tid],@Ins[tid],@Ic[tid],@Inr[tid],@Iaccepted[tid],@Ierrors[tid]); delete(@Ins[tid]); delete(@Ic[tid]); delete(@Inr[tid]); delete(@Iaccepted[tid]); delete(@Ierrors[tid]); printf("B %d %llu %llu %llu %llu %llu %llu\\n",tid,@j[tid],@at[tid],@Sns[tid],@Scount[tid],@Wns[tid],@Wcount[tid]); delete(@Sns[tid]); delete(@Scount[tid]); delete(@Wns[tid]); delete(@Wcount[tid]); printf("X %llu %d %llu %llu %llu %llu\\n",nsecs,tid,@j[tid],@at[tid],@n[tid],retval); delete(@eh[tid]); delete(@j[tid]); delete(@at[tid]); delete(@n[tid]); }}
uprobe:/proc/{pid}/root/lib/aarch64-linux-gnu/libc.so.6:pthread_join /pid == {pid}/ {{ printf("J %llu %d %llu\\n",nsecs,tid,arg0); }}
uretprobe:/proc/{pid}/root/lib/aarch64-linux-gnu/libc.so.6:pthread_join /pid == {pid}/ {{ printf("K %llu %d %llu\\n",nsecs,tid,retval); }}
uprobe:{p}:engram.Hasher.hashes /pid == {pid} && @eh[tid]/ {{ printf("H %llu %d %llu %llu\\n",nsecs,tid,@j[tid],@at[tid]); }}
uretprobe:{p}:engram.Hasher.hashes /pid == {pid} && @eh[tid]/ {{ printf("h %llu %d %llu %llu\\n",nsecs,tid,@j[tid],@at[tid]); }}
uprobe:{p}:prompt.aioGather /pid == {pid} && @eh[tid]/ {{ printf("A %llu %d %llu %llu\\n",nsecs,tid,@j[tid],@at[tid]); }}
uretprobe:{p}:prompt.aioGather /pid == {pid} && @eh[tid]/ {{ printf("a %llu %d %llu %llu\\n",nsecs,tid,@j[tid],@at[tid]); }}
uprobe:{p}:engram.decodeRows /pid == {pid} && @eh[tid]/ {{ printf("D %llu %d %llu %llu\\n",nsecs,tid,@j[tid],@at[tid]); }}
uretprobe:{p}:engram.decodeRows /pid == {pid} && @eh[tid]/ {{ printf("d %llu %d %llu %llu\\n",nsecs,tid,@j[tid],@at[tid]); }}
uprobe:{p}:engram_aio.Aio.start /pid == {pid} && @eh[tid]/ {{ @Sstart[tid]=nsecs; }}
uretprobe:{p}:engram_aio.Aio.start /pid == {pid} && @Sstart[tid]/ {{ @Sns[tid]+=nsecs-@Sstart[tid]; @Scount[tid]+=1; delete(@Sstart[tid]); }}
uprobe:{p}:engram_aio.Aio.wait /pid == {pid} && @eh[tid]/ {{ @Wstart[tid]=nsecs; }}
uretprobe:{p}:engram_aio.Aio.wait /pid == {pid} && @Wstart[tid]/ {{ @Wns[tid]+=nsecs-@Wstart[tid]; @Wcount[tid]+=1; delete(@Wstart[tid]); }}
tracepoint:syscalls:sys_enter_io_submit /pid == {pid} && @eh[tid]/ {{ @Is[tid]=nsecs; if(args->nr > 0) {{ @Inr[tid]+=(uint64)args->nr; }} }}
tracepoint:syscalls:sys_exit_io_submit /pid == {pid} && @Is[tid]/ {{ @Ins[tid]+=nsecs-@Is[tid]; @Ic[tid]+=1; if(args->ret > 0) {{ @Iaccepted[tid]+=(uint64)args->ret; }} else {{ @Ierrors[tid]+=1; }} delete(@Is[tid]); }}
tracepoint:sched:sched_switch /@Is[args->prev_pid]/ {{ @Offstart[args->prev_pid]=nsecs; @Offstate[args->prev_pid]=args->prev_state; @Switches[args->prev_pid]+=1; if(args->prev_state != 0) {{ @SleepStacks[kstack(12)]=count(); }} }}
tracepoint:sched:sched_switch /@Offstart[args->next_pid]/ {{ @Offns[args->next_pid]+=nsecs-@Offstart[args->next_pid]; if(@Offstate[args->next_pid] == 0) {{ @Runqueue[args->next_pid]+=nsecs-@Offstart[args->next_pid]; }} delete(@Offstart[args->next_pid]); delete(@Offstate[args->next_pid]); }}
profile:hz:49 /@Is[tid]/ {{ @RunningStacks[kstack(12)]=count(); }}
BEGIN {{ printf("FORGE_ATTACH_READY\\n"); }}
interval:s:20 {{ exit(); }}
'''
if subprocess.run(['docker','ps','-a','--filter','name=^/'+name+'$','--format','{{.ID}}'],capture_output=True,text=True,check=True).stdout.strip():raise RuntimeError('name exists')
try:
 cmd=['docker','run','--rm','--name',name,'--network','none','--pid','host','--privileged','--memory','512m','--entrypoint','nsenter','tensorfold-dsv41:exl3-tp2-8348338a8d90','-t','1','-m','--','timeout','30s','bpftrace','-e',script]
 p=subprocess.Popen(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
 for line in p.stdout:
  print(line,end='',flush=True)
 p.wait(timeout=5);print('TRACE_RETURN',p.returncode,flush=True)
 if p.returncode:raise RuntimeError('trace failed')
finally:
 q=subprocess.run(['docker','ps','-a','--filter','name=^/'+name+'$','--format','{{.ID}}'],capture_output=True,text=True,check=True)
 if q.stdout.strip():subprocess.run(['docker','rm','-f',name],capture_output=True,text=True,check=True)
 q=subprocess.run(['docker','ps','-a','--filter','name=^/'+name+'$','--format','{{.ID}}'],capture_output=True,text=True,check=True);print('CLEANUP_ABSENT',not bool(q.stdout.strip()))
 if q.stdout.strip():raise RuntimeError('cleanup failed')
"""
p=subprocess.Popen(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=5','ztxjzh@192.168.1.118','timeout 45s python3 -u -c '+shlex.quote(remote)],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
lines=[];probe=None
try:
 for line in p.stdout:
  lines.append(line);print(line,end='',flush=True)
  if 'FORGE_ATTACH_READY' in line and probe is None:
   probe=subprocess.run(['python3','probe.py','--arm','SCHEDSYNC1','--case','prefill8','--rep','37','--match','forge-20261123-host-events'],cwd=ROOT.parent/'forge-20261123-interleave',capture_output=True,text=True,timeout=30)
   print(probe.stdout,probe.stderr,flush=True)
 p.wait(timeout=5)
finally:
 if p.poll() is None:p.terminate();p.wait(timeout=5)
f=ROOT/'engram-sched-trace.txt'
if f.exists():raise RuntimeError('refuse overwrite')
f.write_text(''.join(lines))
if p.returncode:raise RuntimeError('remote trace failed')
if probe is None or probe.returncode:raise RuntimeError('synchronized request failed')
