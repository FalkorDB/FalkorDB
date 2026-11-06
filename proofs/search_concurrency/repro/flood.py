# Flood one graph with N concurrent writes (one per connection), then probe liveness.
import asyncio, sys, time, redis
port=int(sys.argv[1]); N=int(sys.argv[2]); Q=sys.argv[3] if len(sys.argv)>3 else "UNWIND range(1,20000) AS x CREATE (:N)"
def enc(*a):
    s=f"*{len(a)}\r\n"
    for x in a:
        b=x.encode(); s+=f"${len(b)}\r\n"+x+"\r\n"
    return s.encode()
done=0; errs={}
async def one(i):
    global done
    w=None
    try:
        r,w=await asyncio.open_connection('127.0.0.1',port)
        w.write(enc("GRAPH.QUERY","fl",Q)); await w.drain()
        line=await r.readline()
        if line==b'': errs[b'EOF without a reply']=errs.get(b'EOF without a reply',0)+1
        elif line.startswith(b'-'): errs[line[:80]]=errs.get(line[:80],0)+1
    except Exception as e:
        k=repr(e)[:80].encode(); errs[k]=errs.get(k,0)+1
    finally:
        done+=1           # one per task, whatever happened
        if w is not None: w.close()
async def main():
    tasks=[asyncio.create_task(one(i)) for i in range(N)]
    t0=time.time()
    deadline=float(sys.argv[4]) if len(sys.argv)>4 else 60.0
    while time.time()-t0<deadline:
        await asyncio.sleep(2)
        def probe():
            try: return redis.Redis(port=port,socket_timeout=2).ping()
            except Exception as e: return repr(e)[:60]
        ok=await asyncio.to_thread(probe)   # off the loop thread: reads and `done` keep moving
        print(f"t={time.time()-t0:.0f}s done={done}/{N} ping={ok} errs={errs}",flush=True)
        if done==N: break
    for t in tasks: t.cancel()
asyncio.run(main())
