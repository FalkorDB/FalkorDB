# N concurrent writes spread over G graphs (each graph gets < 1024 writes), then probe liveness.
import asyncio, sys, time, redis
port=int(sys.argv[1]); N=int(sys.argv[2]); G=int(sys.argv[3]); Q=sys.argv[4]; DUR=float(sys.argv[5])
def enc(*a):
    s=f"*{len(a)}\r\n"
    for x in a: s+=f"${len(x.encode())}\r\n"+x+"\r\n"
    return s.encode()
done=0; errs={}
async def one(i):
    global done
    r,w=await asyncio.open_connection('127.0.0.1',port)
    w.write(enc("GRAPH.QUERY",f"g{i%G}",Q)); await w.drain()
    line=await r.readline()
    if line.startswith(b'-'): errs[line[:60]]=errs.get(line[:60],0)+1
    done+=1; w.close()
async def ping():
    try:
        r,w=await asyncio.open_connection('127.0.0.1',port); w.write(b"PING\r\n"); await w.drain()
        return (await asyncio.wait_for(r.readline(),2)).strip()
    except Exception as e: return repr(e)[:30]
async def main():
    tasks=[asyncio.create_task(one(i)) for i in range(N)]
    t0=time.time()
    while time.time()-t0<DUR:
        await asyncio.sleep(3)
        print(f"t={time.time()-t0:.0f}s done={done}/{N} ping={await ping()} errs={errs}",flush=True)
        if done==N: break
asyncio.run(main())
