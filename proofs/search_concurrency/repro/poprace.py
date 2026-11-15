# Index population vs concurrent writer: after quiescence the index must agree with a scan.
import redis, threading, time, sys, random
from redis.retry import Retry
from redis.backoff import NoBackoff
port=int(sys.argv[1]); N=int(sys.argv[2]); ROUNDS=int(sys.argv[3])
def conn(): return redis.Redis(port=port, socket_timeout=300, retry=Retry(NoBackoff(),0))
r=conn()
if r.exists("pr"): r.execute_command("GRAPH.DELETE","pr")
r.execute_command("GRAPH.QUERY","pr",f"UNWIND range(0,{N-1}) AS x CREATE (:L {{k:x, v:0}})")
bad=0
for rnd in range(ROUNDS):
    stop=False
    wfail=[]
    def w():
        try:
            c=conn(); i=0
            while not stop:
                m=random.randrange(50)
                c.execute_command("GRAPH.QUERY","pr",f"MATCH (n:L) WHERE n.k % 50 = {m} SET n.v = n.v + 1"); i+=1
        except Exception as e:
            wfail.append(repr(e)[:80])
    # daemon: if this round raises, the writer must not keep the process alive
    t=threading.Thread(target=w, daemon=True); t.start()
    time.sleep(0.05)
    r.execute_command("GRAPH.QUERY","pr","CREATE INDEX FOR (n:L) ON (n.v)")
    # wait for population
    while True:
        st=r.execute_command("GRAPH.QUERY","pr","CALL db.indexes() YIELD status RETURN status")[1]
        if st and st[0][0]==b'OPERATIONAL': break
        time.sleep(0.01)
    time.sleep(0.2); stop=True; t.join()
    if wfail:   # the writer died: this round did not race the index build
        print(f"round {rnd}: WRITER FAILED {wfail[0]}", flush=True); bad+=1
        r.execute_command("GRAPH.QUERY","pr","DROP INDEX FOR (n:L) ON (n.v)"); continue
    vals=r.execute_command("GRAPH.QUERY","pr","MATCH (n:L) RETURN n.v, count(n) ORDER BY n.v")[1]
    mism=[]
    for v,c in vals:
        ci=r.execute_command("GRAPH.QUERY","pr",f"MATCH (n:L) WHERE n.v = {v} RETURN count(n)")[1][0][0]
        if ci!=c: mism.append((v,c,ci))
    # also: index hits whose actual value differs. Not vacuous: `n.v = q` is served by the
    # index (Node By Index Scan in GRAPH.EXPLAIN), and the `WITH ... WHERE n.v <> q` filter after
    # the WITH re-reads the node's stored property, so a stale index entry shows up here.
    wrong=r.execute_command("GRAPH.QUERY","pr","UNWIND range(0,200) AS q MATCH (n:L) WHERE n.v = q WITH q, n WHERE n.v <> q RETURN count(n)")[1][0][0]
    print(f"round {rnd}: distinct v={len(vals)} mismatches={mism[:5]} n_mism={len(mism)} wrong_hits={wrong}", flush=True)
    bad+=len(mism)>0 or wrong>0
    r.execute_command("GRAPH.QUERY","pr","DROP INDEX FOR (n:L) ON (n.v)")
print("BAD ROUNDS", bad)
