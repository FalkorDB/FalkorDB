# Two range indexes on the same label created while the first is still populating.
import redis, time, sys
port=int(sys.argv[1]); N=int(sys.argv[2]); delay=float(sys.argv[3])
r=redis.Redis(port=port, socket_timeout=300)
if r.exists("ti"): r.execute_command("GRAPH.DELETE","ti")
r.execute_command("GRAPH.QUERY","ti",f"UNWIND range(0,{N-1}) AS x CREATE (:L {{a:x, b:x}})")
r.execute_command("GRAPH.QUERY","ti","CREATE INDEX FOR (n:L) ON (n.a)")
time.sleep(delay)
r.execute_command("GRAPH.QUERY","ti","CREATE INDEX FOR (n:L) ON (n.b)")
t0=time.time()
while True:
    st=r.execute_command("GRAPH.QUERY","ti","CALL db.indexes() YIELD status RETURN status")[1]
    if st and st[0][0]==b'OPERATIONAL': break
    time.sleep(0.01)
print("operational after", round(time.time()-t0,2))
plan=r.execute_command("GRAPH.EXPLAIN","ti",f"MATCH (n:L) WHERE n.b < {N} RETURN count(n)")
ca=r.execute_command("GRAPH.QUERY","ti",f"MATCH (n:L) WHERE n.a < {N} RETURN count(n)")[1][0][0]
cb=r.execute_command("GRAPH.QUERY","ti",f"MATCH (n:L) WHERE n.b < {N} RETURN count(n)")[1][0][0]
print(port, "N",N,"via index a:",ca,"via index b:",cb, [p.decode().strip() for p in plan])
