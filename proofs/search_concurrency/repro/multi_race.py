import redis, threading, time, sys
port = int(sys.argv[1])
r = redis.Redis(port=port, socket_timeout=60)
r.execute_command("GRAPH.DELETE", "mr") if r.exists("mr") else None
r.execute_command("GRAPH.QUERY", "mr", "CREATE (:Seed)")
res = {}
def slow():
    t=time.time()
    try:
        res['slow'] = r2.execute_command("GRAPH.QUERY", "mr", "UNWIND range(1,4000000) AS x WITH count(x) AS c CREATE (:N {c:c})")[1]
    except Exception as e:
        res['slow'] = 'ERR ' + str(e)
    res['slow_t'] = time.time()-t
r2 = redis.Redis(port=port, socket_timeout=60)
th = threading.Thread(target=slow); th.start()
time.sleep(0.05)
p = r.pipeline(transaction=True)
p.execute_command("GRAPH.QUERY", "mr", "CREATE (:M)")
try:
    out = p.execute(raise_on_error=False)
    res['multi'] = [str(x)[:120] for x in out]
except Exception as e:
    res['multi'] = 'ERR ' + str(e)
th.join()
res['count'] = r.execute_command("GRAPH.QUERY", "mr", "MATCH (n) RETURN labels(n)[0], count(n) ORDER BY labels(n)[0]")[1]
print(port, res)
