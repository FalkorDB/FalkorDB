import threading, random, sys
from falkordb import FalkorDB
HOST=sys.argv[1] if len(sys.argv)>1 else '127.0.0.1'
PORT=int(sys.argv[2]) if len(sys.argv)>2 else 6399

db = FalkorDB(host=HOST, port=PORT)
g = db.select_graph('uaf')
try: g.delete()
except: pass
g.query("UNWIND range(1,500) AS i CREATE (a:City {id:i, name:'C'+toString(i)})-[:JOINED {ts:i}]->(b:Member {id:i, name:'M'+toString(i)})")

# Generate many distinct edge types and labels
def worker(tid):
    g2 = db.select_graph('uaf')
    for k in range(5000):
        # Mix of writer queries with OPTIONAL MATCH and CALL{} subqueries (matches customer crash pattern)
        l1=f'L{(tid*7+k)%30}'; l2=f'L{(tid*5+k)%30}'; r=f'R{(tid*3+k)%30}'
        try:
            choice = k % 6
            if choice==0:
                g2.query(f"MATCH (this:City) WHERE this.id={k%500} CALL {{ WITH this OPTIONAL MATCH (this)<-[u0:JOINED]-(u1:Member) DELETE u0 RETURN count(u0) AS d }} RETURN d")
            elif choice==1:
                g2.query(f"MATCH (this:City) WHERE this.id={k%500} OPTIONAL MATCH (this)<-[u0:JOINED]-(u1:Member) RETURN this.id, count(u0)")
            elif choice==2:
                g2.query(f"MATCH (a:City)-[e:JOINED]->(b:Member) WHERE a.id={k%500} RETURN e.ts LIMIT 1")
            elif choice==3:
                # unique query to churn cache
                g2.query(f"MATCH (a:{l1})-[e:{r}]->(b:{l2}) RETURN count(*)")
            elif choice==4:
                g2.query(f"MATCH (a:City) WHERE a.id<{k%500} OPTIONAL MATCH (a)-[e:JOINED]->(m:Member {{id:a.id}}) RETURN count(m) LIMIT 5")
            else:
                g2.query(f"MATCH (a:City {{id:{k%500}}})-[e:JOINED]->(b:Member) CREATE (a)-[:NEW{tid%5} {{k:{k}}}]->(b)")
        except Exception as ex:
            pass

T=24
threads=[threading.Thread(target=worker,args=(i,)) for i in range(T)]
for t in threads: t.start()
for t in threads: t.join()
print("done")
