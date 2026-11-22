"""Live repro for runtime_core B1/B2 (Budget.hard_cap_unsound_*): `record_cap` used as a hard stop.

Usage: python repro_limit.py PORT   (server with the Rust module, then again with the C module)
Re-run on main 256ab4d3f: Rust [] / [] / [], C [[2]] / [[30]] / [[20]].
"""
import sys, redis
r = redis.Redis(port=int(sys.argv[1]))
q = lambda s: r.execute_command("GRAPH.QUERY", "g", s)
r.delete("g")
q("CREATE (a1:A {id:1}), (b1:B {id:10}), (a2:A {id:2}), (b2:B {id:20}), (a1)-[:R]->(b1), (a2)-[:R]->(b2), (b2)-[:R]->(a2)")
print("B1a", q("MATCH (a:A)-[:R]->(b)-[:R]->(a) RETURN a.id LIMIT 1")[1])
r.delete("g")
q("CREATE (a1:A {id:1}), (b1:B {id:10}), (a2:A {id:2}), (b2:B {id:20}), (c:C {id:30}), (a1)-[:R]->(b1), (a2)-[:R]->(b2), (b2)-[:R]->(c)")
print("B1b", q("MATCH (a:A)-[:R]->(b)-[:R]->(c) RETURN c.id LIMIT 1")[1])
r.delete("g")
q("CREATE (a1:A {id:1}), (b1:B {id:10}), (a2:A {id:2}), (b2:B {id:20}), (a1)-[:R]->(b1), (a2)-[:R]->(b2)")
print("B2", q("MATCH (a:A)-[:R]->(b) WITH b SKIP 0 RETURN b.id SKIP 1 LIMIT 1")[1])
