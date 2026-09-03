"""The canonical benchmark query set and the graph it runs against.

Data only — no I/O, no imports beyond the value types, so this module is safe
to import from a test. Everything that measures (`measure`, `callgrind`,
`compare`, `coverage.sh`) reads the query set from here.

`Q` is `Query`; see `model.Query` for the field meanings. `cg=True` marks a
query as runnable under callgrind, i.e. it needs nothing beyond what CG_SETUP
provides and it does not consume entities faster than the pool supplies them.
That flag used to be a 93-name list in the callgrind module — a second source
of truth about the query set, in a different file, kept honest by an assert
that only checked the names existed. It lives here now, next to the query it
describes, so adding a query decides its own membership.
"""

from falkorbench.model import Query as Q

SETUP = [
    "CREATE INDEX FOR (p:Person) ON (p.id)",
    "UNWIND range(0, 9999) AS i CREATE (:Person {id: i, name: 'p' + toString(i)})",
    "UNWIND range(0, 9999) AS i MATCH (a:Person {id: i}) MATCH (b:Person {id: (i + 1) % 10000}) CREATE (a)-[:KNOWS]->(b)",
    "MATCH (p:Person) SET p.age = p.id % 80, p.score = p.id * 1.5",
    # ---- Doc corpus for coverage features (fulltext / vector / edge index /
    # constraints). Kept on separate labels/types so the Person/KNOWS perf
    # rows above stay index-free and comparable with the C baseline.
    "CREATE INDEX FOR (d:Doc) ON (d.id)",
    # Vector index must come BEFORE the fulltext index: creating fulltext
    # first triggers a pre-existing indexing bug where numeric value 0
    # vanishes from the range index (Doc{id:0} unfindable, SIMILAR ring
    # loses its 0->1 and 99->0 edges).
    "CREATE VECTOR INDEX FOR (d:Doc) ON (d.embedding) OPTIONS {dimension: 4, similarityFunction: 'euclidean', M: 16, efConstruction: 64, efRuntime: 16}",
    "CREATE FULLTEXT INDEX FOR (d:Doc) ON (d.text)",
    "CREATE INDEX FOR ()-[r:SIMILAR]-() ON (r.weight)",
    "CREATE VECTOR INDEX FOR ()-[r:SIMILAR]-() ON (r.vec) OPTIONS {dimension: 4, similarityFunction: 'euclidean', M: 16, efConstruction: 64, efRuntime: 16}",
    "UNWIND range(0, 99) AS i CREATE (:Doc {id: i, text: 'doc word' + toString(i), embedding: vecf32([toFloat(i), toFloat(i % 10), toFloat(i % 7), 1.0])})",
    "UNWIND range(0, 99) AS i MATCH (a:Doc {id: i}) MATCH (b:Doc {id: (i + 1) % 100}) CREATE (a)-[:SIMILAR {weight: i % 10, vec: vecf32([toFloat(i), 1.0, 0.0, 1.0])}]->(b)",
    # Multi-edge pair (same type, same endpoints) on an isolated label so the
    # tensor multi-edge paths (promotion, me matrix, demote) get exercised
    # without touching the Person/Doc perf rows.
    "CREATE (a:MEnd {id: 0}), (b:MEnd {id: 1}) WITH a, b CREATE (a)-[:MULTI {k: 1}]->(b), (a)-[:MULTI {k: 2}]->(b), (a)-[:MULTI {k: 3}]->(b), (b)-[:MULTI {k: 4}]->(a)",
    # Composite-index corpus (separate label so Person stays single-index).
    "CREATE INDEX FOR (c:CIdx) ON (c.a, c.b)",
    "UNWIND range(0, 99) AS i CREATE (:CIdx {a: i % 10, b: i})",
    # Point corpus for distance index scans (separate label, tiny).
    "CREATE INDEX FOR (g:Geo) ON (g.loc)",
    "UNWIND range(0, 99) AS i CREATE (:Geo {id: i, loc: point({latitude: toFloat(i) / 100.0, longitude: toFloat(i) / 100.0})})",
    # Index DDL round-trip: DROP INDEX has no other coverage.
    "CREATE INDEX FOR (z:ZIdx) ON (z.q)",
    "DROP INDEX ON :ZIdx(q)",
    # MEnd id index backs the create+drop constraint pair in SETUP_COMMANDS.
    "CREATE INDEX FOR (m:MEnd) ON (m.id)",
    # Typed-value index corpus: indexer write arms for bool/float/temporal/
    # point/list values on a range-indexed property.
    "CREATE INDEX FOR (d:IDoc) ON (d.v)",
    "CREATE (:IDoc {v: true}), (:IDoc {v: 2.5}), (:IDoc {v: date('2024-03-15')}), (:IDoc {v: point({latitude: 32.1, longitude: 34.8})}), (:IDoc {v: [1, 2, 3]}), (:IDoc {v: ['x', 'y']}), (:IDoc {v: 'plain'})",
    # String-index corpus. Corpus BEFORE index so index creation backfills
    # existing nodes (the Person/Doc indexes only cover index-then-insert).
    "UNWIND range(1, 100) AS i CREATE (:SIdx {s: 'name_' + toString(i)})",
    "CREATE INDEX FOR (n:SIdx) ON (n.s)",
    # Point-index corpus: distance() rewritten to an index range scan.
    "UNWIND range(1, 50) AS i CREATE (:Place {loc: point({latitude: 32.0 + i * 0.001, longitude: 34.0})})",
    "CREATE INDEX FOR (n:Place) ON (n.loc)",
    # Edge fulltext index + corpus (db.idx.fulltext.queryRelationships).
    "CREATE FULLTEXT INDEX FOR ()-[r:REF]-() ON (r.note)",
    "MATCH (a:Person {id: 1}), (b:Person {id: 2}) CREATE (a)-[:REF {note: 'hello world benchmark'}]->(b), (a)-[:REF {note: 'quick brown fox'}]->(b)",
    # Fulltext OPTIONS parsing: language/stopwords/weight/nostem/phonetic.
    "CREATE FULLTEXT INDEX FOR (n:SW2) ON (n.txt) OPTIONS {language: 'english', stopwords: ['the', 'and'], weight: 2.0, nostem: true, phonetic: 'dm:en'}",
    "CREATE (:SW2 {txt: 'the quick and brown fox'})",
    # Temporal-typed property values: attribute-store write arms for
    # Time/DateTime/Date/Duration, plus their RDB encode via DEBUG RELOAD.
    "CREATE (:TVal {t: localtime('12:30:45'), dt: localdatetime('2020-06-15T12:30:45'), d: date('2020-06-15'), du: duration('P1D')})",
    # Entity-to-entity SET copy targets (SET x = y / SET x += y).
    "CREATE (:SCopy {k: 1})-[:SC {w: 1}]->(:SCopy2 {k: 2, extra: 'x'})",
    # Unique RELATIONSHIP constraint corpus: needs a supporting exact-match
    # edge index or the constraint create is rejected. Values stay unique so
    # the constraint goes OPERATIONAL (unlike the SIMILAR one, which fails).
    "CREATE INDEX FOR ()-[r:UREL]-() ON (r.uid)",
    "MATCH (a:Person {id: 1}), (b:Person {id: 2}) CREATE (a)-[:UREL {uid: 1}]->(b), (a)-[:UREL {uid: 2}]->(b)",
]

# Raw redis commands run after SETUP (redis-cli arg lists). {graph} is
# replaced with the graph key by the harness.
UDF_SCRIPT = (
    "function bench_add(a, b) { return a + b; }"
    "function bench_fib(n) { let a = 0, b = 1;"
    " for (let i = 0; i < n; i++) { const t = a + b; a = b; b = t; }"
    " return a; }"
    "function bench_neighbors(n) { return n.getNeighbors({direction: 'outgoing', types: ['KNOWS'], returnType: 'nodes'}).length; }"
    "function bench_inspect(n) { return n.labels.length + Object.keys(n.attributes).length; }"
    "function bench_echo(x) { return x; }"
    "function bench_traverse(n) { return graph.traverse([n], {maxDepth: 3, direction: 'outgoing', types: ['KNOWS'], returnType: 'nodes'}).length; }"
    "function bench_nbr_filter(n) { return n.getNeighbors({direction: 'both', labels: ['Person'], returnType: 'edges'}).length + n.getNeighbors({direction: 'incoming', types: ['KNOWS'], labels: ['Person']}).length; }"
    "function bench_edge_inspect(e) { return [e.id, e.type, e.source, e.target, Object.keys(e.attributes).length]; }"
    "falkor.register('add', bench_add);"
    "falkor.register('fib', bench_fib);"
    "falkor.register('neighbors', bench_neighbors);"
    "falkor.register('inspect', bench_inspect);"
    "falkor.register('echo', bench_echo);"
    "falkor.register('traverse', bench_traverse);"
    "falkor.register('nbrFilter', bench_nbr_filter);"
    "falkor.register('edgeInspect', bench_edge_inspect);"
)
SETUP_COMMANDS = [
    ["GRAPH.CONSTRAINT", "CREATE", "{graph}", "UNIQUE", "NODE", "Doc", "PROPERTIES", "1", "id"],
    ["GRAPH.CONSTRAINT", "CREATE", "{graph}", "MANDATORY", "NODE", "Doc", "PROPERTIES", "1", "id"],
    # Created here, dropped after DEBUG RELOAD below: covers constraint DROP
    # plus constraint serialization through the RDB round-trip.
    ["GRAPH.CONSTRAINT", "CREATE", "{graph}", "UNIQUE", "NODE", "MEnd", "PROPERTIES", "1", "id"],
    ["GRAPH.UDF", "LOAD", "bench", UDF_SCRIPT],
    # Round-trip the whole graph through RDB encode/decode so the
    # serialization paths (matrix/vector/tensor/attribute-store Encode+Decode)
    # are exercised by the coverage run.
    ["DEBUG", "RELOAD"],
    ["GRAPH.CONSTRAINT", "DROP", "{graph}", "UNIQUE", "NODE", "MEnd", "PROPERTIES", "1", "id"],
    # Relationship + composite constraints. UNIQUE SIMILAR(weight) fails async
    # validation (setup weights repeat i % 10) — that failure path is the
    # coverage target and the constraint then stays inactive. MANDATORY
    # SIMILAR(weight) goes operational and checks every SIMILAR write; the
    # 2-property CIdx constraint covers composite-key build.
    ["GRAPH.CONSTRAINT", "CREATE", "{graph}", "UNIQUE", "RELATIONSHIP", "SIMILAR", "PROPERTIES", "1", "weight"],
    ["GRAPH.CONSTRAINT", "CREATE", "{graph}", "MANDATORY", "RELATIONSHIP", "SIMILAR", "PROPERTIES", "1", "weight"],
    ["GRAPH.CONSTRAINT", "CREATE", "{graph}", "UNIQUE", "NODE", "CIdx", "PROPERTIES", "2", "a", "b"],
    ["GRAPH.CONSTRAINT", "CREATE", "{graph}", "UNIQUE", "RELATIONSHIP", "UREL", "PROPERTIES", "1", "uid"],
    # GRAPH.MEMORY: memory_usage_report, Value::heap_size, index sizes.
    ["GRAPH.MEMORY", "USAGE", "{graph}"],
    ["GRAPH.MEMORY", "USAGE", "{graph}", "SAMPLES", "100"],
    # PROFILE executes AND renders per-op timing, covering the profiling
    # decoration paths in the runtime that EXPLAIN/plain execution skip.
    ["GRAPH.PROFILE", "{graph}", "MATCH (p:Person) WHERE p.id > 5 RETURN count(p), max(p.score)"],
    # EXPLAIN renders the plan (and expressions via Display), covering the
    # AST/plan formatting paths that normal execution never touches.
    ["GRAPH.EXPLAIN", "{graph}", "MATCH (p:Person) WHERE p.id > 5 AND p.name STARTS WITH 'n' RETURN p.id + 1, collect(p)[0], CASE WHEN p.id > 2 THEN 1 ELSE 2 END ORDER BY p.id SKIP 1 LIMIT 2"],
    # `exists(p.name)`, not `exists((p)-[:KNOWS]->())`: the engine rejects a
    # traversal pattern as an argument to exists() ("traversal patterns are not
    # allowed as arguments to exists()"), so this command errored and rendered no
    # plan at all. It went unnoticed because redis-cli writes error replies to
    # stderr and the old harness inspected only stdout, making its
    # setup-command error check dead code. A pattern predicate still gets
    # rendered — see the "EXISTS pattern" query, which uses the bare
    # `(p)-[:KNOWS]->()` form in WHERE.
    ["GRAPH.EXPLAIN", "{graph}", "MERGE (p:Person {id: 0}) ON CREATE SET p.c = 1 ON MATCH SET p += {m: 1} WITH p OPTIONAL MATCH (p)-[r:KNOWS*1..2]->(q) DELETE r RETURN p, q, [x IN range(1, 3) WHERE x > 1 | x * 2], exists(p.name)"],
    ["GRAPH.EXPLAIN", "{graph}", "MATCH (p:Person) WHERE p.id % 2 = 0 XOR any(x IN [1, 2] WHERE x = p.id) RETURN p.id ^ 2, -p.id, p.id <= 3, p.id >= 1"],
    ["GRAPH.EXPLAIN", "{graph}", "MATCH ()-[r:KNOWS|SIMILAR]->() RETURN r"],
    ["GRAPH.EXPLAIN", "{graph}", "MERGE (x:Tmp3 {k: 1}) ON CREATE SET x.c = 1 ON MATCH SET x += {m: 2} RETURN x"],
]

# CSV files the harness writes into IMPORT_DIR before starting the server
# (the server is started with IMPORT_FOLDER pointing there).
IMPORT_DIR = "/tmp/falkordb_bench_import"
CSV_FILES = {
    "data.csv": "id,name,score\n" + "".join(f"{i},row{i},{i * 1.5}\n" for i in range(100)),
}

QUERIES = [
    # ---- expressions only (no graph access) -------------------------------
    Q("RETURN 1",            False, "RETURN 1", cg=True),
    Q("arithmetic",          False, "UNWIND range(0, 999) AS i RETURN sum(i * 3 + i % 7 - i / 2)", cg=True),
    Q("float math",          False, "UNWIND range(0, 999) AS i RETURN sum(sqrt(toFloat(i)) + abs(i - 500) + ceil(i / 3.0) + floor(i * 0.7) + round(i * 1.1))", cg=True),
    Q("split+trim+replace",  False, "UNWIND range(0, 99) AS i RETURN count(split(replace(trim('  a,b,c  '), ',', ';'), ';'))", cg=True),
    Q("list comprehension",  False, "RETURN size([x IN range(0, 9999) WHERE x % 2 = 0 | x * 2])", cg=True),
    Q("reduce",              False, "RETURN reduce(acc = 0, x IN range(0, 9999) | acc + x)", cg=True),
    Q("list ops",            False, "UNWIND range(0, 99) AS i RETURN count(head(range(i, i + 10)) + last(range(i, i + 10)) + size(tail(range(i, i + 10))))", cg=True),
    Q("type conversion",     False, "UNWIND range(0, 999) AS i RETURN count(toInteger(toString(i)) + toInteger(toFloat(i)))", cg=True),
    Q("temporal",            False, "RETURN date('2024-01-15'), duration('P1D'), date().year", cg=True),
    Q("spatial",             False, "RETURN point({latitude: 32.0, longitude: 34.8}), distance(point({latitude: 32.0, longitude: 34.8}), point({latitude: 31.0, longitude: 35.0}))"),
    Q("string extras",       False, "RETURN left('hello', 2), right('hello', 2), lTrim('  x'), rTrim('x  ')"),

    # ---- single-clause reads ----------------------------------------------
    Q("label scan + count",  False, "MATCH (p:Person) RETURN count(p)", cg=True),
    Q("index lookup",        False, "MATCH (p:Person {id: 500}) RETURN p.name", cg=True),
    Q("id seek",             False, "MATCH (n) WHERE id(n) = 5 RETURN n", cg=True),
    Q("label + id scan",     False, "MATCH (n:Person) WHERE id(n) < 100 RETURN count(n)"),
    Q("filter scan",         False, "MATCH (p:Person) WHERE p.age > 45 AND p.score < 2000.0 RETURN count(p)", cg=True),
    Q("string predicates",   False, "MATCH (p:Person) WHERE p.name STARTS WITH 'p1' AND p.name CONTAINS '2' AND p.name ENDS WITH '3' RETURN count(p)", cg=True),
    Q("string funcs",        False, "MATCH (p:Person) WHERE p.id < 1000 RETURN count(toUpper(p.name) + toLower(p.name) + substring(p.name, 1, 2) + reverse(p.name)), sum(size(p.name))", cg=True),
    Q("IN list",             False, "MATCH (p:Person) WHERE p.id IN [1, 17, 4242, 9999] RETURN count(p)"),
    Q("CASE",                False, "MATCH (p:Person) RETURN sum(CASE WHEN p.id % 3 = 0 THEN 1 WHEN p.id % 3 = 1 THEN 2 ELSE 3 END)", cg=True),
    Q("coalesce",            False, "MATCH (p:Person) RETURN count(coalesce(p.missing, p.id))", cg=True),
    Q("entity funcs",        False, "MATCH (p:Person {id: 7}) RETURN id(p), labels(p), keys(p), properties(p)"),
    Q("RETURN DISTINCT",     False, "MATCH (p:Person) RETURN DISTINCT p.age", cg=True),
    Q("ORDER BY + LIMIT",    False, "MATCH (p:Person) RETURN p.name ORDER BY p.score DESC LIMIT 10", cg=True),
    Q("SKIP + LIMIT",        False, "MATCH (p:Person) RETURN p.id ORDER BY p.id SKIP 5000 LIMIT 100", cg=True),
    Q("traversal + count",   False, "MATCH (a:Person)-[:KNOWS]->(b) RETURN count(b)", cg=True),
    Q("two-hop",             False, "MATCH (a:Person)-[:KNOWS]->()-[:KNOWS]->(c) RETURN count(c)", cg=True),
    Q("edge + type()",       False, "MATCH (a:Person)-[r:KNOWS]->(b) RETURN count(type(r))", cg=True),
    Q("var-length 1..3",     False, "MATCH (a:Person {id: 0})-[:KNOWS*1..3]->(b) RETURN count(b)", cg=True),
    Q("var-length 1..50",    False, "MATCH (a:Person {id: 0})-[:KNOWS*1..50]->(b) RETURN count(b)"),
    Q("path + length",       False, "MATCH path = (a:Person {id: 5})-[:KNOWS*1..3]->(b) RETURN sum(length(path))", cg=True),
    Q("path funcs",          False, "MATCH p = (a:Person {id: 5})-[:KNOWS]->(b) RETURN nodes(p), relationships(p)", cg=True),
    Q("OPTIONAL MATCH",      False, "MATCH (p:Person {id: 42}) OPTIONAL MATCH (p)-[:MISSING]->(q) RETURN p.id, q"),
    Q("expand into",         False, "MATCH (a:Person {id: 0}), (b:Person {id: 1}) MATCH (a)-[:KNOWS]->(b) RETURN count(*)", cg=True),

    # ---- aggregation functions, one per query -----------------------------
    Q("agg count",           False, "MATCH (p:Person) RETURN count(p.age)", cg=True),
    Q("agg sum",             False, "MATCH (p:Person) RETURN sum(p.score)", cg=True),
    Q("agg min",             False, "MATCH (p:Person) RETURN min(p.score)", cg=True),
    Q("agg max",             False, "MATCH (p:Person) RETURN max(p.score)", cg=True),
    Q("agg avg",             False, "MATCH (p:Person) RETURN avg(p.score)", cg=True),
    Q("percentileDisc",      False, "MATCH (p:Person) RETURN percentileDisc(p.score, 0.5)", cg=True),
    Q("percentileCont",      False, "MATCH (p:Person) RETURN percentileCont(p.score, 0.5)"),
    Q("stDev",               False, "MATCH (p:Person) RETURN stDev(p.score)", cg=True),
    Q("stDevP",              False, "MATCH (p:Person) RETURN stDevP(p.score)"),
    Q("collect",             False, "MATCH (p:Person) WHERE p.id < 1000 RETURN size(collect(p.name))", cg=True),
    Q("count distinct",      False, "MATCH (p:Person) RETURN count(DISTINCT p.id % 100)", cg=True),

    # ---- single write clauses ---------------------------------------------
    Q("MERGE existing",      True,  "MERGE (p:Person {id: 500}) RETURN p.id", cg=True),
    Q("SET property",        True,  "MATCH (p:Person {id: 500}) SET p.age = 30 RETURN p.age", cg=True),
    Q("REMOVE",              True,  "MATCH (p:Person {id: 3}) SET p.tmp = 1 REMOVE p.tmp", cg=True),
    Q("create node",         True,  "CREATE (:Tmp {x: 1})", cg=True),
    Q("delete node",         True,  "MATCH (t:Tmp) WITH t LIMIT 1 DELETE t", cg=True),
    Q("create edge",         True,  "MATCH (a:Person {id: 0}), (b:Person {id: 1}) CREATE (a)-[:TKNOWS]->(b)"),
    Q("delete edge",         True,  "MATCH (:Person {id: 0})-[r:TKNOWS]->() WITH r LIMIT 1 DELETE r"),

    # ---- mixed clauses -----------------------------------------------------
    Q("aggregates",          False, "MATCH (p:Person) RETURN min(p.id), max(p.id), avg(p.score), sum(p.age), count(*)", cg=True),
    Q("WITH pipeline",       False, "MATCH (p:Person) WITH p.id % 100 AS g, count(*) AS c WHERE c > 50 RETURN g, c ORDER BY c DESC LIMIT 5", cg=True),
    Q("UNION",               False, "MATCH (p:Person {id: 1}) RETURN p.id AS x UNION MATCH (p:Person {id: 2}) RETURN p.id AS x", cg=True),
    Q("EXISTS pattern",      False, "MATCH (p:Person) WHERE p.id < 100 AND (p)-[:KNOWS]->() RETURN count(p)", cg=True),
    Q("pattern OR filter",   False, "MATCH (p:Person) WHERE (p)-[:KNOWS]->(:Person {id: 1}) OR p.id = 0 RETURN count(p)", cg=True),
    Q("hash join",           False, "MATCH (a:Person), (b:Person) WHERE a.id = 9999 - b.id RETURN count(*)", cg=True),

    # ---- cartesian product, by right-branch count --------------------------
    # Nothing above reaches the operator with more than one row a side: every
    # other multi-component `MATCH (a), (b)` here is an id-seek pair, or the
    # `hash join` row, which is the ValueHashJoin rewrite of exactly this shape.
    # A regression in the product itself was therefore invisible to this set.
    #
    # Cost is per output row and the row count is the *product* of the branches,
    # so these are sized by product (~1-2M rows) rather than by node count, and
    # each branch takes a different label so every output column has a distinct
    # owner. `reps` is capped for the same reason the sized writes cap theirs: at
    # the default 1000 a 10 ms query would outweigh the rest of the set.
    # Person x Doc = 10,000 x 100.
    Q("cartesian 1 branch", False, "MATCH (a:Person), (b:Doc) RETURN count(*)", 50),
    # Doc x CIdx x Geo = 100^3.
    Q("cartesian 2 branches", False, "MATCH (a:Doc), (b:CIdx), (c:Geo) RETURN count(*)", 50),
    # Doc x CIdx x Geo x MEnd = 100^3 x 2 — the 2-row innermost branch wraps the
    # odometer's fastest digit on every other row.
    Q("cartesian 3 branches", False, "MATCH (a:Doc), (b:CIdx), (c:Geo), (d:MEnd) RETURN count(*)", 50),
    Q("shortestPath",        False, "MATCH (a:Person {id: 0}), (b:Person {id: 3}) RETURN length(shortestPath((a)-[:KNOWS*..5]->(b)))", cg=True),
    Q("allShortestPaths",    False, "MATCH (a:Person {id: 0}), (b:Person {id: 3}) WITH a, b MATCH p = allShortestPaths((a)-[:KNOWS*..5]->(b)) RETURN count(p)", cg=True),
    Q("CALL procedure",      False, "CALL db.labels() YIELD label RETURN label", cg=True),
    Q("CREATE + DELETE",     True,  "CREATE (t:Tmp {x: 1}) WITH t DELETE t", cg=True),
    Q("FOREACH",             True,  "MATCH (p:Person {id: 3}) FOREACH (x IN [1, 2, 3] | SET p.age = x)", cg=True),
    Q("algo.pageRank",       False, "CALL algo.pageRank('Person', 'KNOWS') YIELD node, score RETURN count(node)", cg=True),
    Q("algo.BFS",            False, "MATCH (a:Person {id: 0}) CALL algo.BFS(a, 3, 'KNOWS') YIELD nodes RETURN size(nodes)", cg=True),
    Q("algo.BFS unbounded",  False, "MATCH (a:Person {id: 0}) CALL algo.BFS(a, -1, 'KNOWS') YIELD nodes RETURN size(nodes)"),
    Q("algo.WCC",            False, "CALL algo.WCC(null) YIELD node, componentId RETURN count(DISTINCT componentId)", cg=True),

    # ---- coverage features (indexes, procedures, algos, temporal, CSV, UDF)
    # These run against the small Doc corpus / expressions, so they are cheap;
    # they exist mainly to keep the query set exercising the whole crate.
    Q("fulltext query",      False, "CALL db.idx.fulltext.queryNodes('Doc', 'word1*') YIELD node RETURN count(node)"),
    Q("vector query",        False, "CALL db.idx.vector.queryNodes('Doc', 'embedding', 5, vecf32([5.0, 5.0, 5.0, 1.0])) YIELD node, score RETURN count(node)"),
    Q("vec distance",        False, "RETURN vec.euclideanDistance(vecf32([1.0, 2.0, 3.0, 4.0]), vecf32([4.0, 3.0, 2.0, 1.0])), vec.cosineDistance(vecf32([1.0, 0.0, 0.0, 1.0]), vecf32([0.0, 1.0, 0.0, 1.0]))"),
    Q("edge index scan",     False, "MATCH ()-[r:SIMILAR]->() WHERE r.weight > 7 RETURN count(r)"),
    Q("LOAD CSV",            False, "LOAD CSV FROM 'file://data.csv' AS row RETURN count(row)"),
    Q("LOAD CSV headers",    False, "LOAD CSV WITH HEADERS FROM 'file://data.csv' AS row RETURN count(row.name), sum(toFloat(row.score))"),
    Q("UDF call",            False, "RETURN bench.add(1, 2), bench.fib(20)"),
    Q("db.indexes",          False, "CALL db.indexes() YIELD label RETURN count(label)", cg=True),
    Q("db.propertyKeys",     False, "CALL db.propertyKeys() YIELD propertyKey RETURN count(propertyKey)", cg=True),
    Q("db.relationshipTypes", False, "CALL db.relationshipTypes() YIELD relationshipType RETURN count(relationshipType)"),
    Q("db.meta.stats",       False, "CALL db.meta.stats() YIELD nodeCount, relCount RETURN nodeCount, relCount"),
    Q("db.constraints",      False, "CALL db.constraints() YIELD type, label RETURN count(*)"),
    Q("dbms.procedures",     False, "CALL dbms.procedures() YIELD name RETURN count(name)"),
    Q("dbms.functions",      False, "CALL dbms.functions() YIELD name RETURN count(name)"),
    Q("algo.betweenness",    False, "CALL algo.betweenness({nodeLabels: ['Doc'], relationshipTypes: ['SIMILAR']}) YIELD node, score RETURN count(node)", 200),
    Q("algo.labelPropagation", False, "CALL algo.labelPropagation({nodeLabels: ['Doc'], relationshipTypes: ['SIMILAR']}) YIELD node, communityId RETURN count(DISTINCT communityId)", 200),
    Q("algo.MSF",            False, "CALL algo.MSF({nodeLabels: ['Doc'], relationshipTypes: ['SIMILAR'], weightAttribute: 'weight'}) YIELD edges RETURN size(edges)", 200),
    Q("algo.HarmonicCentrality", False, "CALL algo.HarmonicCentrality({nodeLabels: ['Doc'], relationshipTypes: ['SIMILAR']}) YIELD node, score RETURN count(node)", 200),
    Q("algo.SPpaths",        False, "MATCH (a:Person {id: 0}), (b:Person {id: 3}) CALL algo.SPpaths({sourceNode: a, targetNode: b, relTypes: ['KNOWS'], maxLen: 5}) YIELD path RETURN count(path)"),
    Q("algo.SSpaths",        False, "MATCH (a:Person {id: 0}) CALL algo.SSpaths({sourceNode: a, relTypes: ['KNOWS'], maxLen: 2}) YIELD path RETURN count(path)"),
    Q("algo.maxFlow",        False, "MATCH (a:Doc {id: 0}), (b:Doc {id: 50}) CALL algo.maxFlow({sourceNodes: [a], targetNodes: [b], relationshipTypes: ['SIMILAR'], capacityProperty: 'weight', defaultCapacity: 1.0}) YIELD nodes, edges, edgeFlows, maxFlow RETURN size(nodes), size(edges), size(edgeFlows), maxFlow", 200),
    Q("temporal constructors", False, "RETURN localdatetime('2024-01-15T10:30:00'), localtime('10:30:00'), date({year: 2024, month: 2, day: 29}), duration({days: 2, hours: 3})"),
    Q("temporal components", False, "WITH localdatetime('2024-03-15T10:30:45') AS dt RETURN dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second, dt.dayOfWeek, dt.quarter", cg=True),
    Q("temporal arithmetic", False, "RETURN date('2024-01-15') + duration('P1M2D'), duration('PT1H30M') + duration('PT45M')", cg=True),
    Q("temporal compare",    False, "RETURN date('2024-01-15') < date('2024-02-01'), localtime('10:00:00') < localtime('11:00:00')"),
    Q("NOT pattern",         False, "MATCH (p:Person) WHERE p.id < 100 AND NOT (p)-[:MISSING]->() RETURN count(p)", cg=True),
    Q("map ops",             False, "WITH {a: 1, b: 'x', c: [1, 2]} AS m RETURN m.a, keys(m), m['b'], m.c[1]", cg=True),
    Q("list slicing",        False, "WITH range(0, 99) AS l RETURN l[10..20], l[-5..], l[..5], l[3]", cg=True),
    Q("null handling",       False, "RETURN null + 1, null = null, 3 IS NULL, coalesce(null, null, 3), toInteger(null)", cg=True),
    Q("boolean ops",         False, "UNWIND range(0, 99) AS i RETURN count(CASE WHEN (i % 2 = 0 XOR i % 3 = 0) AND NOT i % 5 = 0 OR i > 90 THEN 1 END)", cg=True),
    Q("order by mixed types", False, "UNWIND [1, 'a', 2.5, true, null, [1], {a: 1}] AS x RETURN x ORDER BY x"),
    Q("SET += map",          True,  "MATCH (p:Person {id: 501}) SET p += {hobby: 'x', level: 2} RETURN p.level", cg=True),
    Q("SET = map",           True,  "MERGE (c:Cfg {cid: 1}) SET c = {cid: 1, a: 1, b: 2} RETURN c.a"),
    Q("SET/REMOVE label",    True,  "MATCH (p:Person {id: 502}) SET p:VIP REMOVE p:VIP"),
    Q("MERGE on create/match", True, "MERGE (p:Person {id: 600}) ON CREATE SET p.created = 1 ON MATCH SET p.matched = 1 RETURN p.id"),
    Q("DETACH DELETE",       True,  "CREATE (a:Tmp3)-[:TREL]->(b:Tmp3) WITH a, b DETACH DELETE a, b", 500),

    # ---- round 2: expression / function coverage ---------------------------
    Q("trig",                False, "UNWIND range(0, 99) AS i RETURN sum(sin(i) + cos(i) + tan(i) + atan2(toFloat(i), 2.0) + haversin(i)), degrees(pi()), radians(180)", cg=True),
    Q("inverse trig",        False, "RETURN acos(0.5), asin(0.5), atan(1.0), cot(1.0)"),
    Q("math extras",         False, "UNWIND range(1, 99) AS i RETURN sum(exp(i % 5) + log(i) + log10(i) + pow(i, 2) + sign(i - 50)), e(), rand() < 2, size(randomUUID())", cg=True),
    Q("power + float mod",   False, "UNWIND range(1, 99) AS i RETURN sum(i ^ 2 + 7.5 % 2.5)"),
    Q("conversion extras",   False, "RETURN toBoolean('true'), toBooleanOrNull('x'), toIntegerOrNull('a'), toFloatOrNull('b'), toStringOrNull(1), toIntegerList(['1', '2']), toFloatList(['1.5']), toStringList([1, 2]), toBooleanList(['true'])"),
    Q("toJSON + isEmpty",    False, "RETURN toJSON({a: 1, b: [1, 2]}), isEmpty([]), isEmpty(''), isEmpty({a: 1})"),
    Q("regex",               False, "MATCH (p:Person) WHERE p.name =~ 'p1.2' RETURN count(p)", cg=True),
    Q("string namespace",    False, "RETURN string.join(['a', 'b', 'c'], '-'), string.matchRegEx('aBc', '[a-z]'), string.replaceRegEx('a1b2', '[0-9]', '#')", cg=True),
    Q("list namespace",      False, "RETURN list.dedup([1, 1, 2]), list.sort([3, 1, 2]), list.insert([1, 3], 1, 2), list.remove([1, 2, 3], 1), list.insertListElements([1, 4], [2, 3], 1)", cg=True),
    Q("list concat + compare", False, "RETURN [1, 2] + [3], [1, 2] < [1, 3], {a: 1} = {a: 1}"),
    Q("quantifiers",         False, "RETURN all(x IN [1, 2] WHERE x > 0), any(x IN [1, 2] WHERE x > 1), none(x IN [1, 2] WHERE x > 5), single(x IN [1, 2] WHERE x = 1)", cg=True),
    Q("CASE value form",     False, "MATCH (p:Person) RETURN sum(CASE p.id % 3 WHEN 0 THEN 1 WHEN 1 THEN 2 ELSE 3 END)"),
    Q("pattern comprehension", False, "MATCH (a:Person {id: 0}) RETURN [(a)-[:KNOWS]->(b) | b.id]", cg=True),
    Q("point accessors",     False, "WITH point({latitude: 32.0, longitude: 34.8}) AS pt RETURN pt.latitude, pt.longitude"),
    Q("txn temporals",       False, "RETURN timestamp() > 0, date.transaction(), localdatetime.transaction(), localtime.transaction()"),
    Q("duration accessors",  False, "WITH duration({days: 400, hours: 25, minutes: 61}) AS d RETURN d.years, d.months, d.days, d.hours, d.minutes, d.seconds"),
    Q("week/ordinal dates",  False, "RETURN date({year: 2024, week: 10, dayOfWeek: 3}), date({year: 2024, ordinalDay: 60})"),

    # ---- round 2: clause / plan coverage -----------------------------------
    Q("CALL subquery",       False, "MATCH (p:Person) WHERE p.id < 100 CALL { WITH p MATCH (p)-[:KNOWS]->(q) RETURN q.id AS qid } RETURN count(qid)", cg=True),
    Q("UNION ALL",           False, "MATCH (p:Person {id: 1}) RETURN p.id AS x UNION ALL MATCH (p:Person {id: 1}) RETURN p.id AS x", cg=True),
    Q("undirected",          False, "MATCH (a:Person {id: 5})-[:KNOWS]-(b) RETURN count(b)", cg=True),
    Q("multi-type",          False, "MATCH (a:Person {id: 5})-[r:KNOWS|MISSING]->(b) RETURN count(r)"),
    Q("entity extras",       False, "MATCH (a:Person {id: 3})-[r:KNOWS]->(b) RETURN startNode(r).id, endNode(r).id, exists(a.name), hasLabels(a, ['Person']), typeOf(a.id), indegree(a), outdegree(a)"),
    Q("expand into WITH",    False, "MATCH (a:Person {id: 0}), (b:Person {id: 1}) WITH a, b MATCH (a)-[r:KNOWS]->(b) RETURN count(r)"),
    Q("self-loop match",     False, "MATCH (a:Person {id: 0})-[:KNOWS]->(a) RETURN count(a)"),
    Q("return entities",     False, "MATCH (p:Person {id: 9})-[r:KNOWS]->(q) RETURN p, r, q"),
    Q("return path",         False, "MATCH path = (a:Person {id: 9})-[:KNOWS]->(b) RETURN path"),
    Q("UDF node API",        False, "MATCH (p:Person {id: 7}) RETURN bench.neighbors(p), bench.inspect(p)"),
    Q("vector edge query",   False, "CALL db.idx.vector.queryRelationships('SIMILAR', 'vec', 5, vecf32([5.0, 1.0, 0.0, 1.0])) YIELD relationship, score RETURN count(relationship)"),

    # ---- round 2: edge writes ----------------------------------------------
    Q("SET edge prop",       True,  "MATCH (:Person {id: 0})-[r:KNOWS]->() SET r.w = 1 RETURN r.w"),
    Q("edge += map",         True,  "MATCH (:Person {id: 0})-[r:KNOWS]->() SET r += {w2: 2, w3: 3} RETURN r.w2"),
    Q("REMOVE edge prop",    True,  "MATCH (:Person {id: 0})-[r:KNOWS]->() SET r.tmp = 1 REMOVE r.tmp"),
    Q("MERGE edge",          True,  "MATCH (a:Person {id: 0}), (b:Person {id: 1}) MERGE (a)-[r:KNOWS]->(b) RETURN count(r)"),

    # ---- round 3: multi-edges, UDF type round-trips, deletes, comparisons --
    Q("multi-edge read",     False, "MATCH (:MEnd {id: 0})-[r:MULTI]->(:MEnd {id: 1}) RETURN count(r), collect(r.k)"),
    # Aggregate over an *edge* property. Guards the aggregate operator's
    # vectorized path: `sum(r.prop)` classifies as AggInputKind::Property, whose
    # bulk materializer is node-only, so a regression there silently drops the
    # whole batch to the per-row path (an owned Row per row). Uses the existing
    # MULTI corpus so no other row's numbers move.
    Q("multi-edge agg",      False, "MATCH ()-[r:MULTI]->() RETURN sum(r.k)"),
    Q("multi-edge undirected", False, "MATCH (a:MEnd {id: 0})-[r:MULTI]-(b) RETURN count(r)"),
    # Return whole *relationship entities*, not scalars off them, so the compact reply
    # path serializes a relationship's property map (`reply.rs`, Value::Relationship)
    # rather than a single value. `entities` also returns `r`; this row is distinct in
    # returning property-bearing :SIMILAR edges, so the map it serializes is populated
    # — :KNOWS has no edge properties, and returning those would frame an empty map and
    # measure nothing. Every other edge query here returns `r.prop` or an aggregate.
    Q("rel entity return",   False, "MATCH ()-[r:SIMILAR]->() RETURN r"),
    Q("multi-edge demote/promote", True, "MATCH (a:MEnd {id: 0})-[r:MULTI]->(b:MEnd {id: 1}) WHERE r.k > 1 DELETE r WITH DISTINCT a, b CREATE (a)-[:MULTI {k: 2}]->(b), (a)-[:MULTI {k: 3}]->(b)", 500),
    Q("UDF echo scalars",    False, "UNWIND range(0, 99) AS i RETURN count(bench.echo(i) + bench.echo(1.5)), bench.echo(true), bench.echo(null), bench.echo('s')"),
    Q("UDF echo bigint",     False, "RETURN bench.echo(9007199254740993)"),
    Q("UDF echo list+map",   False, "UNWIND range(0, 99) AS i RETURN count(bench.echo([1, 'a', null, 2.5, [i]])[4][0] + bench.echo({a: i, b: {c: 's'}}).a)"),
    Q("UDF echo entities",   False, "MATCH (p:Person {id: 5})-[r:KNOWS]->(q) RETURN bench.echo(p).id, id(bench.echo(r)), size(bench.echo([p, r]))"),
    Q("UDF echo path",       False, "MATCH pth = (p:Person {id: 5})-[:KNOWS]->() RETURN size(nodes(bench.echo(pth)))"),
    Q("UDF echo point+date+vec", False, "RETURN bench.echo(point({latitude: 1.0, longitude: 2.0})).latitude, bench.echo(date('2024-01-15')), bench.echo(vecf32([1.0, 2.0]))"),
    Q("delete path",         True,  "CREATE (a:Tmp3)-[:TREL]->(:Tmp3) WITH a MATCH pth = (a)-[:TREL]->() DELETE pth", 500),
    Q("delete optional null", True, "OPTIONAL MATCH (x:NoSuchLabel) DELETE x RETURN count(x)"),
    Q("equality matrix",     False, "UNWIND [[1, 1.0], ['a', 'a'], [[1, 2], [1, 2]], [{a: 1}, {a: 1}], [true, false], [null, 1]] AS pr RETURN count(pr[0] = pr[1]), count(pr[0] <> pr[1]), count(pr[0] < pr[1])"),
    Q("distinct mixed",      False, "UNWIND [1, 1.0, '1', 1, true, [1], [1], {a: 1}, {a: 1}] AS x RETURN count(DISTINCT x)"),
    Q("edge index eq",       False, "MATCH ()-[r:SIMILAR]->() WHERE r.weight = 5 RETURN count(r)"),
    Q("edge index range",    False, "MATCH ()-[r:SIMILAR]->() WHERE r.weight >= 3 AND r.weight < 7 RETURN count(r)"),

    # ---- round 4: temporal maps, duration arithmetic, map projection,
    # pending-only deletes, CASE forms ---------------------------------------
    Q("date week format",    False, "RETURN date('2024-W05-3'), date('2024-W01-1')"),
    Q("time from map",       False, "RETURN localtime({hour: 12, minute: 30, second: 45}), localtime('12:30:45'), localdatetime({year: 2024, month: 3, day: 10, hour: 6, minute: 30, second: 15})"),
    Q("duration arithmetic", False, "RETURN duration({hours: 1, minutes: 30}) + duration({minutes: 45}), duration({hours: 2}) - duration({minutes: 30}), date('2024-01-01') + duration({days: 5}), date('2024-02-01') - duration({days: 1})"),
    Q("scalar arithmetic",   False, "UNWIND range(0, 99) AS i RETURN count('a' + toString(i)), count([1, i] + [3]), count([i] + 2), count(i + 2.5), count(i - 2), count(5.5 - i), count(i * 2.5), count(i / 2.5), count(i % 7), count(2 ^ 2)", cg=True),
    Q("map projection",      False, "MATCH (p:Person {id: 7}) RETURN p{.id, .name, extra: 1}", cg=True),
    Q("delete pending node", True,  "CREATE (n:Tmp4 {x: 1}) WITH n DELETE n", 500),
    Q("delete pending edge", True,  "CREATE (a:Tmp4)-[r:TREL]->(b:Tmp4) WITH r, a, b DELETE r, a, b", 500),
    Q("CASE forms",          False, "UNWIND range(0, 99) AS i RETURN count(CASE i % 3 WHEN 0 THEN 'zero' WHEN 1 THEN 'one' ELSE 'two' END), count(CASE WHEN i > 50 THEN i ELSE -i END)", cg=True),

    # ---- round 5: JS traversal API, delete-returning-snapshot, time
    # components, row-context eval, lexer variety ----------------------------
    Q("UDF traverse",        False, "MATCH (p:Person {id: 0}) RETURN bench.traverse(p), bench.nbrFilter(p)"),
    Q("UDF edge inspect",    False, "MATCH (:Person {id: 5})-[r:KNOWS]->() RETURN bench.edgeInspect(r)"),
    Q("delete returning",    True,  "CREATE (n:Tmp4 {a: 1, b: 'x'})-[r:TREL {w: 2}]->(m:Tmp4) WITH n, r, m DELETE n, r, m RETURN n.a, r.w", 500),
    Q("time components",     False, "WITH localtime('10:30:45') AS t, localdatetime('2024-06-15T10:30:45') AS dt RETURN t.hour, t.minute, t.second, dt.quarter, dt.dayOfWeek"),
    Q("range step",          False, "UNWIND range(1, 20, 3) AS x RETURN count(x), max(x)"),
    Q("CASE batch",          False, "MATCH (p:Person) RETURN count(CASE WHEN p.id > 5000 THEN 'high' ELSE 'low' END)"),
    Q("map projection star", False, "MATCH (p:Person {id: 7}) RETURN p{.*, plus: p.id + 1}"),
    Q("shortestPath row ctx", False, "MATCH (a:Person {id: 0}), (b:Person {id: 3}) WITH a, b ORDER BY a.id RETURN length(shortestPath((a)-[:KNOWS*..5]->(b)))"),
    Q("lexer variety",       False, "WITH 0x1F AS h, 1e3 AS e, .5 AS hf RETURN h + e + hf, 'esc\\'q\\n\\t', \"dq\\\"s\", true AND NOT false", cg=True),

    # ---- round 6: typed degrees, pending-state degrees, constraint-checked
    # writes, value hash/eq for temporal+point, parser corners, untyped scans,
    # WITH forms, algo option variants, composite index, CSV-driven write ----
    Q("degree by type",      False, "MATCH (a:Person {id: 0}) RETURN indegree(a, 'KNOWS'), outdegree(a, 'KNOWS')", cg=True),
    Q("pending degree",      True,  "CREATE (a:Tmp5)-[:T1]->(b:Tmp5) WITH a, b, outdegree(a) AS o, indegree(b) AS i DETACH DELETE a, b RETURN o, i", 500),
    Q("MERGE constrained SET", True, "MERGE (d:Doc {id: 42}) ON MATCH SET d.seen = 1 RETURN d.id"),
    Q("quarter date",        False, "RETURN date({year: 2024, quarter: 2, dayOfQuarter: 15})"),
    Q("collect DISTINCT",    False, "MATCH (p:Person) RETURN size(collect(DISTINCT p.age))"),
    Q("distinct temporal+point", False, "UNWIND [date('2024-01-01'), date('2024-01-01'), localtime('10:00:00'), duration('P1D'), point({latitude: 1.0, longitude: 2.0}), point({latitude: 1.0, longitude: 2.0})] AS x RETURN count(DISTINCT x)"),
    Q("negation",            False, "UNWIND range(0, 99) AS i RETURN sum(-i), sum(-toFloat(i))", cg=True),
    Q("comments",            False, "/* block */ MATCH (p:Person) // line\nRETURN count(p)", cg=True),
    Q("unicode escapes",     False, "RETURN '\\u0041\\u00e9', 'back\\\\slash'"),
    Q("CYPHER directive",    False, "CYPHER x=1 RETURN $x + 1"),
    Q("pattern predicate props", False, "MATCH (a:Person) WHERE (a)-[:KNOWS]->({id: 5}) RETURN count(a)"),
    Q("OR mixed filter",     False, "MATCH (p:Person) WHERE p.id = 5 OR p.name = 'p6' RETURN count(p)", cg=True),
    Q("all-node scan",       False, "MATCH (n) RETURN count(n)", cg=True),
    Q("untyped edge scan",   False, "MATCH ()-[r]->() RETURN count(r)"),
    Q("untyped var-length",  False, "MATCH (a:Person {id: 0})-[*1..2]->(b) RETURN count(b)"),
    Q("multi-label write",   True,  "CREATE (n:LA:LB {x: 1}) WITH n MATCH (m:LA:LB) DELETE m RETURN count(m)", 500),
    Q("WITH star",           False, "MATCH (p:Person {id: 1}) WITH * RETURN p.id", cg=True),
    Q("WITH order limit",    False, "MATCH (p:Person) WITH p ORDER BY p.score DESC LIMIT 5 RETURN p.name"),
    Q("MERGE bound pattern", True,  "MERGE (a:Person {id: 0})-[:KNOWS]->(b:Person {id: 1}) RETURN b.id"),
    Q("betweenness sampled", False, "CALL algo.betweenness({nodeLabels: ['Doc'], relationshipTypes: ['SIMILAR'], samplingSize: 20, samplingSeed: 42}) YIELD node, score RETURN count(node)", 200),
    Q("labelPropagation iters", False, "CALL algo.labelPropagation({nodeLabels: ['Doc'], relationshipTypes: ['SIMILAR'], maxIterations: 5}) YIELD node, communityId RETURN count(node)", 200),
    Q("MSF maximize",        False, "CALL algo.MSF({nodeLabels: ['Doc'], relationshipTypes: ['SIMILAR'], weightAttribute: 'weight', objective: 'maximize'}) YIELD edges RETURN size(edges)", 200),
    Q("SPpaths cost",        False, "MATCH (a:Person {id: 0}), (b:Person {id: 3}) CALL algo.SPpaths({sourceNode: a, targetNode: b, relTypes: ['KNOWS'], maxLen: 5, weightProp: 'w', costProp: 'w', maxCost: 100}) YIELD path RETURN count(path)"),
    Q("composite index eq",  False, "MATCH (c:CIdx {a: 5, b: 55}) RETURN count(c)"),
    Q("composite index range", False, "MATCH (c:CIdx) WHERE c.a = 5 AND c.b > 10 RETURN count(c)"),
    Q("LOAD CSV set",        True,  "LOAD CSV WITH HEADERS FROM 'file://data.csv' AS row WITH row LIMIT 1 MATCH (p:Person {id: 0}) SET p.loaded = row.name RETURN p.loaded"),
    Q("fulltext score",      False, "CALL db.idx.fulltext.queryNodes('Doc', 'word1*') YIELD node, score RETURN count(node), max(score)"),

    # ---- round 7: uncovered-range targets (toJSON entities, chained
    # arithmetic reduce arms, relationship-matrix builder, typed index) -----
    Q("toJSON node",         False, "MATCH (p:Person {id: 5}) RETURN toJSON(p)"),
    Q("toJSON rel path",     False, "MATCH pth = (p:Person {id: 5})-[r:KNOWS]->() RETURN toJSON(r), toJSON(pth)"),
    Q("toJSON scalars",      False, "RETURN toJSON(vecf32([1.5, 2.5])), toJSON(point({latitude: 1.0, longitude: 2.0})), toJSON(date('2024-01-01'))"),
    Q("chained arithmetic",  False, "UNWIND range(1, 99) AS i RETURN sum(100 - i - 1), sum(2 * i * 3), sum(100 / i / 2), sum(100 % i % 7)", cg=True),
    Q("multi-type traverse", False, "MATCH (a:Person)-[:KNOWS|SIMILAR]->(b:Person) RETURN count(b)"),
    Q("multi-label endpoints", False, "MATCH (a:Person:Doc)-[:KNOWS]->(b:Person:Doc) RETURN count(b)"),
    Q("MSF multi-edge",      False, "CALL algo.MSF({nodeLabels: ['MEnd'], relationshipTypes: ['MULTI'], weightAttribute: 'k'}) YIELD edges RETURN size(edges)", 200),
    Q("typed index read",    False, "MATCH (d:IDoc) WHERE d.v > 1 RETURN count(d)"),

    # ---- round 8: correlated inner operators (Apply set_argument_batch
    # arms in batch.rs: fulltext/vector/csv scans, hash join, paths) --------
    Q("correlated fulltext", False, "MATCH (p:Person) WHERE p.id < 3 CALL { WITH p CALL db.idx.fulltext.queryNodes('Doc', 'word1*') YIELD node RETURN count(node) AS c } RETURN sum(c)"),
    Q("correlated vector",   False, "MATCH (p:Person) WHERE p.id < 3 CALL { WITH p CALL db.idx.vector.queryNodes('Doc', 'embedding', 3, vecf32([5.0, 5.0, 5.0, 1.0])) YIELD node RETURN count(node) AS c } RETURN sum(c)"),
    Q("correlated edge vector", False, "MATCH (p:Person) WHERE p.id < 3 CALL { WITH p CALL db.idx.vector.queryRelationships('SIMILAR', 'vec', 3, vecf32([5.0, 1.0, 0.0, 1.0])) YIELD relationship RETURN count(relationship) AS c } RETURN sum(c)"),
    Q("correlated load csv", False, "MATCH (p:Person {id: 0}) CALL { WITH p LOAD CSV WITH HEADERS FROM 'file://data.csv' AS row RETURN count(row) AS n } RETURN n"),
    Q("correlated hash join", False, "MATCH (m:MEnd {id: 0}) CALL { WITH m MATCH (a:CIdx), (b:Doc) WHERE a.b = b.id RETURN count(*) AS c } RETURN c"),
    Q("optional allShortest", False, "MATCH (a:Person {id: 0}), (b:Person {id: 3}) OPTIONAL MATCH p = allShortestPaths((a)-[:KNOWS*..5]->(b)) RETURN count(p)"),
    Q("optional varlen",     False, "MATCH (a:Person {id: 0}) OPTIONAL MATCH (a)-[:KNOWS*2..3]->(c) RETURN count(c)"),
    Q("optional path build", False, "MATCH (a:Person {id: 0}) OPTIONAL MATCH p2 = (a)-[:KNOWS]->(x) RETURN count(p2)"),
    Q("id eq join",          False, "MATCH (p:Person {id: 5}) MATCH (q:Person) WHERE id(q) = id(p) RETURN count(q)"),

    # ---- round 9: row-context eval via CALL{} projections, temporal parse
    # variants, string/point index scans, relationship constraints, indexed
    # create+delete, algo config arms -----------------------------------------
    Q("temporal now fns",    False, "RETURN localtime(), localdatetime(), date.transaction(), localtime.transaction(), localdatetime.transaction()"),
    Q("temporal maps 2",     False, "RETURN localtime({hour: 12, minute: 30, second: 5}), date({year: 2020, month: 2, day: 29}), date({year: 2020, quarter: 2, dayOfQuarter: 60}), duration({days: 1, hours: 2, minutes: 3, seconds: 4.5})"),
    Q("temporal parse 2",    False, "RETURN date('2015-W30-2'), date('2015202'), localtime('12:30:45.123'), localdatetime('2020-06-15T12:30:45'), duration('P1Y2M3DT4H5M6.5S'), duration('P12W')"),
    Q("shortestPath in CALL", False, "MATCH (a:Person {id: 1}), (b:Person {id: 5}) CALL { WITH a, b RETURN shortestPath((a)-[:KNOWS*..10]->(b)) AS p } RETURN length(p)"),
    Q("map projection row",  False, "MATCH (a:Person {id: 1}) CALL { WITH a RETURN a{.id, .name, double: a.id * 2} AS m } RETURN m"),
    Q("list comp row",       False, "MATCH (a:Person {id: 1}) CALL { WITH a RETURN [x IN range(1, 10) WHERE x % 2 = 0 | x * a.id] AS l } RETURN l"),
    Q("agg in CALL",         False, "MATCH (p:Person) WHERE p.id < 100 CALL { WITH p MATCH (p)-[:KNOWS]->(q) RETURN avg(q.id) AS a, sum(q.id) AS s, min(q.id) AS mn, max(q.id) AS mx, stDev(q.id) AS sd, percentileDisc(q.id, 0.5) AS pd, collect(q.id) AS c } RETURN count(*), avg(a), sum(s), min(mn), max(mx)"),
    Q("CASE projection",     False, "MATCH (p:Person) WHERE p.id < 50 RETURN CASE WHEN p.id % 2 = 0 THEN 'even' ELSE 'odd' END, CASE p.id % 3 WHEN 0 THEN 'z' WHEN 1 THEN 'o' ELSE 'n' END"),
    Q("chained eq",          False, "MATCH (p:Person) WHERE p.id < 20 RETURN p.id = p.id = p.id, p.id <> 3 <> 4"),
    Q("query params",        False, "CYPHER pid=7 lst=[1,2,3] m={a: 1} MATCH (p:Person {id: $pid}) RETURN p.id, $lst, $m", cg=True),
    Q("IN mod filter",       False, "MATCH (p:Person) WHERE (p.id % 10) IN [1, 3, 5] RETURN count(p)"),
    Q("edge IN index",       False, "MATCH ()-[r:SIMILAR]->() WHERE r.weight IN [1, 3, 5] RETURN count(r)"),
    Q("IN on list prop",     False, "MATCH (d:IDoc) WHERE 1 IN d.v RETURN count(d)"),
    Q("ORDER BY agg",        False, "MATCH (p:Person) RETURN p.id % 5 AS k, count(*) ORDER BY count(*) DESC, k LIMIT 3", cg=True),
    Q("AND pattern filter",  False, "MATCH (a:Person {id: 1}) WHERE (a)-[:KNOWS]->() AND a.id > 0 RETURN a.id", cg=True),
    Q("cross product filter", False, "MATCH (a:Person), (b:Person) WHERE a.id < 3 AND b.id < 3 AND a.id < b.id RETURN a.id, b.id", cg=True),
    Q("vlt edge prop filter", False, "MATCH (a:Person {id: 1})-[:KNOWS*1..3 {weight: 1}]->(b) RETURN count(b)"),
    Q("correlated edge dyn idx", False, "MATCH (p:Person) WHERE p.id < 3 CALL { WITH p MATCH ()-[r:SIMILAR]->() WHERE r.weight = p.id * 0.001 RETURN count(r) AS c } RETURN sum(c)"),
    Q("delete return entities", True, "CREATE (x:TmpDel {a: 1, b: 'x'})-[r:TDEL {w: 2}]->(y:TmpDel {c: 3}) WITH x, y, r DELETE r, x, y RETURN x, r", 500),
    Q("create delete indexed", True, "CREATE (d:IDoc {v: 999999}) WITH d DELETE d", 500),
    Q("delete indexed edge", True,  "MATCH (a:Person {id: 1}), (b:Person {id: 2}) CREATE (a)-[r:SIMILAR {weight: 0.987654}]->(b) WITH r DELETE r", 500),
    Q("MERGE edge on create/match", True, "MATCH (a:Person {id: 1}), (b:Person {id: 2}) MERGE (a)-[r:MRG {k: 1}]->(b) ON CREATE SET r.c = 1 ON MATCH SET r.m = 1"),
    Q("fulltext edge query", False, "CALL db.idx.fulltext.queryRelationships('REF', 'hello') YIELD relationship, score RETURN count(relationship), max(score)"),
    Q("fulltext stopwords",  False, "CALL db.idx.fulltext.queryNodes('SW2', 'fox') YIELD node RETURN count(node)"),
    Q("string index range",  False, "MATCH (n:SIdx) WHERE n.s > 'name_95' RETURN count(n)"),
    Q("string index between", False, "MATCH (n:SIdx) WHERE n.s >= 'name_2' AND n.s < 'name_4' RETURN count(n)"),
    Q("distance index scan", False, "MATCH (n:Place) WHERE distance(n.loc, point({latitude: 32.0, longitude: 34.0})) < 3000 RETURN count(n)"),
    Q("multi-label index seek", False, "MATCH (n:Person:MEnd {id: 1}) RETURN count(n)"),
    Q("SPpaths pathCount",   False, "MATCH (a:Person {id: 1}), (b:Person {id: 40}) CALL algo.SPpaths({sourceNode: a, targetNode: b, relTypes: ['KNOWS'], relDirection: 'both', maxLen: 50, pathCount: 2}) YIELD path, pathWeight, pathCost RETURN length(path), pathWeight, pathCost"),
    Q("SSpaths pathCount",   False, "MATCH (a:Person {id: 1}) CALL algo.SSpaths({sourceNode: a, relTypes: ['KNOWS'], relDirection: 'outgoing', maxLen: 3, pathCount: 5}) YIELD path RETURN count(path)"),
    Q("SPpaths weighted",    False, "MATCH (a:Person {id: 1}), (b:Person {id: 10}) CALL algo.SPpaths({sourceNode: a, targetNode: b, relTypes: ['KNOWS'], relDirection: 'both', maxLen: 50, weightProp: 'weight', costProp: 'weight', maxCost: 100.5, pathCount: 3}) YIELD path, pathWeight RETURN count(path), max(pathWeight)"),
    Q("betweenness default", False, "CALL algo.betweenness({samplingSize: 10}) YIELD node, score RETURN count(node)", 20),
    Q("labelPropagation default", False, "CALL algo.labelPropagation({maxIterations: 3}) YIELD node, communityId RETURN count(node)", 20),
    Q("WCC typed",           False, "CALL algo.WCC({relationshipTypes: ['KNOWS']}) YIELD node, componentId RETURN count(node)", 200),
    Q("MSF nodes yield",     False, "CALL algo.MSF({relationshipTypes: ['SIMILAR'], weightAttribute: 'weight'}) YIELD nodes, edges RETURN size(nodes), size(edges)", 200),
    Q("maxFlow KNOWS",       False, "MATCH (a:Person {id: 1}), (b:Person {id: 5}) CALL algo.maxFlow({sourceNodes: [a], targetNodes: [b], relationshipTypes: ['KNOWS'], capacityProperty: 'weight', defaultCapacity: 1.0}) YIELD maxFlow RETURN maxFlow", 200),

    # ---- round 10: comprehension-body (Row-context) eval, null/scalar
    # iterables, chained-compare null arms, temporal component accessors,
    # value ordering arms, CALL{}+UNION binder, pending-entity reads,
    # label predicate, label-filtered algo matrices, lexer corners ----------
    Q("map proj in comprehension", False, "MATCH (p:Person {id: 1}) RETURN [x IN range(1, 3) | p{.id, .name, xx: x}]"),
    Q("shortestPath in comprehension", False, "MATCH (a:Person {id: 1}), (b:Person {id: 5}) RETURN [x IN [1] | shortestPath((a)-[:KNOWS*..10]->(b))]"),
    Q("nested comprehension", False, "MATCH (a:Person {id: 1}) RETURN [x IN range(1, 3) | [y IN range(1, x) | y * a.id]]", cg=True),
    Q("func in comprehension", False, "MATCH (p:Person {id: 1}) RETURN [x IN range(1, 3) | toString(x) + p.name]"),
    Q("rel prop in comprehension", False, "MATCH (a:Person {id: 1})-[r:KNOWS]->(b) RETURN [x IN [1] | r.weight + x]"),
    Q("param in comprehension", False, "CYPHER k=5 RETURN [x IN range(1, 3) | x + $k]"),
    Q("chained cmp in comprehension", False, "RETURN [x IN [1] | 1 < 2 < x]"),
    Q("chained cmp null",    False, "RETURN 1 < null < 3, null = null, 1 <= null >= 2"),
    Q("rel subscript",       False, "MATCH (a:Person {id: 1})-[r:KNOWS]->() RETURN r['weight'] LIMIT 5"),
    Q("list null subscript", False, "RETURN [1, 2, 3][null]"),
    Q("null iterable",       False, "RETURN [x IN null | x], all(x IN null WHERE x > 0), any(x IN null WHERE x > 0), none(x IN null WHERE x > 0), single(x IN null WHERE x > 0)"),
    Q("scalar iterables",    False, "UNWIND 5 AS x RETURN x, [y IN 6 | y]"),
    Q("single multi true",   False, "RETURN single(x IN [1, 2, 3] WHERE x > 1)"),
    Q("unary minus null",    False, "MATCH (p:Person {id: 1}) RETURN -p.noSuchProp"),
    Q("untyped shortestPath", False, "MATCH (a:Person {id: 1}), (b:Person {id: 5}) RETURN shortestPath((a)-[*..5]->(b))"),
    Q("zero hop shortestPath", False, "MATCH (a:Person {id: 1}) RETURN shortestPath((a)-[:KNOWS*0..2]->(a))"),
    Q("date components",     False, "RETURN date('2024-03-15').week, date('2024-03-15').dayOfWeek, date('2024-03-15').quarter, date('2024-03-15').ordinalDay, date('2024-03-15').weekYear, date('2024-03-15').dayOfQuarter", cg=True),
    Q("temporal props read", False, "MATCH (v:TVal) RETURN v.t, v.dt, v.d, v.du LIMIT 1"),
    Q("point order by",      False, "UNWIND [point({latitude: 1.0, longitude: 2.0}), point({latitude: 0.5, longitude: 3.0})] AS p RETURN p ORDER BY p"),
    Q("mixed type order by", False, "MATCH (n:Person) WHERE n.id < 5 UNWIND [n, n.id, 'a'] AS v RETURN v ORDER BY v"),
    Q("call union",          False, "CALL { RETURN 1 AS x UNION RETURN 2 AS x } RETURN x"),
    Q("call union import",   False, "MATCH (p:Person {id: 1}) CALL { WITH p RETURN p.id AS x UNION WITH p RETURN p.id * 2 AS x } RETURN sum(x)"),
    Q("label predicate",     False, "MATCH (n) WHERE n:Person AND n.id < 5 RETURN count(n)"),
    Q("match pending edge",  True,  "MATCH (a:Person {id: 1}), (b:Person {id: 2}) CREATE (a)-[:PEND {v: 1}]->(b) WITH a MATCH (a)-[r:PEND]->() DELETE r RETURN count(r)", 500),
    Q("delete pending connected", True, "CREATE (x:TmpP)-[:TP]->(y:TmpP) WITH x, y DELETE x, y", 500),
    Q("rereferenced node labels", False, "MATCH (a:Person {id: 1})-[:KNOWS]->(b), (b:Person)-[:KNOWS]->(c) RETURN count(c)"),
    Q("rereferenced node attrs", False, "MATCH (a:Person {id: 1})-[:KNOWS]->(b), (b {id: 2})-[:KNOWS]->(c) RETURN count(c)"),
    Q("betweenness labeled", False, "CALL algo.betweenness({nodeLabels: ['Person'], relationshipTypes: ['KNOWS'], samplingSize: 5}) YIELD node, score RETURN count(node)", 20),
    Q("WCC labeled",         False, "CALL algo.WCC({nodeLabels: ['Person'], relationshipTypes: ['KNOWS']}) YIELD node, componentId RETURN count(node)", 200),
    Q("all edges scan",      False, "MATCH ()-[r]->() RETURN count(r)"),
    Q("sci notation",        False, "RETURN 1e5, 1E-5, 1.5e+3, 2e10"),
    Q("string escapes",      False, "RETURN 'a\\'b', \"c\\\"d\", '\\t\\n\\\\'"),
    Q("backtick var",        False, "MATCH (`weird var`:Person {id: 1}) RETURN `weird var`.id"),

    # ---- round 11: DISTINCT aggregations (scalar agg path), compact
    # temporal strings, entity-to-entity SET, pending-created deletes
    # without WITH, unique-rel constraint corpus, all-tensor algos ----------
    Q("distinct aggs",       False, "MATCH (p:Person) WHERE p.id < 100 RETURN count(DISTINCT p.id % 10), sum(DISTINCT p.id % 10), avg(DISTINCT p.id % 10), min(DISTINCT p.id % 10), max(DISTINCT p.id % 10), collect(DISTINCT p.id % 5), stDev(DISTINCT p.id % 10)"),
    Q("chained eq in comprehension", False, "RETURN [x IN [1] | x = 1 = 1], [x IN [2] | x <> 1 <> 3]"),
    Q("compact date strings", False, "RETURN date('2020'), date('202006'), date('20200615')"),
    Q("compact time strings", False, "RETURN localtime('12'), localtime('1230'), localtime('123045')"),
    Q("duration map full",   False, "RETURN duration({years: 1, months: 2, weeks: 3, days: 4, hours: 5, minutes: 6, seconds: 7})"),
    Q("datetime accessors",  False, "RETURN localdatetime('2020-06-15T12:30:45').week, localdatetime('2020-06-15T12:30:45').dayOfWeek, localdatetime('2020-06-15T12:30:45').quarter, localdatetime('2020-06-15T12:30:45').ordinalDay"),
    Q("time accessors",      False, "RETURN localtime('12:30:45.123456789').hour, localtime('12:30:45.123456789').minute, localtime('12:30:45.123456789').second"),
    Q("set copy entity",     True,  "MATCH (x:SCopy)-[r:SC]->(y:SCopy2) SET x = y", 500),
    Q("set merge entity",    True,  "MATCH (x:SCopy)-[r:SC]->(y:SCopy2) SET x += y SET r += {w: 1}", 500),
    Q("create delete no with", True, "CREATE (x:TmpQ) DELETE x", 500),
    Q("create delete rel no with", True, "CREATE (x:TmpQ)-[r:TQ]->(y:TmpQ) DELETE r, x, y", 500),
    Q("urel create delete",  True,  "MATCH (a:Person {id: 1}), (b:Person {id: 2}) CREATE (a)-[r:UREL {uid: 999}]->(b) DELETE r", 500),
    Q("rel mixed order by",  False, "MATCH ()-[r:KNOWS]->() WITH r LIMIT 2 UNWIND [r, 1] AS v RETURN v ORDER BY v"),
    Q("labelPropagation all types", False, "CALL algo.labelPropagation({maxIterations: 2}) YIELD node RETURN count(node)", 20),
    Q("WCC all types",       False, "CALL algo.WCC({}) YIELD node, componentId RETURN count(node)", 200),

    # ---- optimizer / runtime slow-risk paths -------------------------------
    # Each row targets a plan-rewrite or per-edge runtime mechanism that is
    # easy to regress and was previously unexercised by the bench set.
    # Chain reversal (select_scan_node): selective scan at the END of the
    # chain, whole chain reversed with transposed traversals.
    Q("reversed 2hop chain",  False, "MATCH (a)-[:KNOWS]->(b)-[:KNOWS]->(c:Person {id: 1}) RETURN count(a)", cg=True),
    # Same, with a mid-chain filter reinserted at the correct hop.
    Q("reversed chain mid filter", False, "MATCH (a)-[:KNOWS]->(b)-[:KNOWS]->(c:Person {id: 1}) WHERE b.id > 0 RETURN count(a)"),
    # Fused consecutive anonymous bidirectional traverses with cross-row
    # (from,to) dedup (cond_traverse fused-dedup path).
    Q("bidir anon 2hop dedup", False, "MATCH (a:Person {id: 1})-[:KNOWS]-()-[:KNOWS]-(c) RETURN count(c)"),
    # Per-edge inline rel-attr map check during traverse over all 10k edges.
    Q("traverse inline rel attrs", False, "MATCH (a:Person)-[:KNOWS {weight: 1}]->(b) RETURN count(b)"),
    # ExpandInto with inline rel attrs checked per candidate edge (triangle).
    Q("expand-into inline rel attrs", False, "MATCH (a)-[:KNOWS]->(b)-[:KNOWS]->(c)-[:KNOWS {weight: 1}]->(a) RETURN count(a)"),
    # CP-split (push_filters_down): multi-branch conjunct pulls a,b into an
    # inner filtered cartesian product, c stays outer.
    Q("cartesian split filter", False, "MATCH (a:Person), (b:Person), (c:Person) WHERE a.id < 20 AND b.id < 20 AND c.id < 20 AND a.id + b.id = 5 RETURN count(*)"),
    # Upper-bound-only index ranges (Le on node index, Lt on edge index).
    Q("index le range",       False, "MATCH (p:Person) WHERE p.id <= 50 RETURN count(p)"),
    Q("edge index lt range",  False, "MATCH ()-[r:SIMILAR]->() WHERE r.weight < 5 RETURN count(r)"),
    # Two lower bounds on the same attribute: merge_range_queries falls back
    # to IndexQuery::And (can't compare expr values at plan time).
    Q("index same-bound and", False, "MATCH (p:Person) WHERE p.id > 10 AND p.id >= 20 RETURN count(p)"),
    # Runtime-bound index range: b's scan range depends on a row value.
    Q("runtime-bound index range", False, "MATCH (a:Person {id: 9990}) WITH a MATCH (b:Person) WHERE b.id > a.id RETURN count(b)"),
    # Distance index scan over the Geo point index (IndexQuery::Point).
    Q("distance index scan geo", False, "MATCH (g:Geo) WHERE distance(g.loc, point({latitude: 0.0, longitude: 0.0})) < 10000 RETURN count(g)"),

    # ---- columnar expression evaluation ------------------------------------
    # Unindexed 10k-Person scans whose predicate/aggregate input is a computed
    # tree, i.e. the shapes that fall off the columnar path onto per-row
    # expression evaluation. `expr prop cmp` is the control: the same scan with
    # a bare `property <op> constant` predicate, which the kernels already take.
    Q("expr prop cmp",       False, "MATCH (p:Person) WHERE p.age > 45 RETURN count(p)", cg=True),
    Q("expr mod filter",     False, "MATCH (p:Person) WHERE p.age % 7 = 3 RETURN count(p)", cg=True),
    Q("expr arith filter",   False, "MATCH (p:Person) WHERE p.age * 2 + 1 > 100 RETURN count(p)", cg=True),
    Q("expr or filter",      False, "MATCH (p:Person) WHERE p.age > 70 OR p.score < 100.0 RETURN count(p)", cg=True),
    Q("expr not filter",     False, "MATCH (p:Person) WHERE NOT p.age > 70 RETURN count(p)", cg=True),
    Q("expr starts with",    False, "MATCH (p:Person) WHERE p.name STARTS WITH 'p1' RETURN count(p)", cg=True),
    Q("expr func filter",    False, "MATCH (p:Person) WHERE toUpper(p.name) = 'P100' RETURN count(p)", cg=True),
    Q("expr sum computed",   False, "MATCH (p:Person) RETURN sum(p.age * 3 + 1)", cg=True),
    Q("expr case filter",    False, "MATCH (p:Person) WHERE (CASE WHEN p.age > 40 THEN 1 ELSE 2 END) = 1 RETURN count(p)", cg=True),
    Q("expr project computed", False, "MATCH (p:Person) WITH p.age * 2 AS d RETURN count(d)", cg=True),

    # ---- sized writes ------------------------------------------------------
    # Kept LAST: they inflate node capacity / matrix dimension to max(N),
    # which would slow every full-graph query measured after them.
    # "write N" is the mixed create+delete round-trip; the "create N" /
    # "delete N" pairs measure the two halves separately.
    # Ordered by the amount of churn each row causes, ascending, so no row is
    # preceded by one an order of magnitude larger. Grouping the mixed
    # "write N" rows ahead of the create/delete pairs instead put `create 100`
    # and `create 10k` directly after `write 1m`, and a write costs more while
    # the engine still carries a large deletion: creating a single node costs
    # 0.05 ms after a 10k delete and 0.69 ms after a 1M one. Those two rows were
    # measuring recovery from `write 1m` rather than the cost of a create, and
    # read as a 4.4x / 3.0x regression against C that a fresh graph does not
    # show (there Rust is at 1.28x and 0.74x).
    Q("write 1",             True,  "UNWIND range(1, 1) AS i CREATE (t:Tmp {x: i}) WITH t DELETE t", 1000),
    Q("write 10",            True,  "UNWIND range(1, 10) AS i CREATE (t:Tmp {x: i}) WITH t DELETE t", 1000),
    Q("write 100",           True,  "UNWIND range(1, 100) AS i CREATE (t:Tmp {x: i}) WITH t DELETE t", 500),
    Q("create 100",          True,  "UNWIND range(1, 100) AS i CREATE (:Tmp {x: i})", 500),
    Q("delete 100",          True,  "MATCH (t:Tmp) WITH t LIMIT 100 DELETE t", 500),
    Q("write 1k",            True,  "UNWIND range(1, 1000) AS i CREATE (t:Tmp {x: i}) WITH t DELETE t", 200),
    Q("write 10k",           True,  "UNWIND range(1, 10000) AS i CREATE (t:Tmp {x: i}) WITH t DELETE t", 50),
    Q("create 10k",          True,  "UNWIND range(1, 10000) AS i CREATE (:Tmp {x: i})", 50),
    Q("delete 10k",          True,  "MATCH (t:Tmp) WITH t LIMIT 10000 DELETE t", 50),
    Q("write 100k",          True,  "UNWIND range(1, 100000) AS i CREATE (t:Tmp {x: i}) WITH t DELETE t", 10),
    Q("write 1m",            True,  "UNWIND range(1, 1000000) AS i CREATE (t:Tmp {x: i}) WITH t DELETE t", 2),

    # ---- one write against a large EDGE population -------------------------
    # Every sized row above creates and deletes *nodes*, so the edge count
    # stays at SETUP's ~10k for the whole run and no row in the suite is
    # sensitive to the cost of a write scaling with |E|. That hid #2687, where
    # the edge-endpoint index was copied whole on the first edge mutation of
    # every write transaction: 2,676,794 bytes allocated to create one edge
    # against 200k of them, against 52,454 once paged. On this graph the same
    # bug is worth 40 KB, so the suite read it as a 9% row.
    #
    # `edge create at 200k` measures the same create/delete round-trip as
    # `urel create delete`, three orders of magnitude further up the edge count.
    # It DEPENDS on `bulk edges 200k` having run first: a name-filtered run that
    # picks it alone measures it at 10k edges instead, quietly and without
    # failing, exactly as `delete 100` depends on `create 100`. The pairing is
    # held by a test.
    #
    # Both go last so the 200k edges cannot shift any other row, and the
    # measured row follows a large *create* rather than a large delete, so it
    # is not reading recovery from `write 1m` the way `create 100` once did.
    Q("bulk edges 200k",     True,  "UNWIND range(1, 200000) AS i CREATE (:Wide)-[:WIDE]->(:Wide)", 1),
    Q("edge create at 200k", True,  "MATCH (a:Person {id: 1}), (b:Person {id: 2}) CREATE (a)-[r:WIDEX]->(b) WITH r DELETE r", 100),
]

# Expected-error queries: run only in --once (coverage) mode, never timed.
# Each row is (name, redis command, query); the harness passes iff the reply
# is a non-empty error (no "execution time" stat). They cover parse/bind/eval
# error paths and constraint-violation rollback that success queries can't.
ERROR_QUERIES = [
    ("dup rel var match",    "GRAPH.QUERY", "MATCH (a)-[r:KNOWS]->()-[r:KNOWS]->() RETURN r"),
    ("redeclare bound var",  "GRAPH.QUERY", "MATCH ()-[r:KNOWS]->() CREATE ()-[r:KNOWS]->()"),
    ("index no label",       "GRAPH.QUERY", "CREATE INDEX FOR (n) ON (n.x)"),
    ("index wrong var",      "GRAPH.QUERY", "CREATE INDEX FOR (n:Person) ON (m.x)"),
    ("parse trailing op",    "GRAPH.QUERY", "RETURN 1 +"),
    ("parse unclosed paren", "GRAPH.QUERY", "MATCH (n RETURN n"),
    ("undefined var",        "GRAPH.QUERY", "RETURN x"),
    ("unknown function",     "GRAPH.QUERY", "RETURN noSuchFunc(1)"),
    ("arg count",            "GRAPH.QUERY", "RETURN toUpper('a', 'b')"),
    ("arg type",             "GRAPH.QUERY", "RETURN toUpper(1)"),
    ("missing param",        "GRAPH.QUERY", "RETURN $nope"),
    ("bad date",             "GRAPH.QUERY", "RETURN date('not-a-date')"),
    ("bad duration",         "GRAPH.QUERY", "RETURN duration('xyz')"),
    ("add bool int",         "GRAPH.QUERY", "RETURN true + 1"),
    ("sub int string",       "GRAPH.QUERY", "RETURN 1 - 'a'"),
    ("mul string",           "GRAPH.QUERY", "RETURN 'a' * 2"),
    ("div by zero",          "GRAPH.QUERY", "RETURN 1 / 0"),
    ("unique node violation", "GRAPH.QUERY", "CREATE (:Doc {id: 1})"),
    ("mandatory rel violation", "GRAPH.QUERY", "MATCH (a:Person {id: 1}), (b:Person {id: 2}) CREATE (a)-[:SIMILAR]->(b)"),
    ("dup return alias",     "GRAPH.QUERY", "MATCH (n:Person) RETURN n.id AS x, n.name AS x"),
    ("allShortestPaths in return", "GRAPH.QUERY", "MATCH (a:Person {id: 1}), (b:Person {id: 2}) RETURN allShortestPaths((a)-[:KNOWS*..3]->(b))"),
    ("agg in where",         "GRAPH.QUERY", "MATCH (n:Person) WHERE count(n) > 0 RETURN 1"),
    ("set on path",          "GRAPH.QUERY", "MATCH p = (:Person {id: 1})-[:KNOWS]->() SET p.x = 1"),
    ("map property value",   "GRAPH.QUERY", "CREATE (:Bad {p: {a: 1}})"),
    ("merge null prop",      "GRAPH.QUERY", "MERGE (n:Person {id: null}) RETURN n"),
    ("delete property",      "GRAPH.QUERY", "MATCH (n:Person {id: 1}) DELETE n.id"),
    ("reduce non list",      "GRAPH.QUERY", "RETURN reduce(a = 0, x IN 5 | a + x)"),
    ("unknown procedure",    "GRAPH.QUERY", "CALL db.idx.doesnotexist()"),
    ("sppaths missing source", "GRAPH.QUERY", "CALL algo.SPpaths({}) YIELD path RETURN path"),
    ("ro write rejected",    "GRAPH.RO_QUERY", "CREATE (:NopeRO)"),
    ("empty label",          "GRAPH.QUERY", "MATCH (n:) RETURN n"),
    ("urel unique violation", "GRAPH.QUERY", "MATCH (a:Person {id: 1}), (b:Person {id: 2}) CREATE (a)-[:UREL {uid: 1}]->(b)"),
    ("int add bool",         "GRAPH.QUERY", "RETURN 1 + true"),
    ("toupper list",         "GRAPH.QUERY", "RETURN toUpper([1])"),
    ("bad compact date",     "GRAPH.QUERY", "RETURN date('20201')"),
    ("bad compact time",     "GRAPH.QUERY", "RETURN localtime('123')"),
    # REF has two parallel edges between Person 1 and 2 (fulltext corpus),
    # so the multi-edge rejection is deterministic on a fresh graph.
    ("maxflow multi edge",   "GRAPH.QUERY", "MATCH (a:Person {id: 1}), (b:Person {id: 2}) CALL algo.maxFlow({sourceNodes: [a], targetNodes: [b], relationshipTypes: ['REF'], capacityProperty: 'cap', defaultCapacity: 1.0}) YIELD maxFlow RETURN maxFlow"),
    ("maxflow missing capacity", "GRAPH.QUERY", "MATCH (a:Person {id: 1}), (b:Person {id: 5}) CALL algo.maxFlow({sourceNodes: [a], targetNodes: [b], relationshipTypes: ['KNOWS']}) YIELD maxFlow RETURN maxFlow"),
]
