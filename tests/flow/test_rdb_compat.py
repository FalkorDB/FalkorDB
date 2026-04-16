"""
RDB cross-compatibility tests between FalkorDB C (v4.18.1) and FalkorDB Rust.

Both implementations use encoding version 19 and module type "graphdata".
Uses Redis replication (REPLICAOF) to transfer RDB data between servers,
avoiding DUMP/RESTORE version-checksum mismatches.
"""

import os
import time
from random import randint, seed
from common import *

FALKORDB_C_IMAGE = 'falkordb/falkordb:v4.18.1'

# ──────────────────────────── Docker helpers ────────────────────────────

def run_db(image):
    """Start a FalkorDB container on a random port."""
    import docker
    client = docker.from_env()
    port = randint(49152, 65535)
    container = client.containers.run(
        image,
        detach=True,
        ports={'6379/tcp': port},
        extra_hosts={'host.docker.internal': 'host-gateway'},
    )
    return container, port

def stop_db(container):
    """Stop and remove a Docker container."""
    container.stop()
    container.remove()

def wait_for_db(port, timeout=30):
    """Poll until the Redis instance at *port* accepts connections."""
    import redis as _redis
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = _redis.Redis(host='localhost', port=port)
            r.ping()
            return
        except Exception:
            time.sleep(0.5)
    raise RuntimeError(f"FalkorDB container on port {port} did not start in {timeout}s")

def wait_for_replication(conn, timeout=30):
    """Wait until a replica has completed initial sync."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        info = conn.info('replication')
        if info.get('role') == 'slave':
            if info.get('master_link_status') == 'up' and info.get('master_sync_in_progress', 0) == 0:
                return
        time.sleep(0.5)
    raise RuntimeError("Replication did not complete in time")

# ──────────────────────── Graph creation helpers ────────────────────────

SIMPLE_QUERIES = [
    "CREATE (:Person {name: 'Alice', age: 30, score: 9.5, active: true, tags: [1,2,3], loc: POINT({latitude: 32.0816, longitude: 34.7818})})-[:KNOWS {since: 2020}]->(:Person {name: 'Bob', age: 25})",
    "CREATE (:City {name: 'TLV', population: 460613})",
    "CREATE INDEX FOR (p:Person) ON (p.name)",
    "CREATE INDEX FOR (p:Person) ON (p.age)",
]

SIMPLE_VERIFICATION = [
    ("labels", "CALL db.labels() YIELD label RETURN label ORDER BY label"),
    ("rel types", "CALL db.relationshiptypes() YIELD relationshipType RETURN relationshipType ORDER BY relationshipType"),
    ("node count", "MATCH (n) RETURN count(n)"),
    ("edge count", "MATCH ()-[e]->() RETURN count(e)"),
    ("persons", "MATCH (p:Person) RETURN p.name, p.age, p.score, p.active, p.tags ORDER BY p.name"),
    ("city", "MATCH (c:City) RETURN c.name, c.population"),
    ("edges", "MATCH ()-[e]->() RETURN type(e), properties(e) ORDER BY e"),
    ("index scan name", "MATCH (p:Person) WHERE p.name = 'Alice' RETURN p.name, p.age"),
    ("index scan age", "MATCH (p:Person) WHERE p.age > 20 RETURN p.name ORDER BY p.name"),
    ("point exists", "MATCH (p:Person {name: 'Alice'}) RETURN p.loc IS NOT NULL"),
]

def create_simple_graph(g):
    """Populate *g* with the simple test graph and wait for indexes."""
    for q in SIMPLE_QUERIES:
        g.query(q)
    _wait_for_indexes(g)

def _wait_for_indexes(g, timeout=30):
    """Wait until all indexes on *g* are OPERATIONAL."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        result = g.ro_query(
            "CALL db.indexes() YIELD status WHERE status <> 'OPERATIONAL' RETURN count(1)"
        )
        if result.result_set[0][0] == 0:
            return
        time.sleep(0.2)
    raise RuntimeError("Indexes did not become OPERATIONAL in time")

def capture_state(g, queries):
    """Run *queries* on graph *g* and return {label: result_set}."""
    state = {}
    for label, q in queries:
        state[label] = g.ro_query(q).result_set
    return state

def assert_state_eq(env, expected, actual):
    """Assert two captured states are identical."""
    for label in expected:
        if expected[label] != actual.get(label):
            print(f"MISMATCH in '{label}':")
            print(f"  expected: {expected[label]}")
            print(f"  actual:   {actual.get(label)}")
        env.assertEqual(expected[label], actual.get(label))

# ───────────────────────── Random graph helpers ─────────────────────────

RANDOM_LABELS = ['Alpha', 'Beta', 'Gamma', 'Delta']
RANDOM_REL_TYPES = ['LINKS', 'CONNECTS', 'FOLLOWS']

def create_random_graph(g, rng_seed=42):
    """Create a deterministic random graph on *g*."""
    seed(rng_seed)

    # Nodes: deterministic properties per label
    for label in RANDOM_LABELS:
        count = randint(15, 25)
        g.query(
            f"UNWIND range(1, {count}) AS i "
            f"CREATE (:{label} {{id: i, name: '{label}_' + toString(i), "
            f"val: toFloat(i) * 1.5, flag: i % 2 = 0, nums: [i, i+1, i+2]}})"
        )

    # Edges: deterministic cross-label connections
    for rel_type in RANDOM_REL_TYPES:
        src = RANDOM_LABELS[randint(0, len(RANDOM_LABELS) - 1)]
        dst = RANDOM_LABELS[randint(0, len(RANDOM_LABELS) - 1)]
        g.query(
            f"MATCH (a:{src}), (b:{dst}) "
            f"WITH a, b LIMIT 20 "
            f"CREATE (a)-[:{rel_type} {{weight: toFloat(a.id + b.id)}}]->(b)"
        )

    # Range indexes
    for label in RANDOM_LABELS:
        g.query(f"CREATE INDEX FOR (n:{label}) ON (n.id)")

    _wait_for_indexes(g)

RANDOM_VERIFICATION = [
    ("labels", "CALL db.labels() YIELD label RETURN label ORDER BY label"),
    ("rel types", "CALL db.relationshiptypes() YIELD relationshipType RETURN relationshipType ORDER BY relationshipType"),
    ("node count", "MATCH (n) RETURN count(n)"),
    ("edge count", "MATCH ()-[e]->() RETURN count(e)"),
    ("nodes", "MATCH (n) RETURN labels(n), properties(n) ORDER BY n"),
    ("edges", "MATCH ()-[e]->() RETURN type(e), properties(e) ORDER BY e"),
    ("index count", "CALL db.indexes() YIELD label RETURN count(label)"),
]

# ═══════════════════════════ Test class ═══════════════════════════════

class testRdbCompat():
    def __init__(self):
        self.env, self.db = Env(enableDebugCommand=True)
        self.redis_con = self.env.getConnection()
        self.rust_port = self.env.port
        # Allow Docker containers to connect to the Rust server
        self.redis_con.execute_command('CONFIG', 'SET', 'bind', '0.0.0.0')
        self.redis_con.execute_command('CONFIG', 'SET', 'protected-mode', 'no')

    # ── Test 1: C -> Rust (simple) ──

    def test01_c_to_rust_simple(self):
        """C produces RDB, Rust loads it via replication."""
        key = 'G'

        container, c_port = run_db(FALKORDB_C_IMAGE)
        try:
            wait_for_db(c_port)
            c_db = FalkorDB(port=c_port)
            c_graph = c_db.select_graph(key)
            create_simple_graph(c_graph)
            expected = capture_state(c_graph, SIMPLE_VERIFICATION)

            # Rust replicates from C
            self.redis_con.execute_command('REPLICAOF', 'localhost', str(c_port))
            wait_for_replication(self.redis_con)
            self.redis_con.execute_command('REPLICAOF', 'NO', 'ONE')
        finally:
            stop_db(container)

        r_graph = self.db.select_graph(key)
        actual = capture_state(r_graph, SIMPLE_VERIFICATION)
        assert_state_eq(self.env, expected, actual)

    # ── Test 2: C -> Rust (random) ──

    def test02_c_to_rust_random(self):
        """C produces random graph RDB, Rust loads it via replication."""
        key = 'R'

        container, c_port = run_db(FALKORDB_C_IMAGE)
        try:
            wait_for_db(c_port)
            c_db = FalkorDB(port=c_port)
            c_graph = c_db.select_graph(key)
            create_random_graph(c_graph)
            expected = capture_state(c_graph, RANDOM_VERIFICATION)

            self.redis_con.execute_command('REPLICAOF', 'localhost', str(c_port))
            wait_for_replication(self.redis_con)
            self.redis_con.execute_command('REPLICAOF', 'NO', 'ONE')
        finally:
            stop_db(container)

        r_graph = self.db.select_graph(key)
        actual = capture_state(r_graph, RANDOM_VERIFICATION)
        assert_state_eq(self.env, expected, actual)

    # ── Test 3: Rust -> C (simple) ──

    def test03_rust_to_c_simple(self):
        """Rust produces RDB, C loads it via replication."""
        key = 'G'
        self.redis_con.flushall()

        r_graph = self.db.select_graph(key)
        create_simple_graph(r_graph)
        expected = capture_state(r_graph, SIMPLE_VERIFICATION)

        container, c_port = run_db(FALKORDB_C_IMAGE)
        try:
            wait_for_db(c_port)
            import redis as _redis
            c_conn = _redis.Redis(host='localhost', port=c_port)

            # C replicates from Rust
            c_conn.execute_command('REPLICAOF', 'host.docker.internal', str(self.rust_port))
            wait_for_replication(c_conn)
            c_conn.execute_command('REPLICAOF', 'NO', 'ONE')

            c_db = FalkorDB(port=c_port)
            c_graph = c_db.select_graph(key)
            actual = capture_state(c_graph, SIMPLE_VERIFICATION)
            assert_state_eq(self.env, expected, actual)
        finally:
            stop_db(container)

    # ── Test 4: Rust -> C (random) ──

    def test04_rust_to_c_random(self):
        """Rust produces random graph RDB, C loads it via replication."""
        key = 'R'
        self.redis_con.flushall()

        r_graph = self.db.select_graph(key)
        create_random_graph(r_graph)
        expected = capture_state(r_graph, RANDOM_VERIFICATION)

        container, c_port = run_db(FALKORDB_C_IMAGE)
        try:
            wait_for_db(c_port)
            import redis as _redis
            c_conn = _redis.Redis(host='localhost', port=c_port)

            c_conn.execute_command('REPLICAOF', 'host.docker.internal', str(self.rust_port))
            wait_for_replication(c_conn)
            c_conn.execute_command('REPLICAOF', 'NO', 'ONE')

            c_db = FalkorDB(port=c_port)
            c_graph = c_db.select_graph(key)
            actual = capture_state(c_graph, RANDOM_VERIFICATION)
            assert_state_eq(self.env, expected, actual)
        finally:
            stop_db(container)
