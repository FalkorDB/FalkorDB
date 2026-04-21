import pytest
import common


def setup_module(module):
    common.start_redis(release=True)


def teardown_module(module):
    common.shutdown_redis()


def setup_function(function):
    if common.g.name in common.client.list_graphs():
        common.g.delete()

def run_query(q: str, params=None):
    common.g.query(q, params)

def reset_graph():
    if common.g.name in common.client.list_graphs():
        common.g.delete()

def test_return(benchmark):
    benchmark(run_query, "RETURN 1")

@pytest.mark.parametrize("n", [
    1, 10, 100, 1000, 10000, 100000, 1000000
])
def test_unwind(benchmark, n):
    benchmark(run_query, f"UNWIND range(1, {n}) AS x RETURN x")

@pytest.mark.parametrize("n", [
    1, 10, 100, 1000, 10000, 100000, 1000000
])
def test_create_node(benchmark, n):
    def setup():
        reset_graph()
    benchmark.pedantic(run_query, args=(f"UNWIND range(1, {n}) AS x CREATE (:N {{id: x}})",), setup=setup, rounds=5, warmup_rounds=1)

@pytest.mark.parametrize("n", [
    1, 10, 100, 1000, 10000, 100000, 1000000
])
def test_create_relationship(benchmark, n):
    def setup():
        reset_graph()
    benchmark.pedantic(run_query, args=(f"UNWIND range(1, {n}) AS x CREATE (:N {{id: x}})-[:R]->(:N {{id: x + 1}})",), setup=setup, rounds=5, warmup_rounds=1)

@pytest.mark.parametrize("n", [
    1, 10, 100, 1000, 10000, 100000, 1000000
])
def test_match_node(benchmark, n):
    run_query(f"UNWIND range(1, {n}) AS x CREATE (:N {{id: x}})")
    benchmark(run_query, f"MATCH (n:N) RETURN n")

@pytest.mark.parametrize("n", [
    1, 10, 100, 1000, 10000, 100000, 1000000
])
def test_match_relationship(benchmark, n):
    run_query(f"UNWIND range(1, {n}) AS x CREATE (:N {{id: x}})-[:R]->(:N {{id: x + 1}})")
    benchmark(run_query, f"MATCH (n)-[r:R]->(m) RETURN n, r, m")

@pytest.mark.parametrize("n", [
    1, 10, 100, 1000, 10000, 100000, 1000000
])
def test_delete_node(benchmark, n):
    def setup():
        reset_graph()
        run_query(f"UNWIND range(1, {n}) AS x CREATE (:N {{id: x}})")
    benchmark.pedantic(run_query, args=(f"MATCH (n:N) DELETE n",), setup=setup, rounds=5, warmup_rounds=1)

@pytest.mark.parametrize("n", [
    1, 10, 100, 1000, 10000, 100000, 1000000
])
def test_delete_relationship(benchmark, n):
    def setup():
        reset_graph()
        run_query(f"UNWIND range(1, {n}) AS x CREATE (:N {{id: x}})-[:R]->(:N {{id: x + 1}})")
    benchmark.pedantic(run_query, args=(f"MATCH ()-[r:R]->() DELETE r",), setup=setup, rounds=5, warmup_rounds=1)
