import common
from falkordb import FalkorDB
from multiprocessing import Pool


def setup_module(module):
    global is_extra
    from conftest import pytest_config

    is_extra = "extra" in pytest_config.getoption("-m")
    common.start_redis()


def teardown_module(module):
    common.shutdown_redis()


def setup_function(function):
    if common.g.name in common.client.list_graphs():
        common.g.delete()


def run_write_attribute(id):
    db = common.falkordb()
    g = db.select_graph("test")
    if id % 2 == 0:
        res = g.query("MATCH (n:Node) WHERE n.id = $id SET n.id = 0", params={"id": id})
    else:
        res = g.query("MATCH (n:Node) WHERE n.id = $id REMOVE n.id", params={"id": id})
    version = int(res._raw_stats[-1][15:])
    return (id, version)


def run_read_attribute(id):
    db = common.falkordb()
    g = db.select_graph("test")
    res = g.query("MATCH (n) RETURN n.id")
    version = int(res._raw_stats[2][15:])
    return (id, res.result_set, version)

def run_read_indexed_attribute(id):
    db = common.falkordb()
    g = db.select_graph("test")
    res = g.query("MATCH (n:Node) WHERE n.id >= 0 RETURN n.id")
    version = int(res._raw_stats[2][15:])
    return (id, res.result_set, version)


def run_write_label(id):
    db = common.falkordb()
    g = db.select_graph("test")
    if id % 2 == 0:
        res = g.query("MATCH (n:Node) WHERE n.id = $id SET n:L", params={"id": id})
    else:
        res = g.query(
            "MATCH (n:Node) WHERE n.id = $id REMOVE n:Node", params={"id": id}
        )
    version = int(res._raw_stats[-1][15:])
    return (id, version)


def run_read_label(id):
    db = common.falkordb()
    g = db.select_graph("test")
    res = g.query("MATCH (n) RETURN labels(n)")
    version = int(res._raw_stats[-1][15:])
    return (id, res.result_set, version)


def run_write_relationship(id):
    db = common.falkordb()
    g = db.select_graph("test")
    if id % 2 == 0:
        res = g.query(
            "MATCH (n:Node) WHERE id(n) = 0 WITH n MATCH (m:Node) WHERE id(m) = 1 CREATE (n)-[:RELATES_TO {id: 0}]->(m)",
            params={"id": id},
        )
        assert res.relationships_created == 1
    else:
        res = g.query(
            "MATCH (n)-[r:RELATES_TO {id: $id}]->(m) DELETE r",
            params={"id": id},
        )
        assert res.relationships_deleted == 1
    version = int(res._raw_stats[-1][15:])
    return (id, version)


def run_read_relationship(id):
    db = common.falkordb()
    g = db.select_graph("test")
    res = g.query("MATCH (n)-[r]->(m) RETURN r.id")
    version = int(res._raw_stats[2][15:])
    return (id, res.result_set, version)


def mvcc(data_query, run_write, run_read):
    common.g.query(data_query)

    pool1 = Pool(1)
    pool8 = Pool(8)

    res_write = pool1.map_async(run_write, range(1, 101))
    res_read = pool8.map_async(run_read, range(1, 101))

    res_write.wait()
    res_read.wait()

    return (res_write.get(), res_read.get())


def test_mvcc_version_discard_on_error():
    common.g.query("UNWIND range(1, 1000) AS x CREATE (:Node {id: x})")
    try:
        common.g.query(
            "UNWIND [1, 2, 3] AS x MATCH (n:Node {id: x}) SET n.id = 1 / (x - 3)"
        )
    except:
        pass

    res = common.g.query("MATCH (n:Node) RETURN n.id")
    assert res.result_set == [[x] for x in range(1, 1001)]


def test_mvcc_attribute():
    res_write, res_read = mvcc(
        "UNWIND range(1, 1000) AS x CREATE (:Node {id: x})",
        run_write_attribute,
        run_read_attribute,
    )

    res = common.g.query("MATCH (n) RETURN n.id")
    assert res.result_set == [
        [0 if i < 101 and i % 2 == 0 else None if i < 101 and i % 2 == 1 else i]
        for i in range(1, 1001)
    ]

    assert res_write == [(x, x + 1) for x in range(1, 101)]

    for r in res_read:
        version = r[2]
        res = r[1]
        assert res == [
            [
                (
                    0
                    if i < version and i % 2 == 0
                    else None if i < version and i % 2 == 1 else i
                )
            ]
            for i in range(1, 1001)
        ]


def test_mvcc_label():
    res_write, res_read = mvcc(
        "UNWIND range(1, 1000) AS x CREATE (:Node {id: x})",
        run_write_label,
        run_read_label,
    )

    res = common.g.query("MATCH (n) RETURN labels(n)")
    assert res.result_set == [
        [
            (
                ["Node", "L"]
                if i < 101 and i % 2 == 0
                else [] if i < 101 and i % 2 == 1 else ["Node"]
            )
        ]
        for i in range(1, 1001)
    ]

    assert res_write == [(x, x + 1) for x in range(1, 101)]

    for r in res_read:
        version = r[2]
        res = r[1]
        assert res == [
            [
                (
                    ["Node", "L"]
                    if i < version and i % 2 == 0
                    else [] if i < version and i % 2 == 1 else ["Node"]
                )
            ]
            for i in range(1, 1001)
        ]


def test_mvcc_relationship():
    res_write, res_read = mvcc(
        "CREATE (a:Node {id: 0}), (b:Node {id: 1}) WITH a, b UNWIND range(1, 1000) AS x CREATE (a)-[:RELATES_TO {id: x}]->(b)",
        run_write_relationship,
        run_read_relationship,
    )

    res = common.g.query("MATCH (n)-[r]->(m) RETURN r.id")
    assert res.result_set == [
        [i if i < 101 and i % 2 == 0 else 0 if i < 101 and i % 2 == 1 else i]
        for i in range(1, 1001)
    ]

    assert res_write == [(x, x + 1) for x in range(1, 101)]

    for r in res_read:
        version = r[2]
        res = r[1]
        assert res == [
            [
                (
                    i
                    if i < version and i % 2 == 0
                    else 0 if i < version and i % 2 == 1 else i
                )
            ]
            for i in range(1, 1001)
            if version % 2 == 1 or i != version - 1
        ]

# def test_mvcc_index():
#     common.g.query("CREATE INDEX ON :Node(id)")

#     res_write, res_read = mvcc(
#         "UNWIND range(1, 1000) AS x CREATE (:Node {id: x})",
#         run_write_attribute,
#         run_read_indexed_attribute,
#     )

#     res = common.g.query("MATCH (n) RETURN n.id ORDER BY n.id")
#     assert res.result_set == [
#         [0 if i < 101 and i % 2 == 0 else None if i < 101 and i % 2 == 1 else i]
#         for i in range(1, 1001)
#     ]

#     assert res_write == [(x, x + 1) for x in range(1, 101)]

#     for r in res_read:
#         version = r[2]
#         res = r[1]
#         assert res == [
#             [
#                 (
#                     0
#                     if i < version and i % 2 == 0
#                     else None if i < version and i % 2 == 1 else i
#                 )
#             ]
#             for i in range(1, 1001)
#         ]