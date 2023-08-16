import os
from disposableredis import DisposableRedis


REDIS_MODULE_PATH_ENVVAR = 'REDIS_MODULE_PATH'
REDIS_PATH_ENVVAR = 'REDIS_PATH'
REDIS_PORT_ENVVAR = 'REDIS_PORT'


def execute_query(graph, query_desc, query):
    print(query_desc)
    print("query: {query}".format(query=query))
    print("execution plan:\n{plan}".format(plan=graph.execution_plan(query)))
    query_res = graph.query(query)
    query_res.pretty_print()
    print("\n")


def _redis():
    module_path = os.getenv(REDIS_MODULE_PATH_ENVVAR)
    redis_path = os.getenv(REDIS_PATH_ENVVAR)
    fixed_port = os.getenv(REDIS_PORT_ENVVAR)

    if module_path is None:
        print("Undeclared environment variable {}".format(REDIS_MODULE_PATH_ENVVAR))
        print("run: export {}=../../src/falkordb.so".format(REDIS_MODULE_PATH_ENVVAR))
        return None

    if redis_path is None:
        print("Undeclared environment variable {}".format(REDIS_PATH_ENVVAR))
        print("run: export {}=<path_to_redis-server>".format(REDIS_PATH_ENVVAR))
        return None

    _module_path = os.path.abspath(os.path.join(os.getcwd(), module_path))

    port = None
    if fixed_port is not None:
        port = fixed_port

    print("port=%s, path=%s, loadmodule=%s" % (port, redis_path, _module_path))
    dr = DisposableRedis(port=port, path=redis_path, loadmodule=_module_path)
    return dr
