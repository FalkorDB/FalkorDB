import os
import platform
import subprocess
import time

from falkordb import FalkorDB
from redis import Redis

redis_server: subprocess.Popen = None
client = None
g = None
shutdown = False


def start_redis(release=None, moduleEnvs=[]):
    global redis_server, client, g, shutdown
    host = os.environ.get("FALKORDB_HOST", "localhost")
    port = os.environ.get("FALKORDB_PORT", os.environ.get("PORT", "6379"))
    # In CI's services-container mode an external redis is already running with
    # the module loaded; spawning locally would race the port. Fail loudly instead.
    existing_env = os.environ.get("EXISTING_ENV", "").lower() == "1"
    if release is None:
        release = True if os.environ.get("RELEASE", "").lower() == "1" else False
    default_target = "target/debug/libfalkordb.so"
    if platform.system() == "Darwin":
        default_target = default_target.replace(".so", ".dylib")
    if release:
        default_target = default_target.replace("debug", "release")
    target = os.environ.get("TARGET", default_target)
    r = Redis(host=host, port=port)
    try:
        r.ping()
        client = FalkorDB(host=host, port=port)
        g = client.select_graph("test")
        return
    except Exception as e:
        if existing_env:
            raise RuntimeError(
                f"EXISTING_ENV=1 but cannot reach redis at {host}:{port}: {e}"
            ) from e
        shutdown = True
        if os.path.exists("redis-test.log"):
            os.remove("redis-test.log")
        redis_server = subprocess.Popen(
            ["/usr/local/bin/redis-server",
             "--save", "", "--port", port, "--logfile", "redis-test.log",
             "--loadmodule", target] + moduleEnvs,
            stdout=subprocess.PIPE)
    while True:
        try:
            r.ping()
            client = FalkorDB(host=host, port=port)
            g = client.select_graph("test")
            return
        except Exception:
            pass

def falkordb():
    """Construct a FalkorDB client honoring FALKORDB_HOST / FALKORDB_PORT.

    Bare `FalkorDB()` defaults to localhost:6379, which silently bypasses
    the docker-services CI mode (where redis runs as a sibling container).
    Use this helper instead so EXISTING_ENV tests connect to the right place.
    """
    return FalkorDB(
        host=os.environ.get("FALKORDB_HOST", "localhost"),
        port=int(os.environ.get("FALKORDB_PORT", os.environ.get("PORT", "6379"))),
    )


def shutdown_redis():
    if shutdown:
        client.connection.shutdown(nosave=True)
        redis_server.wait()

def wait_for_indices_to_sync(graph):
    q = "CALL db.indexes() YIELD status WHERE status <> 'OPERATIONAL' RETURN count(1)"
    while True:
        result = graph.ro_query(q)
        if result.result_set[0][0] == 0:
            break
        time.sleep(0.5) # sleep 500ms