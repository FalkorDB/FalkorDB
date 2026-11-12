import math
import time
import asyncio
from redis.asyncio import Redis as AsyncRedis

# helpers for tests that hold a graph's READ lock with long reads and check
# what happens to the server, or to a command, waiting behind them


def long_read_query(n):
    # ~n^2 iterations holding the graph READ lock, with bounded memory (a single
    # range(0, n^2) would materialize a huge array). the predicate MUST depend on
    # both loop vars, else the planner pushes it below the inner UNWIND and the
    # read finishes in ~ms holding no lock
    return (f"UNWIND range(0, {n}) AS a UNWIND range(0, {n}) AS b "
            f"WITH a, b WHERE a + b < 0 RETURN count(a)")


def calibrate_long_read(graph, target_sec, start_n=3000, max_n=40000):
    # size a long read so a single one takes ~target_sec on this machine: a
    # fixed size either takes minutes on a slow runner or finishes so fast that
    # a fixed latency bound passes vacuously. graph must exist.
    # returns (n, seconds one read took)
    def timed(n):
        t0 = time.monotonic()
        graph.ro_query(long_read_query(n))
        return time.monotonic() - t0

    dur = timed(start_n)
    n = int(start_n * math.sqrt(target_sec / max(dur, 0.01)))  # cost ~ n^2
    n = max(1000, min(n, max_n))
    return n, timed(n)


class ReadLockHolder():
    # keeps `concurrency` long reads in flight on an async graph, so its READ
    # lock stays held until stop(). keep `concurrency` well under the worker
    # pool size, so the commands under test aren't starved of a thread
    def __init__(self, graph, query, concurrency=3):
        self.graph       = graph
        self.query       = query
        self.concurrency = concurrency
        self.reads       = []
        self._stop       = False
        self._task       = None

    def start(self):
        self._task = asyncio.create_task(self._run())

    async def _run(self):
        while not self._stop:
            while (sum(1 for r in self.reads if not r.done()) < self.concurrency
                   and not self._stop):
                self.reads.append(asyncio.create_task(self.graph.ro_query(self.query)))
            await asyncio.sleep(0.01)

    # stop issuing new reads. the ones in flight keep holding the lock
    async def stop(self):
        self._stop = True
        if self._task is not None:
            await self._task

    # stop, wait for the reads in flight, and return the exceptions they raised.
    # the reads exist only to hold the lock: an exception means the setup broke
    # (or the server died), which would make a test's assertions vacuous
    async def drain(self):
        await self.stop()
        results = await asyncio.gather(*self.reads, return_exceptions=True)
        return [r for r in results if isinstance(r, Exception)]


class PingProbe():
    # samples PING latency on a dedicated connection at a fixed cadence, in its
    # own task, so a server freeze lands in a recorded sample (sampling inline
    # with other blocking calls can let a freeze stall those instead). a failed
    # PING (the server died) records inf, so a crash can't pass as responsive
    def __init__(self, port, interval=0.01):
        self.conn      = AsyncRedis(port=port, decode_responses=True)
        self.interval  = interval
        self.latencies = []
        self._stop     = False
        self._task     = None

    # connects first, so the connection setup isn't sampled
    async def start(self):
        await self.conn.ping()
        self._task = asyncio.create_task(self._run())

    async def _run(self):
        loop = asyncio.get_running_loop()
        while not self._stop:
            t0 = loop.time()
            try:
                await self.conn.ping()
                self.latencies.append(loop.time() - t0)
            except Exception:
                self.latencies.append(float('inf'))
            await asyncio.sleep(self.interval)

    # stop sampling, close the connection and return the WORST latency - a
    # freeze is a single event a median would hide
    async def stop(self):
        self._stop = True
        if self._task is not None:
            await self._task
        try:
            await self.conn.aclose()
        except Exception:
            pass
        return max(self.latencies, default=0.0)
