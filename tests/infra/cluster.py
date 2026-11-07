"""Redis Cluster formation, shared by the local and docker backends.

Pure redis-py — no redis-cli dependency, which the CI job/RC images don't
carry.
"""

import socket
import time

from redis.backoff import NoBackoff
from redis.retry import Retry

import redis

TOTAL_SLOTS = 16384


def bootstrap_cluster(shard_addrs, attempts=100, interval=0.2):
    """Form a cluster from `shard_addrs` (list of (host, port)).

    Three phases:
      1. CLUSTER ADDSLOTSRANGE — split the slot space into N contiguous
         ranges and assign one per shard.
      2. CLUSTER MEET — from shard 0, meet every other shard. Bus gossip
         propagates the full topology to the rest.
      3. Poll CLUSTER INFO on every shard until cluster_state=ok and
         cluster_known_nodes equals the shard count.
    """
    n = len(shard_addrs)
    if n == 0:
        return
    clients = [
        redis.Redis(host=h, port=p, socket_connect_timeout=1,
                    retry=Retry(NoBackoff(), 0))
        for h, p in shard_addrs
    ]
    for i, c in enumerate(clients):
        # Integer slot split: shard i owns [i*total/n, (i+1)*total/n - 1].
        # The last shard absorbs the rounding remainder so the union covers
        # every slot without gaps or overlap.
        start = (i * TOTAL_SLOTS) // n
        end = ((i + 1) * TOTAL_SLOTS) // n - 1
        c.execute_command("CLUSTER", "ADDSLOTSRANGE", start, end)

    head = clients[0]
    for (h, p) in shard_addrs[1:]:
        # CLUSTER MEET rejects hostnames ("Invalid node address"), so resolve
        # first. Gossip still propagates any announced hostname once paired.
        ip = socket.gethostbyname(h)
        head.execute_command("CLUSTER", "MEET", ip, p)

    for c, (h, p) in zip(clients, shard_addrs):
        last_exc = None
        ok = False
        for _ in range(attempts):
            try:
                info = c.execute_command("CLUSTER", "INFO")
                if isinstance(info, bytes):
                    info = info.decode()
                fields = dict(
                    line.split(":", 1)
                    for line in info.splitlines() if ":" in line
                )
                if (fields.get("cluster_state", "").strip() == "ok"
                        and int(fields.get("cluster_known_nodes", "0")) == n):
                    ok = True
                    break
            except Exception as e:
                last_exc = e
            time.sleep(interval)
        if not ok:
            msg = (f"cluster shard {h}:{p} never reached cluster_state=ok "
                   f"with {n} known nodes in {attempts * interval:.1f}s")
            if last_exc is not None:
                msg += f"; last error: {last_exc!r}"
            raise RuntimeError(msg)
