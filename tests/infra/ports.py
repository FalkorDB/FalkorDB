"""Free TCP port allocation that survives xdist parallelism.

RLTest handed out ports from a single in-process counter, which is safe only
because one RLTest process owned every server. Under `pytest -n`, each worker
is a separate process, so a plain "bind :0, close, reuse" allocator races:
two workers can be handed the same port between the probe and the eventual
redis-server bind.

Two mechanisms together make that unlikely enough to be reliable:

  1. Each xdist worker draws from a disjoint band of the ephemeral range,
     derived from its worker id (gw0, gw1, ...). Workers therefore never
     contend with each other at all.
  2. Within a band the port is probed by actually binding it, and the socket
     is held open until the caller has spawned its server (`reserve`), so a
     second allocation in the same worker can't hand out the same number.
"""

import os
import socket

# Ephemeral-ish range chosen to sit above redis's default 6379 and below the
# usual Linux ephemeral floor (32768), so we neither collide with a
# developer's own redis nor with the kernel's outbound port choices.
_BASE = 16000
_BAND = 400
_MAX = 32000


def _worker_index():
    """0 for a serial run, N for xdist worker 'gwN'."""
    wid = os.environ.get("PYTEST_XDIST_WORKER", "")
    if wid.startswith("gw") and wid[2:].isdigit():
        return int(wid[2:])
    return 0


class PortReservation:
    """A port plus the socket holding it, released once the server is up.

    Callers must `release()` after spawning, otherwise the server cannot bind
    the port itself. Usable as a context manager, which is the safe default.
    """

    def __init__(self, port, sock):
        self.port = port
        self._sock = sock

    def release(self):
        if self._sock is not None:
            self._sock.close()
            self._sock = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()


def reserve_port():
    """Reserve a free port within this worker's band.

    The returned reservation keeps the port bound; release it immediately
    before spawning the process that should own it.
    """
    start = _BASE + _worker_index() * _BAND
    if start >= _MAX:
        # More workers than bands. Fall back to kernel-assigned ports; the
        # hold-until-spawn behaviour still prevents intra-worker collisions.
        start = 0
    for offset in range(_BAND if start else 1):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Deliberately NOT SO_REUSEADDR: we want the bind to fail when
        # anything else holds the port, including a socket in TIME_WAIT,
        # because redis-server would fail the same way.
        try:
            s.bind(("127.0.0.1", start + offset if start else 0))
        except OSError:
            s.close()
            continue
        return PortReservation(s.getsockname()[1], s)
    raise RuntimeError(
        f"no free port in band [{start}, {start + _BAND}) for worker "
        f"{os.environ.get('PYTEST_XDIST_WORKER', 'serial')}"
    )
