"""A stand-in for `docker`, so test_probe.py can drive `bench probe` end to end.

It implements just enough of `pull`, `run`, `rm`, `inspect`, `logs` and
`stats`. Each "container" is a small RESP server, started on the published
port, that answers the probe's queries.

Environment:
  FAKE_DOCKER_STATE   directory holding one `<name>.pid` per container
  FAKE_DOCKER_SLOW    image substring: its UNWIND shapes sleep 2 us per node,
                      a cost that grows with the graph
  FAKE_DOCKER_HANG    image substring whose queries containing
                      FAKE_DOCKER_HANG_QUERY sleep 5 s
  FAKE_DOCKER_REFUSE  container-name substring that `docker run` refuses
  FAKE_DOCKER_CLOSE_CONFIG  image substring whose server closes the
                      connection on CONFIG SET
  FAKE_DOCKER_SHORT   image substring whose UNWIND shapes create one
                      relationship fewer than asked (a rows mismatch)

Every measured query sleeps 2 ms, so a ratio of two of them is stable on a
busy machine.
"""

from __future__ import annotations

import contextlib
import os
import re
import signal
import socketserver
import subprocess
import sys
import time
from pathlib import Path

NEVER = "\0"  # no image or name contains it


def state() -> Path:
    return Path(os.environ["FAKE_DOCKER_STATE"])


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def run(args: list[str]) -> int:
    name = args[args.index("--name") + 1]
    port = args[args.index("-p") + 1].split(":")[1]
    image = args[-1]
    if os.environ.get("FAKE_DOCKER_REFUSE", NEVER) in name:
        print("docker: Error response from daemon: Conflict.", file=sys.stderr)
        return 125
    slow = int(os.environ.get("FAKE_DOCKER_SLOW", NEVER) in image)
    close_config = int(os.environ.get("FAKE_DOCKER_CLOSE_CONFIG", NEVER) in image)
    short = int(os.environ.get("FAKE_DOCKER_SHORT", NEVER) in image)
    hang = os.environ.get("FAKE_DOCKER_HANG_QUERY", "")
    if os.environ.get("FAKE_DOCKER_HANG", NEVER) not in image:
        hang = ""  # argv cannot carry NEVER
    server = subprocess.Popen(
        [
            sys.executable,
            "-I",
            __file__,
            "serve",
            port,
            str(slow),
            hang,
            str(close_config),
            str(short),
        ],
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    (state() / f"{name}.pid").write_text(str(server.pid))
    print("f" * 64)
    return 0


def main(argv: list[str]) -> int:
    cmd, args = argv[0], argv[1:]
    record = state() / f"{args[-1]}.pid" if args else None
    if cmd == "pull":
        return 0
    if cmd == "run":
        return run(args)
    if cmd == "serve":
        serve(int(args[0]), args[1] == "1", args[2], args[3] == "1", args[4] == "1")
        return 0
    if cmd in ("rm", "inspect", "logs") and not record.exists():
        print(f"Error: No such container: {args[-1]}", file=sys.stderr)
        return 1
    if cmd == "rm":
        with contextlib.suppress(ProcessLookupError):
            os.kill(int(record.read_text()), signal.SIGKILL)
        record.unlink()
        return 0
    if cmd == "inspect":
        print("running exit 0" if alive(int(record.read_text())) else "exited exit 137")
        return 0
    if cmd == "logs":
        print("fake server log")
        return 0
    if cmd == "stats":
        print("CONTAINER ID   NAME   CPU %")
        return 0
    print(f"fake docker: no {cmd}", file=sys.stderr)
    return 2


# --- the server --------------------------------------------------------------


def encode(value) -> bytes:
    if isinstance(value, list):
        return b"*%d\r\n" % len(value) + b"".join(encode(v) for v in value)
    if isinstance(value, int):
        return b":%d\r\n" % value
    data = str(value).encode()
    return b"$%d\r\n%s\r\n" % (len(data), data)


def size_of(graph: str) -> int:
    m = re.fullmatch(r"probe_(\d+)(k?)", graph)
    return int(m.group(1)) * (1000 if m.group(2) else 1)


def answer(words: list[str], slow: bool, hang: str, short: bool = False) -> bytes:
    cmd = words[0].upper()
    if cmd == "PING":
        return b"+PONG\r\n"
    if cmd == "CONFIG":
        return b"+OK\r\n"
    if cmd == "GRAPH.EXPLAIN":
        return encode(["Results", "    Fake Plan"])
    graph, query = words[1], words[2]
    stats = ["Query internal execution time: 0.010000 milliseconds"]
    if hang and hang in query:
        time.sleep(5)
    if query.startswith("CYPHER"):
        time.sleep(0.002)
    if "db.indexes" in query:
        return encode([["label", "status"], [["N", "OPERATIONAL"]], stats])
    if "count(" in query:
        return encode([["c"], [[size_of(graph)]], stats])
    if query.startswith("CYPHER rows="):
        if slow:
            time.sleep(size_of(graph) * 2e-6)
        n = query.count("{s:") - int(short)
        return encode([[f"Relationships created: {n}", f"Relationships deleted: {n}", *stats]])
    if query.startswith("CYPHER id="):
        node = int(query.split()[1].removeprefix("id="))
        return encode([["n.id"], [[node]], stats])
    if query.startswith("CYPHER v="):
        return encode([["Nodes created: 1", *stats]])
    return encode([stats])


def serve(port: int, slow: bool, hang: str, close_config: bool, short: bool) -> None:
    class Handler(socketserver.StreamRequestHandler):
        def handle(self) -> None:
            with contextlib.suppress(OSError, ValueError):
                while True:
                    head = self.rfile.readline()
                    if not head:
                        return
                    words = []
                    for _ in range(int(head[1:])):
                        n = int(self.rfile.readline()[1:])
                        words.append(self.rfile.read(n + 2)[:-2].decode())
                    if close_config and words[0].upper() == "CONFIG":
                        return  # drop the connection: PING answered, CONFIG SET not
                    self.wfile.write(answer(words, slow, hang, short))

    socketserver.ThreadingTCPServer.allow_reuse_address = True
    socketserver.ThreadingTCPServer.daemon_threads = True
    with socketserver.ThreadingTCPServer(("127.0.0.1", port), Handler) as server:
        server.serve_forever()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
