"""A stand-in for `docker`, so test_kit.py drives kit's docker calls with no Docker.

Adapted from the size probe's fake (perf/size-probe 7d7cefff5). It implements
what kit.py calls: pull, image inspect, run (detached or foreground), port,
inspect, logs, rm and ps. Each "container" is a small RESP server that
answers PING with +PONG (or -LOADING for a while) and anything else with +OK.
Every call is appended to $FAKE_DOCKER_STATE/calls.jsonl.

Environment:
  FAKE_DOCKER_STATE         directory: one <name>.json per container, calls.jsonl
  FAKE_DOCKER_PULL_FAILS    the first N pulls fail
  FAKE_DOCKER_REFUSE        container-name substring that `docker run` refuses
  FAKE_DOCKER_CRASH         image substring: the container exits at once (139)
  FAKE_DOCKER_OOM           image substring: the container is OOM-killed at once
  FAKE_DOCKER_NO_PORT       `docker port` prints nothing
  FAKE_DOCKER_LOADING_S     the server answers -LOADING for this many seconds
  FAKE_DOCKER_INSPECT_DOWN  `docker inspect` cannot reach the daemon
  FAKE_DOCKER_ENV           JSON list `docker image inspect` reports as Config.Env
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import socket
import socketserver
import subprocess
import sys
import time
from pathlib import Path

NEVER = "\0"  # no image or name contains it
# Flags that take no value (kit spells every flag out long; extras in the tests use these).
NO_VALUE = {"--detach", "--rm", "--privileged", "--all", "--quiet", "--force", "--volumes"}


def state() -> Path:
    return Path(os.environ["FAKE_DOCKER_STATE"])


def knob(name: str) -> str:
    return os.environ.get(f"FAKE_DOCKER_{name}", NEVER)


def record(name: str) -> Path:
    return state() / f"{name}.json"


def alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def parse(args: list[str]) -> tuple[dict[str, list[str]], list[str]]:
    """Options (each name -> its values) and the positional words, docker-style."""
    options: dict[str, list[str]] = {}
    words, i = [], 0
    while i < len(args):
        arg = args[i]
        if words or not arg.startswith("-"):
            words.append(arg)
        elif "=" in arg:
            key, _, value = arg.partition("=")
            options.setdefault(key, []).append(value)
        elif arg in NO_VALUE:
            options.setdefault(arg, []).append("")
        else:
            options.setdefault(arg, []).append(args[i + 1])
            i += 1
        i += 1
    return options, words


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def run(args: list[str]) -> int:
    options, words = parse(args)
    name, image, command = options["--name"][0], words[0], words[1:]
    if knob("REFUSE") in name:
        print("docker: Error response from daemon: Conflict.", file=sys.stderr)
        return 125
    labels = dict(v.partition("=")[::2] for v in options.get("--label", []))
    published = "--publish" in options
    port = free_port() if published else int(command[command.index("--port") + 1])
    entry = {"image": image, "labels": labels, "port": port, "published": published, "pid": None}
    entry["exit"], entry["oom"] = 0, knob("OOM") in image
    if knob("CRASH") in image or entry["oom"]:
        entry["exit"] = 137 if entry["oom"] else 139
        if "--rm" not in options:  # with --rm the exited container is gone at once
            record(name).write_text(json.dumps(entry))
        print("f" * 64)
        return 0 if "--detach" in options else entry["exit"]
    loading = os.environ.get("FAKE_DOCKER_LOADING_S", "0")
    if "--detach" not in options:  # foreground: this process is the container
        record(name).write_text(json.dumps({**entry, "pid": os.getpid()}))
        serve(port, float(loading))
        return 0
    argv = [sys.executable, "-I", __file__, "serve", str(port), loading]
    # No inherited pipe: `docker run --detach` returns at once, and so must this.
    null = subprocess.DEVNULL
    server = subprocess.Popen(argv, start_new_session=True, stdin=null, stdout=null, stderr=null)
    record(name).write_text(json.dumps({**entry, "pid": server.pid}))
    print("f" * 64)
    return 0


def no_such(kind: str, name: str) -> int:
    print(f"Error response from daemon: No such {kind}: {name}", file=sys.stderr)
    return 1


def main(argv: list[str]) -> int:
    if argv[0] == "serve":
        serve(int(argv[1]), float(argv[2]))
        return 0
    with (state() / "calls.jsonl").open("a") as calls:
        calls.write(json.dumps(argv) + "\n")
    cmd, (options, words) = argv[0], parse(argv[1:])
    if cmd == "pull":
        count = state() / "pulls"
        done = int(count.read_text()) if count.exists() else 0
        count.write_text(str(done + 1))
        if done < int(os.environ.get("FAKE_DOCKER_PULL_FAILS", "0")):
            print("Error response from daemon: toomanyrequests", file=sys.stderr)
            return 1
        return 0
    if cmd == "image":
        default = '["PATH=/usr/bin", "FALKORDB_ARGS=TIMEOUT 1000", "FALKORDB_SERVER=valkey"]'
        print(os.environ.get("FAKE_DOCKER_ENV", default))
        return 0
    if cmd == "run":
        return run(argv[1:])
    if cmd == "ps":
        key, _, value = options["--filter"][0].removeprefix("label=").partition("=")
        for path in sorted(state().glob("*.json")):
            if json.loads(path.read_text())["labels"].get(key) == value:
                print(path.stem)
        return 0
    if cmd == "rm":
        code = 0
        for name in words:
            if not record(name).exists():
                code = no_such("container", name)
                continue
            with contextlib.suppress(ProcessLookupError, TypeError):
                os.kill(json.loads(record(name).read_text())["pid"], signal.SIGKILL)
            record(name).unlink()
        return code
    if cmd == "inspect" and knob("INSPECT_DOWN") != NEVER:
        print("Cannot connect to the Docker daemon. Is the docker daemon running?", file=sys.stderr)
        return 1
    name = words[0]
    if not record(name).exists():
        return no_such("object" if cmd == "inspect" else "container", name)
    entry = json.loads(record(name).read_text())
    running = alive(entry["pid"])
    if cmd == "port":
        if not running:  # docker clears the port map when a container stops
            print(f"Error: No public port '{words[1]}' published for {name}", file=sys.stderr)
            return 1
        if knob("NO_PORT") == NEVER and entry["published"]:
            print(f"127.0.0.1:{entry['port']}")
        return 0
    if cmd == "inspect":
        if "{{.State.Pid}}" in options["--format"][0]:
            print(entry["pid"] if running else 0)
        else:
            status = "running 0" if running else f"exited {entry['exit'] or 137}"
            print(f"{status} {str(entry['oom']).lower()}")
        return 0
    if cmd == "logs":
        lines = [(sys.stdout, "fake server log"), (sys.stderr, "fake warning")]
        lines += [(sys.stdout, "Ready to accept connections")]
        if "--tail" in options:
            lines = lines[max(0, len(lines) - int(options["--tail"][0])) :]
        for stream, line in lines:
            print(line, file=stream, flush=True)
        return 0
    print(f"fake docker: no {cmd}", file=sys.stderr)
    return 2


def serve(port: int, loading_s: float) -> None:
    ready_at = time.monotonic() + loading_s

    class Handler(socketserver.StreamRequestHandler):
        def handle(self) -> None:
            with contextlib.suppress(OSError, ValueError):
                while head := self.rfile.readline():
                    words = []
                    for _ in range(int(head[1:])):
                        size = int(self.rfile.readline()[1:])
                        words.append(self.rfile.read(size + 2)[:-2].decode())
                    if words[0].upper() != "PING":
                        self.wfile.write(b"+OK\r\n")
                    elif time.monotonic() < ready_at:
                        self.wfile.write(b"-LOADING the dataset is loading\r\n")
                    else:
                        self.wfile.write(b"+PONG\r\n")

    socketserver.ThreadingTCPServer.allow_reuse_address = True
    socketserver.ThreadingTCPServer.daemon_threads = True
    with socketserver.ThreadingTCPServer(("127.0.0.1", port), Handler) as server:
        server.serve_forever()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
