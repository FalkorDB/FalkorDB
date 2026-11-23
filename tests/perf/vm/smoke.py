"""The VM path's self-test: start both images, ask each for PING and a query.

    smoke.py --candidate <image@sha256:…> --baseline <image@sha256:…> \
             --out <dir> --mode <queue|nightly|long> \
             [--candidate-label …] [--baseline-label …] \
             [--candidate-commit …] [--baseline-commit …]

A stand-in check in the shape of contract v1 (tests/perf/README of the
swarm, START-HERE §1): it proves that a VM from `_perf-vm.yml` runs Docker,
pulls both images by digest, pins the servers to `PERF_SERVER_CPUS`, and that
the check's result.json reaches the artifact. It measures nothing that
gates a release, so it is not in `tests/perf/checks.yaml`; every `--mode`
does the same thing.

Each image is started twice, interleaved and with the order swapped
(candidate, baseline, baseline, candidate). A start counts as completed
when the server answers PING and `GRAPH.QUERY smoke "RETURN 1"` returns an
array within 60 s. Verdict: `pass` when every start completed, `regression`
when only the candidate failed, `error` when the baseline failed (nothing to
compare against).

Standard library only; run it with `python3 -I`.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import socket
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

DIGEST = re.compile(r"^[^@\s]+@sha256:[0-9a-f]{64}$")
ORDER = ("candidate", "baseline", "baseline", "candidate")
START_TIMEOUT_S = 60
PINGS = 50


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def default_label(image: str) -> str:
    """`<repo>@<first 12 hex of the digest>`, the contract's label when none is given."""
    name, _, digest = image.partition("@sha256:")
    # Drop a tag, but not a registry port (`host:5000/repo`).
    if name.rfind(":") > name.rfind("/"):
        name = name[: name.rfind(":")]
    return f"{name}@{digest[:12]}"


def completed_status(candidate: int, baseline: int) -> str:
    """Contract v1.4: broken on both sides is `info`, not a regression."""
    if candidate == baseline:
        return "ok" if candidate else "info"
    return "improvement" if candidate else "regression"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def command(sock: socket.socket, *words: str) -> bytes:
    """Send one RESP command and return the first line of the reply."""
    payload = f"*{len(words)}\r\n".encode()
    for word in words:
        data = word.encode()
        payload += b"$%d\r\n%s\r\n" % (len(data), data)
    sock.sendall(payload)
    reply = b""
    while b"\r\n" not in reply:
        chunk = sock.recv(4096)
        if not chunk:
            raise ConnectionError("server closed the connection")
        reply += chunk
    return reply.split(b"\r\n", 1)[0]


def machine() -> dict:
    path = os.environ.get("PERF_MACHINE_JSON")
    if path:
        return json.loads(Path(path).read_text())
    return {
        "runner": "local",
        "type": f"{platform.system()}-{platform.machine()}",
        "cpu": platform.processor() or platform.machine() or "unknown",
        "cores": os.cpu_count() or 1,
    }


def run_once(image: str, name: str, raw: Path) -> tuple[bool, float | None, str]:
    """Start one server, probe it, stop it. Returns (completed, ping max ms, note)."""
    port = free_port()
    docker = ["docker", "run", "-d", "--rm", "--name", name, "-p", f"127.0.0.1:{port}:6379"]
    if os.environ.get("PERF_SERVER_CPUS"):
        docker += ["--cpuset-cpus", os.environ["PERF_SERVER_CPUS"]]
    started = subprocess.run([*docker, image], capture_output=True, text=True)
    if started.returncode != 0:
        return False, None, f"docker run failed: {started.stderr.strip()[:200]}"
    try:
        deadline = time.monotonic() + START_TIMEOUT_S
        while True:
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=5) as sock:
                    if command(sock, "PING") != b"+PONG":
                        raise ConnectionError("no PONG yet")
                    timings = []
                    for _ in range(PINGS):
                        t0 = time.perf_counter()
                        if command(sock, "PING") != b"+PONG":
                            return False, None, "PING stopped answering PONG"
                        timings.append((time.perf_counter() - t0) * 1000)
                    first = command(sock, "GRAPH.QUERY", "smoke", "RETURN 1")
                    if not first.startswith(b"*"):
                        return False, None, f"GRAPH.QUERY replied {first[:120]!r}"
                    return True, max(timings), f"ping median {statistics.median(timings):.3f} ms"
            except (OSError, ConnectionError):
                if time.monotonic() > deadline:
                    return False, None, f"no PONG within {START_TIMEOUT_S} s"
                time.sleep(0.5)
    finally:
        logs = subprocess.run(["docker", "logs", name], capture_output=True, text=True)
        (raw / f"{name}.log").write_text(logs.stdout + logs.stderr)
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--mode", required=True, choices=["queue", "nightly", "long"])
    parser.add_argument("--candidate-label")
    parser.add_argument("--baseline-label")
    parser.add_argument("--candidate-commit")
    parser.add_argument("--baseline-commit")
    args = parser.parse_args(argv)

    images = {"candidate": args.candidate, "baseline": args.baseline}
    for side, image in images.items():
        if not DIGEST.match(image):
            parser.error(f"--{side} must be pinned by digest (name@sha256:<64 hex>): {image}")

    out = Path(args.out)
    raw = out / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    started = now()

    rows = []
    for i, side in enumerate(ORDER):
        image = images[side]
        pulled = subprocess.run(["docker", "pull", "-q", image], capture_output=True, text=True)
        if pulled.returncode != 0:
            ok, ping_ms, note = False, None, f"docker pull failed: {pulled.stderr.strip()[:200]}"
        else:
            ok, ping_ms, note = run_once(image, f"perf-smoke-{side}-{i}", raw)
        print(f"{side:9} run {i}: {'ok' if ok else 'FAILED'} ({note})", flush=True)
        rows.append({"side": side, "run": i, "completed": ok, "ping_max_ms": ping_ms, "note": note})

    with (raw / "smoke.csv").open("w") as csv:
        csv.write("side,run,completed,ping_max_ms,note\n")
        for row in rows:
            ping = "" if row["ping_max_ms"] is None else f"{row['ping_max_ms']:.3f}"
            note = row["note"].replace(",", ";").replace("\n", " ")
            csv.write(f"{row['side']},{row['run']},{int(row['completed'])},{ping},{note}\n")

    def side_rows(side: str) -> list[dict]:
        return [row for row in rows if row["side"] == side]

    completed = {side: int(all(r["completed"] for r in side_rows(side))) for side in images}
    pings = {
        side: max((r["ping_max_ms"] for r in side_rows(side) if r["ping_max_ms"] is not None), default=None)
        for side in images
    }

    if not completed["baseline"]:
        verdict, error = "error", "baseline server did not answer: " + "; ".join(
            r["note"] for r in side_rows("baseline") if not r["completed"]
        )
    elif not completed["candidate"]:
        verdict, error = "regression", None
    else:
        verdict, error = "pass", None

    findings = [
        {
            "name": "server answers PING and GRAPH.QUERY",
            "metric": "completed",
            "candidate": completed["candidate"],
            "baseline": completed["baseline"],
            "ratio": round(completed["candidate"] / completed["baseline"], 4) if completed["baseline"] else None,
            "threshold": None,
            "direction": "lower_is_worse",
            "status": completed_status(completed["candidate"], completed["baseline"]),
        }
    ]
    if pings["candidate"] is not None and pings["baseline"] is not None:
        findings.append(
            {
                "name": f"max of {PINGS} PINGs per start",
                "metric": "ping_max_ms",
                "candidate": round(pings["candidate"], 3),
                "baseline": round(pings["baseline"], 3),
                "ratio": round(pings["candidate"] / pings["baseline"], 4) if pings["baseline"] > 0 else None,
                "threshold": None,
                "direction": "higher_is_worse",
                "status": "info",
            }
        )

    summary = {
        "pass": "both servers started twice and answered PING and a query",
        "regression": "the candidate server did not answer; the baseline did",
        "error": "the baseline server did not answer, so nothing was compared",
    }[verdict]
    if verdict == "error" and not completed["candidate"]:
        summary = "neither server answered (broken on both sides), so nothing was compared"
    result = {
        "schema": 1,
        "check": "smoke",
        "verdict": verdict,
        "confirmed": verdict == "regression",
        "summary": summary,
        "candidate": {
            "image": args.candidate,
            "commit": args.candidate_commit or None,
            "label": args.candidate_label or default_label(args.candidate),
        },
        "baseline": {
            "image": args.baseline,
            "commit": args.baseline_commit or None,
            "label": args.baseline_label or default_label(args.baseline),
        },
        "machine": machine(),
        "started": started,
        "finished": now(),
        "findings": findings,
        "error": error,
        "artifacts": ["raw/smoke.csv"],
    }
    (out / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"verdict: {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
