"""What the nightly perf checks share, and nothing else. Standard library only.

The contract's flags and result.json plumbing, the threshold test, and the
docker calls that start, probe and remove a server from an image. Each check
keeps its own measuring and judging. API and an example: bench/README.md,
"Shared check kit".
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import platform
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
from collections.abc import Callable
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC
from datetime import datetime
from pathlib import Path
from typing import Any

from falkorbench.contract import COMMIT_RE
from falkorbench.contract import FINDING_STATUSES
from falkorbench.contract import IMAGE_RE
from falkorbench.contract import MACHINE_FIELDS
from falkorbench.contract import MAX_FINDINGS

MAX_INFORMATIVE = 10  # info findings kept beside the crossings (contract §2)
MODES = ("queue", "nightly", "long")
SIDES = ("candidate", "baseline")
SUMMARY_LIMIT = 500

# --- the contract's flags -----------------------------------------------------


def add_args(parser: argparse.ArgumentParser) -> None:
    """Add --candidate --baseline --out --mode and the optional labels and commits."""
    for side in SIDES:
        parser.add_argument(f"--{side}", required=True, help="image, name[:tag]@sha256:<64 hex>")
    parser.add_argument("--out", required=True, help="directory for result.json and raw/")
    parser.add_argument("--mode", required=True, choices=MODES, help="the run's length")
    for side in SIDES:
        parser.add_argument(f"--{side}-label", help="result.json label (default <repo>@<12 hex>)")
        parser.add_argument(f"--{side}-commit", help="40-hex commit the image was built from")


def default_label(image: str) -> str:
    """`<repo>@<first 12 hex of the digest>` (contract v1.2)."""
    name, _, digest = image.partition("@")
    repo = name.rsplit(":", 1)[0] if ":" in name.rsplit("/", 1)[-1] else name
    hexpart = digest.removeprefix("sha256:")[:12]
    return f"{repo}@{hexpart}" if hexpart else repo


def side(args: Any, which: str) -> dict:
    """result.json's `candidate` or `baseline`: a bad commit is null, a blank label the default."""
    image = getattr(args, which)
    commit = getattr(args, f"{which}_commit", None)
    label = (getattr(args, f"{which}_label", None) or "").strip()
    return {
        "image": image,
        "commit": commit if commit and COMMIT_RE.fullmatch(commit) else None,
        "label": label or default_label(image),
    }


def input_errors(args: Any) -> list[str]:
    """Why these flags cannot give a valid result: an image not pinned by digest,
    a commit that is not 40 hex, a label that is not one line, an unknown mode."""
    errors = []
    for which in SIDES:
        image = getattr(args, which)
        commit = getattr(args, f"{which}_commit", None)
        label = getattr(args, f"{which}_label", None)
        if not isinstance(image, str) or not IMAGE_RE.fullmatch(image):
            errors.append(f"--{which} {image!r} is not name[:tag]@sha256:<64 hex>, a digest")
        if commit and not COMMIT_RE.fullmatch(commit):
            errors.append(f"--{which}-commit {commit!r} is not 40 lowercase hex")
        if label is not None and (not label.strip() or len(label.splitlines()) > 1):
            errors.append(f"--{which}-label {label!r} is not one non-blank line")
    if getattr(args, "mode", None) not in MODES:
        errors.append(f"--mode {getattr(args, 'mode', None)!r} is not one of {', '.join(MODES)}")
    return errors


def out_dir(out: str | Path) -> Path:
    """`--out` made absolute (the workflow passes `--out out`, contract v1.8), with raw/ in it."""
    path = Path(out).resolve()
    (path / "raw").mkdir(parents=True, exist_ok=True)
    return path


def utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def machine() -> dict:
    """$PERF_MACHINE_JSON (the VM workflow's object, copied), else this host as `local`."""
    path = os.environ.get("PERF_MACHINE_JSON")
    if path:
        try:
            value = json.loads(Path(path).read_text())
            if _is_machine(value):
                return {k: value[k] for k in MACHINE_FIELDS}  # nothing else: no stray NaN
            raise ValueError("not a machine object (runner, type, cpu as text; cores >= 1)")
        except (OSError, ValueError) as exc:
            _say(f"PERF_MACHINE_JSON {path}: {exc}; describing this host")
    cpu = ""
    try:
        if sys.platform == "darwin":
            argv = ["sysctl", "-n", "machdep.cpu.brand_string"]
            cpu = subprocess.run(argv, capture_output=True, text=True, timeout=10).stdout.strip()
        else:
            lines = Path("/proc/cpuinfo").read_text().splitlines()
            models = [ln.split(":", 1)[1] for ln in lines if ln.startswith("model name")]
            cpu = models[0].strip() if models else ""
    except (OSError, subprocess.SubprocessError):
        pass  # no model string here: platform's answer below
    return {
        "runner": "local",
        "type": f"{platform.system().lower()}-{platform.machine()}",
        "cpu": cpu or platform.processor() or platform.machine() or "unknown",
        "cores": os.cpu_count() or 1,
    }


def _is_machine(value: Any) -> bool:
    if not isinstance(value, dict) or not all(k in value for k in MACHINE_FIELDS):
        return False
    texts = [value[k] for k in MACHINE_FIELDS if k != "cores"]
    cores = value["cores"]
    text_ok = all(isinstance(t, str) and t.strip() for t in texts)
    return text_ok and type(cores) is int and cores >= 1


# --- findings, verdict, result.json -------------------------------------------


def _number(value: Any) -> int | float | None:
    if value is None:
        return None
    if isinstance(value, bool | int):
        return int(value)
    value = float(value)
    if not math.isfinite(value):
        return None
    return round(value, 6) if abs(value) < 1e6 else round(value, 1)


def finding(
    name: str,
    metric: str,
    candidate: float | None,
    baseline: float | None,
    status: str,
    *,
    direction: str = "higher_is_worse",
    threshold: float | None = None,
    ratio: float | None = None,
) -> dict:
    """One finding. `ratio` defaults to candidate / baseline; none for a 0/1 metric
    or a zero baseline. NaN and infinity become null (JSON has neither)."""
    candidate, baseline = _number(candidate), _number(baseline)
    if ratio is None and candidate is not None and baseline:
        ratio = candidate / baseline
    if metric in ("completed", "invariant"):
        ratio = None
    return {
        "name": name,
        "metric": metric,
        "candidate": candidate,
        "baseline": baseline,
        "ratio": _number(ratio),
        "threshold": _number(threshold),
        "direction": direction,
        "status": status,
    }


def correctness_status(candidate: int | None, baseline: int | None) -> str:
    """A 0/1 metric (completed, invariant; 1 held), contract v1.4: broken on the candidate
    only is a regression, on both info; a side not measured is info, never ok."""
    if candidate == 0:
        return "info" if baseline == 0 else "regression"
    if candidate is None or baseline is None:
        return "info"
    return "improvement" if baseline == 0 else "ok"


def crossed(
    candidate: float | None,
    baseline: float | None,
    threshold: float,
    floor: float,
    direction: str = "higher_is_worse",
) -> int:
    """+1 worse, -1 better, 0 neither. Crossing needs both, strictly: the worse value over
    the better one > threshold, and their difference > floor (required: with a zero
    baseline the ratio is infinite, so the floor alone decides). A missing, negative or
    non-finite value never crosses (finding() writes it as null)."""
    if direction not in ("higher_is_worse", "lower_is_worse"):
        raise ValueError(f"crossed() compares magnitudes, not direction {direction!r}")
    values = (candidate, baseline)
    if any(v is None or not math.isfinite(v) or v < 0 for v in values):
        return 0
    worse, better = candidate, baseline
    if direction == "lower_is_worse":
        worse, better = baseline, candidate
    if worse > threshold * better and worse - better > floor:
        return 1
    if better > threshold * worse and better - worse > floor:
        return -1
    return 0


def select(findings: Iterable[dict]) -> list[dict]:
    """What result.json carries: regressions, then improvements, then at most 10 info
    rows, 50 in all; each group in the order given (sort it worst first). Other
    `ok` rows are left for raw/rows.csv; a status the contract does not know raises."""
    rows = list(findings)
    unknown = sorted({str(f["status"]) for f in rows} - set(FINDING_STATUSES))
    if unknown:
        raise ValueError(f"finding status {', '.join(unknown)} is not one of {FINDING_STATUSES}")
    groups = [[f for f in rows if f["status"] == s] for s in ("regression", "improvement", "info")]
    return (groups[0] + groups[1] + groups[2][:MAX_INFORMATIVE])[:MAX_FINDINGS]


def verdict(findings: Iterable[dict], errors: Iterable[str]) -> str:
    """regression > error > pass (contract v1.4): a regression outranks an error."""
    if any(f["status"] == "regression" for f in findings):
        return "regression"
    return "error" if any(_one_line(e) for e in errors if e) else "pass"


def _one_line(text: str, limit: int | None = None) -> str:
    text = " ".join(str(text).split())
    return text if limit is None or len(text) <= limit else text[: limit - 1] + "…"


def result(
    check: str,
    args: Any,
    *,
    started: str,
    findings: Iterable[dict],
    summary: str,
    errors: Iterable[str] = (),
    artifacts: Iterable[str | Path] = (),
    machine_info: dict | None = None,
) -> dict:
    """A schema-1 result. The flags' own input_errors() count as errors; `findings`
    go through select(); the verdict comes from verdict() on what is kept."""
    kept = select(findings)
    given = [_one_line(e) for e in errors if e]
    problems = list(dict.fromkeys([*input_errors(args), *(e for e in given if e)]))
    outcome = verdict(kept, problems)
    return {
        "schema": 1,
        "check": check,
        "verdict": outcome,
        "confirmed": outcome == "regression",
        "summary": _one_line(summary, SUMMARY_LIMIT) or outcome,
        "candidate": side(args, "candidate"),
        "baseline": side(args, "baseline"),
        "machine": machine_info if machine_info is not None else machine(),
        "started": started,
        "finished": max(utc_now(), started),
        "findings": kept,
        "error": _one_line("; ".join(problems)) if outcome == "error" else None,
        "artifacts": sorted({str(a) for a in artifacts}),
    }


def error_result(check: str, args: Any, started: str, error: str) -> dict:
    """The smallest valid result that says error (a blank `error` still says error)."""
    error = _one_line(error) or "unknown error"
    summary = f"could not measure: {error}"
    return result(check, args, started=started, findings=[], summary=summary, errors=[error])


def write_result(out: Path, value: dict) -> None:
    """<out>/result.json, atomically: a temp file renamed over it. Serialised first,
    so a value JSON cannot hold (NaN) raises and leaves the old file in place."""
    text = json.dumps(value, indent=2, allow_nan=False) + "\n"
    temp = out / "result.json.tmp"
    temp.write_text(text)
    temp.replace(out / "result.json")


# --- always a result.json -----------------------------------------------------

STOP_SIGNALS = (signal.SIGTERM, signal.SIGINT)
_stopped: str | None = None  # why run_check() is stopping, once a signal arrived


class Stopped(BaseException):
    """SIGTERM (the workflow's timeout) or SIGINT arrived. A BaseException, so no
    `except Exception` on the way up can swallow it."""


def stopping() -> bool:
    """True once run_check() has caught a stop signal: skip graceful stops."""
    return _stopped is not None


def _ignore_stop_signals() -> None:
    for signum in STOP_SIGNALS:
        signal.signal(signum, signal.SIG_IGN)


def _on_stop_signal(signum: int, _frame: object) -> None:
    global _stopped
    _stopped = f"stopped by signal {signum} ({signal.Signals(signum).name}); partial data in raw/"
    _ignore_stop_signals()  # once: `timeout --kill-after` leaves the rest for the write
    raise Stopped(_stopped)


def _say(text: str) -> None:
    with contextlib.suppress(OSError, ValueError):  # a broken or closed stderr must not stop it
        print(text, end="" if text.endswith("\n") or not text else "\n", file=sys.stderr)


def run_check(
    check: str,
    args: Any,
    work: Callable[[], None],
    build: Callable[[str | None], dict],
    cleanup: Callable[[], None] | None = None,
) -> dict:
    """Run `work()` and write <out>/result.json whatever happens; return the result.

    First an error result ("did not finish") is written, so a SIGKILL leaves one.
    During work() the first SIGTERM or SIGINT raises Stopped. Stopped or any other
    exception (SystemExit included) becomes the error passed to build(error), which
    returns the result from what was measured so far; a build() that raises, or
    says pass despite the error, gives error_result(). From the first signal on,
    stop signals are ignored, so the check's own teardown, the write and cleanup()
    finish; then the previous handlers come back. `args.out` is made absolute.
    Handlers are installed only on the main thread.
    """
    global _stopped
    _stopped = None
    out = out_dir(args.out)
    args.out = str(out)
    started = utc_now()
    on_main = threading.current_thread() is threading.main_thread()
    previous = {s: signal.getsignal(s) for s in STOP_SIGNALS} if on_main else {}
    error, trace = None, ""
    try:
        try:
            try:
                for signum in previous:
                    signal.signal(signum, _on_stop_signal)
                write_result(out, error_result(check, args, started, "the check did not finish"))
                work()
            except Stopped:
                pass  # _stopped says why
            except (Exception, SystemExit, KeyboardInterrupt) as exc:  # still a result.json
                error = f"the check crashed: {type(exc).__name__}: {exc}"
                trace = traceback.format_exc()
            if on_main:
                _ignore_stop_signals()
        except Stopped:
            pass  # it landed after work(); the handler ignores the rest
        # Also when the check swallowed Stopped (a bare except): the run was still cut.
        error = _stopped or error
        _say(trace)
        try:
            value = build(error)
            if not isinstance(value, dict):
                raise TypeError(f"build() returned {type(value).__name__}, not a result")
            if error and value.get("verdict") == "pass":  # build() dropped the error
                value = error_result(check, args, started, error)
        except (Exception, SystemExit) as exc:  # a sys.exit in build() too
            _say(traceback.format_exc())
            why = f"building the result failed: {type(exc).__name__}: {exc}"
            value = error_result(check, args, started, f"{error}; {why}" if error else why)
        try:
            write_result(out, value)
        except (TypeError, ValueError, OSError) as exc:
            why = f"writing the result failed: {type(exc).__name__}: {exc}"
            value = error_result(check, args, started, f"{error}; {why}" if error else why)
            write_result(out, value)
        return value
    finally:
        if cleanup is not None:
            try:
                cleanup()
            except (Exception, SystemExit, KeyboardInterrupt) as exc:  # result is written
                _say(f"cleanup failed: {type(exc).__name__}: {exc}")
        for signum, handler in previous.items():
            signal.signal(signum, signal.SIG_DFL if handler is None else handler)


# --- docker -------------------------------------------------------------------


class DockerError(RuntimeError):
    """docker failed or did not answer; the message carries its stderr."""


def _docker(*argv: str, timeout: float = 120.0, merge: bool = False) -> subprocess.CompletedProcess:
    """One docker CLI call; `merge` puts stderr into stdout, in order."""
    err = subprocess.STDOUT if merge else subprocess.PIPE
    try:
        return subprocess.run(
            ["docker", *argv],
            stdout=subprocess.PIPE,
            stderr=err,
            text=True,
            errors="replace",
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        raise DockerError("docker is not on PATH") from exc
    except OSError as exc:  # it could not be started (EAGAIN, ENOMEM, EACCES)
        raise DockerError(f"docker could not be started: {exc}") from exc
    except subprocess.TimeoutExpired as exc:
        raise DockerError(f"`docker {argv[0]}` did not return within {timeout:.0f} s") from exc


def _ok(proc: subprocess.CompletedProcess, what: str) -> str:
    if proc.returncode != 0:
        tail = " | ".join((proc.stderr or proc.stdout or "").strip().splitlines()[-3:])
        raise DockerError(f"{what} failed (exit {proc.returncode}): {tail}")
    return proc.stdout


def pull(image: str, attempts: int = 3, delay_s: float = 15.0) -> None:
    """`docker pull`, retried after delay_s, 2 x delay_s, ... (registries fail transiently).
    A pull that hangs past 15 min raises at once: retrying a hung registry burns the run."""
    for attempt in range(1, max(attempts, 1) + 1):
        proc = _docker("pull", "--quiet", image, timeout=900)
        if proc.returncode == 0:
            return
        if attempt < attempts:
            time.sleep(delay_s * attempt)
    _ok(proc, f"docker pull {image} ({attempts} attempts)")


def image_env(image: str) -> dict[str, str]:
    """The image's ENV (e.g. FALKORDB_ARGS, FALKORDB_SERVER)."""
    found = _docker("image", "inspect", "--format", "{{json .Config.Env}}", image)
    text = _ok(found, f"docker image inspect {image}")
    return dict(item.partition("=")[::2] for item in json.loads(text or "null") or [])


@dataclass
class Container:
    """A started server: its host `port` (0 if it died at start), and for a foreground
    one the `docker run` process."""

    name: str
    port: int
    proc: subprocess.Popen | None = None


def start(
    image: str,
    name: str,
    *,
    env: dict[str, str] | None = None,
    volumes: dict[str | Path, str] | None = None,
    labels: dict[str, str] | None = None,
    cpus: str | None = None,
    entrypoint: str | None = None,
    command: Iterable[str] = (),
    extra: Iterable[str] = (),
    port: int = 6379,
    host_network: bool = False,
    foreground: bool = False,
    output: Path | None = None,
) -> Container:
    """`docker run` a server. By default detached, publishing `port` as
    `-p 127.0.0.1::<port>` and reading the host port back with `docker port`
    (docker picks a free one: no race, never 6379). A server that exits before its
    port is read comes back with port 0, its container kept for state() and logs().
    host_network: no publishing; pass the `port` the server's command listens on
    (never 6379, and nothing may answer there yet). foreground (host_network only):
    `docker run` stays a child process (Container.proc), its output to `output`.
    volumes map a host path to `<path>[:ro]`; `extra` goes before the image,
    `command` after it."""
    if foreground and not host_network:
        raise ValueError("a foreground server needs host_network: its port is known up front")
    if host_network and port == 6379:
        raise ValueError("on the host network the server needs its own port: never 6379")
    if host_network and _listening(port):
        raise DockerError(f"port {port} already has a server (a stale run?); remove it first")
    remove(name)  # a container left by an earlier run under the same name
    argv = ["run", "--name", name, *([] if foreground else ["--detach"])]
    argv += [x for k, v in (labels or {}).items() for x in ("--label", f"{k}={v}")]
    argv += [x for k, v in (env or {}).items() for x in ("--env", f"{k}={v}")]
    for source, target in (volumes or {}).items():
        argv += ["--volume", f"{Path(source).resolve()}:{target}"]
    argv += ["--cpuset-cpus", cpus] if cpus else []
    argv += ["--network", "host"] if host_network else ["--publish", f"127.0.0.1::{port}"]
    argv += ["--entrypoint", entrypoint] if entrypoint else []
    argv += [*extra, image, *command]
    container = Container(name, port if host_network else 0)
    try:
        if foreground:
            sink = output.open("ab") if output else subprocess.DEVNULL
            try:
                container.proc = subprocess.Popen(["docker", *argv], stdout=sink, stderr=sink)
            except OSError as exc:
                raise DockerError(f"docker could not be started: {exc}") from exc
            finally:
                if output:
                    sink.close()
            return container
        _ok(_docker(*argv, timeout=300), f"docker run {image}")
        if not host_network:
            text = _ok(_docker("port", name, f"{port}/tcp", timeout=60), f"docker port {name}")
            ports = [ln.rpartition(":")[2] for ln in text.split() if ln.rpartition(":")[0]]
            if not ports or not ports[0].isdigit():
                raise DockerError(f"no host port for {name}:{port} in {text!r}")
            container.port = int(ports[0])
        return container
    except (Exception, Stopped) as exc:  # the container may exist, and nobody holds it yet
        died = False
        try:  # a Stopped landing in here still removes the container
            died = isinstance(exc, DockerError) and _ended(container, gone=False)
        finally:
            if not died:
                _remove_quietly(name)
        if died:
            return container  # it died at start: state() and logs() say why
        raise


def _listening(port: int) -> bool:
    try:
        socket.create_connection(("127.0.0.1", port), timeout=1.0).close()
    except OSError:
        return False
    return True


def _remove_quietly(name: str) -> None:
    with contextlib.suppress(DockerError):  # never in place of the error being raised
        remove(name)


def _ended(container: Container, gone: bool = True, timeout: float = 60.0) -> bool:
    """Whether the server has stopped: its `docker run` exited (foreground), or its
    container exited, died or (`gone`) was removed, as with --rm. Unknown is False.
    A foreground `docker run` that docker itself refused (exit 125) raises DockerError:
    no server ran, so it is not a server that stopped."""
    if container.proc is not None:
        code = container.proc.poll()
        if code == 125:
            raise DockerError(f"docker run {container.name} failed (exit 125): no server started")
        return code is not None
    ends = ("exited", "dead", *(["gone"] if gone else []))
    try:
        return state(container.name, timeout).startswith(ends)
    except DockerError:
        return False  # inspect failed: says nothing about the server


def ping(port: int, timeout: float = 5.0) -> bool:
    """One PING to 127.0.0.1:port, never re-sent, all within `timeout`. False on
    anything but +PONG (refused, timeout, LOADING, a reply that never ends)."""
    deadline = time.monotonic() + timeout
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=timeout) as sock:
            sock.sendall(b"*1\r\n$4\r\nPING\r\n")
            reply = b""
            while not reply.endswith(b"\r\n") and len(reply) < 64:
                left = deadline - time.monotonic()
                if left <= 0:
                    return False
                sock.settimeout(left)
                chunk = sock.recv(64)
                if not chunk:
                    return False
                reply += chunk
            return reply == b"+PONG\r\n"
    except OSError:
        return False


def wait_ping(port: int, timeout_s: float = 120.0, container: Container | None = None) -> bool:
    """PING every 0.5 s until +PONG (True) or timeout_s (False). With `container`, a
    server that stopped ends the wait at once; a foreground `docker run` that docker
    refused raises DockerError. Every step, `docker inspect` included, keeps to timeout_s."""
    deadline = time.monotonic() + timeout_s
    while True:
        if ping(port, timeout=max(0.1, min(5.0, deadline - time.monotonic()))):
            return True
        left = deadline - time.monotonic()
        if container is not None and left > 0 and _ended(container, timeout=max(0.5, left)):
            return False
        if time.monotonic() >= deadline:
            return False
        time.sleep(max(0.0, min(0.5, deadline - time.monotonic())))


def logs(name: str, tail: int | None = None) -> str:
    """The container's output, stdout and stderr interleaved as written."""
    tail_args = [] if tail is None else ["--tail", str(tail)]
    proc = _docker("logs", *tail_args, name, timeout=60, merge=True)
    return _ok(proc, f"docker logs {name}")


def state(name: str, timeout: float = 60.0) -> str:
    """`running`, how it ended (`exited 139`, `exited 137 (OOM-killed)`), or `gone`.
    A docker that does not answer raises DockerError: it says nothing about the server."""
    fmt = "{{.State.Status}} {{.State.ExitCode}} {{.State.OOMKilled}}"
    proc = _docker("inspect", "--type", "container", "--format", fmt, name, timeout=timeout)
    if proc.returncode != 0 and _no_such_container(proc):
        return "gone"
    status, code, oom = [*_ok(proc, f"docker inspect {name}").split(), "", "", ""][:3]
    if status == "running":
        return "running"
    return f"{status} {code}" + (" (OOM-killed)" if oom == "true" else "")


def _no_such_container(proc: subprocess.CompletedProcess) -> bool:
    text = (proc.stderr or "") + (proc.stdout or "")
    return any(m in text.lower() for m in ("no such container", "no such object"))


def host_pid(name: str) -> int:
    """The server's PID on the host (for psutil), 0 when it is not running or gone."""
    fmt = "{{.State.Pid}}"
    proc = _docker("inspect", "--type", "container", "--format", fmt, name, timeout=60)
    if proc.returncode != 0 and _no_such_container(proc):
        return 0
    return int(_ok(proc, f"docker inspect {name}").strip() or 0)


def remove(*names: str) -> None:
    """`docker rm --force --volumes`; one already gone counts as removed."""
    if not names:
        return
    proc = _docker("rm", "--force", "--volumes", *names, timeout=300)
    said = [ln for ln in proc.stderr.splitlines() if ln.strip()]
    only_gone = bool(said) and all("no such container" in ln.lower() for ln in said)
    if proc.returncode != 0 and not only_gone:  # a silent failure is a failure too
        _ok(proc, f"docker rm {' '.join(names)}")


def remove_labelled(label: str) -> None:
    """Remove every container carrying `label` (`key=value`): what a cut run left."""
    found = _docker("ps", "--all", "--quiet", "--filter", f"label={label}", timeout=60)
    remove(*_ok(found, "docker ps").split())
