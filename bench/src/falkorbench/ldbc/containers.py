"""Running a published falkordb-server image as a throwaway server.

The nightly check compares two *images*, not two module builds, so each side is
the image's own server (its redis or valkey, its allocator, its entrypoint) in a
container of its own. Only three things are set from outside, and each works on
every falkordb-server image back to the C v4.22 line:

* the host port: `-p 127.0.0.1::6379` lets docker pick a free one, so the check
  never takes 6379 and two checks on one host do not collide;
* `FALKORDB_ARGS=IMPORT_FOLDER /import/`: the image's entrypoint appends it to
  `--loadmodule`, and the dataset is bind-mounted read-only at `/import` (the
  trailing slash matters: the C module concatenates the folder and the file
  name without a separator);
* `--cpuset-cpus`, when the VM workflow names the server CPUs
  (`PERF_SERVER_CPUS`).

`BROWSER=0` keeps the images that bundle the browser from starting it.
"""

from __future__ import annotations

import contextlib
import subprocess
from dataclasses import dataclass
from pathlib import Path

#: The port the server listens on inside every falkordb-server image.
_INNER_PORT = "6379/tcp"


class DockerError(RuntimeError):
    """A docker command failed; the message carries its stderr."""


def _docker(*args: str, timeout: float = 120.0) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ["docker", *args], capture_output=True, text=True, timeout=timeout, check=False
        )
    except FileNotFoundError as e:
        raise DockerError("docker is not installed or not on PATH") from e
    except subprocess.TimeoutExpired as e:
        raise DockerError(f"`docker {args[0]}` did not return within {timeout:.0f}s") from e


def _check(proc: subprocess.CompletedProcess[str], what: str) -> str:
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-3:]
        raise DockerError(f"{what} failed (exit {proc.returncode}): {' | '.join(tail)}")
    return proc.stdout


def pull(image: str) -> None:
    """Pull `image` (a digest reference), so a start never waits on the registry."""
    _check(_docker("pull", "--quiet", image, timeout=900.0), f"docker pull {image}")


@dataclass
class Container:
    """One running server. `port` is the host port docker mapped to 6379."""

    name: str
    image: str
    port: int

    def state(self) -> str:
        """`running`, or how it ended (`exited 139`, `exited 137 (OOM-killed)`)."""
        proc = _docker(
            "inspect",
            "--format",
            "{{.State.Status}} {{.State.ExitCode}} {{.State.OOMKilled}}",
            self.name,
        )
        if proc.returncode != 0:
            return "gone"
        status, code, oom = [*proc.stdout.split(), "", "", ""][:3]
        if status == "running":
            return "running"
        return f"{status} {code}" + (" (OOM-killed)" if oom == "true" else "")

    def logs(self) -> str:
        proc = _docker("logs", "--timestamps", self.name, timeout=60.0)
        return proc.stdout + proc.stderr

    def remove(self) -> None:
        remove_by_name(self.name)


def remove_by_name(name: str) -> None:
    """`docker rm --force`; one that is already gone counts as removed."""
    proc = _docker("rm", "--force", "--volumes", name, timeout=60.0)
    if proc.returncode != 0 and "no such container" not in (proc.stderr or "").lower():
        _check(proc, f"docker rm {name}")


def start(
    image: str,
    name: str,
    import_dir: Path,
    *,
    cpus: str | None = None,
) -> Container:
    """Start `image` as `name`, with `import_dir` as the module's IMPORT_FOLDER."""
    _docker("rm", "--force", "--volumes", name, timeout=60.0)
    argv = [
        "run",
        "--detach",
        "--name",
        name,
        "--publish",
        f"127.0.0.1::{_INNER_PORT}",
        "--env",
        "BROWSER=0",
        "--env",
        "FALKORDB_ARGS=IMPORT_FOLDER /import/",
        "--volume",
        f"{import_dir.resolve()}:/import:ro",
    ]
    if cpus:
        argv += ["--cpuset-cpus", cpus]
    argv.append(image)
    _check(_docker(*argv), f"docker run {image}")
    container = Container(name=name, image=image, port=0)
    try:
        container.port = parse_port(_check(_docker("port", name, _INNER_PORT), "docker port"))
    except BaseException:
        # Including an interrupt: the container exists and nobody else owns
        # it yet. The original exception is the one that propagates.
        # A failure here is reported by the caller's teardown, which removes
        # the container by name again.
        with contextlib.suppress(DockerError):
            container.remove()
        raise
    return container


def parse_port(text: str) -> int:
    """The host port from `docker port` output (`127.0.0.1:32771`, one per line)."""
    for line in text.splitlines():
        host, _, port = line.strip().rpartition(":")
        if host and port.isdigit():
            return int(port)
    raise ValueError(f"no host port in `docker port` output: {text!r}")
