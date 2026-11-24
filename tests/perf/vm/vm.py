"""Helpers for `.github/workflows/_perf-vm.yml`.

`env`
    Run on the VM before the check. Splits the VM's physical cores between
    the server and the client, writes the `machine` object a check puts in
    its result.json, and prints the `PERF_*` variables (contract v1.3) as
    `NAME=value` lines for `$GITHUB_ENV`.

`record`
    Writes `perf-vm.json`: where the check ran, or why no VM came up.
    `merge.py` quotes it when the artifact holds no result.json.

Standard library only: it runs on a bare Debian VM and on `ubuntu-latest`.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

METADATA = "http://metadata.google.internal/computeMetadata/v1/instance/"


# --- cores ------------------------------------------------------------------


def parse_cpu_list(text: str) -> list[int]:
    """Parse a kernel CPU list such as `0-3,8,10-11`."""
    cpus: list[int] = []
    for part in text.strip().split(","):
        if not part:
            continue
        lo, _, hi = part.partition("-")
        cpus.extend(range(int(lo), int(hi or lo) + 1))
    return cpus


def format_cpu_list(cpus: list[int]) -> str:
    """Format CPUs as the list `taskset -c` and `docker --cpuset-cpus` take."""
    out: list[str] = []
    ordered = sorted(cpus)
    i = 0
    while i < len(ordered):
        j = i
        while j + 1 < len(ordered) and ordered[j + 1] == ordered[j] + 1:
            j += 1
        out.append(str(ordered[i]) if i == j else f"{ordered[i]}-{ordered[j]}")
        i = j + 1
    return ",".join(out)


def read_cores(sysfs: Path) -> list[list[int]]:
    """Group the online logical CPUs by physical core, in CPU order."""
    online = parse_cpu_list((sysfs / "online").read_text())
    cores: dict[tuple[str, str], list[int]] = {}
    for cpu in online:
        topology = sysfs / f"cpu{cpu}" / "topology"
        key = (
            (topology / "physical_package_id").read_text().strip(),
            (topology / "core_id").read_text().strip(),
        )
        cores.setdefault(key, []).append(cpu)
    return sorted(cores.values(), key=min)


def split_cores(cores: list[list[int]]) -> tuple[list[int], list[int], str | None]:
    """Give the server the first half of the physical cores, the client the rest.

    Hyperthread siblings stay together, so the two sides never share a core.
    With a single core the sides share it, and the third value says so.
    """
    if len(cores) >= 2:
        half = (len(cores) + 1) // 2
        server = [cpu for core in cores[:half] for cpu in core]
        client = [cpu for core in cores[half:] for cpu in core]
        return server, client, None
    threads = cores[0] if cores else [0]
    if len(threads) >= 2:
        return threads[:1], threads[1:], "one physical core: server and client share it"
    return threads, threads, "one CPU: server and client are not separated"


# --- machine ----------------------------------------------------------------


def metadata(key: str) -> str:
    request = urllib.request.Request(METADATA + key, headers={"Metadata-Flavor": "Google"})
    with urllib.request.urlopen(request, timeout=5) as response:
        return response.read().decode().strip()


def cpu_model(cpuinfo: Path) -> str:
    for line in cpuinfo.read_text().splitlines():
        name, _, value = line.partition(":")
        if name.strip() == "model name" and value.strip():
            return value.strip()
    return "unknown"


def cmd_env(args: argparse.Namespace) -> int:
    cores = read_cores(Path(args.sysfs))
    server, client, warning = split_cores(cores)
    if warning:
        print(f"::warning::cpusets: {warning}", file=sys.stderr)

    if args.runner == "gce":
        # `zone` and `machine-type` come back as full resource paths.
        zone = metadata("zone").rsplit("/", 1)[-1]
        machine_type = metadata("machine-type").rsplit("/", 1)[-1]
    else:
        zone, machine_type = args.zone, args.machine_type
    if not machine_type:
        print("machine type unknown: pass --machine-type off GCE", file=sys.stderr)
        return 2

    machine = {
        "runner": args.runner,
        "type": machine_type,
        "cpu": cpu_model(Path(args.cpuinfo)),
        "cores": sum(len(core) for core in cores),
    }
    machine_json = Path(args.machine_json).resolve()
    machine_json.write_text(json.dumps(machine, allow_nan=False) + "\n")

    print(f"PERF_SERVER_CPUS={format_cpu_list(server)}")
    print(f"PERF_CLIENT_CPUS={format_cpu_list(client)}")
    print(f"PERF_ZONE={zone}")
    print(f"PERF_MACHINE_TYPE={machine_type}")
    print(f"PERF_MACHINE_JSON={machine_json}")
    return 0


# --- perf-vm.json -----------------------------------------------------------


def optional_int(text: str) -> int | None:
    return int(text) if text.strip() else None


def cmd_record(args: argparse.Namespace) -> int:
    record = {
        "schema": 1,
        "check": args.check,
        "vm_name": args.vm_name or None,
        "zone": args.zone or None,
        "machine_type": args.machine_type or None,
        "fallback_used": args.fallback_used == "true",
        "create_error": args.create_error or None,
        "command_exit": optional_int(args.command_exit),
        "timed_out": args.timed_out == "true",
        # The vPMU type GCE reports for the VM, and what `perf stat` counted
        # on it (or why it could not): both null without hardware counters.
        "pmu": args.pmu or None,
        "pmu_probe": args.pmu_probe or None,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print(out.read_text(), end="")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    env = sub.add_parser("env", help="print the PERF_* variables for the check")
    env.add_argument("--machine-json", required=True, help="where to write the machine object")
    env.add_argument("--runner", default="gce", choices=["gce", "hosted", "local"])
    env.add_argument("--zone", default="", help="off GCE only")
    env.add_argument("--machine-type", default="", help="off GCE only")
    env.add_argument("--sysfs", default="/sys/devices/system/cpu")
    env.add_argument("--cpuinfo", default="/proc/cpuinfo")
    env.set_defaults(func=cmd_env)

    record = sub.add_parser("record", help="write perf-vm.json")
    record.add_argument("--out", required=True)
    record.add_argument("--check", required=True)
    record.add_argument("--vm-name", default="")
    record.add_argument("--zone", default="")
    record.add_argument("--machine-type", default="")
    record.add_argument("--fallback-used", default="false", help="true or false")
    record.add_argument("--create-error", default="")
    record.add_argument("--command-exit", default="", help="empty when the command never ran")
    record.add_argument("--timed-out", default="false", help="true or false")
    record.add_argument("--pmu", default="", help="vPMU type, empty without one")
    record.add_argument("--pmu-probe", default="", help="what perf stat counted, or why it could not")
    record.set_defaults(func=cmd_record)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
