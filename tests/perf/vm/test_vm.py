"""Tests for the _perf-vm.yml helpers. Run: python3 -I -m unittest discover -s tests/perf/vm"""

from __future__ import annotations

import io
import json
import os
import re
import subprocess
import sys
import tempfile
import textwrap
import unittest
from contextlib import redirect_stdout
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import smoke  # noqa: E402
import vm  # noqa: E402


def fake_sysfs(root: Path, siblings: list[list[int]]) -> Path:
    """Lay out /sys/devices/system/cpu for the given physical cores."""
    cpus = sorted(cpu for core in siblings for cpu in core)
    root.mkdir(parents=True, exist_ok=True)
    (root / "online").write_text(vm.format_cpu_list(cpus) + "\n")
    for core_id, core in enumerate(siblings):
        for cpu in core:
            topology = root / f"cpu{cpu}" / "topology"
            topology.mkdir(parents=True)
            (topology / "physical_package_id").write_text("0\n")
            (topology / "core_id").write_text(f"{core_id}\n")
    return root


class CpuListTest(unittest.TestCase):
    def test_round_trip(self):
        for text in ["0", "0-3", "0-3,8-11", "0,2,4", "1-2,5,7-9"]:
            self.assertEqual(vm.format_cpu_list(vm.parse_cpu_list(text)), text)

    def test_parse_trailing_newline(self):
        self.assertEqual(vm.parse_cpu_list("0-2\n"), [0, 1, 2])


class SplitCoresTest(unittest.TestCase):
    def split(self, siblings):
        with tempfile.TemporaryDirectory() as tmp:
            cores = vm.read_cores(fake_sysfs(Path(tmp), siblings))
        return vm.split_cores(cores)

    def test_gce_layout_siblings_n_apart(self):
        # n4-standard-16: 8 cores, CPU n and n+8 share one.
        server, client, warning = self.split([[i, i + 8] for i in range(8)])
        self.assertEqual(vm.format_cpu_list(server), "0-3,8-11")
        self.assertEqual(vm.format_cpu_list(client), "4-7,12-15")
        self.assertIsNone(warning)

    def test_adjacent_siblings(self):
        server, client, _ = self.split([[2 * i, 2 * i + 1] for i in range(8)])
        self.assertEqual(vm.format_cpu_list(server), "0-7")
        self.assertEqual(vm.format_cpu_list(client), "8-15")

    def test_no_core_is_shared(self):
        siblings = [[i, i + 8] for i in range(8)]
        server, client, _ = self.split(siblings)
        for core in siblings:
            self.assertFalse(set(core) & set(server) and set(core) & set(client), core)

    def test_odd_core_count_gives_server_the_extra_core(self):
        server, client, _ = self.split([[0], [1], [2]])
        self.assertEqual((server, client), ([0, 1], [2]))

    def test_one_core_shares_and_warns(self):
        server, client, warning = self.split([[0, 1]])
        self.assertEqual((server, client), ([0], [1]))
        self.assertIn("share", warning)

    def test_one_cpu_warns(self):
        server, client, warning = self.split([[0]])
        self.assertEqual((server, client), ([0], [0]))
        self.assertIsNotNone(warning)


class EnvCommandTest(unittest.TestCase):
    def test_local_env_writes_machine_and_variables(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            sysfs = fake_sysfs(tmp / "cpu", [[i, i + 2] for i in range(2)])
            cpuinfo = tmp / "cpuinfo"
            cpuinfo.write_text("processor\t: 0\nmodel name\t: Test CPU @ 2.0GHz\n")
            out = io.StringIO()
            with redirect_stdout(out):
                rc = vm.main([
                    "env", "--runner", "local", "--zone", "z", "--machine-type", "t",
                    "--machine-json", str(tmp / "machine.json"),
                    "--sysfs", str(sysfs), "--cpuinfo", str(cpuinfo),
                ])
            self.assertEqual(rc, 0)
            variables = dict(line.split("=", 1) for line in out.getvalue().splitlines())
            self.assertEqual(variables["PERF_SERVER_CPUS"], "0,2")
            self.assertEqual(variables["PERF_CLIENT_CPUS"], "1,3")
            self.assertEqual(variables["PERF_ZONE"], "z")
            self.assertEqual(variables["PERF_MACHINE_TYPE"], "t")
            machine = json.loads(Path(variables["PERF_MACHINE_JSON"]).read_text())
            self.assertEqual(machine, {"runner": "local", "type": "t", "cpu": "Test CPU @ 2.0GHz", "cores": 4})


class RecordCommandTest(unittest.TestCase):
    def record(self, *extra):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "perf-vm.json"
            with redirect_stdout(io.StringIO()):
                vm.main(["record", "--out", str(path), "--check", "smoke", *extra])
            return json.loads(path.read_text())

    def test_no_vm(self):
        got = self.record("--create-error", "no zone had n4-standard-16")
        self.assertEqual(got["create_error"], "no zone had n4-standard-16")
        self.assertIsNone(got["zone"])
        self.assertIsNone(got["command_exit"])
        self.assertFalse(got["fallback_used"])
        self.assertEqual((got["pmu"], got["pmu_probe"]), (None, None))

    def test_ran(self):
        got = self.record("--vm-name", "gce-gh-1", "--zone", "us-central1-b", "--machine-type",
                          "n2-standard-16", "--fallback-used", "true", "--command-exit", "0",
                          "--timed-out", "false")
        self.assertEqual(got["zone"], "us-central1-b")
        self.assertTrue(got["fallback_used"])
        self.assertEqual(got["command_exit"], 0)
        self.assertIsNone(got["create_error"])

    def test_pmu(self):
        got = self.record("--pmu", "STANDARD", "--pmu-probe", "cycles=1203456 instructions=834567")
        self.assertEqual((got["pmu"], got["pmu_probe"]), ("STANDARD", "cycles=1203456 instructions=834567"))


class PmuProbeTest(unittest.TestCase):
    """vm.py pmu-probe: when `perf stat` output means the counters work."""

    GOOD = "661511,,cycles,295712,100.00,,\n592778,,instructions,295712,100.00,0.90,insn per cycle\n"

    def probe(self, text):
        proc = subprocess.run([sys.executable, "-I", str(HERE / "vm.py"), "pmu-probe"], input=text,
                              capture_output=True, text=True)
        return proc.returncode, proc.stdout.strip()

    def test_both_counted(self):
        self.assertEqual(self.probe(self.GOOD), (0, "cycles=661511 instructions=592778"))
        self.assertEqual(self.probe(self.GOOD.replace(",cycles,", ",cycles:u,"))[0], 0)

    def test_anything_else_fails_closed(self):
        bad = {
            "not supported": "<not supported>,,cycles,0,100.00,,\n<not supported>,,instructions,0,100.00,,\n",
            "not counted": "<not counted>,,cycles,0,0.00,,\n592778,,instructions,295712,100.00,,\n",
            "zero": "0,,cycles,295712,100.00,,\n592778,,instructions,295712,100.00,,\n",
            "one missing": "661511,,cycles,295712,100.00,,\n",
            "fell back to cpu-clock": "0.31,msec,cpu-clock,312000,100.00,0.5,CPUs utilized\n",
            "hybrid name": "661511,,cpu_core/cycles/,1,100.00,,\n592778,,cpu_core/instructions/,1,100.00,,\n",
            "duplicate": self.GOOD + "1,,cycles,1,100.00,,\n",
            "nothing": "",
        }
        for name, text in bad.items():
            rc, out = self.probe(text)
            self.assertEqual(rc, 1, name)
            self.assertTrue(out, name)


class RespReplyTest(unittest.TestCase):
    """smoke.read_reply takes a whole reply or none."""

    def read(self, data):
        return smoke.read_reply(io.BytesIO(data))

    def test_complete_replies(self):
        self.assertEqual(self.read(b"+PONG\r\n"), b"PONG")
        self.assertEqual(self.read(b"*2\r\n$3\r\nabc\r\n*1\r\n:1\r\n"), [b"abc", [b"1"]])
        self.assertEqual(self.read(b"%1\r\n+k\r\n:2\r\n"), [b"k", b"2"])
        self.assertIsNone(self.read(b"$-1\r\n"))

    def test_cut_short(self):
        for data in (b"*3\r\n", b"*2\r\n:1\r\n", b"$5\r\nab", b"+PON"):
            with self.assertRaises(ConnectionError, msg=data):
                self.read(data)

    def test_error_reply(self):
        with self.assertRaises(smoke.ReplyError):
            self.read(b"-ERR unknown command\r\n")


class SmokeLabelTest(unittest.TestCase):
    DIGEST = "sha256:" + "ab" * 32

    def test_default_label(self):
        self.assertEqual(smoke.default_label(f"falkordb/falkordb-server@{self.DIGEST}"),
                         "falkordb/falkordb-server@abababababab")
        self.assertEqual(smoke.default_label(f"ghcr.io/falkordb/falkordb-server:rc-x@{self.DIGEST}"),
                         "ghcr.io/falkordb/falkordb-server@abababababab")
        self.assertEqual(smoke.default_label(f"localhost:5000/repo@{self.DIGEST}"),
                         "localhost:5000/repo@abababababab")


# A `docker` stand-in: `run` starts a tiny RESP server on the published port,
# `pull` fails for any image whose name contains "dead".
FAKE_DOCKER = r'''#!/usr/bin/env python3
import os, signal, subprocess, sys
SERVER = r"""
import os, socket, sys, time
mode = os.environ.get("FAKE_SERVER_MODE", "")
s = socket.socket()
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(("127.0.0.1", int(sys.argv[1])))
s.listen()
while True:
    c, _ = s.accept()
    while True:
        data = c.recv(4096)
        if not data:
            break
        if mode == "slow":
            time.sleep(0.1)
        if b"GRAPH.QUERY" in data and mode == "truncate":
            c.sendall(b"*3\r\n")  # an array header whose elements never come
            c.close()
            break
        c.sendall(b"*1\r\n:1\r\n" if b"GRAPH.QUERY" in data else b"+PONG\r\n")
"""
state = os.environ["FAKE_DOCKER_STATE"]
args = sys.argv[1:]
with open(os.path.join(state, "calls"), "a") as f:
    f.write(" ".join(args) + "\n")
if args[0] == "pull":
    sys.exit(1 if "dead" in args[-1] else 0)
if args[0] == "run":
    name = args[args.index("--name") + 1]
    port = args[args.index("-p") + 1].split(":")[1]
    proc = subprocess.Popen([sys.executable, "-c", SERVER, port], start_new_session=True,
                            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    with open(os.path.join(state, name), "w") as f:
        f.write(str(proc.pid))
    print("container-id")
if args[0] == "rm":
    pidfile = os.path.join(state, args[-1])
    if os.path.exists(pidfile):
        os.kill(int(open(pidfile).read()), signal.SIGTERM)
        os.remove(pidfile)
'''

VALIDATOR = Path(os.environ.get("PERF_VALIDATE", HERE.parent / "validate.py"))


class SmokeEndToEndTest(unittest.TestCase):
    """smoke.py against a fake docker; result.json checked by the contract validator."""

    GOOD = "example/falkordb-server@sha256:" + "1" * 64
    DEAD = "example/dead-server@sha256:" + "2" * 64

    def run_smoke(self, candidate, baseline, *extra, env_extra=None, relative_out=False):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            docker = tmp / "bin" / "docker"
            docker.parent.mkdir()
            docker.write_text(FAKE_DOCKER)
            docker.chmod(0o755)
            env = dict(os.environ, PATH=f"{docker.parent}:{os.environ['PATH']}", FAKE_DOCKER_STATE=str(tmp))
            env.pop("PERF_MACHINE_JSON", None)
            env.update(env_extra or {})
            out = tmp / "out"
            proc = subprocess.run(
                [sys.executable, "-I", str(HERE / "smoke.py"), "--candidate", candidate,
                 "--baseline", baseline, "--out", "out" if relative_out else str(out), "--mode", "nightly", *extra],
                capture_output=True, text=True, env=env, timeout=120, cwd=tmp,
            )
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            result_path = out / "result.json"
            if VALIDATOR.exists():
                check = subprocess.run([sys.executable, "-I", str(VALIDATOR), str(result_path)],
                                       capture_output=True, text=True)
                self.assertEqual(check.returncode, 0, check.stdout + check.stderr)
            calls = (tmp / "calls").read_text().splitlines()
            return json.loads(result_path.read_text()), calls, sorted(p.name for p in (out / "raw").iterdir())

    def test_pass_interleaves_and_pins(self):
        machine = {"runner": "gce", "type": "n4-standard-16", "cpu": "x", "cores": 16}
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(machine, f)
        try:
            result, calls, raw = self.run_smoke(
                self.GOOD, self.GOOD.replace("1" * 64, "3" * 64),
                "--candidate-label", "main@4c10889", "--baseline-commit", "a" * 40,
                env_extra={"PERF_SERVER_CPUS": "0-3,8-11", "PERF_MACHINE_JSON": f.name},
            )
        finally:
            os.unlink(f.name)
        self.assertEqual(result["verdict"], "pass")
        self.assertEqual(result["machine"], machine)
        self.assertEqual(result["candidate"]["label"], "main@4c10889")
        self.assertEqual(result["baseline"]["commit"], "a" * 40)
        self.assertEqual(result["baseline"]["label"], "example/falkordb-server@333333333333")
        runs = [c for c in calls if c.startswith("run ")]
        self.assertEqual([r.split()[-1][-3:] for r in runs], ["111", "333", "333", "111"])
        self.assertTrue(all("--cpuset-cpus 0-3,8-11" in r for r in runs))
        self.assertFalse(any("6379:6379" in r for r in runs))
        self.assertIn("smoke.csv", raw)

    def test_dead_candidate_is_a_confirmed_regression(self):
        result, _, _ = self.run_smoke(self.DEAD, self.GOOD)
        self.assertEqual((result["verdict"], result["confirmed"]), ("regression", True))
        self.assertEqual(result["findings"][0]["status"], "regression")
        self.assertEqual(result["machine"]["runner"], "local")

    def test_dead_baseline_is_an_error(self):
        result, _, _ = self.run_smoke(self.GOOD, self.DEAD)
        self.assertEqual(result["verdict"], "error")
        self.assertIn("docker pull failed", result["error"])
        self.assertEqual(result["findings"][0]["status"], "improvement")

    def test_relative_out_is_under_the_start_directory(self):
        # Contract v1.8: _perf-vm.yml passes `--out out`.
        result, _, raw = self.run_smoke(self.GOOD, self.GOOD, relative_out=True)
        self.assertEqual(result["verdict"], "pass")
        self.assertIn("smoke.csv", raw)

    def test_truncated_query_reply_is_not_a_completed_start(self):
        result, _, _ = self.run_smoke(self.GOOD, self.GOOD, env_extra={"FAKE_SERVER_MODE": "truncate"})
        self.assertEqual(result["verdict"], "error")
        self.assertIn("reply cut short", result["error"])

    def test_a_probe_slower_than_the_limit_is_not_a_completed_start(self):
        # 51 replies at 0.1 s each cannot finish inside a 2 s limit, though each one succeeds.
        result, _, _ = self.run_smoke(self.GOOD, self.GOOD,
                                      env_extra={"FAKE_SERVER_MODE": "slow", "SMOKE_START_TIMEOUT_S": "2"})
        self.assertEqual(result["verdict"], "error")
        self.assertRegex(result["error"], r"probe failed|took longer")

    def test_dead_on_both_sides_is_info_not_regression(self):
        result, _, _ = self.run_smoke(self.DEAD, self.DEAD.replace("2" * 64, "4" * 64))
        self.assertEqual(result["verdict"], "error")
        self.assertEqual(result["findings"][0]["status"], "info")


# gce-github-runner's action.sh stand-in, driven by type@zone lists:
#   SUCCEED   creates the VM and reports runner_id;
#   NOTREADY  creates it, waits out the readiness check, deletes it, fails;
#   PARTIAL   creates it, then fails with an error and leaves it behind;
#   anything else is refused with gcloud's two-line error.
FAKE_ACTION = r"""#!/usr/bin/env bash
for a in "$@"; do
  case $a in
    --machine_zone=*) zone=${a#*=} ;;
    --machine_type=*) type=${a#*=} ;;
  esac
done
echo "$*" >> "$STATE/attempts"
n=$(wc -l < "$STATE/attempts" | tr -d ' ')
vm=gce-gh-$n
if [[ " $SUCCEED $NOTREADY $PARTIAL " == *" $type@$zone "* ]]; then
  printf '%s\t%s\t%s\n' "$vm" "$zone" "$type" >> "$STATE/vms"
  echo "runner_id=$vm" >> "$GITHUB_OUTPUT"
fi
if [[ " $SUCCEED " == *" $type@$zone "* ]]; then
  exit 0
fi
if [[ " $NOTREADY " == *" $type@$zone "* ]]; then
  echo "Waited 7 minutes for $vm, without luck, deleting $vm ..."
  grep -v "^$vm	" "$STATE/vms" > "$STATE/vms.new"; mv "$STATE/vms.new" "$STATE/vms"
  echo "Deleted [https://www.googleapis.com/compute/v1/projects/$GCP_PROJECT_ID/zones/$zone/instances/$vm]."
  exit 1
fi
if [[ " $PARTIAL " == *" $type@$zone "* ]]; then
  echo "ERROR: (gcloud.compute.instances.create) something odd after insert"
  exit 1
fi
echo "ERROR: (gcloud.compute.instances.create) Could not fetch resource:"
echo " - The zone 'projects/$GCP_PROJECT_ID/zones/$zone' does not have enough resources available to fulfill the request."
exit 1
"""

# gcloud stand-in over the VMs FAKE_ACTION made: list (honouring whether the
# format asks for the machine type), describe and delete.
FAKE_GCLOUD = r"""#!/usr/bin/env bash
touch "$STATE/vms"
echo "$*" >> "$STATE/gcloud"
[[ $CLOUDSDK_CORE_DISABLE_PROMPTS == 1 ]] || { echo "prompts enabled" >&2; exit 9; }
if [[ $3 == delete && -n ${DELETE_FAILS:-} ]]; then
  echo "ERROR: delete refused" >&2
  exit 1
fi
if [[ $1 == components && $2 == install ]]; then
  [[ -n ${BETA_INSTALLABLE:-} ]] && touch "$STATE/beta"
  exit 0
fi
if [[ $1 == beta ]]; then
  [[ -n ${BETA_MISSING:-} && ! -e $STATE/beta ]] && { echo "beta Not Installed" >&2; exit 1; }
  shift
  if [[ $3 == create && $* == *--help* ]]; then
    [[ -n ${BETA_OLD:-} && ! -e $STATE/beta ]] && { echo "--machine-type --zone"; exit 0; }
    echo "--machine-type --performance-monitoring-unit=PERFORMANCE_MONITORING_UNIT --zone"
    exit 0
  fi
  [[ $3 == describe && $* == *performanceMonitoringUnit* ]] && { echo STANDARD; exit 0; }
fi
case "$3" in
  list)
    if [[ "$*" == *machineType* ]]; then cat "$STATE/vms"; else cut -f1,2 "$STATE/vms"; fi ;;
  describe)
    [[ -z ${DESCRIBE_FAILS:-} ]] || exit 1
    awk -F'\t' -v n="$4" -v z="${5#--zone=}" '$1 == n && $2 == z { print $3; found = 1 } END { exit !found }' "$STATE/vms" ;;
  delete)
    echo "delete $4" >> "$STATE/deletes"
    grep -v "^$4	" "$STATE/vms" > "$STATE/vms.new"; mv "$STATE/vms.new" "$STATE/vms" ;;
esac
"""

# The registration-token check: fails when TOKEN_BAD is set.
FAKE_CURL = r"""#!/usr/bin/env bash
echo "$*" >> "$STATE/curl"
[[ -z ${TOKEN_BAD:-} ]] || { echo "curl: (22) The requested URL returned error: 401" >&2; exit 22; }
"""

PROJECT = "proj-123456"


class CreateVmTest(unittest.TestCase):
    """create-vm.sh against a fake action.sh, gcloud and curl."""

    LABEL = "perf-smoke-1-1"

    def create(self, zones, types, succeed="", notready="", partial="", token_bad="", describe_fails="", pmu="",
               beta_missing="", beta_installable="", beta_old="", delete_fails=""):
        self.attempt_args = []
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            for name, body in [("action/action.sh", FAKE_ACTION), ("bin/gcloud", FAKE_GCLOUD),
                               ("bin/curl", FAKE_CURL)]:
                path = tmp / name
                path.parent.mkdir(exist_ok=True)
                path.write_text(body)
                path.chmod(0o755)
            outputs = tmp / "outputs"
            env = dict(os.environ, PATH=f"{tmp / 'bin'}:{os.environ['PATH']}", STATE=str(tmp),
                       SUCCEED=succeed, NOTREADY=notready, PARTIAL=partial, TOKEN_BAD=token_bad,
                       DESCRIBE_FAILS=describe_fails, PMU=pmu, BETA_MISSING=beta_missing,
                       BETA_INSTALLABLE=beta_installable, BETA_OLD=beta_old, DELETE_FAILS=delete_fails,
                       GITHUB_OUTPUT=str(outputs), GITHUB_REPOSITORY="FalkorDB/FalkorDB",
                       GH_SA_TOKEN="t", GCP_PROJECT_ID=PROJECT, GCP_SA_KEY="{}",
                       DELETE_VMS_POLL_S="0", DELETE_VMS_GIVE_UP_S="2")
            proc = subprocess.run(["bash", str(HERE / "create-vm.sh"), str(tmp / "action"), self.LABEL,
                                   zones, *types], capture_output=True, text=True, env=env)
            out = dict(line.split("=", 1) for line in outputs.read_text().splitlines()) if outputs.exists() else {}
            attempts = []
            lines = (tmp / "attempts").read_text().splitlines() if (tmp / "attempts").exists() else []
            for line in lines:
                words = dict(w.split("=", 1) for w in line.split() if w.startswith("--") and "=" in w)
                attempts.append(f"{words['--machine_type']}@{words['--machine_zone']}")
                self.attempt_args.append(words)
            vms = (tmp / "vms").read_text().splitlines() if (tmp / "vms").exists() else []
            self.gcloud_calls = (tmp / "gcloud").read_text().splitlines() if (tmp / "gcloud").exists() else []
            return proc, out, attempts, vms

    def test_first_zone_failing_falls_through(self):
        proc, out, attempts, _ = self.create("z1,z2,z3", ["n4-standard-16", "n2-standard-16"],
                                             succeed="n4-standard-16@z2")
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertEqual(attempts, ["n4-standard-16@z1", "n4-standard-16@z2"])
        self.assertEqual((out["vm_name"], out["zone"], out["machine_type"], out["fallback_used"]),
                         ("gce-gh-2", "z2", "n4-standard-16", "false"))
        self.assertNotIn("runner_id", out)

    def test_fallback_type_after_every_zone(self):
        proc, out, attempts, _ = self.create("z1 z2", ["n4-standard-16", "n2-standard-16"],
                                             succeed="n2-standard-16@z2")
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertEqual(attempts, ["n4-standard-16@z1", "n4-standard-16@z2",
                                    "n2-standard-16@z1", "n2-standard-16@z2"])
        self.assertEqual((out["machine_type"], out["fallback_used"]), ("n2-standard-16", "true"))

    def test_nothing_anywhere_names_each_reason_without_the_project(self):
        proc, out, attempts, _ = self.create("z1,z2", ["n4-standard-16", "n2-standard-16"])
        self.assertEqual(proc.returncode, 1)
        self.assertEqual(len(attempts), 4)
        error = out["create_error"]
        self.assertTrue(error.startswith("no VM came up: n4-standard-16 in z1: "), error)
        self.assertIn("n2-standard-16 in z2: ERROR: (gcloud.compute.instances.create) Could not fetch resource: "
                      "- The zone 'projects/…/zones/z2' does not have enough resources", error)
        self.assertTrue(error.endswith("; every attempt failed"), error)
        self.assertNotIn(PROJECT, error)
        self.assertNotIn("\n", error)

    def test_no_fallback_type(self):
        proc, _, attempts, _ = self.create("z1", ["n4-standard-16"])
        self.assertEqual((proc.returncode, attempts), (1, ["n4-standard-16@z1"]))

    def test_vm_settings(self):
        self.create("z1", ["n4-standard-16", "n2-standard-16"])
        n4, n2 = self.attempt_args
        self.assertEqual((n4["--boot_disk_type"], n2["--boot_disk_type"]), ("hyperdisk-balanced", "pd-balanced"))
        # M4 has no Persistent Disk at all.
        self.create("z1", ["m4-megamem-28"], pmu="standard")
        self.assertEqual(self.attempt_args[0]["--boot_disk_type"], "hyperdisk-balanced")
        self.create("z1", ["n4-standard-16", "n2-standard-16"])
        n4, n2 = self.attempt_args
        for args in (n4, n2):
            self.assertEqual(args["--disk_size"], "100GB")
            self.assertEqual(args["--ephemeral"], "true")
            self.assertEqual(args["--network"], "gh-runner")
            self.assertEqual(args["--scopes"], "compute-rw")
            self.assertEqual(args["--runner_label"], self.LABEL)
            # action.sh runs `if $actions_preinstalled` and `if $arm`: empty would read as true.
            self.assertEqual((args["--actions_preinstalled"], args["--arm"]), ("false", "false"))

    def test_runner_that_never_comes_up_stops_the_loop(self):
        proc, out, attempts, vms = self.create("z1,z2", ["n4-standard-16", "n2-standard-16"],
                                               notready="n4-standard-16@z1")
        self.assertEqual(proc.returncode, 1)
        self.assertEqual(attempts, ["n4-standard-16@z1"])
        self.assertIn("runner never came up", out["create_error"])
        self.assertNotIn(PROJECT, out["create_error"])
        self.assertEqual(vms, [])

    def test_bad_token_stops_before_any_vm(self):
        proc, out, attempts, _ = self.create("z1,z2", ["n4-standard-16"], succeed="n4-standard-16@z1",
                                             token_bad="1")
        self.assertEqual((proc.returncode, attempts), (1, []))
        self.assertIn("GH_SA_TOKEN cannot create a runner registration token", out["create_error"])

    def test_pmu_flag_only_when_asked_and_type_recorded(self):
        proc, out, _, _ = self.create("z1", ["c4-standard-16"], succeed="c4-standard-16@z1", pmu="standard")
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertEqual(self.attempt_args[0]["--performance_monitoring_unit"], "standard")
        self.assertEqual(self.attempt_args[0]["--boot_disk_type"], "hyperdisk-balanced")
        self.assertEqual(out["pmu"], "STANDARD")
        # The default pin's action.sh exits on a flag it does not know.
        proc, out, _, _ = self.create("z1", ["n4-standard-16"], succeed="n4-standard-16@z1")
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertNotIn("--performance_monitoring_unit", self.attempt_args[0])
        self.assertEqual(out["pmu"], "")
        self.assertFalse([c for c in self.gcloud_calls if c.startswith("beta ")], self.gcloud_calls)

    def test_missing_beta_track_is_installed_once(self):
        proc, out, attempts, _ = self.create("z1", ["c4-standard-16"], succeed="c4-standard-16@z1", pmu="standard",
                                             beta_missing="1", beta_installable="1")
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertEqual((attempts, out["pmu"]), (["c4-standard-16@z1"], "STANDARD"))
        self.assertEqual(len([c for c in self.gcloud_calls if c.startswith("components install")]), 1)

    def test_beta_track_without_the_pmu_flag_counts_as_missing(self):
        proc, out, attempts, _ = self.create("z1", ["c4-standard-16"], succeed="c4-standard-16@z1",
                                             pmu="standard", beta_old="1")
        self.assertEqual((proc.returncode, attempts), (1, []))
        self.assertIn("missing or too old", out["create_error"])

    def test_beta_track_that_cannot_be_installed_stops_before_any_vm(self):
        proc, out, attempts, _ = self.create("z1,z2", ["c4-standard-16"], succeed="c4-standard-16@z1",
                                             pmu="standard", beta_missing="1")
        self.assertEqual((proc.returncode, attempts), (1, []))
        self.assertIn("missing or too old", out["create_error"])

    def test_failed_describe_keeps_the_ready_vm(self):
        proc, out, attempts, _ = self.create("z1,z2", ["n2-standard-16"], succeed="n2-standard-16@z1",
                                             describe_fails="1")
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertEqual(attempts, ["n2-standard-16@z1"])
        self.assertEqual((out["vm_name"], out["machine_type"]), ("gce-gh-1", "n2-standard-16"))

    def test_vm_that_will_not_delete_stops_before_a_second_shares_its_label(self):
        proc, out, attempts, vms = self.create("z1,z2", ["n4-standard-16"], succeed="n4-standard-16@z2",
                                               partial="n4-standard-16@z1", delete_fails="1")
        self.assertEqual(proc.returncode, 1)
        self.assertEqual(attempts, ["n4-standard-16@z1"])
        self.assertIn("could not be deleted", out["create_error"])
        self.assertEqual(len(vms), 1)

    def test_vm_left_by_a_failed_attempt_is_deleted_before_the_next(self):
        proc, out, _, vms = self.create("z1,z2", ["n4-standard-16"],
                                        succeed="n4-standard-16@z2", partial="n4-standard-16@z1")
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertEqual(vms, ["gce-gh-2\tz2\tn4-standard-16"])
        self.assertEqual(out["vm_name"], "gce-gh-2")


class PlanTest(unittest.TestCase):
    """plan.sh: input checks and the machine types it asks for."""

    DIGEST = "falkordb/falkordb-server@sha256:" + "a" * 64

    def plan(self, **env_extra):
        with tempfile.TemporaryDirectory() as tmp:
            outputs = Path(tmp) / "outputs"
            env = dict(os.environ, GITHUB_OUTPUT=str(outputs), CHECK="size-probe", MODE="nightly",
                       TIMEOUT="45", ZONES="us-central1-a,us-central1-b", MACHINE_TYPE="", FALLBACK="",
                       PMU="false", CANDIDATE=self.DIGEST, BASELINE=self.DIGEST,
                       GCE_GITHUB_RUNNER_REF="default-ref", GCE_GITHUB_RUNNER_PMU_REF="pmu-ref")
            env.update(env_extra)
            proc = subprocess.run(["bash", "-e", "-c", str(HERE / "plan.sh")], capture_output=True, text=True, env=env)
            out = dict(line.split("=", 1) for line in outputs.read_text().splitlines()) if outputs.exists() else {}
            return proc.returncode, out

    def test_default_vm(self):
        rc, out = self.plan()
        self.assertEqual(rc, 0)
        self.assertEqual((out["machine_type"], out["fallback"], out["pmu"], out["action_ref"]),
                         ("n4-standard-16", "n2-standard-16", "", "default-ref"))
        self.assertEqual((out["command_timeout_s"], out["step_timeout"], out["job_timeout"], out["deadline"]),
                         ("2700", "50", "65", "75"))

    def test_longest_timeout_fits_the_vm_lifetime(self):
        rc, out = self.plan(TIMEOUT="300")
        self.assertEqual((rc, out["deadline"]), (0, "330"))

    def test_n4_size_falls_back_to_the_same_n2_size(self):
        _, out = self.plan(MACHINE_TYPE="n4-highcpu-32")
        self.assertEqual(out["fallback"], "n2-highcpu-32")

    def test_pmu_default_is_c4_with_a_c4_fallback(self):
        rc, out = self.plan(PMU="true")
        self.assertEqual(rc, 0)
        self.assertEqual((out["machine_type"], out["fallback"], out["pmu"], out["action_ref"]),
                         ("c4-standard-16", "c4-standard-8", "standard", "pmu-ref"))

    def test_pmu_never_falls_back_to_a_type_without_counters(self):
        for extra in ({"MACHINE_TYPE": "n4-standard-16"}, {"FALLBACK": "n2-standard-16"},
                      {"MACHINE_TYPE": "c4d-standard-16"}, {"MACHINE_TYPE": "c4a-standard-16"}):
            rc, out = self.plan(PMU="true", **extra)
            self.assertEqual(rc, 1, extra)
            self.assertIn("C4 or M4", out["error"], extra)

    def test_pmu_named_type_gets_no_silent_smaller_fallback(self):
        for named in ("c4-standard-8", "c4-standard-48", "m4-megamem-28"):
            _, out = self.plan(PMU="true", MACHINE_TYPE=named)
            self.assertEqual((out["machine_type"], out["fallback"]), (named, ""), named)
        _, out = self.plan(PMU="true", MACHINE_TYPE="c4-standard-48", FALLBACK="c4-standard-32")
        self.assertEqual(out["fallback"], "c4-standard-32")

    def test_bad_inputs(self):
        for extra, words in [({"CHECK": "micro--x"}, "single dashes"), ({"CHECK": "Size"}, "lowercase"),
                             ({"MODE": "fast"}, "mode"), ({"CANDIDATE": "falkordb/falkordb-server:6.2.0"}, "digest"),
                             ({"TIMEOUT": "0"}, "timeout_minutes"), ({"TIMEOUT": "301"}, "over 300"),
                             ({"ZONES": " , "}, "no zones"),
                             ({"PMU": "yes"}, "pmu"), ({"MACHINE_TYPE": "c4-standard-16 x", "PMU": "true"}, "name"),
                             ({"FALLBACK": "n2-standard-16;true"}, "name")]:
            rc, out = self.plan(**extra)
            self.assertEqual(rc, 1, extra)
            self.assertIn(words, out["error"], extra)


class RunCheckTest(unittest.TestCase):
    """run-check.sh under `bash -e`, the shell GitHub gives a `run:` step."""

    def run_check(self, command, timeout="30", **extra):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            # Records its arguments into out/argv, one per line, then exits with $EXIT.
            (tmp / "record.sh").write_text('printf "%s\\n" "$@" > out/argv; exit "${EXIT:-0}"\n')
            (tmp / "slow.sh").write_text("sleep 30\n")
            outputs = tmp / "outputs"
            env = dict(os.environ, GITHUB_OUTPUT=str(outputs), CHECK="smoke",
                       COMMAND=command.replace("{dir}", str(tmp)),
                       CANDIDATE="c@sha256:" + "1" * 64, BASELINE="b@sha256:" + "2" * 64,
                       MODE="nightly", TIMEOUT_SECONDS=timeout, PERF_CLIENT_CPUS="")
            for key in ("CANDIDATE_LABEL", "BASELINE_LABEL", "CANDIDATE_COMMIT", "BASELINE_COMMIT", "EXIT",
                        "PERF_VM_TOOLS"):
                env.pop(key, None)
            env.update({key: value.replace("{dir}", str(tmp)) for key, value in extra.items()})
            # The step is `run: "$PERF_VM_TOOLS/run-check.sh"`, which GitHub hands to
            # `bash -e`: the file is executed, so it must be executable.
            proc = subprocess.run(["bash", "-e", "-c", str(HERE / "run-check.sh")], cwd=tmp,
                                  capture_output=True, text=True, env=env)
            out = dict(line.split("=", 1) for line in outputs.read_text().splitlines()) if outputs.exists() else {}
            argv_file = tmp / "out" / "argv"
            argv = argv_file.read_text().splitlines() if argv_file.exists() else []
            return proc, out, argv

    def test_out_that_cannot_be_emptied_stops_before_the_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            work = tmp / "work"
            (work / "out").mkdir(parents=True)
            (work / "out" / "result.json").write_text("{}")  # an earlier run's
            marker = tmp / "ran"
            work.chmod(0o555)  # out/ can be emptied but not removed or recreated
            try:
                env = dict(os.environ, GITHUB_OUTPUT=str(tmp / "outputs"), CHECK="smoke",
                           COMMAND=f"touch {marker}", CANDIDATE="c@sha256:" + "1" * 64,
                           BASELINE="b@sha256:" + "2" * 64, MODE="nightly", TIMEOUT_SECONDS="30",
                           PERF_CLIENT_CPUS="")
                proc = subprocess.run(["bash", "-e", "-c", str(HERE / "run-check.sh")], cwd=work,
                                      capture_output=True, text=True, env=env)
            finally:
                work.chmod(0o755)
            self.assertEqual(proc.returncode, 2, proc.stdout + proc.stderr)
            self.assertIn("could not empty out/", proc.stdout)
            self.assertFalse(marker.exists())
            self.assertFalse((tmp / "outputs").exists())

    def test_failing_command_is_recorded_and_returned(self):
        proc, out, _ = self.run_check("bash {dir}/record.sh", EXIT="3")
        self.assertEqual(proc.returncode, 3)
        self.assertEqual(out, {"started": "true", "command_exit": "3", "timed_out": "false"})
        self.assertIn("wrote no result.json", proc.stdout)

    def test_timeout_is_recorded(self):
        proc, out, _ = self.run_check("bash {dir}/slow.sh", timeout="1")
        self.assertEqual(proc.returncode, 124)
        self.assertEqual(out, {"started": "true", "command_exit": "124", "timed_out": "true"})

    def test_command_may_name_the_workflow_scripts(self):
        for spelling in ("$PERF_VM_TOOLS", "${PERF_VM_TOOLS}"):
            proc, out, argv = self.run_check(f"bash {spelling}/record.sh", PERF_VM_TOOLS="{dir}")
            self.assertEqual(proc.returncode, 0, spelling + proc.stdout + proc.stderr)
            self.assertEqual(argv[:2], ["--candidate", "c@sha256:" + "1" * 64], spelling)

    def test_unset_tools_path_is_an_error_not_a_wrong_path(self):
        proc, out, _ = self.run_check("bash $PERF_VM_TOOLS/record.sh")
        self.assertEqual(proc.returncode, 2)
        self.assertIn("PERF_VM_TOOLS", proc.stdout)
        self.assertEqual(out, {})

    def test_contract_flags_and_only_non_empty_labels(self):
        proc, out, argv = self.run_check("bash {dir}/record.sh", CANDIDATE_LABEL="main@4c10889",
                                         BASELINE_COMMIT="", BASELINE_LABEL="6.2.0")
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertEqual(out["command_exit"], "0")
        self.assertEqual(argv, ["--candidate", "c@sha256:" + "1" * 64, "--baseline", "b@sha256:" + "2" * 64,
                                "--out", "out", "--mode", "nightly",
                                "--candidate-label", "main@4c10889", "--baseline-label", "6.2.0"])


class ScriptModeTest(unittest.TestCase):
    def test_scripts_the_workflow_executes_are_executable(self):
        for script in HERE.glob("*.sh"):
            self.assertTrue(os.access(script, os.X_OK), f"{script.name} is not executable")


class WorkflowScriptsTest(unittest.TestCase):
    """Every job runs tests/perf/vm from the workflow's own commit, not the caller's."""

    def test_scripts_come_from_the_workflow_commit(self):
        text = (HERE.parents[2] / ".github" / "workflows" / "_perf-vm.yml").read_text()
        jobs = dict(zip(*[iter(re.split(r"^  (create|run|cleanup):$", text, flags=re.M)[1:])] * 2))
        self.assertEqual(sorted(jobs), ["cleanup", "create", "run"])
        for name, body in jobs.items():
            self.assertIn("repository: ${{ job.workflow_repository }}", body, name)
            self.assertIn("ref: ${{ job.workflow_sha }}", body, name)
            self.assertIn("path: .perf-vm-tools", body, name)
            self.assertIn('[[ -z $WANT_SHA || $got != "$WANT_SHA" ]]', body, name)
            # PERF_VM_TOOLS is exported before the comparison can fail the step.
            self.assertLess(body.index("PERF_VM_TOOLS=$GITHUB_WORKSPACE"), body.index("[[ -z $WANT_SHA"), name)
        # The calling commit is checked out first: checkout empties a workspace
        # that is not its repository, which would remove .perf-vm-tools.
        run = jobs["run"]
        self.assertLess(run.index("- name: Check out the calling commit"), run.index("path: .perf-vm-tools"))
        # Every cleanup step survives a cancel.
        steps = re.split(r"^      - ", jobs["cleanup"].split("    steps:\n", 1)[1], flags=re.M)[1:]
        self.assertGreaterEqual(len(steps), 5)
        for step in steps:
            self.assertRegex(step, r"\n        if: always\(\)", step.splitlines()[0])
        # Outside comments and the checkout itself, no step names the caller's copy.
        for line in text.splitlines():
            code = line.split("#", 1)[0]
            if "tests/perf/vm" in code:
                self.assertTrue("sparse-checkout:" in code or ".perf-vm-tools/tests/perf/vm" in code, line)


class WorkflowLabelTest(unittest.TestCase):
    """The runner label is spelled the same where it is created, used and deleted."""

    def test_label_template_matches_runs_on(self):
        text = (HERE.parents[2] / ".github" / "workflows" / "_perf-vm.yml").read_text()
        label = re.search(r"^  PERF_VM_LABEL: (.+)$", text, re.M).group(1)
        runs_on = re.findall(r"^    runs-on: (perf-.+)$", text, re.M)
        self.assertEqual(runs_on, [label])
        create, rest = text.split("\n  run:\n")
        self.assertIn('create-vm.sh" "$PWD/.gce-github-runner" "$PERF_VM_LABEL"', create)
        cleanup = rest.split("\n  cleanup:\n")[1]
        # Cleanup deletes every attempt's label: the template with its last part replaced.
        self.assertTrue(label.endswith("-${{ github.run_attempt }}"), label)
        self.assertIn('label=${PERF_VM_LABEL%-*}-$attempt', cleanup)
        self.assertIn("delete=${PERF_VM_TOOLS:-.perf-vm-tools/tests/perf/vm}/delete-vms.sh", cleanup)
        self.assertIn('"$delete" "$label"', cleanup)


class DeleteVmsTest(unittest.TestCase):
    """delete-vms.sh against a fake gcloud that replays scripted listings."""

    def run_script(self, listings: list[str], grace: str = "0"):
        """Each listing is what one `list` call prints; FAIL makes that call fail."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            for i, listing in enumerate(listings):
                (tmp / f"list{i}").write_text(listing)
            gcloud = tmp / "gcloud"
            gcloud.write_text(textwrap.dedent(f"""\
                #!/usr/bin/env bash
                state={tmp}
                echo "$*" >> "$state/calls"
                if [[ "$3" == list ]]; then
                  n=$(cat "$state/n" 2>/dev/null || echo 0)
                  files=("$state"/list*); last=${{#files[@]}}
                  (( n >= last )) && n=$((last - 1))
                  echo $((n + 1)) > "$state/n"
                  [[ $(cat "$state/list$n") == FAIL ]] && exit 3
                  cat "$state/list$n"
                fi
                """))
            gcloud.chmod(0o755)
            env = dict(os.environ, PATH=f"{tmp}:{os.environ['PATH']}",
                       DELETE_VMS_POLL_S="0", DELETE_VMS_GIVE_UP_S="2")
            proc = subprocess.run(["bash", str(HERE / "delete-vms.sh"), "perf-smoke-1-1", grace],
                                  capture_output=True, text=True, env=env)
            calls = (tmp / "calls").read_text().splitlines() if (tmp / "calls").exists() else []
            return proc, calls

    def test_deletes_every_vm_with_the_label(self):
        proc, calls = self.run_script(["vm-a\tus-central1-a\nvm-b\tus-central1-c\n", ""])
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        deletes = [c for c in calls if " delete " in f" {c} "]
        self.assertEqual(deletes, [
            "compute instances delete vm-a --zone=us-central1-a --quiet",
            "compute instances delete vm-b --zone=us-central1-c --quiet",
        ])

    def test_nothing_to_delete(self):
        proc, calls = self.run_script([""])
        self.assertEqual(proc.returncode, 0)
        self.assertFalse([c for c in calls if "delete" in c])

    def test_vm_that_never_goes_away_fails(self):
        proc, _ = self.run_script(["vm-a\tus-central1-a\n"])
        self.assertEqual(proc.returncode, 1)
        self.assertIn("still exist", proc.stdout)

    def test_failed_listing_is_retried_not_read_as_none(self):
        proc, calls = self.run_script(["FAIL", "FAIL", "vm-a\tus-central1-a\n", ""])
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("compute instances delete vm-a --zone=us-central1-a --quiet", calls)

    def test_listing_that_never_works_fails(self):
        proc, _ = self.run_script(["FAIL"])
        self.assertEqual(proc.returncode, 1)
        self.assertIn("cannot list", proc.stdout)

    def test_grace_catches_a_vm_that_appears_late(self):
        # First listing is empty (insert still in flight), then the VM shows up.
        proc, calls = self.run_script(["", "vm-late\tus-central1-b\n", ""], grace="3")
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("compute instances delete vm-late --zone=us-central1-b --quiet", calls)

    def test_invalid_label_is_a_no_op(self):
        proc = subprocess.run(["bash", str(HERE / "delete-vms.sh"), "Perf_Bad Label"],
                              capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0)


if __name__ == "__main__":
    unittest.main()
