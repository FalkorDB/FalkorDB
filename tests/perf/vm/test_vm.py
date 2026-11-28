"""Tests for the _perf-vm.yml helpers, in bench's environment (plan.sh imports falkorbench):
uv run --project bench --frozen --group dev python -I -B -m pytest tests/perf/vm"""

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

from falkorbench.contract import IMAGE_RE

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

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

    def test_split_keeps_hyperthread_siblings_together(self):
        cases = {
            # n4-standard-16 / c4-standard-16: 8 cores, CPU n and n+8 share one.
            "siblings n apart": ([[i, i + 8] for i in range(8)], "0-3,8-11", "4-7,12-15"),
            "adjacent siblings": ([[2 * i, 2 * i + 1] for i in range(8)], "0-7", "8-15"),
        }
        for name, (siblings, want_server, want_client) in cases.items():
            with self.subTest(name):
                server, client, warning = self.split(siblings)
                self.assertEqual((vm.format_cpu_list(server), vm.format_cpu_list(client), warning),
                                 (want_server, want_client, None))
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
                vm.main(["record", "--out", str(path), "--check", "pokec", *extra])
            return json.loads(path.read_text())

    def test_fields(self):
        cases = {
            "no VM": (["--create-error", "no zone had n4-standard-16"],
                      {"create_error": "no zone had n4-standard-16", "zone": None, "command_exit": None,
                       "fallback_used": False, "pmu": None}),
            "ran": (["--vm-name", "gce-gh-1", "--zone", "us-central1-b", "--machine-type", "n2-standard-16",
                     "--fallback-used", "true", "--command-exit", "0", "--timed-out", "false"],
                    {"zone": "us-central1-b", "fallback_used": True, "command_exit": 0, "create_error": None}),
            "pmu": (["--pmu", "STANDARD"], {"pmu": "STANDARD"}),
        }
        for name, (args, want) in cases.items():
            with self.subTest(name):
                got = self.record(*args)
                self.assertEqual({key: got[key] for key in want}, want)

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

    LABEL = "perf-pokec-1-1"

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

    def test_pmu_needs_a_beta_track_with_the_flag(self):
        # (fake gcloud state, VM expected, `components install` calls)
        cases = {
            "missing, installable": ({"beta_missing": "1", "beta_installable": "1"}, True, 1),
            "present but without the flag": ({"beta_old": "1"}, False, 1),
            "missing, not installable": ({"beta_missing": "1"}, False, 1),
        }
        for name, (state, comes_up, installs) in cases.items():
            with self.subTest(name):
                proc, out, attempts, _ = self.create("z1,z2", ["c4-standard-16"], succeed="c4-standard-16@z1",
                                                     pmu="standard", **state)
                installed = [c for c in self.gcloud_calls if c.startswith("components install")]
                self.assertEqual(len(installed), installs)
                if comes_up:
                    self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
                    self.assertEqual((attempts, out["pmu"]), (["c4-standard-16@z1"], "STANDARD"))
                else:
                    self.assertEqual((proc.returncode, attempts), (1, []))
                    self.assertIn("missing or too old", out["create_error"])

    def test_failed_describe_keeps_the_ready_vm(self):
        proc, out, attempts, _ = self.create("z1,z2", ["n2-standard-16"], succeed="n2-standard-16@z1",
                                             describe_fails="1")
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertEqual(attempts, ["n2-standard-16@z1"])
        self.assertEqual((out["vm_name"], out["machine_type"]), ("gce-gh-1", "n2-standard-16"))

    def test_vm_left_by_a_failed_attempt(self):
        with self.subTest("deletes: the next attempt goes ahead"):
            proc, out, _, vms = self.create("z1,z2", ["n4-standard-16"],
                                            succeed="n4-standard-16@z2", partial="n4-standard-16@z1")
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            self.assertEqual(vms, ["gce-gh-2\tz2\tn4-standard-16"])
            self.assertEqual(out["vm_name"], "gce-gh-2")
        with self.subTest("will not delete: stop, so no second VM shares its label"):
            proc, out, attempts, vms = self.create("z1,z2", ["n4-standard-16"], succeed="n4-standard-16@z2",
                                                   partial="n4-standard-16@z1", delete_fails="1")
            self.assertEqual((proc.returncode, attempts, len(vms)), (1, ["n4-standard-16@z1"], 1))
            self.assertIn("could not be deleted", out["create_error"])

class PlanTest(unittest.TestCase):
    """plan.sh: input checks and the machine types it asks for."""

    DIGEST = "falkordb/falkordb-server@sha256:" + "a" * 64

    def plan(self, **env_extra):
        with tempfile.TemporaryDirectory() as tmp:
            outputs = Path(tmp) / "outputs"
            env = dict(os.environ, GITHUB_OUTPUT=str(outputs), CHECK="pokec", MODE="nightly",
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
            # C4A is refused twice over: no counters here, and Arm.
            self.assertRegex(out["error"], "C4 or M4|Arm", extra)

    def test_pmu_named_type_gets_no_silent_smaller_fallback(self):
        for named in ("c4-standard-8", "c4-standard-48", "m4-megamem-28"):
            _, out = self.plan(PMU="true", MACHINE_TYPE=named)
            self.assertEqual((out["machine_type"], out["fallback"]), (named, ""), named)
        _, out = self.plan(PMU="true", MACHINE_TYPE="c4-standard-48", FALLBACK="c4-standard-32")
        self.assertEqual(out["fallback"], "c4-standard-32")

    def test_image_rule_is_the_contracts(self):
        digest = "@sha256:" + "a" * 64
        images = ["falkordb/falkordb-server" + digest, "ghcr.io/falkordb/falkordb-server:rc-x" + digest,
                  "localhost:5000/repo" + digest, "Falkordb/falkordb-server" + digest,
                  "falkordb/falkordb-server:6.2.0", "falkordb/falkordb-server@sha256:" + "a" * 63]
        for image in images:
            with self.subTest(image):
                rc, out = self.plan(CANDIDATE=image)
                self.assertEqual(rc == 0, IMAGE_RE.fullmatch(image) is not None, out.get("error"))

    def test_bad_inputs(self):
        for extra, words in [({"CHECK": "micro--x"}, "single dashes"), ({"CHECK": "Size"}, "lowercase"),
                             ({"MODE": "fast"}, "mode"), ({"CANDIDATE": "falkordb/falkordb-server:6.2.0"}, "digest"),
                             ({"TIMEOUT": "0"}, "timeout_minutes"), ({"TIMEOUT": "301"}, "over 300"),
                             ({"ZONES": " , "}, "no zones"),
                             ({"PMU": "yes"}, "pmu"), ({"MACHINE_TYPE": "c4-standard-16 x", "PMU": "true"}, "name"),
                             ({"FALLBACK": "n2-standard-16;true"}, "name"),
                             ({"MACHINE_TYPE": "c4a-standard-16"}, "Arm"), ({"MACHINE_TYPE": "n4a-standard-16"}, "Arm"),
                             ({"MACHINE_TYPE": "t2a-standard-8"}, "Arm"), ({"FALLBACK": "c4a-standard-16"}, "Arm"),
                             ({"CANDIDATE_COMMIT": "4c10889"}, "--candidate-commit"),
                             ({"BASELINE_LABEL": "two\nlines"}, "--baseline-label")]:
            rc, out = self.plan(**extra)
            self.assertEqual(rc, 1, extra)
            self.assertIn(words, out["error"], extra)


class RunCheckTest(unittest.TestCase):
    """run-check.sh under `bash -e`, the shell GitHub gives a `run:` step."""

    def run_check(self, command, timeout="30", **extra):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            # Records its arguments into out/argv, one per line, and its budget into
            # out/budget, then exits with $EXIT.
            (tmp / "record.sh").write_text('printf "%s\\n" "$@" > out/argv; '
                                           'printf "%s" "${PERF_TIMEOUT_SECONDS-unset}" > out/budget; '
                                           'exit "${EXIT:-0}"\n')
            (tmp / "slow.sh").write_text("sleep 30\n")
            outputs = tmp / "outputs"
            env = dict(os.environ, GITHUB_OUTPUT=str(outputs), CHECK="pokec",
                       COMMAND=command.replace("{dir}", str(tmp)),
                       CANDIDATE="c@sha256:" + "1" * 64, BASELINE="b@sha256:" + "2" * 64,
                       MODE="nightly", TIMEOUT_SECONDS=timeout, PERF_CLIENT_CPUS="")
            for key in ("CANDIDATE_LABEL", "BASELINE_LABEL", "CANDIDATE_COMMIT", "BASELINE_COMMIT", "EXIT",
                        "PERF_VM_TOOLS", "PERF_TIMEOUT_SECONDS"):
                env.pop(key, None)
            env.update({key: value.replace("{dir}", str(tmp)) for key, value in extra.items()})
            # The step is `run: "$PERF_VM_TOOLS/run-check.sh"`, which GitHub hands to
            # `bash -e`: the file is executed, so it must be executable.
            proc = subprocess.run(["bash", "-e", "-c", str(HERE / "run-check.sh")], cwd=tmp,
                                  capture_output=True, text=True, env=env)
            out = dict(line.split("=", 1) for line in outputs.read_text().splitlines()) if outputs.exists() else {}
            argv_file = tmp / "out" / "argv"
            argv = argv_file.read_text().splitlines() if argv_file.exists() else []
            budget_file = tmp / "out" / "budget"
            self.budget = budget_file.read_text() if budget_file.exists() else None
            return proc, out, argv

    def test_command_gets_its_own_budget(self):
        """A check that schedules its own runs (soak) reads the seconds it has left to them."""
        proc, out, _ = self.run_check("bash {dir}/record.sh", timeout="2700")
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertEqual(self.budget, "2700")

    def test_out_that_cannot_be_emptied_stops_before_the_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            work = tmp / "work"
            (work / "out").mkdir(parents=True)
            (work / "out" / "result.json").write_text("{}")  # an earlier run's
            marker = tmp / "ran"
            # An `rm` that fails, as one would on a directory it may not remove.
            # Not chmod: root, which the VM's runner is, ignores permissions.
            fake_rm = tmp / "bin" / "rm"
            fake_rm.parent.mkdir()
            fake_rm.write_text('#!/usr/bin/env bash\necho "rm: out: Permission denied" >&2\nexit 1\n')
            fake_rm.chmod(0o755)
            env = dict(os.environ, GITHUB_OUTPUT=str(tmp / "outputs"), CHECK="pokec",
                       COMMAND=f"touch {marker}", CANDIDATE="c@sha256:" + "1" * 64,
                       BASELINE="b@sha256:" + "2" * 64, MODE="nightly", TIMEOUT_SECONDS="30",
                       PERF_CLIENT_CPUS="", PATH=f"{fake_rm.parent}:{os.environ['PATH']}")
            proc = subprocess.run(["bash", "-e", "-c", str(HERE / "run-check.sh")], cwd=work,
                                  capture_output=True, text=True, env=env)
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
            with self.subTest(spelling):
                proc, out, argv = self.run_check(f"bash {spelling}/record.sh", PERF_VM_TOOLS="{dir}")
                self.assertEqual(proc.returncode, 0, spelling + proc.stdout + proc.stderr)
                self.assertEqual(argv[:2], ["--candidate", "c@sha256:" + "1" * 64])
        with self.subTest("unset: an error, not a wrong path"):
            proc, out, _ = self.run_check("bash $PERF_VM_TOOLS/record.sh")
            self.assertEqual((proc.returncode, out), (2, {}))
            self.assertIn("PERF_VM_TOOLS", proc.stdout)

    def test_contract_flags_and_only_non_empty_labels(self):
        proc, out, argv = self.run_check("bash {dir}/record.sh", CANDIDATE_LABEL="main@4c10889",
                                         BASELINE_COMMIT="", BASELINE_LABEL="6.2.0")
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertEqual(out["command_exit"], "0")
        self.assertEqual(argv, ["--candidate", "c@sha256:" + "1" * 64, "--baseline", "b@sha256:" + "2" * 64,
                                "--out", "out", "--mode", "nightly",
                                "--candidate-label=main@4c10889", "--baseline-label=6.2.0"])

    def test_dash_prefixed_label_stays_one_argument(self):
        proc, _, argv = self.run_check("bash {dir}/record.sh", CANDIDATE_LABEL="--release")
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("--candidate-label=--release", argv)


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
        # The check's code (the calling commit, or code_ref) is checked out
        # first: checkout empties a workspace that is not its repository, which
        # would remove .perf-vm-tools.
        run = jobs["run"]
        self.assertLess(run.index("- name: Check out the check's code"), run.index("path: .perf-vm-tools"))
        self.assertIn("ref: ${{ inputs.code_ref }}", run.split("path: .perf-vm-tools")[0])
        # Every cleanup step survives a cancel.
        steps = re.split(r"^      - ", jobs["cleanup"].split("    steps:\n", 1)[1], flags=re.M)[1:]
        self.assertGreaterEqual(len(steps), 5)
        for step in steps:
            self.assertRegex(step, r"\n        if: always\(\)", step.splitlines()[0])
        # Outside comments and the checkout itself, no step names the caller's copy.
        for line in text.splitlines():
            code = line.split("#", 1)[0]
            if "tests/perf/vm" in code:
                self.assertTrue("sparse-checkout:" in code or code.strip() == "tests/perf/vm"
                                or ".perf-vm-tools/tests/perf/vm" in code, line)
        # plan.sh runs in the workflow's own bench, whose falkorbench.kit judges the inputs.
        create = jobs["create"]
        self.assertRegex(create, r"sparse-checkout: \|\n +tests/perf/vm\n +bench\n")
        self.assertIn('uv run --project .perf-vm-tools/bench --frozen --no-dev bash "$PERF_VM_TOOLS/plan.sh"', create)


class WorkflowArtifactTest(unittest.TestCase):
    """Every run uploads perf-vm.json: the run job once it starts, cleanup when it never does."""

    def test_cleanup_records_a_run_job_that_never_started(self):
        text = (HERE.parents[2] / ".github" / "workflows" / "_perf-vm.yml").read_text()
        jobs = dict(zip(*[iter(re.split(r"^  (create|run|cleanup):$", text, flags=re.M)[1:])] * 2))
        run, cleanup = jobs["run"], jobs["cleanup"]
        self.assertIn("started: ${{ steps.deadline.outputs.started }}", run)
        first = re.split(r"^      - ", run.split("    steps:\n", 1)[1], flags=re.M)[1]
        self.assertIn("id: deadline", first)
        self.assertRegex(first, r"run: \|\n +echo \"started=true\" >> \"\$GITHUB_OUTPUT\"\n")
        condition = ("always() && (needs.create.result != 'success' || "
                     "(needs.run.result != 'skipped' && needs.run.outputs.started != 'true'))")
        for step in ("- name: Record why the check never ran", "- name: Upload perf-result-"):
            body = cleanup.split(step, 1)[1]
            self.assertEqual(re.search(r"\n +if: (.+)\n", body).group(1), condition, step)


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
            proc = subprocess.run(["bash", str(HERE / "delete-vms.sh"), "perf-pokec-1-1", grace],
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

    def test_failed_listing_is_retried_never_read_as_none(self):
        with self.subTest("fails twice, then works"):
            proc, calls = self.run_script(["FAIL", "FAIL", "vm-a\tus-central1-a\n", ""])
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            self.assertIn("compute instances delete vm-a --zone=us-central1-a --quiet", calls)
        with self.subTest("never works"):
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
