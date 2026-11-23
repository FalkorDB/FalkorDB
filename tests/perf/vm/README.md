# Perf checks on a temporary GCE VM

`.github/workflows/_perf-vm.yml` runs one perf check on a VM of its own and
deletes the VM afterwards. `perf-nightly.yml` calls it once for each
`runner: vm` entry in `tests/perf/checks.yaml`.

## Calling it

```yaml
jobs:
  size-probe:
    permissions:
      contents: read   # checkout of the calling commit
      packages: read   # ghcr.io login (the images also pull anonymously)
    uses: ./.github/workflows/_perf-vm.yml
    with:
      check: size-probe                          # [a-z][a-z0-9-]{0,42}, unique in the run
      command: uv run --project bench bench probe
      candidate_image: ghcr.io/falkordb/falkordb-server@sha256:…
      baseline_image: falkordb/falkordb-server@sha256:…
      mode: nightly                              # queue | nightly | long
      timeout_minutes: 45                        # default 60
      machine_type: n4-standard-16               # default; '' means the default
      # optional: candidate_label, baseline_label, candidate_commit,
      # baseline_commit, fallback_machine_type, zones
    secrets: inherit                             # GH_SA_TOKEN, GCP_PROJECT_ID, GCP_SA_KEY
```

A matrix caller needs `fail-fast: false`, or one check's failure cancels the
others mid-run (their VMs are still deleted, but their results are lost).
Re-run a failed run with **Re-run all jobs**. "Re-run failed jobs" does not
re-run the create job, so no VM carries the new attempt's label: the run job is
skipped and the cleanup job fails, saying so.

The VM job runs, in a checkout of the calling commit:

```
<command> --candidate <image> --baseline <image> --out out --mode <mode>
          [--candidate-label …] [--baseline-label …] [--candidate-commit …] [--baseline-commit …]
```

A label or commit flag is passed only when its input is non-empty. `command` is
split on spaces and run directly, so it cannot hold quotes, pipes or `VAR=x`.

## What the check gets (contract v1.3)

| variable | value |
| --- | --- |
| `PERF_SERVER_CPUS` | CPUs for the servers: pass `docker run --cpuset-cpus "$PERF_SERVER_CPUS"` |
| `PERF_CLIENT_CPUS` | CPUs the command already runs on (`taskset`), so the client side needs nothing |
| `PERF_ZONE`, `PERF_MACHINE_TYPE` | where the VM is, read from the metadata server |
| `PERF_MACHINE_JSON` | a file holding the result.json `machine` object: `{"runner": "gce", "type", "cpu", "cores"}` |

The split is by physical core: hyperthread siblings stay together, the
servers get the first half of the cores and the client the rest. On an
n4-standard-16 (8 cores, CPU n and n+8 share a core) that is `0-3,8-11` and
`4-7,12-15`. A check must also run without these variables (on a laptop,
unpinned).

The VM is Debian 12 with Docker, git, python3 and uv, a 100 GB boot disk, and
4 GB of memory per vCPU on the default type. Its service account token is
scoped to Compute only (`compute-rw`): enough to label and delete itself.

## The artifact

`perf-result-<check>` holds the check's `out/` directory plus `perf-vm.json`:

```json
{"schema": 1, "check": "size-probe", "vm_name": "gce-gh-…", "zone": "us-central1-b",
 "machine_type": "n2-standard-16", "fallback_used": true, "create_error": null,
 "command_exit": 0, "timed_out": false}
```

It is uploaded when the command fails or times out too. When no VM came up,
the artifact holds only `perf-vm.json`, with `create_error` naming each zone
and type tried and why it failed. The merge step then stubs the check as
`error` and can quote that. The workflow also outputs `zone`, `machine_type`,
`fallback_used`, `vm_name` and `command_exit`. A matrix caller sees only one
leg's outputs, so read `perf-vm.json` instead.

## Getting a VM

`create-vm.sh` runs `FalkorDB/gce-github-runner`'s `action.sh` at commit
`0caf8406` (`GCE_GITHUB_RUNNER_REF` in the workflow, the same pin C's
`benchmark.yml` uses) directly, not through the composite action, and once
per zone: every zone in `vars.GCP_ZONES` (or the `zones` input) for
`machine_type`, then every zone for the fallback type. Any failure moves on to
the next zone. This loop replaces `action.sh`'s own zone loop, which moves on
only for `ZONE_RESOURCE_POOL_EXHAUSTED_WITH_DETAILS` and stops the whole
create on any other error. The flags are the ones the composite passes, with
`ephemeral=true`, a 100 GB disk, Debian 12, `compute-rw` scope, and a runner
version resolved with a token instead of through the unauthenticated API.

Two failures are not a zone's fault and stop the loop at once instead of
costing every zone: a `GH_SA_TOKEN` that cannot create a runner registration
token (checked once, up front; `action.sh` would read the error as the token
"null"), and a VM whose runner never comes up within the action's 7-minute
wait. No attempt starts after 50 minutes. `create_error` quotes gcloud's reason
line for each attempt, with the project ID replaced by `…`: GitHub drops a job
output that contains a secret.

The fallback is `n2-<size>` for an `n4-<size>` type, the same vCPUs and
memory in a more plentiful family. Set `fallback_machine_type` for any other
choice. A fallback VM is still one machine running both sides, so the
comparison holds. Only its numbers are not comparable with another night's.

## Deleting the VM

Three independent things delete it:

1. **The cleanup job** (`ubuntu-latest`, `if: always()` on the job and on
   every step, so a cancel that lands mid-cleanup does not skip the delete)
   runs after success, failure, timeout and cancel. It deletes every VM
   carrying the run's label in any zone (`delete-vms.sh`), retrying a failed
   listing rather than reading it as "none", and goes red if one remains.
   After a create that did not succeed, it keeps listing for 2 minutes,
   because a cancelled `instances create` finishes server-side.
2. **A systemd timer on the VM**, armed first thing in the run job, deletes
   the VM `timeout_minutes + 30` minutes after the job starts. It covers a run
   whose cleanup job never runs (a force-cancel, an Actions outage). Proven on
   a hand-made VM (2026-11-23): the VM's own service account deleted it 3.5
   minutes after boot. With the gcloud path broken, `systemd-run` refuses to
   arm, the step is flagged (`continue-on-error`) and nothing else changes.
3. **The action's startup script** runs `sleep 6h && gcloud … delete`.

What remains: a VM whose run job never started (its runner went offline
after the action saw it ready) carries no timer (2). If the cleanup job also
never runs (a force-cancel during create), only the 6 h timer (3) deletes it.
GCE's own `--max-run-duration` set at create would bound every path. The
pinned action has no input for it, and `set-scheduling` only works on a
stopped VM; adding that input to `gce-github-runner` is the follow-up.

## Files

| file | what |
| --- | --- |
| `create-vm.sh` | zones × types loop over the action's `action.sh` |
| `run-check.sh` | runs the command with the contract's flags under `timeout` and `taskset`; records exit code and timeout |
| `delete-vms.sh` | delete every VM with a label; retries listing, waits out in-flight inserts |
| `vm.py env` | core split, machine object, `PERF_*` lines for `$GITHUB_ENV` |
| `vm.py record` | writes `perf-vm.json` |
| `smoke.py` | the VM path's self-test check, not in `checks.yaml` (below) |
| `test_vm.py` | `python3 -I -m unittest discover -s tests/perf/vm` |

### smoke.py

A stand-in check in the contract's shape. It starts each image twice,
interleaved with the order swapped, pinned to `PERF_SERVER_CPUS`, on ports it
picks itself, and asks each server for PING and `GRAPH.QUERY smoke "RETURN 1"`.
Its result.json says `pass` when every start answered, `regression` when only
the candidate failed, and `error` when the baseline failed. Every `--mode` does
the same. Use it to try the VM path without a real check:

```yaml
command: python3 -I tests/perf/vm/smoke.py
timeout_minutes: 10
```
