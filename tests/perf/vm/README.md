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
4 GB of memory per vCPU on the default type.

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

`create-vm.sh` runs `FalkorDB/gce-github-runner`'s `action.sh` (pinned by
commit in the workflow) once per zone: every zone in `vars.GCP_ZONES` (or the
`zones` input) for `machine_type`, then every zone for the fallback type. Any
failure moves on to the next zone. The action's own zone loop moves on only
for `ZONE_RESOURCE_POOL_EXHAUSTED_WITH_DETAILS` and stops on any other error,
so it is called with one zone at a time.

The fallback is `n2-<size>` for an `n4-<size>` type, the same vCPUs and
memory in a more plentiful family. Set `fallback_machine_type` for any other
choice. A fallback VM is still one machine running both sides, so the
comparison holds. Only its numbers are not comparable with another night's.

## Deleting the VM

Three independent things delete it:

1. **The cleanup job** (`ubuntu-latest`, `if: always()`) runs after success,
   failure, timeout and cancel. It deletes every VM carrying the run's label in
   any zone (`delete-vms.sh`) and goes red if one remains. After a create that
   did not succeed, it keeps listing for 2 minutes, because a cancelled
   `instances create` finishes server-side.
2. **A systemd timer on the VM**, armed first thing in the run job, deletes
   the VM `timeout_minutes + 30` minutes after the job starts. It covers a run
   whose cleanup job never runs (a force-cancel, an Actions outage).
3. **The action's startup script** runs `sleep 6h && gcloud … delete`.

Not covered: a run job queued for its runner never starts, so the timer
(2) is never armed. That needs the runner to have gone offline after the action
saw it ready. The 6 h timer (3) still deletes that VM. GCE's own
`--max-run-duration` would cover all of these. The pinned action has no input
for it, and `set-scheduling` only works on a stopped VM.

## Files

| file | what |
| --- | --- |
| `create-vm.sh` | zones × types loop over the action's `action.sh` |
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
