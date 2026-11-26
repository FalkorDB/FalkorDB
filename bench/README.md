# bench — per-query performance & coverage loop

Measures, per query:

- **instructions and cycles** of the redis-server process (macOS
  `proc_pid_rusage`, no root; Linux `perf stat -p`, needs PMU access)
- **allocated / deallocated bytes** from `MEMORY MALLOC-STATS` merged-arena
  deltas — works on any jemalloc-built redis, i.e. stock
- **branches, branch-misses, L1D-misses** on Apple Silicon, if `pmc_tool` has
  been built (optional, needs root)
- **wall-clock**, with the caveats in "Reading the output" below

plus deterministic **instruction counts under callgrind** for a curated subset,
and a coverage check that the query set still exercises most of the `graph`
crate.

## Setup

The harness is a [uv](https://docs.astral.sh/uv/) project, separate from the
`tests/` virtualenv, because it also has to run inside a container that ships
nothing but the engine.

```bash
uv sync --project bench          # once: the harness and the dev group (pytest, ruff, pyyaml)
```

uv runs it on the Python `bench/.python-version` pins (3.12), on a laptop, a VM,
the CI image and the runners alike. The soak and fuzz checks need one more
dependency group each: `uv run --project bench --group soak …`, `--group fuzz`
(which includes the soak's, as fuzz uses its kit).

Every command below can be run as `uv run --project bench bench …`. From inside
`bench/` it is just `uv run bench …`. The tests:

```bash
uv run --project bench --group dev --group soak --group fuzz python -I -B -m pytest bench/tests
```

## The loop

```bash
cargo build --release                                   # 1. build
uv run --project bench bench measure                    # 2. measure -> bench/results/current.csv
uv run --project bench bench compare                    # 3. gate vs your baseline (exit 1 on a regression)
uv run --project bench bench coverage                   # 4. verify the query set still covers the code
uv run --project bench bench profile "CASE"             # 5. drill into a hot query
```

After a confirmed improvement, promote the numbers:

```bash
cp bench/results/current.csv bench/baseline/rust.csv    # local only; baseline/ is git-ignored
```

Two things to know about that baseline before trusting a verdict:

- **It is a single file, and switching branches does not change it.** After a
  checkout you are comparing against numbers measured on some other commit,
  silently. Re-measure a baseline when you switch.
- **A baseline from another machine is not comparable at all.** Per-host speed
  differences alone measured 1.46x on byte-identical engines.

Automating both (commit-keyed baselines with recorded provenance) is deliberately
not done yet.

### Subsets and drilling in

```bash
uv run --project bench bench measure "CASE" "WITH pipeline"   # merges into the existing CSV
uv run --project bench bench measure --keep-server "RETURN 1" # leave the server up
uv run --project bench bench profile --reuse "CASE"           # profile against that server
```

`bench measure --c-compat` is required when the module is the C engine: it skips
the setup commands the C engine cannot take (its async validation of a composite
unique constraint crashes it; its RDB round-trip drops numeric 0 from a range
index; UDFs are Rust-only) and skips queries whose warmup reply errors.

### vs the C engine, locally

The C engine is `:edge-c`. **Not `:edge`** — that now resolves to the Rust
engine; `:edge` and `:edge-rs` share a digest.

```bash
docker create --name c falkordb/falkordb-server:edge-c
docker cp c:/var/lib/falkordb/bin/falkordb.so /tmp/falkordb-c.so && docker rm c
# (that copy is Linux — on macOS you need a local `master` build, which lands in bin/)

uv run --project bench bench measure --out /tmp/rust.csv
uv run --project bench bench measure --c-compat --module /tmp/falkordb-c.so --out /tmp/c.csv
uv run --project bench bench compare /tmp/rust.csv /tmp/c.csv
```

### Instruction counts under callgrind

```bash
uv run --project bench bench callgrind --module target/release/libfalkordb.so \
    --module-args THREAD_COUNT --module-args 1
```

Linux-only in practice: valgrind has no Apple-silicon support, and even on Linux
arm64 it cannot execute this module (an ARMv8.1 LSE atomic in RediSearch). Use
`--bare` there to validate the differencing arithmetic against a plain
redis-server with no module loaded.

### Flow-test benchmarking (C vs Rust per flow file)

```bash
uv run --project bench bench flow --out bench/results/flow_rust.csv
uv run --project bench bench flow --module /tmp/falkordb-c.so --out bench/results/flow_c.csv
uv run --project bench bench flow --compare bench/results/flow_c.csv --current bench/results/flow_rust.csv
```

Runs each file in `flow_tests_done.txt` through `./flow.sh` (RLTest, parallelism
1). RLTest spawns transient redis-servers and macOS `wait4` rusage does not fold
grandchildren, so a poller samples every redis-server pid that appears; each row
is the **sum** over the servers that file spawned. Tests may legitimately fail on
C (Rust-specific behaviours), so the compare table shows both failure counts.
macOS-only.

## CI

Labelling a PR **`benchmark-cov`** measures the PR, its base and the C engine and
posts one comparison comment. Nothing is compiled: all three modules come from
prebuilt images (`rc-pr-<N>-valkey`, `edge-rs`, `edge-c`), each measured on its own
runner in parallel inside one image built from the C engine's. Two readings per
side — the full query set on allocated bytes, and the `cg=True` subset with exact
callgrind instruction counts, sharded so it finishes in the same wall-clock as the
full set.

The callgrind table is **PR-vs-base only**. The C engine cannot be measured that
way: it busy-waits on a worker thread that valgrind schedules arbitrarily, which
showed up as 331,579,187 instructions of drift between two identical runs (this
module: ~100k) and rows that cost *more* on the run doing *fewer* queries. vs-C
is on allocated bytes, which thread scheduling does not affect.

Its one caveat: the base side is the `edge-rs` image, i.e. the tip of the trunk
when it was last built, not the PR's merge base. For a borderline row, confirm
locally against a real base build.

## Result contract (nightly perf checks)

Each nightly perf check (micro-hw, pokec, soak-a, soak-b, fuzz, ldbc) compares a
candidate image with a baseline image and writes one `result.json`. The nightly
merges them into one `nightly.json`, which the reporter posts.
`src/falkorbench/contract.py` defines those two files and checks them: it
rejects a result or nightly that cannot be reported as it stands, i.e. a missing
or malformed field the reporter reads, or a verdict that contradicts its
findings. It is standard library only, so the reporting path never depends on a
package.

### The format

`result.json` (one per check):

```json
{
  "schema": 1, "check": "micro-hw", "verdict": "pass | regression | error", "confirmed": false,
  "summary": "one line in plain words",
  "candidate": {"image": "repo@sha256:<64 hex>", "commit": "<40 hex or null>", "label": "main@4c1088949"},
  "baseline":  {"image": "repo@sha256:<64 hex>", "commit": null, "label": "6.2.1"},
  "machine": {"runner": "gce", "type": "n4-standard-16", "cpu": "<model>", "cores": 16},
  "started": "2026-11-24T00:05:00Z", "finished": "2026-11-24T00:41:00Z",
  "findings": [{"name": "…", "metric": "p50_ms", "candidate": 13.3, "baseline": 0.4, "ratio": 33.25,
                "threshold": 1.5, "direction": "higher_is_worse", "status": "regression"}],
  "error": null, "artifacts": ["raw/rows.csv"]
}
```

`nightly.json` holds `schema`, `date`, `run_url`, `candidate`, `baseline`,
`previous_green_commit`, `status` (green, red, error or skipped), `skip_reason`
and `checks`: one entry per planned check, its `result.json` or, when it has
none, a stub `{"check", "verdict": "error", "error"}`.

The rules the contract adds to "every field present and well formed"
(`contract.CODES` lists every problem it can report):

* images are pinned by digest; commits are 40 hex or null;
* a `regression` verdict is confirmed and has a regression finding; `pass` and
  `error` carry none (a regression outranks an error); `error` has a reason and
  any other verdict has `error: null`;
* a run that did not finish (`completed` 0) or a broken invariant (`invariant`
  0) on the candidate is a regression, whether or not the baseline broke it too
  (v1.14, #4065: a night is green only when every pass/fail check holds on the
  candidate); a `rows` finding with direction `mismatch` whose sides differ is a
  regression;
* a result without `broken` was written before v1.14 and is checked by the rule
  then (0 on both sides is `info`), so a stored night stays valid;
* `broken` is optional (v1.14; older results lack it): one entry per `completed`
  or `invariant` regression finding, in the findings' order, with exactly `name`,
  `baseline_too` (the finding's baseline is 0), `issue` (the open issue that tracks
  it, a positive integer or null; reporting only, it never changes a verdict) and
  `introduced` (the release gate's rule: false when the baseline broke it too, or
  when the check's own rule says the candidate did not introduce it, such as a
  known flaky check). A break the baseline shares is never `introduced`;
* `incomplete` is optional: null, or a non-empty one-line string saying what was not
  measured or was cut (a regression outranks an error, so this is where a regressed
  result says its run was partial); validate checks only that it is a string or null;
* no `NaN` or `Infinity` (Python's `json.dump` writes them for a 0 baseline);
* at most 50 findings;
* a nightly's `status` is red if any check regressed, else error if any check
  errored, else green; `skipped` (with a reason) exactly when `checks` is empty;
  each check appears once, and a stub can only say error.

Keys the format does not name are ignored.

### Checking a file

```sh
uv run --project bench bench validate out/result.json          # any number of files
uv run --project bench bench validate --kind nightly nightly.json
```

It prints `<file>: <code>: <json-path>: <message>` per problem, or
`<file>: ok (<kind>)`, and exits 0 (all good), 1 (a problem) or 2 (usage). From
Python: `from falkorbench import contract`, then
`contract.validate_file(path, kind)` returns `(kind, [Problem(code, path, message)])`;
`validate_document` takes a parsed document, and `IMAGE_RE`, `COMMIT_RE`,
`METRICS`, `VERDICTS` and the other constants are the format's own.

### Acceptance pairs

`acceptance.yaml` lists, per check, image pairs whose right answer is known: the
same image compared with itself must pass; 6.2.0 against main@4c1088949 must
flag the rows #3963 fixed; 6.0.4 against 6.0.1 must catch the 3-hop read that
hangs. A check is accepted when an independent run of every pair (the acceptance
run) comes out as listed. The file's header defines each field.

`tests/test_contract.py` runs the checker over the example files
(`tests/fixtures/valid/`: 7 results and one nightly, plus the nightlies and
one-field broken variants `tests/fixtures/examples.py` builds from them) and
checks `acceptance.yaml` against the format and the query set.
`.github/workflows/perf-contract.yml` runs `bench/tests` on pull requests that
touch `bench/`.

### Shared check kit

`src/falkorbench/kit.py` holds what two or more checks would otherwise each
write: the flags, `result.json`, the threshold test and the docker calls. It is
functions only and standard library only. Each check keeps its own measuring
and judging. `args` is anything with the flags as attributes: argparse's
namespace from `add_args`, or `types.SimpleNamespace(**kwargs)` in a click command.

* Flags: `add_args(parser)`; `input_errors(args) -> [str]` (an image that is not
  pinned by digest, a bad commit, a label that is not one line, an unknown mode);
  `side(args, "candidate")`; `default_label(image)`; `out_dir(out) -> Path`
  (absolute path, `raw/` created); `machine()` (the four fields of a well-formed
  `$PERF_MACHINE_JSON`, else this host as `local`); `utc_now()`.
* Findings: `finding(name, metric, candidate, baseline, status, *, direction,
  threshold, ratio)` (it computes the ratio and turns NaN into null);
  `correctness_status(candidate, baseline)` (the 0/1 rule: 0 on the candidate is a
  regression, whatever the baseline did); `introduced(candidate, baseline)` (the
  release gate's default: the candidate broke what the baseline did not), which a
  check overrides per finding with the key `kit.INTRODUCED` (never written out);
  `crossed(candidate, baseline, threshold, floor, direction) -> +1 | -1 | 0`.
  A row crosses when worse/better > threshold *and* the difference > floor, both
  strict. `floor` is required: with a zero baseline the ratio is infinite, so the
  floor alone decides. A missing, negative or non-finite value never crosses.
* Result: `select(findings)` keeps regressions, improvements, then at most 10
  info rows, 50 in all, in the order given. `ok` rows stay in `raw/rows.csv`; a
  status the contract does not know raises. `verdict(findings, errors)` is
  regression > error > pass. `result(check, args, *, started, findings, summary,
  errors, artifacts, machine_info, issues)` also reports `input_errors`, and writes
  `broken` from the kept findings (`issues`: finding name to issue number). Then
  `error_result(check, args, started, error)` and `write_result(out, result)`
  (temp file + rename).
* Always a result: `run_check(check, args, work, build, cleanup=None)`.
  * It writes an error result first, so a SIGKILL still leaves one.
  * The first SIGTERM or SIGINT raises `Stopped`, a `BaseException`, so a
    check's `except Exception` cannot swallow it. Every later signal is ignored,
    so the check's own teardown and the final write finish.
  * `Stopped` (even one the check swallowed) or any exception, `sys.exit`
    included, becomes the error passed to `build(error)`. `build` turns what was measured so far into the result, so a
    confirmed regression still outranks the stop.
  * If `build` fails, says pass despite the error, or its result cannot be
    written, `run_check` writes `error_result` instead.
  * Then `cleanup()` runs and the old signal handlers come back. `args.out` is
    made absolute.
  * `stopping()` says whether a signal arrived, so a check can skip graceful stops.
* Docker (`DockerError` on any failure): `pull(image, attempts=3)`;
  `image_env(image)`; `start(image, name, *, env, volumes, labels, cpus,
  entrypoint, command, extra, port, host_network, foreground, output) ->
  Container(name, port, proc)`.
  * By default `start` runs detached with `-p 127.0.0.1::6379` and reads the
    port back with `docker port`, so docker picks a free one (no race, never 6379).
  * A server that exits at start comes back with `port` 0, and its container is
    kept for `state` and `logs`.
  * `host_network` takes the server's own port, which must not be 6379 or
    already be in use. `foreground` keeps `docker run` as a child process.
  * `ping(port)` and `wait_ping(port, timeout_s, container)`: one PING each
    time, never re-sent. The wait keeps to its timeout and stops early once the
    container is no longer running. A foreground `docker run` that docker itself
    refused (exit 125) raises `DockerError`: no server ran.
  * `logs(name, tail)` (stdout and stderr interleaved), `state(name)`
    (`running`, `exited 139`, `gone`), `host_pid(name)`, `remove(*names)` and
    `remove_labelled("key=value")`.

```python
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="bench example")
    kit.add_args(parser)
    args = parser.parse_args(argv)
    started, run_id = kit.utc_now(), str(os.getpid())
    cpus = os.environ.get("PERF_SERVER_CPUS")
    up, ports, p50 = {}, {}, []  # filled as work() goes; build() judges what it reached

    def work():
        if kit.input_errors(args):
            return  # result() reports them; nothing is pulled
        for side in kit.SIDES:
            image = getattr(args, side)
            kit.pull(image)
            server = kit.start(image, f"ex-{side}-{run_id}", labels={"ex": run_id}, cpus=cpus)
            up[side], ports[side] = int(kit.wait_ping(server.port, 120, server)), server.port
        if all(up.values()):
            for order in (kit.SIDES, kit.SIDES[::-1]):  # the re-run swaps the order
                times = {side: measure_p50_ms(ports[side]) for side in order}  # the check's own
                p50.append((times["candidate"], times["baseline"]))

    def build(error):
        findings = []
        errors = [error]
        if up and not all(up.values()):  # a side that never answered is not a comparison
            errors.append(f"not every server answered: {up}")
        if up:  # a server that never answered is completed 0, even if the run then failed
            c, b = up.get("candidate"), up.get("baseline")
            status = kit.correctness_status(c, b)
            findings.append(
                kit.finding("up", "completed", c, b, status, direction="lower_is_worse")
            )
        if p50:
            crossings = [kit.crossed(c, b, 1.5, 2.0) for c, b in p50]
            status = {1: "regression", -1: "improvement", 0: "ok"}[crossings[0]]
            if crossings[0] and crossings[1:] != [crossings[0]]:
                status = "info"  # the re-run did not repeat it: not confirmed
            findings.append(kit.finding("q1", "p50_ms", *p50[0], status, threshold=1.5))
        summary = f"{len(p50)} passes compared"
        return kit.result(
            "ex", args, started=started, findings=findings, summary=summary, errors=errors
        )

    kit.run_check("ex", args, work, build, cleanup=lambda: kit.remove_labelled(f"ex={run_id}"))
    return 0  # result.json was written
```

## Nightly report and missed-run alarm

Two commands tell the core team's Google Chat space how the nightly perf run went.
Both are standard library only (`src/falkorbench/nightly/`), like the contract.

| command | called by | does |
| --- | --- | --- |
| `bench nightly report --nightly <nightly.json>` | the report job of `perf-nightly.yml` | reports one night |
| `bench nightly missed-run [--date YYYY-MM-DD]` | `perf-missed-run.yml`, 08:30 UTC | alarms when a night left no result |

### `bench nightly report`: one night

The merge step validates `nightly.json` and sets its `status` before the report
runs, so the report trusts that status as written. It checks only what needs no
validator: a file it cannot read, a status that is not `green`, `red`, `error` or
`skipped`, and a crash of the report itself all report the night as ERROR, never
green.

* **Chat:** one line every night, posted to the incoming webhook in
  `$PERF_CHAT_WEBHOOK` when that is set, and printed to stdout, for example
  `Perf nightly 2026-11-24: RED (micro-hw), main@4c10889 vs 6.2.0 · details · dashboard · new issue`.
  The parentheses hold the regressed checks on a red night, the checks that could
  not measure on an error night (either list cut at a whole name, `+N more`, to stay
  under 300 characters), the reason on a skipped night, and the number of checks on a
  green night. "details" links to the GitHub run. "dashboard" links the run's page on
  the dashboard (`<site>/run/<date>-<run id>.html`, `$PERF_PAGES_URL` or the
  repository's github.io site) for a run the dashboard stores: the schedule, and a
  dispatch from the default branch that named no images (`$PERF_ON_DASHBOARD`). The
  page appears within minutes of the run's end, as the step summary says. A red night also gets
  "new issue": a new GitHub issue pre-filled from nightly.json only (title
  `Perf regression <date>: <checks>`, labels `performance` and `perf-regression`, a
  plain-text body with candidate vs baseline, the run link and each regressed
  check's summary), its body cut so the URL stays under 2,000 characters.
* **Run page:** `$GITHUB_STEP_SUMMARY`, when set, gets the details as Markdown: a
  row per check with its verdict and summary (or error), the worst regression rows
  of each regressed check (the first five, in the check's order), and on a red
  night the commits since `previous_green_commit` with their PR numbers (from
  `git log` in this checkout; each PR number is the last `(#N)` of the commit
  title) and the same pre-filled issue link.
* **Exit code:** 0 whenever it reported the night, whatever its status: the merge
  step owns the status and alone decides the run's colour. 1 only when it could not
  report: a file it cannot read, an unknown status, or its own crash (whose ERROR
  line is still posted first).

Text from the file or from commit titles cannot become a Chat link or mention (`<`
is replaced), Markdown is escaped, and a `run_url` that is not a plain `https://`
link of at most 300 characters is replaced by the current run's link.

A failed post is a warning; it never changes the exit code. A post is retried
twice on 429, 5xx and errors while connecting or sending, when nothing has arrived
yet. It is not retried after a timeout, a lost connection or a malformed answer
while waiting for Chat, because the message may have arrived. The webhook URL is
the credential and is never printed. Only `https://` webhooks are used.

### `bench nightly missed-run`: the 08:30 UTC alarm

It lists the scheduled runs of `perf-nightly.yml` on `main` created on `<date>`
between 00:00 UTC, when `perf-nightly.yml` starts, and 08:30, so a late run from the
night before cannot stand in for this one. The longest night its job caps allow
(resolve, the longest check chain, then merge and report) ends before 08:30; a test
adds up those caps from the workflow files and pins the gap. It is quiet when any of
those runs has an unexpired `perf-nightly` artifact; a skipped night uploads one
too, and a re-run of the night's jobs is the same run. A hand-started run does not count: one given its own
images uploads nightly.json but posts nothing. A run still going is "not yet", not
missed: the alarm says so in the job summary, posts nothing and exits 0, and the
run posts its own report when it ends. Otherwise it posts an alarm naming the case
and exits 1, so the run is red even without a webhook:

* no scheduled run in the window;
* the newest run finished (with its conclusion) without the artifact;
* `perf-nightly.yml` does not exist, or GitHub could not be asked.

A re-run of a scheduled alarm checks the night its run was created for. Each run
has its own concurrency group, so no check can displace another one that waits.
`--workflow`, `--branch`, `--event` and `--artifact` point it at other runs for an
acceptance run; the workflow takes `date` through `workflow_dispatch`.

`tests/test_nightly.py` has one test per rule, each failing when its rule breaks;
the webhook and the GitHub API are served from a local stub server. Try the example
red night by hand:

```sh
uv run --project bench bench nightly report --nightly bench/tests/fixtures/valid/nightly-red.json
```

## Dashboard (#4005)

`perf-dashboard.yml` runs when a `Perf nightly` run completes (`workflow_run`, from
the default branch, with its own permissions: `perf-nightly.yml` is also the release
gate's callee and holds no write grant). For a scheduled night, or a dispatch that
named no image of its own, it stores the run at `data/<date>-<run id>/` on the
`gh-pages` branch (the archive) with `bench nightly store`, rebuilds the site with
`bench nightly dashboard` and deploys it with `actions/deploy-pages`. A push
with `GITHUB_TOKEN` does not start a Pages build, hence the explicit deploy. It
never touches a night's verdict: the night has finished.

One-time admin step: **Settings → Pages → Source: GitHub Actions** (not "deploy from
a branch"), and let the `github-pages` environment accept the default branch.

**`bench nightly store [--nightly <nightly.json>] --run-id <id> --data <dir> --trigger T
--scope S [--full]`** downloads the run's `perf-nightly` artifact (unless `--nightly` is
given) and each check's newest `perf-result-<check>` artifact (`gh`, under a timeout, tried
again when a download stalls; after a re-run, the last attempt's) and writes
`<dir>/<date>-<id>/`: `nightly.json`, `run.json` `{"trigger": T, "scope": S}`, and per
check what its page reads or links: `result.json`, `raw/rows.csv` and `perf-vm.json` /
`perf-hosted.json`, of types `.json`, `.csv`, `.txt`, `.log` only. That is about 0.5 MB of
a full night's 80 MB (soak's per-scenario logs are most of it); the full files stay in the
run's artifacts for 90 days. `--full` stores every file each check lists instead (relative
paths only, none through `..`), for reading a run locally. A folder of the same run already
there is replaced, except by a store that could not download a check the old folder has:
then it changes nothing and exits 1. A `/perf` run lays out its own site with the same
command.

**Unlisted pages.** A dispatch that named its own images and a `/perf` run are not on the
runs page. Every deploy also builds, with `bench nightly unlisted --out <site> --days 30`,
one page for each such run created in the last 30 days, at `manual/<date>-<run id>/` and
`pr/<N>/<date>-<run id>/`. Each page is the run's page itself, with `noindex`, over the run
stored as above. No listed page links to them, and they are never stored on `gh-pages`: a
page is gone at the first deploy after its run is 30 days old. A run a deploy cannot rebuild
(a download that failed, or past the build's 20-minute budget) keeps the page the last deployed
site had for it (that deploy's Pages artifact, kept 35 days), so a link lives its 30 days. `bench nightly unlisted url`
prints a run's page address (`$PERF_PAGES_URL`, a repository variable for a custom domain,
else `https://<owner>.github.io/<repo>/`). The `/perf` answer and a manual run's summary link it.

**`bench nightly failures --data <dir> --run <date>-<run id> [--json]`** lists one stored
run's failures: every regression (including rows a capped `result.json` left in
`raw/rows.csv` only), every correctness check the candidate broke (also when it is broken on
the baseline too, a known flaky check, or a timing check that broke in some runs: each says
which), and every check that could not measure. Each is `new` or failing `since` a date
against the scheduled nights stored before it, and the previous night's failures that are
gone are listed as fixed. The core team's `/core:perf-run` skill (FalkorDB/handbook) uses it
to read any run locally (a night, a manual run, a `/perf` run, a release-gate run) and drafts
the missing issues; it posts nothing.

## `/perf` on a pull request (#4037)

A comment whose first line is

```
/perf <checks> [baseline <main | X.Y.Z>]      e.g. /perf micro-hw,pokec baseline 4.22.4
```

runs the nightly flow (`perf-nightly.yml`, called by `perf-comment.yml`) with the pull
request's `rc-pr-<N>-valkey` image as the candidate. `<checks>` is `all` or a comma list
of `micro-hw` and the names in `checks.yaml`. The baseline is `main` (main's newest
`rc-mg` image, found by the nightly's own walk; the default), a release `X.Y.Z` with
X >= 5 (`falkordb/falkordb-server:X.Y.Z`), or a C engine release `4.Y.Z`
(`falkordb/falkordb-server:v4.Y.Z`).

`bench nightly pr-comment resolve` decides whether it runs, and replies when it does
not: only people with write, maintain or admin access (bots are ignored), only a
pull request from this repository that is open, one `/perf` run per pull request at
a time, and the candidate only when `rust-pr.yml`'s Valkey `build-rc-image` succeeded
for the head commit, pinned by digest, with the head re-read after pinning (a head
that moved is refused). A request that runs gets a 🚀 reaction. When it ends,
`bench nightly pr-comment answer` posts one comment: each check's verdict, both
images and commits, the run, and the run's own dashboard (`bench nightly store` +
`bench nightly dashboard`, uploaded as the run's `perf-dashboard` artifact: open its
`index.html`). A run that left no `nightly.json` is answered as an error, never green.

A `/perf` run is a call, not a night: it posts nothing to Chat, never counts for the
skip rule or the previous green commit (those read only scheduled runs of
`perf-nightly.yml` itself), and never reaches the public dashboard
(`perf-dashboard.yml` runs only on completed `Perf nightly` runs). The checks run
from main's copy of the workflows and of `bench/`, as every `issue_comment` run does.

## Release gate (#4004)

A release is one pull request into a version branch X.Y that cherry-picks
commits from main; that pull request is the release candidate. The gate runs
only in the merge queue of X.Y: not on pull requests, not on main's queue, not
at tagging. It calls main's code (`@main`, because the perf code lives only
there) with the queue's image `rc-mg-<sha>-valkey`, in two jobs of `rust-pr.yml`
that start after its CI suites pass:

| job | runs | judges |
| --- | --- | --- |
| `perf` -> `perf-nightly.yml`, `nightly` mode, `gate: true` | the nightly flow: every registered check, gated, with the re-runs a night has | the image against the newest stable release below X.Y+1.0 |
| `perf-cov` -> `benchmark-cov.yml` | callgrind instructions and allocated bytes (`bench cov-gate`) | the same two images |

`gate: true` is what makes a call the release gate: another caller of
`perf-nightly.yml` (`/perf`) passes a `mode` too, but no `gate`, and is judged as a
night is. `ci-gate` requires both in the queue of a version branch and accepts `skipped`
everywhere else. The queue's required-check timeout must cover the CI and the
nightly together, about 180 minutes; an admin sets it in the branch's ruleset.

**`bench cov-gate`** turns benchmark-cov's PR-vs-base CSVs into a `result.json`
(check `benchmark-cov`, findings named `<row> [<reads|writes>]`). Instructions
cross at > 1.10x with no floor: callgrind repeats to about 0.2%, so this is the
instrument for #3996's class (1.30x on a query of about 240k instructions, under
micro-hw's 250k hardware floor). Allocated bytes use micro-hw's Rust rule (> 1.10x
and > 64 KiB per query, or > 8x and > 4 KiB). Both counters are deterministic for
an image, so there is no re-run. A side with no rows, or a row measured on the
base but not on the PR, is an error. On a run of an unchanged engine (benchmark-cov
run 45338149817, 111 callgrind rows, 373 allocation rows), instructions moved
0.995x–1.008x and allocations at most 3.08x at +9 KB: nothing crossed. A call
measures no C side and runs no coverage; the label's run on main is unchanged.

**The decision** is `bench nightly merge conclude --gate --waivers <bench/waivers.yaml
as JSON>` (on `nightly.json`, or `--result` on benchmark-cov's single result). It
passes a green night and a red night whose every regression finding is waived or,
for a pass/fail check, was not introduced by the candidate (result.json's `broken`
says `introduced: false`: the baseline breaks it too, or the check's own rule, such
as a known flaky check; #4065), and fails everything else: an unwaived regression, any check that said `error`,
a regression verdict with no finding to name, a regression from a check that did
not finish cleanly (merge's `unfinished.json`: it may have measured nothing after
the row it flagged), and a skipped night. A waiver names a check, a regex that
must match the whole finding name, an issue and a reason, and may narrow itself
to one `metric` and a `max_ratio`; a waived finding is still reported, as a
warning on the run page. The nightly's own run applies the same waivers. An
infrastructure error fails the gate like a regression: "Re-run all jobs" clears
it (see tests/perf/vm/README.md for why "Re-run failed jobs" does not).

A query a side does not answer is judged by Pokec's rule, from each side's
`bench measure --failed-out`: if neither side answers it (a query main added
that the line's engine lacks), it is info, "not supported on this branch". If
the base answers and the PR does not, it is a regression. If only the base
fails, it is info, "not measured on the base". A row missing with no such
record is unmeasured, an error. `bench cov-gate` also writes `unfinished.json`
when its reading is partial (a row the PR did not measure, or more regressions
than result.json lists), so a waiver never passes it.

A waiver never turns a night green when the result it would waive has a
non-null `incomplete` (a regression that also left parts unmeasured, or whose
findings the cap cut): that night fails, naming the check and the reason. A
result without the field (an older one) is judged as before.

## Nightly micro check (`micro-hw`)

`.github/workflows/_perf-micro.yml` takes a candidate and a baseline
falkordb-server image, both pinned by digest, and runs `bench micro-hw` on a GCE
c4-standard-8 with its PMU exposed, through `_perf-vm.yml` (`pmu: true`).
`perf-nightly.yml` calls it. The result is the nightly contract's `result.json`
with check `micro-hw`, in artifact `perf-result-micro-hw`.

```bash
bench micro-hw --bench-image ghcr.io/falkordb/falkordb-bench@sha256:… \
    --candidate IMAGE@sha256:… --baseline IMAGE@sha256:… --out out --mode nightly
```

**What it measures.** The full query set through `bench measure`, inside the
bench image (the same redis and jemalloc as `benchmark-cov`), with each image's
module copied out and loaded into it. Per query and per execution:

| metric | read from |
|---|---|
| instructions | `perf stat`: `instructions` |
| cycles | `perf stat`: `cycles` |
| branch_misses | `perf stat`: `branch-misses` |
| l2_misses | `perf stat`: `l2_rqsts.all_demand_miss` (demand requests that miss L2) |
| alloc_bytes | jemalloc, `MEMORY MALLOC-STATS` before and after |

The counters are read on the server process from inside the container, which
runs `--privileged`: Docker's default seccomp profile blocks `perf_event_open`.
The VM's PMU type is `standard`, which has "most documented core/L2 events" and
no last-level-cache events (gcloud's help for `--performance-monitoring-unit`),
so `cache-misses` reads `<not supported>` there. The L2 event is the one the C4
counted, and it is never swapped for another. Before the first pass a probe
needs every counter to read above 0; a host that cannot count one gives
`error`, never a pass on the rest.

**How it runs.**

- **Both images on one VM.** Candidate, baseline, baseline, candidate: two passes
  a side, order swapped once.
- **Threads pinned.** `THREAD_COUNT 4` and `OMP_NUM_THREADS=4`. Unpinned, the
  engine's pool and GraphBLAS follow the host, and per-thread workspaces made
  main's write rows allocate ~1.02 MB on a 16-vCPU VM against ~36 KB on a 4-vCPU
  runner. Pinned, the C4 matches a hosted runner: median 1.000 over 373 rows.
- **A row that regresses is measured again.** The whole set runs a second time,
  baseline first, and the row is a regression only if that second run crosses
  too; one that crossed once is `info`. A single query run alone would not
  rebuild the graph state the full set measured against. Improvements are
  reported as measured, not re-run, and never fail the run.
- **A query that errors on the candidate in every pass** is a regression at
  once (metric `completed`), named `<query> [errors on candidate only]`, or
  `<query> [errors on both sides]` when the baseline errs too, always or in some
  passes (#4065: the release gate does not block on that one). One that errors
  on the candidate in some passes only, in either run, is `info`, and so is one
  only the baseline fails sometimes. A pass in which every query errored still
  counts as a pass. A row whose prerequisite (`needs`) errored on one side only,
  or in some passes, is not compared: the two sides measured different graphs.
- **A gap is never a pass.** A missing pass, a row missing a reading in a pass
  it ran in, or a crossed row the second run could not fully measure makes the
  verdict `error`, unless a confirmed regression outranks it. A measured run
  that does not finish is a failed query, never a row.
- **Per-class rows.** A small fixed cost on every query crosses no per-row
  threshold, so reads and writes each get an `info` finding with the median
  per-row ratio and how many rows went up and down, for instructions, cycles and
  allocated bytes.
- **The C engine.** When either image is C, both sides run `--c-compat` (both
  build the same graph) and the C gates below apply.

**The gates.** +10% on instructions, cycles and allocated bytes, +25% on the
miss counters (bench's thresholds), with the passes not overlapping and the
second run repeating it, and an absolute change per execution of more than the
floor; or a steeper ratio over a smaller floor. Every bound is strict: a row
exactly on one does not cross. Every value sits above the worst
split two runs of the same image could produce: the top two of a row's four
readings against the bottom two, over runs 45376924910, 45376924661, 45403484617
and 45403484657 (Rust) and 45376924684 and 45403484855 (C), first and second
runs.

| engine | metric | floor (worst same-image change past the threshold) | steep tier | its basis: worst same-image ratio of a change at least that floor | margin |
|---|---|---|---|---|---|
| Rust | instructions | 250,000 (+103,619, edge index lt range 1.49x) | — | | |
| Rust | cycles | 20,000,000 (+9,781,665, labelPropagation all types 1.113x) | 1.8x over 250,000 | 1.113x (labelPropagation all types) | 1.62x |
| Rust | branch_misses | 12,000 (+5,598, maxFlow KNOWS 1.27x) | 2.8x over 2,000 | 1.716x (WCC typed) | 1.63x |
| Rust | l2_misses | 225,000 (+111,503, labelPropagation all types 1.82x) | 3.0x over 15,000 | 1.816x (labelPropagation all types) | 1.65x |
| Rust | alloc_bytes | 65,536 (+17,655, create delete indexed 2.16x) | 8x over 4,096 | 2.620x (delete path) | 3.05x |
| C | instructions | 1,300,000 (+629,577, cross product filter 3.66x) | — | | |
| C | cycles | 270,000 (+134,843, cross product filter 1.68x) | — | | |
| C | branch_misses | 40,000 (+20,159, betweenness default 1.31x) | — | (algo.pageRank 3.58x at +10,998 rules one out) | |
| C | l2_misses | 665,000 (+332,270, algo.labelPropagation 1.53x) | 3.1x over 15,000 | 1.534x (algo.labelPropagation) | 2.02x |
| C | alloc_bytes | 3,100,000 (none past 1.10x; betweenness default 1.097x at +1,537,039) | 8x over 4,096 | 1.097x (betweenness default) | 7.3x |

A steep tier sits at least 1.6x above its basis. The C engine has its own set
because its worker thread's busy-wait is counted here: it moved `cross product
filter` between 237,470 and 866,391 instructions on one image. C's allocations
are far steadier than Rust's, but one large row came within 0.3% of the
threshold, hence its 3.1 MB floor. Two Rust rows are bimodal by pass position
(`edge index lt range` reads ~314k instructions in passes 1-2 and ~210k in 3-4);
the interleaved order and the floors absorb that.

**What the gates cost.** A change smaller than a metric's floor flags only at its
steep ratio, and a change smaller than the steep floor never does: cycles under
+20M need 1.8x, branch misses under +12,000 need 2.8x, L2 misses under +225,000
need 3x, allocations under 64 KiB need 8x (none under 4 KiB), and instructions
under +250,000 do not flag. So small-query instruction regressions are not caught
at night; `benchmark-cov`'s per-PR callgrind table is where they show. On 6.2.0 vs
main, `UNWIND index MATCH edge 1 row` (2.72x branch misses) does not flag on
branch misses alone; its cycles (4.3x), instructions, L2 misses and allocated
bytes do. L2 demand misses are the noisiest counter: 61% of same-image
row-splits pass 1.25x.

**Modes.** `queue`, `nightly` and `long` are accepted and checked; today they are
one parameter set: the full query set, two passes a side, one second run.
Measured on the acceptance runs: about 15 minutes on the VM when nothing crosses,
29 when the set runs a second time, and 38 for two C images; creating the VM adds
about 2.

**Its limits:**

- It copies each image's module into the bench image's own redis-server, so
  both are measured with the same redis and the same jemalloc. That measures the
  module. A server or Valkey version change inside an image does not show here;
  the other VM checks, which run each image's own server, cover that.
- A night with no C4 capacity in any zone gives no result, which reads as
  `error`, never as a pass.
- A server crash reads as error, never pass. The contract would call a crash a
  regression; telling a candidate crash from an infrastructure failure needs
  crash reporting inside `bench measure`, which it does not have yet.

## Layout

| file | purpose |
|---|---|
| `src/falkorbench/model.py` | `Query`, `Metric` — the value types |
| `src/falkorbench/queries.py` | the canonical query set + graph SETUP (10k Person ring, 10k KNOWS, indexes, constraints, UDFs); `cg=True` marks the callgrind subset |
| `src/falkorbench/metrics.py` | CSV parsing, ratios, thresholds, control-row normalisation — shared by compare and report |
| `src/falkorbench/client.py` | server lifecycle and control plane, over falkordb-py |
| `src/falkorbench/counters.py` | instruction/cycle backends (rusage / perf / none) + `pmc_tool` |
| `src/falkorbench/measure.py` | the full-set measurement loop |
| `src/falkorbench/callgrind.py` | deterministic counts by differencing two instrumented runs |
| `src/falkorbench/compare.py` | local regression gate |
| `src/falkorbench/report.py` | the CI comment, and the run's pass/fail decision |
| `src/falkorbench/verdict.py` | the nightly micro check's `result.json`: its gates, and the crossing, second-run and correctness rules |
| `src/falkorbench/hw.py` | `bench micro-hw`, the nightly micro check, run on a VM with a PMU |
| `src/falkorbench/profile.py` | samply profile of one query |
| `src/falkorbench/flow.py` | per-flow-test-file measurement |
| `src/falkorbench/cli.py` | the `bench` command |
| `src/falkorbench/contract.py` | the nightly perf checks' `result.json` / `nightly.json` format and its checker (`bench validate`) |
| `src/falkorbench/kit.py` | what the nightly perf checks share: flags, `result.json`, the threshold test, docker |
| `src/falkorbench/pokec/` | `bench pokec`, the nightly Pokec check (FalkorDB/benchmark under concurrent writes); its README is there |
| `src/falkorbench/nightly/` | `bench nightly`: the night's bookkeeping, report (Chat line and run page), missed-run alarm, dashboard (`store`, `dashboard`) and the `/perf` comment (`pr-comment`) |
| `src/falkorbench/coverage.py` | instrumented build, run the set once, report graph-crate line coverage |
| `src/falkorbench/ldbc/` | LDBC SNB Interactive v1 — dataset fetch/prepare, loader, parameters, the nightly check, and the vendored query texts |
| `Dockerfile` | the CI measurement image, `FROM …:edge-c` |
| `pmc_tool.c` | Apple Silicon PMU counters (kperf/kperfdata private frameworks) |
| `acceptance.yaml` | the nightly perf checks' acceptance pairs |
| `tests/` | the guards above, over CSV fixtures |

## pmc_tool (optional, for branch/L1D columns)

```bash
clang -O2 -o bench/pmc_tool bench/pmc_tool.c \
  -F /System/Library/PrivateFrameworks -framework kperf -framework kperfdata
sudo chown root:wheel bench/pmc_tool && sudo chmod u+s bench/pmc_tool
```

Without it, instructions, cycles, allocated bytes and wall-clock are still
measured, and the branch/L1D columns stay empty. With it, those columns are
filled in **and gated** like the rest (`GATED_BY_DEFAULT` in `metrics.py`; only
`ms` is left out). PMU numbers are system-wide and include the `redis-benchmark`
client, so on a busy machine they can fail the gate on their own: treat values
under ~1K/query as noise, and narrow the gate with `--metrics
instr,cycles,alloc_bytes,dealloc_bytes` when they do.

`pmc_tool window` opens a counter window, prints `READY`, and waits on stdin; the
harness runs the measured command itself in that gap and then closes the window.
It deliberately does not run the command for you — it is installed setuid-root,
and a setuid binary that execs a caller-supplied command is a local privilege
escalation (put your own `redis-benchmark` earlier in `$PATH` and you have root).
Not exec'ing at all removes the whole class of bug, and the counters are
system-wide so bracketing in time is all that was needed.

## Reading the output

- **Trust the instructions column.** Cycles and wall-clock move 10-60% with
  machine load; on a shared or virtualised host they will invent regressions.
  Always include `RETURN 1` in an isolated batch: it is the fixed per-query
  floor, so if it moves, the whole run's cycle column is load-inflated and every
  other cycle flag in it is void.
- **Micro-queries need 3 reps per build.** Rows near the instruction floor (what
  `RETURN 1` costs: about 175k on Apple Silicon)
  read 1.07x on one shot and ~1.00x over three. The error is *absolute*, so a
  fixed percentage tolerance is simultaneously too strict for expensive queries
  and too lax for cheap ones.
- **Ignore rows where C's instruction count is ~500-2500** — those are rows the C
  engine errors on (regex, week/ordinal dates, `LOAD CSV*`, toJSON scalars), so
  the ratio is meaningless. `--c-compat` skips the ones whose warmup reply errors.
- **Watch for near-zero ratios in Rust's favour.** `cross product filter` and
  `untyped shortestPath` have C burning 1.8B and 3.8B instructions against ~400k
  — a regression there would be invisible as a ratio and obvious in absolute
  terms.
- **id-0 rows against C are suspect**: C's `DEBUG RELOAD` drops id 0 from the
  range index. Re-measure on a fresh C server before believing one.

### How this relates to the team performance toolbox

The [performance toolbox](https://aviavni.github.io/database-learning-path/topics/00-performance-toolbox/index.html)
prescribes criterion for microbenchmarks, samply for profiling, `dhat-rs` for
allocations, and confidence intervals over point estimates. Where this harness
agrees, it agrees for the same reasons: "single-shot timing is fiction" is why
wall-clock never gates here; the idle-rate calibration is "keep the machine
idle"; and the callgrind path **drops** any row it cannot resolve to better than
2% rather than print it, which is that confidence-interval discipline applied to
instruction counts.

Three deliberate deviations:

1. **No criterion.** criterion measures in-process Rust functions; this measures
   whole-query cost through the redis protocol, which criterion cannot reach.
   They are complementary — criterion belongs on `graph`-crate internals, and
   there is no criterion suite in this repo yet.
2. **Allocations from jemalloc merged-arena deltas, not `dhat-rs`.** No
   instrumented build needed, works on any stock jemalloc redis, and measures the
   real server process — which is what makes the vs-C comparison possible.
3. **`ms` is subject to coordinated omission, by construction.**
   `redis-benchmark -c 1` is a closed-loop generator: send, await reply, send
   next. A stall backs up the generator and disappears from the data, exactly as
   the toolbox warns. It does not touch the instruction or allocation columns — a
   stall cannot hide an instruction — and it is acceptable *only* because `ms` is
   never the gate, just a coarse outlier net.

`ms` is also a **mean**, not a percentile, while the project's stated bar (see
CLAUDE.md) is p99 latency. `redis-benchmark` computes a latency distribution and
this harness currently discards it, so the p99 bar is not measured here yet.

## Known gaps

One structural gap is worth recording because closing it needs a design decision
rather than a patch: the MVCC copy-on-write `GrB_Matrix_dup` of delta matrices in
`create_nodes` / `set_nodes_labels_bulk` waits on pending work first, so it scales
with the accumulated delta (up to the 10k flush threshold, avg ~5k) per query,
independent of batch size. C never merges pending tuples on the write path. That
is what keeps the create/delete rows above 1.0x against C.

**A ranked Rust-vs-C table is deliberately not kept here.** It goes stale the
moment anything merges, and a stale ranking is worse than none — it sends people
to work on rows that are already fixed. Generate it from a live run using the
recipe above.

**Coverage**: the query set covers **74.8%** of graph-crate lines
(28,869/38,573, excluding the generated `GraphBLAS.rs` FFI; measured in CI on
2026-08-05). The 0%-coverage areas need infrastructure a Cypher
query cannot reach from this graph: `cow_btree` (~750 lines, appears unwired),
`string_pool`, `vec_distance`, and most of `algo_procedures.rs`. For the hot paths
this set targets (runtime, expressions, planner, matrices), coverage is high.
`bench coverage` reports the number but does not enforce a floor — it is a
validator of the query set, not a coverage gate.

## LDBC SNB (Interactive v1)

A second, complementary workload: the 14 **complex reads** of the LDBC Social
Network Benchmark, run against the real SNB dataset. The micro-benchmark set
above measures narrow operations on a synthetic ring; these measure whole
realistic queries — multi-hop traversals, aggregation, `OPTIONAL MATCH`,
variable-length paths — on a graph with realistic degree skew. A planner
regression that a single-op query cannot see tends to show up here. It runs as
`bench ldbc check`, one of the nightly perf checks (below).

### This is not an auditable LDBC result

Do not publish these as an "LDBC score". A real result requires the official
Java driver with its validation and workload-generation phases, LDBC
membership, and an audit commissioned from a certified auditor. What this gives
you is an internal, repeatable number on a standard dataset and standard
queries.

### Five queries differ from the reference text

Each departure is annotated in the `.cypher` file next to the line it replaced,
and listed in `ldbc/queries.py::REWRITES` so a run always prints them. All five
were confirmed against a running engine, and each was checked against the C
engine too so that a genuine bug is not filed away as a dialect gap:

| query | why it could not run as written |
|---|---|
| IC1, IC13 | `MATCH path = shortestPath(...)` → *"FalkorDB currently only supports shortestPaths in WITH or RETURN clauses"*. Moved into `WITH`. |
| IC14 | `allShortestPaths()` with inline endpoint patterns → *"Source and destination must already be resolved"*. Endpoints pre-bound in a preceding `MATCH`. Also, a pattern comprehension's `WHERE` cannot see the enclosing list comprehension's variable, so the per-relationship weights are computed over `UNWIND`ed relationships with their endpoints hoisted into plain variables. |
| IC10 | `datetime({epochMillis: ...}).month` → *"Unknown function 'datetime'"*. The loader derives `birthdayMonth`/`birthdayDay` instead. |
| IC7 | `not((liker)-[:KNOWS]-(person))` → *"Type mismatch: expected Boolean or Null but was List"*. A traversal pattern is not a boolean-valued expression in a projection, and `exists()` explicitly refuses traversal patterns, so this becomes `size(<pattern>) = 0`. |

In their upstream form, IC1, IC7, IC13 and IC14 fail identically on the C
engine, so those four are dialect gaps rather than regressions. With the
rewrites, C runs IC7 but still not IC1, IC13 or IC14, nor IC10 (the nightly
check below lists each error). IC6 and IC9 were rewritten too at first,
and were not dialect gaps: both ran correctly on C and hit two Rust
wrong-result bugs ([#2556](https://github.com/FalkorDB/FalkorDB/issues/2556),
[#2557](https://github.com/FalkorDB/FalkorDB/issues/2557)). Both are fixed, and
both queries run the upstream text again. That distinction only exists because
every failure was re-checked against `falkordb/falkordb-server:edge-c` before
being classified — without that step two real regressions would have been
written off as dialect gaps, and two dialect gaps filed as false bugs.

`CREATE CONSTRAINT ... ASSERT n.id IS UNIQUE` is also unsupported, so upstream's
`indices.cypher` becomes `GRAPH.CONSTRAINT CREATE` calls — each of which
additionally requires its exact-match index to already exist, and validates
asynchronously (the loader waits for `OPERATIONAL` rather than assuming it;
note `db.constraints()` reports `UNDER CONSTRUCTION`, never `PENDING`).

The CSVs are pipe-separated. FalkorDB has no `DELIMITER` keyword — the
standard-Cypher spelling is `FIELDTERMINATOR`, and it goes *after* `AS <var>`.

### The nightly check: `bench ldbc check`

The nightly perf run (epic #3974) compares main's newest image with the latest
release through several independent checks; this is the LDBC one. It is one
command, with the contract flags every nightly check shares (`kit.add_args`):

```bash
uv run --project bench bench ldbc check \
  --candidate ghcr.io/falkordb/falkordb-server@sha256:… \
  --baseline falkordb/falkordb-server@sha256:… \
  --out out --mode nightly \
  [--candidate-label main@4c10889] [--baseline-label 6.2.0] \
  [--candidate-commit <40 hex>] [--baseline-commit <40 hex>] \
  [--cache DIR] [--param-count N] [IC2 IC7 …]
```

`--cache` (default `bench/ldbc-data`), `--param-count` and a query subset are
for development runs; `bench/checks.yaml` runs it with none of them.

Both arguments are images pinned by digest, and nothing is compiled: each image
runs as its own server in docker (its own redis or valkey and allocator),
side by side on this machine, started with `kit.start` on ports docker picks
(never 6379). The dataset is bind-mounted read-only as the module's
`IMPORT_FOLDER`. On the VM the servers
are pinned to `PERF_SERVER_CPUS`; locally they run unpinned and the result says
`runner: local`.

**How it measures.** The dataset is loaded into the baseline, then the
candidate. Parameters are sampled once, from the baseline, and both sides run
the same rows (`raw/params.json`). Per parameter row the two servers run back to
back, in two rounds with the order swapped, after one warm-up call per query
and side. Each latency is one `GRAPH.RO_QUERY` on a connection that never
re-sends (redis-py's retries are off), with the query's server-side `TIMEOUT`.

**What it reports** (`<out>/result.json`, contract schema 1), per query:

| finding | metric | when |
|---|---|---|
| `IC3 p50`, `IC3 p95` | `p50_ms`, `p95_ms` | the ratio candidate/baseline over the requests both sides finished (the same parameter row in the same round) is more than 2x **and** the difference is more than 0.5 ms (p50) / 2 ms (p95). A crossing is measured again (two more swapped rounds); it is a `regression` only when the re-run crosses too; otherwise it is no finding, and `raw/rows.csv` keeps the re-run as "not repeated". A re-run that returns no rows, or does not time every row the main rounds timed, settles nothing and makes the result an `error`. Wall clock, hence the wide band. |
| `IC9 rows` | `rows` (`mismatch`) | the two sides return a different number of rows for the same request (parameter row and round, re-runs included). A regression at once, and the query gets no latency finding: the two sides did different work (6.0.4's IC9 "ran" 93x faster by returning nothing, #2557). |
| `IC3 completed` | `completed` | judged per request (parameter row and round): the candidate timing out, erroring or losing its server on a request is a regression at once, whether or not the baseline failed it too (#4065; baseline 0 then). The release gate counts it only on a request the baseline finished. A parameter row that timed out is skipped on both sides from then on, and a side stops a query after two timeouts. A query the baseline cannot run at all (a dialect gap) is left out: `info`, the candidate not judged (null). |
| `server up` | `completed` | a server that went down at any point, also after its last reply (its container exited): the candidate's is a regression, the baseline's makes the result an `error`. |
| `IC6 returns rows` | `rows` (0 vs 0, `mismatch`) | zero rows on every parameter row, on both sides: it measured nothing (`info`), and its latency is not compared. |
| `load` | `throughput` | entities loaded per second, per side: one sample each, so `info` only. |

Row contents are not compared, only row counts. A digest of each request's
rows (sorted, with doubles rounded so the engines' float spelling does not
count) is a possible follow-up; it would only ever be `info`, and across
engines it has to allow for the C engine's `LOAD CSV` dropping a field's
trailing whitespace (#4003).

**Thresholds.** The ratio and the floors come from three A/A runs (main@4c1088949
against itself twice, C v4.22.4 against itself once; nightly mode, unpinned,
on an n4-standard-16 dev VM). The worst ratio either way was 1.19x (IC11 p95),
the worst p50 difference on a query under 5 ms 0.061 ms, and the worst p95
difference on a query under 10 ms 0.65 ms; 2x is 1.68x above the first, 0.5 ms
8x the second, 2 ms 3x the third. The judge's pinned A/A runs on the VM
workflow are the independent check of these values.

**Queries an engine cannot run** are left out rather than failing the check: a
query on which the *baseline* errors on every parameter row is reported as one
`info` finding (`completed`, baseline 0) and is not compared. On the C engine
that is IC1, IC10, IC13 and IC14, measured on v4.22.4 (C against itself, every
parameter row):

| query | the C engine's error |
|---|---|
| IC1, IC13 | *"FalkorDB does not currently support undirected shortestPath traversals"* |
| IC14 | *"allShortestPaths(...) does not support a minimal length different from 1"* |
| IC10 | *"Type mismatch: expected Null but was Pointer"* |

The same query failing on the *candidate* only is a regression.

**Verdict.** `regression` (confirmed) if any finding is one; otherwise `error`
when something could not be measured (an image not pinned by digest, a pull or
a dataset download that failed, a baseline that did not start, load or stay
up, no query that finished on both sides, a deadline or a per-round budget
that cut any query's rounds); otherwise `pass`. The summary counts as compared only the queries
with at least one parameter row finished on both sides. The run goes through
`kit.run_check`: `result.json` is written first as an `error` and replaced at
the end, a SIGTERM still leaves one with what was measured so far, and both
containers are removed afterwards. The command exits 0 whenever `result.json`
was written.

**Modes** set only the length, never the method:

| mode | dataset | parameter rows per query | rounds (+ re-run) | per-query timeout | deadline |
|---|---|---|---|---|---|
| `queue` | SF0.1 (18 MB) | 10 | 2 (+2) | 30 s | 25 min |
| `nightly` | SF0.1 (18 MB) | 25 | 2 (+2) | 60 s | 35 min |
| `long` | SF1 (230 MB) | 25 | 2 (+2) | 300 s | 150 min |

How long it takes, measured on an n4-standard-16 dev VM (unpinned), with the
settings above: main@4c1088949 against itself took 13.5 min in `nightly` and
7.7 min in `queue`. Earlier runs, before the per-round budget was raised to
300 s: C v4.22.4 against itself 3.7 min, 6.0.4 against 6.2.0 (nine
regressions confirmed, so nine re-runs) 20.6 min. Most of it is the two loads
(about 70 s each) and the queries that time out (IC3 on every row, #4001, and one IC14
row, on the Rust engine). The 35-minute deadline stops new measurement well
inside the 45 minutes the check gets in `bench/checks.yaml`. `long` (SF1)
is accepted and set as above but not yet run end to end.

The queries run in a fixed order (IC1 to IC14), every query's main rounds
before any re-run. If several queries are pathologically slow, the last ones
in that order (IC12 to IC14) can miss the deadline before their first round;
they are reported as not measured, so the night is `error` or `regression`,
never `pass`.

**Artifacts** under `<out>/raw/`: `rows.csv` (every latency comparison, crossing or not; findings carry only regressions, improvements and `info`), `samples.csv` (every request: phase, round,
query, parameter row, side, outcome, latency, rows), `summary.csv`
(per query and side), `params.json`, `load.json`, `explain/<side>/<IC>.txt`
(`GRAPH.EXPLAIN` of every query on both sides), `<side>-server.log`, and
`run.json` (mode, thresholds, servers, rewrites, notes).

**Tests.** `bench/tests/test_ldbc.py` covers the analysis, the measurement
loop (with scripted servers and kit's docker calls replaced) and every result
shape, holding each result to `falkorbench.contract`. `perf-contract.yml` runs
them with the rest of `bench/tests`.

### Substitution parameters

The check samples the loaded graph with a fixed seed, so two runs compare
against each other; the numbers are not comparable to published LDBC results.
Sampling draws from where the data is dense: people from the most-connected end
of the `KNOWS` degree distribution, and dates from the last quarter of the
corpus. IC3 gets a much wider window than the rest: it counts friends who are
*not* resident in either country yet posted from both inside the window, and at
SF0.1 a 30-day window returns 0 rows where 365 returns 1 and 730 returns 2, on
both engines. `durationDays` is an explicit LDBC parameter, so widening it stays
inside the query's own contract. A uniformly drawn person is usually isolated
and a uniformly drawn window usually lands in the sparse early history; either
returns nothing in microseconds and measures none of the work.
