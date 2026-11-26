---
name: perf-run
description: Read one perf run locally — a Perf nightly (scheduled or manual), a `/perf` run on a pull request, or a version-branch release-gate run — given its link, its id or "last night". Lays the run out with `bench nightly store`, builds its dashboard page with `bench nightly dashboard`, lists every regression and every correctness check broken on the candidate (`bench nightly failures`), compares them with the previous scheduled nights, matches each failure to an issue and drafts the missing ones. Posts nothing. Use when asked what a perf run found, why a night is red, or to triage a /perf or gate run.
---

# Reading a perf run

Everything here reads; nothing posts. No issue, comment, reaction or Chat message: the
issues it finds missing are drafted as text in the report, for a person to file.

The pipeline it reads is described in `bench/README.md` ("Nightly report", "Dashboard",
"`/perf` on a pull request", "Release gate") and in the header of
`.github/workflows/perf-nightly.yml`.

## The rule for a verdict

**Green means every test passes, whatever its issue status.** A correctness check broken on
the candidate is a failure in this report even when it is broken on the baseline too, is a
known flaky check with an open issue, broke in only some of the runs, or the check's own
verdict is `pass`: say which of those apply, never leave it out. A row missing from a check's
findings is not unmeasured: the check kept it in its raw files (ok rows, info rows past the
cap, known flaky rows), which is why `bench nightly failures` reads them.

## 1. Find the run

```bash
REPO=FalkorDB/FalkorDB
# A link https://github.com/FalkorDB/FalkorDB/actions/runs/<id>[/job/...] -> its <id>.
RUN=<id>
# "last night": the newest completed scheduled Perf nightly on main.
RUN=$(gh api "repos/$REPO/actions/workflows/perf-nightly.yml/runs?event=schedule&branch=main&status=completed&per_page=1" --jq '.workflow_runs[0].id')

gh api "repos/$REPO/actions/runs/$RUN" \
  --jq '{name, path, event, head_branch, display_title, status, conclusion, created_at, html_url}'
```

A run still in progress has no result yet: say so and stop. Pick the trigger and scope the
dashboard files the run under from the run's workflow (`path`), then its `event`:

| workflow | event | trigger | scope |
| --- | --- | --- | --- |
| `perf-nightly.yml` ("Perf nightly") | `schedule` | `nightly` | `main` |
| `perf-nightly.yml` | `workflow_dispatch` | `manual` | `main` |
| `perf-nightly.yml` | anything else (a branch's test run of the workflow, measuring main's images) | the event | `main` |
| `perf-comment.yml` ("Perf PR comment", a `/perf` run) | any | `pr-comment` | `pr-<N>` |
| `rust-pr.yml` (the release gate, in a version branch's queue) | `merge_group` | `release gate` | the branch, e.g. `6.2` (`head_branch` is `gh-readonly-queue/6.2/...`) |

For a `/perf` run, take `<N>` from the candidate's label in its nightly.json (`pr-<N>@<sha>`,
step 2): that is the pull request it measured. Its title (`/perf on #<N>`) says the same for a
comment, but `gh run view` may show another pull request (the one of the branch the workflow
file came from), which is not the one measured. Only scheduled nights count as nights below;
every other trigger is a one-off comparison of the images it names.

"Last night" may be a skipped night (main had not moved: its nightly.json has `status:
skipped` and no checks). Say so, then read the newest scheduled night before it that measured.

## 2. Lay it out and build its page

Work in a scratch folder (your session's temporary folder if it names one):

```bash
W=${TMPDIR:-/tmp}/perf-run/$RUN
rm -rf "$W" && mkdir -p "$W/data"
```

First the previous scheduled nights, then the run itself (it replaces any copy of itself).
Once the gh-pages branch exists it holds every stored night:

```bash
if git ls-remote --exit-code --heads "https://github.com/$REPO.git" gh-pages >/dev/null; then
  git clone --quiet --depth 1 --filter=blob:none --sparse --branch gh-pages \
    "https://github.com/$REPO.git" "$W/pages"
  git -C "$W/pages" sparse-checkout set data
  cp -R "$W/pages/data/." "$W/data/"
else
  # No archive yet: up to three scheduled nights before this run, from their own artifacts
  # (a night whose artifacts are gone is passed over).
  before=$(gh api "repos/$REPO/actions/runs/$RUN" --jq .created_at)
  stored=0
  for id in $(gh api "repos/$REPO/actions/workflows/perf-nightly.yml/runs?event=schedule&branch=main&status=completed&per_page=100" \
      --jq ".workflow_runs[] | select(.created_at < \"$before\") | .id"); do
    uv run --project bench --frozen bench nightly store \
      --run-id "$id" --data "$W/data" --trigger nightly --scope main || continue
    stored=$((stored + 1)); [ "$stored" -ge 3 ] && break
  done
fi
```

Then the run (`TRIGGER` and `SCOPE` from the table). `bench nightly store` downloads the run's
`perf-nightly` artifact (its nightly.json) and every `perf-result-<check>` artifact itself (100 MB
is not unusual), each under a timeout and tried again when it stalls:

```bash
FOLDER=$(uv run --project bench --frozen bench nightly store \
  --run-id "$RUN" --data "$W/data" --trigger "$TRIGGER" --scope "$SCOPE") \
  || FOLDER=$(find "$W/data" -maxdepth 1 -name "*-$RUN" -exec basename {} \; | head -1)
[ -n "$FOLDER" ] || { echo "run $RUN could not be stored: read the error above"; exit 1; }
uv run --project bench --frozen bench nightly dashboard --data "$W/data" --out "$W/site"
echo "run page: $W/site/run/$FOLDER.html"
```

Read what the store prints. `no perf-result-<check>` means that check's files are not here:
step 3 reports the check as an error, never a pass. If the store fails but the run is already on
gh-pages (its artifacts expired), the second line takes the stored copy. Otherwise its error says
which case it is:

- **the run has no perf-nightly artifact**: it never merged its checks (cancelled, or its resolve
  job failed), or the artifact expired. Report that as an error for the whole run, with its failed
  jobs (`gh run view "$RUN" -R "$REPO"`), and stop.
- **the artifact did not download**: a network failure, not a fact about the run. Run the store
  again; never report the run as broken for it.

The run page's headline is nightly.json's status (today `green` while correctness checks are
broken on both sides): the report states its own verdict by the rule above.

## 3. Every failure, and how it compares with the nights before

```bash
uv run --project bench --frozen bench nightly failures --data "$W/data" --run "$FOLDER"
uv run --project bench --frozen bench nightly failures --data "$W/data" --run "$FOLDER" --json > "$W/failures.json"
```

It lists, per check, three kinds of failure, read from each check's `result.json` and raw files:

- **regression**: a row the check judged a regression, also one its findings cap left in
  `raw/rows.csv` only;
- **broken**: a correctness check the candidate broke (an invariant, a query or run that did not
  complete, rows that differ, a micro-hw query that errors), each saying all of what applies:
  broken on both sides or the candidate only, a known flaky check (with its issue), a timing
  check, broken in k of n runs;
- **error**: a check that could not measure, or whose files are not stored.

A check line also shows `incomplete:` when the check says it measured or kept only part of
its rows. Each failure is `new`, `since <date>` (failing in every scheduled night that measured,
back to that date) or `no earlier night`; skipped nights are looked through. The list ends with
the previous night's failures that are gone ("fixed"), for the checks this run measured. When the
baseline differs from the previous night's (a `/perf` run, a manual run, a gate run) it says so:
a failure can be new because the pair changed, not the candidate.

**When the candidate and the baseline are different lines** (a `/perf` run against `main`, a
gate run, a manual run naming its images), check how far the candidate is behind the baseline
before blaming the candidate:

```bash
gh api "repos/$REPO/compare/<candidate commit>...<baseline commit>" \
  --jq '{status, ahead_by, behind_by, commits: [.commits[] | .commit.message | split("\n")[0]]}'
```

`ahead_by` commits are on the baseline and not in the candidate: a regression may be a fix the
candidate lacks (a stale pull request branch), not something it broke. Say so per failure. The
count is by commit, so against a release-line baseline it also counts backports whose originals
the candidate has: look the baseline-only commits up by pull request number in the candidate's
history (`git log <candidate commit> --oneline | grep '(#<n>)'`, after `git fetch origin
<candidate commit>`) before writing "<n> commits behind".

The strongest evidence about a row is often the previous night: the same row as an improvement
there (main gained a fix the release lacked), or the candidate's number equal to an older image's.

What the summary row does not say (the invariant's own message, the query's error, the slow
request) is in the check's raw files, under `$W/data/$FOLDER/<check>/`:

| check | where the detail is |
| --- | --- |
| soak-a, soak-b | `raw/<scenario>/<side>-<n>/stdout.log` (each invariant's message and how many probes broke), `raw/runs.csv` |
| fuzz | `raw/runs.json` (each run's checks), `raw/rows.csv` |
| ldbc | `raw/summary.csv` (requests per query and side, `first_failure`), `raw/samples.csv`, `raw/explain/<side>/<query>.txt`, `<side>-server.log` |
| pokec | `raw/rows.csv` (every row's ratio, gated or not), `raw/runs/<dataset>-<workload>-<side>-p<pass>r<rep>/` (`run.log`, `server.log`, `ping.csv`), `raw/runs.csv` |
| micro-hw | `raw/measure-<side>-r<n>.log` (a query's error), `raw/confirm-measure-*` (the re-run), `raw/rows.csv` |

The run page (`$W/site/run/$FOLDER.html`) links each check's files.

## 4. Match each failure to an issue

Start with what the checks already map: `bench/src/falkorbench/soak/README.md` ("Known breaks
on both sides"), the known flaky checks (`bench nightly failures` names their issue), and the
notes on queries a side cannot run in `bench/README.md` (LDBC) and each check's README under
`bench/src/falkorbench/<check>/`. Then list every issue once, open and closed, and search the titles for the **mechanism**, not only
the symptom: the operator, function, index type, procedure or invariant that broke (`index ==
scan`, `CLIENT KILL`, `used_memory`, `writer starv`, `shortestPath`, `UDF`, `IC3`), as well as the
check and scenario names (`s10`, `fuzz`, `LDBC`).

```bash
gh issue list -R "$REPO" --state all --limit 10000 --json number,title,state,labels > "$W/issues.json"
jq length "$W/issues.json"      # equal to the limit: raise it, the list was cut; a network error: run it again
jq -r '.[] | "#\(.number) [\(.state)] \(.title)"' "$W/issues.json" | grep -iE '<mechanism words>'
```

A known flaky check names its issue already. Read a candidate issue's body (`gh issue view <n>
-R "$REPO"`) before calling it a match. Several failures with one cause (all `s10` index checks,
every UDF query that errors) share one issue. A closed issue that matches is one of two things:
its fix is on the baseline but not in the candidate (step 3's compare says so): name it and say
"fixed on the baseline, not in the candidate", no draft; or the failure is back after the fix:
a regression of it, which gets a draft that names the closed issue.

A row that sums up others (pokec's `all queries` latency and `throughput`, a class median) has
no mechanism of its own: tie it to the per-query failure that drives it when one does, say the
link is inferred, and draft nothing separate for it.

Some failures are known behaviour the team decided not to file (a by-design trade-off). If you
know of such a decision, say "not filed, by decision" and name where it was decided; if you are
unsure, draft the issue anyway and say it may be a known, unfiled behaviour.

For each failure with no matching issue, draft one as text in the report, never filed:

```
Title: [perf] <check>: <what broke, by mechanism> (<new | since YYYY-MM-DD | no earlier night>)
Labels: performance, perf-regression (a regression) | bug (a broken correctness check)
Body:
- What: <the row: name, metric, candidate vs baseline, ratio and threshold, or "broken on the
  candidate" / "on both sides", and in how many runs; the message from the raw files>
- Images: candidate <label> <image> (commit <sha>), baseline <label> <image> (commit <sha>)
- Run: <html_url>; files: <check>/raw/..., <check>/result.json (the run's perf-result-<check> artifact)
- Since: <new in this run | failing in every scheduled night since YYYY-MM-DD | no earlier night to compare>
```

## 5. The report

Short, in chat:

```
Perf run <id> (<workflow>, <event>, <created_at>): <STATUS from nightly.json>
candidate <label> (<commit>) vs baseline <label> (<commit>)[; candidate is <n> commits behind the baseline]
Run page: <$W/site/run/<folder>.html>

| check | verdict | regressions | broken on the candidate | error |
(one row per check; a `pass` with broken rows reads "pass, N broken", never "green")

Failures:
- <check> · <kind> · <what> · <every qualifier that applies: both sides | candidate only, known flaky #n, timing, k of n runs> · <new | since date | no earlier night> · <#issue | fixed on the baseline (#n) | not filed, by decision | no issue>
Fixed since <previous night>: <list or none>
Compared with: <the earlier scheduled nights, or "no earlier scheduled night">

Draft issues (not filed): <one per failure with no issue>
```

Say what was not measured (a check that errored, files that did not download) as plainly as
what failed: an unmeasured check is never green.
