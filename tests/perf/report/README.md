# Nightly perf report and missed-run alarm

Two scripts tell the core team's Google Chat space how the nightly perf run went.
Both use only the Python standard library and run as `python3 -I`.

| script | called by | does |
| --- | --- | --- |
| `post.py --nightly <nightly.json>` | the report job of `perf-nightly.yml` | reports one night |
| `missed_run.py [--date YYYY-MM-DD]` | `perf-missed-run.yml`, 05:00 UTC | alarms when a night left no result |

## `post.py`: one night

`merge.py` validates `nightly.json` and sets its `status` before the report runs,
so `post.py` trusts that status as written. It checks only what needs no
validator: a file it cannot read, a status that is not `green`, `red`, `error` or
`skipped`, and a crash of `post.py` itself all report the night as ERROR, never
green.

* **Chat:** one line every night, posted to the incoming webhook in
  `$PERF_CHAT_WEBHOOK` when that is set, and printed to stdout, for example
  `Perf nightly 2026-11-24: RED (micro-hw), main@4c10889 vs 6.2.0 · details · new issue`.
  The parentheses hold the regressed checks on a red night, the checks that could
  not measure on an error night, the reason on a skipped night, and the number of
  checks on a green night. "details" links to the run page. A red night also gets
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
* **Exit code:** 0 whenever it reported the night, whatever its status: `merge.py`
  owns the status and alone decides the run's colour. 1 only when it could not
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

## `missed_run.py`: the 05:00 UTC alarm

It lists the scheduled runs of `perf-nightly.yml` on `main` created in the 24 hours
before `<date> 05:00 UTC` (`perf-nightly.yml` starts at 00:00). It is quiet when
any of them has an unexpired `perf-nightly` artifact; a skipped night uploads one
too, and a re-run of the night's jobs is the same run. A hand-started run does not
count: one given its own images uploads nightly.json but posts nothing. Otherwise
it posts an alarm naming the case and exits 1, so the run is red even without a
webhook:

* no scheduled run in the window;
* the newest run is still going;
* the newest run finished (with its conclusion) without the artifact;
* `perf-nightly.yml` does not exist, or GitHub could not be asked.

A re-run of a scheduled alarm checks the night its run was created for. Each run
has its own concurrency group, so no check can displace another one that waits.
`--workflow`, `--branch`, `--event` and `--artifact` point it at other runs for an
acceptance run; the workflow takes `date` through `workflow_dispatch`.

## Tests

```bash
python -m pytest tests/perf/report        # or: python -m unittest discover tests/perf/report
```

Each test names one rule and fails when that rule is broken. The webhook and the
GitHub API are served from a local stub server. `.github/workflows/perf-contract.yml`
runs them with the rest of `tests/perf`.

Try a fixture by hand:

```bash
python3 -I tests/perf/report/post.py --nightly tests/perf/fixtures/valid/nightly-red.json
```
