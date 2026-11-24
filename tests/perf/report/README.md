# Nightly perf report and missed-run alarm

Two scripts turn the nightly perf run into one message in the core team's Google
Chat space. Both use only the Python standard library and run as `python3 -I`.

| script | called by | does |
| --- | --- | --- |
| `post.py --nightly <nightly.json>` | the last job of `perf-nightly.yml` | reports one night |
| `missed_run.py [--date YYYY-MM-DD]` | `perf-missed-run.yml`, 05:00 UTC | alarms when a night left no result |

## `post.py`: one message per night

Every verdict comes from `nightly.json` (schema 1, epic #3974). Before trusting it,
`post.py` runs perf-judge's validator, `tests/perf/validate.py`, on the file. The
message never shows a status better than the evidence:

* **red** when any check says `regression`;
* otherwise **error** when any check says `error`, when the file breaks the contract
  (the validator's problems are listed as `contract: <code> at <json path>`), when
  its declared `status` disagrees with its checks, when the validator itself cannot
  run, or when a non-skipped night has no checks;
* **skipped** only for `status: "skipped"` with no checks and no problems;
* **green** only when every check passed and nothing above applies.

Two guards hold even without the validator: a check name listed twice, and a
`pass`/`regression` entry that is not a full result (no `schema`), both make the
night an error. If the script itself crashes, it still posts "the reporter crashed"
with the run link, and exits 1.

What each message holds:

| status | message |
| --- | --- |
| green | one line: date, candidate vs baseline label, how many checks passed (and how many rows are broken on both sides), run link |
| skipped | one line with `skip_reason` |
| error | the checks that could not measure and why, the problems with the file, the checks that passed; the headline says "tonight is not green" |
| red | each regressed check (confirmed or not, its summary, its worst regression rows: correctness first, then by ratio), the checks that could not measure, the commits since `previous_green_commit` with their PR numbers, the run link, and a pre-filled new-issue link (labels `performance` and `perf-regression`) |

Findings read as plain words. `completed` 0 is "did not finish", `invariant` 0 is
"invariant broken", a `mismatch` row is "row count differs: 12 vs 14". A null value
prints as `n/a`, and a null ratio or threshold is left out, never computed here. A
correctness row at 0 on both sides (contract v1.4: status `info`) is never a
regression: Chat only counts such rows ("2 rows broken on both sides (not new)"),
because they recur every night, and the job summary names them.

The commit list comes from `git log <previous_green_commit>..<candidate commit>` in
this checkout, else from GitHub's compare API (`GITHUB_API_URL`, token from
`GH_TOKEN` or `GITHUB_TOKEN`). Each PR number is the last `(#N)` at the end of the
commit title (squash merges), so `… (#3887) (#3889)` is PR #3889. When neither
source works the message says why and still links the compare page. When the
candidate commit is the last green night's, there is no list: the message says
nothing new was merged on main.

Delivery:

* stdout: the Chat text;
* `$GITHUB_STEP_SUMMARY`, when set: the same message as GitHub Markdown, with
  more findings and commits;
* `$PERF_CHAT_WEBHOOK`, when set: POSTed as `{"text": …}` to the Chat incoming
  webhook. Chat markup, not Markdown: incoming webhooks default to
  `MARKUP_SYNTAX_CHAT` (developers.google.com/workspace/chat/format-messages), so
  `*bold*`, `` `code` ``, `<url|text>` and `- ` bullets. Text from results and
  commit titles has `<` replaced, so it cannot become a mention or a disguised
  link (a bare URL in it still shows as itself). A link whose URL holds `<`, `>`,
  `|`, quotes, brackets or spaces is shown as plain text, and a `run_url` like that,
  or one longer than 300 characters, is replaced by the current run's. Google caps a message at 32,000 bytes ("The
  maximum message size (including any text or cards) is 32,000 bytes",
  developers.google.com/workspace/chat/create-messages, an API page; the webhook
  page states no limit). The message stays under 4,000 characters: fewer rows, free
  text cut to 300 characters, and as a last resort whole lines dropped from the end
  with a pointer to the job summary.

A failed post is a warning, and the exit code stays 0: only a crash of the script is
a failure. A post is retried twice on 429, 5xx and connection errors (nothing was
sent yet), and not after timing out while waiting for Chat's answer, when it may
have arrived. A trimmed red message still ends with the run and issue links. The webhook URL is the credential and is
never printed. Only `https://` webhooks are used.

## `missed_run.py`: the 05:00 UTC alarm

It lists the scheduled runs of `perf-nightly.yml` on `main` created in the 24 hours
before `<date> 05:00 UTC` (`perf-nightly.yml` starts at 00:00). It is quiet when
any of them has an unexpired `perf-nightly` artifact; a skipped night uploads one
too, and a re-run of the night's jobs is the same run. A hand-started run does not
count: one given its own images uploads nightly.json but posts nothing. Otherwise
it posts an alarm naming the case and exits 1:

* no scheduled run in the window;
* the newest run is still going;
* the newest run finished (with its conclusion) without the artifact;
* `perf-nightly.yml` does not exist, or GitHub could not be asked.

`--workflow`, `--branch`, `--event` and `--artifact` point it at other runs for an
acceptance run; the workflow takes `date` through `workflow_dispatch`.

## Tests

```bash
python -m pytest tests/perf/report        # or: python -m unittest discover tests/perf/report
```

The expected messages come from the contract and perf-judge's fixtures in
`tests/perf/fixtures/`; the tests serve the webhook and the GitHub API from a local
stub server. `.github/workflows/perf-contract.yml` runs them with the rest of
`tests/perf`.

Try a fixture by hand:

```bash
python3 -I tests/perf/report/post.py --nightly tests/perf/fixtures/valid/nightly-red.json
```
