# Nightly perf runs: the result format and its checker

Each nightly perf check (micro-hw, size-probe, pokec, soak-a, soak-b, fuzz,
ldbc) compares a candidate image with a baseline image and writes one
`result.json`. The nightly merges them into one `nightly.json`, which the
reporter posts. This directory defines those two files and holds what the
checks are accepted against. Nothing here runs at night or stores history.

| file | what it is |
| --- | --- |
| `validate.py` | The checker. It rejects a result or nightly that cannot be reported as it stands: a missing or malformed field the reporter reads, or a verdict that contradicts its findings. Standard library only. |
| `fixtures/valid/` | Hand-written example files (7 results, 5 nightlies) for the tests and for the reporter's tests. |
| `invalid_cases.py` | Broken examples, each built from a valid file by changing one field, with the problem it must produce. |
| `acceptance.yaml` | For each check, image pairs whose right answer is known, and that answer. |
| `test_validate.py` | Runs all of the above. |

## The format

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

The rules `validate.py` adds to "every field present and well formed"
(`validate.CODES` lists every problem it can report):

* images are pinned by digest; commits are 40 hex or null;
* a `regression` verdict is confirmed and has a regression finding; `pass` and
  `error` carry none (a regression outranks an error); `error` has a reason and
  any other verdict has `error: null`;
* a run that did not finish (`completed` 0) or a broken invariant (`invariant`
  0) is a regression, unless the baseline broke too (then `info`); a `rows`
  finding with direction `mismatch` whose sides differ is a regression;
* no `NaN` or `Infinity` (Python's `json.dump` writes them for a 0 baseline);
* at most 50 findings;
* a nightly's `status` is red if any check regressed, else error if any check
  errored, else green; `skipped` (with a reason) exactly when `checks` is empty;
  each check appears once, and a stub can only say error.

Keys the format does not name are ignored.

## Using the checker

```sh
python -I tests/perf/validate.py out/result.json          # any number of files
python -I tests/perf/validate.py --kind nightly nightly.json
```

It prints `<file>: <code>: <json-path>: <message>` per problem, or
`<file>: ok (<kind>)`, and exits 0 (all good), 1 (a problem) or 2 (usage). From
Python, load it by path (`python -I` keeps this directory off `sys.path`) and
call `validate_file(path, kind)`.

## Acceptance pairs

`acceptance.yaml` lists, per check, image pairs whose right answer is known: the
same image compared with itself must pass; 6.2.0 against main@4c1088949 must
flag the rows #3963 fixed; 6.0.4 against 6.0.1 must catch the 3-hop read that
hangs. A check is accepted when an independent run of every pair (the acceptance
run) comes out as listed. The file's header defines each field.

## Running the tests

```sh
uv run --no-project --with pytest --with pyyaml python -I -m pytest tests/perf
```

`.github/workflows/perf-contract.yml` runs the same, with pinned versions, on
pull requests that touch `tests/perf/`, `tests/soak/` or `bench/`.
