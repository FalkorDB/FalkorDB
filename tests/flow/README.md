# Flow tests — upstream tracking

The Python flow tests in this directory mirror `FalkorDB/FalkorDB`'s
`tests/flow/`. The C implementation is the upstream source of truth; this
repo carries a (growing) port of the same tests so we exercise the rust
runtime against the same coverage.

## Automated upstream tracking

The `flow-tests-migration` agentic workflow
(`.github/workflows/flow-tests-migration.md`) runs daily at 06:00 UTC. On
each run it:

1. Diffs `tests/flow/test_*.py` here against the same path in
   `FalkorDB/FalkorDB@master`.
2. For each upstream file with no counterpart here, classifies it as
   **services-bucket**, **spawn-bucket**, **cluster**, or **split** using
   the cluster / spawn heuristics documented in the workflow markdown.
3. Materialises the missing file into `tests/flow/` (splitting mixed files
   into `test_xyz.py` + `test_xyz_cluster.py` à la `test_udf.py` /
   `test_udf_cluster.py` when both flavours coexist upstream).
4. Opens **one draft PR per missing file**, labelled
   `flow-tests-migration`, with a body that:
   - links the upstream permalink at a pinned SHA,
   - records the classification,
   - reminds the reviewer to add the new file to the appropriate matrix in
     `.github/workflows/rust-pr.yml`.

Up to 5 PRs are opened per run, and the run is skipped entirely when 10+
`flow-tests-migration` PRs are already open. Files already covered by an
open or recently-closed (<30 days) migration PR are deduplicated.

## Reviewing a migration PR

Migration PRs land as **draft** on purpose: the freshly-copied upstream
test will frequently fail because the rust port has API gaps. The flow is:

1. CI runs `Rust PR` against the new branch and (usually) fails.
2. The companion `flaky-test-rca` workflow detects the failure, sees the
   `flow-tests-migration` label, and posts **one** root-cause comment per
   head SHA with the failing test ID, classification (API gap /
   unsupported feature / harness issue / flake), and suggested next step.
3. Reviewer pushes a fix (port the missing feature, adjust the harness,
   split into `_cluster.py`, …). The next failed CI run earns a fresh
   RCA comment on the new head SHA — same-SHA re-runs do not duplicate.
4. Once CI is green, reviewer adds the file to the matrix in
   `rust-pr.yml` and flips the PR out of draft.

## Manual run

To force a discovery run (e.g. after upstream merges a batch of new
tests):

```bash
gh workflow run flow-tests-migration.lock.yml --repo FalkorDB/falkordb-rs-next-gen
```

`workflow_dispatch` bypasses the open-PR cap.
