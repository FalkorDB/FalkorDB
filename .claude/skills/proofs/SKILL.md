---
name: proofs
description: Work with FalkorDB's Lean 4 proofs of the Rust engine in proofs/ — build and check them, keep them in sync with main, prove new or changed functions, confirm bugs the models find, and review a PR against the proofs. Use when asked to update/run/extend the proofs, check Lean coverage, re-target proofs after main moves, verify a function or PR formally, or when a change touches code listed in proofs/*/COVERAGE.tsv.
allowed-tools: Bash, Read, Edit, Write
---

# Proofs (Lean 4)

`proofs/` holds one standalone Lake project per engine area (`value_order`,
`binder`, `pending_commit`, `index_layer`, …). Each models the Rust code it
covers line by line and proves properties about that model. Lean 4 only, no
Mathlib. The toolchain comes from each project's `lean-toolchain`.

- `proofs/<area>/<Root>.lean` — the header is the report: Lean↔Rust `file:line` table, theorems, bugs, gaps.
- `proofs/<area>/COVERAGE.tsv` — one row per Rust fn: bucket PROVEN / AXIOMATISED / MODELLED / NOT COVERED.
- `proofs/FINDINGS.md` — the bug ledger. Every bug has a confirmed repro, and later notes record which commits fixed it.
- `proofs/lean_ci.sh`, `proofs/coverage.sh` — the two gates.

## Quick start

```bash
curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh   # once
bash proofs/lean_ci.sh                     # build all projects; fails on sorry/admit/native_decide/unjustified axiom
bash proofs/lean_ci.sh binder value_order  # just some projects
bash proofs/coverage.sh                    # per-file + total coverage; --unmatched lists stale rows
```

Proofs cite `origin/main`. If your checkout is on another branch, read the Rust from
a clean worktree: `git worktree add --detach ../fdb-main origin/main` then
`COVERAGE_RUST_ROOT=../fdb-main bash proofs/coverage.sh`.

## Workflows

**Re-target after main moves** (keep the proofs true for the current code):
1. `git diff <old-sha> origin/main --stat -- graph/src src`. The old sha is in FINDINGS.md "Current state".
2. Find the projects to update: `grep -l "<file>" proofs/*/COVERAGE.tsv` for each changed file.
3. For each changed fn: update the Lean model to the new code, re-prove, fix `file:line` cites. Add rows for new fns and delete rows for removed ones.
4. If the commit fixed a bug, turn its counterexample theorem into a correctness theorem. Keep the old one as a labelled historical note: `pre<PR>_…`, fixed by `#PR (sha)`.
5. Run `lean_ci.sh` (all green) and `coverage.sh` (100% non-FFI, 0 unlisted), then update FINDINGS.md "Current state".

**Prove a new or changed function**: model it in the owning project, or start a new
project by copying the shape of an existing one. Then add the COVERAGE row and prove.
Conventions are in [REFERENCE.md](REFERENCE.md).

**When a theorem is false** (the model has a bug): get a concrete counterexample
(`#eval` or `decide`), then reproduce it against the real Rust before calling it a bug:
a Rust test, or a live query compared with the C module. Record it in FINDINGS.md with
`file:line`, the repro, and the Rust vs C output. Keep the counterexample as a theorem.

**Review a PR with the proofs**: in a new `proofs/pr<N>_review/` project, copy the
definitions you need from the owning projects. Model the PR's merged code for the
changed functions, and prove or refute each claim the PR makes. Confirm any refutation
live. Also check that the existing theorems still hold for the new code.

## Hazards

- Never hand-edit Lean to match a guess about the Rust; read the cited lines in `origin/main`.
- Back up `proofs/` before mass edits (`tar --exclude=.lake -czf …`). A bad `perl -pi` once wiped citations.
- Live repros: give each run its own redis port. For flow tests use `RLTest -p <port>`; the harness connects to that port (default 6379), and `--randomize-ports` does not change it. Copy a rebuilt dylib to a fresh path before loading it (macOS kills redis-server when the loaded dylib is overwritten).
- Agents working in parallel: one owner per `proofs/<area>`, private scratch dirs, and no `./flow.sh` on 6379.
