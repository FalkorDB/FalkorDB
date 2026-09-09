# The effects v3 conformance corpus

This branch carries **only artifacts** — no engine code, no build, no history
shared with `main`. It exists so both FalkorDB implementations can point at one
set of bytes instead of each trusting its own round trip.

`README.md` beside this file is generated and describes every case, the wire
layout and what the corpus does and does not guarantee. Read that one for the
format; this one is about the branch.

## Getting it

    git fetch origin effects-v3-corpus
    git worktree add tests/fixtures/effects_v3 origin/effects-v3-corpus

The Rust engine looks for the corpus at `tests/fixtures/effects_v3` and skips
its corpus tests when the directory is absent, so a checkout without it still
builds and tests clean. `EFFECTS_V3_CORPUS=<dir>` points it somewhere else.

## Why it is not in the pull request

These files are output, not source. Carrying 73 generated files through review
buries the code that produced them, and the C engine needs to reach the corpus
without depending on which Rust PR happens to be open. Keeping them here means
one address that outlives any branch.

What that costs: a wire change no longer shows up as a fixture diff inside the
Rust PR. It shows up here instead — every update to this branch names the
commit that generated it, so the pairing is recorded, just not in the PR's own
diff. The generator (`graph/src/effects/v3/fixtures.rs`) and its in-memory
round-trip tests stay with the code and still fail on drift.

## Updating it

From an engine checkout, with the corpus worktree in place:

    UPDATE_EFFECTS_FIXTURES=1 cargo test -p graph effects::v3::fixtures

then commit here, naming the engine commit in the message. A diff on this
branch is a wire break: for the C engine that means a version bump, not a
patch.
