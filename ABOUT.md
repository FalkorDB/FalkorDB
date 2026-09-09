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

These files are output, not source. Carrying 74 generated files through review
buries the code that produced them, and the C engine needs to reach the corpus
without depending on which Rust PR happens to be open. Keeping them here means
one address that outlives any branch.

What that costs: a wire change no longer shows up as a fixture diff inside the
Rust PR. It shows up here instead — every update to this branch names the
commit that generated it, so the pairing is recorded, just not in the PR's own
diff. The generator (`graph/src/effects/v3/fixtures.rs`) and its in-memory
round-trip tests stay with the code and still fail on drift.

## The generator lives here too

`generator/fixtures.rs` is the Rust module that produces every file in this
directory. It is here rather than in the engine for the same reason the bytes
are: it is not production code, and it has no business in a pull request that
reviewers read for the codec.

It needs the engine's crate to compile, so regenerating means putting it back
in place for one run. With the corpus checked out as a worktree inside an
engine tree, from the engine root:

    cp tests/fixtures/effects_v3/generator/fixtures.rs graph/src/effects/v3/
    printf '#[cfg(test)]\nmod fixtures;\n' >> graph/src/effects/v3/mod.rs
    UPDATE_EFFECTS_FIXTURES=1 cargo test -p graph effects::v3::fixtures
    git -C graph checkout src/effects/v3/mod.rs && rm graph/src/effects/v3/fixtures.rs

Then commit here, naming the engine commit that generated it. A diff on this
branch is a wire break: for the C engine that means a version bump, not a
patch.

**What this costs, stated plainly.** With the generator out of the engine, a
change to the encoder no longer fails `cargo test -p graph`. The drift alarm
only rings when someone runs the four steps above. Closing that gap needs a CI
job that checks out both this branch and the engine and runs them — until that
exists, treat regenerating as a required step whenever the wire changes, not
something the test suite will remind you about.
