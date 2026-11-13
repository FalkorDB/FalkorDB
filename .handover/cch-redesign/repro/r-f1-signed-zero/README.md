# R-F1: the scoped recompute is `==`-equal to a full customization, not bit-identical

The live confirmation of finding R-F1 in `proofs/cch_recompute_review/` (Lean: `R_F1_not_bit_identical`,
`R_F1_agrees`, `R_F1_bitwise_fixes_it` in `CchRecomputeReview/Findings.lean`).

## What the test shows

`Cch::recustomize_seeded` (`graph/src/index/falkordb/cch/partial.rs:98-258`) has two kinds of comparison:
- it decides that an arc changed with `f64 !=` (`:221-222`);
- it writes each weight only where `!=` says the weight differs (`:225-230`).

`+0.0 != -0.0` is false. So a seed that only flips the sign of a zero is neither written nor propagated. A full
`customize` stores the new sign.

The test builds an 8-node graph in node-id order. Nodes 3 to 7 are isolated, so one changed arc stays under the
`n / 8` valve. The edges are:
- `1 → 0`, the weight that changes;
- `0 → 1 = 5`;
- `0 → 2 = -0.0`;
- `2 → 0 = 5`.

Eliminating node 0 adds the fill arc `(1, 2)`. Its upward weight is `w(1→0) + w(0→2)`.

The test changes `w(1→0)` from `+0.0` to `-0.0`, runs the scoped pass with `changed = [(1, 0)]`, and compares the
result with a fresh `customize` of the new weights:
1. It prints the bits of the two arcs involved:
   - the changed arc `(0, 1)`, downward;
   - the fill arc `(1, 2)`, upward.
2. It asserts that the two metrics are `==`-equal, middles included. The repo's own tests compare the same way
   (`snapshot`), which is why they cannot see this.
3. It asserts that every weight is bit-identical, which `partial.rs:19-21` and `:83` and 03 §1.8 and §6b claim.

The outcomes:
- On the base, step 2 passes and step 3 fails. The scoped pass keeps `+0.0` (`0x0`) where `customize` stores
  `-0.0` (`0x8000000000000000`). This happens on the changed arc (the write is skipped) and on the fill arc above
  it (propagation is skipped).
- With `fix.patch`, the change test and the two weight writes compare `to_bits()`, and the test passes.

The weights compare equal, so the distances do too; only the sign of a zero weight differs. Severity: minor.

## Base

`31ca1b802` = `origin/cch/08c-build-catch-up`, the commit 03 cites as `@08c`. `partial.rs` is byte-identical from
`cch/08-maint` to the stack tip `cch/10-rdb-v20` (`484e36d4e`), so both patches should apply there unchanged. Only
`31ca1b802` was run.

## Files

- `test.patch`: adds the test `lean_r_f1_negative_zero_seed_bit_identical_to_full` to `partial.rs`'s `tests`
  module.
- `fix.patch`: the `to_bits()` fix to `partial.rs:221-230`. Apply it on top of `test.patch`.
- `run1-base.log`: `test.patch` alone. The test fails at the bit-identity assertion.
- `run2-fix.log`: `test.patch` + `fix.patch`. The test passes.
- `run3-fix-all-partial.log`: `test.patch` + `fix.patch`, with every test in `partial.rs`.

## Output (2026-11-13, Apple M5, macOS, Homebrew clang)

`run1-base.log`, at `31ca1b802` with `test.patch`. Exit 101:

```text
R-F1 arc 0-1 down: scoped 0x0000000000000000 full 0x8000000000000000
R-F1 arc 1-2 up:   scoped 0x0000000000000000 full 0x8000000000000000
R-F1 scoped == full under f64 `==` (snapshot comparison): yes
thread '...::lean_r_f1_negative_zero_seed_bit_identical_to_full' panicked at graph/src/index/falkordb/cch/partial.rs:750:9:
scoped is ==-equal to full but not bit-identical: (slot, direction, scoped bits, full bits) [(0, "down", 0, 8000000000000000), (2, "up", 0, 8000000000000000)]
test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 941 filtered out; finished in 0.01s
```

`run2-fix.log`, with `test.patch` and `fix.patch`. Exit 0:

```text
R-F1 arc 0-1 down: scoped 0x8000000000000000 full 0x8000000000000000
R-F1 arc 1-2 up:   scoped 0x8000000000000000 full 0x8000000000000000
R-F1 scoped == full under f64 `==` (snapshot comparison): yes
R-F1 scoped bit-identical to full: yes
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 941 filtered out; finished in 0.01s
```

`run3-fix-all-partial.log`, with both patches, runs every test in `partial.rs`:
`test result: ok. 7 passed; 0 failed`. So the fix breaks none of the existing six.

Both patches were checked to apply cleanly to `31ca1b802`, in order, and to give the sources that were run.

## How to run

```bash
# a worktree at the base, outside /tmp
git worktree add --detach ../r-f1-tmp/wt 31ca1b802
cd ../r-f1-tmp/wt
git submodule update --init --recursive   # or copy deps/ from a checkout at the same pins
git apply <this folder>/test.patch

# macOS: Homebrew clang (see CLAUDE.md); the CCH module needs the index-falkordb feature
export CC=/opt/homebrew/opt/llvm/bin/clang CXX=/opt/homebrew/opt/llvm/bin/clang++
cargo test -p graph --features index-falkordb --lib -- lean_r_f1 --nocapture   # fails: R-F1

git apply <this folder>/fix.patch
cargo test -p graph --features index-falkordb --lib -- lean_r_f1 --nocapture   # passes
```

Without `--features index-falkordb`, the filter matches no test, because the `cch` module is not compiled.
