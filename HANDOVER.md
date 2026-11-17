# Handover: Creusot on `cow_btree` (spike)

Branch `spike/creusot-cow-btree`, based on main `087b384cf`. A spike to see what it takes to verify
the native index's copy-on-write B⁺-tree with [Creusot](https://github.com/creusot-rs/creusot)
(v0.15.0), a verifier that checks Rust code itself against specs written next to it. Compare
`proofs/cow_btree/`, which proves a hand-written Lean copy of the same code. Not meant to merge.

## What is proven

Each claim is checked against the real Rust code, including freedom from panics and integer overflow.
The six files prove in 5.7 s from scratch.

| Function | Claim |
|---|---|
| `leaf::partition_point` | Returns the boundary between "predicate true" and "predicate false" for a monotone predicate |
| `leaf::merge_sorted` | Output is strictly sorted and holds exactly the elements of both inputs |
| `Branch::child_index` | Result is at most `seps.len()` |
| `Node::insert_one` | Entries after = entries before + the new one (a split hands some to the right sibling); every branch keeps `children = seps + 1`; a `false` flag means the entry was already present |
| `mem_node`, `shaped` | The two logic functions terminate |

`insert_one` is proven at the level of *which entries are present*, not their order. Lean's
`insert_spec` also proves order and the full `TreeWF` invariant. Deliberate bugs, each re-proved:

| Change | Result |
|---|---|
| `insert_one` drops a split's right half | Rejected (70/73 goals) |
| `merge_sorted` keeps duplicates | Rejected (67/70 goals) |
| `insert_one` puts the right half on the wrong side | **Still proves**: order is not in this spec |

Proving "a `true` flag means the entry was absent", and catching the third bug, needs the ordering
invariant (the entry can only be in the child it routes to). That is the next step.

## What changed in the tree

All changes are inside `cow_btree/`, except three lines of crate config. Search for `Creusot spike`.

**Needed just to translate** (behaviour-preserving):

1. `Node` → `Arc<Branch>` → `Vec<Node>` is rejected as an illegal recursive type. Creusot allows
   recursion through `Box`, `&`, or a type declared positive. creusot-std declares `Rc` but not `Arc`
   or `Vec`, so `cow_btree/mod.rs` declares both (trusted).
2. `merge_underfull_below` ↔ `merge_underfull` call each other. Creusot rejects mutual recursion
   (a FIXME in `creusot/src/validate/terminates.rs`), so `merge_underfull_below` is `#[trusted]`.
3. `slice::windows` and `slice::chunks` have no `IteratorSpec`, and the orphan rule blocks adding
   one. Three call sites became index loops (`leaf/mod.rs` ×2, `mod.rs` ×1).
4. `Leaf::iter` returns `impl Iterator`, which hides the spec; a `cfg(creusot)` copy returns
   `impl IteratorSpec`.
5. `RangeIter` gets the weakest sound `IteratorSpec` ("may produce anything"). It says nothing about
   range scans.
6. `LeafFormat` derives `DeepModel`.
7. The creusot-std prelude exports its own `Clone`, `PartialEq` and `vec!`; `leaf/mod.rs` imports
   std's explicitly.

**For the specs:**

- `partition_point` and `gallop_lower_bound` take `impl Fn` instead of `impl FnMut`. All callers pass
  closures that do not mutate.
- `insert_one` takes a ghost height `h: Snapshot<Int>` (zero-sized, erased in normal builds). Its
  two callers pass it. `CowBTree::insert` passes a placeholder, so the top level is not verified.
- `snapshot!` / `proof_assert!` hints in `merge_sorted` and `insert_one` (erased in normal builds).

**Crate config:** `creusot-std = "0.15.0"` in `graph/Cargo.toml` (adds `creusot-std`,
`creusot-std-proc`, `pearlite-syn`, `uuid` to the lockfile), `cfg(creusot)` in the workspace
`check-cfg`, and `#![cfg_attr(creusot, feature(allocator_api))]` in `graph/src/lib.rs`.

In a normal build every annotation is erased. On this branch:
`cargo clippy -p graph --all-targets -- -D warnings` and `cargo fmt --check` are clean, and the 36
`cow_btree` unit tests pass.

## Trusted, not proven

- `Arc` and `Vec` declared positive (recursion)
- `merge_underfull_below` (breaks the mutual recursion)
- `Leaf::insert` (byte layouts), with a spec at the level of which entries are present
- `make_private` (`Arc` sharing): `*result == **arc`, `^result == *(^arc)`
- `Vec::split_off` and `<[T]>::partition_point`: creusot-std has no contract for them
- the leaf's logical view (`Seq<(Int, Int)>`), opaque

Lean makes the same two cuts (leaf bytes, `Arc`) and proves the bytes in `proofs/index_layer`.

## Baseline: Creusot with no specs

Measured on a copy of `cow_btree` with only the translate changes above:
**59 of 120 functions** proved free of panics and overflow, in 5 min 44 s (10 provers in parallel).
Most failures are functions like `read_u64(b, off)` that do panic on arbitrary input; their safety
comes from the caller, which only a precondition can state. There are also 132 calls to std functions
creusot-std has no contract for (`extend_from_slice` ×60, `to_le_bytes` ×19, range indexing ×14,
`as_slice` ×14, `all` ×7, `split_off` ×5, `from_le_bytes` ×5, …). A function making one of those
calls cannot be proven until the std function gets a (trusted) spec. That covers the byte-layout code.

## Rerun

**Install** (macOS; Linux is the same minus Homebrew):

```bash
brew install opam zlib
opam init --bare -n -y
git clone https://github.com/creusot-rs/creusot && cd creusot && git checkout v0.15.0
OPAMVAR_in_creusot_ci=true ./INSTALL   # the variable skips the Why3 GUI and its GTK dependencies
```

This installs the Rust toolchain `nightly-2026-08-03`, `cargo-creusot`, Why3, and the Alt-Ergo, Z3,
CVC4 and CVC5 provers. It does not touch `~/.cargo/config.toml`.

**Build the standalone crate.** Creusot cannot run on `graph` (FFI, native deps), so copy the
module out. From the repo root, on this branch:

```bash
SPIKE=$(mktemp -d)/creusot-cow-btree
mkdir -p "$SPIKE/src"
cp -R graph/src/index/falkordb/data_structures/cow_btree "$SPIKE/src/"
cp graph/src/narrow_int.rs "$SPIKE/src/"   # the one item cow_btree imports from outside itself
cat > "$SPIKE/src/lib.rs" <<'EOF'
#![cfg_attr(creusot, feature(allocator_api))]
#![allow(dead_code)]
extern crate creusot_std;
pub mod cow_btree;
pub mod narrow_int;
EOF
cat > "$SPIKE/Cargo.toml" <<'EOF'
[package]
name = "cow-btree"
version = "0.0.0"
edition = "2024"
publish = false

[dependencies]
creusot-std = "0.15.0"

[lints.rust]
unexpected_cfgs = { level = "warn", check-cfg = ['cfg(creusot)'] }

[workspace]
EOF
cat > "$SPIKE/why3find.json" <<'EOF'
{
  "fast": 0.2,
  "time": 1.0,
  "depth": 6,
  "packages": [ "creusot" ],
  "provers": [ "alt-ergo", "z3", "cvc5", "cvc4" ],
  "tactics": [ "compute_specified", "split_vc" ],
  "drivers": [],
  "warnoff": [ "unused_variable", "axiom_abstract" ],
  "profile": [
    { "prover": "cvc5@", "size": 47, "time": 0.541 },
    { "prover": "z3@", "size": 33, "time": 0.793 },
    { "prover": "alt-ergo@", "size": 17, "time": 0.58 },
    { "prover": "cvc4@", "size": 45, "time": 0.432 }
  ]
}
EOF
cd "$SPIKE"
cargo creusot --only=coma
cargo creusot --only=prove \
  verif/cow_btree_rlib/cow_btree/leaf/partition_point.coma \
  verif/cow_btree_rlib/cow_btree/leaf/merge_sorted.coma \
  verif/cow_btree_rlib/cow_btree/node/impl_Branch/child_index.coma \
  verif/cow_btree_rlib/cow_btree/node/impl_Node/insert_one.coma \
  verif/cow_btree_rlib/cow_btree/node/mem_node.coma \
  verif/cow_btree_rlib/cow_btree/node/shaped.coma
# Expected: "Proved (6 files) ✔". `cargo creusot --only=prove` with no files runs every function,
# most of which fail without specs (see Baseline).
```

To see why a goal fails: `cargo creusot --only=prove --why3find-arg=-X <file.coma>` prints each
failed goal with its hypotheses.

## Notes

- A plain `cargo check` with rustc 1.96.1 fails in `native-deps` (`map_or_default` is unstable
  there), independent of this branch. 1.100.0 builds it.
- One `cargo test` run in the standalone crate failed 1 of 36 while a Creusot build was running, then
  passed three times in a row: the known reader/writer concurrency test that is sensitive to load.

## Next steps

1. Add the ordering invariant to `insert_one`'s spec (`TreeWF`'s separator chain). It should reject
   the wrong-side bug and allow "`true` ⇒ was absent". It also shows the real cost of tree-level
   parity with Lean, which this spike did not measure.
2. Spec `merge_walk` (a closure-driven merge like `merge_sorted`): the next leaf-level function
   that fits.
3. Give `CowBTree::insert` a real height (a ghost field or a type invariant), so the top level is
   verified too.
