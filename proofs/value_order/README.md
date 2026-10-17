# value_order: how FalkorDB compares, sorts, groups and hashes values

This project proves what the engine's value comparison promises, and shows with concrete counterexamples where it used to break. Several user-facing features depend on it:

- `ORDER BY`;
- `DISTINCT` and `count(DISTINCT …)`;
- grouping in aggregations;
- `=` and `<>`;
- `min` and `max`;
- `list.sort`.

Each of them relies on one of a small number of comparison and hashing functions in `graph/src/runtime/value.rs`. If one of those functions breaks a basic law, the bug shows up in every feature built on it.

The full Lean↔Rust mapping (every Lean name next to the Rust `file:line` it models) is in the header of [`FalkorValueOrder.lean`](FalkorValueOrder.lean). This README explains *what* is proven and *why it matters*.

## The questions this project answers

**1. Is `ORDER BY` a valid sort?**

Rust's `slice::sort_by` requires the comparator to be a *total order*:

- **reflexive:** `cmp(a, a) = Equal`;
- **antisymmetric:** `cmp(b, a)` is the reverse of `cmp(a, b)`;
- **transitive:** `a ≤ b` and `b ≤ c` imply `a ≤ c`.

Since Rust 1.81 the sort checks this while it runs, and it panics, killing `redis-server`, when it catches the comparator contradicting itself.

The project answers the question both ways:

- **Before #2912, no.** The old `compare_value` broke all three laws, and Lean found the inputs (see [Bugs found](#bugs-found)).
- **On main, yes.** `Value::sort_cmp` is proven to be a total order on every value Rust can build. That includes nested lists, maps, vectors and points.

**2. Do "equal" and "same hash" agree?**

Grouping (`GroupKey`) puts rows in a `HashMap`. The `HashMap` contract is that equal keys have equal hashes. If two keys compare equal but hash differently, the same group appears twice.

`main_hash_consistent` proves the contract holds on main: if `sort_cmp` says equal, the hash words are equal. Before #2917 this was false for vectors and points containing `-0.0` or NaN.

**3. Does DISTINCT keep exactly one row per value?**

`DISTINCT` stores only a 64-bit FxHash per row. The project proves this is fine for NaN on main: every NaN is one key with one hash (`main_nans_one_key`). It also proves the underlying risk: FxHash is linear, so a collision can be built on purpose. In `fx_row_collision`, rows `(a, b)` and `(a+1, b − K²)` always collide, and the second one is silently dropped. This one is still open.

**4. Do `=`, `<>`, `min` behave like Cypher says?**

`<>` is now exactly `NOT =` (`evalNeq_not_evalEq`). `min` ignores NaN as C does (`min_nan_ignored`).

## The headline theorems

| Theorem | In plain words | File |
|---|---|---|
| `scMain_total` | `sort_cmp` is reflexive, antisymmetric and transitive on every well-formed value, so `ORDER BY` and `list.sort` can never panic | `Main.lean` |
| `sc_laws` | The induction behind it: the laws hold for all values of size ≤ n, for **any** lawful f32 comparator | `SortCmp.lean` |
| `cmpKey_total`, `compareAt_eq_scMain`, `compareRowsAt_eq_scMain` | The three places that sort rows (the sort op, typed batch columns, multi-key rows) all reduce to `sort_cmp` | `Main.lean` |
| `listSort_perm` | `list.sort` returns a permutation of its input | `Main.lean` |
| `main_hash_consistent` | `sort_cmp`-equal ⇒ equal hash words (the `GroupKey` `HashMap` contract) | `Main.lean` |
| `main_nans_one_key` | All NaNs (any sign, any payload) are one key and one hash, for floats and vector components | `Main.lean` |
| `signed_zeros_apart` | Float `-0.0` and `0.0` are one key; inside vectors and points they stay apart, as in C | `Main.lean` |
| `scMain_fixes_2891` | The exact inputs from issue #2891 are now ordered correctly | `Main.lean` |
| `list_fst_eq_lex` | The ordering half of `compare_list` is plain lexicographic order, whatever its four counters do: lists were never the problem, maps were | `Model.lean` |
| `roundF_exact` | Every integer with \|i\| < 2⁵³ is exactly representable as `f64` | `Model.lean` |

### How `scMain_total` is proved

1. **Model the type.** `V` has one constructor per `Value` variant that matters for ordering. `F` models a float: NaN with sign and payload, `-0.0`, exact integers, a non-integral value between two integers, and ±∞.
2. **Model the function.** `SC.sc` is `Value::sort_cmp`, written arm for arm in the same order as the Rust `match`. `thn` is `Ordering::then`.
3. **Prove by induction on size.** `IHn n` bundles the three laws for all values of size ≤ n. Size 0 is empty. For size n+1, a list or map only contains smaller values, so the n case covers its elements, and the scalar cases are direct.
4. **Restrict to well-formed values.** `wf` says Ints fit in i64, float `num n` is exactly representable, and a map has one value per key. Rust guarantees all three, so the restriction loses nothing.

## Bugs found

Every bug was reproduced against a live server before it was recorded (see [`../FINDINGS.md`](../FINDINGS.md)).

| # | Theorem (counterexample) | What it means | Status |
|---|---|---|---|
| 1 | `nan_breaks_antisymmetry` | NaN < 1.0 **and** 1.0 < NaN. `ORDER BY` over stored NaNs (21+ rows) panicked and killed the server | Fixed by #2912 (`pre2912_sortCmp_not_antisymmetric`) |
| 2 | `mixed_numeric_not_transitive` | `2⁵³ = 2⁵³.0 = 2⁵³+1` but `2⁵³ < 2⁵³+1`: ints were rounded through `f64`. Same panic | Fixed by #2912 |
| 3 | `map_order_not_transitive` | `compare_map` returned `Equal` at the first mixed-type value pair, which creates a cycle. Same panic | Fixed by #2912 |
| 4 | `pre2917_map_neq_both_false` | `{a:1} = {a:'x'}` and `{a:1} <> {a:'x'}` were both false | Fixed by #2917 |
| 5 | `hash_not_consistent_with_eq` | `2⁵³+1 == 2⁵³.0` under `PartialEq`, but the hashes differ | Harmless: grouping now uses `sort_cmp` |
| 6 | `nan_eq_irreflexive` | `NaN != NaN`, so each NaN row became its own group | Fixed by #2917 (`main_nans_one_key`) |
| 7 | `fx_row_collision`, `fx_collision_general` | DISTINCT, `count(DISTINCT)` and the MERGE cache keep only the 64-bit Fx hash; a collision is one subtraction away and the second row is dropped | **Present on main** (#2780 closed as not planned) |
| 8 | `pre2917_min_nan_picks_larger` | `min([1.0, NaN, 3.0])` returned 3.0 | Fixed by #2917 (`min_nan_ignored`) |
| W9-hash-1 | `pre2917_float_nan_hash_differs` | `count(DISTINCT x)` over `[NaN, -NaN]` was 2 (C: 1) | Fixed by #2917 |

### The bug this project missed, and what changed

The first version of the float model had one NaN and one zero. On #2917 a code review found that `GroupKey` equality and `Value::hash` disagreed on vectors containing `-0.0` or a NaN with another payload. The proof didn't catch it for two reasons:

- the model couldn't express those values;
- "equal ⇒ same hash" was an *assumption* of the grouping theorem, not something we proved.

Since then:

- the float model has NaN sign and payload and `-0.0`;
- the assumption is the proven theorem `main_hash_consistent`;
- results that still rest on an unproven assumption are tagged **CONDITIONAL** in `COVERAGE.tsv` across the repo.

## Files

| File | What's in it |
|---|---|
| `Model.lean` | `V`, the type order (`Tag`, `orderTag`), the pre-#2912 `compare_value`, `compare_list`/`compare_map`, `PartialEq`, `Hash`, FxHasher; counterexamples 1–8 |
| `SortCmp.lean` | `sort_cmp` (`SC.sc`), well-formedness `wf`, and the induction `sc_laws` |
| `Main.lean` | `sort_cmp` as on main (`scMain`), its total order, the sort call sites, hash consistency, NaN and signed-zero theorems |
| `Glue.lean` | Small `value.rs` glue: `clone`, `get_type`, `name`, `DeletedNode::new`, the deduper |
| `Json.lean` | JSON output of values (`toJsonString`, node/relationship JSON) |
| `COVERAGE.tsv` | One row per Rust function: bucket and the theorem that covers it |

Shared foundations (the float model `F`, `fpcmp`, i64 helpers, comparator laws `Lw`/`thn`/`lexA`) come from [`../core`](../core).

## Coverage

`COVERAGE.tsv` has 84 rows, mostly `value.rs`:

| Bucket | Rows |
|---|---|
| PROVEN | 37 |
| MODELLED | 2 |
| NOT COVERED | 45 |

- **MODELLED:** `compare_vecf32` and `first_difference` are modelled here and proven in `value_math` (`chunked_eq_firstDiff`).
- **NOT COVERED:** mostly geometry and accessors (`Point::distance`, `get_numeric`, …), which are outside this project's question. Some are proven in other projects; `bash proofs/coverage.sh` counts each function once, in its best bucket.

## What the model does not cover

- Non-integral floats are `frac lo r`, a value between `lo` and `lo + 1` ranked by `r`. Their bit pattern is abstract, not the IEEE layout.
- Strings are numbers compared numerically. Rust compares bytes, but any total order gives the same laws. Their hash words are abstract.
- Path, Relationship and the temporal types appear only through the type order: Path shares `compare_list`, and the temporals share Int's `cmp`.
- The model checks the comparator, not Rust's sort algorithm. The Rust standard library guarantees that a valid total order makes `sort_by` behave.

## Build and check

```bash
cd proofs/value_order && lake build    # this project
bash proofs/lean_ci.sh value_order     # build + no sorry/admit/native_decide/unmarked axiom
bash proofs/coverage.sh                # repo-wide coverage report
```

**When you change `value.rs`, `ops/sort.rs`, `batch.rs` or `vectorized.rs`:**

1. Find the function in `COVERAGE.tsv`.
2. Update its Lean model to match.
3. Re-run the build.

If a theorem stops holding, either the change introduced a bug or the theorem needs a new hypothesis. Either way, say which in the PR.
