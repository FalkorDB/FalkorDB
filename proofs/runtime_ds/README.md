# runtime_ds: the small data structures every query runs on

This project proves that the runtime's basic containers do what their callers assume. They are small, but every query touches them:

- `Row` (`runtime/row.rs`): the slots that hold a row's variables, plus a bitmap saying which are bound;
- `BitSet` (`runtime/bitset.rs`): that bitmap;
- `OrderMap` and `OrderSet` (`runtime/ordermap.rs`, `runtime/orderset.rs`): insertion-ordered maps and sets, used for property maps, Cypher map values and label sets;
- `StringPool` (`runtime/string_pool.rs`): interning of attribute and label names;
- `Cow` (`graph/cow.rs`): the copy-on-write wrapper MVCC uses to give a write query its own version of a matrix;
- `narrow_int::width_for` and `identifier_limits`: two small helpers for integer widths and name-length checks.

If one of these breaks its contract, the result is not a single wrong query but a whole class of them: a variable that reads back the wrong value, a map whose `=` disagrees with its hash, or a read query that sees a writer's half-finished matrix.

Each `.lean` file starts with its own Lean↔Rust table (every Lean name next to the Rust `file:line` it models). The root [`FalkorRuntimeDS.lean`](FalkorRuntimeDS.lean) only imports them. This README explains *what* is proven and *why it matters*.

## The questions this project answers

**1. Does a row read back what was written into it?**

Yes. `getById_insertById` says that after `insert_by_id(id, v)`, slot `id` holds `v`, every other existing slot is unchanged, and any gap created by growing the row reads as `Null`. `isBound_insertById` and `isBound_unbind` say the bound bit changes for exactly that one slot. `merge_get_bound` says `Row::merge` copies exactly the bound slots of the other row.

Two details are worth knowing, because they are easy to get wrong in operator code:

- `unbind` clears the bound bit but leaves the old value in the slot (`getById_unbind`).
- `take` moves the value out and leaves `Null`, but keeps the bound bit (`take_leaves_null`, `take_keeps_bound`). A taken slot is "bound to `Null`", not "unbound".

**2. Is the bound bitmap a real set, including across the 64-bit word boundary?**

Yes. `BitSet` keeps the first 64 bits inline and the rest in an overflow vector. `test_set`, `test_clear` and `test_union` prove that `set`, `clear` and `union` add, remove and combine exactly the right bits, for any bit number, so bit 63 and bit 64 are handled correctly. `isEmpty_iff` proves `is_empty` is true exactly when no bit is set, which is what `Row::has_bindings` relies on (`hasBindings_iff`).

**3. Is `OrderMap` a correct map, and do its `=` and hash agree?**

Yes, as long as keys are unique, which every constructor except one guarantees:

- `insert` never creates a duplicate key (`insertL_nodup`), keeps an existing key in its slot and appends a new one (`keys_insertL`);
- `get` after `insert` or `remove` returns what you expect (`getL_insertL`, `getL_removeL`);
- building from a list is "last write wins" per key (`getL_fromList_snoc`), which is what `{a:1, a:2}` relies on;
- `=` ignores order: two maps are equal exactly when they hold the same entries in any order (`eqM_iff_perm`), so `{a:1, b:2} = {b:2, a:1}`;
- equal maps hash the same (`hash_consistent`), because `Hash` sorts the pairs by key before hashing.

The exception is `from_unique_keys`, which checks uniqueness only with a `debug_assert!`. See [Bugs found](#bugs-found).

**4. Is `OrderSet` a correct set?**

Yes, under the same caveat. `insert` never duplicates (`insertS_nodup`), `remove` removes (`containsS_removeS`, given no duplicates), and `get_index_of` shifts by exactly one for every element after a removed one (`indexOf_removeS`). Unlike `OrderMap`, `OrderSet`'s `=` is order-sensitive: it is the derived list equality.

**5. Does string interning give one pointer per string, and keep it stable?**

Yes. `intern_injective` says two different strings never share a pointer. `intern_idem` says interning the same string twice returns the same pointer. `prune_keeps_held` says the pool's cleanup in `stats` never evicts a string that someone still holds. `reintern_fresh` says a string interned again after being pruned gets a brand-new pointer, so an old pointer can never be mistaken for it (no ABA).

**6. Does `Cow::new_version` isolate the committed version from the writer?**

Yes, in one direction. `newVersion_isolated` proves that after `v2 = v1.new_version()`, no sequence of writes or replaces through `v2` changes what `v1` reads. `siblings_isolated` proves the same for two versions taken from one committed value. This is the property MVCC needs: a read query holding the committed matrix never sees the write query's changes.

The other direction does not hold, by design: writing through `v1` after taking `v2` changes what `v2` sees. MVCC relies on the committed version never being written once a new version exists.

## The headline theorems

| Theorem | In plain words | File |
|---|---|---|
| `getById_insertById` | After writing slot `id`, it holds the new value, other slots are unchanged, and padding reads as `Null` | `Row.lean` |
| `take_leaves_null`, `take_keeps_bound` | `take` leaves `Null` in the slot and keeps it marked bound | `Row.lean` |
| `merge_get_bound` | `Row::merge` copies exactly the bound slots of the other row | `Row.lean` |
| `hashed_pad` | Trailing unbound `Null` padding does not change a row's hash | `Row.lean` |
| `test_set`, `test_clear`, `test_union` | `BitSet` operations change exactly the right bits, across word boundaries | `BitSet.lean` |
| `isEmpty_iff` | `is_empty` is exact | `BitSet.lean` |
| `eqM_iff_perm`, `eqM_symm` | For unique-key maps, `OrderMap` `=` means "same entries, any order", so it is symmetric | `OrderMap.lean` |
| `hash_consistent` | Equal `OrderMap`s hash the same | `OrderMapExtra.lean` |
| `getL_fromList_snoc` | Building a map from a list: the last value for a key wins | `OrderMap.lean` |
| `indexM_panics_iff`, `indexS_panics_iff` | `map[&k]` and `set[i]` panic exactly when the key or index is absent | `OrderMapExtra.lean`, `OrderSetExtra.lean` |
| `intern_injective`, `prune_keeps_held`, `reintern_fresh` | One pointer per string; held strings survive cleanup; no pointer reuse | `StringPool.lean` |
| `newVersion_isolated` | Writes through a new version never reach the committed one | `Cow.lean` |
| `widthFor_fits`, `widthFor_minimal`, `narrow_lossless` | `width_for` picks the narrowest of 1/2/4/8 bytes that holds the maximum, and narrowing loses nothing | `NarrowInt.lean` |
| `validate_ok_iff`, `isTooLong_of_validate` | A name is accepted iff its UTF-8 byte length is ≤ 512, and every rejection is recognised by `is_identifier_too_long` | `IdentifierLimits.lean` |

### How `newVersion_isolated` is proved

1. **Model the type.** In Rust, cloning a GraphBLAS `Matrix` is a shallow handle copy (the clone shares the underlying `GrB_Matrix`), and `dup()` is the deep copy. So the model is a heap of handles: `Heap` maps a handle number to contents, and `Cow` is a handle `h` plus the `dup` flag.
2. **Model the functions.** `newVersion` copies the handle and sets `dup := true`. `derefMut` is `DerefMut`: if `dup` is set, allocate a fresh handle with a copy of the contents and clear `dup`. `write` is `derefMut` followed by a mutation of the handle; `replace` always allocates fresh.
3. **State what "not the owner" means.** `NotOwner c x` says `c` either points at another handle or has `dup` set, so it will copy before writing.
4. **One step at a time.** `step_frame` proves that a single write or replace through a `Cow` that does not own handle `x` leaves `x`'s contents unchanged, and the `Cow` still does not own `x` afterwards.
5. **Induction on the list of operations.** A new version starts as a non-owner of `v1`'s handle (its `dup` is set), so by `step_frame` every operation in the list preserves `v1`'s contents.

The only hypothesis is `v1.h < H.next`: `v1`'s handle was allocated before. Every real `Cow` satisfies it.

## Bugs found

None of these is reachable from Cypher today. They are contracts the code does not check, filed together as #2903 (PR #2905, not merged).

| # | Theorem (counterexample) | What it means | Status |
|---|---|---|---|
| 1 | `example` blocks at the end of `OrderMap.lean` | `OrderMap::from_unique_keys` checks uniqueness only in debug builds. In a release build, a duplicate key gives a map where `a == b` but `b != a`, and where `remove(k)` leaves `k` reachable. The in-tree callers build from stored attributes or from CSV headers deduplicated when the file is opened, so their keys are unique (not proven here) | Latent, #2903 |
| 2 | `example` blocks at the end of `OrderSet.lean` | `OrderSet::from_vec` does not deduplicate; with a duplicate, `remove` leaves the element in (this is why `containsS_removeS` needs no duplicates). The one caller, `reorder_labels.rs`, passes a permutation of an existing set, so it is safe | Latent, #2903 |
| 3 | `example` block at the end of `Cow.lean` | Copy-on-write isolation is one-directional: a write through the committed version after `new_version` is seen by the new version. MVCC never does this | Latent, by design, #2903 |

The Rust repros (`orderset_from_vec_duplicates`, `ordermap_from_unique_keys_duplicates`, `cow_isolation`) are in `graph/tests/lean_runtime_ds.rs`, which is kept locally and not committed (see [`../FINDINGS.md`](../FINDINGS.md)).

Also worth knowing, not filed as a bug: the examples after `hashed_pad` in `Row.lean` show that a slot emptied by `take` (bound `Null`), and a stale value left by `unbind`, both still feed `Hash for Row`. Two rows that print the same can hash differently depending on how their slots were emptied.

## Files

| File | What's in it |
|---|---|
| `Row.lean` | `Row` as a list of optional values plus a `BitSet`; `insertById`, `getById`, `take`, `unbind`, `merge`, `cloneWith`, the hash input `hashed` |
| `RowExtra.lean` | The `RowView` trait contract (`rowView_lawful`), the remaining `Row` methods, `Cow::new` (`cowNew`), and the `OnceLock` string-pool global (`globalPool_spec`) |
| `BitSet.lean` | `BitSet` with an inline word and overflow words, the well-formedness invariant `WF` (every word < 2⁶⁴), set/test/clear/union/is_empty |
| `OrderMap.lean` | `OrderMap` as a list of pairs; insert/remove/get, last-write-wins construction, `=` as permutation; the `from_unique_keys` counterexamples |
| `OrderMapExtra.lean` | Constructors, accessors, `Index` (panics iff absent), `IntoIterator`, and `Hash` (`hash_consistent`) |
| `OrderSet.lean` | `OrderSet` as a list; insert/remove/contains/`get_index_of`; the `from_vec` counterexamples |
| `OrderSetExtra.lean` | Constructors, accessors, `clear`, `Index`, `IntoIterator` |
| `StringPool.lean` | The pool as `(content, pointer)` entries plus a count of outside references; intern, prune, release and the invariant `WF` |
| `Cow.lean` | `Cow` over a heap of handles; `newVersion_isolated`, `siblings_isolated` and the one-directional counterexample |
| `NarrowInt.lean` | `width_for` and the truncating cast it guards |
| `IdentifierLimits.lean` | `validate_identifier_len` and `is_identifier_too_long` over UTF-8 byte length |
| `COVERAGE.tsv` | One row per Rust function: bucket and the theorem that covers it |

The project has no dependencies beyond Lean core.

## Coverage

`COVERAGE.tsv` has 74 rows, all PROVEN:

| Bucket | Rows |
|---|---|
| PROVEN | 74 |

By file: `row.rs` 21, `ordermap.rs` 20, `orderset.rs` 15, `bitset.rs` 5, `cow.rs` 5, `string_pool.rs` 5, `identifier_limits.rs` 2, `narrow_int.rs` 1.

## What the model does not cover

- `Value` is abstract: a row slot is `Option β`, with `none` for `Null`. Nothing here depends on what a value is.
- `Hash for Row` and `Hash for OrderMap` are modelled as the sequence fed to the hasher, not the hasher itself. `hash_consistent` proves equal maps feed the same sequence, which is enough for equal hashes.
- `OrderMap` keys are compared with an abstract `BEq`; theorems that need it assume `EquivBEq` (reflexive, symmetric, transitive), which `Arc<String>` equality satisfies.
- `StringPool::stats` is modelled only for its pruning; its counting is not. The `Mutex` is not modelled: operations are taken to run one at a time, which the lock guarantees.
- `Cow` models the handle sharing, not GraphBLAS: `dup()` is "allocate and copy contents".
- Capacity, `SmallVec` inline storage and `ThinVec` allocation are invisible in the model; only contents are.

## Build and check

```bash
cd proofs/runtime_ds && lake build    # this project
bash proofs/lean_ci.sh runtime_ds     # build + no sorry/admit/native_decide/unmarked axiom
bash proofs/coverage.sh               # repo-wide coverage report
```

**When you change `row.rs`, `bitset.rs`, `ordermap.rs`, `orderset.rs`, `string_pool.rs`, `cow.rs`, `narrow_int.rs` or `identifier_limits.rs`:**

1. Find the function in `COVERAGE.tsv`.
2. Update its Lean model to match (the table at the top of each `.lean` file says which definition models it).
3. Re-run the build.

If a theorem stops holding, either the change introduced a bug or the theorem needs a new hypothesis. Either way, say which in the PR.
