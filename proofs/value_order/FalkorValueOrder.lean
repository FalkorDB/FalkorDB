import FalkorValueOrder.Glue
import FalkorValueOrder.Json
import FalkorValueOrder.Model
import FalkorValueOrder.SortCmp
import FalkorValueOrder.Main
/-
# Value order: what `compare_value` promises, and where it breaks

**Re-synced to main 256ab4d3f**: #3861 adds a `List(Any)` skip to
`value_of_type` (modelled in proofs/value_math `vot_list_any`); cites re-pinned; the former NOT
COVERED / MODELLED rows (temporal, arithmetic, memory accounting, codec, vectorized.rs kernels)
point at the projects that prove them.

A model of FalkorDB-rs's value comparison (`Value::compare_value`), equality
(`PartialEq for Value`), hashing (`Hash for Value` fed to `FxHasher`) and the
comparators built on them (ORDER BY, DISTINCT, grouping, `=`/`<>`), with
machine-checked proofs of what holds and `decide`-checked counterexamples of
what does not.

## What is modelled, and where it lives in the tree

| here | there |
| --- | --- |
| `V`                      | `enum Value` (`graph/src/runtime/value.rs:184`) — Null, Bool, Int, Float, String, List, Map, Node, VecF32 (components), Point (lat, lon) |
| `F`                      | `f64`/`f32`: NaN (sign, payload), `-0.0`, integral, non-integral (`frac lo r`), ±inf |
| `fpcmp`, `F.nkey`        | `partial_cmp` on floats (`-0.0 == 0.0`, NaN unordered) |
| `cmpVec`, `firstDifference`, `cmpPoint` | `compare_vecf32` (`value.rs:2009`), the Point arm of `compare_value` (`:1288`) |
| `bits64`, `bits32`, `asI64`, `diffZero` | `to_bits`, `as i64`, the `x - casted == 0` test of the Float hash arm |
| `Tag`, `order`           | `impl OrderedEnum for Value` (`value.rs:1219`) — all 16 variants |
| `roundF`                 | `i as f64` for `i: i64` (round-to-nearest, ties-to-even at 53 bits) |
| `cmpF`                   | `compare_floats` (`value.rs:1919`) — `partial_cmp`, NaN ↦ `(Less, NaN)` |
| `cmpV`                   | `impl CompareValue for Value::compare_value` (`value.rs:1266`) |
| `St`, `step`, `listLoop`, `listFinish` | `Value::compare_list` (`value.rs:1446`) — the four counters, line by line |
| `keysCmp`, `mapVals`     | `Value::compare_map` (`value.rs:1519`), maps held with keys already sorted |
| `eqV`                    | `impl PartialEq for Value` (`value.rs:1256`) — `compare_value(..).0 == Equal` |
| `sortCmp`                | `compare_value(..).0`, the old ORDER BY comparator (kept as history); main sorts by `Main.scMain` (`OrderedKey::cmp_key` `ops/sort.rs:76`, `Column::compare_at` `batch.rs:399`: `Main.cmpKey`, `Main.compareAt`) |
| `evalEq`, `evalNeq`      | `compare_values` `CmpOp::Eq`/`Neq` (`runtime/vectorized.rs:239`) |
| `hwords`, `floatWord`, `f32HashBits` | `impl Hash for Value` (`value.rs:861`) as the word stream it feeds a `Hasher`; `f32_hash_bits` (`:1961`) |
| `fxAdd`, `fxFinish`, `fx`| `rustc_hash::FxHasher` 2.1.3 (`add_to_hash`, `finish` = `rotate_left(26)`) |
| `rowHash`                | `DistinctOp::next` (`ops/distinct.rs:62`), `ValuesDeduper::is_seen` (`value.rs:2061`) |
| `minFold`                | `min_batch` (`functions/aggregation.rs:498`) |

## Findings (each one reproduced against the real engine, see REPORT.md)

Status on main: 1–3 are facts about `compare_value` (the theorems `nan_breaks_antisymmetry`,
`mixed_numeric_not_transitive`, `map_order_not_transitive` remain in `Model.lean`), but nothing sorts by it:
ORDER BY uses `sort_cmp` (`Main.scMain_total`). 5 is closed for grouping only (`GroupKey::eq` is `sort_cmp`);
`Value`'s own `Hash`/`PartialEq` still disagree and the value hash join uses them (see finding 5), so it is
open outside grouping. 7 is open.

1. `nan_breaks_antisymmetry` — NaN is `Less` than 1.0 *and* 1.0 is `Less` than
   NaN, so `compare_value` is not a sound `slice::sort_by` comparator.
2. `mixed_numeric_not_transitive` — `2^53 = 2^53.0 = 2^53+1` but `2^53 < 2^53+1`.
3. `map_order_not_transitive` — `compare_map` answers `Equal` at the first value
   pair that is disjoint or null, without looking further.
5. `hash_not_consistent_with_eq` — `2^53+1 == 2^53.0` but their hashes differ.
   Grouping is not affected: `GroupKey` hashes/compares by `sort_cmp`, and main's hash agrees with `sort_cmp`
   (`Main.main_hash_consistent`). **Open outside grouping**: `Value`'s own `Hash` and `PartialEq`
   (`compare_value`) remain inconsistent, and `ValueHashJoin` (`ops/value_hash_join.rs`) hashes `Value`
   (`hash_value`, and the `i64`-keyed `key_as_i64` fast path) and then checks `keys_match`
   (`compare_value`): `Int(2^53+1)` and `Float(2^53)` compare equal (`compare_floats(i as f64, f)`) but hash
   to different buckets, so the join can miss a match that `=` in a filter finds. Found by reading the
   Rust; not reproduced against a live server.
7. `fx_row_collision`, `fx_collision_general` — DISTINCT, `count(DISTINCT)` and
   the MERGE pattern cache keep *only* the 64-bit Fx hash. Fx is linear, so a
   collision is one subtraction away: rows `(a, b)` and `(a+1, b - K²)` always
   collide, and the second one is silently dropped (or, for MERGE, never
   created).

## What is proved to hold

* `order_injective`, `no_tag_between_int_and_float` — the type order is a
  strict order on tags and nothing sits between Int and Float, which is what
  lets numeric cross-type comparison coexist with the tag order.
* `cmp_sane_eq_key` — on the *sane* scalar fragment (no NaN, every Int exactly
  representable as f64) `compare_value` is a lexicographic compare of a key;
  hence `sane_refl`, `sane_swap`, `sane_le_trans`: a total preorder.
* `roundF_exact` — every |i| < 2^53 is exactly representable, so the sane
  fragment contains every Int a user normally meets.
* `sane_hash_consistent` — on that fragment `a == b ⇒ hash a = hash b`.
* `list_fst_eq_lex` — the ORDER-BY half of `compare_list` *is* plain
  lexicographic order (first unequal element, then length), whatever the four
  counters do; `lex_swap`, `lex_le_trans` then lift any total preorder on the
  elements to lists — so lists are not where ORDER BY goes wrong; maps are.
* Main (Main.lean): `scMain_total` (`sort_cmp` is a total preorder), `cmpKey_total`,
  `compareAt_eq_scMain`, `compareRowsAt_eq_scMain`, `listSort_perm`, `listSort_sorted`, `main_hash_consistent`
  (`sort_cmp`-equal ⇒ equal hash words), `main_nans_one_key`, `signed_zeros_apart`, `min_nan_ignored`.

## Gaps (what the model does NOT cover)

* Non-integral floats are `frac lo r` (between `lo` and `lo+1`, ranked by `r`);
  their `to_bits` is some function of the value, not the IEEE layout.
* Strings are `Nat`s compared numerically (Rust: byte order; any total order
  gives the same laws). Their hash words are abstracted.
* Path, Relationship and the temporals appear only as `Tag`s for the type order;
  `Path` shares `compare_list`, temporals share Int's `cmp`.
* Maps are held with sorted keys and one value per key (`compare_map` sorts
  first).

## Checking it

    cd proofs/value_order && lake build

## Wave 5: value.rs glue and JSON (`FalkorValueOrder/Glue.lean`, `Json.lean`)

| here | there |
| --- | --- |
| `Glue.W` (16 variants), `W.clone`, `W.order`, `W.getType`, `W.valueOfType`, `W.name` | value.rs:184, :223, :1219, :1373, :1323, :1399 |
| `DeletedNode.new`, `DeletedRelationship.new` | value.rs:101, :119 |
| classes `OrderedEnum`, `CompareValue`, `ValueTypeOf`, `ValueGetType`, `Contains` | trait decls value.rs:1214, :1249, :1315, :1368, :1873 |
| `withCapacity`, `checkAndInsert` | value.rs:2051, :2079 |
| `Json.fmtS`/`fmtJson`, `writeJsonString`, `writeNodeJson`, `relJson`, `toJsonString` | value.rs:1700, :1822, :1834, :1740, :1421 |

f64/f32 Display, `escape_str` and the runtime accessors are fields of the `JsonEnv`
structure (no axioms). Theorems: `clone_eq`, `name_eq_iff_order_eq`, `valueOfType_self`,
`getType_concrete`, `partialCmpVia_none_iff`, `containsImpl_false`, `withCapacity_seen_iff`,
`DeletedRelationship.new_injective`, `fmt_float_nonfinite`, `fmt_list`,
`writeNodeJson_includeType`, `fmtS_entityFree`, `writeNodeJson_depth`.

## Main model (`Main.lean`)

* Model.lean: `hwords` is main's hash (Float NaN → one word, f32 components via
  `f32_hash_bits`). `evalNeq` and `minFold` follow main.
* Main.lean: `scF32` (main's `sort_cmp_f32`, `-0.0 < 0.0`),
  `scMain = sc scF32` with `scMain_total`; `main_hash_consistent`; `cmpKey`/`compareAt`/`compareRowsAt`/
  `listSort` for ORDER BY, typed lanes and `list.sort`.
* COVERAGE: rows for `sort_cmp`, `sort_cmp_map`, `sort_cmp_f64`, `sort_cmp_f32`,
  `f32_hash_bits`, `sort_cmp_int_float`, `compare_rows_at`, `list_sort`.

## Wave 9: extended floats, vector/point contents, `sort_cmp`

Model.lean: `F` gained NaN sign/payload, `-0.0`, fractions and ±inf; `V` gained
`vecf32`/`point` contents with `compare_vecf32` and the Point arm of `compare_value`, and
`hwords` their hash arms (`to_bits` per component).

| here | there (value.rs at b033aecc6; see Main.lean for the current lines) |
| --- | --- |
| proofs/core `FalkorCore/Laws.lean`: `Lw`, `thn`, `lexA`, `zipNe`, `cmp3_lw` | generic comparator laws (shared core; `F`, `fpcmp`, `cmpI`, bit layouts also live in `FalkorCore`, re-exported by Model.lean) |
| `SC.scF64`, `SC.scIntFloat`, `SC.sc` | `sort_cmp_f64` (:1888, also `sort_cmp_f32` :1900), `sort_cmp_int_float` (:1914), `Value::sort_cmp` (:1526), `sort_cmp_map` (:1563) |

* `SC.scF64_key`, `SC.scIntFloat_key` — the float and Int/Float arms order by a numeric key
  (NaN on top, `-0.0 = 0.0`, no rounding for any `i64`).
* **`SC.sc_laws`** (then `scMain_total`) — `sort_cmp` is a total preorder on every
  well-formed value, nested lists/maps/vectors/points included (generic in `sort_cmp_f32`).
* `SC.vec_zero_eq_but_hash_differs` — `Value::eq` merges `vecf32([-0.0])`/`vecf32([0.0])` and
  the same points while hashing them apart. `GroupKey::eq` is `sort_cmp`, which keeps them apart.
-/
