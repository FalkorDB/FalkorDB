import FalkorValueOrder.Glue
import FalkorValueOrder.Json
import FalkorValueOrder.Model
import FalkorValueOrder.SortCmp
/-
# Value order: what `compare_value` promises, and where it breaks

A model of FalkorDB-rs's value comparison (`Value::compare_value`), equality
(`PartialEq for Value`), hashing (`Hash for Value` fed to `FxHasher`) and the
comparators built on them (ORDER BY, DISTINCT, grouping, `=`/`<>`), with
machine-checked proofs of what holds and `decide`-checked counterexamples of
what does not.

## What is modelled, and where it lives in the tree

| here | there |
| --- | --- |
| `V`                      | `enum Value` (`graph/src/runtime/value.rs:180`) — Null, Bool, Int, Float, String, List, Map, Node, VecF32 (components), Point (lat, lon) |
| `F`                      | `f64`/`f32`: NaN (sign, payload), `-0.0`, integral, non-integral (`frac lo r`), ±inf |
| `fpcmp`, `F.nkey`        | `partial_cmp` on floats (`-0.0 == 0.0`, NaN unordered) |
| `cmpVec`, `firstDifference`, `cmpPoint` | `compare_vecf32` (`value.rs:1817`), the Point arm of `compare_value` (`:1247`) |
| `bits64`, `bits32`, `asI64`, `diffZero` | `to_bits`, `as i64`, the `x - casted == 0` test of the Float hash arm |
| `Tag`, `order`           | `impl OrderedEnum for Value` (`value.rs:1177`) — all 16 variants |
| `roundF`                 | `i as f64` for `i: i64` (round-to-nearest, ties-to-even at 53 bits) |
| `cmpF`                   | `compare_floats` (`value.rs:1801`) — `partial_cmp`, NaN ↦ `(Less, NaN)` |
| `cmpV`                   | `impl CompareValue for Value::compare_value` (`value.rs:1224`) |
| `St`, `step`, `listLoop`, `listFinish` | `Value::compare_list` (`value.rs:1399`) — the four counters, line by line |
| `keysCmp`, `mapVals`     | `Value::compare_map` (`value.rs:1472`), maps held with keys already sorted |
| `eqV`                    | `impl PartialEq for Value` (`value.rs:1215`) — `compare_value(..).0 == Equal` |
| `sortCmp`                | ORDER BY: `OrderedKey::cmp_key` (`ops/sort.rs:76`), `Column::compare_at` (`batch.rs:408`) |
| `floatLane`              | ORDER BY float lane (`ops/sort.rs:347`): `partial_cmp(..).unwrap_or(Less)` |
| `evalEq`, `evalNeq`      | `compare_values` `CmpOp::Eq`/`Neq` (`runtime/vectorized.rs:239`) |
| `hwords`, `floatWord`    | `impl Hash for Value` (`value.rs:822`) as the word stream it feeds a `Hasher` |
| `fxAdd`, `fxFinish`, `fx`| `rustc_hash::FxHasher` 2.1.3 (`add_to_hash`, `finish` = `rotate_left(26)`) |
| `rowHash`                | `DistinctOp::next` (`ops/distinct.rs:79`), `ValuesDeduper::is_seen` (`value.rs:1869`) |
| `minFold`                | `min_batch` (`functions/aggregation.rs:487`) |

## Findings (each one reproduced against the real engine, see REPORT.md)

1. `nan_breaks_antisymmetry` — NaN is `Less` than 1.0 *and* 1.0 is `Less` than
   NaN. `slice::sort_by` (Rust ≥ 1.81) detects this and panics; the panic kills
   `redis-server`. Twenty-one floats with a NaN among them in `ORDER BY` do it.
2. `mixed_numeric_not_transitive` — `2^53 = 2^53.0 = 2^53+1` but `2^53 < 2^53+1`.
   Same panic from `ORDER BY` over a mixed int/float column near 2^53.
3. `map_order_not_transitive` — `compare_map` answers `Equal` at the first value
   pair that is disjoint or null, without looking further. Same panic.
4. `map_neq_both_false` — hence `{a:1} = {a:'x'}` and `{a:1} <> {a:'x'}` are
   both `false` (C and openCypher: `<>` is `true`).
5. `hash_not_consistent_with_eq` — `2^53+1 == 2^53.0` but their hashes differ.
6. `nan_eq_irreflexive` — `NaN != NaN` under `PartialEq`, but NaNs hash
   alike; grouping (`HashMap<GroupKey,_>`, `aggregate.rs:59`) needs a
   reflexive `Eq`, so every NaN row becomes its own group.
7. `fx_row_collision`, `fx_collision_general` — DISTINCT, `count(DISTINCT)` and
   the MERGE pattern cache keep *only* the 64-bit Fx hash. Fx is linear, so a
   collision is one subtraction away: rows `(a, b)` and `(a+1, b - K²)` always
   collide, and the second one is silently dropped (or, for MERGE, never
   created).
8. `min_nan_picks_larger` — `min([1.0, NaN, 3.0])` is `3.0`.

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
* `floatLane_eq_cmpV` — the float fast lane of ORDER BY is the same comparator,
  so fixing `compare_floats` fixes both.

## Gaps (what the model does NOT cover)

* Non-integral floats are `frac lo r` (between `lo` and `lo+1`, ranked by `r`);
  their `to_bits` is some function of the value, not the IEEE layout. (Until
  2026-10-11 floats were only NaN or integral, with no `-0.0` and one NaN: that
  hid the #2917 vector/point hash bug — see FINDINGS.md "Wave 9".)
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
| `Glue.W` (16 variants), `W.clone`, `W.order`, `W.getType`, `W.valueOfType`, `W.name` | value.rs:180, :221, :1177, :1328, :1282, :1352 |
| `DeletedNode.new`, `DeletedRelationship.new` | value.rs:69, :87 |
| classes `OrderedEnum`, `CompareValue`, `ValueTypeOf`, `ValueGetType`, `Contains` | trait decls value.rs:1174, :1209, :1275, :1324, :1756 |
| `withCapacity`, `checkAndInsert` | value.rs:1859, :1887 |
| `Json.fmtS`/`fmtJson`, `writeJsonString`, `writeNodeJson`, `relJson`, `toJsonString` | value.rs:1582, :1704, :1716, :1622, :1374/:1384 |

f64/f32 Display, `escape_str` and the runtime accessors are fields of the `JsonEnv`
structure (no axioms). Theorems: `clone_eq`, `name_eq_iff_order_eq`, `valueOfType_self`,
`getType_concrete`, `partialCmpVia_none_iff`, `containsImpl_false`, `withCapacity_seen_iff`,
`DeletedRelationship.new_injective`, `fmt_float_nonfinite`, `fmt_list`,
`writeNodeJson_includeType`, `fmtS_entityFree`, `writeNodeJson_depth`.

## Wave 9 (2026-10-11): extended floats, vector/point contents, #2912 `sort_cmp`

Model.lean: `F` gained NaN sign/payload, `-0.0`, fractions and ±inf; `V` gained
`vecf32`/`point` contents with `compare_vecf32` and the Point arm of `compare_value`, and
`hwords` their hash arms (`to_bits` per component). Every earlier theorem still holds
(`F.qnan` is the old `nan`).

| here | there (origin/main b033aecc6, value.rs; ahead of the e8f8a3017 baseline) |
| --- | --- |
| `Laws.lean`: `Lw`, `thn`, `lexA`, `zipNe`, `cmp3_lw` | generic comparator laws |
| `SC.scF64`, `SC.scIntFloat`, `SC.sc` | `sort_cmp_f64` (:1888, also `sort_cmp_f32` :1900), `sort_cmp_int_float` (:1914), `Value::sort_cmp` (:1526), `sort_cmp_map` (:1563) |

* `SC.scF64_key`, `SC.scIntFloat_key` — the float and Int/Float arms order by a numeric key
  (NaN on top, `-0.0 = 0.0`, no rounding for any `i64`).
* **`SC.sc_laws`, `SC.scMain_total`** — #2912's `sort_cmp` is a total preorder on every
  well-formed value, nested lists/maps/vectors/points included (generic in `sort_cmp_f32`).
* **Refuted on main** — `sort_cmp`-equal ⇒ hash-equal: `SC.main_hash_not_consistent`, via
  `main_float_nan_hash_differs` (`NaN` vs `-NaN`), `main_vec_nan_hash_differs`,
  `main_vec_zero_hash_differs`, `main_point_zero_hash_differs`, `main_point_nan_hash_differs`.
  Nothing on main pairs `sort_cmp` with `Hash` yet; PR #2917 does (proofs/pr2917_review).
  The Float NaN half is observable today through hash-only DISTINCT: `UNWIND [0.0/0.0,
  -(0.0/0.0)] AS x RETURN count(DISTINCT x)` is 2 on main, 1 in C (FINDINGS W9-hash-1).
* `SC.vec_zero_eq_but_hash_differs` — main's `GroupKey` equality (`Value::eq`) also merges
  `vecf32([-0.0])`/`vecf32([0.0])` and the same points while hashing them apart (a
  `HashMap` contract violation; live, both land in separate groups, as in C).
-/
