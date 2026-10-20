import FalkorValueOrder.Glue
import FalkorValueOrder.Json
import FalkorValueOrder.Model
import FalkorValueOrder.SortCmp
import FalkorValueOrder.Main
/-
# Value order: what `compare_value` promises, and where it breaks

**Re-targeted to main 8621212fc** (from 289c201fe, before that fb3d516d0): no
change to `compare_value`, `sort_cmp`, `PartialEq` or `Hash`. #2984 moves the
date ± duration month shift into a new `shift_months` and #3007 edits only
`percentile_batch` (both outside this model; earlier #2983/#2985 edited
`Point::distance` and `Time ± Duration`); COVERAGE.tsv line numbers re-pinned.

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
| `cmpVec`, `firstDifference`, `cmpPoint` | `compare_vecf32` (`value.rs:1963`), the Point arm of `compare_value` (`:1250`) |
| `bits64`, `bits32`, `asI64`, `diffZero` | `to_bits`, `as i64`, the `x - casted == 0` test of the Float hash arm |
| `Tag`, `order`           | `impl OrderedEnum for Value` (`value.rs:1181`) — all 16 variants |
| `roundF`                 | `i as f64` for `i: i64` (round-to-nearest, ties-to-even at 53 bits) |
| `cmpF`                   | `compare_floats` (`value.rs:1875`) — `partial_cmp`, NaN ↦ `(Less, NaN)` |
| `cmpV`                   | `impl CompareValue for Value::compare_value` (`value.rs:1228`) |
| `St`, `step`, `listLoop`, `listFinish` | `Value::compare_list` (`value.rs:1402`) — the four counters, line by line |
| `keysCmp`, `mapVals`     | `Value::compare_map` (`value.rs:1475`), maps held with keys already sorted |
| `eqV`                    | `impl PartialEq for Value` (`value.rs:1219`) — `compare_value(..).0 == Equal` |
| `sortCmp`                | `compare_value(..).0`, the ORDER BY comparator *before* #2912; main sorts by `Main.scMain` (`OrderedKey::cmp_key` `ops/sort.rs:76`, `Column::compare_at` `batch.rs:399`: `Main.cmpKey`, `Main.compareAt`) |
| `floatLanePre2912`       | ORDER BY float lane before #2912: `partial_cmp(..).unwrap_or(Less)` (main: `sort_cmp_f64`, `SC.scF64`) |
| `evalEq`, `evalNeq`      | `compare_values` `CmpOp::Eq`/`Neq` (`runtime/vectorized.rs:239`; `Neq` as #2917, old: `evalNeqPre2917`) |
| `hwords`, `floatWord`, `f32HashBits` | `impl Hash for Value` (`value.rs:823`) as the word stream it feeds a `Hasher`; `f32_hash_bits` (`:1919`); before #2917: `hwordsPre2917` |
| `fxAdd`, `fxFinish`, `fx`| `rustc_hash::FxHasher` 2.1.3 (`add_to_hash`, `finish` = `rotate_left(26)`) |
| `rowHash`                | `DistinctOp::next` (`ops/distinct.rs:79`), `ValuesDeduper::is_seen` (`value.rs:2015`) |
| `minFold`                | `min_batch` (`functions/aggregation.rs:495`; before #2917: `minFoldPre2917`) |

## Findings (each one reproduced against the real engine, see REPORT.md)

Status on main 4a4f2a72f: 1–3 are still facts about `compare_value`, but nothing sorts by it
any more (#2912 b033aecc6: `sort_cmp`, `Main.scMain_total`), so the panics are fixed
(`pre2912_sortCmp_not_antisymmetric`, `pre2912_sortCmp_not_transitive`); 4, 6 and 8 are
fixed by #2917 (7fdf06a0e); 5 is harmless since `GroupKey::eq` is `sort_cmp`; 7 is open.

1. `nan_breaks_antisymmetry` — NaN is `Less` than 1.0 *and* 1.0 is `Less` than
   NaN. Before #2912, `slice::sort_by` (Rust ≥ 1.81) detected this in ORDER BY
   and panicked, killing `redis-server`.
2. `mixed_numeric_not_transitive` — `2^53 = 2^53.0 = 2^53+1` but `2^53 < 2^53+1`.
   Same (pre-#2912) panic from `ORDER BY` over a mixed int/float column near 2^53.
3. `map_order_not_transitive` — `compare_map` answers `Equal` at the first value
   pair that is disjoint or null, without looking further. Same (pre-#2912) panic.
4. `pre2917_map_neq_both_false` — `{a:1} = {a:'x'}` and `{a:1} <> {a:'x'}` were
   both `false`. Fixed by #2917: `map_neq_true`, `evalNeq_not_evalEq` (`<>` = NOT `=`).
5. `hash_not_consistent_with_eq` — `2^53+1 == 2^53.0` but their hashes differ.
   Main's hash agrees with `sort_cmp` instead (`Main.main_hash_consistent`).
6. `nan_eq_irreflexive` — `NaN != NaN` under `PartialEq`; before #2917 grouping
   used it, so every NaN row became its own group. `GroupKey::eq` is now `sort_cmp`
   (NaN equal to NaN, `Main.main_nans_one_key`).
7. `fx_row_collision`, `fx_collision_general` — DISTINCT, `count(DISTINCT)` and
   the MERGE pattern cache keep *only* the 64-bit Fx hash. Fx is linear, so a
   collision is one subtraction away: rows `(a, b)` and `(a+1, b - K²)` always
   collide, and the second one is silently dropped (or, for MERGE, never
   created).
8. `pre2917_min_nan_picks_larger` — `min([1.0, NaN, 3.0])` was `3.0`. Fixed by #2917:
   `min_nan_ignored` (1.0; a leading NaN survives, as C).

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
* `pre2912_floatLane_eq_cmpV` — the pre-#2912 float fast lane was the same comparator.
* Main (Main.lean): `scMain_total` (`sort_cmp` is a total preorder), `cmpKey_total`,
  `compareAt_eq_scMain`, `compareRowsAt_eq_scMain`, `listSort_perm`, `main_hash_consistent`
  (`sort_cmp`-equal ⇒ equal hash words), `main_nans_one_key`, `signed_zeros_apart`.

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

## Re-target to 4a4f2a72f (2026-10-12): #2912 b033aecc6, #2917 7fdf06a0e merged

* Model.lean: `hwords` is main's hash (Float NaN → one word, f32 components via
  `f32_hash_bits`); the old one is `hwordsPre2917`. `evalNeq` and `minFold` follow #2917;
  `evalNeqPre2917`, `minFoldPre2917`, `floatLanePre2912` keep the old code.
* Main.lean (new): `scF32` (main's `sort_cmp_f32`, `-0.0 < 0.0`, copied from pr2917_review),
  `scMain = sc scF32` with `scMain_total`; `main_hash_consistent` (pr2917_review's
  `hashPR_consistent`, re-proved on this `hwords`); `cmpKey`/`compareAt`/`compareRowsAt`/
  `listSort` for ORDER BY, typed lanes and `list.sort`.
* SortCmp.lean: `scPre2917 = sc scF64` (`sort_cmp` at b033aecc6) with `pre2917_sc_total`; its
  hash counterexamples are now `pre2917_*` (fixed: W9-hash-1).
* COVERAGE: rows for `sort_cmp`, `sort_cmp_map`, `sort_cmp_f64`, `sort_cmp_f32`,
  `f32_hash_bits`, `sort_cmp_int_float`, `compare_rows_at`, `list_sort`; `cmp_key` now cites
  the correctness theorem (it used to count the refutation `sortCmp_not_antisymmetric` as
  PROVEN).

## Wave 9 (2026-10-11): extended floats, vector/point contents, #2912 `sort_cmp`

Model.lean: `F` gained NaN sign/payload, `-0.0`, fractions and ±inf; `V` gained
`vecf32`/`point` contents with `compare_vecf32` and the Point arm of `compare_value`, and
`hwords` their hash arms (`to_bits` per component). Every earlier theorem still holds
(`F.qnan` is the old `nan`).

| here | there (value.rs at b033aecc6; since 4a4f2a72f see Main.lean) |
| --- | --- |
| proofs/core `FalkorCore/Laws.lean`: `Lw`, `thn`, `lexA`, `zipNe`, `cmp3_lw` | generic comparator laws (shared core; `F`, `fpcmp`, `cmpI`, bit layouts also live in `FalkorCore`, re-exported by Model.lean) |
| `SC.scF64`, `SC.scIntFloat`, `SC.sc` | `sort_cmp_f64` (:1888, also `sort_cmp_f32` :1900), `sort_cmp_int_float` (:1914), `Value::sort_cmp` (:1526), `sort_cmp_map` (:1563) |

* `SC.scF64_key`, `SC.scIntFloat_key` — the float and Int/Float arms order by a numeric key
  (NaN on top, `-0.0 = 0.0`, no rounding for any `i64`).
* **`SC.sc_laws`, `SC.pre2917_sc_total`** (then `scMain_total`) — #2912's `sort_cmp` is a total preorder on every
  well-formed value, nested lists/maps/vectors/points included (generic in `sort_cmp_f32`).
* **Refuted at b033aecc6** (fixed by #2917, see Main.lean) — `sort_cmp`-equal ⇒ hash-equal:
  `SC.pre2917_hash_not_consistent`, via `pre2917_float_nan_hash_differs` (`NaN` vs `-NaN`),
  `pre2917_vec_nan_hash_differs`, `pre2917_vec_zero_hash_differs`,
  `pre2917_point_zero_hash_differs`, `pre2917_point_nan_hash_differs`.
  The Float NaN half was observable then through hash-only DISTINCT: `UNWIND [0.0/0.0,
  -(0.0/0.0)] AS x RETURN count(DISTINCT x)` was 2, 1 in C (FINDINGS W9-hash-1); main: 1 (`Main.main_nans_one_key`).
* `SC.vec_zero_eq_but_hash_differs` — `Value::eq` merges `vecf32([-0.0])`/`vecf32([0.0])` and
  the same points while hashing them apart; before #2917 that was `GroupKey`'s equality (a
  `HashMap` contract violation). Main's `GroupKey::eq` is `sort_cmp`, which keeps them apart.
-/
