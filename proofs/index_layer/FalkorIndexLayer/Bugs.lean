import FalkorIndexLayer.Model

/-! # Counterexamples: where index scan ≠ full scan + filter

Every theorem here is a concrete instance, closed by `decide`, of the model
disagreeing with the Cypher predicate, or (`fixed…`) of a fixed bug's query now agreeing. Each was reproduced on the live Rust
module (and compared with C) by `proofs/index_layer/repro.py`; the case name
is given in brackets. -/

namespace IndexLayer

def allF : Attr → Bool := fun _ => true

-- Unfold the (well-founded, hence kernel-opaque) model and evaluate.
set_option hygiene false in
macro "bugsimp" : tactic => `(tactic| (simp (config := { decide := true }) [indexHit, buildQ, buildEq,
  buildNumRange, buildStrRange, strBound, isStrVal, valueToNumeric, rsMatch, docOf, setRange,
  rsStore, holds, propOr, cyEqScalar, cyLt, cyLe, numOf, isTrue, inNum, inLex, lexLt, tagEnc,
  escaped, encByte, tagEncB, tagKey, boolKey, STRING_RANGE_END, notNull, isBool, ENum.le, ENum.lt, ENum.eq, canUtilize, indexable, intLosesPrecision,
  scanEmits, expandIn, one, allF, fieldsA, fieldsB, both, both', MASK]))
def one (v : Val) : Props := fun k => if k = 0 then some v else none
/-- Label A indexes attribute 0, label B indexes attribute 1. -/
def fieldsA : Attr → Bool := fun k => k == 0
def fieldsB : Attr → Bool := fun k => k == 1
def both : Props := fun k => if k = 0 then some (.int 1) else if k = 1 then some (.int 2) else none
def both' : Props := fun k => if k = 0 then some (.int 1) else if k = 1 then some (.int 3) else none

/-- [bool_int_conflated] FIXED by #2975 (bug 8). `Document::set` stored `true` as numeric 1.0 and
`value_to_numeric` mapped `1` to 1.0, so the index said `true = 1`. A Bool is now a TAG key of
its own (`tag_key`): neither `n.v = 1` finds `true` nor `n.v = true` finds `1`. The general
statements are `boolEqual_exact` and `equal_exact_storedBool`. -/
theorem fixed2975_bool_eq_int :
    indexHit allF (one (.bool true)) (.equal 0 (.int 1)) = false ∧
    holds (one (.bool true)) (.equal 0 (.int 1)) = false ∧
    indexHit allF (one (.int 1)) (.equal 0 (.bool true)) = false ∧
    holds (one (.int 1)) (.equal 0 (.bool true)) = false ∧
    indexHit allF (one (.bool true)) (.equal 0 (.bool true)) = true := by bugsimp

/-- Since #3076 the same query is exact: the date is not in the index, and not in the answer. -/
theorem fixed3076_temporal_in_numeric_range :
    indexHit allF (one (.temporal 0 5)) (.range 0 (some (.int 0)) none false false) = false ∧
    holds (one (.temporal 0 5)) (.range 0 (some (.int 0)) none false false) = false := by bugsimp

/-- [folded_temporal_constant_drops_filter] FIXED by #2975 (bug 1). A temporal *constant*
(constant-folded `date(...)`) is not `can_be_served`, so the scan falls back to a label scan;
`is_non_indexable_subexpr` used to flag only `Constant(Int)`, so no post-filter was kept and
every node of the label was returned. It is now an allow-list (`is_exact_index_literal`), the
filter is kept (`optimizer_scan`, `Cex` C3), and the fallback answers exactly. -/
theorem fixed2975_folded_temporal_constant :
    canUtilize (.equal 0 (.temporal 0 5)) = false ∧
    scanEmits allF true (one (.int 7)) (.equal 0 (.temporal 0 5)) = false ∧
    holds (one (.int 7)) (.equal 0 (.temporal 0 5)) = false := by bugsimp

/-- [point_equality_empty] FIXED by #2975 (bug 2). `is_indexable(Point)` was true but
`build_query_node` has no `Equal`-`Point` arm (null node), so `n.p = point(...)` returned nothing.
`can_be_served` now refuses a Point equality: the scan falls back under the kept filter. -/
theorem fixed2975_point_equality :
    canUtilize (.equal 0 (.point 1 2)) = false ∧
    scanEmits allF true (one (.point 1 2)) (.equal 0 (.point 1 2)) = true ∧
    holds (one (.point 1 2)) (.equal 0 (.point 1 2)) = true := by bugsimp

/-- [in_list_drops_temporal_item] FIXED by #2975 (bug 3). The `InList` expansion used to drop
items that are not Int/Float/String/Bool, so the node whose value was the dropped item was never
produced. `IndexQuery::in_list` now keeps every non-null item, `can_be_served` refuses the list,
and the scan falls back under the kept filter. -/
theorem fixed2975_in_list_keeps_item :
    expandIn 0 (.list [.temporal 0 5, .int 2]) = some (.inList 0 (.list [.temporal 0 5, .int 2])) ∧
    canUtilize (.inList 0 (.list [.temporal 0 5, .int 2])) = false ∧
    scanEmits allF true (one (.temporal 0 5)) (.inList 0 (.list [.temporal 0 5, .int 2])) = true ∧
    holds (one (.temporal 0 5)) (.inList 0 (.list [.temporal 0 5, .int 2])) = true := by bugsimp

/-- [empty_string_not_indexed] FIXED by #2975 (bug 10). RediSearch does not index an empty TAG
value (`rsStore`), so `n.v = ''` used to miss it. The empty string's key is now `0x21`
(`tagEnc_ne_nil`), found by `= ''` and by `< 'a'`. -/
theorem fixed2975_empty_string :
    indexHit allF (one (.str [])) (.equal 0 (.str [])) = true ∧
    holds (one (.str [])) (.equal 0 (.str [])) = true ∧
    indexHit allF (one (.str [])) (.range 0 none (some (.str [97])) false false) = true ∧
    holds (one (.str [])) (.range 0 none (some (.str [97])) false false) = true := by bugsimp

/-- FIXED by #2975 (bug 7): `"a b" < "aA"` and their keys too (the general statement is
`tagEnc_mono`). -/
theorem fixed2975_tagEnc_order :
    lexLt [97, 32, 98] [97, 65] = true ∧ lexLt (tagEnc [97, 32, 98]) (tagEnc [97, 65]) = true := by
  bugsimp

/-- [string_range_encoding_order] FIXED by #2975: `'a b' < 'aA'` is found by the range. -/
theorem fixed2975_string_range_order :
    indexHit allF (one (.str [97, 32, 98])) (.range 0 none (some (.str [97, 65])) false false) = true ∧
    holds (one (.str [97, 32, 98])) (.range 0 none (some (.str [97, 65])) false false) = true := by
  bugsimp

/-- [string_range_encoding_order_gt] FIXED by #2975: `'a b' > 'a!'` is false, and the index
agrees. -/
theorem fixed2975_string_range_order_gt :
    indexHit allF (one (.str [97, 32, 98])) (.range 0 (some (.str [97, 33])) none false false) = false ∧
    holds (one (.str [97, 32, 98])) (.range 0 (some (.str [97, 33])) none false false) = false := by
  bugsimp

/-- #2975: an open-ended string range stops at `STRING_RANGE_END`, below the Bool keys, so a
stored Bool is never returned by `n.v > ''`. -/
theorem fixed2975_string_range_skips_bool :
    indexHit allF (one (.bool true)) (.range 0 (some (.str [])) none false false) = false ∧
    holds (one (.bool true)) (.range 0 (some (.str [])) none false false) = false := by
  bugsimp

/-- Since #3072 the same query selects nothing, as Cypher does. -/
theorem fixed3072_exclusive_equal_bounds :
    indexHit allF (one (.str [97])) (.range 0 (some (.str [97])) (some (.str [97])) false false) = false ∧
    holds (one (.str [97])) (.range 0 (some (.str [97])) (some (.str [97])) false false) = false := by
  bugsimp

/-- Since #3081 the same queries select the stored infinities, as Cypher
does: `n.v > 0` finds `+inf`, `n.v < 0` finds `-inf`. (The general statement
is `numRange_exact`, which now covers stored `±inf`.) -/
theorem fixed3081_open_bound_includes_inf :
    indexHit allF (one (.flt .posInf)) (.range 0 (some (.int 0)) none false false) = true ∧
    holds (one (.flt .posInf)) (.range 0 (some (.int 0)) none false false) = true ∧
    indexHit allF (one (.flt .negInf)) (.range 0 none (some (.int 0)) false false) = true ∧
    holds (one (.flt .negInf)) (.range 0 none (some (.int 0)) false false) = true := by bugsimp

/-- [multilabel_and] `try_filter_pushdown` merges conjuncts found on
*different* labels into one query against the *first* label's index
(`utilize_index.rs`); `build_query_node` returns null for the unknown
field and the whole AND becomes empty (`mod.rs`). (C: correct.) -/
theorem bug_multilabel_and :
    indexHit fieldsA both (.and [.equal 0 (.int 1), .equal 1 (.int 2)]) = false ∧
    holds both (.and [.equal 0 (.int 1), .equal 1 (.int 2)]) = true := by bugsimp

/-- [multilabel_or] same for OR (`utilize_index.rs`); the unknown-field
branch is dropped from the union (`mod.rs`) and the filter is gone. -/
theorem bug_multilabel_or :
    indexHit fieldsB both' (.or [.equal 1 (.int 2), .equal 0 (.int 1)]) = false ∧
    holds both' (.or [.equal 1 (.int 2), .equal 0 (.int 1)]) = true := by bugsimp

end IndexLayer
