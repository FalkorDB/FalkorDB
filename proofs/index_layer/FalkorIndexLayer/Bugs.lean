import FalkorIndexLayer.Model

/-! # Counterexamples: where index scan ≠ full scan + filter

Every theorem here is a concrete instance, closed by `decide`, of the model
disagreeing with the Cypher predicate. Each was reproduced on the live Rust
module (and compared with C) by `proofs/index_layer/repro.py`; the case name
is given in brackets. -/

namespace IndexLayer

def allF : Attr → Bool := fun _ => true

-- Unfold the (well-founded, hence kernel-opaque) model and evaluate.
set_option hygiene false in
macro "bugsimp" : tactic => `(tactic| (simp (config := { decide := true }) [indexHit, buildQ, buildEq,
  buildNumRange, buildStrRange, strBound, isStrVal, valueToNumeric, rsMatch, docOf, setRange,
  rsStore, holds, propOr, cyEqScalar, cyLt, cyLe, numOf, isTrue, inNum, inLex, lexLt, tagEnc,
  tagEncRaw, tagUnit, tagKey, boolKey, STRING_RANGE_END, ENum.le, ENum.lt, ENum.eq, canBeServed,
  indexable, intLosesPrecision, isNull, isBool, inListQ, scanEmits, one, allF, fieldsA, fieldsB,
  both, both', MASK]))
def one (v : Val) : Props := fun k => if k = 0 then some v else none
/-- Label A indexes attribute 0, label B indexes attribute 1. -/
def fieldsA : Attr → Bool := fun k => k == 0
def fieldsB : Attr → Bool := fun k => k == 1
def both : Props := fun k => if k = 0 then some (.int 1) else if k = 1 then some (.int 2) else none
def both' : Props := fun k => if k = 0 then some (.int 1) else if k = 1 then some (.int 3) else none

/-- [bool_int_conflated] fixed by #2975: a Bool is stored as the TAG key
`0xFF t`/`0xFF f` (`mod.rs:888`), not as numeric 1/0, so `n.v = 1` no longer
finds `true`. (`bug_bool_eq_int` was removed; the last revision that has it is
343e077f2.) -/
theorem fixed2975_bool_ne_int :
    indexHit allF (one (.bool true)) (.equal 0 (.int 1)) = false ∧
    holds (one (.bool true)) (.equal 0 (.int 1)) = false ∧
    indexHit allF (one (.bool true)) (.equal 0 (.bool true)) = true ∧
    holds (one (.bool true)) (.equal 0 (.bool true)) = true := by bugsimp

/-- #2975: a stored Bool is not in a numeric range (it is no longer `1.0`). -/
theorem fixed2975_bool_not_in_numeric_range :
    indexHit allF (one (.bool true)) (.range 0 (some (.int 0)) none false false) = false ∧
    holds (one (.bool true)) (.range 0 (some (.int 0)) none false false) = false := by bugsimp

/-- Since #3076 the same query is exact: the date is not in the index, and not in the answer. -/
theorem fixed3076_temporal_in_numeric_range :
    indexHit allF (one (.temporal 0 5)) (.range 0 (some (.int 0)) none false false) = false ∧
    holds (one (.temporal 0 5)) (.range 0 (some (.int 0)) none false false) = false := by bugsimp

/-- [folded_temporal_constant_drops_filter] a temporal *constant*
(constant-folded `date(...)`) is not `can_be_served` (`mod.rs:304`), so the
scan falls back to a label scan, which without the post-filter returns every
node of the label. Before #2975 `is_non_indexable_subexpr` dropped that filter
(`keep = false`); since #2975 the optimizer keeps it (see optimizer_scan
`fixed2975_*`), and the fallback is then exact: -/
theorem bug_folded_temporal_constant :
    canBeServed (.equal 0 (.temporal 0 5)) = false ∧
    scanEmits allF false (one (.int 7)) (.equal 0 (.temporal 0 5)) = true ∧
    holds (one (.int 7)) (.equal 0 (.temporal 0 5)) = false := by bugsimp

theorem fixed2975_folded_temporal_constant :
    scanEmits allF true (one (.int 7)) (.equal 0 (.temporal 0 5)) = false ∧
    scanEmits allF true (one (.temporal 0 5)) (.equal 0 (.temporal 0 5)) = true := by bugsimp

/-- [point_equality_empty] fixed by #2975: a Point is not `is_indexable`
(`mod.rs:309`), so `n.p = point(...)` falls back to the label scan plus the
kept filter and finds the node. (`bug_point_equality` was removed; the last
revision that has it is 343e077f2.) -/
theorem fixed2975_point_equality :
    canBeServed (.equal 0 (.point 1 2)) = false ∧
    scanEmits allF true (one (.point 1 2)) (.equal 0 (.point 1 2)) = true ∧
    holds (one (.point 1 2)) (.equal 0 (.point 1 2)) = true := by bugsimp

/-- [in_list_drops_temporal_item] fixed by #2975: `IndexQuery::in_list`
(`mod.rs:284`) keeps the temporal item, so the list is not servable and the
fallback scan with the kept filter returns the node. (`bug_in_list_drops_item`
was removed; the last revision that has it is 343e077f2.) -/
theorem fixed2975_in_list_keeps_item :
    inListQ 0 [.temporal 0 5, .int 2] = .inList 0 (.list [.temporal 0 5, .int 2]) ∧
    canBeServed (inListQ 0 [.temporal 0 5, .int 2]) = false ∧
    scanEmits allF true (one (.temporal 0 5)) (inListQ 0 [.temporal 0 5, .int 2]) = true ∧
    holds (one (.temporal 0 5)) (.inList 0 (.list [.temporal 0 5, .int 2])) = true := by bugsimp

/-- [empty_string_not_indexed] fixed by #2975: `''` is stored as the key
`0x21` (`mod.rs:670-672`), so `n.v = ''` finds it.
(`bug_empty_string_not_indexed` was removed; the last revision that has it is
343e077f2.) -/
theorem fixed2975_empty_string_indexed :
    indexHit allF (one (.str [])) (.equal 0 (.str [])) = true ∧
    holds (one (.str [])) (.equal 0 (.str [])) = true := by bugsimp

/-- Since #2975 `tag_encode_lower` is order-preserving (in general:
`tagEnc_lexLt`): `"a b" < "aA"` and their keys too. (`tagEnc_not_monotone`
was removed; the last revision that has it is 343e077f2.) -/
theorem fixed2975_tagEnc_monotone :
    lexLt [97, 32, 98] [97, 65] = true ∧ lexLt (tagEnc [97, 32, 98]) (tagEnc [97, 65]) = true := by
  bugsimp

/-- [string_range_encoding_order] fixed by #2975 (in general: `strRange_exact`). -/
theorem fixed2975_string_range_order :
    indexHit allF (one (.str [97, 32, 98])) (.range 0 none (some (.str [97, 65])) false false) = true ∧
    holds (one (.str [97, 32, 98])) (.range 0 none (some (.str [97, 65])) false false) = true := by
  bugsimp

/-- [string_range_encoding_order_gt] fixed by #2975: `'a b' > 'a!'` is false
on both sides. -/
theorem fixed2975_string_range_order_gt :
    indexHit allF (one (.str [97, 32, 98])) (.range 0 (some (.str [97, 33])) none false false) = false ∧
    holds (one (.str [97, 32, 98])) (.range 0 (some (.str [97, 33])) none false false) = false := by
  bugsimp

/-- #2975: an open upper end stops at `STRING_RANGE_END`, below the Bool keys:
`n.v > 'a'` does not return `true`. -/
theorem fixed2975_string_range_skips_bool :
    indexHit allF (one (.bool true)) (.range 0 (some (.str [97])) none false false) = false ∧
    holds (one (.bool true)) (.range 0 (some (.str [97])) none false false) = false := by
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
(`utilize_index.rs:503`); `build_query_node` returns null for the unknown
field and the whole AND becomes empty (`mod.rs:1572`). (C: correct.) -/
theorem bug_multilabel_and :
    indexHit fieldsA both (.and [.equal 0 (.int 1), .equal 1 (.int 2)]) = false ∧
    holds both (.and [.equal 0 (.int 1), .equal 1 (.int 2)]) = true := by bugsimp

/-- [multilabel_or] same for OR (`utilize_index.rs:521`); the unknown-field
branch is dropped from the union (`mod.rs:1589`) and the filter is gone. -/
theorem bug_multilabel_or :
    indexHit fieldsB both' (.or [.equal 1 (.int 2), .equal 0 (.int 1)]) = false ∧
    holds both' (.or [.equal 1 (.int 2), .equal 0 (.int 1)]) = true := by bugsimp

end IndexLayer
