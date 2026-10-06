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
  escaped, hexDigit, HEX, ENum.le, ENum.lt, ENum.eq, canUtilize, indexable, intLosesPrecision,
  scanEmits, expandIn, one, allF, fieldsA, fieldsB, both, both', MASK]))
def one (v : Val) : Props := fun k => if k = 0 then some v else none
/-- Label A indexes attribute 0, label B indexes attribute 1. -/
def fieldsA : Attr → Bool := fun k => k == 0
def fieldsB : Attr → Bool := fun k => k == 1
def both : Props := fun k => if k = 0 then some (.int 1) else if k = 1 then some (.int 2) else none
def both' : Props := fun k => if k = 0 then some (.int 1) else if k = 1 then some (.int 3) else none

/-- [bool_int_conflated] `Document::set` stores `true` as numeric 1.0
(`mod.rs:760`) and `value_to_numeric` maps `1` to 1.0 (`mod.rs:1359`): the
index says `true = 1`. The optimizer drops the filter for a literal, so the
wrong row is returned. (C: same bug.) -/
theorem bug_bool_eq_int :
    indexHit allF (one (.bool true)) (.equal 0 (.int 1)) = true ∧
    holds (one (.bool true)) (.equal 0 (.int 1)) = false := by bugsimp

/-- [temporal_in_numeric_range] temporals are stored as their raw numeric
timestamp (`mod.rs:797`), so `n.v > 0` matches a date. (C: correct.) -/
theorem bug_temporal_in_numeric_range :
    indexHit allF (one (.temporal 0 5)) (.range 0 (some (.int 0)) none false false) = true ∧
    holds (one (.temporal 0 5)) (.range 0 (some (.int 0)) none false false) = false := by bugsimp

/-- [folded_temporal_constant_drops_filter] a temporal *constant*
(constant-folded `date(...)`) is not `can_utilize_index`
(`node_by_index_scan.rs:252`), so the scan falls back to a label scan; but
`is_non_indexable_subexpr` (`utilize_index.rs:631`) only flags
`Constant(Int)` among constants, so no post-filter is kept: every node of
the label is returned. -/
theorem bug_folded_temporal_constant :
    canUtilize (.equal 0 (.temporal 0 5)) = false ∧
    scanEmits allF false (one (.int 7)) (.equal 0 (.temporal 0 5)) = true ∧
    holds (one (.int 7)) (.equal 0 (.temporal 0 5)) = false := by bugsimp

/-- [point_equality_empty] `is_indexable(Point) = true` but
`build_query_node` has no `Equal`-`Point` arm (`mod.rs:1672` → null), so
`n.p = point(...)` returns nothing even with the filter kept. (C: correct.) -/
theorem bug_point_equality :
    canUtilize (.equal 0 (.point 1 2)) = true ∧
    scanEmits allF true (one (.point 1 2)) (.equal 0 (.point 1 2)) = false ∧
    holds (one (.point 1 2)) (.equal 0 (.point 1 2)) = true := by bugsimp

/-- [in_list_drops_temporal_item] the `InList` expansion silently drops list
items that are not Int/Float/String/Bool (`node_by_index_scan.rs:222`), so
the node whose value is the dropped item is never produced. -/
theorem bug_in_list_drops_item :
    expandIn 0 (.list [.temporal 0 5, .int 2]) = some (.or [.equal 0 (.int 2)]) ∧
    scanEmits allF true (one (.temporal 0 5)) (.or [.equal 0 (.int 2)]) = false ∧
    holds (one (.temporal 0 5)) (.inList 0 (.list [.temporal 0 5, .int 2])) = true := by bugsimp

/-- [empty_string_not_indexed] RediSearch does not index an empty TAG value
(`rsStore`), so `n.v = ''` misses it. (C: same.) -/
theorem bug_empty_string_not_indexed :
    indexHit allF (one (.str [])) (.equal 0 (.str [])) = false ∧
    holds (one (.str [])) (.equal 0 (.str [])) = true := by bugsimp

/-- `tag_encode_lower` is not order-preserving: `"a b" < "aA"` but
`"a_20b" > "aA"`. -/
theorem tagEnc_not_monotone :
    lexLt [97, 32, 98] [97, 65] = true ∧ lexLt (tagEnc [97, 32, 98]) (tagEnc [97, 65]) = false := by
  bugsimp

/-- [string_range_encoding_order] hence a lexicographic range over encoded
keys misses (or adds) strings containing escaped bytes. -/
theorem bug_string_range_order :
    indexHit allF (one (.str [97, 32, 98])) (.range 0 none (some (.str [97, 65])) false false) = false ∧
    holds (one (.str [97, 32, 98])) (.range 0 none (some (.str [97, 65])) false false) = true := by
  bugsimp

/-- [string_range_encoding_order_gt] …and returns wrong ones: `'a b' > 'a!'`
is false, the index says true. -/
theorem bug_string_range_order_gt :
    indexHit allF (one (.str [97, 32, 98])) (.range 0 (some (.str [97, 33])) none false false) = true ∧
    holds (one (.str [97, 32, 98])) (.range 0 (some (.str [97, 33])) none false false) = false := by
  bugsimp

/-- [string_exclusive_equal_bounds] `build_string_range_node` turns
`lo == hi` into an exact match ignoring the include flags (`mod.rs:1430`):
`n.v > 'a' AND n.v < 'a'` returns `'a'`. (C: correct.) -/
theorem bug_exclusive_equal_bounds :
    indexHit allF (one (.str [97])) (.range 0 (some (.str [97])) (some (.str [97])) false false) = true ∧
    holds (one (.str [97])) (.range 0 (some (.str [97])) (some (.str [97])) false false) = false := by
  bugsimp

/-- [open_bound_excludes_inf] a missing bound becomes `RSRANGE_INF` with
the (always `false`) include flag of the missing side (`mod.rs:1389-1407`),
so `+inf` fails `v < +inf`: `n.v > 0` misses a stored `+inf`. (C: same.) -/
theorem bug_open_bound_excludes_inf :
    indexHit allF (one (.flt .posInf)) (.range 0 (some (.int 0)) none false false) = false ∧
    holds (one (.flt .posInf)) (.range 0 (some (.int 0)) none false false) = true := by bugsimp

/-- [multilabel_and] `try_filter_pushdown` merges conjuncts found on
*different* labels into one query against the *first* label's index
(`utilize_index.rs:503`); `build_query_node` returns null for the unknown
field and the whole AND becomes empty (`mod.rs:1559`). (C: correct.) -/
theorem bug_multilabel_and :
    indexHit fieldsA both (.and [.equal 0 (.int 1), .equal 1 (.int 2)]) = false ∧
    holds both (.and [.equal 0 (.int 1), .equal 1 (.int 2)]) = true := by bugsimp

/-- [multilabel_or] same for OR (`utilize_index.rs:521`); the unknown-field
branch is dropped from the union (`mod.rs:1576`) and the filter is gone. -/
theorem bug_multilabel_or :
    indexHit fieldsB both' (.or [.equal 1 (.int 2), .equal 0 (.int 1)]) = false ∧
    holds both' (.or [.equal 1 (.int 2), .equal 0 (.int 1)]) = true := by bugsimp

end IndexLayer
