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
(`mod.rs`) and `value_to_numeric` maps `1` to 1.0 (`mod.rs`): the
index says `true = 1`. The optimizer drops the filter for a literal, so the
wrong row is returned. (C: same bug.) -/
theorem bug_bool_eq_int :
    indexHit allF (one (.bool true)) (.equal 0 (.int 1)) = true ∧
    holds (one (.bool true)) (.equal 0 (.int 1)) = false := by bugsimp

/-- Since #3076 the same query is exact: the date is not in the index, and not in the answer. -/
theorem fixed3076_temporal_in_numeric_range :
    indexHit allF (one (.temporal 0 5)) (.range 0 (some (.int 0)) none false false) = false ∧
    holds (one (.temporal 0 5)) (.range 0 (some (.int 0)) none false false) = false := by bugsimp

/-- Since #3064 (W2-index-1 fixed) a folded temporal constant is still not `can_be_served`,
so the scan falls back to the label scan, but `is_non_indexable_subexpr` now flags every constant
that is not an exact index literal (`is_exact_index_literal`, `utilize_index.rs`), so the optimizer
keeps the post-filter (`keep = true`; proven in `optimizer_scan`, `Index.litFlag`): the node
whose `v` is not the date is dropped, as Cypher does. -/
theorem fixed3064_folded_temporal_constant :
    canUtilize (.equal 0 (.temporal 0 5)) = false ∧
    scanEmits allF true (one (.int 7)) (.equal 0 (.temporal 0 5)) = false ∧
    holds (one (.int 7)) (.equal 0 (.temporal 0 5)) = false := by bugsimp

/-- Since #3064 (W2-index-2 fixed) `is_indexable(Point)` is false, so `n.p = point(...)` falls
back to the label scan under the kept filter and finds the point. -/
theorem fixed3064_point_equality :
    canUtilize (.equal 0 (.point 1 2)) = false ∧
    scanEmits allF true (one (.point 1 2)) (.equal 0 (.point 1 2)) = true ∧
    holds (one (.point 1 2)) (.equal 0 (.point 1 2)) = true := by bugsimp

/-- Since #3064 (W2-index-3 fixed) `IndexQuery::in_list` keeps the date member, so the list is not
`can_be_served`, the scan falls back, and the date row is produced. -/
theorem fixed3064_in_list_keeps_item :
    expandIn 0 (.list [.temporal 0 5, .int 2]) = some (.or [.equal 0 (.temporal 0 5), .equal 0 (.int 2)]) ∧
    canUtilize (.inList 0 (.list [.temporal 0 5, .int 2])) = false ∧
    scanEmits allF true (one (.temporal 0 5)) (.inList 0 (.list [.temporal 0 5, .int 2])) = true ∧
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

/-- Why a query must not mix labels: one index answers a conjunction over a field it does not
hold with nothing (`build_query_node` returns null for the unknown field, `mod.rs`)... -/
theorem multilabel_and_unanswerable :
    indexHit fieldsA both (.and [.equal 0 (.int 1), .equal 1 (.int 2)]) = false ∧
    holds both (.and [.equal 0 (.int 1), .equal 1 (.int 2)]) = true := by bugsimp

/-- ...and a disjunction without the unknown branch. -/
theorem multilabel_or_unanswerable :
    indexHit fieldsB both' (.or [.equal 1 (.int 2), .equal 0 (.int 1)]) = false ∧
    holds both' (.or [.equal 1 (.int 2), .equal 0 (.int 1)]) = true := by bugsimp

/-- Since #3064 (W2-index-4 fixed) `try_filter_pushdown` pushes only the conjuncts of the first
label's index (`utilize_index.rs`; `optimizer_scan` `Index.pushAnd_spec`) and keeps the others as
the remaining filter, and refuses an `OR` across labels: the pushed query finds the node, and the
remaining `n.y = 2` keeps it. -/
theorem fixed3064_multilabel_and :
    indexHit fieldsA both (.equal 0 (.int 1)) = true ∧
    holds both (.equal 1 (.int 2)) = true ∧
    holds both (.and [.equal 0 (.int 1), .equal 1 (.int 2)]) = true := by bugsimp

end IndexLayer
