import FalkorIndexLayer.Model

/-! # The `IN` arm of the index scan (`IndexQuery::in_list`, `index/mod.rs`), re-checked at #3064

#2383 (and the scan-op refactors after it) keep an `IN` as one `IndexQuery::InList` instead of
expanding it to `Or(Equal, ..)`. `Model.expandIn` models the expansion; this file
proves the two shapes the same. Since #3064 `in_list` keeps every non-null member (a `date()`
too), so `can_be_served` sees it and the scan falls back instead of losing its rows:
-/
namespace IndexLayer

/-- `not_null` in `IndexQuery::in_list` (`mod.rs`). -/
def keepItem : Val → Bool
  | .null => false
  | _ => true

/-- `IndexQuery::in_list` (`mod.rs`). -/
def inListQ (k : Attr) (xs : List Val) : IQ := .inList k (.list (xs.filter keepItem))

/-- `Model.expandIn` is the member filter of `evaluate_index_query` followed by one `Equal` per
member (what `build_query_node`'s `InList` arm unions). -/
theorem expandIn_eq_filter (k : Attr) (xs : List Val) :
    expandIn k (.list xs) = some (.or ((xs.filter keepItem).map (IQ.equal k))) := by
  simp only [expandIn]
  congr 3

/-- The `InList` arm of `can_be_served`: `!items.is_empty() && items.iter().all(is_indexable)`. -/
def canUtilizeInList (items : List Val) : Bool := !items.isEmpty && items.all indexable

theorem canUtilize_inList (k : Attr) (items : List Val) :
    canUtilize (.inList k (.list items)) = canUtilizeInList items := by
  simp [canUtilize, canUtilizeInList]

/-- `can_be_served` on `Or`: non-empty and every child utilizable. -/
theorem canUtilize_or (qs : List IQ) : canUtilize (.or qs) = (!qs.isEmpty && qs.all canUtilize) := by
  rw [canUtilize]
  congr 1
  apply Bool.eq_iff_iff.mpr
  simp only [List.all_eq_true, List.mem_attach, true_implies, Subtype.forall]

/-- **The two shapes agree**: an `IN` list is utilized exactly when the `Or` of `Equal`s it
used to expand to is (an empty list, after the members were filtered, falls back to the label scan
under both; so does any member the index cannot hold exactly). -/
theorem canUtilizeInList_eq_or (k : Attr) (items : List Val) :
    canUtilizeInList items = canUtilize (.or (items.map (IQ.equal k))) := by
  rw [canUtilize_or]
  simp only [canUtilizeInList, List.isEmpty_map]
  congr 1
  apply Bool.eq_iff_iff.mpr
  simp only [List.all_eq_true, List.mem_map]
  constructor
  · intro h q ⟨v, hv, hq⟩; subst hq; simpa [canUtilize] using h v hv
  · intro h v hv; simpa [canUtilize] using h _ ⟨v, hv, rfl⟩

/-- Hence for the members `keep` lets through, the scan utilizes the index exactly when
`Model.canUtilize` says so of the expansion `Model.expandIn` builds. -/
theorem scan_inlist_faithful (k : Attr) (xs : List Val) :
    canUtilizeInList (xs.filter keepItem) =
      (match expandIn k (.list xs) with | some q => canUtilize q | none => false) := by
  rw [expandIn_eq_filter]
  exact canUtilizeInList_eq_or k _

/-- **PROVEN** (#3064, W2-index-3 fixed): a non-null member the index cannot hold exactly (a
`date()`, a list, a point) is kept by `in_list`, so the scan falls back to the label scan (and the
kept post-filter decides) instead of asking the index about a shorter list. -/
theorem inListQ_unservable_falls_back (k : Attr) (xs : List Val) (v : Val) (hv : v ∈ xs)
    (hn : keepItem v = true) (hi : indexable v = false) :
    canUtilize (inListQ k xs) = false := by
  simp only [inListQ, canUtilize_inList, canUtilizeInList]
  have : v ∈ xs.filter keepItem := List.mem_filter.mpr ⟨hv, hn⟩
  have hne : (xs.filter keepItem).isEmpty = false := by
    cases h : xs.filter keepItem with
    | nil => rw [h] at this; simp at this
    | cons _ _ => rfl
  rw [hne]
  simp only [Bool.not_false, Bool.true_and, List.all_eq_false]
  exact ⟨v, this, by simp [hi]⟩

/-- The scan's `InList` is served exactly when the `Or` of `Equal`s over the same non-null
members is. -/
theorem inListQ_eq_expandIn (k : Attr) (xs : List Val) :
    canUtilize (inListQ k xs) =
      (match expandIn k (.list xs) with | some q => canUtilize q | none => false) := by
  rw [inListQ, canUtilize_inList]
  exact scan_inlist_faithful k xs

end IndexLayer
