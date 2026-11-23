import FalkorIndexLayer.Model

/-! # The `IN` arm of the index scan (`runtime/ops/node_by_index_scan.rs`), re-checked at #2975

#2383 (and the scan-op refactors after it) keep an `IN` as one `IndexQuery::InList` instead of
expanding it to `Or(Equal, ..)`; since #2975 it is built by `IndexQuery::in_list`, which keeps
every non-null member. `Model.expandIn` builds that `InList`; this file proves that
`can_be_served` treats it as the `Or` of its `Equal`s:
-/
namespace IndexLayer

/-- `not_null` in `IndexQuery::in_list` (`mod.rs`, #2975; before, `keep` in
`node_by_index_scan.rs` also dropped every non-scalar member). -/
def keepItem : Val → Bool := notNull

/-- `Model.expandIn` is the member filter of `IndexQuery::in_list`, as one `InList`. -/
theorem expandIn_eq_filter (k : Attr) (xs : List Val) :
    expandIn k (.list xs) = some (.inList k (.list (xs.filter keepItem))) := rfl

/-- The `InList` arm of `can_be_served`: `!items.is_empty() && items.iter().all(is_indexable)`. -/
def canUtilizeInList (items : List Val) : Bool := !items.isEmpty && items.all indexable

theorem canUtilizeInList_eq (k : Attr) (items : List Val) :
    canUtilizeInList items = canUtilize (.inList k (.list items)) := by
  simp [canUtilizeInList, canUtilize]

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

/-- Hence for the members `in_list` keeps, the scan utilizes the index exactly when
`Model.canUtilize` says so of the `InList` `Model.expandIn` builds, which is also the `Or` of
its `Equal`s. -/
theorem scan_inlist_faithful (k : Attr) (xs : List Val) :
    canUtilizeInList (xs.filter keepItem) =
      (match expandIn k (.list xs) with | some q => canUtilize q | none => false) := by
  rw [expandIn_eq_filter]
  exact canUtilizeInList_eq k _

end IndexLayer
