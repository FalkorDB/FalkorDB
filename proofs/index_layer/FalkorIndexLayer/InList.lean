import FalkorIndexLayer.Model

/-! # The `IN` arm of the index scan (`runtime/ops/node_by_index_scan.rs`), re-checked at 256ab4d3f

#2383 (and the scan-op refactors after it) keep an `IN` as one `IndexQuery::InList` instead of
expanding it to `Or(Equal, ..)`. `Model.expandIn`/`canUtilize` still model the expansion; this file
proves the two shapes the same, so the model needs no change:

| Lean | Rust |
| --- | --- |
| `keepItem` | `keep` in `evaluate_index_query`, `node_by_index_scan.rs:226-231` (only `Int/Float/String/Bool` members reach the index) |
| `canUtilizeInList` | the `IndexQuery::InList` arm of `can_utilize_index`, `:279-282` (non-empty, every member indexable) |
| `expandIn` (`Model.lean`) | the `InList { key, list }` that `evaluate_index_query` builds `:216-246`; `Index::build_query_node`'s `InList` arm `index/mod.rs:1607-1624` is a union of `Equal` children, empty list → the empty node |
-/
namespace IndexLayer

/-- `keep` (`node_by_index_scan.rs:226-231`). -/
def keepItem : Val → Bool
  | .int _ | .flt _ | .str _ | .bool _ => true
  | _ => false

/-- `Model.expandIn` is the member filter of `evaluate_index_query` followed by one `Equal` per
member (what `build_query_node`'s `InList` arm unions). -/
theorem expandIn_eq_filter (k : Attr) (xs : List Val) :
    expandIn k (.list xs) = some (.or ((xs.filter keepItem).map (IQ.equal k))) := by
  simp only [expandIn]
  congr 3

/-- The `InList` arm of `can_utilize_index` (`:279-282`): `!items.is_empty() && items.iter().all(is_indexable)`. -/
def canUtilizeInList (items : List Val) : Bool := !items.isEmpty && items.all indexable

/-- `can_utilize_index` on `Or` (`:269-271`): non-empty and every child utilizable. -/
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

end IndexLayer
