import FalkorEffectsEmitApply.Emit
/-!
# The remaining digests of `emit.rs`, column for column

`digest_created_edges` (`:576`), `digest_deleted_edges` (`:637`),
`digest_updates` (`:688`), `digest_labels` (`:855`), `digest_cancelled`
(`:397`). The `FxHashMap` grouping is first-appearance `groupBy` here — the
hash map's iteration order is unspecified in Rust, so every theorem below is
order-insensitive (a permutation or a per-record property).
-/
namespace FalkorEA

/-- `rows.sort_unstable()` on `(id, …)` tuples: ids are distinct, so ordering by
id is the whole order. -/
def sortByFst {β} (l : List (Nat × β)) : List (Nat × β) := l.mergeSort (fun a b => decide (a.1 ≤ b.1))

theorem perm_flatMap_congr {α β} {l : List α} {f g : α → List β} (h : ∀ a ∈ l, (f a).Perm (g a)) :
    (l.flatMap f).Perm (l.flatMap g) := by
  induction l with
  | nil => simp
  | cons x xs ih =>
    simp only [List.flatMap_cons]
    exact (h x (by simp)).append (ih fun a ha => h a (by simp [ha]))

theorem sortByFst_perm {β} (l : List (Nat × β)) : (sortByFst l).Perm l := List.mergeSort_perm _ _

theorem sortByFst_sorted {β} (l : List (Nat × β)) : (sortByFst l).Pairwise (fun a b => a.1 ≤ b.1) := by
  have := List.pairwise_mergeSort (le := fun (a b : Nat × β) => decide (a.1 ≤ b.1))
    (fun a b c h1 h2 => by simp at *; omega) (fun a b => by simp; omega) l
  unfold sortByFst; simpa using this

/-! ### `digest_created_edges` -/

/-- One `CreateEdge`: `(ids, rel, src, dst, attr_ids, rows)`. -/
structure EdgeRec where
  ids : List Nat
  rel : Nat
  src : List Nat
  dst : List Nat
  attrIds : List Nat
  rows : List Val

def triples (r : EdgeRec) : List (Nat × Nat × Nat) := r.ids.zip (r.src.zip r.dst)

def mkEdgeRec (rel : Nat) (attrIds : List Nat) (look : Nat → Option (List (Nat × Val)))
    (rows : List (Nat × Nat × Nat)) : EdgeRec :=
  let rows := sortByFst rows
  let ids := rows.map (·.1)
  ⟨ids, rel, rows.map (·.2.1), rows.map (·.2.2), attrIds, gatherRows attrIds look ids⟩

def attrIdsOf (look : Nat → Option (List (Nat × Val))) (id : Nat) : List Nat :=
  ((look id).getD []).map (·.1)

/-- `digest_created_edges`: per registered type, group by attribute-id vector. -/
def digestCreatedEdges (types : List Name) (byType : List (Name × List (Nat × Nat × Nat)))
    (look : Nat → Option (List (Nat × Val))) : List EdgeRec :=
  byType.flatMap fun (t, entries) =>
    match idxOf types t with
    | none => []
    | some rid => (groupBy (fun e => attrIdsOf look e.1) entries).map fun (aids, rows) => mkEdgeRec rid aids look rows

theorem zip_map3 (l : List (Nat × Nat × Nat)) : (l.map (·.1)).zip ((l.map (·.2.1)).zip (l.map (·.2.2))) = l := by
  induction l with
  | nil => rfl
  | cons x xs ih => simp [ih]

theorem mkEdgeRec_triples (rel : Nat) (aids : List Nat) (look) (rows : List (Nat × Nat × Nat)) :
    (triples (mkEdgeRec rel aids look rows)).Perm rows := by
  simp only [triples, mkEdgeRec, zip_map3]; exact sortByFst_perm rows

/-- Each emitted record's columns are aligned and its rows block has
`ids × attrs` cells (what `check_attr_shape` requires on the replica). -/
theorem mkEdgeRec_shape (rel : Nat) (aids : List Nat) (look) (rows : List (Nat × Nat × Nat)) :
    let r := mkEdgeRec rel aids look rows
    r.src.length = r.ids.length ∧ r.dst.length = r.ids.length ∧
    r.rows.length = r.ids.length * r.attrIds.length ∧ r.ids.Pairwise (· ≤ ·) := by
  refine ⟨by simp [mkEdgeRec], by simp [mkEdgeRec], by simp [mkEdgeRec, gatherRows_length], ?_⟩
  simp only [mkEdgeRec]
  exact List.pairwise_map.mpr (sortByFst_sorted rows)

/-- **Nothing is lost or invented**: the `(id, src, dst)` triples of all
emitted `CreateEdge`s are exactly the created edges of registered types. -/
theorem digestCreatedEdges_perm (types : List Name) (byType : List (Name × List (Nat × Nat × Nat))) (look) :
    ((digestCreatedEdges types byType look).flatMap triples).Perm
      (byType.flatMap fun (t, es) => if (idxOf types t).isSome then es else []) := by
  induction byType with
  | nil => simp [digestCreatedEdges]
  | cons q qs ih =>
    obtain ⟨t, es⟩ := q
    simp only [digestCreatedEdges, List.flatMap_cons, List.flatMap_append] at ih ⊢
    apply List.Perm.append _ ih
    cases h : idxOf types t with
    | none => simp
    | some rid =>
      simp only [Option.isSome_some, ite_true, List.flatMap_map]
      have hg := groupBy_flat (fun e => attrIdsOf look e.1) es
      refine List.Perm.trans ?_ hg
      exact perm_flatMap_congr fun q _ => mkEdgeRec_triples _ _ _ _

/-- Every edge in a record really has that record's attribute-id vector. -/
theorem digestCreatedEdges_keyed (types byType look) :
    ∀ r ∈ digestCreatedEdges types byType look, ∀ i ∈ r.ids, attrIdsOf look i = r.attrIds := by
  intro r hr i hi
  simp only [digestCreatedEdges, List.mem_flatMap] at hr
  obtain ⟨⟨t, es⟩, _, hr⟩ := hr
  split at hr
  · cases hr
  · simp only [List.mem_map] at hr
    obtain ⟨⟨aids, rows⟩, hq, rfl⟩ := hr
    simp only [mkEdgeRec, List.mem_map] at hi
    obtain ⟨e, he, rfl⟩ := hi
    have := groupBy_keyed (fun e => attrIdsOf look e.1) es _ hq e ((sortByFst_perm rows).mem_iff.mp he)
    exact this

/-! ### `digest_deleted_edges` -/

def digestDeletedEdges (deleted : List (Nat × Nat × Nat × Nat)) : List EdgeRec :=
  (groupBy (fun d => d.2.1) deleted).map fun (tid, rows) =>
    let rows := sortByFst (rows.map fun d => (d.1, d.2.2.1, d.2.2.2))
    ⟨rows.map (·.1), tid, rows.map (·.2.1), rows.map (·.2.2), [], []⟩

/-- The deleted edges' `(id, src, dst)` come back exactly, each under its own type. -/
theorem digestDeletedEdges_perm (deleted : List (Nat × Nat × Nat × Nat)) :
    ((digestDeletedEdges deleted).flatMap fun r => (triples r).map fun x => (x.1, r.rel, x.2.1, x.2.2)).Perm deleted := by
  unfold digestDeletedEdges
  refine List.Perm.trans ?_ (groupBy_flat (fun d => d.2.1) deleted)
  simp only [List.flatMap_map]
  refine perm_flatMap_congr ?_
  intro ⟨tid, rows⟩ hq
  simp only [triples, zip_map3]
  have hk := groupBy_keyed (fun d : Nat × Nat × Nat × Nat => d.2.1) deleted _ hq
  have hm : (rows.map fun d => (d.1, d.2.2.1, d.2.2.2)).map (fun x => (x.1, tid, x.2.1, x.2.2)) = rows := by
    rw [List.map_map]
    conv => rhs; rw [← List.map_id rows]
    apply List.map_congr_left
    intro d hd; have := hk d hd
    obtain ⟨a, b, c, e⟩ := d; simp at this ⊢; exact this.symm
  exact ((sortByFst_perm _).map _).trans (List.Perm.of_eq hm)

/-! ### `digest_updates` -/

/-- One update record: `(ids, labels, rel?, attr_ids, rows)`. -/
structure UpdRec where
  ids : List Nat
  labels : List Nat
  rel : Option Nat
  attrIds : List Nat
  rows : List Val

/-- The grouping key of `digest_updates` (`:713-735`). -/
def updKey (node : Bool) (labelsOf : Nat → List Nat) (typeOf : Nat → Nat) (q : Nat × List (Nat × Val)) :
    List Nat × Option Nat × List Nat :=
  (if node then sortDedup (labelsOf q.1) else [], if node then none else some (typeOf q.1), q.2.map (·.1))

/-- `digest_updates`: skip deleted ids; key = (sorted labels | type, attr ids). -/
def digestUpdates (attrs : List (Nat × List (Nat × Val))) (deleted : Nat → Bool) (node : Bool)
    (labelsOf : Nat → List Nat) (typeOf : Nat → Nat) : List UpdRec :=
  let look : Nat → Option (List (Nat × Val)) := fun i => (attrs.find? (·.1 == i)).map (·.2)
  let live := attrs.filter fun q => !deleted q.1
  (groupBy (updKey node labelsOf typeOf) live).map fun (k, qs) =>
    let ids := (qs.map (·.1)).mergeSort (fun a b => decide (a ≤ b))
    ⟨ids, k.1, k.2.1, k.2.2, gatherRows k.2.2 look ids⟩

/-- **Every surviving updated entity is emitted exactly once; a deleted one never.** -/
theorem digestUpdates_perm (attrs : List (Nat × List (Nat × Val))) (deleted node labelsOf typeOf) :
    ((digestUpdates attrs deleted node labelsOf typeOf).flatMap (·.ids)).Perm
      ((attrs.filter fun q => !deleted q.1).map (·.1)) := by
  unfold digestUpdates
  simp only [List.flatMap_map]
  refine List.Perm.trans ?_ (List.Perm.map (fun q : Nat × List (Nat × Val) => q.1)
    (groupBy_flat (updKey node labelsOf typeOf) (attrs.filter fun q => !deleted q.1)))
  rw [List.map_flatMap]
  exact perm_flatMap_congr fun q _ => List.mergeSort_perm _ _

theorem addTo_ne {K} [DecidableEq K] {A} (k : K) (x : A) :
    ∀ acc : List (K × List A), (∀ q ∈ acc, q.2 ≠ []) → ∀ q ∈ addTo k x acc, q.2 ≠ [] := by
  intro acc
  induction acc with
  | nil => intro _ q hq; simp only [addTo, List.mem_singleton] at hq; subst hq; simp
  | cons hd rest ih =>
    obtain ⟨k', g⟩ := hd
    intro h q hq
    simp only [addTo] at hq
    split at hq
    · rcases List.mem_cons.mp hq with he | hr
      · subst he; simp
      · exact h q (List.mem_cons_of_mem _ hr)
    · rcases List.mem_cons.mp hq with he | hr
      · subst he; exact h _ List.mem_cons_self
      · exact ih (fun z hz => h z (List.mem_cons_of_mem _ hz)) q hr

theorem groupBy_ne {K} [DecidableEq K] {A} (key : A → K) (l : List A) : ∀ q ∈ groupBy key l, q.2 ≠ [] := by
  suffices ∀ (l : List A) acc, (∀ q ∈ acc, q.2 ≠ []) → ∀ q ∈ groupAux key l acc, q.2 ≠ [] from
    this l [] (by simp)
  intro l; induction l with
  | nil => intro acc h; exact h
  | cons x rest ih => intro acc h; exact ih _ (addTo_ne (key x) x acc h)

/-- Per record: the rows block is `ids × attrs`; a node record names no type,
an edge record names no labels; every id in it carries the record's key. -/
theorem digestUpdates_shape (attrs deleted node labelsOf typeOf) :
    ∀ r ∈ digestUpdates attrs deleted node labelsOf typeOf,
      r.rows.length = r.ids.length * r.attrIds.length ∧ (node = true → r.rel = none) ∧
      (node = false → r.labels = []) := by
  intro r hr
  simp only [digestUpdates, List.mem_map] at hr
  obtain ⟨⟨k, qs⟩, hq, rfl⟩ := hr
  refine ⟨gatherRows_length _ _ _, fun h => ?_, fun h => ?_⟩ <;>
  · have hk := groupBy_keyed _ _ _ hq
    cases qs with
    | nil => exact absurd rfl (groupBy_ne _ _ _ hq)
    | cons x xs =>
      have hx := hk x (by simp)
      simp only [updKey] at hx ⊢; rw [← hx]; simp [h]

/-! ### `digest_labels` -/

/-- `digest_labels`: skip `skip`ped and empty; group by the sorted-dedup'd shape. -/
def digestLabels (labels : List (Nat × List Nat)) (skip : Nat → Bool) : List (List Nat × List Nat) :=
  let keep := labels.filter fun q => !skip q.1 && !q.2.isEmpty
  (groupBy (fun q => sortDedup q.2) keep).map fun (shape, qs) =>
    (shape, (qs.map (·.1)).mergeSort (fun a b => decide (a ≤ b)))

theorem digestLabels_perm (labels : List (Nat × List Nat)) (skip : Nat → Bool) :
    ((digestLabels labels skip).flatMap (·.2)).Perm
      ((labels.filter fun q => !skip q.1 && !q.2.isEmpty).map (·.1)) := by
  unfold digestLabels
  simp only [List.flatMap_map]
  refine List.Perm.trans ?_ (List.Perm.map (fun q : Nat × List Nat => q.1)
    (groupBy_flat (fun q : Nat × List Nat => sortDedup q.2) (labels.filter fun q => !skip q.1 && !q.2.isEmpty)))
  rw [List.map_flatMap]
  exact perm_flatMap_congr fun q _ => List.mergeSort_perm _ _

/-- No label record is empty (`debug_assert!` at `:880` holds). -/
theorem digestLabels_nonempty (labels skip) : ∀ r ∈ digestLabels labels skip, r.1 ≠ [] := by
  intro r hr
  simp only [digestLabels, List.mem_map] at hr
  obtain ⟨⟨k, qs⟩, hq, rfl⟩ := hr
  have hk := groupBy_keyed _ _ _ hq
  cases qs with
  | nil => exact absurd rfl (groupBy_ne _ _ _ hq)
  | cons x xs =>
    have hx := hk x (by simp)
    have hmem := (List.mem_filter.mp ((groupBy_flat _ _).mem_iff.mp
      (List.mem_flatMap.mpr ⟨_, hq, List.mem_cons_self⟩))).2
    simp only [Bool.and_eq_true, Bool.not_eq_true', List.isEmpty_eq_false_iff] at hmem
    simp only at hx ⊢; rw [← hx]
    intro h0
    obtain ⟨a, ha⟩ := List.exists_mem_of_ne_nil _ hmem.2
    have := (mem_sortDedup a x.2).mpr ha
    rw [h0] at this; simp at this

/-! ### `digest_cancelled` -/

theorem filterMap_flatMap_perm {α β γ} (F : α → Option β) (T : β → List γ) (G : α → List γ) :
    ∀ (l : List α), (∀ a ∈ l, (match F a with | none => [] | some b => T b).Perm (G a)) →
    ((l.filterMap F).flatMap T).Perm (l.flatMap G)
  | [], _ => by simp
  | x :: xs, h => by
    have ih := filterMap_flatMap_perm F T G xs (fun a ha => h a (by simp [ha]))
    have hx := h x (by simp)
    simp only [List.filterMap_cons, List.flatMap_cons]
    cases hF : F x with
    | none =>
      simp only [hF] at hx
      have h0 : G x = [] := List.perm_nil.mp hx.symm
      simp only [h0, List.nil_append]; exact ih
    | some b => simp only [hF] at hx; simp only [List.flatMap_cons]; exact hx.append ih

/-- `digest_cancelled`, columns included: cancelled nodes created bare, every
cancelled edge of a registered type created and deleted (grouped by type,
sorted by id, types ordered by relation id), the nodes deleted last. -/
def digestCancelledF (nodes : List Nat) (rels : List (Nat × Name × Nat × Nat)) (types : List Name) :
    List (Sum (List Nat) EdgeRec) × List EdgeRec × List EdgeRec × List (List Nat) :=
  if nodes.isEmpty && rels.isEmpty then ([], [], [], []) else
  let groups := (groupBy (fun r => r.2.1) rels).filterMap fun (t, rs) =>
    (idxOf types t).map fun rid =>
      let rows := sortByFst (rs.map fun r => (r.1, r.2.2.1, r.2.2.2))
      (⟨rows.map (·.1), rid, rows.map (·.2.1), rows.map (·.2.2), [], []⟩ : EdgeRec)
  let pairs := groups.mergeSort (fun a b => decide (a.rel ≤ b.rel))
  (if nodes.isEmpty then [] else [.inl nodes], pairs, pairs, if nodes.isEmpty then [] else [nodes])

/-- The create and delete halves name exactly the same edges, and those are the
cancelled edges of registered types; the node pair brackets them. -/
theorem digestCancelled_pairs (nodes rels types) :
    let d := digestCancelledF nodes rels types
    d.2.1 = d.2.2.1 ∧ (d.1 = [] ↔ (nodes = [] ∨ (nodes.isEmpty && rels.isEmpty))) ∧
    ((d.2.1.flatMap triples).Perm ((rels.filter fun r => (idxOf types r.2.1).isSome).map
        fun r => (r.1, r.2.2.1, r.2.2.2))) := by
  simp only [digestCancelledF]
  split
  · rename_i h
    simp only [Bool.and_eq_true, List.isEmpty_iff] at h
    obtain ⟨h1, h2⟩ := h; subst h1; subst h2; simp
  · rename_i h
    refine ⟨rfl, ?_, ?_⟩
    · by_cases hn : nodes = [] <;> simp [hn] at h ⊢
    · refine List.Perm.trans ((List.mergeSort_perm _ _).flatMap_right _) ?_
      refine List.Perm.trans ?_ ((((groupBy_flat (fun r : Nat × Name × Nat × Nat => r.2.1) rels).filter
        (fun r => (idxOf types r.2.1).isSome)).map (fun r => (r.1, r.2.2.1, r.2.2.2))))
      rw [List.filter_flatMap, List.map_flatMap]
      apply filterMap_flatMap_perm
      intro ⟨t, rs⟩ hq
      have hk := groupBy_keyed _ _ _ hq
      cases hid : idxOf types t with
      | none =>
        simp only [Option.map_none]
        rw [List.filter_eq_nil_iff.mpr (fun r hr => by have := hk r hr; simp at this; simp [this, hid])]
        exact List.Perm.refl _
      | some rid =>
        simp only [Option.map_some, triples, zip_map3]
        rw [List.filter_eq_self.mpr (fun r hr => by have := hk r hr; simp at this; simp [this, hid])]
        exact sortByFst_perm _

end FalkorEA
