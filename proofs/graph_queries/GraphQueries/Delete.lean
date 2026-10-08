import GraphQueries.Queries
/-
# Edge creation and deletion; the stale type-matrix width (CONFIRMED BUG)

| here | there (graph.rs) |
| --- | --- |
| `resolveRels`  | `delete_relationships` phase 1 (:2777-2800) |
| `deleteRels`   | `delete_relationships` :2767 (phase 2 :2802-2888; `release(rels, resolved)` :2807) |
| `collectImplicit` | `delete_implicit_edges` collection loop (:2904-2948) |
| `deleteImplicit`  | `delete_implicit_edges` :2894 (`release(freed, freed)` :3032-3035) |
| `createRelsBulk`  | `create_relationships_bulk` :2518 (shape refusals :2538/:2420, `create` :2557) |

Every id-space move goes through `O` (IdContract.lean); theorems that need
its behaviour take `IdSpaceContract O`.

`remove_mask` builds its mask as `relationship_cap × relationship_types.len()`
(:2853, :2962). `Mat.removeMask` is a no-op on a dimension mismatch, which is
what the release build does (`GrB_DIMENSION_MISMATCH` is only
`debug_assert`ed, matrix.rs:927).
-/
namespace GQ
variable {V : Type}

/-- Phase 1: `(edge, type, src, dst)` for every id that has endpoints and a type. -/
def resolveRels (g : G V) (rels : List Nat) : List (Nat × Nat × Nat × Nat) :=
  rels.filterMap (fun e => match g.endpoints e, relTypeId g e with
    | some (s, d), some t => some (e, t, s, d)
    | _, _ => none)

def remainingEdge (ms : List Ten) (s d : Nat) : Bool := ms.any (fun tn => tn.any (fun x => x.1 == s && x.2.1 == d))

def deleteRels (O : IdSpaceOps) (g : G V) (rels : List Nat) :
    Except String (G V × List (Nat × Nat × Nat × Nat)) :=
  if rels = [] then .ok (g, []) else
  let res := resolveRels g rels
  let ids := res.map (·.1)
  match O.release (relIds g) rels ids with
  | none => .error "relationship"
  | some sp =>
    let g0 := setRelIds g sp
    let et := res.map (fun r => (r.1, r.2.1))
    let relMs' := g.relMs.mapIdx (fun t tn => tn.filter (fun x => (x.2.2, t) ∉ et))
    let cand := (res.map (fun r => (r.2.2.1, r.2.2.2))).filter (fun p => !remainingEdge relMs' p.1 p.2)
    .ok ({ g0 with endpoints := fun x => if x ∈ ids then none else g.endpoints x,
                   relMs := relMs',
                   relType := g.relType.removeMask g.relCap g.types.length et,
                   adj := g.adj.removeMask g.nodeCap g.nodeCap cand }, res)

/-- Phase 1 does not mutate, so a refusal (an id already free) changes nothing. -/
theorem deleteRels_refused (O : IdSpaceOps) (hC : IdSpaceContract O) (g : G V) (rels : List Nat)
    (e : Nat) (he : e ∈ rels) (hd : e ∈ g.delRels) :
    deleteRels O g rels = .error "relationship" := by
  have hne : rels ≠ [] := by intro h; subst h; cases he
  simp [deleteRels, hne, hC.release_refuses_recycled (relIds g) rels _ e he hd]

/-- With consistent dimensions, every deleted edge loses its type entry, its
endpoints, and its tensor entry, and is binned. -/
theorem deleteRels_clears (O : IdSpaceOps) (hC : IdSpaceContract O) (g : G V) (rels : List Nat)
    (r : G V) (out : List (Nat × Nat × Nat × Nat))
    (h : deleteRels O g rels = .ok (r, out))
    (hw : g.relType.nr = g.relCap ∧ g.relType.nc = g.types.length)
    (e t s d : Nat) (hm : (e, t, s, d) ∈ out) :
    r.relType.get e t = false ∧ r.endpoints e = none ∧ e ∈ r.delRels ∧
    ∀ tn ∈ r.relMs[t]?, (s, d, e) ∉ tn := by
  unfold deleteRels at h
  split at h
  · cases h; cases hm
  · simp only at h
    split at h
    · cases h
    · rename_i sp hsp
      cases h
      have hid : e ∈ (resolveRels g rels).map (·.1) := List.mem_map.2 ⟨_, hm, rfl⟩
      have het : (e, t) ∈ (resolveRels g rels).map (fun r => (r.1, r.2.1)) := List.mem_map.2 ⟨_, hm, rfl⟩
      obtain ⟨hr1, -⟩ := hC.release_ok _ _ _ _ hsp
      refine ⟨?_, by simp [hid], ?_, ?_⟩
      · simp [Mat.get, Mat.removeMask, hw.1.symm, hw.2.symm, het]
      · simp only [setRelIds, hr1, relIds]
        simp only [union, List.mem_append, List.mem_filter, decide_eq_true_eq]
        by_cases hh : e ∈ g.delRels
        · exact Or.inl hh
        · exact Or.inr ⟨hid, hh⟩
      · intro tn htn
        simp only [List.getElem?_mapIdx, Option.mem_def, Option.map_eq_some_iff] at htn
        obtain ⟨tn0, -, rfl⟩ := htn
        simp [het]

/-- **BUG (graph.rs:1229 + :2853).** When `relationship_type_matrix` is
narrower than the type table — exactly what `get_type_id_mut` leaves behind —
the bulk type-matrix removal is a no-op: a deleted edge keeps its type. -/
theorem deleteRels_stale (O : IdSpaceOps) (g : G V) (rels : List Nat) (r : G V)
    (out : List (Nat × Nat × Nat × Nat))
    (h : deleteRels O g rels = .ok (r, out)) (hw : g.relType.nc ≠ g.types.length) :
    r.relType = g.relType := by
  unfold deleteRels at h
  split at h
  · cases h; rfl
  · simp only at h
    split at h
    · cases h
    · cases h; simp [setRelIds, Mat.removeMask, Ne.symm hw]

/-- End to end: register a fresh type through `get_type_id_mut`
(`GRAPH.CONSTRAINT CREATE … RELATIONSHIP <new type>`, or a replica applying
a schema record), then `DELETE` an existing edge: its old type stays in the
type matrix. When the id is reused by an edge of a later type `t2 > t`,
`get_relationship_type_id` (first entry of the row) still answers `t` —
live repro: `type(r)` returns the deleted edge's type. -/
theorem stale_type_after_registration (O : IdSpaceOps) (g : G V) (snew : String) (e t : Nat)
    (hn : snew ∉ g.types) (hw : g.relType.nc = g.types.length)
    (ht : g.relType.get e t = true) (r : G V) (out : List (Nat × Nat × Nat × Nat))
    (h : deleteRels O (getTypeIdMut g snew).1 [e] = .ok (r, out)) :
    r.relType.get e t = true := by
  have hs := getTypeIdMut_stale_width g snew hn hw
  rw [deleteRels_stale O _ _ _ _ h (Nat.ne_of_lt hs)]
  have hk : (getTypeIdMut g snew).1.relType = g.relType := by
    simp [getTypeIdMut, (idx_none g.types snew).2 hn]
  rw [hk]; exact ht

theorem relTypeId_two (g : G V) (e t t2 : Nat) (h : g.relType.row e = [(e, t), (e, t2)]) :
    relTypeId g e = some t := by simp [relTypeId, h]

/-- Pairs that lost their last edge leave the adjacency matrix. -/
theorem deleteRels_adj (O : IdSpaceOps) (g : G V) (rels : List Nat) (r : G V)
    (out : List (Nat × Nat × Nat × Nat))
    (h : deleteRels O g rels = .ok (r, out)) (hw : g.adj.nr = g.nodeCap ∧ g.adj.nc = g.nodeCap)
    (e t s d : Nat) (hm : (e, t, s, d) ∈ out) (hr : remainingEdge r.relMs s d = false) :
    (s, d) ∉ r.adj.ents := by
  unfold deleteRels at h
  split at h
  · cases h; cases hm
  · simp only at h
    split at h
    · cases h
    · cases h
      simp only [Mat.removeMask, hw.1, hw.2, and_self, ite_true, List.mem_filter,
        decide_eq_true_eq]
      rintro ⟨_, hnot⟩
      apply hnot
      refine ⟨List.mem_map.2 ⟨_, hm, rfl⟩, ?_⟩
      have := hr; simp only at this; rw [this]; rfl

/-! ## `delete_implicit_edges` -/

def collectImplicit (tn : Ten) (dn ex : List Nat) : Ten :=
  dn.flatMap (fun n =>
    (Ten.out tn n).filter (fun x => x.2.2 ∉ ex) ++
    (Ten.inc tn n).filter (fun x => x.1 != n && x.1 ∉ dn && x.2.2 ∉ ex))

/-- The collection is exactly the non-explicit edges incident to a deleted
node — none missed (so none dangle), and none outside. -/
theorem collectImplicit_mem (tn : Ten) (dn ex : List Nat) (x : Nat × Nat × Nat) (hx : x ∈ tn) :
    x ∈ collectImplicit tn dn ex ↔ (x.1 ∈ dn ∨ x.2.1 ∈ dn) ∧ x.2.2 ∉ ex := by
  obtain ⟨s, d, e⟩ := x
  simp only [collectImplicit, Ten.out, Ten.inc, List.mem_flatMap, List.mem_append, List.mem_filter,
    beq_iff_eq, Bool.and_eq_true, bne_iff_ne, ne_eq, decide_eq_true_eq]
  constructor
  · rintro ⟨n, hn, (⟨⟨-, rfl⟩, he⟩ | ⟨⟨-, rfl⟩, ⟨-, -⟩, he⟩)⟩
    · exact ⟨Or.inl hn, he⟩
    · exact ⟨Or.inr hn, he⟩
  · rintro ⟨(hs | hd), he⟩
    · exact ⟨s, hs, Or.inl ⟨⟨hx, rfl⟩, he⟩⟩
    · by_cases hs : s ∈ dn
      · exact ⟨s, hs, Or.inl ⟨⟨hx, rfl⟩, he⟩⟩
      · exact ⟨d, hd, Or.inr ⟨⟨hx, rfl⟩, ⟨fun h => hs (h ▸ hd), hs⟩, he⟩⟩

/-- `RoaringTreemap::from_iter`: the ids as a set (first occurrence kept). -/
def toSet (l : List Nat) : List Nat := l.foldl ins []

/-- `delete_implicit_edges` (:2894). The tensors, endpoint index and masks
move first; the freed ids go through `release(freed, freed)` (:3032) at
the end, once, as one set. -/
def deleteImplicit (O : IdSpaceOps) (g : G V) (dn ex : List Nat) :
    Except String (G V × List (Nat × Nat × Nat × Nat)) :=
  if g.relMs = [] then .ok (g, []) else
  let col := g.relMs.mapIdx (fun t tn => (collectImplicit tn dn ex).map (fun x => (x.2.2, t, x.1, x.2.1)))
  let all := col.flatMap id
  let ids := all.map (·.1)
  let et := all.map (fun r => (r.1, r.2.1))
  let relMs' := g.relMs.mapIdx (fun t tn => tn.filter (fun x => (x.2.2, t) ∉ et))
  let cand := (all.map (fun r => (r.2.2.1, r.2.2.2))).filter (fun p =>
    (p.1 ∈ dn ∧ p.2 ∈ dn) || !remainingEdge relMs' p.1 p.2)
  let ep : Nat → Option (Nat × Nat) := fun x => if x ∈ ids then none else g.endpoints x
  let g1 : G V := { g with endpoints := ep, relMs := relMs', relType := g.relType.removeMask g.relCap g.types.length et }
  let freed := toSet ids
  match O.release (relIds g1) freed freed with
  | none => .error "relationship"
  | some sp => .ok ({ setRelIds g1 sp with adj := g.adj.removeMask g.nodeCap g.nodeCap cand }, all)

theorem foldl_ins_mem (l acc : List Nat) (x : Nat) : x ∈ l.foldl ins acc ↔ x ∈ acc ∨ x ∈ l := by
  induction l generalizing acc with
  | nil => simp
  | cons y l ih =>
    simp only [List.foldl_cons, ih, ins, List.mem_cons]
    by_cases hy : y ∈ acc
    · simp only [hy, ite_true]
      constructor
      · rintro (h | h)
        · exact Or.inl h
        · exact Or.inr (Or.inr h)
      · rintro (h | h | h)
        · exact Or.inl h
        · exact Or.inl (h ▸ hy)
        · exact Or.inr h
    · simp only [hy, ite_false, List.mem_append, List.mem_singleton]
      constructor
      · rintro ((h | h) | h)
        · exact Or.inl h
        · exact Or.inr (Or.inl h)
        · exact Or.inr (Or.inr h)
      · rintro (h | h | h)
        · exact Or.inl (Or.inl h)
        · exact Or.inl (Or.inr h)
        · exact Or.inr h

theorem toSet_mem (l : List Nat) (x : Nat) : x ∈ toSet l ↔ x ∈ l := by
  simp [toSet, foldl_ins_mem]

/-- No edge incident to a deleted node survives in any tensor unless it is
an explicit delete (handled by `delete_relationships` in the same commit). -/
theorem deleteImplicit_no_dangling (O : IdSpaceOps) (g : G V) (dn ex : List Nat) (hne : g.relMs ≠ [])
    (r : G V) (out : List (Nat × Nat × Nat × Nat)) (hok : deleteImplicit O g dn ex = .ok (r, out))
    (t : Nat) (tn : Ten) (htn : g.relMs[t]? = some tn) (x : Nat × Nat × Nat) (hx : x ∈ tn)
    (hinc : x.1 ∈ dn ∨ x.2.1 ∈ dn) (hex : x.2.2 ∉ ex) :
    ∀ tn' ∈ r.relMs[t]?, x ∉ tn' := by
  intro tn' h'
  simp only [deleteImplicit, hne, ite_false] at hok
  split at hok
  · cases hok
  · cases hok
    simp only [setRelIds, List.getElem?_mapIdx, htn, Option.map_some,
      Option.mem_def, Option.some.injEq] at h'
    subst h'
    intro hmem
    have hf := (List.mem_filter.1 hmem).2
    simp only [decide_eq_true_eq] at hf
    apply hf
    have hcol : ((collectImplicit tn dn ex).map (fun x => (x.2.2, t, x.1, x.2.1))) ∈
        g.relMs.mapIdx (fun t tn => (collectImplicit tn dn ex).map (fun x => (x.2.2, t, x.1, x.2.1))) := by
      exact List.mem_of_getElem? (i := t) (by simp [List.getElem?_mapIdx, htn])
    have hr : (x.2.2, t, x.1, x.2.1) ∈ (collectImplicit tn dn ex).map (fun x => (x.2.2, t, x.1, x.2.1)) :=
      List.mem_map.2 ⟨x, (collectImplicit_mem tn dn ex x hx).2 ⟨hinc, hex⟩, rfl⟩
    exact List.mem_map.2 ⟨_, List.mem_flatMap.2 ⟨_, hcol, hr⟩, rfl⟩

/-- Every implicitly deleted edge id ends up free (the bin and the count now
move from one set, `freed`). -/
theorem deleteImplicit_freed (O : IdSpaceOps) (hC : IdSpaceContract O) (g : G V) (dn ex : List Nat)
    (r : G V) (out : List (Nat × Nat × Nat × Nat)) (hok : deleteImplicit O g dn ex = .ok (r, out))
    (y : Nat × Nat × Nat × Nat) (hy : y ∈ out) : y.1 ∈ r.delRels := by
  unfold deleteImplicit at hok
  split at hok
  · cases hok; cases hy
  · simp only at hok
    split at hok
    · cases hok
    · rename_i sp hsp
      cases hok
      obtain ⟨h1, -⟩ := hC.release_ok _ _ _ _ hsp
      simp only [setRelIds, h1, union, List.mem_append, List.mem_filter, decide_eq_true_eq, relIds]
      by_cases hh : y.1 ∈ g.delRels
      · exact Or.inl hh
      · exact Or.inr ⟨(toSet_mem _ _).2 (List.mem_map.2 ⟨y, hy, rfl⟩), decide_eq_true hh⟩

/-! ## `create_relationships_bulk` -/

/-- `create_relationships_bulk` (:2518): refuse slices that disagree in
length (:2538) or repeat an id (:2548) — both before anything moves — then
`relationship_ids.create(ids)` (:2557), capacity growth, registration and
the logical effect: tensor, endpoint index, adjacency and type matrix all
gain the batch. -/
def createRelsBulk (O : IdSpaceOps) (chunk : Nat) (hc : 0 < chunk) (g : G V) (ty : String)
    (srcs dsts ids : List Nat) : Except String (G V) :=
  if srcs.length ≠ ids.length ∨ dsts.length ≠ ids.length then .error "graph: lengths" else
  if ¬ ids.Nodup then .error "graph: duplicate id" else
  match O.create (relIds g) ids with
  | none => .error "relationship"
  | some sp =>
  let es := srcs.zip (dsts.zip ids)
  let g1 : G V := setRelIds g sp
  let g2 := match ids.max? with
    | some m => if m + 1 > g1.relCap then
        resizeRelMs { g1 with relCap := growCap chunk g1.relCap (m + 1) hc } else g1
    | none => g1
  let (g3, t) := getRelMatMut chunk hc g2 ty
  let g4 := resize chunk hc g3
  .ok { g4 with
    relMs := g4.relMs.modify t (· ++ es)
    endpoints := fun x => match es.find? (·.2.2 == x) with
      | some y => some (y.1, y.2.1) | none => g4.endpoints x
    adj := es.foldl (fun m y => m.set y.1 y.2.1) g4.adj
    relType := es.foldl (fun m y => m.set y.2.2 t) g4.relType }

/-- The two shape refusals (new in #2846): `[0, 0]` no longer counts one
relationship while building two tensor edges, and a short `srcs` no longer
truncates the zip after every id was counted. -/
theorem createRelsBulk_ragged (O : IdSpaceOps) (chunk : Nat) (hc : 0 < chunk) (g : G V) (ty : String)
    (srcs dsts ids : List Nat) (h : srcs.length ≠ ids.length ∨ dsts.length ≠ ids.length) :
    createRelsBulk O chunk hc g ty srcs dsts ids = .error "graph: lengths" := by
  simp [createRelsBulk, h]
theorem createRelsBulk_dup (O : IdSpaceOps) (chunk : Nat) (hc : 0 < chunk) (g : G V) (ty : String)
    (srcs dsts ids : List Nat) (hl : srcs.length = ids.length ∧ dsts.length = ids.length)
    (h : ¬ ids.Nodup) :
    createRelsBulk O chunk hc g ty srcs dsts ids = .error "graph: duplicate id" := by
  simp [createRelsBulk, hl, h]

theorem foldl_set_mem (m : Mat) (es : List (Nat × Nat × Nat)) (f : Nat × Nat × Nat → Nat × Nat)
    (p : Nat × Nat) (hp : p ∈ m.ents) : p ∈ (es.foldl (fun m y => m.set (f y).1 (f y).2) m).ents := by
  induction es generalizing m with
  | nil => exact hp
  | cons y es ih =>
    simp only [List.foldl_cons]
    apply ih; unfold Mat.set; split
    · simp [hp]
    · exact hp

theorem set_dims (m : Mat) (i j : Nat) : (m.set i j).nr = m.nr ∧ (m.set i j).nc = m.nc := by
  unfold Mat.set; split <;> simp

theorem foldl_set_dims (m : Mat) (es : List (Nat × Nat × Nat)) (f : Nat × Nat × Nat → Nat × Nat) :
    (es.foldl (fun m y => m.set (f y).1 (f y).2) m).nr = m.nr ∧
    (es.foldl (fun m y => m.set (f y).1 (f y).2) m).nc = m.nc := by
  induction es generalizing m with
  | nil => exact ⟨rfl, rfl⟩
  | cons y es ih =>
    simp only [List.foldl_cons]
    rw [(ih _).1, (ih _).2]; exact set_dims _ _ _

theorem foldl_set_has (m : Mat) (es : List (Nat × Nat × Nat)) (f : Nat × Nat × Nat → Nat × Nat)
    (y : Nat × Nat × Nat) (hy : y ∈ es) (h1 : (f y).1 < m.nr) (h2 : (f y).2 < m.nc) :
    f y ∈ (es.foldl (fun m y => m.set (f y).1 (f y).2) m).ents := by
  induction es generalizing m with
  | nil => cases hy
  | cons z es ih =>
    simp only [List.foldl_cons]
    rcases List.mem_cons.1 hy with rfl | hy'
    · apply foldl_set_mem; unfold Mat.set; split
      · simp
      · rename_i hn; simp only [not_and, Classical.not_not] at hn; exact hn h1 h2
    · have hd := set_dims m (f z).1 (f z).2
      exact ih _ hy' (hd.1 ▸ h1) (hd.2 ▸ h2)

theorem foldl_set_get (m : Mat) (es : List (Nat × Nat × Nat)) (f : Nat × Nat × Nat → Nat × Nat)
    (y : Nat × Nat × Nat) (hy : y ∈ es) (h1 : (f y).1 < m.nr) (h2 : (f y).2 < m.nc) :
    (es.foldl (fun m y => m.set (f y).1 (f y).2) m).get (f y).1 (f y).2 = true := by
  have hm := foldl_set_has m es f y hy h1 h2
  obtain ⟨a, b⟩ := foldl_set_dims m es f
  simp [Mat.get, a, b, h1, h2, hm]

/-- Each new edge resolves to its endpoints. Uniqueness of the ids is no
longer a hypothesis: the function refuses anything else. -/
theorem createRelsBulk_endpoints (O : IdSpaceOps) (chunk : Nat) (hc : 0 < chunk) (g r : G V) (ty : String)
    (srcs dsts ids : List Nat) (h : createRelsBulk O chunk hc g ty srcs dsts ids = .ok r)
    (y : Nat × Nat × Nat) (hy : y ∈ srcs.zip (dsts.zip ids)) :
    r.endpoints y.2.2 = some (y.1, y.2.1) := by
  unfold createRelsBulk at h
  split at h
  · cases h
  · rename_i hlen
    split at h
    · cases h
    · rename_i hnd0
      have hnd : ids.Nodup := Classical.not_not.1 hnd0
      split at h
      · cases h
      · cases h
        simp only
        have hm : (srcs.zip (dsts.zip ids)).map (·.2.2) = ids := by
          have h1 : (srcs.zip (dsts.zip ids)).map (·.2.2) = ((srcs.zip (dsts.zip ids)).map Prod.snd).map Prod.snd := by
            simp [List.map_map]
          rw [h1, List.map_snd_zip (by simp; omega), List.map_snd_zip (by omega)]
        have hnd' : ((srcs.zip (dsts.zip ids)).map (·.2.2)).Nodup := by rw [hm]; exact hnd
        generalize srcs.zip (dsts.zip ids) = es at hy hnd'
        have : es.find? (·.2.2 == y.2.2) = some y := by
          induction es with
          | nil => cases hy
          | cons z es ih =>
            simp only [List.map_cons, List.nodup_cons] at hnd'
            rcases List.mem_cons.1 hy with rfl | hy'
            · simp
            · have hz : z.2.2 ≠ y.2.2 := fun he => hnd'.1 (he ▸ List.mem_map_of_mem hy')
              simp [hz, ih hy' hnd'.2]
        rw [this]

end GQ
