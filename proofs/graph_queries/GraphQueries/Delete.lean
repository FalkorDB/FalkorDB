import GraphQueries.Queries
/-
# Edge creation and deletion; the edge type column

| here | there (graph.rs) |
| --- | --- |
| `resolveRels`  | `delete_relationships` phase 1 (the `Phase 1` block from :2964) |
| `deleteRels`   | `delete_relationships`  (phase 2 from :2989; `release(rels, resolved)` ) |
| `collectImplicit` | `delete_implicit_edges` collection loop (from :3088) |
| `deleteImplicit`  | `delete_implicit_edges`  (`release(freed, freed)` ) |
| `createRelsBulk`  | `create_relationships_bulk`  (shape refusals, the type-slot refusal (#3967), `create` ) |

Every id-space move goes through `O` (IdContract.lean); theorems that need
its behaviour take `IdSpaceContract O`.

#3967: an edge's type lives in the endpoint index's type column (`G.relTypes`), written by
`set` with the endpoints and cleared by `clear_edge_endpoint` with them; the
`relationship_type_matrix` and its mask removal are gone. (Before, a type matrix narrower
than the type table made the mask removal a no-op, #3277; there is no width left to get wrong.)
`Mat.removeMask` (the adjacency mask) is a no-op on a dimension mismatch, which is
what the release build does (`GrB_DIMENSION_MISMATCH` is only `debug_assert`ed, matrix.rs).
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
                   relTypes := fun x => if x ∈ ids then none else g.relTypes x,
                   adj := g.adj.removeMask g.nodeCap g.nodeCap cand }, res)

/-- Phase 1 does not mutate, so a refusal (an id already free) changes nothing. -/
theorem deleteRels_refused (O : IdSpaceOps) (hC : IdSpaceContract O) (g : G V) (rels : List Nat)
    (e : Nat) (he : e ∈ rels) (hd : e ∈ g.delRels) :
    deleteRels O g rels = .error "relationship" := by
  have hne : rels ≠ [] := by intro h; subst h; cases he
  simp [deleteRels, hne, hC.release_refuses_recycled (relIds g) rels _ e he hd]

/-- Every deleted edge loses its type, its endpoints, and its tensor entry, and is binned
(#3967: no dimension hypothesis, the type is cleared with the endpoints). -/
theorem deleteRels_clears (O : IdSpaceOps) (hC : IdSpaceContract O) (g : G V) (rels : List Nat)
    (r : G V) (out : List (Nat × Nat × Nat × Nat))
    (h : deleteRels O g rels = .ok (r, out))
    (e t s d : Nat) (hm : (e, t, s, d) ∈ out) :
    r.relTypes e = none ∧ r.endpoints e = none ∧ e ∈ r.delRels ∧
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
      refine ⟨by simp [hid], by simp [hid], ?_, ?_⟩
      · simp only [setRelIds, hr1, relIds]
        simp only [union, List.mem_append, List.mem_filter, decide_eq_true_eq]
        by_cases hh : e ∈ g.delRels
        · exact Or.inl hh
        · exact Or.inr ⟨hid, hh⟩
      · intro tn htn
        simp only [List.getElem?_mapIdx, Option.mem_def, Option.map_eq_some_iff] at htn
        obtain ⟨tn0, -, rfl⟩ := htn
        simp [het]

/-- **`type_cleared_after_registration`** registering a type through
`get_type_id_mut` (constraint creation, a replica's schema record) leaves every edge's type
as it was, so a later `DELETE` of an edge clears its type (#3967: with no hypothesis on a
type-matrix width, which no longer exists). -/
theorem type_cleared_after_registration (O : IdSpaceOps) (hC : IdSpaceContract O) (chunk : Nat)
    (hc : 0 < chunk) (g : G V) (snew : String)
    (rels : List Nat) (r : G V) (out : List (Nat × Nat × Nat × Nat))
    (h : deleteRels O (getTypeIdMut chunk hc g snew).1 rels = .ok (r, out))
    (e t s d : Nat) (hm : (e, t, s, d) ∈ out) :
    r.relTypes e = none :=
  (deleteRels_clears O hC _ rels r out h e t s d hm).1

/-- A deleted edge resolves to no type: `get_relationship_type_id` on it is the `expect`
panic, `edge_has_type_id` is false for every type. -/
theorem deleteRels_no_type (O : IdSpaceOps) (hC : IdSpaceContract O) (g : G V) (rels : List Nat)
    (r : G V) (out : List (Nat × Nat × Nat × Nat)) (h : deleteRels O g rels = .ok (r, out))
    (e t s d : Nat) (hm : (e, t, s, d) ∈ out) (t' : Nat) :
    relTypeId r e = none ∧ edgeHasTypeId r e t' = false := by
  have := (deleteRels_clears O hC g rels r out h e t s d hm).1
  simp [relTypeId, edgeHasTypeId, this]

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

/-- `delete_implicit_edges` (:3088). The tensors, endpoint index and masks
move first; the freed ids go through `release(freed, freed)` (:3232) at
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
  let g1 : G V := { g with endpoints := ep, relMs := relMs', relTypes := fun x => if x ∈ ids then none else g.relTypes x }
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
      simp only [setRelIds, h1, union, List.mem_append, List.mem_filter, relIds]
      by_cases hh : y.1 ∈ g.delRels
      · exact Or.inl hh
      · exact Or.inr ⟨(toSet_mem _ _).2 (List.mem_map.2 ⟨y, hy, rfl⟩), decide_eq_true hh⟩

/-- Every implicitly deleted edge reads no type afterwards (#3967: `clear_edge_endpoint`
clears the type column with the endpoints; there is no type-matrix mask). -/
theorem deleteImplicit_no_type (O : IdSpaceOps) (g : G V) (dn ex : List Nat)
    (r : G V) (out : List (Nat × Nat × Nat × Nat)) (hok : deleteImplicit O g dn ex = .ok (r, out))
    (y : Nat × Nat × Nat × Nat) (hy : y ∈ out) : r.relTypes y.1 = none ∧ r.endpoints y.1 = none := by
  unfold deleteImplicit at hok
  split at hok
  · cases hok; cases hy
  · simp only at hok
    split at hok
    · cases hok
    · cases hok
      have : y.1 ∈ ((g.relMs.mapIdx (fun t tn => (collectImplicit tn dn ex).map
          (fun x => (x.2.2, t, x.1, x.2.1)))).flatMap id).map (·.1) := List.mem_map.2 ⟨y, hy, rfl⟩
      exact ⟨ite_cond_eq_true _ _ (eq_true this), ite_cond_eq_true _ _ (eq_true this)⟩

/-! ## `create_relationships_bulk` -/

/-- `create_relationships_bulk`: refuse slices that disagree in
length or repeat an id, then (#3967) refuse a type whose slot does not fit the type column
(`relationship_type_slot`) — all before anything moves — then
`relationship_ids.create(ids)`, capacity growth, registration and
the logical effect: tensor, endpoint index (endpoints and type), adjacency all
gain the batch. -/
def createRelsBulk (O : IdSpaceOps) (chunk : Nat) (hc : 0 < chunk) (g : G V) (ty : String)
    (srcs dsts ids : List Nat) : Except String (G V) :=
  if srcs.length ≠ ids.length ∨ dsts.length ≠ ids.length then .error "graph: lengths" else
  if ¬ ids.Nodup then .error "graph: duplicate id" else
  match relTypeSlot g ty with
  | none => .error "graph: too many relationship types"
  | some slot =>
  match O.create (relIds g) ids with
  | none => .error "relationship"
  | some sp =>
  let es := srcs.zip (dsts.zip ids)
  let g1 : G V := setRelIds g sp
  let g2 := match ids.max? with
    | some m => if m + 1 > g1.relCap then
        { g1 with relCap := growCap chunk g1.relCap (m + 1) hc } else g1
    | none => g1
  let (g3, t) := getRelMatMut chunk hc g2 ty
  let g4 := resize chunk hc g3
  .ok { g4 with
    relMs := g4.relMs.modify t (· ++ es)
    endpoints := fun x => match es.find? (·.2.2 == x) with
      | some y => some (y.1, y.2.1) | none => g4.endpoints x
    adj := es.foldl (fun m y => m.set y.1 y.2.1) g4.adj
    relTypes := fun x => if x ∈ ids then some slot else g4.relTypes x }

/-- The two shape refusals `[0, 0]` is refused rather than counting one
relationship while building two tensor edges, and a short `srcs` is refused rather than
truncating the zip after every id was counted. -/
theorem createRelsBulk_ragged (O : IdSpaceOps) (chunk : Nat) (hc : 0 < chunk) (g : G V) (ty : String)
    (srcs dsts ids : List Nat) (h : srcs.length ≠ ids.length ∨ dsts.length ≠ ids.length) :
    createRelsBulk O chunk hc g ty srcs dsts ids = .error "graph: lengths" := by
  simp [createRelsBulk, h]
theorem createRelsBulk_dup (O : IdSpaceOps) (chunk : Nat) (hc : 0 < chunk) (g : G V) (ty : String)
    (srcs dsts ids : List Nat) (hl : srcs.length = ids.length ∧ dsts.length = ids.length)
    (h : ¬ ids.Nodup) :
    createRelsBulk O chunk hc g ty srcs dsts ids = .error "graph: duplicate id" := by
  simp [createRelsBulk, hl, h]

/-- **#3967: a type past the limit is refused before anything moves**: no id is taken, no type
registered, no edge written (nothing is returned but the error). -/
theorem createRelsBulk_type_refused (O : IdSpaceOps) (chunk : Nat) (hc : 0 < chunk) (g : G V) (ty : String)
    (srcs dsts ids : List Nat) (hl : srcs.length = ids.length ∧ dsts.length = ids.length)
    (hn : ids.Nodup) (h : relTypeSlot g ty = none) :
    createRelsBulk O chunk hc g ty srcs dsts ids = .error "graph: too many relationship types" := by
  simp [createRelsBulk, hl, hn, h]

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

/-- **#3967: each new edge reads its type**: the type column holds, for every id of the
batch, the index of `ty` in the type table (the slot checked up front is the id
registration gives it, `relTypeSlot_getRelMatMut`), and that slot fits a `u16` below the
empty marker. -/
theorem createRelsBulk_types (O : IdSpaceOps) (chunk : Nat) (hc : 0 < chunk) (g r : G V) (ty : String)
    (hT : TInv g) (srcs dsts ids : List Nat) (h : createRelsBulk O chunk hc g ty srcs dsts ids = .ok r)
    (e : Nat) (he : e ∈ ids) :
    ∃ t, r.relTypes e = some t ∧ r.types[t]? = some ty ∧ t ≤ maxTypeId ∧
      relTypeId r e = some t ∧ edgeHasType r e ty = true := by
  unfold createRelsBulk at h
  split at h
  · cases h
  split at h
  · cases h
  split at h
  · cases h
  rename_i slot hslot
  split at h
  · cases h
  rename_i sp hsp
  cases h
  generalize hg2 : (match ids.max? with
      | some m => if m + 1 > (setRelIds g sp).relCap then
          { setRelIds g sp with relCap := growCap chunk (setRelIds g sp).relCap (m + 1) hc } else setRelIds g sp
      | none => setRelIds g sp) = g2
  have hT1 : TInv g2 := by
    rw [← hg2]; split
    · split <;> simpa [TInv, setRelIds] using hT
    · simpa [TInv, setRelIds] using hT
  have hty : g2.types = g.types := by
    rw [← hg2]; split
    · split <;> rfl
    · rfl
  have hslot2 : relTypeSlot g2 ty = some slot := by
    unfold relTypeSlot getTypeId at hslot ⊢; rw [hty]; exact hslot
  have hidx := relTypeSlot_getRelMatMut chunk hc g2 ty hT1 slot hslot2
  have hspec := getRelMatMut_spec chunk hc g2 ty hT1
  obtain ⟨-, -, -, k4, -⟩ := resize_keeps chunk hc (getRelMatMut chunk hc g2 ty).1
  have hfit : slot ≤ maxTypeId := by
    obtain ⟨hs, hle⟩ := (edgeTypeSlot_spec _ _).1 hslot; rw [hs]; exact hle
  have htys : (resize chunk hc (getRelMatMut chunk hc g2 ty).1).types[slot]? = some ty := by
    rw [k4, ← hidx]; exact hspec.2.1
  have hmem : ty ∈ (getRelMatMut chunk hc g2 ty).1.types := List.mem_of_getElem? hspec.2.1
  obtain ⟨i, hi⟩ := idx_some _ _ hmem
  have h2 : (getRelMatMut chunk hc g2 ty).2 = (idx (getRelMatMut chunk hc g2 ty).1.types ty).getD 0 := rfl
  rw [hi, Option.getD_some, hidx] at h2
  have hid : getTypeId (resize chunk hc (getRelMatMut chunk hc g2 ty).1) ty = some slot := by
    unfold getTypeId; rw [k4, hi, h2]
  refine ⟨slot, ?_, ?_, hfit, ?_, ?_⟩
  · simp [he]
  · exact htys
  · simp [relTypeId, he]
  · show (match getTypeId (resize chunk hc (getRelMatMut chunk hc g2 ty).1) ty with
      | some t => edgeHasTypeId _ e t | none => false) = true
    rw [hid]; simp [edgeHasTypeId, he]

end GQ
