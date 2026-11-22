import PendingCommit.PendingDocs
/-
# Unwinding a pending node, paired with the graph's id space

`delete_pending_node` (pending.rs) and
`remove_pending_relationships_for_node` (:902) take `g: &mut Graph` and hand
every unwound id back through `Graph::cancel_node_id` /
`cancel_relationship_id` (graph.rs) themselves. Each call is `?`:
the first refusal aborts the unwind.

The graph is abstract here (`Gr`); what is needed of its cancel is the
hypothesis `CancelSpec freeN freeR cancel` — each id space has its own free
predicate (`freeN` for node ids, `freeR` for relationship ids): after
`cancel g id = some g'`, `id` is free in `g'` *in the cancel's own space*, and
nothing free in either space stops being free. That is `IdSpace::cancel`'s
`recycled.insert(id)` (id_space.rs) read through `is_free`; proofs/id_space
owns it, and proofs/graph_queries states it as `IdSpaceContract.cancel_ok`.

* `removeRelsG_spec`: on success the `Pending` side is exactly the pure
  `removeRels` (so `removeRels_spec` applies: incident edges gone,
  `RelInv` kept, `taken` loses them) and every cascaded id is free in the graph.
* `deletePendingNode_spec`: the node leaves `created`, enters `cancelled_nodes`,
  its staged labels/attrs are dropped, and its id and every cascaded edge id
  are free in the graph — no id is unwound on one side only.
* `deletePendingNode_refused`: a refused graph cancel is an error.
-/
namespace PendingCommit.PS
open PA

/-- Hypothesis on the graph's cancel (see header). `free` is the free predicate of the space the
cancel acts on, `other` that of the other space (which a cancel may not shrink). -/
structure CancelSpec {Gr : Type} (free other : Gr → List Nat) (cancel : Gr → Nat → Option Gr) : Prop where
  ok : ∀ g id g', cancel g id = some g' →
    id ∈ free g' ∧ (∀ x ∈ free g, x ∈ free g') ∧ ∀ x ∈ other g, x ∈ other g'

/-- **Hypothesis on `Graph::get_type_id_mut`** (graph.rs; proofs/graph_queries
`getTypeIdMut_spec`): afterwards the type is registered, every registered type stays registered,
and neither id space loses a free id (registering a type touches names and tensors only). `types`
is the graph's relationship-type list (names as `Nat`s here). Added for #3641, which makes
`remove_pending_relationships_for_node` call it for every cancelled edge (:974). -/
structure RegSpec {Gr : Type} (types freeN freeR : Gr → List Nat) (regT : Gr → Nat → Gr) : Prop where
  ok : ∀ g t, t ∈ types (regT g t) ∧ (∀ y ∈ types g, y ∈ types (regT g t)) ∧
    (∀ x ∈ freeN g, x ∈ freeN (regT g t)) ∧ ∀ x ∈ freeR g, x ∈ freeR (regT g t)

/-- A graph cancel (`cancel_node_id` / `cancel_relationship_id`) drops no registered type. -/
def KeepsTypes {Gr : Type} (types : Gr → List Nat) (cancel : Gr → Nat → Option Gr) : Prop :=
  ∀ g id g', cancel g id = some g' → ∀ y ∈ types g, y ∈ types g'

/-- `remove_pending_relationships_for_node` (:902): the second loop (:925-955),
each iteration the `Pending` removals (`rmStep`) then
`g.cancel_relationship_id(rel_id)?` (:947) then, since #3641, `g.get_type_id_mut(&type_name)`
(:974; the result is discarded: `regT`). -/
def rmStepG {Gr : Type} (cancelR : Gr → Nat → Option Gr) (regT : Gr → Nat → Gr) (pg : P × Gr)
    (x : (Nat × Nat × Nat) × Nat) : Option (P × Gr) :=
  (cancelR pg.2 x.1.1).map (fun g' => (rmStep pg.1 x, regT g' x.2))

def removeRelsG {Gr : Type} (cancelR : Gr → Nat → Option Gr) (regT : Gr → Nat → Gr) (p : P) (g : Gr)
    (id : Nat) : Option (P × Gr) :=
  (collect p id).foldlM (rmStepG cancelR regT) (p, g)

/-- The type names of a collected list. -/
def Tys (L : List ((Nat × Nat × Nat) × Nat)) : List Nat := L.map (·.2)

theorem foldlM_rmStepG {Gr : Type} (types freeN freeR : Gr → List Nat) (cancelR : Gr → Nat → Option Gr)
    (regT : Gr → Nat → Gr) (hc : CancelSpec freeR freeN cancelR) (hg : RegSpec types freeN freeR regT)
    (hk : KeepsTypes types cancelR) :
    ∀ (L : List ((Nat × Nat × Nat) × Nat)) (p : P) (g : Gr) (p' : P) (g' : Gr),
    L.foldlM (rmStepG cancelR regT) (p, g) = some (p', g') →
    p' = L.foldl rmStep p ∧ (∀ r ∈ Rids L, r ∈ freeR g') ∧ (∀ x ∈ freeR g, x ∈ freeR g') ∧
      (∀ x ∈ freeN g, x ∈ freeN g') ∧ (∀ t ∈ Tys L, t ∈ types g') ∧ ∀ y ∈ types g, y ∈ types g'
  | [], p, g, p', g', h => by
    simp only [List.foldlM_nil, pure, Option.some.injEq, Prod.mk.injEq] at h
    obtain ⟨rfl, rfl⟩ := h
    exact ⟨rfl, by simp [Rids], fun _ hx => hx, fun _ hx => hx, by simp [Tys], fun _ hx => hx⟩
  | x :: L, p, g, p', g', h => by
    simp only [List.foldlM_cons, bind, rmStepG] at h
    cases hcg : cancelR g x.1.1 with
    | none => simp [hcg] at h
    | some g1 =>
      simp only [hcg, Option.map_some, Option.bind_some] at h
      obtain ⟨e1, e2, e3, e4, e5, e6⟩ := foldlM_rmStepG types freeN freeR cancelR regT hc hg hk L _ _ _ _ h
      obtain ⟨c1, c2, c3⟩ := hc.ok _ _ _ hcg
      obtain ⟨r1, r2, r3, r4⟩ := hg.ok g1 x.2
      refine ⟨by simpa using e1, fun r hr => ?_, fun y hy => e3 y (r4 y (c2 y hy)),
        fun y hy => e4 y (r3 y (c3 y hy)), fun t ht => ?_, fun y hy => e6 y (r2 y (hk _ _ _ hcg y hy))⟩
      · simp only [Rids, List.map_cons, List.mem_cons] at hr
        rcases hr with rfl | hr
        · exact e3 _ (r4 _ c1)
        · exact e2 r hr
      · simp only [Tys, List.map_cons, List.mem_cons] at ht
        rcases ht with rfl | ht
        · exact e6 _ r1
        · exact e5 t ht

theorem foldl_rmStep_taken : ∀ (L : List ((Nat × Nat × Nat) × Nat)) (p : P),
    (L.foldl rmStep p).taken = p.taken.filter (· ∉ Rids L)
  | [], p => by simp only [List.foldl_nil]; exact (List.filter_eq_self.2 (by simp [Rids])).symm
  | (e, t) :: L, p => by
    simp only [List.foldl_cons, foldl_rmStep_taken L, rmStep, List.filter_filter]
    congr 1; funext x
    by_cases hx : x = e.1 <;> by_cases h2 : x ∈ Rids L <;> simp [Rids] at h2 <;> simp [Rids, hx, h2]

/-- **`removeRelsG_spec`**: success means the `Pending` side is the pure
`removeRels`, every cascaded relationship id is free in the graph's
relationship space (and nothing free in either space is lost), and (#3641) every cascaded edge's
type is registered in the graph. -/
theorem removeRelsG_spec {Gr : Type} (types freeN freeR : Gr → List Nat) (cancelR : Gr → Nat → Option Gr)
    (regT : Gr → Nat → Gr) (hc : CancelSpec freeR freeN cancelR) (hg : RegSpec types freeN freeR regT)
    (hk : KeepsTypes types cancelR) (p : P) (g : Gr) (id : Nat) (p' : P) (g' : Gr)
    (h : removeRelsG cancelR regT p g id = some (p', g')) :
    p' = removeRels p id ∧ (∀ r ∈ Rids (collect p id), r ∈ freeR g' ∧ r ∉ p'.taken) ∧
    (∀ x ∈ freeR g, x ∈ freeR g') ∧ (∀ x ∈ freeN g, x ∈ freeN g') ∧
    (∀ t ∈ Tys (collect p id), t ∈ types g') := by
  obtain ⟨e1, e2, e3, e4, e5, -⟩ := foldlM_rmStepG types freeN freeR cancelR regT hc hg hk _ p g p' g' h
  refine ⟨e1, fun r hr => ⟨e2 r hr, ?_⟩, e3, e4, e5⟩
  rw [e1]; rw [foldl_rmStep_taken]
  simp [hr]

/-- `delete_pending_node`'s `Pending`-only prefix: `created_nodes.remove(id)`, `set_labels.remove`,
`remove_labels.remove` (#3459). -/
def dropNode (p : P) (id : Nat) : P :=
  { p with created := p.created.filter (· != id), setL := frem p.setL id, remL := frem p.remL id }

/-- The node's staged attrs: `new_nodes_attrs.remove(id)`, else `existing_nodes_attrs.remove(id)`,
else empty (`.or_else(..).unwrap_or_default()`). -/
def nodeAttrs (p : P) (id : Nat) : List Entry := (fget p.newN id).getD ((fget p.existN id).getD [])

/-- Taking them out of their map, and (#3030) `cancelled_properties += count_set(&attrs)`. -/
def takeAttrs (p : P) (id : Nat) : P :=
  match fget p.newN id with
  | some a => { p with newN := frem p.newN id, cancelledP := p.cancelledP + countSet a }
  | none => { p with existN := frem p.existN id,
                     cancelledP := p.cancelledP + countSet ((fget p.existN id).getD []) }

/-- `delete_pending_node` (pending.rs), its effect on `Pending` and the graph: `dropNode`, the
staged attrs taken and counted (`takeAttrs`), `remove_pending_relationships_for_node(id, g)?`,
`cancelled_nodes.insert(id)`, `g.cancel_node_id(id)?`. -/
def deletePendingNode {Gr : Type} (cancelN cancelR : Gr → Nat → Option Gr) (regT : Gr → Nat → Gr)
    (p : P) (g : Gr) (id : Nat) : Option (P × Gr) :=
  match removeRelsG cancelR regT (takeAttrs (dropNode p id) id) g id with
  | none => none
  | some (p3, g3) =>
    let p4 : P := { p3 with cancelledN := sins p3.cancelledN id }
    (cancelN g3 id).map (fun g4 => (p4, g4))

theorem collect_congr (p q : P) (h : p.relsByType = q.relsByType) (id : Nat) :
    collect p id = collect q id := by simp [collect, h]

theorem foldl_rmStep_created : ∀ (L : List ((Nat × Nat × Nat) × Nat)) (p : P),
    (L.foldl rmStep p).created = p.created ∧ (L.foldl rmStep p).cancelledN = p.cancelledN
  | [], _ => ⟨rfl, rfl⟩
  | _ :: L, p => by simp only [List.foldl_cons]; exact foldl_rmStep_created L _

/-- **`deletePendingNode_spec`**: on success the node is no longer created,
is recorded as cancelled, its id is free in the graph's node space, every
cascaded relationship id is free in its relationship space and no longer `taken`
— the unwind is never half-done — and nothing free in either space is lost. Since #3641 every
cascaded edge's type is also registered in the graph (`hty`: the node cancel drops no type). -/
theorem deletePendingNode_spec {Gr : Type} (types freeN freeR : Gr → List Nat)
    (cancelN cancelR : Gr → Nat → Option Gr) (regT : Gr → Nat → Gr)
    (hn : CancelSpec freeN freeR cancelN) (hr : CancelSpec freeR freeN cancelR)
    (hg : RegSpec types freeN freeR regT) (hkr : KeepsTypes types cancelR)
    (hty : KeepsTypes types cancelN) (p : P) (g : Gr) (id : Nat)
    (p' : P) (g' : Gr) (h : deletePendingNode cancelN cancelR regT p g id = some (p', g')) :
    id ∉ p'.created ∧ id ∈ p'.cancelledN ∧ id ∈ freeN g' ∧
    (∀ r ∈ Rids (collect p id), r ∈ freeR g' ∧ r ∉ p'.taken) ∧
    (∀ x ∈ freeN g, x ∈ freeN g') ∧ (∀ x ∈ freeR g, x ∈ freeR g') ∧
    (∀ t ∈ Tys (collect p id), t ∈ types g') := by
  unfold deletePendingNode at h
  simp only at h
  split at h
  · cases h
  · rename_i p3 g3 hrm
    cases hcn : cancelN g3 id with
    | none => simp [hcn] at h
    | some g4 =>
      simp only [hcn, Option.map_some, Option.some.injEq, Prod.mk.injEq] at h
      obtain ⟨rfl, rfl⟩ := h
      obtain ⟨e1, e2, e3, e4, e5⟩ := removeRelsG_spec types freeN freeR cancelR regT hr hg hkr _ g id p3 g3 hrm
      obtain ⟨c1, c2, c3⟩ := hn.ok _ _ _ hcn
      have hcol : ∀ q : P, q.relsByType = p.relsByType → collect q id = collect p id :=
        fun q hq => collect_congr q p hq id
      have hc2 : collect (takeAttrs (dropNode p id) id) id = collect p id := by
        apply hcol; unfold takeAttrs dropNode; split <;> rfl
      have hcr : (removeRels (takeAttrs (dropNode p id) id) id).created = p.created.filter (· != id) := by
        unfold removeRels; rw [(foldl_rmStep_created _ _).1]; unfold takeAttrs dropNode; split <;> rfl
      rw [hc2] at e2 e5
      refine ⟨?_, by simp [mem_sins], c1, fun r hr' => ⟨c3 r (e2 r hr').1, (e2 r hr').2⟩,
        fun x hx => c2 x (e4 x hx), fun x hx => c3 x (e3 x hx), fun t ht => hty _ _ _ hcn t (e5 t ht)⟩
      simp only [e1, hcr, List.mem_filter]; simp

/-- **`deletePendingNode_refused`**: a refused node cancel fails the unwind. -/
theorem deletePendingNode_refused {Gr : Type} (cancelN cancelR : Gr → Nat → Option Gr) (regT : Gr → Nat → Gr)
    (p : P) (g : Gr)
    (id : Nat) (p3 : P) (g3 : Gr)
    (hrm : removeRelsG cancelR regT (takeAttrs (dropNode p id) id) g id = some (p3, g3))
    (hcn : cancelN g3 id = none) :
    deletePendingNode cancelN cancelR regT p g id = none := by
  unfold deletePendingNode; rw [hrm]; simp [hcn]

/-! ## `cancelled_properties` (#3030) -/

/-- What one cascaded edge adds: `attrs.as_deref().map_or(0, count_set)`. -/
def relProps (p : P) (x : (Nat × Nat × Nat) × Nat) : Nat := ((fget p.newR x.1.1).map countSet).getD 0

/-- The ids of a group list's incident edges are distinct when the type keys are, the ids within a
group are, and no id sits in two groups. -/
theorem rids_flatMap_nodup (id : Nat) : ∀ (m : FMap (List (Nat × Nat × Nat))),
    (m.map (·.1)).Nodup → (∀ g ∈ m, (g.2.map (·.1)).Nodup) →
    (∀ g ∈ m, ∀ g' ∈ m, ∀ r, r ∈ g.2.map (·.1) → r ∈ g'.2.map (·.1) → g.1 = g'.1) →
    (Rids (m.flatMap (fun g => (g.2.filter (incident id)).map (·, g.1)))).Nodup
  | [], _, _, _ => by simp [Rids]
  | g :: m, hk, hn, hx => by
    have ih := rids_flatMap_nodup id m (List.nodup_cons.1 hk).2
      (fun g' h => hn g' (List.mem_cons_of_mem _ h))
      (fun a ha b hb => hx a (List.mem_cons_of_mem _ ha) b (List.mem_cons_of_mem _ hb))
    have hg : g.1 ∉ m.map (·.1) := (List.nodup_cons.1 hk).1
    simp only [Rids, List.flatMap_cons, List.map_append] at ih ⊢
    rw [List.nodup_append]
    refine ⟨?_, ih, ?_⟩
    · have e : ((g.2.filter (incident id)).map (·, g.1)).map (·.1.1) = (g.2.filter (incident id)).map (·.1) := by
        simp [List.map_map, Function.comp_def]
      rw [e]; exact (hn g (by simp)).sublist (List.filter_sublist.map _)
    · intro a ha b hb hab
      subst hab
      simp only [List.mem_map, List.mem_filter, List.mem_flatMap] at ha hb
      obtain ⟨⟨e, t⟩, ⟨e', ⟨he', -⟩, he⟩, rfl⟩ := ha
      obtain ⟨⟨f, u⟩, ⟨g', hg', f', ⟨hf', -⟩, hf⟩, hfa⟩ := hb
      simp only [Prod.mk.injEq] at he hf
      obtain ⟨rfl, rfl⟩ := he; obtain ⟨rfl, rfl⟩ := hf
      have := hx g (by simp) g' (List.mem_cons_of_mem _ hg') e'.1 (List.mem_map_of_mem he')
        (by simp only at hfa; rw [← hfa]; exact List.mem_map_of_mem hf')
      exact hg (this ▸ List.mem_map_of_mem hg')

/-- **`collect_rids_nodup`**: under `RelInv` (and distinct type keys) the cascade collects each
relationship id once. -/
theorem collect_rids_nodup (p : P) (h : RelInv p) (hk : (p.relsByType.map (·.1)).Nodup) (id : Nat) :
    (Rids (collect p id)).Nodup := by
  have hgrp : ∀ g ∈ p.relsByType, grp p g.1 = g.2 := fun g hg => by
    simp [grp, fget_of_mem _ hk g.1 g.2 hg]
  apply rids_flatMap_nodup id _ hk
  · intro g hg; rw [← hgrp g hg]; exact h.2 g.1
  · intro g hg g' hg' r hr hr'
    rw [← hgrp g hg, ← (h.1 r g.1)] at hr
    rw [← hgrp g' hg', ← (h.1 r g'.1)] at hr'
    rw [hr] at hr'; exact Option.some.inj hr'

theorem foldl_rmStep_cancelledP : ∀ (L : List ((Nat × Nat × Nat) × Nat)) (p : P), (Rids L).Nodup →
    (L.foldl rmStep p).cancelledP = p.cancelledP + (L.map (relProps p)).sum
  | [], _, _ => by simp
  | (e, t) :: L, p, hn => by
    simp only [Rids, List.map_cons, List.nodup_cons] at hn
    simp only [List.foldl_cons, List.map_cons, List.sum_cons]
    rw [foldl_rmStep_cancelledP L _ hn.2]
    have : L.map (relProps (rmStep p (e, t))) = L.map (relProps p) := by
      apply List.map_congr_left; intro x hx
      have hne : x.1.1 ≠ e.1 := fun hq => hn.1 (hq ▸ List.mem_map_of_mem hx)
      simp [relProps, rmStep, hne]
    rw [this]; simp only [rmStep, relProps]; omega

/-- **`removeRels_cancelledP`** (#3030): `remove_pending_relationships_for_node` adds the
`count_set` of every cascaded edge's staged attrs to `cancelled_properties`, once per edge. -/
theorem removeRels_cancelledP (p : P) (h : RelInv p) (hk : (p.relsByType.map (·.1)).Nodup) (id : Nat) :
    (removeRels p id).cancelledP = p.cancelledP + ((collect p id).map (relProps p)).sum :=
  foldl_rmStep_cancelledP _ _ (collect_rids_nodup p h hk id)

/-- A successful graph-side cascade is the pure one on `Pending`. -/
theorem foldlM_rmStepG_pure {Gr : Type} (cancelR : Gr → Nat → Option Gr) (regT : Gr → Nat → Gr) :
    ∀ (L : List ((Nat × Nat × Nat) × Nat)) (p : P) (g : Gr) (p' : P) (g' : Gr),
    L.foldlM (rmStepG cancelR regT) (p, g) = some (p', g') → p' = L.foldl rmStep p
  | [], p, g, p', g', h => by
    simp only [List.foldlM_nil, pure, Option.some.injEq, Prod.mk.injEq] at h; exact h.1.symm
  | x :: L, p, g, p', g', h => by
    simp only [List.foldlM_cons, bind, rmStepG] at h
    cases hx : cancelR g x.1.1 with
    | none => simp [hx] at h
    | some g1 =>
      simp only [hx, Option.map_some, Option.bind_some] at h
      simpa using foldlM_rmStepG_pure cancelR regT L _ _ _ _ h

/-- **`deletePendingNode_cancelledP`** (#3030): a successful unwind adds to `cancelled_properties`
exactly the `count_set` of the node's staged attrs plus that of every cascaded edge's
(`relProps`), i.e. what `import_attrs` would have counted for them (`countSet_eq_import`), and
records the node as cancelled — so commit, which adds `cancelled_properties` to `properties_set`
(`PC.commit_spec`), counts the properties of an entity created and deleted in one segment. -/
theorem deletePendingNode_cancelledP {Gr : Type} (cancelN cancelR : Gr → Nat → Option Gr)
    (regT : Gr → Nat → Gr) (p : P) (hinv : RelInv p) (hk : (p.relsByType.map (·.1)).Nodup)
    (g : Gr) (id : Nat) (p' : P) (g' : Gr)
    (h : deletePendingNode cancelN cancelR regT p g id = some (p', g')) :
    p'.cancelledP = p.cancelledP + countSet (nodeAttrs p id) + ((collect p id).map (relProps p)).sum ∧
      id ∈ p'.cancelledN := by
  unfold deletePendingNode at h
  split at h
  · cases h
  · rename_i p3 g3 hrm
    cases hcn : cancelN g3 id with
    | none => simp [hcn] at h
    | some g4 =>
      simp only [hcn, Option.map_some, Option.some.injEq, Prod.mk.injEq] at h
      obtain ⟨rfl, rfl⟩ := h
      have hp3 := foldlM_rmStepG_pure cancelR regT _ _ _ _ _ hrm
      have hq : (takeAttrs (dropNode p id) id).relsByType = p.relsByType ∧
          (takeAttrs (dropNode p id) id).newR = p.newR ∧
          (takeAttrs (dropNode p id) id).cancelledP = p.cancelledP + countSet (nodeAttrs p id) := by
        unfold takeAttrs dropNode nodeAttrs; split <;> rename_i hn <;> simp [hn]
      have hcol : collect (takeAttrs (dropNode p id) id) id = collect p id := collect_congr _ _ hq.1 id
      refine ⟨?_, by simp [mem_sins]⟩
      simp only
      rw [hp3, hcol, foldl_rmStep_cancelledP _ _ (collect_rids_nodup p hinv hk id), hq.2.2]
      have : (collect p id).map (relProps (takeAttrs (dropNode p id) id)) = (collect p id).map (relProps p) := by
        apply List.map_congr_left; intro x _; simp [relProps, hq.2.1]
      rw [this]

/-- `CREATE (a {v: 1, w: 2}) DELETE a` (node 0, nothing cascaded): `cancelled_properties` is 2. -/
theorem cancelledP_example : (takeAttrs (dropNode { P0 with created := [0], newN := [(0, [(0, .inl 1), (1, .inl 2)])] } 0) 0).cancelledP = 2 := rfl

end PendingCommit.PS
