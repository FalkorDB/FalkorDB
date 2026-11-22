import PendingCommit.PendingDocs
/-
# Unwinding a pending node, paired with the graph's id space

`delete_pending_node` (pending.rs) and
`remove_pending_relationships_for_node` (:902) take `g: &mut Graph` and hand
every unwound id back through `Graph::cancel_node_id` /
`cancel_relationship_id` (graph.rs/:1562) themselves. Each call is `?`:
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

/-- `delete_pending_node` (:851), its effect on `Pending` and the graph:
`created_nodes.remove(id)` (:893), `set_labels.remove` (:896), `remove_labels.remove`
(:897, #3459), the staged attrs taken from `new_nodes_attrs`, else `existing_nodes_attrs`
(:900-904), `remove_pending_relationships_for_node(id, g)?` (:906),
`cancelled_nodes.insert(id)` (:910), `g.cancel_node_id(id)?` (:912). -/
def deletePendingNode {Gr : Type} (cancelN cancelR : Gr → Nat → Option Gr) (regT : Gr → Nat → Gr)
    (p : P) (g : Gr) (id : Nat) : Option (P × Gr) :=
  let p1 : P := { p with created := p.created.filter (· != id), setL := frem p.setL id, remL := frem p.remL id }
  let p2 : P := match fget p1.newN id with
    | some _ => { p1 with newN := frem p1.newN id }
    | none => { p1 with existN := frem p1.existN id }
  match removeRelsG cancelR regT p2 g id with
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
      have hc2 : collect (match fget ({ p with created := p.created.filter (· != id), setL := frem p.setL id, remL := frem p.remL id } : P).newN id with
          | some _ => ({ ({ p with created := p.created.filter (· != id), setL := frem p.setL id, remL := frem p.remL id } : P) with
              newN := frem p.newN id } : P)
          | none => ({ ({ p with created := p.created.filter (· != id), setL := frem p.setL id, remL := frem p.remL id } : P) with
              existN := frem p.existN id } : P)) id = collect p id := by
        apply hcol; split <;> rfl
      have hcr : (removeRels (match fget ({ p with created := p.created.filter (· != id), setL := frem p.setL id, remL := frem p.remL id } : P).newN id with
          | some _ => ({ ({ p with created := p.created.filter (· != id), setL := frem p.setL id, remL := frem p.remL id } : P) with
              newN := frem p.newN id } : P)
          | none => ({ ({ p with created := p.created.filter (· != id), setL := frem p.setL id, remL := frem p.remL id } : P) with
              existN := frem p.existN id } : P)) id).created = p.created.filter (· != id) := by
        unfold removeRels; rw [(foldl_rmStep_created _ _).1]; split <;> rfl
      rw [hc2] at e2 e5
      refine ⟨?_, by simp [mem_sins], c1, fun r hr' => ⟨c3 r (e2 r hr').1, (e2 r hr').2⟩,
        fun x hx => c2 x (e4 x hx), fun x hx => c3 x (e3 x hx), fun t ht => hty _ _ _ hcn t (e5 t ht)⟩
      simp only [e1, hcr, List.mem_filter]; simp

/-- **`deletePendingNode_refused`**: a refused node cancel fails the unwind. -/
theorem deletePendingNode_refused {Gr : Type} (cancelN cancelR : Gr → Nat → Option Gr) (regT : Gr → Nat → Gr)
    (p : P) (g : Gr)
    (id : Nat) (p3 : P) (g3 : Gr)
    (hrm : removeRelsG cancelR regT (match fget ({ p with created := p.created.filter (· != id), setL := frem p.setL id, remL := frem p.remL id } : P).newN id with
          | some _ => ({ ({ p with created := p.created.filter (· != id), setL := frem p.setL id, remL := frem p.remL id } : P) with
              newN := frem p.newN id } : P)
          | none => ({ ({ p with created := p.created.filter (· != id), setL := frem p.setL id, remL := frem p.remL id } : P) with
              existN := frem p.existN id } : P)) g id = some (p3, g3))
    (hcn : cancelN g3 id = none) :
    deletePendingNode cancelN cancelR regT p g id = none := by
  unfold deletePendingNode; simp only; rw [hrm]; simp [hcn]

end PendingCommit.PS
