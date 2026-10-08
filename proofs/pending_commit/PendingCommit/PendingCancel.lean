import PendingCommit.PendingDocs
/-
# Unwinding a pending node, paired with the graph's id space (#2846)

Since #2846 `delete_pending_node` (pending.rs:659) and
`remove_pending_relationships_for_node` (:710) take `g: &mut Graph` and hand
every unwound id back through `Graph::cancel_node_id` /
`cancel_relationship_id` (graph.rs:1427/:1414) themselves, instead of
leaving `ops/delete.rs` to call `return_*_id` afterwards. Each call is `?`:
the first refusal aborts the unwind.

The graph is abstract here (`Gr`); what is needed of its cancel is the
hypothesis `CancelSpec` — after `cancel g id = some g'`, `id` is free in `g'`
and nothing free in `g` stops being free. That is `IdSpace::cancel`'s
`recycled.insert(id)` (id_space.rs:536) read through `is_free`; proofs/id_space
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

/-- Hypothesis on the graph's cancel (see header). -/
structure CancelSpec {Gr : Type} (free : Gr → List Nat) (cancel : Gr → Nat → Option Gr) : Prop where
  ok : ∀ g id g', cancel g id = some g' → id ∈ free g' ∧ ∀ x ∈ free g, x ∈ free g'

/-- `remove_pending_relationships_for_node` (:710): the second loop (:731-765),
each iteration the `Pending` removals (`rmStep`) then
`g.cancel_relationship_id(rel_id)?` (:755). -/
def rmStepG {Gr : Type} (cancelR : Gr → Nat → Option Gr) (pg : P × Gr)
    (x : (Nat × Nat × Nat) × Nat) : Option (P × Gr) :=
  (cancelR pg.2 x.1.1).map (fun g' => (rmStep pg.1 x, g'))

def removeRelsG {Gr : Type} (cancelR : Gr → Nat → Option Gr) (p : P) (g : Gr) (id : Nat) :
    Option (P × Gr) :=
  (collect p id).foldlM (rmStepG cancelR) (p, g)

theorem foldlM_rmStepG {Gr : Type} (free : Gr → List Nat) (cancelR : Gr → Nat → Option Gr)
    (hc : CancelSpec free cancelR) :
    ∀ (L : List ((Nat × Nat × Nat) × Nat)) (p : P) (g : Gr) (p' : P) (g' : Gr),
    L.foldlM (rmStepG cancelR) (p, g) = some (p', g') →
    p' = L.foldl rmStep p ∧ (∀ r ∈ Rids L, r ∈ free g') ∧ ∀ x ∈ free g, x ∈ free g'
  | [], p, g, p', g', h => by
    simp only [List.foldlM_nil, pure, Option.some.injEq, Prod.mk.injEq] at h
    obtain ⟨rfl, rfl⟩ := h
    exact ⟨rfl, by simp [Rids], fun _ hx => hx⟩
  | x :: L, p, g, p', g', h => by
    simp only [List.foldlM_cons, bind, rmStepG] at h
    cases hcg : cancelR g x.1.1 with
    | none => simp [hcg] at h
    | some g1 =>
      simp only [hcg, Option.map_some, Option.bind_some] at h
      obtain ⟨e1, e2, e3⟩ := foldlM_rmStepG free cancelR hc L _ _ _ _ h
      obtain ⟨c1, c2⟩ := hc.ok _ _ _ hcg
      refine ⟨by simpa using e1, fun r hr => ?_, fun y hy => e3 y (c2 y hy)⟩
      simp only [Rids, List.map_cons, List.mem_cons] at hr
      rcases hr with rfl | hr
      · exact e3 _ c1
      · exact e2 r hr

theorem foldl_rmStep_taken : ∀ (L : List ((Nat × Nat × Nat) × Nat)) (p : P),
    (L.foldl rmStep p).taken = p.taken.filter (· ∉ Rids L)
  | [], p => by simp only [List.foldl_nil]; exact (List.filter_eq_self.2 (by simp [Rids])).symm
  | (e, t) :: L, p => by
    simp only [List.foldl_cons, foldl_rmStep_taken L, rmStep, List.filter_filter]
    congr 1; funext x
    by_cases hx : x = e.1 <;> by_cases h2 : x ∈ Rids L <;> simp [Rids] at h2 <;> simp [Rids, hx, h2]

/-- **`removeRelsG_spec`**: success means the `Pending` side is the pure
`removeRels` and every cascaded relationship id is free in the graph. -/
theorem removeRelsG_spec {Gr : Type} (free : Gr → List Nat) (cancelR : Gr → Nat → Option Gr)
    (hc : CancelSpec free cancelR) (p : P) (g : Gr) (id : Nat) (p' : P) (g' : Gr)
    (h : removeRelsG cancelR p g id = some (p', g')) :
    p' = removeRels p id ∧ (∀ r ∈ Rids (collect p id), r ∈ free g' ∧ r ∉ p'.taken) ∧
    ∀ x ∈ free g, x ∈ free g' := by
  obtain ⟨e1, e2, e3⟩ := foldlM_rmStepG free cancelR hc _ p g p' g' h
  refine ⟨e1, fun r hr => ⟨e2 r hr, ?_⟩, e3⟩
  rw [e1]; rw [foldl_rmStep_taken]
  simp [hr]

/-- `delete_pending_node` (:659), its effect on `Pending` and the graph:
`created_nodes.remove(id)` (:676), `set_labels.remove` (:680), the staged
attrs taken from `new_nodes_attrs`, else `existing_nodes_attrs` (:683-687),
`remove_pending_relationships_for_node(id, g)?` (:690),
`cancelled_nodes.insert(id)` (:694), `g.cancel_node_id(id)?` (:696). -/
def deletePendingNode {Gr : Type} (cancelN cancelR : Gr → Nat → Option Gr) (p : P) (g : Gr)
    (id : Nat) : Option (P × Gr) :=
  let p1 : P := { p with created := p.created.filter (· != id), setL := frem p.setL id }
  let p2 : P := match fget p1.newN id with
    | some _ => { p1 with newN := frem p1.newN id }
    | none => { p1 with existN := frem p1.existN id }
  match removeRelsG cancelR p2 g id with
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
is recorded as cancelled, and its id and every cascaded relationship id are
free in the graph and no longer `taken` — the unwind is never half-done. -/
theorem deletePendingNode_spec {Gr : Type} (free : Gr → List Nat) (cancelN cancelR : Gr → Nat → Option Gr)
    (hn : CancelSpec free cancelN) (hr : CancelSpec free cancelR) (p : P) (g : Gr) (id : Nat)
    (p' : P) (g' : Gr) (h : deletePendingNode cancelN cancelR p g id = some (p', g')) :
    id ∉ p'.created ∧ id ∈ p'.cancelledN ∧ id ∈ free g' ∧
    (∀ r ∈ Rids (collect p id), r ∈ free g' ∧ r ∉ p'.taken) ∧ ∀ x ∈ free g, x ∈ free g' := by
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
      obtain ⟨e1, e2, e3⟩ := removeRelsG_spec free cancelR hr _ g id p3 g3 hrm
      obtain ⟨c1, c2⟩ := hn.ok _ _ _ hcn
      have hcol : ∀ q : P, q.relsByType = p.relsByType → collect q id = collect p id :=
        fun q hq => collect_congr q p hq id
      have hc2 : collect (match fget ({ p with created := p.created.filter (· != id), setL := frem p.setL id } : P).newN id with
          | some _ => ({ ({ p with created := p.created.filter (· != id), setL := frem p.setL id } : P) with
              newN := frem p.newN id } : P)
          | none => ({ ({ p with created := p.created.filter (· != id), setL := frem p.setL id } : P) with
              existN := frem p.existN id } : P)) id = collect p id := by
        apply hcol; split <;> rfl
      have hcr : (removeRels (match fget ({ p with created := p.created.filter (· != id), setL := frem p.setL id } : P).newN id with
          | some _ => ({ ({ p with created := p.created.filter (· != id), setL := frem p.setL id } : P) with
              newN := frem p.newN id } : P)
          | none => ({ ({ p with created := p.created.filter (· != id), setL := frem p.setL id } : P) with
              existN := frem p.existN id } : P)) id).created = p.created.filter (· != id) := by
        unfold removeRels; rw [(foldl_rmStep_created _ _).1]; split <;> rfl
      rw [hc2] at e2
      refine ⟨?_, by simp [mem_sins], c1, fun r hr' => ⟨c2 r (e2 r hr').1, (e2 r hr').2⟩,
        fun x hx => c2 x (e3 x hx)⟩
      simp only [e1, hcr, List.mem_filter]; simp

/-- **`deletePendingNode_refused`**: a refused node cancel fails the unwind. -/
theorem deletePendingNode_refused {Gr : Type} (cancelN cancelR : Gr → Nat → Option Gr) (p : P) (g : Gr)
    (id : Nat) (p3 : P) (g3 : Gr)
    (hrm : removeRelsG cancelR (match fget ({ p with created := p.created.filter (· != id), setL := frem p.setL id } : P).newN id with
          | some _ => ({ ({ p with created := p.created.filter (· != id), setL := frem p.setL id } : P) with
              newN := frem p.newN id } : P)
          | none => ({ ({ p with created := p.created.filter (· != id), setL := frem p.setL id } : P) with
              existN := frem p.existN id } : P)) g id = some (p3, g3))
    (hcn : cancelN g3 id = none) :
    deletePendingNode cancelN cancelR p g id = none := by
  unfold deletePendingNode; simp only; rw [hrm]; simp [hcn]

end PendingCommit.PS
