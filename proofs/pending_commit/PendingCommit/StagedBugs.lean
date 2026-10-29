import PendingCommit.PendingCancel
import PendingCommit.Labels
/-
# Staged state that outlives what it describes (#3447, #3448, #3453)

The existing theorems prove the query-time views (`overlay_eq_ref`,
`pendingWithLabels_spec`, `deletePendingNode_spec`). What they never state is
that the staged maps only name entities that exist when the batch is
committed or replayed. Line numbers are b84473a7f.

* **#3448** `remove_node_labels` (pending.rs:697-709) pushes every label into
  `remove_labels`, also one the node only had staged (a `CREATE`d node, or a
  `SET n:L` earlier in the query); ops/remove.rs filters by the overlay, not the
  committed matrix (`Labels.removeOp`). `delete_pending_node` (:788-821) clears
  `set_labels` but not `remove_labels`. Properties: `RemLCommitted` (every staged
  removal is of a committed label) and `RemLLive` (no created or cancelled node
  has a staged removal). Refuted: `bug3448_remove_of_staged_label`,
  `bug3448_cancelled_node_keeps_remL` (via the general
  `deletePendingNode_keeps_remL`).
* **#3453** `deleted_node` (:772-777) leaves `set_labels`, and
  `get_pending_nodes_with_labels` (:1040-1055) does not filter deleted ids.
  Property `PendingWithLabelsLive`. Refuted: `bug3453_pendingWithLabels_yields_deleted`.
  (The composed IncludePending property is in proofs/ops_aggregate,
  `IncludePendingExact.lean`.)
* **#3447** `remove_pending_relationships_for_node` (:839-896) drops a type's
  `created_rels_by_type` group once it is empty, so `Pending::commit`
  (:1239-1253, `create_relationships_bulk` → `get_type_id_mut`) never registers a
  type whose every edge was cascaded away; `CreateOp::resolve_pattern`
  (ops/create.rs:353-390) registers labels eagerly but not types. Property
  `CancelledTypeRegistered`. Refuted: `bug3447_cascade_drops_type_key`; the
  emitter consequence (an id gap on the replica) is proofs/effects_emit_apply
  `CancelGap.lean`.
-/
namespace PendingCommit.PS
open PA

/-! ## #3448: `remove_labels` -/

/-- **Property** (per node, `Labels.St`): every staged removal is of a label the
committed node carries. -/
def RemLCommitted (C : Nat → Bool) (s : Labels.St) : Prop := ∀ l, s.remL.contains l = true → C l = true

/-- **#3448** `CREATE (n:Foo) REMOVE n:Foo` (the node is new: no committed
label) and `MATCH (n:A) SET n:B REMOVE n:B` (`B` not committed): the overlay says
the node has the label, so REMOVE stages it in `remove_labels`, which commit
counts as `Labels removed` and the emitter ships as `RemoveLabels`. -/
theorem bug3448_remove_of_staged_label :
    (Labels.run (fun _ => false) [.add [0], .rem [0]]).remL = [0] ∧
    ¬ RemLCommitted (fun _ => false) (Labels.run (fun _ => false) [.add [0], .rem [0]]) := by
  refine ⟨by decide, fun h => ?_⟩
  have := h 0 (by decide)
  cases this

/-- **Property** (whole `Pending`): no created or cancelled node has a staged
label removal — such a node is not live on a replica before its `CREATE_NODE`. -/
def RemLLive (p : P) : Prop := ∀ n, n ∈ p.created ∨ n ∈ p.cancelledN → fget p.remL n = none

theorem foldl_rmStep_remL : ∀ (L : List ((Nat × Nat × Nat) × Nat)) (p : P), (L.foldl rmStep p).remL = p.remL
  | [], _ => rfl
  | _ :: L, p => by simp only [List.foldl_cons]; exact foldl_rmStep_remL L _

/-- **`deletePendingNode_keeps_remL`**: whatever the graph's cancels do, a
successful `delete_pending_node` leaves `remove_labels` exactly as it was. -/
theorem deletePendingNode_keeps_remL {Gr : Type} (cancelN cancelR : Gr → Nat → Option Gr) (p : P) (g : Gr)
    (id : Nat) (p' : P) (g' : Gr) (h : deletePendingNode cancelN cancelR p g id = some (p', g')) :
    p'.remL = p.remL := by
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
      unfold removeRelsG at hrm
      have hc : ∀ (L : List ((Nat × Nat × Nat) × Nat)) (q : P) (gq : Gr) (q' : P) (gq' : Gr),
          L.foldlM (rmStepG cancelR) (q, gq) = some (q', gq') → q'.remL = q.remL := by
        intro L
        induction L with
        | nil => intro q gq q' gq' h; simp only [List.foldlM_nil, pure, Option.some.injEq, Prod.mk.injEq] at h
                 rw [← h.1]
        | cons x L ih =>
          intro q gq q' gq' h
          simp only [List.foldlM_cons, bind, rmStepG] at h
          cases hx : cancelR gq x.1.1 with
          | none => simp [hx] at h
          | some g1 =>
            simp only [hx, Option.map_some, Option.bind_some] at h
            rw [ih _ _ _ _ h]; simp [rmStep]
      rw [hc _ _ _ _ _ hrm]
      split <;> rfl

/-- `CREATE (n:Foo) REMOVE n:Foo` staged for node 0: created, `Foo` (id 0) in
`remove_labels` (`bug3448_remove_of_staged_label`), no `set_labels`. -/
def pCreateRemove : P := ⟨[0], [], [], [], [], [], [], [], [], [], [], [], [], [(0, [0])]⟩

/-- **#3448** `CREATE (n:Foo) REMOVE n:Foo DELETE n`: after `delete_pending_node`
the node is cancelled and still has its `remove_labels` entry, whatever the graph
does (live: the payload is `AddLabel Foo, RemoveLabels [0], CreateNode [0],
DeleteNode [0]` and the replica refuses node 0 as not live). -/
theorem bug3448_cancelled_node_keeps_remL {Gr : Type} (cancelN cancelR : Gr → Nat → Option Gr) (g : Gr)
    (p' : P) (g' : Gr) (h : deletePendingNode cancelN cancelR pCreateRemove g 0 = some (p', g')) :
    0 ∈ p'.cancelledN ∧ fget p'.remL 0 = some [0] ∧ ¬ RemLLive p' := by
  have hr := deletePendingNode_keeps_remL cancelN cancelR pCreateRemove g 0 p' g' h
  have hc : 0 ∈ p'.cancelledN := by
    unfold deletePendingNode at h
    simp only at h
    split at h
    · cases h
    · rename_i p3 g3 _
      cases hcn : cancelN g3 0 with
      | none => simp [hcn] at h
      | some g4 =>
        simp only [hcn, Option.map_some, Option.some.injEq, Prod.mk.injEq] at h
        rw [← h.1]; simp [mem_sins]
  have hf : fget p'.remL 0 = some [0] := by rw [hr]; rfl
  exact ⟨hc, hf, fun hl => by rw [hl 0 (.inr hc)] at hf; cases hf⟩

/-- The unwind does succeed when the graph's cancels do (a trivial graph). -/
theorem bug3448_cancelled_node_keeps_remL_reached :
    ∃ p' g', deletePendingNode (fun (g : Unit) _ => some g) (fun g _ => some g) pCreateRemove () 0 = some (p', g') :=
  ⟨_, _, rfl⟩

/-! ## #3453: `get_pending_nodes_with_labels` -/

/-- **Property**: Phase 2's node list holds no pending-deleted node. -/
def PendingWithLabelsLive (p : P) (ls : List Nat) : Prop := ∀ n ∈ pendingWithLabels p ls, n ∉ p.deletedN

/-- `MATCH (n:A) SET n:B` on committed node 0 (`B` = 1 staged). -/
def pSetB : P := ⟨[], [], [], [], [], [], [], [], [], [], [], [], [(0, [1])], []⟩

/-- **#3453 (1)** `MATCH (n:A) SET n:B DELETE n MERGE (m:B)`: `deleted_node`
leaves `set_labels`, so the deleted node is a pending `:B` node. -/
theorem bug3453_pendingWithLabels_yields_deleted :
    pendingWithLabels (deletedNode pSetB 0) [1] = [0] ∧ 0 ∈ (deletedNode pSetB 0).deletedN ∧
    ¬ PendingWithLabelsLive (deletedNode pSetB 0) [1] := by
  refine ⟨rfl, by simp [deletedNode, mem_sins], fun h => h 0 (by decide) (by simp [deletedNode, mem_sins])⟩

/-! ## #3447: a cascaded edge's type -/

/-- **Property**: every cancelled relationship's type is registered after
commit — already in the graph (`reg`) or committed by a surviving edge of the
same type (a key of `created_rels_by_type`). -/
def CancelledTypeRegistered (reg : Nat → Bool) (p : P) : Prop :=
  ∀ c ∈ p.cancelledR, reg c.2.1 = true ∨ (fget p.relsByType c.2.1).isSome

/-- **#3447** `CREATE (a)-[r:NEW_TYPE]->(b) DETACH DELETE a`: edge 0 of the new
type 7 is cascaded away with `a`; its group is dropped, so commit registers no
type 7 although `cancelled_relationships` names it. -/
theorem bug3447_cascade_drops_type_key :
    (removeRels (createdRel P0 0 0 1 7) 0).cancelledR = [(0, 7, 0, 1)] ∧
    (removeRels (createdRel P0 0 0 1 7) 0).relsByType = [] ∧
    ¬ CancelledTypeRegistered (fun _ => false) (removeRels (createdRel P0 0 0 1 7) 0) := by
  refine ⟨by decide, by decide, fun h => ?_⟩
  have := h (0, 7, 0, 1) (by decide)
  simp at this
  revert this; decide

end PendingCommit.PS
