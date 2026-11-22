import PendingCommit.PendingCancel
import PendingCommit.Labels
/-
# Staged state that outlives what it describes (#3447, #3448, #3453)

The existing theorems prove the query-time views (`overlay_eq_ref`,
`pendingWithLabels_spec`, `deletePendingNode_spec`). What they never state is
that the staged maps only name entities that exist when the batch is
committed or replayed. Line numbers are current main (5c7af54f2).

* **#3448** (*fixed by #3459*) `remove_node_labels` (pending.rs:741-768) used to push
  every label into `remove_labels`, also one the node only had staged (a `CREATE`d
  node, or a `SET n:L` earlier in the query), and `delete_pending_node` cleared
  `set_labels` but not `remove_labels`. Now a removal that only cancels a staged
  add bumps `staged_labels_removed` instead (`Labels.removeOneC`), and
  `delete_pending_node` (:897) clears `remove_labels`. Properties: `RemLCommitted`
  (every staged removal is of a committed label), now proven for every run of
  REMOVE/SET clauses (`fixed3448_run_remL_committed`), and `RemLLive` (no created or
  cancelled node has a staged removal), now kept by the unwind
  (`deletePendingNode_clears_remL`, `fixed3448_cancelled_node_clears_remL`).
  The two repros are `fixed3448_remove_of_staged_label` (before #3459:
  `remL = [0]`) and `fixed3448_cancelled_node_clears_remL` (before: `remL = [(0, [0])]`).
* **#3453** (*fixed by #3467, a477d9aa3*) `deleted_node` leaves `set_labels`, and
  `get_pending_nodes_with_labels` used to ignore deletion and committed labels. Property
  `PendingWithLabelsLive`, now proven (`pendingWithLabels_live`); `fixed3453_*` are the three repros.
  (The composed IncludePending property is in proofs/ops_aggregate,
  `IncludePendingExact.lean`.)
* **#3447** `remove_pending_relationships_for_node` (:902-960) drops a type's
  `created_rels_by_type` group once it is empty, so `Pending::commit`
  (:1348-1361, `create_relationships_bulk` → `get_type_id_mut`) never registers a
  type whose every edge was cascaded away; `CreateOp::resolve_pattern`
  (ops/create.rs:353-400) registers labels eagerly but not types. Property
  `CancelledTypeRegistered`. Refuted: `bug3447_cascade_drops_type_key`; the
  emitter consequence (an id gap on the replica) is proofs/effects_emit_apply
  `CancelGap.lean`. **Fixed by #3461** (f203972c5) at commit, not here: the group
  is still dropped (the theorem still holds), but `Pending::commit` now registers
  every `cancelled_relationships` type (`PC.stCancelTypes_registers`).
-/
namespace PendingCommit.PS
open PA

/-! ## #3448: `remove_labels` -/

/-- **Property** (per node, `Labels.St`): every staged removal is of a label the
committed node carries. -/
def RemLCommitted (C : Nat → Bool) (s : Labels.St) : Prop := ∀ l, s.remL.contains l = true → C l = true

/-- `remove_node_labels` before #3459 (pending.rs:729-746 on 3f4ea8f40): every label left
`set_labels` and was pushed onto `remove_labels`. History for `pre3459_*`. -/
def removeLabelsPre3459 (s : Labels.St) (ls : List Nat) : Labels.St :=
  ls.foldl (fun s l => { setL := s.setL.filter (· != l), remL := s.remL ++ [l] }) s

/-- History (#3448, fixed by #3459): `CREATE (n:Foo) REMOVE n:Foo` staged `Foo` in `remove_labels`
although no committed node carried it. -/
theorem pre3459_bug3448_remove_of_staged_label :
    (removeLabelsPre3459 ⟨[0], []⟩ [0]).remL = [0] ∧
    ¬ RemLCommitted (fun _ => false) (removeLabelsPre3459 ⟨[0], []⟩ [0]) := by
  refine ⟨by decide, fun h => ?_⟩
  have := h 0 (by decide)
  cases this

theorem step_remL_committed (C : Nat → Bool) (s : Labels.St) (op : Labels.Op)
    (hnd : ∀ ls, op = .rem ls → ls.Nodup) (h : RemLCommitted C s) :
    RemLCommitted C (Labels.step C s op) := by
  cases op with
  | add ls =>
    intro l hl
    simp only [Labels.step, Labels.stage, List.contains_iff_mem, List.mem_filter] at hl
    exact h l (by simpa using hl.1)
  | rem ls => exact Labels.removeOp_committed C s ls (hnd ls rfl) h

/-- **#3448, fixed by #3459**: after any sequence of `SET n:…` / `REMOVE n:…` clauses (each
REMOVE's labels distinct, an `OrderSet` in ops/remove.rs), every label staged in `remove_labels`
is one the committed node carries. -/
theorem fixed3448_run_remL_committed (C : Nat → Bool) (ops : List Labels.Op)
    (hnd : ∀ ls, Labels.Op.rem ls ∈ ops → ls.Nodup) : RemLCommitted C (Labels.run C ops) := by
  unfold Labels.run
  suffices ∀ s, RemLCommitted C s → RemLCommitted C (ops.foldl (Labels.step C) s) from
    this _ (by intro l hl; simp at hl)
  induction ops with
  | nil => intro s h; exact h
  | cons op ops ih =>
    intro s h
    exact ih (fun ls hm => hnd ls (List.mem_cons_of_mem _ hm)) _
      (step_remL_committed C s op (fun ls he => hnd ls (by subst he; exact List.mem_cons_self ..)) h)

/-- **#3448 repro, fixed by #3459**: `CREATE (n:Foo) REMOVE n:Foo` (the node is new: no committed
label). The removal only cancels the staged add: nothing reaches `remove_labels` (before #3459:
`[0]`, shipped as `RemoveLabels`), and `staged_labels_removed` counts it, so `Labels removed` is
still 1. -/
theorem fixed3448_remove_of_staged_label :
    (Labels.run (fun _ => false) [.add [0], .rem [0]]).remL = [] ∧
    (Labels.run (fun _ => false) [.add [0], .rem [0]]).setL = [] ∧
    (Labels.removeLabelsC (fun _ => false) ⟨[0], []⟩ 0 [0]).2 = 1 ∧
    RemLCommitted (fun _ => false) (Labels.run (fun _ => false) [.add [0], .rem [0]]) :=
  ⟨by decide, by decide, by decide, fixed3448_run_remL_committed _ _ (fun ls hm => by
      simp only [List.mem_cons, List.not_mem_nil, or_false] at hm
      rcases hm with h | h <;> cases h; exact List.nodup_cons.2 ⟨by simp, List.nodup_nil⟩)⟩

/-- `MATCH (n:A) SET n:A REMOVE n:A` (`A` committed): the add was staged but the committed node
carries `A`, so the removal is still staged and committed. -/
theorem remove_of_committed_label_staged :
    (Labels.run (fun l => l == 0) [.add [0], .rem [0]]).remL = [0] := by decide

/-- **Property** (whole `Pending`): no created or cancelled node has a staged
label removal — such a node is not live on a replica before its `CREATE_NODE`. -/
def RemLLive (p : P) : Prop := ∀ n, n ∈ p.created ∨ n ∈ p.cancelledN → fget p.remL n = none

theorem foldl_rmStep_remL : ∀ (L : List ((Nat × Nat × Nat) × Nat)) (p : P), (L.foldl rmStep p).remL = p.remL
  | [], _ => rfl
  | _ :: L, p => by simp only [List.foldl_cons]; exact foldl_rmStep_remL L _

/-- **`deletePendingNode_clears_remL`** (#3459): whatever the graph's cancels do, a
successful `delete_pending_node` drops the node's `remove_labels` entry and leaves every
other node's (before #3459 it left `remove_labels` exactly as it was). -/
theorem deletePendingNode_clears_remL {Gr : Type} (cancelN cancelR : Gr → Nat → Option Gr) (p : P) (g : Gr)
    (id : Nat) (p' : P) (g' : Gr) (h : deletePendingNode cancelN cancelR p g id = some (p', g')) :
    p'.remL = frem p.remL id := by
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

/-- `CREATE (n:Foo) REMOVE n:Foo` as staged **before #3459** for node 0: created, `Foo` (id 0) in
`remove_labels`, no `set_labels`. -/
def pCreateRemove : P := ⟨[0], [], [], [], [], [], [], [], [], [], [], [], [], [(0, [0])], 0⟩

/-- **#3448 repro, fixed by #3459**: `CREATE (n:Foo) REMOVE n:Foo DELETE n`. Even from the state the
old `remove_node_labels` left (`pCreateRemove`), `delete_pending_node` now drops the node's
`remove_labels` entry, so the cancelled node satisfies `RemLLive` and no `RemoveLabels [0]` precedes
its `CreateNode [0]` (before #3459: `fget p'.remL 0 = some [0]`, `¬ RemLLive p'`, and the replica
refused node 0 as not live). -/
theorem fixed3448_cancelled_node_clears_remL {Gr : Type} (cancelN cancelR : Gr → Nat → Option Gr) (g : Gr)
    (p' : P) (g' : Gr) (h : deletePendingNode cancelN cancelR pCreateRemove g 0 = some (p', g')) :
    0 ∈ p'.cancelledN ∧ fget p'.remL 0 = none ∧ RemLLive p' := by
  have hr := deletePendingNode_clears_remL cancelN cancelR pCreateRemove g 0 p' g' h
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
  have he : p'.remL = [] := by rw [hr]; decide
  refine ⟨hc, by rw [he]; rfl, fun n _ => by rw [he]; rfl⟩

/-- The unwind does succeed when the graph's cancels do (a trivial graph). -/
theorem fixed3448_cancelled_node_clears_remL_reached :
    ∃ p' g', deletePendingNode (fun (g : Unit) _ => some g) (fun g _ => some g) pCreateRemove () 0 = some (p', g') :=
  ⟨_, _, rfl⟩

/-! ## #3453: `get_pending_nodes_with_labels` (fixed by #3467, a477d9aa3) -/

/-- **Property**: Phase 2's node list holds no pending-deleted node. -/
def PendingWithLabelsLive (p : P) (C : Nat → Nat → Bool) (ls : List Nat) : Prop :=
  ∀ n ∈ pendingWithLabels p C ls, n ∉ p.deletedN

/-- **`pendingWithLabels_live`**: for a non-empty pattern the new `get_pending_nodes_with_labels` skips every
pending-deleted node, whatever `set_labels` still says about it. -/
theorem pendingWithLabels_live (p : P) (C : Nat → Nat → Bool) (hk : (p.setL.map (·.1)).Nodup)
    (ls : List Nat) (hls : ls ≠ []) : PendingWithLabelsLive p C ls := by
  intro n hn
  rw [pendingWithLabels_spec p C hk] at hn
  rcases hn with ⟨h, _⟩ | ⟨_, hd, _⟩
  · exact absurd h hls
  · exact hd

/-- `MATCH (n:A) SET n:B` on committed node 0 (`B` = 1 staged). -/
def pSetB : P := ⟨[], [], [], [], [], [], [], [], [], [], [], [], [(0, [1])], [], 0⟩

/-- **#3453 (1), fixed**: `MATCH (n:A) SET n:B DELETE n MERGE (m:B)`: the deleted node is not a pending `:B`
node any more (before the fix the list was `[0]`). -/
theorem fixed3453_deleted_not_pending :
    pendingWithLabels (deletedNode pSetB 0) (fun _ l => l == 0) [1] = [] := by decide

/-- **#3453 (2), fixed**: `MATCH (n:A) SET n:A`: the node already has `A` committed, so it is left to the
scan, not listed again (before the fix: a second row). -/
theorem fixed3453_resets_committed_label :
    pendingWithLabels ⟨[], [], [], [], [], [], [], [], [], [], [], [], [(0, [0])], [], 0⟩
      (fun _ l => l == 0) [0] = [] := by decide

/-- **#3453 (3), fixed**: `MATCH (n:A) SET n:B MERGE (m:A:B)`: node 0 has `A` committed and `B` staged; it
is listed here (and skipped by the scan), so MERGE matches it (before the fix: nothing, a duplicate). -/
theorem fixed3453_split_multilabel :
    pendingWithLabels pSetB (fun _ l => l == 0) [0, 1] = [0] := by decide

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
