/-
# `Nodes created` / `Nodes deleted` statistics vs. C

`Pending::commit` (pending.rs:1105, :1190) reports
`nodes_created += created_nodes.len()` and `nodes_deleted += deleted_nodes.len()`.
A node created and deleted in the same segment is unwound out of
`created_nodes` by `delete_pending_node` (pending.rs:621) into
`cancelled_nodes`, which no statistic reads, so it is reported as neither
created nor deleted. The C engine counts every create and every delete.

`c_eq_rust_plus_cancelled` proves the discrepancy is *exactly* the cancelled
set, for every well-formed segment: adding `cancelled_nodes.len()` to both
counters is the whole fix. `cancel_example` is the Lean witness of
`CREATE (a) DELETE a` (Rust: nothing; C: `Nodes created: 1, Nodes deleted: 1`).
The same holds for relationships cascaded away with a cancelled node
(`cancelled_relationships`), and for the properties/labels those entities
carried — confirmed live, not modelled separately.
-/
namespace PendingCommit.Stats

structure P where
  created : List Nat
  deleted : List Nat
  cancelled : List Nat
  /-- C's counters, accumulated per operation -/
  cCreated : Nat
  cDeleted : Nat

inductive Op where
  | create (id : Nat)
  | delete (id : Nat)

/-- `Pending::created_nodes` / `delete_nodes_bulk` + `delete_entity` (ops/delete.rs):
already pending-deleted or already cancelled (its id is back in the graph's
bin, so `is_node_deleted`) ⇒ skip; pending-created ⇒ cancel; else stage. -/
def step (p : P) : Op → P
  | .create id => { p with created := id :: p.created, cCreated := p.cCreated + 1 }
  | .delete id =>
      if id ∈ p.deleted ∨ id ∈ p.cancelled then p
      else if id ∈ p.created then
        { p with created := p.created.erase id, cancelled := id :: p.cancelled,
                 cDeleted := p.cDeleted + 1 }
      else { p with deleted := id :: p.deleted, cDeleted := p.cDeleted + 1 }

/-- A CREATE never reuses an id this segment already holds: `IdSpace::reserve`
excludes its `taken` set (every id the batch created or cancelled) and the
`issued` set `Pending` lends it (`created_nodes`, create.rs:158) —
proofs/id_space. -/
def WF (p : P) : Op → Prop
  | .create id => id ∉ p.created ∧ id ∉ p.cancelled ∧ id ∉ p.deleted
  | .delete _ => True

def run : P → List Op → P
  | p, [] => p
  | p, op :: ops => run (step p op) ops

def RunWF : P → List Op → Prop
  | _, [] => True
  | p, op :: ops => WF p op ∧ RunWF (step p op) ops

def Inv (p : P) : Prop :=
  p.cCreated = p.created.length + p.cancelled.length ∧
  p.cDeleted = p.deleted.length + p.cancelled.length

theorem step_inv (p : P) (op : Op) (h : Inv p) (_hw : WF p op) : Inv (step p op) := by
  obtain ⟨h1, h2⟩ := h
  cases op with
  | create id => exact ⟨by simp [step, h1]; omega, by simp [step, h2]⟩
  | delete id =>
    simp only [step]
    split
    · exact ⟨h1, h2⟩
    · split
      · rename_i _ hc
        have := List.length_erase_of_mem hc
        have hpos : 0 < p.created.length := List.length_pos_of_mem hc
        exact ⟨by simp; omega, by simp; omega⟩
      · exact ⟨by simp [h1], by simp; omega⟩

theorem run_inv : ∀ (ops : List Op) (p : P), Inv p → RunWF p ops → Inv (run p ops)
  | [], _, hp, _ => hp
  | op :: ops, p, hp, hw => run_inv ops (step p op) (step_inv p op hp hw.1) hw.2

/-- **`c_eq_rust_plus_cancelled`**: over any well-formed segment, C's
`Nodes created` is Rust's plus the cancelled count, and likewise for
`Nodes deleted`. -/
theorem c_eq_rust_plus_cancelled (ops : List Op) (h : RunWF ⟨[], [], [], 0, 0⟩ ops) :
    let r := run ⟨[], [], [], 0, 0⟩ ops
    r.cCreated = r.created.length + r.cancelled.length ∧
    r.cDeleted = r.deleted.length + r.cancelled.length := by
  exact run_inv ops _ ⟨rfl, rfl⟩ h

/-- `CREATE (a) DELETE a`: Rust reports 0/0, C reports 1/1. -/
theorem cancel_example :
    let r := run ⟨[], [], [], 0, 0⟩ [.create 0, .delete 0]
    (r.created.length, r.deleted.length) = (0, 0) ∧ (r.cCreated, r.cDeleted) = (1, 1) := by
  decide

end PendingCommit.Stats
