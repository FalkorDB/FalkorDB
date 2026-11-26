/-
# `Nodes created` / `Nodes deleted` statistics vs. C

A node created and deleted in the same segment is unwound out of `created_nodes` by
`delete_pending_node` (pending.rs) into `cancelled_nodes`; since #3030 `Pending::commit` adds
`cancelled_nodes.len()` to both `nodes_created` and `nodes_deleted` (and likewise
`cancelled_relationships.len()` to the relationship counters, `cancelled_properties` to
`properties_set`: `PC.commit_spec`, `PC.commit_cancelled_props`), on top of
`created_nodes.len()` / `deleted_nodes.len()`. The C engine counts every create and every delete.

`rust_eq_c` proves the two agree for every well-formed segment; `cancel_counted` is
`CREATE (a) DELETE a` (Rust and C: `Nodes created: 1, Nodes deleted: 1`).
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
`issued` set `Pending` lends it (`created_nodes`, create.rs) —
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

/-- The counters `Pending::commit` reports for the segment (`PC.commit_spec`): the cancelled
nodes once as created and once as deleted, plus `created_nodes` / `deleted_nodes`. -/
def rustCreated (p : P) : Nat := p.cancelled.length + p.created.length
def rustDeleted (p : P) : Nat := p.cancelled.length + p.deleted.length

/-- **`rust_eq_c`** (#3030): over any well-formed segment, Rust's `Nodes created` and
`Nodes deleted` are C's. -/
theorem rust_eq_c (ops : List Op) (h : RunWF ⟨[], [], [], 0, 0⟩ ops) :
    let r := run ⟨[], [], [], 0, 0⟩ ops
    rustCreated r = r.cCreated ∧ rustDeleted r = r.cDeleted := by
  obtain ⟨h1, h2⟩ := run_inv ops _ ⟨rfl, rfl⟩ h
  exact ⟨by simp only [rustCreated]; omega, by simp only [rustDeleted]; omega⟩

/-- `CREATE (a) DELETE a`: Rust and C both report 1/1. -/
theorem cancel_counted :
    let r := run ⟨[], [], [], 0, 0⟩ [.create 0, .delete 0]
    (rustCreated r, rustDeleted r) = (1, 1) ∧ (r.cCreated, r.cDeleted) = (1, 1) := by
  decide

end PendingCommit.Stats
