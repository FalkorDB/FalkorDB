/-
# Staged label changes: `set_labels` / `remove_labels`

`graph/src/runtime/pending.rs`. Per node, `Pending` keeps the labels this query
added (`set_labels`) and removed (`remove_labels`). Maps are per node and nodes
never interact, so one node is modelled; `C` is the committed label set.

* `stage` — `stage_node_labels` (pending.rs:691): push the labels onto
  `set_labels`, drop them from `remove_labels`.
* `removeLabels` — `remove_node_labels` (pending.rs:729): per label, drop it
  from `set_labels`, push it onto `remove_labels`. The REMOVE operator
  (`ops/remove.rs:128-142`) first filters the labels to those the node
  currently has (`overlay`) — modelled as `removeOp`.
* `nodeHas` — `node_has_label` (pending.rs:755): removal first, then add, else
  `None` (ask the matrix).
* `stagedNodeLabels` — `Pending::staged_node_labels` (pending.rs:826, #3288): the two staged
  lists as they are, `(set_labels, remove_labels)`; `update_node_labels` (:806) is now
  written over it.
* `stagedLabelIds` — `Runtime::staged_label_ids` (ops/delete.rs:196, #3288): the committed
  ids, plus the staged adds, minus the staged removes. The deleted-node snapshot
  (`delete_nodes_bulk` :306, `delete_entity` :486) is built from it.
* `commitLabels` — `set_nodes_labels_bulk` then `remove_nodes_labels`
  (`Pending::commit`, pending.rs:1362-1378): `(C ∪ set) \ remove`.

Reference: a label's state is decided by the last clause that mentions it,
else the committed matrix (`refHas`).
-/
namespace PendingCommit.Labels

structure St where
  setL : List Nat
  remL : List Nat
  deriving Repr

def stage (s : St) (ls : List Nat) : St :=
  { setL := s.setL ++ ls, remL := s.remL.filter (fun l => !(ls.contains l)) }

/-- One iteration of the `for label in labels` loop in `remove_node_labels`. -/
def removeOne (s : St) (l : Nat) : St :=
  { setL := s.setL.filter (· != l), remL := s.remL ++ [l] }

def removeLabels (s : St) (ls : List Nat) : St := ls.foldl removeOne s

/-- What the query sees: `update_node_labels` over the committed set. -/
def overlay (C : Nat → Bool) (s : St) (l : Nat) : Bool :=
  if s.remL.contains l then false else if s.setL.contains l then true else C l

def nodeHas (s : St) (l : Nat) : Option Bool :=
  if s.remL.contains l then some false else if s.setL.contains l then some true else none

/-- `ops/remove.rs`: only labels the node currently carries are staged. -/
def removeOp (C : Nat → Bool) (s : St) (ls : List Nat) : St :=
  removeLabels s (ls.filter (overlay C s))

/-- `Pending::staged_node_labels`: the lists as staged. -/
def stagedNodeLabels (s : St) : List Nat × List Nat := (s.setL, s.remL)

/-- `Pending::update_node_labels` after #3288: insert every staged add, then remove every
staged removal, both read through `staged_node_labels`. -/
def updateNodeLabels (C : Nat → Bool) (s : St) (l : Nat) : Bool :=
  let (added, removed) := stagedNodeLabels s
  (C l || added.contains l) && !removed.contains l

/-- `Runtime::staged_label_ids` (ops/delete.rs:196): `g.get_node_label_ids(id)` collected into a
set, then `labels.extend(added)`, then `labels.remove(l)` for each removed label; the same
staged-label read as `update_node_labels`, so it shares its model. -/
def stagedLabelIds (C : Nat → Bool) (s : St) (l : Nat) : Bool :=
  updateNodeLabels C s l

/-- The committed label set after `Pending::commit`. -/
def commitLabels (C : Nat → Bool) (s : St) (l : Nat) : Bool :=
  !(s.remL.contains l) && (C l || s.setL.contains l)

inductive Op where
  | add (ls : List Nat)
  | rem (ls : List Nat)

def step (C : Nat → Bool) (s : St) : Op → St
  | .add ls => stage s ls
  | .rem ls => removeOp C s ls

def run (C : Nat → Bool) (ops : List Op) : St := ops.foldl (step C) ⟨[], []⟩

/-- Reference: the last clause that mentions `l` wins. -/
def refStep (b : Bool) (l : Nat) : Op → Bool
  | .add ls => if ls.contains l then true else b
  | .rem ls => if ls.contains l then false else b

def refHas (C : Nat → Bool) (ops : List Op) (l : Nat) : Bool := ops.foldl (refStep · l) (C l)

theorem contains_filter (p : Nat → Bool) (ls : List Nat) (l : Nat) :
    (ls.filter p).contains l = (ls.contains l && p l) := by
  induction ls with
  | nil => simp
  | cons x xs ih =>
    by_cases hx : l = x
    · subst hx; cases hp : p l <;> simp [List.filter_cons, hp, ih]
    · have : (x == l) = false := by simp; omega
      cases hp : p x <;> simp [List.filter_cons, hp, ih, hx] <;> cases p l <;> simp

def Disjoint (s : St) : Prop := ∀ l, s.remL.contains l = true → s.setL.contains l = false

/-! ## Proofs -/

theorem removeLabels_closed (s : St) (ls : List Nat) :
    (∀ l, (removeLabels s ls).remL.contains l = (s.remL.contains l || ls.contains l)) ∧
    (∀ l, (removeLabels s ls).setL.contains l = (s.setL.contains l && !ls.contains l)) := by
  induction ls generalizing s with
  | nil => simp [removeLabels]
  | cons x xs ih =>
    have := ih (removeOne s x)
    simp only [removeLabels, List.foldl_cons] at this ⊢
    refine ⟨fun l => ?_, fun l => ?_⟩
    · rw [this.1]; simp [removeOne]; by_cases h : l = x <;> simp [h, Bool.or_comm, Bool.or_assoc]
    · rw [this.2]; simp [removeOne, List.contains_iff_mem, List.mem_filter]
      by_cases h : l = x <;> simp [h]

theorem step_disjoint (C : Nat → Bool) (s : St) (op : Op) (h : Disjoint s) :
    Disjoint (step C s op) := by
  intro l hl
  cases op with
  | add ls =>
    simp [step, stage, List.contains_iff_mem, List.mem_filter] at hl ⊢
    exact ⟨by have := h l; simp [List.contains_iff_mem] at this; exact this hl.1, hl.2⟩
  | rem ls =>
    simp only [step, removeOp] at hl ⊢
    have ⟨h1, h2⟩ := removeLabels_closed s (ls.filter (overlay C s))
    rw [h1] at hl; rw [h2]
    cases hr : s.remL.contains l
    · rw [hr, Bool.false_or] at hl; rw [hl]; simp
    · rw [h l hr]; simp

theorem step_overlay (C : Nat → Bool) (s : St) (op : Op) (l : Nat) :
    overlay C (step C s op) l = refStep (overlay C s l) l op := by
  cases op with
  | add ls =>
    simp only [step, stage, overlay, refStep]
    by_cases hl : ls.contains l = true
    · simp [hl, List.contains_iff_mem, List.mem_filter] at *
      simp [hl]
    · simp at hl
      simp [List.contains_iff_mem, List.mem_filter, hl]
  | rem ls =>
    simp only [step, removeOp, refStep]
    have hFc : ∀ l, (ls.filter (overlay C s)).contains l = (ls.contains l && overlay C s l) :=
      fun l => contains_filter _ _ l
    have ⟨h1, h2⟩ := removeLabels_closed s (ls.filter (overlay C s))
    generalize ls.filter (overlay C s) = F at h1 h2 hFc
    conv => lhs; unfold overlay
    rw [h1, h2, hFc]
    simp only [overlay]
    clear h1 h2 hFc
    by_cases h1' : s.remL.contains l = true <;> by_cases h2' : s.setL.contains l = true <;>
      by_cases h3' : ls.contains l = true <;> by_cases h4' : C l = true <;> simp_all

theorem run_overlay_aux (C : Nat → Bool) (ops : List Op) (s : St) (b : Nat → Bool)
    (h : ∀ l, overlay C s l = b l) (l : Nat) :
    overlay C (ops.foldl (step C) s) l = ops.foldl (refStep · l) (b l) := by
  induction ops generalizing s b with
  | nil => simpa using h l
  | cons op ops ih =>
    simp only [List.foldl_cons]
    exact ih (step C s op) (fun l => refStep (b l) l op) (fun l => by rw [step_overlay, h l])

/-- **`overlay_eq_ref`**: after any sequence of `SET n:…` / `REMOVE n:…`
clauses, what the query reads for a label is the last clause that mentioned
it (`REMOVE n:L SET n:L` keeps `L`, `SET n:L REMOVE n:L` drops it), else the
committed matrix. -/
theorem overlay_eq_ref (C : Nat → Bool) (ops : List Op) (l : Nat) :
    overlay C (run C ops) l = refHas C ops l := by
  unfold run refHas
  exact run_overlay_aux C ops ⟨[], []⟩ C (fun l => by simp [overlay]) l

/-- **`run_disjoint`**: `set_labels` and `remove_labels` stay disjoint per node,
the invariant `node_has_label`'s doc comment relies on. -/
theorem run_disjoint (C : Nat → Bool) (ops : List Op) : Disjoint (run C ops) := by
  unfold run
  suffices ∀ s, Disjoint s → Disjoint (ops.foldl (step C) s) from this _ (by intro l h; simp at h)
  induction ops with
  | nil => intro s h; exact h
  | cons op ops ih => intro s h; exact ih _ (step_disjoint C s op h)

/-- **`commit_eq_ref`**: the committed labels after `Pending::commit` are what
the query read before committing (read-your-writes is preserved by commit). -/
theorem commit_eq_ref (C : Nat → Bool) (ops : List Op) (l : Nat) :
    commitLabels C (run C ops) l = refHas C ops l := by
  rw [← overlay_eq_ref]
  unfold commitLabels overlay
  cases h1 : (run C ops).remL.contains l <;> cases h2 : (run C ops).setL.contains l <;>
    cases C l <;> simp

/-- **`nodeHas_eq_ref`**: `Runtime::node_has_label_id`'s pending-first answer,
falling back to the committed matrix, is the reference answer. -/
theorem nodeHas_eq_ref (C : Nat → Bool) (ops : List Op) (l : Nat) :
    (nodeHas (run C ops) l).getD (C l) = refHas C ops l := by
  rw [← overlay_eq_ref]; unfold nodeHas overlay
  split <;> (try split) <;> simp_all

/-- **The #3288 refactor of `update_node_labels` keeps the overlay**: reading the staged lists
through `staged_node_labels`, adds first and removals second, is `overlay` (removal wins). -/
theorem updateNodeLabels_eq_overlay (C : Nat → Bool) (s : St) (l : Nat) :
    updateNodeLabels C s l = overlay C s l := by
  simp only [updateNodeLabels, stagedNodeLabels, overlay]
  cases h1 : s.remL.contains l <;> cases h2 : s.setL.contains l <;> cases C l <;> simp

/-- **#3288 (fixes #3245, W4-fn-1): the deleted-node snapshot's label set is the query's view.**
`staged_label_ids` equals `update_node_labels`/`node_has_label` over the committed set, so a node
deleted after `SET n:C` keeps `C` and one deleted after `REMOVE n:A` does not keep `A`. -/
theorem stagedLabelIds_eq_overlay (C : Nat → Bool) (s : St) (l : Nat) :
    stagedLabelIds C s l = overlay C s l := by
  unfold stagedLabelIds; exact updateNodeLabels_eq_overlay C s l

/-- After any sequence of `SET n:…` / `REMOVE n:…` the snapshot a `DELETE` takes holds exactly
the labels the last clause that mentioned each label left (`overlay_eq_ref`), i.e. the labels
`labels(n)` showed just before the DELETE. -/
theorem stagedLabelIds_run (C : Nat → Bool) (ops : List Op) (l : Nat) :
    stagedLabelIds C (run C ops) l = refHas C ops l := by
  rw [stagedLabelIds_eq_overlay, overlay_eq_ref]

/-- The two shapes of #3245: `MATCH (n:A) SET n:C DELETE n` keeps `C` (and `A`); `MATCH (n:A)
REMOVE n:A DELETE n` keeps nothing (label 1 = A, 2 = C, committed `{A}`). -/
theorem deleted_snapshot_examples :
    (let C : Nat → Bool := fun l => l == 1
     stagedLabelIds C (run C [.add [2]]) 2 = true ∧ stagedLabelIds C (run C [.add [2]]) 1 = true ∧
     stagedLabelIds C (run C [.rem [1]]) 1 = false) := by decide

/-- **`removeOp_nodup`**: because REMOVE stages only labels the node currently
has, `remove_labels` never holds a label twice — so the commit's
`labels_removed += rows.len()` (pending.rs:1376) counts each removed label once. -/
theorem removeOp_no_dup_push (C : Nat → Bool) (s : St) (ls : List Nat) (l : Nat)
    (h : s.remL.contains l = true) : (ls.filter (overlay C s)).contains l = false := by
  rw [contains_filter]; have hm : l ∈ s.remL := by simpa using h
  simp [overlay, hm]

end PendingCommit.Labels
