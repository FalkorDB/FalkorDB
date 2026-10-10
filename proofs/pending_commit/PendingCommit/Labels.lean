/-
# Staged label changes: `set_labels` / `remove_labels`

`graph/src/runtime/pending.rs`. Per node, `Pending` keeps the labels this query
added (`set_labels`) and removed (`remove_labels`). Maps are per node and nodes
never interact, so one node is modelled; `C` is the committed label set.

* `stage` — `stage_node_labels` (pending.rs:529): push the labels onto
  `set_labels`, drop them from `remove_labels`.
* `removeLabels` — `remove_node_labels` (pending.rs:565): per label, drop it
  from `set_labels`, push it onto `remove_labels`. The REMOVE operator
  (`ops/remove.rs:128-139`) first filters the labels to those the node
  currently has (`overlay`) — modelled as `removeOp`.
* `nodeHas` — `node_has_label` (pending.rs:590): removal first, then add, else
  `None` (ask the matrix).
* `commitLabels` — `set_nodes_labels_bulk` then `remove_nodes_labels`
  (pending.rs:1126-1146): `(C ∪ set) \ remove`.

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

/-- **`removeOp_nodup`**: because REMOVE stages only labels the node currently
has, `remove_labels` never holds a label twice — so the commit's
`labels_removed += rows.len()` (pending.rs:1144) counts each removed label once. -/
theorem removeOp_no_dup_push (C : Nat → Bool) (s : St) (ls : List Nat) (l : Nat)
    (h : s.remL.contains l = true) : (ls.filter (overlay C s)).contains l = false := by
  rw [contains_filter]; have hm : l ∈ s.remL := by simpa using h
  simp [overlay, hm]

end PendingCommit.Labels
