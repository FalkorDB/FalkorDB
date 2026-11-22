/-
# Staged label changes: `set_labels` / `remove_labels`

`graph/src/runtime/pending.rs`. Per node, `Pending` keeps the labels this query
added (`set_labels`) and removed (`remove_labels`). Maps are per node and nodes
never interact, so one node is modelled; `C` is the committed label set.

* `stage` — `stage_node_labels` (pending.rs): push the labels onto
  `set_labels`, drop them from `remove_labels`.
* `removeLabels` — `remove_node_labels` (pending.rs): per label, drop it
  from `set_labels`; if it was there (`was_staged`) and the committed node lacks
  it (a created node, whose `C` is empty, or `!g.node_has_label_id`), only bump
  `staged_labels_removed`; else push it onto
  `remove_labels`. `removeLabelsC` is the loop with that counter, `removeLabels`
  its map half. The REMOVE operator (`ops/remove.rs`) first filters the
  labels to those the node currently has (`overlay`) — modelled as `removeOp`.
* `nodeHas` — `node_has_label` (pending.rs): removal first, then add, else
  `None` (ask the matrix).
* `stagedNodeLabels` — `Pending::staged_node_labels` (pending.rs): the two staged
  lists as they are, `(set_labels, remove_labels)`; `update_node_labels` (:806) is now
  written over it.
* `stagedLabelIds` — `Runtime::staged_label_ids` (ops/delete.rs): the committed
  ids, plus the staged adds, minus the staged removes. The deleted-node snapshot
  (`delete_nodes_bulk` :306, `delete_entity` :486) is built from it.
* `commitLabels` — `set_nodes_labels_bulk` then `remove_nodes_labels`
  (`Pending::commit`, pending.rs): `(C ∪ set) \ remove`.

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

/-- One iteration of the `for label in labels` loop in `remove_node_labels`
(pending.rs), with `staged_labels_removed` as the `Nat`. `C` is the
committed label set of the node: `g.node_has_label_id(id, l)`, and empty for a
node in `created_nodes` (whose `||` short-circuits the matrix). -/
def removeOneC (C : Nat → Bool) (sc : St × Nat) (l : Nat) : St × Nat :=
  let s := sc.1
  let wasStaged := s.setL.contains l
  let s1 : St := { s with setL := s.setL.filter (· != l) }
  if wasStaged && !C l then (s1, sc.2 + 1)
  else ({ s1 with remL := s.remL ++ [l] }, sc.2)

def removeLabelsC (C : Nat → Bool) (s : St) (n : Nat) (ls : List Nat) : St × Nat :=
  ls.foldl (removeOneC C) (s, n)

/-- The staged maps after `remove_node_labels` (the counter does not feed back). -/
def removeLabels (C : Nat → Bool) (s : St) (ls : List Nat) : St := (removeLabelsC C s 0 ls).1

/-- What the query sees: `update_node_labels` over the committed set. -/
def overlay (C : Nat → Bool) (s : St) (l : Nat) : Bool :=
  if s.remL.contains l then false else if s.setL.contains l then true else C l

def nodeHas (s : St) (l : Nat) : Option Bool :=
  if s.remL.contains l then some false else if s.setL.contains l then some true else none

/-- `ops/remove.rs`: only labels the node currently carries are staged. -/
def removeOp (C : Nat → Bool) (s : St) (ls : List Nat) : St :=
  removeLabels C s (ls.filter (overlay C s))

/-- `Pending::staged_node_labels`: the lists as staged. -/
def stagedNodeLabels (s : St) : List Nat × List Nat := (s.setL, s.remL)

/-- `Pending::update_node_labels` after #3288: insert every staged add, then remove every
staged removal, both read through `staged_node_labels`. -/
def updateNodeLabels (C : Nat → Bool) (s : St) (l : Nat) : Bool :=
  let (added, removed) := stagedNodeLabels s
  (C l || added.contains l) && !removed.contains l

/-- `Runtime::staged_label_ids` (ops/delete.rs): `g.get_node_label_ids(id)` collected into a
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
    · subst hx; cases hp : p l <;> simp [hp]
    · have : (x == l) = false := by simp; omega
      cases hp : p x <;> simp [hp, hx] <;> cases p l <;> simp

def Disjoint (s : St) : Prop := ∀ l, s.remL.contains l = true → s.setL.contains l = false

/-! ## Proofs -/

theorem removeOneC_setL (C : Nat → Bool) (sc : St × Nat) (x l : Nat) :
    (removeOneC C sc x).1.setL.contains l = (sc.1.setL.contains l && !decide (l = x)) := by
  unfold removeOneC; dsimp only
  split <;> by_cases hlx : l = x <;> simp [List.mem_filter, hlx]

theorem removeOneC_remL (C : Nat → Bool) (sc : St × Nat) (x l : Nat) :
    (removeOneC C sc x).1.remL.contains l =
      (sc.1.remL.contains l || (decide (l = x) && !(sc.1.setL.contains x && !C x))) := by
  unfold removeOneC; dsimp only
  split
  · rename_i h; simp only at h ⊢; rw [h]; simp
  · rename_i h; simp only at h ⊢
    have h' : (sc.1.setL.contains x && !C x) = false := by simpa using h
    rw [h']; by_cases hlx : l = x <;> simp [hlx]

theorem removeOneC_count (C : Nat → Bool) (sc : St × Nat) (x : Nat) :
    (removeOneC C sc x).2 + (removeOneC C sc x).1.remL.length = sc.2 + sc.1.remL.length + 1 := by
  unfold removeOneC; dsimp only; split <;> simp <;> omega

theorem fold_setL (C : Nat → Bool) (ls : List Nat) (sc : St × Nat) (l : Nat) :
    (ls.foldl (removeOneC C) sc).1.setL.contains l = (sc.1.setL.contains l && !ls.contains l) := by
  induction ls generalizing sc with
  | nil => simp
  | cons x xs ih =>
    rw [List.foldl_cons, ih, removeOneC_setL]
    by_cases h : l = x <;> simp [h]

theorem fold_remL_sub (C : Nat → Bool) (ls : List Nat) (sc : St × Nat) (l : Nat)
    (h : (ls.foldl (removeOneC C) sc).1.remL.contains l = true) :
    sc.1.remL.contains l = true ∨ ls.contains l = true := by
  induction ls generalizing sc with
  | nil => exact .inl h
  | cons x xs ih =>
    rw [List.foldl_cons] at h
    rcases ih _ h with h1 | h1
    · rw [removeOneC_remL] at h1
      by_cases hx : l = x
      · subst hx; exact .inr (by simp)
      · simp only [hx, decide_false, Bool.false_and, Bool.or_false] at h1; exact .inl h1
    · exact .inr (by simp at h1 ⊢; exact .inr h1)

/-- **`removeLabels_closed`** (#3459): for the distinct labels REMOVE passes
(an `OrderSet`, ops/remove.rs), a label lands in `remove_labels` iff it
was there, or it is removed now and either the committed node carries it or this
query had not staged it; every removed label leaves `set_labels`. -/
theorem removeLabels_closed (C : Nat → Bool) (s : St) (ls : List Nat) (hnd : ls.Nodup) :
    (∀ l, (removeLabels C s ls).remL.contains l =
      (s.remL.contains l || (ls.contains l && (C l || !s.setL.contains l)))) ∧
    (∀ l, (removeLabels C s ls).setL.contains l = (s.setL.contains l && !ls.contains l)) := by
  refine ⟨fun l => ?_, fun l => fold_setL C ls (s, 0) l⟩
  unfold removeLabels removeLabelsC
  generalize (0 : Nat) = n
  induction ls generalizing s n with
  | nil => simp
  | cons x xs ih =>
    have hx : x ∉ xs := (List.nodup_cons.1 hnd).1
    rw [List.foldl_cons]
    have e := ih (removeOneC C (s, n) x).1 (List.nodup_cons.1 hnd).2 (removeOneC C (s, n) x).2
    rw [show ∀ q : St × Nat, (q.1, q.2) = q from fun _ => rfl] at e
    rw [e, removeOneC_remL, removeOneC_setL]
    simp only
    by_cases h : l = x
    · subst h
      have : xs.contains l = false := by simpa using hx
      rw [this]; cases s.remL.contains l <;> cases s.setL.contains l <;> cases C l <;> simp
    · simp [h]

/-- **`removeLabelsC_count`**: every label passed is counted exactly once,
either by `staged_labels_removed` or as a `remove_labels` entry — so the
`Labels removed` statistic (`commit`, pending.rs) is what it was
before #3459, when every label was pushed onto `remove_labels`. -/
theorem removeLabelsC_count (C : Nat → Bool) (s : St) (n : Nat) (ls : List Nat) :
    (removeLabelsC C s n ls).2 + (removeLabelsC C s n ls).1.remL.length = n + s.remL.length + ls.length := by
  unfold removeLabelsC
  induction ls generalizing s n with
  | nil => simp
  | cons x xs ih =>
    rw [List.foldl_cons]
    have e := ih (removeOneC C (s, n) x).1 (removeOneC C (s, n) x).2
    rw [show ∀ q : St × Nat, (q.1, q.2) = q from fun _ => rfl] at e
    have := removeOneC_count C (s, n) x
    simp only at this
    simp only [List.length_cons]; omega

theorem fold_overlay (C : Nat → Bool) (ls : List Nat) (sc : St × Nat) (l : Nat) :
    overlay C (ls.foldl (removeOneC C) sc).1 l = (!ls.contains l && overlay C sc.1 l) := by
  induction ls generalizing sc with
  | nil => simp
  | cons x xs ih =>
    rw [List.foldl_cons, ih]
    have h1 := removeOneC_setL C sc x l
    have h2 := removeOneC_remL C sc x l
    simp only [overlay] at *
    by_cases hx : l = x
    · subst hx
      cases hr : sc.1.remL.contains l <;> cases hs : sc.1.setL.contains l <;> cases hc : C l <;>
        simp_all
    · simp [hx] at h1 h2
      simp [h1, h2, hx]

theorem step_disjoint (C : Nat → Bool) (s : St) (op : Op) (h : Disjoint s) :
    Disjoint (step C s op) := by
  intro l hl
  cases op with
  | add ls =>
    simp [step, stage, List.mem_filter] at hl ⊢
    exact ⟨by have := h l; simp at this; exact this hl.1, hl.2⟩
  | rem ls =>
    simp only [step, removeOp, removeLabels, removeLabelsC] at hl ⊢
    rw [fold_setL]
    rcases fold_remL_sub C _ _ l hl with hr | hr
    · simp only at hr; rw [h l hr]; simp
    · rw [hr]; simp

theorem step_overlay (C : Nat → Bool) (s : St) (op : Op) (l : Nat) :
    overlay C (step C s op) l = refStep (overlay C s l) l op := by
  cases op with
  | add ls =>
    simp only [step, stage, overlay, refStep]
    by_cases hl : ls.contains l = true
    · simp [List.mem_filter] at *
      simp [hl]
    · simp at hl
      simp [List.mem_filter, hl]
  | rem ls =>
    simp only [step, removeOp, refStep, removeLabels, removeLabelsC]
    rw [fold_overlay, contains_filter]
    simp only
    cases h1 : ls.contains l <;> cases h2 : overlay C s l <;> simp

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
`labels_removed += rows.len()` (pending.rs) counts each removed label once. -/
theorem removeOp_no_dup_push (C : Nat → Bool) (s : St) (ls : List Nat) (l : Nat)
    (h : s.remL.contains l = true) : (ls.filter (overlay C s)).contains l = false := by
  rw [contains_filter]; have hm : l ∈ s.remL := by simpa using h
  simp [overlay, hm]

/-- **`removeOp_committed`** (#3459, fixes #3448): REMOVE (`removeOp`, distinct labels)
pushes onto `remove_labels` only labels the committed node carries: a label that is only staged
(any label of a created node, a `SET n:L` of an uncommitted `L`) just cancels its add. -/
theorem removeOp_committed (C : Nat → Bool) (s : St) (ls : List Nat) (hnd : ls.Nodup)
    (h : ∀ l, s.remL.contains l = true → C l = true) :
    ∀ l, (removeOp C s ls).remL.contains l = true → C l = true := by
  intro l hl
  unfold removeOp at hl
  rw [(removeLabels_closed C s _ (hnd.filter _)).1, contains_filter] at hl
  cases hr : s.remL.contains l
  · rw [hr] at hl
    cases hc : C l
    · cases hs : s.setL.contains l <;> simp only [overlay, hr, hs, hc] at hl <;> simp at hl
    · rfl
  · exact h l hr

end PendingCommit.Labels
