import PendingCommit.PendingRels
/-
# Degrees and label scans over `Pending` (pending.rs:1114-1340)

* `pendingDeg_spec` — `pending_indegree` / `pending_outdegree`: with type
  names, the sum over the listed types of the matching created edges of that
  type that are not pending-deleted (a name listed twice is counted twice; the
  callers dedupe, `functions/entity.rs` `parse_degree_args`); without, all such
  created edges. Since #3520 (9ffa9ab04) an edge created and deleted in the
  query segment (since the last commit) is not counted (`pendingDeg_skips_deleted`).
* `deletedDeg_some` / `deletedDeg_none` — `pending_deleted_*degree` count the
  deleted edges with the right endpoint (and type) that are *committed*: an
  edge created since the last commit is skipped (`deletedDeg_skips_created`),
  so `Graph::get_relationship_endpoints` (which panics for an id the committed
  graph does not have) is only asked for committed ids (`deletedDeg_some`).
  `fixed2769_created_deleted`: after `CREATE (a)-[r:R]->(b) DELETE r` the
  degrees are 0 and nothing panics — #2769, fixed by #3520.
* `pendingWithLabels_spec`, `withLabelRemoves_spec` —
  `get_pending_nodes_with_labels` (:1114, now taking the graph, #3467
  a477d9aa3): every created node when no label is given, else the *live*
  nodes (not pending-deleted) that gain a pattern label: each pattern label is
  either committed (the node is not created, the label is not staged for
  removal, the graph has it) or staged, and at least one is only staged. And
  `nodes_with_pending_label_removes`: nodes with a staged removal of any of
  them, none for an empty list.
-/
namespace PendingCommit.PS
open PA

def pendingDegRaw (p : P) (sel : Nat × Nat × Nat → Bool) (types : List Nat) : Nat :=
  if types = [] then (p.relsByType.flatMap (·.2)).countP sel
  else ((types.filterMap (fget p.relsByType)).flatten).countP sel

def inSel (node : Nat) (e : Nat × Nat × Nat) : Bool := e.2.2 == node
def outSel (node : Nat) (e : Nat × Nat × Nat) : Bool := e.2.1 == node

/-- The edge is not pending-deleted (`!is_relationship_deleted(id)`, #3520). -/
def liveSel (p : P) (sel : Nat × Nat × Nat → Bool) (e : Nat × Nat × Nat) : Bool :=
  sel e && !decide (e.1 ∈ p.deletedR)

/-- `pending_indegree` / `pending_outdegree` (:1231, :1257): created edges at the node that are not deleted. -/
def pendingDeg (p : P) (sel : Nat × Nat × Nat → Bool) (types : List Nat) : Nat :=
  pendingDegRaw p (liveSel p sel) types

theorem pendingDegRaw_spec (p : P) (sel : Nat × Nat × Nat → Bool) (types : List Nat) (h : types ≠ []) :
    pendingDegRaw p sel types = (types.map (fun t => (grp p t).countP sel)).sum := by
  unfold pendingDegRaw; rw [if_neg h]
  clear h
  induction types with
  | nil => simp
  | cons t ts ih =>
    simp only [grp] at ih ⊢
    simp only [List.filterMap_cons, List.map_cons, List.sum_cons]
    cases e : fget p.relsByType t with
    | none => simp [ih]
    | some es => simp [List.countP_append, ih]

/-- **`pendingDeg_spec`**. -/
theorem pendingDeg_spec (p : P) (sel : Nat × Nat × Nat → Bool) (types : List Nat) (h : types ≠ []) :
    pendingDeg p sel types = (types.map (fun t => (grp p t).countP (liveSel p sel))).sum :=
  pendingDegRaw_spec p (liveSel p sel) types h

/-- **`pendingDeg_skips_deleted`** (#2769, #3520): a deleted edge adds nothing, whatever its endpoints. -/
theorem pendingDeg_skips_deleted (p : P) (sel : Nat × Nat × Nat → Bool) (e : Nat × Nat × Nat)
    (hd : e.1 ∈ p.deletedR) : liveSel p sel e = false := by
  simp [liveSel, hd]

/-- `pending_deleted_indegree` / `_outdegree` (:1284, :1314): the deleted edges that were not created since the last
commit, each asked for its committed endpoints (`ep`, `none` = `get_relationship_endpoints` panics). -/
def deletedDeg (p : P) (ep : Nat → Option (Nat × Nat)) (ty : Nat → Nat) (types : List Nat)
    (useDst : Bool) (node : Nat) : Option Nat :=
  (p.deletedR.filter (fun r => !(fget p.relTypes r).isSome)).foldl (fun acc r => acc.bind fun c => (ep r).map fun (f, t) =>
    if (if useDst then t else f) = node ∧ (types = [] ∨ ty r ∈ types) then c + 1 else c) (some 0)

def delMatch (ep : Nat → Option (Nat × Nat)) (ty : Nat → Nat) (types : List Nat) (useDst : Bool) (node r : Nat) : Bool :=
  match ep r with
  | some (f, t) => decide ((if useDst then t else f) = node ∧ (types = [] ∨ ty r ∈ types))
  | none => false

theorem fold_none (ep : Nat → Option (Nat × Nat)) (g : Nat → Nat → Nat × Nat → Nat) :
    ∀ (rs : List Nat), rs.foldl (fun acc r => acc.bind fun c => (ep r).map (g c r)) none = none
  | [] => rfl
  | _ :: rs => by simp only [List.foldl_cons, Option.bind_none]; exact fold_none ep g rs

theorem deletedDeg_aux (ep : Nat → Option (Nat × Nat)) (ty : Nat → Nat) (types : List Nat) (useDst : Bool)
    (node : Nat) : ∀ (rs : List Nat) (c : Nat),
    rs.foldl (fun acc r => acc.bind fun c => (ep r).map fun (f, t) =>
      if (if useDst then t else f) = node ∧ (types = [] ∨ ty r ∈ types) then c + 1 else c) (some c) =
    if rs.all (fun r => (ep r).isSome) then some (c + rs.countP (delMatch ep ty types useDst node)) else none
  | [], c => by simp
  | r :: rs, c => by
    simp only [List.foldl_cons, Option.bind_some, List.all_cons, List.countP_cons]
    cases e : ep r with
    | none =>
      simp only [Option.map_none, Option.isSome_none, Bool.false_and, Bool.false_eq_true, ite_false]
      exact fold_none ep (fun c r x => if (if useDst = true then x.2 else x.1) = node ∧
        (types = [] ∨ ty r ∈ types) then c + 1 else c) rs
    | some ft =>
      obtain ⟨f, t⟩ := ft
      simp only [Option.map_some, Option.isSome_some, Bool.true_and]
      rw [deletedDeg_aux ep ty types useDst node rs]
      simp only [delMatch, e]
      by_cases hall : (rs.all fun r => (ep r).isSome) = true
      · simp only [hall, ite_true]
        by_cases hc : (if useDst = true then t else f) = node ∧ (types = [] ∨ ty r ∈ types)
        · simp only [hc, ite_true, decide_true]; simp; omega
        · simp only [hc, ite_false, decide_false]; simp
      · simp [hall]

/-- **`deletedDeg_some`**: when every committed deleted edge has endpoints, the count is that of the
matching ones among them. -/
theorem deletedDeg_some (p : P) (ep : Nat → Option (Nat × Nat)) (ty : Nat → Nat) (types : List Nat)
    (useDst : Bool) (node : Nat) (h : ∀ r ∈ p.deletedR, (fget p.relTypes r).isSome = false → (ep r).isSome) :
    deletedDeg p ep ty types useDst node =
      some ((p.deletedR.filter (fun r => !(fget p.relTypes r).isSome)).countP (delMatch ep ty types useDst node)) := by
  unfold deletedDeg; rw [deletedDeg_aux]
  have : (p.deletedR.filter (fun r => !(fget p.relTypes r).isSome)).all (fun r => (ep r).isSome) = true := by
    simp only [List.all_eq_true, List.mem_filter]
    intro r ⟨hr, hc⟩
    exact h r hr (by simpa using hc)
  rw [if_pos this]; simp

/-- **`deletedDeg_none`**: one deleted, committed-side id without committed endpoints = panic. Only reachable
for an id that is neither created since the last commit nor committed (a stale id). -/
theorem deletedDeg_none (p : P) (ep : Nat → Option (Nat × Nat)) (ty : Nat → Nat) (types : List Nat)
    (useDst : Bool) (node r : Nat) (hr : r ∈ p.deletedR) (hc : (fget p.relTypes r).isSome = false)
    (he : ep r = none) :
    deletedDeg p ep ty types useDst node = none := by
  unfold deletedDeg; rw [deletedDeg_aux]
  have : ¬ (p.deletedR.filter (fun r => !(fget p.relTypes r).isSome)).all (fun r => (ep r).isSome) = true := by
    simp only [List.all_eq_true, List.mem_filter]; intro hall
    have := hall r ⟨hr, by simp [hc]⟩; simp [he] at this
  rw [if_neg this]

/-- **`deletedDeg_skips_created`** (#3520): an edge created since the last commit contributes nothing, and its
endpoints are never asked for: if the only deleted edges were created here, the answer is `some 0` for any
`ep`. -/
theorem deletedDeg_skips_created (p : P) (ep : Nat → Option (Nat × Nat)) (ty : Nat → Nat) (types : List Nat)
    (useDst : Bool) (node : Nat) (h : ∀ r ∈ p.deletedR, (fget p.relTypes r).isSome = true) :
    deletedDeg p ep ty types useDst node = some 0 := by
  have h2 : ∀ r ∈ p.deletedR, (fget p.relTypes r).isSome = false → (ep r).isSome := by
    intro r hr hf; have := h r hr; rw [hf] at this; cases this
  rw [deletedDeg_some p ep ty types useDst node h2]
  have : p.deletedR.filter (fun r => !(fget p.relTypes r).isSome) = [] := by
    apply List.filter_eq_nil_iff.2
    intro r hr; simp [h r hr]
  rw [this]; rfl

def P0 : P := ⟨[], [], [], [], [], [], [], [], [], [], [], [], [], []⟩

/-- **`fixed2769_created_deleted`** (#2769, fixed by #3520): create edge 0 between nodes 0 and 1 in this
batch, delete it, ask for `indegree(1)`: no panic (the committed graph is not asked about edge 0), the
degree is 0, as in C. -/
theorem fixed2769_created_deleted :
    deletedDeg (deletedRel (createdRel P0 0 0 1 7) 0) (fun _ => none) (fun _ => 7) [] true 1 = some 0 ∧
    pendingDeg (deletedRel (createdRel P0 0 0 1 7) 0) (inSel 1) [] = 0 := by
  constructor
  · exact deletedDeg_skips_created _ _ _ _ _ _ (by
      intro r hr
      simp only [deletedRel, sins, createdRel, P0] at hr ⊢
      simp at hr
      subst hr; decide)
  · decide

/-! ## Label scans -/

/-- Is `l` committed on node `n` and still there after the staged removals: not a created node, not in
`remove_labels`, and the graph has it (`C n l` = `Graph::node_has_label_id`). -/
def committedLabel (p : P) (C : Nat → Nat → Bool) (n l : Nat) : Bool :=
  !decide (n ∈ p.created) && !((fget p.remL n).any (fun r => decide (l ∈ r))) && C n l

/-- `get_pending_nodes_with_labels(label_ids, g)` (:1114): the nodes of `set_labels` that are live and gain
a pattern label (all of them carried after the pending changes, at least one only staged). -/
def pendingWithLabels (p : P) (C : Nat → Nat → Bool) (ls : List Nat) : List Nat :=
  if ls = [] then p.created else
    (p.setL.filter (fun e =>
      !decide (e.1 ∈ p.deletedN) &&
      ls.all (fun l => committedLabel p C e.1 l || decide (l ∈ e.2)) &&
      !ls.all (fun l => committedLabel p C e.1 l))).map (·.1)

def withLabelRemoves (p : P) (ls : List Nat) : List Nat :=
  if ls = [] then [] else (p.remL.filter (fun e => ls.any (· ∈ e.2))).map (·.1)

theorem pendingWithLabels_spec (p : P) (C : Nat → Nat → Bool) (hk : (p.setL.map (·.1)).Nodup)
    (ls : List Nat) (n : Nat) :
    n ∈ pendingWithLabels p C ls ↔
      (ls = [] ∧ n ∈ p.created) ∨
      (ls ≠ [] ∧ n ∉ p.deletedN ∧ ∃ staged, fget p.setL n = some staged ∧
        (∀ l ∈ ls, committedLabel p C n l = true ∨ l ∈ staged) ∧ ∃ l ∈ ls, committedLabel p C n l = false) := by
  unfold pendingWithLabels
  by_cases h : ls = []
  · simp [h]
  · simp only [h, ite_false, false_and, false_or, ne_eq, not_false_eq_true, true_and, List.mem_map,
      List.mem_filter, Bool.and_eq_true, Bool.not_eq_true', List.all_eq_true, decide_eq_true_eq,
      Bool.or_eq_true, decide_eq_false_iff_not]
    constructor
    · rintro ⟨⟨a, labs⟩, ⟨hm, ⟨hd, hl⟩, hnc⟩, rfl⟩
      refine ⟨hd, labs, fget_of_mem _ hk _ _ hm, hl, ?_⟩
      apply Classical.byContradiction
      intro hno
      have hno : ∀ l ∈ ls, ¬ committedLabel p C a l = false := fun l hl' hc => hno ⟨l, hl', hc⟩
      have : (ls.all fun l => committedLabel p C a l) = true := by
        simp only [List.all_eq_true]; intro l hl'; have := hno l hl'; simpa using this
      rw [this] at hnc; cases hnc
    · rintro ⟨hd, labs, hf, hl, l, hl', hc⟩
      refine ⟨(n, labs), ⟨mem_of_fget _ _ _ hf, ⟨hd, hl⟩, ?_⟩, rfl⟩
      have : (ls.all fun l => committedLabel p C n l) = false := by
        simp only [Bool.eq_false_iff, ne_eq, List.all_eq_true]; intro hall; have := hall l hl'; simp [hc] at this
      simp [this]

theorem withLabelRemoves_spec (p : P) (hk : (p.remL.map (·.1)).Nodup) (ls : List Nat) (n : Nat) :
    n ∈ withLabelRemoves p ls ↔ ls ≠ [] ∧ ∃ rem, fget p.remL n = some rem ∧ ∃ l ∈ ls, l ∈ rem := by
  unfold withLabelRemoves
  by_cases h : ls = []
  · simp [h]
  · simp only [h, ite_false, ne_eq, not_false_eq_true, true_and, List.mem_map, List.mem_filter,
      List.any_eq_true, decide_eq_true_eq]
    constructor
    · rintro ⟨⟨a, rem⟩, ⟨hm, hl⟩, rfl⟩; exact ⟨rem, fget_of_mem _ hk _ _ hm, hl⟩
    · rintro ⟨rem, hf, hl⟩; exact ⟨(n, rem), ⟨mem_of_fget _ _ _ hf, hl⟩, rfl⟩

end PendingCommit.PS
