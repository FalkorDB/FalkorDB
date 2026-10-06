import PendingCommit.PendingRels
/-
# Degrees and label scans over `Pending` (pending.rs:911-1096)

* `pendingDeg_spec` — `pending_indegree` / `pending_outdegree` (:1010,
  :1034): with type names, the sum over the listed types of the matching
  created edges of that type (a name listed twice is counted twice; the
  callers dedupe, `functions/entity.rs` `parse_degree_args`); without, all
  created edges.
* `deletedDeg_some` / `deletedDeg_none` — `pending_deleted_*degree` (:1059,
  :1081) count the deleted edges with the right endpoint (and type), but call
  `Graph::get_relationship_endpoints`, which panics for an id the committed
  graph does not have (graph.rs:3140-3149). `deg_panics_on_pending_deleted`:
  after `CREATE (a)-[r:R]->(b) DELETE r` the deleted set holds `r`, which was
  never committed, so `indegree(b)` in the same segment panics — #2769,
  reproduced live: `CREATE (a)-[r:R]->(b) DELETE r SET b.d = indegree(b)`
  kills the Rust server (`relationship 0 not found`), C returns `0`.
* `pendingWithLabels_spec`, `withLabelRemoves_spec` —
  `get_pending_nodes_with_labels` (:911): every created node when no label is
  given, else the nodes whose staged label adds include all of them (note:
  this includes existing nodes with a staged `SET n:L`); and
  `nodes_with_pending_label_removes` (:931): nodes with a staged removal of
  any of them, none for an empty list.
-/
namespace PendingCommit.PS
open PA

def pendingDeg (p : P) (sel : Nat × Nat × Nat → Bool) (types : List Nat) : Nat :=
  if types = [] then (p.relsByType.flatMap (·.2)).countP sel
  else ((types.filterMap (fget p.relsByType)).flatten).countP sel

def inSel (node : Nat) (e : Nat × Nat × Nat) : Bool := e.2.2 == node
def outSel (node : Nat) (e : Nat × Nat × Nat) : Bool := e.2.1 == node

/-- **`pendingDeg_spec`**. -/
theorem pendingDeg_spec (p : P) (sel : Nat × Nat × Nat → Bool) (types : List Nat) (h : types ≠ []) :
    pendingDeg p sel types = (types.map (fun t => (grp p t).countP sel)).sum := by
  unfold pendingDeg; rw [if_neg h]
  clear h
  induction types with
  | nil => simp
  | cons t ts ih =>
    simp only [grp] at ih ⊢
    simp only [List.filterMap_cons, List.map_cons, List.sum_cons]
    cases e : fget p.relsByType t with
    | none => simp [ih]
    | some es => simp [List.countP_append, ih]

def deletedDeg (p : P) (ep : Nat → Option (Nat × Nat)) (ty : Nat → Nat) (types : List Nat)
    (useDst : Bool) (node : Nat) : Option Nat :=
  p.deletedR.foldl (fun acc r => acc.bind fun c => (ep r).map fun (f, t) =>
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

/-- **`deletedDeg_some`**. -/
theorem deletedDeg_some (p : P) (ep : Nat → Option (Nat × Nat)) (ty : Nat → Nat) (types : List Nat)
    (useDst : Bool) (node : Nat) (h : ∀ r ∈ p.deletedR, (ep r).isSome) :
    deletedDeg p ep ty types useDst node = some (p.deletedR.countP (delMatch ep ty types useDst node)) := by
  unfold deletedDeg; rw [deletedDeg_aux]; simp [List.all_eq_true.2 h]

/-- **`deletedDeg_none`**: one deleted id without committed endpoints = panic. -/
theorem deletedDeg_none (p : P) (ep : Nat → Option (Nat × Nat)) (ty : Nat → Nat) (types : List Nat)
    (useDst : Bool) (node r : Nat) (hr : r ∈ p.deletedR) (he : ep r = none) :
    deletedDeg p ep ty types useDst node = none := by
  unfold deletedDeg; rw [deletedDeg_aux]
  have : ¬ p.deletedR.all (fun r => (ep r).isSome) = true := by
    simp only [List.all_eq_true]; intro hall; have := hall r hr; simp [he] at this
  simp [this]

def P0 : P := ⟨[], [], [], [], [], [], [], [], [], [], [], [], [], []⟩

/-- **`deg_panics_on_pending_deleted`** (#2769): create edge 0 between nodes 0
and 1 in this batch, delete it, ask for `indegree(1)`: the committed graph has
no edge 0, so the call panics. -/
theorem deg_panics_on_pending_deleted :
    deletedDeg (deletedRel (createdRel P0 0 0 1 7) 0) (fun _ => none) (fun _ => 7) [] true 1 = none :=
  deletedDeg_none _ _ _ _ _ _ 0 (by simp [deletedRel, sins, createdRel, P0]) rfl

/-! ## Label scans -/

def pendingWithLabels (p : P) (ls : List Nat) : List Nat :=
  if ls = [] then p.created else (p.setL.filter (fun e => ls.all (· ∈ e.2))).map (·.1)

def withLabelRemoves (p : P) (ls : List Nat) : List Nat :=
  if ls = [] then [] else (p.remL.filter (fun e => ls.any (· ∈ e.2))).map (·.1)

theorem pendingWithLabels_spec (p : P) (hk : (p.setL.map (·.1)).Nodup) (ls : List Nat) (n : Nat) :
    n ∈ pendingWithLabels p ls ↔
      (ls = [] ∧ n ∈ p.created) ∨ (ls ≠ [] ∧ ∃ labs, fget p.setL n = some labs ∧ ∀ l ∈ ls, l ∈ labs) := by
  unfold pendingWithLabels
  by_cases h : ls = []
  · simp [h]
  · simp only [h, ite_false, false_and, false_or, ne_eq, not_false_eq_true, true_and, List.mem_map,
      List.mem_filter, List.all_eq_true, decide_eq_true_eq]
    constructor
    · rintro ⟨⟨a, labs⟩, ⟨hm, hl⟩, rfl⟩; exact ⟨labs, fget_of_mem _ hk _ _ hm, hl⟩
    · rintro ⟨labs, hf, hl⟩; exact ⟨(n, labs), ⟨mem_of_fget _ _ _ hf, hl⟩, rfl⟩

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
