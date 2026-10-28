import OpsTraverse.CondTraverse
import OpsTraverse.ShortestPaths
/-
# ExpandInto over uncommitted deletes (#3452)

`CondTraverse.expandInto` models `ExpandIntoOp::expand_row` over one committed
graph. Within a commit segment the operator also reads `Pending`; this file adds
the pending deletes the operator sees.

| here | there (origin/main b29a2885c) |
| --- | --- |
| `Pending.delNodes` | `Pending::deleted_nodes` (`runtime/pending.rs:304`), read by `is_node_deleted` (`:1115-1120`); `DETACH DELETE` of a committed node only inserts it here (`runtime/ops/delete.rs:191-216`, `pending.rs:771-777`) and defers its incident edges to commit (`delete_implicit_edges`, `pending.rs:1325-1330`) |
| `Pending.delRels`  | `Pending::deleted_relationships`, read by `is_relationship_deleted` (`pending.rs:1129-1134`) |
| `expandIntoP`      | `ExpandIntoOp::expand_row` (`runtime/ops/expand_into.rs:125-245`): no row when `pending.is_node_deleted(src) \|\| is_node_deleted(dst)` (`:158-160`, added by #3463); pairs `:183-184`; collapse `:204-222` and emit `:224-235` skip `pending.is_relationship_deleted(id)` and sibling-used edges |
| `expandIntoPre`    | the operator before #3463 (b29a2885c^): the same without the endpoint check; kept for the `pre3463_` counterexamples |

`expandIntoP` with empty `Pending` is literally `CondTraverse.expandInto`
(`expandIntoP_empty`), so every earlier ExpandInto theorem is the
no-pending-deletes instance of this model.

## The bug (fixed by #3463, b29a2885c)

`CREATE (:A)-[:R]->(:B)`; `MATCH (a:A), (b:B) DETACH DELETE a MERGE (a)-[r:R]->(b)
RETURN count(r)`: Merge sits directly over ExpandInto (no Commit between), `a`
is in `deleted_nodes` but the edge `a→b` is not yet in `deleted_relationships`,
so the pre-#3463 ExpandInto matched it: Rust returned 1 with "Relationships
created: 0". The edge is doomed (deleted implicitly at commit). Main now yields
no row for a pending-deleted endpoint, so MERGE takes its create branch, which
refuses a deleted endpoint. Property `LiveExpandInto`: proved for main
(`expandIntoP_live`); the old counterexamples are `pre3463_*` on `expandIntoPre`.

The issue's other half (the `synthetic_label` branch, `expand_into.rs:162-173`,
reads committed labels only) has no valid-Cypher route to pending label state
(a MATCH after a write needs `WITH`, which plans a Commit; MERGE cannot
re-declare a bound node's labels), so it is not claimed here. (The new endpoint
check runs before that branch, so it applies there too; the model covers
relationship mode only.)
-/

namespace OpsTraverse.EIP

open OpsTraverse.ASP (eq_of_mem_of_id)

/-- The part of `Pending` ExpandInto can observe: uncommitted deletes. -/
structure Pending where
  delNodes : List Nat
  delRels  : List Nat
deriving DecidableEq, Repr

/-- `ExpandIntoOp::expand_row` (relationship mode) with `Pending` as it was
before #3463: no endpoint check. -/
def expandIntoPre (g : Graph) (p : Pending) (bidir emit : Bool) (src dst : Nat)
    (used : List Nat) : List Nat :=
  let ps := if bidir && src != dst then [(src, dst), (dst, src)] else [(src, dst)]
  ps.flatMap fun q =>
    let es := (edgesBetween g q.1 q.2).filter fun e =>
      !(p.delRels.contains e) && !(used.contains e)
    if emit then es else es.take 1

/-- `ExpandIntoOp::expand_row` (relationship mode) on main (`expand_into.rs:158-235`):
`is_node_deleted(src) || is_node_deleted(dst)` ⇒ no row (#3463). -/
def expandIntoP (g : Graph) (p : Pending) (bidir emit : Bool) (src dst : Nat)
    (used : List Nat) : List Nat :=
  if p.delNodes.contains src || p.delNodes.contains dst then []
  else expandIntoPre g p bidir emit src dst used

/-- With nothing pending, the model is the committed-graph `expandInto`. -/
theorem expandIntoP_empty (g : Graph) (b em : Bool) (s d : Nat) (u : List Nat) :
    expandIntoP g ⟨[], []⟩ b em s d u = expandInto g b em s d u := by
  simp [expandIntoP, expandIntoPre, expandInto]

/-- An edge is live in the effective graph: not deleted itself, and neither
endpoint deleted (a deleted node's edges go with it at commit). -/
def Live (g : Graph) (p : Pending) (e : Nat) : Prop :=
  e ∉ p.delRels ∧ ∀ x ∈ g, x.id = e → x.src ∉ p.delNodes ∧ x.dst ∉ p.delNodes

/-- **Property**: ExpandInto only matches edges of the effective graph. -/
def LiveExpandInto
    (op : Graph → Pending → Bool → Bool → Nat → Nat → List Nat → List Nat) : Prop :=
  ∀ g p b em s d u, g.WF → ∀ e ∈ op g p b em s d u, Live g p e

theorem mem_expandIntoPre {g : Graph} {p : Pending} {b em : Bool} {s d e : Nat}
    {u : List Nat} (h : e ∈ expandIntoPre g p b em s d u) :
    e ∉ p.delRels ∧ e ∉ u ∧ ∃ x ∈ g, x.id = e ∧
      ((x.src = s ∧ x.dst = d) ∨ (x.src = d ∧ x.dst = s)) := by
  unfold expandIntoPre at h
  simp only [List.mem_flatMap] at h
  obtain ⟨q, hq, he⟩ := h
  have he' : e ∈ (edgesBetween g q.1 q.2).filter fun e =>
      !(p.delRels.contains e) && !(u.contains e) := by
    cases em
    · exact List.mem_of_mem_take he
    · exact he
  simp only [List.mem_filter, Bool.and_eq_true, Bool.not_eq_true',
    List.contains_eq_mem, decide_eq_false_iff_not] at he'
  obtain ⟨hb, hdr, hu⟩ := he'
  obtain ⟨x, hx, hid, h1, h2⟩ := mem_edgesBetween.1 hb
  refine ⟨hdr, hu, x, hx, hid, ?_⟩
  split at hq
  · simp only [List.mem_cons, List.not_mem_nil, or_false] at hq
    rcases hq with rfl | rfl
    · exact Or.inl ⟨h1, h2⟩
    · exact Or.inr ⟨h1, h2⟩
  · simp only [List.mem_cons, List.not_mem_nil, or_false] at hq
    subst hq; exact Or.inl ⟨h1, h2⟩

/-- **Main satisfies the property** (#3463). -/
theorem expandIntoP_live : LiveExpandInto expandIntoP := by
  intro g p b em s d u hwf e he
  unfold expandIntoP at he
  split at he
  · simp at he
  · rename_i hn
    simp only [Bool.or_eq_true, List.contains_eq_mem, decide_eq_true_eq, not_or] at hn
    obtain ⟨hdr, _, x, hx, hid, hdir⟩ := mem_expandIntoPre he
    refine ⟨hdr, fun y hy hyid => ?_⟩
    have : y = x := eq_of_mem_of_id hwf hy hx (hyid.trans hid.symm)
    subst this
    rcases hdir with ⟨h1, h2⟩ | ⟨h1, h2⟩ <;> rw [h1, h2]
    · exact ⟨hn.1, hn.2⟩
    · exact ⟨hn.2, hn.1⟩

/-- Main drops explicitly deleted edges. -/
theorem expandIntoP_not_delRel {g : Graph} {p : Pending} {b em : Bool} {s d e : Nat}
    {u : List Nat} (h : e ∈ expandIntoP g p b em s d u) : e ∉ p.delRels := by
  unfold expandIntoP at h
  split at h
  · simp at h
  · exact (mem_expandIntoPre h).1

/-! ## #3452 (fixed by #3463): `DETACH DELETE a MERGE (a)-[r:R]->(b)` -/

/-- `CREATE (:A)-[:R]->(:B)`: node `a = 0`, `b = 1`, edge `0`. -/
def gAB1 : Graph := [⟨0, 0, 1⟩]

/-- After `DETACH DELETE a` in the same segment: `a` pending-deleted, the edge
not (it is deleted implicitly at commit). -/
def pDetach : Pending := ⟨[0], []⟩

/-- Pre-#3463 ExpandInto (the MERGE's match) found the doomed edge, so MERGE
created nothing and `count(r)` was 1. -/
theorem pre3463_matches_doomed_edge : expandIntoPre gAB1 pDetach false true 0 1 [] = [0] := by
  decide

/-- Collapse mode (`r` unread) matched it too. -/
theorem pre3463_collapse_matches_doomed_edge :
    expandIntoPre gAB1 pDetach false false 0 1 [] = [0] := by decide

/-- Main matches nothing for a pending-deleted endpoint (either side). -/
theorem main_3463_empty : expandIntoP gAB1 pDetach false true 0 1 [] = [] := by decide

theorem main_3463_empty_dst : expandIntoP gAB1 ⟨[1], []⟩ false true 0 1 [] = [] := by decide

/-- **The pre-#3463 operator violated `LiveExpandInto`.** -/
theorem pre3463_violates : ¬ LiveExpandInto expandIntoPre := by
  intro h
  have := h gAB1 pDetach false true 0 1 [] (by unfold Graph.WF; decide) 0
    (by rw [pre3463_matches_doomed_edge]; simp)
  exact (this.2 ⟨0, 0, 1⟩ (by simp [gAB1]) rfl).1 (by simp [pDetach])

/-! ## Open: the CondTraverse path of the same query (#3463 follow-up)

`CondTraverseOp` (one endpoint unbound, e.g. `DETACH DELETE a MERGE (a)-[r:R]->(x:B)`)
never reads `Pending`: `cond_traverse.rs` has no `is_node_deleted` /
`is_relationship_deleted` call at all (`expand_row` `:749`, `process_pairs` `:944`
read the relationship matrices only). So its model is `CondTraverse.expandRow`
unchanged, and it still matches the doomed edge of a pending-deleted `a` (and the
edge of an explicitly `DELETE`d `r`, and an edge into a pending-deleted target).
Expected: no match (the pending-deleted endpoint is gone), so MERGE takes its
create branch, which refuses a deleted endpoint, exactly as #3463 makes ExpandInto
do. Not fixed in Rust here. -/

/-- `CondTraverseOp::expand_row` with `Pending`: it ignores `Pending`, so this is
`expandRow`. Rows are `(from, to, edge)`. -/
def condTraverseP (g : Graph) (_p : Pending) (bidir emit : Bool) (r : RowIn) :
    List (Nat × Nat × Nat) :=
  expandRow g bidir emit r

/-- Property: CondTraverse only matches edges of the effective graph. -/
def LiveCondTraverse
    (op : Graph → Pending → Bool → Bool → RowIn → List (Nat × Nat × Nat)) : Prop :=
  ∀ g p b em r, g.WF → ∀ x ∈ op g p b em r, Live g p x.2.2

/-- `a = 0` bound, `x` unbound. -/
def rowA : RowIn := ⟨some 0, none, []⟩

/-- `MATCH (a:A) DETACH DELETE a MERGE (a)-[r:R]->(x:B)`: CondTraverse returns the
doomed edge `0 -> 1`. -/
theorem bug_condTraverse_matches_doomed_edge :
    condTraverseP gAB1 pDetach false true rowA = [(0, 1, 0)] := by decide

/-- **Main's CondTraverse violates `LiveCondTraverse`.** -/
theorem bug_condTraverse_main_violates : ¬ LiveCondTraverse condTraverseP := by
  intro h
  have := h gAB1 pDetach false true rowA (by unfold Graph.WF; decide) (0, 1, 0)
    (by rw [bug_condTraverse_matches_doomed_edge]; simp)
  exact (this.2 ⟨0, 0, 1⟩ (by simp [gAB1]) rfl).1 (by simp [pDetach])

end OpsTraverse.EIP
