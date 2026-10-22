import OpsTraverse.CondTraverse
import OpsTraverse.ShortestPaths
/-
# ExpandInto over uncommitted deletes (#3452)

`CondTraverse.expandInto` models `ExpandIntoOp::expand_row` over one committed
graph. Within a commit segment the operator also reads `Pending`; this file adds
the pending deletes the operator sees.

| here | there (origin/main b84473a7f) |
| --- | --- |
| `Pending.delNodes` | `Pending::deleted_nodes` (`runtime/pending.rs:304`), read by `is_node_deleted` (`:1115-1120`); `DETACH DELETE` of a committed node only inserts it here (`runtime/ops/delete.rs:191-216`, `pending.rs:771-777`) and defers its incident edges to commit (`delete_implicit_edges`, `pending.rs:1325-1330`) |
| `Pending.delRels`  | `Pending::deleted_relationships`, read by `is_relationship_deleted` (`pending.rs:1129-1134`) |
| `expandIntoP`      | `ExpandIntoOp::expand_row` (`runtime/ops/expand_into.rs:125-238`): pairs `:176-177`; collapse `:197-215` and emit `:217-228` both skip `pending.is_relationship_deleted(id)` and sibling-used edges — and nothing else: no `is_node_deleted(src/dst)` check |
| `expandIntoFix`    | the fix: no row when an endpoint is pending-deleted |

`expandIntoP` with empty `Pending` is literally `CondTraverse.expandInto`
(`expandIntoP_empty`), so every earlier ExpandInto theorem is the
no-pending-deletes instance of this model.

## The bug

`CREATE (:A)-[:R]->(:B)`; `MATCH (a:A), (b:B) DETACH DELETE a MERGE (a)-[r:R]->(b)
RETURN count(r)`: Merge sits directly over ExpandInto (no Commit between), `a`
is in `deleted_nodes` but the edge `a→b` is not yet in `deleted_relationships`,
so ExpandInto matches it: Rust returns 1 with "Relationships created: 0". The
edge is doomed (it is deleted implicitly at commit). C creates a new edge from
the deleted node instead (dangling edge). Property `LiveExpandInto`; Lean
`bug3452_*`.

The issue's other half (the `synthetic_label` branch, `expand_into.rs:155-166`,
reads committed labels only) has no valid-Cypher route to pending label state
(a MATCH after a write needs `WITH`, which plans a Commit; MERGE cannot
re-declare a bound node's labels), so it is not claimed here.
-/

namespace OpsTraverse.EIP

open OpsTraverse.ASP (eq_of_mem_of_id)

/-- The part of `Pending` ExpandInto can observe: uncommitted deletes. -/
structure Pending where
  delNodes : List Nat
  delRels  : List Nat
deriving DecidableEq, Repr

/-- `ExpandIntoOp::expand_row` (relationship mode) with `Pending`
(`expand_into.rs:176-228`). -/
def expandIntoP (g : Graph) (p : Pending) (bidir emit : Bool) (src dst : Nat)
    (used : List Nat) : List Nat :=
  let ps := if bidir && src != dst then [(src, dst), (dst, src)] else [(src, dst)]
  ps.flatMap fun q =>
    let es := (edgesBetween g q.1 q.2).filter fun e =>
      !(p.delRels.contains e) && !(used.contains e)
    if emit then es else es.take 1

/-- With nothing pending, the model is the committed-graph `expandInto`. -/
theorem expandIntoP_empty (g : Graph) (b em : Bool) (s d : Nat) (u : List Nat) :
    expandIntoP g ⟨[], []⟩ b em s d u = expandInto g b em s d u := by
  simp [expandIntoP, expandInto]

/-- An edge is live in the effective graph: not deleted itself, and neither
endpoint deleted (a deleted node's edges go with it at commit). -/
def Live (g : Graph) (p : Pending) (e : Nat) : Prop :=
  e ∉ p.delRels ∧ ∀ x ∈ g, x.id = e → x.src ∉ p.delNodes ∧ x.dst ∉ p.delNodes

/-- **Property**: ExpandInto only matches edges of the effective graph. -/
def LiveExpandInto
    (op : Graph → Pending → Bool → Bool → Nat → Nat → List Nat → List Nat) : Prop :=
  ∀ g p b em s d u, g.WF → ∀ e ∈ op g p b em s d u, Live g p e

/-- The fix: `is_node_deleted(src) || is_node_deleted(dst)` ⇒ no match. -/
def expandIntoFix (g : Graph) (p : Pending) (bidir emit : Bool) (src dst : Nat)
    (used : List Nat) : List Nat :=
  if p.delNodes.contains src || p.delNodes.contains dst then []
  else expandIntoP g p bidir emit src dst used

theorem mem_expandIntoP {g : Graph} {p : Pending} {b em : Bool} {s d e : Nat}
    {u : List Nat} (h : e ∈ expandIntoP g p b em s d u) :
    e ∉ p.delRels ∧ e ∉ u ∧ ∃ x ∈ g, x.id = e ∧
      ((x.src = s ∧ x.dst = d) ∨ (x.src = d ∧ x.dst = s)) := by
  unfold expandIntoP at h
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

/-- **The fix satisfies the property.** -/
theorem expandIntoFix_live : LiveExpandInto expandIntoFix := by
  intro g p b em s d u hwf e he
  unfold expandIntoFix at he
  split at he
  · simp at he
  · rename_i hn
    simp only [Bool.or_eq_true, List.contains_eq_mem, decide_eq_true_eq, not_or] at hn
    obtain ⟨hdr, _, x, hx, hid, hdir⟩ := mem_expandIntoP he
    refine ⟨hdr, fun y hy hyid => ?_⟩
    have : y = x := eq_of_mem_of_id hwf hy hx (hyid.trans hid.symm)
    subst this
    rcases hdir with ⟨h1, h2⟩ | ⟨h1, h2⟩ <;> rw [h1, h2]
    · exact ⟨hn.1, hn.2⟩
    · exact ⟨hn.2, hn.1⟩

/-- Main already drops explicitly deleted edges. -/
theorem expandIntoP_not_delRel {g : Graph} {p : Pending} {b em : Bool} {s d e : Nat}
    {u : List Nat} (h : e ∈ expandIntoP g p b em s d u) : e ∉ p.delRels :=
  (mem_expandIntoP h).1

/-! ## #3452: `DETACH DELETE a MERGE (a)-[r:R]->(b)` -/

/-- `CREATE (:A)-[:R]->(:B)`: node `a = 0`, `b = 1`, edge `0`. -/
def gAB1 : Graph := [⟨0, 0, 1⟩]

/-- After `DETACH DELETE a` in the same segment: `a` pending-deleted, the edge
not (it is deleted implicitly at commit). -/
def pDetach : Pending := ⟨[0], []⟩

/-- Main's ExpandInto (the MERGE's match) finds the doomed edge, so MERGE
creates nothing and `count(r)` is 1. -/
theorem bug3452_matches_doomed_edge : expandIntoP gAB1 pDetach false true 0 1 [] = [0] := by
  decide

/-- Collapse mode (`r` unread) matches it too. -/
theorem bug3452_collapse_matches_doomed_edge :
    expandIntoP gAB1 pDetach false false 0 1 [] = [0] := by decide

/-- The fixed operator matches nothing for a pending-deleted endpoint. -/
theorem bug3452_fix_empty : expandIntoFix gAB1 pDetach false true 0 1 [] = [] := by decide

/-- **Main violates `LiveExpandInto`.** -/
theorem bug3452_main_violates : ¬ LiveExpandInto expandIntoP := by
  intro h
  have := h gAB1 pDetach false true 0 1 [] (by unfold Graph.WF; decide) 0
    (by rw [bug3452_matches_doomed_edge]; simp)
  exact (this.2 ⟨0, 0, 1⟩ (by simp [gAB1]) rfl).1 (by simp [pDetach])

end OpsTraverse.EIP
