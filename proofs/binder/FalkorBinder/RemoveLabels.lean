import FalkorBinder.Scopes
/-!
# `REMOVE n:L` and the accumulated label set across `WITH` (#3450) — gap closed 2026-10-22

`accumulate` / `specScanLabels` assumed labels only accumulate: `LClause` had no
REMOVE, so a label removed between clauses was invisible to the model. Rust source of
truth: origin/main b84473a7f, `graph/src/planner/binder.rs`.

* `bind_ir` `QueryIR::Remove` (:477-482) binds the items and never touches `node_labels`.
* `bind_projection` copies the parent variable's label set to the projected child
  (`WITH a`, :1132-1141 / :1159-1168), so the child starts from `accumulate before`.
* A later `MATCH (a)…` extends the child's set (`bind_graph` :1356-1360), and
  `update_all_node_labels` (:122-134, called from `bind` :104) stamps the whole set onto
  that MATCH's node, turning `(a)` into `(a:A)`.

`rsStamp before own` is the label set stamped on the node of the first MATCH after the
`WITH`, `before` the clauses ahead of the `WITH` (as seen by this variable) and `own` the
labels written in that MATCH. The property: every stamped label is one the MATCH wrote, or
one the node is guaranteed to still carry when the MATCH runs (`held`, point-in-time: MATCH
and CREATE add, REMOVE removes; OPTIONAL MATCH, predicates and comprehensions guarantee
nothing). The `WITH` before the MATCH commits the REMOVE, so the MATCH reads the graph
without `L`.

Live (main b84473a7f vs C), after `CREATE (:A)-[:R]->(:B)`:
`MATCH (a:A) REMOVE a:A WITH a MATCH (a)-[:R]->(b) RETURN labels(b)`: Rust 0 rows, C `[B]`.
(The CREATE/MERGE half of the issue — `CREATE (a) WITH a MATCH (a:Person)` — does not
reproduce: 0 rows in both, and `CREATE (a) MATCH (a:Person)` is a parse error in both.)
-/
namespace FalkorBinder

/-- Labels the node is guaranteed to carry after the clauses (point-in-time). -/
def held (acc : List Name) : List LClause → List Name
  | [] => acc
  | .match_ ls :: r | .create ls :: r => held (acc ++ ls.filter (fun l => !acc.contains l)) r
  | .remove ls :: r => held (acc.filter (fun l => !ls.contains l)) r
  | .optMatch _ :: r | .predicate _ :: r | .comprehension _ :: r => held acc r

/-- The label set `update_all_node_labels` stamps on the first MATCH after `WITH`: the
child inherits `accumulate [] before` (binder.rs:1159-1168) and the MATCH extends it
(:1356-1360). -/
def rsStamp (before : List LClause) (own : List Name) : List Name :=
  accumulate (accumulate [] before) [.match_ own]

/-- The fixed stamp: the MATCH's own labels plus what is still held. -/
def specStamp (before : List LClause) (own : List Name) : List Name :=
  own ++ (held [] before).filter (fun l => !own.contains l)

/-- **Property `StampedLabelsHeld`**: a label stamped onto a post-`WITH` MATCH node is one
that MATCH wrote or one the node still carries at that point. -/
def StampedLabelsHeld (stamp : List LClause → List Name → List Name) : Prop :=
  ∀ before own, ∀ l ∈ stamp before own, l ∈ own ∨ l ∈ held [] before

theorem specStamp_held : StampedLabelsHeld specStamp := by
  intro before own l hl
  simp only [specStamp, List.mem_append, List.mem_filter] at hl
  rcases hl with h | ⟨h, _⟩
  · exact .inl h
  · exact .inr h

/-- Without a REMOVE the two stamps agree on the case below (no regression for the
accumulation the planner relies on). -/
theorem stamp_agree_no_remove :
    rsStamp [.match_ ["A"]] [] = specStamp [.match_ ["A"]] [] := by decide

/-- **BUG #3450 (binder half)**: `MATCH (a:A) REMOVE a:A WITH a MATCH (a)-[:R]->(b)` stamps
`(a:A)` on the second MATCH although `a` no longer has `A`; `CondTraverse` then checks
`:A` and drops the row (Rust 0 rows, C 1). -/
theorem bug3450_remove_label_kept_across_with :
    rsStamp [.match_ ["A"], .remove ["A"]] [] = ["A"] ∧
    held [] [.match_ ["A"], .remove ["A"]] = [] ∧
    specStamp [.match_ ["A"], .remove ["A"]] [] = [] ∧
    ¬ StampedLabelsHeld rsStamp := by
  refine ⟨by decide, by decide, by decide, fun h => ?_⟩
  have := h [.match_ ["A"], .remove ["A"]] [] "A" (by decide)
  revert this; decide

end FalkorBinder
