import FalkorBinder.Scopes
/-!
# `REMOVE n:L` and the accumulated label set across `WITH` (#3450) — gap closed 2026-10-22

`accumulate` / `specScanLabels` assumed labels only accumulate: `LClause` had no
REMOVE, so a label removed between clauses was invisible to the model. Rust source of
truth: origin/main 5c7af54f2 (the bug found on b84473a7f), `graph/src/planner/binder.rs`.

* `bind_ir` `QueryIR::Remove` (:477-482) binds the items and never touches `node_labels`.
* `bind_projection` copies the parent variable's label set to the projected child
  (`WITH a`, :1135-1146 / :1162-1173), so the child starts from `accumulate before`.
* A later `MATCH (a)…` extends the child's set (`bind_graph` :1360-1366), and
  `update_all_node_labels` (:122-153, called from `bind` :104) stamps the whole set onto
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

**Fixed by #3457** (PR head 75d5cdfd2): `bind_ir`'s REMOVE arm (binder.rs:504-514) adds the
labels `removed_node_labels` (:105-119) reads off each item to the query-wide
`removed_labels`, a UNION branch's set is merged into the caller's (`bind_call_body`
:991-995), and every projection (`bind_projection` :1161-1163 / :1185-1187 / :1206-1208,
`build_import_projections` :1048-1050) goes through `carry_node_labels` (:1020-1034), which
copies the parent's set minus every removed label. `rsStamp` is that code; `others` is the
part of `removed_labels` that comes from other variables or UNION branches (the set is per
label, never per variable). The pre-#3457 stamp is kept as `rsStampPre3457`.
`rsStamp_held`: on every clause list without a pattern comprehension the stamp satisfies
`StampedLabelsHeld`, whatever `others` is; `rsStamp_comprehension_leak` is what is left, the
pre-existing root cause 5 (`labels_leak`), not a REMOVE case.
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

/-- Before #3457: the label set `update_all_node_labels` stamps on the first MATCH after
`WITH`: the child inherits `accumulate [] before` (binder.rs:1162-1173 at b84473a7f) and the
MATCH extends it (:1360-1366). -/
def rsStampPre3457 (before : List LClause) (own : List Name) : List Name :=
  accumulate (accumulate [] before) [.match_ own]

/-- The labels the REMOVE clauses add to `removed_labels` (`bind_ir` binder.rs:509-513). -/
def removedIn : List LClause → List Name
  | [] => []
  | .remove ls :: r => ls ++ removedIn r
  | _ :: r => removedIn r

/-- `carry_node_labels` (binder.rs:1020-1034): no entry for `from` → nothing written;
else `to` gets `from`'s set minus every label in `removed_labels`. -/
def carryNodeLabels (removed : List Name) (fromLabels : Option (List Name)) : Option (List Name) :=
  fromLabels.map (·.filter (fun l => !removed.contains l))

/-- The label set stamped on the first MATCH after `WITH` (#3457): the child inherits
`carry_node_labels` of the parent's accumulated set, `removed_labels` holding this
variable's removed labels plus `others` (other variables', UNION branches'). -/
def rsStamp (others : List Name) (before : List LClause) (own : List Name) : List Name :=
  accumulate ((carryNodeLabels (others ++ removedIn before) (some (accumulate [] before))).getD [])
    [.match_ own]

/-- The fixed stamp: the MATCH's own labels plus what is still held. -/
def specStamp (before : List LClause) (own : List Name) : List Name :=
  own ++ (held [] before).filter (fun l => !own.contains l)

/-- **Property `StampedLabelsHeld`**: a label stamped onto a post-`WITH` MATCH node is one
that MATCH wrote or one the node still carries at that point. -/
def StampedLabelsHeld (stamp : List LClause → List Name → List Name) : Prop :=
  ∀ before own, ∀ l ∈ stamp before own, l ∈ own ∨ l ∈ held [] before

/-- The fixed stamp satisfies the property. **Caveat: this holds by construction.** `specStamp`
is defined from the same `held` that `StampedLabelsHeld` mentions, so the theorem only checks
the bookkeeping of that definition; it does not show that `held` models what `binder.rs` and the
runtime do for REMOVE (that REMOVE clears a label point-in-time while `bind_ir`'s REMOVE arm
leaves `node_labels` alone is read off the Rust by inspection). The evidence for the bug is the
refutation `bug3450_remove_label_kept_across_with` of the *Rust* stamp `rsStamp`, which is
independent of `specStamp`. -/
theorem specStamp_held : StampedLabelsHeld specStamp := by
  intro before own l hl
  simp only [specStamp, List.mem_append, List.mem_filter] at hl
  rcases hl with h | ⟨h, _⟩
  · exact .inl h
  · exact .inr h

/-- Without a REMOVE the two stamps agree on the case below (no regression for the
accumulation the planner relies on). -/
theorem stamp_agree_no_remove :
    rsStamp [] [.match_ ["A"]] [] = specStamp [.match_ ["A"]] [] := by decide

/-- **BUG #3450 (binder half), before #3457**: `MATCH (a:A) REMOVE a:A WITH a MATCH
(a)-[:R]->(b)` stamped `(a:A)` on the second MATCH although `a` no longer has `A`;
`CondTraverse` then checks `:A` and drops the row (Rust 0 rows, C 1). Historical: fixed by
#3457 (75d5cdfd2), see `fixed3457_remove_label_dropped`. -/
theorem bug3450_remove_label_kept_across_with :
    rsStampPre3457 [.match_ ["A"], .remove ["A"]] [] = ["A"] ∧
    held [] [.match_ ["A"], .remove ["A"]] = [] ∧
    specStamp [.match_ ["A"], .remove ["A"]] [] = [] ∧
    ¬ StampedLabelsHeld rsStampPre3457 := by
  refine ⟨by decide, by decide, by decide, fun h => ?_⟩
  have := h [.match_ ["A"], .remove ["A"]] [] "A" (by decide)
  revert this; decide

/-! ## #3457 -/

/-- Labels a clause list adds to the accumulated set (MATCH, CREATE, comprehension). -/
def addedIn : List LClause → List Name
  | [] => []
  | .match_ ls :: r | .create ls :: r | .comprehension ls :: r => ls ++ addedIn r
  | _ :: r => addedIn r

/-- Labels a clause list adds to `held` (MATCH, CREATE). -/
def addedHeld : List LClause → List Name
  | [] => []
  | .match_ ls :: r | .create ls :: r => ls ++ addedHeld r
  | _ :: r => addedHeld r

/-- No pattern comprehension mentions the variable. -/
def noComp : List LClause → Bool
  | [] => true
  | .comprehension _ :: _ => false
  | _ :: r => noComp r

theorem mem_accumulate (l : Name) : ∀ (cs : List LClause) (acc : List Name),
    l ∈ accumulate acc cs ↔ l ∈ acc ∨ l ∈ addedIn cs
  | [], acc => by simp [accumulate, addedIn]
  | .match_ ls :: r, acc | .create ls :: r, acc | .comprehension ls :: r, acc => by
    simp only [accumulate, addedIn, mem_accumulate l r, List.mem_append, List.mem_filter]
    by_cases h : l ∈ acc <;> simp [h]
  | .optMatch _ :: r, acc | .predicate _ :: r, acc | .remove _ :: r, acc => by
    simp only [accumulate, addedIn, mem_accumulate l r]

theorem mem_held (l : Name) : ∀ (cs : List LClause) (acc : List Name), l ∉ removedIn cs →
    (l ∈ acc ∨ l ∈ addedHeld cs) → l ∈ held acc cs
  | [], acc, _, h => by simpa [held, addedHeld] using h
  | .match_ ls :: r, acc, hr, h | .create ls :: r, acc, hr, h => by
    simp only [held]
    apply mem_held l r _ (by simpa [removedIn] using hr)
    simp only [addedHeld, List.mem_append, List.mem_filter] at h ⊢
    by_cases ha : l ∈ acc
    · exact .inl (.inl ha)
    · rcases h with h | h | h
      · exact absurd h ha
      · exact .inl (.inr ⟨h, by simpa using ha⟩)
      · exact .inr h
  | .remove ls :: r, acc, hr, h => by
    simp only [removedIn, List.mem_append, not_or] at hr
    simp only [held]
    apply mem_held l r _ hr.2
    simp only [addedHeld, List.mem_filter] at h ⊢
    rcases h with h | h
    · exact .inl ⟨h, by simpa using hr.1⟩
    · exact .inr h
  | .optMatch _ :: r, acc, hr, h | .predicate _ :: r, acc, hr, h
  | .comprehension _ :: r, acc, hr, h => by
    simp only [held]
    exact mem_held l r _ (by simpa [removedIn] using hr) (by simpa [addedHeld] using h)

theorem addedIn_noComp : ∀ (cs : List LClause), noComp cs = true → addedIn cs = addedHeld cs
  | [], _ => rfl
  | .match_ ls :: r, h | .create ls :: r, h => by
    simp only [addedIn, addedHeld, addedIn_noComp r (by simpa [noComp] using h)]
  | .optMatch _ :: r, h | .predicate _ :: r, h | .remove _ :: r, h => by
    simp only [addedIn, addedHeld, addedIn_noComp r (by simpa [noComp] using h)]
  | .comprehension _ :: _, h => by simp [noComp] at h

/-- **#3457, the fix**: with the query-wide `removed_labels` (any `others`), every label
stamped on the post-`WITH` MATCH is one it wrote or one the node still holds, on every
clause list without a pattern comprehension (that one is the pre-existing root cause 5). -/
theorem rsStamp_held (others : List Name) (before : List LClause) (own : List Name)
    (hc : noComp before = true) : ∀ l ∈ rsStamp others before own, l ∈ own ∨ l ∈ held [] before := by
  intro l hl
  simp only [rsStamp, carryNodeLabels, Option.map_some, Option.getD_some, mem_accumulate,
    addedIn, List.mem_append, List.not_mem_nil, or_false, List.mem_filter] at hl
  rcases hl with ⟨h, hr⟩ | h
  · right
    have hr' : l ∉ removedIn before := by
      intro hm; simp [hm] at hr
    rw [addedIn_noComp _ hc] at h
    exact mem_held l before [] hr' (by simpa using h)
  · exact .inl h

/-- The #3450 query is fixed: `MATCH (a:A) REMOVE a:A WITH a MATCH (a)-[:R]->(b)` stamps
no label on the second MATCH's `(a)`. -/
theorem fixed3457_remove_label_dropped :
    rsStamp [] [.match_ ["A"], .remove ["A"]] [] = [] ∧
    rsStamp [] [.match_ ["A"], .remove ["A"]] ["A"] = ["A"] := by decide

/-- Without any REMOVE in the query the new stamp is the old one (no regression). -/
theorem rsStamp_no_remove (before : List LClause) (own : List Name) (h : removedIn before = []) :
    rsStamp [] before own = rsStampPre3457 before own := by
  simp only [rsStamp, rsStampPre3457, carryNodeLabels, h, List.append_nil, Option.map_some,
    Option.getD_some]
  congr 1
  exact List.filter_eq_self.2 (by simp)

/-- `removed_labels` is per label: a label removed from another variable (`MATCH (a:A),
(c:A) REMOVE c:A WITH a …`) is dropped from `a`'s carried set too — sound (fewer stamped
labels), only a lost redundant check. -/
theorem rsStamp_other_variable :
    rsStamp ["A"] [.match_ ["A"]] [] = [] ∧ held [] [.match_ ["A"]] = ["A"] := by decide

/-- `carry_node_labels` only ever drops labels, and copies the set when nothing was removed. -/
theorem carryNodeLabels_sub (removed : List Name) (ls : List Name) :
    ∀ c, carryNodeLabels removed (some ls) = some c → ∀ l ∈ c, l ∈ ls ∧ l ∉ removed := by
  intro c h l hl
  simp only [carryNodeLabels, Option.map_some, Option.some.injEq] at h
  subst h
  simpa using hl

theorem carryNodeLabels_none (removed : List Name) : carryNodeLabels removed none = none := rfl

theorem carryNodeLabels_nil (o : Option (List Name)) : carryNodeLabels [] o = o := by
  cases o <;> simp [carryNodeLabels]

/-- What #3457 leaves: the comprehension leak (root cause 5, `labels_leak`) still breaks
`StampedLabelsHeld`; it does not involve REMOVE and is not a regression
(`rsStampPre3457` stamps the same). -/
theorem rsStamp_comprehension_leak :
    rsStamp [] [.comprehension ["A"]] [] = ["A"] ∧
    rsStampPre3457 [.comprehension ["A"]] [] = ["A"] ∧ ¬ StampedLabelsHeld (rsStamp []) := by
  refine ⟨by decide, by decide, fun h => ?_⟩
  have := h [.comprehension ["A"]] [] "A" (by decide)
  revert this; decide

end FalkorBinder
