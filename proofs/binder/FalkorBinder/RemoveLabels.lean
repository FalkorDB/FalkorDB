import FalkorBinder.Stack
/-!
# `REMOVE n:L` and the accumulated label set (#3450, fixed by #3457)

Rust source of truth: `graph/src/planner/binder.rs` (PR #3457).

* `removed_node_labels` reads the labels of a bound `REMOVE n:L1:L2` item (a `hasLabels`
  call); `bind_ir`'s `QueryIR::Remove` arm adds them to the binder-wide `removed_labels`
  (per label, not per variable: `n` may be reachable under other names). `node_labels` itself
  is left alone, so the first MATCH of `n` still scans with the full accumulated set.
* `carry_node_labels` (called by `bind_projection` for `WITH`/`RETURN` and by
  `build_import_projections` for a `CALL {}` import) gives the projected variable the
  accumulated labels of its source minus `removed_labels`.
* `bind_call_body` adds every UNION branch's `removed_labels` to the outer binder's.
* `drop_removed_labels` (the MERGE arm of `bind_ir`) drops removed labels from the bound MERGE
  pattern unless the clause writes them itself, and `update_all_node_labels`, once a label was
  removed, stamps CREATE/MERGE nodes with their own labels plus the accumulated ones that were
  not removed (`union = true`).

`rsStamp before own extra` is the label set stamped on the node of the first MATCH after the
`WITH`: `before` are the clauses ahead of the `WITH` (as seen by this variable), `own` the
labels written in that MATCH and `extra` the labels other variables' REMOVEs added to the
binder-wide set. The property `StampedLabelsHeld`: every stamped label is one the MATCH wrote,
or one the node is guaranteed to still carry when the MATCH runs (`held`, point-in-time: MATCH
and CREATE add, REMOVE removes; OPTIONAL MATCH, predicates and comprehensions guarantee
nothing). The `WITH` before the MATCH commits the REMOVE.

Repro of the fixed bug, after `CREATE (:A)-[:R]->(:B)`:
`MATCH (a:A) REMOVE a:A WITH a MATCH (a)-[:R]->(b) RETURN labels(b)`: before #3457 0 rows,
now `[B]` as in C (`remove_label_dropped_across_with`).
-/
namespace FalkorBinder

/-! ## `removed_node_labels` -/

/-- A child of the `hasLabels` label list: a string constant or anything else. -/
inductive LArg
  | str (s : Name)
  | other
  deriving Repr, DecidableEq

def LArg.str? : LArg → Option Name
  | .str s => some s
  | .other => none

/-- The root of a bound REMOVE item: a function call `name(children…)`, each child given by
its own children (only child 1's, the label list, are read), or any other expression. -/
inductive RItem
  | func (name : String) (children : List (List LArg))
  | other
  deriving Repr

/-- `removed_node_labels`: `Some` the string constants of child 1 of a `hasLabels` call,
`None` for a property removal or a malformed call without child 1. -/
def removedNodeLabels : RItem → Option (List Name)
  | .func n cs => if n = "hasLabels" then (cs[1]?).map (·.filterMap LArg.str?) else none
  | .other => none

theorem removedNodeLabels_spec :
    removedNodeLabels (.func "hasLabels" [[], [.str "A", .str "B"]]) = some ["A", "B"] ∧
    removedNodeLabels (.func "hasLabels" [[]]) = none ∧
    removedNodeLabels (.func "hasLabels" [[], [.str "A", .other]]) = some ["A"] ∧
    removedNodeLabels .other = none := by decide

theorem removedNodeLabels_not_hasLabels (n : String) (cs : List (List LArg)) (h : n ≠ "hasLabels") :
    removedNodeLabels (.func n cs) = none := by
  simp [removedNodeLabels, h]

/-- Every label returned is a string constant of the label list. -/
theorem removedNodeLabels_mem (it : RItem) (ls : List Name) (h : removedNodeLabels it = some ls)
    (l : Name) (hl : l ∈ ls) : ∃ n cs, it = .func n cs ∧ ∃ c, cs[1]? = some c ∧ LArg.str l ∈ c := by
  cases it with
  | other => simp [removedNodeLabels] at h
  | func n cs =>
    simp only [removedNodeLabels] at h
    split at h
    · cases hc : cs[1]? with
      | none => simp [hc] at h
      | some c =>
        rw [hc] at h
        simp only [Option.map_some, Option.some.injEq] at h
        subst h
        obtain ⟨a, ha, he⟩ := List.mem_filterMap.1 hl
        cases a with
        | other => simp [LArg.str?] at he
        | str s =>
          simp only [LArg.str?, Option.some.injEq] at he
          subst he
          exact ⟨n, cs, rfl, c, hc, ha⟩
    · cases h

/-! ## `bind_ir` `QueryIR::Remove`: the binder-wide `removed_labels` -/

def recordRemoved (removed : List Name) : List RItem → List Name
  | [] => removed
  | it :: r => match removedNodeLabels it with
    | some ls => recordRemoved (addLabels removed ls) r
    | none => recordRemoved removed r

theorem mem_addLabels (acc ls : List Name) (l : Name) : l ∈ addLabels acc ls ↔ l ∈ acc ∨ l ∈ ls := by
  simp only [addLabels, List.mem_append, List.mem_filter]
  constructor
  · rintro (h | ⟨h, _⟩)
    · exact .inl h
    · exact .inr h
  · rintro (h | h)
    · exact .inl h
    · by_cases ha : l ∈ acc
      · exact .inl ha
      · exact .inr ⟨h, by simpa using ha⟩

theorem recordRemoved_mono (removed : List Name) (its : List RItem) (l : Name) (h : l ∈ removed) :
    l ∈ recordRemoved removed its := by
  induction its generalizing removed with
  | nil => exact h
  | cons it r ih =>
    simp only [recordRemoved]
    split
    · exact ih _ ((mem_addLabels _ _ _).2 (.inl h))
    · exact ih _ h

/-- After the REMOVE every label of every `n:L…` item is in `removed_labels`. -/
theorem recordRemoved_covers (removed : List Name) (its : List RItem) (it : RItem) (hit : it ∈ its)
    (ls : List Name) (h : removedNodeLabels it = some ls) (l : Name) (hl : l ∈ ls) :
    l ∈ recordRemoved removed its := by
  induction its generalizing removed with
  | nil => cases hit
  | cons i r ih =>
    rcases List.mem_cons.1 hit with e | e
    · subst e
      simp only [recordRemoved, h]
      exact recordRemoved_mono _ _ _ ((mem_addLabels _ _ _).2 (.inr hl))
    · simp only [recordRemoved]
      split
      · exact ih _ e
      · exact ih _ e

/-! ## `bind_call_body` (UNION): branches' removals reach the outer binder -/

def unionRemoved (outer : List Name) (branches : List (List Name)) : List Name :=
  branches.foldl addLabels outer

theorem unionRemoved_covers (outer : List Name) (branches : List (List Name)) (l : Name) :
    (l ∈ outer ∨ ∃ b ∈ branches, l ∈ b) → l ∈ unionRemoved outer branches := by
  induction branches generalizing outer with
  | nil => rintro (h | ⟨b, hb, _⟩); exact h; cases hb
  | cons b bs ih =>
    rintro (h | ⟨c, hc, hl⟩)
    · exact ih _ (.inl ((mem_addLabels _ _ _).2 (.inl h)))
    · rcases List.mem_cons.1 hc with e | e
      · subst e; exact ih _ (.inl ((mem_addLabels _ _ _).2 (.inr hl)))
      · exact ih _ (.inr ⟨c, e, hl⟩)

/-! ## `carry_node_labels` -/

/-- The labels the projected variable gets: its source's accumulated set minus the removed
labels. -/
def heldLabels (removed acc : List Name) : List Name := acc.filter (fun l => !removed.contains l)

def carryNodeLabels (nl : List (VKey × List Name)) (removed : List Name) (src dst : VKey) :
    List (VKey × List Name) :=
  match nl.lookup src with
  | some ls => (dst, heldLabels removed ls) :: nl
  | none => nl

theorem carry_dst (nl : List (VKey × List Name)) (removed : List Name) (src dst : VKey)
    (ls : List Name) (h : nl.lookup src = some ls) :
    (carryNodeLabels nl removed src dst).lookup dst = some (heldLabels removed ls) := by
  simp [carryNodeLabels, h]

theorem carry_none (nl : List (VKey × List Name)) (removed : List Name) (src dst : VKey)
    (h : nl.lookup src = none) : carryNodeLabels nl removed src dst = nl := by
  simp [carryNodeLabels, h]

theorem mem_heldLabels (removed acc : List Name) (l : Name) :
    l ∈ heldLabels removed acc ↔ l ∈ acc ∧ l ∉ removed := by
  simp [heldLabels]

/-- Without a REMOVE the carry copies the set as before #3457. -/
theorem heldLabels_nil (acc : List Name) : heldLabels [] acc = acc := by simp [heldLabels]

/-! ## The stamp on the first MATCH after `WITH` -/

/-- Labels the node is guaranteed to carry after the clauses (point-in-time). -/
def held (acc : List Name) : List LClause → List Name
  | [] => acc
  | .match_ ls :: r | .create ls :: r => held (acc ++ ls.filter (fun l => !acc.contains l)) r
  | .remove ls :: r => held (acc.filter (fun l => !ls.contains l)) r
  | .optMatch _ :: r | .predicate _ :: r | .comprehension _ :: r => held acc r

/-- The labels this variable's REMOVEs record (`bind_ir` QueryIR::Remove). -/
def removedIn : List LClause → List Name
  | [] => []
  | .remove ls :: r => ls ++ removedIn r
  | _ :: r => removedIn r

/-- The stamp `update_all_node_labels` puts on the first MATCH after `WITH`: the child gets
`carry_node_labels` of `accumulate [] before` (with the binder-wide removed set, this
variable's removals plus `extra`) and the MATCH extends it (`bind_graph`). -/
def rsStamp (before : List LClause) (own extra : List Name) : List Name :=
  accumulate (heldLabels (removedIn before ++ extra) (accumulate [] before)) [.match_ own]

/-- **Property `StampedLabelsHeld`**: a label stamped onto a post-`WITH` MATCH node is one
that MATCH wrote or one the node still carries at that point. -/
def StampedLabelsHeld (stamp : List LClause → List Name → List Name) : Prop :=
  ∀ before own, ∀ l ∈ stamp before own, l ∈ own ∨ l ∈ held [] before

theorem held_sub (l : Name) (b : Bool) (acc acc' : List Name) (cs : List LClause)
    (hr : l ∉ removedIn cs) (h : l ∈ acc → l ∈ acc') :
    l ∈ accumulateFrom b acc cs → l ∈ held acc' cs := by
  induction cs generalizing b acc acc' with
  | nil => exact h
  | cons c r ih =>
    have step : ∀ ls : List Name, l ∈ addLabels acc ls → l ∈ acc' ++ ls.filter (fun l => !acc'.contains l) :=
      fun ls hl => (mem_addLabels acc' ls l).2 (((mem_addLabels _ _ _).1 hl).imp h id)
    cases c with
    | match_ ls => exact ih _ _ _ hr (step ls)
    | create ls =>
      simp only [accumulateFrom]
      split
      · exact ih _ _ _ hr fun hl => (mem_addLabels acc' ls l).2 (.inl (h hl))
      · exact ih _ _ _ hr (step ls)
    | optMatch ls => exact ih _ _ _ hr h
    | predicate ls => exact ih _ _ _ hr h
    | comprehension ls => exact ih _ _ _ hr h
    | remove ls =>
      simp only [removedIn, List.mem_append, not_or] at hr
      exact ih _ _ _ hr.2 fun hl => List.mem_filter.2 ⟨h hl, by simpa using hr.1⟩

/-- **#3457 fixes #3450 (binder half)**: whatever other variables removed (`extra`), the
stamp is held. -/
theorem rsStamp_held (extra : List Name) : StampedLabelsHeld (fun b o => rsStamp b o extra) := by
  intro before own l hl
  simp only [rsStamp, accumulate, accumulateFrom] at hl
  rcases (mem_addLabels _ _ _).1 hl with h | h
  · obtain ⟨ha, hr⟩ := (mem_heldLabels _ _ _).1 h
    simp only [List.mem_append, not_or] at hr
    exact .inr (held_sub l false [] [] before hr.1 id ha)
  · exact .inl h

/-- The repro of #3450: `MATCH (a:A) REMOVE a:A WITH a MATCH (a)-[:R]->(b)` no longer stamps
`(a:A)` on the second MATCH (before #3457 the stamp was `["A"]`: 0 rows, C 1), and a label the
MATCH writes itself is still stamped. -/
theorem remove_label_dropped_across_with :
    rsStamp [.match_ ["A"], .remove ["A"]] [] [] = [] ∧
    rsStamp [.match_ ["A"], .remove ["A"]] ["A"] [] = ["A"] ∧
    held [] [.match_ ["A"], .remove ["A"]] = [] := by decide

/-- `WITH n AS a, n AS c REMOVE c:A WITH a MATCH (a)…`: the removal through the other name
(`extra`) drops the label too. -/
theorem remove_through_alias_dropped :
    rsStamp [.match_ ["A"]] [] ["A"] = [] := by decide

/-- Without a REMOVE the stamp is the accumulation as before #3457. -/
theorem stamp_no_remove (before : List LClause) (own : List Name) (h : removedIn before = []) :
    rsStamp before own [] = accumulate (accumulate [] before) [.match_ own] := by
  simp [rsStamp, h, heldLabels_nil]

/-! ## MERGE after REMOVE: `drop_removed_labels` and the CREATE/MERGE stamp -/

/-- `drop_removed_labels`, per bound node: keep a label unless it was removed and the clause
does not write it (`written`: the raw pattern's labels for the node's alias, `none` if the
alias is not in the raw pattern). -/
def dropLabels (removed : List Name) (written : Option (List Name)) (labels : List Name) :
    List Name :=
  labels.filter fun l => !removed.contains l || (written.map (·.contains l)).getD false

theorem dropLabels_nil (w : Option (List Name)) (labels : List Name) :
    dropLabels [] w labels = labels := by simp [dropLabels]

theorem dropLabels_sub (removed : List Name) (w : Option (List Name)) (labels : List Name)
    (l : Name) (h : l ∈ dropLabels removed w labels) : l ∈ labels :=
  (List.mem_filter.1 h).1

/-- A removed label survives only if the MERGE writes it. -/
theorem dropLabels_removed_written (removed : List Name) (w : Option (List Name))
    (labels : List Name) (l : Name) (h : l ∈ dropLabels removed w labels) (hr : l ∈ removed) :
    ∃ ws, w = some ws ∧ l ∈ ws := by
  have h2 := (List.mem_filter.1 h).2
  have hc : removed.contains l = true := List.contains_iff_mem.2 hr
  simp only [hc, Bool.not_true, Bool.false_or] at h2
  cases w with
  | none => simp at h2
  | some ws => exact ⟨ws, rfl, by simpa using h2⟩

/-- A label the MERGE writes is kept. -/
theorem dropLabels_keeps_written (removed : List Name) (ws labels : List Name) (l : Name)
    (h : l ∈ labels) (hw : l ∈ ws) : l ∈ dropLabels removed (some ws) labels := by
  simp only [dropLabels, List.mem_filter, h, true_and]
  simp [hw]

/-- `update_all_node_labels` on a CREATE/MERGE: the accumulated map as before while no label
was removed; else its held part, merged into the clause's own labels. -/
def cmMap (removed : List Name) (nl : List (VKey × List Name)) : List (VKey × List Name) :=
  if removed.isEmpty then nl else nl.map fun p => (p.1, heldLabels removed p.2)

def cmStamp (removed : List Name) (nl : List (VKey × List Name)) (n : VKey × List Name) :
    VKey × List Name :=
  updNode (cmMap removed nl) (!removed.isEmpty) n

theorem cmStamp_no_remove (nl : List (VKey × List Name)) (n : VKey × List Name) :
    cmStamp [] nl n = updNode nl false n := rfl

theorem lookup_map_held (removed : List Name) (nl : List (VKey × List Name)) (k : VKey) :
    (nl.map fun p => (p.1, heldLabels removed p.2)).lookup k =
      (nl.lookup k).map (heldLabels removed) := by
  induction nl with
  | nil => rfl
  | cons p t ih =>
    simp only [List.map_cons, List.lookup]
    split <;> simp_all

/-- Once a label was removed, a CREATE/MERGE node is stamped only with its own labels and
accumulated labels that were not removed. -/
theorem cmStamp_removed_own (removed : List Name) (nl : List (VKey × List Name))
    (n : VKey × List Name) (hne : removed ≠ []) (l : Name) (hl : l ∈ (cmStamp removed nl n).2)
    (hr : l ∈ removed) : l ∈ n.2 := by
  have hne' : removed.isEmpty = false := by cases removed <;> simp_all
  simp only [cmStamp, cmMap, hne', Bool.not_false, Bool.false_eq_true, ↓reduceIte, updNode,
    lookup_map_held] at hl
  cases hk : nl.lookup n.1 with
  | none => simp only [hk, Option.map_none] at hl; exact hl
  | some acc =>
    simp only [hk, Option.map_some, mergedLabels, ↓reduceIte, List.mem_append,
      List.mem_filter] at hl
    rcases hl with h | ⟨h, _⟩
    · exact h
    · exact absurd hr ((mem_heldLabels _ _ _).1 h).2

/-- **MERGE after `REMOVE n:L`** (#3457): a removed label is stamped on a MERGE node only if
the MERGE writes it itself (`drop_removed_labels`, then the stamp), so
`MATCH (a:A), (b:B) REMOVE a:A MERGE (a)-[:R]->(b)` matches the existing edge. -/
theorem merge_removed_label_written (removed : List Name) (nl : List (VKey × List Name))
    (k : VKey) (w : Option (List Name)) (labels : List Name) (hne : removed ≠ []) (l : Name)
    (hl : l ∈ (cmStamp removed nl (k, dropLabels removed w labels)).2) (hr : l ∈ removed) :
    ∃ ws, w = some ws ∧ l ∈ ws :=
  dropLabels_removed_written removed w labels l
    (cmStamp_removed_own removed nl _ hne l hl hr) hr

theorem merge_after_remove_example :
    cmStamp ["A"] [((0, 0), ["A"])] ((0, 0), dropLabels ["A"] (some []) ["A"]) = ((0, 0), []) ∧
    cmStamp ["A"] [((1, 0), ["A"])] ((1, 0), dropLabels ["A"] (some ["A"]) ["A"]) =
      ((1, 0), ["A"]) := by decide

end FalkorBinder
