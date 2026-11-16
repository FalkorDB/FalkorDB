import FalkorBinder.Pattern
import FalkorBinder.BindExpr
/-
# Scope tables across clauses: UNION, CALL {}, FOREACH, comprehensions, labels
-/
namespace FalkorBinder

/-! ## `merge_scope_vars` (binder.rs:2694-2706) -/

/-- Per scope, keep the longer table. Equal to the Rust loop: resizing with
empty tables and then replacing index `i < other.len()` when `other[i]` is
longer yields, at every index, the longer of the two (an index past either
list counts as empty). -/
def mergeSV : List (List Var) → List (List Var) → List (List Var)
  | [], b => b
  | a, [] => a
  | x :: xs, y :: ys => (if y.length > x.length then y else x) :: mergeSV xs ys

def lenAt (t : List (List Var)) (s : Nat) : Nat := (t.getD s []).length

theorem mergeSV_ge (a b : List (List Var)) (s : Nat) :
    lenAt a s ≤ lenAt (mergeSV a b) s ∧ lenAt b s ≤ lenAt (mergeSV a b) s := by
  induction a generalizing b s with
  | nil => simp [mergeSV, lenAt]
  | cons x xs ih =>
    cases b with
    | nil => simp [mergeSV, lenAt]
    | cons y ys =>
      cases s with
      | zero => simp only [mergeSV, lenAt, List.getD_cons_zero]; split <;> omega
      | succ s => simpa [mergeSV, lenAt] using ih ys s

/-- **Branches never collide with the planner.** If each UNION branch (or
retired CALL body) uses only ids below its own table lengths, the merged
table lengths cover all of them, so the planner's `len()`-minted ids are
fresh for every branch (`mint_fresh_iff`). -/
theorem mergeSV_covers (a b : List (List Var)) (s i : Nat)
    (h : i < lenAt a s ∨ i < lenAt b s) : i < lenAt (mergeSV a b) s := by
  have := mergeSV_ge a b s; omega

/-! ## FOREACH restore (binder.rs:851-866) -/

/-- The restored table: the saved outer table plus every inner entry whose
id the outer table does not hold, reserved nameless (`Scope::reserve`, which
lists an id once). Before #3016 they were re-keyed `_foreach_{scope}_{id}`. -/
def foreachRestore (saved inner : Env) (_scope : Nat) : Env :=
  (inner.filter (fun p => !(saved.ids.contains p.2.id))).foldl (fun e p => e.reserve p.2) saved

theorem rsvHas_false_of_not_mem (e : Env) (i : Nat) (h : i ∉ e.ids) : e.rsvHas i = false := by
  unfold Env.rsvHas
  rw [List.any_eq_false]
  intro p hp hc
  simp only [Bool.and_eq_true, beq_iff_eq] at hc
  exact h (hc.2 ▸ List.mem_map_of_mem hp)

/-- Reserving entries of pairwise distinct ids that the table does not hold
appends each one. -/
theorem foldl_reserve_eq (l : Env) :
    ∀ e : Env, (l.map (·.2.id)).Nodup → (∀ p ∈ l, p.2.id ∉ e.ids) →
      l.foldl (fun e p => e.reserve p.2) e = e ++ l.map (fun p => (Key.rsv, p.2)) := by
  induction l with
  | nil => intro e _ _; simp
  | cons q r ih =>
    intro e hn hd
    simp only [List.map_cons, List.nodup_cons] at hn
    simp only [List.foldl_cons]
    have h1 : e.reserve q.2 = e ++ [(.rsv, q.2)] := by
      simp [Env.reserve, rsvHas_false_of_not_mem e _ (hd q (by simp))]
    rw [h1, ih _ hn.2]
    · simp
    · intro p hp hm
      simp only [Env.ids, List.map_append, List.mem_append, List.map_cons, List.map_nil,
        List.mem_singleton] at hm
      rcases hm with hm | hm
      · exact hd p (by simp [hp]) hm
      · exact hn.1 (hm ▸ List.mem_map_of_mem hp)

/-- When the body only *added* entries, each minted at or above the outer
size with distinct ids (`inner = saved ++ added`; ids of a `Scope` are
distinct), the restored table holds exactly the inner ids and has the inner
size — so a covering inner table stays covering: no id used inside the
FOREACH is handed out again after it. -/
theorem foreachRestore_covers (saved added : Env) (scope : Nat)
    (hs : saved.Covers) (ha : ∀ p ∈ added, saved.length ≤ p.2.id)
    (hn : (added.map (·.2.id)).Nodup)
    (hi : (saved ++ added).Covers) :
    (foreachRestore saved (saved ++ added) scope).Covers ∧
    (foreachRestore saved (saved ++ added) scope).ids = (saved ++ added).ids := by
  have hS : saved.filter (fun p => !(saved.ids.contains p.2.id)) = [] := by
    rw [List.filter_eq_nil_iff]
    intro p hp
    simp [Env.ids]
    exact ⟨p.1, p.2, hp, rfl⟩
  have hnot : ∀ p ∈ added, p.2.id ∉ saved.ids := by
    intro p hp hm
    have := ha p hp
    have := hs _ hm
    omega
  have hA : added.filter (fun p => !(saved.ids.contains p.2.id)) = added := by
    rw [List.filter_eq_self]
    intro p hp
    simp only [Bool.not_eq_eq_eq_not, Bool.not_true, List.contains_eq_mem,
      decide_eq_false_iff_not]
    exact hnot p hp
  have heq : foreachRestore saved (saved ++ added) scope =
      saved ++ added.map (fun p => (Key.rsv, p.2)) := by
    simp only [foreachRestore, List.filter_append, hS, hA, List.nil_append]
    exact foldl_reserve_eq added saved hn hnot
  have hids : (foreachRestore saved (saved ++ added) scope).ids = (saved ++ added).ids := by
    rw [heq]; simp [Env.ids, List.map_map, Function.comp_def]
  have hlen : (foreachRestore saved (saved ++ added) scope).length = (saved ++ added).length := by
    rw [heq]; simp
  refine ⟨?_, hids⟩
  intro i hmem
  rw [hids] at hmem; rw [hlen]; exact hi i hmem

/-! ## Pattern comprehension / pattern predicate cleanup (binder.rs:2070-2071, 2245-2246) -/

/-- `MATCH (n:A) RETURN [(n)-[r]->(m) | m.v] AS l`, scope 0 while the
comprehension is bound: `bind_graph` mints `m ↦ 1`, `r ↦ 2`, which the
comprehension's sub-plan uses; the cleanup then keeps only the names that
were there before (and `_anon*`). -/
def compScope : Env :=
  [(.name "n", ⟨"n", 0, 0, .node⟩), (.name "m", ⟨"m", 1, 0, .node⟩), (.name "r", ⟨"r", 2, 0, .rel⟩)]

/-- The cleanup before #3016: `retain` dropped the entries. -/
def compCleaned : Env := compScope.retain (fun k => k == "n" || k.startsWith "_anon")

/-- Since #3016 (`hide_names_except`, binder.rs:417-428): the names go, the
slots stay reserved. -/
def compHidden : Env := hideExcept ["n"] compScope

/-- **Root cause 2 (comprehensions), fixed by #3016.** The hidden table keeps
`m` and `r` out of scope but keeps their slots: size 3, every id below it, so
the planner's and the next binder variable's fresh ids are new. -/
theorem comprehension_cleanup_covers :
    compScope.Covers ∧ compHidden.names = ["n"] ∧ compHidden.length = 3 ∧
    compHidden.Covers ∧ (∀ i ∈ compScope.ids, i < compHidden.length) := by
  refine ⟨?_, by decide, by decide, ?_, ?_⟩
  · intro i hi; simp [compScope, Env.ids] at hi; rcases hi with rfl | rfl | rfl <;> decide
  · intro i hi
    have : compHidden.ids = [0, 1, 2] := by decide
    rw [this] at hi; simp at hi; rcases hi with rfl | rfl | rfl <;> decide
  · intro i hi; simp [compScope, Env.ids] at hi; rcases hi with rfl | rfl | rfl <;> decide

/-- **Root cause 2 (comprehensions), before #3016.** The cleaned table had
size 1 while ids 1 and 2 were still in the IR: the planner's first two fresh
ids in scope 0 — or the next binder variable (`x`, `y` of a following
`MATCH (x)-->(y)`) — were exactly `m`'s and `r`'s. `mint_fresh_iff` says that
is a collision. -/
theorem pre3016_comprehension_cleanup_breaks :
    compScope.Covers ∧ compCleaned.length = 1 ∧
    ¬ (∀ i ∈ compScope.ids, i < compCleaned.length) := by
  refine ⟨?_, by decide, ?_⟩
  · intro i hi; simp [compScope, Env.ids] at hi; rcases hi with rfl | rfl | rfl <;> decide
  · intro h; exact absurd (h 1 (by decide)) (by decide)

/-! ## `CALL { … UNION … }` scope numbering (`bind_call_body`, binder.rs:824-925) -/

/-- The `env_stack` of a plain CALL body: the `outerLen` live outer scopes (only their number
matters here) with the body's own scope pushed on top (`self.env_stack.push(HashMap::new())`,
binder.rs:845). -/
def queryBodyStack (outerLen : Nat) : List Env := List.replicate outerLen [] ++ [[]]

/-- The `env_stack` of a UNION branch's own `Binder` since #3027:
`vec![HashMap::new(); self.env_stack.len() + 1]` (binder.rs:903), the outer scopes as empty
placeholders and one more on top. (Before #3027 it was a one-scope stack, so the branch's first
scope had the id of outer scope 0.) -/
def unionBranchStack (outerLen : Nat) : List Env := List.replicate (outerLen + 1) []

/-- Scope id of the first scope a CALL body binds into: the top of its `env_stack`
(`current_env`), with `outerLen = env_stack.len()` at the CALL. Each case is read off its own
stack construction, so a regression of either constructor changes the matching theorem. -/
def callBodyScope (outerLen : Nat) (isUnion : Bool) : Nat :=
  (if isUnion then unionBranchStack outerLen else queryBodyStack outerLen).length - 1

/-- A plain body's scopes are disjoint from the live outer scopes … -/
theorem callBody_query_disjoint (outerLen : Nat) (s : Nat) (h : s < outerLen) :
    callBodyScope outerLen false ≠ s := by
  simp [callBodyScope, queryBodyStack]; omega

/-- … and, since #3027 (fixes W2-binder-4, formerly `callBody_union_aliases`:
a UNION branch's first scope was outer scope 0), so are a UNION branch's. -/
theorem callBody_union_disjoint (outerLen : Nat) (s : Nat) (h : s < outerLen) :
    callBodyScope outerLen true ≠ s := by
  simp [callBodyScope, unionBranchStack]; omega

/-! ## Label accumulation (binder.rs:1356-1360, 122-154) -/

inductive LClause
  | match_ (labels : List Name)        -- MATCH (n:L…)
  | optMatch (labels : List Name)      -- OPTIONAL MATCH (n:L…): saved/restored (binder.rs:338-346)
  | predicate (labels : List Name)     -- WHERE (n:L)-->(): saved/restored (binder.rs:2090-2092)
  | create (labels : List Name)        -- CREATE (n:L)-[..]->(): not saved
  | comprehension (labels : List Name) -- [(n:L)-->() | …]: not saved
  | remove (labels : List Name)        -- REMOVE n:L…: `bind_ir` QueryIR::Remove (b84473a7f :477-482)
                                       -- binds the items and leaves `node_labels` alone (#3450)
  deriving Repr, DecidableEq

/-- The accumulated label set of one node variable after the clauses. -/
def accumulate (acc : List Name) : List LClause → List Name
  | [] => acc
  | .match_ ls :: r | .create ls :: r | .comprehension ls :: r =>
    accumulate (acc ++ ls.filter (fun l => !acc.contains l)) r
  | .optMatch _ :: r | .predicate _ :: r | .remove _ :: r => accumulate acc r

/-- `update_all_node_labels` writes the accumulation into the *first* MATCH's
scan. openCypher: that scan carries the labels of the (non-optional) MATCH
clauses only. -/
def specScanLabels : List LClause → List Name
  | [] => []
  | .match_ ls :: r => ls ++ (specScanLabels r).filter (fun l => !ls.contains l)
  | _ :: r => specScanLabels r

/-- Agreement when only MATCH / OPTIONAL MATCH / predicates mention labels. -/
theorem labels_agree_match_only :
    accumulate [] [.match_ ["A"], .optMatch ["B"], .predicate ["C"], .match_ ["D"]] =
      specScanLabels [.match_ ["A"], .optMatch ["B"], .predicate ["C"], .match_ ["D"]] := by
  decide

/-- **Root cause 5.** `MATCH (a) CREATE (a:Z)-[:T]->(:C)` scans `(a:Z)`, and
`MATCH (a) RETURN [(a:A)-->(b) | b.v]` scans `(a:A)`. -/
theorem labels_leak :
    accumulate [] [.match_ [], .create ["Z"]] = ["Z"] ∧ specScanLabels [.match_ [], .create ["Z"]] = [] ∧
    accumulate [] [.match_ [], .comprehension ["A"]] = ["A"] ∧
      specScanLabels [.match_ [], .comprehension ["A"]] = [] := by
  decide

/-! ## UNION column agreement (binder.rs:267-292) -/

def unionCheck : List (List Name) → Except String Unit
  | [] => .ok ()
  | first :: rest =>
    if rest.all (· == first) then .ok ()
    else .error "All sub queries in a UNION must have the same column names."

/-- UNION accepts iff every branch has the first branch's column list —
names *and order* (openCypher requires the same names; order-sensitivity
matches C, which also rejects `RETURN a, b UNION RETURN b, a`). -/
theorem unionCheck_ok_iff (first : List Name) (rest : List (List Name)) :
    unionCheck (first :: rest) = .ok () ↔ ∀ c ∈ rest, c = first := by
  simp only [unionCheck]
  split
  · rename_i h; simp only [true_iff]; intro c hc
    have := List.all_eq_true.1 h c hc; simpa using this
  · rename_i h; simp only [reduceCtorEq, false_iff]; intro hall; apply h
    rw [List.all_eq_true]; intro c hc; simp [hall c hc]

/-! ## CALL {} import WITH (binder.rs:1003-1056) -/

structure ImportItem where
  alias : Name
  /-- `Some v` iff the expression is a childless `ExprIR::Variable(v)` -/
  var : Option Name

def validateImport (hasModifiers star : Bool) (items : List ImportItem) (outer : List Name) :
    Except String (List Name) :=
  if hasModifiers then .error "import"
  else if star then .ok outer
  else items.foldlM (init := []) fun acc it =>
    match it.var with
    | some v => if v == it.alias then (if outer.contains v then .ok (acc ++ [v])
                                       else .error s!"'{v}' not defined")
                else .error "import"
    | none => .error "import"

/-- The import WITH accepts exactly `WITH *` or a list of `v AS v` (written
`WITH v`) of outer variables, with no DISTINCT/ORDER BY/SKIP/LIMIT/WHERE —
openCypher's importing-WITH rule. -/
theorem validateImport_ok (items : List ImportItem) (outer : List Name)
    (h : ∀ it ∈ items, it.var = some it.alias ∧ it.alias ∈ outer) :
    validateImport false false items outer = .ok (items.map (·.alias)) := by
  simp only [validateImport, Bool.false_eq_true, ite_false]
  suffices ∀ acc, items.foldlM (m := Except String) (init := acc) (fun acc it =>
      match it.var with
      | some v => if v == it.alias then (if outer.contains v then .ok (acc ++ [v])
                                         else .error s!"'{v}' not defined")
                  else .error "import"
      | none => .error "import") = .ok (acc ++ items.map (·.alias)) by
    simpa using this []
  induction items with
  | nil => intro acc; simp; rfl
  | cons it rest ih =>
    intro acc
    obtain ⟨hv, ho⟩ := h it (by simp)
    simp only [List.foldlM_cons, hv, beq_self_eq_true, ite_true, List.contains_eq_mem, ho,
      decide_true]
    have := ih (fun it' h' => h it' (by simp [h'])) (acc ++ [it.alias])
    simpa [bind, Except.bind] using this

theorem validateImport_rejects_expr :
    validateImport false false [⟨"m", some "n"⟩] ["n"] = .error "import" := by decide

end FalkorBinder
