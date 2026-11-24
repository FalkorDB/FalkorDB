import FalkorBinder.Pattern
import FalkorBinder.BindExpr
/-
# Scope tables across clauses: UNION, CALL {}, FOREACH, comprehensions, labels
-/
namespace FalkorBinder

/-! ## `merge_scope_vars` (binder.rs) -/

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

/-! ## FOREACH restore (binder.rs) -/

/-- The restored table (binder.rs): the saved outer table, then every inner variable whose id the
outer table does not hold, reserved nameless (`Scope::reserve`, #3016; before, re-keyed
`_foreach_{scope}_{id}`). -/
def foreachRestore (saved inner : Env) : Env :=
  (inner.filter (fun p => !(saved.ids.contains p.2.id))).foldl (fun e p => e.reserve p.2) saved

/-- Reserving variables whose ids the table does not hold, and that are pairwise distinct, appends
them, nameless. -/
theorem Env.foldl_reserve_fresh : ∀ (l : Env) (e : Env), (e.ids ++ l.map (·.2.id)).Nodup →
    l.foldl (fun e p => e.reserve p.2) e = e ++ l.map (fun p => (Key.un, p.2))
  | [], e, _ => by simp
  | p :: l, e, h => by
    have hni : p.2.id ∉ e.unnamedIds := fun hm => by
      have h1 := Env.unnamedIds_sub _ _ hm
      exact (List.nodup_append.1 h).2.2 _ h1 _ (by simp) rfl
    simp only [List.foldl_cons, Env.reserve_eq_append hni]
    rw [Env.foldl_reserve_fresh l _ (by simpa [Env.ids] using h)]
    simp

/-- When the body only *added* entries, each minted at or above the outer
size (`inner = saved ++ added`), the restored table holds exactly the inner
ids and has the inner size — so a covering inner table stays covering: no id
used inside the FOREACH is handed out again after it. Since `reserve` lists an id once
(#3016), this needs the scope invariant that the inner table lists each id once
(`Env.WF`; `lift` unreserves before naming, `hide_names` moves rather than copies). -/
theorem foreachRestore_covers (saved added : Env)
    (hs : saved.Covers) (ha : ∀ p ∈ added, saved.length ≤ p.2.id)
    (hi : (saved ++ added).Covers) (hn : (saved ++ added).ids.Nodup) :
    (foreachRestore saved (saved ++ added)).Covers ∧
    (foreachRestore saved (saved ++ added)).ids = (saved ++ added).ids := by
  have hS : saved.filter (fun p => !(saved.ids.contains p.2.id)) = [] := by
    rw [List.filter_eq_nil_iff]
    intro p hp
    simp [Env.ids]
    exact ⟨p.1, p.2, hp, rfl⟩
  have hA : added.filter (fun p => !(saved.ids.contains p.2.id)) = added := by
    rw [List.filter_eq_self]
    intro p hp
    have := ha p hp
    simp only [Bool.not_eq_eq_eq_not, Bool.not_true, List.contains_eq_mem,
      decide_eq_false_iff_not]
    intro hm
    have := hs _ hm
    omega
  have hr : foreachRestore saved (saved ++ added) = saved ++ added.map (fun p => (Key.un, p.2)) := by
    simp only [foreachRestore, List.filter_append, hS, hA, List.nil_append]
    exact Env.foldl_reserve_fresh added saved (by simpa [Env.ids] using hn)
  have hids : (foreachRestore saved (saved ++ added)).ids = (saved ++ added).ids := by
    rw [hr]; simp [Env.ids]
  have hlen : (foreachRestore saved (saved ++ added)).length = (saved ++ added).length := by
    rw [hr]; simp
  refine ⟨?_, hids⟩
  intro i hmem
  rw [hids] at hmem; rw [hlen]; exact hi i hmem

/-! ## Pattern comprehension / pattern predicate cleanup (binder.rs) -/

/-- `MATCH (n:A) RETURN [(n)-[r]->(m) | m.v] AS l`, scope 0 while the
comprehension is bound: `bind_graph` mints `m ↦ 1`, `r ↦ 2`, which the
comprehension's sub-plan uses; the cleanup then hides the names that were not there before
(`hide_names_except`, #3016). -/
def compScope : Env :=
  [("n", ⟨"n", 0, 0, .node, none⟩), ("m", ⟨"m", 1, 0, .node, none⟩), ("r", ⟨"r", 2, 0, .rel, none⟩)]

def compCleaned : Env := cleanup ["n"] [] compScope

/-- **The cleanup keeps the pattern's slots** (#3016, fixes #3004 for comprehensions;
before, `m` and `r` were removed, the table shrank to size 1 and the next variables of scope 0 —
the planner's or `x`, `y` of a following `MATCH (x)-->(y)` — got `m`'s and `r`'s ids). Now only
`n` is reachable by name, the table still holds ids 0, 1, 2 and the next id is 3. -/
theorem comprehension_cleanup_covers :
    compScope.Covers ∧ compCleaned.keys = ["n"] ∧ compCleaned.length = 3 ∧
    (∀ i ∈ compScope.ids, i < compCleaned.length) ∧ compCleaned.Covers ∧
    (freshVar compCleaned "x" .node 0).id = 3 := by
  have hc : compScope.Covers := by
    intro i hi; simp [compScope, Env.ids] at hi; rcases hi with rfl | rfl | rfl <;> decide
  have hids : compCleaned.ids = [0, 1, 2] := by decide
  refine ⟨hc, by decide, by decide, ?_, ?_, by decide⟩
  · intro i hi; simp [compScope, Env.ids] at hi; rcases hi with rfl | rfl | rfl <;> decide
  · intro i hi; rw [hids] at hi; simp at hi; rcases hi with rfl | rfl | rfl <;> decide

/-! ## `CALL { … UNION … }` scope numbering (`bind_call_body`, binder.rs) -/

/-- The `env_stack` of a plain CALL body: the `outerLen` live outer scopes (only their number
matters here) with the body's own scope pushed on top (`self.env_stack.push(HashMap::new())`,
binder.rs). -/
def queryBodyStack (outerLen : Nat) : List Env := List.replicate outerLen [] ++ [[]]

/-- The `env_stack` of a UNION branch's own `Binder` since #3027:
`vec![HashMap::new(); self.env_stack.len() + 1]` (binder.rs), the outer scopes as empty
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

/-! ## Label accumulation (binder.rs; the rollback in `with_node_labels_rolled_back`) -/

inductive LClause
  | match_ (labels : List Name)        -- MATCH (n:L…)
  | optMatch (labels : List Name)      -- OPTIONAL MATCH (n:L…): saved/restored (binder.rs)
  | predicate (labels : List Name)     -- WHERE (n:L)-->(): saved/restored (binder.rs)
  | create (labels : List Name)        -- CREATE (n:L)-[..]->(): `with_node_labels_rolled_back(true, …)` (:552)
  | comprehension (labels : List Name) -- [(n:L)-->() | …]: `with_node_labels_rolled_back(false, …)` (:2104)
  | remove (labels : List Name)        -- REMOVE n:L…: `bind_ir` QueryIR::Remove (b84473a7f :477-482)
                                       -- binds the items and leaves `node_labels` alone (#3450)
  deriving Repr, DecidableEq

/-- `acc ∪ ls` as `OrderSet::extend` does it (binder.rs). -/
def addLabels (acc ls : List Name) : List Name := acc ++ ls.filter (fun l => !acc.contains l)

/-- The accumulated label set of one node variable after the clauses (since #3049).
`bound`: the variable is already in `current_env()` when the clause runs (bound by an earlier
MATCH, OPTIONAL MATCH or CREATE). `with_node_labels_rolled_back` (binder.rs) restores
`node_labels` after a CREATE for every in-scope key and keeps only the entries of nodes the CREATE
declared itself (`keep_new = true`); after a pattern comprehension it restores everything
(`keep_new = false`). -/
def accumulateFrom (bound : Bool) (acc : List Name) : List LClause → List Name
  | [] => acc
  | .match_ ls :: r => accumulateFrom true (addLabels acc ls) r
  | .create ls :: r =>
    if bound then accumulateFrom true acc r else accumulateFrom true (addLabels acc ls) r
  | .optMatch _ :: r => accumulateFrom true acc r
  | .predicate _ :: r | .comprehension _ :: r | .remove _ :: r => accumulateFrom bound acc r

/-- Accumulation for a variable not yet in scope, starting from `acc`. -/
def accumulate (acc : List Name) (cs : List LClause) : List Name := accumulateFrom false acc cs

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

def LClause.isMatch : LClause → Bool
  | .match_ _ => true
  | _ => false

/-- Since #3049: once a variable is in scope only its MATCH clauses add to its label set;
CREATE, OPTIONAL MATCH, predicates, comprehensions and REMOVE leave it alone. -/
theorem accumulate_bound_match_only (acc : List Name) (cs : List LClause) :
    accumulateFrom true acc cs = accumulateFrom true acc (cs.filter LClause.isMatch) := by
  induction cs generalizing acc with
  | nil => rfl
  | cons c r ih =>
    cases c <;> simp [accumulateFrom, LClause.isMatch, List.filter_cons, ih]

/-- The two repros of W2-binder-6 (and the OPTIONAL MATCH one of test21) now scan with the
spec's labels. -/
theorem labels_leak_fixed :
    accumulate [] [.match_ [], .create ["Z"]] = specScanLabels [.match_ [], .create ["Z"]] ∧
    accumulate [] [.match_ [], .comprehension ["A"]] =
      specScanLabels [.match_ [], .comprehension ["A"]] ∧
    accumulate [] [.optMatch [], .create ["Z"]] = [] ∧
    accumulate [] [.match_ ["A"], .create ["Z"], .comprehension ["B"], .match_ ["D"]] =
      specScanLabels [.match_ ["A"], .create ["Z"], .comprehension ["B"], .match_ ["D"]] := by
  decide

/-- A node the CREATE declares itself keeps its labels (`keep_new`), so a later MERGE on it
still sees them (VariableAlreadyBound check, binder.rs). -/
theorem create_new_keeps_labels :
    accumulate [] [.create ["Z"]] = ["Z"] ∧ accumulate [] [.create ["Z"], .create ["Y"]] = ["Z"] := by
  decide

/-! ## UNION column agreement (binder.rs) -/

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

/-! ## CALL {} import WITH (binder.rs) -/

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
