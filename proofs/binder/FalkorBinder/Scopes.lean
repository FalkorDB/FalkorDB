import FalkorBinder.Pattern
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

/-! ## FOREACH restore (binder.rs:678-709) -/

/-- The restored table: the saved outer table plus every inner entry whose
id the outer table does not hold, re-keyed `_foreach_{scope}_{id}`. -/
def foreachRestore (saved inner : Env) (scope : Nat) : Env :=
  saved ++ (inner.filter (fun p => !(saved.ids.contains p.2.id))).map
    (fun p => (s!"_foreach_{scope}_{p.2.id}", p.2))

/-- When the body only *added* entries, each minted at or above the outer
size (`inner = saved ++ added`), the restored table holds exactly the inner
ids and has the inner size — so a covering inner table stays covering: no id
used inside the FOREACH is handed out again after it. -/
theorem foreachRestore_covers (saved added : Env) (scope : Nat)
    (hs : saved.Covers) (ha : ∀ p ∈ added, saved.length ≤ p.2.id)
    (hi : (saved ++ added).Covers) :
    (foreachRestore saved (saved ++ added) scope).Covers ∧
    (foreachRestore saved (saved ++ added) scope).ids = (saved ++ added).ids := by
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
  have hids : (foreachRestore saved (saved ++ added) scope).ids = (saved ++ added).ids := by
    simp only [foreachRestore, List.filter_append, hS, hA, List.nil_append]
    simp [Env.ids, List.map_map, Function.comp_def]
  have hlen : (foreachRestore saved (saved ++ added) scope).length = (saved ++ added).length := by
    simp only [foreachRestore, List.filter_append, hS, hA, List.nil_append]
    simp
  refine ⟨?_, hids⟩
  intro i hmem
  rw [hids] at hmem; rw [hlen]; exact hi i hmem

/-! ## Pattern comprehension / pattern predicate cleanup (binder.rs:1922, 2097) -/

/-- `MATCH (n:A) RETURN [(n)-[r]->(m) | m.v] AS l`, scope 0 while the
comprehension is bound: `bind_graph` mints `m ↦ 1`, `r ↦ 2`, which the
comprehension's sub-plan uses; the cleanup then keeps only the names that
were there before (and `_anon*`). -/
def compScope : Env :=
  [("n", ⟨"n", 0, 0, .node⟩), ("m", ⟨"m", 1, 0, .node⟩), ("r", ⟨"r", 2, 0, .rel⟩)]

def compCleaned : Env := compScope.retain (fun k => k == "n" || k.startsWith "_anon")

/-- **Root cause 2 (comprehensions).** The cleaned table has size 1 while
ids 1 and 2 are still in the IR: the planner's first two fresh ids in scope
0 — or the next binder variable (`x`, `y` of a following `MATCH (x)-->(y)`)
— are exactly `m`'s and `r`'s. `mint_fresh_iff` says that is a collision. -/
theorem comprehension_cleanup_breaks :
    compScope.Covers ∧ compCleaned.length = 1 ∧
    ¬ (∀ i ∈ compScope.ids, i < compCleaned.length) := by
  refine ⟨?_, by decide, ?_⟩
  · intro i hi; simp [compScope, Env.ids] at hi; rcases hi with rfl | rfl | rfl <;> decide
  · intro h; exact absurd (h 1 (by decide)) (by decide)

/-! ## `CALL { … UNION … }` scope numbering (binder.rs:845 vs 897-905) -/

/-- Scope id of the first scope of a CALL body, with `outerLen =
env_stack.len()` at the CALL. A plain body is pushed on the stack; each
UNION branch gets `Binder { env_stack: vec![HashMap::new()], .. }`. -/
def callBodyScope (outerLen : Nat) (isUnion : Bool) : Nat :=
  if isUnion then 0 else outerLen

/-- A plain body's scopes are disjoint from the live outer scopes … -/
theorem callBody_query_disjoint (outerLen : Nat) (s : Nat) (h : s < outerLen) :
    callBodyScope outerLen false ≠ s := by
  simp [callBodyScope]; omega

/-- … a UNION branch's first scope **is** outer scope 0 (root cause 3): its
first variable has id 0 in scope 0, the same slot as the outer query's first
variable. Rust: `MATCH (n:A) CALL { MATCH (q:B) RETURN q UNION MATCH (q:B)
RETURN q } RETURN n.v, q.v` returns `[1, 1]` (q reads n); C `[1,0],[1,5]`. -/
theorem callBody_union_aliases (outerLen : Nat) (h : 0 < outerLen) :
    callBodyScope outerLen true < outerLen := by
  simp [callBodyScope]; exact h

/-! ## Label accumulation (binder.rs:1356-1360, 122-154) -/

inductive LClause
  | match_ (labels : List Name)        -- MATCH (n:L…)
  | optMatch (labels : List Name)      -- OPTIONAL MATCH (n:L…): saved/restored (binder.rs:338-346)
  | predicate (labels : List Name)     -- WHERE (n:L)-->(): saved/restored (binder.rs:2090-2092)
  | create (labels : List Name)        -- CREATE (n:L)-[..]->(): not saved
  | comprehension (labels : List Name) -- [(n:L)-->() | …]: not saved
  deriving Repr, DecidableEq

/-- The accumulated label set of one node variable after the clauses. -/
def accumulate (acc : List Name) : List LClause → List Name
  | [] => acc
  | .match_ ls :: r | .create ls :: r | .comprehension ls :: r =>
    accumulate (acc ++ ls.filter (fun l => !acc.contains l)) r
  | .optMatch _ :: r | .predicate _ :: r => accumulate acc r

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
