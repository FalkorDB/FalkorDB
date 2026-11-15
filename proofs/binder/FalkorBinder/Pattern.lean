import FalkorBinder.Projection
/-
# Pattern binding order, and the parser folding MATCH clauses together

This is the general cause behind issue #2923 (an inline property map that
references an earlier variable): `MATCH (a)-[r]->(b) MATCH (c:B {v: id(r)*0})`
fails with "'r' not defined".

* The parser folds consecutive `MATCH` (and `CREATE`) clauses into one
  pattern: after a pattern, `if token == *clause { next; continue }`
  (parser/cypher.rs:1540-1547). A pattern *predicate* is parsed with the same
  routine and `Keyword::Match` (cypher.rs:2102), so `WHERE (n)-->() MATCH (x)`
  also swallows the next clause.
* `bind_graph` defines and binds **all nodes** (with their inline attrs)
  before **any relationship** (binder.rs:1351-1365, then 1368-1520), so a
  node's attrs can only see relationships of *earlier clauses*.

C (the reference) binds a pattern element by element in textual order: an
element's property map sees the scope before the clause, the elements to its
left, and itself.
-/
namespace FalkorBinder

inductive Elem
  | node (alias : Name) (attrs : List Name)
  | rel  (alias : Name) (attrs : List Name)
  deriving Repr, DecidableEq

def Elem.alias : Elem → Name
  | .node a _ => a
  | .rel a _ => a

def Elem.attrs : Elem → List Name
  | .node _ as => as
  | .rel _ as => as

def Elem.isNode : Elem → Bool
  | .node .. => true
  | .rel .. => false

/-- What the parser hands the binder: `QueryGraph.nodes()` in first-occurrence
order and `relationships()` in textual order — the interleaving is lost. -/
def split (p : List Elem) : List Elem × List Elem :=
  (p.filter (·.isNode), p.filter (fun e => !e.isNode))

def grow (scope : List Name) (a : Name) : List Name :=
  if scope.contains a then scope else scope ++ [a]

theorem mem_grow {scope : List Name} {n : Name} (a : Name) (h : n ∈ scope) : n ∈ grow scope a := by
  unfold grow; split <;> simp [h]

/-- Scope checking of one element (`define_name_in_scope`, then `bind_expr`
of its attrs): define its alias, then its attrs must be visible. -/
def step (scope : List Name) (e : Elem) : Except String (List Name) :=
  let s := grow scope e.alias
  if e.attrs.all (fun n => s.contains n) then .ok s
  else .error s!"'{(e.attrs.find? (fun n => !s.contains n)).getD ""}' not defined"

def stepAll (scope : List Name) : List Elem → Except String (List Name)
  | [] => .ok scope
  | e :: es => do stepAll (← step scope e) es

/-- C / spec: textual order. -/
def specBind (scope : List Name) (p : List Elem) : Except String (List Name) :=
  stepAll scope p

/-- `bind_graph` (binder.rs:1328): nodes, then relationships. -/
def rustBind (scope : List Name) (p : List Elem) : Except String (List Name) :=
  do stepAll (← stepAll scope (split p).1) (split p).2

/-- #2923's shape inside one MATCH: `(a)-[r]->(b), (c {v: id(r)})`. -/
def pRef : List Elem := [.node "a" [], .rel "r" [], .node "b" [], .node "c" ["r"]]

theorem node_attr_on_rel_rust_rejects :
    rustBind [] pRef = .error "'r' not defined" ∧ (specBind [] pRef).isOk = true := by
  decide

/-- `(a)-[r]->(b {v: r.w - 2})`: C accepts, Rust rejects. -/
theorem node_attr_on_own_rel :
    rustBind [] [.node "a" [], .rel "r" [], .node "b" ["r"]] = .error "'r' not defined" ∧
    (specBind [] [.node "a" [], .rel "r" [], .node "b" ["r"]]).isOk = true := by
  decide

/-- `(a)-[r {w: b.v + 2}]->(b)`: C rejects ("'b' not defined"), Rust
accepts — the other direction of the same reordering. -/
theorem rel_attr_forward_node :
    (rustBind [] [.node "a" [], .rel "r" ["b"], .node "b" []]).isOk = true ∧
    specBind [] [.node "a" [], .rel "r" ["b"], .node "b" []] = .error "'b' not defined" := by
  decide

/-- Both orders accept a self-reference (`MATCH (a {v: a.v})`). -/
theorem self_reference_ok :
    (rustBind [] [.node "a" ["a"]]).isOk = true ∧ (specBind [] [.node "a" ["a"]]).isOk = true := by
  decide

theorem step_ok_of_outer {scope : List Name} {e : Elem} (h : ∀ n ∈ e.attrs, n ∈ scope) :
    step scope e = .ok (grow scope e.alias) := by
  unfold step
  have : e.attrs.all (fun n => (grow scope e.alias).contains n) = true := by
    rw [List.all_eq_true]; intro x hx; simpa using mem_grow e.alias (h x hx)
  simp only [this, ite_true]

theorem stepAll_ok_of_outer (scope : List Name) (p : List Elem)
    (h : ∀ e ∈ p, ∀ n ∈ e.attrs, n ∈ scope) : (stepAll scope p).isOk = true := by
  induction p generalizing scope with
  | nil => rfl
  | cons e es ih =>
    simp only [stepAll]
    rw [step_ok_of_outer (h e (by simp))]
    apply ih
    intro e' he' n hn
    exact mem_grow _ (h e' (by simp [he']) n hn)

/-- **Agreement.** When every property map only mentions variables bound
*before* the clause, the two orders accept the same patterns — so the
divergence is exactly intra-pattern references. -/
theorem agree_on_outer_refs (scope : List Name) (p : List Elem)
    (h : ∀ e ∈ p, ∀ n ∈ e.attrs, n ∈ scope) :
    (rustBind scope p).isOk = true ∧ (specBind scope p).isOk = true := by
  refine ⟨?_, stepAll_ok_of_outer scope p h⟩
  unfold rustBind
  have h1 := stepAll_ok_of_outer scope (split p).1 (fun e he n hn =>
    h e ((List.mem_filter.1 he).1) n hn)
  cases hs : stepAll scope (split p).1 with
  | error _ => rw [hs] at h1; cases h1
  | ok s1 =>
    simp only [bind, Except.bind]
    -- every name in `scope` survives into `s1`
    have hsub : ∀ n ∈ scope, n ∈ s1 := stepAll_mono scope _ s1 hs
    exact stepAll_ok_of_outer s1 _ (fun e he n hn =>
      hsub n (h e ((List.mem_filter.1 he).1) n hn))
where
  stepAll_mono (scope : List Name) (p : List Elem) (s : List Name)
      (hs : stepAll scope p = .ok s) : ∀ n ∈ scope, n ∈ s := by
    induction p generalizing scope with
    | nil => simp [stepAll] at hs; subst hs; exact fun _ h => h
    | cons e es ih =>
      simp only [stepAll] at hs
      cases h1 : step scope e with
      | error _ => rw [h1] at hs; cases hs
      | ok s' =>
        rw [h1] at hs
        have := ih s' hs
        intro n hn
        apply this
        unfold step at h1
        dsimp only at h1
        split at h1
        · cases h1; exact mem_grow _ hn
        · cases h1

/-! ## The parser fold -/

inductive RawClause
  | match_ (p : List Elem) (hasWhere : Bool)
  | with_ (items : List Name)
  deriving Repr, DecidableEq

/-- `parse_pattern`'s loop (cypher.rs:1540-1547): a MATCH whose pattern is
directly followed by `MATCH` absorbs it — WHERE comes after the pattern
(`parse_match_clause`, cypher.rs:1109-1117), so only a WHERE-less clause
absorbs; the absorbed clause's own WHERE becomes the merged clause's. -/
def fold : List RawClause → List RawClause
  | [] => []
  | c :: rest =>
    match c, fold rest with
    | .match_ p false, .match_ q w :: rest' => .match_ (p ++ q) w :: rest'
    | c, r => c :: r

/-- Clause-by-clause scope checking with a pattern binder. -/
def bindClauses (bindPat : List Name → List Elem → Except String (List Name))
    (scope : List Name) : List RawClause → Except String (List Name)
  | [] => .ok scope
  | .match_ p _ :: rest => do bindClauses bindPat (← bindPat scope p) rest
  | .with_ items :: rest => bindClauses bindPat items rest

def q2923 : List RawClause :=
  [.match_ [.node "a" [], .rel "r" [], .node "b" []] false, .match_ [.node "c" ["r"]] false]

/-- **#2923, general cause.** The spec accepts; the binder *would* accept the
two clauses as written (the second clause sees `r` from the first); it is the
fold into one pattern, bound nodes-first, that rejects it. Neither half alone
does: with a `WITH` between the clauses (no fold) the query binds. -/
theorem q2923_cause :
    (bindClauses specBind [] q2923).isOk = true ∧
    (bindClauses rustBind [] q2923).isOk = true ∧
    bindClauses rustBind [] (fold q2923) = .error "'r' not defined" ∧
    (bindClauses specBind [] (fold q2923)).isOk = true := by
  decide

/-- The same fold is what makes relationship uniqueness (a per-MATCH rule in
openCypher) apply across clauses — `MATCH (a)-[r]->(b) MATCH (a)-[s]->(b)`
gets one uniqueness group — and what turns `OPTIONAL MATCH A MATCH B` into
`OPTIONAL MATCH A, B`: the fold does not preserve the clause list. -/
theorem fold_merges :
    fold [.match_ [.rel "r" []] false, .match_ [.rel "s" []] false] =
      [.match_ [.rel "r" [], .rel "s" []] false] := by
  decide

/-- A WHERE (or any other clause) between them keeps the boundary. -/
theorem fold_keeps_where :
    fold [.match_ [.rel "r" []] true, .match_ [.rel "s" []] false] =
      [.match_ [.rel "r" []] true, .match_ [.rel "s" []] false] := by
  decide

end FalkorBinder
