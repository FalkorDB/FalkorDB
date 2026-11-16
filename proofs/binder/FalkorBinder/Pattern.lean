import FalkorBinder.Projection
/-
# Pattern binding order, and the parser folding MATCH clauses together

This is the general cause behind issue #2923 (an inline property map that
references an earlier variable): `MATCH (a)-[r]->(b) MATCH (c:B {v: id(r)*0})`
fails with "'r' not defined".

* Before #2981 the parser folded consecutive `MATCH` (and `CREATE`) clauses
  into one pattern (`if token == *clause { next; continue }`), and a pattern
  *predicate*, parsed with the same routine, swallowed the next MATCH. #2981
  ends the pattern on any token but `,` (cypher.rs:1581-1587): `fold` below is
  the pre-#2981 parser, kept for the historical theorems.
* Before #2981 `bind_graph` defined and bound **all nodes** (with their inline
  attrs) before **any relationship**, so a node's attrs could only see
  relationships of *earlier clauses*. Since #2981 it first defines every named
  relationship alias (binder.rs:1380-1395), then the nodes (1404-1418), then
  the relationships (1421-1557): a node's attrs see every relationship of the
  pattern, and the planner checks such attrs once all they read is bound
  (`lift_cross_entity_attrs`).

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

/-- The named relationship aliases `bind_graph` defines first (binder.rs:1387-1395; an
`_anon` alias is skipped). -/
def defRels (scope : List Name) (p : List Elem) : List Name :=
  ((split p).2.filter (fun e => !e.alias.startsWith "_anon")).foldl (fun s e => grow s e.alias) scope

/-- `bind_graph` (binder.rs:1364): named relationship aliases, then nodes, then
relationships (#2981; before it, nodes then relationships). -/
def rustBind (scope : List Name) (p : List Elem) : Except String (List Name) :=
  do stepAll (← stepAll (defRels scope p) (split p).1) (split p).2

/-- #2923's shape inside one MATCH: `(a)-[r]->(b), (c {v: id(r)})`. -/
def pRef : List Elem := [.node "a" [], .rel "r" [], .node "b" [], .node "c" ["r"]]

/-- **Fixed by #2981** (pre-#2981 `node_attr_on_rel_rust_rejects`: Rust "'r' not defined"):
a node's attrs see a relationship of the same MATCH, as in C. -/
theorem node_attr_on_rel_rust_accepts :
    (rustBind [] pRef).isOk = true ∧ (specBind [] pRef).isOk = true := by
  decide

/-- `(a)-[r]->(b {v: r.w - 2})`: C accepts, and since #2981 Rust too (pre-#2981 `node_attr_on_own_rel`:
Rust "'r' not defined"). -/
theorem node_attr_on_own_rel_accepts :
    (rustBind [] [.node "a" [], .rel "r" [], .node "b" ["r"]]).isOk = true ∧
    (specBind [] [.node "a" [], .rel "r" [], .node "b" ["r"]]).isOk = true := by
  decide

/-- Since #2981 Rust also accepts a node whose attrs read a relationship that appears *later* in
the pattern (`(a {v: id(r)})-[r]->(b)`), where the textual order rejects it: the same reordering
as `rel_attr_forward_node`. The planner applies such an attr above the pattern. -/
theorem node_attr_later_rel :
    (rustBind [] [.node "a" ["r"], .rel "r" [], .node "b" []]).isOk = true ∧
    specBind [] [.node "a" ["r"], .rel "r" [], .node "b" []] = .error "'r' not defined" := by
  decide

/-- Nodes are still defined one by one: a node's attrs cannot read a node later in the pattern
(`(a {v: b.v})-[r]->(b)`), in Rust as in C. -/
theorem node_attr_later_node_rejected :
    rustBind [] [.node "a" ["b"], .rel "r" [], .node "b" []] = .error "'b' not defined" ∧
    specBind [] [.node "a" ["b"], .rel "r" [], .node "b" []] = .error "'b' not defined" := by
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

theorem foldGrow_mono : ∀ (l : List Elem) (scope : List Name) (n : Name), n ∈ scope →
    n ∈ l.foldl (fun s e => grow s e.alias) scope
  | [], _, _, h => h
  | e :: es, scope, n, h => foldGrow_mono es _ n (mem_grow _ h)

/-- **Agreement.** When every property map only mentions variables bound
*before* the clause, the two orders accept the same patterns — so the
divergence is exactly intra-pattern references. -/
theorem agree_on_outer_refs (scope : List Name) (p : List Elem)
    (h : ∀ e ∈ p, ∀ n ∈ e.attrs, n ∈ scope) :
    (rustBind scope p).isOk = true ∧ (specBind scope p).isOk = true := by
  refine ⟨?_, stepAll_ok_of_outer scope p h⟩
  unfold rustBind
  have hdef : ∀ n ∈ scope, n ∈ defRels scope p := fun n hn => foldGrow_mono _ scope n hn
  have h1 := stepAll_ok_of_outer (defRels scope p) (split p).1 (fun e he n hn =>
    hdef n (h e ((List.mem_filter.1 he).1) n hn))
  cases hs : stepAll (defRels scope p) (split p).1 with
  | error _ => rw [hs] at h1; cases h1
  | ok s1 =>
    simp only [bind, Except.bind]
    -- every name in `scope` survives into `s1`
    have hsub : ∀ n ∈ scope, n ∈ s1 := fun n hn => stepAll_mono _ _ s1 hs n (hdef n hn)
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

/-! ## The parser fold (pre-#2981) -/

inductive RawClause
  | match_ (p : List Elem) (hasWhere : Bool)
  | with_ (items : List Name)
  deriving Repr, DecidableEq

/-- `parse_pattern`'s loop before #2981 (`if token == *clause { next; continue }`, removed at
cypher.rs:1581-1587): a MATCH whose pattern was directly followed by `MATCH` absorbed it — WHERE comes after the pattern
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

/-- **#2923, general cause, fixed by #2981.** The spec accepts, and so does the binder on the
two clauses as written (the second clause sees `r` from the first), which is what the parser
hands it since #2981. Before it (`q2923_cause`) the parser folded them into one pattern
(`fold q2923`), which the nodes-first binder rejected with "'r' not defined"; the relationships-
first binder of #2981 accepts even the folded pattern. -/
theorem q2923_fixed :
    (bindClauses specBind [] q2923).isOk = true ∧
    (bindClauses rustBind [] q2923).isOk = true ∧
    (bindClauses rustBind [] (fold q2923)).isOk = true ∧
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
