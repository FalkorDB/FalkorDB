import FalkorBinder.Projection
/-
# Pattern binding order

This is the general cause behind issue #2923 (an inline property map that
references an earlier variable): `MATCH (a)-[r]->(b) MATCH (c:B {v: id(r)*0})`
fails with "'r' not defined".

* The parser ends a pattern on any token but `,` (cypher.rs): consecutive `MATCH` (and `CREATE`)
  clauses stay separate, so the second clause sees the first one's variables.
* `bind_graph` first defines every named relationship alias (binder.rs), then the nodes
  (1522-1534), then the relationships (1527-1706): a node's attrs see every relationship of the
  pattern, and the planner checks such attrs once all they read is bound
  (`lift_cross_entity_attrs`).

C (the reference) binds a pattern element by element in textual order: an
element's property map sees the scope before the clause, the elements to its
left, and itself.
-/
namespace FalkorBinder

/-- A written name stands for its `Alias::Named` (the test patterns below name every entity). -/
instance : Coe Name Alias := ⟨Alias.named⟩

/-- `Alias::name(..).is_some()` (ast.rs). -/
def Alias.isNamed : Alias → Bool
  | .named _ => true
  | .anon _ => false

/-- A pattern element: its `Alias` (a written name, or the parser's number for an unnamed
entity, which no attr can mention) and the names its inline property map reads. -/
inductive Elem
  | node (alias : Alias) (attrs : List Name)
  | rel  (alias : Alias) (attrs : List Name)
  deriving Repr, DecidableEq

def Elem.alias : Elem → Alias
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

def grow (scope : List Alias) (a : Alias) : List Alias :=
  if scope.contains a then scope else scope ++ [a]

theorem mem_grow {scope : List Alias} {n : Alias} (a : Alias) (h : n ∈ scope) : n ∈ grow scope a := by
  unfold grow; split <;> simp [h]

/-- Scope checking of one element (`define_name_in_scope`, then `bind_expr`
of its attrs): define its alias, then its attrs must be visible. -/
def step (scope : List Alias) (e : Elem) : Except String (List Alias) :=
  let s := grow scope e.alias
  if e.attrs.all (fun n => s.contains (.named n)) then .ok s
  else .error s!"'{(e.attrs.find? (fun n => !s.contains (.named n))).getD ""}' not defined"

def stepAll (scope : List Alias) : List Elem → Except String (List Alias)
  | [] => .ok scope
  | e :: es => do stepAll (← step scope e) es

/-- C / spec: textual order. -/
def specBind (scope : List Alias) (p : List Elem) : Except String (List Alias) :=
  stepAll scope p

/-- The named relationship aliases `bind_graph` defines first (binder.rs; an
unnamed one, `alias.name()` = `None`, is skipped). -/
def defRels (scope : List Alias) (p : List Elem) : List Alias :=
  ((split p).2.filter (fun e => e.alias.isNamed)).foldl (fun s e => grow s e.alias) scope

/-- `bind_graph` (binder.rs): named relationship aliases, then nodes, then
relationships (#2981; before it, nodes then relationships). A node `Elem`'s attrs are
those of every occurrence of its alias (`Parser::add_pattern_node` merges them for MATCH), so
a repeated endpoint that shares the node's bound map (`merged_occurrences`, binder.rs) checks no name the node step did not already check. -/
def rustBind (scope : List Alias) (p : List Elem) : Except String (List Alias) :=
  do stepAll (← stepAll (defRels scope p) (split p).1) (split p).2

/-- #2923's shape inside one MATCH: `(a)-[r]->(b), (c {v: id(r)})`. -/
def pRef : List Elem := [.node "a" [], .rel "r" [], .node "b" [], .node "c" ["r"]]

/-- **#2981**: a node's attrs see a relationship of the same MATCH, as in C. -/
theorem node_attr_on_rel_rust_accepts :
    (rustBind [] pRef).isOk = true ∧ (specBind [] pRef).isOk = true := by
  decide

/-- `(a)-[r]->(b {v: r.w - 2})`: C accepts, and since #2981 Rust too. -/
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

/-- Only named relationships are defined first (`alias.name().is_some()`): an unnamed one is
left to the relationship loop, and a written name is a name whatever its spelling, so
`(a {v: id(_anon_0)})-[_anon_0]->(b)` is accepted like any other later relationship. -/
theorem defRels_named_only :
    defRels [] [.node "a" [], .rel (.anon 0) [], .rel "_anon_0" [], .node "b" []] = [.named "_anon_0"] ∧
    (rustBind [] [.node "a" ["_anon_0"], .rel "_anon_0" [], .node "b" []]).isOk = true := by
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

theorem step_ok_of_outer {scope : List Alias} {e : Elem} (h : ∀ n ∈ e.attrs, Alias.named n ∈ scope) :
    step scope e = .ok (grow scope e.alias) := by
  unfold step
  have : e.attrs.all (fun n => (grow scope e.alias).contains n) = true := by
    rw [List.all_eq_true]; intro x hx; simpa using mem_grow e.alias (h x hx)
  simp only [this, ite_true]

theorem stepAll_ok_of_outer (scope : List Alias) (p : List Elem)
    (h : ∀ e ∈ p, ∀ n ∈ e.attrs, Alias.named n ∈ scope) : (stepAll scope p).isOk = true := by
  induction p generalizing scope with
  | nil => rfl
  | cons e es ih =>
    simp only [stepAll]
    rw [step_ok_of_outer (h e (by simp))]
    apply ih
    intro e' he' n hn
    exact mem_grow _ (h e' (by simp [he']) n hn)

theorem foldGrow_mono : ∀ (l : List Elem) (scope : List Alias) (n : Alias), n ∈ scope →
    n ∈ l.foldl (fun s e => grow s e.alias) scope
  | [], _, _, h => h
  | e :: es, scope, n, h => foldGrow_mono es _ n (mem_grow _ h)

/-- **Agreement.** When every property map only mentions variables bound
*before* the clause, the two orders accept the same patterns — so the
divergence is exactly intra-pattern references. -/
theorem agree_on_outer_refs (scope : List Alias) (p : List Elem)
    (h : ∀ e ∈ p, ∀ n ∈ e.attrs, Alias.named n ∈ scope) :
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
  stepAll_mono (scope : List Alias) (p : List Elem) (s : List Alias)
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

/-! ## Clause-by-clause scope checking -/

inductive RawClause
  | match_ (p : List Elem) (hasWhere : Bool)
  | with_ (items : List Name)
  deriving Repr, DecidableEq

/-- Clause-by-clause scope checking with a pattern binder. -/
def bindClauses (bindPat : List Alias → List Elem → Except String (List Alias))
    (scope : List Alias) : List RawClause → Except String (List Alias)
  | [] => .ok scope
  | .match_ p _ :: rest => do bindClauses bindPat (← bindPat scope p) rest
  | .with_ items :: rest => bindClauses bindPat (items.map Alias.named) rest

def q2923 : List RawClause :=
  [.match_ [.node "a" [], .rel "r" [], .node "b" []] false, .match_ [.node "c" ["r"]] false]

/-- **#2923** (fixed by #2981): the spec accepts, and so does the binder on the two clauses as
written (the second clause sees `r` from the first). -/
theorem q2923_fixed :
    (bindClauses specBind [] q2923).isOk = true ∧
    (bindClauses rustBind [] q2923).isOk = true := by
  decide

end FalkorBinder
