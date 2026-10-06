/-
# parser/ast.rs: variables, pattern graphs, validation

| here | there |
| --- | --- |
| `AV`, `AV.fmt`, `AV.eqv`, `AV.hashKey`, `AV.asStr` | `Variable` Display/PartialEq/Hash/`as_str` ast.rs:92-137 |
| `QN`, `QR`, `QP`, `QG` | `QueryNode`, `QueryRelationship`, `QueryPath`, `QueryGraph` ast.rs:473-640 |
| `fmtNode`, `fmtRel`, `fmtGraph` | their `Display` ast.rs:481-489, 542-566, 643-660 |
| `mergeAttrs` | `merge_attr_maps` ast.rs:667-683 |
| `addNode`, `mergeNode`, `replaceNode`, `addRel`, `addPath`, `variables`, accessors | ast.rs:685-791 |
| `filterVisited` | `filter_visited` ast.rs:795-821 |
| `dfs`, `components` | `dfs` ast.rs:844-883, `connected_components` ast.rs:823-842 |
| `QC`, `innerValidate`, `validateSetItems`, `inlinedOk`, `returnColumns` | `validate` / `inner_validate` / `validate_set_items` / `validate_inlined_properties` / `return_column_names` ast.rs:1151-1393 |

* CONFIRMED bug (`call_skips_rest`): `inner_validate`'s procedure-call arm
  (ast.rs:1166-1206) returns `Ok(())` without validating the clauses after it, so
  `CALL db.labels() YIELD label MATCH (n)` reaches the optimizer with a scan at the
  plan root and the server **panics** (`utilize_node_by_id.rs:118` `parent().unwrap()`);
  `… UNWIND [1] AS x` is accepted, `… CREATE (a)-[:R|S]->(b)` creates an edge; C rejects all.
* `with_last_accepted`: a query ending in `WITH` is accepted (`MATCH (n) WITH n` returns
  rows); C rejects it ("Query cannot conclude with WITH"), as openCypher does.
* `validate_ends`: away from those, a query cannot end in MATCH / UNWIND / a
  returning subquery.
-/
namespace PlannerBuild.Ast

/-! ## `Variable` -/

structure AV where
  name : Option String
  id : Nat
  scope : Nat
  deriving DecidableEq, Repr

def AV.fmt (v : AV) : String := match v.name with
  | some n => n
  | none => "?" ++ toString v.id

/-- `PartialEq for Variable`: by id only (ast.rs:113-119). -/
def AV.eqv (a b : AV) : Bool := a.id == b.id

/-- `Hash for Variable`: hashes the id only (ast.rs:124-130). -/
def AV.hashKey (v : AV) : Nat := v.id

def AV.asStr (v : AV) : String := v.name.getD "?"

theorem AV.fmt_named (n : String) (i s : Nat) : AV.fmt ⟨some n, i, s⟩ = n := rfl
theorem AV.fmt_anon (i s : Nat) : AV.fmt ⟨none, i, s⟩ = "?" ++ toString i := rfl
theorem AV.asStr_spec (v : AV) : v.asStr = v.name.getD "?" := rfl

/-- Equality ignores scope and name — two variables of different scopes with
the same id are "equal" (the root of the id-only issues noted elsewhere). -/
theorem AV.eqv_iff (a b : AV) : a.eqv b = true ↔ a.id = b.id := by simp [AV.eqv]

theorem AV.eqv_other_scope : AV.eqv ⟨some "a", 3, 0⟩ ⟨some "b", 3, 1⟩ = true := rfl

/-- `Hash` is consistent with `Eq` (as `HashMap`/`HashSet` require). -/
theorem AV.hash_consistent (a b : AV) (h : a.eqv b = true) : a.hashKey = b.hashKey := by
  simpa [AV.eqv, AV.hashKey] using h

theorem AV.eqv_equivalence : (∀ a, AV.eqv a a = true) ∧ (∀ a b, AV.eqv a b = AV.eqv b a) ∧
    (∀ a b c, AV.eqv a b = true → AV.eqv b c = true → AV.eqv a c = true) := by
  refine ⟨fun a => by simp [AV.eqv], fun a b => by simp [AV.eqv, Bool.beq_comm], fun a b c h1 h2 => ?_⟩
  simp [AV.eqv] at *; omega

/-! ## Expressions as far as ast.rs inspects them -/

inductive AD
  | var (v : AV) | const (c : String) | map | func (name : String) (agg : Bool) | other (t : Nat)
  | property (k : String)

inductive AE
  | node (d : AD) (cs : List AE)

mutual
def anyA (p : AD → Bool) : AE → Bool
  | .node d cs => p d || anyAL p cs
def anyAL (p : AD → Bool) : List AE → Bool
  | [] => false
  | c :: cs => anyA p c || anyAL p cs
end

def isAggD : AD → Bool | .func _ true => true | _ => false

/-- `is_aggregation` (both impls, ast.rs:446-466): some node of the tree is an aggregate call (DFS any). -/
def isAggregation (e : AE) : Bool := anyA isAggD e

theorem isAggregation_root (n : String) (cs : List AE) : isAggregation (.node (.func n true) cs) = true := by
  simp [isAggregation, anyA, isAggD]

theorem isAggregation_child (d : AD) (c : AE) (cs : List AE) (h : isAggregation c = true) :
    isAggregation (.node d (c :: cs)) = true := by
  simp only [isAggregation, anyA, anyAL] at *; simp [h]

theorem isAggregation_leaf (d : AD) (h : isAggD d = false) : isAggregation (.node d []) = false := by
  simp [isAggregation, anyA, anyAL, h]

/-! ## Pattern graphs -/

structure QN where
  alias : AV
  labels : List String
  /-- inline map entries `(key, value)` under the `Map` root -/
  attrs : List (String × AE)

structure QR where
  alias : AV
  types : List String
  attrs : List (String × AE)
  frm : QN
  to : QN
  bidirectional : Bool
  minHops : Option Nat
  maxHops : Option Nat
  shortest : Bool := false

structure QP where
  var : AV
  vars : List AV

structure QG where
  nodes : List QN := []
  rels : List QR := []
  paths : List QP := []

/-- `QueryNode::new`, `QueryRelationship::new` (no allShortestPaths), `QueryPath::new`, `QueryGraph::default`. -/
def QN.new (a : AV) (ls : List String) (at' : List (String × AE)) : QN := ⟨a, ls, at'⟩
def QR.new (a : AV) (ts : List String) (at' : List (String × AE)) (f t : QN) (b : Bool) (mn mx : Option Nat) : QR :=
  ⟨a, ts, at', f, t, b, mn, mx, false⟩
def QP.new (v : AV) (vs : List AV) : QP := ⟨v, vs⟩
def QG.default : QG := {}

theorem new_spec (a : AV) (ls : List String) (at' : List (String × AE)) (ts : List String) (f t : QN) (b : Bool)
    (mn mx : Option Nat) (v : AV) (vs : List AV) :
    (QN.new a ls at').alias = a ∧ (QN.new a ls at').labels = ls ∧
    (QR.new a ts at' f t b mn mx).shortest = false ∧ (QR.new a ts at' f t b mn mx).frm = f ∧
    (QR.new a ts at' f t b mn mx).to = t ∧ (QP.new v vs).vars = vs ∧
    QG.default.nodes = [] ∧ QG.default.rels = [] ∧ QG.default.paths = [] := by
  simp [QN.new, QR.new, QP.new, QG.default]

def fmtNode (n : QN) : String :=
  if n.labels.isEmpty then "(" ++ n.alias.fmt ++ ")" else "(" ++ n.alias.fmt ++ ":" ++ ":".intercalate n.labels ++ ")"

def fmtRel (r : QR) : String :=
  let dir := if r.bidirectional then "" else ">"
  if r.types.isEmpty then "(" ++ r.frm.alias.fmt ++ ")-[" ++ r.alias.fmt ++ "]-" ++ dir ++ "(" ++ r.to.alias.fmt ++ ")"
  else "(" ++ r.frm.alias.fmt ++ ")-[" ++ r.alias.fmt ++ ":" ++ "|".intercalate r.types ++ "]-" ++ dir ++
    "(" ++ r.to.alias.fmt ++ ")"

def fmtGraph (g : QG) : String :=
  String.join (g.nodes.map (fun n => fmtNode n ++ ", ") ++ g.rels.map (fun r => fmtRel r ++ ", ") ++
    g.paths.map (fun p => p.var.fmt ++ ", "))

theorem fmtNode_plain (n : QN) (h : n.labels = []) : fmtNode n = "(" ++ n.alias.fmt ++ ")" := by
  simp [fmtNode, h]

theorem fmtRel_untyped (r : QR) (h : r.types = []) (hb : r.bidirectional = false) :
    fmtRel r = "(" ++ r.frm.alias.fmt ++ ")-[" ++ r.alias.fmt ++ "]->(" ++ r.to.alias.fmt ++ ")" := by
  simp [fmtRel, h, hb, String.append_assoc]

theorem fmtGraph_empty : fmtGraph QG.default = "" := rfl

/-- `merge_attr_maps`: entries concatenated (duplicate keys kept → `n.v = 1 AND n.v = 2`). -/
def mergeAttrs (l r : List (String × AE)) : List (String × AE) :=
  if r.isEmpty then l else if l.isEmpty then r else l ++ r

theorem mergeAttrs_eq (l r : List (String × AE)) : mergeAttrs l r = l ++ r := by
  unfold mergeAttrs; cases l <;> cases r <;> simp

variable (eqv : AV → AV → Bool)

def addNode (g : QG) (n : QN) : QG × Bool :=
  if g.nodes.any (fun m => eqv m.alias n.alias) then (g, false) else ({ g with nodes := g.nodes ++ [n] }, true)

def addRel (g : QG) (r : QR) : QG × Bool :=
  if g.rels.any (fun m => eqv m.alias r.alias) then (g, false) else ({ g with rels := g.rels ++ [r] }, true)

def addPath (g : QG) (p : QP) : QG × Bool :=
  if g.paths.any (fun m => eqv m.var p.var) then (g, false) else ({ g with paths := g.paths ++ [p] }, true)

theorem addNode_spec (g : QG) (n : QN) :
    (addNode eqv g n).2 = !(g.nodes.any fun m => eqv m.alias n.alias) ∧
    (addNode eqv g n).1.nodes = (if (addNode eqv g n).2 then g.nodes ++ [n] else g.nodes) ∧
    (addNode eqv g n).1.rels = g.rels := by
  unfold addNode; split <;> simp_all

theorem addRel_spec (g : QG) (r : QR) :
    (addRel eqv g r).2 = !(g.rels.any fun m => eqv m.alias r.alias) ∧
    (addRel eqv g r).1.rels = (if (addRel eqv g r).2 then g.rels ++ [r] else g.rels) := by
  unfold addRel; split <;> simp_all

theorem addPath_spec (g : QG) (p : QP) :
    (addPath eqv g p).2 = !(g.paths.any fun m => eqv m.var p.var) ∧
    (addPath eqv g p).1.paths = (if (addPath eqv g p).2 then g.paths ++ [p] else g.paths) := by
  unfold addPath; split <;> simp_all

/-- `OrderSet::extend`: append the labels not yet present. -/
def extendLabels : List String → List String → List String
  | a, [] => a
  | a, x :: xs => extendLabels (if a.contains x then a else a ++ [x]) xs

def mergeNode (g : QG) (n : QN) : QG :=
  match g.nodes.findIdx? (fun m => eqv m.alias n.alias) with
  | none => { g with nodes := g.nodes ++ [n] }
  | some i =>
    match g.nodes[i]? with
    | some m => { g with nodes := g.nodes.set i ⟨m.alias, extendLabels m.labels n.labels, mergeAttrs m.attrs n.attrs⟩ }
    | none => g

theorem extendLabels_mem (a b : List String) (l : String) : l ∈ extendLabels a b ↔ l ∈ a ∨ l ∈ b := by
  induction b generalizing a with
  | nil => simp [extendLabels]
  | cons x xs ih =>
    simp only [extendLabels]
    rw [ih]
    by_cases hx : a.contains x = true
    · simp only [hx, ↓reduceIte, List.mem_cons]
      have : x ∈ a := by simpa using hx
      constructor
      · rintro (h | h)
        · exact Or.inl h
        · exact Or.inr (Or.inr h)
      · rintro (h | rfl | h)
        · exact Or.inl h
        · exact Or.inl this
        · exact Or.inr h
    · simp only [hx, Bool.false_eq_true, ↓reduceIte, List.mem_append, List.mem_cons, List.not_mem_nil, or_false]
      constructor
      · rintro ((h | h) | h)
        · exact Or.inl h
        · exact Or.inr (Or.inl h)
        · exact Or.inr (Or.inr h)
      · rintro (h | h | h)
        · exact Or.inl (Or.inl h)
        · exact Or.inl (Or.inr h)
        · exact Or.inr h

/-- Merging a repeated alias: same position, labels unioned (old first), attrs concatenated. -/
theorem mergeNode_present (g : QG) (n m : QN) (i : Nat) (hi : g.nodes.findIdx? (fun m => eqv m.alias n.alias) = some i)
    (hm : g.nodes[i]? = some m) :
    (mergeNode eqv g n).nodes[i]? = some ⟨m.alias, extendLabels m.labels n.labels, m.attrs ++ n.attrs⟩ ∧
    (mergeNode eqv g n).nodes.length = g.nodes.length := by
  obtain ⟨hlt, hm'⟩ := List.getElem?_eq_some_iff.1 hm
  simp [mergeNode, hi, hm', mergeAttrs_eq, hlt]

theorem mergeNode_absent (g : QG) (n : QN) (h : g.nodes.findIdx? (fun m => eqv m.alias n.alias) = none) :
    (mergeNode eqv g n).nodes = g.nodes ++ [n] := by
  simp [mergeNode, h]

def replaceNode (g : QG) (a : AV) (n : QN) : QG :=
  match g.nodes.findIdx? (fun m => eqv m.alias a) with
  | some i => { g with nodes := g.nodes.set i n }
  | none => g

theorem replaceNode_spec (g : QG) (a : AV) (n : QN) :
    (replaceNode eqv g a n).nodes.length = g.nodes.length ∧ (replaceNode eqv g a n).rels = g.rels := by
  unfold replaceNode; split <;> simp

/-- `variables()`: nodes, then relationships, then paths. -/
def variables (g : QG) : List AV := g.nodes.map (·.alias) ++ g.rels.map (·.alias) ++ g.paths.map (·.var)

theorem variables_order (g : QG) :
    (variables g).take g.nodes.length = g.nodes.map (·.alias) ∧
    (variables g).length = g.nodes.length + g.rels.length + g.paths.length := by
  simp [variables]; omega

/-- The accessors return the fields (`nodes`, `nodes_mut`, `relationships`, `relationships_mut`, `paths`). -/
theorem accessors (g : QG) (ns : List QN) (rs : List QR) :
    ({ g with nodes := ns }).nodes = ns ∧ ({ g with rels := rs }).rels = rs ∧
    ({ g with nodes := ns }).rels = g.rels ∧ ({ g with rels := rs }).nodes = g.nodes ∧
    ({ g with nodes := ns }).paths = g.paths := ⟨rfl, rfl, rfl, rfl, rfl⟩

/-! ## `filter_visited` -/

def notVisited (vis : List (Nat × Nat)) (v : AV) : Bool := !vis.contains (v.id, v.scope)

def filterVisited (g : QG) (vis : List (Nat × Nat)) : QG :=
  let g1 := (g.nodes.filter (fun n => notVisited vis n.alias)).foldl (fun acc n => (addNode eqv acc n).1) QG.default
  let g2 := (g.rels.filter (fun r => notVisited vis r.alias)).foldl (fun acc r => (addRel eqv acc r).1) g1
  (g.paths.filter (fun p => notVisited vis p.var)).foldl (fun acc p => (addPath eqv acc p).1) g2

theorem foldl_addNode_mem (l : List QN) (acc : QG) (n : QN) (h : n ∈ (l.foldl (fun acc n => (addNode eqv acc n).1) acc).nodes) :
    n ∈ acc.nodes ∨ n ∈ l := by
  induction l generalizing acc with
  | nil => left; exact h
  | cons m ms ih =>
    rcases ih _ h with h | h
    · have := (addNode_spec eqv acc m).2.1
      rw [this] at h
      split at h
      · rcases List.mem_append.1 h with h | h
        · left; exact h
        · simp at h; right; simp [h]
      · left; exact h
    · right; exact List.mem_cons_of_mem _ h

/-- Only unvisited entities survive (`(id, scope)` checked). -/
theorem filterVisited_nodes (g : QG) (vis : List (Nat × Nat)) (n : QN) (h : n ∈ (filterVisited eqv g vis).nodes) :
    n ∈ g.nodes ∧ notVisited vis n.alias = true := by
  simp only [filterVisited] at h
  -- paths and rels folds keep the nodes
  have hp : ∀ (l : List QP) (acc : QG), (l.foldl (fun acc p => (addPath eqv acc p).1) acc).nodes = acc.nodes := by
    intro l; induction l with
    | nil => intro; rfl
    | cons p ps ih => intro acc; simp only [List.foldl_cons]; rw [ih]; unfold addPath; split <;> rfl
  have hr : ∀ (l : List QR) (acc : QG), (l.foldl (fun acc r => (addRel eqv acc r).1) acc).nodes = acc.nodes := by
    intro l; induction l with
    | nil => intro; rfl
    | cons p ps ih => intro acc; simp only [List.foldl_cons]; rw [ih]; unfold addRel; split <;> rfl
  rw [hp, hr] at h
  rcases foldl_addNode_mem eqv _ _ n h with h | h
  · simp [QG.default] at h
  · exact List.mem_filter.1 h

/-! ## Validation -/

/-- What the validator looks at in each clause. -/
inductive QC
  | call (proc : String) (arg0 : Option String)  -- "str" / "map+label" / "map" / other
  | match_ (inlineMaps : Bool)
  | unwind
  | merge (inlineMaps : Bool) (relTypes : List Nat) (setOk : Bool)
  | create (inlineMaps : Bool) (relTypes : List Nat)
  | delete
  | set (setOk : Bool)
  | remove (nullTarget : Bool)
  | loadCsv | with_ | return_ | createIndex | dropIndex
  | query (cs : List QC)
  | union (bs : List QC) (cols : List (List String))
  | foreach (body : List QC)
  | callSub (body : QC) (returning : Bool)

def callCheck (proc : String) (arg0 : Option String) : Except String Unit :=
  if proc = "db.idx.fulltext.createNodeIndex" then
    match arg0 with
    | some "str" | some "map+label" => .ok ()
    | some "map" => .error "Label is missing"
    | _ => .error "The first argument of a procedure call must be a string or a map with a 'label' key"
  else if proc = "db.idx.fulltext.drop" ∧ arg0 ≠ some "str" then
    .error "The first argument of db.idx.fulltext.drop must be a string literal"
  else .ok ()

def inlinedOk (b : Bool) : Except String Unit := if b then .ok () else .error "Encountered unhandled type in inlined properties."

def relTypesOk (kind : String) (ts : List Nat) : Except String Unit :=
  if ts.all (· == 1) then .ok ()
  else .error s!"Exactly one relationship type must be specified for each relation in a {kind} pattern."

def setItemsOk (b : Bool) : Except String Unit :=
  if b then .ok () else .error "FalkorDB does not currently support non-alias references on the left-hand side of SET expressions"

theorem szL {α : Type} [SizeOf α] (l : List α) : 1 ≤ sizeOf l := by cases l <;> simp <;> omega

mutual
def innerValidate : QC → List QC → Except String Unit
  | .call p a, _ => callCheck p a
  | .match_ m, rest => (inlinedOk m).bind fun _ => match rest with
    | [] => .error "Query cannot conclude with MATCH (must be a RETURN clause, an update clause, a procedure call or a non-returning subquery)"
    | c :: cs => innerValidate c cs
  | .unwind, rest => match rest with
    | [] => .error "Query cannot conclude with UNWIND (must be a RETURN clause, an update clause, a procedure call or a non-returning subquery)"
    | c :: cs => innerValidate c cs
  | .merge m ts s, rest => (inlinedOk m).bind fun _ => (relTypesOk "MERGE" ts).bind fun _ =>
      (setItemsOk s).bind fun _ => next rest
  | .create m ts, rest => (inlinedOk m).bind fun _ => (relTypesOk "CREATE" ts).bind fun _ => next rest
  | .delete, rest => next rest
  | .set s, rest => (setItemsOk s).bind fun _ => next rest
  | .remove nt, rest => if nt then .error "Type mismatch: expected Node or Relationship but was Null" else next rest
  | .loadCsv, rest | .with_, rest | .return_, rest | .createIndex, rest | .dropIndex, rest => next rest
  | .query cs, _ => match cs with
    | [] => .error "Error: empty query."
    | c :: rest => innerValidate c rest
  | .union bs cols, _ => (validateAll bs).bind fun _ =>
      match cols with
      | [] => .ok ()
      | c0 :: more => if more.all (· == c0) then .ok () else .error "All sub queries in a UNION must have the same column names."
  | .foreach body, rest => (validateAll body).bind fun _ => next rest
  | .callSub body ret, rest => (innerValidate body []).bind fun _ =>
      if ret then match rest with
        | [] => .error "Query cannot conclude with a returning subquery (must be a RETURN clause, an update clause, a procedure call or a non-returning subquery)"
        | c :: cs => innerValidate c cs
      else next rest
  termination_by q rest => sizeOf q + sizeOf rest
  decreasing_by all_goals first | (simp_wf; omega) | (simp_wf; have := szL cs; omega) | (simp_wf; have := szL cols; omega) | simp_wf | omega
def next : List QC → Except String Unit
  | [] => .ok ()
  | c :: cs => innerValidate c cs
  termination_by l => sizeOf l
  decreasing_by all_goals first | (simp_wf; omega) | (simp_wf; have := szL cs; omega) | (simp_wf; have := szL cols; omega) | simp_wf | omega
def validateAll : List QC → Except String Unit
  | [] => .ok ()
  | c :: cs => (innerValidate c []).bind fun _ => validateAll cs
  termination_by l => sizeOf l
  decreasing_by all_goals first | (simp_wf; omega) | (simp_wf; have := szL cs; omega) | (simp_wf; have := szL cols; omega) | simp_wf | omega
end

def validate (q : QC) : Except String Unit := innerValidate q []

/-- The procedure-call arm never looks at what follows (ast.rs:1166-1206). -/
theorem call_skips_rest (p : String) (a : Option String) (rest : List QC) :
    innerValidate (.call p a) rest = callCheck p a := by rw [innerValidate]

/-- `CALL db.labels() YIELD label MATCH (n)` validates (Rust then panics in
`utilize_node_by_id`); `MATCH (n)` alone does not. -/
theorem call_then_match_accepted :
    validate (.query [.call "db.labels" none, .match_ true]) = .ok () ∧
    validate (.query [.match_ true]) ≠ .ok () := by
  constructor
  · simp [validate, innerValidate, callCheck]
  · simp [validate, innerValidate, inlinedOk, Except.bind]

/-- `MATCH (n) WITH n` validates (C and openCypher reject a final WITH). -/
theorem with_last_accepted : validate (.query [.match_ true, .with_]) = .ok () := by
  simp [validate, innerValidate, inlinedOk, next, Except.bind]

/-- Without a procedure call before it, a final MATCH / UNWIND is rejected. -/
theorem validate_ends :
    validate (.query [.unwind]) = .error "Query cannot conclude with UNWIND (must be a RETURN clause, an update clause, a procedure call or a non-returning subquery)" ∧
    validate (.query [.match_ true]) = .error "Query cannot conclude with MATCH (must be a RETURN clause, an update clause, a procedure call or a non-returning subquery)" := by
  constructor <;> simp [validate, innerValidate, inlinedOk, Except.bind]

theorem create_multi_type :
    innerValidate (.create true [2]) [] = .error "Exactly one relationship type must be specified for each relation in a CREATE pattern." := by
  simp [innerValidate, inlinedOk, relTypesOk, Except.bind]; rfl

theorem setItemsOk_spec (b : Bool) : setItemsOk b = .ok () ↔ b = true := by
  cases b <;> simp [setItemsOk]

theorem inlinedOk_spec (b : Bool) : inlinedOk b = .ok () ↔ b = true := by
  cases b <;> simp [inlinedOk]

/-! ## `return_column_names` -/

inductive CC
  | ret (names : List String)
  | call (yields : List String)
  | callSub (body : CC) (returning : Bool)
  | other
  | query (cs : List CC)

mutual
def returnColumns : CC → List String
  | .query cs => (lastCols cs).getD []
  | _ => []
def colsOf : CC → Option (List String)
  | .ret ns => some ns
  | .call ys => some ys
  | .callSub b true => some (returnColumns b)
  | _ => none
/-- The last clause (scanning from the end) that is a RETURN, a CALL's yields,
or a returning subquery. -/
def lastCols : List CC → Option (List String)
  | [] => none
  | c :: cs => (lastCols cs).orElse fun _ => colsOf c
end

theorem returnColumns_last_return (cs : List CC) (ns : List String) :
    returnColumns (.query (cs ++ [.ret ns])) = ns := by
  have : ∀ l : List CC, lastCols (l ++ [.ret ns]) = some ns := by
    intro l; induction l with
    | nil => rfl
    | cons c cs ih => simp [lastCols, ih]
  simp [returnColumns, this]

theorem returnColumns_skips_updates (cs : List CC) (ns : List String) :
    returnColumns (.query (cs ++ [.ret ns, .other, .callSub .other false])) = ns := by
  have : ∀ l : List CC, lastCols (l ++ [.ret ns, .other, .callSub .other false]) = some ns := by
    intro l; induction l with
    | nil => rfl
    | cons c cs ih => simp [lastCols, ih]
  simp [returnColumns, this]

theorem returnColumns_nonquery : returnColumns (.ret ["a"]) = [] := rfl

end PlannerBuild.Ast
