import FalkorBinder.BindExpr
import FalkorBinder.Scopes
/-
# `bind_ir`: the per-clause checks, CALL {} and CREATE

| here | there |
| --- | --- |
| `shortestOk` | allShortestPaths endpoint check, MATCH arm binder.rs:306-332 |
| `withOptionalLabels` | OPTIONAL MATCH label save/restore binder.rs:336-345 |
| `mergeOk` | MERGE redeclaration checks binder.rs:371-414 |
| `deleteOk` | DELETE entity check binder.rs:461-466 |
| `returnStarOk` | `RETURN *` check binder.rs:531-540 |
| `yieldOk`, `bindYield` | CALL … YIELD validation and binding binder.rs:562-662 |
| `RC`, `bindIR` | the dispatcher `bind_ir` binder.rs:262-713 (sub-binders as parameters) |
| `subqueryReturns`, `bindCallSubquery` | `bind_call_subquery` binder.rs:716-791 |
| `importProjections` | `build_import_projections` binder.rs:979-1001 |
| `createCheck`, `bindGraphCreate` | `bind_graph_create` binder.rs:1557-1680 |
-/
namespace FalkorBinder

/-! ## MATCH -/

/-- `(from, to, isShortest)` per relationship. -/
abbrev RelEnds := Name × Name × Bool

def shortestOk (env : Env) (rels : List RelEnds) : Bool :=
  let inner := (rels.filter (fun r => !r.2.2)).flatMap fun r => [r.1, r.2.1]
  rels.all fun r => !r.2.2 || ((env.has r.1 || inner.contains r.1) && (env.has r.2.1 || inner.contains r.2.1))

/-- An allShortestPaths needs both endpoints bound before, or bound by another
(ordinary) relationship of the same pattern. -/
theorem shortestOk_iff (env : Env) (rels : List RelEnds) :
    shortestOk env rels = true ↔ ∀ r ∈ rels, r.2.2 = true →
      (env.has r.1 = true ∨ ∃ q ∈ rels, q.2.2 = false ∧ (q.1 = r.1 ∨ q.2.1 = r.1)) ∧
      (env.has r.2.1 = true ∨ ∃ q ∈ rels, q.2.2 = false ∧ (q.1 = r.2.1 ∨ q.2.1 = r.2.1)) := by
  unfold shortestOk
  simp only [List.all_eq_true, Bool.or_eq_true, Bool.not_eq_true', Bool.and_eq_true,
    List.contains_iff_mem, List.mem_flatMap, List.mem_filter, List.mem_cons, List.not_mem_nil, or_false]
  constructor
  · intro h r hr hs
    rcases h r hr with h1 | h1
    · simp [hs] at h1
    · refine ⟨?_, ?_⟩
      · rcases h1.1 with h2 | ⟨q, ⟨hq, hq2⟩, hq3⟩
        · exact Or.inl h2
        · exact Or.inr ⟨q, hq, by simpa using hq2, by rcases hq3 with e | e <;> simp [e]⟩
      · rcases h1.2 with h2 | ⟨q, ⟨hq, hq2⟩, hq3⟩
        · exact Or.inl h2
        · exact Or.inr ⟨q, hq, by simpa using hq2, by rcases hq3 with e | e <;> simp [e]⟩
  · intro h r hr
    by_cases hs : r.2.2 = true
    · right
      obtain ⟨h1, h2⟩ := h r hr hs
      refine ⟨?_, ?_⟩
      · rcases h1 with h1 | ⟨q, hq, hq2, hq3⟩
        · exact Or.inl h1
        · exact Or.inr ⟨q, ⟨hq, by simpa using hq2⟩, by rcases hq3 with e | e <;> simp [e]⟩
      · rcases h2 with h2 | ⟨q, hq, hq2, hq3⟩
        · exact Or.inl h2
        · exact Or.inr ⟨q, ⟨hq, by simpa using hq2⟩, by rcases hq3 with e | e <;> simp [e]⟩
    · left; simpa using hs

/-- OPTIONAL MATCH: whatever binding the pattern does to the accumulated labels
is undone (the labels stay clause-local); a plain MATCH keeps them. -/
def withOptionalLabels {L : Type} (optional : Bool) (before : L) (bindPat : L → L) : L :=
  if optional then before else bindPat before

theorem optional_labels_restored {L : Type} (before : L) (f : L → L) :
    withOptionalLabels true before f = before := rfl

/-! ## MERGE -/

/-- Accumulated labels of a bound name (none if not bound / no labels). -/
abbrev LabelsOf := Name → Option (List Name)

/-- Nodes `(alias, labels)`, relationships `(alias, from, to)`. -/
def mergeOk (env : Env) (lab : LabelsOf) (nodes : List (Name × List Name))
    (rels : List (Name × (Name × List Name) × (Name × List Name))) : Bool :=
  let nodeOk := fun (n : Name × List Name) =>
    !env.has n.1 || (match lab n.1 with
      | some ex => n.2.all ex.contains
      | none => true)
  nodes.all nodeOk && rels.all fun r => !env.has r.1 && nodeOk r.2.1 && nodeOk r.2.2

/-- MERGE may not re-declare a bound relationship, nor add labels to a bound node. -/
theorem mergeOk_rel_bound (env : Env) (lab : LabelsOf) (nodes : List (Name × List Name))
    (r : Name × (Name × List Name) × (Name × List Name)) (rest : List _) (h : env.has r.1 = true) :
    mergeOk env lab nodes (r :: rest) = false := by
  simp [mergeOk, h]

theorem mergeOk_new_label (env : Env) (lab : LabelsOf) (n : Name) (ls ex : List Name) (l : Name)
    (rest : List (Name × List Name)) (rels : List _) (hb : env.has n = true) (hl : lab n = some ex)
    (hm : l ∈ ls) (hn : l ∉ ex) : mergeOk env lab ((n, ls) :: rest) rels = false := by
  simp only [mergeOk, List.all_cons, hb, Bool.not_true, hl, Bool.false_or, Bool.and_eq_false_iff]
  left; left
  rw [List.all_eq_false]
  exact ⟨l, hm, by simpa using hn⟩

theorem mergeOk_unbound (env : Env) (lab : LabelsOf) (nodes : List (Name × List Name))
    (rels : List (Name × (Name × List Name) × (Name × List Name)))
    (h : ∀ n, env.has n = false) : mergeOk env lab nodes rels = true := by
  simp [mergeOk, h]

/-! ## DELETE, RETURN * -/

def deleteOk {B : Type} (mayEntity : B → Bool) (es : List B) : Bool := es.all mayEntity

theorem deleteOk_iff {B : Type} (m : B → Bool) (es : List B) : deleteOk m es = true ↔ ∀ e ∈ es, m e = true := by
  simp [deleteOk]

def returnStarOk (all : Bool) (env : Env) : Bool := !all || env.any fun kv => !kv.1.startsWith "_"

theorem returnStarOk_iff (all : Bool) (env : Env) :
    returnStarOk all env = false ↔ all = true ∧ ∀ kv ∈ env, kv.1.startsWith "_" = true := by
  simp [returnStarOk]

/-! ## CALL … YIELD -/

/-- Each yield `(name, alias?)`: the source field is the alias if present. -/
def fieldOf (y : Name × Option Name) : Name := y.2.getD y.1

def requiredEntity : String → Option Name
  | "db.idx.fulltext.queryNodes" | "db.idx.vector.queryNodes" => some "node"
  | "db.idx.fulltext.queryRelationships" | "db.idx.vector.queryRelationships" => some "relationship"
  | _ => none

/-- The validation loop: field exists, projected name not repeated (case-insensitively). -/
def yieldLoop (fields : List Name) : List Name → List (Name × Option Name) → Except String Unit
  | _, [] => .ok ()
  | seen, y :: ys =>
    if !fields.contains (fieldOf y) then .error s!"does not yield output `{fieldOf y}`"
    else if seen.contains y.1.toLower then .error s!"Duplicate yield field '{y.1}'"
    else yieldLoop fields (y.1.toLower :: seen) ys

def yieldOk (proc : String) (fields : List Name) (ys : List (Name × Option Name)) : Except String Unit :=
  match yieldLoop fields [] ys with
  | .error e => .error e
  | .ok () => match requiredEntity proc with
    | some ent => if ys.any (fun y => fieldOf y == ent) then .ok () else .error s!"requires YIELD of '{ent}'"
    | none => .ok ()

/-- No projected name repeats (case-insensitively), given those already seen. -/
def freshNames : List Name → List (Name × Option Name) → Bool
  | _, [] => true
  | seen, y :: ys => !seen.contains y.1.toLower && freshNames (y.1.toLower :: seen) ys

theorem yieldLoop_ok (fields : List Name) : ∀ (seen : List Name) (ys : List (Name × Option Name)),
    yieldLoop fields seen ys = .ok () ↔
      (∀ y ∈ ys, fields.contains (fieldOf y) = true) ∧ freshNames seen ys = true
  | _, [] => by simp [yieldLoop, freshNames]
  | seen, y :: ys => by
    have ih := yieldLoop_ok fields (y.1.toLower :: seen) ys
    unfold yieldLoop
    cases h1 : fields.contains (fieldOf y) <;> cases h2 : seen.contains y.1.toLower <;>
      simp only [Bool.not_true, Bool.not_false, Bool.false_eq_true, ↓reduceIte, freshNames, h1, h2,
        List.mem_cons, forall_eq_or_imp, ih] <;> simp

/-- `YIELD node, node` and `YIELD node AS x, score AS X` are rejected. -/
theorem yield_duplicate (fields : List Name) (n : Name) (a b : Option Name)
    (h1 : fields.contains (fieldOf (n, a)) = true) :
    yieldLoop fields [] [(n, a), (n, b)] ≠ .ok () := by
  intro h; rw [yieldLoop_ok] at h; simp [freshNames] at h

/-- A required entity field must be yielded (fulltext / vector scans). -/
theorem yield_requires_entity (fields : List Name) (ys : List (Name × Option Name))
    (hl : yieldLoop fields [] ys = .ok ()) (h : ys.all (fun y => fieldOf y != "node") = true) :
    yieldOk "db.idx.fulltext.queryNodes" fields ys ≠ .ok () := by
  simp only [yieldOk, hl, requiredEntity]
  have : ys.any (fun y => fieldOf y == "node") = false := by
    simpa [List.all_eq_true, List.any_eq_false] using h
  simp [this]

/-- Binding one yield (binder.rs:625-656): `YIELD f AS a` registers `a` with the
variable's name set to the source field; without `YIELD` the variable is hidden
under `_hidden_{id}_{name}` so `RETURN *` ignores it. -/
def bindYield (explicit : Bool) (hidden : Nat → Name → Name) (e : Env) (depth : Nat)
    (y : Name × Option Name) : Except String (Env × Var) :=
  if explicit then
    match defineName e depth y.1 .any true with
    | .error err => .error err
    | .ok (e1, v) => match y.2 with
      | some f => let v' := { v with name := f }; .ok (e1.insert y.1 v', v')
      | none => .ok (e1, v)
  else
    let v := freshVar e y.1 .any depth
    .ok (e.insert (hidden v.id y.1) v, v)

theorem bindYield_alias (hidden : Nat → Name → Name) (e : Env) (d : Nat) (n f : Name) (e1 : Env) (v : Var)
    (h : defineName e d n .any true = .ok (e1, v)) :
    bindYield true hidden e d (n, some f) = .ok (e1.insert n { v with name := f }, { v with name := f }) := by
  simp [bindYield, h]

theorem bindYield_hidden (hidden : Nat → Name → Name) (e : Env) (d : Nat) (y : Name × Option Name) :
    bindYield false hidden e d y =
      .ok (e.insert (hidden e.length y.1) (freshVar e y.1 .any d), freshVar e y.1 .any d) := rfl

/-! ## The dispatcher -/

/-- Raw clauses, with what `bind_ir` itself inspects; sub-binders are parameters. -/
inductive RC
  | union (branches : List RC)
  | query (clauses : List RC)
  | match_ (rels : List RelEnds) (optional : Bool) (pat : Nat) (filter : Option NE)
  | unwind (e : NE) (x : Name)
  | merge (nodes : List (Name × List Name)) (rels : List (Name × (Name × List Name) × (Name × List Name))) (pat : Nat)
  | create (pat : Nat)
  | createIndex (opts : Option NE)
  | dropIndex
  | delete (es : List NE)
  | set (items : List SetItem)
  | remove (es : List NE)
  | loadCsv (path delim : NE) (x : Name)
  | with_ (p : Nat)
  | return_ (all : Bool) (p : Nat)
  | call (proc : String) (fields : List Name) (args : List NE) (yields : List (Name × Option Name))
      (explicit : Bool) (filter : Option NE)
  | foreach (list : NE) (x : Name) (body : List RC)
  | callSub (body : RC) (returning : Bool)

/-- The sub-binders and checks `bind_ir` delegates to. -/
structure Subs where
  expr : St → NE → Except String St
  graph : St → Nat → Except String St
  graphCreate : St → Nat → Except String St
  projection : St → Nat → Except String St
  setItems : St → List SetItem → Except String St
  mayBool : NE → Bool
  mayEntity : NE → Bool
  columns : RC → List Name
  labels : LabelsOf
  hidden : Nat → Name → Name
  callSub : St → RC → Bool → Except String St
  bindBranch : RC → Except String Unit

def seqE {A : Type} (f : St → A → Except String St) : St → List A → Except String St
  | st, [] => .ok st
  | st, a :: as => match f st a with
    | .error e => .error e
    | .ok st' => seqE f st' as

variable (S : Subs)

def bindIR : St → RC → Except String St
  | st, .union bs =>
    match unionCheck (bs.map S.columns) with
    | .error e => .error e
    | .ok () => match seqE (fun st b => (S.bindBranch b).map fun _ => st) st bs with
      | .error e => .error e
      | .ok _ => .ok st
  | st, .query cs => bindIRs st cs
  | st, .match_ rels _ pat filter =>
    if !shortestOk st.cur rels then .error "Source and destination must already be resolved to call allShortestPaths"
    else match S.graph st pat with
      | .error e => .error e
      | .ok st1 => match filter with
        | none => .ok st1
        | some f => match S.expr st1 f with
          | .error e => .error e
          | .ok st2 => if S.mayBool f then .ok st2 else .error "Expected boolean predicate"
  | st, .unwind e x => match S.expr st e with
    | .error err => .error err
    | .ok st1 => match defineName st1.cur st1.depth x .any false with
      | .error err => .error err
      | .ok (c, _) => .ok { st1 with cur := c }
  | st, .merge nodes rels pat =>
    if !mergeOk st.cur S.labels nodes rels then .error "can't be redeclared in a MERGE clause"
    else S.graph st pat
  | st, .create pat => S.graphCreate st pat
  | st, .createIndex none => .ok st
  | st, .createIndex (some o) => S.expr st o
  | st, .dropIndex => .ok st
  | st, .delete es => match seqE S.expr st es with
    | .error e => .error e
    | .ok st1 => if deleteOk S.mayEntity es then .ok st1
                 else .error "DELETE can only be called on nodes, paths and relationships"
  | st, .set items => S.setItems st items
  | st, .remove es => seqE S.expr st es
  | st, .loadCsv p d x => match seqE S.expr st [p, d] with
    | .error e => .error e
    | .ok st1 => match defineName st1.cur st1.depth x .any true with
      | .error e => .error e
      | .ok (c, _) => .ok { st1 with cur := c }
  | st, .with_ p => S.projection st p
  | st, .return_ all p =>
    if !returnStarOk all st.cur then .error "RETURN * is not allowed when there are no variables in scope"
    else S.projection st p
  | st, .call proc fields args ys explicit filter =>
    match seqE S.expr st args with
    | .error e => .error e
    | .ok st1 =>
      match (if explicit then yieldOk proc fields ys else .ok ()) with
      | .error e => .error e
      | .ok () =>
        match seqE (fun st y => (bindYield explicit S.hidden st.cur st.depth y).map
            fun r => { st with cur := r.1 }) st1 ys with
        | .error e => .error e
        | .ok st2 => match filter with
          | none => .ok st2
          | some f => match S.expr st2 f with
            | .error e => .error e
            | .ok st3 => if S.mayBool f then .ok st3 else .error "Expected boolean predicate"
  | st, .foreach l x body =>
    match S.expr st l with
    | .error e => .error e
    | .ok st1 => match defineName st1.cur st1.depth x .any true with
      | .error e => .error e
      | .ok (c, _) => match bindIRs { st1 with cur := c } body with
        | .error e => .error e
        | .ok st2 => .ok { st2 with cur := foreachRestore st1.cur st2.cur st1.depth }
  | st, .callSub body ret => S.callSub st body ret
where
  bindIRs : St → List RC → Except String St
    | st, [] => .ok st
    | st, c :: cs => match bindIR st c with
      | .error e => .error e
      | .ok st' => bindIRs st' cs

/-- `UNWIND e AS x`: `e` is bound before `x` exists (so `UNWIND x AS x` needs an outer `x`),
and `x` may not be already declared. -/
theorem unwind_order (st st1 : St) (e : NE) (x : Name) (h1 : S.expr st e = .ok st1) (v : Var)
    (hx : st1.cur.find? x = some v) (hn : x.startsWith "_anon" = false) :
    bindIR S st (.unwind e x) = .error s!"Variable `{x}` already declared" := by
  simp [bindIR, h1, defineName, hn, hx]

/-- MATCH: a non-boolean WHERE is rejected after binding. -/
theorem match_nonbool (st st1 st2 : St) (rels : List RelEnds) (o : Bool) (p : Nat) (f : NE)
    (hs : shortestOk st.cur rels = true) (hg : S.graph st p = .ok st1) (he : S.expr st1 f = .ok st2)
    (hb : S.mayBool f = false) :
    bindIR S st (.match_ rels o p (some f)) = .error "Expected boolean predicate" := by
  simp [bindIR, hs, hg, he, hb]

theorem match_shortest_unbound (st : St) (rels : List RelEnds) (o : Bool) (p : Nat) (f : Option NE)
    (hs : shortestOk st.cur rels = false) :
    bindIR S st (.match_ rels o p f) = .error "Source and destination must already be resolved to call allShortestPaths" := by
  simp [bindIR, hs]

theorem delete_nonentity (st st1 : St) (es : List NE) (h : seqE S.expr st es = .ok st1)
    (hd : deleteOk S.mayEntity es = false) :
    bindIR S st (.delete es) = .error "DELETE can only be called on nodes, paths and relationships" := by
  simp [bindIR, h, hd]

theorem return_star_empty (st : St) (p : Nat) (h : ∀ kv ∈ st.cur, kv.1.startsWith "_" = true) :
    bindIR S st (.return_ true p) = .error "RETURN * is not allowed when there are no variables in scope" := by
  have : returnStarOk true st.cur = false := (returnStarOk_iff true st.cur).2 ⟨rfl, h⟩
  simp [bindIR, this]

theorem query_seq (st : St) (c : RC) (cs : List RC) :
    bindIR S st (.query (c :: cs)) = match bindIR S st c with
      | .error e => .error e
      | .ok st' => bindIR S st' (.query cs) := by
  simp [bindIR, bindIR.bindIRs]

/-- UNION: binding fails on a column mismatch before any branch is bound, and
the outer state is untouched (each branch has its own binder). -/
theorem union_columns (st : St) (bs : List RC) (e : String) (h : unionCheck (bs.map S.columns) = .error e) :
    bindIR S st (.union bs) = .error e := by
  simp [bindIR, h]

theorem union_state (st st' : St) (bs : List RC) (h : bindIR S st (.union bs) = .ok st') : st' = st := by
  simp only [bindIR] at h
  split at h
  · simp at h
  · split at h
    · simp at h
    · simp at h; exact h.symm

/-! ## CALL {} -/

def internalPrefixes : List String := ["_slot_", "__agg_placeholder_", "_quant_", "_lc_", "_reduce_"]

def isInternal (n : Name) : Bool := internalPrefixes.any fun p => n.startsWith p

/-- What a returning CALL body hands back: its final table minus internal reservations. -/
def subqueryReturns (inner : Env) : Env := inner.filter fun kv => !isInternal kv.1

/-- Steps 3-4 of `bind_call_subquery`: restore the outer table, reject a returned
name the outer scope already has, then project each returned name to a fresh
outer id. Returns the new outer table and the inner→outer remap. -/
def bindCallSubquery (outer inner : Env) (depth : Nat) (returning : Bool) :
    Except String (Env × List (Var × Var)) :=
  if !returning then .ok (outer, [])
  else
    let ret := subqueryReturns inner
    if ret.any (fun kv => outer.has kv.1) then .error "already declared in outer scope"
    else .ok (ret.foldl (fun acc kv =>
        let p := projectName acc.1 depth kv.1 kv.2.ty
        (p.1, acc.2 ++ [(kv.2, p.2)])) (outer, []))

theorem callSubquery_nonreturning (outer inner : Env) (d : Nat) :
    bindCallSubquery outer inner d false = .ok (outer, []) := rfl

theorem callSubquery_collision (outer inner : Env) (d : Nat) (kv : Name × Var) (hk : kv ∈ subqueryReturns inner)
    (h : outer.has kv.1 = true) : ∃ e, bindCallSubquery outer inner d true = .error e := by
  have : (subqueryReturns inner).any (fun kv => outer.has kv.1) = true := List.any_eq_true.2 ⟨kv, hk, h⟩
  exact ⟨"already declared in outer scope", by simp [bindCallSubquery, this]⟩

/-- No internal reservation is ever returned. -/
theorem subqueryReturns_noInternal (inner : Env) (kv : Name × Var) (h : kv ∈ subqueryReturns inner) :
    isInternal kv.1 = false := by
  simp only [subqueryReturns, List.mem_filter, Bool.not_eq_true'] at h; exact h.2

theorem foldProject_covers (depth : Nat) : ∀ (ret : List (Name × Var)) (acc : Env × List (Var × Var)),
    acc.1.Covers → (∀ kv ∈ ret, acc.1.has kv.1 = false) → (ret.map (·.1)).Nodup →
    (ret.foldl (fun acc kv => let p := projectName acc.1 depth kv.1 kv.2.ty; (p.1, acc.2 ++ [(kv.2, p.2)])) acc).1.Covers
  | [], acc, hc, _, _ => hc
  | kv :: rest, acc, hc, hn, hd => by
    simp only [List.foldl_cons]
    simp only [List.map_cons, List.nodup_cons, List.mem_map] at hd
    apply foldProject_covers depth rest
    · exact projectName_new_covers hc (hn kv (by simp)) _ _
    · intro kv' hkv'
      simp only [projectName, Env.insert_new_eq (hn kv (by simp)), Env.has, List.any_append,
        List.any_cons, List.any_nil, Bool.or_false, Bool.or_eq_false_iff]
      refine ⟨by simpa [Env.has] using hn kv' (by simp [hkv']), ?_⟩
      rw [beq_eq_false_iff_ne]; intro e; exact hd.1 ⟨kv', hkv', e.symm⟩
    · exact hd.2

/-- With distinct returned names, the outer table stays covering: every returned
variable gets a fresh outer id (binder.rs:777-787). -/
theorem callSubquery_covers (outer inner : Env) (d : Nat) (hc : outer.Covers)
    (hd : ((subqueryReturns inner).map (·.1)).Nodup) (r : Env × List (Var × Var))
    (h : bindCallSubquery outer inner d true = .ok r) : r.1.Covers := by
  simp only [bindCallSubquery, Bool.not_true, Bool.false_eq_true, ↓reduceIte] at h
  split at h
  · simp at h
  · rename_i hn
    simp at h; rw [← h]
    apply foldProject_covers d _ _ hc _ hd
    intro kv hkv
    simp only [List.any_eq_true, not_exists, not_and, Bool.not_eq_true] at hn
    exact hn kv hkv

/-! ## `build_import_projections` -/

/-- Each imported `(name, outer)` gets a fresh inner variable in the (new, empty)
body table; the pairs are sorted by name. -/
def importProjections (imported : List (Name × Var)) (depth : Nat) : Env × List (Var × Var) :=
  imported.foldl (fun acc kv =>
    let p := projectName acc.1 depth kv.1 kv.2.ty
    (p.1, acc.2 ++ [(p.2, kv.2)])) ([], [])

def sortProj (ps : List (Var × Var)) : List (Var × Var) :=
  ps.mergeSort fun a b => decide (a.1.name ≤ b.1.name)

theorem importProjections_covers (imported : List (Name × Var)) (d : Nat) (hd : (imported.map (·.1)).Nodup) :
    (importProjections imported d).1.Covers := by
  unfold importProjections
  have hc := foldProject_covers d imported ([], []) (by intro i hi; simp [Env.ids] at hi) (by simp [Env.has]) hd
  -- the same fold, pairs in the other order
  suffices ∀ (l : List (Name × Var)) (acc : Env × List (Var × Var)) (acc' : Env × List (Var × Var)),
      acc.1 = acc'.1 →
      (l.foldl (fun acc kv => let p := projectName acc.1 d kv.1 kv.2.ty; (p.1, acc.2 ++ [(p.2, kv.2)])) acc).1 =
      (l.foldl (fun acc kv => let p := projectName acc.1 d kv.1 kv.2.ty; (p.1, acc.2 ++ [(kv.2, p.2)])) acc').1 by
    rw [this imported ([], []) ([], []) rfl]; exact hc
  intro l
  induction l with
  | nil => intro acc acc' h; exact h
  | cons kv rest ih => intro acc acc' h; simp only [List.foldl_cons]; apply ih; simp [h]

theorem sortProj_spec (ps : List (Var × Var)) :
    (sortProj ps).Perm ps ∧ (sortProj ps).Pairwise (fun a b => a.1.name ≤ b.1.name) := by
  refine ⟨List.mergeSort_perm _ _, ?_⟩
  have := List.pairwise_mergeSort (le := fun a b : Var × Var => decide (a.1.name ≤ b.1.name))
    (fun a b c h1 h2 => by simp at *; exact String.le_trans h1 h2)
    (fun a b => by simp; exact String.le_total _ _) ps
  exact this.imp (fun h => by simpa using h)

/-! ## `bind_graph_create` -/

/-- The redeclaration check (binder.rs:1575-1644): a node that is a
relationship endpoint is a reference; a bare node or a named relationship
may not reuse a bound name at its first occurrence in the clause. -/
def createStepN (env : Env) (endpoints : List Name) (acc : Except String (List Name)) (n : Name) :
    Except String (List Name) :=
  match acc with
  | .error e => .error e
  | .ok defd =>
    if isAnon n then .ok defd
    else if endpoints.contains n then .ok (n :: defd)
    else if !defd.contains n && env.has n then .error s!"The bound variable '{n}' can't be redeclared in a CREATE clause"
    else .ok (n :: defd)

def createStepR (env : Env) (acc : Except String (List Name)) (n : Name) : Except String (List Name) :=
  match acc with
  | .error e => .error e
  | .ok defd =>
    if isAnon n then .ok defd
    else if !defd.contains n && env.has n then .error s!"The bound variable '{n}' can't be redeclared in a CREATE clause"
    else .ok (n :: defd)

def createCheck (env : Env) (nodes rels : List Name) (endpoints : List Name) : Except String Unit :=
  match rels.foldl (createStepR env) (nodes.foldl (createStepN env endpoints) (.ok [])) with
  | .error e => .error e
  | .ok _ => .ok ()

theorem createCheck_fresh (env : Env) (nodes rels endpoints : List Name) (h : ∀ n, env.has n = false) :
    createCheck env nodes rels endpoints = .ok () := by
  have hN : ∀ (l : List Name) (d : List Name), ∃ d', l.foldl (createStepN env endpoints) (.ok d) = .ok d' := by
    intro l
    induction l with
    | nil => intro d; exact ⟨d, rfl⟩
    | cons n ns ih =>
      intro d; simp only [List.foldl_cons, createStepN, h, Bool.and_false]
      split
      · exact ih _
      · split <;> exact ih _
  have hR : ∀ (l : List Name) (d : List Name), ∃ d', l.foldl (createStepR env) (.ok d) = .ok d' := by
    intro l
    induction l with
    | nil => intro d; exact ⟨d, rfl⟩
    | cons n ns ih => intro d; simp only [List.foldl_cons, createStepR, h, Bool.and_false]; split <;> exact ih _
  obtain ⟨d1, h1⟩ := hN nodes []
  obtain ⟨d2, h2⟩ := hR rels d1
  simp [createCheck, h1, h2]

/-- `MATCH (a) CREATE (a:L)`: a bare node reusing a bound name is rejected. -/
theorem createCheck_bare_bound (env : Env) (a : Name) (ha : isAnon a = false) (hb : env.has a = true) :
    createCheck env [a] [] [] = .error s!"The bound variable '{a}' can't be redeclared in a CREATE clause" := by
  simp [createCheck, createStepN, createStepR, ha, hb]

/-- …but as a relationship endpoint it is a reference (`CREATE (a)-[:R]->(b)`). -/
theorem createCheck_endpoint (env : Env) (a : Name) (ha : isAnon a = false) :
    createCheck env [a] [] [a] = .ok () := by
  simp [createCheck, createStepN, createStepR, ha]

/-- The whole of `bind_graph_create`: redeclaration check, self-reference
check on every entity, attribute maps bound in the clause's incoming scope,
then the pattern. -/
def bindGraphCreate (S : Subs) (st : St) (nodes rels endpoints : List Name) (attrs : List (Name × NE))
    (pat : Nat) : Except String St :=
  match createCheck st.cur nodes rels endpoints with
  | .error e => .error e
  | .ok () =>
    match attrs.find? (fun a => !isAnon a.1 && selfRef a.1 a.2) with
    | some a => .error s!"'{a.1}' not defined; undefined attribute"
    | none => match seqE S.expr st (attrs.map (·.2)) with
      | .error e => .error e
      | .ok st1 => S.graph st1 pat

theorem bindGraphCreate_selfRef (st : St) (a : Name) (e : NE) (ha : isAnon a = false) (hs : selfRef a e = true)
    (hc : createCheck st.cur [] [] [] = .ok ()) :
    bindGraphCreate S st [] [] [] [(a, e)] 0 = .error s!"'{a}' not defined; undefined attribute" := by
  simp [bindGraphCreate, hc, ha, hs]

end FalkorBinder
