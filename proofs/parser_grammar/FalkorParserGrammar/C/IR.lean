/-
# `QueryIR` and its validation (`graph/src/parser/ast.rs:885-1395`)
-/
import FalkorParserGrammar.C.Graph

namespace FalkorParserGrammar.C

/-- `SetItem` (ast.rs:891-898). -/
inductive SetItem
  | attr (target value : ET) (replace : Bool)
  | label (v : Nat) (ls : List Nat)

/-- The projection part shared by `With` and `Return`. -/
structure Proj where
  distinct : Bool
  all : Bool
  exprs : List (Nat × ET)
  order : List (ET × Bool)
  skip : Option ET
  limit : Option ET

inductive IdxTy | range | fulltext | vector
  deriving DecidableEq, Repr

/-- `QueryIR<Arc<String>>` (ast.rs:928-1040); names are ids, a procedure is
its dotted name. -/
inductive IR
  | call (f : List Nat) (args : List ET) (yields : List Nat) (aliases : List (Option Nat))
      (filter : Option ET) (explicit : Bool)
  | match_ (p : QG Nat) (filter : Option ET) (optional : Bool)
  | unwind (e : ET) (v : Nat)
  | merge (p : QG Nat) (onCreate onMatch : List SetItem)
  | create (p : QG Nat)
  | delete (es : List ET) (detach : Bool)
  | set (items : List SetItem)
  | remove (items : List ET)
  | loadCsv (path : ET) (headers : Bool) (delim : ET) (v : Nat)
  | with_ (p : Proj) (filter : Option ET) (write : Bool)
  | return_ (p : Proj) (write : Bool)
  | createIndex (label : Nat) (attrs : List Nat) (ity : IdxTy) (rel : Bool) (options : Option ET)
  | dropIndex (label : Nat) (attrs : List Nat) (ity : IdxTy) (rel : Bool)
  | union (bs : List IR) (all : Bool)
  | query (cs : List IR) (write : Bool)
  | forEach (l : ET) (v : Nat) (body : List IR)
  | callSub (body : IR) (ret : Bool)

/-- The two procedure names `inner_validate` looks at (ast.rs:1171, 1194). -/
def procCNI : List Nat := [3000001]    -- db.idx.fulltext.createNodeIndex
def procFTDrop : List Nat := [3000002] -- db.idx.fulltext.drop
/-- The string `"label"` (ast.rs:1177). -/
def strLabel : Nat := 3000003

/-- Validation outcome; `panic` is an out-of-bounds `args[0]` / `child(0)`. -/
inductive V | ok | err | panic
  deriving DecidableEq, Repr

def V.andThen : V → V → V
  | .ok, b => b
  | a, _ => a

/-- `validate_set_items` (ast.rs:1329-1345). -/
def vSetItems : List SetItem → V
  | [] => .ok
  | .label _ _ :: is => vSetItems is
  | .attr t _ _ :: is =>
    match t with
    | .node (.prop _) [] => .panic                          -- `child(0)`
    | .node (.prop _) (.node (.var _) _ :: _) => vSetItems is
    | .node (.prop _) _ => .err
    | .node (.var _) _ => vSetItems is
    | _ => .err

/-- `validate_inlined_properties` (ast.rs:1348-1367). -/
def vInlined (p : QG Nat) : V :=
  if p.nodes.all (fun n => n.attrs.root == .map) && p.rels.all (fun r => r.attrs.root == .map)
  then .ok else .err

/-- `validate_inlined_properties`: every node and relationship property part
is a map literal (a `$param` is rejected, as in C). -/
theorem vInlined_spec (p : QG Nat) :
    vInlined p = .ok ↔ (∀ n ∈ p.nodes, n.attrs.root = .map) ∧ (∀ r ∈ p.rels, r.attrs.root = .map) := by
  unfold vInlined
  split <;> simp_all

/-- The REMOVE check (ast.rs:1235-1240). -/
def vRemove : List ET → V
  | [] => .ok
  | .node (.prop _) [] :: _ => .panic
  | .node (.prop _) (.node .null _ :: _) :: _ => .err
  | _ :: is => vRemove is

/-- The CALL check (ast.rs:1167-1201). It returns without looking at the
clauses after the CALL. -/
def vCall (f : List Nat) (args : List ET) : V :=
  let first : V :=
    if f = procCNI then
      match args with
      | [] => .panic                                          -- `args[0]`
      | .node (.str _) _ :: _ => .ok
      | .node .map es :: _ =>
        if es.any (fun e => e.root == .str strLabel) then .ok else .err
      | _ => .err
    else .ok
  first.andThen
    (if f = procFTDrop ∧ !(match args with | .node (.str _) _ :: _ => true | _ => false) then .err
     else .ok)

def relTypesOne (p : QG Nat) : Bool := p.rels.all (fun r => r.types.length == 1)

mutual
/-- `return_column_names` (ast.rs:1370-1392). -/
def returnCols : IR → List Nat
  | .query cs _ => (colsL cs).getD []
  | _ => []
/-- The scan `for clause in clauses.iter().rev()`: the last clause that names
columns wins. -/
def colsL : List IR → Option (List Nat)
  | [] => none
  | c :: cs => match colsL cs with
    | some r => some r
    | none => clauseCols c
def clauseCols : IR → Option (List Nat)
  | .return_ p _ => some (p.exprs.map (·.1))
  | .call _ _ ys _ _ _ => some ys
  | .callSub b true => some (returnCols b)
  | _ => none
end

theorem sz_list_pos (l : List IR) : 0 < sizeOf l := by cases l <;> simp <;> omega
theorem sz_ir_pos (c : IR) : 0 < sizeOf c := by cases c <;> simp <;> omega

mutual
/-- `inner_validate(self, iter)` (ast.rs:1157-1326). -/
def vClause : IR → List IR → V
  | .call f args _ _ _ _, _ => vCall f args
  | .match_ p _ _, rest => (vInlined p).andThen (match rest with
      | [] => V.err | c :: r => vClause c r)
  | .unwind _ _, rest => match rest with
      | [] => V.err | c :: r => vClause c r
  | .merge p oc om, rest =>
      (vInlined p).andThen ((if relTypesOne p then V.ok else V.err).andThen
        ((vSetItems oc).andThen ((vSetItems om).andThen (vNext rest))))
  | .create p, rest =>
      (vInlined p).andThen ((if relTypesOne p then V.ok else V.err).andThen (vNext rest))
  | .delete _ _, rest => vNext rest
  | .set items, rest => (vSetItems items).andThen (vNext rest)
  | .remove items, rest => (vRemove items).andThen (vNext rest)
  | .loadCsv _ _ _ _, rest => vNext rest
  | .with_ _ _ _, rest => vNext rest
  | .return_ _ _, rest => vNext rest
  | .createIndex _ _ _ _ _, rest => vNext rest
  | .dropIndex _ _ _ _, rest => vNext rest
  | .query [] _, _ => .err                                   -- "Error: empty query."
  | .query (c :: cs) _, _ => vClause c cs
  | .union bs _, _ => vUnion bs none
  | .forEach _ _ body, rest => (vAll body).andThen (vNext rest)
  | .callSub b r, rest => (vClause b []).andThen (if r then (match rest with
      | [] => V.err | c :: r => vClause c r) else vNext rest)
termination_by c rest => sizeOf c + sizeOf rest
decreasing_by
  all_goals simp_wf
  all_goals first
    | omega
    | (have := sz_list_pos rest; have := sz_ir_pos c; omega)
    | (have := sz_list_pos rest; omega)
    | skip
/-- `iter.next().map_or(Ok(()), |first| first.inner_validate(iter))`. -/
def vNext : List IR → V
  | [] => .ok
  | c :: r => vClause c r
termination_by l => sizeOf l
/-- `for clause in body { clause.validate()? }`. -/
def vAll : List IR → V
  | [] => .ok
  | c :: r => (vClause c []).andThen (vAll r)
termination_by l => sizeOf l
decreasing_by all_goals (simp_wf; have := sz_list_pos r; omega)
/-- The UNION arm: validate each branch, compare column names. -/
def vUnion : List IR → Option (List Nat) → V
  | [], _ => .ok
  | b :: bs, cols => (vClause b []).andThen (match cols with
      | none => vUnion bs (some (returnCols b))
      | some e => if returnCols b = e then vUnion bs cols else .err)
termination_by l => sizeOf l
decreasing_by all_goals (simp_wf; have := sz_list_pos bs; omega)
end

/-- `QueryIR::validate` (ast.rs:1151-1153). -/
def validate (q : IR) : V := vClause q []

/-! ## What validation is meant to establish, and where it falls short -/

/-- A clause that may end a query (the "must be a RETURN clause, an update
clause, a procedure call or a non-returning subquery" rule). -/
def canConclude : IR → Bool
  | .match_ _ _ _ | .unwind _ _ => false
  | .callSub _ true => false
  | _ => true

/-- **The query-end rule holds when no CALL precedes the end**: for a query
made of MATCH / UNWIND / RETURN / WITH clauses only, `validate` succeeds iff
the list is non-empty and the last clause may conclude. -/
def simpleClause : IR → Bool
  | .match_ _ _ _ | .unwind _ _ | .with_ _ _ _ | .return_ _ _ => true
  | _ => false

theorem vClause_simple (p : QG Nat) (hp : vInlined p = .ok) :
    ∀ (c : IR) (rest : List IR), (∀ x ∈ c :: rest, simpleClause x = true) →
      (∀ x ∈ c :: rest, ∀ q f o, x = .match_ q f o → q = p) →
      (vClause c rest = .ok ↔ canConclude ((c :: rest).getLast (by simp)) = true)
  | c, [], hs, hm => by
    have := hs c (by simp)
    cases c <;> simp_all [simpleClause, vClause, vNext, canConclude, V.andThen]
  | c, d :: rest, hs, hm => by
    have ih := vClause_simple p hp d rest (fun x h => hs x (List.mem_cons_of_mem _ h))
      (fun x h => hm x (List.mem_cons_of_mem _ h))
    have hc := hs c (by simp)
    have hmc := hm c (by simp)
    cases c <;> simp_all [simpleClause, vClause, vNext, V.andThen] <;> (try subst_vars) <;>
      simp_all [V.andThen]

/-- **BUG (ast.rs:1167-1201)**: the CALL arm returns `Ok(())` and never
validates the clauses after it. A query ending in MATCH after a CALL, or a
CREATE of an untyped relationship after a CALL, passes validation:
`CALL db.labels() YIELD label MATCH (n)` and
`CALL db.labels() YIELD label CREATE ()-[r]->() RETURN label` both crash
the server (live; C rejects both). -/
theorem call_skips_rest_validation (f : List Nat) (hf : f ≠ procCNI) (hd : f ≠ procFTDrop)
    (rest : List IR) : validate (.query (.call f [] [] [] none true :: rest) false) = .ok := by
  simp [validate, vClause, vCall, hf, hd, V.andThen]

theorem match_last_rejected (p : QG Nat) (hp : vInlined p = .ok) :
    validate (.query [.match_ p none false] false) = .err := by
  simp [validate, vClause, hp, V.andThen]

/-- The same MATCH-terminated tail is accepted once a CALL precedes it. -/
theorem match_last_after_call_accepted (p : QG Nat) :
    validate (.query [.call [5] [] [] [] none true, .match_ p none false] false) = .ok := by
  simp [validate, vClause, vCall, procCNI, procFTDrop, V.andThen]

/-- `body_has_return` (cypher.rs:1089-1097). -/
def bodyHasReturn : IR → Bool
  | .query cs _ => match cs.getLast? with
    | some (.return_ _ _) => true
    | _ => false
  | .union bs _ => bsAll bs
  | _ => false
where
  bsAll : List IR → Bool
  | [] => true
  | b :: bs => bodyHasReturn b && bsAll bs

theorem bodyHasReturn_query (cs : List IR) (w : Bool) :
    bodyHasReturn (.query cs w) = true ↔ ∃ p w', cs.getLast? = some (.return_ p w') := by
  simp only [bodyHasReturn]
  split <;> simp_all

theorem bodyHasReturn_union (bs : List IR) (a : Bool) :
    bodyHasReturn (.union bs a) = true ↔ ∀ b ∈ bs, bodyHasReturn b = true := by
  simp only [bodyHasReturn]
  induction bs with
  | nil => simp [bodyHasReturn.bsAll]
  | cons b bs ih => simp [bodyHasReturn.bsAll, ih]

/-- `return_column_names`: the last RETURN's names, a CALL's yields, or a
returning subquery's columns, scanning from the end. -/
theorem colsL_append_return (cs : List IR) (p : Proj) (w : Bool) :
    colsL (cs ++ [.return_ p w]) = some (p.exprs.map (·.1)) := by
  induction cs with
  | nil => simp [colsL, clauseCols]
  | cons c cs ih => simp [colsL, ih]

theorem returnCols_last_return (cs : List IR) (p : Proj) (w b : Bool) :
    returnCols (.query (cs ++ [.return_ p w]) b) = p.exprs.map (·.1) := by
  simp [returnCols, colsL_append_return]

/-- **No panic in validation of a parsed query**: the `child(0)` /
`args[0]` sites cannot fire when every property node has a child (the
parser builds properties only through `parse_property_lookup`, which
always attaches one) and createNodeIndex has an argument (its registry
entry requires one: `args: [Type::Map]`, procedures.rs:328-330, checked by
`func.validate` at cypher.rs:1048). -/
def ET.propOk : ET → Bool
  | .node (.prop _) [] => false
  | _ => true

theorem vSetItems_np : ∀ is : List SetItem,
    (∀ t v r, SetItem.attr t v r ∈ is → t.propOk = true) → vSetItems is ≠ .panic
  | [], _ => by simp [vSetItems]
  | .label _ _ :: is, h => by
    simp only [vSetItems]; exact vSetItems_np is (fun t v r m => h t v r (by simp [m]))
  | .attr t v r :: is, h => by
    have ht := h t v r (by simp)
    have ih := vSetItems_np is (fun t v r m => h t v r (by simp [m]))
    simp only [vSetItems]
    split <;> simp_all [ET.propOk]

theorem vRemove_np : ∀ is : List ET, (∀ t ∈ is, t.propOk = true) → vRemove is ≠ .panic
  | [], _ => by simp [vRemove]
  | t :: is, h => by
    have ht := h t (by simp)
    have ih := vRemove_np is (fun x m => h x (by simp [m]))
    unfold vRemove; split <;> simp_all [ET.propOk]

theorem V.andThen_np (x y : V) (hx : x ≠ .panic) (hy : y ≠ .panic) : x.andThen y ≠ .panic := by
  cases x <;> simp_all [V.andThen]

theorem vCall_np (f : List Nat) (args : List ET) (h : f = procCNI → args ≠ []) :
    vCall f args ≠ .panic := by
  unfold vCall
  apply V.andThen_np
  · split
    · rename_i hf
      cases args with
      | nil => exact absurd rfl (h hf)
      | cons a as =>
        split <;> (try split) <;> simp_all
    · simp
  · split <;> (try split) <;> simp

/-! ## `is_aggregation` (ast.rs:440-466) -/

/-- Both `SupportAggregation::is_aggregation` impls (bound and raw trees)
are `find_aggregate_name(..).is_some()` up to the tree's variable type. -/
def isAggregation (t : ET) : Bool := (aggName t).isSome
theorem isAggregation_spec (t : ET) : isAggregation t = true ↔ HasAgg t :=
  find_aggregate_name_spec t

end FalkorParserGrammar.C
