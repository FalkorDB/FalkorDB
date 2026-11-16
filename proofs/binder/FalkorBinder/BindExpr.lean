import FalkorBinder.Resolve
/-
# Binding expressions: `bind_expr_node` and its wrappers

| here | there |
| --- | --- |
| `NE` / `BE` | raw `ExprIR<Arc<String>>` / bound `ExprIR<Variable>` trees (the variants the binder treats specially) |
| `reserve` | `fresh_var` + insert under `_quant_{s}_{id}` / `_lc_…` / `_reduce_acc_…` / `_reduce_iter_…` (binder.rs:1762-1765, 1796-1799, 1838-1844) |
| `bindNode`, `bindL` | `bind_expr_node` binder.rs:1836-2260 |
| `Alias`, `defineAlias`, `lookupAlias`, `resolveAlias`, `resolveNamed` | `define_alias_in_scope` :2287, `lookup_alias` :2312, `resolve_alias` :2324, the Pattern arm's `resolve_bound_name` loop :2182-2186 (#3528) |
| `NE.paren`, `bindNode` on it | the `Paren` early return binder.rs:1955-1965 (#3335): `(x)` binds to `x`'s tree |
| `NE.andOr`, `BE.andOr`, `flatB` | `And`/`Or` and the `ExprIR::and_of`/`or_of` return binder.rs:2137-2144, ast.rs:342-380 (#3335) |
| `bindExpr`, `bindExprLocals` | `bind_expr` binder.rs:1725-1730, `bind_expr_with_locals` binder.rs:1732-1739 |
| `bindSetItems` | `bind_set_items` binder.rs:1700-1723 |
| `selfRef`, `checkSelfRef` | `check_unbound_self_referential` binder.rs:1772-1789 |

`bind_graph` (for pattern predicates / comprehensions) defines each pattern
name in the current table (`defineName … true`, Pattern.lean has the finer
node/relationship order). The constant fold and call rewrites are Fold.lean;
labels are Scopes.lean.

Theorems: `reserve_covers` / `bindNode_covers` (every reservation keeps the
table covering; only pattern cleanup can break it — `comprehension_cleanup_breaks`),
`bind_iterable_outer` (a comprehension's list is bound before its variable),
`loop_var_shadows`, `reduce_same_name`, `prop_on_path`, `exists_pattern`,
`func_arg_mismatch`, `bindExpr_eq`, `bindSetItems_*`, `checkSelfRef_iff`.
#3528: `exists_pattern_first`/`exists_paren_pattern` (checked before binding, through parens),
`patternPred_declares_nothing` (#1334 fixed), `defineAlias_anon_*`, `anon_spelled_name_unbound`.

#3335: `bind_paren` (`(x)` binds exactly like `x`: same tree, same state, same
errors — so no bound tree holds a `Paren`, which `BE` has no constructor for),
`prop_on_paren_path` (the `(p).prop` rejection now comes from that, not from a
look-through in the `Property` arm), `bind_andOr_paren` / `flatB_assoc` (a
parenthesised run of the same connective binds flat: `(a AND b) AND c` gives
the bound tree of `a AND b AND c`), `spliceB_other` (a different connective is a
single operand: it splices as itself).
-/
namespace FalkorBinder

inductive NE
  | var (n : Name)
  | quant (x : Name) (cs : List NE)
  | listComp (x : Name) (cs : List NE)
  | reduce (acc it : Name) (cs : List NE)
  | pattern (vars : List Alias)
  | patComp (vars : List Alias) (cs : List NE)
  | prop (k : Name) (cs : List NE)
  | func (name : Name) (argTys : List Ty) (cs : List NE)
  /-- `(…)`: one child (the parser's grouping) -/
  | paren (cs : List NE)
  /-- `AND` (`isOr = false`) / `OR` (`isOr = true`) -/
  | andOr (isOr : Bool) (cs : List NE)
  /-- every other payload (not `Paren`, `And`, `Or`) -/
  | other (tag : Nat) (cs : List NE)

inductive BE
  | var (v : Var)
  | quant (v : Var) (cs : List BE)
  | listComp (v : Var) (cs : List BE)
  | reduce (a i : Var) (cs : List BE)
  | pattern (vs : List Var)
  | patComp (vs : List Var) (cs : List BE)
  | prop (k : Name) (cs : List BE)
  | func (name : Name) (cs : List BE)
  | andOr (isOr : Bool) (cs : List BE)
  /-- `Constant(Bool(b))` -/
  | bool (b : Bool)
  | other (tag : Nat) (cs : List BE)

/-- What the binder reads of its own state while binding an expression. -/
structure Ctx where
  /-- the reserved key for a minted loop variable: prefix, scope, id -/
  rkey : String → Nat → Nat → Name
  /-- `expr_may_return_boolean` on the bound WHERE of a list comprehension -/
  mayBool : BE → Bool
  /-- `(scope, id)` of variable-length relationship variables -/
  varlen : List (Nat × Nat)
  /-- `Type::is_compatible_with` -/
  compat : Ty → Ty → Bool

variable (C : Ctx)

/-- Mint a loop variable and reserve its slot under a unique key. -/
def reserve (st : St) (pfx : String) (x : Name) : St × Var :=
  let v := freshVar st.cur x .any st.depth
  ({ st with cur := st.cur.insert (C.rkey pfx st.depth v.id) v }, v)

/-- `Binder::define_alias_in_scope` (binder.rs:2287-2309, #3528), with `allow_reuse`: a name is
`define_name_in_scope`d; the `k`-th unnamed entity reuses its `anon_vars` entry, else mints a
nameless variable (name `""` stands for `None`), pushed unnamed and recorded in `anon_vars`. -/
def defineAlias (st : St) : Alias → Except String (St × Var)
  | .named n => match defineName st.cur st.depth n .any true with
    | .error e => .error e
    | .ok (e, v) => .ok ({ st with cur := e }, v)
  | .anon k => match st.anon.lookup k with
    | some v => .ok (st, v)
    | none =>
      let v := freshVar st.cur "" .any st.depth
      .ok ({ st with cur := st.cur.pushUnnamed v, anon := (k, v) :: st.anon }, v)

/-- `bind_graph` on a pattern's aliases: each defined (reused if present). -/
def bindNames (st : St) : List Alias → Except String (St × List Var)
  | [] => .ok (st, [])
  | a :: as => match defineAlias st a with
    | .error e => .error e
    | .ok (st1, v) => match bindNames st1 as with
      | .error e => .error e
      | .ok (st', vs) => .ok (st', v :: vs)

/-- `Binder::lookup_alias` (binder.rs:2312-2320). -/
def lookupAlias (st : St) : Alias → Option Var
  | .named n => st.cur.find? n
  | .anon k => st.anon.lookup k

/-- `Binder::resolve_alias` (binder.rs:2324-2336). -/
def resolveAlias (st : St) : Alias → Except String (St × Var)
  | .named n => resolve st [] n
  | .anon k => match st.anon.lookup k with
    | some v => .ok (st, v)
    | none => .error s!"unnamed entity @anon_{k} is not bound"

/-- An unnamed entity's first occurrence mints a nameless variable at `len()`, which no name
reaches; later ones (`anon_vars`) reuse it. -/
theorem defineAlias_anon_fresh (st : St) (k : Nat) (h : st.anon.lookup k = none) :
    defineAlias st (.anon k) =
      .ok ({ st with cur := st.cur.pushUnnamed (freshVar st.cur "" .any st.depth),
                     anon := (k, freshVar st.cur "" .any st.depth) :: st.anon },
           freshVar st.cur "" .any st.depth) ∧
    ∀ n, (st.cur.pushUnnamed (freshVar st.cur "" .any st.depth)).find? n = st.cur.find? n := by
  refine ⟨by simp [defineAlias, h], fun n => Env.find?_pushUnnamed _ _ n⟩

theorem defineAlias_anon_reuse (st : St) (k : Nat) (v : Var) (h : st.anon.lookup k = some v) :
    defineAlias st (.anon k) = .ok (st, v) := by
  simp [defineAlias, h]

theorem defineAlias_named (st : St) (n : Name) :
    defineAlias st (.named n) = (defineName st.cur st.depth n .any true).map
      (fun r => ({ st with cur := r.1 }, r.2)) := by
  simp only [defineAlias]; cases defineName st.cur st.depth n .any true <;> rfl

theorem lookupAlias_spec (st : St) (n : Name) (k : Nat) :
    lookupAlias st (.named n) = st.cur.find? n ∧ lookupAlias st (.anon k) = st.anon.lookup k :=
  ⟨rfl, rfl⟩

theorem resolveAlias_named (st : St) (n : Name) : resolveAlias st (.named n) = resolve st [] n := rfl

theorem resolveAlias_anon (st : St) (k : Nat) :
    (∃ r, resolveAlias st (.anon k) = .ok r) ↔ (st.anon.lookup k).isSome := by
  simp only [resolveAlias]; cases st.anon.lookup k <;> simp

def injectLocals (e : Env) (locals : List Env) : Env :=
  locals.foldl (fun e l => l.foldl (fun e kv => match kv.1 with
    | some k => e.insert k kv.2
    | none => e) e) e

/-- The cleanup after a pattern: keep the outer names (`Scope::retain` keeps every unnamed entry). -/
def cleanup (outer : List Name) (e : Env) : Env := e.retain fun k => outer.contains k

/-- The pattern predicate's first step (binder.rs:2182-2186, #3528): every named alias is
resolved with `resolve_bound_name`, threading the state (a parent copy is kept). -/
def resolveNamed (st : St) (locals : List Env) : List Alias → Except String St
  | [] => .ok st
  | .anon _ :: as => resolveNamed st locals as
  | .named n :: as => match resolve st locals n with
    | .error e => .error e
    | .ok (st1, _) => resolveNamed st1 locals as

/-- `exists`'s argument check (binder.rs:2044-2054): a `Pattern`, looking through parentheses. -/
def isPatNE : NE → Bool
  | .pattern _ => true
  | .paren (c :: _) => isPatNE c
  | _ => false

def firstVar : List BE → Option Var
  | BE.var v :: _ => some v
  | _ => none

/-- What one bound operand contributes to `flat`: its children when it is the same connective. -/
def spliceB (o : Bool) : BE → List BE
  | .andOr o' cs => if o' = o then cs else [.andOr o' cs]
  | b => [b]

/-- `ExprIR::flat` (ast.rs:355-380) on bound operands: a lone operand of another
kind is returned as is; else same-connective operands are spliced in place;
none left gives the identity (`true` for AND, `false` for OR), one is returned alone. -/
def flatB (o : Bool) (bs : List BE) : BE :=
  let body := match bs.flatMap (spliceB o) with
    | [] => .bool (!o)
    | [c] => c
    | cs => .andOr o cs
  match bs with
  | [b] => match b with
    | .andOr o' _ => if o' = o then body else b
    | _ => b
  | _ => body

mutual
def bindNode (st : St) (locals : List Env) : NE → Except String (BE × St)
  | .var n => match resolve st locals n with
    | .ok (st', v) => .ok (.var v, st')
    | .error e => .error e
  | .quant x cs => match cs with
    | c0 :: rest =>
      let r := reserve C st "_quant_" x
      match bindNode r.1 locals c0 with
      | .error e => .error e
      | .ok (b0, st1) => match bindL st1 (locals ++ [[(some x, r.2)]]) rest with
        | .error e => .error e
        | .ok (bs, st2) => .ok (.quant r.2 (b0 :: bs), st2)
    | [] => .error "unreachable: quantifier without list"
  | .listComp x cs => match cs with
    | c0 :: rest =>
      let r := reserve C st "_lc_" x
      match bindNode r.1 locals c0 with
      | .error e => .error e
      | .ok (b0, st1) => match bindL st1 (locals ++ [[(some x, r.2)]]) rest with
        | .error e => .error e
        | .ok (bs, st2) =>
          match bs with
          | w :: _ => if C.mayBool w then .ok (.listComp r.2 (b0 :: bs), st2)
                      else .error "Expected boolean predicate"
          | [] => .ok (.listComp r.2 [b0], st2)
    | [] => .error "unreachable: list comprehension without list"
  | .reduce acc it cs =>
    if acc = it then .error s!"Variable `{acc}` already declared"
    else bindReduce st locals acc it cs
  | .pattern vars =>
    match resolveNamed st locals vars with
    | .error e => .error e
    | .ok st0 =>
    let outer := st0.cur.names
    match bindNames { st0 with cur := injectLocals st0.cur locals } vars with
    | .error e => .error e
    | .ok (st1, vs) => .ok (.pattern vs, { st1 with cur := cleanup outer st1.cur })
  | .patComp vars cs =>
    let outer := st.cur.names
    match bindNames { st with cur := injectLocals st.cur locals } vars with
    | .error e => .error e
    | .ok (st1, vs) => match bindL st1 locals cs with
      | .error e => .error e
      | .ok (bs, st2) => .ok (.patComp vs bs, { st2 with cur := cleanup outer st2.cur })
  | .prop k cs => match bindL st locals cs with
    | .error e => .error e
    | .ok (bs, st1) =>
      match firstVar bs with
      | some v => if v.ty = .path ∧ (v.scope, v.id) ∉ C.varlen then
                    .error "Type mismatch: expected Map, Node, Edge, Datetime, Date, Time, Duration, Null, or Point but was Path"
                  else .ok (.prop k bs, st1)
      | none => .ok (.prop k bs, st1)
  | .func name tys cs =>
    if name = "exists" ∧ cs.any isPatNE then
      .error "Invalid input: traversal patterns are not allowed as arguments to exists()"
    else match bindL st locals cs with
    | .error e => .error e
    | .ok (bs, st1) =>
      if name ≠ "hasLabels" ∧ ((tys.zip bs).any fun tb => match tb.2 with
          | .var v => !C.compat v.ty tb.1
          | _ => false) then .error "Type mismatch"
      else .ok (.func name bs, st1)
  | .paren cs => match bindL st locals cs with
    | .error e => .error e
    | .ok (bs, st1) => match bs.getLast? with
      | some b => .ok (b, st1)
      | none => .error "Empty parenthesised expression"
  | .andOr o cs => match bindL st locals cs with
    | .error e => .error e
    | .ok (bs, st1) => .ok (flatB o bs, st1)
  | .other t cs => match bindL st locals cs with
    | .error e => .error e
    | .ok (bs, st1) => .ok (.other t bs, st1)
/-- `reduce(acc = init, it IN list | body)`: init and list in the outer scope,
the body with both names local. -/
def bindReduce (st : St) (locals : List Env) (acc it : Name) : List NE → Except String (BE × St)
  | [i0, l0, b0] =>
    let ra := reserve C st "_reduce_acc_" acc
    let ri := reserve C ra.1 "_reduce_iter_" it
    match bindNode ri.1 locals i0 with
    | .error e => .error e
    | .ok (bi, st1) => match bindNode st1 locals l0 with
      | .error e => .error e
      | .ok (bl, st2) => match bindNode st2 (locals ++ [[(some acc, ra.2), (some it, ri.2)]]) b0 with
        | .error e => .error e
        | .ok (bb, st3) => .ok (.reduce ra.2 ri.2 [bi, bl, bb], st3)
  | _ => .error "unreachable: reduce without three children"
def bindL (st : St) (locals : List Env) : List NE → Except String (List BE × St)
  | [] => .ok ([], st)
  | c :: cs => match bindNode st locals c with
    | .error e => .error e
    | .ok (b, st1) => match bindL st1 locals cs with
      | .error e => .error e
      | .ok (bs, st2) => .ok (b :: bs, st2)
end

def bindExprLocals (st : St) (locals : List Env) (e : NE) : Except String (BE × St) := bindNode C st locals e
def bindExpr (st : St) (e : NE) : Except String (BE × St) := bindExprLocals C st [] e

theorem bindExpr_eq (st : St) (e : NE) : bindExpr C st e = bindNode C st [] e := rfl

/-! ## Scoping facts -/

theorem lookupLocals_last (locals : List Env) (x : Name) (v : Var) :
    lookupLocals (locals ++ [[(some x, v)]]) x = some v := by
  induction locals with
  | nil => simp [lookupLocals, Env.find?, List.lookup]
  | cons l ls ih => simp [lookupLocals, ih]

/-- The loop variable shadows everything inside the comprehension body. -/
theorem loop_var_shadows (st : St) (locals : List Env) (x : Name) (v : Var) :
    resolve st (locals ++ [[(some x, v)]]) x = .ok (st, v) := by
  unfold resolve; rw [lookupLocals_last]

/-- The list of `[x IN list | …]` is bound with the *outer* locals (before
`x` exists), in the state after the slot reservation. -/
theorem bind_iterable_outer (st : St) (locals : List Env) (x : Name) (c0 : NE) (rest : List NE)
    (b : BE) (st' : St) (h : bindNode C st locals (.listComp x (c0 :: rest)) = .ok (b, st')) :
    ∃ b0 st1, bindNode C (reserve C st "_lc_" x).1 locals c0 = .ok (b0, st1) := by
  simp only [bindNode] at h
  cases h0 : bindNode C (reserve C st "_lc_" x).1 locals c0 with
  | error e => rw [h0] at h; simp at h
  | ok p => exact ⟨p.1, p.2, rfl⟩

theorem reduce_same_name (st : St) (locals : List Env) (a : Name) (cs : List NE) :
    bindNode C st locals (.reduce a a cs) = .error s!"Variable `{a}` already declared" := by
  simp [bindNode]

/-- `r.prop` on a named path (not a `[r*]` variable) is rejected. -/
theorem prop_on_path (st : St) (locals : List Env) (k : Name) (n : Name) (v : Var) (st1 : St)
    (hr : resolve st locals n = .ok (st1, v)) (hp : v.ty = .path) (hv : (v.scope, v.id) ∉ C.varlen) :
    ∃ e, bindNode C st locals (.prop k [.var n]) = .error e := by
  simp [bindNode, bindL, hr, firstVar, hp, hv]

theorem exists_pattern (st : St) (locals : List Env) (vars : List Alias) (bp : BE) (st1 : St)
    (h : bindNode C st locals (.pattern vars) = .ok (bp, st1)) (hp : ∃ vs, bp = .pattern vs) :
    ∃ e, bindNode C st locals (.func "exists" [] [.pattern vars]) = .error e := by
  simp [bindNode, isPatNE]

/-- #3528: `exists` rejects a pattern argument before binding it, whatever its variables
(so the error is this one even when the pattern names an unbound variable), and looks through
parentheses: `exists(((p)-->()))` too. -/
theorem exists_pattern_first (st : St) (locals : List Env) (tys : List Ty) (cs : List NE)
    (h : cs.any isPatNE = true) :
    bindNode C st locals (.func "exists" tys cs) =
      .error "Invalid input: traversal patterns are not allowed as arguments to exists()" := by
  simp [bindNode, h]

theorem exists_paren_pattern (st : St) (locals : List Env) (vars : List Alias) :
    bindNode C st locals (.func "exists" [] [.paren [.paren [.pattern vars]]]) =
      .error "Invalid input: traversal patterns are not allowed as arguments to exists()" :=
  exists_pattern_first C st locals [] _ (by simp [isPatNE])

/-- A variable argument of an incompatible declared type is rejected (except `hasLabels`). -/
theorem func_arg_mismatch (st : St) (locals : List Env) (name n : Name) (t : Ty) (v : Var) (st1 : St)
    (hr : resolve st locals n = .ok (st1, v)) (hc : C.compat v.ty t = false) (hn : name ≠ "exists")
    (hh : name ≠ "hasLabels") : bindNode C st locals (.func name [t] [.var n]) = .error "Type mismatch" := by
  simp [bindNode, bindL, hr, hc, hn, hh]

/-! ## #3335: parentheses and AND/OR flattening -/

/-- **`(x)` binds exactly like `x`** (binder.rs:1955-1965): same bound tree, same
state, same errors. -/
theorem bind_paren (st : St) (locals : List Env) (c : NE) :
    bindNode C st locals (.paren [c]) = bindNode C st locals c := by
  simp only [bindNode, bindL]
  cases bindNode C st locals c with
  | error e => rfl
  | ok p => rfl

/-- `(p).prop` on a named path is still rejected — the child binds to the path
variable itself, so the `Property` arm's first-child test sees it (the
pre-#3335 look-through of a `Paren` child is no longer needed). -/
theorem prop_on_paren_path (st : St) (locals : List Env) (k : Name) (n : Name) (v : Var) (st1 : St)
    (hr : resolve st locals n = .ok (st1, v)) (hp : v.ty = .path) (hv : (v.scope, v.id) ∉ C.varlen) :
    ∃ e, bindNode C st locals (.prop k [.paren [.var n]]) = .error e := by
  simp [bindNode, bindL, hr, firstVar, hp, hv]

/-- The early return of `flat` is a shortcut for the general body. -/
theorem flatB_eq_body (o : Bool) (bs : List BE) :
    flatB o bs = match bs.flatMap (spliceB o) with
      | [] => .bool (!o)
      | [c] => c
      | cs => .andOr o cs := by
  unfold flatB
  match bs with
  | [] => rfl
  | [b] =>
    cases b with
    | andOr o' cs =>
      by_cases h : o' = o
      · simp [h]
      · simp [h, spliceB]
    | _ => rfl
  | _ :: _ :: _ => rfl

theorem spliceB_same (o : Bool) (cs : List BE) : spliceB o (.andOr o cs) = cs := by simp [spliceB]

/-- AND under OR (and OR under AND) is never spliced: it stays one operand. -/
theorem spliceB_other (o : Bool) (cs : List BE) : spliceB o (.andOr (!o) cs) = [.andOr (!o) cs] := by
  cases o <;> simp [spliceB]

/-- With two or more operands after splicing, the result is the flat node. -/
theorem flatB_many (o : Bool) (bs : List BE) (h : 2 ≤ (bs.flatMap (spliceB o)).length) :
    flatB o bs = .andOr o (bs.flatMap (spliceB o)) := by
  rw [flatB_eq_body]
  generalize bs.flatMap (spliceB o) = l at h ⊢
  match l, h with
  | _ :: _ :: _, _ => rfl

/-- Associativity: a flat run nested as the first operand joins its parent. -/
theorem flatB_assoc (o : Bool) (xs ys : List BE) (h : 2 ≤ (xs.flatMap (spliceB o)).length) :
    flatB o (flatB o xs :: ys) = flatB o (xs ++ ys) := by
  rw [flatB_many o xs h, flatB_eq_body, flatB_eq_body, List.flatMap_cons, spliceB_same,
    List.flatMap_append]

theorem bindL_append (locals : List Env) (xs ys : List NE) : ∀ st : St,
    bindL C st locals (xs ++ ys) =
      match bindL C st locals xs with
      | .error e => .error e
      | .ok (bs, st1) => match bindL C st1 locals ys with
        | .error e => .error e
        | .ok (bs2, st2) => .ok (bs ++ bs2, st2) := by
  induction xs with
  | nil => intro st; simp only [List.nil_append, bindL]; cases bindL C st locals ys <;> rfl
  | cons x xs ih =>
    intro st
    simp only [List.cons_append, bindL]
    rcases bindNode C st locals x with e | ⟨b, s1⟩
    · rfl
    · simp only [ih]
      rcases bindL C s1 locals xs with e | ⟨bs, s2⟩
      · rfl
      · simp only
        rcases bindL C s2 locals ys with e | ⟨bs2, s3⟩
        · rfl
        · rfl

/-- **A parenthesised run of the same connective binds flat**: `(a AND b) AND c`
binds to the tree of `a AND b AND c` (likewise OR), whenever the grouped operands
leave at least two after splicing (always, for the parser's two-operand
connectives). Filters are then split into all their conjuncts (#3331, #3187). -/
theorem bind_andOr_paren (st : St) (locals : List Env) (o : Bool) (xs ys : List NE)
    (h : ∀ bs st1, bindL C st locals xs = .ok (bs, st1) → 2 ≤ (bs.flatMap (spliceB o)).length) :
    bindNode C st locals (.andOr o (.paren [.andOr o xs] :: ys)) =
      bindNode C st locals (.andOr o (xs ++ ys)) := by
  simp only [bindNode, bindL, bindL_append]
  cases hx : bindL C st locals xs with
  | error e => rfl
  | ok p =>
    simp only [List.getLast?_singleton]
    cases bindL C p.2 locals ys with
    | error e => rfl
    | ok q =>
      simp only
      rw [flatB_assoc o p.1 q.1 (h p.1 p.2 hx)]

/-! ## Reservations keep the table covering -/

/-- The reserved key is new whenever the table covers its ids: it carries the
minted id, which is `len()` (formatting contract of the `_quant_{s}_{id}` keys). -/
def KeysFresh : Prop := ∀ (e : Env) (p : String) (s : Nat), e.Covers → e.has (C.rkey p s e.length) = false

theorem reserve_covers (hk : KeysFresh C) (st : St) (p : String) (x : Name) (h : st.cur.Covers) :
    (reserve C st p x).1.cur.Covers := mint_new_covers h (hk _ _ _ h) _ _ _

theorem reserve_frame (st : St) (p : String) (x : Name) :
    (reserve C st p x).1.depth = st.depth ∧ (reserve C st p x).1.parent = st.parent ∧
    (reserve C st p x).2.id = st.cur.length ∧ (reserve C st p x).2.scope = st.depth := ⟨rfl, rfl, rfl, rfl⟩

mutual
def patFree : NE → Bool
  | .pattern _ | .patComp _ _ => false
  | .var _ => true
  | .quant _ cs | .listComp _ cs | .reduce _ _ cs | .prop _ cs | .func _ _ cs | .paren cs | .andOr _ cs
  | .other _ cs => patFreeL cs
def patFreeL : List NE → Bool
  | [] => true
  | c :: cs => patFree c && patFreeL cs
end

/-- Parent copies happen under a new key (resolve_covers); its hypothesis. -/
def CopyNew (st : St) : Prop := ∀ n, st.cur.find? n = none → st.cur.has n = false

theorem find_none_has (e : Env) (n : Name) (h : e.find? n = none) : e.has n = false := by
  unfold Env.find? at h; unfold Env.has
  induction e with
  | nil => rfl
  | cons p ps ih =>
    simp only [List.lookup] at h
    by_cases hp : n == p.1
    · simp [hp] at h
    · simp only [hp] at h
      simp only [List.any_cons, Bool.or_eq_false_iff]
      have hne : n ≠ p.1 := fun e => hp (by rw [e]; exact beq_self_eq_true _)
      exact ⟨by rw [beq_eq_false_iff_ne]; exact fun e => hne e.symm, ih h⟩

mutual
/-- Pattern-free expressions keep the current table covering. -/
theorem bindNode_covers (hk : KeysFresh C) : ∀ (st : St) (locals : List Env) (e : NE),
    st.cur.Covers → patFree e = true → ∀ b st', bindNode C st locals e = .ok (b, st') → st'.cur.Covers
  | st, locals, .var n, hc, _, b, st', h => by
    simp only [bindNode] at h
    cases hr : resolve st locals n with
    | error e => rw [hr] at h; simp at h
    | ok p =>
      rw [hr] at h; simp at h; obtain ⟨-, rfl⟩ := h
      exact resolve_covers st locals n p.1 p.2 hc (find_none_has _ _) hr
  | st, locals, .quant x cs, hc, hp, b, st', h => by
    match cs, hp, h with
    | c0 :: rest, hp, h =>
      simp only [patFree, patFreeL, Bool.and_eq_true] at hp
      simp only [bindNode] at h
      have h0 := reserve_covers C hk st "_quant_" x hc
      cases h1 : bindNode C (reserve C st "_quant_" x).1 locals c0 with
      | error e => rw [h1] at h; simp at h
      | ok p1 =>
        rw [h1] at h; simp only at h
        have c1 := bindNode_covers hk _ locals c0 h0 hp.1 p1.1 p1.2 h1
        cases h2 : bindL C p1.2 (locals ++ [[(some x, (reserve C st "_quant_" x).2)]]) rest with
        | error e => rw [h2] at h; simp at h
        | ok p2 =>
          rw [h2] at h; simp at h; obtain ⟨-, rfl⟩ := h
          exact bindL_covers hk _ _ rest c1 hp.2 p2.1 p2.2 h2
    | [], _, h => simp [bindNode] at h
  | st, locals, .listComp x cs, hc, hp, b, st', h => by
    match cs, hp, h with
    | c0 :: rest, hp, h =>
      simp only [patFree, patFreeL, Bool.and_eq_true] at hp
      simp only [bindNode] at h
      have h0 := reserve_covers C hk st "_lc_" x hc
      cases h1 : bindNode C (reserve C st "_lc_" x).1 locals c0 with
      | error e => rw [h1] at h; simp at h
      | ok p1 =>
        rw [h1] at h; simp only at h
        have c1 := bindNode_covers hk _ locals c0 h0 hp.1 p1.1 p1.2 h1
        cases h2 : bindL C p1.2 (locals ++ [[(some x, (reserve C st "_lc_" x).2)]]) rest with
        | error e => rw [h2] at h; simp at h
        | ok p2 =>
          rw [h2] at h; simp only at h
          have c2 := bindL_covers hk _ _ rest c1 hp.2 p2.1 p2.2 h2
          split at h
          · split at h
            · simp at h; rw [← h.2]; exact c2
            · simp at h
          · simp at h; rw [← h.2]; exact c2
    | [], _, h => simp [bindNode] at h
  | st, locals, .reduce a i cs, hc, hp, b, st', h => by
    simp only [bindNode] at h
    split at h
    · simp at h
    · simp only [patFree] at hp
      exact bindReduce_covers hk st locals a i cs hc hp b st' h
  | st, locals, .pattern vars, _, hp, _, _, _ => by simp [patFree] at hp
  | st, locals, .patComp vars cs, _, hp, _, _, _ => by simp [patFree] at hp
  | st, locals, .prop k cs, hc, hp, b, st', h => by
    simp only [patFree] at hp
    simp only [bindNode] at h
    cases h1 : bindL C st locals cs with
    | error e => rw [h1] at h; simp at h
    | ok p =>
      rw [h1] at h; simp only at h
      have c1 := bindL_covers hk st locals cs hc hp p.1 p.2 h1
      split at h
      · split at h
        · simp at h
        · simp at h; rw [← h.2]; exact c1
      · simp at h; rw [← h.2]; exact c1
  | st, locals, .func n tys cs, hc, hp, b, st', h => by
    simp only [patFree] at hp
    simp only [bindNode] at h
    split at h
    · simp at h
    cases h1 : bindL C st locals cs with
    | error e => rw [h1] at h; simp at h
    | ok p =>
      rw [h1] at h; simp only at h
      have c1 := bindL_covers hk st locals cs hc hp p.1 p.2 h1
      split at h
      · simp at h
      · simp at h; rw [← h.2]; exact c1
  | st, locals, .paren cs, hc, hp, b, st', h => by
    simp only [patFree] at hp
    simp only [bindNode] at h
    cases h1 : bindL C st locals cs with
    | error e => rw [h1] at h; simp at h
    | ok p =>
      rw [h1] at h; simp only at h
      have c1 := bindL_covers hk st locals cs hc hp p.1 p.2 h1
      split at h
      · simp at h; rw [← h.2]; exact c1
      · simp at h
  | st, locals, .andOr o cs, hc, hp, b, st', h => by
    simp only [patFree] at hp
    simp only [bindNode] at h
    cases h1 : bindL C st locals cs with
    | error e => rw [h1] at h; simp at h
    | ok p =>
      rw [h1] at h; simp at h; obtain ⟨-, rfl⟩ := h
      exact bindL_covers hk st locals cs hc hp p.1 p.2 h1
  | st, locals, .other t cs, hc, hp, b, st', h => by
    simp only [patFree] at hp
    simp only [bindNode] at h
    cases h1 : bindL C st locals cs with
    | error e => rw [h1] at h; simp at h
    | ok p =>
      rw [h1] at h; simp at h; obtain ⟨-, rfl⟩ := h
      exact bindL_covers hk st locals cs hc hp p.1 p.2 h1
theorem bindReduce_covers (hk : KeysFresh C) : ∀ (st : St) (locals : List Env) (a i : Name) (l : List NE),
    st.cur.Covers → patFreeL l = true → ∀ b st', bindReduce C st locals a i l = .ok (b, st') → st'.cur.Covers
  | st, locals, a, i, [i0, l0, b0], hc, hp, b, st', h => by
    simp only [patFreeL, Bool.and_eq_true] at hp
    have h0 := reserve_covers C hk _ "_reduce_iter_" i (reserve_covers C hk st "_reduce_acc_" a hc)
    simp only [bindReduce] at h
    cases h1 : bindNode C (reserve C (reserve C st "_reduce_acc_" a).1 "_reduce_iter_" i).1 locals i0 with
    | error e => rw [h1] at h; simp at h
    | ok p1 =>
      rw [h1] at h; simp only at h
      have c1 := bindNode_covers hk _ locals i0 h0 hp.1 p1.1 p1.2 h1
      cases h2 : bindNode C p1.2 locals l0 with
      | error e => rw [h2] at h; simp at h
      | ok p2 =>
        rw [h2] at h; simp only at h
        have c2 := bindNode_covers hk _ locals l0 c1 hp.2.1 p2.1 p2.2 h2
        cases h3 : bindNode C p2.2 (locals ++ [[(some a, (reserve C st "_reduce_acc_" a).2),
            (some i, (reserve C (reserve C st "_reduce_acc_" a).1 "_reduce_iter_" i).2)]]) b0 with
        | error e => rw [h3] at h; simp at h
        | ok p3 =>
          rw [h3] at h; simp at h; obtain ⟨-, rfl⟩ := h
          exact bindNode_covers hk _ _ b0 c2 hp.2.2.1 p3.1 p3.2 h3
  | st, locals, a, i, [], _, _, b, st', h => by simp [bindReduce] at h
  | st, locals, a, i, [_], _, _, b, st', h => by simp [bindReduce] at h
  | st, locals, a, i, [_, _], _, _, b, st', h => by simp [bindReduce] at h
  | st, locals, a, i, _ :: _ :: _ :: _ :: _, _, _, b, st', h => by simp [bindReduce] at h
theorem bindL_covers (hk : KeysFresh C) : ∀ (st : St) (locals : List Env) (l : List NE),
    st.cur.Covers → patFreeL l = true → ∀ bs st', bindL C st locals l = .ok (bs, st') → st'.cur.Covers
  | st, _, [], hc, _, bs, st', h => by simp [bindL] at h; rw [← h.2]; exact hc
  | st, locals, c :: cs, hc, hp, bs, st', h => by
    simp only [patFreeL, Bool.and_eq_true] at hp
    simp only [bindL] at h
    cases h1 : bindNode C st locals c with
    | error e => rw [h1] at h; simp at h
    | ok p1 =>
      rw [h1] at h; simp only at h
      have c1 := bindNode_covers hk st locals c hc hp.1 p1.1 p1.2 h1
      cases h2 : bindL C p1.2 locals cs with
      | error e => rw [h2] at h; simp at h
      | ok p2 =>
        rw [h2] at h; simp at h; obtain ⟨-, rfl⟩ := h
        exact bindL_covers hk _ locals cs c1 hp.2 p2.1 p2.2 h2
end

/-! ## `bind_set_items` -/

inductive SetItem
  | attr (target value : NE)
  | label (n : Name)

inductive BSetItem
  | attr (target value : BE)
  | label (v : Var)

def bindSetItems (st : St) : List SetItem → Except String (List BSetItem × St)
  | [] => .ok ([], st)
  | .attr t v :: is => match bindExpr C st t with
    | .error e => .error e
    | .ok (bt, st1) => match bindExpr C st1 v with
      | .error e => .error e
      | .ok (bv, st2) => match bindSetItems st2 is with
        | .error e => .error e
        | .ok (bs, st3) => .ok (.attr bt bv :: bs, st3)
  | .label n :: is => match resolve st [] n with
    | .error e => .error e
    | .ok (st1, v) => match bindSetItems st1 is with
      | .error e => .error e
      | .ok (bs, st2) => .ok (.label v :: bs, st2)

theorem bindSetItems_length (st : St) (is : List SetItem) (bs : List BSetItem) (st' : St)
    (h : bindSetItems C st is = .ok (bs, st')) : bs.length = is.length := by
  induction is generalizing st bs st' with
  | nil => simp [bindSetItems] at h; obtain ⟨rfl, -⟩ := h; rfl
  | cons i is ih =>
    cases i with
    | attr t v =>
      simp only [bindSetItems] at h
      split at h
      · simp at h
      · split at h
        · simp at h
        · split at h
          · simp at h
          · rename_i bs' st3 h3
            simp at h; obtain ⟨rfl, -⟩ := h; simp [ih _ _ _ h3]
    | label n =>
      simp only [bindSetItems] at h
      split at h
      · simp at h
      · split at h
        · simp at h
        · rename_i bs' st2 h2
          simp at h; obtain ⟨rfl, -⟩ := h; simp [ih _ _ _ h2]

/-- `SET n:L` on a name the clause cannot see fails exactly as `resolve_name` does. -/
theorem bindSetItems_label_unbound (st : St) (n : Name) (is : List SetItem) (h : ¬ visible st [] n) :
    ∃ e, bindSetItems C st (.label n :: is) = .error e := by
  have : ∀ r, resolve st [] n ≠ .ok r := fun r hr => h ((resolve_ok_iff st [] n).1 ⟨r, hr⟩)
  cases hr : resolve st [] n with
  | error e => exact ⟨e, by simp [bindSetItems, hr]⟩
  | ok r => exact absurd hr (this r)

/-! ## `check_unbound_self_referential` -/

/- DFS: some `Property` whose first child is the entity's own alias. -/
mutual
def selfRef (a : Name) : NE → Bool
  | .prop _ (.var n :: _) => n == a
  | .prop _ cs => selfRefL a cs
  | .var _ | .pattern _ => false
  | .quant _ cs | .listComp _ cs | .reduce _ _ cs | .patComp _ cs | .func _ _ cs | .paren cs | .andOr _ cs
  | .other _ cs => selfRefL a cs
def selfRefL (a : Name) : List NE → Bool
  | [] => false
  | c :: cs => selfRef a c || selfRefL a cs
end

/-- `check_unbound_self_referential` (binder.rs:1772-1789): an unnamed entity has no name to
reference itself by (#3528: no user-written name, `_anon…` included, is exempt). -/
def Alias.selfRef : Alias → NE → Bool
  | .named n, e => FalkorBinder.selfRef n e
  | .anon _, _ => false

def checkSelfRef (a : Alias) (attrs : NE) : Except String Unit :=
  match a with
  | .anon _ => .ok ()
  | .named n => if selfRef n attrs then .error s!"'{n}' not defined; undefined attribute" else .ok ()

theorem checkSelfRef_iff (a : Alias) (attrs : NE) :
    checkSelfRef a attrs = .ok () ↔ (a.isAnon = true ∨ a.selfRef attrs = false) := by
  cases a with
  | anon k => simp [checkSelfRef, Alias.isAnon]
  | named n => by_cases h2 : selfRef n attrs <;> simp [checkSelfRef, Alias.isAnon, Alias.selfRef, h2]

/-- `CREATE (a {v: a.v})` is rejected; `CREATE (a {v: [a.v][0]})` too (the walk is deep).
Since #3528 this holds for every name, `_anon_0` included. -/
theorem selfRef_direct (a k : Name) :
    checkSelfRef (.named a) (.other 0 [.prop k [.var a]]) = .error s!"'{a}' not defined; undefined attribute" := by
  simp [checkSelfRef, selfRef, selfRefL]


/-! ## #1334: a pattern predicate declares nothing (fixed by #3528)

Before #3528 the `ExprIR::Pattern` arm of `bind_expr_node` bound the predicate with
`bind_graph(&pattern, false)` alone, which `define_name_in_scope`s every alias (`allow_reuse`),
minting a fresh variable for a name that is not in scope: `WHERE () -- (n1)` with no `n1` bound
ran as "some edge exists anywhere" (the former counterexample `bug1334_pattern_predicate_declares_var`).
Now (binder.rs:2174-2186) every named alias is first resolved with `resolve_bound_name`, so an
unknown one is "not defined" (`resolveNamed`). -/

/-- **A pattern predicate declares nothing** (openCypher; C rejects it, issue #1334): whenever
`bind_expr_node` accepts a pattern predicate, every named variable in it was already visible where
the predicate stands. -/
def PatternPredDeclaresNothing (st : St) (locals : List Env) (vars : List Alias) : Prop :=
  ∀ r, bindNode C st locals (.pattern vars) = .ok r →
    ∀ n, .named n ∈ vars → visible st locals n

/-- A parent copy only adds a name the parent already showed: whatever is visible after a
`resolve` step was visible before it. -/
theorem resolve_visible_back (st st1 : St) (locals : List Env) (n m : Name) (v : Var)
    (h : resolve st locals n = .ok (st1, v)) (hv : visible st1 locals m) : visible st locals m := by
  unfold resolve at h
  split at h
  · cases h; exact hv
  split at h
  · cases h; exact hv
  split at h
  · rename_i hup
    split at h
    · cases h; exact hv
    split at h
    · cases h; exact hv
    split at h
    · rename_i pv hpv
      cases h
      unfold visible at hv ⊢
      simp only at hv
      rcases hv with h1 | h2 | ⟨h3, h4 | h5 | h6⟩
      · exact Or.inl h1
      · rw [Env.find?_insert] at h2
        by_cases hm : m = n
        · subst hm; exact Or.inr (Or.inr ⟨hup, Or.inr (Or.inr (by simp [hpv]))⟩)
        · simp [hm] at h2; exact Or.inr (Or.inl h2)
      · exact Or.inr (Or.inr ⟨h3, Or.inl h4⟩)
      · by_cases hm : m = n
        · subst hm; exact Or.inr (Or.inr ⟨hup, Or.inr (Or.inr (by simp [hpv]))⟩)
        · have : (m == n) = false := by simp [hm]
          simp [List.lookup, this] at h5
          exact Or.inr (Or.inr ⟨h3, Or.inr (Or.inl (by simp [h5]))⟩)
      · exact Or.inr (Or.inr ⟨h3, Or.inr (Or.inr h6)⟩)
    · cases h
  · cases h

theorem resolveNamed_visible (locals : List Env) : ∀ (vars : List Alias) (st st0 : St),
    resolveNamed st locals vars = .ok st0 → ∀ n, .named n ∈ vars → visible st locals n
  | [], _, _, _, _, hn => by simp at hn
  | .anon _ :: as, st, st0, h, n, hn => by
    simp only [resolveNamed] at h
    simp at hn
    exact resolveNamed_visible locals as st st0 h n hn
  | .named m :: as, st, st0, h, n, hn => by
    simp only [resolveNamed] at h
    cases hr : resolve st locals m with
    | error e => rw [hr] at h; cases h
    | ok p =>
      rw [hr] at h; simp only at h
      simp at hn
      rcases hn with rfl | hn
      · exact (resolve_ok_iff st locals n).1 ⟨p, hr⟩
      · exact resolve_visible_back st p.1 locals m n p.2 hr
          (resolveNamed_visible locals as p.1 st0 h n hn)

/-- **#1334 fixed** (#3528): every pattern predicate the binder accepts declares nothing. -/
theorem patternPred_declares_nothing (st : St) (locals : List Env) (vars : List Alias) :
    PatternPredDeclaresNothing C st locals vars := by
  intro r h n hn
  simp only [bindNode] at h
  cases h0 : resolveNamed st locals vars with
  | error e => rw [h0] at h; cases h
  | ok st0 => exact resolveNamed_visible locals vars st st0 h0 n hn

/-- The binder state while `MATCH (n0) WITH * WHERE …` binds its WHERE: the projection scope
`{n0}` (scope 1), the parent `{n0}`, `use_parent_scope` set; `WITH *` adds nothing to
`parent_to_child_scope` (binder.rs:1102-1150). -/
def st1334 : St :=
  { cur := [("n0", ⟨"n0", 0, 1, .node⟩)], parent := [("n0", ⟨"n0", 0, 0, .node⟩)],
    depth := 1, useParent := true, p2c := [], cfp := [] }

/-- **#1334**, the issue's query: `MATCH (n0) WITH * WHERE () -- (n1) RETURN *` is now rejected,
"'n1' not defined", as C does (before #3528 Rust minted `n1` and returned every row). -/
theorem fixed1334_pattern_predicate_unbound :
    bindNode C st1334 [] (.pattern [.anon 0, .named "n1"]) = .error "'n1' not defined" ∧
    ¬ visible st1334 [] "n1" := by
  refine ⟨by simp (config := { decide := true }) [bindNode, st1334, resolveNamed, resolve, lookupLocals,
    bindNames, defineAlias, defineName, injectLocals, cleanup, Env.find?, Env.has, Env.insert,
    Env.retain, Env.pushUnnamed, Env.names, freshVar, List.lookup], ?_⟩
  unfold visible; decide

/-- …while a predicate over bound names still binds, its unnamed entity minted unnamed
(id 1, kept by the cleanup) and recorded in `anon_vars`. -/
theorem pattern_predicate_bound :
    bindNode C st1334 [] (.pattern [.named "n0", .anon 0]) =
      .ok (.pattern [⟨"n0", 0, 1, .node⟩, ⟨"", 1, 1, .any⟩],
           { st1334 with cur := [("n0", ⟨"n0", 0, 1, .node⟩), (none, ⟨"", 1, 1, .any⟩)],
                         anon := [(0, ⟨"", 1, 1, .any⟩)] }) := by
  simp (config := { decide := true }) [bindNode, st1334, resolveNamed, resolve, lookupLocals,
    bindNames, defineAlias, defineName, injectLocals, cleanup, Env.find?, Env.has, Env.insert,
    Env.retain, Env.pushUnnamed, Env.names, freshVar, List.lookup]

/-- #3528: a user-written `_anon_0` is an ordinary name — unbound, it is "not defined" too
(before, any name starting `_anon` was treated as the parser's own and minted). -/
theorem anon_spelled_name_unbound :
    bindNode C st1334 [] (.pattern [.named "_anon_0"]) = .error "'_anon_0' not defined" := by
  simp (config := { decide := true }) [bindNode, st1334, resolveNamed, resolve, lookupLocals,
    bindNames, defineAlias, defineName, injectLocals, cleanup, Env.find?, Env.has, Env.insert,
    Env.retain, Env.pushUnnamed, Env.names, freshVar, List.lookup]

end FalkorBinder
