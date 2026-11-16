import FalkorBinder.Resolve
/-
# Binding expressions: `bind_expr_node` and its wrappers

| here | there |
| --- | --- |
| `NE` / `BE` | raw `ExprIR<Arc<String>>` / bound `ExprIR<Variable>` trees (the variants the binder treats specially) |
| `reserve` | `fresh_var` + `Scope::reserve` (nameless; before #3016 keyed `_quant_…`/`_lc_…`/`_reduce_…`) for the quantifier / comprehension / reduce variables and the ORDER BY aggregate placeholder |
| `liftLocals`, `hideExcept`, `restoreShadowed`, `cleanup` | `lift_locals` binder.rs:373, `hide_names_except` :417, `restore_shadowed` :397 (pattern / pattern-comprehension arms) |
| `bindNode`, `bindL` | `bind_expr_node` binder.rs:1893-2315 |
| `NE.paren`, `bindNode` on it | the `Paren` early return binder.rs:1955-1965 (#3335): `(x)` binds to `x`'s tree |
| `NE.andOr`, `BE.andOr`, `flatB` | `And`/`Or` and the `ExprIR::and_of`/`or_of` return binder.rs:2137-2144, ast.rs:342-380 (#3335) |
| `bindExpr`, `bindExprLocals` | `bind_expr` binder.rs:1725-1730, `bind_expr_with_locals` binder.rs:1732-1739 |
| `bindSetItems` | `bind_set_items` binder.rs:1700-1723 |
| `selfRef` | `check_unbound_self_referential` binder.rs:1682-1698 |

`bind_graph` (for pattern predicates / comprehensions) defines each pattern
name in the current table (`defineName … true`, Pattern.lean has the finer
node/relationship order). The constant fold and call rewrites are Fold.lean;
labels are Scopes.lean.

Theorems: `reserve_covers` / `bindNode_covers` (every reservation keeps the
table covering; since #3016 the pattern cleanup hides instead of dropping —
`comprehension_cleanup_covers`), `bind_aggPh` (the placeholder takes a nameless
fresh slot), `lift_restore_shadowed` (a comprehension variable no longer
shadows the scope's own after a pattern),
`bind_iterable_outer` (a comprehension's list is bound before its variable),
`loop_var_shadows`, `reduce_same_name`, `prop_on_path`, `exists_pattern`,
`func_arg_mismatch`, `bindExpr_eq`, `bindSetItems_*`, `selfRef_iff`.

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
  /-- `ExprIR::AggOrderPlaceholder` (#3016): the parser's ORDER BY aggregate placeholder -/
  | aggPh
  | quant (x : Name) (cs : List NE)
  | listComp (x : Name) (cs : List NE)
  | reduce (acc it : Name) (cs : List NE)
  | pattern (vars : List Name)
  | patComp (vars : List Name) (cs : List NE)
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
  /-- `expr_may_return_boolean` on the bound WHERE of a list comprehension -/
  mayBool : BE → Bool
  /-- `(scope, id)` of variable-length relationship variables -/
  varlen : List (Nat × Nat)
  /-- `Type::is_compatible_with` -/
  compat : Ty → Ty → Bool

variable (C : Ctx)

/-- Mint a loop variable and reserve its slot, nameless (`fresh_var` +
`Scope::reserve`, binder.rs:1917-1920, 1950-1953, 1991-1997). -/
def reserve (st : St) (x : Name) : St × Var :=
  let v := freshVar st.cur x .any st.depth
  ({ st with cur := st.cur.reserve v }, v)

def isAnon (n : Name) : Bool := n.startsWith "_anon"

/-- `bind_graph` on a pattern's names: each defined (reused if present). -/
def bindNames (st : St) : List Name → Except String (St × List Var)
  | [] => .ok (st, [])
  | n :: ns => match defineName st.cur st.depth n .any true with
    | .error e => .error e
    | .ok (e, v) => match bindNames { st with cur := e } ns with
      | .error e => .error e
      | .ok (st', vs) => .ok (st', v :: vs)

/-- `Binder::lift_locals` (binder.rs:373-391): each comprehension variable is
put under its name (`Scope::lift`); a scope variable of that name is reserved
and recorded as shadowed. -/
def liftLocals (e : Env) (locals : List Env) : Env × List (Name × Option Var) :=
  locals.foldl (fun acc l => l.foldl (fun (acc : Env × List (Name × Option Var)) kv =>
    match kv.1 with
    | .name n =>
      let prev := acc.1.find? n
      let e1 := match prev with
        | some p => acc.1.reserve p
        | none => acc.1
      (e1.lift n kv.2, acc.2 ++ [(n, prev)])
    | .rsv => acc) acc) (e, [])

/-- `Binder::hide_names_except` (binder.rs:417-428): hide every name not in
`keep`, except the generated `_anon*` entries. -/
def hideExcept (keep : List Name) (e : Env) : Env :=
  e.hideAll (e.names.filter fun k => !keep.contains k && !isAnon k)

/-- `Binder::restore_shadowed` (binder.rs:397-411), last lifted first. -/
def restoreShadowed (e : Env) (sh : List (Name × Option Var)) : Env :=
  sh.reverse.foldl (fun e (np : Name × Option Var) =>
    let e1 := e.hide np.1
    match np.2 with
    | some p => (e1.unreserve p.id).insert np.1 p
    | none => e1) e

/-- The cleanup after a pattern (binder.rs:2070-2071, 2245-2246). -/
def cleanup (outer : List Name) (sh : List (Name × Option Var)) (e : Env) : Env :=
  restoreShadowed (hideExcept outer e) sh

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
  | .aggPh =>
    let r := reserve st "__agg_order_by_placeholder__"
    .ok (.var r.2, r.1)
  | .var n => match resolve st locals n with
    | .ok (st', v) => .ok (.var v, st')
    | .error e => .error e
  | .quant x cs => match cs with
    | c0 :: rest =>
      let r := reserve st x
      match bindNode r.1 locals c0 with
      | .error e => .error e
      | .ok (b0, st1) => match bindL st1 (locals ++ [[(.name x, r.2)]]) rest with
        | .error e => .error e
        | .ok (bs, st2) => .ok (.quant r.2 (b0 :: bs), st2)
    | [] => .error "unreachable: quantifier without list"
  | .listComp x cs => match cs with
    | c0 :: rest =>
      let r := reserve st x
      match bindNode r.1 locals c0 with
      | .error e => .error e
      | .ok (b0, st1) => match bindL st1 (locals ++ [[(.name x, r.2)]]) rest with
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
    let outer := st.cur.names
    let lifted := liftLocals st.cur locals
    match bindNames { st with cur := lifted.1 } vars with
    | .error e => .error e
    | .ok (st1, vs) => .ok (.pattern vs, { st1 with cur := cleanup outer lifted.2 st1.cur })
  | .patComp vars cs =>
    let outer := st.cur.names
    let lifted := liftLocals st.cur locals
    match bindNames { st with cur := lifted.1 } vars with
    | .error e => .error e
    | .ok (st1, vs) => match bindL st1 locals cs with
      | .error e => .error e
      | .ok (bs, st2) => .ok (.patComp vs bs, { st2 with cur := cleanup outer lifted.2 st2.cur })
  | .prop k cs => match bindL st locals cs with
    | .error e => .error e
    | .ok (bs, st1) =>
      match firstVar bs with
      | some v => if v.ty = .path ∧ (v.scope, v.id) ∉ C.varlen then
                    .error "Type mismatch: expected Map, Node, Edge, Datetime, Date, Time, Duration, Null, or Point but was Path"
                  else .ok (.prop k bs, st1)
      | none => .ok (.prop k bs, st1)
  | .func name tys cs => match bindL st locals cs with
    | .error e => .error e
    | .ok (bs, st1) =>
      if name = "exists" ∧ bs.any (fun b => match b with | .pattern _ => true | _ => false) then
        .error "Invalid input: traversal patterns are not allowed as arguments to exists()"
      else if name ≠ "hasLabels" ∧ ((tys.zip bs).any fun tb => match tb.2 with
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
    let ra := reserve st acc
    let ri := reserve ra.1 it
    match bindNode ri.1 locals i0 with
    | .error e => .error e
    | .ok (bi, st1) => match bindNode st1 locals l0 with
      | .error e => .error e
      | .ok (bl, st2) => match bindNode st2 (locals ++ [[(.name acc, ra.2), (.name it, ri.2)]]) b0 with
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
    lookupLocals (locals ++ [[(.name x, v)]]) x = some v := by
  induction locals with
  | nil => simp [lookupLocals, Env.find?, List.lookup]
  | cons l ls ih => simp [lookupLocals, ih]

/-- The loop variable shadows everything inside the comprehension body. -/
theorem loop_var_shadows (st : St) (locals : List Env) (x : Name) (v : Var) :
    resolve st (locals ++ [[(.name x, v)]]) x = .ok (st, v) := by
  unfold resolve; rw [lookupLocals_last]

/-- The list of `[x IN list | …]` is bound with the *outer* locals (before
`x` exists), in the state after the slot reservation. -/
theorem bind_iterable_outer (st : St) (locals : List Env) (x : Name) (c0 : NE) (rest : List NE)
    (b : BE) (st' : St) (h : bindNode C st locals (.listComp x (c0 :: rest)) = .ok (b, st')) :
    ∃ b0 st1, bindNode C (reserve st x).1 locals c0 = .ok (b0, st1) := by
  simp only [bindNode] at h
  cases h0 : bindNode C (reserve st x).1 locals c0 with
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

theorem exists_pattern (st : St) (locals : List Env) (vars : List Name) (bp : BE) (st1 : St)
    (h : bindNode C st locals (.pattern vars) = .ok (bp, st1)) (hp : ∃ vs, bp = .pattern vs) :
    ∃ e, bindNode C st locals (.func "exists" [] [.pattern vars]) = .error e := by
  obtain ⟨vs, rfl⟩ := hp
  simp [bindNode, bindL] at h ⊢
  rw [h]; simp

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

/-- A nameless reservation of the minted variable keeps the table covering
(no key to collide with, since #3016). -/
theorem reserve_covers (st : St) (x : Name) (h : st.cur.Covers) :
    (reserve st x).1.cur.Covers := reserve_fresh_covers h _ _ _

theorem reserve_frame (st : St) (x : Name) :
    (reserve st x).1.depth = st.depth ∧ (reserve st x).1.parent = st.parent ∧
    (reserve st x).2.id = st.cur.length ∧ (reserve st x).2.scope = st.depth := ⟨rfl, rfl, rfl, rfl⟩

mutual
def patFree : NE → Bool
  | .pattern _ | .patComp _ _ => false
  | .var _ | .aggPh => true
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
theorem bindNode_covers : ∀ (st : St) (locals : List Env) (e : NE),
    st.cur.Covers → patFree e = true → ∀ b st', bindNode C st locals e = .ok (b, st') → st'.cur.Covers
  | st, _, .aggPh, hc, _, b, st', h => by
    simp only [bindNode, Except.ok.injEq, Prod.mk.injEq] at h
    obtain ⟨-, rfl⟩ := h
    exact reserve_covers st _ hc
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
      have h0 := reserve_covers st x hc
      cases h1 : bindNode C (reserve st x).1 locals c0 with
      | error e => rw [h1] at h; simp at h
      | ok p1 =>
        rw [h1] at h; simp only at h
        have c1 := bindNode_covers _ locals c0 h0 hp.1 p1.1 p1.2 h1
        cases h2 : bindL C p1.2 (locals ++ [[(.name x, (reserve st x).2)]]) rest with
        | error e => rw [h2] at h; simp at h
        | ok p2 =>
          rw [h2] at h; simp at h; obtain ⟨-, rfl⟩ := h
          exact bindL_covers _ _ rest c1 hp.2 p2.1 p2.2 h2
    | [], _, h => simp [bindNode] at h
  | st, locals, .listComp x cs, hc, hp, b, st', h => by
    match cs, hp, h with
    | c0 :: rest, hp, h =>
      simp only [patFree, patFreeL, Bool.and_eq_true] at hp
      simp only [bindNode] at h
      have h0 := reserve_covers st x hc
      cases h1 : bindNode C (reserve st x).1 locals c0 with
      | error e => rw [h1] at h; simp at h
      | ok p1 =>
        rw [h1] at h; simp only at h
        have c1 := bindNode_covers _ locals c0 h0 hp.1 p1.1 p1.2 h1
        cases h2 : bindL C p1.2 (locals ++ [[(.name x, (reserve st x).2)]]) rest with
        | error e => rw [h2] at h; simp at h
        | ok p2 =>
          rw [h2] at h; simp only at h
          have c2 := bindL_covers _ _ rest c1 hp.2 p2.1 p2.2 h2
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
      exact bindReduce_covers st locals a i cs hc hp b st' h
  | st, locals, .pattern vars, _, hp, _, _, _ => by simp [patFree] at hp
  | st, locals, .patComp vars cs, _, hp, _, _, _ => by simp [patFree] at hp
  | st, locals, .prop k cs, hc, hp, b, st', h => by
    simp only [patFree] at hp
    simp only [bindNode] at h
    cases h1 : bindL C st locals cs with
    | error e => rw [h1] at h; simp at h
    | ok p =>
      rw [h1] at h; simp only at h
      have c1 := bindL_covers st locals cs hc hp p.1 p.2 h1
      split at h
      · split at h
        · simp at h
        · simp at h; rw [← h.2]; exact c1
      · simp at h; rw [← h.2]; exact c1
  | st, locals, .func n tys cs, hc, hp, b, st', h => by
    simp only [patFree] at hp
    simp only [bindNode] at h
    cases h1 : bindL C st locals cs with
    | error e => rw [h1] at h; simp at h
    | ok p =>
      rw [h1] at h; simp only at h
      have c1 := bindL_covers st locals cs hc hp p.1 p.2 h1
      split at h
      · simp at h
      · split at h
        · simp at h
        · simp at h; rw [← h.2]; exact c1
  | st, locals, .paren cs, hc, hp, b, st', h => by
    simp only [patFree] at hp
    simp only [bindNode] at h
    cases h1 : bindL C st locals cs with
    | error e => rw [h1] at h; simp at h
    | ok p =>
      rw [h1] at h; simp only at h
      have c1 := bindL_covers st locals cs hc hp p.1 p.2 h1
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
      exact bindL_covers st locals cs hc hp p.1 p.2 h1
  | st, locals, .other t cs, hc, hp, b, st', h => by
    simp only [patFree] at hp
    simp only [bindNode] at h
    cases h1 : bindL C st locals cs with
    | error e => rw [h1] at h; simp at h
    | ok p =>
      rw [h1] at h; simp at h; obtain ⟨-, rfl⟩ := h
      exact bindL_covers st locals cs hc hp p.1 p.2 h1
theorem bindReduce_covers : ∀ (st : St) (locals : List Env) (a i : Name) (l : List NE),
    st.cur.Covers → patFreeL l = true → ∀ b st', bindReduce C st locals a i l = .ok (b, st') → st'.cur.Covers
  | st, locals, a, i, [i0, l0, b0], hc, hp, b, st', h => by
    simp only [patFreeL, Bool.and_eq_true] at hp
    have h0 := reserve_covers _ i (reserve_covers st a hc)
    simp only [bindReduce] at h
    cases h1 : bindNode C (reserve (reserve st a).1 i).1 locals i0 with
    | error e => rw [h1] at h; simp at h
    | ok p1 =>
      rw [h1] at h; simp only at h
      have c1 := bindNode_covers _ locals i0 h0 hp.1 p1.1 p1.2 h1
      cases h2 : bindNode C p1.2 locals l0 with
      | error e => rw [h2] at h; simp at h
      | ok p2 =>
        rw [h2] at h; simp only at h
        have c2 := bindNode_covers _ locals l0 c1 hp.2.1 p2.1 p2.2 h2
        cases h3 : bindNode C p2.2 (locals ++ [[(.name a, (reserve st a).2),
            (.name i, (reserve (reserve st a).1 i).2)]]) b0 with
        | error e => rw [h3] at h; simp at h
        | ok p3 =>
          rw [h3] at h; simp at h; obtain ⟨-, rfl⟩ := h
          exact bindNode_covers _ _ b0 c2 hp.2.2.1 p3.1 p3.2 h3
  | st, locals, a, i, [], _, _, b, st', h => by simp [bindReduce] at h
  | st, locals, a, i, [_], _, _, b, st', h => by simp [bindReduce] at h
  | st, locals, a, i, [_, _], _, _, b, st', h => by simp [bindReduce] at h
  | st, locals, a, i, _ :: _ :: _ :: _ :: _, _, _, b, st', h => by simp [bindReduce] at h
theorem bindL_covers : ∀ (st : St) (locals : List Env) (l : List NE),
    st.cur.Covers → patFreeL l = true → ∀ bs st', bindL C st locals l = .ok (bs, st') → st'.cur.Covers
  | st, _, [], hc, _, bs, st', h => by simp [bindL] at h; rw [← h.2]; exact hc
  | st, locals, c :: cs, hc, hp, bs, st', h => by
    simp only [patFreeL, Bool.and_eq_true] at hp
    simp only [bindL] at h
    cases h1 : bindNode C st locals c with
    | error e => rw [h1] at h; simp at h
    | ok p1 =>
      rw [h1] at h; simp only at h
      have c1 := bindNode_covers st locals c hc hp.1 p1.1 p1.2 h1
      cases h2 : bindL C p1.2 locals cs with
      | error e => rw [h2] at h; simp at h
      | ok p2 =>
        rw [h2] at h; simp at h; obtain ⟨-, rfl⟩ := h
        exact bindL_covers _ locals cs c1 hp.2 p2.1 p2.2 h2
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

/-! ## Comprehension variables inside a pattern (#3016, binder.rs:373-428) -/

/-- Scope `{x ↦ 0}` with the comprehension variable `x` (id 1) reserved, as
`[x IN … | (n)-->(m)]` leaves it while binding the pattern. -/
def shScope : Env := [(.name "x", ⟨"x", 0, 0, .any⟩), (.rsv, ⟨"x", 1, 0, .any⟩)]
def shLocals : List Env := [[(.name "x", ⟨"x", 1, 0, .any⟩)]]

/-- `lift_locals` puts the comprehension's `x` under its name and keeps the
scope's `x` reserved; `hide_names_except` + `restore_shadowed` give the scope
its own `x` back and keep both slots. (Before #3016 the comprehension's `x`
stayed under the name after the pattern, shadowing the scope variable.) -/
theorem lift_restore_shadowed :
    (liftLocals shScope shLocals).1.find? "x" = some ⟨"x", 1, 0, .any⟩ ∧
    (cleanup ["x"] (liftLocals shScope shLocals).2 (liftLocals shScope shLocals).1).find? "x" =
      some ⟨"x", 0, 0, .any⟩ ∧
    (cleanup ["x"] (liftLocals shScope shLocals).2 (liftLocals shScope shLocals).1).ids = [1, 0] := by
  decide

/-! ## The ORDER BY aggregate placeholder (#3016, binder.rs:1904-1912) -/

/-- `ExprIR::AggOrderPlaceholder` binds to a fresh variable at `len()` in a
nameless slot of its own: the table stays covering, gains no name (so no user
alias, `RETURN *` or CALL {} return can reach it), and grows by one. -/
theorem bind_aggPh (st : St) (locals : List Env) (h : st.cur.Covers) :
    bindNode C st locals .aggPh =
      .ok (.var (freshVar st.cur "__agg_order_by_placeholder__" .any st.depth),
           { st with cur := st.cur ++ [(.rsv, freshVar st.cur "__agg_order_by_placeholder__" .any st.depth)] }) ∧
    (st.cur ++ [(Key.rsv, freshVar st.cur "__agg_order_by_placeholder__" .any st.depth)]).names = st.cur.names := by
  refine ⟨?_, by rw [Env.names_append]; simp [Env.names]⟩
  simp only [bindNode, reserve, reserve_fresh_eq h]

/-! ## `check_unbound_self_referential` -/

/- DFS: some `Property` whose first child is the entity's own alias. -/
mutual
def selfRef (a : Name) : NE → Bool
  | .prop _ (.var n :: _) => n == a
  | .prop _ cs => selfRefL a cs
  | .var _ | .aggPh | .pattern _ => false
  | .quant _ cs | .listComp _ cs | .reduce _ _ cs | .patComp _ cs | .func _ _ cs | .paren cs | .andOr _ cs
  | .other _ cs => selfRefL a cs
def selfRefL (a : Name) : List NE → Bool
  | [] => false
  | c :: cs => selfRef a c || selfRefL a cs
end

def checkSelfRef (a : Name) (attrs : NE) : Except String Unit :=
  if isAnon a then .ok ()
  else if selfRef a attrs then .error s!"'{a}' not defined; undefined attribute" else .ok ()

theorem checkSelfRef_iff (a : Name) (attrs : NE) :
    checkSelfRef a attrs = .ok () ↔ (isAnon a = true ∨ selfRef a attrs = false) := by
  unfold checkSelfRef
  by_cases h1 : isAnon a <;> by_cases h2 : selfRef a attrs <;> simp [h1, h2]

/-- `CREATE (a {v: a.v})` is rejected; `CREATE (a {v: [a.v][0]})` too (the walk is deep). -/
theorem selfRef_direct (a k : Name) (h : isAnon a = false) :
    checkSelfRef a (.other 0 [.prop k [.var a]]) = .error s!"'{a}' not defined; undefined attribute" := by
  simp [checkSelfRef, h, selfRef, selfRefL]


/-! ## #1334: a pattern predicate may declare a new variable (gap closed 2026-10-20)

The `ExprIR::Pattern` arm of `bind_expr_node` (binder.rs:2082-2113) binds the predicate with
`bind_graph(&pattern, false)`, which `define_name_in_scope`s every alias (`allow_reuse`), minting
a fresh variable for a name that is not in scope; the cleanup then only hides it again. Nothing
checks that a *named* alias already existed, so `WHERE () -- (n1)` with no `n1` in scope binds
and runs as "some edge exists anywhere". -/

/-- **A pattern predicate declares nothing** (openCypher; C rejects it, issue #1334; the
exact C error message is not recorded here): whenever `bind_expr_node` accepts a pattern predicate, every named
(non-`_anon`) variable in it was already visible where the predicate stands. A fix proves this
by rejecting the pattern otherwise. -/
def PatternPredDeclaresNothing (st : St) (locals : List Env) (vars : List Name) : Prop :=
  ∀ r, bindNode C st locals (.pattern vars) = .ok r →
    ∀ n ∈ vars, isAnon n = false → visible st locals n

/-- The binder state while `MATCH (n0) WITH * WHERE …` binds its WHERE: the projection scope
`{n0}` (scope 1), the parent `{n0}`, `use_parent_scope` set; `WITH *` adds nothing to
`parent_to_child_scope` (binder.rs:1083-1110). -/
def st1334 : St :=
  { cur := [(.name "n0", ⟨"n0", 0, 1, .node⟩)], parent := [(.name "n0", ⟨"n0", 0, 0, .node⟩)],
    depth := 1, useParent := true, p2c := [], cfp := [] }

/-- **#1334**: `MATCH (n0) WITH * WHERE () -- (n1) RETURN *` binds — `n1` is minted fresh (id 2,
after the anonymous node's 1) and hidden again (since #3016 its slot stays reserved) — although `n1` is visible nowhere. Live: Rust main
returns 2 rows on a 2-node graph with an edge; C errors. -/
theorem bug1334_pattern_predicate_declares_var :
    bindNode C st1334 [] (.pattern ["_anon_0", "n1"]) =
      .ok (.pattern [⟨"_anon_0", 1, 1, .any⟩, ⟨"n1", 2, 1, .any⟩],
           { st1334 with cur := [(.name "n0", ⟨"n0", 0, 1, .node⟩), (.name "_anon_0", ⟨"_anon_0", 1, 1, .any⟩),
             (.rsv, ⟨"n1", 2, 1, .any⟩)] }) ∧
    ¬ visible st1334 [] "n1" ∧
    ¬ PatternPredDeclaresNothing C st1334 [] ["_anon_0", "n1"] := by
  have hb : bindNode C st1334 [] (.pattern ["_anon_0", "n1"]) =
      .ok (.pattern [⟨"_anon_0", 1, 1, .any⟩, ⟨"n1", 2, 1, .any⟩],
           { st1334 with cur := [(.name "n0", ⟨"n0", 0, 1, .node⟩), (.name "_anon_0", ⟨"_anon_0", 1, 1, .any⟩),
             (.rsv, ⟨"n1", 2, 1, .any⟩)] }) := by
    have ha : "_anon_0".startsWith "_anon" = true := by simp
    have hn : "n1".startsWith "_anon" = false := by simp
    simp (config := { decide := true }) [bindNode, st1334, liftLocals, bindNames, defineName,
      Env.find?, freshVar, Env.insert, Env.has, cleanup, hideExcept, restoreShadowed, Env.hideAll,
      Env.hide, Env.erase, Env.reserve, Env.rsvHas, Env.names, isAnon, List.lookup, ha, hn,
      beq_iff_eq]
  have hv : ¬ visible st1334 [] "n1" := by unfold visible; decide
  exact ⟨hb, hv, fun h => hv (h _ hb "n1" (by simp) (by decide))⟩

end FalkorBinder
