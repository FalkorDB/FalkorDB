import FalkorBinder.Resolve
/-
# Binding expressions: `bind_expr_node` and its wrappers

`bind_graph` (for pattern predicates / comprehensions) defines each pattern
name in the current table (`defineName … true`, Pattern.lean has the finer
node/relationship order). The constant fold and call rewrites are Fold.lean;
labels are Scopes.lean.

Theorems: `reserveLoop_covers` / `bindNode_covers` (every reservation keeps the
table covering; since #3016 a reservation is a nameless `Scope::reserve`, so no key can clash),
`bind_aggPlaceholder` (the parser's `AggOrderPlaceholder` gets a fresh reserved variable),
`hideExcept_find` (a pattern's own names do not leak out; `Scopes.comprehension_cleanup_covers`),
`bind_iterable_outer` (a comprehension's list is bound before its variable),
`loop_var_shadows`, `reduce_same_name`, `prop_on_path`, `exists_pattern`,
`func_arg_mismatch`, `bindExpr_eq`, `bindSetItems_*`, `selfRef_iff`.

`bind_paren` (`(x)` binds exactly like `x`: same tree, same state, same
errors — so no bound tree holds a `Paren`, which `BE` has no constructor for),
`prop_on_paren_path` (the `(p).prop` rejection comes from that, not from a
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
  /-- a pattern predicate: its entities' aliases (`Alias`, #3303) -/
  | pattern (vars : List Alias)
  | patComp (vars : List Alias) (cs : List NE)
  /-- `ExprIR::AggOrderPlaceholder` (#3016): the parser's ORDER BY slot of an aggregate -/
  | aggPh
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

/-- Mint a loop variable (or the aggregate placeholder) and reserve its slot, nameless
(`Scope::reserve`, #3016; before, under a `_quant_{s}_{id}`-style key). -/
def reserveLoop (st : St) (x : Name) : St × Var :=
  let v := freshVar st.cur x .any st.depth
  ({ st with cur := st.cur.reserve v }, v)

/-- `bind_graph` on a pattern's aliases: each `define_alias_in_scope`d (reused if present). -/
def bindNames (st : St) : List Alias → Except String (St × List Var)
  | [] => .ok (st, [])
  | n :: ns => match defineAlias st n .any true with
    | .error e => .error e
    | .ok (st1, v) => match bindNames st1 ns with
      | .error e => .error e
      | .ok (st', vs) => .ok (st', v :: vs)

/-- `Binder::lift_locals` on one local `(n, v)`: a scope variable of that name keeps its slot
(reserved), then `v` is lifted under `n`; what `n` meant before is recorded. -/
def liftOne (acc : Env × List (Name × Option Var)) (n : Name) (v : Var) :
    Env × List (Name × Option Var) :=
  let prev := acc.1.find? n
  let e1 := match prev with
    | some p => acc.1.reserve p
    | none => acc.1
  (e1.lift n v, acc.2 ++ [(n, prev)])

/-- `Binder::lift_locals` (binder.rs): every comprehension local, outermost first, put under its
name (a local table holds named entries only). -/
def liftLocals (e : Env) (locals : List Env) : Env × List (Name × Option Var) :=
  locals.foldl (fun acc l => l.foldl (fun acc kv => match kv.1 with
    | .nm n => liftOne acc n kv.2
    | .un => acc) acc) (e, [])

/-- `Binder::hide_names_except` (binder.rs): hide every name not in `keep`; the unnamed
entries stay. -/
def hideExcept (keep : List Name) (e : Env) : Env :=
  e.hideNames (e.keys.filter fun k => !keep.contains k)

/-- `Binder::restore_shadowed` (binder.rs): last lifted first, hide the name, and give a
shadowed variable its name back. -/
def restoreShadowed (e : Env) (sh : List (Name × Option Var)) : Env :=
  sh.reverse.foldl (fun e np => match np.2 with
    | some p => (e.hideName np.1).lift np.1 p
    | none => e.hideName np.1) e

/-- The cleanup after a pattern (binder.rs, #3016): hide the pattern's own names, keeping their
slots (before #3016 they were removed, freeing ids the bound pattern still used), then undo the
lifting. -/
def cleanup (outer : List Name) (sh : List (Name × Option Var)) (e : Env) : Env :=
  restoreShadowed (hideExcept outer e) sh

def firstVar : List BE → Option Var
  | BE.var v :: _ => some v
  | _ => none

/-- What one bound operand contributes to `flat`: its children when it is the same connective. -/
def spliceB (o : Bool) : BE → List BE
  | .andOr o' cs => if o' = o then cs else [.andOr o' cs]
  | b => [b]

/-- `ExprIR::flat` (ast.rs) on bound operands: a lone operand of another
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
  | .aggPh =>
    let r := reserveLoop st "__agg_order_by_placeholder__"
    .ok (.var r.2, r.1)
  | .quant x cs => match cs with
    | c0 :: rest =>
      let r := reserveLoop st x
      match bindNode r.1 locals c0 with
      | .error e => .error e
      | .ok (b0, st1) => match bindL st1 (locals ++ [[(.nm x, r.2)]]) rest with
        | .error e => .error e
        | .ok (bs, st2) => .ok (.quant r.2 (b0 :: bs), st2)
    | [] => .error "unreachable: quantifier without list"
  | .listComp x cs => match cs with
    | c0 :: rest =>
      let r := reserveLoop st x
      match bindNode r.1 locals c0 with
      | .error e => .error e
      | .ok (b0, st1) => match bindL st1 (locals ++ [[(.nm x, r.2)]]) rest with
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
    let outer := st.cur.keys
    let lifted := liftLocals st.cur locals
    match bindNames { st with cur := lifted.1 } vars with
    | .error e => .error e
    | .ok (st1, vs) => .ok (.pattern vs, { st1 with cur := cleanup outer lifted.2 st1.cur })
  | .patComp vars cs =>
    let outer := st.cur.keys
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
    let ra := reserveLoop st acc
    let ri := reserveLoop ra.1 it
    match bindNode ri.1 locals i0 with
    | .error e => .error e
    | .ok (bi, st1) => match bindNode st1 locals l0 with
      | .error e => .error e
      | .ok (bl, st2) => match bindNode st2 (locals ++ [[(.nm acc, ra.2), (.nm it, ri.2)]]) b0 with
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
    lookupLocals (locals ++ [[(.nm x, v)]]) x = some v := by
  induction locals with
  | nil => simp [lookupLocals, Env.find?, List.lookup]
  | cons l ls ih => simp [lookupLocals, ih]

/-- The loop variable shadows everything inside the comprehension body. -/
theorem loop_var_shadows (st : St) (locals : List Env) (x : Name) (v : Var) :
    resolve st (locals ++ [[(.nm x, v)]]) x = .ok (st, v) := by
  unfold resolve; rw [lookupLocals_last]

/-- The list of `[x IN list | …]` is bound with the *outer* locals (before
`x` exists), in the state after the slot reservation. -/
theorem bind_iterable_outer (st : St) (locals : List Env) (x : Name) (c0 : NE) (rest : List NE)
    (b : BE) (st' : St) (h : bindNode C st locals (.listComp x (c0 :: rest)) = .ok (b, st')) :
    ∃ b0 st1, bindNode C (reserveLoop st x).1 locals c0 = .ok (b0, st1) := by
  simp only [bindNode] at h
  cases h0 : bindNode C (reserveLoop st x).1 locals c0 with
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
  obtain ⟨vs, rfl⟩ := hp
  simp [bindNode, bindL] at h ⊢
  rw [h]; simp

/-- A variable argument of an incompatible declared type is rejected (except `hasLabels`). -/
theorem func_arg_mismatch (st : St) (locals : List Env) (name n : Name) (t : Ty) (v : Var) (st1 : St)
    (hr : resolve st locals n = .ok (st1, v)) (hc : C.compat v.ty t = false) (hn : name ≠ "exists")
    (hh : name ≠ "hasLabels") : bindNode C st locals (.func name [t] [.var n]) = .error "Type mismatch" := by
  simp [bindNode, bindL, hr, hc, hn, hh]

/-! ## Parentheses and AND/OR flattening -/

/-- **`(x)` binds exactly like `x`** (binder.rs): same bound tree, same
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

theorem reserveLoop_covers (st : St) (x : Name) (h : st.cur.Covers) :
    (reserveLoop st x).1.cur.Covers := (reserve_covers h _ rfl).2

theorem reserveLoop_frame (st : St) (x : Name) :
    (reserveLoop st x).1.depth = st.depth ∧ (reserveLoop st x).1.parent = st.parent ∧
    (reserveLoop st x).2.id = st.cur.length ∧ (reserveLoop st x).2.scope = st.depth ∧
    (∀ n, (reserveLoop st x).1.cur.find? n = st.cur.find? n) :=
  ⟨rfl, rfl, rfl, rfl, fun n => Env.reserve_find _ _ n⟩

/-- **The aggregate placeholder** (`ExprIR::AggOrderPlaceholder`, #3016): a fresh variable at
`len()` in the current scope, reserved nameless, whatever the scope or the locals hold; no name
reaches it and the table stays covering. -/
theorem bind_aggPlaceholder (st : St) (locals : List Env) (hc : st.cur.Covers) :
    ∃ v, bindNode C st locals .aggPh = .ok (.var v, { st with cur := st.cur ++ [(.un, v)] }) ∧
      v.id = st.cur.length ∧ v.scope = st.depth ∧ v.name = some "__agg_order_by_placeholder__" ∧
      (∀ n, Env.find? (st.cur ++ [(.un, v)]) n = st.cur.find? n) ∧
      Env.Covers (st.cur ++ [(.un, v)]) := by
  have hr := reserve_covers hc (freshVar st.cur "__agg_order_by_placeholder__" .any st.depth) rfl
  refine ⟨freshVar st.cur "__agg_order_by_placeholder__" .any st.depth, ?_, rfl, rfl, rfl, ?_, ?_⟩
  · simp only [bindNode, reserveLoop, Env.reserve_eq_append hr.1]
  · intro n; rw [← Env.reserve_eq_append hr.1]; exact Env.reserve_find _ _ n
  · rw [← Env.reserve_eq_append hr.1]; exact hr.2

theorem Env.find_some_keys : ∀ (e : Env) (n : Name) (v : Var), e.find? n = some v → n ∈ e.keys
  | [], _, _, h => by simp [Env.find?] at h
  | (k, w) :: rest, n, v, h => by
    by_cases hk : k = .nm n
    · subst hk; simp [Env.keys]
    · have : (Key.nm n == k) = false := by simp; exact fun h => hk h.symm
      have hr : Env.find? rest n = some v := by simpa [Env.find?, List.lookup, this] using h
      have := Env.find_some_keys rest n v hr
      cases k <;> simp_all [Env.keys]

/-- **A pattern's own names do not leak out** (`hide_names_except`): after the cleanup a name
resolves in the scope only if it was there before the pattern (and then as before), so a
pattern-local alias is gone, though its slot is kept (`Env.hideNames_covers`). -/
theorem hideExcept_find (keep : List Name) (e : Env) (m : Name) :
    (hideExcept keep e).find? m = if keep.contains m then e.find? m else none := by
  unfold hideExcept
  rw [Env.hideNames_find]
  by_cases hk : m ∈ keep
  · have hb : keep.contains m = true := by simp [hk]
    have : m ∉ e.keys.filter fun k => !keep.contains k := by simp [List.mem_filter, hk]
    rw [if_neg this]; simp [hk]
  · have hb : keep.contains m = false := by simp [hk]
    simp only [hb, Bool.false_eq_true, ite_false]
    by_cases hm : m ∈ e.keys.filter fun k => !keep.contains k
    · rw [if_pos hm]
    · rw [if_neg hm]
      cases hf : e.find? m with
      | none => rfl
      | some v =>
        exact absurd (List.mem_filter.2 ⟨Env.find_some_keys e m v hf, by simp [hk]⟩) hm

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
      have h0 := reserveLoop_covers st x hc
      cases h1 : bindNode C (reserveLoop st x).1 locals c0 with
      | error e => rw [h1] at h; simp at h
      | ok p1 =>
        rw [h1] at h; simp only at h
        have c1 := bindNode_covers _ locals c0 h0 hp.1 p1.1 p1.2 h1
        cases h2 : bindL C p1.2 (locals ++ [[(.nm x, (reserveLoop st x).2)]]) rest with
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
      have h0 := reserveLoop_covers st x hc
      cases h1 : bindNode C (reserveLoop st x).1 locals c0 with
      | error e => rw [h1] at h; simp at h
      | ok p1 =>
        rw [h1] at h; simp only at h
        have c1 := bindNode_covers _ locals c0 h0 hp.1 p1.1 p1.2 h1
        cases h2 : bindL C p1.2 (locals ++ [[(.nm x, (reserveLoop st x).2)]]) rest with
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
  | st, locals, .aggPh, hc, _, b, st', h => by
    simp only [bindNode, Except.ok.injEq, Prod.mk.injEq] at h
    obtain ⟨-, rfl⟩ := h
    exact reserveLoop_covers st _ hc
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
    have h0 := reserveLoop_covers _ i (reserveLoop_covers st a hc)
    simp only [bindReduce] at h
    cases h1 : bindNode C (reserveLoop (reserveLoop st a).1 i).1 locals i0 with
    | error e => rw [h1] at h; simp at h
    | ok p1 =>
      rw [h1] at h; simp only at h
      have c1 := bindNode_covers _ locals i0 h0 hp.1 p1.1 p1.2 h1
      cases h2 : bindNode C p1.2 locals l0 with
      | error e => rw [h2] at h; simp at h
      | ok p2 =>
        rw [h2] at h; simp only at h
        have c2 := bindNode_covers _ locals l0 c1 hp.2.1 p2.1 p2.2 h2
        cases h3 : bindNode C p2.2 (locals ++ [[(Key.nm a, (reserveLoop st a).2),
            (Key.nm i, (reserveLoop (reserveLoop st a).1 i).2)]]) b0 with
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

/-! ## `check_unbound_self_referential` -/

/- DFS: some `Property` whose first child is the entity's own alias. -/
mutual
def selfRef (a : Name) : NE → Bool
  | .prop _ (.var n :: _) => n == a
  | .prop _ cs => selfRefL a cs
  | .var _ | .pattern _ | .aggPh => false
  | .quant _ cs | .listComp _ cs | .reduce _ _ cs | .patComp _ cs | .func _ _ cs | .paren cs | .andOr _ cs
  | .other _ cs => selfRefL a cs
def selfRefL (a : Name) : List NE → Bool
  | [] => false
  | c :: cs => selfRef a c || selfRefL a cs
end

/-- `check_unbound_self_referential` (binder.rs) takes the entity's `Alias` (#3303):
an unnamed entity has no name its attributes could mention. -/
def checkSelfRef (a : Alias) (attrs : NE) : Except String Unit :=
  match a with
  | .anon _ => .ok ()
  | .named n => if selfRef n attrs then .error s!"'{n}' not defined; undefined attribute" else .ok ()

theorem checkSelfRef_iff (a : Alias) (attrs : NE) :
    checkSelfRef a attrs = .ok () ↔ (∀ n, a = .named n → selfRef n attrs = false) := by
  unfold checkSelfRef
  cases a with
  | anon k => simp
  | named n => by_cases h2 : selfRef n attrs <;> simp [h2]

/-- `CREATE (a {v: a.v})` is rejected; `CREATE (a {v: [a.v][0]})` too (the walk is deep). Since
#3303 this holds for every written name, `_anon…` included. -/
theorem selfRef_direct (a k : Name) :
    checkSelfRef (.named a) (.other 0 [.prop k [.var a]]) =
      .error s!"'{a}' not defined; undefined attribute" := by
  simp [checkSelfRef, selfRef, selfRefL]

/-- #3841: a map-projection selector (the `.a` of `x {.a}`) is a childless `Property`; it has no
operand, so it is never a self-reference (before #3841 the Rust indexed the missing operand and
panicked: `CREATE (a {f: 1 {.a}})`). -/
theorem selfRef_leaf_prop (a b k : Name) (n : Nat) :
    checkSelfRef (.named a) (.other n [.var b, .prop k []]) = .ok () := by
  simp [checkSelfRef, selfRef, selfRefL]


/-! ## #1334: a pattern predicate may declare a new variable (gap closed 2026-10-20)

The `ExprIR::Pattern` arm of `bind_expr_node` (binder.rs) binds the predicate with
`bind_graph(&pattern, false)`, which `define_name_in_scope`s every alias (`allow_reuse`), minting
a fresh variable for a name that is not in scope; the cleanup then only hides it again (since
#3016 keeping its slot: `n1` moves to the scope's nameless list). Nothing
checks that a *named* alias already existed, so `WHERE () -- (n1)` with no `n1` in scope binds
and runs as "some edge exists anywhere". -/

/-- **A pattern predicate declares nothing** (openCypher; C rejects it, issue #1334; the
exact C error message is not recorded here): whenever `bind_expr_node` accepts a pattern predicate, every named
(`Alias.named`) variable in it was already visible where the predicate stands. A fix proves
this by rejecting the pattern otherwise. -/
def PatternPredDeclaresNothing (st : St) (locals : List Env) (vars : List Alias) : Prop :=
  ∀ r, bindNode C st locals (.pattern vars) = .ok r →
    ∀ n, .named n ∈ vars → visible st locals n

/-- The binder state while `MATCH (n0) WITH * WHERE …` binds its WHERE: the projection scope
`{n0}` (scope 1), the parent `{n0}`, `use_parent_scope` set; `WITH *` adds nothing to
`parent_to_child_scope` (binder.rs). -/
def st1334 : St :=
  { cur := [("n0", ⟨"n0", 0, 1, .node, none⟩)], parent := [("n0", ⟨"n0", 0, 0, .node, none⟩)],
    depth := 1, useParent := true, p2c := [], cfp := [] }

/-- **#1334**: `MATCH (n0) WITH * WHERE () -- (n1) RETURN *` binds — `n1` is minted fresh (id 2,
after the anonymous node's 1) and hidden — although `n1` is visible nowhere. Live: Rust main
returns 2 rows on a 2-node graph with an edge; C errors. -/
theorem bug1334_pattern_predicate_declares_var :
    bindNode C st1334 [] (.pattern [.anon 0, .named "n1"]) =
      .ok (.pattern [⟨none, 1, 1, .any, some 0⟩, ⟨"n1", 2, 1, .any, none⟩],
           { st1334 with cur := [("n0", ⟨"n0", 0, 1, .node, none⟩), (.un, ⟨none, 1, 1, .any, some 0⟩),
                                 (.un, ⟨"n1", 2, 1, .any, none⟩)] }) ∧
    ¬ visible st1334 [] "n1" ∧
    ¬ PatternPredDeclaresNothing C st1334 [] [.anon 0, .named "n1"] := by
  have hb : bindNode C st1334 [] (.pattern [.anon 0, .named "n1"]) =
      .ok (.pattern [⟨none, 1, 1, .any, some 0⟩, ⟨"n1", 2, 1, .any, none⟩],
           { st1334 with cur := [("n0", ⟨"n0", 0, 1, .node, none⟩), (.un, ⟨none, 1, 1, .any, some 0⟩),
                                 (.un, ⟨"n1", 2, 1, .any, none⟩)] }) := by
    simp (config := { decide := true }) [bindNode, st1334, liftLocals, bindNames, defineAlias,
      defineName, Env.find?, freshVar, freshAnon, Env.insert, Env.has, Env.reserve, Env.keys,
      Env.lookupAnon, Env.unnamed, Env.unnamedIds, cleanup, hideExcept, restoreShadowed,
      Env.hideNames, Env.hideName, Env.erase, List.lookup, List.filter, List.filterMap]
  have hv : ¬ visible st1334 [] "n1" := by unfold visible; decide
  exact ⟨hb, hv, fun h => hv (h _ hb "n1" (by simp))⟩

end FalkorBinder
