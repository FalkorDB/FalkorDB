import FalkorBinder.Resolve
/-
# Binding expressions: `bind_expr_node` and its wrappers

| here | there |
| --- | --- |
| `NE` / `BE` | raw `ExprIR<Arc<String>>` / bound `ExprIR<Variable>` trees (the variants the binder treats specially) |
| `reserve` | `fresh_var` + insert under `_quant_{s}_{id}` / `_lc_…` / `_reduce_acc_…` / `_reduce_iter_…` (binder.rs:1762-1765, 1796-1799, 1838-1844) |
| `bindNode`, `bindL` | `bind_expr_node` binder.rs:1746-2141 |
| `bindExpr`, `bindExprLocals` | `bind_expr` binder.rs:1725-1730, `bind_expr_with_locals` binder.rs:1732-1739 |
| `bindSetItems` | `bind_set_items` binder.rs:1700-1723 |
| `selfRef` | `check_unbound_self_referential` binder.rs:1682-1698 |

`bind_graph` (for pattern predicates / comprehensions) defines each pattern
name in the current table (`defineName … true`, Pattern.lean has the finer
node/relationship order). The constant fold and call rewrites are Fold.lean;
labels are Scopes.lean.

Theorems: `reserve_covers` / `bindNode_covers` (every reservation keeps the
table covering; only pattern cleanup can break it — `comprehension_cleanup_breaks`),
`bind_iterable_outer` (a comprehension's list is bound before its variable),
`loop_var_shadows`, `reduce_same_name`, `prop_on_path`, `exists_pattern`,
`func_arg_mismatch`, `bindExpr_eq`, `bindSetItems_*`, `selfRef_iff`.
-/
namespace FalkorBinder

inductive NE
  | var (n : Name)
  | quant (x : Name) (cs : List NE)
  | listComp (x : Name) (cs : List NE)
  | reduce (acc it : Name) (cs : List NE)
  | pattern (vars : List Name)
  | patComp (vars : List Name) (cs : List NE)
  | prop (k : Name) (cs : List NE)
  | func (name : Name) (argTys : List Ty) (cs : List NE)
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

def isAnon (n : Name) : Bool := n.startsWith "_anon"

/-- `bind_graph` on a pattern's names: each defined (reused if present). -/
def bindNames (st : St) : List Name → Except String (St × List Var)
  | [] => .ok (st, [])
  | n :: ns => match defineName st.cur st.depth n .any true with
    | .error e => .error e
    | .ok (e, v) => match bindNames { st with cur := e } ns with
      | .error e => .error e
      | .ok (st', vs) => .ok (st', v :: vs)

def injectLocals (e : Env) (locals : List Env) : Env :=
  locals.foldl (fun e l => l.foldl (fun e kv => e.insert kv.1 kv.2) e) e

/-- The cleanup after a pattern: keep the outer names and every `_anon` key. -/
def cleanup (outer : List Name) (e : Env) : Env := e.retain fun k => outer.contains k || isAnon k

def firstVar : List BE → Option Var
  | BE.var v :: _ => some v
  | _ => none

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
      | .ok (b0, st1) => match bindL st1 (locals ++ [[(x, r.2)]]) rest with
        | .error e => .error e
        | .ok (bs, st2) => .ok (.quant r.2 (b0 :: bs), st2)
    | [] => .error "unreachable: quantifier without list"
  | .listComp x cs => match cs with
    | c0 :: rest =>
      let r := reserve C st "_lc_" x
      match bindNode r.1 locals c0 with
      | .error e => .error e
      | .ok (b0, st1) => match bindL st1 (locals ++ [[(x, r.2)]]) rest with
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
    let outer := st.cur.map (·.1)
    match bindNames { st with cur := injectLocals st.cur locals } vars with
    | .error e => .error e
    | .ok (st1, vs) => .ok (.pattern vs, { st1 with cur := cleanup outer st1.cur })
  | .patComp vars cs =>
    let outer := st.cur.map (·.1)
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
  | .func name tys cs => match bindL st locals cs with
    | .error e => .error e
    | .ok (bs, st1) =>
      if name = "exists" ∧ bs.any (fun b => match b with | .pattern _ => true | _ => false) then
        .error "Invalid input: traversal patterns are not allowed as arguments to exists()"
      else if name ≠ "hasLabels" ∧ ((tys.zip bs).any fun tb => match tb.2 with
          | .var v => !C.compat v.ty tb.1
          | _ => false) then .error "Type mismatch"
      else .ok (.func name bs, st1)
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
      | .ok (bl, st2) => match bindNode st2 (locals ++ [[(acc, ra.2), (it, ri.2)]]) b0 with
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
    lookupLocals (locals ++ [[(x, v)]]) x = some v := by
  induction locals with
  | nil => simp [lookupLocals, Env.find?, List.lookup]
  | cons l ls ih => simp [lookupLocals, ih]

/-- The loop variable shadows everything inside the comprehension body. -/
theorem loop_var_shadows (st : St) (locals : List Env) (x : Name) (v : Var) :
    resolve st (locals ++ [[(x, v)]]) x = .ok (st, v) := by
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
  | .quant _ cs | .listComp _ cs | .reduce _ _ cs | .prop _ cs | .func _ _ cs | .other _ cs => patFreeL cs
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
        cases h2 : bindL C p1.2 (locals ++ [[(x, (reserve C st "_quant_" x).2)]]) rest with
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
        cases h2 : bindL C p1.2 (locals ++ [[(x, (reserve C st "_lc_" x).2)]]) rest with
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
    cases h1 : bindL C st locals cs with
    | error e => rw [h1] at h; simp at h
    | ok p =>
      rw [h1] at h; simp only at h
      have c1 := bindL_covers hk st locals cs hc hp p.1 p.2 h1
      split at h
      · simp at h
      · split at h
        · simp at h
        · simp at h; rw [← h.2]; exact c1
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
        cases h3 : bindNode C p2.2 (locals ++ [[(a, (reserve C st "_reduce_acc_" a).2),
            (i, (reserve C (reserve C st "_reduce_acc_" a).1 "_reduce_iter_" i).2)]]) b0 with
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
  | .quant _ cs | .listComp _ cs | .reduce _ _ cs | .patComp _ cs | .func _ _ cs | .other _ cs => selfRefL a cs
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

end FalkorBinder
