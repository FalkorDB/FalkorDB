import Columnar.Arith
import FalkorExprSemantics.Frame

/-!
# `VectorEval::eval` over expr_semantics' expression trees (vector_expr.rs:211-283)

`Dispatch.lean` models the tree walk over an abstract `Kind`/`Node`; this module models it over
proofs/expr_semantics' `E` — the `ExprIR` model whose per-row evaluator is `evalNode`
(`ExprEval::eval_node`) — so that the per-row reference of the soundness contract can *be*
`evalNode` (`CoreBridge.lean`). Arms in source order, guards included:

| `E` | arm |
| --- | --- |
| `const`, `param`, `var` | `Scalar(v)` (:219), `parameters.get` (:220-225), `eval_variable` (:226) |
| `prop attr ch` | `eval_property`, else `eval_per_row` (:227-232) |
| `and`, `or` | `eval_and_or(.., false/true)` (:233-234) |
| `not ch` | `negate_bools` (:235-238) |
| `add`, `arith` | `arithmetic` (:239-245) |
| `case` | `eval_case` (:246) |
| `eq`, `neq`, `cmp` | `compare_columns` (:247-252; `CmpOp::from_expr_ir` is `kindOp`) |
| `func f cs` | guard `!Aggregation && struct_fn.is_none() && !write && !any(Distinct)`, then `eval_has_labels` for `hasLabels`, else `eval_function` (:253-266) |
| everything else | `eval_per_row` (:267) |

`E`'s binary constructors are exactly the `num_children() == 2` shapes and `prop`/`not` the
one-child shapes; an `Add`…`Eq` node with another arity is `E.compound` and so per row, as in
Rust. **`eval_sound`**: if every arm is sound given sound children (what the lane theorems
state), `eval` is sound on every tree whose variables are in scope (`Scope`; beyond the
batch's column space the columnar `eval_variable` reads `Null` where the per-row view errors,
`Lanes.evalVariable_beyond`) and whose `hasLabels` calls are the registry's (`HL`), for
every row set of valid rows (`Rows`).
-/

namespace Columnar.VecEval
open FalkorExpr.Eval (E Fn Compound)

variable {Val C : Type}

/-- The arithmetic node kinds (`Add | Sub | Mul | Div | Modulo`, two children). -/
def aopOf : FalkorExpr.Eval.Arith → AOp
  | .sub => .sub | .mul => .mul | .div => .div | .rem => .rem

/-- `CmpOp::from_expr_ir` (vectorized.rs:70) on the comparison arms of `E`. -/
def cmpOf : FalkorExpr.Eval.CmpOp → CmpOp
  | .lt => .lt | .gt => .gt | .le => .le | .ge => .ge

def isDistinct : E Val → Bool
  | .distinct _ => true
  | _ => false

/-- The `FuncInvocation` guard (vector_expr.rs:253-258) on a non-aggregate call. -/
def fnGuard (f : Fn Val) (cs : List (E Val)) : Bool :=
  f.structFn.isNone && !f.write && !cs.any isDistinct

/-- Everything `eval` delegates to. Arms that recurse receive the recursive evaluator. -/
structure Arms (Val C : Type) where
  scalar : Val → C
  params : String → Option Val
  evalVariable : Nat → List Nat → C
  evalProperty : String → E Val → List Nat → Option C
  perRow : E Val → List Nat → Except String C
  andOr : (E Val → List Nat → Except String C) → List (E Val) → List Nat → Bool → Except String C
  negate : C → Nat → Except String C
  arith : AOp → C → C → Nat → Except String C
  case : (E Val → List Nat → Except String C) → Option (E Val) → List (E Val) → E Val → List Nat →
    Except String C
  compare : C → C → CmpOp → Nat → C
  hasLabels : List (E Val) → List Nat → Option C
  evalFunction : (E Val → List Nat → Except String C) → Fn Val → List (E Val) → List Nat → Except String C

variable (A : Arms Val C)

/-- `VectorEval::eval` (vector_expr.rs:211). -/
noncomputable def eval (e : E Val) (rows : List Nat) : Except String C :=
  let len := rows.length
  -- the recursive evaluator, available on strictly smaller trees
  let sub : E Val → List Nat → Except String C := fun ch r =>
    if h : sizeOf ch < sizeOf e then eval ch r else .error "not a subtree"
  match e with
  | .const v => .ok (A.scalar v)
  | .param name =>
    match A.params name with
    | some v => .ok (A.scalar v)
    | none => .error s!"Parameter {name} not found"
  | .var id _ => .ok (A.evalVariable id rows)
  | .prop attr ch =>
    match A.evalProperty attr ch rows with
    | some col => .ok col
    | none => A.perRow e rows
  | .and cs => A.andOr sub cs rows false
  | .or cs => A.andOr sub cs rows true
  | .not ch => do let x ← sub ch rows; A.negate x len
  | .add a b => do let x ← sub a rows; let y ← sub b rows; A.arith .add x y len
  | .arith op a b => do let x ← sub a rows; let y ← sub b rows; A.arith (aopOf op) x y len
  | .case s arms els => A.case sub s arms els rows
  | .eq a b => do let x ← sub a rows; let y ← sub b rows; pure (A.compare x y .eq len)
  | .neq a b => do let x ← sub a rows; let y ← sub b rows; pure (A.compare x y .neq len)
  | .cmp op a b => do let x ← sub a rows; let y ← sub b rows; pure (A.compare x y (cmpOf op) len)
  | .func f cs =>
    if fnGuard f cs then
      if f.name = "hasLabels" then
        match A.hasLabels cs rows with
        | some col => .ok col
        | none => A.evalFunction sub f cs rows
      else A.evalFunction sub f cs rows
    else A.perRow e rows
  | _ => A.perRow e rows
termination_by sizeOf e
decreasing_by all_goals exact h

/-- The per-row semantics and the soundness contract: `wf` is "the column has `len` rows". -/
structure RowSem (Val C : Type) where
  get : C → Nat → Val
  wf : C → Nat → Prop
  rowEval : E Val → Nat → Except String Val

variable (S : RowSem Val C)

/-- Column `col` answers `e` on `rows`: entry `i` = per-row value of row `rows[i]`. -/
def Sound (e : E Val) (rows : List Nat) (col : C) : Prop :=
  S.wf col rows.length ∧ ∀ i (h : i < rows.length), S.rowEval e rows[i] = .ok (S.get col i)

/-- A (sub-)evaluator is sound on `e` over every set of valid rows. -/
def SoundEv (Rows : Nat → Prop) (ev : E Val → List Nat → Except String C) (e : E Val) : Prop :=
  ∀ rows col, (∀ r ∈ rows, Rows r) → ev e rows = .ok col → Sound S e rows col

/-- The children an arm hands to the recursive evaluator. -/
def caseKids (s : Option (E Val)) (arms : List (E Val)) (els : E Val) : List (E Val) :=
  s.toList ++ arms ++ [els]

/-- What the leaves and lanes guarantee. `Scope id`: variable `id` is in the batch;
`Rows r`: `r` is a row of the batch; `HL f`: `f` is the registry's `hasLabels`. -/
structure Lawful (Scope : Nat → Prop) (Rows : Nat → Prop) (HL : Fn Val → Prop) : Prop where
  scalar : ∀ v e rows, (∀ r ∈ rows, S.rowEval e r = .ok v) → Sound S e rows (A.scalar v)
  const : ∀ v r, S.rowEval (.const v) r = .ok v
  param : ∀ name v r, A.params name = some v → S.rowEval (.param name) r = .ok v
  var : ∀ id n rows, Scope id → (∀ r ∈ rows, Rows r) → Sound S (.var id n) rows (A.evalVariable id rows)
  prop : ∀ attr ch rows col, (∀ id ∈ ch.vars, Scope id) → (∀ r ∈ rows, Rows r) →
    A.evalProperty attr ch rows = some col → Sound S (.prop attr ch) rows col
  perRow : ∀ e rows col, A.perRow e rows = .ok col → Sound S e rows col
  andOr : ∀ ev cs rows sc col, (∀ c ∈ cs, SoundEv S Rows ev c) → (∀ r ∈ rows, Rows r) →
    A.andOr ev cs rows sc = .ok col → Sound S (if sc then .or cs else .and cs) rows col
  negate : ∀ ch rows x col, Sound S ch rows x → A.negate x rows.length = .ok col → Sound S (.not ch) rows col
  add : ∀ a b rows x y col, Sound S a rows x → Sound S b rows y →
    A.arith .add x y rows.length = .ok col → Sound S (.add a b) rows col
  arith : ∀ op a b rows x y col, Sound S a rows x → Sound S b rows y →
    A.arith (aopOf op) x y rows.length = .ok col → Sound S (.arith op a b) rows col
  case : ∀ ev s arms els rows col, (∀ c ∈ caseKids s arms els, SoundEv S Rows ev c) → (∀ r ∈ rows, Rows r) →
    A.case ev s arms els rows = .ok col → Sound S (.case s arms els) rows col
  eq : ∀ a b rows x y, Sound S a rows x → Sound S b rows y → Sound S (.eq a b) rows (A.compare x y .eq rows.length)
  neq : ∀ a b rows x y, Sound S a rows x → Sound S b rows y → Sound S (.neq a b) rows (A.compare x y .neq rows.length)
  cmp : ∀ op a b rows x y, Sound S a rows x → Sound S b rows y →
    Sound S (.cmp op a b) rows (A.compare x y (cmpOf op) rows.length)
  hasLabels : ∀ f cs rows col, fnGuard f cs = true → HL f → (∀ id ∈ E.varsL cs, Scope id) →
    (∀ r ∈ rows, Rows r) → A.hasLabels cs rows = some col → Sound S (.func f cs) rows col
  func : ∀ ev f cs rows col, fnGuard f cs = true → (∀ c ∈ cs, SoundEv S Rows ev c) → (∀ r ∈ rows, Rows r) →
    A.evalFunction ev f cs rows = .ok col → Sound S (.func f cs) rows col

/-! ## The functions a tree calls -/

-- `E.funcs` (the `Fn`s a tree calls) is proofs/expr_semantics' (`Frame.lean`).

theorem mem_varsL_of {c : E Val} : ∀ {cs : List (E Val)}, c ∈ cs → ∀ id ∈ c.vars, id ∈ E.varsL cs
  | [], h, _, _ => by simp at h
  | x :: xs, h, id, hi => by
    simp only [E.varsL, List.mem_append]
    rcases List.mem_cons.1 h with rfl | h'
    · exact .inl hi
    · exact .inr (mem_varsL_of h' id hi)

theorem mem_funcsL_of {c : E Val} : ∀ {cs : List (E Val)}, c ∈ cs → ∀ f ∈ c.funcs, f ∈ E.funcsL cs
  | [], h, _, _ => by simp at h
  | x :: xs, h, f, hi => by
    simp only [E.funcsL, List.mem_append]
    rcases List.mem_cons.1 h with rfl | h'
    · exact .inl hi
    · exact .inr (mem_funcsL_of h' f hi)

theorem sizeOf_lt_of_mem' {c : E Val} {cs : List (E Val)} (h : c ∈ cs) : sizeOf c < sizeOf cs :=
  List.sizeOf_lt_of_mem h

variable {A S}

theorem bind_ok {α β} {x : Except String α} {f : α → Except String β} {b : β}
    (h : (x >>= f) = .ok b) : ∃ a, x = .ok a ∧ f a = .ok b := by
  cases x with
  | error e => simp [bind, Except.bind] at h
  | ok a => exact ⟨a, rfl, h⟩

/-- The premises `eval_sound` carries down a tree. -/
def InScope (Scope : Nat → Prop) (HL : Fn Val → Prop) (e : E Val) : Prop :=
  (∀ id ∈ e.vars, Scope id) ∧ (∀ f ∈ e.funcs, f.name = "hasLabels" → HL f)

/-- A child in the list of an n-ary node inherits the premises. -/
theorem inScope_mem {Scope : Nat → Prop} {HL : Fn Val → Prop} {cs : List (E Val)} {c : E Val} (hc : c ∈ cs)
    (hv : ∀ id ∈ E.varsL cs, Scope id) (hf : ∀ f ∈ E.funcsL cs, f.name = "hasLabels" → HL f) :
    InScope Scope HL c :=
  ⟨fun id hi => hv id (mem_varsL_of hc id hi), fun f hi => hf f (mem_funcsL_of hc f hi)⟩

/-- **`eval` is sound for every expression tree** in scope, over every set of valid rows. -/
theorem eval_sound {Scope : Nat → Prop} {Rows : Nat → Prop} {HL : Fn Val → Prop}
    (L : Lawful A S Scope Rows HL) : ∀ (e : E Val) (rows : List Nat) (col : C),
    InScope Scope HL e → (∀ r ∈ rows, Rows r) → eval A e rows = .ok col → Sound S e rows col
  | e, rows, col, hsc, hrows, h => by
    -- the recursive evaluator is sound on every smaller tree in scope
    have sub_sound : ∀ ch, sizeOf ch < sizeOf e → InScope Scope HL ch →
        SoundEv S Rows (fun c r => if _ : sizeOf c < sizeOf e then eval A c r else .error "not a subtree") ch := by
      intro ch hlt hch r col' hr hc'
      simp only [hlt, dite_true] at hc'
      exact eval_sound L ch r col' hch hr hc'
    have hv := hsc.1
    have hf := hsc.2
    unfold eval at h
    cases e with
    | const v =>
      simp only [Except.ok.injEq] at h; subst h
      exact L.scalar v _ rows (fun r _ => L.const v r)
    | param name =>
      simp only at h
      split at h
      · rename_i v hv'; simp only [Except.ok.injEq] at h; subst h
        exact L.scalar v _ rows (fun r _ => L.param name v r hv')
      · simp at h
    | var id n =>
      simp only [Except.ok.injEq] at h; subst h
      exact L.var id n rows (hv id (by simp [E.vars])) hrows
    | prop attr ch =>
      simp only at h
      split at h
      · rename_i hp; simp only [Except.ok.injEq] at h; subst h
        exact L.prop attr ch rows _ (fun id hi => hv id (by simpa [E.vars] using hi)) hrows hp
      · exact L.perRow _ rows col h
    | and cs =>
      have := L.andOr _ cs rows false col (fun c hc => sub_sound c
        (by have := sizeOf_lt_of_mem' hc; simp; omega)
        (inScope_mem hc (by simpa [E.vars] using hv) (by simpa [E.funcs] using hf))) hrows h
      simpa using this
    | or cs =>
      have := L.andOr _ cs rows true col (fun c hc => sub_sound c
        (by have := sizeOf_lt_of_mem' hc; simp; omega)
        (inScope_mem hc (by simpa [E.vars] using hv) (by simpa [E.funcs] using hf))) hrows h
      simpa using this
    | not ch =>
      simp only at h
      obtain ⟨x, hx, hn⟩ := bind_ok h
      exact L.negate ch rows x col (sub_sound ch (by simp)
        ⟨by simpa [E.vars] using hv, by simpa [E.funcs] using hf⟩ rows x hrows hx) hn
    | add a b =>
      simp only at h
      obtain ⟨x, hx, h⟩ := bind_ok h; obtain ⟨y, hy, h⟩ := bind_ok h
      simp only [E.vars, List.mem_append] at hv; simp only [E.funcs, List.mem_append] at hf
      exact L.add a b rows x y col
        (sub_sound a (by simp; omega) ⟨fun i hi => hv i (.inl hi), fun g hg => hf g (.inl hg)⟩ rows x hrows hx)
        (sub_sound b (by simp; omega) ⟨fun i hi => hv i (.inr hi), fun g hg => hf g (.inr hg)⟩ rows y hrows hy) h
    | arith op a b =>
      simp only at h
      obtain ⟨x, hx, h⟩ := bind_ok h; obtain ⟨y, hy, h⟩ := bind_ok h
      simp only [E.vars, List.mem_append] at hv; simp only [E.funcs, List.mem_append] at hf
      exact L.arith op a b rows x y col
        (sub_sound a (by simp; omega) ⟨fun i hi => hv i (.inl hi), fun g hg => hf g (.inl hg)⟩ rows x hrows hx)
        (sub_sound b (by simp; omega) ⟨fun i hi => hv i (.inr hi), fun g hg => hf g (.inr hg)⟩ rows y hrows hy) h
    | eq a b =>
      simp only at h
      obtain ⟨x, hx, h⟩ := bind_ok h; obtain ⟨y, hy, h⟩ := bind_ok h
      simp only [pure, Except.pure, Except.ok.injEq] at h; subst h
      simp only [E.vars, List.mem_append] at hv; simp only [E.funcs, List.mem_append] at hf
      exact L.eq a b rows x y
        (sub_sound a (by simp; omega) ⟨fun i hi => hv i (.inl hi), fun g hg => hf g (.inl hg)⟩ rows x hrows hx)
        (sub_sound b (by simp; omega) ⟨fun i hi => hv i (.inr hi), fun g hg => hf g (.inr hg)⟩ rows y hrows hy)
    | neq a b =>
      simp only at h
      obtain ⟨x, hx, h⟩ := bind_ok h; obtain ⟨y, hy, h⟩ := bind_ok h
      simp only [pure, Except.pure, Except.ok.injEq] at h; subst h
      simp only [E.vars, List.mem_append] at hv; simp only [E.funcs, List.mem_append] at hf
      exact L.neq a b rows x y
        (sub_sound a (by simp; omega) ⟨fun i hi => hv i (.inl hi), fun g hg => hf g (.inl hg)⟩ rows x hrows hx)
        (sub_sound b (by simp; omega) ⟨fun i hi => hv i (.inr hi), fun g hg => hf g (.inr hg)⟩ rows y hrows hy)
    | cmp op a b =>
      simp only at h
      obtain ⟨x, hx, h⟩ := bind_ok h; obtain ⟨y, hy, h⟩ := bind_ok h
      simp only [pure, Except.pure, Except.ok.injEq] at h; subst h
      simp only [E.vars, List.mem_append] at hv; simp only [E.funcs, List.mem_append] at hf
      exact L.cmp op a b rows x y
        (sub_sound a (by simp; omega) ⟨fun i hi => hv i (.inl hi), fun g hg => hf g (.inl hg)⟩ rows x hrows hx)
        (sub_sound b (by simp; omega) ⟨fun i hi => hv i (.inr hi), fun g hg => hf g (.inr hg)⟩ rows y hrows hy)
    | case s arms els =>
      refine L.case _ s arms els rows col (fun c hc => ?_) hrows h
      simp only [caseKids, List.mem_append, List.mem_singleton] at hc
      simp only [E.vars, List.mem_append] at hv; simp only [E.funcs, List.mem_append] at hf
      rcases hc with (hc | hc) | rfl
      · cases s with
        | none => simp at hc
        | some s0 =>
          simp only [Option.toList_some, List.mem_singleton] at hc; subst hc
          exact sub_sound _ (by simp; omega)
            ⟨fun i hi => hv i (.inl (.inl (by simpa [E.varsO] using hi))),
             fun g hg => hf g (.inl (.inl (by simpa [E.funcsO] using hg)))⟩
      · exact sub_sound c (by have := sizeOf_lt_of_mem' hc; simp; omega)
          ⟨fun i hi => hv i (.inl (.inr (mem_varsL_of hc i hi))),
           fun g hg => hf g (.inl (.inr (mem_funcsL_of hc g hg)))⟩
      · exact sub_sound _ (by simp; omega) ⟨fun i hi => hv i (.inr hi), fun g hg => hf g (.inr hg)⟩
    | func f cs =>
      simp only at h
      have hvs : ∀ id ∈ E.varsL cs, Scope id := by simpa [E.vars] using hv
      have hfs : ∀ g ∈ E.funcsL cs, g.name = "hasLabels" → HL g := fun g hg => hf g (by simp [E.funcs, hg])
      have kids : ∀ c ∈ cs, SoundEv S Rows _ c := fun c hc => sub_sound c
        (by have := sizeOf_lt_of_mem' hc; simp; omega) (inScope_mem hc hvs hfs)
      split at h
      · rename_i hg
        split at h
        · rename_i hn
          split at h
          · rename_i col' hl; simp only [Except.ok.injEq] at h; subst h
            exact L.hasLabels f cs rows col' hg (hf f (by simp [E.funcs]) hn) hvs hrows hl
          · exact L.func _ f cs rows col hg kids hrows h
        · exact L.func _ f cs rows col hg kids hrows h
      · exact L.perRow _ rows col h
    | map _ | key _ _ | compound _ _ | xor _ | inList _ _ | list _ | getElement _ _ | mapProj _ _
    | projAll | projProp _ | distinct _ => exact L.perRow _ rows col h
termination_by e => sizeOf e
decreasing_by all_goals simp_wf; omega

/-! ## `eval_values` (vector_expr.rs:285-307) -/

variable (A)

/-- `eval_values`: the bulk property values, the variable passthrough
(`batch.value_at(var.id, row).unwrap_or(Null)`), else `eval(..).into_values(len)`. -/
noncomputable def evalValues (intoValues : C → Nat → List Val)
    (propVals : String → E Val → List Nat → Option (List Val)) (varVals : Nat → List Nat → List Val)
    (e : E Val) (rows : List Nat) : Except String (List Val) :=
  let fallback := (eval A e rows).map fun c => intoValues c rows.length
  match e with
  | .prop attr ch => match propVals attr ch rows with
    | some vs => .ok vs
    | none => fallback
  | .var id _ => .ok (varVals id rows)
  | _ => fallback

variable {A}

/-- Each `eval_values` arm returns the per-row values. -/
theorem evalValues_sound {Scope : Nat → Prop} {Rows : Nat → Prop} {HL : Fn Val → Prop}
    (L : Lawful A S Scope Rows HL) (intoValues : C → Nat → List Val)
    (propVals : String → E Val → List Nat → Option (List Val)) (varVals : Nat → List Nat → List Val)
    (null : Val)
    (hInto : ∀ c rows e, Sound S e rows c → ∀ i (hi : i < rows.length),
        S.rowEval e rows[i] = .ok ((intoValues c rows.length).getD i null))
    (hVar : ∀ id n rows, Scope id → (∀ r ∈ rows, Rows r) → ∀ i (hi : i < rows.length),
        S.rowEval (.var id n) rows[i] = .ok ((varVals id rows).getD i null))
    (hProp : ∀ attr ch rows vs, (∀ id ∈ ch.vars, Scope id) → (∀ r ∈ rows, Rows r) →
        propVals attr ch rows = some vs →
        ∀ i (hi : i < rows.length), S.rowEval (.prop attr ch) rows[i] = .ok (vs.getD i null))
    (e : E Val) (rows : List Nat) (vs : List Val) (hsc : InScope Scope HL e) (hrows : ∀ r ∈ rows, Rows r)
    (h : evalValues A intoValues propVals varVals e rows = .ok vs) :
    ∀ i (hi : i < rows.length), S.rowEval e rows[i] = .ok (vs.getD i null) := by
  have fb : ∀ (vs : List Val), ((eval A e rows).map fun c => intoValues c rows.length) = .ok vs →
      ∀ i (hi : i < rows.length), S.rowEval e rows[i] = .ok (vs.getD i null) := by
    intro vs h
    cases he : eval A e rows with
    | error err => simp [he, Except.map] at h
    | ok c =>
      simp only [he, Except.map, Except.ok.injEq] at h; subst h
      exact hInto c rows e (eval_sound L e rows c hsc hrows he)
  unfold evalValues at h
  split at h
  · rename_i attr ch
    split at h
    · rename_i hp; simp only [Except.ok.injEq] at h; subst h
      exact hProp attr ch rows _ (fun id hi => hsc.1 id (by simpa [E.vars] using hi)) hrows hp
    · exact fb vs h
  · rename_i id n
    simp only [Except.ok.injEq] at h; subst h
    exact hVar id n rows (hsc.1 id (by simp [E.vars])) hrows
  · exact fb vs h

end Columnar.VecEval
