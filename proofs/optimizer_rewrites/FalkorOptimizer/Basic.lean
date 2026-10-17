/-
# Rows, expressions and the Filter operator

Reference semantics shared by every pass model.

| here | there |
| --- | --- |
| `Var`            | `parser::ast::Variable` (`graph/src/parser/ast.rs:92`) — `(id, scope_id)`; `id` is scope-local |
| `Val`            | `runtime::value::Value` (only the constructors the passes care about) |
| `Row`, `Row.get` | a row of the batch env; an unbound column reads as NULL |
| `Expr`, `eval`   | `ExprEval::eval_node` (`graph/src/runtime/eval.rs`) — `And` at `eval.rs:674` |
| `andL`           | the n-ary `ExprIR::And` loop (`eval.rs:674-699`), as a right fold of the binary step |
| `passes`         | per-row verdict of `FilterOp::eval` (`graph/src/runtime/ops/filter.rs:48-80`) |
| `filterRows`     | `FilterOp` over the whole input: any row error fails the query |
| `cp`             | `CartesianProduct` (`runtime/ops/cartesian_product.rs`) |
| `extend`         | any per-row expanding operator that keeps its input columns (`Unwind`, `CondTraverse`, `Apply` right side, `ProcedureCall`, …) |
| `List.take`/`drop` | `Limit` / `Skip` (`runtime/ops/limit.rs`, `skip.rs`) |
| `List.mergeSort` | `Sort` (stable) |
-/

namespace Falkor.Opt

structure Var where
  id : Nat
  scope : Nat
deriving DecidableEq, Repr

inductive Val where
  | null
  | bool (b : Bool)
  | int (i : Int)
  | str (s : String)
deriving DecidableEq, Repr

inductive Err where
  | divZero
  | typeMismatch
deriving DecidableEq, Repr

abbrev Row := List (Var × Val)

/-- First binding wins (a row built as `x ++ y` sees `x`'s columns first). -/
def Row.get (r : Row) (v : Var) : Val :=
  match r.find? (fun p => p.1 == v) with
  | some p => p.2
  | none => .null

def Row.binds (r : Row) (v : Var) : Prop := v ∈ r.map Prod.fst

inductive Expr where
  | lit (v : Val)
  | var (v : Var)
  | eq (a b : Expr)
  | neq (a b : Expr)
  | lt (a b : Expr)
  | gt (a b : Expr)
  | sub (a b : Expr)
  | div (a b : Expr)
  /-- one step of the `ExprIR::And` loop: `and a rest` -/
  | and (a b : Expr)
deriving Repr

def Expr.vars : Expr → List Var
  | .lit _ => []
  | .var v => [v]
  | .eq a b | .neq a b | .lt a b | .gt a b | .sub a b | .div a b | .and a b => a.vars ++ b.vars

def cmpEq : Val → Val → Val
  | .null, _ | _, .null => .null
  | .int a, .int b => .bool (a == b)
  | .bool a, .bool b => .bool (a == b)
  | .str a, .str b => .bool (a == b)
  | _, _ => .bool false

def cmpLt : Val → Val → Val
  | .int a, .int b => .bool (decide (a < b))
  | .str a, .str b => .bool (decide (a < b))
  | _, _ => .null   -- ComparedNull / Disjoint → NULL (eval.rs `Lt` arm)

def notV : Val → Val
  | .bool b => .bool (!b)
  | v => v

/-- The `ExprIR::And` step (eval.rs:674): evaluate left; `false` short-circuits,
an error propagates immediately (`?`), `null` is remembered, a non-boolean is a
type error. -/
def andStep (a : Except Err Val) (rest : Unit → Except Err Val) : Except Err Val :=
  match a with
  | .error e => .error e
  | .ok (.bool false) => .ok (.bool false)
  | .ok (.bool true) => rest ()
  | .ok .null =>
    match rest () with
    | .ok (.bool false) => .ok (.bool false)
    | .ok (.bool true) => .ok .null
    | .ok .null => .ok .null
    | .ok _ => .error .typeMismatch
    | .error e => .error e
  | .ok _ => .error .typeMismatch

def eval (r : Row) : Expr → Except Err Val
  | .lit v => .ok v
  | .var v => .ok (r.get v)
  | .eq a b => do let x ← eval r a; let y ← eval r b; pure (cmpEq x y)
  | .neq a b => do let x ← eval r a; let y ← eval r b; pure (notV (cmpEq x y))
  | .lt a b => do let x ← eval r a; let y ← eval r b; pure (cmpLt x y)
  | .gt a b => do let x ← eval r a; let y ← eval r b; pure (cmpLt y x)
  | .sub a b => do
      let x ← eval r a; let y ← eval r b
      match x, y with
      | .int i, .int j => pure (.int (i - j))
      | .null, _ | _, .null => pure .null
      | _, _ => throw .typeMismatch
  | .div a b => do
      let x ← eval r a; let y ← eval r b
      match x, y with
      | .int _, .int 0 => throw .divZero
      | .int i, .int j => pure (.int (i.tdiv j))
      | .null, _ | _, .null => pure .null
      | _, _ => throw .typeMismatch
  | .and a b => andStep (eval r a) (fun _ => eval r b)

/-- n-ary AND: `And(c₁,…,cₙ)` — the loop over children, base `true`. -/
def andL : List Expr → Expr
  | [] => .lit (.bool true)
  | c :: cs => .and c (andL cs)

/-- The literal loop of eval.rs:674-699, for comparison with `andL`. -/
def andLoop (r : Row) : List Expr → Bool → Except Err Val
  | [], isNull => .ok (if isNull then .null else .bool true)
  | c :: cs, isNull =>
    match eval r c with
    | .error e => .error e
    | .ok (.bool false) => .ok (.bool false)
    | .ok (.bool true) => andLoop r cs isNull
    | .ok .null => andLoop r cs true
    | .ok _ => .error .typeMismatch

theorem andLoop_shape (r : Row) (cs : List Expr) (n : Bool) (v : Val)
    (h : andLoop r cs n = .ok v) : v = .null ∨ ∃ b, v = .bool b := by
  induction cs generalizing n with
  | nil => cases n <;> simp [andLoop] at h <;> subst h <;> simp
  | cons c cs ih =>
    simp only [andLoop] at h
    split at h
    · cases h
    · cases h; exact Or.inr ⟨false, rfl⟩
    · exact ih _ h
    · exact ih _ h
    · cases h

theorem andLoop_null_absorb (r : Row) (cs : List Expr) :
    andLoop r cs true =
      match andLoop r cs false with
      | .ok (.bool true) => .ok .null
      | x => x := by
  induction cs with
  | nil => rfl
  | cons c cs ih =>
    simp only [andLoop]
    split
    · rfl
    · rfl
    · exact ih
    · rcases hs : andLoop r cs true with e | v
      · rfl
      · rcases andLoop_shape r cs true v hs with rfl | ⟨b, rfl⟩
        · rfl
        · cases b
          · rfl
          · rw [ih] at hs; split at hs <;> simp_all
    · rfl

/-- The fold `andL` is exactly the Rust loop. -/
theorem eval_andL (r : Row) (cs : List Expr) : eval r (andL cs) = andLoop r cs false := by
  induction cs with
  | nil => rfl
  | cons c cs ih =>
    simp only [andL, eval, andStep, andLoop, ih]
    split
    · rfl
    · rfl
    · rfl
    · rw [andLoop_null_absorb]
      rcases hs : andLoop r cs false with e | v
      · rfl
      · rcases andLoop_shape r cs false v hs with rfl | ⟨b, rfl⟩
        · rfl
        · cases b <;> rfl
    · rfl

/-- FilterOp per-row verdict (filter.rs:48-80): `true` keeps, `false`/`null`
drop, anything else is "expected Boolean". -/
def passes (p : Expr) (r : Row) : Except Err Bool :=
  match eval r p with
  | .ok (.bool b) => .ok b
  | .ok .null => .ok false
  | .ok _ => .error .typeMismatch
  | .error e => .error e

def filterRows (p : Expr) : List Row → Except Err (List Row)
  | [] => .ok []
  | r :: rs => do
      let k ← passes p r
      let rest ← filterRows p rs
      pure (if k then r :: rest else rest)

/-- Outcome equivalence: same rows, or both fail (error message may differ). -/
def Same {α} : Except Err α → Except Err α → Prop
  | .ok a, .ok b => a = b
  | .error _, .error _ => True
  | _, _ => False

/-- `p` never errors on the given rows. -/
def NoErr (p : Expr) (xs : List Row) : Prop := ∀ r ∈ xs, ∃ b, passes p r = .ok b

def passB (p : Expr) (r : Row) : Bool :=
  match passes p r with
  | .ok b => b
  | .error _ => false

theorem filterRows_noErr (p : Expr) (xs : List Row) (h : NoErr p xs) :
    filterRows p xs = .ok (xs.filter (passB p)) := by
  induction xs with
  | nil => rfl
  | cons r rs ih =>
    obtain ⟨b, hb⟩ := h r (by simp)
    have ih' := ih (fun r' hr' => h r' (by simp [hr']))
    simp only [filterRows, hb, ih', List.filter, passB]
    cases b <;> rfl

/-! ## Expressions read only their variables -/

theorem eval_congr (e : Expr) (r₁ r₂ : Row) (h : ∀ v ∈ e.vars, r₁.get v = r₂.get v) :
    eval r₁ e = eval r₂ e := by
  induction e with
  | lit => rfl
  | var v => simp [eval, h v (by simp [Expr.vars])]
  | eq a b iha ihb | neq a b iha ihb | lt a b iha ihb | gt a b iha ihb
  | sub a b iha ihb | div a b iha ihb | and a b iha ihb =>
    simp only [Expr.vars, List.mem_append] at h
    simp only [eval]
    rw [iha (fun v hv => h v (Or.inl hv)), ihb (fun v hv => h v (Or.inr hv))]

theorem get_append_left (x y : Row) (v : Var) (h : x.binds v) : (x ++ y).get v = x.get v := by
  unfold Row.get
  rw [List.find?_append]
  have : (x.find? fun p => p.1 == v).isSome := by
    rw [List.find?_isSome]
    simp only [Row.binds, List.mem_map] at h
    obtain ⟨p, hp, rfl⟩ := h
    exact ⟨p, hp, by simp⟩
  cases hx : x.find? (fun p => p.1 == v) with
  | none => simp [hx] at this
  | some p => simp

theorem get_append_right (x y : Row) (v : Var) (h : ¬ x.binds v) : (x ++ y).get v = y.get v := by
  unfold Row.get
  rw [List.find?_append]
  have : x.find? (fun p => p.1 == v) = none := by
    rw [List.find?_eq_none]
    intro p hp hpv
    apply h
    simp only [Row.binds, List.mem_map]
    exact ⟨p, hp, by simpa using hpv⟩
  simp [this]

/-! ## Relational operators -/

def cp (xs ys : List Row) : List Row := xs.flatMap (fun x => ys.map (fun y => x ++ y))

/-- A per-row expanding operator that keeps the input columns in front. -/
def extend (g : Row → List Row) (xs : List Row) : List Row :=
  xs.flatMap (fun x => (g x).map (fun e => x ++ e))

/-- Every variable `p` reads is bound in every row of `xs`. -/
def Covers (p : Expr) (xs : List Row) : Prop := ∀ x ∈ xs, ∀ v ∈ p.vars, x.binds v

theorem passB_append_left (p : Expr) (x y : Row) (h : ∀ v ∈ p.vars, x.binds v) :
    passB p (x ++ y) = passB p x := by
  unfold passB passes
  rw [eval_congr p (x ++ y) x (fun v hv => get_append_left x y v (h v hv))]

theorem passB_append_right (p : Expr) (x y : Row) (h : ∀ v ∈ p.vars, ¬ x.binds v) :
    passB p (x ++ y) = passB p y := by
  unfold passB passes
  rw [eval_congr p (x ++ y) y (fun v hv => get_append_right x y v (h v hv))]

/-- **push_filters_down, CP left child**: a conjunct covered by the left branch
may be evaluated below the product. -/
theorem push_cp_left (p : Expr) (xs ys : List Row) (hc : Covers p xs) :
    (cp xs ys).filter (passB p) = cp (xs.filter (passB p)) ys := by
  unfold cp
  rw [List.filter_flatMap]
  induction xs with
  | nil => rfl
  | cons x xs ih =>
    simp only [List.flatMap_cons, List.filter_cons]
    rw [ih (fun x' hx' => hc x' (by simp [hx']))]
    have hx : ∀ y, passB p (x ++ y) = passB p x :=
      fun y => passB_append_left p x y (hc x (by simp))
    cases h : passB p x
    · simp [List.filter_map, hx, h, Function.comp_def, List.filter_eq_self.mpr]
    · simp [List.filter_map, hx, h, Function.comp_def, List.filter_eq_self.mpr]

/-- **push_filters_down, CP right child**: needs the other branch not to bind
any variable the conjunct reads (rows of distinct branches are disjoint). -/
theorem push_cp_right (p : Expr) (xs ys : List Row)
    (hdis : ∀ x ∈ xs, ∀ v ∈ p.vars, ¬ x.binds v) :
    (cp xs ys).filter (passB p) = cp xs (ys.filter (passB p)) := by
  unfold cp
  rw [List.filter_flatMap]
  induction xs with
  | nil => rfl
  | cons x xs ih =>
    simp only [List.flatMap_cons]
    rw [ih (fun x' hx' => hdis x' (by simp [hx']))]
    congr 1
    rw [List.filter_map]
    congr 1
    apply List.filter_congr
    intro y _
    simp [Function.comp_def, passB_append_right p x y (hdis x (by simp))]

/-- **push_filters_down through a per-row operator** (Unwind, CondTraverse,
Apply, ProcedureCall …): conjunct covered by the input. -/
theorem push_extend (p : Expr) (g : Row → List Row) (xs : List Row) (hc : Covers p xs) :
    (extend g xs).filter (passB p) = extend g (xs.filter (passB p)) := by
  unfold extend
  rw [List.filter_flatMap]
  induction xs with
  | nil => rfl
  | cons x xs ih =>
    simp only [List.flatMap_cons, List.filter_cons]
    rw [ih (fun x' hx' => hc x' (by simp [hx']))]
    have hx : ∀ y, passB p (x ++ y) = passB p x :=
      fun y => passB_append_left p x y (hc x (by simp))
    cases h : passB p x
    · simp [List.filter_map, hx, h, Function.comp_def, List.filter_eq_self.mpr]
    · simp [List.filter_map, hx, h, Function.comp_def, List.filter_eq_self.mpr]

/-- Pushing below `Sort` keeps the multiset … -/
theorem push_sort_perm (p : Expr) (le : Row → Row → Bool) (xs : List Row) :
    ((xs.mergeSort le).filter (passB p)).Perm ((xs.filter (passB p)).mergeSort le) :=
  ((List.mergeSort_perm xs le).filter _).trans (List.mergeSort_perm _ le).symm

/-- … and both sides are sorted, so ORDER BY is still honoured. -/
theorem push_sort_sorted (p : Expr) (le : Row → Row → Bool)
    (trans : ∀ a b c, le a b = true → le b c = true → le a c = true)
    (total : ∀ a b, (le a b || le b a) = true) (xs : List Row) :
    ((xs.mergeSort le).filter (passB p)).Pairwise (fun a b => le a b = true) ∧
    ((xs.filter (passB p)).mergeSort le).Pairwise (fun a b => le a b = true) :=
  ⟨(List.pairwise_mergeSort trans total xs).filter _, List.pairwise_mergeSort trans total _⟩

/-- `Limit` commutes with a filter only when the verdict is constant over the
limited stream (the Apply-argument case: every conjunct reads only variables
the enclosing Apply fixed for this invocation). -/
theorem push_limit_const (p : Expr) (n : Nat) (xs : List Row) (b : Bool)
    (hconst : ∀ x ∈ xs, passB p x = b) :
    (xs.take n).filter (passB p) = (xs.filter (passB p)).take n := by
  have h1 : xs.filter (passB p) = if b then xs else [] := by
    cases b
    · simp only [Bool.false_eq_true, ↓reduceIte]; rw [List.filter_eq_nil_iff]; intro x hx; simp [hconst x hx]
    · simp only [↓reduceIte]; rw [List.filter_eq_self]; intro x hx; simp [hconst x hx]
  have h2 : (xs.take n).filter (passB p) = if b then xs.take n else [] := by
    have hc : ∀ x ∈ xs.take n, passB p x = b := fun x hx => hconst x (List.mem_of_mem_take hx)
    cases b
    · simp only [Bool.false_eq_true, ↓reduceIte]; rw [List.filter_eq_nil_iff]; intro x hx; simp [hc x hx]
    · simp only [↓reduceIte]; rw [List.filter_eq_self]; intro x hx; simp [hc x hx]
  rw [h1, h2]; cases b <;> simp

/-! ### Counterexamples: LIMIT / SKIP are not transparent (push_filters_down.rs:165-178
does not exclude them). -/

def vN : Var := ⟨0, 0⟩
def rowV (i : Int) : Row := [(vN, .int i)]
def gt1 : Expr := .gt (.var vN) (.lit (.int 1))

/-- `WITH n ORDER BY n.v LIMIT 2 MATCH (m) WHERE n.v > 1`: filter-after-limit keeps
one row, filter-before-limit keeps two. -/
theorem limit_not_transparent :
    ([rowV 1, rowV 2, rowV 3].take 2).filter (passB gt1) ≠
    ([rowV 1, rowV 2, rowV 3].filter (passB gt1)).take 2 := by decide

theorem skip_not_transparent :
    ([rowV 1, rowV 5].drop 1).filter (passB (.eq (.var vN) (.lit (.int 5)))) ≠
    ([rowV 1, rowV 5].filter (passB (.eq (.var vN) (.lit (.int 5))))).drop 1 := by decide

/-! ## Filter merging (push_filters_down.rs:113-140)

Stacked `Filter(outer p) → Filter(inner q)` is merged into one `AND`. The Rust code
lists the *outer* conjuncts first (`for f in [&filter, &child_filter]`, line 122).
-/

def stacked (p q : Expr) (xs : List Row) : Except Err (List Row) :=
  filterRows q xs >>= filterRows p

/-- The two orders give different outcomes on a concrete row: the query
`MATCH (n) WHERE n.v <> 1 MATCH (m) WHERE 1/(n.v-1) = 0` — Rust merges to
`AND(1/(n.v-1)=0, n.v<>1)` and fails with division by zero, C returns rows. -/
def qInner : Expr := .neq (.var vN) (.lit (.int 1))
def pOuter : Expr := .eq (.div (.lit (.int 1)) (.sub (.var vN) (.lit (.int 1)))) (.lit (.int 0))

theorem merge_order_rust_errors :
    stacked pOuter qInner [rowV 1, rowV 3] = .ok [rowV 3] ∧
    filterRows (.and pOuter qInner) [rowV 1, rowV 3] = .error .divZero ∧
    filterRows (.and qInner pOuter) [rowV 1, rowV 3] = .ok [rowV 3] := ⟨rfl, rfl, rfl⟩

/-- Per-row agreement of stacked vs child-first AND when the inner verdict is a
proper boolean (not NULL). -/
theorem merge_child_first_row (p q : Expr) (r : Row) (b : Bool) (hq : eval r q = .ok (.bool b)) :
    (do let k ← passes q r; if k then passes p r else pure false) = passes (.and q p) r := by
  unfold passes
  simp only [eval, hq, andStep]
  cases b
  · rfl
  · simp only [bind, Except.bind]
    split <;> simp_all

/-! ## eliminate_true_filters -/

theorem filter_true (xs : List Row) : filterRows (.lit (.bool true)) xs = .ok xs := by
  induction xs with
  | nil => rfl
  | cons r rs ih => simp [filterRows, passes, eval, ih, bind, Except.bind, pure, Except.pure]

/-- Dropping conjuncts that evaluate to `true` on the row leaves the AND loop's
result unchanged (eliminate_true_filters.rs:123-143). -/
theorem andLoop_drop_true (r : Row) (keep : Expr → Bool) (cs : List Expr)
    (htrue : ∀ c ∈ cs, keep c = false → eval r c = .ok (.bool true)) (n : Bool) :
    andLoop r (cs.filter keep) n = andLoop r cs n := by
  induction cs generalizing n with
  | nil => rfl
  | cons c cs ih =>
    have ih' := fun n => ih (fun c' hc' hk => htrue c' (by simp [hc']) hk) n
    cases hk : keep c
    · simp [List.filter, hk, andLoop, htrue c (by simp) hk, ih']
    · simp only [List.filter, hk, andLoop]
      split <;> (try rfl) <;> exact ih' _

/-- `remaining.len() == 1`: `Filter(c)` for `Filter(AND(c))`. -/
theorem passes_and_single (c : Expr) (r : Row) :
    passes (andL [c]) r = passes c r := by
  unfold passes
  simp only [andL, eval]
  rcases eval r c with e | (_ | b | i | s) <;> (try cases b) <;> rfl

theorem filter_and_single (c : Expr) (xs : List Row) :
    filterRows (andL [c]) xs = filterRows c xs := by
  induction xs with
  | nil => rfl
  | cons r rs ih => simp only [filterRows, passes_and_single, ih]

/-- Whole-filter removal is sound under the hypothesis that the plan-time
constant evaluator (`ExprEval::constant`, eval_true_filters.rs:49-72) agrees with
the runtime evaluator. That hypothesis is *false* for `^` (issue #2902). -/
theorem eliminate_whole_filter (c : Expr) (xs : List Row)
    (hconst : ∀ r, eval r c = .ok (.bool true)) : filterRows c xs = .ok xs := by
  induction xs with
  | nil => rfl
  | cons r rs ih =>
    simp [filterRows, passes, hconst r, ih, bind, Except.bind, pure, Except.pure]

end Falkor.Opt
