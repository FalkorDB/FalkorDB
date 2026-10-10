/-!
# Columnar AND/OR agrees with the per-row evaluator

| here | there |
| --- | --- |
| `V`                 | a child's `Value` for one row: `Bool`, `Null`, anything else (type error), or an eval error |
| `St`                | per-row state of `eval_and_or`: `live`, `result[pos]`, `nulls[pos]` — `vector_expr.rs:392-465` |
| `step`              | one child for one row: the `Bools` fast path (`:428-441`) and the generic path (`:445-460`), which are the same function of the row's value |
| `rowEval`           | `ExprIR::And` / `ExprIR::Or` in `ExprEval::eval_compound` — `eval.rs:629-697` (loop with `break` on the settling value) |
| `colEval`           | `VectorEval::eval_and_or` — children outer, rows inner, settled rows dropped from `live` |
| `finish`            | how `ExprColumn::Bools(result, nulls)` reads back as a `Value` |
| `intLane`           | `arithmetic` int lane — `vector_expr.rs:878-915`; `valueArith` is `Add/Sub/Mul/Div/Rem for Value` (`value.rs`) |

`colEval_eq` is the theorem: evaluating child-by-child over all rows, dropping settled rows,
yields exactly the per-row results, and fails iff some row fails (the error surfaces for the
whole batch either way; *which* row's message is reported may differ).
-/
namespace FalkorTemporal.VExpr

inductive V where
  | b (x : Bool)
  | null
  | bad        -- non-boolean, non-null value: "Type mismatch: expected Bool"
  | err        -- the child's own evaluation failed
  deriving DecidableEq

structure St where
  settled : Bool
  null : Bool
  deriving DecidableEq

def St.init : St := ⟨false, false⟩

/-- One child's value applied to one row's state; `sc` is `short_circuit_on`
(`false` for AND, `true` for OR). A settled row is not evaluated (`none` = error). -/
def step (sc : Bool) (s : St) (v : V) : Option St :=
  if s.settled then some s else
  match v with
  | .b x => if x = sc then some ⟨true, false⟩ else some s
  | .null => some ⟨false, true⟩
  | .bad => none
  | .err => none

/-- Per-row evaluator (`eval.rs`): walk the children in order. -/
def rowEval (sc : Bool) : St → List V → Option St
  | s, [] => some s
  | s, v :: vs => step sc s v >>= fun s' => rowEval sc s' vs

/-- The value `ExprColumn::Bools` reads back as. -/
def finish (sc : Bool) (s : St) : V :=
  if s.settled then .b sc else if s.null then .null else .b (!sc)

/-- Columnar evaluator: `k` children; each row is `(state, remaining child values)`.
One child is applied to *all* rows (settled rows untouched), then the next. -/
def colEval (sc : Bool) : Nat → List (St × List V) → Option (List St)
  | 0, xs => some (xs.map Prod.fst)
  | k + 1, xs =>
    xs.mapM (fun p => (step sc p.1 (p.2.headD .null)).map (fun s => (s, p.2.tail))) >>=
      colEval sc k

/-- `rowEval` restricted to the first `k` children (all rows have exactly `k`). -/
def rowK (sc : Bool) : Nat → St × List V → Option St
  | 0, p => some p.1
  | k + 1, p => step sc p.1 (p.2.headD .null) >>= fun s => rowK sc k (s, p.2.tail)

theorem rowK_eq (sc : Bool) : ∀ (vs : List V) (s : St), rowK sc vs.length (s, vs) = rowEval sc s vs
  | [], s => rfl
  | v :: vs, s => by
    simp only [List.length_cons, rowK, List.headD_cons, List.tail_cons, rowEval]
    cases step sc s v with
    | none => rfl
    | some s' => exact rowK_eq sc vs s'

theorem mapM_bind_mapM {α β γ : Type} (f : α → Option β) (g : β → Option γ) :
    ∀ xs : List α, (xs.mapM f >>= fun ys => ys.mapM g) = xs.mapM (fun x => f x >>= g)
  | [] => rfl
  | x :: xs => by
    have ih := mapM_bind_mapM f g xs
    simp only [List.mapM_cons]
    cases hf : f x with
    | none => rfl
    | some y =>
      cases hm : xs.mapM f with
      | none =>
        rw [hm] at ih
        have ih' : List.mapM (fun x => (f x).bind g) xs = none := by simpa using ih.symm
        cases hg : g y <;> simp [hg, ih']
      | some ys =>
        rw [hm] at ih
        have ih' : List.mapM (fun x => (f x).bind g) xs = List.mapM g ys := by simpa using ih.symm
        cases hg : g y <;> simp [hg, ih']

theorem colEval_rows (sc : Bool) : ∀ (k : Nat) (xs : List (St × List V)),
    colEval sc k xs = xs.mapM (rowK sc k)
  | 0, xs => by
    simp only [colEval, rowK]
    induction xs with
    | nil => rfl
    | cons x xs ih => simp only [List.map_cons, List.mapM_cons]; rw [← ih]; rfl
  | k + 1, xs => by
    simp only [colEval]
    rw [show colEval sc k = fun ys => ys.mapM (rowK sc k) from funext (colEval_rows sc k),
      mapM_bind_mapM]
    congr 1; funext p
    simp only [rowK]
    cases step sc p.1 (p.2.headD .null) <;> rfl

/-- **Columnar AND/OR = per-row AND/OR**, for a batch whose rows each carry the values of
the same `k` children: the batch succeeds iff every row succeeds, and then row `i` of the
column is row `i`'s per-row result. -/
theorem colEval_eq (sc : Bool) (k : Nat) (rows : List (List V)) (hk : ∀ r ∈ rows, r.length = k) :
    (colEval sc k (rows.map (fun r => (St.init, r)))).map (List.map (finish sc)) =
    (rows.mapM (fun r => rowEval sc St.init r)).map (List.map (finish sc)) := by
  rw [colEval_rows]
  congr 1
  induction rows with
  | nil => rfl
  | cons r rs ih =>
    simp only [List.map_cons, List.mapM_cons]
    have hr : r.length = k := hk r (by simp)
    have e : rowK sc k (St.init, r) = rowEval sc St.init r := by rw [← hr]; exact rowK_eq sc r _
    rw [e, ih (fun r' h => hk r' (by simp [h]))]

/-- The per-row loop is Kleene's AND (`sc = false`) / OR (`sc = true`) with left-to-right
short-circuit: a `false` (resp. `true`) settles the row and later children — even erroring
ones — are not evaluated. -/
theorem rowEval_short (sc : Bool) (vs : List V) :
    rowEval sc St.init (.b sc :: vs) = some ⟨true, false⟩ := by
  simp only [rowEval, step, St.init, Bool.false_eq_true, ite_false, ite_true]
  show rowEval sc ⟨true, false⟩ vs = _
  induction vs with
  | nil => rfl
  | cons v vs ih => simp only [rowEval, step, ite_true, Option.bind_eq_bind, Option.bind_some]; exact ih

/-! ## Int lane of `arithmetic` agrees with `Value` arithmetic -/

def wrap (x : Int) : Int := (x + 2 ^ 63) % 2 ^ 64 - 2 ^ 63

inductive Op where | add | sub | mul | div | rem

/-- `Value` Int/Int arithmetic (`value.rs` `Add`/`Sub`/`Mul`/`Div`/`Rem`): wrapping, and
`Division by zero` (`none`) for `/` and `%` by 0. `Int.tdiv`/`tmod` are Rust's `/`,`%`. -/
def valueArith : Op → Int → Int → Option Int
  | .add, a, b => some (wrap (a + b))
  | .sub, a, b => some (wrap (a - b))
  | .mul, a, b => some (wrap (a * b))
  | .div, a, b => if b = 0 then none else some (wrap (a.tdiv b))
  | .rem, a, b => if b = 0 then none else some (wrap (a.tmod b))

/-- The columnar int lane over rows `(a, b, isNull)`: enters `Div/Rem` only if no
non-null divisor is 0 (`vector_expr.rs:893`), else falls back to the generic lane
(= `valueArith` per row). Null rows yield null either way. -/
def laneRow (op : Op) (a b : Int) (isNull : Bool) : Option (Option Int) :=
  if isNull then some none else (valueArith op a b).map some

def intLane (op : Op) (rows : List (Int × Int × Bool)) : Option (List (Option Int)) :=
  match op with
  | .div | .rem =>
    if rows.all (fun r => r.2.1 != 0 || r.2.2) then
      some (rows.map fun r => if r.2.2 then none else valueArith op r.1 r.2.1)
    else rows.mapM (fun r => laneRow op r.1 r.2.1 r.2.2)
  | _ => some (rows.map fun r => if r.2.2 then none else valueArith op r.1 r.2.1)

theorem intLane_eq (op : Op) (rows : List (Int × Int × Bool)) :
    intLane op rows = rows.mapM (fun r => laneRow op r.1 r.2.1 r.2.2) := by
  have gen : ∀ rows : List (Int × Int × Bool), (∀ r ∈ rows, r.2.2 = false → (valueArith op r.1 r.2.1).isSome) →
      some (rows.map fun r => if r.2.2 then none else valueArith op r.1 r.2.1) =
      rows.mapM (fun r => laneRow op r.1 r.2.1 r.2.2) := by
    intro rows h
    induction rows with
    | nil => rfl
    | cons r rs ih =>
      simp only [List.map_cons, List.mapM_cons]
      rw [← ih (fun x hx => h x (by simp [hx]))]
      unfold laneRow
      cases hn : r.2.2
      · have := h r (by simp) hn
        cases hv : valueArith op r.1 r.2.1 with
        | none => simp [hv] at this
        | some v => simp [hv]
      · simp
  cases op
  case div | rem =>
    unfold intLane
    by_cases hall : (rows.all fun r => r.2.1 != 0 || r.2.2) = true
    · rw [if_pos hall]
      apply gen; intro r hr hn
      have := List.all_eq_true.mp hall r hr
      simp only [hn, Bool.or_false, bne_iff_ne, ne_eq] at this
      simp [valueArith, this]
    · rw [if_neg hall]
  all_goals (unfold intLane; apply gen; intro r _ _; simp [valueArith])

end FalkorTemporal.VExpr
