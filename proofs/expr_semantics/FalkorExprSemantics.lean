import FalkorExprSemantics.Eval
import FalkorExprSemantics.EvalMore
import FalkorExprSemantics.BFS
/-
# Expression evaluation: the columnar path agrees with the per-row path, and
# where the per-row path itself departs from openCypher / C FalkorDB

A model of FalkorDB-rs scalar expression semantics and of the columnar
evaluator (`VectorEval`) that Filter / Project / Aggregate actually run, with
machine-checked proofs that

* **every typed comparison kernel** (`compare_i64_column`, `compare_f64_column`,
  the promoted mixed int/float lanes, `op.flip` for constant-on-the-left) and the
  generic `compare_values` lane answer exactly what the per-row `eval_node`
  answers — `vec_cmp_*`;
* **the arithmetic lanes** (`Int` wrapping lane with the divide-by-zero gate,
  `f64` lane with the "one operand really is a float" guard, generic lane)
  produce, row for row and error for error, `Value`'s `+ - * / %` — `arith_agrees`;
* **`AND` / `OR` with row narrowing** is the per-row short-circuit Kleene fold
  on every row, and errors exactly when some row's per-row evaluation errors —
  `andOr_agrees`;
* **`CASE`** evaluates each `WHEN`/`THEN`/`ELSE` for exactly the rows the
  per-row path does, and the value form matches on the same terms —
  `case_agrees`, `case_match_agrees`;
* **whole-column comparison** (`compare_columns`, every typed arm + fallback)
  is the per-row comparison — `compareColumns_agrees`;
* `IN`, `AND`, `OR`, `NOT`, `XOR` are openCypher's three-valued logic —
  `contains_is_kleene_any`, `and_kleene`, `or_kleene`, `deMorgan_*`;
* i64 arithmetic is Rust's wrapping arithmetic, and agrees with mathematical
  arithmetic whenever the mathematical result is in range — `wrap_id`, `add_exact`.

and with counterexamples, each reproduced against the real engine
(`graph/tests/lean_expr_semantics.rs`), for

* `^` silently dropping an operand's error (`pow_drops_error`), which also makes
  the optimizer delete `WHERE x ^ true` as constant-true (`pow_constant_true`);
* the lazy `UNWIND range(..)` iterator overflowing after yielding `i64::MAX`
  (`rangeIter_wraps`) — panic in debug, wrong rows and non-termination in release;
* `sign(<float>)` returning a Float — fixed by #2907 (`f04c3557a`), now
  `sign_float_is_int` (historical `pre2907_sign_float_is_float`);
* mixed Int/Float equality not being transitive (`int_float_eq_not_transitive`).

## What is modelled, and where it lives in the tree

| here | there |
| --- | --- |
| `wrap`, `wAdd`/`wSub`/`wMul`/`wDiv`/`wRem` | `i64::wrapping_*` (`Value` ops, `graph/src/runtime/value.rs:904-1175`) |
| `checkedNeg`             | `ExprIR::Negate` (`graph/src/runtime/eval.rs:709`) |
| `FloatModel`             | `f64` — abstract; IEEE comparisons defined from `partial_cmp` |
| `V`                      | `runtime::value::Value` (scalar subset: Null, Bool, Int, Float, String) |
| `V.order`                | `OrderedEnum::order` (`value.rs:1177`) |
| `compareFloats`          | `compare_floats` (`value.rs:1801`) |
| `compareValue`           | `CompareValue::compare_value` (`value.rs:1225`) |
| `vAdd` … `vRem`          | `impl Add/Sub/Mul/Div/Rem for Value` (`value.rs:904-1175`) |
| `evalCmp`                | `all_equals`, `all_not_equals` (`eval.rs:1665,1688`), `Lt/Gt/Le/Ge` (`eval.rs:436-454`, `733-772`) |
| `compareValues`          | `vectorized::compare_values` (`vectorized.rs:239`) |
| `kI64`, `kF64`, `cmpRow` | `compare_i64_column` / `compare_f64_column` (`vectorized.rs:105,162`), `compare_*_pairs` (`vector_expr.rs:795-838`) |
| `CmpOp.flip`             | `CmpOp::flip` (`vectorized.rs:84`) |
| `Col`, `Col.vals`        | `vector_expr::ExprColumn`, `ExprColumn::get` (`vector_expr.rs:64-105`) |
| `typedLane`, `compareColumns` | `compare_columns` (`vector_expr.rs:731`) |
| `arith`, `intLane`, `floatLane`, `floatOperand` | `arithmetic`, `int_lane`, `float_lane`, `float_operand` (`vector_expr.rs:878-985`) |
| `scalarAndOr`            | `ExprIR::Or` / `ExprIR::And` in `eval_compound` (`eval.rs:629`, `674`) |
| `scalarXor`              | `ExprIR::Xor` (`eval.rs:654`) |
| `RowSt`, `stepRow`, `vecAndOr` | `VectorEval::eval_and_or` (`vector_expr.rs:392`) — per-position state |
| `scalarCase`, `caseMatchScalar` | `ExprEval::eval_case` (`eval.rs:485`) |
| `CSt`, `whenPhase`/`thenPhase`/`elsePhase`, `vecCase`, `caseMatchVec` | `VectorEval::eval_case` (`vector_expr.rs:475`) |
| `omap`, `ofold`          | Rust's `?` over a row set / over the children |
| `containsGo`, `contains` | `impl Contains for ThinVec<Value>` (`value.rs:1762`), used by `ExprIR::In` (`eval.rs:774`) |
| `applyPow`, `powFold`    | `apply_pow` (`math.rs:329`); `ExprIR::Pow` in `eval_compound` (`eval.rs:819-825`, the `flat_map`) |
| `pow_constant_true`      | `is_constant_true` (`graph/src/planner/optimizer/eliminate_true_filters.rs:49`) |
| `rangeIterRel` / `rangeIterDbg` | `RangeIter::next` (`eval.rs:108-122`), release / debug build |
| `rangeMath`              | `range()` (`graph/src/runtime/functions/math.rs:251`) |
| `signRust`               | `sign` (`math.rs:210`) |
| `toIntegerFloat` / `toIntegerString` | `tointeger` (`graph/src/runtime/functions/conversion.rs:41-88`) |
| `coalesce`               | `coalesce` (`math.rs:314`) |

## Modelling choices (what this does not cover)

* `f64` is abstract (`FloatModel`): the proofs hold for *any* `partial_cmp`
  and any arithmetic, which is exactly what makes them robust — both paths call
  the same primitive. The one assumption is that Rust's `f64` `<`, `<=`, `==`,
  `!=`, `>`, `>=` are `partial_cmp` read the IEEE way (`!=` is `!(==)`), which is
  what `PartialOrd for f64` documents.
* A typed column's null bitmap is modelled as `Option` per row. Rust keeps a
  placeholder (0 / 0.0) in null rows and masks afterwards; the wrapping ops on a
  placeholder cannot panic and the division gate skips null rows, so this is
  observationally the same.
* `AND`/`OR`: the Rust `result`/`nulls`/`live` triple is represented per position
  by `RowSt`; a child's columnar evaluation over the live rows is assumed to be
  the per-row evaluation mapped over them (that is the induction hypothesis for
  the enclosing tree, and it is what `eval_per_row` does literally). Agreement is
  proven on `Option` (ok/err), not on the error *message*: the columnar path
  reports the first error in child-major order, the per-row path in row-major
  order.
* Lists, maps, temporal values, points and vectors are out of scope (`V.other`).

## Wave 5 (`Eval.lean`, `EvalMore.lean`, `BFS.lean`)

All fns of runtime/eval.rs PROVEN. Highlights: `evalOperand_eq` (inline leaf path = full
evaluator), `evalBatch_eq_perRow` (bulk property path = per-row evaluation),
`classifyJoinKeys_lossless`, `compiledRegex_spec`, `mapProjection_prop`,
`newReversed_typed` (exactly the in-neighbours), and **`bfs_sound`**: whenever the
bidirectional BFS meets, the parent-chain `expect`s never fire and the reconstructed
node sequence is a walk from src to dst over followed edges (invariants `scan_inv`,
`level_inv`, `search_sound`, `chain_sound`, `chain_total`). Minimality of the BFS is argued
in the source comment, not proved here.
-/

set_option linter.unusedSectionVars false

namespace FalkorExpr

/-! ## i64 -/

def I64MIN : Int := -9223372036854775808
def I64MAX : Int := 9223372036854775807

/-- Two's-complement wrap of a mathematical integer into i64. -/
def wrap (x : Int) : Int := (x + 9223372036854775808) % 18446744073709551616 - 9223372036854775808

def InI64 (x : Int) : Prop := I64MIN ≤ x ∧ x ≤ I64MAX

theorem wrap_inRange (x : Int) : InI64 (wrap x) := by
  unfold InI64 wrap I64MIN I64MAX; omega

/-- Wrapping is the identity on in-range results. -/
theorem wrap_id {x : Int} (h : InI64 x) : wrap x = x := by
  unfold InI64 I64MIN I64MAX at h; unfold wrap; omega

def wAdd (a b : Int) : Int := wrap (a + b)
def wSub (a b : Int) : Int := wrap (a - b)
def wMul (a b : Int) : Int := wrap (a * b)
/-- `i64::wrapping_div` — truncating division; only `MIN / -1` wraps. -/
def wDiv (a b : Int) : Int := wrap (Int.tdiv a b)
/-- `i64::wrapping_rem` — truncating remainder. -/
def wRem (a b : Int) : Int := wrap (Int.tmod a b)
/-- `i64::checked_neg`. -/
def checkedNeg (a : Int) : Option Int := if a = I64MIN then none else some (-a)
def checkedAbs (a : Int) : Option Int := if a = I64MIN then none else some (if a < 0 then -a else a)

/-- `+` is exact whenever the mathematical sum is representable. -/
theorem add_exact {a b : Int} (h : InI64 (a + b)) : wAdd a b = a + b := wrap_id h

theorem div_exact {a b : Int} (h : InI64 (Int.tdiv a b)) : wDiv a b = Int.tdiv a b := wrap_id h

/-- `i64::MIN / -1` wraps to `i64::MIN` (C traps / is UB here). -/
theorem min_div_neg_one : wDiv I64MIN (-1) = I64MIN := by decide

/-- `i64::MIN % -1` is `0` (no trap). -/
theorem min_rem_neg_one : wRem I64MIN (-1) = 0 := by decide

/-- `i64::MAX + 1` silently wraps (C does the same; Neo4j raises an overflow). -/
theorem max_add_one : wAdd I64MAX 1 = I64MIN := by decide

/-- Unary minus is *checked* but binary minus *wraps*: `-x` errors on
`i64::MIN` while `0 - x` returns `i64::MIN`. Everywhere else they agree. -/
theorem neg_vs_sub_zero :
    checkedNeg I64MIN = none ∧ wSub 0 I64MIN = I64MIN := by decide

theorem neg_eq_sub_zero {x : Int} (hx : InI64 x) (hne : x ≠ I64MIN) :
    checkedNeg x = some (wSub 0 x) := by
  unfold checkedNeg wSub
  simp only [hne, ↓reduceIte]
  rw [wrap_id]
  · simp
  · unfold InI64 I64MIN I64MAX at *; omega

/-! ## f64, abstractly -/

class FloatModel (F : Type) where
  pcmp : F → F → Option Ordering
  add : F → F → F
  sub : F → F → F
  mul : F → F → F
  div : F → F → F
  rem : F → F → F
  ofInt : Int → F

variable {F : Type} [FM : FloatModel F]

/-- IEEE comparisons, read off `partial_cmp`: `NaN` makes every one false
except `!=`. -/
def fLt (a b : F) : Bool := FM.pcmp a b == some .lt
def fGt (a b : F) : Bool := FM.pcmp a b == some .gt
def fEq (a b : F) : Bool := FM.pcmp a b == some .eq
def fNe (a b : F) : Bool := !(fEq a b)
def fLe (a b : F) : Bool := fLt a b || fEq a b
def fGe (a b : F) : Bool := fGt a b || fEq a b

/-! ## Values and comparison -/

inductive V (F : Type) where
  | null
  | bool (b : Bool)
  | int (i : Int)
  | flt (f : F)
  | str (s : String)
  | other (tag : Nat)   -- list / map / temporal / …: carries only its type order

/-- `OrderedEnum::order` (value.rs:1177). -/
def V.order : V F → Nat
  | .null => 32768 | .bool _ => 4096 | .int _ => 8192 | .flt _ => 16384
  | .str _ => 2048 | .other t => t

inductive Flag where
  | disjoint | comparedNull | nan | none
  deriving DecidableEq, Repr

def cmpNat (a b : Nat) : Ordering := if a < b then .lt else if a = b then .eq else .gt
def cmpInt (a b : Int) : Ordering := if a < b then .lt else if a = b then .eq else .gt
def cmpBool (a b : Bool) : Ordering := cmpNat a.toNat b.toNat

/-- `compare_floats` (value.rs:1801). -/
def compareFloats (a b : F) : Ordering × Flag :=
  match FM.pcmp a b with
  | some o => (o, .none)
  | none => (.lt, .nan)

/-- `compare_value` restricted to scalars (value.rs:1225). Arm order matters:
same-type arms first, then mixed numerics, then null, then disjoint. -/
def compareValue : V F → V F → Ordering × Flag
  | .bool a, .bool b => (cmpBool a b, .none)
  | .flt a, .flt b => compareFloats a b
  | .str a, .str b => (compare a b, .none)
  | .int a, .int b => (cmpInt a b, .none)
  | .int i, .flt f => compareFloats (FM.ofInt i) f
  | .flt f, .int i => compareFloats f (FM.ofInt i)
  | a, b =>
    if (match a, b with | .null, _ => true | _, .null => true | _, _ => false)
    then (cmpNat a.order b.order, .comparedNull)
    else (cmpNat a.order b.order, .disjoint)

@[simp] theorem cv_ii (a b : Int) : compareValue (.int a : V F) (.int b) = (cmpInt a b, .none) := rfl
@[simp] theorem cv_ff (a b : F) : compareValue (.flt a : V F) (.flt b) = compareFloats a b := rfl
@[simp] theorem cv_if (a : Int) (b : F) :
    compareValue (.int a : V F) (.flt b) = compareFloats (FM.ofInt a) b := rfl
@[simp] theorem cv_fi (a : F) (b : Int) :
    compareValue (.flt a : V F) (.int b) = compareFloats a (FM.ofInt b) := rfl

def oEq : Ordering → Bool | .eq => true | _ => false
def oLt : Ordering → Bool | .lt => true | _ => false
def oGt : Ordering → Bool | .gt => true | _ => false

inductive CmpOp where
  | eq | neq | lt | le | gt | ge
  deriving DecidableEq, Repr

def CmpOp.flip : CmpOp → CmpOp
  | .eq => .eq | .neq => .neq | .lt => .gt | .le => .ge | .gt => .lt | .ge => .le

/-- The per-row evaluator (eval.rs): `=` is `all_equals`, `<>` is
`all_not_equals` (`partial_cmp`, `None` only for `ComparedNull`), the orderings
are the `Lt/Gt/Le/Ge` arms. -/
def evalCmp (op : CmpOp) (a b : V F) : V F :=
  let (o, fl) := compareValue a b
  match op with
  | .eq => match fl with
    | .comparedNull => .null
    | .nan | .disjoint => .bool false
    | .none => .bool (oEq o)
  | .neq => match fl with
    | .comparedNull => .null
    | _ => .bool (!oEq o)
  | .lt => match fl with
    | .comparedNull | .disjoint => .null | .nan => .bool false | .none => .bool (oLt o)
  | .gt => match fl with
    | .comparedNull | .disjoint => .null | .nan => .bool false | .none => .bool (oGt o)
  | .le => match fl with
    | .comparedNull | .disjoint => .null | .nan => .bool false | .none => .bool (!oGt o)
  | .ge => match fl with
    | .comparedNull | .disjoint => .null | .nan => .bool false | .none => .bool (!oLt o)

/-- `vectorized::compare_values` (vectorized.rs:239): `none` is null. -/
def compareValues (a b : V F) (op : CmpOp) : Option Bool :=
  let (o, fl) := compareValue a b
  match op with
  | .eq => match fl with
    | .comparedNull => none
    | .nan | .disjoint => some false
    | .none => some (oEq o)
  | .neq => match fl with
    | .comparedNull => none
    | _ => some (!oEq o)
  | _ => match fl with
    | .comparedNull | .disjoint => none
    | .nan => some false
    | .none => some (match op with
        | .lt => oLt o
        | .le => !oGt o
        | .gt => oGt o
        | _ => !oLt o)

def ofOB : Option Bool → V F
  | none => .null
  | some b => .bool b

/-- **Generic comparison lane = per-row evaluator**, for every pair of values
and every operator. -/
theorem vec_cmp_generic (a b : V F) (op : CmpOp) :
    ofOB (compareValues a b op) = evalCmp op a b := by
  unfold compareValues evalCmp
  obtain ⟨o, fl⟩ := compareValue a b
  cases op <;> cases fl <;> rfl

/-- The `i64` kernel for one row (`compare_i64_column` / `compare_i64_pairs`). -/
def kI64 (op : CmpOp) (x t : Int) : Bool :=
  match op with
  | .eq => decide (x = t) | .neq => !decide (x = t) | .lt => decide (x < t) | .le => decide (x ≤ t)
  | .gt => decide (x > t) | .ge => decide (x ≥ t)

/-- **Int lane = per-row evaluator**: an `Int` never compares to `null` or
disjointly with an `Int`, so the kernel's plain bool is the whole answer. -/
theorem vec_cmp_int (op : CmpOp) (x t : Int) :
    evalCmp op (.int x : V F) (.int t) = .bool (kI64 op x t) := by
  rcases Int.lt_trichotomy x t with h | h | h
  · have h1 : x ≠ t := by omega
    have h2 : ¬ t < x := by omega
    have h3 : x ≤ t := by omega
    have h4 : ¬ t ≤ x := by omega
    cases op <;> simp [evalCmp, cmpInt, kI64, h, h1, h2, h3, h4] <;> simp [oEq, oLt, oGt]
  · subst h; cases op <;> simp [evalCmp, cmpInt, kI64, oEq, oLt, oGt]
  · have h1 : x ≠ t := by omega
    have h2 : ¬ x < t := by omega
    have h3 : t ≤ x := by omega
    have h4 : ¬ x ≤ t := by omega
    cases op <;> simp [evalCmp, cmpInt, kI64, h, h1, h2, h3, h4] <;> simp [oEq, oLt, oGt]

/-- **Constant on the left** (`(Scalar(Int t), Ints)` arm calls the kernel with
`op.flip`): `t op x` is `x (flip op) t`. -/
theorem vec_cmp_int_flip (op : CmpOp) (x t : Int) :
    evalCmp op (.int t : V F) (.int x) = .bool (kI64 op.flip x t) := by
  rw [vec_cmp_int]
  cases op <;> simp only [kI64, CmpOp.flip, V.bool.injEq] <;> by_cases hxt : x = t <;> simp [hxt, eq_comm] <;> omega

/-- The `f64` kernel for one row (`compare_f64_column` / `compare_f64_pairs`). -/
def kF64 (op : CmpOp) (x t : F) : Bool :=
  match op with
  | .eq => fEq x t | .neq => fNe x t | .lt => fLt x t | .le => fLe x t
  | .gt => fGt x t | .ge => fGe x t

/-- **Float lane = per-row evaluator**, `NaN` included: every kernel answer is
the per-row answer (`NaN <> x` is `true` on both paths, every other `NaN`
comparison `false`). -/
theorem vec_cmp_float (op : CmpOp) (x t : F) :
    evalCmp op (.flt x : V F) (.flt t) = .bool (kF64 op x t) := by
  simp only [evalCmp, cv_ff, compareFloats, kF64, fEq, fNe, fLt, fLe, fGt, fGe]
  cases h : FM.pcmp x t with
  | none => cases op <;> rfl
  | some o => cases op <;> cases o <;> rfl

/-- Constant on the left for floats. This one needs the IEEE law that
`partial_cmp` is antisymmetric (`b.partial_cmp(a) == a.partial_cmp(b).map(reverse)`),
which Rust guarantees for `f64`; it is taken as a hypothesis, not an axiom. -/
theorem vec_cmp_float_flip (op : CmpOp) (x t : F)
    (swap : FM.pcmp t x = (FM.pcmp x t).map Ordering.swap) :
    evalCmp op (.flt t : V F) (.flt x) = .bool (kF64 op.flip x t) := by
  simp only [evalCmp, cv_ff, compareFloats, kF64, fEq, fNe, fLt, fLe, fGt, fGe, CmpOp.flip]
  rw [swap]
  cases h : FM.pcmp x t with
  | none => cases op <;> rfl
  | some o => cases op <;> cases o <;> rfl

/-! ## Arithmetic: `Value` operators and the columnar lanes -/

inductive AOp where
  | add | sub | mul | div | rem
  deriving DecidableEq, Repr

def AOp.int : AOp → Int → Int → Int
  | .add => wAdd | .sub => wSub | .mul => wMul | .div => wDiv | .rem => wRem

def AOp.flt : AOp → F → F → F
  | .add => FM.add | .sub => FM.sub | .mul => FM.mul | .div => FM.div | .rem => FM.rem

def AOp.isDiv : AOp → Bool
  | .div | .rem => true | _ => false

def typeMismatch : String := "Type mismatch: expected Integer, Float, or Null"

/-- `impl Add/Sub/Mul/Div/Rem for Value` on scalars (value.rs:904-1175). `+`
tests `Int/Float` before `Null` and the others test `Null` first; for these
shapes the order is unobservable. Strings concatenate under `+`. -/
def vArith (op : AOp) : V F → V F → Except String (V F)
  | .null, _ => .ok .null
  | _, .null => .ok .null
  | .int a, .int b =>
    if op.isDiv && b == 0 then .error "Division by zero" else .ok (.int (op.int a b))
  | .flt a, .flt b => .ok (.flt (op.flt a b))
  | .flt a, .int b => .ok (.flt (op.flt a (FM.ofInt b)))
  | .int a, .flt b => .ok (.flt (op.flt (FM.ofInt a) b))
  | .str a, .str b => if op == .add then .ok (.str (a ++ b)) else .error typeMismatch
  | _, _ => .error typeMismatch

/-- A numeric cell of a typed lane: `Int` or `Float`. -/
inductive Cell (F : Type) where
  | i (x : Int)
  | f (x : F)

def Cell.v : Cell F → V F
  | .i x => .int x
  | .f x => .flt x

def Cell.toF : Cell F → F
  | .i x => FM.ofInt x
  | .f x => x

def optV {α : Type} (g : α → V F) : Option α → V F
  | none => .null
  | some x => g x

/-- `ExprColumn` (vector_expr.rs:64). A typed lane's null bitmap is the `Option`. -/
inductive Col (F : Type) where
  | scalar (v : V F)
  | ints (xs : List (Option Int))
  | floats (xs : List (Option F))
  | values (xs : List (V F))

/-- `ExprColumn::get` for every row / `into_values`. -/
def Col.vals (len : Nat) : Col F → List (V F)
  | .scalar v => List.replicate len v
  | .ints xs => xs.map (optV .int)
  | .floats xs => xs.map (optV .flt)
  | .values xs => xs

def Col.WF (len : Nat) : Col F → Prop
  | .scalar _ => True
  | .ints xs => xs.length = len
  | .floats xs => xs.length = len
  | .values xs => xs.length = len

/-- `int_lane` (vector_expr.rs:953). -/
def intLane (len : Nat) : Col F → Option (List (Option Int))
  | .ints xs => some xs
  | .scalar (.int v) => some (List.replicate len (some v))
  | _ => none

/-- `float_lane` (vector_expr.rs:965), keeping which cells were promoted. -/
def floatLane (len : Nat) : Col F → Option (List (Option (Cell F)))
  | .floats xs => some (xs.map (Option.map Cell.f))
  | .ints xs => some (xs.map (Option.map Cell.i))
  | .scalar (.flt v) => some (List.replicate len (some (Cell.f v)))
  | .scalar (.int v) => some (List.replicate len (some (Cell.i v)))
  | _ => none

/-- `float_operand` (vector_expr.rs:980). -/
def floatOperand : Col F → Bool
  | .floats _ => true
  | .scalar (.flt _) => true
  | _ => false

def intRow (op : AOp) (x y : Option Int) : Option Int := do
  let x ← x; let y ← y; pure (op.int x y)

def fltRow (op : AOp) (x y : Option (Cell F)) : Option F := do
  let x ← x; let y ← y; pure (op.flt x.toF y.toF)

/-- The divide-by-zero gate: no non-null row divides by zero. -/
def divGate (a b : List (Option Int)) : Bool :=
  (List.zip a b).all fun p => match p with
    | (some _, some 0) => false
    | _ => true

def generic (op : AOp) (len : Nat) (l r : Col F) : Except String (Col F) :=
  ((List.zip (l.vals len) (r.vals len)).mapM fun p => vArith op p.1 p.2).map Col.values

/-- `arithmetic` (vector_expr.rs:878), lane by lane in the Rust order. -/
def arith (op : AOp) (len : Nat) (l r : Col F) : Except String (Col F) :=
  match intLane len l, intLane len r with
  | some a, some b =>
    if !op.isDiv || divGate a b then .ok (.ints (List.zipWith (intRow op) a b))
    else generic op len l r
  | _, _ =>
    if floatOperand l || floatOperand r then
      match floatLane len l, floatLane len r with
      | some a, some b => .ok (.floats (List.zipWith (fltRow op) a b))
      | _, _ => generic op len l r
    else generic op len l r

/-- The per-row reference: `Value`'s operator applied to every row. -/
def perRow (op : AOp) (len : Nat) (l r : Col F) : Except String (List (V F)) :=
  (List.zip (l.vals len) (r.vals len)).mapM fun p => vArith op p.1 p.2

theorem zip_mapM {α β γ : Type} (op : AOp) (vx : α → V F) (vy : β → V F) (vo : γ → V F)
    (f : α → β → γ) (P : α → β → Prop)
    (hrow : ∀ x y, P x y → vArith op (vx x) (vy y) = .ok (vo (f x y))) :
    ∀ (a : List α) (b : List β), a.length = b.length → (∀ p ∈ List.zip a b, P p.1 p.2) →
      (List.zip (a.map vx) (b.map vy)).mapM (fun p => vArith op p.1 p.2)
        = .ok ((List.zipWith f a b).map vo)
  | [], [], _, _ => rfl
  | x :: a, y :: b, hl, hP => by
    have h1 : P x y := hP (x, y) (by simp)
    have h2 : ∀ p ∈ List.zip a b, P p.1 p.2 := fun p hp => hP p (by simp [hp])
    have ih := zip_mapM op vx vy vo f P hrow a b (by simpa using hl) h2
    simp only [List.map_cons, List.zip_cons_cons, List.mapM_cons, List.zipWith_cons_cons]
    rw [hrow x y h1, ih]; rfl
  | [], _ :: _, hl, _ => by simp at hl
  | _ :: _, [], hl, _ => by simp at hl

theorem intLane_vals {len : Nat} {c : Col F} {a : List (Option Int)}
    (h : intLane len c = some a) (wf : c.WF len) :
    c.vals len = a.map (optV .int) ∧ a.length = len := by
  cases c with
  | scalar v =>
    cases v <;> simp [intLane] at h
    subst h; simp [Col.vals, optV, List.map_replicate]
  | ints xs => simp [intLane] at h; subst h; exact ⟨rfl, wf⟩
  | floats _ => simp [intLane] at h
  | values _ => simp [intLane] at h

theorem floatLane_vals {len : Nat} {c : Col F} {a : List (Option (Cell F))}
    (h : floatLane len c = some a) (wf : c.WF len) :
    c.vals len = a.map (optV Cell.v) ∧ a.length = len := by
  cases c with
  | scalar v =>
    cases v <;> simp [floatLane] at h <;> subst h <;>
      simp [Col.vals, optV, List.map_replicate, Cell.v]
  | ints xs =>
    simp [floatLane] at h; subst h
    refine ⟨?_, by simpa [Col.WF] using wf⟩
    simp only [Col.vals, List.map_map]; congr 1; funext o; cases o <;> rfl
  | floats xs =>
    simp [floatLane] at h; subst h
    refine ⟨?_, by simpa [Col.WF] using wf⟩
    simp only [Col.vals, List.map_map]; congr 1; funext o; cases o <;> rfl
  | values _ => simp [floatLane] at h

/-- A float operand's lane holds only `Float` cells (or nulls). -/
def AllF : List (Option (Cell F)) → Prop
  | xs => ∀ c ∈ xs, ∀ z, c = some (Cell.i z) → False

theorem floatOperand_allF {len : Nat} {c : Col F} {a : List (Option (Cell F))}
    (hf : floatOperand c = true) (h : floatLane len c = some a) : AllF a := by
  intro x hx z hz
  cases c with
  | scalar v =>
    cases v <;> simp [floatOperand] at hf
    simp [floatLane] at h; subst h
    have := List.eq_of_mem_replicate hx; subst this; simp at hz
  | floats xs =>
    simp [floatLane] at h; subst h
    simp at hx; obtain ⟨o, _, ho⟩ := hx; subst hz; cases o <;> simp at ho
  | ints _ => simp [floatOperand] at hf
  | values _ => simp [floatOperand] at hf

def mixedOK (x y : Option (Cell F)) : Prop :=
  ∀ p q, x = some p → y = some q → (∃ u, p = Cell.f u) ∨ (∃ u, q = Cell.f u)

theorem intRow_ok (op : AOp) (x y : Option Int)
    (hg : op.isDiv = true → ∀ a, x = some a → y ≠ some 0) :
    vArith op (optV .int x : V F) (optV .int y) = .ok (optV .int (intRow op x y)) := by
  cases x with
  | none => cases y <;> rfl
  | some a =>
    cases y with
    | none => rfl
    | some b =>
      simp only [optV, vArith, intRow]
      by_cases hd : op.isDiv = true
      · have hb : b ≠ 0 := fun hb => hg hd a rfl (by rw [hb])
        simp [hd, hb]
      · have : op.isDiv = false := by simpa using hd
        simp [this]

theorem fltRow_ok (op : AOp) (x y : Option (Cell F)) (hm : mixedOK x y) :
    vArith op (optV Cell.v x : V F) (optV Cell.v y) = .ok (optV .flt (fltRow op x y)) := by
  cases x with
  | none => cases y <;> rfl
  | some p =>
    cases y with
    | none => cases p <;> rfl
    | some q =>
      cases p with
      | i a =>
        cases q with
        | i b =>
          rcases hm _ _ rfl rfl with ⟨u, hu⟩ | ⟨u, hu⟩ <;> cases hu
        | f b => rfl
      | f a => cases q <;> rfl

theorem divGate_spec (a b : List (Option Int)) (hg : divGate a b = true) :
    ∀ p ∈ List.zip a b, ∀ z, p.1 = some z → p.2 ≠ some 0 := by
  intro p hp z hz h0
  unfold divGate at hg
  have := List.all_eq_true.mp hg p hp
  obtain ⟨p1, p2⟩ := p
  simp at hz h0; subst hz; subst h0; simp at this

/-- **The columnar arithmetic is the per-row `Value` arithmetic**: same rows,
same values, and an error exactly when (and with exactly the message) the
per-row evaluation errors. -/
theorem arith_agrees (op : AOp) (len : Nat) (l r : Col F) (wl : l.WF len) (wr : r.WF len) :
    (arith op len l r).map (Col.vals len) = perRow op len l r := by
  have gen : (generic op len l r).map (Col.vals len) = perRow op len l r := by
    unfold generic perRow
    cases (List.zip (l.vals len) (r.vals len)).mapM (fun p => vArith op p.1 p.2) <;> rfl
  unfold arith
  cases hl : intLane len l with
  | some a =>
    cases hr : intLane len r with
    | some b =>
      obtain ⟨vl, la⟩ := intLane_vals hl wl
      obtain ⟨vr, lb⟩ := intLane_vals hr wr
      simp only
      by_cases hc : (!op.isDiv || divGate a b) = true
      · simp only [hc, ↓reduceIte]
        show Except.ok ((List.zipWith (intRow op) a b).map (optV .int)) = _
        unfold perRow; rw [vl, vr]
        refine (zip_mapM op (optV .int) (optV .int) (optV .int) (intRow op)
          (fun x y => op.isDiv = true → ∀ z, x = some z → y ≠ some 0)
          (fun x y h => intRow_ok op x y h) a b (by omega) ?_).symm
        intro p hp hd z hz
        have : divGate a b = true := by simpa [hd] using hc
        exact divGate_spec a b this p hp z hz
      · simp only [hc, ↓reduceIte, Bool.false_eq_true]; exact gen
    | none =>
      simp only
      split
      · rename_i hfo
        cases fl : floatLane len l with
        | none => simp only; exact gen
        | some fa =>
          cases fr : floatLane len r with
          | none => simp only; exact gen
          | some fb =>
            simp only
            obtain ⟨vl, la⟩ := floatLane_vals fl wl
            obtain ⟨vr, lb⟩ := floatLane_vals fr wr
            show Except.ok ((List.zipWith (fltRow op) fa fb).map (optV .flt)) = _
            unfold perRow; rw [vl, vr]
            refine (zip_mapM op (optV Cell.v) (optV Cell.v) (optV .flt) (fltRow op)
              mixedOK (fun x y h => fltRow_ok op x y h) fa fb (by omega) ?_).symm
            intro p hp u w hu hw
            rcases Bool.or_eq_true_iff.mp hfo with h | h
            · have := floatOperand_allF h fl
              cases u with
              | f u => exact Or.inl ⟨u, rfl⟩
              | i z => exact absurd (this p.1 (List.of_mem_zip hp).1 z hu) id
            · have := floatOperand_allF h fr
              cases w with
              | f w => exact Or.inr ⟨w, rfl⟩
              | i z => exact absurd (this p.2 (List.of_mem_zip hp).2 z hw) id
      · exact gen
  | none =>
    simp only
    split
    · rename_i hfo
      cases fl : floatLane len l with
      | none => simp only; exact gen
      | some fa =>
        cases fr : floatLane len r with
        | none => simp only; exact gen
        | some fb =>
          simp only
          obtain ⟨vl, la⟩ := floatLane_vals fl wl
          obtain ⟨vr, lb⟩ := floatLane_vals fr wr
          show Except.ok ((List.zipWith (fltRow op) fa fb).map (optV .flt)) = _
          unfold perRow; rw [vl, vr]
          refine (zip_mapM op (optV Cell.v) (optV Cell.v) (optV .flt) (fltRow op)
            mixedOK (fun x y h => fltRow_ok op x y h) fa fb (by omega) ?_).symm
          intro p hp u w hu hw
          rcases Bool.or_eq_true_iff.mp hfo with h | h
          · have := floatOperand_allF h fl
            cases u with
            | f u => exact Or.inl ⟨u, rfl⟩
            | i z => exact absurd (this p.1 (List.of_mem_zip hp).1 z hu) id
          · have := floatOperand_allF h fr
            cases w with
            | f w => exact Or.inr ⟨w, rfl⟩
            | i z => exact absurd (this p.2 (List.of_mem_zip hp).2 z hw) id
    · exact gen

/-! ## Comparisons over whole columns (`compare_columns`) -/

theorem evalCmp_null_l (op : CmpOp) (v : V F) : evalCmp op .null v = .null := by
  cases v <;> cases op <;> rfl

theorem evalCmp_null_r (op : CmpOp) (v : V F) : evalCmp op v .null = .null := by
  cases v <;> cases op <;> rfl

theorem evalCmp_if (op : CmpOp) (x : Int) (t : F) :
    evalCmp op (.int x : V F) (.flt t) = evalCmp op (.flt (FM.ofInt x)) (.flt t) := by
  simp only [evalCmp, cv_if, cv_ff]

theorem evalCmp_fi (op : CmpOp) (x : F) (t : Int) :
    evalCmp op (.flt x : V F) (.int t) = evalCmp op (.flt x) (.flt (FM.ofInt t)) := by
  simp only [evalCmp, cv_fi, cv_ff]

/-- One kernel row: a null on either side is null (the bitmap), otherwise the
kernel's bool. -/
def cmpRow {α β : Type} (k : α → β → Bool) : Option α → Option β → Option Bool
  | some x, some y => some (k x y)
  | _, _ => none

/-- `compare_columns`' typed arms (vector_expr.rs:737-765); `none` is the
`_ => None` fallthrough to the generic lane. The `Scalar` operand of a kernel
is the threshold broadcast to every row. -/
def typedLane (op : CmpOp) (len : Nat) : Col F → Col F → Option (List (Option Bool))
  | .ints a, .scalar (.int t) =>
    some (List.zipWith (cmpRow (kI64 op)) a (List.replicate len (some t)))
  | .scalar (.int t), .ints a =>
    some (List.zipWith (cmpRow (kI64 op.flip)) a (List.replicate len (some t)))
  | .floats a, .scalar (.flt t) =>
    some (List.zipWith (cmpRow (kF64 op)) a (List.replicate len (some t)))
  | .scalar (.flt t), .floats a =>
    some (List.zipWith (cmpRow (kF64 op.flip)) a (List.replicate len (some t)))
  | .ints a, .scalar (.flt t) =>
    some (List.zipWith (cmpRow (kF64 op)) (a.map (Option.map FM.ofInt)) (List.replicate len (some t)))
  | .floats a, .scalar (.int t) =>
    some (List.zipWith (cmpRow (kF64 op)) a (List.replicate len (some (FM.ofInt t))))
  | .ints a, .ints b => some (List.zipWith (cmpRow (kI64 op)) a b)
  | .floats a, .floats b => some (List.zipWith (cmpRow (kF64 op)) a b)
  | _, _ => none

def compareColumns (op : CmpOp) (len : Nat) (l r : Col F) : List (Option Bool) :=
  match typedLane op len l r with
  | some m => m
  | none => List.zipWith (fun a b => compareValues a b op) (l.vals len) (r.vals len)

theorem rows_agree {α β : Type} (e : V F → V F → V F)
    (hnl : ∀ v, e .null v = .null) (hnr : ∀ v, e v .null = .null)
    (gx : α → V F) (gy : β → V F) (k : α → β → Bool)
    (hk : ∀ x y, e (gx x) (gy y) = .bool (k x y)) :
    ∀ (a : List (Option α)) (b : List (Option β)),
      (List.zipWith (cmpRow k) a b).map ofOB
        = List.zipWith e (a.map (optV gx)) (b.map (optV gy))
  | [], _ => by simp
  | _ :: _, [] => by simp
  | x :: a, y :: b => by
    simp only [List.zipWith_cons_cons, List.map_cons]
    rw [rows_agree e hnl hnr gx gy k hk a b]
    congr 1
    cases x <;> cases y <;> simp [cmpRow, ofOB, optV, hk, hnl, hnr]

theorem cmpRow_map_left {α α' β : Type} (k : α' → β → Bool) (g : α → α') :
    (fun x y => cmpRow k (Option.map g x) y) = cmpRow (fun x y => k (g x) y) := by
  funext x y; cases x <;> cases y <;> rfl

theorem cmpRow_map_right {α β β' : Type} (k : α → β' → Bool) (g : β → β') :
    (fun x y => cmpRow k x (Option.map g y)) = cmpRow (fun x y => k x (g y)) := by
  funext x y; cases x <;> cases y <;> rfl

theorem repl_map {α : Type} (len : Nat) (g : α → V F) (t : α) :
    List.replicate len (g t) = (List.replicate len (some t)).map (optV g) := by
  simp [List.map_replicate, optV]

/-- The mask a filter reads, row for row, `null` included. -/
theorem zipWith_swap {α β γ : Type} (f : α → β → γ) :
    ∀ (a : List α) (b : List β), List.zipWith f a b = List.zipWith (fun y x => f x y) b a
  | [], [] => rfl
  | [], _ :: _ => rfl
  | _ :: _, [] => rfl
  | x :: a, y :: b => by simp [zipWith_swap f a b]

/-- **`compare_columns` = the per-row comparison**, for every column shape the
typed lanes enter and for the generic lane. The float lane with the constant on
the left needs `partial_cmp`'s antisymmetry (`swap`). -/
theorem compareColumns_agrees (op : CmpOp) (len : Nat) (l r : Col F)
    (swap : ∀ x t : F, FM.pcmp t x = (FM.pcmp x t).map Ordering.swap) :
    (compareColumns op len l r).map ofOB = List.zipWith (evalCmp op) (l.vals len) (r.vals len) := by
  unfold compareColumns
  have nl := evalCmp_null_l (F := F) op
  have nr := evalCmp_null_r (F := F) op
  have snl : ∀ v : V F, (fun y x => evalCmp op x y) .null v = .null := fun v => nr v
  have snr : ∀ v : V F, (fun y x => evalCmp op x y) v .null = .null := fun v => nl v
  split
  · rename_i m hm
    unfold typedLane at hm
    split at hm <;> (try simp at hm) <;> subst hm
    · simp only [Col.vals]; rw [repl_map len .int]
      exact rows_agree _ nl nr (.int) (.int) (kI64 op) (vec_cmp_int op) _ _
    · simp only [Col.vals]; rw [repl_map len .int, zipWith_swap (evalCmp op)]
      exact rows_agree _ snl snr (.int) (.int) (kI64 op.flip)
        (fun x t => vec_cmp_int_flip op x t) _ _
    · simp only [Col.vals]; rw [repl_map len .flt]
      exact rows_agree _ nl nr (.flt) (.flt) (kF64 op) (vec_cmp_float op) _ _
    · simp only [Col.vals]; rw [repl_map len .flt, zipWith_swap (evalCmp op)]
      exact rows_agree _ snl snr (.flt) (.flt) (kF64 op.flip)
        (fun x t => vec_cmp_float_flip op x t (swap x t)) _ _
    · simp only [Col.vals]; rw [repl_map len .flt, List.zipWith_map_left, cmpRow_map_left]
      exact rows_agree _ nl nr (.int) (.flt) (fun x t => kF64 op (FM.ofInt x) t)
        (fun x t => by rw [evalCmp_if, vec_cmp_float]) _ _
    · simp only [Col.vals]; rw [repl_map len .int]
      have : ∀ (n : Nat) (u : Int), List.replicate n (some (FM.ofInt u))
          = (List.replicate n (some u)).map (Option.map FM.ofInt) := fun n u => by
        simp [List.map_replicate]
      rw [this, List.zipWith_map_right, cmpRow_map_right]
      exact rows_agree _ nl nr (.flt) (.int) (fun x t => kF64 op x (FM.ofInt t))
        (fun x t => by rw [evalCmp_fi, vec_cmp_float]) _ _
    · exact rows_agree _ nl nr (.int) (.int) (kI64 op) (vec_cmp_int op) _ _
    · exact rows_agree _ nl nr (.flt) (.flt) (kF64 op) (vec_cmp_float op) _ _
  · rw [List.map_zipWith]
    congr 1; funext a b; exact vec_cmp_generic a b op

/-! ## `AND` / `OR`: three-valued logic, short-circuit, and row narrowing -/

/-- `?` over a list: the first `none` wins. -/
def omap {α β : Type} (f : α → Option β) : List α → Option (List β)
  | [] => some []
  | x :: xs => match f x with
    | none => none
    | some y => match omap f xs with
      | none => none
      | some ys => some (y :: ys)

/-- A fold that stops at the first `none`. -/
def ofold {σ γ : Type} (f : γ → σ → Option σ) : List γ → σ → Option σ
  | [], s => some s
  | c :: cs, s => match f c s with
    | none => none
    | some s' => ofold f cs s'

theorem omap_some {α : Type} : ∀ xs : List α, omap some xs = some xs
  | [] => rfl
  | x :: xs => by simp [omap, omap_some xs]

theorem omap_bind {α β γ : Type} (F : α → Option β) (G : β → Option γ) :
    ∀ xs : List α,
      (match omap F xs with | none => none | some ys => omap G ys)
        = omap (fun x => match F x with | none => none | some y => G y) xs
  | [] => rfl
  | x :: xs => by
    have ih := omap_bind F G xs
    simp only [omap]
    cases hF : F x with
    | none => rfl
    | some y =>
      simp only
      cases hFs : omap F xs with
      | none =>
        rw [hFs] at ih; simp only at ih; rw [← ih]
        cases G y <;> rfl
      | some ys =>
        rw [hFs] at ih; simp only at ih
        simp only [omap]
        cases hG : G y with
        | none => rfl
        | some z => rw [← ih]

/-- **Traversing rows and folding children commute** (for `?`-propagation):
columnar = "for each child, all live rows"; per-row = "for each row, all
children". Both succeed iff every step succeeds, and then agree. -/
theorem fold_omap_comm {α σ γ : Type} (f : γ → α → σ → Option σ) :
    ∀ (cs : List γ) (xs : List (α × σ)),
      ofold (fun c ys => omap (fun p => (f c p.1 p.2).map (p.1, ·)) ys) cs xs
        = omap (fun p => (ofold (fun c s => f c p.1 s) cs p.2).map (p.1, ·)) xs
  | [], xs => by
    simp only [ofold, Option.map_some]
    exact (omap_some xs).symm
  | c :: cs, xs => by
    simp only [ofold]
    have ih := fold_omap_comm f cs
    cases h : omap (fun p => (f c p.1 p.2).map (p.1, ·)) xs with
    | none =>
      -- some row fails at child `c`: find it on the right-hand side too
      simp only
      symm
      have key := omap_bind (fun p => (f c p.1 p.2).map (p.1, ·))
        (fun p => (ofold (fun c s => f c p.1 s) cs p.2).map (p.1, ·)) xs
      rw [h] at key; simp only at key
      rw [key]; congr 1; funext p
      cases hp : f c p.1 p.2 <;> simp
    | some ys =>
      simp only
      rw [ih ys]
      have key := omap_bind (fun p => (f c p.1 p.2).map (p.1, ·))
        (fun p => (ofold (fun c s => f c p.1 s) cs p.2).map (p.1, ·)) xs
      rw [h] at key; simp only at key
      rw [key]; congr 1; funext p
      cases hp : f c p.1 p.2 <;> simp

variable {R : Type}

def boolMismatch : String := "Type mismatch: expected Bool"

/-- Per-row `AND` (`sc = false`) / `OR` (`sc = true`) of `eval_compound`
(eval.rs:629-700): children left to right, stop at the first child equal to
`sc`, remember a `null`, reject any other type. -/
def scalarAndOr (sc : Bool) (r : R) : List (R → Except String (V F)) → Bool → Except String (V F)
  | [], isNull => .ok (if isNull then .null else .bool (!sc))
  | c :: cs, isNull => match c r with
    | .error e => .error e
    | .ok (.bool b) => if b == sc then .ok (.bool sc) else scalarAndOr sc r cs isNull
    | .ok .null => scalarAndOr sc r cs true
    | .ok _ => .error boolMismatch

/-- One position of `eval_and_or`'s `result` / `nulls` / `live` triple:
still live (with its null flag), or settled. -/
inductive RowSt where
  | live (isNull : Bool)
  | done

/-- What one child does to one position (vector_expr.rs:421-454). A settled
position is not in `live`, so the child is not evaluated for it. -/
def stepRow (sc : Bool) (c : R → Except String (V F)) (r : R) : RowSt → Option RowSt
  | .done => some .done
  | .live n => match c r with
    | .error _ => none
    | .ok (.bool b) => if b == sc then some .done else some (.live n)
    | .ok .null => some (.live true)
    | .ok _ => none

/-- `ExprColumn::Bools(result, nulls)` read back per position. -/
def finalV (sc : Bool) : RowSt → V F
  | .done => .bool sc
  | .live n => if n then .null else .bool (!sc)

/-- `eval_and_or`: children in order, each over the rows still live. -/
def vecAndOr (sc : Bool) (cs : List (R → Except String (V F))) (rows : List R) :
    Option (List (V F)) :=
  (ofold (fun c ys => omap (fun p => (stepRow sc c p.1 p.2).map (p.1, ·)) ys) cs
      (rows.map (·, RowSt.live false))).map (fun ys => ys.map (fun p => finalV sc p.2))

/-- `stepRow` with the row fixed, argument order for `ofold`. -/
def rstep (sc : Bool) (r : R) (c : R → Except String (V F)) (s : RowSt) : Option RowSt :=
  stepRow sc c r s

theorem rowFold_done (sc : Bool) (r : R) :
    ∀ cs : List (R → Except String (V F)), ofold (rstep sc r) cs .done = some .done
  | [] => rfl
  | _ :: cs => by simp only [ofold, rstep, stepRow]; exact rowFold_done sc r cs

theorem rowFold_scalar (sc : Bool) (r : R) :
    ∀ (cs : List (R → Except String (V F))) (n : Bool),
      (scalarAndOr sc r cs n).toOption = (ofold (rstep sc r) cs (.live n)).map (finalV sc)
  | [], n => by cases n <;> rfl
  | c :: cs, n => by
    simp only [scalarAndOr, ofold, rstep, stepRow]
    cases h : c r with
    | error e => rfl
    | ok v =>
      cases v with
      | bool b =>
        by_cases hb : (b == sc) = true
        · simp only [hb, ↓reduceIte]; rw [rowFold_done]; rfl
        · simp only [hb]; exact rowFold_scalar sc r cs n
      | null => exact rowFold_scalar sc r cs true
      | int _ => rfl
      | flt _ => rfl
      | str _ => rfl
      | other _ => rfl

theorem omap_map {α β γ : Type} (f : β → Option γ) (g : α → β) :
    ∀ xs : List α, omap f (xs.map g) = omap (fun x => f (g x)) xs
  | [] => rfl
  | x :: xs => by simp [omap, omap_map f g xs]

theorem omap_post {α β γ : Type} (f : α → Option β) (g : β → γ) :
    ∀ xs : List α, (omap f xs).map (·.map g) = omap (fun x => (f x).map g) xs
  | [] => rfl
  | x :: xs => by
    simp only [omap]
    cases f x with
    | none => rfl
    | some y =>
      have ih := omap_post f g xs
      cases h : omap f xs with
      | none => rw [h] at ih; simp at ih; rw [← ih]; rfl
      | some ys => rw [h] at ih; simp at ih; simp [← ih]

/-- **`AND`/`OR` with row narrowing = the per-row short-circuit fold**, on
every row: the columnar path succeeds exactly when every row's per-row
evaluation succeeds, and then produces the same three-valued result for every
row. -/
theorem andOr_agrees (sc : Bool) (cs : List (R → Except String (V F))) (rows : List R) :
    vecAndOr sc cs rows = omap (fun r => (scalarAndOr sc r cs false).toOption) rows := by
  unfold vecAndOr
  rw [fold_omap_comm (fun c r s => stepRow sc c r s) cs, omap_map, omap_post]
  congr 1; funext r
  rw [rowFold_scalar]
  show Option.map _ (Option.map _ (ofold (rstep sc r) cs (RowSt.live false)))
    = Option.map _ (ofold (rstep sc r) cs (RowSt.live false))
  cases ofold (rstep sc r) cs (RowSt.live false) <;> rfl

/-! ### Kleene truth tables -/

def k3 : Option Bool → V F
  | none => .null
  | some b => .bool b

def constChild (v : V F) : R → Except String (V F) := fun _ => .ok v

def kAnd : Option Bool → Option Bool → Option Bool
  | some false, _ => some false
  | _, some false => some false
  | some true, some true => some true
  | _, _ => none

def kOr : Option Bool → Option Bool → Option Bool
  | some true, _ => some true
  | _, some true => some true
  | some false, some false => some false
  | _, _ => none

def kNot : Option Bool → Option Bool
  | none => none
  | some b => some (!b)

/-- `a AND b` on booleans and nulls is Kleene conjunction. -/
theorem and_kleene (r : R) (a b : Option Bool) :
    scalarAndOr false r [constChild (k3 (F := F) a), constChild (k3 b)] false = .ok (k3 (kAnd a b)) := by
  cases a with
  | none => cases b with
    | none => rfl
    | some b => cases b <;> rfl
  | some a => cases a <;> cases b with
    | none => rfl
    | some b => cases b <;> rfl

/-- `a OR b` on booleans and nulls is Kleene disjunction. -/
theorem or_kleene (r : R) (a b : Option Bool) :
    scalarAndOr true r [constChild (k3 (F := F) a), constChild (k3 b)] false = .ok (k3 (kOr a b)) := by
  cases a with
  | none => cases b with
    | none => rfl
    | some b => cases b <;> rfl
  | some a => cases a <;> cases b with
    | none => rfl
    | some b => cases b <;> rfl

theorem deMorgan_and (a b : Option Bool) : kNot (kAnd a b) = kOr (kNot a) (kNot b) := by
  cases a with
  | none => cases b with
    | none => rfl
    | some b => cases b <;> rfl
  | some a => cases a <;> cases b with
    | none => rfl
    | some b => cases b <;> rfl

theorem deMorgan_or (a b : Option Bool) : kNot (kOr a b) = kAnd (kNot a) (kNot b) := by
  cases a with
  | none => cases b with
    | none => rfl
    | some b => cases b <;> rfl
  | some a => cases a <;> cases b with
    | none => rfl
    | some b => cases b <;> rfl

/-- Short-circuit: `false AND <error>` is `false` and `true OR <error>` is `true`
— the right operand is never evaluated. -/
theorem and_short_circuit (r : R) (e : String) :
    scalarAndOr false r [constChild (.bool false : V F), fun _ => .error e] false = .ok (.bool false) := rfl

/-- …but `null AND <error>` evaluates the right operand and fails. -/
theorem and_null_evaluates_rhs (r : R) (e : String) :
    scalarAndOr false r [constChild (.null : V F), fun _ => .error e] false = .error e := rfl

/-- `XOR` (eval.rs:604-625): any null makes it null, otherwise parity. -/
def scalarXor : List (V F) → Option Bool → Except String (V F)
  | [], last => .ok (.bool (last.getD false))
  | .bool b :: vs, last => scalarXor vs (some (last.elim b (fun l => l != b)))
  | .null :: _, _ => .ok .null
  | _ :: _, _ => .error boolMismatch

theorem xor_kleene (a b : Option Bool) :
    scalarXor [k3 a, k3 b] none = .ok (k3 (match a, b with
      | some x, some y => some (x != y) | _, _ => none) : V F) := by
  cases a with
  | none => rfl
  | some a => cases b with
    | none => rfl
    | some b => cases a <;> cases b <;> rfl

/-! ## `CASE` -/

/-- The value form's match test in the per-row `eval_case` (eval.rs:500):
`compare_value == (Equal, None)`. -/
def caseMatchScalar (w subj : V F) : Bool :=
  match compareValue w subj with
  | (.eq, .none) => true
  | _ => false

/-- …and in the columnar `eval_case` (vector_expr.rs:505):
`compare_values(.., Eq) == Some(true)`. -/
def caseMatchVec (w subj : V F) : Bool := compareValues w subj .eq == some true

/-- **The two `CASE` value-form match tests agree** on every pair. -/
theorem case_match_agrees (w subj : V F) : caseMatchScalar w subj = caseMatchVec w subj := by
  unfold caseMatchScalar caseMatchVec compareValues
  obtain ⟨o, fl⟩ := compareValue w subj
  cases o <;> cases fl <;> rfl

/-- Searched form (both paths): anything but `false`/`null` selects the arm. -/
def truthy : V F → Bool
  | .bool false => false
  | .null => false
  | _ => true

def caseMatch (subj : Option (V F)) (w : V F) : Bool :=
  match subj with
  | some s => caseMatchScalar w s
  | none => truthy w

structure Arm (R F : Type) where
  whenE : R → Except String (V F)
  thenE : R → Except String (V F)

/-- Per-row `eval_case` (eval.rs:485): arms in order, lazily; `ELSE` last. -/
def scalarCase (subj : Option (V F)) (r : R) (els : R → Except String (V F)) :
    List (Arm R F) → Except String (V F)
  | [] => els r
  | a :: as => match a.whenE r with
    | .error e => .error e
    | .ok w => if caseMatch subj w then a.thenE r else scalarCase subj r els as

/-- One position of the columnar `eval_case`: unclaimed, claimed by the arm
whose `WHEN` just matched (its `THEN` still to evaluate), or finished. -/
inductive CSt (F : Type) where
  | un
  | pend
  | got (v : V F)

/-- Phase 1 of an arm: `WHEN` over the unclaimed rows (vector_expr.rs:493-520). -/
def whenPhase (subj : Option (V F)) (a : Arm R F) (r : R) : CSt F → Option (CSt F)
  | .un => match a.whenE r with
    | .error _ => none
    | .ok w => some (if caseMatch subj w then .pend else .un)
  | st => some st

/-- Phase 2 of an arm: `THEN` over the rows it claimed (vector_expr.rs:522-528). -/
def thenPhase (a : Arm R F) (r : R) : CSt F → Option (CSt F)
  | .pend => (a.thenE r).toOption.map CSt.got
  | st => some st

/-- The final `ELSE` over what no arm claimed (vector_expr.rs:535-541). -/
def elsePhase (els : R → Except String (V F)) (r : R) : CSt F → Option (CSt F)
  | .un => (els r).toOption.map CSt.got
  | st => some st

def CSt.out : CSt F → V F
  | .got v => v
  | _ => .null

/-- The phases the columnar `CASE` runs, in order. -/
inductive Phase (R F : Type) where
  | w (a : Arm R F)
  | t (a : Arm R F)
  | e (els : R → Except String (V F))

def phaseStep (subj : Option (V F)) : Phase R F → R → CSt F → Option (CSt F)
  | .w a => whenPhase subj a
  | .t a => thenPhase a
  | .e els => elsePhase els

def phases (els : R → Except String (V F)) : List (Arm R F) → List (Phase R F)
  | [] => [.e els]
  | a :: as => .w a :: .t a :: phases els as

/-- Columnar `CASE` with a fixed subject (the value form evaluates the subject
over all rows first — one more phase of the same shape). -/
def vecCase (subj : Option (V F)) (els : R → Except String (V F)) (arms : List (Arm R F))
    (rows : List R) : Option (List (V F)) :=
  (ofold (fun ph ys => omap (fun p => (phaseStep subj ph p.1 p.2).map (p.1, ·)) ys)
      (phases els arms) (rows.map (·, CSt.un))).map (fun ys => ys.map (fun p => p.2.out))

def cstep (subj : Option (V F)) (r : R) (ph : Phase R F) (st : CSt F) : Option (CSt F) :=
  phaseStep subj ph r st

theorem caseFold_got (subj : Option (V F)) (r : R) (v : V F) (els : R → Except String (V F)) :
    ∀ arms : List (Arm R F), ofold (cstep subj r) (phases els arms) (.got v) = some (.got v)
  | [] => rfl
  | _ :: as => by
    simp only [phases, ofold, cstep, phaseStep, whenPhase, thenPhase]
    exact caseFold_got subj r v els as

theorem caseFold_scalar (subj : Option (V F)) (r : R) (els : R → Except String (V F)) :
    ∀ arms : List (Arm R F),
      (scalarCase subj r els arms).toOption
        = (ofold (cstep subj r) (phases els arms) .un).map CSt.out
  | [] => by
    simp only [scalarCase, phases, ofold, cstep, phaseStep, elsePhase]
    cases els r <;> rfl
  | a :: as => by
    simp only [scalarCase, phases, ofold, cstep, phaseStep, whenPhase]
    cases hw : a.whenE r with
    | error e => rfl
    | ok w =>
      simp only
      by_cases hm : caseMatch subj w = true
      · simp only [hm, ↓reduceIte, thenPhase]
        cases ht : a.thenE r with
        | error e => rfl
        | ok v =>
          simp only [Except.toOption, Option.map_some]
          rw [caseFold_got]; rfl
      · simp only [hm, Bool.false_eq_true, ↓reduceIte, thenPhase]
        exact caseFold_scalar subj r els as

/-- **Columnar `CASE` = per-row `CASE`** on every row: a `THEN` / `ELSE` is
evaluated for exactly the rows the per-row path evaluates it for, so the
columnar path fails iff some row fails (`CASE WHEN n.d <> 0 THEN 1 / n.d …`
cannot divide by zero on the columnar path either). -/
theorem case_agrees (subj : Option (V F)) (els : R → Except String (V F))
    (arms : List (Arm R F)) (rows : List R) :
    vecCase subj els arms rows = omap (fun r => (scalarCase subj r els arms).toOption) rows := by
  unfold vecCase
  rw [fold_omap_comm (fun ph r st => phaseStep subj ph r st) (phases els arms), omap_map, omap_post]
  congr 1; funext r
  rw [caseFold_scalar]
  show Option.map _ (Option.map _ (ofold (cstep subj r) (phases els arms) CSt.un))
    = Option.map _ (ofold (cstep subj r) (phases els arms) CSt.un)
  cases ofold (cstep subj r) (phases els arms) CSt.un <;> rfl

/-! ## `IN` -/

/-- `impl Contains for ThinVec<Value>` (value.rs:1762). -/
def containsGo (v : V F) : List (V F) → Bool → V F
  | [], isNull => if isNull then .null else .bool false
  | x :: xs, isNull =>
    let (o, fl) := compareValue v x
    let isNull' := isNull || fl == .comparedNull
    if o == .eq then (if fl == .comparedNull then .null else .bool true)
    else containsGo v xs isNull'

def contains (v : V F) (l : List (V F)) : V F := containsGo v l false

def isScalar : V F → Bool
  | .other _ => false
  | _ => true

def eqOB (a b : V F) : Option Bool :=
  match evalCmp .eq a b with
  | .bool b => some b
  | _ => none

theorem evalCmp_eq_shape (a b : V F) :
    evalCmp .eq a b = .null ∨ evalCmp .eq a b = .bool true ∨ evalCmp .eq a b = .bool false := by
  unfold evalCmp
  obtain ⟨o, fl⟩ := compareValue a b
  cases fl <;> simp <;> cases o <;> simp [oEq]

/-- Kleene "any": `true` if some element is `true`, else `null` if some is
`null`, else `false`. -/
def kAny : List (Option Bool) → Option Bool
  | [] => some false
  | x :: xs => kOr x (kAny xs)

theorem compareFloats_flag (a b : F) :
    (compareFloats a b).2 = .none ∨ ((compareFloats a b).2 = .nan ∧ (compareFloats a b).1 = .lt) := by
  unfold compareFloats; cases FM.pcmp a b <;> simp

theorem cf_ne (a b : F) : (compareFloats a b).2 ≠ .none → (compareFloats a b).1 ≠ .eq := by
  unfold compareFloats; cases FM.pcmp a b <;> simp

/-- For a non-null scalar on the left, an inconclusive comparison (null,
disjoint, NaN) never reports `Equal`. -/
theorem cv_inconclusive_ne (v x : V F) (hv : v ≠ .null) (hs : isScalar v = true)
    (hsx : isScalar x = true) :
    (compareValue v x).2 ≠ .none → (compareValue v x).1 ≠ .eq := by
  intro hfl
  cases v <;> cases x <;> simp only [isScalar] at hs hsx <;>
    (try simp only [cv_ff, cv_if, cv_fi] at hfl ⊢)
  all_goals first
    | contradiction
    | exact cf_ne _ _ hfl
    | (simp [compareValue, cmpNat, V.order] at hfl ⊢)

theorem eqOB_of {v x : V F} {o : Ordering} {fl : Flag} (h : compareValue v x = (o, fl)) :
    eqOB v x = (match fl with
      | .comparedNull => none | .nan => some false | .disjoint => some false
      | .none => some (oEq o)) := by
  unfold eqOB evalCmp; simp only [h]; cases fl <;> rfl

theorem kOr_false_l (b : Option Bool) : kOr (some false) b = b := by
  cases b with
  | none => rfl
  | some b => cases b <;> rfl

/-- Absorbing an inconclusive element: `null` (from a null comparison) or
`false` (from NaN / disjoint types). -/
theorem kOr_step (n : Bool) (e : Option Bool) (he : e = none ∨ e = some false) (rest : Option Bool) :
    kOr (if n then none else some false) (kOr e rest)
      = kOr (if (n || e == none) then none else some false) rest := by
  rcases he with h | h <;> subst h <;> cases n <;>
    (cases rest with
      | none => rfl
      | some b => cases b <;> rfl)

/-- **`x IN list` is Kleene "any" of `x = element`** (for a non-null scalar
`x` over scalar elements — the shapes where `compare_value` never reports
`Equal` together with an inconclusive flag). -/
theorem contains_is_kleene_any (v : V F) (hv : v ≠ .null) (hs : isScalar v = true) :
    ∀ (l : List (V F)) (n : Bool), (∀ x ∈ l, isScalar x = true) →
      containsGo v l n = k3 (kOr (if n then none else some false) (kAny (l.map (eqOB v))))
  | [], n, _ => by cases n <;> rfl
  | x :: xs, n, hl => by
    have hx : isScalar x = true := hl x (by simp)
    have ih := contains_is_kleene_any v hv hs xs
    have hne := cv_inconclusive_ne v x hv hs hx
    simp only [containsGo, List.map_cons, kAny]
    generalize hc : compareValue v x = pr at hne ⊢
    obtain ⟨o, fl⟩ := pr
    rw [eqOB_of hc]
    simp only at hne ⊢
    have ih' := fun m => ih m (fun y hy => hl y (by simp [hy]))
    cases fl with
    | none =>
      cases o with
      | eq => cases n <;> rfl
      | lt =>
        have : (Flag.none == Flag.comparedNull) = false := rfl
        have e1 : (Ordering.lt == Ordering.eq) = false := rfl
        simp only [this, e1, Bool.or_false, oEq, Bool.false_eq_true, ↓reduceIte]
        rw [ih' n, kOr_step n (some false) (Or.inr rfl)]
        cases n <;> rfl
      | gt =>
        have : (Flag.none == Flag.comparedNull) = false := rfl
        have e1 : (Ordering.gt == Ordering.eq) = false := rfl
        simp only [this, e1, Bool.or_false, oEq, Bool.false_eq_true, ↓reduceIte]
        rw [ih' n, kOr_step n (some false) (Or.inr rfl)]
        cases n <;> rfl
    | comparedNull =>
      have ho : (o == .eq) = false := by cases o <;> simp_all
      have : (Flag.comparedNull == Flag.comparedNull) = true := rfl
      simp only [ho, this, Bool.false_eq_true, ↓reduceIte, Bool.or_true]
      rw [ih' true, kOr_step n none (Or.inl rfl)]
      cases n <;> rfl
    | nan =>
      have ho : (o == .eq) = false := by cases o <;> simp_all
      have : (Flag.nan == Flag.comparedNull) = false := rfl
      simp only [ho, this, Bool.false_eq_true, ↓reduceIte, Bool.or_false]
      rw [ih' n, kOr_step n (some false) (Or.inr rfl)]
      cases n <;> rfl
    | disjoint =>
      have ho : (o == .eq) = false := by cases o <;> simp_all
      have : (Flag.disjoint == Flag.comparedNull) = false := rfl
      simp only [ho, this, Bool.false_eq_true, ↓reduceIte, Bool.or_false]
      rw [ih' n, kOr_step n (some false) (Or.inr rfl)]
      cases n <;> rfl

theorem cv_null_l (x : V F) : compareValue .null x = (cmpNat 32768 x.order, .comparedNull) := by
  cases x <;> rfl

theorem containsGo_null (n : Bool) :
    ∀ l : List (V F), containsGo .null l n = if l = [] ∧ n = false then .bool false else .null
  | [] => by cases n <;> rfl
  | x :: xs => by
    simp only [containsGo, cv_null_l]
    have : (Flag.comparedNull == Flag.comparedNull) = true := rfl
    simp only [this, ↓reduceIte, Bool.or_true, reduceCtorEq, false_and]
    split
    · rfl
    · rw [containsGo_null true xs]; simp

/-- `null IN list` is `null` for a non-empty list and `false` for `[]`. -/
theorem null_in (l : List (V F)) :
    contains .null l = if l = [] then .bool false else .null := by
  unfold contains; rw [containsGo_null]; simp

/-! ## `^`: the `flat_map` that drops errors (BUG) -/

/-- `apply_pow` (math.rs:329): numeric pairs become a float power, anything
else — including a non-numeric operand — is `null`, never a type error. -/
def applyPow (pw : F → F → F) : V F → V F → V F
  | .int a, .int b => .flt (pw (FM.ofInt a) (FM.ofInt b))
  | .flt a, .flt b => .flt (pw a b)
  | .int a, .flt b => .flt (pw (FM.ofInt a) b)
  | .flt a, .int b => .flt (pw a (FM.ofInt b))
  | _, _ => .null

/-- `ExprIR::Pow` in `eval_compound` (eval.rs:818-825):
`children.flat_map(|c| self.eval_node(c)).reduce(apply_pow)`. `flat_map` over a
`Result` yields the `Ok` value and *nothing* for an `Err`. -/
def powFold (pw : F → F → F) (xs : List (Except String (V F))) : Except String (V F) :=
  match xs.filterMap Except.toOption with
  | [] => .error "Pow operator requires at least one argument"
  | y :: ys => .ok (ys.foldl (applyPow pw) y)

/-- What every other binary operator does (`?` on each child), and what C does. -/
def powSpec (pw : F → F → F) (xs : List (Except String (V F))) : Except String (V F) :=
  match xs.mapM id with
  | .error e => .error e
  | .ok [] => .error "Pow operator requires at least one argument"
  | .ok (y :: ys) => .ok (ys.foldl (applyPow pw) y)

/-- **BUG**: `2 ^ (1/0)` is `2` — the right operand's error vanishes and the
left operand is returned *unpowered* (an `Int`, not even a `Float`). -/
theorem pow_drops_error (pw : F → F → F) :
    powFold pw [.ok (.int 2), .error "Division by zero"] = .ok (.int 2) ∧
    powSpec pw [.ok (.int 2), .error "Division by zero"] = .error "Division by zero" :=
  ⟨rfl, rfl⟩

/-- **BUG, consequence**: the optimizer's `eliminate_true_filters` evaluates a
`WHERE` predicate with the constant evaluator, where every variable is an
error ("Variable not found"). For `x ^ true` that error is dropped, the fold
yields the constant `true`, and the whole `Filter` is deleted from the plan —
so `WHERE x ^ true` keeps every row although `x ^ true` is `null` for each. -/
theorem pow_constant_true (pw : F → F → F) :
    powFold pw [.error "Variable not found", .ok (.bool true)] = .ok (.bool true) := rfl

/-- On error-free operands the two agree: the bug is exactly error dropping. -/
theorem pow_agrees_without_errors (pw : F → F → F) (vs : List (V F)) :
    powFold pw (vs.map .ok) = powSpec pw (vs.map .ok) := by
  have h2 : ∀ ws : List (V F), (ws.map (Except.ok : V F → Except String (V F))).mapM id = .ok ws := by
    intro ws
    induction ws with
    | nil => rfl
    | cons v vs ih => simp only [List.map_cons, List.mapM_cons, id]; rw [ih]; rfl
  have h1 : ∀ ws : List (V F),
      (ws.map (Except.ok : V F → Except String (V F))).filterMap Except.toOption = ws := by
    intro ws
    induction ws with
    | nil => rfl
    | cons v vs ih => simp [Except.toOption, ih]
  unfold powFold powSpec
  rw [h1, h2]
  cases vs <;> rfl

/-! ## `UNWIND range(..)`: the lazy iterator overflows (BUG) -/

/-- `RangeIter::next` (eval.rs:108-122) in a release build (wrapping `+=`),
run for at most `fuel` items. `step` is stored as `step.unsigned_abs() as i64`
and negated for a descending range — `wrap` models both casts. -/
def rangeIterRel (up : Bool) (stepAbs endv : Int) : Nat → Int → List Int
  | 0, _ => []
  | fuel + 1, cur =>
    if (up && decide (cur > endv)) || (!up && decide (cur < endv)) then []
    else cur :: rangeIterRel up stepAbs endv fuel
      (wrap (cur + (if up then wrap stepAbs else wrap (-(wrap stepAbs)))))

/-- The same iterator in a debug build: `+=` and unary `-` are checked and
panic ("attempt to add/negate with overflow"). -/
def rangeIterDbg (up : Bool) (stepAbs endv : Int) : Nat → Int → Except String (List Int)
  | 0, _ => .ok []
  | fuel + 1, cur =>
    if (up && decide (cur > endv)) || (!up && decide (cur < endv)) then .ok []
    else
      let st := wrap stepAbs   -- `step.unsigned_abs() as i64`
      let d := if up then st else -st
      if !up && ¬ InI64' (-st) then .error "attempt to negate with overflow"
      else if ¬ InI64' (cur + d) then .error "attempt to add with overflow"
      else (rangeIterDbg up stepAbs endv fuel (cur + d)).map (cur :: ·)
where
  InI64' (x : Int) : Bool := decide (I64MIN ≤ x) && decide (x ≤ I64MAX)

/-- The mathematical sequence, which is what the eager `range()` function
(math.rs:251, `(start..=end).step_by(step)`) returns. -/
def rangeMath (up : Bool) (stepAbs endv : Int) : Nat → Int → List Int
  | 0, _ => []
  | fuel + 1, cur =>
    if (up && decide (cur > endv)) || (!up && decide (cur < endv)) then []
    else cur :: rangeMath up stepAbs endv fuel (cur + (if up then stepAbs else -stepAbs))

/-- When the step past `end` is still representable, the lazy iterator is the
mathematical range (and so agrees with the eager `range()`). -/
theorem rangeIter_ok_up (stepAbs endv : Int) (hs : 0 < stepAbs) (hs' : stepAbs ≤ I64MAX)
    (hroom : endv + stepAbs ≤ I64MAX) :
    ∀ (fuel : Nat) (cur : Int), I64MIN ≤ cur →
      rangeIterRel true stepAbs endv fuel cur = rangeMath true stepAbs endv fuel cur
  | 0, _, _ => rfl
  | fuel + 1, cur, hc => by
    simp only [rangeIterRel, rangeMath, Bool.true_and, Bool.not_true, Bool.false_and,
      Bool.or_false, ↓reduceIte]
    by_cases h : cur > endv
    · simp [h]
    · simp only [h, decide_false, Bool.false_eq_true, ↓reduceIte, List.cons.injEq, true_and]
      have hin : InI64 (cur + stepAbs) := by unfold InI64 I64MIN I64MAX at *; omega
      have hst : InI64 stepAbs := by unfold InI64 I64MIN I64MAX at *; omega
      rw [wrap_id hst, wrap_id hin]
      exact rangeIter_ok_up stepAbs endv hs hs' hroom fuel (cur + stepAbs)
        (by unfold I64MIN at *; omega)

/-- **BUG (release)**: `UNWIND range(i64::MAX - 1, i64::MAX)` yields
`MAX-1, MAX, MIN, MIN+1, …` and never stops (`cur > MAX` is never true);
exactly the rows `redis-cli GRAPH.QUERY g "UNWIND range(9223372036854775806,
9223372036854775807) AS x RETURN x LIMIT 5"` returns. The eager `range()` gives
the two elements. -/
theorem rangeIter_wraps :
    rangeIterRel true 1 I64MAX 5 (I64MAX - 1)
      = [I64MAX - 1, I64MAX, I64MIN, I64MIN + 1, I64MIN + 2] ∧
    rangeMath true 1 I64MAX 5 (I64MAX - 1) = [I64MAX - 1, I64MAX] := by
  decide

/-- **BUG (debug)**: the same query panics. -/
theorem rangeIter_panics :
    rangeIterDbg true 1 I64MAX 5 (I64MAX - 1) = .error "attempt to add with overflow" := by
  rfl

/-- **BUG (debug)**: descending to `i64::MIN` panics too, and a step of
`i64::MIN` panics on the negation before the second item. -/
theorem rangeIter_panics_down :
    rangeIterDbg false 1 I64MIN 5 (I64MIN + 1) = .error "attempt to add with overflow" ∧
    rangeIterDbg false (-I64MIN) (-10) 5 0 = .error "attempt to negate with overflow" :=
  ⟨rfl, rfl⟩

/-! ## Numeric comparison across Int and Float is not transitive -/

/-- Whenever two distinct integers round to the same `f64`, `=` stops being
transitive: `a = f`, `b = f`, yet `a <> b`. (Concretely `a = 2^53 + 1`,
`b = 2^53`, `f = 2^53.0`; see `#eval`s below. C FalkorDB promotes the same
way; Neo4j compares exactly.) -/
theorem int_float_eq_not_transitive (a b : Int) (f : F) (hab : a ≠ b)
    (ha : FM.pcmp (FM.ofInt a) f = some .eq) (hb : FM.pcmp (FM.ofInt b) f = some .eq) :
    evalCmp .eq (.int a : V F) (.flt f) = .bool true ∧
    evalCmp .eq (.int b : V F) (.flt f) = .bool true ∧
    evalCmp .eq (.int a : V F) (.int b) = .bool false := by
  refine ⟨?_, ?_, ?_⟩
  · simp [evalCmp, compareFloats, ha, oEq]
  · simp [evalCmp, compareFloats, hb, oEq]
  · rw [vec_cmp_int]; simp [kI64, hab]

/-! ## `sign`, `toInteger`, `coalesce` -/

/-- `sign` (math.rs:210-219, since #2907 `f04c3557a`): the Float arm is
`Value::Int(i64::from(*f > 0.0) - i64::from(*f < 0.0))` (math.rs:214), read
through `partial_cmp` against `0.0` (`FM.ofInt 0`), so NaN gives `0`. -/
def signRust : V F → V F
  | .int n => .int (if n > 0 then 1 else if n < 0 then -1 else 0)
  | .flt f => .int ((if fGt f (FM.ofInt 0) then 1 else 0) - (if fLt f (FM.ofInt 0) then 1 else 0))
  | _ => .null

/-- **#2906 fixed** (#2907 `f04c3557a`): `sign` of any Float is an Integer, as in C's
`AR_SIGN` and openCypher. -/
theorem sign_float_is_int (f : F) : ∃ n, signRust (.flt f) = .int n ∧ (n = 1 ∨ n = -1 ∨ n = 0) := by
  refine ⟨_, rfl, ?_⟩
  unfold fGt fLt
  cases h : FM.pcmp f (FM.ofInt 0) with
  | none => simp
  | some o => cases o <;> simp

/-- A Float sign follows the IEEE comparison with `0.0`; incomparable (NaN) and equal
(`±0.0`) give `0`. -/
theorem sign_float_cases (f : F) :
    (FM.pcmp f (FM.ofInt 0) = some .gt → signRust (.flt f) = .int 1) ∧
    (FM.pcmp f (FM.ofInt 0) = some .lt → signRust (.flt f) = .int (-1)) ∧
    (FM.pcmp f (FM.ofInt 0) = some .eq → signRust (.flt f) = .int 0) ∧
    (FM.pcmp f (FM.ofInt 0) = none → signRust (.flt f) = .int 0) := by
  refine ⟨fun h => ?_, fun h => ?_, fun h => ?_, fun h => ?_⟩ <;> simp [signRust, fGt, fLt, h]

/-- Historical: `sign` before #2907 (`f04c3557a`) answered `Float(f.signum().round())`
for a non-zero float. -/
def signRustPre2907 (signum : F → F) (isZero : F → Bool) : V F → V F
  | .int n => .int (if n > 0 then 1 else if n < 0 then -1 else 0)
  | .flt f => if isZero f then .int 0 else .flt (signum f)
  | _ => .null

/-- Historical counterexample (pre-#2907): a non-zero float gave a **Float**. Fixed by
#2907 (`f04c3557a`); see `sign_float_is_int`. -/
theorem pre2907_sign_float_is_float (signum : F → F) (isZero : F → Bool) (f : F) (h : isZero f = false) :
    ∃ g, signRustPre2907 signum isZero (.flt f) = .flt g := ⟨signum f, by simp [signRustPre2907, h]⟩

/-- `toInteger` of a finite float, with `m = floor(f)` as a mathematical
integer: `m as i64` saturates. The *string* path (`toInteger('1e30')`) instead
returns `null` out of range, so the two disagree. -/
def toIntegerFloat (m : Int) : Option Int := some (max I64MIN (min I64MAX m))
def toIntegerString (m : Int) : Option Int := if I64MIN ≤ m ∧ m < 9223372036854775808 then some m else none

theorem toInteger_float_vs_string (m : Int) (h : m > I64MAX) :
    toIntegerFloat m = some I64MAX ∧ toIntegerString m = none := by
  unfold toIntegerFloat toIntegerString I64MAX I64MIN at *
  constructor
  · congr 1; omega
  · simp only [ite_eq_right_iff, reduceCtorEq, imp_false]; omega

/-- `coalesce` (math.rs:314) tests `*arg == Value::Null` through `PartialEq`,
i.e. the ordering half of `compare_value`. For every value whose type order is
not `Null`'s, that is exactly "is not null". -/
def coalesce : List (V F) → V F
  | [] => .null
  | a :: as => if (compareValue a .null).1 == .eq then coalesce as else a

theorem coalesce_first_nonnull :
    ∀ l : List (V F), (∀ x ∈ l, x.order = 32768 → x = .null) →
      coalesce l = (l.find? (fun x => match x with | .null => false | _ => true)).getD .null
  | [], _ => rfl
  | a :: as, h => by
    have ih := coalesce_first_nonnull as (fun x hx => h x (by simp [hx]))
    cases a with
    | null => simp [coalesce, compareValue, cmpNat, V.order, ih]
    | other t =>
      by_cases ht : t = 32768
      · have := h (.other t) (by simp) (by simp [V.order, ht]); cases this
      · by_cases hlt : t < 32768
        · simp [coalesce, compareValue, cmpNat, V.order, hlt]
        · simp [coalesce, compareValue, cmpNat, V.order, hlt, ht]
    | bool b => simp [coalesce, compareValue, cmpNat, V.order]
    | int i => simp [coalesce, compareValue, cmpNat, V.order]
    | flt f => simp [coalesce, compareValue, cmpNat, V.order]
    | str x => simp [coalesce, compareValue, cmpNat, V.order]

end FalkorExpr


/-! ## Concrete counterexamples on Lean's `Float` (IEEE binary64) -/

instance : FalkorExpr.FloatModel Float where
  pcmp a b := if a < b then some .lt else if a == b then some .eq else if b < a then some .gt else none
  add := (· + ·)
  sub := (· - ·)
  mul := (· * ·)
  div := (· / ·)
  rem a b := a - b * (a / b).floor  -- only for `#eval`; no proof uses it
  ofInt := Float.ofInt

open FalkorExpr in
/-- `(2^53+1 : f64) == 2^53.0`, `(2^53 : f64) == 2^53.0`, and `2^53+1 == 2^53`
as integers: the hypotheses of `int_float_eq_not_transitive` hold on IEEE
binary64. -/
def nonTransitiveWitness : Bool × Bool × Bool :=
  (FloatModel.pcmp (FloatModel.ofInt 9007199254740993 : Float) 9007199254740992.0 == some .eq,
   FloatModel.pcmp (FloatModel.ofInt 9007199254740992 : Float) 9007199254740992.0 == some .eq,
   (9007199254740993 : Int) == 9007199254740992)

#eval nonTransitiveWitness  -- (true, true, false)
