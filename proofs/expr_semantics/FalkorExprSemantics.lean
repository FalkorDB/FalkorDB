import FalkorExprSemantics.Eval
import FalkorExprSemantics.EvalMore
import FalkorExprSemantics.BFS
import FalkorExprSemantics.FnRegistry
import FalkorCore.Int64
import FalkorCore.Float
/-
# Expression evaluation: the columnar path agrees with the per-row path, and
# where the per-row path itself departs from openCypher / C FalkorDB

**Re-targeted to main 737dfb5da** (from 8621212fc). #2899 (`e63165cb5`, closes #2894):
`RangeIter` now stores the exact element count `remaining` and the signed `step`, and
advances only while another element is due — re-modelled as `rangeIter` (debug, checked
`+=`) / `rangeIterRel` (release); `rangeIter_correct`: for every `i64` start/end and
nonzero step the caller accepts, it yields exactly the eager range (`rangeList`, and
`rangeMath` with any fuel ≥ the length) and never overflows; `evalIter_range_is_iter`,
`rangeIter_sizeHint_exact` (new `size_hint`), `rangeIter_bounds_examples`,
`rangeIter_max_fixed`. The old model and its counterexamples are kept as
`pre2899_rangeIterRel`/`pre2899_rangeIterDbg`, `pre2899_rangeIter_wraps`,
`pre2899_rangeIter_panics`, `pre2899_rangeIter_panics_down`, `pre2899_rangeIter_ok_up`.
#2950 (temporal.rs) and #3041 (parser) touch nothing modelled here. COVERAGE.tsv
eval.rs line numbers re-pinned to 737dfb5da.

Earlier: re-targeted to main 8621212fc (from 289c201fe). #2904 (#2902): `^` propagates
operand errors (`?` instead of `flat_map`) and `apply_pow` type-checks its operands
(`Type mismatch: expected Integer, Float, or Null but was …`) — `applyPow`/`powFold`
re-modelled, `pow_propagates_error`, `pow_constant_not_true`, `pow_bool_type_error`,
`pow_binary_eq_spec`, `pow_agrees_without_errors`; the old model is kept as
`pre2904_applyPow`/`pre2904_powFold` with `pre2904_pow_drops_error`,
`pre2904_pow_constant_true`. COVERAGE.tsv line numbers re-pinned to 8621212fc.

Earlier: re-targeted to main 289c201fe (from fb3d516d0). #3294 (#3292): the `AND`/`OR`/`XOR`
operand type error now names the operand's type like C (`expected Boolean or Null but
was Integer`), on both the per-row and the columnar path — `boolMismatch` takes the
value, `and_or_mismatch_names_type`, `xor_mismatch_names_type`. #2951 (#2945) changes
only `eval_compound`'s aggregate-finalizer arm, which this project does not model
(proofs/ops_aggregate does). The `init_functions` name set (`FnRegistry`) is unchanged.

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

* `^` silently dropping an operand's error — fixed by #2904 (`fd2c76a6c`), now
  `pow_propagates_error` (historical `pre2904_pow_drops_error`; the optimizer
  deleting `WHERE x ^ true` as constant-true is `pre2904_pow_constant_true`);
* the lazy `UNWIND range(..)` iterator overflowing after yielding `i64::MAX` — fixed by
  #2899 (`e63165cb5`), now `rangeIter_correct` (historical `pre2899_rangeIter_wraps`:
  panic in debug, wrong rows and non-termination in release);
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
| `applyPow`, `powStep`, `powFold` | `apply_pow` (`math.rs:328`); `ExprIR::Pow` in `eval_compound` (`eval.rs:828-834`, `reduce` calling `apply_pow(acc?, value?)`) |
| `pre2904_applyPow`, `pre2904_powFold` | the same before #2904 (the `flat_map`) |
| `pow_constant_not_true`  | `is_constant_true` (`graph/src/planner/optimizer/eliminate_true_filters.rs:48`) |
| `rangeIter` / `rangeIterRel` | `RangeIter::next` (`eval.rs:114-127`), debug / release build |
| `rangeIterSizeHint`      | `RangeIter::size_hint` (`eval.rs:129-132`) |
| `rangeLen`, `rangeList`  | `length` in `eval_iter_expr` (`eval.rs:1224-1233`); the eager `range()` list |
| `pre2899_rangeIterRel` / `pre2899_rangeIterDbg` | `RangeIter::next` before #2899, release / debug |
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

## Gap closed 2026-10-20

* #2724 (`FnRegistry.lean`): property `CoversC` — every user-callable C built-in (the 104
  non-internal `AR_FuncDesc`s of origin/master `builtin_funcs.c`) resolves in the Rust
  registry. `bug2724_prev_missing`: `prev` does not (live: "Unknown function 'prev'"; C
  returns null); `covers_all_but_prev`: it is the only one. **Open**.
-/

set_option linter.unusedSectionVars false

namespace FalkorExpr
open FalkorCore

/-! ## i64

`I64MIN`/`I64MAX`, `InI64`, `wrap`, `wrap_inRange`, `wrap_id` and the `wrapping_*` ops
(`wAdd`…`wRem`) are proofs/core's (`FalkorCore/Int64.lean`). -/

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
  · simp only [InI64, I64MIN, I64MAX] at *; omega

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

-- `Flag` (`compare_value`'s second component) is proofs/core's `FalkorCore.Flag`.

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
`all_not_equals` (eval.rs:1688; since #2917 7fdf06a0e: null makes it null, disjoint
types and NaN are unequal, else `ord != Equal` — before, `partial_cmp`, which read a
map's disjoint `(Equal, Disjoint)` as equal), the orderings are the `Lt/Gt/Le/Ge` arms. -/
def evalCmp (op : CmpOp) (a b : V F) : V F :=
  let (o, fl) := compareValue a b
  match op with
  | .eq => match fl with
    | .comparedNull => .null
    | .nan | .disjoint => .bool false
    | .none => .bool (oEq o)
  | .neq => match fl with
    | .comparedNull => .null
    | .nan | .disjoint => .bool true
    | .none => .bool (!oEq o)
  | .lt => match fl with
    | .comparedNull | .disjoint => .null | .nan => .bool false | .none => .bool (oLt o)
  | .gt => match fl with
    | .comparedNull | .disjoint => .null | .nan => .bool false | .none => .bool (oGt o)
  | .le => match fl with
    | .comparedNull | .disjoint => .null | .nan => .bool false | .none => .bool (!oGt o)
  | .ge => match fl with
    | .comparedNull | .disjoint => .null | .nan => .bool false | .none => .bool (!oLt o)

/-- `vectorized::compare_values` (vectorized.rs:239; `Neq` arm as #2917): `none` is null. -/
def compareValues (a b : V F) (op : CmpOp) : Option Bool :=
  let (o, fl) := compareValue a b
  match op with
  | .eq => match fl with
    | .comparedNull => none
    | .nan | .disjoint => some false
    | .none => some (oEq o)
  | .neq => match fl with
    | .comparedNull => none
    | .nan | .disjoint => some true
    | .none => some (!oEq o)
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

/-- Bool negation through a Cypher result (null stays null). -/
def negB : V F → V F
  | .bool x => .bool (!x)
  | v => v

/-- **`<>` is `NOT (=)`** (since #2917 7fdf06a0e; C's `AR_NE`): for every pair of values,
`<>` is null exactly when `=` is, and otherwise its negation. Before #2917, a map that
stopped at a disjoint pair (`{a:1}` vs `{a:'x'}`) made both `false` (value_order
`pre2917_map_neq_both_false`). -/
theorem neq_eq_not_eq (a b : V F) : evalCmp .neq a b = negB (evalCmp .eq a b) := by
  unfold evalCmp
  obtain ⟨o, fl⟩ := compareValue a b
  cases fl <;> rfl

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

/-- `Value::name` (value.rs:1384) on the scalar subset. `.other` stands for every
non-scalar variant and carries only its type order, not its name: the model does not
say what message a non-scalar operand produces, and no theorem below depends on it. -/
def V.name : V F → String
  | .null => "Null" | .bool _ => "Boolean" | .int _ => "Integer" | .flt _ => "Float"
  | .str _ => "String" | .other _ => "Other"

/-- The `AND`/`OR`/`XOR` operand type error (eval.rs:642, :669, :693; vector_expr.rs:456).
Since #3294 (#3292) it names the operand's type like C — `… but was {ir.name()}`; before,
it was `expected Bool but was {ir:?}` (the `Debug` dump of the value). -/
def boolMismatch (v : V F) : String :=
  s!"Type mismatch: expected Boolean or Null but was {v.name}"

/-- Per-row `AND` (`sc = false`) / `OR` (`sc = true`) of `eval_compound`
(eval.rs:629-700): children left to right, stop at the first child equal to
`sc`, remember a `null`, reject any other type. -/
def scalarAndOr (sc : Bool) (r : R) : List (R → Except String (V F)) → Bool → Except String (V F)
  | [], isNull => .ok (if isNull then .null else .bool (!sc))
  | c :: cs, isNull => match c r with
    | .error e => .error e
    | .ok (.bool b) => if b == sc then .ok (.bool sc) else scalarAndOr sc r cs isNull
    | .ok .null => scalarAndOr sc r cs true
    | .ok v => .error (boolMismatch v)

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
  | v :: _, _ => .error (boolMismatch v)

/-- #3294: a non-Boolean, non-null `AND`/`OR` operand fails with C's message naming its
type, e.g. `1 AND true` → `Type mismatch: expected Boolean or Null but was Integer`,
and so does an `XOR` operand. -/
theorem and_or_mismatch_names_type (sc : Bool) (r : R) (v : V F) (cs : List (R → Except String (V F)))
    (n : Bool) (hb : ∀ b, v ≠ .bool b) (hn : v ≠ .null) :
    scalarAndOr sc r (constChild v :: cs) n =
      .error s!"Type mismatch: expected Boolean or Null but was {v.name}" := by
  cases v with
  | bool b => exact absurd rfl (hb b)
  | null => exact absurd rfl hn
  | _ => rfl

theorem xor_mismatch_names_type (v : V F) (vs : List (V F)) (last : Option Bool)
    (hb : ∀ b, v ≠ .bool b) (hn : v ≠ .null) :
    scalarXor (v :: vs) last =
      .error s!"Type mismatch: expected Boolean or Null but was {v.name}" := by
  cases v with
  | bool b => exact absurd rfl (hb b)
  | null => exact absurd rfl hn
  | _ => rfl

example (r : R) : scalarAndOr false r [constChild (.int 1 : V F), constChild (.bool true)] false =
    .error "Type mismatch: expected Boolean or Null but was Integer" := rfl
example (r : R) : scalarAndOr true r [constChild (.bool false), constChild (.str "a" : V F)] false =
    .error "Type mismatch: expected Boolean or Null but was String" := rfl

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

/-! ## `^`: operand errors propagate and operands are type-checked (#2904)

Before #2904 (main 289c201fe) the `Pow` arm folded its children with `flat_map`,
which drops an `Err` (W-expr bug, issue #2902); that model is kept below as
`pre2904_*` history. -/

/-- The `apply_pow` type error (math.rs:328): `… but was {v.name()}`. -/
def powMismatch (v : V F) : String :=
  s!"Type mismatch: expected Integer, Float, or Null but was {v.name}"

def V.numOrNull : V F → Bool
  | .null | .int _ | .flt _ => true
  | _ => false

/-- `apply_pow` (math.rs:328, since #2904): a non-numeric, non-null operand (base
first) is a type error; else `null` if either side is null; else `powf` of the two
operands as `f64` (`get_numeric`). -/
def applyPow (pw : F → F → F) (a b : V F) : Except String (V F) :=
  if !a.numOrNull then .error (powMismatch a)
  else if !b.numOrNull then .error (powMismatch b)
  else match a, b with
    | .int x, .int y => .ok (.flt (pw (FM.ofInt x) (FM.ofInt y)))
    | .flt x, .flt y => .ok (.flt (pw x y))
    | .int x, .flt y => .ok (.flt (pw (FM.ofInt x) y))
    | .flt x, .int y => .ok (.flt (pw x (FM.ofInt y)))
    | _, _ => .ok .null

/-- One step of the `Pow` arm's `reduce(|acc, value| apply_pow(acc?, value?))`. -/
def powStep (pw : F → F → F) (acc v : Except String (V F)) : Except String (V F) :=
  match acc, v with
  | .error e, _ => .error e
  | .ok _, .error e => .error e
  | .ok x, .ok y => applyPow pw x y

/-- `ExprIR::Pow` in `eval_compound` (eval.rs:828-834, since #2904):
`children.map(eval_node).reduce(|acc, value| apply_pow(acc?, value?))`, then
`ok_or_else(..)??`. A single child is returned as is. -/
def powFold (pw : F → F → F) : List (Except String (V F)) → Except String (V F)
  | [] => .error "Pow operator requires at least one argument"
  | y :: ys => ys.foldl (powStep pw) y

/-- What every other operator does (`?` on each child first) — and C. -/
def powSpec (pw : F → F → F) (xs : List (Except String (V F))) : Except String (V F) :=
  match xs.mapM id with
  | .error e => .error e
  | .ok [] => .error "Pow operator requires at least one argument"
  | .ok (y :: ys) => ys.foldlM (applyPow pw) y

/-- `2 ^ (1/0)` is now the division error (was `2`, see `pre2904_pow_drops_error`). -/
theorem pow_propagates_error (pw : F → F → F) :
    powFold pw [.ok (.int 2), .error "Division by zero"] = .error "Division by zero" ∧
    powFold pw [.error "Division by zero", .ok (.int 2)] = .error "Division by zero" :=
  ⟨rfl, rfl⟩

theorem powStep_foldl_error (pw : F → F → F) (e : String) :
    ∀ ys : List (Except String (V F)), ys.foldl (powStep pw) (.error e) = .error e
  | [] => rfl
  | _ :: ys => powStep_foldl_error pw e ys

/-- On error-free operands the fold is `foldlM apply_pow`: it stops at the first
type error, exactly like the spec. -/
theorem powStep_foldl_ok (pw : F → F → F) :
    ∀ (ws : List (V F)) (a : V F),
      (ws.map Except.ok).foldl (powStep pw) (.ok a) = ws.foldlM (applyPow pw) a
  | [], a => rfl
  | w :: ws, a => by
    simp only [List.map_cons, List.foldl_cons, List.foldlM_cons]
    show List.foldl (powStep pw) (applyPow pw a w) (ws.map Except.ok) = (applyPow pw a w) >>= _
    cases applyPow pw a w with
    | error e => rw [powStep_foldl_error]; rfl
    | ok b => exact powStep_foldl_ok pw ws b

theorem mapM_ok (ws : List (V F)) :
    (ws.map (Except.ok : V F → Except String (V F))).mapM id = .ok ws := by
  induction ws with
  | nil => rfl
  | cons v vs ih => simp only [List.map_cons, List.mapM_cons, id]; rw [ih]; rfl

/-- On error-free operands main's fold *is* the spec, error for error. -/
theorem pow_agrees_without_errors (pw : F → F → F) (vs : List (V F)) :
    powFold pw (vs.map .ok) = powSpec pw (vs.map .ok) := by
  unfold powSpec
  rw [mapM_ok]
  cases vs with
  | nil => rfl
  | cons v vs => exact powStep_foldl_ok pw vs v

/-- The binary operator `a ^ b` is exactly the `?`-on-each-operand spec. -/
theorem pow_binary_eq_spec (pw : F → F → F) (a b : Except String (V F)) :
    powFold pw [a, b] = powSpec pw [a, b] := by
  cases a with
  | error e => rfl
  | ok x =>
    cases b with
    | error e => rfl
    | ok y => exact pow_agrees_without_errors pw [x, y]

/-- The optimizer's constant evaluator (every variable an error) no longer folds
`x ^ true` to `true`: the error reaches `is_constant_true`, which keeps the Filter. -/
theorem pow_constant_not_true (pw : F → F → F) :
    powFold pw [.error "Variable not found", .ok (.bool true)] = .error "Variable not found" := rfl

/-- `x ^ true` at runtime is now the C type error (it was `null` for every row). -/
theorem pow_bool_type_error (pw : F → F → F) (x : V F) (hx : x.numOrNull = true) :
    powFold pw [.ok x, .ok (.bool true)] =
      .error "Type mismatch: expected Integer, Float, or Null but was Boolean" := by
  cases x <;> simp_all [powFold, List.foldl, powStep, applyPow, V.numOrNull, powMismatch, V.name] <;> rfl

/-! ### History: the `flat_map` that dropped errors (fixed by #2904) -/

/-- `apply_pow` before #2904: anything non-numeric was `null`, never a type error. -/
def pre2904_applyPow (pw : F → F → F) : V F → V F → V F
  | .int a, .int b => .flt (pw (FM.ofInt a) (FM.ofInt b))
  | .flt a, .flt b => .flt (pw a b)
  | .int a, .flt b => .flt (pw (FM.ofInt a) b)
  | .flt a, .int b => .flt (pw a (FM.ofInt b))
  | _, _ => .null

/-- `ExprIR::Pow` before #2904: `children.flat_map(eval_node).reduce(apply_pow)`.
`flat_map` over a `Result` yields the `Ok` value and *nothing* for an `Err`. -/
def pre2904_powFold (pw : F → F → F) (xs : List (Except String (V F))) : Except String (V F) :=
  match xs.filterMap Except.toOption with
  | [] => .error "Pow operator requires at least one argument"
  | y :: ys => .ok (ys.foldl (pre2904_applyPow pw) y)

def pre2904_powSpec (pw : F → F → F) (xs : List (Except String (V F))) : Except String (V F) :=
  match xs.mapM id with
  | .error e => .error e
  | .ok [] => .error "Pow operator requires at least one argument"
  | .ok (y :: ys) => .ok (ys.foldl (pre2904_applyPow pw) y)

/-- **BUG (fixed by #2904)**: `2 ^ (1/0)` was `2` — the right operand's error
vanished and the left operand was returned *unpowered*. -/
theorem pre2904_pow_drops_error (pw : F → F → F) :
    pre2904_powFold pw [.ok (.int 2), .error "Division by zero"] = .ok (.int 2) ∧
    pre2904_powSpec pw [.ok (.int 2), .error "Division by zero"] = .error "Division by zero" :=
  ⟨rfl, rfl⟩

/-- **BUG, consequence (fixed by #2904)**: the constant evaluator folded `x ^ true`
to `true`, so `eliminate_true_filters` deleted `WHERE x ^ true`. -/
theorem pre2904_pow_constant_true (pw : F → F → F) :
    pre2904_powFold pw [.error "Variable not found", .ok (.bool true)] = .ok (.bool true) := rfl

/-- Before #2904, error dropping was the only difference from `?`-semantics. -/
theorem pre2904_pow_agrees_without_errors (pw : F → F → F) (vs : List (V F)) :
    pre2904_powFold pw (vs.map .ok) = pre2904_powSpec pw (vs.map .ok) := by
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
  unfold pre2904_powFold pre2904_powSpec
  rw [h1, h2]
  cases vs <;> rfl

/-! ## `UNWIND range(..)`: the lazy iterator before #2899 (BUG, fixed)

Kept for history: before #2899 (`e63165cb5`) `RangeIter` stored `end`, the step's
magnitude and a direction flag, and stepped past the last element. -/

/-- `RangeIter::next` before #2899 (eval.rs:108-122 at 8621212fc) in a release build (wrapping `+=`),
run for at most `fuel` items. `step` is stored as `step.unsigned_abs() as i64`
and negated for a descending range — `wrap` models both casts. -/
def pre2899_rangeIterRel (up : Bool) (stepAbs endv : Int) : Nat → Int → List Int
  | 0, _ => []
  | fuel + 1, cur =>
    if (up && decide (cur > endv)) || (!up && decide (cur < endv)) then []
    else cur :: pre2899_rangeIterRel up stepAbs endv fuel
      (wrap (cur + (if up then wrap stepAbs else wrap (-(wrap stepAbs)))))

/-- The same iterator in a debug build: `+=` and unary `-` are checked and
panic ("attempt to add/negate with overflow"). -/
def pre2899_rangeIterDbg (up : Bool) (stepAbs endv : Int) : Nat → Int → Except String (List Int)
  | 0, _ => .ok []
  | fuel + 1, cur =>
    if (up && decide (cur > endv)) || (!up && decide (cur < endv)) then .ok []
    else
      let st := wrap stepAbs   -- `step.unsigned_abs() as i64`
      let d := if up then st else -st
      if !up && ¬ InI64' (-st) then .error "attempt to negate with overflow"
      else if ¬ InI64' (cur + d) then .error "attempt to add with overflow"
      else (pre2899_rangeIterDbg up stepAbs endv fuel (cur + d)).map (cur :: ·)
where
  InI64' (x : Int) : Bool := decide (I64MIN ≤ x) && decide (x ≤ I64MAX)

/-- The mathematical sequence, which is what the eager `range()` function
(math.rs:251, `(start..=end).step_by(step)`) returns. -/
def rangeMath (up : Bool) (stepAbs endv : Int) : Nat → Int → List Int
  | 0, _ => []
  | fuel + 1, cur =>
    if (up && decide (cur > endv)) || (!up && decide (cur < endv)) then []
    else cur :: rangeMath up stepAbs endv fuel (cur + (if up then stepAbs else -stepAbs))

/-- Before #2899: when the step past `end` is still representable, the lazy iterator was the
mathematical range (and so agrees with the eager `range()`). -/
theorem pre2899_rangeIter_ok_up (stepAbs endv : Int) (hs : 0 < stepAbs) (hs' : stepAbs ≤ I64MAX)
    (hroom : endv + stepAbs ≤ I64MAX) :
    ∀ (fuel : Nat) (cur : Int), I64MIN ≤ cur →
      pre2899_rangeIterRel true stepAbs endv fuel cur = rangeMath true stepAbs endv fuel cur
  | 0, _, _ => rfl
  | fuel + 1, cur, hc => by
    simp only [pre2899_rangeIterRel, rangeMath, Bool.true_and, Bool.not_true, Bool.false_and,
      Bool.or_false, ↓reduceIte]
    by_cases h : cur > endv
    · simp [h]
    · simp only [h, decide_false, Bool.false_eq_true, ↓reduceIte, List.cons.injEq, true_and]
      have hin : InI64 (cur + stepAbs) := by simp only [InI64, I64MIN, I64MAX] at *; omega
      have hst : InI64 stepAbs := by simp only [InI64, I64MIN, I64MAX] at *; omega
      rw [wrap_id hst, wrap_id hin]
      exact pre2899_rangeIter_ok_up stepAbs endv hs hs' hroom fuel (cur + stepAbs)
        (by unfold I64MIN at *; omega)

/-- **BUG (release; fixed by #2899)**: `UNWIND range(i64::MAX - 1, i64::MAX)` yields
`MAX-1, MAX, MIN, MIN+1, …` and never stops (`cur > MAX` is never true);
exactly the rows `redis-cli GRAPH.QUERY g "UNWIND range(9223372036854775806,
9223372036854775807) AS x RETURN x LIMIT 5"` returns. The eager `range()` gives
the two elements. -/
theorem pre2899_rangeIter_wraps :
    pre2899_rangeIterRel true 1 I64MAX 5 (I64MAX - 1)
      = [I64MAX - 1, I64MAX, I64MIN, I64MIN + 1, I64MIN + 2] ∧
    rangeMath true 1 I64MAX 5 (I64MAX - 1) = [I64MAX - 1, I64MAX] := by
  decide

/-- **BUG (debug; fixed by #2899)**: the same query panics. -/
theorem pre2899_rangeIter_panics :
    pre2899_rangeIterDbg true 1 I64MAX 5 (I64MAX - 1) = .error "attempt to add with overflow" := by
  rfl

/-- **BUG (debug; fixed by #2899)**: descending to `i64::MIN` panics too, and a step of
`i64::MIN` panics on the negation before the second item. -/
theorem pre2899_rangeIter_panics_down :
    pre2899_rangeIterDbg false 1 I64MIN 5 (I64MIN + 1) = .error "attempt to add with overflow" ∧
    pre2899_rangeIterDbg false (-I64MIN) (-10) 5 0 = .error "attempt to negate with overflow" :=
  ⟨rfl, rfl⟩

/-! ## `UNWIND range(..)` after #2899: exactly the eager range, never overflows -/

/-- `RangeIter::next` after #2899 (`eval.rs:114-127`), debug build (checked `+=`):
`remaining` counts the elements still due and `current` advances only while
another one is (`if self.remaining > 0 { self.current += self.step }`). -/
def rangeIter (step : Int) : Nat → Int → Except String (List Int)
  | 0, _ => .ok []
  | r + 1, cur =>
    if r = 0 then .ok [cur]
    else if I64MIN ≤ cur + step ∧ cur + step ≤ I64MAX then
      (rangeIter step r (cur + step)).map (cur :: ·)
    else .error "attempt to add with overflow"

/-- The same in a release build (wrapping `+=`). -/
def rangeIterRel (step : Int) : Nat → Int → List Int
  | 0, _ => []
  | r + 1, cur => if r = 0 then [cur] else cur :: rangeIterRel step r (wrap (cur + step))

/-- `length` as `eval_iter_expr` (`eval.rs:1224-1233`) computes it (in `i128`/`u128`,
so exactly): `|end - start| / |step| + 1`; it becomes `remaining`. -/
def rangeLen (a e s : Int) : Nat := (e - a).natAbs / s.natAbs + 1

/-- The eager `range()` list: `start + k*step` for `k < n`. -/
def rangeList (a s : Int) (n : Nat) : List Int := (List.range n).map fun (k : Nat) => a + (k : Int) * s

theorem rangeIter_of_inRange (step : Int) :
    ∀ (r : Nat) (cur : Int), (∀ k : Nat, k < r → InI64 (cur + k * step)) →
      rangeIter step r cur = .ok (rangeList cur step r) ∧
      rangeIterRel step r cur = rangeList cur step r
  | 0, _, _ => ⟨rfl, rfl⟩
  | r + 1, cur, h => by
    by_cases hr : r = 0
    · subst hr; simp [rangeIter, rangeIterRel, rangeList]
    · have h1 : InI64 (cur + step) := by have := h 1 (by omega); simpa using this
      have ih := rangeIter_of_inRange step r (cur + step) (fun k hk => by
        have := h (k + 1) (by omega)
        have e : cur + step + (k : Int) * step = cur + ((k + 1 : Nat) : Int) * step := by
          push_cast; rw [Int.add_mul]; omega
        rwa [e])
      have hl : rangeList cur step (r + 1) = cur :: rangeList (cur + step) step r := by
        simp only [rangeList, List.range_succ_eq_map, List.map_cons, List.map_map]
        congr 1
        · simp
        · apply List.map_congr_left; intro k _
          simp only [Function.comp]; push_cast; rw [Int.add_mul]; omega
      have hw : wrap (cur + step) = cur + step := wrap_id h1
      have h1' : I64MIN ≤ cur + step ∧ cur + step ≤ I64MAX := h1
      refine ⟨?_, ?_⟩
      · simp only [rangeIter, hr, ↓reduceIte, h1', and_self, ih.1, hl]; rfl
      · simp only [rangeIterRel, hr, ↓reduceIte, hw, ih.2, hl]

/-- Arithmetic facts about the element count `length` the caller computes. -/
theorem rangeLen_bounds (a e s : Int) (hs : s ≠ 0)
    (hdir : ¬ ((a > e ∧ s > 0) ∨ (a < e ∧ s < 0))) :
    (∀ k : Nat, k < rangeLen a e s → min a e ≤ a + k * s ∧ a + k * s ≤ max a e) ∧
    (s > 0 → a + (rangeLen a e s : Int) * s > e) ∧
    (s < 0 → a + (rangeLen a e s : Int) * s < e) := by
  unfold rangeLen
  have hS : 0 < s.natAbs := Int.natAbs_pos.mpr hs
  have hqd := Nat.div_mul_le_self (e - a).natAbs s.natAbs
  have hlt := Nat.lt_mul_div_succ (e - a).natAbs hS
  have hlt' : (e - a).natAbs < ((e - a).natAbs / s.natAbs + 1) * s.natAbs := by
    rw [Nat.mul_comm]; exact hlt
  generalize hq : (e - a).natAbs / s.natAbs = q at hqd hlt'
  have hD : ((e - a).natAbs : Int) = if e ≥ a then e - a else a - e := by split <;> omega
  have hSi : (s.natAbs : Int) = if s > 0 then s else -s := by split <;> omega
  have key : ∀ k : Nat, (k : Int) * s = if s > 0 then ((k * s.natAbs : Nat) : Int)
      else -((k * s.natAbs : Nat) : Int) := by
    intro k; push_cast; rw [hSi]; split
    · rfl
    · rw [Int.mul_neg, Int.neg_neg]
  refine ⟨fun k hk => ?_, fun hp => ?_, fun hn => ?_⟩
  · have hkq : k * s.natAbs ≤ q * s.natAbs := Nat.mul_le_mul_right _ (by omega)
    have hc : ((k * s.natAbs : Nat) : Int) ≤ ((e - a).natAbs : Int) := by exact_mod_cast Nat.le_trans hkq hqd
    rw [key k]; rw [hD] at hc
    have h0 : (0 : Int) ≤ ((k * s.natAbs : Nat) : Int) := Int.natCast_nonneg _
    split at hc <;> split <;> omega
  · have hc : ((e - a).natAbs : Int) < (((q + 1) * s.natAbs : Nat) : Int) := by exact_mod_cast hlt'
    rw [key (q + 1)]; rw [hD] at hc
    have h0 : (0 : Int) ≤ (((q+1) * s.natAbs : Nat) : Int) := Int.natCast_nonneg _
    simp only [hp, ↓reduceIte]; push_cast at hc ⊢
    split at hc <;> omega
  · have hc : ((e - a).natAbs : Int) < (((q + 1) * s.natAbs : Nat) : Int) := by exact_mod_cast hlt'
    rw [key (q + 1)]; rw [hD] at hc
    have hnp : ¬ s > 0 := by omega
    simp only [hnp, ↓reduceIte]; push_cast at hc ⊢
    split at hc <;> omega

/-- `rangeMath` with enough fuel is the explicit list of `m` elements. -/
theorem rangeMath_eq (up : Bool) (stepAbs endv : Int) :
    ∀ (m fuel : Nat) (cur : Int), m ≤ fuel →
      (∀ k : Nat, k < m → ¬ ((up && decide (cur + k * (if up then stepAbs else -stepAbs) > endv)) ||
          (!up && decide (cur + k * (if up then stepAbs else -stepAbs) < endv))) = true) →
      ((up && decide (cur + m * (if up then stepAbs else -stepAbs) > endv)) ||
          (!up && decide (cur + m * (if up then stepAbs else -stepAbs) < endv))) = true →
      rangeMath up stepAbs endv fuel cur = rangeList cur (if up then stepAbs else -stepAbs) m
  | 0, 0, _, _, _, _ => rfl
  | 0, fuel + 1, cur, _, _, hout => by
    simp only [Int.natCast_zero, Int.zero_mul, Int.add_zero] at hout
    simp [rangeMath, hout, rangeList]
  | m + 1, 0, _, h, _, _ => absurd h (by omega)
  | m + 1, fuel + 1, cur, hf, hin, hout => by
    have h0 := hin 0 (by omega)
    simp only [Int.natCast_zero, Int.zero_mul, Int.add_zero] at h0
    have ih := rangeMath_eq up stepAbs endv m fuel (cur + (if up then stepAbs else -stepAbs)) (by omega)
      (fun k hk => by
        have := hin (k + 1) (by omega)
        have e : cur + (if up then stepAbs else -stepAbs) + (k : Int) * (if up then stepAbs else -stepAbs)
            = cur + ((k + 1 : Nat) : Int) * (if up then stepAbs else -stepAbs) := by
          push_cast; rw [Int.add_mul]; omega
        rwa [e])
      (by have e : cur + (if up then stepAbs else -stepAbs) + (m : Int) * (if up then stepAbs else -stepAbs)
            = cur + ((m + 1 : Nat) : Int) * (if up then stepAbs else -stepAbs) := by
            push_cast; rw [Int.add_mul]; omega
          rwa [e])
    have hl : rangeList cur (if up then stepAbs else -stepAbs) (m + 1) =
        cur :: rangeList (cur + (if up then stepAbs else -stepAbs)) (if up then stepAbs else -stepAbs) m := by
      simp only [rangeList, List.range_succ_eq_map, List.map_cons, List.map_map]
      congr 1
      · simp
      · apply List.map_congr_left; intro k _
        simp only [Function.comp]; push_cast; rw [Int.add_mul]; omega
    rw [hl, rangeMath]
    simp only [Bool.not_eq_true] at h0
    rw [h0]; simp only [Bool.false_eq_true, ↓reduceIte, ih]

/-- **#2899 (fixed)**: for every `i64` `start`, `end`, nonzero `step` the
caller does not reject, the lazy iterator yields exactly the eager range and
never overflows, in debug and release. -/
theorem rangeIter_correct (a e s : Int) (ha : InI64 a) (he : InI64 e) (hs : s ≠ 0)
    (hdir : ¬ ((a > e ∧ s > 0) ∨ (a < e ∧ s < 0))) :
    rangeIter s (rangeLen a e s) a = .ok (rangeList a s (rangeLen a e s)) ∧
    rangeIterRel s (rangeLen a e s) a = rangeList a s (rangeLen a e s) ∧
    (∀ fuel, rangeLen a e s ≤ fuel →
      rangeMath (decide (s > 0)) s.natAbs e fuel a = rangeList a s (rangeLen a e s)) := by
  obtain ⟨hb, hup, hdn⟩ := rangeLen_bounds a e s hs hdir
  have hin : ∀ k : Nat, k < rangeLen a e s → InI64 (a + k * s) := by
    intro k hk; have := hb k hk; unfold InI64 at *; omega
  refine ⟨(rangeIter_of_inRange s _ a hin).1, (rangeIter_of_inRange s _ a hin).2, fun fuel hf => ?_⟩
  have hsa : (if decide (s > 0) then (s.natAbs : Int) else -(s.natAbs : Int)) = s := by
    by_cases h : s > 0 <;> simp [h] <;> omega
  have := rangeMath_eq (decide (s > 0)) s.natAbs e (rangeLen a e s) fuel a hf
    (fun k hk => by rw [hsa]; have := hb k hk; by_cases h : s > 0 <;> simp [h] <;> omega)
    (by rw [hsa]; by_cases h : s > 0
        · simp [h, hup h]
        · simp [h]; exact hdn (by omega))
  rw [this, hsa]

/-- `RangeIter::size_hint` (`eval.rs:129-132`) is `(remaining, Some(remaining))`:
exact, because the iterator yields exactly `remaining` items. -/
def rangeIterSizeHint (r : Nat) : Nat × Option Nat := (r, some r)

theorem rangeIter_sizeHint_exact (step : Int) :
    ∀ (r : Nat) (cur : Int), (rangeIterRel step r cur).length = (rangeIterSizeHint r).1 ∧
      (rangeIterSizeHint r).2 = some (rangeIterRel step r cur).length
  | 0, _ => ⟨rfl, rfl⟩
  | r + 1, cur => by
    by_cases hr : r = 0
    · subst hr; simp [rangeIterRel, rangeIterSizeHint]
    · have ih := (rangeIter_sizeHint_exact step r (wrap (cur + step))).1
      simp only [rangeIterSizeHint] at ih ⊢
      simp [rangeIterRel, hr, ih]

/-- The four cases of `range_iter_stops_at_i64_bounds` (eval.rs tests): the last
element may sit on `i64::MAX`/`i64::MIN`, and a step of `i64::MIN` is fine
(the pre-#2899 negation is gone). -/
theorem rangeIter_bounds_examples :
    rangeIter 1 2 (I64MAX - 1) = .ok [I64MAX - 1, I64MAX] ∧
    rangeIter (-1) 2 (I64MIN + 1) = .ok [I64MIN + 1, I64MIN] ∧
    rangeIter I64MIN 1 0 = .ok [0] ∧
    rangeIter I64MAX 2 (-1) = .ok [-1, I64MAX - 1] := by
  refine ⟨?_, ?_, ?_, ?_⟩ <;> rfl

/-- **#2894 fixed by #2899**: `UNWIND range(i64::MAX - 1, i64::MAX)` now yields the two
elements the eager `range()` gives (contrast `pre2899_rangeIter_wraps`). -/
theorem rangeIter_max_fixed :
    rangeIter 1 (rangeLen (I64MAX - 1) I64MAX 1) (I64MAX - 1) = .ok [I64MAX - 1, I64MAX] ∧
    rangeIter (-1) (rangeLen (I64MIN + 1) I64MIN (-1)) (I64MIN + 1) = .ok [I64MIN + 1, I64MIN] :=
  ⟨rfl, rfl⟩

/-- `eval_iter_expr`'s `range` arm (`eval.rs:1205-1243`) wired to the real iterator:
whenever it emits rows, they are exactly what `RangeIter` yields from
`remaining = length`, with no overflow (debug) — the same list as the eager `range()`. -/
theorem evalIter_range_is_iter (isList : Int → Option (List Int)) (isNull : Int → Bool)
    (a e s : Int) (ha : InI64 a) (he : InI64 e) (l : List Int)
    (h : Eval.evalIterExpr isList isNull (fun a e s => rangeList a s (rangeLen a e s))
      (.range a e s) = .ok (some l)) :
    rangeIter s (rangeLen a e s) a = .ok l ∧ rangeLen a e s ≤ 2 ^ 32 - 1 := by
  unfold Eval.evalIterExpr at h
  by_cases hs : s = 0
  · simp [hs] at h
  · by_cases hdir : (a > e ∧ s > 0) ∨ (a < e ∧ s < 0)
    · simp [hs, hdir] at h
    · by_cases hlen : (e - a).natAbs / s.natAbs + 1 > 2 ^ 32 - 1
      · simp [hs, hdir, hlen] at h
      · simp only [hs, hdir, hlen, ↓reduceIte, Except.ok.injEq, Option.some.injEq] at h
        subst h
        exact ⟨(rangeIter_correct a e s ha he hs hdir).1, by unfold rangeLen; omega⟩

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
