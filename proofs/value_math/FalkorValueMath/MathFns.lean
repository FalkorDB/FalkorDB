import FalkorValueMath.TypeCheck
/-!
# `graph/src/runtime/functions/math.rs`

Floats are abstracted to their IEEE-754 *class* (`FC`): NaN, ±∞, ±0, positive or
negative finite non-zero. The class-level behaviour of `sqrt`, `fabs`, `>= 0.0`, `== 0.0`
and `signum` is fixed by IEEE 754-2019 §5.4.1/§5.5.1/§5.11 and C99 Annex F.9.4.5 —
these are *definitions* here (`ieeeSqrt`, …), and Lean's own `Float` (the same libm) is
used to spot-check them with `#eval`. Functions whose result class depends on magnitude
(`exp`, `ln`, `floor`, …) are parameters (`MathOps`); only their *type* behaviour and the
branches the Rust code adds around them are proved.

`Res` distinguishes an `Err` return from a panic (`unreachable!()`).
-/

namespace ValueMath

inductive FC where
  | nan | pinf | ninf | pzero | nzero | pos | neg
  deriving DecidableEq, Repr

/-- `i as f64` at class level (exact sign; |i| ≤ 2^63 is finite). -/
def fcOfInt (i : Int) : FC := if i = 0 then .pzero else if i > 0 then .pos else .neg

/-- IEEE `x >= 0.0`. -/
def fcGe0 : FC → Bool
  | .pzero | .nzero | .pos | .pinf => true
  | _ => false

/-- IEEE `x == 0.0`. -/
def fcEq0 : FC → Bool
  | .pzero | .nzero => true
  | _ => false

/-- IEEE `x < 0.0`. -/
def fcLt0 : FC → Bool
  | .neg | .ninf => true
  | _ => false

/-- IEEE `squareRoot` (C99 F.9.4.5): `sqrt(-0) = -0`, negative → NaN. -/
def ieeeSqrt : FC → FC
  | .nan => .nan | .pinf => .pinf | .ninf => .nan | .pzero => .pzero | .nzero => .nzero
  | .pos => .pos | .neg => .nan

/-- IEEE `abs` (clears the sign bit). -/
def ieeeAbs : FC → FC
  | .ninf => .pinf | .nzero => .pzero | .neg => .pos | x => x

/-- `x * -1` at class level. -/
def fcNeg : FC → FC
  | .pinf => .ninf | .ninf => .pinf | .pzero => .nzero | .nzero => .pzero
  | .pos => .neg | .neg => .pos | .nan => .nan

/-- Rust `f64::signum`: `±1` for everything but NaN (including `±0`). -/
def fcSignum : FC → FC
  | .nan => .nan
  | .pinf | .pos | .pzero => .pos
  | .ninf | .neg | .nzero => .neg

#eval (Float.sqrt (-0.0), Float.sqrt (-1.0), Float.abs (-0.0), Float.log 0.0, Float.log (-1.0))

inductive Res where
  | ok (v : V FC)
  | err (msg : String)
  | panic

/-- Magnitude-dependent libm functions (class in, class out). -/
structure MathOps where
  ceil : FC → FC
  floor : FC → FC
  round : FC → FC
  exp : FC → FC
  ln : FC → FC
  log10 : FC → FC

variable (m : MathOps)

/-- `abs` (math.rs:46). -/
def mAbs : V FC → Res
  | .int n => if n = I64MIN then .err "ArgumentError: integer overflow in abs()" else .ok (.int n.natAbs)
  | .float f => .ok (.float (ieeeAbs f))
  | .null => .ok .null
  | _ => .panic

def mCeil : V FC → Res
  | .int n => .ok (.int n) | .float f => .ok (.float (m.ceil f)) | .null => .ok .null | _ => .panic
def mFloor : V FC → Res
  | .int n => .ok (.int n) | .float f => .ok (.float (m.floor f)) | .null => .ok .null | _ => .panic
def mRound : V FC → Res
  | .int n => .ok (.int n) | .float f => .ok (.float (m.round f)) | .null => .ok .null | _ => .panic
def mExp : V FC → Res
  | .int n => .ok (.float (m.exp (fcOfInt n))) | .float f => .ok (.float (m.exp f))
  | .null => .ok .null | _ => .panic
def mLog : V FC → Res
  | .int n => .ok (.float (m.ln (fcOfInt n))) | .float f => .ok (.float (m.ln f))
  | .null => .ok .null | _ => .panic
def mLog10 : V FC → Res
  | .int n => .ok (.float (m.log10 (fcOfInt n))) | .float f => .ok (.float (m.log10 f))
  | .null => .ok .null | _ => .panic

/-- `sign` (math.rs:210). -/
def mSign : V FC → Res
  | .int n => .ok (.int (signumI n))
  | .float f => if fcEq0 f then .ok (.int 0) else .ok (.float (fcSignum f))
  | .null => .ok .null
  | _ => .panic

/-- `sqrt` (math.rs:228): an explicit negative check before `f64::sqrt`. -/
def mSqrt : V FC → Res
  | .int n => if n < 0 then .ok (.float .nan) else .ok (.float (ieeeSqrt (fcOfInt n)))
  | .float f => if fcGe0 f then .ok (.float (ieeeSqrt f)) else .ok (.float .nan)
  | .null => .ok .null
  | _ => .panic

def unaryFns : List (V FC → Res) :=
  [mAbs, mCeil m, mFloor m, mRound m, mExp m, mLog m, mLog10 m, mSign, mSqrt]

def Res.isPanic : Res → Bool
  | .panic => true
  | _ => false

/-- **No numeric function panics after argument validation** (their `_ =>
unreachable!()` arms are dead): validation against `Int | Float | Null` leaves only the
three handled variants. -/
theorem unary_no_panic (v : V FC) (h : validateArgsType (.fixed [numArg]) [v] = none) :
    ∀ f ∈ unaryFns m, f v ≠ .panic := by
  rcases numeric_body_cases v h with ⟨i, rfl⟩ | ⟨f, rfl⟩ | rfl <;>
    intro g hg <;> simp [unaryFns] at hg <;>
    rcases hg with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;>
    simp [mAbs, mCeil, mFloor, mRound, mExp, mLog, mLog10, mSign, mSqrt] <;>
    (split <;> simp)

/-- Integer inputs of `ceil`/`floor`/`round` are returned unchanged and stay Integers
(no f64 round trip, so `ceil(2^53+1) = 2^53+1`; C agrees, live). -/
theorem int_rounding_identity (n : Int) :
    mCeil m (.int n) = .ok (.int n) ∧ mFloor m (.int n) = .ok (.int n) ∧ mRound m (.int n) = .ok (.int n) :=
  ⟨rfl, rfl, rfl⟩

/-- The pre-check in `sqrt` never changes the IEEE answer (class-wise): Rust's
`sqrt(x)` is exactly `f64::sqrt(x)` and C's `sqrt(x)` on Floats. -/
theorem sqrt_float_eq_ieee (f : FC) : mSqrt (.float f) = .ok (.float (ieeeSqrt f)) := by
  cases f <;> rfl

/-- …and on Integers it is `sqrt(toFloat(n))`. -/
theorem sqrt_int_eq_sqrt_float (n : Int) : mSqrt (.int n) = mSqrt (.float (fcOfInt n)) := by
  unfold mSqrt fcOfInt
  by_cases h0 : n = 0
  · subst h0; rfl
  · by_cases hp : n > 0
    · simp [h0, hp, fcGe0]; omega
    · have : n < 0 := by omega
      simp [h0, hp, this, fcGe0]

/-- `sqrt(-0.0) = -0.0` (both engines print `-0`, live). -/
theorem sqrt_nzero : mSqrt (.float .nzero) = .ok (.float .nzero) := rfl

/-- `sqrt` is NaN exactly on NaN and on negative non-zero inputs. -/
theorem sqrt_nan_iff (f : FC) : mSqrt (.float f) = .ok (.float .nan) ↔ (f = .nan ∨ f = .neg ∨ f = .ninf) := by
  cases f <;> simp [mSqrt, fcGe0, ieeeSqrt]

/-! ## `abs` against C -/

/-- C `AR_ABS`: `if (x < 0) return x * -1; return x` (numeric_funcs.c). -/
def cAbs : V FC → Res
  | .int n => .ok (.int (cAbsInt n))
  | .float f => .ok (.float (if fcLt0 f then fcNeg f else f))
  | .null => .ok .null
  | _ => .panic

/-- Rust and C `abs` agree on every Float class except `-0.0` (Rust `0`, C `-0`; live)
and on every Integer except `i64::MIN` (Rust errors, C returns `MIN`; live). -/
theorem abs_agree_except (v : V FC) (h1 : v ≠ .float .nzero) (h2 : v ≠ .int I64MIN)
    (h3 : ∀ n, v = .int n → InI64 n) :
    mAbs v = cAbs v := by
  cases v with
  | int n =>
    have : n ≠ I64MIN := fun e => h2 (by rw [e])
    have hr := h3 n rfl
    unfold InI64 at hr
    simp only [mAbs, cAbs, this, if_false, cAbsInt]
    unfold wMul wrap
    split
    · congr 2; omega
    · congr 2; omega
  | float f => cases f <;> simp_all [mAbs, cAbs, ieeeAbs, fcLt0, fcNeg]
  | _ => rfl

theorem abs_nzero_diverges : mAbs (.float .nzero) = .ok (.float .pzero) ∧ cAbs (.float .nzero) = .ok (.float .nzero) :=
  ⟨rfl, rfl⟩

theorem abs_min_diverges :
    mAbs (.int I64MIN) = .err "ArgumentError: integer overflow in abs()" ∧ cAbs (.int I64MIN) = .ok (.int I64MIN) :=
  ⟨rfl, by simp [cAbs, cAbs_min]⟩

/-! ## `sign` (also seen: issue #2906) -/

/-- `sign` of a non-zero Float is a *Float* (`-1.0`/`1.0`; NaN for NaN), while the
declared return type is `Integer | Null` and C returns an Integer. -/
theorem sign_float_is_float (f : FC) (h : fcEq0 f = false) : mSign (.float f) = .ok (.float (fcSignum f)) := by
  simp [mSign, h]

theorem sign_zero_is_int (f : FC) (h : fcEq0 f = true) : mSign (.float f) = .ok (.int 0) := by
  simp [mSign, h]

theorem sign_int (n : Int) : mSign (.int n) = .ok (.int (signumI n)) := rfl

/-! ## `log` of non-positive Integers -/

/-- `log(0)` goes through `ln(+0)`, `log(-n)` through `ln(negative)`: with IEEE
`ln(±0) = -∞` and `ln(neg) = NaN` this is `-inf`/`nan` in both engines (live). -/
theorem log_int_nonpos (n : Int) (hn : n ≤ 0) (hz : m.ln .pzero = .ninf) (hneg : m.ln .neg = .nan) :
    mLog m (.int n) = .ok (.float (if n = 0 then .ninf else .nan)) := by
  unfold mLog fcOfInt
  by_cases h : n = 0
  · simp [h, hz]
  · have : ¬ n > 0 := by omega
    simp [h, this, hneg]

/-! ## `coalesce` (math.rs:317) -/

/-- `OrderedEnum::order` (value.rs:1178). -/
def order : V FC → Nat
  | .null => 2 ^ 15 | .bool _ => 2 ^ 12 | .int _ => 2 ^ 13 | .float _ => 2 ^ 14
  | .str _ => 2 ^ 11 | .list _ => 2 ^ 3 | .map _ => 2 ^ 0 | .node _ => 2 ^ 1
  | .rel _ => 2 ^ 2 | .path _ => 2 ^ 4 | .point .. => 2 ^ 5 | .datetime _ => 2 ^ 6
  | .date _ => 2 ^ 7 | .time _ => 2 ^ 8 | .duration _ => 2 ^ 10 | .vec _ => 2 ^ 18

/-- `compare_value(x, Null).0` (value.rs:1224): no same-variant or Int/Float arm matches
a `Null` right operand unless `x` is `Null`, so the `(_, Null)` arm compares type
orders. -/
def cmpWithNull (x : V FC) : Ordering := compare (order x) (order .null)

/-- `*arg == Value::Null` is exactly "is `Null`": no other variant shares Null's order. -/
theorem eq_null_iff (x : V FC) : cmpWithNull x = .eq ↔ x = .null := by
  cases x <;> simp [cmpWithNull, order, compare, compareOfLessAndEq] <;> decide

def coalesce : List (V FC) → V FC
  | [] => .null
  | a :: rest => if cmpWithNull a = .eq then coalesce rest else a

/-- `coalesce` returns the first non-null argument, or null. -/
theorem coalesce_spec (args : List (V FC)) :
    coalesce args = ((args.find? (fun a => !a.isNull)).getD .null) := by
  induction args with
  | nil => rfl
  | cons a rest ih =>
    by_cases h : a = .null
    · subst h; simp [coalesce, cmpWithNull, ih, V.isNull]
    · have : cmpWithNull a ≠ .eq := fun e => h ((eq_null_iff a).1 e)
      have hn : a.isNull = false := by cases a <;> simp_all [V.isNull]
      simp [coalesce, this, hn]

/-! ## `randomUUID` (math.rs:143) -/

def hi (b : Nat) : Nat := b / 16
def lo (b : Nat) : Nat := b % 16

/-- `{:0wx}` of `u32/u16::from_be_bytes(bytes)`: `w` nibbles, most significant first. -/
def hexW (w n : Nat) : List Nat := (List.range w).map (fun i => (n / 16 ^ (w - 1 - i)) % 16)

theorem hex4_be16 (b0 b1 : Nat) (h0 : b0 < 256) (h1 : b1 < 256) :
    hexW 4 (b0 * 256 + b1) = [hi b0, lo b0, hi b1, lo b1] := by
  have hr : List.range 4 = [0, 1, 2, 3] := rfl
  rw [hexW, hr]
  simp only [List.map, List.cons.injEq, hi, lo, and_true]
  refine ⟨?_, ?_, ?_, ?_⟩ <;> omega

theorem hex8_be32 (b0 b1 b2 b3 : Nat) (h0 : b0 < 256) (h1 : b1 < 256) (h2 : b2 < 256) (h3 : b3 < 256) :
    hexW 8 (((b0 * 256 + b1) * 256 + b2) * 256 + b3) =
      [hi b0, lo b0, hi b1, lo b1, hi b2, lo b2, hi b3, lo b3] := by
  have hr : List.range 8 = [0, 1, 2, 3, 4, 5, 6, 7] := rfl
  rw [hexW, hr]
  simp only [List.map, List.cons.injEq, hi, lo, and_true]
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩ <;> omega

set_option maxRecDepth 100000 in
/-- Version byte: `(b & 0x0F) | 0x40` always has high nibble 4. -/
theorem uuid_version_nibble : ∀ b : Fin 256, hi (Nat.lor (Nat.land b.val 0x0F) 0x40) = 4 := by decide

set_option maxRecDepth 100000 in
/-- Variant byte: `(b & 0x3F) | 0x80` has high nibble 8, 9, a or b (RFC 4122 variant). -/
theorem uuid_variant_nibble : ∀ b : Fin 256, 8 ≤ hi (Nat.lor (Nat.land b.val 0x3F) 0x80) ∧
    hi (Nat.lor (Nat.land b.val 0x3F) 0x80) ≤ 11 := by decide

/-- The 36-symbol layout, `none` = '-': nibbles of bytes 0-3, 4-5, 6-7, 8-9, 10-15. -/
def uuidLayout (bs : List Nat) : List (Option Nat) :=
  let n := fun (i : Nat) => bs.getD i 0
  (([hi (n 0), lo (n 0), hi (n 1), lo (n 1), hi (n 2), lo (n 2), hi (n 3), lo (n 3)].map some) ++ [none] ++
   ([hi (n 4), lo (n 4), hi (n 5), lo (n 5)].map some) ++ [none] ++
   ([hi (n 6), lo (n 6), hi (n 7), lo (n 7)].map some) ++ [none] ++
   ([hi (n 8), lo (n 8), hi (n 9), lo (n 9)].map some) ++ [none] ++
   ((List.range 12).map (fun i => some (if i % 2 = 0 then hi (n (10 + i / 2)) else lo (n (10 + i / 2))))))

/-- Length 36, hyphens at 8/13/18/23, `4` at 14, variant nibble at 19 (the live
`substring(randomUUID(), 14, 1) = '4'`). -/
theorem uuid_shape (bs : List Nat) (b6 b8 : Fin 256)
    (h6 : bs.getD 6 0 = Nat.lor (Nat.land b6.val 0x0F) 0x40)
    (h8 : bs.getD 8 0 = Nat.lor (Nat.land b8.val 0x3F) 0x80) :
    (uuidLayout bs).length = 36 ∧ (uuidLayout bs)[8]? = some none ∧ (uuidLayout bs)[13]? = some none ∧
    (uuidLayout bs)[18]? = some none ∧ (uuidLayout bs)[23]? = some none ∧
    (uuidLayout bs)[14]? = some (some 4) ∧
    (∃ k, 8 ≤ k ∧ k ≤ 11 ∧ (uuidLayout bs)[19]? = some (some k)) := by
  have hv := uuid_version_nibble b6
  have hw := uuid_variant_nibble b8
  rw [← h6] at hv; rw [← h8] at hw
  simp only [List.getD_eq_getElem?_getD] at hv hw
  refine ⟨by simp [uuidLayout], by simp [uuidLayout], by simp [uuidLayout], by simp [uuidLayout],
    by simp [uuidLayout], by simp [uuidLayout, hv], ⟨_, hw.1, hw.2, by simp [uuidLayout]⟩⟩

/-! ## `pow` / `^` (math.rs:332; also seen: issue #2902) -/

def applyPow (powf : FC → FC → FC) : V FC → V FC → V FC
  | .int a, .int b => .float (powf (fcOfInt a) (fcOfInt b))
  | .float a, .float b => .float (powf a b)
  | .int a, .float b => .float (powf (fcOfInt a) b)
  | .float a, .int b => .float (powf a (fcOfInt b))
  | _, _ => .null

/-- Any non-numeric operand yields `null` rather than a type error. -/
theorem pow_nonnumeric_null (powf : FC → FC → FC) (a b : V FC) (h : a.isNum = false ∨ b.isNum = false) :
    applyPow powf a b = .null := by
  cases a <;> cases b <;> simp_all [applyPow, V.isNum]

end ValueMath
