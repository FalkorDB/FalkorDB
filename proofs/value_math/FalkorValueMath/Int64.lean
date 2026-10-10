/-!
# `i64` arithmetic of `Value` (Int lanes of `Add/Sub/Mul/Div/Rem for Value`)

Rust's `i64` is modelled as a mathematical `Int` together with the two's-complement
wrap `wrap : Int → Int` onto `[-2^63, 2^63)`. Every Int lane in
`graph/src/runtime/value.rs:903-1171` uses the `wrapping_*` methods, which are exactly
`wrap ∘ (exact op)` (Rust reference, `i64::wrapping_add` etc.; `wrapping_div` wraps only
`MIN / -1`, `wrapping_rem` returns 0 for `MIN % -1`).

C FalkorDB's integer division (`SIValue_Divide`, `src/value.c`, master) is
`SI_LongVal(SI_GET_NUMERIC(a) / SI_GET_NUMERIC(b))`: both operands are converted to
`double` first. We model `i as f64` exactly on integers with `roundF` (round to 53
significant bits, ties to even — IEEE 754 roundTiesToEven) and the `double → int64_t`
conversion as the arm64 `fcvtzs` saturating truncation `satTrunc`.
-/

namespace ValueMath

/-- 2^63 and 2^64 as literals, so `omega` can reason about them. -/
notation "TWO63" => (9223372036854775808 : Int)
notation "TWO64" => (18446744073709551616 : Int)
notation "I64MIN" => (-9223372036854775808 : Int)
notation "I64MAX" => (9223372036854775807 : Int)

/-- In `i64` range. -/
def InI64 (x : Int) : Prop := I64MIN ≤ x ∧ x ≤ I64MAX

instance : DecidablePred InI64 := fun x => by unfold InI64; infer_instance

/-- Two's-complement wrap onto `i64`. -/
def wrap (x : Int) : Int := (x + TWO63) % TWO64 - TWO63

theorem wrap_inI64 (x : Int) : InI64 (wrap x) := by
  unfold InI64 wrap; omega

theorem wrap_id {x : Int} (h : InI64 x) : wrap x = x := by
  unfold InI64 at h; unfold wrap; omega

theorem wrap_wrap (x : Int) : wrap (wrap x) = wrap x := wrap_id (wrap_inI64 x)

/-- `wrap` only depends on the residue mod 2^64. -/
theorem wrap_add_mul (x k : Int) : wrap (x + k * TWO64) = wrap x := by
  unfold wrap; omega

/-! ## Rust `wrapping_*` (value.rs:915, 1048, 1097, 1130, 1158) -/

def wAdd (a b : Int) : Int := wrap (a + b)
def wSub (a b : Int) : Int := wrap (a - b)
def wMul (a b : Int) : Int := wrap (a * b)
/-- `i64::wrapping_div` for `b ≠ 0`: truncating division, only `MIN / -1` wraps. -/
def wDiv (a b : Int) : Int := wrap (Int.tdiv a b)
/-- `i64::wrapping_rem` for `b ≠ 0`: truncating remainder (`MIN % -1 = 0`). -/
def wRem (a b : Int) : Int := wrap (Int.tmod a b)

theorem wAdd_comm (a b : Int) : wAdd a b = wAdd b a := by
  unfold wAdd; rw [Int.add_comm]

theorem wAdd_assoc (a b c : Int) : wAdd (wAdd a b) c = wAdd a (wAdd b c) := by
  unfold wAdd wrap; omega

theorem wAdd_zero {a : Int} (h : InI64 a) : wAdd a 0 = a := by
  unfold wAdd; simp [wrap_id h]

theorem wSub_eq_wAdd_neg (a b : Int) : wSub a b = wAdd a (wrap (-b)) := by
  unfold wSub wAdd wrap; omega

theorem wMul_comm (a b : Int) : wMul a b = wMul b a := by
  unfold wMul; rw [Int.mul_comm]

/-- `i64::MAX + 1` wraps to `i64::MIN` (C does the same: signed overflow, which on every
supported target wraps). -/
theorem add_max_one : wAdd I64MAX 1 = I64MIN := by decide

theorem mul_max_two : wMul I64MAX 2 = -2 := by decide

/-- The Rust result is the exact result whenever the exact result fits. -/
theorem wAdd_exact {a b : Int} (h : InI64 (a + b)) : wAdd a b = a + b := wrap_id h
theorem wMul_exact {a b : Int} (h : InI64 (a * b)) : wMul a b = a * b := wrap_id h

/-- The truncated quotient of two `i64`s fits in `i64` except for `MIN / -1`. -/
theorem tdiv_inI64 {a b : Int} (ha : InI64 a) (hb : b ≠ 0) (hx : ¬ (a = I64MIN ∧ b = -1)) :
    InI64 (Int.tdiv a b) := by
  unfold InI64 at *
  have hn := Int.natAbs_tdiv_le_natAbs a b
  rcases Int.lt_or_gt_of_ne hb with hb | hb
  · -- b < 0
    by_cases hb1 : b = -1
    · subst hb1
      have : a.tdiv (-1) = -a := by rw [Int.tdiv_neg, Int.tdiv_one]
      rw [this]; constructor <;> omega
    · have hb2 : b ≤ -2 := by omega
      have h2 : (Int.tdiv a b).natAbs ≤ a.natAbs / 2 := by
        rw [Int.natAbs_tdiv]
        exact Nat.div_le_div_left (by omega) (by decide)
      omega
  · rcases Int.le_total 0 a with h0 | h0
    · have := Int.tdiv_nonneg h0 (Int.le_of_lt hb)
      omega
    · have h1 : Int.tdiv a b = -(Int.tdiv (-a) b) := by rw [Int.neg_tdiv, Int.neg_neg]
      have h2 := Int.tdiv_nonneg (Int.neg_nonneg_of_nonpos h0) (Int.le_of_lt hb)
      omega

theorem wDiv_exact {a b : Int} (ha : InI64 a) (hb : b ≠ 0) (hx : ¬ (a = I64MIN ∧ b = -1)) :
    wDiv a b = Int.tdiv a b := wrap_id (tdiv_inI64 ha hb hx)

/-- `MIN / -1` wraps back to `MIN` in Rust (`wrapping_div`). -/
theorem wDiv_min_neg1 : wDiv I64MIN (-1) = I64MIN := by decide

theorem tmod_inI64 {a b : Int} (ha : InI64 a) (hb : InI64 b) (hb0 : b ≠ 0) :
    InI64 (Int.tmod a b) := by
  unfold InI64 at *
  have h1 := Int.natAbs_tmod a b
  have h2 : a.natAbs % b.natAbs < b.natAbs := Nat.mod_lt _ (by omega)
  have h3 : a.natAbs % b.natAbs ≤ a.natAbs := Nat.mod_le _ _
  omega

theorem wRem_exact {a b : Int} (ha : InI64 a) (hb : InI64 b) (hb0 : b ≠ 0) :
    wRem a b = Int.tmod a b := wrap_id (tmod_inI64 ha hb hb0)

/-- The division identity that ties `/` and `%` together holds for Rust's pair,
modulo 2^64 always, and exactly whenever the quotient fits. -/
theorem div_rem_identity {a b : Int} (ha : InI64 a) (hb : InI64 b) (hb0 : b ≠ 0)
    (hx : ¬ (a = I64MIN ∧ b = -1)) :
    wAdd (wMul (wDiv a b) b) (wRem a b) = a := by
  rw [wDiv_exact ha hb0 hx, wRem_exact ha hb hb0]
  have hd := Int.tmod_def a b
  unfold wAdd wMul
  have hq : InI64 (Int.tdiv a b * b) := by
    have : Int.tdiv a b * b = a - Int.tmod a b := by rw [hd, Int.mul_comm]; omega
    have := tmod_inI64 ha hb hb0
    unfold InI64 at *
    have h1 := Int.natAbs_tmod a b
    have h3 : a.natAbs % b.natAbs ≤ a.natAbs := Nat.mod_le _ _
    -- sign of tmod follows a
    rcases Int.le_total 0 a with h | h
    · have := Int.tmod_nonneg b h; omega
    · have : Int.tmod a b ≤ 0 := by
        have := Int.tmod_nonneg b (Int.neg_nonneg_of_nonpos h)
        rw [Int.neg_tmod] at this; omega
      omega
  rw [wrap_id hq]
  have : Int.tdiv a b * b + Int.tmod a b = a := by rw [hd, Int.mul_comm]; omega
  rw [this]; exact wrap_id ha

/-- Remainder sign follows the dividend and its magnitude is below the divisor's
(same as C's `%`, C99 6.5.5). -/
theorem rem_bounds (a b : Int) (hb : b ≠ 0) :
    (Int.tmod a b).natAbs < b.natAbs ∧ (0 ≤ a → 0 ≤ Int.tmod a b) := by
  refine ⟨?_, fun h => Int.tmod_nonneg b h⟩
  rw [Int.natAbs_tmod]; exact Nat.mod_lt _ (by omega)

/-! ## `abs` / `sign` on integers (math.rs:48, 212) -/

/-- `i64::checked_abs`. -/
def checkedAbs (a : Int) : Option Int := if a = I64MIN then none else some a.natAbs

theorem checkedAbs_none_iff {a : Int} : checkedAbs a = none ↔ a = I64MIN := by
  unfold checkedAbs; split <;> simp_all

theorem checkedAbs_some {a : Int} (ha : InI64 a) (h : a ≠ I64MIN) :
    checkedAbs a = some a.natAbs ∧ InI64 a.natAbs := by
  unfold checkedAbs InI64 at *; simp [h]; omega

/-- C `AR_ABS`: `if (SI_GET_NUMERIC(x) < 0) return x * -1` — so `abs(MIN)` wraps to MIN. -/
def cAbsInt (a : Int) : Int := if a < 0 then wMul a (-1) else a

theorem cAbs_min : cAbsInt I64MIN = I64MIN := by decide

/-- `i64::signum`. -/
def signumI (a : Int) : Int := if a > 0 then 1 else if a < 0 then -1 else 0

/-! ## C's integer division goes through `double` -/

/-- Round an integer to the nearest `double` (53-bit significand, ties to even).
Exact integer model of `i as f64` (copied from proofs/value_order `roundF`, generalised
to use a bit-length search). -/
def bitLen : Nat → Nat
  | 0 => 0
  | n + 1 => Nat.log2 (n + 1) + 1

def roundNat (n : Nat) : Nat :=
  let L := bitLen n
  if L ≤ 53 then n else
    let s := L - 53
    let q := n / 2 ^ s
    let r := n % 2 ^ s
    let half := 2 ^ (s - 1)
    let q' := if r > half ∨ (r = half ∧ q % 2 = 1) then q + 1 else q
    q' * 2 ^ s

def roundF (i : Int) : Int := if i < 0 then -(roundNat i.natAbs : Int) else (roundNat i.natAbs : Int)

theorem roundNat_exact {n : Nat} (h : n < 2 ^ 53) : roundNat n = n := by
  unfold roundNat
  have : bitLen n ≤ 53 := by
    unfold bitLen
    split
    · omega
    · rename_i m
      have := (Nat.log2_lt (n := m + 1) (by omega)).2 h
      simp only [Nat.succ_eq_add_one] at *
      omega
  simp [this]

/-- Integers below 2^53 in magnitude are doubles. -/
theorem roundF_exact {i : Int} (h : i.natAbs < 2 ^ 53) : roundF i = i := by
  unfold roundF
  rw [roundNat_exact h]
  split <;> omega

/-- arm64 `fcvtzs` (the `(int64_t)` cast of a double already integral): saturating. -/
def satTrunc (x : Int) : Int := if x > I64MAX then I64MAX else if x < I64MIN then I64MIN else x

/-- C `SIValue_Divide` on two integers: `(int64_t)((double)a / (double)b)`. The model
truncates the quotient of the *rounded operands*, which is faithful whenever that
quotient is an integer (in particular for `b = ±1`, the only case used below). -/
def cDivExact (a b : Int) : Int := satTrunc (Int.tdiv (roundF a) (roundF b))

/-- 2^53 + 1 / 1: Rust returns 9007199254740993, C returns 9007199254740992 (live). -/
theorem c_div_loses_precision :
    wDiv 9007199254740993 1 = 9007199254740993 ∧ cDivExact 9007199254740993 1 = 9007199254740992 := by
  decide

/-- `MIN / -1`: Rust `MIN` (wraps), C `MAX` (double 2^63 saturates) — live-confirmed. -/
theorem min_div_neg1_diverges : wDiv I64MIN (-1) = I64MIN ∧ cDivExact I64MIN (-1) = I64MAX := by
  decide

/-- Dividing by 1: C agrees with Rust exactly when |a| < 2^53 (the double is exact),
and loses the low bits above that. -/
theorem c_rust_div_one_agree {a : Int} (ha : a.natAbs < 2 ^ 53) :
    cDivExact a 1 = wDiv a 1 := by
  unfold cDivExact wDiv
  rw [roundF_exact ha, show roundF 1 = 1 by decide, Int.tdiv_one]
  have hA : InI64 a := by unfold InI64; omega
  rw [wrap_id hA]
  unfold satTrunc InI64 at *
  split
  · omega
  · split
    · omega
    · rfl

end ValueMath
