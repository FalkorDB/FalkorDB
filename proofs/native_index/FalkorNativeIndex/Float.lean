set_option linter.unusedSimpArgs false
/-!
# `f64` at the bit level, and `encode_f64` (`graph/src/index/falkordb/encode.rs:50-55`)

An `f64` is modelled as **its IEEE-754 binary64 encoding** (`f64::to_bits`), a `Nat < 2^64`. This
represents every value the Rust can hold: both zeros (`0` and `2^63`), every NaN sign and payload,
both infinities and every finite value. It refines value_order's extended `F`
(`proofs/value_order/FalkorValueOrder/Model.lean:46`): `F.negZero` is `2^63`, `F.nan neg payload` is a
pattern with the exponent field all ones and a non-zero fraction, `F.negInf`/`F.posInf` are
`0xFFF0…0`/`0x7FF0…0`. The bit level is needed here because `encode_f64` *is* a bit trick.

`f64::partial_cmp` is IEEE 754's comparison (§5.11) and the binary64 encoding is designed so that it
is the sign-magnitude order of the bits (§3.4: for operands of one sign, the 63-bit
exponent‖fraction field orders as the integer it spells; `±0` compare equal; NaN is unordered). We
*define* `fpcmp` that way (`pos` below is the position on the real line). No axiom.
-/

namespace NI

/-- `2^63`, the IEEE sign bit `SIGN` (`encode.rs:21`). -/
def SIGN : Nat := 9223372036854775808
/-- `2^64`. -/
def W : Nat := 18446744073709551616
/-- `0x7FF0_0000_0000_0000` = `+inf`; also the largest non-NaN magnitude. -/
def INF : Nat := 9218868437227405312
/-- `-inf` = `SIGN + INF`. -/
def NINF : Nat := 18442240474082181120

/-- `f64::is_nan` on the encoding: exponent all ones and fraction non-zero, i.e. magnitude `> INF`. -/
def isNaN (x : Nat) : Bool := decide (x % SIGN > INF)

/-- Position of a non-NaN encoding on the real line (`±0` share `SIGN`). Positives (`x < SIGN`) sit
    at `SIGN + x`, negatives at `SIGN - |x|` = `W - x`. -/
def pos (x : Nat) : Nat := if x < SIGN then SIGN + x else W - x

/-- `Ordering` of two naturals: `u64::cmp`. -/
def cmpN (a b : Nat) : Ordering := if a < b then .lt else if a = b then .eq else .gt

/-- `f64::partial_cmp`: `none` iff either side is NaN, else the IEEE order. -/
def fpcmp (a b : Nat) : Option Ordering :=
  if isNaN a || isNaN b then none else some (cmpN (pos a) (pos b))

/-- `x == 0.0` (IEEE equality with `+0.0`): true for both zeros. -/
def isZero (x : Nat) : Bool := fpcmp x 0 == some .eq

/-- `u64` `!x`, via `BitVec`. -/
def u64not (x : Nat) : Nat := (~~~(BitVec.ofNat 64 x)).toNat

/-- `encode_f64` (`encode.rs:50-55`), line by line:
    `let x = if x == 0.0 { 0.0 } else { x }; let bits = x.to_bits();
     if bits & SIGN == 0 { bits | SIGN } else { !bits }`. -/
def encodeF64 (x : Nat) : Nat :=
  let x := if isZero x then 0 else x
  let bits := x
  if bits &&& SIGN = 0 then bits ||| SIGN else u64not bits

/-! ## Bit lemmas -/

theorem and_sign (b : Nat) (hb : b < W) : (b &&& SIGN = 0 ↔ b < SIGN) := by
  have hS : SIGN = 2 ^ 63 := rfl
  rw [hS]
  constructor
  · intro h
    have h63 : (b &&& 2^63).testBit 63 = false := by rw [h]; simp
    rw [Nat.testBit_and, Nat.testBit_two_pow] at h63
    simp only [decide_true, Bool.and_true] at h63
    rw [Nat.testBit_eq_decide_div_mod_eq] at h63
    simp at h63
    have : b / 2^63 < 2 := by simp [W] at hb; omega
    omega
  · intro h
    apply Nat.eq_of_testBit_eq
    intro i
    rw [Nat.testBit_and, Nat.testBit_two_pow]
    by_cases hi : 63 = i
    · subst hi; simp [Nat.testBit_lt_two_pow h]
    · simp [hi]

theorem or_sign (b : Nat) (hb : b < SIGN) : b ||| SIGN = b + SIGN :=
  Nat.or_two_pow_eq_add_of_lt (n := 63) hb

theorem u64not_eq (b : Nat) (hb : b < W) : u64not b = W - 1 - b := by
  unfold u64not
  rw [BitVec.toNat_not, BitVec.toNat_ofNat, Nat.mod_eq_of_lt (show b < 2 ^ 64 from hb)]
  rfl

/-! ## NaN and zero on the encoding -/

theorem isNaN_false_iff (x : Nat) : isNaN x = false ↔ x % SIGN ≤ INF := by
  simp [isNaN]

theorem isZero_iff (x : Nat) (hx : x < W) : isZero x = true ↔ x % SIGN = 0 := by
  have h0 : isNaN 0 = false := by decide
  have hp0 : pos 0 = SIGN := by decide
  unfold isZero fpcmp
  rw [h0, hp0]
  cases hn : isNaN x
  · have := (isNaN_false_iff x).1 hn
    simp only [Bool.or_false, Bool.false_eq_true, ite_false, cmpN, pos]
    simp only [SIGN, W, INF] at *
    by_cases h1 : x < 9223372036854775808
    · simp only [h1, ite_true]
      by_cases h2 : x = 0
      · subst h2; decide
      · have h3 : ¬ (9223372036854775808 + x < 9223372036854775808) := by omega
        have h4 : ¬ (9223372036854775808 + x = 9223372036854775808) := by omega
        simp only [h3, h4, ite_false]
        constructor
        · intro h; simp at h
        · intro h; omega
    · simp only [h1, ite_false]
      by_cases h2 : x = 9223372036854775808
      · subst h2; decide
      · have h3 : 18446744073709551616 - x < 9223372036854775808 := by omega
        simp only [h3, ite_true]
        constructor
        · intro h; simp at h
        · intro h; omega
  · have : x % SIGN > INF := by simpa [isNaN] using hn
    simp only [SIGN, INF] at this
    simp only [Bool.true_or, ite_true]
    simp only [SIGN]
    constructor
    · intro h; simp at h
    · intro h; omega

/-- The closed form of `encode_f64` on a `u64` encoding. -/
def ek (x : Nat) : Nat := if x % SIGN = 0 then SIGN else if x < SIGN then x + SIGN else W - 1 - x

/-- The encoding of `x` in closed form: `+SIGN` for non-negatives, `W - 1 - x` for negatives, both
    zeros at `SIGN`. -/
theorem encodeF64_eq (x : Nat) (hx : x < W) : encodeF64 x = ek x := by
  unfold encodeF64 ek
  by_cases hz : x % SIGN = 0
  · have : isZero x = true := (isZero_iff x hx).2 hz
    simp only [this, ite_true, hz]
    rw [if_pos ((and_sign 0 (by decide)).2 (by decide)), or_sign 0 (by decide)]
    rfl
  · have : isZero x = false := by
      cases h : isZero x
      · rfl
      · exact absurd ((isZero_iff x hx).1 h) hz
    simp only [this, hz, Bool.false_eq_true, ite_false]
    by_cases hs : x < SIGN
    · rw [if_pos ((and_sign x hx).2 hs), or_sign x hs, if_pos hs]
    · rw [if_neg (fun h => hs ((and_sign x hx).1 h)), u64not_eq x hx, if_neg hs]

/-- Case split of a `u64` encoding into the three closed-form regimes. -/
theorem ek_cases (x : Nat) :
    (x % SIGN = 0 ∧ ek x = SIGN) ∨ (x % SIGN ≠ 0 ∧ x < SIGN ∧ ek x = x + SIGN) ∨
    (x % SIGN ≠ 0 ∧ ¬ x < SIGN ∧ ek x = W - 1 - x) := by
  unfold ek
  by_cases h1 : x % SIGN = 0
  · left; simp [h1]
  · by_cases h2 : x < SIGN
    · right; left; simp [h1, h2]
    · right; right; simp [h1, h2]

theorem pos_cases (x : Nat) : (x < SIGN ∧ pos x = SIGN + x) ∨ (¬ x < SIGN ∧ pos x = W - x) := by
  unfold pos
  by_cases h : x < SIGN
  · left; simp [h]
  · right; simp [h]

/-- Non-NaN encodings are bounded: `0 < encode_f64 x < u64::MAX`, so the `checked_add(1)` /
    `checked_sub(1)` of `NumericIndex::range` never fail on them. -/
theorem encodeF64_bounds (x : Nat) (hx : x < W) (hn : isNaN x = false) :
    0 < encodeF64 x ∧ encodeF64 x < W - 1 := by
  rw [encodeF64_eq x hx]
  have := (isNaN_false_iff x).1 hn
  rcases ek_cases x with ⟨_, h⟩ | ⟨_, _, h⟩ | ⟨_, _, h⟩ <;> rw [h] <;> simp only [SIGN, W, INF] at * <;> omega

theorem cmpN_congr {a b c d : Nat} (h1 : a < b ↔ c < d) (h2 : a = b ↔ c = d) : cmpN a b = cmpN c d := by
  unfold cmpN
  by_cases x : a < b
  · simp [x, h1.1 x]
  · have y : ¬ c < d := fun y => x (h1.2 y)
    by_cases z : a = b
    · simp [z, h2.1 z]
    · have w : ¬ c = d := fun w => z (h2.2 w)
      simp [x, y, z, w]

/-- **Order preservation and reflection** (`encode.rs:6-11`, `:43-44`): for non-NaN `a`, `b`,
    `f64::partial_cmp a b = Some (encode_f64 a).cmp(&encode_f64 b)`. Equal keys ⇔ IEEE-equal values. -/
theorem encodeF64_cmp (a b : Nat) (ha : a < W) (hb : b < W) (hna : isNaN a = false) (hnb : isNaN b = false) :
    fpcmp a b = some (cmpN (encodeF64 a) (encodeF64 b)) := by
  have hna' := (isNaN_false_iff a).1 hna
  have hnb' := (isNaN_false_iff b).1 hnb
  simp only [fpcmp, hna, hnb, Bool.or_false, Bool.false_eq_true, ite_false]
  rw [encodeF64_eq a ha, encodeF64_eq b hb]
  congr 1
  apply cmpN_congr <;>
  rcases ek_cases a with ⟨_, h⟩ | ⟨_, _, h⟩ | ⟨_, _, h⟩ <;> rw [h] <;>
  rcases ek_cases b with ⟨_, h'⟩ | ⟨_, _, h'⟩ | ⟨_, _, h'⟩ <;> rw [h'] <;>
  rcases pos_cases a with ⟨_, p⟩ | ⟨_, p⟩ <;> rw [p] <;>
  rcases pos_cases b with ⟨_, p'⟩ | ⟨_, p'⟩ <;> rw [p'] <;>
  simp only [SIGN, W, INF, iff_true, true_iff, iff_false, false_iff] at * <;>
  first | omega | (constructor <;> intro <;> omega)

/-- **Injectivity** (up to the deliberate `-0.0 → +0.0` collapse): two non-NaN encodings share a key
    iff they are the same bits or both are zeros. -/
theorem encodeF64_inj (a b : Nat) (ha : a < W) (hb : b < W) (hna : isNaN a = false) (hnb : isNaN b = false) :
    encodeF64 a = encodeF64 b ↔ (a = b ∨ (a % SIGN = 0 ∧ b % SIGN = 0)) := by
  have hna' := (isNaN_false_iff a).1 hna
  have hnb' := (isNaN_false_iff b).1 hnb
  rw [encodeF64_eq a ha, encodeF64_eq b hb]
  rcases ek_cases a with ⟨_, h⟩ | ⟨_, _, h⟩ | ⟨_, _, h⟩ <;> rw [h] <;>
  rcases ek_cases b with ⟨_, h'⟩ | ⟨_, _, h'⟩ | ⟨_, _, h'⟩ <;> rw [h'] <;>
  simp only [SIGN, W, INF, iff_true, true_iff, iff_false, false_iff] at * <;>
  first | omega | (constructor <;> intro <;> omega)

/-- The two zeros share a key (`encode.rs:51`, test `zeros_collide`). -/
theorem encodeF64_negZero : encodeF64 SIGN = encodeF64 0 := by decide

/-- `-inf` has the least non-NaN key and `+inf` the greatest. -/
theorem encodeF64_inf_extremes (x : Nat) (hx : x < W) (hn : isNaN x = false) :
    encodeF64 NINF ≤ encodeF64 x ∧ encodeF64 x ≤ encodeF64 INF := by
  rw [encodeF64_eq x hx, encodeF64_eq NINF (by decide), encodeF64_eq INF (by decide)]
  have := (isNaN_false_iff x).1 hn
  have e1 : ek NINF = 4503599627370495 := by decide
  have e2 : ek INF = 18442240474082181120 := by decide
  rw [e1, e2]
  rcases ek_cases x with ⟨_, h⟩ | ⟨_, _, h⟩ | ⟨_, _, h⟩ <;> rw [h] <;> simp only [SIGN, W, INF] at * <;> omega

/-- IEEE `fpcmp` really is NaN-unordered whatever the payload or sign. -/
theorem fpcmp_nan (a b : Nat) (h : isNaN a = true) : fpcmp a b = none := by
  simp [fpcmp, h]

/-- `fpcmp` with the `pos` order: both zeros compare equal. -/
theorem fpcmp_zeros : fpcmp SIGN 0 = some .eq := by decide

end NI
