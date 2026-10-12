import FalkorNativeIndex.Float
set_option linter.unusedSimpArgs false

/-!
# `i as f64` at the bit level (`encode.rs:31`, `value.rs:1267-1268`)

`bitlenAux` / `bitlen` / `roundNat` are copied from value_order
(`proofs/value_order/FalkorValueOrder/Model.lean:111-128`, IEEE round-to-nearest-even to 53
significant bits). `magBits m` is the binary64 encoding of the positive double nearest `m`: biased
exponent `1022 + L` (`L` = bit length of the rounded value) and the 52 fraction bits below the
implicit one.

Proven here: the conversion is exact and strictly monotone below `2^53`, every converted `i64` lies
at or above `2^52` once `|i| ≥ 2^52` (so it can never tie or cross a value below `2^52`), and every
converted `i64` is finite (not NaN). `2^53` and `2^53 + 1` convert to the same bits.
-/

namespace NI

def bitlenAux : Nat → Nat → Nat
  | 0, _ => 0
  | fuel + 1, n => if n = 0 then 0 else 1 + bitlenAux fuel (n / 2)

/-- Number of significant bits (`0` for `0`); fuel 70 covers every `i64`. -/
def bitlen (n : Nat) : Nat := bitlenAux 70 n

/-- Round a natural to 53 significant bits, ties to even. -/
def roundNat (m : Nat) : Nat :=
  let L := bitlen m
  if L ≤ 53 then m else
    let e := L - 53
    let q := m / 2 ^ e
    let r := m % 2 ^ e
    let half := 2 ^ (e - 1)
    let q' := if r > half ∨ (r = half ∧ q % 2 = 1) then q + 1 else q
    q' * 2 ^ e

/-- binary64 encoding of the positive double nearest `m > 0`. -/
def magBits (m : Nat) : Nat :=
  let r := roundNat m
  let L := bitlen r
  let frac := if L ≤ 53 then r * 2 ^ (53 - L) - 2 ^ 52 else r / 2 ^ (L - 53) - 2 ^ 52
  (1022 + L) * 2 ^ 52 + frac

/-- `i as f64` (`f64::to_bits`), for an `i64`. -/
def i64ToF64 (i : Int) : Nat :=
  if i = 0 then 0 else if 0 < i then magBits i.toNat else SIGN + magBits (-i).toNat

/-! ## `bitlen` -/

theorem bitlenAux_spec : ∀ (f n : Nat), n < 2 ^ f → 0 < n →
    0 < bitlenAux f n ∧ 2 ^ (bitlenAux f n - 1) ≤ n ∧ n < 2 ^ bitlenAux f n
  | 0, n, h, hn => by simp at h; omega
  | f + 1, n, h, hn => by
    have hne : n ≠ 0 := by omega
    simp only [bitlenAux, hne, if_false]
    by_cases h2 : n / 2 = 0
    · have : bitlenAux f (n / 2) = 0 := by rw [h2]; cases f <;> simp [bitlenAux]
      rw [this]; simp; omega
    · have hlt : n / 2 < 2 ^ f := by rw [Nat.pow_succ] at h; omega
      obtain ⟨h1, h2', h3⟩ := bitlenAux_spec f (n / 2) hlt (by omega)
      refine ⟨by omega, ?_, ?_⟩
      · have : 1 + bitlenAux f (n / 2) - 1 = (bitlenAux f (n / 2) - 1) + 1 := by omega
        rw [this, Nat.pow_succ]; omega
      · rw [Nat.add_comm, Nat.pow_succ]; omega

theorem bitlen_spec (n : Nat) (h : n < 2 ^ 70) (hn : 0 < n) :
    0 < bitlen n ∧ 2 ^ (bitlen n - 1) ≤ n ∧ n < 2 ^ bitlen n :=
  bitlenAux_spec 70 n h hn

theorem bitlen_zero : bitlen 0 = 0 := rfl

theorem pow_lt_pow_of_lt' {a b : Nat} (h : a < b) : 2 ^ a < 2 ^ b :=
  Nat.pow_lt_pow_right (by decide) h

theorem pow_le_pow' {a b : Nat} (h : a ≤ b) : 2 ^ a ≤ 2 ^ b :=
  Nat.pow_le_pow_right (by decide) h

/-- `bitlen` is determined by the binade. -/
theorem bitlen_le_of_lt (n k : Nat) (h : n < 2 ^ 70) (hn : 0 < n) (hk : n < 2 ^ k) : bitlen n ≤ k := by
  obtain ⟨h1, h2, _⟩ := bitlen_spec n h hn
  apply Classical.byContradiction
  intro hc
  have : 2 ^ k ≤ 2 ^ (bitlen n - 1) := pow_le_pow' (by omega)
  omega

theorem bitlen_ge_of_le (n k : Nat) (h : n < 2 ^ 70) (hn : 0 < n) (hk : 2 ^ k ≤ n) : k + 1 ≤ bitlen n := by
  obtain ⟨_, _, h3⟩ := bitlen_spec n h hn
  apply Classical.byContradiction
  intro hc
  have : 2 ^ bitlen n ≤ 2 ^ k := pow_le_pow' (by omega)
  omega

theorem bitlen_mono (m n : Nat) (hn : n < 2 ^ 70) (hm : 0 < m) (hmn : m ≤ n) : bitlen m ≤ bitlen n := by
  obtain ⟨_, _, h3⟩ := bitlen_spec n hn (by omega)
  exact bitlen_le_of_lt m _ (by omega) hm (by omega)

/-! ## Exact range: `0 < m < 2^53` -/

theorem roundNat_small (m : Nat) (hm : 0 < m) (h : m < 2 ^ 53) : roundNat m = m := by
  have : bitlen m ≤ 53 := bitlen_le_of_lt m 53 (by omega) hm h
  simp [roundNat, this]

/-- The normalised significand `m * 2^(53-L)` lies in `[2^52, 2^53)`. -/
theorem sig_bounds (m : Nat) (hm : 0 < m) (h : m < 2 ^ 53) :
    2 ^ 52 ≤ m * 2 ^ (53 - bitlen m) ∧ m * 2 ^ (53 - bitlen m) < 2 ^ 53 := by
  obtain ⟨h1, h2, h3⟩ := bitlen_spec m (by omega) hm
  have hL : bitlen m ≤ 53 := bitlen_le_of_lt m 53 (by omega) hm h
  have e1 : 2 ^ (bitlen m - 1) * 2 ^ (53 - bitlen m) = 2 ^ 52 := by
    rw [← Nat.pow_add]; congr 1; omega
  have e2 : 2 ^ bitlen m * 2 ^ (53 - bitlen m) = 2 ^ 53 := by
    rw [← Nat.pow_add]; congr 1; omega
  constructor
  · rw [← e1]; exact Nat.mul_le_mul_right _ h2
  · rw [← e2]; exact Nat.mul_lt_mul_of_pos_right h3 (Nat.two_pow_pos _)

theorem magBits_small (m : Nat) (hm : 0 < m) (h : m < 2 ^ 53) :
    magBits m = (1022 + bitlen m) * 2 ^ 52 + (m * 2 ^ (53 - bitlen m) - 2 ^ 52) := by
  have hL : bitlen m ≤ 53 := bitlen_le_of_lt m 53 (by omega) hm h
  simp [magBits, roundNat_small m hm h, hL]

/-- **Exact and strictly monotone** below `2^53`. -/
theorem magBits_strictMono (m n : Nat) (hm : 0 < m) (hmn : m < n) (hn : n < 2 ^ 53) :
    magBits m < magBits n := by
  rw [magBits_small m hm (by omega), magBits_small n (by omega) hn]
  have sm := sig_bounds m hm (by omega)
  have sn := sig_bounds n (by omega) hn
  have hle := bitlen_mono m n (by omega) hm (by omega)
  by_cases hL : bitlen m = bitlen n
  · rw [hL] at sm ⊢
    have : m * 2 ^ (53 - bitlen n) < n * 2 ^ (53 - bitlen n) :=
      Nat.mul_lt_mul_of_pos_right hmn (Nat.two_pow_pos _)
    have : (2:Nat) ^ 52 = 4503599627370496 := by decide
    have : (2:Nat) ^ 53 = 9007199254740992 := by decide
    omega
  · have hlt : bitlen m + 1 ≤ bitlen n := by omega
    have : (2:Nat) ^ 52 = 4503599627370496 := by decide
    have : (2:Nat) ^ 53 = 9007199254740992 := by decide
    generalize m * 2 ^ (53 - bitlen m) = A at *
    generalize n * 2 ^ (53 - bitlen n) = B at *
    generalize bitlen m = Lm at *
    generalize bitlen n = Ln at *
    have : (1022 + Lm) * 4503599627370496 + 2 * 4503599627370496 ≤ (1022 + Ln) * 4503599627370496 + 4503599627370496 := by
      have := Nat.mul_le_mul_right 4503599627370496 hlt
      omega
    omega

/-- Below `2^52` the exponent field is at most `1074` (value `< 2^52`). -/
theorem magBits_lt_2p52 (m : Nat) (hm : 0 < m) (h : m < 2 ^ 52) : magBits m < 1075 * 2 ^ 52 := by
  rw [magBits_small m hm (by omega)]
  have sm := sig_bounds m hm (by omega)
  have hL : bitlen m ≤ 52 := bitlen_le_of_lt m 52 (by omega) hm h
  have : (2:Nat) ^ 52 = 4503599627370496 := by decide
  have : (2:Nat) ^ 53 = 9007199254740992 := by decide
  generalize m * 2 ^ (53 - bitlen m) = A at *
  generalize bitlen m = L at *
  have : (1022 + L) * 4503599627370496 ≤ 1074 * 4503599627370496 := Nat.mul_le_mul_right _ (by omega)
  omega

/-! ## Large magnitudes -/

theorem roundNat_ge (m : Nat) (h : m < 2 ^ 64) (hm : 2 ^ 52 ≤ m) : 2 ^ 52 ≤ roundNat m := by
  unfold roundNat
  simp only
  split
  · exact hm
  · rename_i hL
    have hL' : 54 ≤ bitlen m := by omega
    obtain ⟨_, h2, _⟩ := bitlen_spec m (by omega) (by omega)
    have e : 2 ^ (bitlen m - 1) = 2 ^ 52 * 2 ^ (bitlen m - 53) := by
      rw [← Nat.pow_add]; congr 1; omega
    have hq : 2 ^ 52 ≤ m / 2 ^ (bitlen m - 53) := by
      rw [Nat.le_div_iff_mul_le (Nat.two_pow_pos _)]
      rw [← e]; exact h2
    have hp : 1 ≤ 2 ^ (bitlen m - 53) := Nat.one_le_two_pow
    split
    · calc 2 ^ 52 ≤ (m / 2 ^ (bitlen m - 53) + 1) * 1 := by omega
        _ ≤ _ := Nat.mul_le_mul_left _ hp
    · calc 2 ^ 52 ≤ (m / 2 ^ (bitlen m - 53)) * 1 := by omega
        _ ≤ _ := Nat.mul_le_mul_left _ hp

theorem roundNat_le (m : Nat) (h : m < 2 ^ 64) : roundNat m ≤ 2 * m := by
  unfold roundNat
  simp only
  split
  · omega
  · have hp : 0 < 2 ^ (bitlen m - 53) := Nat.two_pow_pos _
    have hd := Nat.div_mul_le_self m (2 ^ (bitlen m - 53))
    have hq : 2 ^ (bitlen m - 53) ≤ m := by
      obtain ⟨_, h2, _⟩ := bitlen_spec m (by omega) (by
        apply Classical.byContradiction; intro h0
        have : m = 0 := by omega
        subst this; rw [bitlen_zero] at *; omega)
      exact Nat.le_trans (pow_le_pow' (by omega)) h2
    split
    · rw [Nat.add_mul]; omega
    · omega

/-- Once `m ≥ 2^52` the converted value has exponent field `≥ 1075`. -/
theorem magBits_ge (m : Nat) (h : m < 2 ^ 64) (hm : 2 ^ 52 ≤ m) : 1075 * 2 ^ 52 ≤ magBits m := by
  have r1 := roundNat_ge m h hm
  have r2 := roundNat_le m h
  have hL : 53 ≤ bitlen (roundNat m) := bitlen_ge_of_le _ 52 (by omega) (by omega) r1
  have : 1075 * 2 ^ 52 ≤ (1022 + bitlen (roundNat m)) * 2 ^ 52 := Nat.mul_le_mul_right _ (by omega)
  have key : ∀ F : Nat, 1075 * 2 ^ 52 ≤ (1022 + bitlen (roundNat m)) * 2 ^ 52 + F := fun F =>
    Nat.le_trans this (Nat.le_add_right _ _)
  unfold magBits
  exact key _

/-- Every converted magnitude is finite: below `+inf`'s encoding `INF`. -/
theorem magBits_finite (m : Nat) (h : m < 2 ^ 64) (hm : 0 < m) : magBits m < INF := by
  have r2 := roundNat_le m h
  have r0 : 0 < roundNat m := by
    by_cases hs : m < 2 ^ 53
    · rw [roundNat_small m hm hs]; exact hm
    · have := roundNat_ge m h (by omega); omega
  obtain ⟨_, h2, h3⟩ := bitlen_spec (roundNat m) (by omega) r0
  have hL : bitlen (roundNat m) ≤ 65 := bitlen_le_of_lt _ 65 (by omega) r0 (by omega)
  unfold magBits
  simp only
  have hf : (if bitlen (roundNat m) ≤ 53 then roundNat m * 2 ^ (53 - bitlen (roundNat m)) - 2 ^ 52
      else roundNat m / 2 ^ (bitlen (roundNat m) - 53) - 2 ^ 52) < 2 ^ 53 := by
    split
    · have e2 : 2 ^ bitlen (roundNat m) * 2 ^ (53 - bitlen (roundNat m)) = 2 ^ 53 := by
        rw [← Nat.pow_add]; congr 1; omega
      have := Nat.mul_lt_mul_of_pos_right h3 (Nat.two_pow_pos (53 - bitlen (roundNat m)))
      omega
    · have e2 : 2 ^ bitlen (roundNat m) = 2 ^ 53 * 2 ^ (bitlen (roundNat m) - 53) := by
        rw [← Nat.pow_add]; congr 1; omega
      have : roundNat m / 2 ^ (bitlen (roundNat m) - 53) < 2 ^ 53 := by
        rw [Nat.div_lt_iff_lt_mul (Nat.two_pow_pos _)]; omega
      omega
  have : (1022 + bitlen (roundNat m)) * 2 ^ 52 ≤ 1087 * 2 ^ 52 := Nat.mul_le_mul_right _ (by omega)
  simp only [INF]
  have : (2:Nat) ^ 52 = 4503599627370496 := by decide
  have : (2:Nat) ^ 53 = 9007199254740992 := by decide
  omega

theorem magBits_pos (m : Nat) : 0 < magBits m := by
  have key : ∀ L F : Nat, 0 < (1022 + L) * 2 ^ 52 + F := fun L F =>
    Nat.lt_of_lt_of_le (Nat.mul_pos (by omega) (Nat.two_pow_pos 52)) (Nat.le_add_right _ _)
  unfold magBits
  exact key _ _

/-! ## `i as f64` on `i64` -/

/-- An `i64`. -/
def I64 (i : Int) : Prop := -9223372036854775808 ≤ i ∧ i < 9223372036854775808

theorem i64ToF64_lt (i : Int) (hi : I64 i) : i64ToF64 i < W ∧ isNaN (i64ToF64 i) = false := by
  unfold i64ToF64 I64 at *
  split
  · decide
  · split
    · have := magBits_finite i.toNat (by omega) (by omega)
      simp only [W, INF, isNaN, SIGN] at *
      refine ⟨by omega, ?_⟩
      exact decide_eq_false (by omega)
    · have := magBits_finite (-i).toNat (by omega) (by omega)
      simp only [W, INF, isNaN, SIGN] at *
      refine ⟨by omega, ?_⟩
      exact decide_eq_false (by omega)

/-- `2^53` and `2^53 + 1` convert to the same `f64` (the rounding collision). -/
theorem i64ToF64_collide : i64ToF64 9007199254740992 = i64ToF64 9007199254740993 := by decide

/-- ...while `2^53 - 1` and `2^53` stay apart. -/
theorem i64ToF64_apart : i64ToF64 9007199254740991 ≠ i64ToF64 9007199254740992 := by decide

end NI
