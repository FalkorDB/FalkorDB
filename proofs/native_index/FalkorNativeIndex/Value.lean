import FalkorNativeIndex.IntF
set_option linter.unusedSimpArgs false

/-!
# Values, `encode_numeric` (`encode.rs:29-41`) and the evaluator's comparison

`Value` keeps every variant the index path can meet with its real payload: `Int` (an `i64`), `Float`
(the `f64` bits, so `-0.0` and every NaN payload are distinct values), `Bool`, the four temporals
(raw `i64`), `String`, `Null`, and `other` for the variants no numeric index ever stores or queries
(List/Map/Path/Node/Relationship/Point/VecF32, `tag` = their `order()`).

`compareValue` is `CompareValue::compare_value` (`value.rs:1227-1270`) on these variants;
`cmpPred` is the Filter's `<`/`>`/`<=`/`>=` (`eval.rs:446-462`) and `=` (`all_equals`,
`eval.rs:1665-1686`). A row survives a Filter iff the predicate is `some true`.
-/

namespace NI

inductive Value where
  | null
  | bool (b : Bool)
  | int (i : Int)
  | float (x : Nat)
  | string (s : String)
  | date (i : Int)
  | datetime (i : Int)
  | time (i : Int)
  | duration (i : Int)
  | other (tag : Nat)
deriving DecidableEq, Repr

/-- Payload well-formedness: `i64` / `u64`-bit ranges. -/
def Value.WF : Value → Prop
  | .int i => I64 i
  | .float x => x < W
  | _ => True

/-- `OrderedEnum::order` (`value.rs:1181-1199`). -/
def Value.order : Value → Nat
  | .null => 2 ^ 15 | .bool _ => 2 ^ 12 | .int _ => 2 ^ 13 | .float _ => 2 ^ 14
  | .string _ => 2 ^ 11 | .datetime _ => 2 ^ 6 | .date _ => 2 ^ 7 | .time _ => 2 ^ 8
  | .duration _ => 2 ^ 10 | .other t => t

/-- `DisjointOrNull` (`value.rs:1204`). -/
inductive Flag where
  | disjoint | comparedNull | nan | none
deriving DecidableEq, Repr

def cmpI (a b : Int) : Ordering := if a < b then .lt else if a = b then .eq else .gt
def cmpB (a b : Bool) : Ordering := cmpN a.toNat b.toNat
def cmpS (a b : String) : Ordering := compare a b

/-- `compare_floats` (`value.rs:1875-1885`). -/
def compareFloats (a b : Nat) : Ordering × Flag :=
  match fpcmp a b with
  | some o => (o, .none)
  | none => (.lt, .nan)

/-- `compare_value` (`value.rs:1227-1270`), arm by arm. -/
def compareValue : Value → Value → Ordering × Flag
  | .bool a, .bool b => (cmpB a b, .none)
  | .float a, .float b => compareFloats a b
  | .string a, .string b => (cmpS a b, .none)
  | .int a, .int b => (cmpI a b, .none)
  | .datetime a, .datetime b => (cmpI a b, .none)
  | .date a, .date b => (cmpI a b, .none)
  | .time a, .time b => (cmpI a b, .none)
  | .duration a, .duration b => (cmpI a b, .none)
  | .int i, .float f => compareFloats (i64ToF64 i) f
  | .float f, .int i => compareFloats f (i64ToF64 i)
  | .null, b => (cmpN Value.null.order b.order, .comparedNull)
  | a, .null => (cmpN a.order Value.null.order, .comparedNull)
  | a, b => (cmpN a.order b.order, .disjoint)

/-- Comparison operators of a Filter. -/
inductive Op where
  | lt | gt | le | ge | eq
deriving DecidableEq, Repr

/-- The Filter's value for `d <op> b` (`eval.rs:446-462`; `=` is `all_equals`, `eval.rs:1665`).
    `none` is `Value::Null`. -/
def cmpPred (op : Op) (d b : Value) : Option Bool :=
  match op with
  | .eq =>
    match compareValue d b with
    | (_, .comparedNull) => none
    | (_, .nan) | (_, .disjoint) => some false
    | (.eq, _) => some true
    | _ => some false
  | _ =>
    match compareValue d b with
    | (_, .comparedNull) | (_, .disjoint) => none
    | (_, .nan) => some false
    | (o, _) => some (match op with
      | .lt => o == .lt
      | .gt => o == .gt
      | .le => o != .gt
      | _ => o != .lt)

/-- `encode_numeric` (`encode.rs:29-41`): `Int` → `i as f64`, `Float` as is, `Bool` → `0.0`/`1.0`
    (`f64::from(b)`: `+1.0` is `0x3FF0_0000_0000_0000`); everything else, and NaN, is `None`. -/
def encodeNumeric : Value → Option Nat
  | v =>
    let x? : Option Nat := match v with
      | .int i => some (i64ToF64 i)
      | .float f => some f
      | .bool b => some (if b then 4607182418800017408 else 0)
      | _ => none
    match x? with
    | none => none
    | some x => if !isNaN x then some (encodeF64 x) else none

/-- The same predicate on encoded keys: what a `(key, doc)` range scan decides. -/
def keyPred (op : Op) (k kb : Nat) : Bool :=
  match op with
  | .lt => k < kb | .gt => kb < k | .le => k ≤ kb | .ge => kb ≤ k | .eq => k == kb

/-- `int_loses_f64_precision` (`index/mod.rs:1371-1373`): `i.unsigned_abs() & 0x7FF0…0 != 0`. -/
def intLosesPrecision (i : Int) : Bool := i.natAbs &&& 0x7FF0000000000000 != 0

/-! ## The precision guard -/

theorem mask_bits : ∀ i, i < 63 → 52 ≤ i → (0x7FF0000000000000 : Nat).testBit i = true := by decide

/-- What the guard lets through: `|i| < 2^52`, or `|i| ≥ 2^63` — i.e. `i64::MIN`.
    (Same statement as index_layer's `intLosesPrecision_sound`, re-proven here.) -/
theorem guard_sound (i : Int) (hm : intLosesPrecision i = false) :
    i.natAbs < 2 ^ 52 ∨ 2 ^ 63 ≤ i.natAbs := by
  simp only [intLosesPrecision, bne_eq_false_iff_eq] at hm
  by_cases h63 : 2 ^ 63 ≤ i.natAbs
  · exact Or.inr h63
  · left
    apply Nat.lt_pow_two_of_testBit
    intro j hj
    by_cases hj2 : j < 63
    · have := congrArg (fun x => x.testBit j) hm
      simp only [Nat.testBit_and, Nat.zero_testBit, mask_bits j hj2 hj, Bool.and_true] at this
      exact this
    · apply Nat.testBit_lt_two_pow
      have : 2 ^ 63 ≤ 2 ^ j := Nat.pow_le_pow_right (by decide) (by omega)
      omega

/-- `i64::MIN` passes the guard. -/
theorem guard_admits_i64_min : intLosesPrecision (-9223372036854775808) = false := by decide

/-! ## Order agreement -/

theorem pos_i64 (i : Int) (hi : I64 i) :
    (i = 0 → pos (i64ToF64 i) = SIGN) ∧
    (0 < i → pos (i64ToF64 i) = SIGN + magBits i.toNat) ∧
    (i < 0 → pos (i64ToF64 i) = SIGN - magBits (-i).toNat) := by
  unfold I64 at hi
  refine ⟨fun h => by subst h; decide, fun h => ?_, fun h => ?_⟩
  · have hne : i ≠ 0 := by omega
    have := magBits_finite i.toNat (by omega) (by omega)
    simp only [i64ToF64, hne, h, if_false, if_true, pos, INF, SIGN] at *
    rw [if_pos (by omega)]
  · have hne : i ≠ 0 := by omega
    have hn : ¬ 0 < i := by omega
    have := magBits_finite (-i).toNat (by omega) (by omega)
    simp only [i64ToF64, hne, hn, if_false, pos, INF, SIGN, W] at *
    rw [if_neg (by omega)]
    omega

theorem cmpI_congr {a b : Int} {c d : Nat} (h1 : a < b ↔ c < d) (h2 : a = b ↔ c = d) :
    cmpI a b = cmpN c d := by
  unfold cmpI cmpN
  by_cases x : a < b
  · simp [x, h1.1 x]
  · have y : ¬ c < d := fun y => x (h1.2 y)
    by_cases z : a = b
    · simp [z, h2.1 z]
    · have w : ¬ c = d := fun w => z (h2.2 w)
      simp [x, y, z, w]

/-- **The guard is sound for `|j| < 2^52`**: against such a bound, the exact `i64` order
    (`a.cmp(b)`, `value.rs:1260`) equals the order of the `f64` images, whatever the data `i`. -/
theorem int_cmp_agrees (i j : Int) (hi : I64 i) (hj : I64 j) (hjs : j.natAbs < 2 ^ 52) :
    cmpI i j = cmpN (pos (i64ToF64 i)) (pos (i64ToF64 j)) := by
  have Pi := pos_i64 i hi
  have Pj := pos_i64 j hj
  unfold I64 at hi hj
  have h52 : (2:Nat) ^ 52 = 4503599627370496 := by decide
  have h53 : (2:Nat) ^ 53 = 9007199254740992 := by decide
  have h64 : (2:Nat) ^ 64 = 18446744073709551616 := by decide
  have small : ∀ m : Nat, 0 < m → m < 2 ^ 52 → magBits m < 1075 * 2 ^ 52 := magBits_lt_2p52
  have big : ∀ m : Nat, m < 2 ^ 64 → 2 ^ 52 ≤ m → 1075 * 2 ^ 52 ≤ magBits m := magBits_ge
  have mono : ∀ m n : Nat, 0 < m → m < n → n < 2 ^ 53 → magBits m < magBits n := magBits_strictMono
  have fin : ∀ m : Nat, m < 2 ^ 64 → 0 < m → magBits m < INF := magBits_finite
  rw [h52] at small big hjs
  rw [h53] at mono
  rw [h64] at big fin
  simp only [INF, SIGN] at Pi Pj fin ⊢
  have t1 := Int.toNat_eq_max i
  have t2 := Int.toNat_eq_max (-i)
  have t3 := Int.toNat_eq_max j
  have t4 := Int.toNat_eq_max (-j)
  have C : ∀ (P Q : Nat), (i < j ↔ P < Q) → (i = j ↔ P = Q) → cmpI i j = cmpN P Q :=
    fun _ _ h1 h2 => cmpI_congr h1 h2
  have m0 := magBits_pos
  rcases Int.lt_trichotomy j 0 with hj0 | hj0 | hj0
  · rw [Pj.2.2 hj0]
    have mj := small (-j).toNat (by omega) (by omega)
    have mj0 := m0 (-j).toNat
    rcases Int.lt_trichotomy i 0 with hi0 | hi0 | hi0
    · rw [Pi.2.2 hi0]
      have fi := fin (-i).toNat (by omega) (by omega)
      by_cases hb : (-i).toNat < 4503599627370496
      · rcases Int.lt_trichotomy i j with h | h | h
        · have := mono (-j).toNat (-i).toNat (by omega) (by omega) (by omega)
          apply C <;> constructor <;> intro <;> omega
        · subst h; apply C <;> constructor <;> intro <;> omega
        · have := mono (-i).toNat (-j).toNat (by omega) (by omega) (by omega)
          apply C <;> constructor <;> intro <;> omega
      · have := big (-i).toNat (by omega) (by omega)
        apply C <;> constructor <;> intro <;> omega
    · rw [Pi.1 hi0]; apply C <;> constructor <;> intro <;> omega
    · rw [Pi.2.1 hi0]
      have := fin i.toNat (by omega) (by omega)
      apply C <;> constructor <;> intro <;> omega
  · rw [Pj.1 hj0]
    rcases Int.lt_trichotomy i 0 with hi0 | hi0 | hi0
    · rw [Pi.2.2 hi0]
      have := fin (-i).toNat (by omega) (by omega)
      have := m0 (-i).toNat
      apply C <;> constructor <;> intro <;> omega
    · rw [Pi.1 hi0]; apply C <;> constructor <;> intro <;> omega
    · rw [Pi.2.1 hi0]
      have := m0 i.toNat
      have := fin i.toNat (by omega) (by omega)
      apply C <;> constructor <;> intro <;> omega
  · rw [Pj.2.1 hj0]
    have mj := small j.toNat (by omega) (by omega)
    have fj := fin j.toNat (by omega) (by omega)
    have mj0 := m0 j.toNat
    rcases Int.lt_trichotomy i 0 with hi0 | hi0 | hi0
    · rw [Pi.2.2 hi0]
      have := fin (-i).toNat (by omega) (by omega)
      apply C <;> constructor <;> intro <;> omega
    · rw [Pi.1 hi0]; apply C <;> constructor <;> intro <;> omega
    · rw [Pi.2.1 hi0]
      have fi := fin i.toNat (by omega) (by omega)
      by_cases hb : i.toNat < 4503599627370496
      · rcases Int.lt_trichotomy i j with h | h | h
        · have := mono i.toNat j.toNat (by omega) (by omega) (by omega)
          apply C <;> constructor <;> intro <;> omega
        · subst h; apply C <;> constructor <;> intro <;> omega
        · have := mono j.toNat i.toNat (by omega) (by omega) (by omega)
          apply C <;> constructor <;> intro <;> omega
      · have := big i.toNat (by omega) (by omega)
        apply C <;> constructor <;> intro <;> omega

end NI
