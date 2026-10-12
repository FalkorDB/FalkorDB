import FalkorNativeIndex.Value
set_option linter.unusedSimpArgs false

/-!
# The Filter agrees with the encoded keys (`encode.rs:152-246`, the "no post-filter" contract)

`Node By Index Scan` answers `Equal` / `Range` with **no post-filter**, so the key order must
decide each comparison exactly as the evaluator does. Proven (`filter_iff_key`): for every data
value `d` — any variant, any NaN payload, `-0.0`, `±inf`, any `i64` — and a numeric bound `b` that is
a non-NaN `Float` or an `Int` with `|b| < 2^52`, `d <op> b` holds in the Filter iff `d` is indexed
and its key satisfies `<op>` against the bound's key. `Bool` is excluded on both sides.

Refuted (counterexamples):
* `bool_divergence` — a `Bool` and a number share a key, the evaluator never equates them. Pinned in
  the Rust (`encode.rs:248-294`) as the engine-wide behaviour C and the RediSearch path share.
* `i64min_hole` — the precision guard (`int_loses_f64_precision`, `index/mod.rs:1371`) admits
  `i64::MIN` (its `unsigned_abs` is `2^63`, outside the mask), but `i64::MIN + 1 … i64::MIN + 512`
  round to the same `f64`: `n.v > -9223372036854775808` misses them, `=` returns them.
-/

namespace NI

/-- The bound a numeric index query can carry and the guard lets through. -/
def GoodBound : Value → Prop
  | .int j => I64 j ∧ j.natAbs < 2 ^ 52
  | .float f => f < W ∧ isNaN f = false
  | _ => False

/-- A data value the Filter and the index treat alike: anything but a `Bool`. -/
def NotBool : Value → Prop
  | .bool _ => False
  | _ => True

/-- `Ordering` vs key predicate. -/
theorem ord_keyPred (op : Op) (k kb : Nat) (hop : op ≠ .eq) :
    (match op with
      | .lt => cmpN k kb == .lt
      | .gt => cmpN k kb == .gt
      | .le => cmpN k kb != .gt
      | _ => cmpN k kb != .lt) = keyPred op k kb := by
  have a1 : k < kb → ¬ kb < k := fun h => by omega
  have a2 : k < kb → ¬ k = kb := fun h => by omega
  have a3 : ¬ k < kb → ¬ k = kb → kb < k := fun h h' => by omega
  cases op
  all_goals
    by_cases h1 : k < kb <;> by_cases h2 : k = kb
  all_goals first
    | exact absurd rfl hop
    | (subst h2; simp [cmpN, keyPred])
    | simp [cmpN, keyPred, h1, h2, a1 h1, a2 h1, Nat.le_of_lt h1]
    | simp [cmpN, keyPred, h1, h2, a3 h1 h2, Nat.le_of_lt (a3 h1 h2), Nat.not_le_of_lt (a3 h1 h2)]

/-- The numeric image of a numeric value, as `f64` bits (what `encode_numeric` encodes). -/
def fimg : Value → Option Nat
  | .int i => some (i64ToF64 i)
  | .float f => some f
  | _ => none

theorem encodeNumeric_int (i : Int) : encodeNumeric (.int i) =
    if !isNaN (i64ToF64 i) then some (encodeF64 (i64ToF64 i)) else none := rfl
theorem encodeNumeric_float (f : Nat) : encodeNumeric (.float f) =
    if !isNaN f then some (encodeF64 f) else none := rfl

/-- Both sides numeric and non-NaN: `compare_value` is the key order, flag `None`. -/
theorem compare_numeric (d b : Value) (hd : d.WF) (hb : GoodBound b) (x : Nat) (hx : fimg d = some x)
    (hxn : isNaN x = false) :
    ∃ kd kb, encodeNumeric d = some kd ∧ encodeNumeric b = some kb ∧
      compareValue d b = (cmpN kd kb, .none) := by
  cases d <;> simp [fimg] at hx <;> cases b <;> simp [GoodBound] at hb
  · -- int, int
    rename_i i j
    subst hx
    have ⟨hiW, hiN⟩ := i64ToF64_lt i hd
    have ⟨hjW, hjN⟩ := i64ToF64_lt j hb.1
    refine ⟨encodeF64 (i64ToF64 i), encodeF64 (i64ToF64 j), by simp [encodeNumeric_int, hiN], by simp [encodeNumeric_int, hjN], ?_⟩
    simp only [compareValue]
    have h1 := int_cmp_agrees i j hd hb.1 hb.2
    have h2 := encodeF64_cmp _ _ hiW hjW hiN hjN
    simp only [fpcmp, hiN, hjN, Bool.or_false, Bool.false_eq_true, ite_false, Option.some.injEq] at h2
    rw [h1, h2]
  · -- int, float
    rename_i i f
    subst hx
    have ⟨hiW, hiN⟩ := i64ToF64_lt i hd
    refine ⟨encodeF64 (i64ToF64 i), encodeF64 f, by simp [encodeNumeric_int, hiN], by simp [encodeNumeric_float, hb.2], ?_⟩
    simp only [compareValue, compareFloats, encodeF64_cmp _ _ hiW hb.1 hiN hb.2]
  · -- float, int
    rename_i f j
    subst hx
    have ⟨hjW, hjN⟩ := i64ToF64_lt j hb.1
    refine ⟨encodeF64 f, encodeF64 (i64ToF64 j), by simp [encodeNumeric_float, hxn], by simp [encodeNumeric_int, hjN], ?_⟩
    simp only [compareValue, compareFloats, encodeF64_cmp _ _ hd hjW hxn hjN]
  · -- float, float
    rename_i f g
    subst hx
    refine ⟨encodeF64 f, encodeF64 g, by simp [encodeNumeric_float, hxn], by simp [encodeNumeric_float, hb.2], ?_⟩
    simp only [compareValue, compareFloats, encodeF64_cmp _ _ hd hb.1 hxn hb.2]

/-- **Filter ≡ key predicate.** -/
theorem filter_iff_key (op : Op) (d b : Value) (hd : d.WF) (hnb : NotBool d) (hb : GoodBound b)
    (kb : Nat) (hkb : encodeNumeric b = some kb) :
    cmpPred op d b = some true ↔ ∃ k, encodeNumeric d = some k ∧ keyPred op k kb = true := by
  -- numeric, non-NaN data
  have numeric : ∀ x, fimg d = some x → isNaN x = false →
      (cmpPred op d b = some true ↔ ∃ k, encodeNumeric d = some k ∧ keyPred op k kb = true) := by
    intro x hx hxn
    obtain ⟨kd, kb', h1, h2, h3⟩ := compare_numeric d b hd hb x hx hxn
    rw [hkb] at h2; cases h2
    rw [h1]
    simp only [cmpPred, h3, Option.some.injEq, exists_eq_left']
    cases op <;> simp only [keyPred] <;>
    by_cases h : kd < kb <;> by_cases h' : kd = kb <;> simp [cmpN, h, h'] <;> omega
  -- a NaN float: the Filter says false, the index never stored it
  have nanCase : ∀ x, d = .float x → isNaN x = true →
      (cmpPred op d b = some true ↔ ∃ k, encodeNumeric d = some k ∧ keyPred op k kb = true) := by
    intro x hdx hn
    subst hdx
    have : encodeNumeric (.float x) = none := by simp [encodeNumeric_float, hn]
    rw [this]
    cases b <;> simp [GoodBound] at hb
    · simp [cmpPred, compareValue, compareFloats, fpcmp, hn]; cases op <;> simp
    · simp [cmpPred, compareValue, compareFloats, fpcmp, hn]; cases op <;> simp
  cases d with
  | int i =>
    exact numeric _ rfl (i64ToF64_lt i hd).2
  | float x =>
    cases hn : isNaN x
    · exact numeric _ rfl hn
    · exact nanCase x rfl hn
  | bool _ => exact absurd hnb (by simp [NotBool])
  | null => cases b <;> simp [GoodBound] at hb <;> cases op <;> simp [cmpPred, compareValue, encodeNumeric]
  | string _ => cases b <;> simp [GoodBound] at hb <;> cases op <;> simp [cmpPred, compareValue, encodeNumeric]
  | date _ => cases b <;> simp [GoodBound] at hb <;> cases op <;> simp [cmpPred, compareValue, encodeNumeric]
  | datetime _ => cases b <;> simp [GoodBound] at hb <;> cases op <;> simp [cmpPred, compareValue, encodeNumeric]
  | time _ => cases b <;> simp [GoodBound] at hb <;> cases op <;> simp [cmpPred, compareValue, encodeNumeric]
  | duration _ => cases b <;> simp [GoodBound] at hb <;> cases op <;> simp [cmpPred, compareValue, encodeNumeric]
  | other _ => cases b <;> simp [GoodBound] at hb <;> cases op <;> simp [cmpPred, compareValue, encodeNumeric]

/-! ## Counterexamples -/

/-- **Bool divergence** (pinned in `encode.rs:272-294`, shared with C and the RediSearch path):
    `true` and `1` get one key, but `true = 1` is `false` in the Filter. -/
theorem bool_divergence :
    encodeNumeric (.bool true) = encodeNumeric (.int 1) ∧ cmpPred .eq (.bool true) (.int 1) = some false := by
  decide

/-- **`i64::MIN` passes the guard but is not safe**: `i64::MIN + 1 > i64::MIN` in the Filter, yet both
    encode to one key, so the range scan (`lo = key + 1`) drops the row and an `Equal` scan adds it. -/
theorem i64min_hole :
    intLosesPrecision (-9223372036854775808) = false ∧
    cmpPred .gt (.int (-9223372036854775807)) (.int (-9223372036854775808)) = some true ∧
    cmpPred .eq (.int (-9223372036854775807)) (.int (-9223372036854775808)) = some false ∧
    encodeNumeric (.int (-9223372036854775807)) = encodeNumeric (.int (-9223372036854775808)) := by
  decide

/-- The rounding window: `i64::MIN + 512` still collides, `i64::MIN + 513` does not. -/
theorem i64min_window :
    encodeNumeric (.int (-9223372036854775808 + 512)) = encodeNumeric (.int (-9223372036854775808)) ∧
    encodeNumeric (.int (-9223372036854775808 + 1025)) ≠ encodeNumeric (.int (-9223372036854775808)) := by
  decide

/-- `int_loses_f64_precision` does reject `2^53` (the test's exempted pair). -/
theorem guard_rejects_2p53 : intLosesPrecision 9007199254740992 = true := by decide

/-- ...whose collision with `2^53 + 1` is real. -/
theorem p53_collision :
    cmpPred .lt (.int 9007199254740992) (.int 9007199254740993) = some true ∧
    encodeNumeric (.int 9007199254740992) = encodeNumeric (.int 9007199254740993) := by
  decide

/-- `-0.0` and `+0.0`: one key, and the Filter agrees they are equal. -/
theorem negZero_agrees :
    encodeNumeric (.float SIGN) = encodeNumeric (.float 0) ∧ cmpPred .eq (.float SIGN) (.float 0) = some true := by
  decide

/-- Every NaN (any sign, any payload) is unindexed. -/
theorem nan_unindexed (x : Nat) (h : isNaN x = true) : encodeNumeric (.float x) = none := by
  simp [encodeNumeric_float, h]

/-- Temporals are not indexed (`encode.rs:34-38`). -/
theorem temporal_unindexed (i : Int) :
    encodeNumeric (.date i) = none ∧ encodeNumeric (.datetime i) = none ∧
    encodeNumeric (.time i) = none ∧ encodeNumeric (.duration i) = none := by
  simp [encodeNumeric]

end NI
