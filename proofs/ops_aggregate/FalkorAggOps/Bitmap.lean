import Std.Tactic.BVDecide
/-
`NullBitmap` (graph/src/runtime/batch.rs:89-167): bit `i` of word `i / 64`
is set iff row `i` is null. Words are `u64` = `BitVec 64`; the `Vec<u64>` is
modelled as a total function `Nat → BitVec 64` (indices ≥ `num_words` are
never read: every accessor `debug_assert!`s `idx < len`).
-/
namespace FalkorAggOps

structure Bitmap where
  words : Nat → BitVec 64
  len : Nat

/-- `NullBitmap::none` (batch.rs:98). -/
def Bitmap.none (len : Nat) : Bitmap := ⟨fun _ => 0, len⟩

/-- `NullBitmap::is_null` (batch.rs:133): `(words[idx/64] >> (idx%64)) & 1 != 0`. -/
def Bitmap.isNull (b : Bitmap) (i : Nat) : Bool :=
  ((b.words (i / 64) >>> (i % 64)) &&& 1) != 0

/-- `NullBitmap::set` (batch.rs:150): `words[idx/64] |= 1 << (idx%64)`. -/
def Bitmap.set (b : Bitmap) (i : Nat) : Bitmap :=
  { b with words := fun w => if w = i / 64 then b.words w ||| ((1 : BitVec 64) <<< (i % 64)) else b.words w }

/-- `NullBitmap::clear` (batch.rs:160): `words[idx/64] &= !(1 << (idx%64))`. -/
def Bitmap.clear (b : Bitmap) (i : Nat) : Bitmap :=
  { b with words := fun w => if w = i / 64 then b.words w &&& ~~~((1 : BitVec 64) <<< (i % 64)) else b.words w }

/-- `NullBitmap::all` (batch.rs:108): `set` every row. -/
def Bitmap.all (len : Nat) : Bitmap := (List.range len).foldl Bitmap.set (Bitmap.none len)

/-- `NullBitmap::from_values` (batch.rs:118). -/
def Bitmap.fromNulls (nulls : List Bool) : Bitmap :=
  (List.range nulls.length).foldl
    (fun b i => if nulls.getD i false then b.set i else b) (Bitmap.none nulls.length)

theorem bit_test (w : BitVec 64) (a : Nat) (ha : a < 64) :
    ((w >>> a) &&& 1) != 0 ↔ w.getLsbD a = true := by
  have key : ∀ (v : BitVec 64), (v &&& 1) != 0 ↔ v.getLsbD 0 = true := by
    intro v; simp only [bne_iff_ne, ne_eq]; bv_decide
  rw [key]; simp [BitVec.getLsbD_ushiftRight]

theorem getLsbD_one_shl (a b : Nat) (ha : a < 64) (hb : b < 64) :
    ((1 : BitVec 64) <<< a).getLsbD b = (a == b) := by
  simp [BitVec.getLsbD_shiftLeft, hb]
  by_cases h : a = b
  · subst h; simp
  · have : b - a ≠ 0 ∨ b < a := by omega
    rcases this with h1 | h1
    · by_cases hlt : b < a
      · simp [hlt, h]
      · simp [hlt, h1, h]
    · simp [h1, h]

theorem isNull_iff (b : Bitmap) (i : Nat) : b.isNull i = (b.words (i / 64)).getLsbD (i % 64) := by
  unfold Bitmap.isNull
  have := bit_test (b.words (i / 64)) (i % 64) (Nat.mod_lt _ (by decide))
  cases h : (b.words (i / 64)).getLsbD (i % 64) <;> simp_all

theorem none_isNull (len i : Nat) : (Bitmap.none len).isNull i = false := by
  rw [isNull_iff]; simp [Bitmap.none]

theorem same_slot (i j : Nat) (h1 : j / 64 = i / 64) (h2 : i % 64 = j % 64) : i = j := by omega

/-- `set i` makes row `i` null and leaves every other row alone. -/
theorem set_isNull (b : Bitmap) (i j : Nat) :
    (b.set i).isNull j = (b.isNull j || i == j) := by
  rw [isNull_iff, isNull_iff]
  simp only [Bitmap.set]
  by_cases hw : j / 64 = i / 64
  · simp only [hw, ite_true, BitVec.getLsbD_or]
    rw [getLsbD_one_shl _ _ (Nat.mod_lt _ (by decide)) (Nat.mod_lt _ (by decide))]
    rw [← hw]
    by_cases hij : i = j
    · subst hij; simp
    · have : ¬ (i % 64 = j % 64) := fun e => hij (same_slot i j hw e)
      have e1 : (i % 64 == j % 64) = false := by simp [this]
      have e2 : (i == j) = false := by simp [hij]
      rw [e1, e2]
  · have : i ≠ j := by intro e; subst e; exact hw rfl
    simp [hw, this]

/-- `clear i` makes row `i` non-null and leaves every other row alone. -/
theorem clear_isNull (b : Bitmap) (i j : Nat) :
    (b.clear i).isNull j = (b.isNull j && i != j) := by
  rw [isNull_iff, isNull_iff]
  simp only [Bitmap.clear]
  by_cases hw : j / 64 = i / 64
  · simp only [hw, ite_true, BitVec.getLsbD_and, BitVec.getLsbD_not]
    rw [getLsbD_one_shl _ _ (Nat.mod_lt _ (by decide)) (Nat.mod_lt _ (by decide))]
    rw [← hw]
    by_cases hij : i = j
    · subst hij; simp [Nat.mod_lt]
    · have : ¬ (i % 64 = j % 64) := fun e => hij (same_slot i j hw e)
      have e1 : (i % 64 == j % 64) = false := by simp [this]
      have e2 : (i != j) = true := by simp [hij]
      rw [e1, e2]; simp [Nat.mod_lt j (show 64 > 0 by decide)]
  · have : i ≠ j := by intro e; subst e; exact hw rfl
    simp [hw, this]

theorem foldl_set_isNull (xs : List Nat) (b : Bitmap) (j : Nat) :
    (xs.foldl Bitmap.set b).isNull j = (b.isNull j || j ∈ xs) := by
  induction xs generalizing b with
  | nil => simp
  | cons x xs ih =>
    simp only [List.foldl, ih, set_isNull, List.mem_cons]
    by_cases h : x = j
    · subst h; simp
    · have : decide (j = x) = false := by simp [Ne.symm h]
      have h2 : (x == j) = false := by simp [h]
      simp [this, h2]

/-- `all len` marks exactly rows `0..len` null. -/
theorem all_isNull (len j : Nat) : (Bitmap.all len).isNull j = decide (j < len) := by
  unfold Bitmap.all; rw [foldl_set_isNull, none_isNull]; simp

/-- `from_values` marks exactly the `Null` positions. -/
theorem foldl_setIf_isNull (p : Nat → Bool) (j : Nat) : ∀ (xs : List Nat) (b : Bitmap),
    (xs.foldl (fun b i => if p i then b.set i else b) b).isNull j =
      (b.isNull j || (decide (j ∈ xs) && p j))
  | [], b => by simp
  | x :: xs, b => by
    simp only [List.foldl]
    rw [foldl_setIf_isNull p j xs]
    by_cases hx : x = j
    · subst hx
      cases hp : p x <;> simp [hp, set_isNull]
    · have e1 : decide (j = x) = false := by simp [Ne.symm hx]
      have e2 : (x == j) = false := by simp [hx]
      cases hp : p x <;> simp [set_isNull, e1, e2, List.mem_cons]

/-- `from_values` marks exactly the `Null` positions. -/
theorem fromNulls_isNull (nulls : List Bool) (j : Nat) (hj : j < nulls.length) :
    (Bitmap.fromNulls nulls).isNull j = nulls.getD j false := by
  unfold Bitmap.fromNulls
  rw [foldl_setIf_isNull (fun i => nulls.getD i false), none_isNull]; simp [hj]

end FalkorAggOps
