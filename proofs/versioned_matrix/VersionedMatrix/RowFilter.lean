import Std.Tactic.BVDecide
/-
# `RowFilter` (`versioned_matrix.rs:218-293`)

Faithful bit-level model: rows are `u64` (`BitVec 64`), the Fibonacci hash
wraps (`wrapping_mul`), the bitmap is 512 words of `u64`.

| here | there |
| --- | --- |
| `RF`        | `enum RowFilter` (:235) |
| `slotH`/`slot` | `RowFilter::slot` (:250) |
| `add`       | `RowFilter::add` (:255) |
| `mayHold`   | `RowFilter::may_hold` (:277) |
| `holds`     | "row r may be present" — the bit `slot r` is set |

Safety property (the doc's "never says no when the answer is yes"):
`add_self` + `add_mono` (every added row stays recorded) and
`mayHold_sound` (a recorded row inside the range makes `may_hold` true).
`slot_word_lt`: the word index is always `< ROW_FILTER_WORDS = 512`, so
`bits[w]` never panics.
-/
namespace VMRowFilter

def K : BitVec 64 := 0x9E3779B97F4A7C15#64

inductive RF where
  | empty
  | bits (w : Nat → BitVec 64)
  | unknown

/-- `row.wrapping_mul(K) >> (64 - 15)`. -/
def slotH (row : BitVec 64) : BitVec 64 := (row * K) >>> 49

/-- `(h / 64, 1u64 << (h % 64))`. -/
def slot (row : BitVec 64) : Nat × BitVec 64 :=
  ((slotH row).toNat / 64, 1#64 <<< ((slotH row).toNat % 64))

theorem slotH_lt (row : BitVec 64) : slotH row < 32768#64 := by
  unfold slotH K; bv_decide

theorem slot_word_lt (row : BitVec 64) : (slot row).1 < 512 := by
  have h := slotH_lt row
  have : (slotH row).toNat < 32768 := by
    rw [BitVec.lt_def] at h; simpa using h
  simp [slot]; omega

theorem bit_ne_zero (k : Nat) (hk : k < 64) : (1#64 <<< k) ≠ 0 := by
  intro h
  have := congrArg BitVec.toNat h
  simp [BitVec.toNat_shiftLeft] at this
  have h2 : 2 ^ k < 2 ^ 64 := Nat.pow_lt_pow_right (by decide) hk
  rw [Nat.mod_eq_of_lt (by simpa [Nat.shiftLeft_eq] using h2)] at this
  simp [Nat.shiftLeft_eq] at this

theorem slot_bit_ne_zero (row : BitVec 64) : (slot row).2 ≠ 0 :=
  bit_ne_zero _ (Nat.mod_lt _ (by decide))

def upd (f : Nat → BitVec 64) (w : Nat) (x : BitVec 64) : Nat → BitVec 64 :=
  fun i => if i = w then x else f i

def test (f : Nat → BitVec 64) (row : BitVec 64) : Bool :=
  f (slot row).1 &&& (slot row).2 != 0

/-- `RowFilter::add` (:255). -/
def add (F : RF) (row : BitVec 64) : RF :=
  match F with
  | .unknown => .unknown
  | .empty => let (w, b) := slot row; .bits (upd (fun _ => 0) w (0 ||| b))
  | .bits f => let (w, b) := slot row; .bits (upd f w (f w ||| b))

/-- `RowFilter::may_hold` (:277). Rows are the `u64` values `min..=max`. -/
def mayHold (F : RF) (minr maxr : Nat) : Bool :=
  match F with
  | .unknown => true
  | .empty => false
  | .bits f =>
    if maxr < minr then false            -- checked_sub → None
    else if maxr - minr ≥ 64 then true
    else (List.range' minr (maxr - minr + 1)).any (fun r => test f (BitVec.ofNat 64 r))

/-- What the filter claims about row `r`. -/
def holds (F : RF) (row : BitVec 64) : Prop :=
  match F with
  | .unknown => True
  | .empty => False
  | .bits f => test f row = true

theorem or_and_self (x b : BitVec 64) (hb : b ≠ 0) : (x ||| b) &&& b ≠ 0 := by bv_decide
theorem or_and_mono (x y b : BitVec 64) (h : x &&& b ≠ 0) : (x ||| y) &&& b ≠ 0 := by bv_decide

theorem add_self (F : RF) (row : BitVec 64) : holds (add F row) row := by
  cases F <;> simp [add, holds, test, upd]
  · exact slot_bit_ne_zero row
  · exact or_and_self _ _ (slot_bit_ne_zero row)

theorem add_mono {F : RF} {r : BitVec 64} (h : holds F r) (row : BitVec 64) : holds (add F row) r := by
  cases F with
  | unknown => trivial
  | empty => exact absurd h id
  | bits f =>
    simp only [holds, add, test, upd] at *
    split
    · next heq =>
      rw [← heq]
      simp at h ⊢; exact or_and_mono _ _ _ h
    · exact h

/-- Soundness: a recorded row inside `min..=max` makes `may_hold` answer yes. -/
theorem mayHold_sound {F : RF} {r minr maxr : Nat}
    (h : holds F (BitVec.ofNat 64 r)) (h1 : minr ≤ r) (h2 : r ≤ maxr) : mayHold F minr maxr = true := by
  cases F with
  | unknown => rfl
  | empty => exact absurd h id
  | bits f =>
    simp only [mayHold]
    rw [if_neg (show ¬ maxr < minr by omega)]
    split
    · rfl
    · rw [List.any_eq_true]
      exact ⟨r, List.mem_range'_1.2 ⟨h1, by omega⟩, h⟩

/-- An empty filter answers no, which is right exactly when the layer is empty. -/
theorem mayHold_empty (minr maxr : Nat) : mayHold .empty minr maxr = false := rfl
theorem mayHold_unknown (minr maxr : Nat) : mayHold .unknown minr maxr = true := rfl
theorem mayHold_wide (f : Nat → BitVec 64) {minr maxr : Nat} (h : minr + 64 ≤ maxr) :
    mayHold (.bits f) minr maxr = true := by
  simp [mayHold]; omega

/-- Filter soundness for a set of stored rows. -/
def Sound (F : RF) (rows : List Nat) : Prop := ∀ r ∈ rows, holds F (BitVec.ofNat 64 r)

theorem sound_add {F : RF} {rows : List Nat} (h : Sound F rows) (i : Nat) :
    Sound (add F (BitVec.ofNat 64 i)) (i :: rows) := by
  intro r hr
  rcases List.mem_cons.1 hr with rfl | hr
  · exact add_self _ _
  · exact add_mono (h r hr) _

theorem sound_sub {F : RF} {rows rows' : List Nat} (h : Sound F rows) (hs : ∀ r ∈ rows', r ∈ rows) :
    Sound F rows' := fun r hr => h r (hs r hr)

theorem sound_unknown (rows : List Nat) : Sound .unknown rows := fun _ _ => trivial
theorem sound_empty : Sound .empty [] := fun _ h => absurd h (List.not_mem_nil)

end VMRowFilter
