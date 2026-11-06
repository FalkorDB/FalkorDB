/-
# `BitSet` (`graph/src/runtime/bitset.rs`) is a set of naturals

Words are modelled as `Nat` with the invariant `< 2^64` (`WF`), so `u64`
`|`, `&` are `|||`, `&&&`, and `!mask` is `(2^64 - 1) ^^^ mask`.
`1u64 << (bit % 64)` never overflows (the shift is < 64), so it is `2^(bit%64)`.

| here | there |
| --- | --- |
| `BitSet`      | `struct BitSet { inline: u64, overflow: ThinVec<u64> }` (bitset.rs:25) |
| `resizeZ`     | `ThinVec::resize(n, 0)` |
| `set`         | `BitSet::set`   (bitset.rs:31) |
| `test`        | `BitSet::test`  (bitset.rs:49) |
| `union`       | `BitSet::union` (bitset.rs:63) — `iter_mut().zip(..)` is `zipOr` |
| `clear`       | `BitSet::clear` (bitset.rs:76) |
| `isEmpty`     | `BitSet::is_empty` (bitset.rs:94) |
-/
namespace FalkorRuntimeDS.BitSetModel

def U64MAX : Nat := 2 ^ 64 - 1

structure BitSet where
  inline : Nat
  overflow : List Nat

/-- `BitSet::default()` -/
def empty : BitSet := ⟨0, []⟩

/-- `let mask = 1u64 << (bit % 64);` -/
def mask (bit : Nat) : Nat := 1 <<< (bit % 64)

/-- `Vec::resize(n, 0)` -/
def resizeZ (l : List Nat) (n : Nat) : List Nat :=
  if n ≤ l.length then l.take n else l ++ List.replicate (n - l.length) 0

/-- bitset.rs:31 -/
def set (s : BitSet) (bit : Nat) : BitSet :=
  let word := bit / 64
  let m := mask bit
  if word = 0 then { s with inline := s.inline ||| m }
  else
    let idx := word - 1
    let ov := if idx ≥ s.overflow.length then resizeZ s.overflow (idx + 1) else s.overflow
    { s with overflow := ov.set idx ((ov[idx]?.getD 0) ||| m) }

/-- bitset.rs:49 (Rust `a & m != 0` parses as `(a & m) != 0`) -/
def test (s : BitSet) (bit : Nat) : Bool :=
  let word := bit / 64
  let m := mask bit
  if word = 0 then (s.inline &&& m) != 0
  else
    let idx := word - 1
    decide (idx < s.overflow.length) && (((s.overflow[idx]?.getD 0) &&& m) != 0)

/-- `for (a, b) in self.overflow.iter_mut().zip(other.overflow.iter()) { *a |= b }` -/
def zipOr : List Nat → List Nat → List Nat
  | a :: as, b :: bs => (a ||| b) :: zipOr as bs
  | as, [] => as
  | [], _ => []

/-- bitset.rs:63 -/
def union (s o : BitSet) : BitSet :=
  let ov := if o.overflow.length > s.overflow.length then resizeZ s.overflow o.overflow.length
            else s.overflow
  { inline := s.inline ||| o.inline, overflow := zipOr ov o.overflow }

/-- bitset.rs:76 -/
def clear (s : BitSet) (bit : Nat) : BitSet :=
  let word := bit / 64
  let m := mask bit
  if word = 0 then { s with inline := s.inline &&& (U64MAX ^^^ m) }
  else
    let idx := word - 1
    if idx < s.overflow.length then
      { s with overflow := s.overflow.set idx ((s.overflow[idx]?.getD 0) &&& (U64MAX ^^^ m)) }
    else s

/-- bitset.rs:94 -/
def isEmpty (s : BitSet) : Bool := s.inline == 0 && s.overflow.all (· == 0)

/-- Every word is a `u64`. -/
def WF (s : BitSet) : Prop := s.inline < 2 ^ 64 ∧ ∀ w ∈ s.overflow, w < 2 ^ 64

/-! ## Word view -/

/-- The `n`-th 64-bit word (absent words read as 0). -/
def wordAt (s : BitSet) (n : Nat) : Nat :=
  if n = 0 then s.inline else s.overflow[n - 1]?.getD 0

theorem and_two_pow_ne_zero (w k : Nat) : ((w &&& 2 ^ k) != 0) = w.testBit k := by
  cases h : w.testBit k
  · have : w &&& 2 ^ k = 0 := by
      apply Nat.eq_of_testBit_eq; intro i
      simp only [Nat.testBit_and, Nat.testBit_two_pow, Nat.zero_testBit]
      by_cases hk : k = i
      · subst hk; simp [h]
      · simp [hk]
    simp [this]
  · have : w &&& 2 ^ k ≠ 0 := by
      intro heq
      have := congrArg (fun x => x.testBit k) heq
      simp [Nat.testBit_and, Nat.testBit_two_pow_self, h] at this
    simp [this]

theorem mask_eq (b : Nat) : mask b = 2 ^ (b % 64) := by
  simp [mask, Nat.one_shiftLeft]

theorem test_eq_wordAt (s : BitSet) (b : Nat) :
    test s b = (wordAt s (b / 64)).testBit (b % 64) := by
  unfold test wordAt
  rw [mask_eq]
  by_cases h0 : b / 64 = 0
  · simp only [h0, ite_true]; exact and_two_pow_ne_zero _ _
  · simp only [h0, ite_false]
    by_cases hl : b / 64 - 1 < s.overflow.length
    · simp [hl, and_two_pow_ne_zero]
    · have : s.overflow[b / 64 - 1]? = none := by
        rw [List.getElem?_eq_none_iff]; omega
      simp [hl, this]

theorem getElem?_resizeZ (l : List Nat) (n j : Nat) (hn : l.length ≤ n) :
    (resizeZ l n)[j]? = if j < l.length then l[j]? else if j < n then some 0 else none := by
  unfold resizeZ
  by_cases h : n ≤ l.length
  · have : n = l.length := by omega
    subst this
    simp only [Nat.le_refl, ite_true, List.take_length]
    by_cases hj : j < l.length
    · simp [hj]
    · simp [hj]
  · simp only [h, ite_false]
    rw [List.getElem?_append]
    by_cases hj : j < l.length
    · simp [hj]
    · simp only [hj, ite_false, dite_false]
      rw [List.getElem?_replicate]
      by_cases hj2 : j < n
      · simp [hj2]; omega
      · simp [hj2]; omega

theorem length_resizeZ (l : List Nat) (n : Nat) (hn : l.length ≤ n) :
    (resizeZ l n).length = n := by
  unfold resizeZ; split <;> simp <;> omega

theorem wordAt_set (s : BitSet) (b n : Nat) :
    wordAt (set s b) n = if n = b / 64 then wordAt s n ||| 2 ^ (b % 64) else wordAt s n := by
  unfold set wordAt
  rw [mask_eq]
  by_cases h0 : b / 64 = 0
  · simp only [h0, ite_true]
    by_cases hn : n = 0 <;> simp [hn]
  · simp only [h0, ite_false]
    by_cases hn : n = 0
    · simp [hn]; omega
    · simp only [hn, ite_false]
      -- the (possibly resized) overflow vector
      generalize hov : (if b / 64 - 1 ≥ s.overflow.length then
          resizeZ s.overflow (b / 64 - 1 + 1) else s.overflow) = ov
      have hlen : b / 64 - 1 < ov.length := by
        subst hov; split
        · rw [length_resizeZ _ _ (by omega)]; omega
        · omega
      have hget : ∀ j : Nat, (ov[j]?.getD 0) = (s.overflow[j]?.getD 0) ∨ j ≥ s.overflow.length := by
        intro j; subst hov; split
        · rw [getElem?_resizeZ _ _ _ (by omega)]
          by_cases hj : j < s.overflow.length
          · simp [hj]
          · right; omega
        · left; rfl
      have hget' : ∀ j : Nat, (ov[j]?.getD 0) = (s.overflow[j]?.getD 0) := by
        intro j
        rcases hget j with h | h
        · exact h
        · have : s.overflow[j]? = none := by rw [List.getElem?_eq_none_iff]; omega
          rw [this]
          subst hov; split
          · rw [getElem?_resizeZ _ _ _ (by omega)]
            simp only [show ¬ j < s.overflow.length by omega, ite_false]
            split <;> simp
          · have : s.overflow[j]? = none := by rw [List.getElem?_eq_none_iff]; omega
            rw [this]
      rw [List.getElem?_set]
      by_cases hnb : n = b / 64
      · simp only [hnb, ite_true]
        have : b / 64 - 1 < ov.length := hlen
        have e := hget' (b / 64 - 1)
        rw [List.getElem?_eq_getElem this] at e
        simp only [Option.getD_some] at e
        simp [this, ← e]
      · have : b / 64 - 1 ≠ n - 1 := by omega
        simp only [this, ite_false, hnb]
        exact hget' _

theorem wordAt_clear (s : BitSet) (b n : Nat) :
    wordAt (clear s b) n =
      if n = b / 64 then wordAt s n &&& (U64MAX ^^^ 2 ^ (b % 64)) else wordAt s n := by
  unfold clear wordAt
  rw [mask_eq]
  by_cases h0 : b / 64 = 0
  · simp only [h0, ite_true]
    by_cases hn : n = 0 <;> simp [hn]
  · simp only [h0, ite_false]
    by_cases hl : b / 64 - 1 < s.overflow.length
    · simp only [hl, ite_true]
      by_cases hn : n = 0
      · simp [hn]; omega
      · simp only [hn, ite_false]
        rw [List.getElem?_set]
        by_cases hnb : n = b / 64
        · simp [hnb, hl]
        · have : b / 64 - 1 ≠ n - 1 := by omega
          simp [this, hnb]
    · simp only [hl, ite_false]
      by_cases hn : n = 0
      · simp [hn]; omega
      · simp only [hn, ite_false]
        by_cases hnb : n = b / 64
        · have : s.overflow[n - 1]? = none := by rw [List.getElem?_eq_none_iff]; omega
          simp [hnb, this]
          rw [List.getElem?_eq_none_iff.mpr (by omega)]
          simp
        · simp [hnb]

theorem getElem?_zipOr (a b : List Nat) (j : Nat) (h : b.length ≤ a.length) :
    ((zipOr a b)[j]?.getD 0) = (a[j]?.getD 0) ||| (b[j]?.getD 0) := by
  induction a generalizing b j with
  | nil =>
    cases b with
    | nil => simp [zipOr]
    | cons => simp at h
  | cons x xs ih =>
    cases b with
    | nil => simp [zipOr]
    | cons y ys =>
      cases j with
      | zero => simp [zipOr]
      | succ j => simp [zipOr]; exact ih ys j (by simp at h; omega)

theorem wordAt_union (s o : BitSet) (n : Nat) :
    wordAt (union s o) n = wordAt s n ||| wordAt o n := by
  unfold union wordAt
  by_cases hn : n = 0
  · simp [hn]
  · simp only [hn, ite_false]
    split
    · rename_i hgt
      rw [getElem?_zipOr _ _ _ (by rw [length_resizeZ _ _ (by omega)]; exact Nat.le_refl _)]
      rw [getElem?_resizeZ _ _ _ (by omega)]
      by_cases hj : n - 1 < s.overflow.length
      · simp [hj]
      · have : s.overflow[n - 1]? = none := by rw [List.getElem?_eq_none_iff]; omega
        rw [this]
        by_cases hj2 : n - 1 < o.overflow.length <;> simp [hj2, hj]
    · rw [getElem?_zipOr _ _ _ (by omega)]

/-! ## Reference semantics: `test` is membership in a set of naturals -/

theorem test_empty (c : Nat) : test empty c = false := by
  rw [test_eq_wordAt]; simp [wordAt, empty]

/-- `set` adds exactly `b` — including across the 63/64 word boundary. -/
theorem test_set (s : BitSet) (b c : Nat) : test (set s b) c = (test s c || decide (c = b)) := by
  rw [test_eq_wordAt, test_eq_wordAt, wordAt_set]
  by_cases h : c / 64 = b / 64
  · simp only [h, ite_true, Nat.testBit_or, Nat.testBit_two_pow]
    congr 1
    apply Bool.eq_iff_iff.mpr; simp; omega
  · simp only [h, ite_false]
    have : c ≠ b := fun e => h (e ▸ rfl)
    simp [this]

/-- `clear` removes exactly `b` and nothing else. -/
theorem test_clear (s : BitSet) (b c : Nat) : test (clear s b) c = (test s c && !decide (c = b)) := by
  rw [test_eq_wordAt, test_eq_wordAt, wordAt_clear]
  by_cases h : c / 64 = b / 64
  · simp only [h, ite_true, Nat.testBit_and, Nat.testBit_xor, U64MAX, Nat.testBit_two_pow_sub_one,
      Nat.testBit_two_pow]
    have h64 : c % 64 < 64 := Nat.mod_lt _ (by decide)
    congr 1
    apply Bool.eq_iff_iff.mpr; simp [h64]; omega
  · simp only [h, ite_false]
    have : c ≠ b := fun e => h (e ▸ rfl)
    simp [this]

/-- `union` is set union. -/
theorem test_union (s o : BitSet) (c : Nat) : test (union s o) c = (test s c || test o c) := by
  rw [test_eq_wordAt, test_eq_wordAt, test_eq_wordAt, wordAt_union, Nat.testBit_or]

/-! ## Well-formedness (every word stays a `u64`) and `is_empty` -/

theorem getD_lt {l : List Nat} (h : ∀ w ∈ l, w < 2 ^ 64) (j : Nat) : l[j]?.getD 0 < 2 ^ 64 := by
  cases hj : l[j]? with
  | none => simp
  | some w => simp; exact h w (List.mem_of_getElem? hj)

theorem wf_iff_wordAt (s : BitSet) :
    WF s ↔ ∀ n, wordAt s n < 2 ^ 64 := by
  constructor
  · rintro ⟨hi, ho⟩ n
    unfold wordAt; split
    · exact hi
    · exact getD_lt ho _
  · intro h
    refine ⟨by simpa [wordAt] using h 0, ?_⟩
    intro w hw
    obtain ⟨j, hj, rfl⟩ := List.getElem_of_mem hw
    have := h (j + 1)
    simp [wordAt, List.getElem?_eq_getElem hj] at this
    exact this

theorem mask_lt (b : Nat) : 2 ^ (b % 64) < 2 ^ 64 :=
  Nat.pow_lt_pow_right (by decide) (Nat.mod_lt _ (by decide))

theorem wf_empty : WF empty := by simp [WF, empty]

theorem wf_set {s : BitSet} (h : WF s) (b : Nat) : WF (set s b) := by
  rw [wf_iff_wordAt] at *
  intro n; rw [wordAt_set]; split
  · exact Nat.or_lt_two_pow (h n) (mask_lt b)
  · exact h n

theorem wf_clear {s : BitSet} (h : WF s) (b : Nat) : WF (clear s b) := by
  rw [wf_iff_wordAt] at *
  intro n; rw [wordAt_clear]; split
  · exact Nat.lt_of_le_of_lt Nat.and_le_left (h n)
  · exact h n

theorem wf_union {s o : BitSet} (hs : WF s) (ho : WF o) : WF (union s o) := by
  rw [wf_iff_wordAt] at *
  intro n; rw [wordAt_union]; exact Nat.or_lt_two_pow (hs n) (ho n)

theorem word_eq_zero_of_bits {w : Nat} (hw : w < 2 ^ 64) (h : ∀ i < 64, w.testBit i = false) :
    w = 0 := by
  apply Nat.eq_of_testBit_eq; intro i
  by_cases hi : i < 64
  · simp [h i hi]
  · simp only [Nat.zero_testBit]
    exact Nat.testBit_lt_two_pow (Nat.lt_of_lt_of_le hw (Nat.pow_le_pow_right (by decide) (by omega)))

/-- `is_empty` is exact: it is true iff no bit tests true (for well-formed sets). -/
theorem isEmpty_iff (s : BitSet) (hs : WF s) :
    isEmpty s = true ↔ ∀ c, test s c = false := by
  constructor
  · intro h c
    simp only [isEmpty, Bool.and_eq_true, beq_iff_eq, List.all_eq_true] at h
    rw [test_eq_wordAt]
    have : wordAt s (c / 64) = 0 := by
      unfold wordAt; split
      · exact h.1
      · cases hj : s.overflow[c / 64 - 1]? with
        | none => simp
        | some w => simpa using h.2 w (List.mem_of_getElem? hj)
    simp [this]
  · intro h
    have hw : ∀ n, wordAt s n = 0 := by
      intro n
      apply word_eq_zero_of_bits ((wf_iff_wordAt s).mp hs n)
      intro i hi
      have := h (n * 64 + i)
      rw [test_eq_wordAt] at this
      have e1 : (n * 64 + i) / 64 = n := by omega
      have e2 : (n * 64 + i) % 64 = i := by omega
      rwa [e1, e2] at this
    simp only [isEmpty, Bool.and_eq_true, beq_iff_eq, List.all_eq_true]
    refine ⟨by simpa [wordAt] using hw 0, ?_⟩
    intro w hmem
    obtain ⟨j, hj, rfl⟩ := List.getElem_of_mem hmem
    have := hw (j + 1)
    simpa [wordAt, List.getElem?_eq_getElem hj] using this

/-! ## Word-boundary sanity checks (63 / 64 / 127 / 128) -/

example : test (set empty 63) 63 = true := by decide
example : test (set empty 63) 64 = false := by decide
example : test (set empty 64) 64 = true := by decide
example : test (set empty 64) 0 = false := by decide
example : test (set empty 128) 128 = true := by decide
example : test (set empty 128) 64 = false := by decide
example : isEmpty (clear (set empty 64) 64) = true := by decide
example : test (union (set empty 200) (set empty 3)) 200 = true := by decide

end FalkorRuntimeDS.BitSetModel
