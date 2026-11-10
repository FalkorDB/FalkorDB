/-
# Finite sets of ids, and counting them

A `RoaringTreemap` is a finite set of `u64`. Here it is a membership predicate
`Nat → Bool` read over a universe `[0, N)`, with `N = 2^64` for the engine and a
small `N` for `#eval`. `cnt n s` is `|s ∩ [0, n)|` — which is also roaring's
`rank(n - 1)` — and `len` is `cnt N`.
-/

namespace IdSpaceModel

abbrev IdSet := Nat → Bool

/-- `|{ i < n | s i }|`. -/
def cnt : Nat → IdSet → Nat
  | 0, _ => 0
  | n + 1, s => cnt n s + (if s n then 1 else 0)

@[simp] theorem cnt_zero (s : IdSet) : cnt 0 s = 0 := rfl
theorem cnt_succ (n : Nat) (s : IdSet) :
    cnt (n + 1) s = cnt n s + (if s n then 1 else 0) := rfl

theorem cnt_le (n : Nat) (s : IdSet) : cnt n s ≤ n := by
  induction n with
  | zero => simp
  | succ n ih => rw [cnt_succ]; split <;> omega

theorem cnt_mono_n {m n : Nat} (s : IdSet) (h : m ≤ n) : cnt m s ≤ cnt n s := by
  induction n with
  | zero => have : m = 0 := by omega
            subst this; simp
  | succ n ih =>
    rcases Nat.lt_or_ge m (n + 1) with h' | h'
    · have := ih (by omega); rw [cnt_succ]; omega
    · have : m = n + 1 := by omega
      subst this; omega

/-- Counting is monotone in the set. -/
theorem cnt_mono {s t : IdSet} (n : Nat) (h : ∀ i, i < n → s i = true → t i = true) :
    cnt n s ≤ cnt n t := by
  induction n with
  | zero => simp
  | succ n ih =>
    rw [cnt_succ, cnt_succ]
    have ih' := ih (fun i hi => h i (by omega))
    by_cases hs : s n = true
    · have ht := h n (by omega) hs
      simp [hs, ht]; omega
    · simp only [Bool.not_eq_true] at hs; simp [hs]; split <;> omega

theorem cnt_congr {s t : IdSet} (n : Nat) (h : ∀ i, i < n → s i = t i) :
    cnt n s = cnt n t := by
  have a := cnt_mono (s := s) (t := t) n (fun i hi hs => by rw [← h i hi]; exact hs)
  have b := cnt_mono (s := t) (t := s) n (fun i hi hs => by rw [h i hi]; exact hs)
  omega

/-- Inclusion–exclusion. -/
theorem cnt_or_and (n : Nat) (s t : IdSet) :
    cnt n (fun i => s i || t i) + cnt n (fun i => s i && t i) = cnt n s + cnt n t := by
  induction n with
  | zero => simp
  | succ n ih =>
    simp only [cnt_succ]
    cases hs : s n <;> cases ht : t n <;> simp <;> omega

theorem cnt_false (n : Nat) : cnt n (fun _ => false) = 0 := by
  induction n with
  | zero => rfl
  | succ n ih => rw [cnt_succ]; simp [ih]

/-- Disjoint union adds. -/
theorem cnt_or_disjoint (n : Nat) (s t : IdSet) (h : ∀ i, i < n → ¬ (s i = true ∧ t i = true)) :
    cnt n (fun i => s i || t i) = cnt n s + cnt n t := by
  have e := cnt_or_and n s t
  have := cnt_mono (s := fun i => s i && t i) (t := fun _ => false) n (by
    intro i hi hst; simp only [Bool.and_eq_true] at hst; exact absurd hst (h i hi))
  rw [cnt_false] at this
  omega

/-- A set splits into the part in `t` and the part outside it. -/
theorem cnt_split (n : Nat) (s t : IdSet) :
    cnt n s = cnt n (fun i => s i && t i) + cnt n (fun i => s i && !t i) := by
  induction n with
  | zero => simp
  | succ n ih =>
    simp only [cnt_succ]
    cases hs : s n <;> cases ht : t n <;> simp <;> omega

theorem cnt_eq_zero {n : Nat} {s : IdSet} : cnt n s = 0 ↔ ∀ i, i < n → s i = false := by
  induction n with
  | zero => simp
  | succ n ih =>
    rw [cnt_succ]
    constructor
    · intro h i hi
      have h1 : cnt n s = 0 := by split at h <;> omega
      rcases Nat.lt_or_ge i n with h' | h'
      · exact ih.mp h1 i h'
      · have : i = n := by omega
        subst this; split at h <;> simp_all
    · intro h
      have h1 := ih.mpr (fun i hi => h i (by omega))
      simp [h1, h n (by omega)]

theorem cnt_pos {n : Nat} {s : IdSet} : cnt n s ≠ 0 ↔ ∃ i, i < n ∧ s i = true := by
  constructor
  · intro h
    apply Classical.byContradiction
    intro hn
    apply h
    apply cnt_eq_zero.mpr
    intro i hi
    cases hs : s i
    · rfl
    · exact absurd ⟨i, hi, hs⟩ hn
  · intro ⟨i, hi, hs⟩ h
    have := cnt_eq_zero.mp h i hi
    simp_all

/-- The interval `[a, b)` has `b - a` members below any `n ≥ b`. -/
theorem cnt_interval (a b n : Nat) (hb : b ≤ n) :
    cnt n (fun i => decide (a ≤ i) && decide (i < b)) = b - a := by
  induction n generalizing b with
  | zero => have : b = 0 := by omega
            subst this; simp
  | succ n ih =>
    rw [cnt_succ]
    rcases Nat.lt_or_ge b (n + 1) with h | h
    · rw [ih b (by omega)]
      have : ¬ (n < b) := by omega
      simp [this]
    · have hbn : b = n + 1 := by omega
      subst hbn
      have e : cnt n (fun i => decide (a ≤ i) && decide (i < n + 1))
             = cnt n (fun i => decide (a ≤ i) && decide (i < n)) :=
        cnt_congr n (fun i hi => by
          have : i < n + 1 := by omega
          simp [this, hi])
      rw [e, ih n (Nat.le_refl n)]
      by_cases ha : a ≤ n
      · simp [ha]; omega
      · simp [ha]; omega

/-- **Pigeonhole.** A subset of `t` (within `[0, n)`) with as many members is
    all of `t`. What makes `verify`'s two scalar comparisons a statement about a
    whole set. -/
theorem cnt_sub_eq {s t : IdSet} (n : Nat) (hsub : ∀ i, i < n → s i = true → t i = true)
    (heq : cnt n s = cnt n t) : ∀ i, i < n → t i = true → s i = true := by
  intro i hi hti
  apply Classical.byContradiction
  intro hsi
  simp only [Bool.not_eq_true] at hsi
  -- s ⊆ t \ {i}, so cnt s ≤ cnt t - 1
  have h1 := cnt_mono (s := s) (t := fun j => t j && !(decide (j = i))) n (by
    intro j hj hs
    have : j ≠ i := fun e => by subst e; simp_all
    simp [hsub j hj hs, this])
  have h2 := cnt_split n t (fun j => decide (j = i))
  have h3 : cnt n (fun j => t j && decide (j = i)) ≠ 0 :=
    cnt_pos.mpr ⟨i, hi, by simp [hti]⟩
  omega

/-- The lowest member below `n`, as roaring's `min()`. -/
def lowest (n : Nat) (s : IdSet) : Option Nat := (asc' n s).head?
where asc' (n : Nat) (s : IdSet) : List Nat := (List.range n).filter s

/-- The members below `n`, ascending — roaring's `iter()`. -/
def asc (n : Nat) (s : IdSet) : List Nat := (List.range n).filter s

theorem mem_asc {n : Nat} {s : IdSet} {i : Nat} : i ∈ asc n s ↔ i < n ∧ s i = true := by
  simp [asc]

theorem asc_nodup (n : Nat) (s : IdSet) : (asc n s).Nodup :=
  List.filter_sublist.nodup List.nodup_range

theorem asc_length (n : Nat) (s : IdSet) : (asc n s).length = cnt n s := by
  induction n with
  | zero => rfl
  | succ n ih =>
    simp only [asc, List.range_succ, List.filter_append, List.length_append] at *
    rw [ih, cnt_succ]
    cases h : s n <;> simp [h]

theorem asc_congr {n : Nat} {s t : IdSet} (h : ∀ i, i < n → s i = t i) : asc n s = asc n t := by
  unfold asc
  apply List.filter_congr
  intro x hx
  exact h x (List.mem_range.mp hx)

end IdSpaceModel
