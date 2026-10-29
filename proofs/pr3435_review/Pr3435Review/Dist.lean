/-! # BFS distances and the `count` of a GraphBLAS vector

Shared foundations for `PushPull.lean`: an adjacency pattern `A : Pat` (the
structure of a GraphBLAS matrix; the BFS semiring `ANY_PAIR` never reads
values), walks of exactly `d` steps (`R`), the shortest-path distance
(`IsDist`), and the facts a level-synchronous BFS needs. `count n f` is
`GrB_Vector_nvals` of a size-`n` vector whose pattern is `f`.
-/
namespace Pr3435.Bfs

abbrev BVec := Nat → Bool
abbrev LVec := Nat → Option Int
abbrev Pat := Nat → Nat → Bool

/-! ## `count` -/

def count (n : Nat) (f : Nat → Bool) : Nat := ((List.range n).filter f).length

theorem count_zero (f : Nat → Bool) : count 0 f = 0 := rfl

theorem count_succ (n : Nat) (f : Nat → Bool) :
    count (n + 1) f = count n f + (if f n then 1 else 0) := by
  unfold count
  rw [List.range_succ, List.filter_append, List.length_append]
  cases h : f n <;> simp [h]

theorem count_le (n : Nat) (f : Nat → Bool) : count n f ≤ n := by
  unfold count
  exact Nat.le_trans (List.length_filter_le _ _) (by simp)

theorem count_eq_zero_iff (n : Nat) (f : Nat → Bool) :
    count n f = 0 ↔ ∀ j, j < n → f j = false := by
  induction n with
  | zero => simp [count_zero]
  | succ n ih =>
    rw [count_succ]
    constructor
    · intro h j hj
      cases hf : f n
      · simp [hf] at h
        rcases Nat.lt_succ_iff_lt_or_eq.1 hj with hj | hj
        · exact ih.1 h j hj
        · subst hj; exact hf
      · simp [hf] at h
    · intro h
      have h1 := ih.2 (fun j hj => h j (Nat.lt_succ_of_lt hj))
      have h2 := h n (Nat.lt_succ_self n)
      simp [h1, h2]

/-- Disjoint union: `nvals` of the union is the sum. -/
theorem count_or (n : Nat) (f g : Nat → Bool) (hd : ∀ j, g j = true → f j = false) :
    count n (fun j => f j || g j) = count n f + count n g := by
  induction n with
  | zero => rfl
  | succ n ih =>
    rw [count_succ, count_succ, count_succ, ih]
    cases hf : f n <;> cases hg : g n <;> simp_all <;> omega

theorem count_single (n src : Nat) (h : src < n) :
    count n (fun j => decide (j = src)) = 1 := by
  induction n with
  | zero => omega
  | succ n ih =>
    rw [count_succ]
    rcases Nat.lt_succ_iff_lt_or_eq.1 h with h' | h'
    · rw [ih h']; have : n ≠ src := by omega
      simp [this]
    · subst h'
      have : count src (fun j => decide (j = src)) = 0 :=
        (count_eq_zero_iff _ _).2 (fun j hj => by simp; omega)
      simp [this]

theorem count_pos_of {n : Nat} {f : Nat → Bool} (h : count n f ≠ 0) : 0 < count n f := by omega

/-! ## Walks and distances -/

section
variable (A : Pat) (src : Nat)

/-- A walk of exactly `d` edges from `src` to `j`. -/
def R : Nat → Nat → Prop
  | 0, j => j = src
  | d + 1, j => ∃ i, R d i ∧ A i j = true

/-- `j` is at shortest-path distance exactly `d` from `src`. -/
def IsDist (j d : Nat) : Prop := R A src d j ∧ ∀ d', d' < d → ¬ R A src d' j

/-- Every node with a walk of length `d` has a distance `≤ d`. -/
theorem exists_dist : ∀ (B d j : Nat), d ≤ B → R A src d j → ∃ d', d' ≤ d ∧ IsDist A src j d' := by
  intro B
  induction B with
  | zero =>
    intro d j hd hr
    have : d = 0 := by omega
    subst this
    exact ⟨0, Nat.le_refl _, hr, fun d' h => absurd h (by omega)⟩
  | succ B ih =>
    intro d j hd hr
    by_cases hm : ∀ d', d' < d → ¬ R A src d' j
    · exact ⟨d, Nat.le_refl _, hr, hm⟩
    · have : ∃ d', d' < d ∧ R A src d' j := by
        apply Classical.byContradiction; intro hne
        exact hm (fun d' h1 h2 => hne ⟨d', h1, h2⟩)
      obtain ⟨d', h1, h2⟩ := this
      obtain ⟨d'', h3, h4⟩ := ih d' j (by omega) h2
      exact ⟨d'', by omega, h4⟩

theorem dist_unique {j d d' : Nat} (h : IsDist A src j d) (h' : IsDist A src j d') : d = d' := by
  rcases Nat.lt_trichotomy d d' with hl | he | hg
  · exact absurd h.1 (h'.2 d hl)
  · exact he
  · exact absurd h'.1 (h.2 d' hg)

theorem dist_zero_iff (j : Nat) : IsDist A src j 0 ↔ j = src := by
  constructor
  · intro h; exact h.1
  · intro h; exact ⟨h, fun d' hd => absurd hd (by omega)⟩

/-- The predecessor on a shortest walk is at distance one less. -/
theorem dist_pred {j K : Nat} (h : IsDist A src j (K + 1)) :
    ∃ i, IsDist A src i K ∧ A i j = true := by
  obtain ⟨⟨i, hi, he⟩, hmin⟩ := h
  refine ⟨i, ⟨hi, fun d' hd hr => hmin (d' + 1) (by omega) ⟨i, hr, he⟩⟩, he⟩

/-- **The BFS level step.** For `K ≥ 1`: `j` is at distance `K` iff it is at no
smaller distance and has an in-edge from a node at distance `K - 1`. -/
theorem dist_step (j K : Nat) (hK : 1 ≤ K) :
    IsDist A src j K ↔ (∀ d, d < K → ¬ IsDist A src j d) ∧ ∃ i, IsDist A src i (K - 1) ∧ A i j = true := by
  obtain ⟨K', rfl⟩ : ∃ K', K = K' + 1 := ⟨K - 1, by omega⟩
  constructor
  · intro h
    refine ⟨fun d hd hd' => absurd (dist_unique A src h hd') (by omega), ?_⟩
    simpa using dist_pred A src h
  · rintro ⟨hno, i, hi, he⟩
    refine ⟨⟨i, by simpa using hi.1, he⟩, fun d hd hr => ?_⟩
    obtain ⟨d'', h1, h2⟩ := exists_dist A src d d j (Nat.le_refl _) hr
    exact hno d'' (by omega) h2

/-- **No gaps.** If some node is at distance `d ≥ K`, some node is at distance `K`. -/
theorem dist_gap : ∀ (d j K : Nat), K ≤ d → IsDist A src j d → ∃ i, IsDist A src i K := by
  intro d
  induction d with
  | zero => intro j K hK h; exact ⟨j, by have : K = 0 := by omega
                                         subst this; exact h⟩
  | succ d ih =>
    intro j K hK h
    rcases Nat.lt_or_ge d K with hl | hge
    · have : K = d + 1 := by omega
      subst this; exact ⟨j, h⟩
    · obtain ⟨i, hi, _⟩ := dist_pred A src h
      exact ih i K hge hi

/-- Adjacency entries are in bounds (an `n × n` matrix). -/
def Bounded (n : Nat) : Prop := ∀ i j, A i j = true → i < n ∧ j < n

theorem dist_lt {n j d : Nat} (hb : Bounded A n) (hs : src < n) (h : IsDist A src j d) : j < n := by
  cases d with
  | zero => rw [(dist_zero_iff A src j).1 h]; exact hs
  | succ d => obtain ⟨i, _, he⟩ := dist_pred A src h; exact (hb i j he).2

end

end Pr3435.Bfs
