/-
# Walks, distances and folds shared by the path searches of `algo_procedures.rs` and `eval.rs`

`Step ε` is the adjacency every search reads: `step u` lists `(far, payload)` for each edge leaving
`u` in the order the neighbour iterator yields them (for `algo_procedures.rs` it is
`node_successors`, see `AlgoSearch.stepOf`; `ε` is the edge triple or the edge id). A walk is a list
of `(far, payload)` taken from `step`. `Dst s x k` says `k` is the length of a shortest walk `s → x`.
-/
namespace OpsTraverse.Search

variable {ε : Type}

abbrev Step (ε : Type) := Nat → List (Nat × ε)

/-- `es` is a walk from `u` to `w`: each element is one of the current node's successors. -/
def WalkTo (step : Step ε) : Nat → List (Nat × ε) → Nat → Prop
  | u, [], w => u = w
  | u, x :: r, w => x ∈ step u ∧ WalkTo step x.1 r w

/-- The nodes a walk visits, source first. -/
def nodesOf (u : Nat) (es : List (Nat × ε)) : List Nat := u :: es.map (·.1)

theorem walk_append (step : Step ε) : ∀ (u : Nat) (a b : List (Nat × ε)) (w : Nat),
    WalkTo step u (a ++ b) w ↔ ∃ m, WalkTo step u a m ∧ WalkTo step m b w
  | u, [], b, w => by simp [WalkTo]
  | u, x :: a, b, w => by
    simp only [List.cons_append, WalkTo]
    constructor
    · rintro ⟨h1, h2⟩
      obtain ⟨m, hm1, hm2⟩ := (walk_append step x.1 a b w).mp h2
      exact ⟨m, ⟨h1, hm1⟩, hm2⟩
    · rintro ⟨m, ⟨h1, hm1⟩, hm2⟩
      exact ⟨h1, (walk_append step x.1 a b w).mpr ⟨m, hm1, hm2⟩⟩

/-- A walk of exactly `k` steps from `s` to `x`. -/
def R (step : Step ε) (s k x : Nat) : Prop := ∃ es, es.length = k ∧ WalkTo step s es x

/-- `k` is the length of a shortest walk from `s` to `x`. -/
def Dst (step : Step ε) (s x k : Nat) : Prop := R step s k x ∧ ∀ j, j < k → ¬ R step s j x

theorem R_zero (step : Step ε) (s x : Nat) : R step s 0 x ↔ x = s := by
  constructor
  · rintro ⟨es, hl, hw⟩
    cases es with
    | nil => exact hw.symm
    | cons _ _ => simp at hl
  · rintro rfl; exact ⟨[], rfl, rfl⟩

theorem R_succ (step : Step ε) (s k x : Nat) :
    R step s (k + 1) x ↔ ∃ u e, R step s k u ∧ (x, e) ∈ step u := by
  constructor
  · rintro ⟨es, hl, hw⟩
    have : es ≠ [] := by rintro rfl; simp at hl
    obtain ⟨a, y, rfl⟩ := List.eq_nil_or_concat es |>.resolve_left this
    rw [List.concat_eq_append] at hw hl
    obtain ⟨m, hm1, hm2⟩ := (walk_append step s a [y] x).mp hw
    simp only [WalkTo] at hm2
    obtain ⟨hy, rfl⟩ := hm2
    refine ⟨m, y.2, ⟨a, by simpa using hl, hm1⟩, ?_⟩
    simpa using hy
  · rintro ⟨u, e, ⟨es, hl, hw⟩, he⟩
    refine ⟨es ++ [(x, e)], by simp [hl], (walk_append step s es _ x).mpr ⟨u, hw, ?_⟩⟩
    simpa [WalkTo] using he

theorem exists_dist (step : Step ε) (s : Nat) : ∀ k x, R step s k x → ∃ j ≤ k, Dst step s x j := by
  intro k
  induction k using Nat.strongRecOn with
  | _ k ih =>
    intro x hr
    by_cases h : ∃ j, j < k ∧ R step s j x
    · obtain ⟨j, hj, hrj⟩ := h
      obtain ⟨d, hd, hdd⟩ := ih j hj x hrj
      exact ⟨d, by omega, hdd⟩
    · exact ⟨k, Nat.le_refl _, hr, fun j hj hr' => h ⟨j, hj, hr'⟩⟩

theorem dist_unique (step : Step ε) (s x : Nat) {j k : Nat} (h1 : Dst step s x j) (h2 : Dst step s x k) :
    j = k := by
  rcases Nat.lt_trichotomy j k with h | h | h
  · exact absurd h1.1 (h2.2 j h)
  · exact h
  · exact absurd h2.1 (h1.2 k h)

theorem dist_zero (step : Step ε) (s x : Nat) : Dst step s x 0 ↔ x = s := by
  constructor
  · intro h; exact (R_zero step s x).mp h.1
  · intro h; rw [h]; exact ⟨(R_zero step s _).mpr rfl, fun j hj => by omega⟩

theorem dist_pred (step : Step ε) (s x : Nat) (k : Nat) (h : Dst step s x (k + 1)) :
    ∃ u e, (x, e) ∈ step u ∧ Dst step s u k := by
  obtain ⟨u, e, hu, he⟩ := (R_succ step s k x).mp h.1
  obtain ⟨d, hd, hdd⟩ := exists_dist step s k u hu
  have : d = k := by
    refine Classical.byContradiction fun hne => ?_
    have hlt : d < k := by omega
    exact h.2 (d + 1) (by omega) ((R_succ step s d x).mpr ⟨u, e, hdd.1, he⟩)
  subst this
  exact ⟨u, e, he, hdd⟩

theorem dist_succ (step : Step ε) (s u x : Nat) (k : Nat) (e : ε) (hu : Dst step s u k)
    (he : (x, e) ∈ step u) (hn : ∀ j, j ≤ k → ¬ R step s j x) : Dst step s x (k + 1) :=
  ⟨(R_succ step s k x).mpr ⟨u, e, hu.1, he⟩, fun j hj hr => hn j (by omega) hr⟩

theorem no_level (step : Step ε) (s h : Nat) (hn : ∀ u, ¬ Dst step s u h) :
    ∀ k, h ≤ k → ∀ x, ¬ Dst step s x k := by
  intro k hk
  obtain ⟨n, rfl⟩ : ∃ n, k = h + n := ⟨k - h, by omega⟩
  induction n with
  | zero => exact hn
  | succ n ih =>
    intro x hx
    obtain ⟨u, e, _, hu⟩ := dist_pred step s x (h + n) hx
    exact ih (by omega) u hu

theorem R_ne_source (step : Step ε) (s x k : Nat) (hd : Dst step s x k) (hk : 1 ≤ k) : x ≠ s := by
  intro h; rw [h] at hd
  exact hd.2 0 (by omega) ((R_zero step s _).mpr rfl)

/-- Left fold `0 + g e₁ + g e₂ + …`, the order every Rust accumulator uses. -/
def fsum {F : Type} (add : F → F → F) (zero : F) (g : ε → F) (es : List (Nat × ε)) : F :=
  es.foldl (fun acc x => add acc (g x.2)) zero

theorem fsum_append {F : Type} (add : F → F → F) (zero : F) (g : ε → F) (a b : List (Nat × ε)) :
    fsum add zero g (a ++ b) = b.foldl (fun acc x => add acc (g x.2)) (fsum add zero g a) := by
  simp [fsum, List.foldl_append]

/-- A shortest walk visits distinct nodes. -/
theorem shortest_nodup (step : Step ε) (s : Nat) :
    ∀ (es : List (Nat × ε)) (x : Nat), WalkTo step s es x → Dst step s x es.length →
      (nodesOf s es).Nodup ∧ ∀ v ∈ nodesOf s es, ∃ i, i ≤ es.length ∧ Dst step s v i ∧
        (v = x → i = es.length) := by
  suffices H : ∀ n (es : List (Nat × ε)) (x : Nat), es.length = n → WalkTo step s es x →
      Dst step s x es.length → (nodesOf s es).Nodup ∧ ∀ v ∈ nodesOf s es, ∃ i, i ≤ es.length ∧
        Dst step s v i ∧ (v = x → i = es.length) from fun es x => H es.length es x rfl
  intro n
  induction n with
  | zero =>
    intro es x hl hw hd
    have : es = [] := List.length_eq_zero_iff.mp hl
    subst this
    simp only [WalkTo] at hw
    have hx : x = s := hw.symm
    subst hx
    refine ⟨by simp [nodesOf], fun v hv => ?_⟩
    simp [nodesOf] at hv
    rw [hv]
    exact ⟨0, Nat.le_refl _, (dist_zero step _ _).mpr rfl, fun _ => rfl⟩
  | succ n ih =>
    intro es x hl hw hd
    have hne : es ≠ [] := by rintro rfl; simp at hl
    obtain ⟨a, y, rfl⟩ := List.eq_nil_or_concat es |>.resolve_left hne
    rw [List.concat_eq_append] at hw hd hl ⊢
    have hal : a.length = n := by simpa using hl
    obtain ⟨m, hm1, hm2⟩ := (walk_append step s a [y] x).mp hw
    simp only [WalkTo] at hm2
    obtain ⟨hy, rfl⟩ := hm2
    simp only [List.length_append, List.length_singleton] at hd
    have hdm : Dst step s m a.length := by
      obtain ⟨d, hd', hdd⟩ := exists_dist step s a.length m ⟨a, rfl, hm1⟩
      have : d = a.length := by
        refine Classical.byContradiction fun hne => ?_
        exact hd.2 (d + 1) (by omega) ((R_succ step s d _).mpr ⟨m, y.2, hdd.1, by simpa using hy⟩)
      subst this; exact hdd
    obtain ⟨hnd, hv⟩ := ih a m hal hm1 hdm
    have hnodes : nodesOf s (a ++ [y]) = nodesOf s a ++ [y.1] := by simp [nodesOf]
    have hnot : y.1 ∉ nodesOf s a := by
      intro hmem
      obtain ⟨i, hi, hdi, _⟩ := hv _ hmem
      exact hd.2 i (by omega) hdi.1
    refine ⟨?_, fun v hvm => ?_⟩
    · rw [hnodes, List.nodup_append]
      refine ⟨hnd, by simp, ?_⟩
      intro a1 ha1 b hb
      simp at hb; subst hb
      intro heq; subst heq; exact hnot ha1
    · rw [hnodes] at hvm
      rcases List.mem_append.mp hvm with h | h
      · obtain ⟨i, hi, hdi, _⟩ := hv v h
        refine ⟨i, by simp; omega, hdi, fun hvx => ?_⟩
        subst hvx
        exact absurd hdi.1 (fun hr => hd.2 i (by omega) hr)
      · simp at h; subst h
        exact ⟨a.length + 1, by simp, hd, fun _ => by simp⟩

end OpsTraverse.Search
