import FalkorIdSpace.Model
/-! Lemmas about the roaring-set model: `min`/`max`/`rank`/`len`. -/
namespace IdSpaceModel

theorem asc_sorted (n : Nat) (s : IdSet) : (asc n s).Pairwise (· < ·) :=
  (List.pairwise_lt_range).sublist List.filter_sublist

theorem sMin_some {N : Nat} {s : IdSet} {l : Nat} :
    sMin N s = some l ↔ (l < N ∧ s l = true ∧ ∀ i, i < l → s i = false) := by
  unfold sMin
  constructor
  · intro h
    obtain ⟨t, ht⟩ : ∃ t, asc N s = l :: t := by
      cases e : asc N s with
      | nil => simp [e] at h
      | cons a t => simp [e] at h; exact ⟨t, by rw [h]⟩
    have hm : l ∈ asc N s := by rw [ht]; simp
    obtain ⟨hl, hs⟩ := mem_asc.mp hm
    refine ⟨hl, hs, fun i hi => ?_⟩
    cases hsi : s i with
    | false => rfl
    | true =>
      have : i ∈ asc N s := mem_asc.mpr ⟨by omega, hsi⟩
      rw [ht] at this
      have hp := asc_sorted N s; rw [ht] at hp
      rcases List.mem_cons.mp this with e | e
      · omega
      · have := (List.pairwise_cons.mp hp).1 i e; omega
  · rintro ⟨hl, hs, hmin⟩
    cases e : asc N s with
    | nil => have : l ∈ asc N s := mem_asc.mpr ⟨hl, hs⟩
             rw [e] at this; simp at this
    | cons a t =>
      have ha : a ∈ asc N s := by rw [e]; simp
      obtain ⟨_, hsa⟩ := mem_asc.mp ha
      have hlm : l ∈ asc N s := mem_asc.mpr ⟨hl, hs⟩
      rw [e] at hlm
      have hp := asc_sorted N s; rw [e] at hp
      simp only [List.head?_cons, Option.some.injEq]
      rcases List.mem_cons.mp hlm with e' | e'
      · exact e'.symm
      · have h1 := (List.pairwise_cons.mp hp).1 l e'
        have := hmin a h1; simp_all

theorem sMin_none {N : Nat} {s : IdSet} : sMin N s = none ↔ ∀ i, i < N → s i = false := by
  unfold sMin
  constructor
  · intro h i hi
    cases hsi : s i with
    | false => rfl
    | true =>
      have : i ∈ asc N s := mem_asc.mpr ⟨hi, hsi⟩
      cases e : asc N s with
      | nil => rw [e] at this; simp at this
      | cons a t => rw [e] at h; simp at h
  · intro h
    cases e : asc N s with
    | nil => rfl
    | cons a t =>
      have : a ∈ asc N s := by rw [e]; simp
      have := mem_asc.mp this; have := h a this.1; simp_all

theorem sMax_some {N : Nat} {s : IdSet} {h : Nat} :
    sMax N s = some h ↔ (h < N ∧ s h = true ∧ ∀ i, h < i → i < N → s i = false) := by
  unfold sMax
  rw [List.getLast?_eq_some_iff]
  constructor
  · rintro ⟨ys, hys⟩
    have hm : h ∈ asc N s := by rw [hys]; simp
    obtain ⟨hl, hs⟩ := mem_asc.mp hm
    refine ⟨hl, hs, fun i hi hiN => ?_⟩
    cases hsi : s i with
    | false => rfl
    | true =>
      have : i ∈ asc N s := mem_asc.mpr ⟨hiN, hsi⟩
      have hp := asc_sorted N s
      rw [hys] at this hp
      rw [List.pairwise_append] at hp
      rcases List.mem_append.mp this with e | e
      · have := hp.2.2 i e h (by simp); omega
      · simp at e; omega
  · rintro ⟨hl, hs, hmax⟩
    have hm : h ∈ asc N s := mem_asc.mpr ⟨hl, hs⟩
    obtain ⟨as, bs, hab⟩ := List.append_of_mem hm
    refine ⟨as, ?_⟩
    rw [hab]
    cases bs with
    | nil => rfl
    | cons b t =>
      have : b ∈ asc N s := by rw [hab]; simp
      obtain ⟨hbN, hsb⟩ := mem_asc.mp this
      have hp := asc_sorted N s
      rw [hab, List.pairwise_append] at hp
      have hhb : h < b := by
        have := hp.2.1; exact (List.pairwise_cons.mp this).1 b (by simp)
      have := hmax b hhb hbN; simp_all

theorem sMax_none {N : Nat} {s : IdSet} : sMax N s = none ↔ ∀ i, i < N → s i = false := by
  unfold sMax
  rw [List.getLast?_eq_none_iff]
  constructor
  · intro h i hi
    cases hsi : s i with
    | false => rfl
    | true => have : i ∈ asc N s := mem_asc.mpr ⟨hi, hsi⟩
              rw [h] at this; simp at this
  · intro h
    cases e : asc N s with
    | nil => rfl
    | cons a t =>
      have : a ∈ asc N s := by rw [e]; simp
      have := mem_asc.mp this; have := h a this.1; simp_all

theorem sMin_none_iff_sMax_none {N : Nat} {s : IdSet} : sMin N s = none ↔ sMax N s = none := by
  rw [sMin_none, sMax_none]

theorem sDisjoint_iff {N : Nat} {a b : IdSet} :
    sDisjoint N a b = true ↔ ∀ i, i < N → ¬ (a i = true ∧ b i = true) := by
  unfold sDisjoint
  rw [List.isEmpty_iff]
  constructor
  · intro h i hi ⟨ha, hb⟩
    have : i ∈ asc N (sInter a b) := mem_asc.mpr ⟨hi, by simp [sInter, ha, hb]⟩
    rw [h] at this; simp at this
  · intro h
    cases e : asc N (sInter a b) with
    | nil => rfl
    | cons x t =>
      have : x ∈ asc N (sInter a b) := by rw [e]; simp
      obtain ⟨hx, hs⟩ := mem_asc.mp this
      simp [sInter] at hs
      exact absurd hs (h x hx)

/-- `cnt n` counts the members below `n`, so for `b ≤ n` it splits at `b`. -/
theorem cnt_split_at (s : IdSet) {b n : Nat} (hb : b ≤ n) :
    cnt n s = cnt b s + cnt n (fun i => s i && decide (b ≤ i)) := by
  induction n with
  | zero => have : b = 0 := by omega
            subst this; simp
  | succ n ih =>
    rcases Nat.lt_or_ge n b with h | h
    · have : b = n + 1 := by omega
      subst this
      have e : cnt (n + 1) (fun i => s i && decide (n + 1 ≤ i)) = 0 :=
        cnt_eq_zero.mpr (fun i hi => by simp; omega)
      rw [e]; simp
    · rw [cnt_succ, cnt_succ, ih (by omega)]
      by_cases hs : s n <;> simp [hs, h] <;> omega

/-- `above` really is "how many sit at or above `bound`" (rank arithmetic). -/
theorem above_spec (N : Nat) (ids : IdSet) (b : Nat) (hb : b ≤ N) :
    above N ids b = cnt N (fun i => ids i && decide (b ≤ i)) := by
  unfold above sLen
  rw [cnt_split_at ids hb]
  by_cases h0 : b = 0
  · subst h0; simp
  · simp [h0]


/-- `rank`-style split of the ascending listing at `b ≤ n`. -/
theorem asc_split (s : IdSet) {b n : Nat} (hb : b ≤ n) :
    asc n s = asc b s ++ (List.range' b (n - b)).filter s := by
  unfold asc
  have : List.range n = List.range b ++ List.range' b (n - b) := by
    rw [List.range_eq_range', List.range_eq_range']
    have := (List.range'_append (s := 0) (m := b) (n := n - b) (step := 1))
    simp only [Nat.zero_add, Nat.one_mul] at this
    rw [this]; congr 1; omega
  rw [this, List.filter_append]

/-- `s` restricted to `[b, ∞)` (`remove_range(..b)` on a copy). -/
def sFrom (s : IdSet) (b : Nat) : IdSet := fun i => s i && decide (b ≤ i)

theorem asc_sFrom (s : IdSet) {b n : Nat} (hb : b ≤ n) :
    asc n (sFrom s b) = (List.range' b (n - b)).filter s := by
  rw [asc_split _ hb]
  have h1 : asc b (sFrom s b) = [] := by
    unfold asc; apply List.filter_eq_nil_iff.mpr
    intro x hx; simp at hx; simp [sFrom]; omega
  rw [h1, List.nil_append]
  apply List.filter_congr
  intro x hx; simp [List.mem_range'] at hx; simp [sFrom]; omega

/-- `taken.select(taken.len() - above)` is the lowest member at or above `b`. -/
theorem select_above (N : Nat) (s : IdSet) {b : Nat} (hb : b ≤ N) :
    sSelect N s (sLen N s - above N s b) = sMin N (sFrom s b) := by
  have hc : sLen N s - above N s b = cnt b s := by
    unfold above sLen
    have := cnt_mono_n s hb
    by_cases h0 : b = 0
    · subst h0; simp
    · simp [h0]; omega
  rw [hc]; unfold sSelect sMin
  rw [asc_split s hb, asc_sFrom s hb, ← asc_length b s]
  rw [List.getElem?_append_right (Nat.le_refl _), Nat.sub_self, List.head?_eq_getElem?]

/-- When the part at or above `b` is non-empty, the max of the whole set is its max. -/
theorem sMax_sFrom (N : Nat) (s : IdSet) {b : Nat} (hb : b ≤ N) {lo : Nat}
    (hlo : sMin N (sFrom s b) = some lo) : sMax N s = sMax N (sFrom s b) := by
  unfold sMax
  rw [asc_split s hb, asc_sFrom s hb]
  rw [List.getLast?_append]
  have hne : ((List.range' b (N - b)).filter s) ≠ [] := by
    intro h; unfold sMin at hlo; rw [asc_sFrom s hb, h] at hlo; cases hlo
  cases e : ((List.range' b (N - b)).filter s).getLast? with
  | none => exact absurd (List.getLast?_eq_none_iff.mp e) hne
  | some x => simp

theorem above_sFrom (N : Nat) (s : IdSet) {b : Nat} (hb : b ≤ N) :
    above N s b = sLen N (sFrom s b) := above_spec N s b hb

end IdSpaceModel
