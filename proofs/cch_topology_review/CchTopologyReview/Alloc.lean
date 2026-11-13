import CchTopologyReview.Inv

/-!
# Rank allocation keeps I1′ and P (03 §6 "Rank allocation")

`allocOne_inv`: one allocation keeps `Inv`. `alloc_inv`, `alloc_ranks_all`: the phase keeps `Inv` and
leaves every endpoint with a rank, which is the premise 03 §6b's Lemma A and §6's "only then classify"
rely on. `allocOne_keeps_holder` is 03's "taking a pool rank never strips a holder"; with
`allocOne_free_sub` it is "a rank never goes back to the pool".
-/

namespace CchTopo

variable {α : Type}

theorem allocTop_arcs (t : Cch α) (x : Nat) : (allocTop t x).arcs = t.arcs := rfl

theorem allocOne_arcs (t : Cch α) (x d : Nat) : (allocOne t x d).arcs = t.arcs := by
  unfold allocOne
  split
  · rfl
  · split
    · split
      · rfl
      · rfl
    · rfl

/-- The pool case of `allocOne`, as a rank-side record. -/
theorem rinv_pool {k : Ranks} (h : RInv k) (x r : Nat) (rest : List Nat) (hx : k.rankOf x = none)
    (hf : k.free = r :: rest) :
    RInv ⟨k.nr, upd k.nodeOf r (some x), upd k.rankOf x (some r), rest⟩ := by
  have hr : r < k.nr ∧ k.nodeOf r = none := (h.free_iff r).1 (by rw [hf]; exact List.mem_cons_self ..)
  have hasc : (r :: rest).Pairwise (· < ·) := hf ▸ h.free_asc
  rw [List.pairwise_cons] at hasc
  refine ⟨?_, ?_, ?_, hasc.2⟩
  · intro z y hz
    simp only [upd] at hz
    split at hz
    · rename_i e; exact e ▸ hr.1
    · exact h.node_lt z y hz
  · intro y s
    simp only [upd]
    by_cases hy : y = x <;> by_cases hs : s = r
    · rw [ite_eq_left hy, ite_eq_left hs]; subst hy hs; simp
    · rw [ite_eq_left hy, ite_eq_right hs]
      constructor
      · intro e; exact absurd (Option.some.inj e).symm hs
      · intro e; subst hy; have := (h.rank_node y s).2 e; rw [hx] at this; cases this
    · rw [ite_eq_right hy, ite_eq_left hs]
      constructor
      · intro e; subst hs; have := (h.rank_node y s).1 e; rw [hr.2] at this; cases this
      · intro e; exact absurd (Option.some.inj e).symm hy
    · rw [ite_eq_right hy, ite_eq_right hs]
      exact h.rank_node y s
  · intro z
    simp only [upd]
    constructor
    · intro hz
      have hzr : r < z := hasc.1 z hz
      have := (h.free_iff z).1 (by rw [hf]; exact List.mem_cons_of_mem _ hz)
      rw [ite_eq_right (by omega)]
      exact this
    · rintro ⟨hz, hn⟩
      split at hn
      · cases hn
      · rename_i hzr
        have := (h.free_iff z).2 ⟨hz, hn⟩
        rw [hf] at this
        rcases List.mem_cons.1 this with e | e
        · exact absurd e hzr
        · exact e

/-- The top-rank case of `allocOne`. -/
theorem rinv_top {k : Ranks} (h : RInv k) (x : Nat) (hx : k.rankOf x = none) :
    RInv ⟨k.nr + 1, upd k.nodeOf k.nr (some x), upd k.rankOf x (some k.nr), k.free⟩ := by
  have hnr : k.nodeOf k.nr = none := by
    cases e : k.nodeOf k.nr with
    | none => rfl
    | some y => have := h.node_lt _ _ e; omega
  refine ⟨?_, ?_, ?_, h.free_asc⟩
  · intro z y hz
    simp only [upd] at hz ⊢
    split at hz
    · omega
    · have := h.node_lt z y hz; omega
  · intro y s
    simp only [upd]
    by_cases hy : y = x <;> by_cases hs : s = k.nr
    · rw [ite_eq_left hy, ite_eq_left hs]; subst hy hs; simp
    · rw [ite_eq_left hy, ite_eq_right hs]
      constructor
      · intro e; exact absurd (Option.some.inj e).symm hs
      · intro e; subst hy; have := (h.rank_node y s).2 e; rw [hx] at this; cases this
    · rw [ite_eq_right hy, ite_eq_left hs]
      constructor
      · intro e; subst hs; have := (h.rank_node y k.nr).1 e; rw [hnr] at this; cases this
      · intro e; exact absurd (Option.some.inj e).symm hy
    · rw [ite_eq_right hy, ite_eq_right hs]
      exact h.rank_node y s
  · intro z
    simp only [upd]
    constructor
    · intro hz
      have := (h.free_iff z).1 hz
      rw [ite_eq_right (by omega)]
      exact ⟨by omega, this.2⟩
    · rintro ⟨hz, hn⟩
      split at hn
      · cases hn
      · rename_i hzr
        exact (h.free_iff z).2 ⟨by omega, hn⟩

theorem upd_isSome (f : Nat → Option Nat) (a z : Nat) (b : Nat) (h : (f z).isSome) : (upd f a (some b) z).isSome := by
  simp only [upd]; split <;> simp_all

theorem allocOne_ranks_cases (t : Cch α) (x d : Nat) :
    (allocOne t x d).ranks = t.ranks ∨
    (t.rankOf x = none ∧ ∃ r rest, t.free = r :: rest ∧
      (allocOne t x d).ranks = ⟨t.nr, upd t.nodeOf r (some x), upd t.rankOf x (some r), rest⟩) ∨
    (t.rankOf x = none ∧ (allocOne t x d).ranks = ⟨t.nr + 1, upd t.nodeOf t.nr (some x), upd t.rankOf x (some t.nr), t.free⟩) := by
  unfold allocOne
  split
  · exact Or.inl rfl
  · rename_i hx
    split
    · rename_i r rest hf
      split
      · exact Or.inr (Or.inl ⟨hx, r, rest, hf, rfl⟩)
      · exact Or.inr (Or.inr ⟨hx, rfl⟩)
    · exact Or.inr (Or.inr ⟨hx, rfl⟩)

/-- **One allocation keeps the invariant.** -/
theorem allocOne_inv (t : Cch α) (x d : Nat) (h : Inv t) : Inv (allocOne t x d) := by
  obtain ⟨hr, ha, hh⟩ := h
  rw [Inv, allocOne_arcs]
  rcases allocOne_ranks_cases t x d with e | ⟨hx, r, rest, hf, e⟩ | ⟨hx, e⟩ <;> rw [e]
  · exact ⟨hr, ha, hh⟩
  · exact ⟨rinv_pool hr x r rest hx hf, ha, fun a hu => ⟨upd_isSome _ _ _ _ (hh a hu).1, upd_isSome _ _ _ _ (hh a hu).2⟩⟩
  · exact ⟨rinv_top hr x hx, ha, fun a hu => ⟨upd_isSome _ _ _ _ (hh a hu).1, upd_isSome _ _ _ _ (hh a hu).2⟩⟩

/-- **Taking a rank never strips a holder** (03 §6: "Under P, a pool rank has no holder, so step 3 never
    strips a node"). -/
theorem allocOne_keeps_holder (t : Cch α) (x d : Nat) (h : Inv t) (r y : Nat) (hy : t.nodeOf r = some y) :
    (allocOne t x d).nodeOf r = some y := by
  obtain ⟨hr, _, _⟩ := h
  have key : (allocOne t x d).ranks.nodeOf r = some y := by
    rcases allocOne_ranks_cases t x d with e | ⟨hx, s, rest, hf, e⟩ | ⟨hx, e⟩ <;> rw [e]
    · exact hy
    · have hs : t.nodeOf s = none :=
        ((hr.free_iff s).1 (by show s ∈ t.free; rw [hf]; exact List.mem_cons_self ..)).2
      show upd t.nodeOf s (some x) r = some y
      simp only [upd]; rw [ite_eq_right (by intro e; subst e; rw [hs] at hy; cases hy)]; exact hy
    · have hlt : r < t.nr := hr.node_lt r y hy
      show upd t.nodeOf t.nr (some x) r = some y
      simp only [upd]; rw [ite_eq_right (by omega)]; exact hy
  exact key

/-- A node that has a rank keeps it. -/
theorem allocOne_keeps_rank (t : Cch α) (x d : Nat) (h : Inv t) (y r : Nat) (hy : t.rankOf y = some r) :
    (allocOne t x d).rankOf y = some r := by
  have h' := allocOne_inv t x d h
  have hn := allocOne_keeps_holder t x d h r y ((h.1.rank_node y r).1 hy)
  exact (h'.1.rank_node y r).2 hn

/-- The pool only shrinks. -/
theorem allocOne_free_sub (t : Cch α) (x d : Nat) (r : Nat) (hr : r ∈ (allocOne t x d).free) : r ∈ t.free := by
  have key : r ∈ (allocOne t x d).ranks.free := hr
  rcases allocOne_ranks_cases t x d with e | ⟨hx, s, rest, hf, e⟩ | ⟨hx, e⟩ <;> rw [e] at key
  · exact key
  · rw [hf]; exact List.mem_cons_of_mem _ key
  · exact key

/-- The rank space only grows. -/
theorem allocOne_nr_mono (t : Cch α) (x d : Nat) : t.nr ≤ (allocOne t x d).nr := by
  have key : t.ranks.nr ≤ (allocOne t x d).ranks.nr := by
    rcases allocOne_ranks_cases t x d with e | ⟨hx, s, rest, hf, e⟩ | ⟨hx, e⟩ <;> rw [e] <;> simp [Cch.ranks]
  exact key

/-- After its allocation, `x` has a rank. -/
theorem allocOne_ranked (t : Cch α) (x d : Nat) : ((allocOne t x d).rankOf x).isSome := by
  cases hx : t.rankOf x with
  | some r => simp [allocOne, hx]
  | none =>
    cases hf : t.free with
    | nil => simp [allocOne, hx, hf, allocTop, upd]
    | cons r rest =>
      by_cases hd : d ≤ 1
      · simp [allocOne, hx, hf, hd, upd]
      · simp [allocOne, hx, hf, hd, allocTop, upd]

/-- **The rule as 03 §6 step 3 states it**: a rank-less node of degree ≤ 1 takes the lowest free rank
    when one is free; otherwise it takes the next top rank, index `nr` (03's `n_r' + 1`). -/
theorem allocOne_rule (t : Cch α) (x d : Nat) (hx : t.rankOf x = none) :
    (d ≤ 1 → ∀ r rest, t.free = r :: rest → (allocOne t x d).rankOf x = some r ∧ (allocOne t x d).free = rest ∧
        (allocOne t x d).nr = t.nr) ∧
    ((1 < d ∨ t.free = []) → (allocOne t x d).rankOf x = some t.nr ∧ (allocOne t x d).nr = t.nr + 1 ∧
        (allocOne t x d).free = t.free) := by
  constructor
  · intro hd r rest hf
    simp [allocOne, hx, hf, hd, upd]
  · intro hc
    rcases hc with hd | hf
    · cases hf : t.free with
      | nil => simp [allocOne, hx, hf, allocTop, upd]
      | cons r rest => simp [allocOne, hx, hf, allocTop, upd, show ¬ d ≤ 1 by omega]
    · simp [allocOne, hx, hf, allocTop, upd]

/-! ## The phase -/

theorem alloc_cons (t : Cch α) (p : Nat × Nat) (E : List (Nat × Nat)) :
    alloc t (p :: E) = alloc (allocOne t p.1 p.2) E := rfl

theorem alloc_arcs (t : Cch α) : ∀ (E : List (Nat × Nat)), (alloc t E).arcs = t.arcs
  | [] => rfl
  | p :: E => by rw [alloc_cons, alloc_arcs _ E, allocOne_arcs]

theorem alloc_inv : ∀ (E : List (Nat × Nat)) (t : Cch α), Inv t → Inv (alloc t E)
  | [], _, h => h
  | p :: E, t, h => by rw [alloc_cons]; exact alloc_inv E _ (allocOne_inv t p.1 p.2 h)

theorem alloc_keeps_rank : ∀ (E : List (Nat × Nat)) (t : Cch α), Inv t → ∀ y r, t.rankOf y = some r →
    (alloc t E).rankOf y = some r
  | [], _, _, _, _, hy => hy
  | p :: E, t, h, y, r, hy => by
    rw [alloc_cons]
    exact alloc_keeps_rank E _ (allocOne_inv t p.1 p.2 h) y r (allocOne_keeps_rank t p.1 p.2 h y r hy)

theorem alloc_keeps_holder : ∀ (E : List (Nat × Nat)) (t : Cch α), Inv t → ∀ r y, t.nodeOf r = some y →
    (alloc t E).nodeOf r = some y
  | [], _, _, _, _, hy => hy
  | p :: E, t, h, r, y, hy => by
    rw [alloc_cons]
    exact alloc_keeps_holder E _ (allocOne_inv t p.1 p.2 h) r y (allocOne_keeps_holder t p.1 p.2 h r y hy)

theorem alloc_free_sub : ∀ (E : List (Nat × Nat)) (t : Cch α) (r : Nat), r ∈ (alloc t E).free → r ∈ t.free
  | [], _, _, h => h
  | p :: E, t, r, h => by
    rw [alloc_cons] at h
    exact allocOne_free_sub t p.1 p.2 r (alloc_free_sub E _ r h)

theorem alloc_nr_mono : ∀ (E : List (Nat × Nat)) (t : Cch α), t.nr ≤ (alloc t E).nr
  | [], _ => Nat.le_refl _
  | p :: E, t => by
    rw [alloc_cons]; exact Nat.le_trans (allocOne_nr_mono t p.1 p.2) (alloc_nr_mono E _)

/-- **After the allocation phase every endpoint has a rank** (03 §6 step 4's premise, and §6b's "every
    edge has an arc before the recompute" on the rank side). -/
theorem alloc_ranks_all (E : List (Nat × Nat)) (t : Cch α) (h : Inv t) :
    ∀ p ∈ E, ((alloc t E).rankOf p.1).isSome := by
  induction E generalizing t with
  | nil => intro p hp; simp at hp
  | cons q E ih =>
    intro p hp
    rw [alloc_cons]
    rcases List.mem_cons.1 hp with e | e
    · subst e
      have hs := allocOne_ranked t p.1 p.2
      obtain ⟨r, hr⟩ := Option.isSome_iff_exists.1 hs
      rw [alloc_keeps_rank E _ (allocOne_inv t p.1 p.2 h) p.1 r hr]; rfl
    · exact ih _ (allocOne_inv t q.1 q.2 h) p e

end CchTopo
