import AlgoUdf.BfsModel
/-! # `bfs_find_bound`: level lemma, soundness and completeness (see `AlgoUdf.BfsModel`) -/
namespace AlgoUdf.BfsBound
open AlgoUdf.Paths (Dir farEndpoint)
open AlgoUdf.Graph

theorem reach_le {g : G} {src tgt h : Nat} {fr : List Nat} {st : BS} (I : LInv g src tgt h fr st) :
    ∀ {y es x}, HWalk g y es x → ∀ d, st.seen y = true → st.dist y ≤ d → d + es.length ≤ h →
      st.seen x = true ∧ st.dist x ≤ d + es.length := by
  intro y es x hw
  induction hw with
  | nil u => intro d hs hd _; exact ⟨hs, by simpa using hd⟩
  | @cons u v w r rest ha _ ih =>
    intro d hs hd hle
    simp only [List.length_cons] at hle ⊢
    obtain ⟨hv, hdv⟩ := I.closed u hs (by omega) v r ha
    have := ih (d + 1) hv (by omega) (by omega)
    exact ⟨this.1, by omega⟩

theorem reach_all {g : G} {src tgt h : Nat} {st : BS} (I : LInv g src tgt h [] st) :
    ∀ {y es x}, HWalk g y es x → st.seen y = true → st.seen x = true := by
  intro y es x hw
  induction hw with
  | nil u => exact id
  | @cons u v w r rest ha _ ih =>
    intro hs
    have hlt : st.dist u < h := by
      have h1 := I.distLe u hs
      have h2 : st.dist u ≠ h := fun he => by have := I.front u hs he; cases this
      omega
    exact ih (I.closed u hs hlt v r ha).1

theorem levels_spec {g : G} {src tgt : Nat} :
    ∀ (k h : Nat) (fr : List Nat) (st : BS), LInv g src tgt h fr st →
      match levels g tgt h k fr st with
      | some st' => PInv g src st' ∧ st'.seen tgt = true ∧ st'.dist tgt ≤ h + k
      | none => ∀ es, HWalk g src es tgt → h + k < es.length := by
  intro k
  induction k with
  | zero =>
    intro h fr st I
    simp only [levels]
    intro es hw
    apply Nat.lt_of_not_le; intro hle
    have := (reach_le I hw 0 I.srcSeen (by simp [I.srcDist]) (by omega)).1
    rw [I.tgtUnseen] at this; cases this
  | succ k ih =>
    intro h fr st I
    by_cases hfr : fr = []
    · subst hfr
      simp only [levels, if_pos]
      intro es hw
      have := reach_all I hw I.srcSeen
      rw [I.tgtUnseen] at this; cases this
    · have I0 : SInv g src tgt h fr [] 0 [] { st with next := [] } := by
        refine ⟨⟨I.srcSeen, I.srcDist, I.par⟩, ?_, ?_, ?_, ?_, ?_, I.front, I.frontSeen, I.tgtUnseen⟩
        · intro x hx; dsimp only at hx ⊢; have := I.distLe x hx; omega
        · exact I.closed
        · intro y hy; cases hy
        · intro r hr; cases hr
        · intro x; dsimp only; simp only [List.not_mem_nil, iff_false, not_and]
          intro hx he; have := I.distLe x hx; omega
      have := scanFrontier_spec fr [] _ rfl I0
      cases hres : scanFrontier g tgt (h + 1) fr { st with next := [] } with
      | inl st' =>
        rw [hres] at this
        simp only [levels, if_neg hfr, hres]
        exact ⟨this.1, this.2.1, by omega⟩
      | inr st' =>
        rw [hres] at this
        simp only [levels, if_neg hfr, hres]
        have I1 : LInv g src tgt (h + 1) st'.next st' := by
          refine ⟨this.toPInv, this.distLe, ?_, ?_, ?_, this.tgtUnseen⟩
          · intro y hy hdy x r ha
            by_cases hlt : st'.dist y < h
            · exact this.closedOld y hy hlt x r ha
            · have hyfr : y ∈ fr := this.front y hy (by have := this.distLe y hy; omega)
              obtain ⟨h1, h2⟩ := this.closedDone y hyfr x r ha
              exact ⟨h1, by omega⟩
          · intro x hx hd; exact (this.nextSpec x).mp ⟨hx, hd⟩
          · intro x hx; exact (this.nextSpec x).mpr hx
        have := ih (h + 1) st'.next st' I1
        cases hl : levels g tgt (h + 1) k st'.next st' with
        | some s => rw [hl] at this; exact ⟨this.1, this.2.1, by omega⟩
        | none =>
          rw [hl] at this
          intro es hw; have := this es hw; omega

theorem linv_init (g : G) (src tgt : Nat) (hne : src ≠ tgt) : LInv g src tgt 0 [src] (init src) := by
  refine ⟨⟨by simp [init], rfl, ?_⟩, ?_, ?_, ?_, ?_, ?_⟩
  · intro x hx hxs; simp [init, upd_ne _ _ _ _ hxs] at hx
  · intro x _; simp [init]
  · intro y _ h; omega
  · intro x hx _
    by_cases h : x = src
    · subst h; simp
    · simp [init, upd_ne _ _ _ _ h] at hx
  · intro x hx; simp at hx; subst hx; simp [init]
  · simp [init, upd_ne _ _ _ _ (Ne.symm hne)]

/-- The parent walk from a seen node never panics and spells a simple path of
exactly `dist` hops. -/
theorem chain_ok {g : G} {src : Nat} {st : BS} (P : PInv g src st) :
    ∀ x acc, st.seen x = true →
      ∃ es ns, chain src st.parents (st.dist x + 1) x acc = some (es ++ acc) ∧
        NWalk g src es ns x ∧ es.length = st.dist x ∧ (src :: ns).Nodup ∧
        ∀ n ∈ src :: ns, st.seen n = true ∧ st.dist n ≤ st.dist x := by
  suffices H : ∀ d x acc, st.dist x = d → st.seen x = true →
      ∃ es ns, chain src st.parents (d + 1) x acc = some (es ++ acc) ∧
        NWalk g src es ns x ∧ es.length = d ∧ (src :: ns).Nodup ∧
        ∀ n ∈ src :: ns, st.seen n = true ∧ st.dist n ≤ d by
    intro x acc hx; exact H _ x acc rfl hx
  intro d
  induction d with
  | zero =>
    intro x acc hd hx
    by_cases hxs : x = src
    · subst hxs
      exact ⟨[], [], by simp [chain], .nil _, rfl, by simp, by simp [P.srcSeen, P.srcDist]⟩
    · obtain ⟨p, r, _, _, _, h4⟩ := P.par x hx hxs; omega
  | succ d ih =>
    intro x acc hd hx
    by_cases hxs : x = src
    · subst hxs; rw [P.srcDist] at hd; omega
    · obtain ⟨p, r, h1, h2, h3, h4⟩ := P.par x hx hxs
      obtain ⟨es, ns, hc, hw, hl, hnd, hall⟩ := ih p (r :: acc) (by omega) h3
      refine ⟨es ++ [r], ns ++ [x], ?_, hw.snoc h2, by simp [hl], ?_, ?_⟩
      · rw [chain, if_neg hxs, h1]; dsimp only; rw [hc]; simp
      · rw [← List.cons_append, List.nodup_append]
        refine ⟨hnd, by simp, ?_⟩
        intro a ha b hb; simp at hb; subst hb
        intro he; subst he
        have := (hall a ha).2; omega
      · intro n hn
        rw [← List.cons_append] at hn
        simp only [List.mem_append, List.mem_singleton] at hn
        rcases hn with hn | rfl
        · have := hall n hn; exact ⟨this.1, by omega⟩
        · exact ⟨hx, by omega⟩

/-- **Soundness** of `bfs_find_bound`: never panics; `Some` is a simple path
source→target with 1..maxLen hops. -/
theorem bfs_sound (g : G) (src tgt maxLen : Nat) (hne : src ≠ tgt) :
    bfsFindBound g tgt src maxLen ≠ none ∧
    ∀ es, bfsFindBound g tgt src maxLen = some (some es) →
      ∃ ns, NWalk g src es ns tgt ∧ 1 ≤ es.length ∧ es.length ≤ maxLen ∧ (src :: ns).Nodup := by
  unfold bfsFindBound
  by_cases h0 : maxLen = 0
  · simp [h0]
  · simp only [h0, if_false]
    have := levels_spec maxLen 0 [src] (init src) (linv_init g src tgt hne)
    split
    · simp
    · rename_i st heq
      rw [heq] at this
      obtain ⟨P, hs, hd⟩ := this
      obtain ⟨es, ns, hc, hw, hl, hnd, _⟩ := chain_ok P tgt [] hs
      simp only [List.append_nil] at hc
      rw [hc]
      refine ⟨by simp, ?_⟩
      intro es' he; simp at he; subst he
      refine ⟨ns, hw, ?_, by omega, hnd⟩
      cases es with
      | nil => have := hw.hwalk.nil_eq; exact absurd this hne
      | cons _ _ => simp

/-- **Completeness** of `bfs_find_bound`: `None` means no source→target walk of
at most `maxLen` hops, so the enumeration would find nothing. -/
theorem bfs_complete (g : G) (src tgt maxLen : Nat) (hne : src ≠ tgt) :
    bfsFindBound g tgt src maxLen = some none →
      ∀ es, HWalk g src es tgt → maxLen < es.length := by
  unfold bfsFindBound
  intro h es hw
  by_cases h0 : maxLen = 0
  · subst h0
    cases es with
    | nil => exact absurd hw.nil_eq hne
    | cons _ _ => simp
  · simp only [h0, if_false] at h
    have := levels_spec maxLen 0 [src] (init src) (linv_init g src tgt hne)
    split at h
    · rename_i heq; rw [heq] at this; have := this es hw; omega
    · rename_i st heq
      rw [heq] at this
      obtain ⟨es', _, hc, _⟩ := chain_ok this.1 tgt [] this.2.1
      rw [hc] at h; simp at h

/-- `source = target`: the source is pre-seen, so the target is never found. -/
theorem scanRels_seen_tgt {g : G} {tgt u h : Nat} :
    ∀ (rs : List Rel) (st : BS), st.seen tgt = true →
      ∃ st', scanRels g tgt u h rs st = .inr st' ∧ st'.seen tgt = true := by
  intro rs
  induction rs with
  | nil => intro st hs; exact ⟨st, rfl, hs⟩
  | cons r rs ih =>
    intro st hs
    simp only [scanRels]
    split
    · exact ih st hs
    · rename_i far _
      split
      · exact ih st hs
      · rename_i hsf
        have hft : far ≠ tgt := by intro h; subst h; exact hsf hs
        rw [if_neg hft]
        apply ih
        have : tgt ≠ far := Ne.symm hft
        simp [mark, upd_ne _ _ _ _ this, hs]

theorem scanFrontier_seen_tgt {g : G} {tgt h : Nat} :
    ∀ (us : List Nat) (st : BS), st.seen tgt = true →
      ∃ st', scanFrontier g tgt h us st = .inr st' ∧ st'.seen tgt = true := by
  intro us
  induction us with
  | nil => intro st hs; exact ⟨st, rfl, hs⟩
  | cons u us ih =>
    intro st hs
    obtain ⟨st1, h1, h2⟩ := scanRels_seen_tgt (u := u) (h := h) (g.rels u) st hs
    rw [scanFrontier, h1]
    exact ih st1 h2

theorem levels_seen_tgt {g : G} {tgt : Nat} :
    ∀ (k h : Nat) (fr : List Nat) (st : BS), st.seen tgt = true → levels g tgt h k fr st = none := by
  intro k
  induction k with
  | zero => intro _ _ _ _; rfl
  | succ k ih =>
    intro h fr st hs
    simp only [levels]
    split
    · rfl
    · obtain ⟨st1, h1, h2⟩ := scanFrontier_seen_tgt (g := g) (tgt := tgt) (h := h + 1) fr
        { st with next := [] } hs
      rw [h1]; exact ih _ _ _ h2

theorem bfs_src_eq_tgt (g : G) (src maxLen : Nat) : bfsFindBound g src src maxLen = some none := by
  unfold bfsFindBound
  split
  · rfl
  · rw [levels_seen_tgt _ _ _ _ (by simp [init])]

end AlgoUdf.BfsBound
