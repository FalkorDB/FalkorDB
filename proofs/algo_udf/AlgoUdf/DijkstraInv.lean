import AlgoUdf.DijkstraModel
/-! # `dijkstra_single_path`: invariant preservation (see `AlgoUdf.DijkstraModel`) -/
namespace AlgoUdf.Dijkstra
open AlgoUdf.Paths (Dir farEndpoint)
open AlgoUdf.Graph

/-! ### Preservation -/

section relax
variable {g : G} {src tgt : Nat}

/-- Settling the popped node `v` at weight `w`, before its relaxations. -/
theorem inv_settle {st : St} {w v : Nat} (I : Inv g src tgt noEx st) (hd : st.done = false)
    (hm : (w, v) ∈ st.heap) (hmin : IsMin st.heap w) (hs : st.settled v = false) :
    Inv g src tgt (fun x _ => x = v) (settle st v (st.heap.erase (w, v))) ∧
      D src st.labels v = w := by
  have pa := pop_analysis I hm hmin hs
  have hDv : D src st.labels v = w := by
    rcases pa with ⟨rfl, rfl⟩ | ⟨hne, l, hl, rfl⟩
    · simp [D]
    · exact D_lbl hne hl
  -- if src is still unsettled we are popping it
  have hsrc : st.settled src = false → v = src := by
    intro h; have := (I.h0 h).1; rw [this] at hm; simp at hm; exact hm.2
  refine ⟨⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩, hDv⟩
  · -- h0
    intro h
    simp only [settle] at h
    by_cases hv : src = v
    · subst hv; simp at h
    · rw [upd_ne _ _ _ _ hv] at h; exact absurd (hsrc h) (Ne.symm hv)
  · -- lbl
    intro x l hl
    simp only [settle] at hl ⊢
    obtain ⟨h1, h2, h3, h4⟩ := I.lbl x l hl
    refine ⟨h1, ?_, ?_, h4⟩
    · by_cases hp : l.parent = v
      · rw [hp]; simp
      · rw [upd_ne _ _ _ _ hp]; exact h2
    · intro hx
      have hp : l.parent ≠ v := by intro hp; rw [← hp] at hs; rw [hs] at h2; cases h2
      rw [upd_ne _ _ _ _ hp]
      by_cases hxv : x = v
      · subst hxv; simp; exact I.ordLt _ h2
      · rw [upd_ne _ _ _ _ hxv] at hx ⊢; exact h3 hx
  · -- heapE
    intro w' x hx; exact I.heapE w' x (List.mem_of_mem_erase hx)
  · -- latest
    intro x l hl hx
    simp only [settle] at hx ⊢
    have hxv : x ≠ v := by intro h; subst h; simp at hx
    rw [upd_ne _ _ _ _ hxv] at hx
    exact mem_erase_of_ne (I.latest x l hl hx) (by intro h; injection h with _ h; exact hxv h)
  · -- setLbl
    intro x hx hxs
    simp only [settle] at hx ⊢
    by_cases hxv : x = v
    · subst hxv
      rcases pa with ⟨h, _⟩ | ⟨_, l, hl, _⟩
      · exact absurd h hxs
      · rw [hl]; simp
    · rw [upd_ne _ _ _ _ hxv] at hx; exact I.setLbl x hx hxs
  · -- ordLt
    intro x hx
    simp only [settle] at hx ⊢
    by_cases hxv : x = v
    · subst hxv; simp
    · rw [upd_ne _ _ _ _ hxv] at hx ⊢; have := I.ordLt x hx; omega
  · -- mono
    intro u w' x hu hx
    simp only [settle] at hu ⊢
    have hx' := List.mem_of_mem_erase hx
    by_cases huv : u = v
    · subst huv; rw [hDv]; exact hmin _ hx'
    · rw [upd_ne _ _ _ _ huv] at hu; exact I.mono u w' x hu hx'
  · -- edge
    intro _ x y r c hx hy he
    simp only [settle] at hx hy ⊢
    by_cases hxv : x = v <;> by_cases hyv : y = v
    · subst hxv; subst hyv; omega
    · subst hxv; rw [upd_ne _ _ _ _ hyv] at hy
      have := I.mono y w x hy hm; omega
    · subst hyv; rw [upd_ne _ _ _ _ hxv] at hx
      rcases I.fr hd x y r c hx he (fun h => h) with h | ⟨l, hl, hle⟩
      · rw [h] at hs; cases hs
      · rcases pa with ⟨h1, _⟩ | ⟨h1, l', hl', hw⟩
        · -- popping src means nothing was settled yet
          have := (I.h0 (h1 ▸ hs)).2 x; rw [this] at hx; cases hx
        · rw [hl] at hl'; cases hl'; omega
    · rw [upd_ne _ _ _ _ hxv] at hx; rw [upd_ne _ _ _ _ hyv] at hy
      exact I.edge hd x y r c hx hy he
  · -- fr
    intro _ x y r c hx he hne
    simp only [settle] at hx ⊢
    have hxv : x ≠ v := hne
    rw [upd_ne _ _ _ _ hxv] at hx
    rcases I.fr hd x y r c hx he (fun h => h) with h | h
    · left; by_cases hyv : y = v
      · subst hyv; simp
      · rw [upd_ne _ _ _ _ hyv]; exact h
    · exact Or.inr h
  · -- opt
    intro u es W hu hw
    simp only [settle] at hu ⊢
    by_cases huv : u = v
    · subst huv
      rw [hDv]
      cases hss : st.settled src with
      | false =>
        rcases pa with ⟨_, rfl⟩ | ⟨h, _⟩
        · omega
        · exact absurd (hsrc hss) h
      | true =>
        obtain ⟨z, l, hz, hl, hle⟩ := frontier_lemma I hd (fun _ _ => Or.inl id) hw hss hs
        have := hmin _ (I.latest z l hl hz)
        simp [D] at hle this; omega
    · rw [upd_ne _ _ _ _ huv] at hu; exact I.opt u es W hu hw

/-- One `relaxOne` keeps the invariant and discharges the waiver for `r`. -/
theorem inv_relaxOne {st : St} {v w : Nat} {P : List Rel} {r : Rel}
    (I : Inv g src tgt (fun x r' => x = v ∧ r' ∉ P) st) (hd : st.done = false)
    (hv : st.settled v = true) (hDv : D src st.labels v = w)
    (hle : ∀ u, st.settled u = true → D src st.labels u ≤ w) (hr : r ∈ g.rels v) :
    Inv g src tgt (fun x r' => x = v ∧ r' ∉ P ++ [r]) (relaxOne g v w st r) ∧
      (relaxOne g v w st r).settled = st.settled ∧ (relaxOne g v w st r).done = st.done ∧
      (∀ x, st.settled x = true → (relaxOne g v w st r).labels x = st.labels x) := by
  -- src is settled (v is)
  have hsrc : st.settled src = true := by
    cases h : st.settled src with
    | true => rfl
    | false => have := (I.h0 h).2 v; rw [this] at hv; cases hv
  -- the unchanged cases all reduce to: fr for the new pair (v, r)
  have keep : (∀ y c, Edge g v y r c → st.settled y = true ∨
        ∃ l, st.labels y = some l ∧ l.w ≤ D src st.labels v + c) →
      Inv g src tgt (fun x r' => x = v ∧ r' ∉ P ++ [r]) st := by
    intro hnew
    refine ⟨I.h0, I.lbl, I.heapE, I.latest, I.setLbl, I.ordLt, I.mono, I.edge, ?_, I.opt⟩
    intro hd' x y r' c hx he hne
    by_cases hxr : x = v ∧ r' = r
    · obtain ⟨rfl, rfl⟩ := hxr; exact hnew y c he
    · apply I.fr hd' x y r' c hx he
      rintro ⟨h1, h2⟩; apply hne; refine ⟨h1, ?_⟩
      simp only [List.mem_append, List.mem_singleton, not_or]; exact ⟨h2, fun h => hxr ⟨h1, h⟩⟩
  unfold relaxOne
  split
  · rename_i hf
    refine ⟨keep ?_, rfl, rfl, fun _ _ => rfl⟩
    intro y c he; rw [he.2.1] at hf; cases hf
  · rename_i far hf
    split
    · rename_i hsf
      refine ⟨keep ?_, rfl, rfl, fun _ _ => rfl⟩
      intro y c he; rw [he.2.1] at hf; cases hf; exact Or.inl hsf
    · rename_i hsf
      have hsf' : st.settled far = false := by simpa using hsf
      split
      · rename_i hw
        refine ⟨keep ?_, rfl, rfl, fun _ _ => rfl⟩
        intro y c he; rw [he.2.2] at hw; cases hw
      · rename_i c hw
        have heF : Edge g v far r c := ⟨hr, hf, hw⟩
        split
        · rename_i hlt
          have hfs : far ≠ src := by intro h; rw [h, hsrc] at hsf'; cases hsf'
          have hfv : far ≠ v := by intro h; rw [h, hv] at hsf'; cases hsf'
          -- labels of every node but `far` are unchanged
          have Lsame : ∀ x, x ≠ far → upd st.labels far (some ⟨v, r, w + c⟩) x = st.labels x :=
            fun x hx => upd_ne _ _ _ _ hx
          have Dsame : ∀ x, x ≠ far → D src (upd st.labels far (some ⟨v, r, w + c⟩)) x =
              D src st.labels x := fun x hx => D_congr (Lsame x hx)
          have setNeFar : ∀ x, st.settled x = true → x ≠ far := by
            intro x hx h; rw [h, hsf'] at hx; cases hx
          have oldLt : ∀ l, st.labels far = some l → w + c < l.w := by
            intro l hl; rw [hl] at hlt; simpa using hlt
          refine ⟨⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩, rfl, rfl, ?_⟩
          · intro h; rw [hsrc] at h; cases h
          · intro x l hl
            dsimp only at hl ⊢
            by_cases hx : x = far
            · subst hx; simp at hl; subst hl
              refine ⟨hfs, hv, (fun h => by rw [h] at hsf'; cases hsf'), c, heF, ?_⟩
              rw [Dsame v (Ne.symm hfv), hDv]
            · rw [Lsame x hx] at hl
              obtain ⟨h1, h2, h3, c', h4, h5⟩ := I.lbl x l hl
              refine ⟨h1, h2, h3, c', h4, ?_⟩
              rw [Dsame _ (setNeFar _ h2)]; exact h5
          · intro w' x hx
            dsimp only at hx ⊢
            simp only [List.mem_cons, Prod.mk.injEq] at hx
            rcases hx with ⟨rfl, rfl⟩ | hx
            · right; exact ⟨_, upd_same _ _ _, Nat.le_refl _⟩
            · rcases I.heapE w' x hx with h | ⟨l, hl, hle'⟩
              · left; exact h
              · right
                by_cases hxf : x = far
                · subst hxf; refine ⟨_, upd_same _ _ _, ?_⟩; have := oldLt l hl; simp; omega
                · exact ⟨l, by rw [Lsame x hxf]; exact hl, hle'⟩
          · intro x l hl hx
            dsimp only at hl hx ⊢
            by_cases hxf : x = far
            · subst hxf; simp at hl; subst hl; simp
            · rw [Lsame x hxf] at hl; exact List.mem_cons_of_mem _ (I.latest x l hl hx)
          · intro x hx hxs
            dsimp only at hx ⊢
            rw [Lsame x (setNeFar x hx)]; exact I.setLbl x hx hxs
          · exact I.ordLt
          · intro u w' x hu hx
            dsimp only at hu hx ⊢
            rw [Dsame u (setNeFar u hu)]
            simp only [List.mem_cons, Prod.mk.injEq] at hx
            rcases hx with ⟨rfl, rfl⟩ | hx
            · have := hle u hu; omega
            · exact I.mono u w' x hu hx
          · intro hd' x y r' c' hx hy he
            dsimp only at hx hy ⊢
            rw [Dsame x (setNeFar x hx), Dsame y (setNeFar y hy)]
            exact I.edge hd' x y r' c' hx hy he
          · intro hd' x y r' c' hx he hne
            dsimp only at hx ⊢
            rw [Dsame x (setNeFar x hx)]
            by_cases hxr : x = v ∧ r' = r
            · obtain ⟨rfl, rfl⟩ := hxr
              obtain ⟨rfl, rfl⟩ := edge_det heF he
              right; refine ⟨_, upd_same _ _ _, ?_⟩; simp; omega
            · have hne' : ¬ (x = v ∧ r' ∉ P) := by
                rintro ⟨h1, h2⟩; apply hne; refine ⟨h1, ?_⟩
                simp only [List.mem_append, List.mem_singleton, not_or]
                exact ⟨h2, fun h => hxr ⟨h1, h⟩⟩
              rcases I.fr hd' x y r' c' hx he hne' with h | ⟨l, hl, hle'⟩
              · exact Or.inl h
              · right
                by_cases hyf : y = far
                · subst hyf; refine ⟨_, upd_same _ _ _, ?_⟩; have := oldLt l hl; simp; omega
                · exact ⟨l, by rw [Lsame y hyf]; exact hl, hle'⟩
          · intro u es W hu hw
            dsimp only at hu ⊢
            rw [Dsame u (setNeFar u hu)]; exact I.opt u es W hu hw
          · intro x hx; exact Lsame x (setNeFar x hx)
        · rename_i hge
          refine ⟨keep ?_, rfl, rfl, fun _ _ => rfl⟩
          intro y c' he
          obtain ⟨rfl, rfl⟩ := edge_det heF he
          right
          cases hl : st.labels far with
          | none => rw [hl] at hge; simp at hge
          | some l => rw [hl] at hge; simp at hge; exact ⟨l, rfl, by rw [hDv]; omega⟩

theorem relaxOne_settled (g : G) (v w : Nat) (st : St) (r : Rel) :
    (relaxOne g v w st r).settled = st.settled := by
  unfold relaxOne; split
  · rfl
  · split
    · rfl
    · split
      · rfl
      · split <;> rfl

theorem fold_settled (rs : List Rel) (v w : Nat) (st : St) :
    (rs.foldl (relaxOne g v w) st).settled = st.settled := by
  induction rs generalizing st with
  | nil => rfl
  | cons r t ih => simp only [List.foldl_cons]; rw [ih, relaxOne_settled]

/-- The whole relaxation loop. -/
theorem inv_fold {v w : Nat} :
    ∀ (rest P : List Rel) (st : St), P ++ rest = g.rels v →
      Inv g src tgt (fun x r' => x = v ∧ r' ∉ P) st → st.done = false →
      st.settled v = true → D src st.labels v = w →
      (∀ u, st.settled u = true → D src st.labels u ≤ w) →
      Inv g src tgt (fun x r' => x = v ∧ r' ∉ g.rels v) (rest.foldl (relaxOne g v w) st) ∧
        (rest.foldl (relaxOne g v w) st).done = false := by
  intro rest
  induction rest with
  | nil => intro P st hP I hd _ _ _; simp at hP; subst hP; exact ⟨I, hd⟩
  | cons r rest ih =>
    intro P st hP I hd hv hDv hle
    have hr : r ∈ g.rels v := by rw [← hP]; simp
    obtain ⟨I', hs, hdn, hl⟩ := inv_relaxOne I hd hv hDv hle hr
    simp only [List.foldl_cons]
    apply ih (P ++ [r]) _ (by simpa using hP) I' (by rw [hdn]; exact hd) (by rw [hs]; exact hv)
    · rw [D_congr (hl v hv)]; exact hDv
    · intro u hu; rw [hs] at hu; rw [D_congr (hl u hu)]; exact hle u hu

end relax
end AlgoUdf.Dijkstra
