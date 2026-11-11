import AlgoUdf.DfsComplete
/-! # The subtree lemma for `enumerate_paths` and its corollaries (see `AlgoUdf.DfsComplete`) -/
namespace AlgoUdf.Dfs
open AlgoUdf.Paths (Dir farEndpoint)
open AlgoUdf.Graph

theorem explore {g : G} {cfg : Cfg} (hm : BoundMono g cfg) :
    ∀ m (L : List (Rel × Nat)) (st : DS) (fr : Frame) (below : List Frame),
      Inv g cfg st → st.frames = fr :: below → fr.lv = some L → cfg.maxLen - below.length = m →
      Explored g cfg st fr below L := by
  intro m
  induction m using Nat.strongRecOn with
  | _ m OH =>
  intro L
  induction L with
  | nil =>
    intro st fr below _ hfs hlv _
    refine ⟨st, .refl _, ?_, fun _ => rfl, ?_⟩
    · rw [hfs]; cases fr; simp at hlv; subst hlv; rfl
    · intro es W C h; cases h <;> simp_all
  | cons p rest IH =>
    intro st fr below I hfs hlv hmeq
    obtain ⟨r, x⟩ := p
    have vis_iff : ∀ y, st.onPath y = true ↔ y ∈ (fr :: below).map (·.node) := by
      intro y; rw [I.onPath y, hfs]
    -- the skipped-successor case, shared by four guards
    have skipCase : step g cfg st = some { st with frames := { fr with lv := some rest } :: below } →
        (∀ es W C, Ext g cfg [(r, x)] fr.pw fr.pc below.length ((fr :: below).map (·.node)) es W C →
          ∀ b, BoundLe b st.bound → BLe W b → False) →
        Explored g cfg st fr below ((r, x) :: rest) := by
      intro hstep hno
      have I1 := inv_step I hstep
      obtain ⟨st', h1, h2, h3, h4⟩ := IH _ { fr with lv := some rest } below I1 rfl rfl hmeq
      have hb := Star.bound hm I1 h1
      refine ⟨st', Star.head hstep h1, by rw [h2], h3, ?_⟩
      intro es W C hext hW
      rcases hext.cons_split with h | h
      · have := h4 es W C h hW; rwa [pathOf_lv] at this
      · exact (hno es W C h st'.bound hb hW).elim
    by_cases hon : st.onPath x = true
    · apply skipCase (step_skip hfs hlv (Or.inl hon))
      intro es W C h _ _ _
      obtain ⟨r', x', _, _, _, hmem, hnv, _⟩ := h.first_le
      simp at hmem; obtain ⟨rfl, rfl⟩ := hmem
      exact hnv ((vis_iff x').mp hon)
    have hon' : st.onPath x = false := by simpa using hon
    cases hw : g.wt r.2.2 with
    | none =>
      apply skipCase (step_skip hfs hlv (Or.inr (Or.inl hw)))
      intro es W C h _ _ _
      obtain ⟨r', x', c, _, _, hmem, _, hc, _⟩ := h.first_le
      simp at hmem; obtain ⟨rfl, rfl⟩ := hmem
      rw [hw] at hc; cases hc
    | some c =>
      by_cases hbd : st.bound.any (· < fr.pw + c) = true
      · apply skipCase (step_skip hfs hlv (Or.inr (Or.inr ⟨c, hw, Or.inl hbd⟩)))
        intro es W C h b hb hW
        obtain ⟨r', x', c', _, _, hmem, _, hc, _, hle⟩ := h.first_le
        simp at hmem; obtain ⟨rfl, rfl⟩ := hmem
        rw [hw] at hc; cases hc
        have := BLe.mono hW hb
        unfold BLe at this
        cases hsb : st.bound with
        | none => rw [hsb] at hbd; cases hbd
        | some y => rw [hsb] at hbd this; simp at hbd this; omega
      by_cases hmc : cfg.maxCost.any (· < fr.pc + g.cost r.2.2) = true
      · apply skipCase (step_skip hfs hlv (Or.inr (Or.inr ⟨c, hw, Or.inr hmc⟩)))
        intro es W C h _ _ _
        obtain ⟨r', x', c', _, _, hmem, _, _, hco, _⟩ := h.first_le
        simp at hmem; obtain ⟨rfl, rfl⟩ := hmem
        cases hm' : cfg.maxCost with
        | none => rw [hm'] at hmc; cases hmc
        | some y => rw [hm'] at hmc hco; simp at hmc hco; omega
      -- accepted: push the successor
      have hbd' : st.bound.any (· < fr.pw + c) = false := by simpa using hbd
      have hmc' : cfg.maxCost.any (· < fr.pc + g.cost r.2.2) = false := by simpa using hmc
      have hstep := step_push hfs hlv hon' hw hbd' hmc'
      generalize hs1 : (⟨_, _, _, _, _⟩ : DS) = s1 at hstep
      have I1 := inv_step I hstep
      have hs1f : s1.frames = ⟨if (!(cfg.tgt.all (· == x) && cfg.tgt.isSome) &&
            decide (below.length + 1 < cfg.maxLen)) then some (succsRev g x) else none, x, fr.pw + c,
            fr.pc + g.cost r.2.2, some r⟩ :: { fr with lv := some rest } :: below := by rw [← hs1]
      have hs1o : s1.onPath = upd st.onPath x true := by rw [← hs1]
      have hs1l : ∀ f, cfg.tgt.all (· == x) = true →
          f = (⟨pathOf (fr :: below) ++ [r], fr.pw + c, fr.pc + g.cost r.2.2⟩ : FP) → f ∈ s1.log := by
        intro f hat hf; rw [← hs1]; simp only [hat, if_true, pathOf_lv]; rw [hf]; simp
      -- from the state after returning to this frame (successor popped), finish with `IH`
      have finish : ∀ s3, Star g cfg st s3 → Inv g cfg s3 →
          s3.frames = { fr with lv := some rest } :: below →
          (∀ y, s3.onPath y = st.onPath y) →
          (∀ es W C, Ext g cfg [(r, x)] fr.pw fr.pc below.length ((fr :: below).map (·.node)) es W C →
            ∀ b, BoundLe b s3.bound → BLe W b → (⟨pathOf (fr :: below) ++ es, W, C⟩ : FP) ∈ s3.log) →
          Explored g cfg st fr below ((r, x) :: rest) := by
        intro s3 h3 I3 hf3 ho3 hfirst
        obtain ⟨st', h1, h2, h3', h4⟩ := IH s3 { fr with lv := some rest } below I3 hf3 rfl hmeq
        have hb := Star.bound hm I3 h1
        refine ⟨st', h3.trans h1, by rw [h2], fun y => by rw [h3', ho3], ?_⟩
        intro es W C hext hW
        rcases hext.cons_split with h | h
        · have := h4 es W C h hW; rwa [pathOf_lv] at this
        · exact log_sub h1 (hfirst es W C h st'.bound hb hW)
      by_cases hexp : (!(cfg.tgt.all (· == x) && cfg.tgt.isSome) &&
          decide (below.length + 1 < cfg.maxLen)) = true
      · -- expanded: explore the new frame (outer induction), then backtrack
        have hlen : below.length + 1 < cfg.maxLen := by
          simp only [Bool.and_eq_true, decide_eq_true_eq] at hexp; exact hexp.2
        have hnf : s1.frames = ⟨some (succsRev g x), x, fr.pw + c, fr.pc + g.cost r.2.2, some r⟩ ::
            { fr with lv := some rest } :: below := by rw [hs1f, if_pos hexp]
        obtain ⟨s2, h12, h2f, h2o, h2e⟩ := OH (cfg.maxLen - (below.length + 1)) (by omega)
          (succsRev g x) s1 _ _ I1 hnf rfl (by simp)
        have I2 := Star.inv I1 h12
        have hback := step_back (g := g) (cfg := cfg) h2f (Or.inr rfl) (by simp)
        have I3 := inv_step I2 hback
        have hb21 := Star.bound hm I1 h12
        apply finish _ ((Star.head hstep h12).tail hback) I3 rfl
        · intro y; dsimp only
          by_cases hy : y = x
          · subst hy; simp [hon']
          · rw [upd_ne _ _ _ _ hy, h2o, hs1o, upd_ne _ _ _ _ hy]
        · intro es W C h b hb hW
          cases h with
          | last hmem _ hc _ _ hat =>
            simp at hmem; obtain ⟨rfl, rfl⟩ := hmem
            rw [hw] at hc; cases hc
            exact log_sub h12 (hs1l _ hat rfl)
          | @more _ _ _ _ _ r' x' c' rest' _ _ hmem _ hc _ _ _ hsub =>
            simp at hmem; obtain ⟨hr, hx⟩ := hmem; subst r'; subst x'
            rw [hw] at hc; cases hc
            have := h2e rest' W C (by simpa using hsub) (BLe.mono hW (hb.trans (BoundLe.refl _)))
            have hp : pathOf (⟨some (succsRev g x), x, fr.pw + c, fr.pc + g.cost r.2.2, some r⟩ ::
                { fr with lv := some rest } :: below) = pathOf (fr :: below) ++ [r] := by
              simp [pathOf, List.filterMap_cons]
            rw [hp, List.append_assoc] at this
            exact this
      · -- not expanded: the next step backtracks at once
        have hnf : s1.frames = ⟨none, x, fr.pw + c, fr.pc + g.cost r.2.2, some r⟩ ::
            { fr with lv := some rest } :: below := by rw [hs1f, if_neg hexp]
        have hback := step_back (g := g) (cfg := cfg) hnf (Or.inl rfl) (by simp)
        have I3 := inv_step I1 hback
        apply finish _ ((Star.refl _).tail hstep |>.tail hback) I3 rfl
        · intro y; dsimp only
          by_cases hy : y = x
          · subst hy; simp [hon']
          · rw [upd_ne _ _ _ _ hy, hs1o, upd_ne _ _ _ _ hy]
        · intro es W C h b hb hW
          cases h with
          | last hmem _ hc _ _ hat =>
            simp at hmem; obtain ⟨rfl, rfl⟩ := hmem
            rw [hw] at hc; cases hc
            exact hs1l _ hat rfl
          | more hmem _ _ _ hnt hlt _ =>
            simp at hmem; obtain ⟨rfl, rfl⟩ := hmem
            exfalso; apply hexp
            simp only [Bool.and_eq_true, Bool.not_eq_true', decide_eq_true_eq]
            exact ⟨hnt, hlt⟩

end AlgoUdf.Dfs
