import CchRedesignReview.WriterStep
set_option linter.unusedVariables false
set_option linter.unusedSimpArgs false

/-!
# Builder publishes keep the invariant (03 §5 publish rules and adoption, §6 Built-entry jobs)

`batch_core`: a batch read on `S0` that publishes `e` onto `cur`. Its fit passed, so `cur` has `S0`'s generation
and count: by `XRel.chain`, `cur`'s log is `S0`'s with only writers' nodes after the fork on top (or an Overflow
replacing it). So the relink (`relinked F cur.log`) is exactly those nodes, every write after the fork is in them,
and every write up to the fork is resolved by the batch or still in its copy.

`job_core`: a fold or S6 publish fitted against generation and ticket. The ticket names one `upkeep`
(`XRel.ticket`), so `since ≤ F`, and the retained log holds every write after `F`; S6 replays them.
-/
namespace CchReview

theorem fit_ok {S0 cur : Pub} {e1 : Entry} (h : IsDiscarded (fit S0 cur e1) = false) :
    cur.e.gen = e1.gen ∧ cur.e.commits + 1 = e1.commits ∧
    (fit S0 cur e1 = .catchUp .refresh ↔ cur.ver ≠ S0.ver) := by
  unfold fit at h ⊢
  cases hd : NI.discardedAgainst e1.toSE (some cur.e.toSE) with
  | some why => simp [hd, IsDiscarded] at h
  | none =>
    obtain ⟨q, hq, hg, hc⟩ := (NI.discardedAgainst_none_iff _ _).1 hd
    simp at hq; subst hq
    simp [Entry.toSE] at hg hc
    refine ⟨hg, hc, ?_⟩
    simp only [hd]
    by_cases hv : cur.ver = S0.ver
    · simp [hv]; split <;> simp
    · simp [hv]

/-- The consumer resolves every pair of its copy at its fork (03 §5 Consumption). -/
theorem consume_res {B : Nat} {w : List (Nat × Pair)} {S0 : Pub} (hS : PInv B w S0) (s : Nat) (p : Pair)
    (hsp : (s, p) ∈ w) (hs : s ≤ S0.ver) : s ≤ (consume S0.ver S0.e).res p := by
  simp only [consume]
  split
  · exact hs
  · rename_i hn
    rcases hS.nlw s p hsp hs with h1 | ⟨n, hn', h1 | ⟨h1, h2⟩⟩
    · exact h1
    · exfalso; exact hn (Or.inr ((mem_allPairs _ _).2 ⟨n, hn', h1.2⟩))
    · exfalso; exact hn (Or.inl ((hasOv_iff _).2 ⟨n, hn', h1⟩))

theorem consume_res_le {B : Nat} {w : List (Nat × Pair)} {S0 : Pub} (hS : PInv B w S0) (p : Pair) :
    (consume S0.ver S0.e).res p ≤ S0.ver := by
  simp only [consume]; split
  · exact Nat.le_refl _
  · exact hS.res_le p

/-- Facts about the batch's copy. -/
theorem batchCopy_facts {B : Nat} {w : List (Nat × Pair)} {S0 : Pub} (hS : PInv B w S0) (ph : Phase) (adv : Bool) :
    (batchCopy S0.ver ph adv S0.e).gen = S0.e.gen ∧ (batchCopy S0.ver ph adv S0.e).lineage = S0.e.lineage ∧
    (batchCopy S0.ver ph adv S0.e).commits = S0.e.commits + 1 ∧
    (batchCopy S0.ver ph adv S0.e).st = .building (nextPhase ph adv) ∧
    (∀ p, (batchCopy S0.ver ph adv S0.e).res p ≤ S0.ver) ∧
    (∀ s p, (s, p) ∈ w → s ≤ S0.ver →
      s ≤ (batchCopy S0.ver ph adv S0.e).res p ∨ covered (batchCopy S0.ver ph adv S0.e).log s p) ∧
    (consumes ph = true → (batchCopy S0.ver ph adv S0.e).log = [] ∧ (batchCopy S0.ver ph adv S0.e).pairs = 0) ∧
    (consumes ph = false →
      (batchCopy S0.ver ph adv S0.e).log = S0.e.log ∧ (batchCopy S0.ver ph adv S0.e).pairs = S0.e.pairs) := by
  cases hc : consumes ph
  · have he : batchCopy S0.ver ph adv S0.e = { S0.e with st := .building (nextPhase ph adv), commits := S0.e.commits + 1 } := by
      simp [batchCopy, hc]
    rw [he]
    exact ⟨rfl, rfl, rfl, rfl, hS.res_le, hS.nlw, fun h => by simp at h, fun _ => ⟨rfl, rfl⟩⟩
  · have he : batchCopy S0.ver ph adv S0.e =
        { consume S0.ver S0.e with st := .building (nextPhase ph adv), commits := S0.e.commits + 1 } := by
      simp [batchCopy, hc]
    rw [he]
    exact ⟨rfl, rfl, rfl, rfl, consume_res_le hS, fun s p hsp hs => Or.inl (consume_res hS s p hsp hs),
      fun _ => ⟨rfl, rfl⟩, fun h => by simp at h⟩

/-- A write after the fork that the published log covers is under one of the relinked nodes. -/
theorem covered_newer {F s : Nat} {p : Pair} {N L0 L : List Node} (hs : F < s) (hc : covered L s p)
    (hL : L = N ++ L0 ∨ L = N) (h0 : ∀ n ∈ L0, n.seq ≤ F) : covered N s p := by
  obtain ⟨n, hn, hh⟩ := hc
  have hgt : F < n.seq := by rcases hh with ⟨h1, _⟩ | ⟨_, h1⟩ <;> omega
  refine ⟨n, ?_, hh⟩
  rcases hL with rfl | rfl
  · rcases List.mem_append.1 hn with h | h
    · exact h
    · have := h0 n h; omega
  · exact hn

/-- **A batch publish keeps `PInv`** (03 §5: Same/Swap, CatchUp's relink, consumption, adoption or Restart). -/
theorem batch_core {B : Nat} {σ : Sys} (hI : Inv B σ) {S0 : Pub} (h0 : S0 ∈ σ.cur :: σ.hist) (b adv : Bool)
    {e : Entry} (he : batchResult b adv S0 σ.cur = some e) :
    PInv B σ.writes ⟨σ.cur.ver, σ.cur.batches + 1, e⟩ ∧ Fresh σ.next e ∧ e.gen = σ.cur.e.gen ∧
      e.commits = σ.cur.e.commits + 1 ∧ σ.cur.e.isBuilt = false ∧ e.upkeep = none ∧
      (∀ ph, S0.e.st = .building ph → consumes ph = true → ∀ n ∈ e.log, S0.ver < n.seq) := by
  obtain ⟨hS, hSf⟩ := hI.pub S0 h0
  obtain ⟨hC, hCf⟩ := hI.pub σ.cur (List.mem_cons_self ..)
  cases hst : S0.e.st with
  | built u => simp [batchResult, hst] at he
  | building ph =>
  have hSb : S0.e.isBuilt = false := isBuilt_building _ ph hst
  obtain ⟨g1, l1, c1, s1, r1, n1, cons1, ncons1⟩ := batchCopy_facts hS ph adv
  simp only [batchResult, hst] at he
  split at he
  · simp at he
  rename_i hnd
  have hnd' : IsDiscarded (fit S0 σ.cur (batchCopy S0.ver ph adv S0.e)) = false := by simpa using hnd
  obtain ⟨fg, fc, fcu⟩ := fit_ok hnd'
  -- the fork and the published entry are the same index, the same count: `cur` extends `S0`
  have hsame : S0.e.gen = σ.cur.e.gen ∧ S0.e.commits = σ.cur.e.commits := ⟨by omega, by omega⟩
  have hF : S0.ver ≤ σ.cur.ver ∧ (S0.ver = σ.cur.ver → S0.e = σ.cur.e) ∧ σ.cur.e.isBuilt = false ∧
      Ext B S0.ver S0.e σ.cur.e := by
    rcases List.mem_cons.1 h0 with rfl | hh
    · exact ⟨Nat.le_refl _, fun _ => rfl, hSb, Ext.refl _ _ _⟩
    · have hx := hI.rel S0 hh
      have hcb : σ.cur.e.isBuilt = false := by rw [← (hx.same_gen hsame.1).2.2 hsame.2]; exact hSb
      refine ⟨by rcases hx.ver_le with h | ⟨h, _⟩ <;> omega, fun hv => (hx.same_ver hv).2 hsame.2, hcb,
        hx.chain hSb hcb hsame.1 hsame.2⟩
  obtain ⟨hFle, hFeq, hcb, hext⟩ := hF
  obtain ⟨_, hres, _, _, _, N, hN, hlog⟩ := hext
  -- the relinked nodes are exactly the writers' nodes after the fork
  have hnew : ∃ N' : List Node, (if fit S0 σ.cur (batchCopy S0.ver ph adv S0.e) = .catchUp .refresh
        then relinked S0.ver σ.cur.e.log else []) = N' ∧ (∀ n ∈ N', S0.ver < n.seq) ∧
      ((σ.cur.e.log = N' ++ S0.e.log ∧ (N' = [] ∨ sumPairs σ.cur.e.log ≤ B)) ∨ (σ.cur.e.log = N' ∧ hasOv N' = true)) := by
    by_cases hv : σ.cur.ver = S0.ver
    · have heq := hFeq hv.symm
      have : ¬ fit S0 σ.cur (batchCopy S0.ver ph adv S0.e) = .catchUp .refresh := fun h => (fcu.1 h) hv
      refine ⟨[], by simp [this], by simp, Or.inl ⟨by simp [heq], Or.inl rfl⟩⟩
    · have : fit S0 σ.cur (batchCopy S0.ver ph adv S0.e) = .catchUp .refresh := fcu.2 hv
      refine ⟨N, ?_, hN, hlog⟩
      simp only [this, ite_true]
      rcases hlog with ⟨hl, _⟩ | ⟨hl, _⟩
      · rw [hl]; exact relinked_seam _ _ _ hN hS.seq_le
      · rw [hl]; exact relinked_all _ _ hN
  obtain ⟨N', hN'eq, hN'gt, hN'log⟩ := hnew
  rw [hN'eq] at he
  have hN'sub : ∀ n ∈ N', n ∈ σ.cur.e.log := by
    intro n hn; rcases hN'log with ⟨hl, _⟩ | ⟨hl, _⟩ <;> rw [hl]
    · exact List.mem_append_left _ hn
    · exact hn
  have hLor : σ.cur.e.log = N' ++ S0.e.log ∨ σ.cur.e.log = N' := by
    rcases hN'log with ⟨hl, _⟩ | ⟨hl, _⟩
    · exact Or.inl hl
    · exact Or.inr hl
  -- every write after the fork is under a relinked node
  have hafter : ∀ s p, (s, p) ∈ σ.writes → S0.ver < s → s ≤ σ.cur.ver → covered N' s p := by
    intro s p hsp h1 h2
    rcases hC.nlw s p hsp h2 with h3 | h3
    · have : σ.cur.e.res p ≤ S0.ver := by rw [hres]; exact hS.res_le p
      omega
    · exact covered_newer h1 h3 hLor hS.seq_le
  have hN'seq : ∀ n ∈ N', n.seq ≤ σ.cur.ver := fun n hn => hC.seq_le n (hN'sub n hn)
  have hN'snd : ∀ n ∈ N', ∀ p ∈ n.ps, (n.seq, p) ∈ σ.writes := fun n hn => hC.sound n (hN'sub n hn)
  -- the bound for a consumer's (empty) remainder
  have hbndN : sumPairs N' ≤ B ∨ hasOv N' = true := by
    rcases hN'log with ⟨hl, rfl | hb⟩ | ⟨_, ho⟩
    · left; simp
    · left; rw [hl] at hb; exact Nat.le_trans (sumPairs_le_append_left _ _) hb
    · exact Or.inr ho
  have fresh_of : ∀ e' : Entry, e'.gen = S0.e.gen → e'.lineage = S0.e.lineage → e'.upkeep = none → Fresh σ.next e' :=
    fun e' hg hl hu => ⟨by rw [hg]; exact hSf.gen, by rw [hl]; exact hSf.lineage, by simp [hu]⟩
  split at he
  · -- adoption
    rename_i hadopt
    have hcons : consumes ph = true := by cases ph <;> first | rfl | (simp [adopts] at hadopt)
    obtain ⟨hl1, hp1⟩ := cons1 hcons
    split at he
    · rename_i hfits
      have hov : hasOv N' = false := by simp at hfits; exact hfits.2
      simp at he; subst he
      refine ⟨⟨?_, by simp, by simp, by simp, ?_, ?_, fun _ => by simp, fun _ => rfl, fun u hu => by simp at hu,
        fun h => by simp [Entry.isBuilt] at h⟩, fresh_of _ g1 l1 (by simp [Entry.upkeep]), g1.trans hsame.1,
        by simp; omega, hcb, by simp [Entry.upkeep], fun _ _ _ n hn => by simp at hn⟩
      · intro p; exact resolve_le p (Nat.le_trans (r1 p) hFle)
      · -- built: every write resolved
        intro s p hsp hs
        left
        show s ≤ resolve _ _ _ p
        by_cases hsF : s ≤ S0.ver
        · rcases n1 s p hsp hsF with h | h
          · exact Nat.le_trans h (resolve_ge p (Nat.le_trans (r1 p) hFle))
          · rw [hl1] at h; obtain ⟨_, hn, _⟩ := h; simp at hn
        · rw [resolve_mem (allPairs_of_covered (hafter s p hsp (by omega) hs) hov)]; exact hs
      · intro _ s p hsp hs
        show s ≤ resolve _ _ _ p
        by_cases hsF : s ≤ S0.ver
        · rcases n1 s p hsp hsF with h | h
          · exact Nat.le_trans h (resolve_ge p (Nat.le_trans (r1 p) hFle))
          · rw [hl1] at h; obtain ⟨_, hn, _⟩ := h; simp at hn
        · rw [resolve_mem (allPairs_of_covered (hafter s p hsp (by omega) hs) hov)]; exact hs
    · -- Absorbed::Restart: Building{Repair}, the relinked nodes as its log
      simp at he; subst he
      refine ⟨⟨fun p => Nat.le_trans (r1 p) hFle, hN'seq, rfl, hN'snd, ?_, fun h => by simp [Entry.isBuilt] at h,
        fun h => by simp [Entry.isBuilt] at h, fun h => by simp at h, fun u hu => by simp at hu, fun _ => hbndN⟩,
        fresh_of _ g1 l1 (by simp [Entry.upkeep]), g1.trans hsame.1, by simp; omega, hcb, by simp [Entry.upkeep],
        fun _ _ _ n hn => hN'gt n hn⟩
      intro s p hsp hs
      by_cases hsF : s ≤ S0.ver
      · rcases n1 s p hsp hsF with h | h
        · exact Or.inl h
        · rw [hl1] at h; obtain ⟨_, hn, _⟩ := h; simp at hn
      · exact Or.inr (hafter s p hsp (by omega) hs)
  · -- a batch that does not adopt: its copy, the relinked nodes on top
    simp at he; subst he
    have hb1 : (batchCopy S0.ver ph adv S0.e).isBuilt = false := by simp [Entry.isBuilt, s1]
    refine ⟨⟨fun p => Nat.le_trans (r1 p) hFle, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩,
      fresh_of _ g1 l1 (by simp [Entry.upkeep, s1]), g1.trans hsame.1, by simp; omega, hcb,
      by simp [Entry.upkeep, s1], ?_⟩
    · intro n hn; rcases List.mem_append.1 hn with hn | hn
      · exact hN'seq n hn
      · cases hc : consumes ph
        · rw [(ncons1 hc).1] at hn; exact Nat.le_trans (hS.seq_le n hn) hFle
        · rw [(cons1 hc).1] at hn; simp at hn
    · show sumPairs N' + _ = sumPairs (N' ++ _)
      rw [sumPairs_append]
      cases hc : consumes ph
      · rw [(ncons1 hc).1, (ncons1 hc).2, hS.cnt]
      · rw [(cons1 hc).1, (cons1 hc).2]; rfl
    · intro n hn p hp; rcases List.mem_append.1 hn with hn | hn
      · exact hN'snd n hn p hp
      · cases hc : consumes ph
        · rw [(ncons1 hc).1] at hn; exact hS.sound n hn p hp
        · rw [(cons1 hc).1] at hn; simp at hn
    · intro s p hsp hs
      by_cases hsF : s ≤ S0.ver
      · rcases n1 s p hsp hsF with h | h
        · exact Or.inl h
        · exact Or.inr (covered_append_right _ h)
      · exact Or.inr (covered_append_left _ (hafter s p hsp (by omega) hs))
    · intro h; simp [Entry.isBuilt, s1] at h
    · intro h; simp [Entry.isBuilt, s1] at h
    · intro h; simp [s1] at h
    · intro u hu; simp [s1] at hu
    · intro _
      show sumPairs N' + _ ≤ B ∨ hasOv (N' ++ _) = true
      cases hc : consumes ph
      · rw [(ncons1 hc).1, (ncons1 hc).2, hS.cnt, ← sumPairs_append]
        rcases hN'log with ⟨hl, rfl | hb⟩ | ⟨_, ho⟩
        · simpa using hS.bound hSb |>.imp (fun h => by rw [← hS.cnt]; exact h) id
        · left; rw [← hl]; exact hb
        · right; simp [hasOv_append, ho]
      · rw [(cons1 hc).1, (cons1 hc).2]; simpa using hbndN
    · intro ph' hph' hc' n hn
      injection hph' with hph'; subst hph'
      rcases List.mem_append.1 hn with hn | hn
      · exact hN'gt n hn
      · rw [(cons1 hc').1] at hn; simp at hn

end CchReview
