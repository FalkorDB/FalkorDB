import CchRedesignReview.Builder
set_option linter.unusedVariables false
set_option linter.unusedSimpArgs false

/-!
# Every step keeps the invariant; every reachable state satisfies it

`job_core` (Built-entry jobs, 03 §6 D6), then `inv_init`, `inv_writer`, `inv_batch`, `inv_job`, `inv_other`,
and `reach_inv`: `Inv` is an inductive invariant of `Step D B` from `init`, for every bound `B` and every
interleaving.
-/
namespace CchReview

/-- **A Built-entry job publish keeps `PInv`** (03 §6: fitted against generation and ticket; S6 replays the
    retained nodes after its fork onto the new lineage; a fold keeps the published hierarchy). -/
theorem job_core {B : Nat} {σ : Sys} (hI : Inv B σ) {S0 : Pub} (h0 : S0 ∈ σ.cur :: σ.hist) (b : Bool)
    {e : Entry} {n : Nat} (he : jobResult b S0 σ.cur σ.next = some (e, n)) :
    PInv B σ.writes ⟨σ.cur.ver, σ.cur.batches + 1, e⟩ ∧ Fresh n e ∧ σ.next ≤ n ∧ e.gen = σ.cur.e.gen ∧
      e.commits = σ.cur.e.commits + 1 ∧ e.st = .built none ∧ σ.cur.e.isBuilt = true := by
  obtain ⟨hS, hSf⟩ := hI.pub S0 h0
  obtain ⟨hC, hCf⟩ := hI.pub σ.cur (List.mem_cons_self ..)
  have hFle : S0.ver ≤ σ.cur.ver := by
    rcases List.mem_cons.1 h0 with rfl | hh
    · exact Nat.le_refl _
    · rcases (hI.rel S0 hh).ver_le with h | ⟨h, _⟩ <;> omega
  cases h1 : S0.e.st with
  | building _ => simp [jobResult, h1] at he
  | built up0 =>
  cases up0 with
  | none => simp [jobResult, h1] at he
  | some u0 =>
  cases h2 : σ.cur.e.st with
  | building _ => simp [jobResult, h1, h2] at he
  | built up =>
  cases up with
  | none => simp [jobResult, h1, h2] at he
  | some u =>
  have hCb : σ.cur.e.isBuilt = true := isBuilt_built _ _ h2
  simp only [jobResult, h1, h2] at he
  split at he
  · rename_i hfit
    have hu : u0 = u := by
      rcases List.mem_cons.1 h0 with rfl | hh
      · rw [h1] at h2; simp at h2; exact h2
      · exact ((hI.rel S0 hh).ticket u0 u h1 h2 hfit.2).1
    subst hu
    have hsince : u0.since ≤ S0.ver := (hS.upSome u0 h1).1
    cases hk : u0.kind with
    | fold =>
      simp only [hk] at he; simp at he; obtain ⟨rfl, rfl⟩ := he
      refine ⟨⟨hC.res_le, by simp, by simp, by simp, fun s p hsp hs => Or.inl (hC.built hCb s p hsp hs),
        fun _ => hC.built hCb, fun _ => by simp, fun _ => rfl, fun u' hu' => by simp at hu',
        fun h => by simp [Entry.isBuilt] at h⟩, ⟨hCf.gen, hCf.lineage, fun u' hu' => by simp [Entry.upkeep] at hu'⟩,
        Nat.le_refl _, rfl, rfl, rfl, hCb⟩
    | s6 =>
      simp only [hk] at he
      split at he
      · simp at he; obtain ⟨rfl, rfl⟩ := he
        have hres : ∀ s p, (s, p) ∈ σ.writes → s ≤ σ.cur.ver → s ≤ s6Res S0.ver σ.cur.ver σ.cur.e.log p := by
          intro s p hsp hs
          by_cases hsF : s ≤ S0.ver
          · exact Nat.le_trans hsF (resolve_ge (r := fun _ => S0.ver) p hFle)
          · obtain ⟨_, _, cov⟩ := hC.upSome u0 h2
            obtain ⟨m, hm, hms, hmp⟩ := cov s p hsp (by omega) hs
            have : p ∈ allPairs (σ.cur.e.log.filter (fun n => decide (S0.ver < n.seq))) :=
              (mem_allPairs _ _).2 ⟨m, List.mem_filter.2 ⟨hm, by simp; omega⟩, hmp⟩
            simp only [s6Res]; rw [resolve_mem this]; exact hs
        refine ⟨⟨fun p => resolve_le (r := fun _ => S0.ver) p hFle, by simp, by simp, by simp,
          fun s p hsp hs => Or.inl (hres s p hsp hs), fun _ => hres, fun _ => by simp, fun _ => rfl,
          fun u' hu' => by simp at hu', fun h => by simp [Entry.isBuilt] at h⟩,
          ⟨Nat.lt_trans hCf.gen (Nat.lt_succ_self _), Nat.lt_succ_self _, fun u' hu' => by simp [Entry.upkeep] at hu'⟩,
          Nat.le_succ _, rfl, rfl, rfl, hCb⟩
      · simp at he
  · simp at he

theorem PInv.congr {B : Nat} {w : List (Nat × Pair)} {S S' : Pub} (h : PInv B w S) (hv : S'.ver = S.ver)
    (he : S'.e = S.e) : PInv B w S' := by
  obtain ⟨v, bt, e⟩ := S; obtain ⟨v', bt', e'⟩ := S'
  simp at hv he; subst hv; subst he
  exact ⟨h.res_le, h.seq_le, h.cnt, h.sound, h.nlw, h.built, h.noOv, h.upNone, h.upSome, h.bound⟩

/-- The old states under a writer's new writes. -/
theorem pinv_old_writer {B : Nat} {σ : Sys} (hI : Inv B σ) {S : Pub} (hS : S ∈ σ.cur :: σ.hist)
    (extra : List (Nat × Pair)) (hx : ∀ x ∈ extra, x.1 = σ.cur.ver + 1) : PInv B (σ.writes ++ extra) S := by
  have hle : S.ver ≤ σ.cur.ver := by
    rcases List.mem_cons.1 hS with rfl | hh
    · exact Nat.le_refl _
    · rcases (hI.rel S hh).ver_le with h | ⟨h, _⟩ <;> omega
  refine (hI.pub S hS).1.mono (fun x hx' => List.mem_append_left _ hx') ?_
  intro x hx'
  rcases List.mem_append.1 hx' with h | h
  · exact Or.inl h
  · right; rw [hx x h]; omega

theorem inv_init (B : Nat) : Inv B init where
  pub := by
    intro S hS; simp [init] at hS; subst hS
    refine ⟨⟨fun _ => Nat.le_refl _, by simp, rfl, by simp, by simp [init], fun h => by simp [Entry.isBuilt] at h,
      fun h => by simp [Entry.isBuilt] at h, fun h => by simp at h, fun u hu => by simp at hu,
      fun _ => Or.inl (Nat.zero_le _)⟩, ⟨by simp [init], by simp [init], fun u hu => by simp [Entry.upkeep] at hu⟩⟩
  rel := by intro S0 h; simp [init] at h
  wver := by intro x h; simp [init] at h

theorem inv_writer {B : Nat} {σ : Sys} (hI : Inv B σ) {fork : Pub} (hf : fork ∈ σ.cur :: σ.hist)
    (hv : fork.ver = σ.cur.ver) (pre post : List (List Pair × Out)) (b : Bool) :
    Inv B (writerSys D B fork pre post b σ) := by
  obtain ⟨hP, hF, hnx, hT⟩ := writer_core hI hf hv pre post b
  have hx : ∀ x ∈ (segPairs (pre ++ post)).map (fun p => (σ.cur.ver + 1, p)), x.1 = σ.cur.ver + 1 := by
    intro x h; obtain ⟨p, _, rfl⟩ := List.mem_map.1 h; rfl
  refine ⟨?_, ?_, ?_⟩
  · intro S hS
    simp only [writerSys] at hS
    rcases List.mem_cons.1 hS with rfl | hS
    · exact ⟨hP, hF⟩
    · exact ⟨pinv_old_writer hI hS _ hx, (hI.pub S hS).2.mono hnx⟩
  · intro S0 hS0
    simp only [writerSys] at hS0 ⊢
    have hS0' : S0 ∈ σ.cur :: σ.hist := hS0
    have hle : S0.ver ≤ σ.cur.ver := by
      rcases List.mem_cons.1 hS0' with rfl | hh
      · exact Nat.le_refl _
      · rcases (hI.rel S0 hh).ver_le with h | ⟨h, _⟩ <;> omega
    have hS0f := (hI.pub S0 hS0').2
    -- `S0` against the old `cur` (itself, or a historical relation)
    have rel0 : S0.e.gen = σ.cur.e.gen → S0.e.commits ≤ σ.cur.e.commits ∧
        (S0.e.isBuilt = true → σ.cur.e.isBuilt = true) ∧ (S0.e.commits = σ.cur.e.commits → S0.e.isBuilt = σ.cur.e.isBuilt) := by
      rcases List.mem_cons.1 hS0' with rfl | hh
      · intro _; exact ⟨Nat.le_refl _, id, fun _ => rfl⟩
      · exact (hI.rel S0 hh).same_gen
    have chain0 : S0.e.isBuilt = false → σ.cur.e.isBuilt = false → S0.e.gen = σ.cur.e.gen →
        S0.e.commits = σ.cur.e.commits → Ext B S0.ver S0.e σ.cur.e := by
      rcases List.mem_cons.1 hS0' with rfl | hh
      · intro _ _ _ _; exact Ext.refl _ _ _
      · exact (hI.rel S0 hh).chain
    have ticket0 : ∀ u0 u, S0.e.st = .built (some u0) → σ.cur.e.st = .built (some u) → u0.ticket = u.ticket →
        u0 = u ∧ S0.e.gen = σ.cur.e.gen := by
      rcases List.mem_cons.1 hS0' with rfl | hh
      · intro u0 u h1 h2 _; rw [h1] at h2; simp at h2; exact ⟨h2, rfl⟩
      · exact (hI.rel S0 hh).ticket
    refine ⟨?_, ?_, ?_, ?_, ?_⟩ <;> dsimp only
    · exact Or.inl (by omega)
    · intro h; omega
    · intro hg
      rcases hT with ⟨hgen, _⟩ | ⟨hg', hc', hb', _, _⟩
      · have := hS0f.gen; omega
      · obtain ⟨a, b', c'⟩ := rel0 (hg.trans hg')
        refine ⟨by omega, fun h => hb'.trans (b' h), fun h => ?_⟩
        rw [hb']; exact c' (by omega)
    · intro hb0 hb hg hc
      rcases hT with ⟨hgen, _⟩ | ⟨hg', hc', hb', hx', _⟩
      · have := hS0f.gen; omega
      · have hcb : σ.cur.e.isBuilt = false := hb'.symm.trans hb
        exact (chain0 hb0 hcb (hg.trans hg') (by omega)).trans ((hx' hb).weaken hle)
    · intro u0 u h1 h2 ht
      rcases hT with ⟨_, hb⟩ | ⟨hg', _, _, _, hu⟩
      · simp [Entry.isBuilt, h2] at hb
      · rcases hu u (by simp [Entry.upkeep, h2]) with hcu | hfr
        · have hcu' : σ.cur.e.st = .built (some u) := (upkeep_some _ _).1 hcu
          obtain ⟨h3, h4⟩ := ticket0 u0 u h1 hcu' ht
          exact ⟨h3, h4.trans hg'.symm⟩
        · have := hS0f.ticket u0 (by simp [Entry.upkeep, h1]); omega
  · intro x hx'
    simp only [writerSys] at hx' ⊢
    rcases List.mem_append.1 hx' with h | h
    · have := hI.wver x h; omega
    · exact Nat.le_of_eq (hx x h)

theorem inv_other {B : Nat} {σ : Sys} (hI : Inv B σ) : Inv B (otherSys σ) := by
  refine ⟨?_, ?_, ?_⟩
  · intro S hS
    simp only [otherSys] at hS
    rcases List.mem_cons.1 hS with rfl | hS
    · obtain ⟨h1, h2⟩ := hI.pub σ.cur (List.mem_cons_self ..)
      exact ⟨h1.congr rfl rfl, h2⟩
    · exact hI.pub S hS
  · intro S0 hS0
    simp only [otherSys] at hS0 ⊢
    rcases List.mem_cons.1 hS0 with rfl | hh
    · exact ⟨Or.inr ⟨rfl, Nat.lt_succ_self _⟩, fun _ => ⟨rfl, fun _ => rfl⟩, fun _ => ⟨Nat.le_refl _, id, fun _ => rfl⟩,
        fun _ _ _ _ => Ext.refl _ _ _, fun u0 u h1 h2 _ => by rw [h1] at h2; simp at h2; exact ⟨h2, rfl⟩⟩
    · have hx := hI.rel S0 hh
      exact ⟨by rcases hx.ver_le with h | ⟨h1, h2⟩
                · exact Or.inl h
                · exact Or.inr ⟨h1, Nat.lt_succ_of_lt h2⟩,
        hx.same_ver, hx.same_gen, hx.chain, hx.ticket⟩
  · intro x hx; exact hI.wver x hx

/-- A builder publish (batch or job) keeps the data version and bumps the batch counter: what every earlier
    version relates to it. -/
theorem rel_builder {B : Nat} {σ : Sys} (hI : Inv B σ) {e : Entry} (hg : e.gen = σ.cur.e.gen)
    (hc : e.commits = σ.cur.e.commits + 1) (hbb : σ.cur.e.isBuilt = true → e.isBuilt = true)
    (hu : e.upkeep = none) (hcb : e.isBuilt = false → σ.cur.e.isBuilt = false) :
    ∀ S0 ∈ σ.cur :: σ.hist, XRel B ⟨σ.cur.ver, σ.cur.batches + 1, e⟩ S0 := by
  intro S0 hS0
  have base : S0.e.gen = σ.cur.e.gen → S0.e.commits ≤ σ.cur.e.commits ∧
      (S0.e.isBuilt = true → σ.cur.e.isBuilt = true) := by
    rcases List.mem_cons.1 hS0 with rfl | hh
    · intro _; exact ⟨Nat.le_refl _, id⟩
    · intro h; obtain ⟨a, b, _⟩ := (hI.rel S0 hh).same_gen h; exact ⟨a, b⟩
  have hgv : S0.ver = σ.cur.ver → S0.e.gen = σ.cur.e.gen := by
    rcases List.mem_cons.1 hS0 with rfl | hh
    · intro _; rfl
    · intro h; exact ((hI.rel S0 hh).same_ver h).1
  refine ⟨?_, ?_, ?_, ?_, ?_⟩
  · rcases List.mem_cons.1 hS0 with rfl | hh
    · exact Or.inr ⟨rfl, Nat.lt_succ_self _⟩
    · rcases (hI.rel S0 hh).ver_le with h | ⟨h1, h2⟩
      · exact Or.inl h
      · exact Or.inr ⟨h1, Nat.lt_succ_of_lt h2⟩
  · intro hv
    refine ⟨(hgv hv).trans hg.symm, fun h => ?_⟩
    have := (base (hgv hv)).1; simp at h; omega
  · intro h
    have h' : S0.e.gen = σ.cur.e.gen := h.trans hg
    obtain ⟨a, b⟩ := base h'
    refine ⟨by simp; omega, fun hb => hbb (b hb), fun h2 => ?_⟩
    simp at h2; omega
  · intro _ _ h1 h2
    have := (base (h1.trans hg)).1; simp at h2; omega
  · intro u0 u _ h2
    have h2' : e.st = .built (some u) := h2
    simp [Entry.upkeep, h2'] at hu

theorem inv_batch {B : Nat} {σ : Sys} (hI : Inv B σ) {S0 : Pub} (h0 : S0 ∈ σ.cur :: σ.hist) (b adv : Bool) :
    Inv B (batchSys b adv S0 σ) := by
  unfold batchSys
  split
  · rename_i e he
    obtain ⟨hP, hF, hg, hc, hcb, hu, _⟩ := batch_core hI h0 b adv he
    refine ⟨?_, rel_builder hI hg hc (fun h => by simp [hcb] at h) hu (fun _ => hcb), hI.wver⟩
    intro S hS
    rcases List.mem_cons.1 hS with rfl | hS
    · exact ⟨hP, hF⟩
    · exact hI.pub S hS
  · exact hI

theorem inv_job {B : Nat} {σ : Sys} (hI : Inv B σ) {S0 : Pub} (h0 : S0 ∈ σ.cur :: σ.hist) (b : Bool) :
    Inv B (jobSys b S0 σ) := by
  unfold jobSys
  split
  · rename_i e n he
    obtain ⟨hP, hF, hnx, hg, hc, hst, hcb⟩ := job_core hI h0 b he
    have hu : e.upkeep = none := by simp [Entry.upkeep, hst]
    have heb : e.isBuilt = true := isBuilt_built _ _ hst
    have hrel := rel_builder (B := B) hI hg hc (fun _ => heb) hu (fun h => by simp [heb] at h)
    refine ⟨?_, fun S0' hS0' => hrel S0' hS0', hI.wver⟩
    intro S hS
    rcases List.mem_cons.1 hS with rfl | hS
    · exact ⟨hP, hF⟩
    · exact ⟨(hI.pub S hS).1, (hI.pub S hS).2.mono hnx⟩
  · exact hI

theorem inv_step {B : Nat} {σ σ' : Sys} (hI : Inv B σ) (hs : Step D B σ σ') : Inv B σ' := by
  cases hs with
  | writer fork hf hv pre post b => exact inv_writer hI hf hv pre post b
  | batch S0 h0 b adv => exact inv_batch hI h0 b adv
  | job S0 h0 b => exact inv_job hI h0 b
  | other => exact inv_other hI

/-- **The invariant holds in every reachable state**, for every bound `B` and every interleaving of writers,
    batches, Built-entry jobs and other indexes' batches. -/
theorem reach_inv {B : Nat} {σ : Sys} (h : Reach D B σ) : Inv B σ := by
  induction h with
  | init => exact inv_init B
  | step _ hs ih => exact inv_step ih hs

end CchReview
